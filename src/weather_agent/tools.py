import json
import logging
import time
import traceback
import uuid
from collections.abc import Callable
from typing import Annotated

from jakarta.inject import Singleton
from micronaut.mcp.annotations import Tool, ToolArg
from weather_core.expr.help import describe_language
from weather_core.expr.redact import Redacted
from weather_core.sources.base import SourceError

from .forecast_service import ForecastService
from .geocoding import GeocodeService

log = logging.getLogger(__name__)

# Annotation arguments must be literals: Pyronaut reads them from source without evaluating.


def _run(fn: Callable[[], dict], call: str, scrub: Callable[[str], str] = lambda m: m,
         summarize: Callable[[dict], str] = lambda r: "") -> str:
    """Run a tool and log the call. `call` and what `scrub`/`summarize` let through must not locate anyone:
    the log shows what agents ask, not where."""
    started = time.monotonic()
    outcome = "failed"
    try:
        result = fn()
        outcome = f"ok{summarize(result)}"
        return json.dumps(result, ensure_ascii=False)
    except (SourceError, ValueError) as e:  # includes QueryError/ExprError: bad queries, timeouts, limits
        outcome = f"rejected ({type(e).__name__}: {scrub(str(e))})"
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    except BaseException as e:  # Java exceptions surface here as foreign exceptions, not Python Exceptions
        if isinstance(e, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        reference = uuid.uuid4().hex[:8]
        outcome = f"internal error [{reference}]"
        log.error("tool failed [%s]: %s: %s\n%s", reference, type(e).__name__, e, traceback.format_exc())
        # Details stay in the server log; the client gets a reference to quote.
        return json.dumps({"error": f"internal error (reference {reference})"}, ensure_ascii=False)
    finally:
        log.info("%s: %s in %.1f s", call, outcome, time.monotonic() - started)


def _forecast_summary(scrub: Callable[[str], str]) -> Callable[[dict], str]:
    def summarize(result: dict) -> str:
        models = ", ".join(m.get("model", "?") for m in result.get("models", []))
        out = f", models [{models}], {result.get('downloaded_mb', 0)} MB downloaded"
        if result.get("unavailable_sources"):
            out += f", unavailable {[s['source'] for s in result['unavailable_sources']]}"
        if result.get("warnings"):
            out += f", warnings {[scrub(w) for w in result['warnings']]}"
        return out
    return summarize


@Singleton
class ForecastTools:
    def __init__(self, service: ForecastService):
        self.forecaster = service.forecaster

    @Tool(description="Reference for forecast queries: the dataset, its dimensions and variables, selecting "
                      "locations and times, reductions, and examples. Read this before the first forecast query.")
    def weather_query_help(self) -> str:
        log.info("weather_query_help")
        return describe_language()

    @Tool(description="Ensemble weather forecast queried with a subset of xarray (Python syntax) over one dataset, "
                      "forecast(), with dimensions model, member, time, lat and lon. The query picks its location "
                      "(.interp(lat=…, lon=…), .interp(places(…)), .interp(route(…)) or an area with "
                      ".sel(lat=slice(…), lon=slice(…))), its times and models with .sel(…), and reduces dimensions "
                      "explicitly, e.g. (forecast().interp(lat=50.94, lon=6.96).sel(time=slice('2026-10-10T10:00', "
                      "'2026-10-10T16:00')).precip.sum('time') < 0.5).mean('member') gives each model's probability "
                      "of a dry afternoon. Returns the answer, its units, warnings about likely mistakes, a "
                      "per-model table (run, lead hours, members) and attribution. Read weather_query_help first.")
    def forecast(self,
                 query: Annotated[str, ToolArg(description="The query; see weather_query_help")],
                 gpx: Annotated[str | None, ToolArg(description="Optional GPX document; the query refers to it as route(gpx, start=…, …)")] = None) -> str:
        redacted = Redacted(query, keep=(s.name for s in self.forecaster.sources))
        call = f"forecast {redacted}" + (f" (gpx: {len(gpx)} characters)" if gpx is not None else "")
        return _run(lambda: self.forecaster.forecast(query, gpx=gpx), call, redacted.scrub,
                    _forecast_summary(redacted.scrub))

    @Tool(description="Summarise a route from GPX text or a polyline: length, bounding box, duration at a speed, "
                      "and a compact encoded polyline to use in forecast queries (route(polyline=…)) instead of the full GPX.")
    def describe_route(self,
                       polyline: Annotated[str | None, ToolArg(description="Encoded polyline (precision 5)")] = None,
                       gpx: Annotated[str | None, ToolArg(description="GPX document text")] = None,
                       speed_kmh: Annotated[float | None, ToolArg(description="Average speed in km/h")] = None) -> str:
        given = [f"{name}: {len(value)} characters" for name, value in (("polyline", polyline), ("gpx", gpx))
                 if value is not None]
        call = f"describe_route ({', '.join(given) or 'no route'}, speed {speed_kmh} km/h)"
        return _run(lambda: self.forecaster.describe_route(polyline=polyline, gpx=gpx, speed_kmh=speed_kmh), call,
                    lambda m: "…")


@Singleton
class PlaceTools:
    def __init__(self, service: GeocodeService):
        self.geocoder = service.geocoder

    @Tool(description="Resolve place names to coordinates with OpenStreetMap's Nominatim (best match only). Give "
                      "complete names, ideally with region or country ('Freiburg im Breisgau', 'Bonn, Germany'), "
                      "not prefixes to autocomplete. Several names separated by ';' (max 10). Each result has lat, "
                      "lon, the matched label (check it is the place you meant) and a 'place' string for "
                      "forecast().interp(places('…')) in queries. Uncached names take about a second each. Credit "
                      "the attribution when showing results.")
    def resolve_place(self,
                      query: Annotated[str, ToolArg(description="Place name(s), e.g. 'Cologne, Germany; Bonn, Germany'")]) -> str:
        names = [n for n in query.split(";") if n.strip()]
        return _run(lambda: self.geocoder.resolve_many(query), f"resolve_place ({len(names)} name(s))",
                    lambda m: "…",
                    lambda r: f", {sum('error' in p for p in r['results'])} not found")
