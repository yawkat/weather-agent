import json
import logging
import traceback
import uuid
from collections.abc import Callable
from typing import Annotated

from jakarta.inject import Singleton
from micronaut.mcp.annotations import Tool, ToolArg
from weather_core.expr.help import describe_language
from weather_core.sources.base import SourceError

from .forecast_service import ForecastService
from .geocoding import GeocodeService

log = logging.getLogger(__name__)

# Annotation arguments must be literals: Pyronaut reads them from source without evaluating.


def _run(fn: Callable[[], dict]) -> str:
    try:
        return json.dumps(fn(), ensure_ascii=False)
    except (SourceError, ValueError) as e:  # includes QueryError/ExprError: bad queries, timeouts, limits
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    except BaseException as e:  # Java exceptions surface here as foreign exceptions, not Python Exceptions
        if isinstance(e, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        reference = uuid.uuid4().hex[:8]
        log.error("tool failed [%s]: %s: %s\n%s", reference, type(e).__name__, e, traceback.format_exc())
        # Details stay in the server log; the client gets a reference to quote.
        return json.dumps({"error": f"internal error (reference {reference})"}, ensure_ascii=False)


@Singleton
class ForecastTools:
    def __init__(self, service: ForecastService):
        self.forecaster = service.forecaster

    @Tool(description="Reference for forecast queries: the dataset, its dimensions and variables, selecting "
                      "locations and times, reductions, and examples. Read this before the first forecast query.")
    def weather_query_help(self) -> str:
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
        return _run(lambda: self.forecaster.forecast(query, gpx=gpx))

    @Tool(description="Summarise a route from GPX text or a polyline: length, bounding box, duration at a speed, "
                      "and a compact encoded polyline to use in forecast queries (route(polyline=…)) instead of the full GPX.")
    def describe_route(self,
                       polyline: Annotated[str | None, ToolArg(description="Encoded polyline (precision 5)")] = None,
                       gpx: Annotated[str | None, ToolArg(description="GPX document text")] = None,
                       speed_kmh: Annotated[float | None, ToolArg(description="Average speed in km/h")] = None) -> str:
        return _run(lambda: self.forecaster.describe_route(polyline=polyline, gpx=gpx, speed_kmh=speed_kmh))


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
        return _run(lambda: self.geocoder.resolve_many(query))
