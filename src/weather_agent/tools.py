import json
import logging
import traceback
import uuid
from collections.abc import Callable
from typing import Annotated

from jakarta.inject import Singleton
from micronaut.mcp.annotations import Tool, ToolArg
from weather_core.cube import describe_schema
from weather_core.sources.base import SourceError

from .forecast_service import ForecastService

log = logging.getLogger(__name__)

# Annotation arguments must be literals: Pyronaut reads them from source without evaluating.


def _run(fn: Callable[[], dict]) -> str:
    try:
        return json.dumps(fn(), ensure_ascii=False)
    except (SourceError, ValueError) as e:  # includes QueryError: bad SQL, timeouts, limits
        return json.dumps({"error": str(e)}, ensure_ascii=False)
    except BaseException as e:  # Java exceptions surface here as foreign exceptions, not Python Exceptions
        if isinstance(e, (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        reference = uuid.uuid4().hex[:8]
        log.error("tool failed [%s]: %s: %s\n%s", reference, type(e).__name__, e, traceback.format_exc())
        # Details stay in the server log; the client gets a reference to quote.
        return json.dumps({"error": f"internal error (reference {reference})"}, ensure_ascii=False)


def _sources(value: str | None) -> list[str] | None:
    return [s.strip() for s in value.split(",") if s.strip()] if value else None


def _bbox(value: str | None) -> list[float] | None:
    if not value:
        return None
    parts = value.split(",")
    try:
        box = [float(x) for x in parts]
    except ValueError:
        box = []
    if len(box) != 4:
        raise ValueError(f"bbox must be four comma-separated numbers south,west,north,east, got {value!r}")
    return box


@Singleton
class ForecastTools:
    def __init__(self, service: ForecastService):
        self.forecaster = service.forecaster

    @Tool(description="Schema, columns, helper macros and example SQL for the forecast tool. Read this before the "
                      "first forecast query.")
    def weather_query_help(self) -> str:
        return describe_schema()

    @Tool(description="Ensemble weather forecast: run a DuckDB SQL query over every model's members for one "
                      "location form (exactly one): a point (lat, lon), named places (places), an area (lat, lon, "
                      "radius_km, or bbox) or a route (polyline/gpx with speed_kmh or use_gpx_times; start is the "
                      "departure). Views: samples, per_member, models. With window_hours, start..end is a search "
                      "range and every hourly window of that length is loaded (window_start/window_end), so the "
                      "query can rank windows. Returns the result, a per-model table (run, lead hours, members, "
                      "notes) and attribution. Results are per model; compare them yourself. Read "
                      "weather_query_help first.")
    def forecast(self,
                 query: Annotated[str, ToolArg(description="DuckDB SQL, e.g. SELECT model, prob(rain < 0.5 AND tmax <= 26) AS p FROM per_member GROUP BY model")],
                 start: Annotated[str, ToolArg(description="Window start (route: departure), ISO 8601 local time, e.g. 2026-10-10T10:00 (Europe/Berlin unless an offset is given)")],
                 end: Annotated[str | None, ToolArg(description="Window end, ISO 8601 local time; required except for routes")] = None,
                 lat: Annotated[float | None, ToolArg(description="Point latitude, or circle centre with radius_km")] = None,
                 lon: Annotated[float | None, ToolArg(description="Point longitude, or circle centre with radius_km")] = None,
                 places: Annotated[str | None, ToolArg(description="Several named points: 'Cologne@50.94,6.96; Bonn@50.73,7.10' (max 20); adds a place column")] = None,
                 radius_km: Annotated[float | None, ToolArg(description="Area: circle radius in km around lat/lon")] = None,
                 bbox: Annotated[str | None, ToolArg(description="Area: south,west,north,east")] = None,
                 polyline: Annotated[str | None, ToolArg(description="Route as encoded polyline (precision 5); see describe_route")] = None,
                 gpx: Annotated[str | None, ToolArg(description="Route as GPX document text")] = None,
                 speed_kmh: Annotated[float | None, ToolArg(description="Route: average speed in km/h")] = None,
                 use_gpx_times: Annotated[bool | None, ToolArg(description="Route: pace by the GPX timestamps (shifted to start) instead of a speed")] = None,
                 window_hours: Annotated[float | None, ToolArg(description="Load every hourly window of this length within start..end (not for routes)")] = None,
                 sources: Annotated[str | None, ToolArg(description="Optional comma-separated model names to restrict to (default: all)")] = None) -> str:
        return _run(lambda: self.forecaster.forecast(
            query, start, end, lat=lat, lon=lon, places=places, radius_km=radius_km, bbox=_bbox(bbox),
            polyline=polyline, gpx=gpx, speed_kmh=speed_kmh, use_gpx_times=bool(use_gpx_times),
            window_hours=window_hours, sources=_sources(sources)))

    @Tool(description="Summarise a route from GPX text or a polyline: length, bounding box, duration at a speed, "
                      "and a compact encoded polyline to pass to forecast_route instead of the full GPX.")
    def describe_route(self,
                       polyline: Annotated[str | None, ToolArg(description="Encoded polyline (precision 5)")] = None,
                       gpx: Annotated[str | None, ToolArg(description="GPX document text")] = None,
                       speed_kmh: Annotated[float | None, ToolArg(description="Average speed in km/h")] = None) -> str:
        return _run(lambda: self.forecaster.describe_route(polyline=polyline, gpx=gpx, speed_kmh=speed_kmh))
