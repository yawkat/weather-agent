"""Answering forecast questions: sample every suitable source into SQL tables, then run the client's query."""

import logging
import math
import traceback
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Protocol
from zoneinfo import ZoneInfo

import numpy as np

from .cube import CubeData, ModelEntry, QueryEngine, QueryError, referenced_variables
from .geometry import LatLon, area_points, decode_polyline, encode_polyline, parse_gpx, sample_route, simplify
from .grid import OutsideRegion
from .sources.base import Prepared, Query, SourceError
from .timeaxis import OutsideForecast
from .variables import CATALOG

log = logging.getLogger(__name__)

# Ensembles reach ~15 days ahead; anything outside this span can't be answered and only costs work.
MAX_PAST = timedelta(days=2)
MAX_AHEAD = timedelta(days=16)
MAX_WINDOW = timedelta(days=16)
MAX_PLACES = 20
MAX_SQL_CHARS = 8000
# Bound on rows loaded for one query (members × windows × time samples × points), checked before downloading.
MAX_ROWS = 4_000_000
AREA_MAX_POINTS = 500  # ~10 km spacing is already finer than the 0.25° ECMWF grid
ASSUMED_MEMBERS = 50  # per model, for the size estimate


class Source(Protocol):
    name: str

    def provides(self) -> set[str]: ...

    def prepare(self, query: Query, variables: Sequence[str]) -> Prepared: ...

    def describe(self) -> dict: ...


_EXPECTED = (SourceError, OutsideForecast, OutsideRegion)
_FATAL = (KeyboardInterrupt, SystemExit, GeneratorExit)


def _attempt(source: Source, problems: list[dict], fn):
    """Run one source's part of a query; a failing source is reported, not fatal to the whole answer."""
    try:
        return fn()
    except _EXPECTED as e:
        problems.append({"source": source.name, "reason": str(e)})
    except BaseException as e:  # Java exceptions (network, decoding) arrive as foreign exceptions
        if isinstance(e, _FATAL):
            raise
        # log.exception's traceback is dropped by the Python→Logback bridge, so put it into the message.
        log.error("source %s failed: %s: %s\n%s", source.name, type(e).__name__, e, traceback.format_exc())
        # Details stay in the server log; clients only learn the kind of failure.
        problems.append({"source": source.name, "reason": f"failed ({type(e).__name__}); see server log"})
    return None


class Forecaster:
    def __init__(self, sources: Sequence[Source], engine: QueryEngine, default_tz: str = "Europe/Berlin",
                 clock=lambda: datetime.now(timezone.utc)):
        self.sources = list(sources)
        self.engine = engine
        self.default_tz = ZoneInfo(default_tz)
        self.clock = clock

    # -- public operations ---------------------------------------------------------------------------------------

    def forecast(self, sql: str, start: str, end: str | None = None, *, lat: float | None = None,
                 lon: float | None = None, places: str | None = None, radius_km: float | None = None,
                 bbox: Sequence[float] | None = None, polyline: str | None = None, gpx: str | None = None,
                 speed_kmh: float | None = None, use_gpx_times: bool = False, window_hours: float | None = None,
                 sources: Sequence[str] | None = None) -> dict:
        """Run `sql` over the samples for one location form: a point, named places, an area, or a route."""
        tz = self._tz(start)
        route = polyline is not None or gpx is not None
        modes = [route, places is not None, bbox is not None or radius_km is not None]
        stray_coordinates = (route or places is not None or bbox is not None) and (lat is not None or lon is not None)
        if sum(modes) > 1 or stray_coordinates or (not any(modes) and (lat is None or lon is None)):
            raise ValueError("give exactly one location: lat+lon (point), places, lat+lon+radius_km or bbox "
                             "(area), or polyline/gpx (route)")
        if route:
            if end is not None or window_hours is not None:
                raise ValueError("routes take only a departure time (start); end and window_hours don't apply")
            return self._route(sql, start, tz, polyline, gpx, speed_kmh, use_gpx_times, sources)
        if end is None:
            raise ValueError("end is required (except for routes)")
        range_start, range_end = self._window(start, end)
        windows = self._windows(range_start, range_end, window_hours)
        extra = {}
        names = None
        if places is not None:
            names, points = _parse_places(places)
            kind = "places"
        elif bbox is not None or radius_km is not None:
            if (lat is None) != (lon is None):
                raise ValueError("give both lat and lon for a circle")
            if lat is not None:
                _check_point(lat, lon)
            points = area_points(center=LatLon(lat, lon) if lat is not None else None, radius_km=radius_km,
                                 bbox=tuple(bbox) if bbox else None, spacing_km=10.0, max_points=AREA_MAX_POINTS)
            kind = "area"
            extra["area_points"] = len(points)
        else:
            _check_point(lat, lon)
            points = [LatLon(lat, lon)]
            kind = "point"
        query = Query(np.array([p.lat for p in points]), np.array([p.lon for p in points]),
                      window=(range_start, range_end), area=kind != "point")
        answer = self._run(query, sql, sources, tz, kind, windows, place_names=names)
        answer.update(extra)
        return answer

    def _route(self, sql: str, start: str, tz: tzinfo, polyline: str | None, gpx: str | None,
               speed_kmh: float | None, use_gpx_times: bool, sources: Sequence[str] | None) -> dict:
        points, track_times = self._route_points(polyline, gpx)
        start_time = self._time(start)
        self._check_range(start_time, start_time)
        route = sample_route(points, start_time, speed_kmh=None if use_gpx_times else speed_kmh,
                             track_times=track_times if use_gpx_times else None)
        self._check_range(route.times[0], route.times[-1])
        query = Query(route.lat, route.lon, times=route.times, dt_hours=route.dt_hours, bearing=route.bearing)
        answer = self._run(query, sql, sources, tz, "route", [(route.times[0], route.times[-1])],
                           distance_km=route.distance_km)
        answer["route"] = {
            "length_km": round(route.total_km, 1),
            "start": self._format(route.times[0], tz),
            "end": self._format(route.times[-1], tz),
            "samples": len(route.times),
        }
        return answer

    @staticmethod
    def _windows(start: datetime, end: datetime, window_hours: float | None) -> list[tuple[datetime, datetime]]:
        """The whole range, or every hourly window of `window_hours` within it."""
        if window_hours is None:
            return [(start, end)]
        if not (math.isfinite(window_hours) and 0 < window_hours <= (end - start).total_seconds() / 3600):
            raise ValueError("window_hours must be positive and fit inside start..end")
        length = timedelta(hours=window_hours)
        windows = []
        t = start
        while t + length <= end:
            windows.append((t, t + length))
            t += timedelta(hours=1)
        return windows

    def describe_route(self, polyline: str | None = None, gpx: str | None = None,
                       speed_kmh: float | None = None) -> dict:
        points, track_times = self._route_points(polyline, gpx)
        start = datetime(2000, 1, 1, tzinfo=timezone.utc)
        route = sample_route(points, start, speed_kmh=speed_kmh or 20.0)
        out = {
            "polyline": encode_polyline(simplify(points)),
            "length_km": round(route.total_km, 1),
            "start": {"lat": points[0].lat, "lon": points[0].lon},
            "end": {"lat": points[-1].lat, "lon": points[-1].lon},
            "bbox": [float(route.lat.min()), float(route.lon.min()), float(route.lat.max()), float(route.lon.max())],
            "duration_hours_at_speed": round(route.duration.total_seconds() / 3600, 2),
            "speed_kmh_assumed": speed_kmh or 20.0,
        }
        if track_times:
            out["gpx_duration_hours"] = round((track_times[-1] - track_times[0]).total_seconds() / 3600, 2)
        return out

    # -- internals -----------------------------------------------------------------------------------------------

    def variables(self, kind: str) -> set[str]:
        provided = set().union(*(s.provides() for s in self.sources)) if self.sources else set()
        names = {n for n in provided if n in CATALOG}
        if kind != "route":
            names -= {n for n, v in CATALOG.items() if v.route_only}
        return names

    def _select(self, names: Sequence[str] | None) -> list[Source]:
        if not names:
            return self.sources
        unknown = set(names) - {s.name for s in self.sources}
        if unknown:
            raise ValueError(f"unknown sources {sorted(unknown)}; available: {[s.name for s in self.sources]}")
        return [s for s in self.sources if s.name in names]

    def _run(self, query: Query, sql: str, sources: Sequence[str] | None, tz: tzinfo, kind: str,
             windows: list[tuple[datetime, datetime]], distance_km: np.ndarray | None = None,
             place_names: list[str] | None = None) -> dict:
        sql = sql.strip().rstrip(";").strip()
        if not sql:
            raise ValueError("query is empty")
        if len(sql) > MAX_SQL_CHARS:
            raise ValueError(f"query longer than {MAX_SQL_CHARS} characters")
        selected = self._select(sources)
        self._check_size(query, kind, windows, len(selected))
        variables = sorted(referenced_variables(sql, self.variables(kind)))

        problems: list[dict] = []
        prepared: list[tuple[Source, Prepared]] = []
        for source in selected:
            p = _attempt(source, problems, lambda: source.prepare(query, variables))
            if p is not None:
                prepared.append((source, p))

        cube = CubeData(variables, windows, query.lat, query.lon, distance_km, kind, tz, models=[],
                        place_names=place_names)
        models = []
        attribution = []
        for source, p in prepared:
            model_id = len(cube.models)
            leads = []
            for w, window in enumerate(windows):
                try:
                    s = p.samples() if len(windows) == 1 else p.samples(window=window)
                except OutsideForecast:
                    continue  # this model doesn't reach this window; others may
                cube.entries.append(ModelEntry(model_id, w, s))
                leads += [(s.times[0] - p.run).total_seconds() / 3600, (s.times[-1] - p.run).total_seconds() / 3600]
            if not leads:
                problems.append({"source": source.name, "reason": "no window lies within this model's range"})
                continue
            description = source.describe()
            cube.models.append({"model": source.name, "provider": p.info.provider, "run": p.run,
                                "members": p.info.members, **description})
            missing = sorted(set(variables) - source.provides())
            models.append({
                "model": source.name,
                "name": p.info.model,
                "run": p.run.strftime("%Y-%m-%dT%H:%MZ"),
                "members": p.info.members,
                "lead_hours": [round(min(leads)), round(max(leads))],
                **description,
                **({"missing_variables": missing} if missing else {}),
            })
            attribution.append(_attribution(p))

        if not cube.entries:
            raise QueryError("no model could provide data for this query: "
                             + "; ".join(f"{p['source']}: {p['reason']}" for p in problems))
        result = self.engine.run(cube.setup(), cube.tables(), sql)
        return {
            "result": result,
            "models": models,
            "unavailable_sources": problems,
            "attribution": attribution,
            "downloaded_mb": round(sum(p.bytes_downloaded for _, p in prepared) / 1e6, 1),
        }

    def _check_size(self, query: Query, kind: str, windows: list[tuple[datetime, datetime]], n_sources: int):
        if kind == "route":
            samples = len(query.times)
            points = 1
        else:
            # Two samples per model step (half-intervals); steps are at least hourly.
            samples = sum(2 * (b - a).total_seconds() / 3600 + 2 for a, b in windows)
            points = len(query.lat)
        rows = samples * points * ASSUMED_MEMBERS * n_sources
        if rows > MAX_ROWS:
            raise ValueError(f"this query would load about {rows / 1e6:.0f}M rows (limit {MAX_ROWS / 1e6:.0f}M); "
                             f"use a shorter window, a smaller area, or fewer candidate windows")

    def _route_points(self, polyline: str | None, gpx: str | None):
        if (polyline is None) == (gpx is None):
            raise ValueError("give either polyline or gpx")
        if polyline is not None:
            return decode_polyline(polyline), None
        track = parse_gpx(gpx)
        return track.points, track.times

    def _tz(self, text: str) -> tzinfo:
        parsed = datetime.fromisoformat(text)
        return parsed.tzinfo or self.default_tz

    def _time(self, text: str) -> datetime:
        try:
            parsed = datetime.fromisoformat(text)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=self.default_tz)
            return parsed.astimezone(timezone.utc)
        except (ValueError, OverflowError):
            raise ValueError(f"invalid time {text!r}; use ISO 8601 like 2026-10-10T10:00") from None

    def _check_range(self, a: datetime, b: datetime) -> None:
        now = self.clock()
        if a < now - MAX_PAST or b > now + MAX_AHEAD:
            raise ValueError(f"times must lie between {MAX_PAST.days} days ago and {MAX_AHEAD.days} days ahead; "
                             f"ensembles don't forecast further")
        if b - a > MAX_WINDOW:
            raise ValueError(f"window longer than {MAX_WINDOW.days} days")

    def _window(self, start: str, end: str) -> tuple[datetime, datetime]:
        a, b = self._time(start), self._time(end)
        if b <= a:
            raise ValueError("end must be after start")
        self._check_range(a, b)
        return a, b

    @staticmethod
    def _format(t: datetime, tz: tzinfo) -> str:
        return t.astimezone(tz).isoformat(timespec="minutes")


def _attribution(p: Prepared) -> dict:
    return p.attribution([p.query.window[0], p.query.window[1]] if p.query.window else p.query.times)


def _check_point(lat: float, lon: float) -> None:
    if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("lat must be within ±90 and lon within ±180 degrees")


def _parse_places(text: str) -> tuple[list[str], list[LatLon]]:
    """"Cologne@50.94,6.96; Bonn@50.73,7.10" → names and points."""
    names, points = [], []
    for part in (p.strip() for p in text.split(";")):
        if not part:
            continue
        name, at, coords = part.rpartition("@")
        try:
            lat_text, lon_text = coords.split(",")
            point = LatLon(float(lat_text), float(lon_text))
        except ValueError:
            raise ValueError(f"places must look like 'Name@lat,lon; Other@lat,lon', got {part!r}") from None
        _check_point(point.lat, point.lon)
        name = " ".join((name if at else coords).split())[:60] or f"place {len(names) + 1}"
        if not name.isprintable():
            raise ValueError("place names must be printable text")
        names.append(name)
        points.append(point)
    if not points:
        raise ValueError("places is empty")
    if len(points) > MAX_PLACES:
        raise ValueError(f"at most {MAX_PLACES} places")
    if len(set(names)) != len(names):
        raise ValueError("place names must be unique")
    return names, points
