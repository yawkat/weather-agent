"""Answering forecast queries (weather_core.expr): fetch what a query needs from every model, then evaluate it."""

import logging
import math
import threading
import time
import traceback
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Protocol
from zoneinfo import ZoneInfo

import numpy as np

from .budget import DownloadBudget
from .chart import check_type, encode_chart
from .expr import runtime as rt
from .expr.axes import BOOL, LABEL, LAT, LON, POINT, RECORD, ExprError, Type
from .expr.language import Demand, Env, Location, Sizes, compile_query, parse
from .geometry import LatLon, area_grid, decode_polyline, encode_polyline, parse_gpx, sample_route, simplify
from .grid import OutsideRegion
from .sources.base import Prepared, Query, SourceError
from .timeaxis import OutsideForecast

log = logging.getLogger(__name__)

# Ensembles reach ~15 days ahead; anything outside this span can't be answered and only costs work.
MAX_PAST = timedelta(days=2)
MAX_AHEAD = timedelta(days=16)
MAX_WINDOW = timedelta(days=16)
MAX_PLACES = 20
# Bound on values loaded per variable and location (members × hours × points × models), checked before downloading.
MAX_ROWS = 4_000_000
AREA_MAX_POINTS = 500  # ~10 km spacing is already finer than the 0.25° ECMWF grid
ASSUMED_MEMBERS = 50  # per model, for the size estimate


class QueryError(ValueError):
    """A query can't be answered (no model has data, the server is busy)."""


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


@dataclass
class _Resolved:
    """A query location turned into a source query and coordinates."""
    query: Query
    time: rt.Coord
    space: dict[str, rt.Coord]  # spatial dimensions: none, point, or lat and lon
    sources: list[Source]
    variables: list[str]
    lat: float = math.nan
    lon: float = math.nan
    summary: dict = field(default_factory=dict)


class Forecaster:
    def __init__(self, sources: Sequence[Source], default_tz: str = "Europe/Berlin",
                 clock=lambda: datetime.now(timezone.utc), budget: DownloadBudget | None = None,
                 max_concurrent: int = 4, busy_timeout_s: float = 30.0, eval_bytes: int = 1 << 30,
                 eval_timeout_s: float = 15.0, max_rows: int = 500):
        self.sources = list(sources)
        self.budget = budget  # the one the sources reserve from; a query's sources share its per-query limit
        self.default_tz = ZoneInfo(default_tz)
        self.clock = clock
        # Evaluations (not downloads) run at most this many at a time: each may hold eval_bytes.
        self._evaluations = threading.BoundedSemaphore(max_concurrent)
        self.busy_timeout_s = busy_timeout_s
        self.eval_bytes = eval_bytes
        self.eval_timeout_s = eval_timeout_s
        self.max_rows = max_rows  # result table rows

    # -- public operations ---------------------------------------------------------------------------------------

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

    def forecast(self, text: str, gpx: str | None = None) -> dict:
        """Answer a forecast query (weather_core.expr). Locations, times and models are part of the query; `gpx` is
        a GPX document the query can refer to as route(gpx, …)."""
        return self._answer(text, gpx, chart=False)[0]

    def visualize(self, text: str, gpx: str | None = None) -> tuple[dict, dict]:
        """Answer a query for display: the answer as `forecast` gives it (a summary instead of the table if that
        would be too long) and every value as a chart (weather_core.chart)."""
        return self._answer(text, gpx, chart=True)

    def _answer(self, text: str, gpx: str | None, chart: bool) -> tuple[dict, dict | None]:
        program = parse(text)
        env = Env(tuple(s.name for s in self.sources), {s.name: s.provides() for s in self.sources},
                  self.default_tz, gpx is not None)
        compiled = compile_query(program, env)
        if chart:
            check_type(compiled.result.type)  # before downloading anything
        resolved = {loc: self._resolve(loc, demand, gpx) for loc, demand in compiled.demands.items()}
        for r in resolved.values():
            cells = len(r.time) * math.prod(len(c) for c in r.space.values()) * ASSUMED_MEMBERS * len(r.sources)
            if cells > MAX_ROWS:
                raise ExprError(f"this query would load about {cells / 1e6:.0f}M values per variable (limit "
                                f"{MAX_ROWS / 1e6:.0f}M); select fewer hours, a smaller area or fewer models")
        def largest(dim: str) -> int:
            return max((len(r.space[dim]) for r in resolved.values() if dim in r.space), default=1)
        sizes = Sizes(len(self.sources), ASSUMED_MEMBERS, max(len(r.time) for r in resolved.values()),
                      largest(POINT), largest(LAT), largest(LON))
        estimate = compiled.estimate(sizes)
        if estimate > self.eval_bytes:
            raise ExprError(f"evaluating this query would need about {estimate / 2**20:.0f} MB (limit "
                            f"{self.eval_bytes // 2**20} MB); reduce dimensions earlier, or select fewer hours, "
                            f"a smaller area or fewer models")

        problems: list[dict] = []
        fetched: dict[Location, list[tuple[Source, Prepared]]] = {}
        with self.budget.query() if self.budget is not None else nullcontext():
            for loc, r in resolved.items():
                for source in r.sources:
                    p = _attempt(source, problems, lambda: source.prepare(r.query, r.variables))
                    if p is not None:
                        fetched.setdefault(loc, []).append((source, p))

        warnings = list(compiled.warnings)
        table: dict[tuple, dict] = {}
        attribution = []
        # Hold the permit from building the arrays on: waiting queries keep nothing but the prepared fields.
        with self._evaluating():
            locations = {}
            for loc, r in resolved.items():
                per_model = []
                for source, p in fetched.get(loc, []):
                    s = p.samples()
                    n = len(s.times)
                    read = [(t, ok) for t, ok, unused in zip(s.times, s.valid if s.valid is not None else [True] * n,
                                                             s.unused if s.unused is not None else [False] * n)
                            if not unused]
                    valid = [t for t, ok in read if ok]
                    if len(valid) < len(read):
                        covered = [rt.format_time(t.timestamp() / 60, self.default_tz)
                                   for t in (valid[0], valid[-1] + timedelta(hours=1))]
                        warnings.append(f"{source.name} covers only {covered[0]} to {covered[1]} of the selected "
                                        f"time at {loc.describe()}")
                    per_model.append((source.name, s))
                    self._model_entry(table, source, p, valid, r.variables)
                    entry = p.attribution(valid)  # only the hours the run really covers
                    if entry not in attribution:
                        attribution.append(entry)
                if not per_model:
                    raise QueryError(f"no model could provide data for {loc.describe()}: "
                                     + "; ".join(f"{p['source']}: {p['reason']}" for p in problems))
                locations[loc] = rt.location_data(per_model, r.variables, r.time, r.space, r.lat, r.lon)
            ctx = rt.Context(self.default_tz, self.eval_bytes, time.monotonic() + self.eval_timeout_s)
            ctx.locations = locations
            value = compiled.evaluate(ctx)
            drawn = encode_chart(value, compiled.result.type, ctx) if chart else None
            try:
                result = rt.encode(value, compiled.result.type, ctx, self.max_rows)
            except rt.TooManyRows:
                if not chart:
                    raise
                result = _summary(value, compiled.result.type)  # the chart has every value
        answer = {
            "result": result,
            "units": _units(compiled.result.type),
            "warnings": warnings + ctx.warnings.messages,
            "models": list(table.values()),
            "unavailable_sources": problems,
            "attribution": attribution,
            "downloaded_mb": round(sum(p.bytes_downloaded for ps in fetched.values() for _, p in ps) / 1e6, 1),
        }
        summaries = [r.summary for r in resolved.values() if r.summary]
        if summaries:
            answer["locations"] = summaries
        if drawn is not None:
            drawn["models"] = answer["models"]
            drawn["attribution"] = attribution
            drawn["warnings"] = answer["warnings"]
            # Where the graphs of a single place are (no point dimension carries it): the view shades its nights.
            # Only when it's the query's one location: a graph from an area elsewhere has other nights.
            if len(resolved) == 1 and not math.isnan((only := next(iter(resolved.values()))).lat):
                drawn["location"] = {"lat": round(only.lat, 4), "lon": round(only.lon, 4)}
        return answer, drawn

    # -- internals -----------------------------------------------------------------------------------------------

    @contextmanager
    def _evaluating(self) -> Iterator[None]:
        """Bound how many evaluations run at once (memory); waiting ones give up after busy_timeout_s."""
        if not self._evaluations.acquire(timeout=self.busy_timeout_s):
            raise QueryError("the server is busy with other queries; try again shortly")
        try:
            yield
        finally:
            self._evaluations.release()

    def _resolve(self, loc: Location, demand: Demand, gpx: str | None) -> _Resolved:
        sources = [s for s in self.sources if demand.models is None or s.name in demand.models]
        variables = sorted(demand.variables)
        if loc.kind == "route":
            track, start, speed, use_times = loc.args[:-3], loc.args[-3], loc.args[-2], loc.args[-1]
            points, track_times = self._route_points(track[1] if track[0] == "polyline" else None,
                                                     gpx if track[0] == "gpx" else None)
            departure = datetime.fromtimestamp(start * 60, timezone.utc)
            self._check_range(departure, departure)
            route = sample_route(points, departure, speed_kmh=None if use_times else speed,
                                 track_times=track_times if use_times else None)
            self._check_range(route.times[0], route.times[-1])
            query = Query(route.lat, route.lon, times=route.times, dt_hours=route.dt_hours, bearing=route.bearing)
            minutes = [t.timestamp() / 60 for t in route.times]
            time_coord = rt.Coord(rt.labels(minutes), {"dt": np.asarray(route.dt_hours, dtype=np.float64),
                                                       "distance_km": np.asarray(route.distance_km),
                                                       "lat": np.asarray(route.lat), "lon": np.asarray(route.lon)},
                                  "route")
            summary = {"route": {"length_km": round(route.total_km, 1),
                                 "start": self._format(route.times[0], self.default_tz),
                                 "end": self._format(route.times[-1], self.default_tz),
                                 "samples": len(route.times)}}
            return _Resolved(query, time_coord, {}, sources, variables, summary=summary)

        start = datetime.fromtimestamp(demand.start * 60, timezone.utc)
        end = datetime.fromtimestamp(demand.end * 60, timezone.utc)
        self._check_range(start, end)
        first = start.replace(minute=0, second=0, microsecond=0)
        first += timedelta(hours=1) if first < start else timedelta(0)
        hours = []
        while first + timedelta(hours=len(hours)) < end:
            hours.append(first + timedelta(hours=len(hours)))
        if not hours:
            raise ExprError(f"{loc.describe()}: the selected time range contains no whole hour")
        time_coord = rt.Coord(rt.labels(int(t.timestamp() // 60) for t in hours), {"dt": np.ones(len(hours))}, "1h")
        space: dict[str, rt.Coord] = {}
        summary: dict = {}
        if loc.kind == "point":
            lat, lon = loc.args
            _check_point(lat, lon)
            points = [LatLon(lat, lon)]
        elif loc.kind == "points":
            points = [LatLon(lat, lon) for _, lat, lon in loc.args]
            if all(name for name, _, _ in loc.args):
                names, points = _check_places([name for name, _, _ in loc.args], points)
                extra = {"place": np.array(names, dtype=object)}
            else:
                for p in points:
                    _check_point(p.lat, p.lon)
                # Position and full coordinates: unique even for repeated points, and points of different
                # locations only align where they really are the same point.
                names, extra = [f"{i}:{p.lat!r},{p.lon!r}" for i, p in enumerate(points)], {}
            space[POINT] = rt.Coord(rt.labels(names), {**extra, "lat": np.array([p.lat for p in points]),
                                                       "lon": np.array([p.lon for p in points])})
        else:
            lats, lons = area_grid(loc.args, spacing_km=10.0, max_points=AREA_MAX_POINTS)
            space = {LAT: rt.Coord(rt.labels(round(float(x), 4) for x in lats)),
                     LON: rt.Coord(rt.labels(round(float(x), 4) for x in lons))}
            points = [LatLon(float(a), float(b)) for a in lats for b in lons]  # lat-major, as location_data expects
            summary = {"area": {"lat_points": len(lats), "lon_points": len(lons)}}
        # Hours between the selected windows (e.g. the night between two afternoons) stay on the axis, unread.
        minutes = [int(t.timestamp() // 60) for t in hours]
        wanted = np.array([any(a <= m < b for a, b in demand.windows) for m in minutes]) if demand.windows else None
        if wanted is not None and not wanted.any():
            raise ExprError(f"{loc.describe()}: the selected time range contains no whole hour")
        query = Query(np.array([p.lat for p in points]), np.array([p.lon for p in points]), area=bool(space),
                      hours=hours, wanted=None if wanted is None or wanted.all() else wanted)
        lat, lon = (points[0].lat, points[0].lon) if loc.kind == "point" else (math.nan, math.nan)
        return _Resolved(query, time_coord, space, sources, variables, lat, lon, summary)

    @staticmethod
    def _model_entry(table: dict, source: Source, p: Prepared, valid: list[datetime], variables: list[str]) -> None:
        """Add one location's answer to the model table: one row per model and run (locations of one query may get
        different runs, e.g. when a new run is published in between)."""
        leads = [(valid[0] - p.run).total_seconds() / 3600, (valid[-1] - p.run).total_seconds() / 3600]
        entry = table.get((source.name, p.run))
        missing = sorted(set(variables) - source.provides())
        if entry is None:
            table[(source.name, p.run)] = {
                "model": source.name,
                "name": p.info.model,
                "run": p.run.strftime("%Y-%m-%dT%H:%MZ"),
                "members": p.info.members,
                "lead_hours": [round(min(leads)), round(max(leads))],
                **source.describe(),
                **({"missing_variables": missing} if missing else {}),
            }
            return
        entry["lead_hours"] = [min(entry["lead_hours"][0], round(min(leads))),
                               max(entry["lead_hours"][1], round(max(leads)))]
        missing = sorted(set(entry.get("missing_variables", [])) | set(missing))
        if missing:
            entry["missing_variables"] = missing

    def _route_points(self, polyline: str | None, gpx: str | None):
        if (polyline is None) == (gpx is None):
            raise ValueError("give either polyline or gpx")
        if polyline is not None:
            return decode_polyline(polyline), None
        track = parse_gpx(gpx)
        return track.points, track.times

    def _check_range(self, a: datetime, b: datetime) -> None:
        now = self.clock()
        if a < now - MAX_PAST or b > now + MAX_AHEAD:
            raise ValueError(f"times must lie between {MAX_PAST.days} days ago and {MAX_AHEAD.days} days ahead; "
                             f"ensembles don't forecast further")
        if b - a > MAX_WINDOW:
            raise ValueError(f"time range longer than {MAX_WINDOW.days} days")

    @staticmethod
    def _format(t: datetime, tz: tzinfo) -> str:
        return t.astimezone(tz).isoformat(timespec="minutes")


def _summary(value, typ: Type) -> dict:
    """Instead of a table that would be too long: each value's dimensions and range."""
    types = dict(typ.fields) if isinstance(value, rt.Rec) else {None: typ}
    fields = value.fields.items() if isinstance(value, rt.Rec) else [(None, value)]
    out = {}
    for name, v in fields:
        finite = v.data[np.isfinite(v.data)]
        entry = {"dims": {d: len(v.coords[d]) for d in v.dims}}
        if finite.size and types[name].kind != LABEL:  # labels are indices: no meaningful range
            entry |= {"min": round(float(finite.min()), 3), "mean": round(float(finite.mean()), 3),
                      "max": round(float(finite.max()), 3)}
        out[name or "value"] = entry
    return {"too_long_for_a_table": out}


def _units(t: Type):
    """Unit of each answer value ("" = dimensionless, e.g. probabilities); None for labels."""
    if t.kind == RECORD:
        return {k: _units(f) for k, f in t.fields}
    if t.kind == LABEL:
        return None
    if t.kind == BOOL or t.unit is None:
        return ""
    return str(t.unit)


def _check_point(lat: float, lon: float) -> None:
    if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("lat must be within ±90 and lon within ±180 degrees")


def _check_places(names: list[str], points: list[LatLon]) -> tuple[list[str], list[LatLon]]:
    """Validate places; names are normalised (whitespace, length, a default for empty ones)."""
    names = list(names)
    for i, point in enumerate(points):
        _check_point(point.lat, point.lon)
        names[i] = " ".join(names[i].split())[:60] or f"place {i + 1}"
        if not names[i].isprintable():
            raise ValueError("place names must be printable text")
    if not points:
        raise ValueError("places is empty")
    if len(points) > MAX_PLACES:
        raise ValueError(f"at most {MAX_PLACES} places")
    if len(set(names)) != len(names):
        raise ValueError("place names must be unique")
    return names, points
