"""Answers as charts (the show_forecast tool): every value of a result, as dense arrays the client's view draws.

The view picks the form from the dimensions left in each field, as docs/query-language.md describes: `time`
(or `hour`, distance bins, or `lat` or `lon` alone: a profile) is the x axis of a graph, `lat` × `lon` is a map
(over time: a map with a time slider),
`point` is places on a map or lines per place. `model` becomes colours, `member` thin lines and `quantile` bands.
The query decides what is shown; the view only draws it.
"""

import io
import json
import math
from functools import cache
from importlib import resources

import numpy as np

from .expr import runtime as rt
from .expr.axes import (BINS, BOOL, HOUR, LABEL, LAT, LON, MEMBER, MODEL, POINT, QUANTILE, RECORD, TIME, ExprError,
                        Type)

# Values over all fields (about 6 bytes of JSON each).
MAX_VALUES = 400_000
_MAP_MARGIN = 0.5  # degrees of basemap around a map's places and routes, at least


def check_type(typ: Type) -> None:
    """Reject answers no chart can show, from the query's type alone (before anything is downloaded)."""
    items = list(typ.fields) if typ.kind == RECORD else [(None, typ)]
    for name, t in items:
        what = name or "the answer"
        if t.kind == LABEL:
            continue
        if LAT in t.dims and LON in t.dims and MEMBER in t.dims:
            raise ExprError(f"{what}: a map can't show every member; reduce member first, e.g. .mean('member') "
                            f"of a condition (a probability) or .quantile(0.9, 'member')")
        if MEMBER in t.dims and QUANTILE in t.dims:
            raise ExprError(f"{what}: a chart shows either members or quantiles; reduce member or take the "
                            f"quantiles over member")
    if all(t.kind == LABEL for _, t in items):
        raise ExprError("nothing to draw: the answer has only labels (idxmax/idxmin); use forecast instead")


def encode_chart(value, typ: Type, ctx: rt.Context, max_values: int = MAX_VALUES) -> dict:
    """{"fields": [...], "basemap": {...}?}: one entry per answer value, with its dims, coords and flat data."""
    if isinstance(value, rt.Rec):
        types = dict(typ.fields)
        items = [(name, v, types[name]) for name, v in value.fields.items()]
    else:
        items = [(None, value, typ)]
    total = sum(v.data.size for _, v, _ in items)
    if total > max_values:
        raise ExprError(f"the chart would have {total} values (limit {max_values}): reduce a dimension (e.g. "
                        f"member with .quantile([0.1, 0.5, 0.9], 'member')), resample time, or select a smaller "
                        f"area or fewer models")
    check_type(typ)
    fields, omitted = [], []
    for name, v, t in items:
        if t.kind == LABEL:
            omitted.append(name)  # idxmax/idxmin answers are labels, not values to draw; the agent's table has them
            continue
        fields.append({
            "name": name,
            "kind": "condition" if t.kind == BOOL else "number",
            "unit": "" if t.kind == BOOL or t.unit is None else str(t.unit),
            "dims": list(v.dims),
            "coords": {d: _coord(d, v.coords[d], ctx) for d in v.dims},
            "data": _values(v.data),
        })
    out: dict = {"fields": fields, "timezone": getattr(ctx.tz, "key", None)}
    if omitted:
        out["omitted"] = omitted
    bounds = _map_bounds([v for _, v, _ in items])
    if bounds is not None:
        out["basemap"] = basemap(*bounds)
    return out


def _values(data: np.ndarray) -> list:
    """Rounded values; missing and infinite ones (x / 0) as null, as JSON has no NaN or Infinity."""
    flat = np.round(np.asarray(data, dtype=np.float64).reshape(-1), 3)
    return [x if math.isfinite(x) else None for x in flat.tolist()]


def _coord(dim: str, coord: rt.Coord, ctx: rt.Context) -> dict:
    if dim == TIME:
        out = {"minutes": [int(t) for t in coord.labels], "step": coord.flavour}
        for extra in ("distance_km", "lat", "lon"):  # routes
            if extra in coord.extra:
                out[extra] = [round(float(x), 4) for x in coord.extra[extra]]
        return out
    if dim == POINT:
        out = {"lat": [round(float(x), 4) for x in coord.extra["lat"]],
               "lon": [round(float(x), 4) for x in coord.extra["lon"]]}
        if "place" in coord.extra:
            out["labels"] = [str(x) for x in coord.extra["place"]]
        return out
    if dim in (LAT, LON, QUANTILE):
        return {"values": [float(x) for x in coord.labels]}
    if dim in (MEMBER, HOUR):
        return {"values": [int(x) for x in coord.labels]}
    if dim in (MODEL, BINS):
        return {"labels": [str(x) for x in coord.labels]}
    return {"labels": [str(x) for x in coord.labels]}


def _map_bounds(values: list[rt.Arr]) -> tuple[float, float, float, float] | None:
    """South, west, north, east of the map, or None. A grid's map is its cells, to their outer edges: the values
    fill it, and it shows no edge but where a model has no data. Maps of places and routes only get room around
    them, as there's nothing to fill."""
    boxes, lats, lons = [], [], []
    for v in values:
        if LAT in v.dims and LON in v.dims:  # one of them alone is a profile (a graph), not a map
            la = [float(x) for x in v.coords[LAT].labels]
            lo = [float(x) for x in v.coords[LON].labels]
            hla, hlo = _half_step(la), _half_step(lo)
            boxes.append((min(la) - hla, min(lo) - hlo, max(la) + hla, max(lo) + hlo))
        if POINT in v.dims:
            lats += [float(x) for x in v.coords[POINT].extra["lat"]]
            lons += [float(x) for x in v.coords[POINT].extra["lon"]]
        if TIME in v.dims and "lat" in v.coords[TIME].extra:
            lats += [float(x) for x in v.coords[TIME].extra["lat"]]
            lons += [float(x) for x in v.coords[TIME].extra["lon"]]
    if boxes:
        boxes += [(a, b, a, b) for a, b in zip(lats, lons)]  # places on a grid's map stay on it
        return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes),
                max(b[3] for b in boxes))
    if not lats:
        return None
    south, west, north, east = min(lats), min(lons), max(lats), max(lons)
    # At least half a degree, so a single place still shows where it is.
    margin = max(_MAP_MARGIN, 0.15 * max(north - south, east - west))
    return south - margin, west - margin, north + margin, east + margin


def _half_step(axis: list[float]) -> float:
    """Half a grid cell along an axis: half its smallest step, or half a tenth of a degree for a single row (as
    the view draws it)."""
    steps = [abs(b - a) for a, b in zip(axis, axis[1:]) if b != a]
    return min(steps) / 2 if steps else 0.05


# The map under maps: Natural Earth 1:10m and GeoNames places, built by scripts/make_basemap.py.
_MAP_SOURCE = "Natural Earth (public domain), GeoNames (CC BY 4.0)"
_MAP_AREAS = ("land", "lake")
_MAP_LINES = ("lake_shore", "coastline", "border", "river_major", "river", "motorway", "road")
_MAP_WIDTH = 600  # drawing units of the view's maps; the map is simplified to about half of one
_MAP_PLACES = 150  # largest places in the map; the view labels those that fit


@cache
def _basemap_data() -> dict[str, np.ndarray]:
    raw = resources.files("weather_core").joinpath("data/basemap.npz").read_bytes()
    with np.load(io.BytesIO(raw)) as npz:
        data = {name: npz[name] for name in npz.files}
    data["places.names"] = data["places.names"].tobytes().decode().split("\n")
    return data


def basemap(south: float, west: float, north: float, east: float) -> dict:
    """The map of a box (see _map_bounds): land and lakes as rings (cut at the box and between tiles, so only for
    filling), lake shores, coastlines, borders, rivers and roads as lines (each
    a list of [lat, lon, lat, lon, …] in degrees), and the largest places ([lat, lon, name], largest first).

    Detail follows the map's scale, like a web map's zoom level: features Natural Earth shows from that zoom on,
    simplified to about half a drawing unit."""
    box = (south, west, north, east)
    k = math.cos(math.radians((box[0] + box[2]) / 2))
    degrees_per_unit = max(box[2] - box[0], (box[3] - box[1]) * k) / _MAP_WIDTH
    # Web map zoom with the same scale (256-pixel tiles of 360° of longitude at zoom 0), plus one: the map has
    # little else on it, so it can show more than a web map would.
    zoom = math.log2(360 * k / (256 * degrees_per_unit)) + 1
    tolerance = 50 * degrees_per_unit  # half a unit, in hundredths of a degree
    data = _basemap_data()
    ibox = tuple(round(x * 100) for x in box)
    out = {"bounds": [round(x, 4) for x in box], "source": _MAP_SOURCE}
    for name in _MAP_AREAS + _MAP_LINES:
        out[name] = _map_layer(data, name, ibox, zoom, tolerance, area=name in _MAP_AREAS)
    s, w, n, e = ibox
    lat, lon = data["places.lat"], data["places.lon"]
    hits = np.nonzero((lat >= s) & (lat <= n) & (lon >= w) & (lon <= e))[0][:_MAP_PLACES]
    names = data["places.names"]
    out["places"] = [[int(lat[i]) / 100, int(lon[i]) / 100, names[i]] for i in hits.tolist()]
    return out


def _map_layer(data: dict, name: str, box: tuple[int, ...], zoom: float, tolerance: float, area: bool) -> list:
    s, w, n, e = box
    start, fbox = data[f"{name}.start"], data[f"{name}.box"].reshape(-1, 4)
    # Features shown at this zoom, overlapping the box and at least about a unit across.
    size = np.maximum(fbox[:, 2] - fbox[:, 0], fbox[:, 3] - fbox[:, 1])
    features = np.nonzero((data[f"{name}.zoom"] <= round(zoom * 10)) & (fbox[:, 0] <= n) & (fbox[:, 2] >= s)
                          & (fbox[:, 1] <= e) & (fbox[:, 3] >= w) & (size >= tolerance))[0]
    if not len(features):
        return []
    # Their points, with the feature each belongs to, simplified to the tolerance.
    lengths = start[features + 1] - start[features]
    first = np.cumsum(lengths) - lengths
    points = np.arange(int(lengths.sum())) + np.repeat(start[features] - first, lengths)
    owner = np.repeat(np.arange(len(features)), lengths)
    keep = data[f"{name}.sig"][points] >= tolerance
    points, owner = points[keep], owner[keep]
    lat, lon = data[f"{name}.lat"][points], data[f"{name}.lon"][points]
    breaks = np.nonzero(owner[1:] != owner[:-1])[0] + 1
    if area:
        out = []
        for a, b in zip(np.r_[0, breaks], np.r_[breaks, len(points)]):
            ring = np.column_stack((lat[a:b], lon[a:b])).astype(np.float64)
            lo, hi = ring.min(axis=0), ring.max(axis=0)
            if lo[0] < s or hi[0] > n or lo[1] < w or hi[1] > e:
                ring = _clip_ring(ring, box)
            if len(ring) >= 3:
                out.append((np.round(ring.reshape(-1)) / 100).tolist())
        return out
    # Lines: the segments that may cross the box (their bounding box overlaps it), so lines reach its edge.
    segment = ((owner[1:] == owner[:-1]) & (np.minimum(lat[1:], lat[:-1]) <= n) & (np.maximum(lat[1:], lat[:-1]) >= s)
               & (np.minimum(lon[1:], lon[:-1]) <= e) & (np.maximum(lon[1:], lon[:-1]) >= w))
    flat = (np.column_stack((lat, lon)).reshape(-1) / 100).tolist()
    cuts = np.nonzero(~segment)[0] + 1
    out = []
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(points)]):
        if b - a > 1:
            out.append(flat[2 * a:2 * b])
    return out


def _clip_ring(ring: np.ndarray, box: tuple[int, ...]) -> np.ndarray:
    """Sutherland-Hodgman: the part of a ring (rows of lat, lon) inside the box, which may run along its edge."""
    s, w, n, e = box
    for axis, value, sign in ((0, s, 1), (0, n, -1), (1, w, 1), (1, e, -1)):
        if not len(ring):
            break
        keep = (ring[:, axis] - value) * sign >= 0
        if keep.all():
            continue
        prev, keep_prev = np.roll(ring, 1, axis=0), np.roll(keep, 1)
        cross = keep != keep_prev
        delta = np.where(cross, ring[:, axis] - prev[:, axis], 1.0)
        crossing = prev + (ring - prev) * ((value - prev[:, axis]) / delta)[:, None]
        crossing[:, axis] = value
        # Per point: where the edge crosses into or out of the box, then the point if it's inside.
        counts = cross.astype(np.int64) + keep
        at = np.cumsum(counts) - counts
        out = np.empty((int(counts.sum()), 2))
        out[at[cross]] = crossing[cross]
        out[(at + cross)[keep]] = ring[keep]
        ring = out
    return ring
