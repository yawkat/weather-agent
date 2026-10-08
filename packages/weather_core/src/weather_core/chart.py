"""Answers as charts (the show_forecast tool): every value of a result, as dense arrays the client's view draws.

The view picks the form from the dimensions left in each field, as docs/query-language.md describes: `time`
(or `hour`, distance bins) is the x axis of a graph, `lat` × `lon` is a map (over time: a map with a time slider),
`point` is places on a map or lines per place. `model` becomes colours, `member` thin lines and `quantile` bands.
The query decides what is shown; the view only draws it.
"""

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
_MAP_MARGIN = 0.5  # degrees of basemap around a map's points, at least


def check_type(typ: Type) -> None:
    """Reject answers no chart can show, from the query's type alone (before anything is downloaded)."""
    items = list(typ.fields) if typ.kind == RECORD else [(None, typ)]
    for name, t in items:
        what = name or "the answer"
        if t.kind == LABEL:
            continue
        if LAT in t.dims and MEMBER in t.dims:
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
    fields = []
    for name, v, t in items:
        if t.kind == LABEL:
            continue  # idxmax/idxmin answers are labels, not values to draw; the table shows them
        fields.append({
            "name": name,
            "kind": "condition" if t.kind == BOOL else "number",
            "unit": "" if t.kind == BOOL or t.unit is None else str(t.unit),
            "dims": list(v.dims),
            "coords": {d: _coord(d, v.coords[d], ctx) for d in v.dims},
            "data": _values(v.data),
        })
    out: dict = {"fields": fields, "timezone": getattr(ctx.tz, "key", None)}
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
    """South, west, north, east around every mapped location (grids, points, routes), or None."""
    lats, lons = [], []
    for v in values:
        if LAT in v.dims:
            lats += [float(x) for x in v.coords[LAT].labels]
            lons += [float(x) for x in v.coords[LON].labels]
        if POINT in v.dims:
            lats += [float(x) for x in v.coords[POINT].extra["lat"]]
            lons += [float(x) for x in v.coords[POINT].extra["lon"]]
        if TIME in v.dims and "lat" in v.coords[TIME].extra:
            lats += [float(x) for x in v.coords[TIME].extra["lat"]]
            lons += [float(x) for x in v.coords[TIME].extra["lon"]]
    if not lats:
        return None
    return min(lats), min(lons), max(lats), max(lons)


@cache
def _basemap_data() -> dict:
    return json.loads(resources.files("weather_core").joinpath("data/basemap.json").read_text())


def basemap(south: float, west: float, north: float, east: float) -> dict:
    """Coastlines and borders (Natural Earth) around a map, as lists of [lat, lon, lat, lon, …] in degrees."""
    # At least half a degree, so a small area or a single place still shows where it is.
    margin = max(_MAP_MARGIN, 0.15 * max(north - south, east - west))
    box = (south - margin, west - margin, north + margin, east + margin)
    data = _basemap_data()
    return {
        "bounds": [round(x, 4) for x in box],
        "coastline": _clip(data["coastline"], box),
        "borders": _clip(data["borders"], box),
        "source": data["source"],
    }


def _clip(lines: list[list[int]], box: tuple[float, float, float, float]) -> list[list[float]]:
    """Parts of the lines inside the box, plus one point beyond each end so lines reach the edge."""
    s, w, n, e = (round(x * 100) for x in box)
    out = []
    for line in lines:
        lat = line[0::2]
        lon = line[1::2]
        inside = [s <= a <= n and w <= b <= e for a, b in zip(lat, lon)]
        i = 0
        while i < len(inside):
            if not inside[i]:
                i += 1
                continue
            j = i
            while j < len(inside) and inside[j]:
                j += 1
            a, b = max(i - 1, 0), min(j + 1, len(inside))
            if b - a > 1:
                out.append([x / 100 for k in range(a, b) for x in (lat[k], lon[k])])
            i = j
    return out
