"""Evaluating expressions: labelled arrays, alignment, reductions, and result encoding.

NaN means "missing" throughout: a variable a model doesn't provide, members beyond a model's member count (the
member dimension is padded to the largest model), hours a model doesn't reach, points outside its domain.
Comparisons with a missing value give missing (not false), `&`/`|` follow SQL's three-valued logic, and
reductions skip missing values (an all-missing slice gives missing). Conditions are stored as 1.0/0.0/NaN.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, tzinfo

import numpy as np

from .axes import BOOL, HOUR, LABEL, LAT, LON, MEMBER, MODEL, POINT, QUANTILE, TIME, ExprError, Type, Warnings, ordered


@dataclass
class Coord:
    """Labels along one dimension, plus auxiliary coordinates of the same length (time: dt; points: lat/lon)."""
    labels: np.ndarray  # object array: model names, member numbers, UTC minutes, point keys, hours, bins, levels
    extra: dict[str, np.ndarray] = field(default_factory=dict)
    flavour: str = ""  # time: "1h", "3h", "1D" or "route"

    def take(self, index) -> "Coord":
        return Coord(self.labels[index], {k: v[index] for k, v in self.extra.items()}, self.flavour)

    def same(self, other: "Coord") -> bool:
        return self is other or (len(self.labels) == len(other.labels) and bool(np.all(self.labels == other.labels)))

    def __len__(self) -> int:
        return len(self.labels)


@dataclass
class Arr:
    data: np.ndarray  # float64, one axis per dim
    dims: tuple[str, ...]
    coords: dict[str, Coord]
    label_coord: Coord | None = None  # LABEL arrays: data are indices into these labels

    def axis(self, dim: str) -> int:
        return self.dims.index(dim)


@dataclass
class Rec:
    fields: dict[str, Arr]


def labels(values) -> np.ndarray:
    values = list(values)
    out = np.empty(len(values), dtype=object)
    out[:] = values
    return out


class Context:
    def __init__(self, tz: tzinfo, max_bytes: int, deadline: float, clock=time.monotonic):
        self.tz = tz
        self.max_bytes = max_bytes
        self.deadline = deadline
        self.clock = clock
        self.used = 0
        self.memo: dict[int, object] = {}
        self.locations: dict[object, "LocationData"] = {}
        self.warnings = Warnings()

    def charge(self, shape, copies: int = 1) -> None:
        """Account for an allocation before making it; also where long evaluations stop."""
        self.used += int(np.prod(shape, dtype=np.int64)) * 8 * copies
        if self.used > self.max_bytes:
            raise ExprError(f"evaluation needs more than {self.max_bytes // 2**20} MB; reduce dimensions earlier "
                            f"or select a smaller area, fewer hours or fewer models")
        if self.clock() > self.deadline:
            raise ExprError("evaluation took too long")


# -- alignment ------------------------------------------------------------------------------------------------------

def join(*values: Arr) -> tuple[tuple[str, ...], dict[str, Coord], dict[str, list]]:
    """Dimensions and coordinates of a combination: shared dimensions keep the labels all values have (an inner
    join, as in xarray). Also returns, per joined dimension, the labels to keep."""
    dims = ordered(d for v in values for d in v.dims)
    coords: dict[str, Coord] = {}
    index: dict[str, list] = {}
    for d in dims:
        present = [v.coords[d] for v in values if d in v.dims]
        base = present[0]
        if all(base.same(c) for c in present[1:]):
            coords[d] = base
            continue
        others = [set(c.labels.tolist()) for c in present[1:]]
        common = [x for x in base.labels.tolist() if all(x in o for o in others)]
        if not common:
            raise ExprError(f"these values share no {d} labels (different time ranges, models or places?)")
        index[d] = common
        position = {x: i for i, x in enumerate(base.labels.tolist())}
        coords[d] = base.take([position[x] for x in common])
    return dims, coords, index


def align(ctx: Context, *values: Arr) -> tuple[tuple[str, ...], dict[str, Coord], list[np.ndarray]]:
    """Broadcast arrays against each other by dimension name and label (see join)."""
    dims, coords, index = join(*values)
    arrays = []
    for v in values:
        data = v.data
        for d in v.dims:
            if d in index:
                position = {x: i for i, x in enumerate(v.coords[d].labels.tolist())}
                data = np.take(data, [position[x] for x in index[d]], axis=v.axis(d))
        shape = [data.shape[v.dims.index(d)] if d in v.dims else 1 for d in dims]
        order = [v.dims.index(d) for d in dims if d in v.dims]
        arrays.append(np.transpose(data, order).reshape(shape))
    ctx.charge(np.broadcast_shapes(*(a.shape for a in arrays)) if arrays else ())
    return dims, coords, arrays


def elementwise(ctx: Context, fn, *values: Arr) -> Arr:
    dims, coords, arrays = align(ctx, *values)
    with np.errstate(all="ignore"):
        out = np.asarray(fn(*arrays), dtype=np.float64)
    shape = tuple(len(coords[d]) for d in dims)
    if out.shape != shape:
        out = np.broadcast_to(out, shape).copy()
    return Arr(out, dims, coords)


def constant(value: float) -> Arr:
    return Arr(np.array(value, dtype=np.float64), (), {})


# -- three-valued logic ---------------------------------------------------------------------------------------------

def missing(*arrays: np.ndarray) -> np.ndarray:
    mask = np.isnan(arrays[0])
    for a in arrays[1:]:
        mask = mask | np.isnan(a)
    return mask


def compare(op):
    return lambda a, b: np.where(missing(a, b), np.nan, op(a, b).astype(np.float64))


def logic_and(a, b):
    return np.where((a == 0) | (b == 0), 0.0, np.where(missing(a, b), np.nan, 1.0))


def logic_or(a, b):
    return np.where((a == 1) | (b == 1), 1.0, np.where(missing(a, b), np.nan, 0.0))


def logic_not(a):
    return 1.0 - a


def where(cond, a, b):
    return np.where(cond == 1, a, np.where(cond == 0, b, np.nan))


# -- reductions -----------------------------------------------------------------------------------------------------

def reduce(ctx: Context, a: Arr, dims: list[str], how: str) -> Arr:
    """Reduce `dims` jointly, skipping NaN. sum and mean weight samples along time by their duration (dt hours), so
    a sum over time is the time integral in hours."""
    keep = [d for d in a.dims if d not in dims]
    ctx.charge(a.data.shape, 3)
    x = _move_last(a, keep, dims)
    weights = None
    if TIME in dims and how in ("sum", "mean"):
        dt = a.coords[TIME].extra["dt"]
        w = np.ones([a.data.shape[a.axis(d)] for d in dims]) * dt.reshape([-1 if d == TIME else 1 for d in dims])
        weights = w.reshape(-1)
    return Arr(_reduce_last(x, how, weights), tuple(keep), {d: a.coords[d] for d in keep})


def _move_last(a: Arr, keep: list[str], dims: list[str]) -> np.ndarray:
    x = np.transpose(a.data, [a.axis(d) for d in keep + list(dims)])
    # The reduced size explicitly: -1 can't be inferred when an axis is empty.
    reduced = int(np.prod([a.data.shape[a.axis(d)] for d in dims], dtype=np.int64))
    return x.reshape([a.data.shape[a.axis(d)] for d in keep] + [reduced])


def _reduce_last(x: np.ndarray, how: str, weights: np.ndarray | None = None) -> np.ndarray:
    if x.shape[-1] == 0:
        return np.full(x.shape[:-1], np.nan)
    valid = ~np.isnan(x)
    n = valid.sum(axis=-1)
    with np.errstate(all="ignore"):
        if how in ("sum", "mean"):
            w = np.ones(x.shape[-1]) if weights is None else weights
            total = np.where(valid, x * w, 0.0).sum(axis=-1)
            if how == "sum":
                return np.where(n > 0, total, np.nan)
            covered = np.where(valid, w, 0.0).sum(axis=-1)
            return np.where(covered > 0, total / np.where(covered > 0, covered, 1), np.nan)
        if how in ("max", "any"):
            return np.fmax.reduce(x, axis=-1)
        if how in ("min", "all"):
            return np.fmin.reduce(x, axis=-1)
        if how == "count":
            return n.astype(np.float64)
        if how == "std":
            mean = np.where(valid, x, 0.0).sum(axis=-1) / np.where(n > 0, n, 1)
            var = np.where(valid, (x - mean[..., None]) ** 2, 0.0).sum(axis=-1) / np.where(n > 0, n, 1)
            return np.where(n > 0, np.sqrt(var), np.nan)
        if how == "median":
            return quantiles_last(x, [0.5])[..., 0]
    raise AssertionError(how)


def quantiles_last(x: np.ndarray, levels: list[float]) -> np.ndarray:
    """Linear-interpolated quantiles over the last axis, skipping NaN; levels become the new last axis.

    Sort-based: np.nanquantile falls back to a per-slice loop that takes seconds on padded member axes.
    """
    if x.shape[-1] == 0:  # e.g. after isel(member=[])
        return np.full(x.shape[:-1] + (len(levels),), np.nan)
    s = np.sort(x, axis=-1)  # NaN sorts last
    n = (~np.isnan(x)).sum(axis=-1)
    top = np.maximum(n - 1, 0)
    out = []
    for q in levels:
        pos = q * top
        lo = np.floor(pos).astype(np.int64)
        hi = np.minimum(lo + 1, top)
        a = np.take_along_axis(s, lo[..., None], axis=-1)[..., 0]
        b = np.take_along_axis(s, hi[..., None], axis=-1)[..., 0]
        with np.errstate(invalid="ignore"):
            out.append(np.where(n > 0, a + (b - a) * (pos - lo), np.nan))
    return np.stack(out, axis=-1)


def quantile(ctx: Context, a: Arr, levels: list[float], dims: list[str], as_dim: bool) -> Arr:
    keep = [d for d in a.dims if d not in dims]
    ctx.charge(a.data.shape, 2 + len(levels))
    out = quantiles_last(_move_last(a, keep, dims), levels)
    coords = {d: a.coords[d] for d in keep}
    if not as_dim:
        return Arr(out[..., 0], tuple(keep), coords)
    coords[QUANTILE] = Coord(labels(levels))
    return Arr(out, tuple(keep) + (QUANTILE,), coords)


def rolling(ctx: Context, a: Arr, n: int, how: str) -> Arr:
    """Windows of n samples along time, labelled at their last sample (as in xarray); windows that are incomplete
    or contain a missing value are missing."""
    k = a.axis(TIME)
    ctx.charge(a.data.shape, n + 1)
    out = np.full(a.data.shape, np.nan)
    if a.data.shape[k] >= n:
        x = np.moveaxis(a.data, k, -1)
        windows = np.lib.stride_tricks.sliding_window_view(x, n, axis=-1)
        if how in ("sum", "mean"):
            wdt = np.lib.stride_tricks.sliding_window_view(a.coords[TIME].extra["dt"], n)
            total = (windows * wdt).sum(axis=-1)
            values = total if how == "sum" else total / wdt.sum(axis=-1)
        else:
            values = windows.max(axis=-1) if how == "max" else windows.min(axis=-1)
        np.moveaxis(out, k, -1)[..., n - 1:] = values
    return Arr(out, a.dims, a.coords)


def segments(ctx: Context, a: Arr, keys: np.ndarray, new: Coord, how: str, new_dim: str) -> Arr:
    """Reduce samples along time that share a key (keys[i] = index into `new`, -1 = dropped) into `new_dim`."""
    keep = [d for d in a.dims if d != TIME]
    x = _move_last(a, keep, [TIME])
    weights = a.coords[TIME].extra["dt"] if how in ("sum", "mean") else None
    ctx.charge(a.data.shape, 3)
    parts = [_reduce_last(x[..., keys == g], how, None if weights is None else weights[keys == g])
             for g in range(len(new))]
    out = np.stack(parts, axis=-1) if parts else np.zeros(x.shape[:-1] + (0,))
    coords = {d: a.coords[d] for d in keep}
    coords[new_dim] = new
    return transpose(Arr(out, tuple(keep) + (new_dim,), coords))


def transpose(a: Arr) -> Arr:
    dims = ordered(a.dims)
    if dims == a.dims:
        return a
    return Arr(np.transpose(a.data, [a.axis(d) for d in dims]), dims, a.coords, a.label_coord)


def resample(ctx: Context, a: Arr, freq: str, hours: int, how: str) -> Arr:
    """Buckets of `hours` local wall-clock hours (24 = local days), labelled by their first sample."""
    order = sorted(range(len(a.coords[TIME])), key=lambda i: a.coords[TIME].labels[i])
    if order != list(range(len(order))):  # reordered by sortby, top or isel
        a = take(a, TIME, order)
    times = a.coords[TIME]
    bucket = np.array([_wall_minutes(t, ctx.tz) // (hours * 60) for t in times.labels], dtype=np.int64)
    change = np.concatenate([[True], bucket[1:] != bucket[:-1]]) if len(bucket) else np.zeros(0, dtype=bool)
    keys = np.cumsum(change) - 1
    starts = np.flatnonzero(change)
    dt = np.array([times.extra["dt"][keys == g].sum() for g in range(len(starts))])
    return segments(ctx, a, keys, Coord(times.labels[starts], {"dt": dt}, freq), how, TIME)


def groupby_hour(ctx: Context, a: Arr, how: str) -> Arr:
    hours = [int(_wall_minutes(t, ctx.tz) // 60 % 24) for t in a.coords[TIME].labels]
    present = sorted(set(hours))
    keys = np.array([present.index(h) for h in hours], dtype=np.int64)
    return segments(ctx, a, keys, Coord(labels(present)), how, HOUR)


def groupby_bins(ctx: Context, a: Arr, edges: list[float], how: str, dim: str) -> Arr:
    """Route samples by distance into (a, b] bins (the start, at distance 0, joins the first bin)."""
    distance = a.coords[TIME].extra["distance_km"]
    keys = np.full(len(distance), -1, dtype=np.int64)
    for g in range(len(edges) - 1):
        keys[(distance > edges[g]) & (distance <= edges[g + 1])] = g
    keys[distance == edges[0]] = 0
    names = labels(f"({edges[g]:g}, {edges[g + 1]:g}]" for g in range(len(edges) - 1))
    return segments(ctx, a, keys, Coord(names), how, dim)


# -- selection and ranking ------------------------------------------------------------------------------------------

def take(a: Arr, dim: str, index, drop: bool = False) -> Arr:
    k = a.axis(dim)
    if drop:
        dims = tuple(d for d in a.dims if d != dim)
        return Arr(np.take(a.data, index, axis=k), dims, {d: a.coords[d] for d in dims}, a.label_coord)
    coords = dict(a.coords)
    coords[dim] = a.coords[dim].take(index)
    return Arr(np.take(a.data, index, axis=k), a.dims, coords, a.label_coord)


def rank(values: np.ndarray, descending: bool) -> np.ndarray:
    """Indices ordering a 1-D array, NaN last, ties in their original order."""
    filled = np.where(np.isnan(values), 0.0, values)
    return np.lexsort((-filled if descending else filled, np.isnan(values)))


def idx(ctx: Context, a: Arr, dim: str, largest: bool) -> Arr:
    """Label (as an index into the dimension's labels) of the largest/smallest value along `dim`."""
    ctx.charge(a.data.shape)
    x = np.moveaxis(a.data, a.axis(dim), -1)
    keep = tuple(d for d in a.dims if d != dim)
    if x.shape[-1] == 0:
        return Arr(np.full(x.shape[:-1], np.nan), keep, {d: a.coords[d] for d in keep}, a.coords[dim])
    filled = np.where(np.isnan(x), -np.inf if largest else np.inf, x)
    index = (filled.argmax(axis=-1) if largest else filled.argmin(axis=-1)).astype(np.float64)
    keep = tuple(d for d in a.dims if d != dim)
    return Arr(np.where(np.isnan(x).all(axis=-1), np.nan, index), keep, {d: a.coords[d] for d in keep},
               a.coords[dim])


# -- time labels (UTC epoch minutes) ---------------------------------------------------------------------------------

def _local(utc_minutes, tz: tzinfo) -> datetime:
    return datetime.fromtimestamp(float(utc_minutes) * 60, timezone.utc).astimezone(tz)


def _wall_minutes(utc_minutes, tz: tzinfo) -> int:
    return int(_local(utc_minutes, tz).replace(tzinfo=timezone.utc).timestamp() // 60)


def format_time(utc_minutes, tz: tzinfo, flavour: str = "") -> str:
    t = _local(utc_minutes, tz)
    return t.strftime("%Y-%m-%d") if flavour.endswith("D") else t.strftime("%Y-%m-%dT%H:%M")


# -- results --------------------------------------------------------------------------------------------------------

def encode(value, typ: Type, ctx: Context, max_rows: int):
    """JSON for a result: scalars as numbers/booleans, arrays as tables with columns for their dimensions."""
    if isinstance(value, Rec):
        types = dict(typ.fields)
        shapes = {tuple(d for d in v.dims if d != QUANTILE) for v in value.fields.values()}
        if len(shapes) == 1 and next(iter(shapes)):
            return _table([(k, v, types[k]) for k, v in value.fields.items()], ctx, max_rows)
        return {k: _encode_one(v, types[k], ctx, max_rows) for k, v in value.fields.items()}
    return _encode_one(value, typ, ctx, max_rows)


def _encode_one(value: Arr, typ: Type, ctx: Context, max_rows: int):
    if not [d for d in value.dims if d != QUANTILE]:
        if QUANTILE not in value.dims:
            return _scalar(value.data, typ, value.label_coord, ctx)
        return {_level(q): _scalar(value.data[i], typ, value.label_coord, ctx)
                for i, q in enumerate(value.coords[QUANTILE].labels)}
    return _table([(None, value, typ)], ctx, max_rows)


def _table(fields: list, ctx: Context, max_rows: int) -> dict:
    dims = tuple(d for d in fields[0][1].dims if d != QUANTILE)
    # Fields may differ in labels after selections: keep the rows all of them have.
    _, coords, _ = join(*[_without_quantile(v) for _, v, _ in fields])
    sizes = [len(coords[d]) for d in dims]
    n_rows = int(np.prod(sizes))
    if n_rows > max_rows:
        raise ExprError(f"the result has {n_rows} rows (limit {max_rows}): reduce a dimension "
                        f"({', '.join(dims)}), resample time, or rank with top()")
    label_columns = [(i, name, values) for i, d in enumerate(dims) for name, values in _labels(d, coords[d], ctx)]
    value_columns = []
    for name, v, typ in fields:
        v = _reindex(v, dims, coords)
        if QUANTILE in v.dims:
            data = np.moveaxis(v.data, v.axis(QUANTILE), 0)
            for i, q in enumerate(v.coords[QUANTILE].labels):
                value_columns.append((f"{name}_{_level(q)}" if name else _level(q), data[i], typ, v.label_coord))
        else:
            value_columns.append((name or "value", v.data, typ, v.label_coord))
    rows = []
    for index in np.ndindex(*sizes):
        row = [values[index[i]] for i, _, values in label_columns]
        row += [_scalar(data[index], typ, lc, ctx) for _, data, typ, lc in value_columns]
        rows.append(row)
    return {"columns": [c[1] for c in label_columns] + [c[0] for c in value_columns], "rows": rows}


def _without_quantile(v: Arr) -> Arr:
    return take(v, QUANTILE, 0, drop=True) if QUANTILE in v.dims else v


def _reindex(v: Arr, dims: tuple[str, ...], coords: dict[str, Coord]) -> Arr:
    for d in dims:
        if not v.coords[d].same(coords[d]):
            position = {x: i for i, x in enumerate(v.coords[d].labels.tolist())}
            v = take(v, d, [position[x] for x in coords[d].labels.tolist()])
    return v


def _labels(dim: str, coord: Coord, ctx: Context) -> list[tuple[str, list]]:
    if dim == TIME:
        out = [("time", [format_time(t, ctx.tz, coord.flavour) for t in coord.labels])]
        if "distance_km" in coord.extra:
            out.append(("distance_km", [round(float(x), 1) for x in coord.extra["distance_km"]]))
        return out
    if dim == POINT:
        if "place" in coord.extra:
            return [("place", list(coord.extra["place"]))]
        return [("lat", [round(float(x), 3) for x in coord.extra["lat"]]),
                ("lon", [round(float(x), 3) for x in coord.extra["lon"]])]
    if dim in (MEMBER, HOUR):
        return [(dim, [int(x) for x in coord.labels])]
    if dim in (LAT, LON):
        return [(dim, [round(float(x), 3) for x in coord.labels])]
    if dim == QUANTILE:
        return [(dim, [float(x) for x in coord.labels])]
    return [(dim, list(coord.labels))]


def _level(q) -> str:
    return f"p{float(q) * 100:g}"


def _scalar(x, typ: Type, label_coord: Coord | None, ctx: Context):
    v = float(x)
    if np.isnan(v):
        return None
    if typ.kind == LABEL and label_coord is not None:
        i = int(v)
        if label_coord.flavour:
            return format_time(label_coord.labels[i], ctx.tz, label_coord.flavour)
        if "place" in label_coord.extra:
            return label_coord.extra["place"][i]
        if "lat" in label_coord.extra:
            return [round(float(label_coord.extra["lat"][i]), 3), round(float(label_coord.extra["lon"][i]), 3)]
        label = label_coord.labels[i]
        return label.item() if hasattr(label, "item") else label
    if typ.kind == BOOL:
        return v == 1.0
    return round(v, 4)


# -- loading --------------------------------------------------------------------------------------------------------

@dataclass
class LocationData:
    """One location's variables over (model, member, time, *space), NaN-padded to the largest member count."""
    arrays: dict[str, np.ndarray]
    coords: dict[str, Coord]
    space: tuple[str, ...]  # (), (POINT,) or (LAT, LON)
    lat: float  # single points: the location (lat/lon coordinates)
    lon: float


def location_data(per_model: list, variables: list[str], time: Coord, space: dict[str, Coord],
                  lat: float = np.nan, lon: float = np.nan) -> LocationData:
    """per_model: (model name, SourceSamples) of every model that answered. Sources sample a flat list of points;
    for a grid it is lat-major and becomes the lat and lon dimensions."""
    members = max(s.info.members for _, s in per_model)
    shape = tuple(len(c) for c in space.values())
    arrays = {}
    for name in variables:
        out = np.full((len(per_model), members, len(time)) + shape, np.nan)
        for i, (_, s) in enumerate(per_model):
            values = s.samples.variables.get(name)
            if values is not None:
                out[i, :values.shape[0]] = values.reshape(values.shape[:2] + shape)
        arrays[name] = out
    coords = {MODEL: Coord(labels(m for m, _ in per_model)), MEMBER: Coord(labels(range(members))), TIME: time,
              **space}
    return LocationData(arrays, coords, tuple(space), lat, lon)


def variable(data: LocationData, name: str) -> Arr:
    dims = (MODEL, MEMBER, TIME) + data.space
    return Arr(data.arrays[name], dims, {d: data.coords[d] for d in dims})


def coordinate(data: LocationData, name: str, tz: tzinfo) -> Arr:
    times = data.coords[TIME]
    if name in ("hour", "dayofweek"):
        local = [_local(t, tz) for t in times.labels]
        values = [t.hour + t.minute / 60 if name == "hour" else float(t.weekday()) for t in local]
        return Arr(np.array(values, dtype=np.float64), (TIME,), {TIME: times})
    if name in times.extra and name != "dt":  # routes: lat, lon, distance_km per sample
        return Arr(np.asarray(times.extra[name], dtype=np.float64), (TIME,), {TIME: times})
    own = LAT if name == "lat" else LON
    if own in data.space:
        return Arr(np.asarray(data.coords[own].labels, dtype=np.float64), (own,), {own: data.coords[own]})
    if POINT in data.space:
        point = data.coords[POINT]
        return Arr(np.asarray(point.extra[name], dtype=np.float64), (POINT,), {POINT: point})
    return constant(data.lat if name == "lat" else data.lon)
