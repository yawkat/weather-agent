"""Forecast queries: a subset of xarray, parsed with `ast` and interpreted (never executed as Python).

A query is optional bindings (`fc = forecast()`) and a final expression, the answer. `forecast()` is one dataset
whose variables are arrays over (model, member, time, lat, lon); a query picks a location with `.interp(lat=…,
lon=…)`, `.interp(places(…))`, `.interp(route(…))` or an area with `.sel(lat=slice(…), lon=slice(…))`, and times
and models with `.sel` too, anywhere in the expression; reductions name the dimension they remove. See
docs/query-language.md.

Locations are compile-time contexts: `x.interp(L)` compiles x's syntax tree again with L as the location of every
variable in it that has none yet, so each operation is type-checked for the actual location (an hourly axis or a
route's samples, points or a grid).

Compiling type-checks the query, works out for every location which variables, models and time range it needs
(walking back from the answer through selections, rolling windows and resampling), and estimates the memory
evaluation will need, all before anything is downloaded.
"""

import ast
import math
import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, tzinfo

import numpy as np

from ..geometry import haversine_km
from ..variables import CATALOG
from . import runtime as rt
from .axes import (BINS, BOOL, DATASET, DIMS, HOUR, HOURLY, HOURS, LABEL, LAT, LON, MEMBER, MODEL, NO_UNIT, NUM,
                   POINT, QUANTILE, RECORD, ROUTE, SPEC, TIME, ExprError, Type, Unit, Warnings, ordered)

MAX_CHARS = 120_000  # polylines are long; complexity is bounded by MAX_NODES and MAX_DEPTH
MAX_NODES = 500
MAX_DEPTH = 40
MAX_LEVELS = 20
MAX_TOP = 100
MAX_FIELDS = 30
MAX_ROLLING = 24 * 16
MAX_BINS = 200
MAX_LOCATIONS = 10
MAX_POINTS = 500  # interp(lat=("point", […]), …)
MAX_DEMAND_VISITS = 20_000

REDUCTIONS = ("sum", "mean", "median", "min", "max", "std", "any", "all", "count")
GROUP_REDUCTIONS = ("sum", "mean", "max", "min")
FUNCTIONS = ("forecast", "places", "route", "distance_from")
RESAMPLE = {"1h": 1, "2h": 2, "3h": 3, "4h": 4, "6h": 6, "8h": 8, "12h": 12, "24h": 24, "1D": 24, "D": 24}
STEP_MINUTES = {HOURLY: 60, **{freq: hours * 60 for freq, hours in RESAMPLE.items()}}
# Variables whose time integral from zero means little (as opposed to rates like precipitation).
STATE = {"t2m", "td2m", "rh", "feels_like", "wind", "wind_dir", "gust", "cloud", "cape", "headwind", "crosswind"}
# Compass directions: reducing them as numbers (mean, quantiles, …) is wrong where they straddle north.
DIRECTIONS = {"wind_dir"}
NUMPY = {"maximum": np.maximum, "minimum": np.minimum, "abs": np.abs, "sqrt": np.sqrt, "hypot": np.hypot,
         "sin": np.sin, "cos": np.cos, "arctan2": np.arctan2, "deg2rad": np.deg2rad, "rad2deg": np.rad2deg,
         "where": None}
DEGREES = Unit.parse("°")
RADIANS = Unit.parse("rad")
COORDINATES = ("lat", "lon", "distance_km")
SPECIAL = {"gpx", "np", "xr", "top", "bottom", "abs", *FUNCTIONS}
LOCATE_HINT = (".interp(lat=…, lon=…), .interp(places(…)), .interp(route(…)) or .sel(lat=slice(…), "
               "lon=slice(…))")


@dataclass
class Program:
    bindings: list[tuple[str, ast.expr]]
    result: ast.expr


@dataclass(frozen=True)
class Env:
    models: tuple[str, ...]  # configured sources, in order
    provided: dict  # model → variables it provides
    tz: tzinfo
    has_gpx: bool = False


@dataclass(frozen=True)
class Location:
    kind: str  # "point", "points" (a point dimension), "grid" (lat and lon dimensions) or "route"
    args: tuple  # canonical arguments; equal locations share one download

    def describe(self) -> str:
        if self.kind == "point":
            return f"the point ({self.args[0]}, {self.args[1]})"
        if self.kind == "points":
            names = [name for name, _, _ in self.args if name]
            return "places " + ", ".join(names) if names else f"{len(self.args)} points"
        if self.kind == "grid":
            south, west, north, east = self.args
            return f"the area lat {south}…{north}, lon {west}…{east}"
        return "the route"

    @property
    def space(self) -> tuple[str, ...]:
        """Spatial dimensions of its arrays."""
        return {"points": (POINT,), "grid": (LAT, LON)}.get(self.kind, ())


@dataclass
class Demand:
    """What one location must provide: variables, models (None = all) and time range (UTC minutes)."""
    variables: set[str] = field(default_factory=set)
    models: set[str] | None = field(default_factory=set)
    start: int | None = None  # span of every window
    end: int | None = None
    windows: set[tuple[int, int]] = field(default_factory=set)  # [start, end) of each time selection
    unbounded: bool = False


@dataclass(frozen=True)
class Need:
    time: tuple[int, int] | None = None  # UTC minutes [start, end); None = all the input has
    models: frozenset[str] | None = None


# -- parsing -----------------------------------------------------------------------------------------------------------

def parse(text: str) -> Program:
    text = text.strip()
    if not text:
        raise ExprError("query is empty")
    if len(text) > MAX_CHARS:
        raise ExprError(f"query longer than {MAX_CHARS} characters")
    try:
        tree = ast.parse(text, mode="exec")
    except SyntaxError as e:
        raise ExprError(f"syntax error at line {e.lineno}, column {e.offset}: {e.msg}") from None
    except (RecursionError, MemoryError, ValueError):
        raise ExprError("query can't be parsed (nested too deeply, or invalid characters)") from None
    _check_size(tree)
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        raise ExprError("the query must end with an expression: the answer")
    bindings: list[tuple[str, ast.expr]] = []
    for statement in tree.body[:-1]:
        if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)):
            raise _error(statement, "only `name = expression` lines may come before the answer")
        name = statement.targets[0].id
        if any(name == n for n, _ in bindings):
            raise _error(statement, f"{name} is assigned twice")
        if name in SPECIAL:
            raise _error(statement, f"{name} is a built-in name; choose another")
        bindings.append((name, statement.value))
    return Program(bindings, tree.body[-1].value)


def _check_size(tree: ast.AST) -> None:
    count = 0
    stack = [(tree, 0)]
    while stack:
        node, depth = stack.pop()
        count += 1
        if count > MAX_NODES:
            raise ExprError(f"query too complex (more than {MAX_NODES} syntax elements)")
        if depth > MAX_DEPTH:
            raise ExprError(f"query nested too deeply (more than {MAX_DEPTH} levels)")
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))


def _error(node: ast.AST, message: str) -> ExprError:
    line = getattr(node, "lineno", None)
    error = ExprError(f"line {line}, column {node.col_offset + 1}: {message}" if line else message)
    if line:
        error.span = (line, node.col_offset, node.end_lineno, node.end_col_offset)
    return error


# -- literals --------------------------------------------------------------------------------------------------------

def _number(node: ast.expr, what: str) -> float:
    sign = 1.0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        sign = -1.0 if isinstance(node.op, ast.USub) else 1.0
        node = node.operand
    if not (isinstance(node, ast.Constant) and type(node.value) in (int, float)):
        raise _error(node, f"{what} must be a number written in the query")
    try:
        value = sign * float(node.value)
    except OverflowError:
        raise _error(node, f"{what} is out of range") from None
    if not math.isfinite(value) or abs(value) > 1e12:
        raise _error(node, f"{what} is out of range")
    return value


def _integer(node: ast.expr, what: str) -> int:
    value = _number(node, what)
    if value != int(value):
        raise _error(node, f"{what} must be a whole number")
    return int(value)


def _string(node: ast.expr, what: str) -> str:
    if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
        raise _error(node, f"{what} must be a string")
    return node.value


def _strings(node: ast.expr, what: str) -> list[str]:
    items = node.elts if isinstance(node, (ast.List, ast.Tuple)) else [node]
    return [_string(n, what) for n in items]


def _pair(node: ast.expr, what: str) -> tuple[float, float]:
    if not (isinstance(node, (ast.Tuple, ast.List)) and len(node.elts) == 2):
        raise _error(node, f"{what} must be (lat, lon)")
    return _number(node.elts[0], what), _number(node.elts[1], what)


def _parse_time(node: ast.expr, tz: tzinfo) -> tuple[int, bool]:
    """UTC minutes, and whether the string was a date only."""
    text = _string(node, "a time")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise _error(node, f"invalid time {text!r}; use ISO 8601 like 2026-10-10T10:00 or 2026-10-10") from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=tz)
    date_only = len(text) <= 10
    return int(parsed.astimezone(timezone.utc).timestamp() // 60), date_only


def _local_floor(minutes: int, hours: int, tz: tzinfo) -> int:
    """Start (UTC minutes) of the local `hours`-bucket containing `minutes`."""
    local = datetime.fromtimestamp(minutes * 60, timezone.utc).astimezone(tz)
    wall = local.replace(tzinfo=None)
    floored = wall.replace(minute=0, second=0, microsecond=0, hour=wall.hour - wall.hour % hours if hours < 24 else 0)
    return int(floored.replace(tzinfo=tz).astimezone(timezone.utc).timestamp() // 60)


# -- typed nodes --------------------------------------------------------------------------------------------------------

class Node:
    type: Type
    children: tuple["Node", ...] = ()

    def eval(self, ctx: rt.Context):
        raise NotImplementedError

    def demand(self, need: Need, walk: "_DemandWalk") -> None:
        for child in self.children:
            walk.visit(child, need)

    def cost(self, size: Callable[[tuple[str, ...]], int]) -> int:
        """Bytes this node allocates (estimate); Context.charge meters the same allocations when evaluating."""
        return size(self.type.dims) * 8 if self.type.kind in (NUM, BOOL, LABEL) else 0


class Const(Node):
    def __init__(self, value: float):
        self.value = value
        self.type = Type(NUM)

    def eval(self, ctx):
        return rt.constant(self.value)

    def cost(self, size):
        return 0


class Dataset(Node):
    """forecast(), possibly located, with selections still to apply; only its variables and coordinates are
    values."""

    def __init__(self, location: Location | None, selections: tuple = ()):
        self.location = location
        self.selections = selections  # ((dim, value, drop, ast node), …), applied to every variable
        self.type = Type(DATASET)


class Spec(Node):
    """places(…) or route(…): a location, only usable as the argument of .interp."""

    def __init__(self, location: Location):
        self.location = location
        self.type = Type(SPEC)


class Leaf(Node):
    """A variable or dataset coordinate; without a location (yet) while it isn't inside .interp/.sel(lat, lon)."""

    def __init__(self, location: Location | None, name: str, typ: Type, variable: bool):
        self.location = location
        self.name = name
        self.variable = variable
        self.type = typ

    def eval(self, ctx):
        data = ctx.locations[self.location]
        return rt.variable(data, self.name) if self.variable else rt.coordinate(data, self.name, ctx.tz)

    def demand(self, need, walk):
        walk.record(self, need)

    def cost(self, size):
        return 0  # variables are charged when loaded


class CoordOf(Node):
    """A coordinate of a computed array (x.time.dt.hour, x.lat, x.distance_km)."""

    def __init__(self, x: Node, name: str, typ: Type):
        self.x = x
        self.name = name
        self.children = (x,)
        self.type = typ

    def eval(self, ctx):
        a = self.x.eval(ctx)
        if self.name in ("hour", "dayofweek"):
            times = a.coords[TIME]
            local = [rt._local(t, ctx.tz) for t in times.labels]
            values = [t.hour + t.minute / 60 if self.name == "hour" else float(t.weekday()) for t in local]
            return rt.Arr(np.array(values, dtype=np.float64), (TIME,), {TIME: times})
        dim = self.type.dims[0]
        coord = a.coords[dim]
        values = coord.labels if dim in (LAT, LON) else coord.extra[self.name]
        return rt.Arr(np.asarray(values, dtype=np.float64), (dim,), {dim: coord})


class Binding(Node):
    def __init__(self, name: str, node: Node):
        self.name = name
        self.node = node
        self.children = (node,)
        self.type = node.type

    def eval(self, ctx):
        if id(self) not in ctx.memo:
            ctx.memo[id(self)] = self.node.eval(ctx)
        return ctx.memo[id(self)]

    def cost(self, size):
        return 0


class Elementwise(Node):
    def __init__(self, fn, children: list[Node], typ: Type):
        self.fn = fn
        self.children = tuple(children)
        self.type = typ

    def eval(self, ctx):
        return rt.elementwise(ctx, self.fn, *(c.eval(ctx) for c in self.children))


class Reduce(Node):
    def __init__(self, x: Node, dims: list[str], how: str, typ: Type):
        self.x = x
        self.dims = dims
        self.how = how
        self.children = (x,)
        self.type = typ

    def eval(self, ctx):
        return rt.reduce(ctx, self.x.eval(ctx), self.dims, self.how)

    def demand(self, need, walk):
        walk.visit(self.x, _release(need, self.dims))

    def cost(self, size):
        return size(self.x.type.dims) * 8 * 3


class Quantile(Node):
    def __init__(self, x: Node, levels: list[float], dims: list[str], as_dim: bool, typ: Type):
        self.x, self.levels, self.dims, self.as_dim = x, levels, dims, as_dim
        self.children = (x,)
        self.type = typ

    def eval(self, ctx):
        return rt.quantile(ctx, self.x.eval(ctx), self.levels, self.dims, self.as_dim)

    def demand(self, need, walk):
        walk.visit(self.x, _release(need, self.dims))

    def cost(self, size):
        return size(self.x.type.dims) * 8 * (2 + len(self.levels))


class Rolling(Node):
    def __init__(self, x: Node, n: int, how: str, typ: Type):
        self.x, self.n, self.how = x, n, how
        self.children = (x,)
        self.type = typ

    def eval(self, ctx):
        return rt.rolling(ctx, self.x.eval(ctx), self.n, self.how)

    def demand(self, need, walk):
        if need.time is not None:
            extend = (self.n - 1) * STEP_MINUTES[self.x.type.time]
            need = Need((need.time[0] - extend, need.time[1]), need.models)
        walk.visit(self.x, need)

    def cost(self, size):
        return size(self.x.type.dims) * 8 * (self.n + 1)


class Group(Node):
    """resample(time=…), groupby("time.hour") and groupby_bins("distance_km", …), each with an aggregation."""

    def __init__(self, x: Node, kind: str, arg, how: str, typ: Type, tz: tzinfo):
        self.x, self.kind, self.arg, self.how, self.tz = x, kind, arg, how, tz
        self.children = (x,)
        self.type = typ

    def eval(self, ctx):
        a = self.x.eval(ctx)
        if self.kind == "resample":
            return rt.resample(ctx, a, self.arg, RESAMPLE[self.arg], self.how)
        if self.kind == "hour":
            return rt.groupby_hour(ctx, a, self.how)
        return rt.groupby_bins(ctx, a, self.arg, self.how, BINS)

    def demand(self, need, walk):
        if self.kind == "resample" and need.time is not None:
            hours = RESAMPLE[self.arg]
            start = _local_floor(need.time[0], hours, self.tz)
            end = _local_floor(need.time[1] - 1, hours, self.tz) + hours * 60
            need = Need((start, end), need.models)
        elif self.kind != "resample":
            need = _release(need, [TIME])
        walk.visit(self.x, need)

    def cost(self, size):
        return size(self.x.type.dims) * 8 * 3


class Select(Node):
    """sel/isel along one dimension. value: label(s), a (start, end) time range in UTC minutes, or positions."""

    def __init__(self, x: Node, dim: str, value, mode: str, drop: bool, typ: Type, where: ast.AST):
        self.x, self.dim, self.value, self.mode, self.drop = x, dim, value, mode, drop
        self.children = (x,)
        self.type = typ
        self.where = where

    def eval(self, ctx):
        a = self.x.eval(ctx)
        coord = a.coords[self.dim]
        if self.mode == "between":
            lo, hi = self.value
            index = [i for i, x in enumerate(coord.labels) if lo <= x <= hi]
            if not index:
                raise _error(self.where, f"this {self.dim} range selects nothing")
            return rt.take(a, self.dim, index)
        if self.mode == "range":
            start, end = self.value
            index = [i for i, t in enumerate(coord.labels) if start <= t < end]
            if not index:
                raise _error(self.where, "this time range selects no samples")
            return rt.take(a, self.dim, index)
        if self.mode == "positions":
            n = len(coord)
            try:
                if isinstance(self.value, tuple):
                    index = list(range(n))[slice(*self.value)]
                else:
                    index = [range(n)[i] for i in self.value]
            except IndexError:
                raise _error(self.where, f"isel: position out of range ({self.dim} has {n} entries)") from None
            return rt.take(a, self.dim, index[0] if self.drop else index, self.drop)
        keys = coord.labels.tolist()
        present = [v for v in self.value if v in keys]
        absent = [v for v in self.value if v not in keys]
        # Models that didn't answer are reported in unavailable_sources; a list just goes without them.
        if absent and not (self.dim == MODEL and present):
            if self.dim == MODEL:
                raise _error(self.where, f"sel: model {absent[0]!r} has no data here (see unavailable_sources)")
            shown = rt.format_time(absent[0], ctx.tz) if self.dim == TIME else absent[0]
            raise _error(self.where, f"sel: no {self.dim} {shown!r} here")
        index = [keys.index(v) for v in present]
        return rt.take(a, self.dim, index[0] if self.drop else index, self.drop)

    def demand(self, need, walk):
        if self.dim == TIME and self.mode == "range":
            start, end = self.value
            if need.time is not None:
                start, end = max(start, need.time[0]), min(end, need.time[1])
            need = Need((start, max(start, end)), need.models)
        elif self.dim == TIME and self.mode == "labels":
            need = Need((self.value[0], self.value[0] + 60), need.models)
        elif self.dim == MODEL and self.mode == "labels":
            models = frozenset(self.value) if need.models is None else need.models & frozenset(self.value)
            need = Need(need.time, models)
        elif self.mode == "positions":
            need = _release(need, [self.dim])
        walk.visit(self.x, need)


class Idx(Node):
    def __init__(self, x: Node, dim: str, largest: bool, typ: Type):
        self.x, self.dim, self.largest = x, dim, largest
        self.children = (x,)
        self.type = typ

    def eval(self, ctx):
        return rt.idx(ctx, self.x.eval(ctx), self.dim, self.largest)

    def demand(self, need, walk):
        walk.visit(self.x, _release(need, [self.dim]))


class Rank(Node):
    """sortby(key) and top/bottom(x, n, dim): reorder (and cut) one dimension by a 1-D key."""

    def __init__(self, x: Node, key_of: Callable, dim: str, descending: bool, n: int | None, typ: Type,
                 children: tuple):
        self.x, self.key_of, self.dim, self.descending, self.n = x, key_of, dim, descending, n
        self.children = children
        self.type = typ

    def eval(self, ctx):
        x = self.x.eval(ctx)
        key = self.key_of(ctx, x)
        if self.dim not in key.dims or len(key.dims) != 1:
            raise ExprError(f"the sort key must have only the {self.dim} dimension")
        order = rt.rank(key.data, self.descending)
        labels = key.coords[self.dim].labels[order]
        if self.n is not None:
            labels = labels[:self.n]

        def reorder(a: rt.Arr) -> rt.Arr:
            position = {v: i for i, v in enumerate(a.coords[self.dim].labels.tolist())}
            return rt.take(a, self.dim, [position[v] for v in labels.tolist() if v in position])
        if isinstance(x, rt.Rec):
            return rt.Rec({k: reorder(v) for k, v in x.fields.items()})
        return reorder(x)

    def demand(self, need, walk):
        need = _release(need, [self.dim])
        for child in self.children:
            walk.visit(child, need)

    def cost(self, size):
        return 0


class Record(Node):
    def __init__(self, fields: list[tuple[str, Node]]):
        self.fields = fields
        self.children = tuple(n for _, n in fields)
        self.type = Type(RECORD, fields=tuple((k, n.type) for k, n in fields))

    def eval(self, ctx):
        return rt.Rec({k: n.eval(ctx) for k, n in self.fields})


class Pending(Node):
    """x.rolling(…), x.resample(…), x.groupby(…), x.time, x.time.dt: only usable through a following method or
    attribute."""

    def __init__(self, x: Node, kind: str, arg=None, where: ast.AST | None = None):
        self.x, self.kind, self.arg, self.where = x, kind, arg, where
        self.type = Type("pending")


def _release(need: Need, dims) -> Need:
    """The need of an input whose `dims` are consumed (reduced, ranked): all of them."""
    return Need(None if TIME in dims else need.time, None if MODEL in dims else need.models)


class _DemandWalk:
    def __init__(self):
        self.seen: set = set()
        self.visits = 0
        self.demands: dict[Location, Demand] = {}

    def visit(self, node: Node, need: Need) -> None:
        key = (id(node), need)
        if key in self.seen:
            return
        self.seen.add(key)
        self.visits += 1
        if self.visits > MAX_DEMAND_VISITS:
            raise ExprError("query too complex")
        node.demand(need, self)

    def record(self, leaf: Leaf, need: Need) -> None:
        d = self.demands.setdefault(leaf.location, Demand())
        if leaf.variable:  # coordinates have no model dimension: they don't decide which models to fetch
            d.variables.add(leaf.name)
            if d.models is not None:
                d.models = None if need.models is None else d.models | need.models
        if TIME in leaf.type.dims:
            if need.time is None:
                d.unbounded = True
            else:
                d.start = need.time[0] if d.start is None else min(d.start, need.time[0])
                d.end = need.time[1] if d.end is None else max(d.end, need.time[1])
                d.windows.add(need.time)


# -- compiling --------------------------------------------------------------------------------------------------------

@dataclass
class Sizes:
    models: int
    members: int
    hours: int  # longest time axis of any location
    points: int  # most points of any location (point dimension)
    lat: int  # largest grid
    lon: int


@dataclass
class Compiled:
    result: Node
    demands: dict[Location, Demand]
    warnings: list[str]
    nodes: list[Node]

    def estimate(self, sizes: Sizes) -> int:
        def size(dims) -> int:
            n = 1
            for d in dims:
                n *= {MODEL: sizes.models, MEMBER: sizes.members, TIME: sizes.hours, POINT: sizes.points,
                      LAT: sizes.lat, LON: sizes.lon, HOUR: 24, BINS: MAX_BINS, QUANTILE: MAX_LEVELS}[d]
            return n
        leaves = {(n.location, n.name): n.type.dims for n in self.nodes if isinstance(n, Leaf) and n.variable}
        return sum(n.cost(size) for n in self.nodes) + sum(size(dims) * 8 for dims in leaves.values())

    def evaluate(self, ctx: rt.Context):
        for data in ctx.locations.values():
            for a in data.arrays.values():
                ctx.charge(a.shape)
        return self.result.eval(ctx)


def compile_query(program: Program, env: Env) -> Compiled:
    compiler = _Compiler(env)
    for i, (name, expr) in enumerate(program.bindings):
        compiler.binding_asts[name] = (i, expr)
    compiler.visible = len(program.bindings)
    result = compiler.expr(program.result, record_ok=True)
    if compiler.is_open(result):
        raise ExprError(f"the query uses forecast data without a location; pick one with {LOCATE_HINT}")
    walk = _DemandWalk()
    walk.visit(result, Need())
    for location, d in walk.demands.items():
        if d.unbounded and location.kind != "route":
            raise ExprError(f"{location.describe()}: select a time range, e.g. .sel(time=slice(\"2026-10-10T10:00\", "
                            f"\"2026-10-10T16:00\")); the query uses its time dimension without one")
        if not d.variables:
            raise ExprError(f"{location.describe()}: the query uses only coordinates of this location; use one of "
                            f"its weather variables, or drop it")
        if d.end is not None and d.end <= d.start:
            raise ExprError(f"{location.describe()}: the selected time range is empty")
    if not any(d.variables for d in walk.demands.values()):
        raise ExprError("the query uses no weather variable")
    return Compiled(result, walk.demands, compiler.warnings.messages, _reachable(result))


def isel_spec(value: ast.expr) -> tuple[str, object, bool]:
    """(mode, positions, drop) for isel: a position (drops the dimension), a list, or slice(start, stop[, step])."""
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "slice":
        parts = [None if isinstance(a, ast.Constant) and a.value is None else _integer(a, "position")
                 for a in value.args]
        if not 1 <= len(parts) <= 3:
            raise _error(value, "slice(start, stop[, step])")
        return "positions", tuple(parts) if len(parts) > 1 else (None, parts[0]), False
    if isinstance(value, (ast.List, ast.Tuple)):
        return "positions", [_integer(e, "position") for e in value.elts], False
    return "positions", [_integer(value, "position")], True


def _reachable(root: Node) -> list[Node]:
    seen: dict[int, Node] = {}
    stack = [root]
    while stack:
        node = stack.pop()
        if id(node) not in seen:
            seen[id(node)] = node
            stack.extend(node.children)
    return list(seen.values())


_ARITHMETIC = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
               ast.FloorDiv: np.floor_divide, ast.Mod: np.mod, ast.Pow: np.power}
_COMPARISONS = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
                ast.Eq: operator.eq, ast.NotEq: operator.ne}
_DIM_HINTS = {
    POINT: " (only .interp(places(…)) or .interp(lat=(\"point\", […]), …) give a point dimension)",
    LAT: " (only areas, .sel(lat=slice(…), lon=slice(…)), have it)",
    LON: " (only areas, .sel(lat=slice(…), lon=slice(…)), have it)",
    MODEL: " (it was already reduced or selected)",
    MEMBER: " (it was already reduced)",
    TIME: " (it was already reduced or grouped)",
}


class _Compiler:
    def __init__(self, env: Env):
        self.env = env
        self.binding_asts: dict[str, tuple[int, ast.expr]] = {}
        self.visible = 0  # bindings defined before the statement being compiled
        self.bindings: dict[tuple, Node] = {}  # (name, location context) → compiled binding
        self.memo: dict[tuple, Node] = {}  # (syntax node, location context) → compiled node
        self.open_cache: dict[int, bool] = {}
        self.context: Location | None = None  # location for variables without one (inside .interp etc.)
        self.locations: dict[tuple, Location] = {}
        self.warnings = Warnings()

    def is_open(self, node: Node) -> bool:
        """Whether some variable or coordinate in it has no location (cached: nested locates ask repeatedly)."""
        key = id(node)
        if key not in self.open_cache:
            self.open_cache[key] = (isinstance(node, Leaf) and node.location is None) or any(
                self.is_open(child) for child in node.children)
        return self.open_cache[key]

    def in_context(self, location: Location | None, fn):
        saved, self.context = self.context, location
        try:
            return fn()
        finally:
            self.context = saved

    # -- expressions

    def expr(self, node: ast.expr, record_ok: bool = False, dataset_ok: bool = False, spec_ok: bool = False) -> Node:
        result = self._expr(node)
        kind = result.type.kind
        if kind == SPEC and not spec_ok:
            raise _error(node, "places(…) and route(…) are locations: use them as forecast().interp(places(…))")
        if kind == RECORD and not record_ok:
            raise _error(node, "a dict ({…}) can only be the answer, a binding, or the argument of top()/bottom()")
        if kind == DATASET and not dataset_ok:
            raise _error(node, "this is a dataset; pick a variable, e.g. .t2m or .precip")
        if kind == "pending":
            raise _error(node, self._pending_hint(result))
        return result

    def array(self, node: ast.expr, *kinds: str, what: str = "this") -> Node:
        result = self.expr(node)
        if kinds and result.type.kind not in kinds:
            expected = "a condition (a comparison)" if kinds == (BOOL,) else "numbers"
            raise _error(node, f"{what} must be {expected}, not {result.type.describe()}")
        return result

    def _pending_hint(self, p: "Pending") -> str:
        return {"rolling": "rolling(…) needs an aggregation: .sum(), .mean(), .max() or .min()",
                "resample": "resample(…) needs an aggregation: .sum(), .mean(), .max() or .min()",
                "hour": "groupby(…) needs an aggregation: .sum(), .mean(), .max() or .min()",
                "bins": "groupby_bins(…) needs an aggregation: .sum(), .mean(), .max() or .min()",
                "time": "x.time is a coordinate; use x.time.dt.hour or x.time.dt.dayofweek, or .sel(time=…)",
                "dt": "use x.time.dt.hour or x.time.dt.dayofweek"}.get(p.kind, "incomplete expression")

    def _expr(self, node: ast.expr) -> Node:
        key = (id(node), self.context)
        if key not in self.memo:
            self.memo[key] = self._compile(node)
        return self.memo[key]

    def _compile(self, node: ast.expr) -> Node:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise _error(node, "only numbers are values; text is only allowed as an argument")
            return Const(_number(node, "number"))
        if isinstance(node, ast.Name):
            return self.name(node)
        if isinstance(node, ast.Attribute):
            return self.attribute(node)
        if isinstance(node, ast.Subscript):
            target = self.expr(node.value, dataset_ok=True)
            if target.type.kind == DATASET and isinstance(node.slice, ast.Constant):
                return self.variable(target, _string(node.slice, "variable name"), node)
            raise _error(node, "indexing isn't supported; use .sel(…) or .isel(…)")
        if isinstance(node, ast.Call):
            return self.call(node)
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                raise _error(node, "`not` doesn't work elementwise on arrays (as in xarray); use ~(condition)")
            if isinstance(node.op, ast.Invert):
                x = self.array(node.operand, BOOL, what="the operand of ~")
                return Elementwise(rt.logic_not, [x], x.type)
            x = self.array(node.operand, NUM, BOOL)
            if isinstance(node.op, ast.USub):
                return Elementwise(operator.neg, [x], Type(NUM, x.type.dims, x.type.unit, x.type.state, x.type.time))
            return x
        if isinstance(node, ast.BinOp):
            return self.binop(node)
        if isinstance(node, ast.BoolOp):
            raise _error(node, "`and`/`or` don't work elementwise on arrays (as in xarray); use & and | with "
                               "parentheses: (a < 1) & (b > 2)")
        if isinstance(node, ast.Compare):
            if len(node.ops) > 1:
                raise _error(node, "chained comparisons don't work on arrays; write (a < b) & (b < c)")
            fn = _COMPARISONS.get(type(node.ops[0]))
            if fn is None:
                raise _error(node, "use < <= > >= == != to compare")
            parts = [self.array(n, NUM, BOOL) for n in (node.left, node.comparators[0])]
            typ = self.combine(parts, node, unit_rule="same")
            return Elementwise(rt.compare(fn), parts, Type(BOOL, typ.dims, NO_UNIT, False, typ.time))
        if isinstance(node, ast.IfExp):
            raise _error(node, "`a if c else b` doesn't work on arrays; use x.where(cond, other) or "
                               "np.where(cond, a, b)")
        if isinstance(node, ast.Dict):
            return self.record(node)
        raise _error(node, f"{type(node).__name__} isn't supported; see weather_query_help for what is")

    def name(self, node: ast.Name) -> Node:
        if node.id in self.binding_asts and self.binding_asts[node.id][0] < self.visible:
            key = (node.id, self.context)
            if key not in self.bindings:
                index, expr = self.binding_asts[node.id]
                saved, self.visible = self.visible, index
                try:
                    compiled = self.expr(expr, record_ok=True, dataset_ok=True, spec_ok=True)
                finally:
                    self.visible = saved
                self.bindings[key] = compiled if isinstance(compiled, (Dataset, Spec)) else Binding(node.id, compiled)
            return self.bindings[key]
        if node.id in self.binding_asts:
            raise _error(node, f"{node.id} is assigned later in the query")
        if node.id in FUNCTIONS:
            raise _error(node, f"{node.id} is a function; call it, e.g. {node.id}(…)")
        if node.id in ("np", "xr"):
            raise _error(node, f"use {node.id}.maximum(a, b), {node.id}.where(cond, a, b), …")
        if node.id in CATALOG or node.id in ("time", "model", "member", "lat", "lon"):
            raise _error(node, f"unknown name {node.id!r}: variables belong to the dataset, e.g. "
                               f"forecast().{node.id}")
        raise _error(node, f"unknown name {node.id!r}")

    def record(self, node: ast.Dict) -> Node:
        if len(node.keys) > MAX_FIELDS:
            raise _error(node, f"at most {MAX_FIELDS} fields")
        fields: list[tuple[str, Node]] = []
        for key, value in zip(node.keys, node.values):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str) and 0 < len(key.value) <= 40):
                raise _error(node, "dict keys must be short strings, e.g. {'p_dry': …}")
            if any(key.value == k for k, _ in fields):
                raise _error(key, f"duplicate key {key.value!r}")
            fields.append((key.value, self.array(value, NUM, BOOL, LABEL)))
        return Record(fields)

    # -- types

    def combine(self, parts: list[Node], node: ast.AST, unit_rule: str) -> Type:
        """Dimensions, unit and state of an elementwise combination (kind is set by the caller)."""
        dims = ordered(d for p in parts for d in p.type.dims)
        flavours = {p.type.time for p in parts if p.type.time}
        if len(flavours) > 1:
            raise _error(node, f"can't combine different time resolutions ({', '.join(sorted(flavours))}); "
                               f"resample both the same way")
        units = [p.type.unit for p in parts if p.type.unit is not None]
        if unit_rule == "same":
            if len({str(u) for u in units}) > 1:
                self.warnings.add(f"line {node.lineno}: combining values in {' and '.join(str(u) or '1' for u in units)}")
            unit = units[0] if units else None
            state = any(p.type.state for p in parts)
        elif unit_rule == "mul":
            unit = None
            for p in parts:
                if p.type.unit is not None:
                    unit = p.type.unit if unit is None else unit * p.type.unit
            arrays = [p for p in parts if p.type.dims or not isinstance(p, Const)]
            state = len(arrays) == 1 and arrays[0].type.state
        elif unit_rule == "div":
            a, b = parts
            unit = a.type.unit / b.type.unit if a.type.unit is not None and b.type.unit is not None else (
                a.type.unit if b.type.unit is None else NO_UNIT / b.type.unit)
            state = a.type.state and isinstance(b, Const)
        else:  # nonlinear: clip, where, comparisons
            unit = units[0] if units else None
            state = False
        return Type(NUM, dims, unit, state, next(iter(flavours), None))

    def binop(self, node: ast.BinOp) -> Node:
        if isinstance(node.op, (ast.BitAnd, ast.BitOr)):
            parts = [self.expr(n) for n in (node.left, node.right)]
            for side, p in zip((node.left, node.right), parts):
                if p.type.kind != BOOL:
                    raise _error(side, "& and | combine conditions; Python binds them before comparisons, so put "
                                       "each comparison in parentheses: (a < 1) & (b < 2)")
            typ = self.combine(parts, node, "other")
            fn = rt.logic_and if isinstance(node.op, ast.BitAnd) else rt.logic_or
            return Elementwise(fn, parts, Type(BOOL, typ.dims, NO_UNIT, False, typ.time))
        fn = _ARITHMETIC.get(type(node.op))
        if fn is None:
            raise _error(node, f"operator {type(node.op).__name__} isn't supported")
        parts = [self.array(n, NUM, BOOL) for n in (node.left, node.right)]
        if isinstance(node.op, (ast.Add, ast.Sub)):
            typ = self.combine(parts, node, "same")
        elif isinstance(node.op, ast.Mult):
            typ = self.combine(parts, node, "mul")
        elif isinstance(node.op, ast.Div):
            typ = self.combine(parts, node, "div")
        elif isinstance(node.op, ast.Pow):
            typ = self.combine(parts, node, "other")
            base = parts[0].type.unit
            unit = base ** parts[1].value if isinstance(parts[1], Const) and base is not None else (
                None if base is None else Unit((), True))
            typ = Type(NUM, typ.dims, unit, False, typ.time)
        else:
            typ = self.combine(parts, node, "other")
        return Elementwise(fn, parts, typ)

    # -- attributes and calls

    def attribute(self, node: ast.Attribute) -> Node:
        target = self._expr(node.value)
        attr = node.attr
        if attr.startswith("_"):
            raise _error(node, f"attribute {attr} isn't supported")
        if target.type.kind == DATASET:
            if attr == "time":
                return Pending(target, "time")
            return self.variable(target, attr, node)
        if isinstance(target, Pending):
            if target.kind == "time" and attr == "dt":
                return Pending(target.x, "dt")
            if target.kind == "dt" and attr in ("hour", "dayofweek"):
                return self.coordinate(target.x, attr, node)
            raise _error(node, self._pending_hint(target))
        if target.type.kind in (NUM, BOOL):
            if attr == "time":
                return Pending(target, "time")
            if attr in COORDINATES:
                return self.coordinate(target, attr, node)
        raise _error(node, f"attribute {attr!r} isn't supported here")

    def variable(self, dataset: Dataset, name: str, node: ast.AST) -> Node:
        location = dataset.location or self.context
        if name in COORDINATES:
            return self.coordinate(dataset, name, node)
        if name not in CATALOG:
            raise _error(node, f"unknown variable {name!r}; variables: {', '.join(CATALOG)}")
        if CATALOG[name].route_only and location is not None and location.kind != "route":
            raise _error(node, f"{name} is only available on routes")
        if not any(name in provided for provided in self.env.provided.values()):
            raise _error(node, f"no model provides {name}")
        space = (LAT, LON) if location is None else location.space
        unit = Unit.parse(CATALOG[name].unit)
        flavour = ROUTE if location is not None and location.kind == "route" else HOURLY
        typ = Type(NUM, (MODEL, MEMBER, TIME) + space, unit, name in STATE, flavour, direction=name in DIRECTIONS)
        return self.apply_selections(Leaf(location, name, typ, variable=True), dataset, location)

    def apply_selections(self, result: Node, dataset: Dataset, location: Location | None) -> Node:
        """Apply the dataset's selections. A coordinate lacks some of the dataset's dimensions (lat has no time),
        so those are skipped; a selection along a dimension the location doesn't have at all is an error."""
        space = (LAT, LON) if location is None else location.space
        for dim, value, drop, where in dataset.selections:
            if dim in result.type.dims:
                result = self.select(result, dim, value, drop, where)
            elif dim in (LAT, LON, POINT) and dim not in space:
                raise _error(where, f"{location.describe()} has no {dim} dimension; this selection doesn't apply")
        return result

    def coordinate(self, target: Node, name: str, node: ast.AST) -> Node:
        if isinstance(target, Dataset):
            location = target.location or self.context
            route = location is not None and location.kind == "route"
            if name in ("hour", "dayofweek"):
                typ = Type(NUM, (TIME,), NO_UNIT, False, ROUTE if route else HOURLY)
            elif name == "distance_km":
                if location is not None and not route:
                    raise _error(node, "distance_km is only available on routes")
                typ = Type(NUM, (TIME,), Unit.parse("km"), False, ROUTE)
            else:
                space = (LAT, LON) if location is None else location.space
                dims = (TIME,) if route else (POINT,) if POINT in space else ((LAT,) if name == "lat" else (LON,)) \
                    if LAT in space else ()
                typ = Type(NUM, dims, Unit.parse("°"), False, ROUTE if route else None)
            return self.apply_selections(Leaf(location, name, typ, variable=False), target, location)
        t = target.type
        if name in ("hour", "dayofweek"):
            if not t.has(TIME):
                raise _error(node, "this value has no time dimension")
            return CoordOf(target, name, Type(NUM, (TIME,), NO_UNIT, False, t.time))
        if name == "distance_km":
            if t.time != ROUTE:
                raise _error(node, "distance_km is only available on routes")
            return CoordOf(target, name, Type(NUM, (TIME,), Unit.parse("km"), False, ROUTE))
        if t.time == ROUTE:
            return CoordOf(target, name, Type(NUM, (TIME,), Unit.parse("°"), False, ROUTE))
        own = LAT if name == "lat" else LON
        if t.has(own):
            return CoordOf(target, name, Type(NUM, (own,), Unit.parse("°"), False, None))
        if t.has(POINT):
            return CoordOf(target, name, Type(NUM, (POINT,), Unit.parse("°"), False, None))
        # No spatial dimension: a single point (or not located yet, if it is still to be located).
        locations = {n.location for n in _reachable(target) if isinstance(n, Leaf)}
        if None in locations and not t.has(LAT):
            raise _error(node, f"{name}: locate this value first")
        if len(locations) != 1 or next(iter(locations)).kind != "point":
            raise _error(node, f"{name}: this value combines several locations, so it has no single {name}")
        lat, lon = next(iter(locations)).args
        return Const(lat if name == "lat" else lon)

    def call(self, node: ast.Call) -> Node:
        if any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):
            raise _error(node, "*args and **kwargs aren't supported")
        kwargs = {k.arg: k.value for k in node.keywords}
        func = node.func
        if isinstance(func, ast.Name):
            if func.id == "forecast":
                if node.args or kwargs:
                    raise _error(node, "forecast() takes no arguments; select with .interp/.sel")
                return Dataset(None)
            if func.id in ("places", "route"):
                return Spec(self.register(node, *self.spec(node, func.id, node.args, kwargs)))
            if func.id == "distance_from":
                return self.distance_from(node, node.args, kwargs)
            if func.id in ("top", "bottom"):
                return self.top(node, func.id == "top", node.args, kwargs)
            if func.id == "abs":
                (x,) = self.positional(node, 1)
                x = self.array(x, NUM, BOOL)
                return Elementwise(np.abs, [x], Type(NUM, x.type.dims, x.type.unit, x.type.state, x.type.time))
            if func.id in ("max", "min", "sum", "len", "round", "any", "all"):
                raise _error(node, f"Python's {func.id}() doesn't work on arrays; use x.{func.id}(\"dim\") or "
                                   f"np.maximum(a, b)" if func.id != "len" else "len() isn't supported")
            if func.id in ("slice", "range"):
                raise _error(node, f"{func.id}(…) is only allowed as an argument, e.g. .sel(time=slice(…)) or "
                                   f"bins=range(…)")
            raise _error(node, f"unknown function {func.id!r}")
        if not isinstance(func, ast.Attribute):
            raise _error(node, "only functions and methods listed in weather_query_help can be called")
        if isinstance(func.value, ast.Name) and func.value.id in ("np", "xr") and func.value.id not in self.bindings:
            return self.numpy(node, func.attr, node.args, kwargs)
        method = func.attr
        if method == "interp" or (method == "sel" and (LAT in kwargs or LON in kwargs)):
            return self.locate(node, func.value, method, node.args, kwargs)
        target = self._expr(func.value)
        if isinstance(target, Pending):
            return self.pending_method(node, target, method, node.args, kwargs)
        if target.type.kind == DATASET:
            if method in ("sel", "isel"):
                return self.dataset_select(node, target, method, node.args, kwargs)
            raise _error(node, f"datasets have .sel/.isel/.interp and variables; {method}() applies to a variable, "
                               f"e.g. .t2m.{method}(…)")
        if target.type.kind == RECORD:
            raise _error(node, "methods apply to arrays, not to dicts")
        return self.method(node, target, method, node.args, kwargs)

    def positional(self, node: ast.Call, n: int, at_most: int | None = None) -> list[ast.expr]:
        at_most = n if at_most is None else at_most
        if not n <= len(node.args) <= at_most:
            name = getattr(node.func, "attr", getattr(node.func, "id", "this"))
            raise _error(node, f"{name}() takes {n if n == at_most else f'{n} to {at_most}'} positional "
                               f"argument{'s' if at_most != 1 else ''}")
        return node.args

    def only(self, node: ast.Call, kwargs: dict, allowed) -> None:
        extra = set(kwargs) - set(allowed)
        if extra:
            name = getattr(node.func, "attr", getattr(node.func, "id", "this"))
            raise _error(node, f"{name}() has no argument {sorted(extra)[0]!r}")

    def dims_argument(self, node: ast.Call, x: Node, args, kwargs, name: str) -> list[str]:
        if "dim" in kwargs:
            spec = kwargs["dim"]
        elif args:
            spec = args[0]
        else:
            raise _error(node, f"{name}() needs the dimension(s) to reduce, e.g. .{name}(\"member\") or "
                               f".{name}([\"time\", \"point\"]); nothing is reduced implicitly")
        dims = list(dict.fromkeys(_strings(spec, "dim")))
        for d in dims:
            if d not in DIMS:
                raise _error(node, f"unknown dimension {d!r}; dimensions: {', '.join(DIMS)}")
            if not x.type.has(d):
                raise _error(node, f"{name}(): no {d} dimension here{_DIM_HINTS.get(d, '')}; the value has "
                                   f"({', '.join(x.type.dims)})")
        return dims

    def method(self, node: ast.Call, x: Node, method: str, args, kwargs) -> Node:
        t = x.type
        if method in REDUCTIONS:
            self.only(node, kwargs, ("dim",))
            if len(args) > 1:
                raise _error(node, f"{method}() takes one argument, the dimension(s)")
            dims = self.dims_argument(node, x, args, kwargs, method)
            return self.reduce(node, x, dims, method)
        if method == "quantile":
            self.only(node, kwargs, ("q", "dim"))
            q = kwargs.get("q", args[0] if args else None)
            if q is None:
                raise _error(node, "quantile(q, dim)")
            dims = self.dims_argument(node, x, args[1:], kwargs, "quantile")
            as_dim = isinstance(q, (ast.List, ast.Tuple))
            levels = [_number(e, "quantile level") for e in (q.elts if as_dim else [q])]
            if not (1 <= len(levels) <= MAX_LEVELS and all(0 <= v <= 1 for v in levels)):
                raise _error(node, f"quantile levels must lie in 0..1 (at most {MAX_LEVELS})")
            if t.has(QUANTILE) and as_dim:
                raise _error(node, "this value already has a quantile dimension")
            self.reduction_warnings(node, x, dims, "quantile")
            keep = tuple(d for d in t.dims if d not in dims) + ((QUANTILE,) if as_dim else ())
            unit = NO_UNIT if t.kind == BOOL else t.unit
            typ = Type(NUM, keep, unit, t.state, t.time if TIME not in dims else None, direction=t.direction)
            return Quantile(x, levels, dims, as_dim, typ)
        if method in ("sel", "isel"):
            if args:
                raise _error(node, f"{method}() takes keyword arguments, e.g. .{method}(time=…)")
            result = x
            for dim, value in kwargs.items():
                result = self.selection(node, result, method, dim, value)
            return result
        if method == "where":
            self.only(node, kwargs, ("other", "cond"))
            cond = kwargs.get("cond", args[0] if args else None)
            if cond is None or len(args) > 2:
                raise _error(node, "where(cond, other=missing)")
            other = kwargs.get("other", args[1] if len(args) > 1 else None)
            parts = [x, self.array(cond, BOOL, what="where()'s condition")]
            if other is not None:
                parts.append(self.array(other, NUM, BOOL))
            typ = self.combine(parts, node, "other")
            # The value branches must agree in unit (the condition's doesn't matter), as in np.where.
            unit = self.combine([x, parts[2]], node, "same").unit if other is not None else t.unit
            kind = t.kind if other is None or parts[2].type.kind == t.kind else NUM
            direction = t.direction and (other is None or parts[2].type.direction)
            fn = (lambda a, c: rt.where(c, a, np.nan)) if other is None else (lambda a, c, o: rt.where(c, a, o))
            return Elementwise(fn, parts, Type(kind, typ.dims, unit if kind == NUM else NO_UNIT, False, typ.time,
                                               direction=direction))
        if method == "clip":
            self.only(node, kwargs, ("min", "max"))
            if len(args) > 2:
                raise _error(node, "clip(min=…, max=…)")
            bounds = [kwargs.get("min", args[0] if args else None), kwargs.get("max", args[1] if len(args) > 1 else None)]
            lo = -np.inf if bounds[0] is None else _number(bounds[0], "clip bound")
            hi = np.inf if bounds[1] is None else _number(bounds[1], "clip bound")
            self.array_kind(node, x, NUM, BOOL)
            return Elementwise(lambda a: np.clip(a, lo, hi), [x], Type(NUM, t.dims, t.unit, False, t.time))
        if method == "round":
            self.only(node, kwargs, ("decimals",))
            n = _integer(kwargs.get("decimals", args[0] if args else ast.Constant(0)), "decimals")
            return Elementwise(lambda a: np.round(a, n), [x], Type(NUM, t.dims, t.unit, t.state, t.time,
                                                                   direction=t.direction))
        if method in ("rolling", "resample"):
            if args or len(kwargs) != 1 or TIME not in kwargs:
                raise _error(node, f"{method}(time=…)")
            if not t.has(TIME):
                raise _error(node, f"{method}(): no time dimension here")
            if t.time == ROUTE:
                raise _error(node, f"{method}() isn't available on routes (samples are irregular); use "
                                   f"groupby_bins(\"distance_km\", bins=[…])")
            if method == "rolling":
                n = _integer(kwargs[TIME], "window length")
                if not 1 <= n <= MAX_ROLLING:
                    raise _error(node, f"rolling windows must be 1 to {MAX_ROLLING} samples")
                return Pending(x, "rolling", n, node)
            freq = _string(kwargs[TIME], "frequency")
            if freq not in RESAMPLE:
                raise _error(node, f"resample frequencies: {', '.join(RESAMPLE)} (local time)")
            if t.time != HOURLY:
                raise _error(node, "resample() works on hourly values only")
            return Pending(x, "resample", freq, node)
        if method == "groupby":
            self.positional(node, 1)
            self.only(node, kwargs, ())
            if _string(args[0], "group") != "time.hour":
                raise _error(node, "groupby() supports \"time.hour\" (use resample for days)")
            if not t.has(TIME) or t.time != HOURLY:
                raise _error(node, "groupby(\"time.hour\") needs hourly values")
            return Pending(x, "hour", None, node)
        if method == "groupby_bins":
            self.only(node, kwargs, ("bins",))
            if not args or _string(args[0], "group") != "distance_km" or "bins" not in kwargs and len(args) < 2:
                raise _error(node, "groupby_bins(\"distance_km\", bins=[0, 10, 20, …])")
            if t.time != ROUTE:
                raise _error(node, "groupby_bins(\"distance_km\", …) is for routes")
            edges = self.bins(kwargs.get("bins", args[1] if len(args) > 1 else None))
            return Pending(x, "bins", edges, node)
        if method in ("idxmax", "idxmin"):
            self.only(node, kwargs, ("dim",))
            dims = self.dims_argument(node, x, args, kwargs, method)
            if len(dims) != 1:
                raise _error(node, f"{method}() takes exactly one dimension")
            (dim,) = dims
            keep = tuple(d for d in t.dims if d != dim)
            return Idx(x, dim, method == "idxmax", Type(LABEL, keep, None, False, t.time if dim != TIME else None,
                                                        label_dim=dim))
        if method == "sortby":
            self.only(node, kwargs, ("ascending",))
            (key,) = self.positional(node, 1)
            key = self.array(key, NUM, BOOL)
            ascending = True
            if "ascending" in kwargs:
                value = kwargs["ascending"]
                if not (isinstance(value, ast.Constant) and isinstance(value.value, bool)):
                    raise _error(node, "ascending must be True or False")
                ascending = value.value
            if len(key.type.dims) != 1 or not t.has(key.type.dims[0]):
                raise _error(node, "sortby(key): the key must have exactly one dimension, which the value has too")
            dim = key.type.dims[0]
            return Rank(x, lambda ctx, _x, key=key: key.eval(ctx), dim, not ascending, None, t, (x, key))
        raise _error(node, f"method {method}() isn't supported; see weather_query_help")

    def array_kind(self, node, x: Node, *kinds) -> None:
        if x.type.kind not in kinds:
            raise _error(node, f"needs numbers, not {x.type.describe()}")

    def bins(self, node: ast.expr | None) -> list[float]:
        if node is None:
            raise ExprError("groupby_bins needs bins=[…]")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
            values = [_integer(a, "range argument") for a in node.args]
            if not 1 <= len(values) <= 3 or (len(values) == 3 and values[2] <= 0):
                raise _error(node, "range(start, stop, step)")
            start, stop, step = (0, values[0], 1) if len(values) == 1 else (*values, 1)[:3]
            if -(-(stop - start) // step) > MAX_BINS + 1:  # count before materialising the range
                raise _error(node, f"bins: 2 to {MAX_BINS + 1} increasing edges")
            edges = [float(v) for v in range(start, stop, step)]
        elif isinstance(node, (ast.List, ast.Tuple)):
            edges = [_number(e, "bin edge") for e in node.elts]
        else:
            raise _error(node, "bins must be a list of distances or range(…)")
        if not 2 <= len(edges) <= MAX_BINS + 1 or any(b <= a for a, b in zip(edges, edges[1:])):
            raise _error(node, f"bins: 2 to {MAX_BINS + 1} increasing edges")
        return edges

    def pending_method(self, node: ast.Call, p: Pending, method: str, args, kwargs) -> Node:
        if p.kind not in ("rolling", "resample", "hour", "bins") or method not in GROUP_REDUCTIONS:
            raise _error(node, self._pending_hint(p))
        if args or kwargs:
            raise _error(node, f"{method}() takes no arguments here")
        t = p.x.type
        unit = NO_UNIT if t.kind == BOOL else t.unit
        if method == "sum":
            unit = (unit or NO_UNIT) * HOURS
            if t.state:
                self.warn_state_sum(node, p.x)
        self.warn_linear_direction(node, p.x, method)
        kind = BOOL if t.kind == BOOL and method in ("max", "min") else NUM
        state = t.state and method != "sum"
        direction = t.direction and method != "sum"
        if p.kind == "rolling":
            return Rolling(p.x, p.arg, method, Type(kind, t.dims, unit, state, t.time, direction=direction))
        if p.kind == "resample":
            return Group(p.x, "resample", p.arg, method, Type(kind, t.dims, unit, state, p.arg, direction=direction),
                         self.env.tz)
        new = HOUR if p.kind == "hour" else BINS
        dims = ordered([d for d in t.dims if d != TIME] + [new])
        return Group(p.x, p.kind, p.arg, method, Type(kind, dims, unit, state, None, direction=direction),
                     self.env.tz)

    def reduce(self, node: ast.Call, x: Node, dims: list[str], how: str) -> Node:
        t = x.type
        if how in ("any", "all") and t.kind != BOOL:
            raise _error(node, f"{how}() needs a condition")
        self.reduction_warnings(node, x, dims, how)
        keep = tuple(d for d in t.dims if d not in dims)
        unit = NO_UNIT if t.kind == BOOL else t.unit
        if how == "sum" and TIME in dims:
            unit = (unit or NO_UNIT) * HOURS
        if how == "count":
            unit = NO_UNIT
        kind = BOOL if how in ("any", "all") or (how in ("max", "min") and t.kind == BOOL) else NUM
        state = t.state and not (how == "sum" and TIME in dims) and how != "count"
        direction = t.direction and how in ("mean", "median", "min", "max")
        typ = Type(kind, keep, unit, state, t.time if TIME not in dims else None, direction=direction)
        return Reduce(x, dims, how, typ)

    def reduction_warnings(self, node: ast.AST, x: Node, dims: list[str], how: str) -> None:
        if how == "sum" and TIME in dims and x.type.state:
            self.warn_state_sum(node, x)
        self.warn_linear_direction(node, x, how)
        if MODEL in dims and x.type.has(MEMBER):
            if MEMBER in dims:
                self.warnings.add(f"line {node.lineno}: pooling members across models weighs each model by its "
                                  f"member count (e.g. IFS 50, ICON-D2 20); reduce member first for per-model "
                                  f"values, then combine models")
            else:
                self.warnings.add(f"line {node.lineno}: reducing model while members remain combines unrelated "
                                  f"members (member 3 of one model has nothing to do with member 3 of another); "
                                  f"reduce member first")

    def warn_linear_direction(self, node: ast.AST, x: Node, how: str) -> None:
        if x.type.direction and how != "count":
            self.warnings.add(f"line {node.lineno}: {how}() treats directions as numbers on a line, where 350° and "
                              f"10° are 340° apart (their mean is 180°): only right while they don't straddle north. "
                              f"Average unit vectors instead: r = np.deg2rad(d), then np.rad2deg(np.arctan2("
                              f"np.sin(r).mean(\"member\"), np.cos(r).mean(\"member\"))) % 360. For a spread, take "
                              f"quantiles of the deviation from that mean m, (d - m + 180) % 360 - 180; or count a "
                              f"sector, ((d >= 315) | (d < 45)).mean(\"member\")")

    def warn_state_sum(self, node: ast.AST, x: Node) -> None:
        self.warnings.add(f"line {node.lineno}: summing a state variable over time adds up "
                          f"{x.type.unit}·h from zero, which is rarely meaningful; did you mean .mean(\"time\"), "
                          f"hours of a condition ((x > 25).sum(\"time\")) or degree-hours "
                          f"((x - 18).clip(min=0).sum(\"time\"))?")

    def selection(self, node: ast.Call, x: Node, method: str, dim: str, value: ast.expr) -> Node:
        if dim not in DIMS:
            raise _error(node, f"unknown dimension {dim!r}")
        if not x.type.has(dim):
            raise _error(node, f"{method}(): no {dim} dimension here{_DIM_HINTS.get(dim, '')}")
        if method == "isel":
            return self.select(x, dim, isel_spec(value), None, value)
        return self.select(x, dim, value, None, value)

    def select(self, x: Node, dim: str, value, drop, where: ast.AST) -> Node:
        """sel along one dimension; `value` is the ast argument (or a prepared (mode, value, drop) tuple)."""
        if isinstance(value, tuple):
            mode, spec, drop = value
            typ = self.dropped(x.type, dim) if drop else x.type
            return Select(x, dim, spec, mode, drop, typ, where)
        mode, spec, drop = self.sel_spec(dim, value)
        typ = self.dropped(x.type, dim) if drop else x.type
        return Select(x, dim, spec, mode, drop, typ, where)

    def sel_spec(self, dim: str, value: ast.expr) -> tuple[str, object, bool]:
        tz = self.env.tz
        if dim == TIME:
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "slice":
                if len(value.args) != 2:
                    raise _error(value, "slice(start, end): both ends, as times; the end is excluded")
                start, _ = _parse_time(value.args[0], tz)
                end, end_date = _parse_time(value.args[1], tz)
                if end <= start:
                    raise _error(value, "the end of a time slice must be after its start")
                return "range", (start, end), False
            t, date_only = _parse_time(value, tz)
            if date_only:  # a whole local day, as xarray's partial string indexing
                end = int((datetime.fromtimestamp(t * 60, timezone.utc).astimezone(tz) + timedelta(days=1))
                          .replace(hour=0).astimezone(timezone.utc).timestamp() // 60)
                return "range", (t, end), False
            return "labels", [t], True
        if dim == MODEL:
            names = _strings(value, "model name")
            unknown = [n for n in names if n not in self.env.models]
            if unknown:
                raise _error(value, f"unknown model {unknown[0]!r}; models: {', '.join(self.env.models)}")
            return "labels", names, not isinstance(value, (ast.List, ast.Tuple))
        if dim == POINT:
            return "labels", _strings(value, "place name"), not isinstance(value, (ast.List, ast.Tuple))
        if dim in (LAT, LON):
            return "between", self.bounds(value, dim), False
        if dim in (MEMBER, HOUR):
            items = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
            return "labels", [_integer(e, dim) for e in items], not isinstance(value, (ast.List, ast.Tuple))
        if dim == QUANTILE:
            items = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
            return "labels", [_number(e, dim) for e in items], not isinstance(value, (ast.List, ast.Tuple))
        return "labels", _strings(value, dim), not isinstance(value, (ast.List, ast.Tuple))

    @staticmethod
    def dropped(t: Type, dim: str) -> Type:
        return Type(t.kind, tuple(d for d in t.dims if d != dim), t.unit, t.state, t.time if dim != TIME else None,
                    t.fields, t.label_dim, t.direction)

    def dataset_select(self, node: ast.Call, ds: Dataset, method: str, args, kwargs) -> Node:
        if args:
            raise _error(node, f"{method}() takes keyword arguments, e.g. .{method}(time=…)")
        selections = list(ds.selections)
        location = ds.location or self.context
        space = (LAT, LON) if location is None else location.space
        for dim, value in kwargs.items():
            if dim not in (TIME, MODEL, MEMBER, POINT, LAT, LON):
                raise _error(node, f"{method}(): unknown dimension {dim!r}")
            if dim in (POINT, LAT, LON) and dim not in space:
                if location is None:
                    raise _error(node, f"{method}(): no {dim} dimension here; locate first with {LOCATE_HINT}")
                raise _error(node, f"{method}(): {location.describe()} has no {dim} dimension")
            spec = self.sel_spec(dim, value) if method == "sel" else isel_spec(value)
            selections.append((dim, spec, None, value))
        return Dataset(ds.location, tuple(selections))

    def locate(self, node: ast.Call, receiver: ast.expr, method: str, args, kwargs) -> Node:
        """x.interp(…) or x.sel(lat=…, lon=…, …): give x's variables a location (or, for sel on a located grid,
        select within it)."""
        # Compiled without the surrounding location, to see whether x still needs one.
        target = self.in_context(None, lambda: self._expr(receiver))
        if isinstance(target, Pending) or target.type.kind not in (DATASET, NUM, BOOL):
            raise _error(node, f"{method}() applies to forecast() or an array")
        open_ = target.location is None if isinstance(target, Dataset) else self.is_open(target)
        if method == "sel" and not open_:  # within a located grid: an ordinary selection
            if isinstance(target, Dataset):
                return self.dataset_select(node, target, "sel", args, kwargs)
            return self.method(node, self._expr(receiver), "sel", args, kwargs)
        if not open_:
            raise _error(node, f"{method}(): this already has a location")
        if method == "interp":
            location = self.interp_location(node, args, kwargs)
            rest = {}
        else:
            if args or LAT not in kwargs or LON not in kwargs:
                raise _error(node, "an area is .sel(lat=slice(south, north), lon=slice(west, east)); for a single "
                                   "point use .interp(lat=…, lon=…)")
            bounds = [self.bounds(kwargs[d], d) for d in (LAT, LON)]
            (south, north), (west, east) = bounds
            location = self.register(node, "grid", (south, west, north, east))
            rest = {k: v for k, v in kwargs.items() if k not in (LAT, LON)}
        if isinstance(target, Dataset):
            located = Dataset(location, target.selections)
            return self.dataset_select(node, located, "sel", (), rest) if rest else located
        result = self.in_context(location, lambda: self._expr(receiver))
        for dim, value in rest.items():
            result = self.selection(node, result, "sel", dim, value)
        return result

    def bounds(self, value: ast.expr, dim: str) -> tuple[float, float]:
        if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "slice"
                and len(value.args) == 2):
            raise _error(value, f"{dim}=slice(start, end) for an area; for a single point use .interp(lat=…, lon=…)")
        a, b = (_number(v, dim) for v in value.args)
        return min(a, b), max(a, b)

    def interp_location(self, node: ast.Call, args, kwargs) -> Location:
        if args:
            if len(args) != 1 or kwargs:
                raise _error(node, "interp(places(…)), interp(route(…)) or interp(lat=…, lon=…)")
            spec = self.expr(args[0], spec_ok=True)
            if not isinstance(spec, Spec):
                raise _error(args[0], "interp's argument must be places(…) or route(…)")
            return spec.location
        self.only(node, kwargs, (LAT, LON))
        if LAT not in kwargs or LON not in kwargs:
            raise _error(node, "interp(lat=…, lon=…) needs both")
        lat, lon = kwargs[LAT], kwargs[LON]
        if all(isinstance(v, ast.Constant) or isinstance(v, ast.UnaryOp) for v in (lat, lon)):
            return self.register(node, "point", (_number(lat, "lat"), _number(lon, "lon")))
        values = [self.point_list(v, d) for v, d in ((lat, LAT), (lon, LON))]
        if len(values[0]) != len(values[1]):
            raise _error(node, "interp: lat and lon need the same number of points")
        return self.register(node, "points", tuple((None, a, b) for a, b in zip(*values)))

    def point_list(self, value: ast.expr, dim: str) -> list[float]:
        if not (isinstance(value, ast.Tuple) and len(value.elts) == 2
                and isinstance(value.elts[1], (ast.List, ast.Tuple))):
            raise _error(value, f"{dim}=… is a number, or (\"point\", [a, b, …]) for several points")
        if _string(value.elts[0], "dimension name") != POINT:
            raise _error(value, "the dimension of several points must be called \"point\"")
        values = [_number(e, dim) for e in value.elts[1].elts]
        if not 1 <= len(values) <= MAX_POINTS:
            raise _error(value, f"1 to {MAX_POINTS} points")
        return values

    def register(self, node: ast.AST, kind: str, args: tuple) -> Location:
        location = self.locations.setdefault((kind, args), Location(kind, args))
        if len(self.locations) > MAX_LOCATIONS:
            raise _error(node, f"at most {MAX_LOCATIONS} locations per query")
        return location

    def distance_from(self, node: ast.Call, args, kwargs) -> Node:
        self.only(node, kwargs, ())
        x, lat0, lon0 = self.positional(node, 3)
        target = self.expr(x, dataset_ok=True)
        lat0, lon0 = _number(lat0, "lat"), _number(lon0, "lon")
        parts = [self.coordinate(target, name, node) for name in ("lat", "lon")]
        dims = ordered(d for p in parts for d in p.type.dims)
        return Elementwise(lambda la, lo: haversine_km(lat0, lon0, la, lo), parts,
                           Type(NUM, dims, Unit.parse("km"), False, parts[0].type.time))

    def numpy(self, node: ast.Call, name: str, args, kwargs) -> Node:
        if name not in NUMPY:
            raise _error(node, f"np.{name} isn't supported; available: {', '.join('np.' + n for n in NUMPY)}")
        self.only(node, kwargs, ())
        if name == "where":
            cond, a, b = self.positional(node, 3)
            parts = [self.array(cond, BOOL, what="np.where()'s condition"), self.array(a, NUM, BOOL),
                     self.array(b, NUM, BOOL)]
            unit = self.combine(parts[1:], node, "same").unit
            typ = self.combine(parts, node, "other")
            typ = Type(NUM, typ.dims, unit, False, typ.time)
            kind = BOOL if parts[1].type.kind == parts[2].type.kind == BOOL else NUM
            direction = parts[1].type.direction and parts[2].type.direction
            return Elementwise(rt.where, parts, Type(kind, typ.dims, typ.unit, False, typ.time, direction=direction))
        n = 2 if name in ("maximum", "minimum", "hypot", "arctan2") else 1
        parts = [self.array(a, NUM, BOOL) for a in self.positional(node, n)]
        if name in ("maximum", "minimum", "hypot", "arctan2"):
            typ = self.combine(parts, node, "same" if name != "hypot" else "other")
            typ = Type(NUM, typ.dims, RADIANS if name == "arctan2" else typ.unit, False, typ.time)
        elif name in ("sin", "cos", "deg2rad", "rad2deg"):
            t = parts[0].type
            if name in ("sin", "cos") and t.unit == DEGREES:
                self.warnings.add(f"line {node.lineno}: np.{name} takes radians; convert degrees with np.deg2rad(…)")
            unit = {"sin": NO_UNIT, "cos": NO_UNIT, "deg2rad": RADIANS, "rad2deg": DEGREES}[name]
            # An angle from rad2deg (of an arctan2) is a direction again, e.g. a per-member circular mean.
            typ = Type(NUM, t.dims, unit, False, t.time, direction=name == "rad2deg")
        elif name == "sqrt":
            t = parts[0].type
            typ = Type(NUM, t.dims, None if t.unit is None else t.unit ** 0.5, False, t.time)
        else:
            t = parts[0].type
            typ = Type(NUM, t.dims, t.unit, t.state, t.time)
        return Elementwise(NUMPY[name], parts, typ)

    def top(self, node: ast.Call, largest: bool, args, kwargs) -> Node:
        name = "top" if largest else "bottom"
        self.only(node, kwargs, ("n", "dim", "by"))
        if not 1 <= len(args) <= 3:
            raise _error(node, f"{name}(value, n, dim, by='field')")
        x = self.expr(args[0], record_ok=True)
        n_node = args[1] if len(args) > 1 else kwargs.get("n")
        n = _integer(n_node, "n") if n_node is not None else 10
        if not 1 <= n <= MAX_TOP:
            raise _error(node, f"n must lie between 1 and {MAX_TOP}")
        dim_node = args[2] if len(args) > 2 else kwargs.get("dim")
        if x.type.kind == RECORD:
            fields = dict(x.type.fields)
            by = _string(kwargs["by"], "by") if "by" in kwargs else next(iter(fields))
            if by not in fields:
                raise _error(node, f"by= must name a field: {', '.join(fields)}")
            key_type = fields[by]
            types = list(fields.values())
        else:
            if "by" in kwargs:
                raise _error(node, "by= only applies to dicts")
            by, key_type, types = None, x.type, [x.type]
        ranked = {tuple(d for d in t.dims if d != QUANTILE) for t in types}
        if len(ranked) != 1 or len(next(iter(ranked))) != 1 or key_type.has(QUANTILE):
            raise _error(node, f"{name}() ranks along exactly one remaining dimension; reduce the others first "
                               f"(e.g. .mean(\"member\").min(\"model\")). Got {x.type.describe()}")
        dim = next(iter(ranked))[0]
        if dim_node is not None and _string(dim_node, "dim") != dim:
            raise _error(node, f"{name}(): the value's remaining dimension is {dim}")

        def key_of(ctx, value, by=by):
            return value.fields[by] if isinstance(value, rt.Rec) else value
        return Rank(x, key_of, dim, largest, n, x.type, (x,))

    # -- locations

    def spec(self, node: ast.Call, kind: str, args, kwargs) -> tuple[str, tuple]:
        if kind == "places":
            entries = []
            for a in args:
                if isinstance(a, ast.Dict):
                    entries += [(_string(k, "place name"), _pair(v, "place")) for k, v in zip(a.keys, a.values)]
                else:
                    text = _string(a, "places")
                    for part in filter(None, (p.strip() for p in text.split(";"))):
                        name, at, coords = part.rpartition("@")
                        try:
                            lat, lon = (float(c) for c in coords.split(","))
                        except ValueError:
                            raise _error(a, f"places must look like 'Name@lat,lon; Other@lat,lon', got {part!r}") \
                                from None
                        entries.append((" ".join(name.split()) if at else coords, (lat, lon)))
            entries += [(k, _pair(v, "place")) for k, v in kwargs.items()]
            if not entries:
                raise _error(node, "places(Köln=(50.94, 6.96), Bonn=(50.73, 7.10)) or places('Köln@50.94,6.96; …')")
            return "points", tuple((name, lat, lon) for name, (lat, lon) in entries)
        self.only(node, kwargs, ("polyline", "start", "speed_kmh", "use_gpx_times", "gpx"))
        source = kwargs.get("polyline", kwargs.get("gpx", args[0] if args else None))
        if isinstance(source, ast.Name) and source.id == "gpx":
            if not self.env.has_gpx:
                raise _error(node, "route(gpx, …) refers to the tool's gpx argument, which wasn't given")
            track = ("gpx",)
        elif source is not None:
            track = ("polyline", _string(source, "polyline"))
        else:
            raise _error(node, "route(polyline=\"…\", start=\"…\", speed_kmh=…) or route(gpx, start=\"…\", …)")
        if "start" not in kwargs:
            raise _error(node, "route(…, start=\"2026-10-10T09:00\"): the departure time")
        start, _ = _parse_time(kwargs["start"], self.env.tz)
        speed = _number(kwargs["speed_kmh"], "speed_kmh") if "speed_kmh" in kwargs else None
        use_times = kwargs.get("use_gpx_times")
        use_times = isinstance(use_times, ast.Constant) and use_times.value is True
        if speed is None and not use_times:
            raise _error(node, "route(…): give speed_kmh=… or use_gpx_times=True")
        return "route", track + (start, speed, use_times)
