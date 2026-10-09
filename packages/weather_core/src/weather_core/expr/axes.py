"""Static types of forecast expressions: dimensions, kinds and units.

Arrays have named dimensions, always kept in the order of DIMS (operations align by name, so the order is only a
convention). Their static type also carries a unit, whether the value is still a linear function of a state
variable (temperature, wind, …), which makes summing it over time suspicious, and whether it is a compass direction,
which reductions treat as a number on a line (as xarray does: 350° and 10° average to 180°, not 0°).
"""

from dataclasses import dataclass, field

MODEL = "model"
MEMBER = "member"
TIME = "time"
LAT = "lat"  # areas: a regular grid, from .sel(lat=slice(…), lon=slice(…))
LON = "lon"
POINT = "point"  # several points, from .interp(lat=("point", […]), …) or .interp(places(…))
HOUR = "hour"  # from groupby("time.hour")
BINS = "distance_km_bins"  # from groupby_bins("distance_km", …)
QUANTILE = "quantile"
DIMS = (MODEL, MEMBER, TIME, LAT, LON, POINT, HOUR, BINS, QUANTILE)

NUM = "number"
BOOL = "condition"
LABEL = "label"  # coordinate labels, from idxmax/idxmin
RECORD = "record"
DATASET = "dataset"
SPEC = "location"  # places(…) / route(…): only an argument of .interp

# Time axis flavours: hourly samples, resampled buckets ("3h", "1D"), or a route's rider positions.
HOURLY = "1h"
ROUTE = "route"


class ExprError(ValueError):
    """The client's query is invalid (syntax, names, types, selections) or exceeds a limit."""
    span: tuple[int, int, int, int] | None = None  # (lineno, col_offset, end_lineno, end_col_offset) it points to


@dataclass(frozen=True)
class Unit:
    """Product of symbols with integer powers, e.g. mm·h⁻¹. None-valued units (constants) adopt their partner's."""
    powers: tuple[tuple[str, int], ...] = ()
    unknown: bool = False

    @staticmethod
    def parse(text: str) -> "Unit":
        num, _, den = text.partition("/")
        powers: dict[str, int] = {}
        for part, sign in ((num, 1), (den, -1)):
            for symbol in filter(None, part.split("·")):
                power = 2 if symbol.endswith("²") else 1
                symbol = symbol.rstrip("²")
                powers[symbol] = powers.get(symbol, 0) + sign * power
        return Unit._of(powers)

    @staticmethod
    def _of(powers: dict[str, int], unknown: bool = False) -> "Unit":
        return Unit(tuple(sorted((s, p) for s, p in powers.items() if p)), unknown)

    def __mul__(self, other: "Unit") -> "Unit":
        powers = dict(self.powers)
        for s, p in other.powers:
            powers[s] = powers.get(s, 0) + p
        return Unit._of(powers, self.unknown or other.unknown)

    def __truediv__(self, other: "Unit") -> "Unit":
        return self * Unit(tuple((s, -p) for s, p in other.powers), other.unknown)

    def __pow__(self, n: float) -> "Unit":
        if n != int(n):
            return Unit((), True) if self.powers else self
        return Unit._of({s: p * int(n) for s, p in self.powers}, self.unknown)

    def __str__(self) -> str:
        if self.unknown:
            return "?"
        powers = sorted(self.powers, key=lambda sp: sp[0] == "h")  # hours last: °C·h, W·h/m²
        num = [s + ("" if p == 1 else _superscript(p)) for s, p in powers if p > 0]
        den = [s + ("" if p == -1 else _superscript(-p)) for s, p in powers if p < 0]
        if not num and not den:
            return ""
        return "·".join(num or ["1"]) + ("/" + "·".join(den) if den else "")


HOURS = Unit.parse("h")
NO_UNIT = Unit()


def _superscript(n: int) -> str:
    return str(n).translate(str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))


@dataclass(frozen=True)
class Type:
    kind: str  # NUM, BOOL, LABEL, RECORD or DATASET
    dims: tuple[str, ...] = ()
    unit: Unit | None = None  # None: a constant, which takes the unit of what it is combined with
    state: bool = False  # linear in a state variable: summing over time is likely a mistake
    time: str | None = None  # flavour of the time dimension, if any (HOURLY, "3h", "1D", ROUTE)
    fields: tuple[tuple[str, "Type"], ...] = ()  # records
    label_dim: str | None = None  # LABEL: which dimension's labels
    direction: bool = False  # a compass direction in ° (wind_dir): reducing it is suspicious; arithmetic drops this
    # The catalog variable the value still measures (precip, cloud, …): kept by selections and by reductions that
    # keep its meaning (mean, min, a rate's sum over time); arithmetic, spreads and counts drop it. Charts draw by it.
    measure: str | None = None

    def has(self, dim: str) -> bool:
        return dim in self.dims

    def describe(self) -> str:
        if self.kind == RECORD:
            return "{" + ", ".join(f"{k!r}: {t.describe()}" for k, t in self.fields) + "}"
        if self.kind == DATASET:
            return "a dataset (pick a variable, e.g. .t2m)"
        what = {NUM: "numbers", BOOL: "conditions", LABEL: f"{self.label_dim} labels"}[self.kind]
        return f"{what} over ({', '.join(self.dims)})" if self.dims else f"a single {what[:-1]}"


def ordered(dims) -> tuple[str, ...]:
    present = set(dims)
    return tuple(d for d in DIMS if d in present)


@dataclass
class Warnings:
    """Non-fatal findings, reported with the answer."""
    messages: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        if message not in self.messages:
            self.messages.append(message)
