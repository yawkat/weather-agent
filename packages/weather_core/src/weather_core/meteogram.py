"""Per-member time series at one point, for meteogram charts, and the daily summary the LLM reads with them."""

import math
import warnings
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import date, datetime, tzinfo

import numpy as np

from .sources.base import SourceSamples
from .variables import CATALOG

DEFAULT_VARIABLES = ("t2m", "precip", "wind", "gust", "cloud")
MAX_VARIABLES = 6
# Wind direction needs arrows rather than lines, and route variables need a route.
CHARTABLE = frozenset(n for n, v in CATALOG.items() if not v.route_only and n != "wind_dir")
# Decimals per variable in the chart data; the rest get one.
_DECIMALS = {"precip": 2, "snow": 2}
# Daily statistics per member for the text summary: min/max of the day, total (interval rates × hours), max, mean.
_DAILY = {
    "t2m": ("min", "max"), "td2m": ("min", "max"), "feels_like": ("min", "max"),
    "precip": ("total",), "snow": ("total",),
    "wind": ("max",), "gust": ("max",), "cape": ("max",),
    "cloud": ("mean",), "rh": ("mean",), "radiation": ("mean",),
}
QUANTILES = (0.1, 0.5, 0.9)


def parse_variables(text: str | None) -> list[str]:
    if not text:
        return list(DEFAULT_VARIABLES)
    names = list(dict.fromkeys(n.strip() for n in text.split(",") if n.strip()))
    unknown = [n for n in names if n not in CHARTABLE]
    if unknown:
        raise ValueError(f"unknown or unchartable variables {unknown}; choose from {sorted(CHARTABLE)}")
    if not names:
        raise ValueError("variables is empty")
    if len(names) > MAX_VARIABLES:
        raise ValueError(f"at most {MAX_VARIABLES} variables")
    return names


@dataclass
class Series:
    """One model's members along a window, split at its model steps.

    Instant variables have a value at every boundary (model steps and the window edges); interval variables
    (rates, gust maxima) have one value per piece between consecutive boundaries.
    """
    boundaries: list[datetime]
    instant: dict[str, np.ndarray]  # [member, boundary]
    interval: dict[str, np.ndarray]  # [member, piece]

    @property
    def piece_hours(self) -> np.ndarray:
        return np.array([(b - a).total_seconds() / 3600 for a, b in zip(self.boundaries, self.boundaries[1:])])


def series(samples: SourceSamples, interval_vars: Collection[str]) -> Series:
    """Undo the half-interval sampling of a window (timeaxis.sample_window): two samples per piece, at its ends."""
    n = len(samples.times)
    if n % 2:
        raise ValueError("window samples come in pairs")
    ends = list(range(1, n, 2))
    boundaries = [samples.times[0]] + [samples.times[i] for i in ends]
    instant, interval = {}, {}
    for name, values in samples.samples.variables.items():
        if name not in CHARTABLE:
            continue
        if name in interval_vars:
            interval[name] = values[:, 0::2]
        else:
            instant[name] = values[:, [0] + ends]
    return Series(boundaries, instant, interval)


def chart_values(values: np.ndarray, name: str) -> list[list[float | None]]:
    """[member, time] → nested lists, rounded, NaN as null."""
    decimals = _DECIMALS.get(name, 1)
    rounded = np.round(np.asarray(values, dtype=np.float64), decimals)
    return [[None if math.isnan(x) else x for x in row] for row in rounded.tolist()]


def daily_summary(s: Series, variables: Sequence[str], tz: tzinfo) -> list[dict]:
    """Per local day: covered hours and, per variable, member quantiles (10/50/90 %) of daily statistics."""
    hours = s.piece_hours
    piece_days = [_local_day(a + (b - a) / 2, tz) for a, b in zip(s.boundaries, s.boundaries[1:])]
    boundary_days = [_local_day(t, tz) for t in s.boundaries]
    out = []
    for day in sorted(set(piece_days)):
        pieces = np.array([d == day for d in piece_days])
        points = np.array([d == day for d in boundary_days])
        entry: dict = {"date": day.isoformat(), "hours": round(float(hours[pieces].sum()), 1)}
        for name in variables:
            for stat in _DAILY.get(name, ()):
                per_member = _daily(s, name, stat, pieces, points, hours)
                if per_member is None or np.all(np.isnan(per_member)):
                    continue
                entry[f"{name}_{stat}"] = _quantiles(per_member, _DECIMALS.get(name, 1))
                if name == "precip" and stat == "total":
                    valid = per_member[~np.isnan(per_member)]
                    entry["p_precip_1mm"] = round(float(np.mean(valid >= 1.0)), 2)
        out.append(entry)
    return out


def _daily(s: Series, name: str, stat: str, pieces: np.ndarray, points: np.ndarray,
           hours: np.ndarray) -> np.ndarray | None:
    if name in s.interval:
        values, weights = s.interval[name][:, pieces], hours[pieces]
    elif name in s.instant:
        if stat == "total":
            return None
        values, weights = s.instant[name][:, points], None
    else:
        return None
    if values.shape[1] == 0:
        return None
    # All-NaN members (a variable this model lacks) give NaN; numpy warns about them.
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        if stat == "total":
            return np.nansum(values * weights, axis=1)
        if stat == "min":
            return np.nanmin(values, axis=1)
        if stat == "max":
            return np.nanmax(values, axis=1)
        if weights is not None:
            return np.nansum(values * weights, axis=1) / weights.sum()
        return np.nanmean(values, axis=1)


def _quantiles(values: np.ndarray, decimals: int) -> list[float]:
    q = np.nanquantile(values, QUANTILES)
    return [round(float(x), decimals) for x in q]


def _local_day(t: datetime, tz: tzinfo) -> date:
    return t.astimezone(tz).date()
