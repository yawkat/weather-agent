"""Keeping recently asked forecasts warm: once a model publishes a new run, fetch what recent queries needed from
it before anyone asks again. A cold query waits for its downloads, which takes minutes for ICON-EU-EPS.

The warm set has one entry per model, rough place, variables and time window that a query got an answer for in the
last `ttl`. It lives in memory only. Downloads depend only on the run, steps and parameters, not on the points, so
an entry keeps a single point (the query's first; it lies inside the domain of every model that answered) and its
hours. Entries are bounded in number and in hours (queries reach at most 16 days), and every refresh goes through
the download budget.

Models can also be kept fully warm (off by default): every step of their newest run for a fixed set of variables,
fetched in chunks of `FULL_CHUNK` so each stays within the per-query download limit.
"""

import logging
import threading
import time
import traceback
from collections.abc import Collection, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol

import numpy as np

from .budget import DownloadBudget
from .grid import OutsideRegion
from .sources.base import Query, SourceError
from .timeaxis import OutsideForecast

log = logging.getLogger(__name__)

_EXPECTED = (SourceError, OutsideForecast, OutsideRegion)
FULL_CHUNK = timedelta(hours=12)
# Inside every model's domain (ICON-D2 covers Germany and its neighbours); downloads don't depend on the point.
FULL_POINT = (51.0, 10.0)
_FATAL = (KeyboardInterrupt, SystemExit, GeneratorExit)


class FetchingSource(Protocol):
    name: str

    def fetch(self, query: Query, variables: Collection[str]) -> int: ...

    def provides(self) -> set[str]: ...

    def max_lead_hours(self) -> int: ...


@dataclass
class _Entry:
    lat: float
    lon: float
    variables: tuple[str, ...]
    start: datetime  # first and last whole hour
    end: datetime
    asked: datetime


class WarmSet:
    def __init__(self, models: Collection[str], max_entries: int = 64, ttl: timedelta = timedelta(days=2),
                 clock=lambda: datetime.now(timezone.utc)):
        self.models = set(models)
        self.max_entries = max_entries
        self.ttl = ttl
        self.clock = clock
        self._lock = threading.Lock()
        self._entries: dict[tuple, _Entry] = {}

    def record(self, source: str, query: Query, variables: Collection[str]) -> None:
        """Remember a query a model answered."""
        if source not in self.models or not variables or len(query.lat) == 0:
            return
        if query.is_route:
            end = _floor_hour(query.times[-1])
            end += timedelta(hours=1) if end < query.times[-1] else timedelta(0)
            windows = [(_floor_hour(query.times[0]), end)]
        else:
            windows = _windows(query.hours, query.wanted)
        lat, lon = float(query.lat[0]), float(query.lon[0])
        variables = tuple(sorted(variables))
        with self._lock:
            for start, end in windows:
                key = (source, round(lat, 1), round(lon, 1), variables, start, end)
                self._entries.pop(key, None)  # re-insert: dicts keep insertion order, oldest asked first
                self._entries[key] = _Entry(lat, lon, variables, start, end, self.clock())
            while len(self._entries) > self.max_entries:
                del self._entries[next(iter(self._entries))]

    def due(self) -> list[tuple[str, Query, tuple[str, ...]]]:
        """(model, query, variables) to keep warm, soonest first; windows are trimmed to the hours still ahead."""
        now = self.clock()
        first = _floor_hour(now)
        out = []
        with self._lock:
            for key, e in list(self._entries.items()):
                if e.end < first or e.asked < now - self.ttl:
                    del self._entries[key]
                    continue
                start = max(e.start, first)
                hours = [start + timedelta(hours=i) for i in range(int((e.end - start) / timedelta(hours=1)) + 1)]
                out.append((key[0], Query(np.array([e.lat]), np.array([e.lon]), hours=hours), e.variables))
        out.sort(key=lambda item: item[1].hours[0])
        return out

    def __len__(self) -> int:
        return len(self._entries)


class Prefetcher:
    """Fetches the warm set's fields, and everything of the fully warm models, from each model's newest run that
    covers them."""

    def __init__(self, sources: Sequence[FetchingSource], warm: WarmSet, budget: DownloadBudget | None = None,
                 keep_free_bytes: int = 0, full_models: Collection[str] = (),
                 full_variables: Collection[str] = ("precip", "t2m", "td2m", "wind", "gust", "cloud")):
        self.sources = {s.name: s for s in sources}
        self.warm = warm
        self.budget = budget
        self.keep_free_bytes = keep_free_bytes  # of the hourly download limit, for interactive queries
        unknown = set(full_models) - self.sources.keys()
        if unknown:
            raise ValueError(f"unknown models to keep fully warm: {sorted(unknown)}")
        self.full_models = list(full_models)
        self.full_variables = tuple(sorted(full_variables))

    def refresh(self) -> int:
        """One pass over the warm set, then the fully warm models; bytes downloaded. A failing entry is logged and
        skipped."""
        started = time.monotonic()
        downloaded = 0
        entries = self.warm.due() + self._full()
        failed: dict[str, int] = {}
        for name, query, variables in entries:
            source = self.sources.get(name)
            if source is None:
                continue
            try:
                with self.budget.query(self.keep_free_bytes) if self.budget is not None else nullcontext():
                    downloaded += source.fetch(query, variables)
            except _EXPECTED as e:
                # E.g. the budget, or no run covers the window any more; the reason names no place.
                log.debug("prefetch %s: %s", name, e)
                failed[name] = failed.get(name, 0) + 1
            except BaseException as e:  # Java exceptions (network, decoding) arrive as foreign exceptions
                if isinstance(e, _FATAL):
                    raise
                log.error("prefetch %s failed: %s: %s\n%s", name, type(e).__name__, e, traceback.format_exc())
                failed[name] = failed.get(name, 0) + 1
        if downloaded or failed:
            # Skips alone repeat every pass (e.g. while the budget is used up): only worth a line with downloads.
            (log.info if downloaded else log.debug)("prefetch: %d entries, %.1f MB downloaded%s in %.1f s",
                                                    len(entries), downloaded / 1e6,
                                                    f", skipped {failed}" if failed else "",
                                                    time.monotonic() - started)
        return downloaded

    def _full(self) -> list[tuple[str, Query, tuple[str, ...]]]:
        """Every hour from now to each fully warm model's last step, in chunks, soonest first. Hours past a run's
        end are only marked invalid, so the newest run's chunks need no more than it has."""
        first = _floor_hour(self.warm.clock())
        out = []
        for name in self.full_models:
            source = self.sources[name]
            variables = tuple(v for v in self.full_variables if v in source.provides())
            end = first + timedelta(hours=source.max_lead_hours())
            start = first
            while start < end and variables:
                n = int(min(FULL_CHUNK, end - start) / timedelta(hours=1))
                hours = [start + timedelta(hours=i) for i in range(n)]
                out.append((name, Query(np.array([FULL_POINT[0]]), np.array([FULL_POINT[1]]), hours=hours),
                            variables))
                start += FULL_CHUNK
        return out


def _floor_hour(t: datetime) -> datetime:
    return t.replace(minute=0, second=0, microsecond=0)


def _windows(hours: list[datetime], wanted: np.ndarray | None) -> list[tuple[datetime, datetime]]:
    """First and last hour of each run of wanted hours."""
    out: list[tuple[datetime, datetime]] = []
    for i, t in enumerate(hours):
        if wanted is not None and not wanted[i]:
            continue
        if out and out[-1][1] == t - timedelta(hours=1):
            out[-1] = (out[-1][0], t)
        else:
            out.append((t, t))
    return out
