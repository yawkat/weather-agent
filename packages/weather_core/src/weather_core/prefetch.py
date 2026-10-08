"""Keeping recently asked forecasts warm: once a model publishes a new run, fetch what recent queries needed from
it before anyone asks again. A cold query waits for its downloads, which takes minutes for ICON-EU-EPS.

The warm set has one entry per model, rough place, variables and time window that a query got an answer for in the
last `ttl`. It lives in memory only. Downloads depend only on the run, steps and parameters, not on the points, so
an entry keeps a single point (the query's first; it lies inside the domain of every model that answered) and its
hours. Entries are bounded in number and in hours (queries reach at most 16 days), and every refresh goes through
the download budget.
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
_FATAL = (KeyboardInterrupt, SystemExit, GeneratorExit)


class FetchingSource(Protocol):
    name: str

    def fetch(self, query: Query, variables: Collection[str]) -> int: ...


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
            start = _floor_hour(query.times[0])
            end = _floor_hour(query.times[-1])
            end += timedelta(hours=1) if end < query.times[-1] else timedelta(0)
        else:
            start, end = query.hours[0], query.hours[-1]
        lat, lon = float(query.lat[0]), float(query.lon[0])
        variables = tuple(sorted(variables))
        key = (source, round(lat, 1), round(lon, 1), variables, start, end)
        with self._lock:
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
    """Fetches the warm set's fields from each model's newest run that covers them."""

    def __init__(self, sources: Sequence[FetchingSource], warm: WarmSet, budget: DownloadBudget | None = None,
                 keep_free_bytes: int = 0):
        self.sources = {s.name: s for s in sources}
        self.warm = warm
        self.budget = budget
        self.keep_free_bytes = keep_free_bytes  # of the hourly download limit, for interactive queries

    def refresh(self) -> int:
        """One pass over the warm set; bytes downloaded. A failing entry is logged and skipped."""
        started = time.monotonic()
        downloaded = 0
        entries = self.warm.due()
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


def _floor_hour(t: datetime) -> datetime:
    return t.replace(minute=0, second=0, microsecond=0)
