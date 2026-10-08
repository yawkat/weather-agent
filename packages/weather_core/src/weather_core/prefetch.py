"""Keeping models warm: fetch every step of each listed model's newest run for a fixed set of variables, so queries
for them don't wait for downloads (a cold query takes up to a minute).

Downloads don't depend on where a query is (fields are whole: ECMWF's cropped to Europe, DWD's whole domain), only
on the run, steps and parameters, and agents ask for the same few variables. So there is nothing to learn from
queries: a pass walks each model's forecast hours from now to its last step, in chunks of `CHUNK`, soonest first,
and fetches what isn't cached yet. Passes aren't driven by clients and their volume is fixed by configuration, so
they don't count against the download budget.
"""

import logging
import time
import traceback
from collections.abc import Collection, Sequence
from contextlib import nullcontext
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
CHUNK = timedelta(hours=12)  # one fetch: bounds the temporary files of a download batch
# Inside every model's domain (ICON-D2 covers Germany and its neighbours); downloads don't depend on the point.
POINT = (51.0, 10.0)
VARIABLES = ("precip", "t2m", "td2m", "wind", "gust", "cloud")


class FetchingSource(Protocol):
    name: str

    def fetch(self, query: Query, variables: Collection[str]) -> int: ...

    def provides(self) -> set[str]: ...

    def max_lead_hours(self) -> int: ...


class Prefetcher:
    def __init__(self, sources: Sequence[FetchingSource], models: Collection[str],
                 variables: Collection[str] = VARIABLES, budget: DownloadBudget | None = None,
                 clock=lambda: datetime.now(timezone.utc)):
        by_name = {s.name: s for s in sources}
        unknown = set(models) - by_name.keys()
        if unknown:
            raise ValueError(f"unknown models to prefetch: {sorted(unknown)}")
        self.sources = [by_name[name] for name in models]
        self.variables = tuple(sorted(variables))
        self.budget = budget
        self.clock = clock

    def refresh(self) -> int:
        """One pass; bytes downloaded. A failing chunk is logged and skipped (the next pass tries again)."""
        started = time.monotonic()
        downloaded = 0
        chunks = self._chunks()
        failed: dict[str, int] = {}
        for source, query, variables in chunks:
            try:
                with self.budget.background() if self.budget is not None else nullcontext():
                    downloaded += source.fetch(query, variables)
            except _EXPECTED as e:
                log.debug("prefetch %s: %s", source.name, e)  # e.g. a run still uploading
                failed[source.name] = failed.get(source.name, 0) + 1
            except BaseException as e:  # Java exceptions (network, decoding) arrive as foreign exceptions
                if isinstance(e, _FATAL):
                    raise
                log.error("prefetch %s failed: %s: %s\n%s", source.name, type(e).__name__, e, traceback.format_exc())
                failed[source.name] = failed.get(source.name, 0) + 1
        if downloaded or failed:
            # Skips alone repeat every pass: only worth a line with downloads.
            (log.info if downloaded else log.debug)("prefetch: %d chunks, %.1f MB downloaded%s in %.1f s",
                                                    len(chunks), downloaded / 1e6,
                                                    f", skipped {failed}" if failed else "",
                                                    time.monotonic() - started)
        return downloaded

    def _chunks(self) -> list[tuple[FetchingSource, Query, tuple[str, ...]]]:
        """Every hour from now to each model's last step, soonest first. Hours past a run's end are only marked
        invalid, so a chunk takes the newest run that reaches into it."""
        first = self.clock().replace(minute=0, second=0, microsecond=0)
        out = []
        for source in self.sources:
            variables = tuple(v for v in self.variables if v in source.provides())
            end = first + timedelta(hours=source.max_lead_hours())
            start = first
            while start < end and variables:
                n = int(min(CHUNK, end - start) / timedelta(hours=1))
                hours = [start + timedelta(hours=i) for i in range(n)]
                out.append((source, Query(np.array([POINT[0]]), np.array([POINT[1]]), hours=hours), variables))
                start += CHUNK
        out.sort(key=lambda chunk: chunk[1].hours[0])  # every model's next hours before anyone's day 10
        return out
