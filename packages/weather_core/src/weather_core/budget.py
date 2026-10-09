"""Limits on upstream download volume, so no single request or burst of requests can pull unbounded data."""

import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from .sources.base import SourceError


class BudgetExceeded(SourceError):
    """A download would exceed a limit; another run of the same model wouldn't fit either."""


@dataclass
class Reservation:
    time: float
    nbytes: int
    query: list[int] | None  # running total of the query it was made in


class DownloadBudget:
    """A per-query limit (summed over every source a query uses) and a rolling hourly limit.

    Sources reserve an estimate before downloading and settle the actual size afterwards. Reservations made inside
    `query()` count towards that query's limit; outside one, each reservation is checked on its own.
    """

    def __init__(self, per_request_bytes: int, per_hour_bytes: int, clock=time.monotonic):
        self.per_request_bytes = per_request_bytes
        self.per_hour_bytes = per_hour_bytes
        self.clock = clock
        self._lock = threading.Lock()
        self._recent: deque[Reservation] = deque()
        self._query: ContextVar[list[int] | None] = ContextVar("download_query", default=None)
        self._background: ContextVar[bool] = ContextVar("download_background", default=False)

    @contextmanager
    def query(self) -> Iterator[None]:
        """Scope of one query: its reservations share the per-query limit."""
        token = self._query.set([0])
        try:
            yield
        finally:
            self._query.reset(token)

    @contextmanager
    def background(self) -> Iterator[None]:
        """Scope of work clients don't drive (prefetch): its downloads are neither limited nor counted."""
        token = self._background.set(True)
        try:
            yield
        finally:
            self._background.reset(token)

    def reserve(self, nbytes: int) -> Reservation:
        """Account for a download before it starts; raises BudgetExceeded if it would exceed a limit."""
        if self._background.get():
            return Reservation(self.clock(), nbytes, None)
        used_by_query = self._query.get()
        before = used_by_query[0] if used_by_query is not None else 0
        if before + nbytes > self.per_request_bytes:
            already = f" ({before / 1e6:.0f} MB already for other models)" if before else ""
            raise BudgetExceeded(f"this query would download {(before + nbytes) / 1e6:.0f} MB{already}, more than the "
                              f"per-query limit of {self.per_request_bytes / 1e6:.0f} MB; use a shorter time range, "
                              f"fewer variables or fewer models")
        with self._lock:
            now = self._prune()
            used = sum(r.nbytes for r in self._recent)
            if used + nbytes > self.per_hour_bytes:
                raise BudgetExceeded(f"hourly download limit reached ({used / 1e6:.0f} of "
                                  f"{self.per_hour_bytes / 1e6:.0f} MB used); cached data is still served")
            reservation = Reservation(now, nbytes, used_by_query)
            self._recent.append(reservation)
            if used_by_query is not None:
                used_by_query[0] += nbytes
        return reservation

    def settle(self, reservation: Reservation, actual: int) -> None:
        """Replace an estimate by what was actually downloaded (never raises: the bytes are already here)."""
        with self._lock:
            if reservation.query is not None:
                reservation.query[0] += actual - reservation.nbytes
            reservation.nbytes = actual

    def used_last_hour(self) -> int:
        """Bytes counted against the hourly limit now."""
        with self._lock:
            self._prune()
            return sum(r.nbytes for r in self._recent)

    def _prune(self) -> float:
        now = self.clock()
        while self._recent and self._recent[0].time < now - 3600:
            self._recent.popleft()
        return now
