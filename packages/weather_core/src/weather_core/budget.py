"""Limits on upstream download volume, so no single request or burst of requests can pull unbounded data."""

import threading
import time
from collections import deque

from .sources.base import SourceError


class DownloadBudget:
    def __init__(self, per_request_bytes: int, per_hour_bytes: int, clock=time.monotonic):
        self.per_request_bytes = per_request_bytes
        self.per_hour_bytes = per_hour_bytes
        self.clock = clock
        self._lock = threading.Lock()
        self._recent: deque[tuple[float, int]] = deque()

    def reserve(self, nbytes: int) -> None:
        """Account for a download before it starts; raises SourceError if it would exceed a limit."""
        if nbytes > self.per_request_bytes:
            raise SourceError(f"this query would download {nbytes / 1e6:.0f} MB, more than the per-query limit of "
                              f"{self.per_request_bytes / 1e6:.0f} MB; use a shorter window or fewer variables")
        with self._lock:
            now = self.clock()
            while self._recent and self._recent[0][0] < now - 3600:
                self._recent.popleft()
            used = sum(n for _, n in self._recent)
            if used + nbytes > self.per_hour_bytes:
                raise SourceError(f"hourly download limit reached ({used / 1e6:.0f} of "
                                  f"{self.per_hour_bytes / 1e6:.0f} MB used); cached data is still served")
            self._recent.append((now, nbytes))
