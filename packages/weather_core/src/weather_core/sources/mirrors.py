"""Fallback between hosts that serve the same files, e.g. data.ecmwf.int and its cloud copies.

The origin comes first and is used whenever it works. A host that fails (after the fetcher's own retries, e.g. on
HTTP 429) is skipped for a while, so a throttled origin isn't hammered by every query. Mirrors copy new runs a few
minutes after the origin, so a 404 from the origin is final, but one from a mirror may just be lag: it isn't held
against the mirror.
"""

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from urllib.parse import urlsplit

from .base import Download, Fetcher, NotPublished, SourceError

log = logging.getLogger(__name__)


class Mirrors:
    def __init__(self, origin: str, mirrors: Sequence[str] = (), cooldown_s: float = 300, clock=time.monotonic):
        bases = [base.rstrip("/") for base in (origin, *mirrors)]
        for base in bases:
            if not base.startswith("https://"):
                raise ValueError(f"base URL must use https: {base}")
        self.bases = bases
        self.origin = bases[0]
        self.cooldown_s = cooldown_s
        self.clock = clock
        self._failed_at: dict[str, float] = {}

    def _order(self) -> list[tuple[str, bool]]:
        """(base, cooling down) to try: working hosts in configured order, then the ones failing longest ago."""
        now = self.clock()
        cooling = [b for b in self.bases if b in self._failed_at and now - self._failed_at[b] < self.cooldown_s]
        cooling.sort(key=self._failed_at.get)  # stable: ties keep the configured order
        return [(b, False) for b in self.bases if b not in cooling] + [(b, True) for b in cooling]

    def _failed(self, base: str, error: Exception) -> None:
        self._failed_at[base] = self.clock()
        if len(self.bases) > 1:
            log.warning("%s failed (%s); preferring other hosts for %.0f s", urlsplit(base).netloc, error,
                        self.cooldown_s)

    def get(self, fetcher: Fetcher, path: str) -> bytes | None:
        """Body of `path` (relative to the base URLs), or None if it isn't published. After a mirror's 404 the
        other working hosts are asked; hosts cooling down are only tried if none of them answered."""
        missing = False
        error: Exception | None = None
        for base, cooling in self._order():
            if cooling and missing:
                break
            try:
                body = fetcher.get(f"{base}/{path}")
            except SourceError as e:
                self._failed(base, e)
                error = e
                continue
            if body is not None or base == self.origin:
                return body
            missing = True
        if missing:
            return None
        raise error

    def download_many(self, fetcher: Fetcher, downloads: list[Download],
                      reserve: Callable[[], None] | None = None) -> list[int]:
        """`fetcher.download_many` for downloads whose URLs are relative to the base URLs. If a host fails, the
        next one fetches the whole batch again (destination files are rewritten from the start), after `reserve`
        accounts for the repeat; it raises to stop. Callers only ask for files an index listed, so after a
        mirror's 404 (lag) every other host is tried, including ones cooling down."""
        error: Exception | None = None
        for attempt, (base, _) in enumerate(self._order()):
            if attempt and reserve is not None:
                try:
                    reserve()
                except SourceError as e:
                    raise e from error
            try:
                return fetcher.download_many([replace(d, url=f"{base}/{d.url}") for d in downloads])
            except NotPublished as e:
                if base == self.origin:
                    raise
                error = e
            except SourceError as e:
                self._failed(base, e)
                error = e
        raise error
