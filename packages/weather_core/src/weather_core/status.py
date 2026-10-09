"""What agents should know before querying: which models there are, how far each reaches, what is cached and what
an uncached query costs. Rendered as text for the help tool; reading it does no I/O beyond listing the cache."""

import logging
import threading
import traceback
from collections import deque
from collections.abc import Collection, Sequence
from datetime import datetime, tzinfo
from typing import Protocol

from .budget import DownloadBudget
from .variables import CATALOG

log = logging.getLogger(__name__)

KEEP = 20  # downloads per source the estimate averages over


class DownloadStats:
    """Recent downloads per source, as reported by the sources' on_download (sources.base.OnDownload)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._recent: dict[str, deque[tuple[int, int, float]]] = {}

    def record(self, source: str, run: datetime, nbytes: int, fields: int, seconds: float) -> None:
        if fields <= 0:
            return
        with self._lock:
            self._recent.setdefault(source, deque(maxlen=KEEP)).append((nbytes, fields, seconds))

    def per_field(self, source: str) -> tuple[float, float] | None:
        """Average (bytes, seconds) of one field (a parameter at one step, every member), or None if unmeasured."""
        with self._lock:
            recent = list(self._recent.get(source, ()))
        fields = sum(f for _, f, _ in recent)
        if not fields:
            return None
        return sum(b for b, _, _ in recent) / fields, sum(s for _, _, s in recent) / fields


class StatusSource(Protocol):
    name: str

    def status(self) -> dict: ...


def describe_models(sources: Sequence[StatusSource], now: datetime, tz: tzinfo, stats: DownloadStats | None = None,
                    budget: DownloadBudget | None = None, warm: Collection[str] = (),
                    warm_variables: Collection[str] = ()) -> str:
    """The models section of the query help. `warm` are the models prefetch keeps cached, for `warm_variables`."""
    lines = [f"Models now ({_time(now, tz)}, {tz}). A query uses each model's newest published run that covers its "
             "hours; what isn't cached is downloaded first, which takes seconds to minutes. The cache only "
             "decides speed, not quality: an uncached model that suits the question (e.g. icon-d2-ruc-eps for the "
             "next hours) is often worth the wait."]
    if budget is not None:
        lines.append(f"Download limits: {budget.per_request_bytes / 1e6:.0f} MB per query, "
                     f"{budget.per_hour_bytes / 1e6:.0f} MB per hour ({budget.used_last_hour() / 1e6:.0f} MB used "
                     "in the last hour). Cached data is always served.")
    for source in sources:
        try:
            s = source.status()
        except BaseException as e:  # Java exceptions (file system) arrive as foreign exceptions
            if isinstance(e, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                raise
            # The help around this section must still answer.
            log.error("status of %s failed: %s: %s\n%s", source.name, type(e).__name__, e, traceback.format_exc())
            lines += ["", f"{source.name}: status unavailable (see server log)."]
            continue
        lines.append("")
        lines.append(f"{s['model']}: {s['name']} ({s['provider']}), {s['members']} members, {s['resolution']}; "
                     f"{s['domain']}.")
        lines.append(f"  {s['note']}")
        lines.append(f"  Schedule: {s['schedule']}.")
        lacks = [v for v in CATALOG if v not in s["variables"]]
        if lacks:
            lines.append(f"  Lacks: {', '.join(lacks)}.")
        lines += _runs(s["runs"], now, tz, s["stale_after_hours"])
        kept = sorted(set(warm_variables) & set(s["variables"])) if s["model"] in warm else []
        if kept:
            lines.append(f"  Kept cached in the background: {', '.join(kept)} for every new run.")
        per_field = stats.per_field(s["model"]) if stats is not None else None
        if per_field is not None:
            nbytes, seconds = per_field
            lines.append(f"  Uncached: about {nbytes / 1e6:.0f} MB and {_seconds(seconds)} per variable and model "
                         f"step ({s['steps_per_day']} steps a day early on; wind needs two fields), measured.")
    return "\n".join(lines)


def _runs(runs: list[dict], now: datetime, tz: tzinfo, stale_after_hours: float) -> list[str]:
    if not runs:
        return ["  Nothing cached: the first query downloads everything it selects."]
    newest = runs[0]["run"]
    age = (now - newest).total_seconds() / 3600
    if age > stale_after_hours:
        return [f"  Newest cached run: {newest:%Y-%m-%d %H:%M} UTC ({age:.0f} h ago). Newer runs are likely published: "
                "queries download what they select from those."]
    out = []
    # The newest run, and an older one only where it still reaches further (ECMWF's long 00/12 UTC runs).
    shown = [runs[0]] + [r for r in runs[1:] if r["reaches"] > runs[0]["reaches"]]
    for i, run in enumerate(shown):
        age = (now - run["run"]).total_seconds() / 3600
        label = "Newest cached run" if i == 0 else "Older cached run, reaching further"
        out.append(f"  {label}: {run['run']:%Y-%m-%d %H:%M} UTC ({age:.0f} h ago), reaches {_time(run['reaches'], tz)}.")
        by_until: dict[datetime | None, list[str]] = {}
        for variable in CATALOG:
            if variable in run["cached"]:
                by_until.setdefault(run["cached"][variable], []).append(variable)
        cached = sorted((until, names) for until, names in by_until.items() if until is not None)
        if not cached:
            out.append("    Nothing cached from now on.")
            continue
        parts = [f"{', '.join(names)} until {_time(until, tz)}" for until, names in reversed(cached)]
        if None in by_until:
            parts.append(f"not {', '.join(by_until[None])}")
        out.append(f"    Cached from now: {'; '.join(parts)}.")
    return out


def _time(t: datetime, tz: tzinfo) -> str:
    return t.astimezone(tz).strftime("%a %Y-%m-%d %H:%M")


def _seconds(s: float) -> str:
    return f"{s:.1f} s" if s < 10 else f"{s:.0f} s"
