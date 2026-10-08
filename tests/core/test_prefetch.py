"""Prefetcher: every step of each listed model's newest run, chunked, outside the download budget."""

from datetime import datetime, timedelta, timezone

import pytest

from weather_core.budget import DownloadBudget
from weather_core.prefetch import Prefetcher
from weather_core.sources.base import SourceError

NOW = datetime(2026, 10, 8, 9, 30, tzinfo=timezone.utc)
FIRST = datetime(2026, 10, 8, 9, tzinfo=timezone.utc)
H = timedelta(hours=1)


class Fetching:
    def __init__(self, name, lead_hours=30, nbytes=5, budget=None, error=None):
        self.name, self.lead_hours, self.nbytes, self.budget, self.error = name, lead_hours, nbytes, budget, error
        self.calls = []

    def provides(self):
        return {"precip", "t2m"}

    def max_lead_hours(self):
        return self.lead_hours

    def fetch(self, query, variables):
        self.calls.append((query.hours[0], len(query.hours), variables))
        if self.error is not None:
            raise self.error
        if self.budget is not None:
            self.budget.reserve(self.nbytes)
        return self.nbytes


def prefetcher(sources, models, **kwargs):
    return Prefetcher(sources, models, clock=lambda: NOW, **kwargs)


def test_models_are_fetched_in_chunks_up_to_their_last_step():
    a = Fetching("a", lead_hours=30)
    prefetcher([a, Fetching("b")], ["a"], variables=["precip", "t2m", "cape"]).refresh()
    assert a.calls == [(FIRST, 12, ("precip", "t2m")), (FIRST + 12 * H, 12, ("precip", "t2m")),
                       (FIRST + 24 * H, 6, ("precip", "t2m"))]


def test_every_models_next_hours_come_first():
    order = []

    class Recording(Fetching):
        def fetch(self, query, variables):
            order.append((self.name, query.hours[0]))
            return 0

    prefetcher([Recording("long", lead_hours=36), Recording("short", lead_hours=12)], ["long", "short"]).refresh()
    assert order == [("long", FIRST), ("short", FIRST), ("long", FIRST + 12 * H), ("long", FIRST + 24 * H)]


def test_failing_chunks_are_skipped():
    a, b = Fetching("a", error=SourceError("no published run covers this time")), Fetching("b")
    assert prefetcher([a, b], ["a", "b"]).refresh() == 15
    assert len(a.calls) == 3 and len(b.calls) == 3


def test_prefetch_isnt_limited_by_the_download_budget():
    budget = DownloadBudget(per_request_bytes=100, per_hour_bytes=250)
    a = Fetching("a", nbytes=1000, budget=budget)
    assert prefetcher([a], ["a"], budget=budget).refresh() == 3000
    budget.reserve(100)  # clients' allowance is untouched


def test_unknown_models_are_rejected():
    with pytest.raises(ValueError, match="unknown models"):
        prefetcher([Fetching("a")], ["icon-d2-ruc"])
