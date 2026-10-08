"""Warm set (what recent answers needed) and the prefetcher that refreshes it from new runs."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from test_forecast import DRY, Broken, Synthetic
from weather_core.budget import DownloadBudget
from weather_core.forecast import Forecaster
from weather_core.prefetch import Prefetcher, WarmSet
from weather_core.sources.base import Query, SourceError

NOW = datetime(2026, 10, 8, 9, 30, tzinfo=timezone.utc)
H = timedelta(hours=1)


def hours(start, n, lat=50.0, lon=7.0):
    return Query(np.array([lat, lat + 1]), np.array([lon, lon]), hours=[start + i * H for i in range(n)])


def warm_set(now=None, **kwargs):
    clock = now if now is not None else [NOW]
    return WarmSet(["a", "b"], clock=lambda: clock[0], **kwargs), clock


def test_records_one_entry_per_model_place_variables_and_window():
    warm, _ = warm_set()
    start = datetime(2026, 10, 10, 10, tzinfo=timezone.utc)
    warm.record("a", hours(start, 8), ["t2m", "precip"])
    warm.record("a", hours(start, 8, lat=50.01), ["precip", "t2m"])  # same rough place: the same entry
    warm.record("b", hours(start, 8), ["precip"])
    warm.record("ruc", hours(start, 8), ["precip"])  # not kept warm
    due = warm.due()
    assert [(name, variables) for name, _, variables in due] == [("a", ("precip", "t2m")), ("b", ("precip",))]
    _, query, _ = due[0]
    assert query.lat.tolist() == [50.01] and query.hours[0] == start and query.hours[-1] == start + 7 * H


def test_windows_are_trimmed_to_the_hours_ahead_and_dropped_once_past_or_stale():
    warm, clock = warm_set(ttl=timedelta(days=2))
    today = datetime(2026, 10, 8, 8, tzinfo=timezone.utc)
    warm.record("a", hours(today, 4), ["precip"])  # 08–11
    warm.record("a", hours(today + 48 * H, 4), ["precip"])
    first = warm.due()
    assert [q.hours[0] for _, q, _ in first] == [today + H, today + 48 * H]  # from 09:00 on, soonest first
    assert len(first[0][1].hours) == 3
    clock[0] = NOW + 3 * H
    assert len(warm.due()) == 1  # the morning window is past
    clock[0] = NOW + 2 * 24 * H + H
    assert warm.due() == []  # nobody asked for two days


def test_routes_become_their_hours():
    warm, _ = warm_set()
    t0 = datetime(2026, 10, 9, 7, 40, tzinfo=timezone.utc)
    times = [t0 + timedelta(minutes=50 * i) for i in range(4)]  # 07:40 … 10:10
    route = Query(np.array([50.0] * 4), np.array([7.0] * 4), times=times, dt_hours=np.ones(4), bearing=np.zeros(4))
    warm.record("a", route, ["wind"])
    (_, query, _), = warm.due()
    assert query.hours[0] == t0.replace(minute=0) and query.hours[-1] == datetime(2026, 10, 9, 11, tzinfo=timezone.utc)


def test_separate_windows_of_one_axis_are_separate_entries():
    warm, _ = warm_set()
    start = datetime(2026, 10, 10, 13, tzinfo=timezone.utc)
    query = hours(start, 30)
    query = Query(query.lat, query.lon, hours=query.hours, wanted=np.array([True] * 6 + [False] * 18 + [True] * 6))
    warm.record("a", query, ["precip"])
    assert [(q.hours[0], len(q.hours)) for _, q, _ in warm.due()] == [(start, 6), (start + 24 * H, 6)]


def test_the_set_keeps_the_most_recently_asked_entries():
    warm, clock = warm_set(max_entries=2)
    start = datetime(2026, 10, 10, tzinfo=timezone.utc)
    for i in range(3):
        clock[0] = NOW + i * timedelta(minutes=1)
        warm.record("a", hours(start + i * H, 2), ["precip"])
    clock[0] = NOW + timedelta(minutes=3)
    warm.record("a", hours(start + H, 2), ["precip"])  # asked again: now the newest
    assert len(warm) == 2
    assert sorted(q.hours[0] for _, q, _ in warm.due()) == [start + H, start + 2 * H]


class Fetching:
    def __init__(self, name, budget=None, nbytes=0, error=None, lead_hours=30):
        self.name, self.budget, self.nbytes, self.error = name, budget, nbytes, error
        self.lead_hours = lead_hours
        self.calls = []

    def provides(self):
        return {"precip", "t2m"}

    def max_lead_hours(self):
        return self.lead_hours

    def fetch(self, query, variables):
        self.calls.append((query.hours[0], variables))
        if self.error is not None:
            raise self.error
        if self.budget is not None:
            self.budget.reserve(self.nbytes)
        return self.nbytes


def test_refresh_fetches_every_entry_and_skips_failures():
    warm, _ = warm_set()
    start = datetime(2026, 10, 10, tzinfo=timezone.utc)
    warm.record("a", hours(start, 2), ["precip"])
    warm.record("b", hours(start, 2), ["t2m"])
    warm.record("b", hours(start + 24 * H, 2), ["t2m"])
    a, b = Fetching("a", error=SourceError("no published run covers this time")), Fetching("b", nbytes=5)
    assert Prefetcher([a, b], warm).refresh() == 10
    assert len(a.calls) == 1 and len(b.calls) == 2


def test_refresh_leaves_budget_for_interactive_queries():
    budget = DownloadBudget(per_request_bytes=100, per_hour_bytes=250)
    warm, _ = warm_set()
    start = datetime(2026, 10, 10, tzinfo=timezone.utc)
    for i in range(3):
        warm.record("a", hours(start + 24 * i * H, 2), ["precip"])
    source = Fetching("a", budget, nbytes=100)
    assert Prefetcher([source], warm, budget, keep_free_bytes=100).refresh() == 100
    budget.reserve(100)  # an interactive query still fits


def test_answered_queries_are_recorded_for_the_models_that_answered():
    warm = WarmSet(["synthetic", "broken"], clock=lambda: NOW)
    f = Forecaster([Broken(), Synthetic()], default_tz="UTC", clock=lambda: NOW, warm=warm)
    f.forecast(DRY)
    (name, query, variables), = warm.due()
    assert name == "synthetic" and variables == ("precip",)
    assert query.hours[0] == datetime(2026, 10, 8, 10, tzinfo=timezone.utc) and len(query.hours) == 2


def test_fully_warm_models_are_fetched_in_chunks_up_to_their_last_step():
    warm, _ = warm_set()
    source = Fetching("a", lead_hours=30)
    Prefetcher([source, Fetching("b")], warm, full_models=["a"], full_variables=["precip", "t2m", "cape"]).refresh()
    first = NOW.replace(minute=0)
    assert source.calls == [(first, ("precip", "t2m")), (first + 12 * H, ("precip", "t2m")),
                            (first + 24 * H, ("precip", "t2m"))]


def test_unknown_fully_warm_models_are_rejected():
    warm, _ = warm_set()
    with pytest.raises(ValueError, match="unknown models"):
        Prefetcher([Fetching("a")], warm, full_models=["icon-d2-ruc"])
