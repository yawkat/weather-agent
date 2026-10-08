"""Forecaster around the query language: downloads, failing sources, validation and limits (synthetic sources)."""

from datetime import datetime, timezone

import numpy as np
import pytest

from weather_core.budget import DownloadBudget
from weather_core.evaluate import SourceInfo
from weather_core.forecast import Forecaster, QueryError
from weather_core.sources.base import Prepared

RUN = datetime(2026, 10, 7, 0, tzinfo=timezone.utc)
MEMBERS = 10
STEPS = list(range(0, 73))  # hourly to +72 h
DRY = ('(forecast().interp(lat=50.0, lon=7.0).sel(time=slice("2026-10-08T10:00", "2026-10-08T12:00"))'
       '.precip.sum("time") < 0.5).mean("member")')


class Synthetic:
    """Member m rains m × 0.1 mm/h everywhere; t2m = 10 + 0.5 m + 0.1 × hour."""

    def __init__(self, name="synthetic"):
        self.name = name

    def provides(self):
        return {"precip", "t2m"}

    def describe(self):
        return {"resolution": "test", "note": "synthetic"}

    def prepare(self, query, variables):
        m = np.arange(MEMBERS)[:, None] * np.ones((1, len(query.lat)))
        per_step = {}
        if "precip" in variables:
            per_step["precip"] = {i: m * 0.1 for i in range(len(STEPS))}
        if "t2m" in variables:
            per_step["t2m"] = {i: 10 + 0.5 * m + 0.1 * STEPS[i] for i in range(len(STEPS))}
        info = SourceInfo(self.name, self.name.title(), "test", RUN, MEMBERS)
        return Prepared(query, RUN, STEPS, per_step, {"precip"}, set(), info, 0,
                        lambda times: {"provider": "test", "model": self.name})


class Broken:
    name = "broken"

    def provides(self):
        return {"precip", "t2m"}

    def describe(self):
        return {}

    def prepare(self, query, variables):
        raise RuntimeError("connection reset")


class Downloading(Synthetic):
    """Reserves `nbytes` from a download budget before answering, like a source with data to fetch."""

    def __init__(self, name, budget, nbytes):
        super().__init__(name)
        self.budget = budget
        self.nbytes = nbytes

    def prepare(self, query, variables):
        self.budget.reserve(self.nbytes)
        return super().prepare(query, variables)


def forecaster(*sources, **kwargs):
    return Forecaster(list(sources) or [Synthetic()], default_tz="UTC", clock=lambda: RUN, **kwargs)


def by_model(answer):
    return dict(answer["result"]["rows"])


def test_sources_of_one_query_share_the_per_query_download_limit():
    budget = DownloadBudget(per_request_bytes=100, per_hour_bytes=10**6)
    f = forecaster(Downloading("a", budget, 60), Downloading("b", budget, 60), budget=budget)
    answer = f.forecast(DRY)
    assert [m["model"] for m in answer["models"]] == ["a"]
    assert answer["unavailable_sources"][0]["source"] == "b"
    assert "per-query limit" in answer["unavailable_sources"][0]["reason"]
    # The next query has its own allowance.
    assert [m["model"] for m in f.forecast(DRY)["models"]] == ["a"]


def test_two_windows_read_only_their_hours():
    class Recording(Synthetic):
        def prepare(self, query, variables):
            self.query = query
            return super().prepare(query, variables)

    source = Recording()
    point = 'forecast().interp(lat=50.0, lon=7.0)'
    answer = forecaster(source).forecast(
        f'{{"sat": {point}.sel(time=slice("2026-10-08T10:00", "2026-10-08T12:00")).precip.sum("time").mean("member"), '
        f'"sun": {point}.sel(time=slice("2026-10-09T10:00", "2026-10-09T12:00")).t2m.max("time").mean("member")}}')
    assert len(source.query.hours) == 26  # one axis from the first window's start to the second's end
    assert source.query.wanted.tolist() == [True] * 2 + [False] * 22 + [True] * 2
    row = answer["result"]["rows"][0]
    assert row[1] == pytest.approx(0.9)  # 2 h × the mean member's 0.45 mm/h
    assert row[2] == pytest.approx(10 + 0.5 * 4.5 + 0.1 * 59)  # +59 h: the second window's last hour


def test_failing_source_does_not_fail_the_answer():
    answer = forecaster(Broken(), Synthetic()).forecast(DRY)
    assert by_model(answer) == {"synthetic": 0.3}
    assert answer["unavailable_sources"] == [{"source": "broken", "reason": "failed (RuntimeError); see server log"}]
    with pytest.raises(QueryError, match="no model could provide data"):
        forecaster(Broken()).forecast(DRY)


def test_answers_carry_models_and_attribution():
    answer = forecaster(Synthetic("a"), Synthetic("b")).forecast(DRY)
    assert [(m["model"], m["run"], m["members"], m["lead_hours"]) for m in answer["models"]] == [
        ("a", "2026-10-07T00:00Z", 10, [34, 35]), ("b", "2026-10-07T00:00Z", 10, [34, 35])]
    assert answer["attribution"] == [{"provider": "test", "model": "a"}, {"provider": "test", "model": "b"}]


@pytest.mark.parametrize("start, end", [("0001-01-01T00:00", "0001-01-02T00:00"),
                                        ("2026-11-30T00:00", "2026-12-01T00:00"),
                                        ("2026-10-01T00:00", "2026-10-02T00:00"),
                                        ("2026-10-07T00:00", "2026-10-25T00:00")])
def test_times_outside_the_forecast_span_are_rejected(start, end):
    with pytest.raises(ValueError):
        forecaster().forecast(DRY.replace("2026-10-08T10:00", start).replace("2026-10-08T12:00", end))


@pytest.mark.parametrize("location", [
    "interp(lat=95.0, lon=7.0)",
    "interp(places(A=(50, 7), B=(95, 7)))",
    "interp(places('A@50,7; A@51,7'))",  # duplicate names
    "interp(places(" + ", ".join(f"p{i}=(50, 7)" for i in range(21)) + "))",
    "sel(lat=slice(-90, 90), lon=slice(-180, 180))",
])
def test_locations_are_validated(location):
    with pytest.raises(ValueError):
        forecaster().forecast(DRY.replace("interp(lat=50.0, lon=7.0)", location).replace(
            '.mean("member")', '.max(["member", "point"])' if "places" in location else '.mean("member")'))


def test_busy_server_refuses_instead_of_queueing_forever():
    f = forecaster(max_concurrent=1, busy_timeout_s=0)
    with f._evaluating():
        with pytest.raises(QueryError, match="busy"):
            f.forecast(DRY)
    assert by_model(f.forecast(DRY)) == {"synthetic": 0.3}
