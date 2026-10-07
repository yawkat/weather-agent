"""Forecaster end to end over synthetic sources, through the real DuckDB engine."""

from datetime import datetime, timezone

import numpy as np
import pytest
from weather_agent.sql_engine import DuckDbEngine

from weather_core.cube import QueryError, referenced_variables
from weather_core.evaluate import SourceInfo
from weather_core.forecast import Forecaster
from weather_core.sources.base import Prepared

RUN = datetime(2026, 10, 7, 0, tzinfo=timezone.utc)
MEMBERS = 10
STEPS = list(range(0, 73))  # hourly to +72 h


class Synthetic:
    """Member m rains m × 0.1 mm/h everywhere; t2m = 10 + 0.5 m + 0.1 × hour."""

    def __init__(self, name="synthetic", rain_scale=0.1):
        self.name = name
        self.rain_scale = rain_scale
        self.requested = None

    def provides(self):
        return {"precip", "t2m"}

    def describe(self):
        return {"resolution": "test", "note": "synthetic"}

    def prepare(self, query, variables):
        self.requested = sorted(variables)
        points = len(query.lat)
        m = np.arange(MEMBERS)[:, None] * np.ones((1, points))
        per_step = {}
        if "precip" in variables:
            per_step["precip"] = {i: m * self.rain_scale for i in range(len(STEPS))}
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


class _Forecaster(Forecaster):
    """Test shorthands for the common location forms."""

    def point(self, lat, lon, start, end, sql, **kwargs):
        return self.forecast(sql, start, end, lat=lat, lon=lon, **kwargs)


def forecaster(*sources):
    return _Forecaster(list(sources) or [Synthetic()], DuckDbEngine(512, 2, 10000, 500), default_tz="UTC",
                       clock=lambda: RUN)


def test_sources_of_one_query_share_the_per_query_download_limit():
    from weather_core.budget import DownloadBudget

    budget = DownloadBudget(per_request_bytes=100, per_hour_bytes=10**6)
    sources = [Downloading("a", budget, 60), Downloading("b", budget, 60)]
    f = _Forecaster(sources, DuckDbEngine(512, 2, 10000, 500), default_tz="UTC", clock=lambda: RUN, budget=budget)
    sql = "SELECT model, count(*) AS n FROM per_member GROUP BY model"
    answer = f.point(50.0, 7.0, "2026-10-08T10:00", "2026-10-08T12:00", sql)
    assert [m["model"] for m in answer["models"]] == ["a"]
    assert answer["unavailable_sources"][0]["source"] == "b"
    assert "per-query limit" in answer["unavailable_sources"][0]["reason"]
    # The next query has its own allowance.
    assert [m["model"] for m in f.point(50.0, 7.0, "2026-10-08T10:00", "2026-10-08T12:00", sql)["models"]] == ["a"]


def rows(answer):
    result = answer["result"]
    return [dict(zip(result["columns"], row)) for row in result["rows"]]


def test_joint_probability_per_model():
    # Two hours: member m gets 0.2 m mm, so rain < 0.5 for m = 0, 1, 2.
    answer = forecaster(Synthetic("a"), Synthetic("b", rain_scale=0.05)).point(
        50.0, 7.0, "2026-10-08T10:00", "2026-10-08T12:00",
        "SELECT model, prob(rain < 0.5) AS p FROM per_member GROUP BY model ORDER BY model")
    assert rows(answer) == [{"model": "a", "p": 0.3}, {"model": "b", "p": 0.5}]
    assert [m["model"] for m in answer["models"]] == ["a", "b"]
    assert answer["models"][0]["lead_hours"] == [34, 36]
    assert answer["models"][0]["note"] == "synthetic"


def test_outcome_breakdown_and_magnitude():
    answer = forecaster().point(50.0, 7.0, "2026-10-08T10:00", "2026-10-08T12:00", """
        SELECT CASE WHEN rain >= 0.5 THEN 'wet' ELSE 'dry' END AS outcome, share(model) AS p, max(rain) AS rain_max
        FROM per_member GROUP BY model, outcome ORDER BY outcome""")
    assert rows(answer) == [{"outcome": "dry", "p": 0.3, "rain_max": 0.4},
                            {"outcome": "wet", "p": 0.7, "rain_max": 1.8}]


def test_hourly_detail_in_local_time():
    f = _Forecaster([Synthetic()], DuckDbEngine(512, 2, 10000, 500), default_tz="Europe/Berlin", clock=lambda: RUN)
    answer = f.point(50.0, 7.0, "2026-10-08T10:00", "2026-10-08T12:00", """
        SELECT time, round(median(t2m), 2) AS t2m FROM samples GROUP BY time ORDER BY time""")
    times = [r["time"] for r in rows(answer)]
    assert times[0] == "2026-10-08T10:00" and times[-1] == "2026-10-08T12:00", rows(answer)
    # 10:00 CEST = 08:00 UTC = +32 h → 10 + 0.5 × 4.5 + 3.2
    assert rows(answer)[0]["t2m"] == pytest.approx(15.45)


def test_only_referenced_variables_are_fetched():
    source = Synthetic()
    forecaster(source).point(50.0, 7.0, "2026-10-08T10:00", "2026-10-08T12:00",
                             "SELECT model, prob(rain < 1) FROM per_member GROUP BY model")
    assert source.requested == ["precip"]
    assert referenced_variables("SELECT * FROM samples", {"t2m", "precip", "cape"}) == {"t2m", "precip"}


def test_failing_source_does_not_fail_the_answer():
    answer = forecaster(Broken(), Synthetic()).point(
        50.0, 7.0, "2026-10-08T10:00", "2026-10-08T12:00", "SELECT model, prob(rain < 1) AS p FROM per_member "
                                                          "GROUP BY model")
    assert rows(answer) == [{"model": "synthetic", "p": 0.5}]
    assert answer["unavailable_sources"] == [
        {"source": "broken", "reason": "failed (RuntimeError); see server log"}]


@pytest.mark.parametrize("sql, fragment", [
    ("SELECT nonsense FROM per_member", "Binder Error"),
    ("SELECT * FROM read_csv('/etc/passwd')", "disabled by configuration"),
    ("COPY (SELECT 1) TO '/tmp/claude-1000/escape.csv'", "only SELECT statements"),
])
def test_sql_errors_are_readable(sql, fragment):
    with pytest.raises(QueryError) as error:
        forecaster().point(50.0, 7.0, "2026-10-08T10:00", "2026-10-08T12:00", sql)
    assert fragment in str(error.value)
    assert "pending query result" not in str(error.value)


def test_window_ranking_in_sql():
    # t2m rises 0.1 K per hour, so later windows are warmer: the 16:00–18:00 window scores highest.
    answer = forecaster().point(50.0, 7.0, "2026-10-08T06:00", "2026-10-08T18:00", """
        SELECT window_start, window_end, prob(tmax > 15.5) AS p FROM per_member GROUP BY ALL
        ORDER BY p DESC, window_start DESC LIMIT 3""", window_hours=2)
    assert [r["window_start"] for r in rows(answer)] == ["2026-10-08T16:00", "2026-10-08T15:00", "2026-10-08T14:00"]
    assert [r["p"] for r in rows(answer)] == [0.7, 0.7, 0.6]
    windows = forecaster().point(50.0, 7.0, "2026-10-08T06:00", "2026-10-08T18:00",
                                 "SELECT count(DISTINCT window_start) AS n FROM samples", window_hours=2)
    assert rows(windows) == [{"n": 11}]


def test_places_are_compared_in_one_query():
    answer = forecaster().forecast(
        "SELECT place, prob(rain < 0.5) AS p FROM per_member GROUP BY place ORDER BY place",
        "2026-10-08T10:00", "2026-10-08T12:00", places="Köln@50.94,6.96; Bonn @ 50.73,7.10")
    assert rows(answer) == [{"place": "Bonn", "p": 0.3}, {"place": "Köln", "p": 0.3}]


@pytest.mark.parametrize("kwargs", [
    dict(),  # no location
    dict(lat=50.0),  # incomplete point
    dict(lat=50.0, lon=7.0, places="A@50,7"),  # two forms
    dict(polyline="_p~iF~ps|U", end="2026-10-08T12:00"),  # routes take no end
    dict(places="A@50,7; A@51,7"),  # duplicate names
    dict(places="nonsense"),
    dict(lat=50.0, lon=7.0, window_hours=0),
])
def test_location_forms_are_validated(kwargs):
    kwargs.setdefault("end", "2026-10-08T12:00")
    with pytest.raises(ValueError):
        forecaster().forecast("SELECT 1", "2026-10-08T10:00", **kwargs)


def test_times_outside_the_forecast_span_are_rejected():
    for start, end in [("0001-01-01T00:00", "0001-01-02T00:00"), ("2026-11-30T00:00", "2026-12-01T00:00"),
                       ("2026-10-01T00:00", "2026-10-02T00:00")]:
        with pytest.raises(ValueError):
            forecaster().point(50.0, 7.0, start, end, "SELECT 1")


def test_invalid_coordinates_are_rejected():
    with pytest.raises(ValueError):
        forecaster().point(95.0, 7.0, "2026-10-08T10:00", "2026-10-08T12:00", "SELECT 1")


def test_area_per_member_is_per_point():
    # Every point gets the same values, so "anywhere" equals the point result; a sum across points would not.
    answer = forecaster().forecast("""
        SELECT model, prob(wettest < 0.5) AS p FROM (
          SELECT model, member, max(rain) AS wettest FROM per_member GROUP BY model, member) GROUP BY model""",
        "2026-10-08T10:00", "2026-10-08T12:00", lat=50.0, lon=7.0, radius_km=30)
    assert rows(answer) == [{"model": "synthetic", "p": 0.3}]
    assert answer["area_points"] > 1


def test_prob_ignores_missing_variables():
    # The synthetic source has no gusts: prob over an all-NULL condition is NULL, not 0.
    source = Synthetic()
    source.provides = lambda: {"precip", "t2m", "gust"}
    answer = forecaster(source).point(50.0, 7.0, "2026-10-08T10:00", "2026-10-08T12:00",
                                      "SELECT model, prob(gust_max > 40) AS p FROM per_member GROUP BY model")
    assert rows(answer) == [{"model": "synthetic", "p": None}]
