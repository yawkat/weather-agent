"""Meteogram series and daily summaries over synthetic sources."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from weather_agent.sql_engine import DuckDbEngine

from weather_core.evaluate import SourceInfo
from weather_core.forecast import Forecaster
from weather_core.sources.base import Prepared, SourceError

RUN = datetime(2026, 10, 7, 0, tzinfo=timezone.utc)
MEMBERS = 10
STEPS = list(range(0, 73))  # hourly to +72 h


class Steady:
    """Member m rains m × 0.1 mm/h; t2m = 10 + 0.5 m + 0.1 × hour. Covers `hours` after RUN."""

    def __init__(self, name="synthetic", hours=72):
        self.name = name
        self.hours = hours
        self.windows = []

    def provides(self):
        return {"precip", "t2m"}

    def describe(self):
        return {"resolution": "test", "note": "synthetic"}

    def coverage_ends(self):
        return [RUN + timedelta(hours=self.hours)]

    def prepare(self, query, variables):
        self.windows.append(query.window)
        if query.window[1] > RUN + timedelta(hours=self.hours):
            raise SourceError(f"{self.name}: no published run covers this time")
        m = np.arange(MEMBERS)[:, None] * np.ones((1, len(query.lat)))
        per_step = {}
        if "precip" in variables:
            per_step["precip"] = {i: m * 0.1 for i in range(len(STEPS))}
        if "t2m" in variables:
            per_step["t2m"] = {i: 10 + 0.5 * m + 0.1 * STEPS[i] for i in range(len(STEPS))}
        info = SourceInfo(self.name, self.name.title(), "test", RUN, MEMBERS)
        return Prepared(query, RUN, STEPS, per_step, {"precip"}, set(), info, 0,
                        lambda times: {"provider": "test", "model": self.name})


class Failing(Steady):
    def prepare(self, query, variables):
        raise RuntimeError("connection reset")


def forecaster(*sources, tz="UTC"):
    return Forecaster(list(sources) or [Steady()], DuckDbEngine(512, 2, 10000, 500), default_tz=tz,
                      clock=lambda: RUN + timedelta(minutes=20))


def test_series_per_member_at_model_steps():
    summary, chart = forecaster().meteogram(50.0, 7.0, variables="t2m,precip", label="Bonn")
    model = chart["models"][0]
    # Default window: from the current hour, three days.
    assert chart["start"] == "2026-10-07T00:00+00:00" and chart["end"] == "2026-10-10T00:00+00:00"
    assert chart["location"] == {"lat": 50.0, "lon": 7.0, "label": "Bonn"}
    assert len(model["times"]) == 73 and model["times"][1] - model["times"][0] == 3600
    # Instants at every boundary, interval rates per piece between boundaries.
    assert model["series"]["t2m"][2][10] == 12.0  # 10 + 0.5 × 2 + 0.1 × 10
    assert len(model["series"]["precip"][3]) == 72 and model["series"]["precip"][3][0] == 0.3
    assert [v["kind"] for v in chart["variables"]] == ["instant", "interval"]
    assert chart["timezone"] == "UTC" and chart["utc_offset_minutes"] is None
    assert summary["models"][0]["lead_hours"] == [0, 72]
    assert summary["attribution"] == chart["attribution"] == [{"provider": "test", "model": "synthetic"}]


def test_daily_summary():
    summary, _ = forecaster().meteogram(50.0, 7.0, variables="precip,t2m")
    day = summary["models"][0]["daily"][0]
    assert day["date"] == "2026-10-07" and day["hours"] == 24
    # Member m: 2.4 m mm a day; quantiles over m = 0…9.
    assert day["precip_total"] == [2.16, 10.8, 19.44]
    assert day["p_precip_1mm"] == 0.9
    # Day minimum at 00:00 and maximum at 23:00 (24:00 belongs to the next day).
    assert day["t2m_min"][0] == pytest.approx(10.4, abs=0.05)
    assert day["t2m_max"][2] == pytest.approx(16.4, abs=0.05)
    assert [d["date"] for d in summary["models"][0]["daily"]] == ["2026-10-07", "2026-10-08", "2026-10-09"]


def test_missing_variables_are_listed_not_charted():
    summary, chart = forecaster().meteogram(50.0, 7.0)
    assert set(chart["models"][0]["series"]) == {"t2m", "precip"}
    assert chart["models"][0]["missing_variables"] == ["cloud", "gust", "wind"]
    assert "wind_max" not in summary["models"][0]["daily"][0]


def test_short_model_shows_what_it_covers():
    short = Steady("short", hours=24)
    summary, chart = forecaster(Steady("long"), short).meteogram(50.0, 7.0, variables="t2m")
    long_model, short_model = chart["models"]
    assert "until" not in long_model
    assert short_model["until"] == "2026-10-08T00:00+00:00"
    assert "no published run" in short_model["until_reason"]
    assert len(short_model["times"]) == 25
    # Whole window first, then the run's end.
    assert [w[1] for w in short.windows] == [RUN + timedelta(hours=72), RUN + timedelta(hours=24)]
    assert summary["models"][1]["until"] == short_model["until"]


def test_window_halves_when_nothing_else_fits():
    tiny = Steady("tiny", hours=10)
    tiny.coverage_ends = lambda: []
    _, chart = forecaster(tiny).meteogram(50.0, 7.0, variables="t2m")
    # 72 h → 36 h → 18 h → 9 h
    assert [w[1] - w[0] for w in tiny.windows] == [timedelta(hours=h) for h in (72, 36, 18, 9)]
    assert chart["models"][0]["until"] == "2026-10-07T09:00+00:00"


def test_failures_are_reported():
    _, chart = forecaster(Failing("broken"), Steady()).meteogram(50.0, 7.0, variables="t2m")
    assert [m["model"] for m in chart["models"]] == ["synthetic"]
    assert chart["unavailable_sources"] == [{"source": "broken", "reason": "failed (RuntimeError); see server log"}]
    with pytest.raises(SourceError, match="no model could provide"):
        forecaster(Steady("never", hours=1)).meteogram(50.0, 7.0, variables="t2m")


def test_local_days_and_explicit_window():
    summary, chart = forecaster(tz="Europe/Berlin").meteogram(
        50.0, 7.0, start="2026-10-07T12:00", end="2026-10-08T12:00", variables="precip")
    assert chart["start"] == "2026-10-07T12:00+02:00" and chart["timezone"] == "Europe/Berlin"
    assert [(d["date"], d["hours"]) for d in summary["models"][0]["daily"]] == [("2026-10-07", 12),
                                                                                ("2026-10-08", 12)]
    _, fixed = forecaster().meteogram(50.0, 7.0, start="2026-10-07T12:00+02:00", variables="precip")
    assert fixed["timezone"] is None and fixed["utc_offset_minutes"] == 120


@pytest.mark.parametrize("kwargs", [
    dict(variables="t2m,nonsense"),
    dict(variables="wind_dir"),
    dict(variables="t2m,precip,wind,gust,cloud,rh,cape"),
    dict(label="a\x00b"),
    dict(start="2026-10-08T12:00", end="2026-10-08T10:00"),
    dict(start="2026-12-01T00:00"),
    dict(sources="nonsense"),
])
def test_arguments_are_validated(kwargs):
    if "sources" in kwargs:
        kwargs["sources"] = [kwargs["sources"]]
    with pytest.raises(ValueError):
        forecaster().meteogram(50.0, 7.0, **kwargs)
