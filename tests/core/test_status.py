"""The models section of the query help: model facts, cache coverage and measured download costs."""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from weather_core.budget import DownloadBudget
from weather_core.status import DownloadStats, describe_models

NOW = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
RUN = datetime(2026, 10, 9, 0, tzinfo=timezone.utc)
TZ = ZoneInfo("Europe/Berlin")


class Fake:
    name = "fake-eps"

    def __init__(self, runs):
        self.runs = runs

    def status(self):
        return {"model": self.name, "name": "Fake EPS", "provider": "DWD", "members": 20, "resolution": "~2 km",
                "domain": "Germany", "note": "A test model.", "schedule": "runs every 3 h to +48 h",
                "variables": ["precip", "t2m", "wind"], "steps_per_day": 24, "stale_after_hours": 7,
                "runs": self.runs}


def test_download_stats_average_per_field():
    stats = DownloadStats()
    assert stats.per_field("fake-eps") is None
    stats.record("fake-eps", RUN, 30_000_000, 2, 4.0)
    stats.record("fake-eps", RUN, 10_000_000, 2, 2.0)
    stats.record("fake-eps", RUN, 0, 0, 1.0)  # nothing downloaded: ignored
    assert stats.per_field("fake-eps") == (10_000_000, 1.5)


def test_describes_models_cache_and_costs():
    older = RUN - timedelta(hours=6)
    runs = [{"run": RUN, "reaches": RUN + timedelta(hours=48),
             "cached": {"precip": RUN + timedelta(hours=48), "t2m": RUN + timedelta(hours=48),
                        "wind": None}},
            {"run": older, "reaches": older + timedelta(hours=30), "cached": {"precip": None}}]
    stats = DownloadStats()
    stats.record("fake-eps", RUN, 32_000_000, 2, 3.0)
    budget = DownloadBudget(4_000_000_000, 20_000_000_000)
    now = RUN + timedelta(hours=5)
    text = describe_models([Fake(runs)], now, TZ, stats, budget, warm=["fake-eps"], warm_variables=["precip", "t2m"])
    assert "Models now (Fri 2026-10-09 07:00, Europe/Berlin)" in text
    assert "4000 MB per query, 20000 MB per hour (0 MB used in the last hour)" in text
    assert "fake-eps: Fake EPS (DWD), 20 members, ~2 km; Germany." in text
    assert "Lacks: snow, td2m, rh," in text
    # The older run doesn't reach further: only the newest is listed.
    assert "Newest cached run: 2026-10-09 00:00 UTC (5 h ago), reaches Sun 2026-10-11 02:00." in text
    assert "Older" not in text
    assert "Cached from now: precip, t2m until Sun 2026-10-11 02:00; not wind." in text
    assert "Kept cached in the background: precip, t2m for every new run." in text
    assert "about 16 MB and 1.5 s per variable and model step (24 steps a day" in text


def test_older_run_reaching_further():
    older = RUN - timedelta(hours=6)
    runs = [{"run": RUN, "reaches": RUN + timedelta(hours=48), "cached": {"precip": RUN + timedelta(hours=12)}},
            {"run": older, "reaches": older + timedelta(hours=360), "cached": {"precip": None}}]
    text = describe_models([Fake(runs)], RUN + timedelta(hours=3), TZ)
    assert "(3 h ago), reaches Sun 2026-10-11 02:00.\n" in text
    assert "Older cached run, reaching further: 2026-10-08 18:00 UTC (9 h ago)" in text
    assert "    Nothing cached from now on." in text


def test_stale_cache():
    runs = [{"run": RUN, "reaches": RUN + timedelta(hours=48), "cached": {"precip": RUN + timedelta(hours=48)}}]
    text = describe_models([Fake(runs)], NOW, TZ)
    assert ("Newest cached run: 2026-10-09 00:00 UTC (12 h ago). Newer runs are likely published: queries download "
            "what they select from those.") in text
    assert "Cached from now" not in text


def test_kept_cached_lists_only_variables_the_model_has():
    text = describe_models([Fake([])], NOW, TZ, warm=["fake-eps"], warm_variables=["gust", "precip", "wind"])
    assert "Kept cached in the background: precip, wind for every new run." in text


def test_a_failing_status_leaves_the_others():
    class Failing(Fake):
        name = "broken-eps"

        def status(self):
            raise OSError("disk")

    text = describe_models([Failing([]), Fake([])], NOW, TZ)
    assert "broken-eps: status unavailable (see server log)." in text
    assert "fake-eps: Fake EPS (DWD)" in text


def test_nothing_cached():
    text = describe_models([Fake([])], NOW, TZ)
    assert "Nothing cached: the first query downloads everything it selects." in text
    assert "Uncached" not in text and "limits" not in text
