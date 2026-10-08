import pytest

from weather_core.budget import DownloadBudget
from weather_core.sources.base import SourceError


def test_per_query_and_hourly_limits():
    now = [0.0]
    budget = DownloadBudget(per_request_bytes=100, per_hour_bytes=250, clock=lambda: now[0])
    with pytest.raises(SourceError, match="per-query"):
        budget.reserve(101)
    budget.reserve(100)
    budget.reserve(100)
    with pytest.raises(SourceError, match="hourly"):
        budget.reserve(100)
    now[0] = 3601.0  # the first reservations age out
    budget.reserve(100)


def test_a_query_shares_one_limit_across_reservations():
    budget = DownloadBudget(per_request_bytes=100, per_hour_bytes=10**6)
    with budget.query():
        budget.reserve(60)  # e.g. one model
        with pytest.raises(SourceError, match="already for other models"):
            budget.reserve(60)  # the next model of the same query
        budget.reserve(40)
    with budget.query():  # a new query starts from zero
        budget.reserve(100)


def test_settling_replaces_the_estimate():
    now = [0.0]
    budget = DownloadBudget(per_request_bytes=100, per_hour_bytes=150, clock=lambda: now[0])
    with budget.query():
        reservation = budget.reserve(100)
        budget.settle(reservation, 30)  # e.g. the download failed early
        budget.reserve(70)  # the query's total is 30 + 70
    budget.reserve(50)  # hourly: 30 + 70 + 50 = 150
    with pytest.raises(SourceError, match="hourly"):
        budget.reserve(1)
    now[0] = 3601.0  # everything ages out together with its reservation
    budget.reserve(100)


def test_background_downloads_are_neither_limited_nor_counted():
    budget = DownloadBudget(per_request_bytes=100, per_hour_bytes=250)
    with budget.background():
        budget.settle(budget.reserve(1000), 900)
        budget.reserve(1000)
    with budget.query():  # clients keep their whole allowance
        budget.reserve(100)
    budget.reserve(100)
    with pytest.raises(SourceError, match="hourly"):
        budget.reserve(51)
