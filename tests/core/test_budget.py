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
