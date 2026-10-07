from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from weather_core.samples import Samples
from weather_core.timeaxis import OutsideForecast, sample_window

RUN = datetime(2026, 10, 7, 0, tzinfo=timezone.utc)


def window_samples(steps, instant_by_step, rate_by_interval, start_h, end_h):
    sampling = sample_window(RUN, steps, RUN + timedelta(hours=start_h), RUN + timedelta(hours=end_h))
    per_step = {i: np.array([[v]]) for i, v in enumerate(instant_by_step)}
    rates = {i: np.array([[v]]) for i, v in enumerate(rate_by_interval)}
    t = sampling.instant_values(per_step)[:, :, 0]
    precip = sampling.interval_values(rates)[:, :, 0]
    return Samples({"t2m": t, "precip": precip}, sampling.dt_hours)


def test_peak_at_a_model_step_is_seen():
    # t2m 15/25/15 at 09/12/15 h: the 12 h value must reach max().
    s = window_samples([9, 12, 15], [15, 25, 15], [0, 1, 2], 9, 15)
    assert s.variables["t2m"].max() == 25


def test_frost_inside_one_long_interval_is_seen():
    # One 6-hourly interval from -2 to +3 °C.
    s = window_samples([0, 6], [-2, 3], [0, 0], 0, 6)
    assert s.variables["t2m"].min() == -2


def test_sums_and_hours_are_exact_over_partial_intervals():
    # 1 mm/h in (0, 3], 4 mm/h in (3, 6]; window 2–5 h: 1 h × 1 + 2 h × 4 = 9 mm, 2 h above 2 mm/h.
    s = window_samples([0, 3, 6], [0, 0, 0], [0, 1, 4], 2, 5)
    precip, dt = s.variables["precip"][0], s.dt_hours
    assert float(np.sum(precip * dt)) == 9
    assert float(np.sum(dt[precip > 2])) == 2


def test_window_edges_are_interpolated():
    s = window_samples([0, 6], [0, 6], [0, 0], 1, 2)
    assert s.variables["t2m"][0].tolist() == pytest.approx([1.0, 2.0])


def test_outside_forecast():
    with pytest.raises(OutsideForecast):
        sample_window(RUN, [0, 3], RUN + timedelta(hours=2), RUN + timedelta(hours=5))
