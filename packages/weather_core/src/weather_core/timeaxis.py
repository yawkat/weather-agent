"""Mapping query times onto model steps.

Model output comes in two kinds:

- *interval* variables (precipitation, gust maxima, radiation) describe the period ending at a step;
- *instant* variables (temperature, wind, cloud) are values at a step.

A point window [start, end] is cut at the model steps inside it. Each piece is split in two halves; interval
variables keep the piece's rate in both halves, and instant variables take their value at the half's outer end
(a model step or a window edge). So min()/max() see every model value inside the window, not just
interpolated midpoints, while sum()/hours() still integrate exactly over the window. Route samples (individual
times) take the interval containing them, or interpolate instants to their time.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np


@dataclass(frozen=True)
class StepSampling:
    """For each query sample: which interval (by its ending step index) and how to interpolate instants."""
    interval_step: np.ndarray  # index into `steps` of the step ending the containing interval
    instant_lo: np.ndarray  # index into `steps`
    instant_weight_hi: np.ndarray  # weight of instant_lo + 1
    dt_hours: np.ndarray
    times: list[datetime]  # representative time of each sample (midpoint for windows)

    def needed_steps(self) -> set[int]:
        """Step indices whose fields must be available."""
        needed = set(self.interval_step.tolist()) | set(self.instant_lo.tolist())
        needed |= {i + 1 for i, w in zip(self.instant_lo.tolist(), self.instant_weight_hi.tolist()) if w > 0}
        return needed

    def interval_values(self, per_step: dict[int, np.ndarray], diagonal: bool = False) -> np.ndarray:
        """per_step: step index → [member, point]. Returns [member, sample, point].

        With `diagonal` (routes), sample k uses only point k and the result is [member, sample]; building the
        full [member, sample, point] array first would be quadratic in route length.
        """
        if diagonal:
            return np.stack([per_step[i][:, k] for k, i in enumerate(self.interval_step.tolist())], axis=1)
        return np.stack([per_step[i] for i in self.interval_step.tolist()], axis=1)

    def instant_values(self, per_step: dict[int, np.ndarray], diagonal: bool = False) -> np.ndarray:
        out = []
        for k, (lo, w) in enumerate(zip(self.instant_lo.tolist(), self.instant_weight_hi.tolist())):
            a = per_step[lo][:, k] if diagonal else per_step[lo]
            if w == 0:
                out.append(a)
            else:
                b = per_step[lo + 1][:, k] if diagonal else per_step[lo + 1]
                out.append((1 - w) * a + w * b)
        return np.stack(out, axis=1)


class OutsideForecast(ValueError):
    pass


def _hours(run: datetime, t: datetime) -> float:
    return (t - run).total_seconds() / 3600.0


def _check_range(steps: list[int], first: float, last: float) -> None:
    if first < steps[0] or last > steps[-1]:
        raise OutsideForecast(f"times span +{first:.0f}h…+{last:.0f}h of the run; the forecast covers "
                              f"+{steps[0]}h…+{steps[-1]}h")


def _instant(steps_arr: np.ndarray, hours: float) -> tuple[int, float]:
    hi = int(np.searchsorted(steps_arr, hours, side="left"))
    if hi < len(steps_arr) and steps_arr[hi] == hours:
        return hi, 0.0
    lo = hi - 1
    return lo, float((hours - steps_arr[lo]) / (steps_arr[hi] - steps_arr[lo]))


def sample_window(run: datetime, steps: list[int], start: datetime, end: datetime) -> StepSampling:
    """Samples covering [start, end] for a source whose output steps (hours after `run`) are `steps`."""
    if end <= start:
        raise ValueError("window end must be after its start")
    a, b = _hours(run, start), _hours(run, end)
    _check_range(steps, a, b)
    steps_arr = np.asarray(steps, dtype=np.float64)
    interval_step, lo, w_hi, dt, times = [], [], [], [], []
    for k in range(1, len(steps)):
        piece_start, piece_end = max(steps_arr[k - 1], a), min(steps_arr[k], b)
        if piece_end <= piece_start:
            continue
        for anchor in (piece_start, piece_end):
            i, w = _instant(steps_arr, anchor)
            interval_step.append(k)
            lo.append(i)
            w_hi.append(w)
            dt.append((piece_end - piece_start) / 2)
            times.append(run + timedelta(hours=float(anchor)))
    return StepSampling(np.array(interval_step), np.array(lo), np.array(w_hi), np.array(dt), times)


def sample_times(run: datetime, steps: list[int], times: list[datetime], dt_hours: np.ndarray) -> StepSampling:
    """Samples at individual times (route samples)."""
    hours = np.array([_hours(run, t) for t in times])
    _check_range(steps, float(hours.min()), float(hours.max()))
    steps_arr = np.asarray(steps, dtype=np.float64)
    interval_step = np.clip(np.searchsorted(steps_arr, hours, side="left"), 1, len(steps) - 1)
    lo, w_hi = zip(*(_instant(steps_arr, h) for h in hours))
    return StepSampling(interval_step, np.array(lo), np.array(w_hi), np.asarray(dt_hours, dtype=np.float64),
                        list(times))
