"""Mapping query times onto model steps.

Model output comes in two kinds:

- *interval* variables (precipitation, gust maxima, radiation) describe the period ending at a step;
- *instant* variables (temperature, wind, cloud) are values at a step.

Queries sample whole hours (sample_hours): each sample stands for the hour starting at its label. Interval
variables take the rate of the step interval containing the hour, instants their value at the label (interpolated
between steps, so 3- and 6-hourly models are on the same hourly axis and their step values stay on it). Route
samples (individual times, sample_times) take the interval containing them, or interpolate instants to their
time.
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True)
class StepSampling:
    """For each query sample: which interval (by its ending step index) and how to interpolate instants."""
    interval_step: np.ndarray  # index into `steps` of the step ending the containing interval
    instant_lo: np.ndarray  # index into `steps`
    instant_weight_hi: np.ndarray  # weight of instant_lo + 1
    dt_hours: np.ndarray
    times: list[datetime]  # time of each sample (hourly: the start of its hour)
    valid: np.ndarray | None = None  # per sample: within the run's steps (sample_hours only; else all are)
    unused: np.ndarray | None = None  # per sample: outside every selected window; reads no steps, values are NaN

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


def sample_times(run: datetime, steps: list[int], times: list[datetime], dt_hours: np.ndarray) -> StepSampling:
    """Samples at individual times (route samples)."""
    hours = np.array([_hours(run, t) for t in times])
    _check_range(steps, float(hours.min()), float(hours.max()))
    steps_arr = np.asarray(steps, dtype=np.float64)
    interval_step = np.clip(np.searchsorted(steps_arr, hours, side="left"), 1, len(steps) - 1)
    lo, w_hi = zip(*(_instant(steps_arr, h) for h in hours))
    return StepSampling(interval_step, np.array(lo), np.array(w_hi), np.asarray(dt_hours, dtype=np.float64),
                        list(times))


def sample_hours(run: datetime, steps: list[int], hours: list[datetime],
                 wanted: np.ndarray | None = None) -> StepSampling:
    """One sample per hour, labelled by the hour's start: interval variables take the step interval containing the
    hour, instants their value at the label. Hours the run doesn't cover are marked invalid (not an error), so a
    model can answer for the part of a range it reaches.

    `wanted` marks the hours a query reads (e.g. two afternoons of one place's time axis): the others take the
    steps of a wanted hour, so they need no fields of their own, and are marked unused."""
    steps_arr = np.asarray(steps, dtype=np.float64)
    starts = np.array([_hours(run, t) for t in hours])
    valid = (starts >= steps_arr[0]) & (starts + 1 <= steps_arr[-1])
    wanted = np.ones(len(hours), dtype=bool) if wanted is None else np.asarray(wanted, dtype=bool)
    if not (valid & wanted).any():
        hours_read = starts[wanted] if wanted.any() else starts
        raise OutsideForecast(f"hours +{hours_read.min():.0f}h…+{hours_read.max():.0f}h of the run; the forecast "
                              f"covers +{steps[0]}h…+{steps[-1]}h")
    starts = np.clip(starts, steps_arr[0], steps_arr[-1] - 1)  # invalid hours: any index in range, masked later
    starts[~wanted] = starts[np.argmax(wanted)]
    interval_step = np.clip(np.searchsorted(steps_arr, starts + 0.5, side="left"), 1, len(steps) - 1)
    lo, w_hi = zip(*(_instant(steps_arr, h) for h in starts))
    return StepSampling(interval_step, np.array(lo), np.array(w_hi), np.ones(len(hours)), list(hours), valid,
                        None if wanted.all() else ~wanted)
