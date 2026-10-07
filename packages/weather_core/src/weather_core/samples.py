"""Ensemble sample data: per-member values over time (and space) for one source and query.

Every variable is an array with the dimensions ``("member", "time")`` or ``("member", "time", "space")``:

- ``member``: ensemble members of one model run.
- ``time``: samples along the query, in chronological order. For a point query these are the model steps
  overlapping the window. For a route they are space-time samples: each one is where the rider is at that time.
- ``space``: optional, for area queries (grid points inside the area).

``dt_hours[t]`` is the duration each time sample stands for. Time aggregations weight by it, so
``sum(precip)`` is the window total in mm, whatever the step length.
"""

from dataclasses import dataclass, field

import numpy as np

MEMBER = "member"
TIME = "time"
SPACE = "space"


@dataclass
class Samples:
    variables: dict[str, np.ndarray]
    dt_hours: np.ndarray
    has_space: bool = False
    # Variables not available from this source; referencing one is reported as missing data, not as an error.
    unavailable: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.dt_hours = np.asarray(self.dt_hours, dtype=np.float64)
        expected_ndim = 3 if self.has_space else 2
        for name, values in self.variables.items():
            if values.ndim != expected_ndim:
                raise ValueError(f"variable {name!r} has {values.ndim} dimensions, expected {expected_ndim}")
            if values.shape[1] != self.dt_hours.shape[0]:
                raise ValueError(f"variable {name!r} has {values.shape[1]} time samples, dt_hours has "
                                 f"{self.dt_hours.shape[0]}")

    @property
    def dims(self) -> tuple[str, ...]:
        return (MEMBER, TIME, SPACE) if self.has_space else (MEMBER, TIME)

    @property
    def members(self) -> int:
        return next(iter(self.variables.values())).shape[0]

    def slice_time(self, start: int, stop: int) -> "Samples":
        return Samples(
            variables={name: values[:, start:stop] for name, values in self.variables.items()},
            dt_hours=self.dt_hours[start:stop],
            has_space=self.has_space,
            unavailable=self.unavailable,
        )
