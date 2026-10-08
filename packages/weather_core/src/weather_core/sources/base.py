"""Common source machinery: queries, I/O interfaces, and turning cached fields into Samples."""

from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import numpy as np

from ..evaluate import SourceInfo
from ..samples import Samples
from ..timeaxis import StepSampling, sample_hours, sample_times
from ..variables import derive


class Fetcher(Protocol):
    def get(self, url: str) -> bytes | None:
        """Body of a small resource, or None if it doesn't exist (404)."""

    def download_many(self, requests: list["Download"]) -> list[int]:
        """Run downloads (in parallel, if the implementation can); bytes written per request. Raises NotPublished
        if a file doesn't exist.

        Python code here can't start threads (GraalPy context policy), so parallelism lives in the fetcher.
        """


@dataclass(frozen=True)
class Download:
    url: str
    ranges: list[tuple[int, int]] | None  # inclusive byte ranges, concatenated in order; None = whole resource
    dest: str


class Decoder(Protocol):
    def decode(self, path: str, extract: Callable[[np.ndarray], np.ndarray] | None = None) -> list[np.ndarray]:
        """Values of every GRIB message in the file, in file order, flattened, float32 (NaN = missing).

        `extract` (e.g. a crop) is applied to each field as it is read, so only one full field is in memory at a
        time: a step of a global ensemble is over a gigabyte.
        """

    def decode_files(self, paths: list[str],
                     extract: Callable[[np.ndarray], np.ndarray] | None = None) -> list[np.ndarray]:
        """Like `decode`, for several files at once (decoded in parallel, if the implementation can)."""


@dataclass(frozen=True)
class Query:
    """Where and when to sample: whole hours at points (hours), or explicit sample times along a route (times)."""
    lat: np.ndarray
    lon: np.ndarray
    hours: list[datetime] | None = None  # one sample per hour (see sample_hours)
    times: list[datetime] | None = None  # route: one time per point
    dt_hours: np.ndarray | None = None  # route: riding time per point
    bearing: np.ndarray | None = None  # route: direction of travel per point
    area: bool = False  # several points: keep the space dimension

    @property
    def is_route(self) -> bool:
        return self.times is not None


@dataclass
class SourceSamples:
    samples: Samples
    info: SourceInfo
    times: list[datetime]
    bytes_downloaded: int = 0
    attribution: dict = field(default_factory=dict)
    valid: np.ndarray | None = None  # per sample, for hourly queries: the run covers it (else values are NaN)


@dataclass
class Prepared:
    """Point values of every model step a query needs, loaded once; `samples()` turns them into Samples."""
    query: Query
    run: datetime
    steps: list[int]
    base_per_step: dict[str, dict[int, np.ndarray]]  # variable → step index → [member, point]
    interval_vars: Collection[str]
    unavailable: set[str]
    info: SourceInfo
    bytes_downloaded: int
    attribution: Callable[[list[datetime]], dict]

    def samples(self) -> SourceSamples:
        sampling = sampling_for(self.query, self.run, self.steps)
        samples = assemble(self.query, sampling, self.base_per_step, self.interval_vars, self.unavailable)
        return SourceSamples(samples, self.info, sampling.times, self.bytes_downloaded,
                             self.attribution(sampling.times), sampling.valid)


class SourceError(Exception):
    """A source can't answer this query (out of range, outside domain, data not published yet)."""


class NotPublished(SourceError):
    """A download's file doesn't exist upstream (HTTP 404)."""


def runs_by_coverage(candidates: list[datetime], sampling: Callable[[datetime], StepSampling | None]) -> list:
    """(run, sampling) for candidates that cover the query: those covering every sample first (newest first), then
    those covering part of it. `sampling` returns None for runs that don't cover it at all."""
    full, partial = [], []
    for run in candidates:
        s = sampling(run)
        if s is not None:
            (full if s.valid is None or s.valid.all() else partial).append((run, s))
    return full + partial


def sampling_for(query: Query, run: datetime, steps: list[int]) -> StepSampling:
    if query.is_route:
        return sample_times(run, steps, query.times, query.dt_hours)
    return sample_hours(run, steps, query.hours)


def assemble(query: Query, sampling: StepSampling, base_per_step: dict[str, dict[int, np.ndarray]],
             interval_vars: Collection[str], unavailable: set[str]) -> Samples:
    """Build Samples from per-step point values.

    base_per_step: variable → step index → [member, point]. For routes, each sample uses only its own point;
    for areas, all points become the space dimension; for a single point, the point dimension is dropped.
    """
    arrays = {}
    for name, per_step in base_per_step.items():
        extract = sampling.interval_values if name in interval_vars else sampling.instant_values
        # Routes: [member, sample] (each sample at its own point). Otherwise [member, sample, point].
        values = extract(per_step, diagonal=query.is_route)
        if not query.is_route and not query.area:
            values = values[:, :, 0]
        if sampling.valid is not None and not sampling.valid.all():
            values = np.array(values, dtype=np.float64)
            values[:, ~sampling.valid] = np.nan
        arrays[name] = values
    bearing = query.bearing if query.is_route else None
    variables = derive(arrays, bearing=bearing)
    return Samples(variables, sampling.dt_hours, has_space=query.area, unavailable=set(unavailable))


# Which base variables each catalogue variable needs.
NEEDS = {
    "precip": {"precip"}, "snow": {"snow"}, "radiation": {"radiation"}, "t2m": {"t2m"}, "td2m": {"td2m"},
    "rh": {"t2m", "td2m"}, "wind": {"wind_u", "wind_v"}, "wind_dir": {"wind_u", "wind_v"},
    "headwind": {"wind_u", "wind_v"}, "crosswind": {"wind_u", "wind_v"},
    "feels_like": {"t2m", "td2m", "wind_u", "wind_v"}, "gust": {"gust"}, "cloud": {"cloud"}, "cape": {"cape"},
}


def step_list(steps: list[int]) -> str:
    """'+0h…+24h' style summary, or the full list when short."""
    if len(steps) <= 6:
        return ", ".join(f"+{s}h" for s in steps)
    return f"+{steps[0]}h…+{steps[-1]}h"


def with_previous(indices: Collection[int]) -> set[int]:
    return set(indices) | {i - 1 for i in indices if i > 0}
