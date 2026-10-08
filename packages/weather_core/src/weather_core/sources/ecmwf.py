"""ECMWF open data ensembles (IFS ENS, AIFS ENS) at 0.25°, fetched lazily by step and parameter.

Each step is one GRIB2 file with a JSON-lines ``.index`` of byte ranges, so we only download the fields a
query needs. Fields are global; we crop them to the storage region before caching.
"""

import json
import logging
import os
import tempfile
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from ..evaluate import SourceInfo
from ..grid import EUROPE, Crop, Region, RegularGrid
from ..budget import DownloadBudget
from ..store import FieldStore
from ..timeaxis import OutsideForecast
from .base import (NEEDS, Decoder, Download, Fetcher, Prepared, Query, SourceError, SourceSamples, runs_by_coverage,
                   sampling_for, step_list, with_previous)
from .mirrors import Mirrors

log = logging.getLogger(__name__)

GRID = RegularGrid(lat0=90.0, dlat=-0.25, nlat=721, lon0=-180.0, dlon=0.25, nlon=1440)

# Canonical base variable → (ECMWF parameter, kind). "accum" parameters are totals since the run start.
PARAMS = {
    "precip": ("tp", "accum"),
    "snow": ("sf", "accum"),
    "radiation": ("ssrd", "accum"),
    "t2m": ("2t", "instant"),
    "td2m": ("2d", "instant"),
    "wind_u": ("10u", "instant"),
    "wind_v": ("10v", "instant"),
    "gust": ("10fg", "interval"),  # max gust since the previous step
    "cloud": ("tcc", "instant"),
    "cape": ("mucape", "instant"),
}
INTERVAL_VARS = {name for name, (_, kind) in PARAMS.items() if kind != "instant"}

# Some parameters are published under different names depending on lead time: IFS ENS gusts are "10fg3"
# (max over the last 3 h) at some steps and "10fg" or "10fg6" (last 6 h) at others. Stored under the first name.
ALIASES = {"10fg": ("10fg", "10fg3", "10fg6")}


class MissingParameter(SourceError):
    def __init__(self, model: str, param: str, step: int):
        super().__init__(f"{model}: parameter {param} missing at +{step}h")
        self.param = param


@dataclass(frozen=True)
class EcmwfModel:
    source: str
    model: str
    path: str  # URL path segment between run and file name
    file_suffix: str  # "<stream>-<suffix>" at the end of file names
    members: int
    params: frozenset[str]  # ECMWF parameters this model publishes
    step_hours: int  # step spacing up to 144 h (6-hourly afterwards)
    long_runs: tuple[int, ...] = (0, 12)
    resolution: str = "0.25° (~25 km)"
    note: str = ""
    water_in_mm: bool = False  # tp/sf in kg/m² (= mm) instead of metres
    cloud_in_percent: bool = False  # tcc in % instead of a 0–1 fraction

    def steps(self, run: datetime) -> list[int]:
        short = list(range(0, 145, self.step_hours))
        if run.hour in self.long_runs:
            return short + list(range(150, 361, 6))
        return short


IFS_ENS = EcmwfModel("ecmwf-ens", "IFS ENS 0.25°", "ifs/0p25/enfo", "enfo-ef", 50,
                     frozenset({"tp", "sf", "ssrd", "2t", "2d", "10u", "10v", "10fg", "tcc", "mucape"}), 3,
                     note="ECMWF's physics-based global ensemble; the reference for days 2–15. 3-hourly to +144 h, "
                          "then 6-hourly. Too coarse for local showers and terrain effects.")
AIFS_ENS = EcmwfModel("ecmwf-aifs-ens", "AIFS ENS 0.25°", "aifs-ens/0p25/enfo", "enfo-pf", 50,
                      frozenset({"tp", "sf", "ssrd", "2t", "2d", "10u", "10v", "tcc"}), 6, water_in_mm=True,
                      cloud_in_percent=True,
                      note="ECMWF's machine-learned global ensemble; competitive with IFS for large-scale "
                           "patterns, smoother in the details. 6-hourly steps, no gusts or CAPE.")


class EcmwfSource:
    def __init__(self, model: EcmwfModel, fetcher: Fetcher, decoder: Decoder, store: FieldStore,
                 hosts: Mirrors | None = None, region: Region = EUROPE,
                 clock=lambda: datetime.now(timezone.utc), grid: RegularGrid = GRID,
                 on_download: Callable[[str, datetime, int], None] | None = None,
                 budget: DownloadBudget | None = None):
        self.model = model
        self.fetcher = fetcher
        self.decoder = decoder
        self.store = store
        # Share one between sources, so they all skip the same failing hosts.
        self.hosts = hosts or Mirrors("https://data.ecmwf.int/forecasts")
        self.crop: Crop = grid.crop(region)
        self.clock = clock
        self.on_download = on_download
        self.budget = budget
        self._index_cache: dict[tuple[datetime, int], list[dict] | None] = {}
        self._index_cache_time: dict[tuple[datetime, int], float] = {}

    @property
    def name(self) -> str:
        return self.model.source

    def describe(self) -> dict:
        return {"resolution": self.model.resolution, "note": self.model.note}

    def provides(self) -> set[str]:
        base = {name for name, (param, _) in PARAMS.items() if param in self.model.params}
        return {name for name, needs in NEEDS.items() if needs <= base}

    # -- run discovery -------------------------------------------------------------------------------------------

    def _path(self, run: datetime, step: int, ext: str) -> str:
        """Path of a step file relative to the base URL."""
        stamp = run.strftime("%Y%m%d%H%M%S")
        return f"{run:%Y%m%d}/{run:%H}z/{self.model.path}/{stamp}-{step}h-{self.model.file_suffix}.{ext}"

    def _index(self, run: datetime, step: int) -> list[dict] | None:
        key = (run, step)
        cached = self._index_cache.get(key, ...)
        # Positive results never change; re-check missing ones after a few minutes.
        if cached is not ... and (cached is not None or time.monotonic() - self._index_cache_time[key] < 300):
            return cached
        body = self.hosts.get(self.fetcher, self._path(run, step, "index"))
        entries = None if body is None else [json.loads(line) for line in body.decode().splitlines() if line.strip()]
        self._index_cache[key] = entries
        self._index_cache_time[key] = time.monotonic()
        return entries

    def candidate_runs(self) -> list[datetime]:
        now = self.clock()
        latest = now.replace(minute=0, second=0, microsecond=0, hour=now.hour - now.hour % 6)
        return [latest - timedelta(hours=6 * i) for i in range(5)]

    def find_run(self, query: Query) -> datetime:
        """Newest run that covers the query and has published every needed step."""
        candidates = self.candidate_runs()
        for key in [k for k in self._index_cache if k[0] not in candidates]:
            del self._index_cache[key], self._index_cache_time[key]
        def sampling(run):
            try:
                return sampling_for(query, run, self.model.steps(run))
            except OutsideForecast:
                return None
        for run, s in runs_by_coverage(candidates, sampling):
            steps = self.model.steps(run)
            last_step = steps[max(with_previous(s.needed_steps()))]
            if self._index(run, last_step) is not None:
                return run
        raise SourceError(f"{self.model.model}: no published run covers this time")

    # -- fetching ------------------------------------------------------------------------------------------------

    def _plan(self, run: datetime, step: int, params: Collection[str]) -> tuple[Download, list[dict]] | None:
        """What to download for one step, or None if everything is cached."""
        missing = [p for p in params if not self.store.has(self.name, run, p, step)]
        if not missing:
            return None
        entries = self._index(run, step)
        if entries is None:
            raise SourceError(f"{self.model.model} {run:%Y-%m-%d %HZ} +{step}h is not published")
        published = {e.get("param") for e in entries if e.get("type") == "pf"}
        wanted = []
        for p in missing:
            name = next((alias for alias in ALIASES.get(p, (p,)) if alias in published), None)
            if name is None:
                # Step 0 has no accumulations or gusts; they are zero by definition.
                if step == 0:
                    self.store.put(self.name, run, p, step, np.zeros((self.model.members, self.crop.size)))
                    continue
                raise MissingParameter(self.model.model, p, step)
            matches = [dict(e, _canonical=p) for e in entries if e.get("param") == name and e.get("type") == "pf"]
            if len({e.get("number") for e in matches}) != self.model.members or len(matches) != self.model.members:
                raise SourceError(f"{self.model.model}: index lists {len(matches)} members of {name} at +{step}h, "
                                  f"expected {self.model.members}")
            wanted += matches
        wanted.sort(key=lambda e: e["_offset"])
        if not wanted:
            return None
        ranges = _merge([(e["_offset"], e["_offset"] + e["_length"] - 1) for e in wanted])
        fd, path = tempfile.mkstemp(suffix=".grib2", dir=self.store.root)
        os.close(fd)
        return Download(self._path(run, step, "grib2"), ranges, path), wanted  # URL relative to self.hosts

    def _ensure_all(self, run: datetime, by_step: dict[int, set[str]]) -> int:
        plans: dict[int, tuple[Download, list[dict]]] = {}
        try:
            for step, params in by_step.items():
                if plan := self._plan(run, step, params):
                    plans[step] = plan
            if not plans:
                return 0
            size = sum(b - a + 1 for download, _ in plans.values() for a, b in download.ranges)
            params = sorted({e["_canonical"] for _, wanted in plans.values() for e in wanted})
            log.info("%s run %s: fetching %d step(s) %s, params %s, %.1f MB", self.name, run.strftime("%Y-%m-%d %HZ"),
                     len(plans), step_list(sorted(plans)), ",".join(params), size / 1e6)
            reserve = None if self.budget is None else lambda: self.budget.reserve(size)
            if reserve is not None:
                reserve()
            # Falling back to another host downloads the batch again, so that is reserved too.
            written = sum(self.hosts.download_many(self.fetcher, [download for download, _ in plans.values()], reserve))
            for step, (download, wanted) in plans.items():
                self._store_step(run, step, download.dest, wanted)
        finally:
            for download, _ in plans.values():
                if os.path.exists(download.dest):
                    os.unlink(download.dest)
        if written and self.on_download is not None:
            self.on_download(self.name, run, written)
        return written

    def _store_step(self, run: datetime, step: int, path: str, wanted: list[dict]) -> None:
        fields = self.decoder.decode(path, self.crop.extract)
        if len(fields) != len(wanted):
            raise SourceError(f"decoded {len(fields)} fields, expected {len(wanted)}")
        by_param: dict[str, list[tuple[int, np.ndarray]]] = {}
        for entry, values in zip(wanted, fields):
            by_param.setdefault(entry["_canonical"], []).append((int(entry["number"]), values))
        for param, members in by_param.items():
            members.sort(key=lambda m: m[0])
            self.store.put(self.name, run, param, step, np.stack([v for _, v in members]))

    # -- sampling ------------------------------------------------------------------------------------------------

    def samples(self, query: Query, variables: Collection[str]) -> SourceSamples:
        return self.prepare(query, variables).samples()

    def fetch(self, query: Query, variables: Collection[str]) -> int:
        """Download (into the store) what `prepare` would need, without loading it; bytes downloaded."""
        return self._fetch(query, variables)[-1]

    def prepare(self, query: Query, variables: Collection[str]) -> Prepared:
        run, steps, interp, needed, unavailable, downloaded = self._fetch(query, variables)
        base_per_step: dict[str, dict[int, np.ndarray]] = {}
        for name, idxs in needed.items():
            param, kind = PARAMS[name]
            per_step = {}
            for i in sorted(idxs):
                current = interp.apply(self._load(run, param, steps[i]))
                if kind == "accum":
                    previous = interp.apply(self._load(run, param, steps[i - 1]))
                    current = np.maximum(current - previous, 0.0) / (steps[i] - steps[i - 1])
                per_step[i] = _convert(name, current, self.model)
            base_per_step[name] = per_step
        info = SourceInfo(self.name, self.model.model, "ECMWF", run, self.model.members)
        return Prepared(query, run, steps, base_per_step, INTERVAL_VARS, unavailable, info, downloaded,
                        lambda times: attribution(info, times, run))

    def _fetch(self, query: Query, variables: Collection[str]):
        provided = self.provides()
        usable = [v for v in variables if v in provided]
        base_vars = set().union(*(NEEDS[v] for v in usable)) if usable else set()
        unavailable = {v for v in variables if v not in provided}
        run = self.find_run(query)
        steps = self.model.steps(run)
        sampling = sampling_for(query, run, steps)
        interp = self.crop.interpolation(query.lat, query.lon)
        downloaded = 0
        while True:
            needed = {name: self._step_indices(name, sampling) for name in base_vars}
            by_step: dict[int, set[str]] = {}
            for name, idxs in needed.items():
                param, kind = PARAMS[name]
                for i in (with_previous(idxs) if kind == "accum" else idxs):
                    by_step.setdefault(steps[i], set()).add(param)
            try:
                downloaded += self._ensure_all(run, by_step)
                break
            except MissingParameter as e:
                # Report variables needing it as unavailable instead of failing the whole source.
                lost = {name for name in base_vars if PARAMS[name][0] == e.param}
                if not lost:
                    raise
                base_vars -= lost
                unavailable |= {v for v in variables if NEEDS.get(v, set()) & lost}
        return run, steps, interp, needed, unavailable, downloaded

    @staticmethod
    def _step_indices(name: str, sampling) -> set[int]:
        _, kind = PARAMS[name]
        if kind == "instant":
            return sampling.needed_steps()
        return set(sampling.interval_step.tolist())

    def _load(self, run: datetime, param: str, step: int) -> np.ndarray:
        try:
            return self.store.get(self.name, run, param, step)
        except FileNotFoundError:
            # Evicted by a concurrent newer run; fetch it again once.
            self._ensure_all(run, {step: {param}})
            return self.store.get(self.name, run, param, step)


def _merge(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _convert(name: str, values: np.ndarray, model: EcmwfModel) -> np.ndarray:
    if name in ("precip", "snow"):
        return values if model.water_in_mm else values * 1000.0  # → mm/h
    if name in ("t2m", "td2m"):
        return values - 273.15
    if name in ("wind_u", "wind_v", "gust"):
        return values * 3.6
    if name == "radiation":
        return values / 3600.0  # J/m² per hour → W/m²
    if name == "cloud":
        return values if model.cloud_in_percent else values * 100.0
    return values


def attribution(info: SourceInfo, times: list[datetime], run: datetime) -> dict:
    first = (times[0] - run).total_seconds() / 3600
    last = (times[-1] - run).total_seconds() / 3600
    return {
        "provider": "ECMWF",
        "model": info.model,
        "run": run.strftime("%Y-%m-%dT%H:%MZ"),
        "members": info.members,
        "lead_hours": f"{first:.0f}–{last:.0f}",
        "license": "CC-BY-4.0",
        "notice": "Based on data and products of the European Centre for Medium-Range Weather Forecasts "
                  "(www.ecmwf.int), published under CC BY 4.0. ECMWF does not accept any liability whatsoever "
                  "for any error or omission in the data. Data modified (interpolated, derived ensemble "
                  "probabilities).",
    }
