"""DWD ICON ensembles (ICON-EU-EPS, ICON-D2-EPS, ICON-D2-RUC-EPS) from opendata.dwd.de, fetched lazily.

All three are published in DWD's "v1" layout: one uncompressed GRIB2 file (a single message) per parameter, run,
member and step, ``<base>/<model>/p/<PARAM>/r/<run>/e/<member>/s/PT<hhh>H<mm>M.grib2``. EU-EPS and D2-EPS also
exist in the older ``grib/<HH>/<param>/`` layout as bz2 files holding all members, but those are no smaller (the
GRIB data is already CCSDS-packed; D2 files also carry 15-minute steps nobody here needs), bunzip2 costs ~3 s of
CPU per D2 file, and EU-EPS is only hourly to +48 h there instead of +75 h.

Directory listings name the published steps of a run and their sizes: they drive run discovery (runs upload
step by step, so we take the newest run that has every step a query needs) and the download budget.

Fields are on ICON's unstructured icosahedral grid. Cell centres come from each product's own CLAT/CLON fields,
fetched once and kept next to the field cache.
"""

import logging
import os
import re
import tempfile
import threading
import time
from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ..budget import DownloadBudget
from ..evaluate import SourceInfo
from ..grid import EUROPE, Region, UnstructuredGrid
from ..store import FieldStore, atomic_replace
from ..timeaxis import OutsideForecast
from .base import (NEEDS, Decoder, Download, Fetcher, OnDownload, Prepared, Query, SourceError, SourceSamples,
                   cached_runs, runs_by_coverage, sampling_for, step_list)

log = logging.getLogger(__name__)

NOTICE = "Datenbasis: Deutscher Wetterdienst, eigene Bearbeitung"

# Canonical base variable → (DWD parameters, kind). A variable sums the parameters a model publishes (ICON-D2
# resolves convection, so RUC-EPS has no convective snow). Kinds:
# - accum: total since the run start (kg/m² = mm);
# - avg: mean since the run start (radiation, W/m²);
# - interval: value for the hour ending at the step (gust maximum);
# - instant: value at the step.
PARAMS = {
    "precip": (("TOT_PREC",), "accum"),
    "snow": (("SNOW_GSP", "SNOW_CON"), "accum"),
    "radiation": (("ASWDIR_S", "ASWDIFD_S"), "avg"),
    "t2m": (("T_2M",), "instant"),
    "td2m": (("TD_2M",), "instant"),
    "wind_u": (("U_10M",), "instant"),
    "wind_v": (("V_10M",), "instant"),
    "gust": (("VMAX_10M",), "interval"),
    "cloud": (("CLCT",), "instant"),
    "cape": (("CAPE_ML",), "instant"),
}
INTERVAL_VARS = {name for name, (_, kind) in PARAMS.items() if kind != "instant"}

_COMMON = frozenset({"T_2M", "U_10M", "V_10M", "VMAX_10M", "TOT_PREC", "SNOW_GSP", "ASWDIR_S", "ASWDIFD_S", "CLCT",
                     "CAPE_ML"})

# Hourly files in a step listing, with their size: `<a href="PT001H00M.grib2">…</a>  07-Oct-2026 14:37  801357`.
_LISTING = re.compile(r'href="PT(\d{3})H00M\.grib2">[^<]*</a>\s+\S+\s+\S+\s+(\d+)')
# A listing that misses steps is fetched again after this long; RUC runs upload over about two hours.
_LISTING_TTL = 60.0
# A run is complete on opendata.dwd.de within about this long after its nominal time.
PUBLISHED_AFTER_HOURS = 4
# Files per download_many call, bounding the temporary disk space a large query uses at once.
_FILES_PER_BATCH = 160


@dataclass(frozen=True)
class IconModel:
    source: str
    model: str
    path: str  # product directory under the v1 base URL
    members: int
    run_every: int  # hours between runs
    last_step: int
    hourly_until: int  # 3-hourly steps afterwards
    params: frozenset[str]
    resolution: str
    domain: str
    max_km: float  # a point farther than this from the nearest cell lies outside the domain
    note: str

    def steps(self) -> list[int]:
        return list(range(0, self.hourly_until + 1)) + list(range(self.hourly_until + 3, self.last_step + 1, 3))


ICON_EU_EPS = IconModel(
    "icon-eu-eps", "ICON-EU-EPS", "icon-eu-eps", 40, 6, 120, 75, _COMMON | {"SNOW_CON"}, "~13 km",
    "Europe and the North Atlantic, about 29–71°N, 24°W–63°E (served for 30–72°N, 30°W–45°E)", 15.0,
    note="DWD's European ensemble (40 members), runs every 6 h to +5 days; hourly steps to +75 h, then 3-hourly. "
         "Finer than ECMWF but still parametrises showers. No dew point (no rh; feels_like is wind chill only). "
         "After +75 h, gusts are the maximum of the last hour of each 3-hour step.")
ICON_D2_EPS = IconModel(
    "icon-d2-eps", "ICON-D2-EPS", "icon-d2-eps", 20, 3, 48, 48, _COMMON | {"SNOW_CON", "TD_2M"}, "~2 km",
    "Germany and its neighbours, about 43–58°N, 4°W–20°E", 3.0,
    note="DWD's convection-permitting ensemble (20 members) for Germany and neighbours, runs every 3 h to +48 h, "
         "hourly. Best for showers, thunderstorms, gusts and terrain effects in the first two days.")
ICON_D2_RUC_EPS = IconModel(
    "icon-d2-ruc-eps", "ICON-D2-RUC-EPS", "icon-d2-ruc-eps", 20, 1, 27, 27, _COMMON | {"TD_2M"}, "~2 km",
    "Germany and its neighbours, about 43–58°N, 4°W–20°E", 3.0,
    note="DWD's rapid-update convection-permitting ensemble (20 members) for Germany and neighbours: a new run "
         "every hour to about +27 h. The most recent data; best for the next hours (showers, storms, when rain "
         "starts or stops).")


class DwdIconSource:
    def __init__(self, model: IconModel, fetcher: Fetcher, decoder: Decoder, store: FieldStore,
                 base_url: str = "https://opendata.dwd.de/weather/nwp/v1/m", region: Region = EUROPE,
                 clock=lambda: datetime.now(timezone.utc),
                 on_download: OnDownload | None = None, budget: DownloadBudget | None = None):
        if not base_url.startswith("https://"):
            raise ValueError("DWD base URL must use https")
        self.model = model
        self.fetcher = fetcher
        self.decoder = decoder
        self.store = store
        self.base_url = base_url.rstrip("/")
        self.region = region
        self.clock = clock
        self.on_download = on_download
        self.budget = budget
        self._listings: dict[tuple[datetime, str], tuple[float, dict[int, int] | None]] = {}
        # Queries and prefetch passes run on several threads; guards `_listings` (never held for I/O).
        self._lock = threading.Lock()
        # The grid and the field size it was built for, replaced as one: another thread may drop it any time.
        self._grid: tuple[UnstructuredGrid, int] | None = None

    @property
    def name(self) -> str:
        return self.model.source

    def describe(self) -> dict:
        return {"resolution": self.model.resolution, "note": self.model.note}

    def _params(self, name: str) -> tuple[str, ...]:
        return tuple(p for p in PARAMS[name][0] if p in self.model.params)

    def _base(self) -> set[str]:
        return {name for name in PARAMS if self._params(name)}

    def _needs(self, variable: str) -> set[str]:
        needs = NEEDS[variable]
        if variable == "feels_like":
            needs = needs - ({"td2m"} - self._base())  # humidity only refines the heat index
        return needs

    def status(self) -> dict:
        """What the model is and what of it is cached (weather_core.status renders it)."""
        m = self.model
        hourly = "hourly steps" if m.hourly_until == m.last_step else \
            f"hourly steps to +{m.hourly_until} h, then 3-hourly"
        needs = {v: {p for b in self._needs(v) for p in self._params(b)} for v in self.provides()}
        return {
            "model": self.name, "name": m.model, "provider": "DWD", "members": m.members,
            "resolution": m.resolution, "note": m.note, "variables": sorted(needs), "domain": m.domain,
            "schedule": f"runs every {m.run_every} h to +{m.last_step} h; {hourly}",
            "steps_per_day": 24,
            "stale_after_hours": m.run_every + PUBLISHED_AFTER_HOURS,
            "runs": cached_runs(self.store, self.name, lambda run: m.steps(), needs, self.clock()),
        }

    def max_lead_hours(self) -> int:
        return self.model.last_step

    def provides(self) -> set[str]:
        base = self._base()
        return {name for name in NEEDS if self._needs(name) <= base}

    # -- URLs and listings ---------------------------------------------------------------------------------------

    def _dir(self, run: datetime, param: str, member: int) -> str:
        return f"{self.base_url}/{self.model.path}/p/{param}/r/{run:%Y-%m-%dT%H:%M}/e/{member:02d}/s/"

    def _url(self, run: datetime, param: str, member: int, step: int) -> str:
        return f"{self._dir(run, param, member)}PT{step:03d}H00M.grib2"

    def _listing(self, run: datetime, param: str) -> dict[int, int]:
        """Hourly steps of a parameter published for both the first and the last member → the larger file size.

        A step reaches the members over about half a minute, mostly from the last member down to the first (member
        01 was last in every upload checked), so a step the last member has may still 404 for the others. Sizes
        serve as the download estimate; members differ by about 1%.
        """
        key = (run, param)
        with self._lock:
            cached = self._listings.get(key)
        complete = cached is not None and cached[1] is not None and len(cached[1]) == len(self.model.steps())
        if cached is not None and (complete or time.monotonic() - cached[0] < _LISTING_TTL):
            return cached[1] or {}
        sizes = None
        for member in (self.model.members, 1):
            body = self.fetcher.get(self._dir(run, param, member))
            if body is None:
                sizes = None
                break
            published = set(self.model.steps()) if sizes is None else sizes.keys()
            listed = {int(step): int(size) for step, size in _LISTING.findall(body.decode(errors="replace"))}
            sizes = {step: max(size, sizes[step] if sizes else 0) for step, size in listed.items()
                     if step in published}
        with self._lock:
            self._listings[key] = (time.monotonic(), sizes)
        return sizes or {}

    # -- run discovery -------------------------------------------------------------------------------------------

    def candidate_runs(self) -> list[datetime]:
        now = self.clock()
        every = self.model.run_every
        latest = now.replace(minute=0, second=0, microsecond=0, hour=now.hour - now.hour % every)
        # Runs take one to three hours to upload; look back far enough to find a complete one.
        return [latest - timedelta(hours=every * i) for i in range(max(4, 6 // every))]

    def find_run(self, query: Query, params: Collection[str]) -> datetime:
        """Newest run that covers the query and has published every needed step of every needed parameter."""
        candidates = self.candidate_runs()
        with self._lock:
            for key in [k for k in self._listings if k[0] not in candidates]:
                del self._listings[key]
        steps = self.model.steps()
        probe = sorted(params) or ["T_2M"]
        def sampling(run):
            try:
                return sampling_for(query, run, steps)
            except OutsideForecast:
                return None
        for run, s in runs_by_coverage(candidates, sampling):
            last_step = steps[max(s.needed_steps())]
            if all(last_step in self._listing(run, p) for p in probe):
                return run
        raise SourceError(f"{self.model.model}: no published run covers this time")

    # -- grid ----------------------------------------------------------------------------------------------------

    def _grid_path(self) -> Path:
        # Outside the per-source run directories, which the store evicts.
        return self.store.root / "_grids" / f"{self.name}.f32"

    def _ensure_grid(self, run: datetime) -> tuple[UnstructuredGrid, int, int]:
        """The grid and the field size it was built for, fetched on first use; and bytes downloaded."""
        loaded = self._grid
        if loaded is not None:
            return *loaded, 0
        path = self._grid_path()
        written = self._fetch_grid(run, path) if not path.exists() else 0
        with open(path, "rb") as f:
            coords = np.frombuffer(f.read(), dtype="<f4")
        lat, lon = coords[:coords.size // 2], coords[coords.size // 2:]
        loaded = self._grid = (UnstructuredGrid(lat, lon, region=self.region, max_km=self.model.max_km), lat.size)
        return *loaded, written

    def _fetch_grid(self, run: datetime, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        params = ("CLAT", "CLON")
        sizes = [self._listing(run, param).get(0) for param in params]
        if None in sizes:
            raise SourceError(f"{self.model.model}: grid coordinates of run {run:%Y-%m-%d %HZ} are not published")
        reservation = self.budget.reserve(sum(sizes)) if self.budget is not None else None
        downloads = []
        for param in params:
            fd, tmp = tempfile.mkstemp(suffix=".grib2", dir=self.store.root)
            os.close(fd)
            downloads.append(Download(self._url(run, param, 1, 0), None, tmp))
        try:
            log.info("%s: fetching grid coordinates from run %s", self.name, run.strftime("%Y-%m-%d %HZ"))
            written = sum(self.fetcher.download_many(downloads))
            if reservation is not None:
                self.budget.settle(reservation, written)
            fields = self.decoder.decode_files([d.dest for d in downloads])
        finally:
            for d in downloads:
                if os.path.exists(d.dest):
                    os.unlink(d.dest)
        if len(fields) != 2 or fields[0].size != fields[1].size:
            raise SourceError(f"{self.model.model}: unexpected CLAT/CLON fields")
        # A unique name: concurrent first queries for the same source may both get here.
        fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=path.parent)
        os.close(fd)
        try:
            with open(tmp, "wb") as f:
                f.write(np.concatenate(fields).astype("<f4").tobytes())
            atomic_replace(Path(tmp), path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return written

    def _forget_grid(self) -> None:
        """DWD changed the grid (fields no longer match it): fetch it again next time."""
        self._grid = None
        self._grid_path().unlink(missing_ok=True)

    # -- fetching ------------------------------------------------------------------------------------------------

    def _ensure_all(self, run: datetime, by_step: dict[int, set[str]]) -> int:
        groups = [(step, param) for step, params in sorted(by_step.items()) for param in sorted(params)
                  if not self.store.has(self.name, run, param, step)]
        if not groups:
            return 0
        size = 0
        for step, param in groups:
            listed = self._listing(run, param)
            if step not in listed:
                raise SourceError(f"{self.model.model} {run:%Y-%m-%d %HZ} {param} +{step}h is not published")
            size += listed[step] * self.model.members
        params = sorted({param for _, param in groups})
        log.info("%s run %s: fetching %d field(s) at %s, params %s, ~%.1f MB", self.name,
                 run.strftime("%Y-%m-%d %HZ"), len(groups), step_list(sorted({s for s, _ in groups})),
                 ",".join(params), size / 1e6)
        grid, points, grid_bytes = self._ensure_grid(run)  # normally loaded by prepare() already
        reservation = self.budget.reserve(size) if self.budget is not None else None
        started = time.monotonic()
        per_batch = max(1, _FILES_PER_BATCH // self.model.members)
        written = 0
        try:
            for i in range(0, len(groups), per_batch):
                written += self._fetch_groups(run, groups[i:i + per_batch], grid, points)
        finally:
            if reservation is not None:
                self.budget.settle(reservation, written)
        if written and self.on_download is not None:
            self.on_download(self.name, run, written, len(groups), time.monotonic() - started)
        return written + grid_bytes

    def _fetch_groups(self, run: datetime, groups: list[tuple[int, str]], grid: UnstructuredGrid,
                      points: int) -> int:
        members = range(1, self.model.members + 1)
        downloads: list[Download] = []
        try:
            for step, param in groups:
                for member in members:
                    fd, path = tempfile.mkstemp(suffix=".grib2", dir=self.store.root)
                    os.close(fd)
                    downloads.append(Download(self._url(run, param, member, step), None, path))
            written = sum(self.fetcher.download_many(downloads))
            for g, (step, param) in enumerate(groups):
                paths = [d.dest for d in downloads[g * len(members):(g + 1) * len(members)]]
                fields = self.decoder.decode_files(paths)
                if len(fields) != len(members):
                    raise SourceError(f"{self.model.model}: {param} +{step}h has {len(fields)} fields for "
                                      f"{len(members)} members")
                if any(f.size != points for f in fields):
                    self._forget_grid()
                    raise SourceError(f"{self.model.model}: field size doesn't match the grid; grid will be "
                                      f"refetched, try again")
                self.store.put(self.name, run, param, step, np.stack([grid.extract(f) for f in fields]))
        finally:
            for d in downloads:
                if os.path.exists(d.dest):
                    os.unlink(d.dest)
        return written

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
            kind = PARAMS[name][1]
            per_step = {}
            for i in sorted(idxs):
                total = 0.0
                for param in self._params(name):
                    current = interp.apply(self._load(run, param, steps[i]))
                    if kind in ("accum", "avg"):
                        current = self._rate(run, param, kind, steps[i - 1], steps[i], current, interp)
                    total = total + current
                per_step[i] = _convert(name, total)
            base_per_step[name] = per_step
        info = SourceInfo(self.name, self.model.model, "DWD", run, self.model.members)
        return Prepared(query, run, steps, base_per_step, INTERVAL_VARS, unavailable, info, downloaded,
                        lambda times: attribution(info, times, run))

    def _fetch(self, query: Query, variables: Collection[str]):
        provided = self.provides()
        usable = [v for v in variables if v in provided]
        base_vars = set().union(*(self._needs(v) for v in usable)) if usable else set()
        unavailable = {v for v in variables if v not in provided}
        params = {p for name in base_vars for p in self._params(name)}
        run = self.find_run(query, params)
        steps = self.model.steps()
        sampling = sampling_for(query, run, steps)
        grid, _, downloaded = self._ensure_grid(run)
        interp = grid.interpolation(query.lat, query.lon)

        needed = {name: self._step_indices(name, sampling) for name in base_vars}
        by_step: dict[int, set[str]] = {}
        for name, idxs in needed.items():
            kind = PARAMS[name][1]
            for i in idxs:
                fetch = [i - 1, i] if kind in ("accum", "avg") else [i]
                for j in fetch:
                    if kind in ("accum", "avg") and steps[j] == 0:
                        continue  # totals and means since the start are zero there
                    by_step.setdefault(steps[j], set()).update(self._params(name))
        downloaded += self._ensure_all(run, by_step)
        return run, steps, interp, needed, unavailable, downloaded

    def _rate(self, run: datetime, param: str, kind: str, previous_step: int, step: int, current: np.ndarray,
              interp) -> np.ndarray:
        """Per-hour rate over (previous_step, step] from totals (accum) or means (avg) since the run start."""
        previous = 0.0 if previous_step == 0 else interp.apply(self._load(run, param, previous_step))
        if kind == "avg":
            current, previous = current * step, previous * previous_step
        return np.maximum(current - previous, 0.0) / (step - previous_step)

    @staticmethod
    def _step_indices(name: str, sampling) -> set[int]:
        if PARAMS[name][1] == "instant":
            return sampling.needed_steps()
        return set(sampling.interval_step.tolist())

    def _load(self, run: datetime, param: str, step: int) -> np.ndarray:
        try:
            return self.store.get(self.name, run, param, step)
        except FileNotFoundError:
            # Evicted by a concurrent newer run; fetch it again once.
            self._ensure_all(run, {step: {param}})
            return self.store.get(self.name, run, param, step)


def _convert(name: str, values: np.ndarray) -> np.ndarray:
    if name in ("t2m", "td2m"):
        return values - 273.15
    if name in ("wind_u", "wind_v", "gust"):
        return values * 3.6
    return values  # precip/snow mm/h (from kg/m²), radiation W/m², cloud %, cape J/kg


def attribution(info: SourceInfo, times: list[datetime], run: datetime) -> dict:
    first = (times[0] - run).total_seconds() / 3600
    last = (times[-1] - run).total_seconds() / 3600
    return {
        "provider": "DWD",
        "model": info.model,
        "run": run.strftime("%Y-%m-%dT%H:%MZ"),
        "members": info.members,
        "lead_hours": f"{first:.0f}–{last:.0f}",
        "license": "CC-BY-4.0",
        "notice": NOTICE,
    }
