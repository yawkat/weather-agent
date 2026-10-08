"""DWD ICON adapter against a fake opendata server: synthetic listings, grids and fields with known values."""

import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from weather_core.budget import DownloadBudget
from weather_core.grid import OutsideRegion
from weather_core.sources.base import Query, SourceError
from weather_core.sources.dwd import ICON_D2_EPS, ICON_D2_RUC_EPS, ICON_EU_EPS, NOTICE, DwdIconSource
from weather_core.store import FieldStore

RUN = datetime(2026, 10, 7, 12, tzinfo=timezone.utc)
# A tiny "icosahedral" grid: cells every 0.1° around (50.25, 7.25). The real ones have ~0.3–0.5M cells.
LAT, LON = (a.ravel() for a in np.meshgrid(np.arange(50.0, 50.55, 0.1), np.arange(7.0, 7.55, 0.1), indexing="ij"))
_URL = re.compile(r"/(?P<model>[^/]+)/p/(?P<param>[A-Z0-9_]+)/r/(?P<run>[^/]+)/e/(?P<member>\d\d)/s/"
                  r"(?:PT(?P<step>\d{3})H00M\.grib2)?$")


def field(param, member, step):
    """Known synthetic values per DWD parameter (SI units, totals and means since the run start)."""
    n = LAT.size
    values = {
        "CLAT": LAT, "CLON": LON,
        "TOT_PREC": np.full(n, 0.5 * member * step),  # 0.5 mm/h × member
        "SNOW_GSP": np.full(n, 0.1 * step), "SNOW_CON": np.full(n, 0.2 * step),
        "ASWDIR_S": np.full(n, 10.0 * step),  # mean since start; energy 10·s², so 10·(2s − 1) W/m² per hour
        "ASWDIFD_S": np.zeros(n),
        "T_2M": 283.15 + 0.1 * step + 0 * LAT, "TD_2M": np.full(n, 278.15),
        "U_10M": np.full(n, 5.0), "V_10M": np.full(n, 5.0), "VMAX_10M": np.full(n, 10.0),
        "CLCT": np.full(n, 50.0), "CAPE_ML": np.full(n, 100.0),
    }[param]
    return np.asarray(values, dtype=np.float32)


class FakeDwdServer:
    """`published[run]` is the last step uploaded for that run; files hold a 'PARAM|member|step' token.

    `member_one_behind[run]` holds member 01 back at an earlier step, as while DWD is still uploading a step.
    """

    def __init__(self, model, published, member_one_behind=None):
        self.model = model
        self.published = published
        self.member_one_behind = member_one_behind or {}
        self.urls = []
        self.listings = 0

    def _parse(self, url):
        m = _URL.search(url)
        assert m and m["model"] == self.model.path, url
        run = datetime.strptime(m["run"], "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
        return m["param"], run, int(m["member"]), None if m["step"] is None else int(m["step"])

    def _last(self, run, member):
        if member == 1 and run in self.member_one_behind:
            return self.member_one_behind[run]
        return self.published.get(run)

    def get(self, url):
        self.listings += 1
        param, run, member, _ = self._parse(url)
        last = self._last(run, member)
        if last is None:
            return None
        lines = [f'<a href="PT{s:03d}H00M.grib2">PT{s:03d}H00M.grib2</a>   07-Oct-2026 14:37:16   1000'
                 for s in self.model.steps() if s <= last]
        # Sub-hourly files (RUC precipitation every 5 minutes) must be ignored.
        lines.append('<a href="PT000H05M.grib2">PT000H05M.grib2</a>   07-Oct-2026 14:37:16   1000')
        return "\n".join(lines).encode()

    def download_many(self, requests):
        out = []
        for r in requests:
            param, run, member, step = self._parse(r.url)
            assert step is not None and step <= self._last(run, member), r.url  # the real server: 404
            self.urls.append(r.url)
            data = f"{param}|{member}|{step}".encode()
            with open(r.dest, "wb") as f:
                f.write(data)
            out.append(len(data))
        return out


class FakeDwdDecoder:
    def decode(self, path, extract=None):
        return self.decode_files([path], extract)

    def decode_files(self, paths, extract=None):
        out = []
        for path in paths:
            param, member, step = open(path, "rb").read().decode().split("|")
            out.append(field(param, int(member), int(step)))
        return out if extract is None else [extract(f) for f in out]


def make(tmp_path, model=ICON_D2_EPS, published=None, now=RUN + timedelta(hours=2), member_one_behind=None,
         **kwargs):
    server = FakeDwdServer(model, published if published is not None else {RUN: model.last_step}, member_one_behind)
    src = DwdIconSource(model, server, FakeDwdDecoder(), FieldStore(tmp_path), base_url="https://dwd.test/m",
                        clock=lambda: now, **kwargs)
    src.server = server
    return src


def hours(hours_from_run, length, lat=50.2, lon=7.3, run=RUN):
    start = run + timedelta(hours=hours_from_run)
    return Query(np.array([lat]), np.array([lon]), hours=[start + timedelta(hours=i) for i in range(length)])


def test_point_hours(tmp_path):
    source = make(tmp_path)
    result = source.samples(hours(24, 2), ["precip", "snow", "radiation", "t2m", "gust", "wind", "rh", "cloud"])
    s = result.samples
    assert result.info.run == RUN and result.info.members == 20 and result.info.provider == "DWD"
    assert s.dt_hours.tolist() == [1.0, 1.0]
    # De-accumulated: member m has 0.5·m mm/h; snow sums grid-scale and convective snow.
    assert s.variables["precip"][:, 0] == pytest.approx(0.5 * np.arange(1, 21), rel=1e-5)
    assert s.variables["snow"][0] == pytest.approx([0.3] * 2, rel=1e-4)
    # De-averaged radiation: the hours ending at +25 h and +26 h.
    assert s.variables["radiation"][0] == pytest.approx([490, 510], rel=1e-4)
    # Instants at the start of each hour.
    assert s.variables["t2m"][0] == pytest.approx([12.4, 12.5], abs=1e-3)
    assert s.variables["gust"][0, 0] == pytest.approx(36.0)
    assert s.variables["wind"][0, 0] == pytest.approx(np.hypot(18, 18))
    assert s.variables["cloud"][0, 0] == pytest.approx(50.0)
    assert "rh" in s.variables
    assert result.attribution["notice"] == NOTICE and result.attribution["model"] == "ICON-D2-EPS"
    assert result.attribution["run"] == "2026-10-07T12:00Z"
    assert result.bytes_downloaded > 0
    assert not list(tmp_path.glob("*.grib2"))


def test_only_needed_steps_and_members_are_downloaded(tmp_path):
    source = make(tmp_path)
    source.samples(hours(24, 2), ["precip"])
    # Totals at +24 h (start of the first hour), +25 h and +26 h, one file per member each; CLAT and CLON once.
    fields = [u for u in source.server.urls if "/TOT_PREC/" in u]
    assert sorted({int(re.search(r"PT(\d+)H", u)[1]) for u in fields}) == [24, 25, 26]
    assert len(fields) == 3 * 20
    assert len(source.server.urls) == len(fields) + 2


def test_first_hour_needs_no_step_zero_totals(tmp_path):
    source = make(tmp_path)
    s = source.samples(hours(0, 1), ["precip"]).samples
    assert s.variables["precip"][:, 0] == pytest.approx(0.5 * np.arange(1, 21), rel=1e-5)
    assert not [u for u in source.server.urls if "PT000H" in u and "/TOT_PREC/" in u]


def test_cached_fields_are_not_downloaded_again(tmp_path):
    source = make(tmp_path)
    source.samples(hours(24, 2), ["precip", "t2m"])
    count = len(source.server.urls)
    again = make(tmp_path)  # a new process: grid and fields come from disk
    result = again.samples(hours(24, 2), ["precip", "t2m"])
    assert again.server.urls == [] and result.bytes_downloaded == 0


def test_fetch_downloads_what_prepare_reads(tmp_path):
    source = make(tmp_path)
    assert source.fetch(hours(24, 2), ["precip", "t2m"]) > 0
    count = len(source.server.urls)
    assert source.prepare(hours(24, 2), ["precip", "t2m"]).bytes_downloaded == 0
    assert len(source.server.urls) == count


def test_fetch_rejects_points_outside_the_domain(tmp_path):
    source = make(tmp_path)
    with pytest.raises(OutsideRegion):
        source.fetch(hours(3, 1, lon=9.0), ["precip"])
    assert all("/CLAT/" in u or "/CLON/" in u for u in source.server.urls)


def test_newest_run_missing_steps_falls_back_to_older_run(tmp_path):
    # RUC: the 14z run has uploaded up to +5 h, the 13z run is complete.
    newest = RUN + timedelta(hours=2)
    published = {newest: 5, newest - timedelta(hours=1): 27}
    source = make(tmp_path, ICON_D2_RUC_EPS, published, now=newest + timedelta(minutes=50))
    result = source.samples(hours(8, 1, run=newest), ["precip"])
    assert result.info.run == newest - timedelta(hours=1)
    # Within its uploaded steps the newest run wins.
    near = source.samples(hours(2, 1, run=newest), ["precip"])
    assert near.info.run == newest


def test_step_still_uploading_to_some_members_uses_older_run(tmp_path):
    # Member 20 already has +9 h of the 14z run, member 01 only +8 h: the 13z run must be used for +8…+9 h.
    newest = RUN + timedelta(hours=2)
    source = make(tmp_path, ICON_D2_RUC_EPS, {newest: 9, newest - timedelta(hours=1): 27},
                  now=newest + timedelta(minutes=50), member_one_behind={newest: 8})
    result = source.samples(hours(8, 1, run=newest), ["precip"])
    assert result.info.run == newest - timedelta(hours=1)


def test_beyond_the_last_step_is_reported(tmp_path):
    source = make(tmp_path)
    with pytest.raises(SourceError, match="no published run"):
        source.samples(hours(50, 2), ["precip"])


def test_point_outside_the_domain_is_rejected_before_downloading_fields(tmp_path):
    source = make(tmp_path)
    query = hours(3, 1, lon=9.0)
    with pytest.raises(OutsideRegion):
        source.samples(query, ["precip"])
    assert all("/CLAT/" in u or "/CLON/" in u for u in source.server.urls)


def test_download_budget_uses_listed_sizes(tmp_path):
    # 2 fields × 20 members × 1000 bytes listed = 40 kB.
    source = make(tmp_path, budget=DownloadBudget(per_request_bytes=30_000, per_hour_bytes=10**9))
    with pytest.raises(SourceError, match="per-query limit"):
        source.samples(hours(24, 1), ["precip"])
    assert not [u for u in source.server.urls if "/TOT_PREC/" in u]


def test_grid_download_is_budgeted_and_counted(tmp_path):
    # CLAT and CLON are listed at 1000 bytes each: 2 kB is more than this allows.
    source = make(tmp_path, budget=DownloadBudget(per_request_bytes=1500, per_hour_bytes=10**9))
    with pytest.raises(SourceError, match="per-query limit"):
        source.samples(hours(24, 1), ["t2m"])
    assert source.server.urls == []

    source = make(tmp_path / "b")
    first = source.samples(hours(24, 1), ["t2m"]).bytes_downloaded
    grid_bytes = len(b"CLAT|1|0") + len(b"CLON|1|0")
    fields = source.samples(hours(30, 1), ["t2m"]).bytes_downloaded  # same number of fields, grid cached
    assert first == fields + grid_bytes


def test_changed_grid_is_fetched_again(tmp_path):
    source = make(tmp_path)
    source.samples(hours(3, 1), ["t2m"])
    source._grid_points += 1  # pretend DWD changed the grid since it was cached
    with pytest.raises(SourceError, match="grid"):
        source.samples(hours(5, 1), ["t2m"])
    assert not (tmp_path / "_grids" / "icon-d2-eps.f32").exists()
    assert source.samples(hours(5, 2), ["t2m"]).samples.variables["t2m"].shape == (20, 2)


def test_eu_eps_steps_and_variables(tmp_path):
    steps = ICON_EU_EPS.steps()
    assert steps[:3] == [0, 1, 2] and steps[75] == 75 and steps[76:78] == [78, 81] and steps[-1] == 120
    source = make(tmp_path, ICON_EU_EPS)
    # No dew point: no rh, but wind chill still gives feels_like.
    assert "rh" not in source.provides() and "td2m" not in source.provides()
    assert "feels_like" in source.provides()
    # RUC has no convective snow parameter, but still provides snow.
    assert "snow" in make(tmp_path, ICON_D2_RUC_EPS).provides()

    result = source.samples(hours(78, 3), ["precip", "feels_like"])
    assert result.info.members == 40
    # 3-hourly steps after +75 h: still 0.5·m mm/h.
    assert result.samples.variables["precip"][:, 0] == pytest.approx(0.5 * np.arange(1, 41), rel=1e-5)
    assert "feels_like" in result.samples.variables


def test_route_samples_use_their_own_point(tmp_path):
    source = make(tmp_path)
    t0 = RUN + timedelta(hours=6)
    times = [t0, t0 + timedelta(minutes=30), t0 + timedelta(hours=1)]
    query = Query(np.array([50.1, 50.2, 50.3]), np.array([7.1, 7.2, 7.3]), times=times,
                  dt_hours=np.array([0.25, 0.5, 0.25]), bearing=np.zeros(3))
    s = source.samples(query, ["t2m", "headwind"]).samples
    assert s.variables["t2m"].shape == (20, 3)
    assert s.variables["t2m"][0] == pytest.approx([10.6, 10.65, 10.7], abs=1e-3)
    assert "headwind" in s.variables
