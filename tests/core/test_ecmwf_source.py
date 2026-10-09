"""ECMWF adapter against a fake server: synthetic index files and fields with known values."""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from weather_core.grid import Region, RegularGrid
from weather_core.sources.base import NotPublished, Query, SourceError
from weather_core.sources.ecmwf import IFS_ENS, EcmwfSource
from weather_core.sources.mirrors import Mirrors
from weather_core.store import FieldStore

RUN = datetime(2026, 10, 7, 0, tzinfo=timezone.utc)
MEMBERS = 50
# A small regional grid keeps the fake fields cheap; the real one is global 0.25°.
GRID = RegularGrid(lat0=55.0, dlat=-1.0, nlat=11, lon0=0.0, dlon=1.0, nlon=16)
LAT, LON = np.meshgrid(GRID.lats, GRID.lons, indexing="ij")


def field(param, step, member):
    """Known synthetic values. tp accumulates 1 mm/h × member number; 2t rises 0.1 K per hour."""
    if param == "tp":
        return np.full(LAT.size, step * 0.001 * member, dtype=np.float32)
    if param == "2t":
        return (283.15 + 0.1 * step + 0.0 * LAT).ravel().astype(np.float32)
    if param in ("10fg", "10fg3"):
        return np.full(LAT.size, 10.0, dtype=np.float32)
    if param in ("10u", "10v"):
        return np.full(LAT.size, 5.0, dtype=np.float32)
    raise KeyError(param)


class FakeServer:
    """Each 'message' in a step file is a 16-byte header naming param, member and step."""

    def __init__(self, published_until: int):
        self.published_until = published_until
        self.downloads = 0

    def _messages(self, step):
        # Like IFS ENS: 3-hourly gusts are "10fg3", 6-hourly ones "10fg"; at +45h gusts are missing entirely.
        gust = [] if step == 45 else ["10fg3" if step % 6 else "10fg"]
        params = ["2t", "10u", "10v"] + gust + (["tp"] if step > 0 else [])
        return [(p, m) for p in params for m in range(1, MEMBERS + 1)]

    def get(self, url):
        step = int(url.rsplit("-", 3)[-3].rstrip("h"))
        if not url.endswith("-enfo-ef.index") or step > self.published_until or "20261007000000" not in url:
            return None
        lines = [json.dumps({"param": p, "number": str(m), "type": "pf", "step": str(step),
                             "_offset": i * 16, "_length": 16})
                 for i, (p, m) in enumerate(self._messages(step))]
        return "\n".join(lines).encode()

    def download_many(self, requests):
        return [self._download(r.url, r.ranges, r.dest) for r in requests]

    def _download(self, url, ranges, dest):
        self.downloads += 1
        step = int(url.rsplit("-", 3)[-3].rstrip("h"))
        blob = b"".join(f"{p:>6}|{m:03d}|{step:04d}|".encode() for p, m in self._messages(step))
        data = b"".join(blob[a:b + 1] for a, b in ranges)
        with open(dest, "wb") as f:
            f.write(data)
        return len(data)


class FakeDecoder:
    def decode(self, path, extract=None):
        data = open(path, "rb").read()
        out = []
        for i in range(0, len(data), 16):
            p, m, step, _ = data[i:i + 16].decode().split("|")
            out.append(field(p.strip(), int(step), int(m)))
        return out if extract is None else [extract(f) for f in out]


def hourly(start, length):
    return Query(np.array([50.1]), np.array([7.3]), hours=[start + timedelta(hours=i) for i in range(length)])


@pytest.fixture
def source(tmp_path):
    server = FakeServer(published_until=60)
    src = EcmwfSource(IFS_ENS, server, FakeDecoder(), FieldStore(tmp_path),
                      region=Region(45, 0, 55, 15), clock=lambda: RUN + timedelta(hours=8), grid=GRID)
    src.server = server
    return src


def test_point_hours(source):
    start = RUN + timedelta(hours=24)
    query = hourly(start, 6)
    result = source.samples(query, ["precip", "t2m", "gust", "wind"])
    s = result.samples
    assert result.info.run == RUN and result.info.members == MEMBERS
    # Six hours across two 3-hourly intervals; member m has m mm/h.
    assert s.dt_hours.tolist() == [1.0] * 6
    assert s.variables["precip"][:, 0] == pytest.approx(np.arange(1, MEMBERS + 1), rel=1e-5)
    # Instant temperature at the start of each hour (+24 … +29 h, interpolated between steps): 10 °C + 0.1 K/h.
    assert s.variables["t2m"][0] == pytest.approx([12.4, 12.5, 12.6, 12.7, 12.8, 12.9], abs=1e-3)
    assert s.variables["gust"][0, 0] == pytest.approx(36.0)
    assert s.variables["wind"][0, 0] == pytest.approx(np.hypot(18, 18))
    assert result.attribution["run"] == "2026-10-07T00:00Z"
    assert result.bytes_downloaded > 0


def test_cached_steps_are_not_downloaded_again(source):
    start = RUN + timedelta(hours=24)
    query = hourly(start, 6)
    source.samples(query, ["precip"])
    count = source.server.downloads
    again = source.samples(query, ["precip"])
    assert source.server.downloads == count and again.bytes_downloaded == 0


def test_fetch_downloads_what_prepare_reads(source):
    query = hourly(RUN + timedelta(hours=24), 6)
    assert source.fetch(query, ["precip", "t2m"]) > 0
    count = source.server.downloads
    assert source.prepare(query, ["precip", "t2m"]).bytes_downloaded == 0
    assert source.server.downloads == count


def test_hours_between_windows_are_not_downloaded(source):
    start = RUN + timedelta(hours=24)
    hours = [start + timedelta(hours=i) for i in range(27)]
    wanted = np.array([True] * 3 + [False] * 21 + [True] * 3)  # +24…+26 h and +48…+50 h
    query = Query(np.array([50.1]), np.array([7.3]), hours=hours, wanted=wanted)
    result = source.samples(query, ["precip"])
    assert source.server.downloads == 4  # steps +24/+27 and +48/+51, not the 7 in between
    precip = result.samples.variables["precip"][0]
    assert np.isnan(precip[3:24]).all() and np.isfinite(precip[:3]).all() and np.isfinite(precip[24:]).all()


def test_unpublished_steps_are_reported(source):
    start = RUN + timedelta(hours=100)
    query = hourly(start, 3)
    with pytest.raises(SourceError):
        source.samples(query, ["precip"])


def test_hourly_samples_between_3_hourly_steps(source):
    from weather_core.forecast import Forecaster

    forecaster = Forecaster([source], default_tz="UTC", clock=lambda: RUN + timedelta(hours=8))
    # Member m rains m mm/h, so each hour has m mm: P(rain <= 10.5) = 10/50 for every hour, including hours
    # between 3-hourly model steps.
    result = forecaster.forecast(
        'wx = forecast().interp(lat=50.1, lon=7.3).sel(time=slice("2026-10-08T00:00+00:00", '
        '"2026-10-08T06:00+00:00"))\n(wx.precip <= 10.5).mean("member")')["result"]
    assert len(result["rows"]) == 6, result
    assert all(p == pytest.approx(0.2) for _, _, p in result["rows"]), result


def test_temp_files_are_removed_when_planning_fails(source, tmp_path):
    start = RUN + timedelta(hours=57)
    # +57h…+63h: step 60 is published, step 63 isn't, so planning fails after the first temp file exists.
    query = hourly(start, 6)
    with pytest.raises(SourceError):
        source.samples(query, ["precip"])
    assert not list(tmp_path.glob("*.grib2"))


def test_missing_parameter_makes_only_its_variables_unavailable(source):
    start = RUN + timedelta(hours=42)
    query = hourly(start, 6)
    result = source.samples(query, ["precip", "gust"])
    assert "gust" in result.samples.unavailable
    assert result.samples.variables["precip"][:, 0] == pytest.approx(np.arange(1, MEMBERS + 1), rel=1e-5)


def test_download_budget_is_enforced(tmp_path):
    from weather_core.budget import DownloadBudget

    budget = DownloadBudget(per_request_bytes=10_000, per_hour_bytes=10**9)
    src = EcmwfSource(IFS_ENS, FakeServer(published_until=60), FakeDecoder(), FieldStore(tmp_path),
                      region=Region(45, 0, 55, 15), clock=lambda: RUN + timedelta(hours=8), grid=GRID, budget=budget)
    start = RUN + timedelta(hours=24)
    query = hourly(start, 24)
    with pytest.raises(SourceError, match="per-query limit"):
        src.samples(query, ["precip", "t2m", "wind", "gust"])


class Uploading(FakeServer):
    """The 06z run's index is out (on the origin), but its files can't be fetched yet: the origin throttles and
    mirrors lag. Its index has the 00z run's layout."""

    def __init__(self):
        super().__init__(published_until=60)
        self.failed = 0

    def get(self, url):
        return super().get(url.replace("20261007060000", "20261007000000"))

    def _download(self, url, ranges, dest):
        if "20261007060000" in url:
            self.failed += 1
            if url.startswith(ORIGIN):
                raise SourceError(f"{url}: HTTP 429")
            raise NotPublished(f"{url}: HTTP 404")
        return super()._download(url, ranges, dest)


ORIGIN = "https://data.ecmwf.int/forecasts"


def uploading_source(tmp_path, server, **kwargs):
    return EcmwfSource(IFS_ENS, server, FakeDecoder(), FieldStore(tmp_path), region=Region(45, 0, 55, 15),
                       clock=lambda: RUN + timedelta(hours=8), grid=GRID, **kwargs)


def test_a_listed_run_that_cant_be_fetched_falls_back_to_the_previous_run(tmp_path):
    server = Uploading()
    src = uploading_source(tmp_path, server)
    query = hourly(RUN + timedelta(hours=24), 6)
    assert src.find_run(query) == RUN + timedelta(hours=6)
    assert src.samples(query, ["precip"]).info.run == RUN
    assert server.failed == 1
    # For a while, the next query goes straight to the run that works.
    assert src.samples(hourly(RUN + timedelta(hours=30), 3), ["t2m"]).info.run == RUN
    assert server.failed == 1


def test_budget_refusals_dont_fall_back(tmp_path):
    from weather_core.budget import BudgetExceeded, DownloadBudget

    server = Uploading()
    src = uploading_source(tmp_path, server, budget=DownloadBudget(per_request_bytes=10, per_hour_bytes=10**9))
    with pytest.raises(BudgetExceeded):
        src.samples(hourly(RUN + timedelta(hours=24), 6), ["precip"])
    assert server.downloads == 0 and server.failed == 0


def test_failed_attempts_dont_count_against_the_query(tmp_path):
    from weather_core.budget import DownloadBudget

    query = hourly(RUN + timedelta(hours=24), 6)
    size = uploading_source(tmp_path / "probe", FakeServer(published_until=60)).samples(query, ["precip"])
    size = size.bytes_downloaded
    server = Uploading()
    # The 06z run fails on the origin (429) and the mirror (404); the 00z run then has to fit the same query.
    src = uploading_source(tmp_path / "src", server, hosts=Mirrors(ORIGIN, ["https://mirror.example"]),
                           budget=DownloadBudget(per_request_bytes=size * 3 // 2, per_hour_bytes=10**9))
    assert src.samples(query, ["precip"]).info.run == RUN
    assert server.failed == 2


def test_older_runs_are_only_looked_up_when_needed(tmp_path):
    class OlderIndexFails(FakeServer):
        """The 06z run works; asking for the 00z run's index fails on every host."""

        def get(self, url):
            if "20261007000000" in url:
                raise SourceError(f"{url}: HTTP 503")
            return super().get(url.replace("20261007060000", "20261007000000"))

    src = uploading_source(tmp_path, OlderIndexFails(published_until=60))
    assert src.samples(hourly(RUN + timedelta(hours=24), 6), ["precip"]).info.run == RUN + timedelta(hours=6)


def test_route_samples_use_their_own_point(source):
    # Instant temperature is uniform in space here, so the route value must equal the point value at that time.
    t0 = RUN + timedelta(hours=24)
    times = [t0, t0 + timedelta(hours=1), t0 + timedelta(hours=2)]
    query = Query(np.array([50.0, 50.1, 50.2]), np.array([7.0, 7.0, 7.0]), times=times,
                  dt_hours=np.array([0.5, 1.0, 0.5]), bearing=np.zeros(3))
    s = source.samples(query, ["t2m"]).samples
    assert s.variables["t2m"].shape == (MEMBERS, 3)
    assert s.variables["t2m"][0] == pytest.approx([12.4, 12.5, 12.6], abs=1e-3)


def test_fields_are_cropped_to_the_region(tmp_path):
    class PositionDecoder(FakeDecoder):
        """Every value is its index in the full field, so the stored window shows which points were kept."""

        def decode(self, path, extract=None):
            out = [np.arange(LAT.size, dtype=np.float32) for _ in super().decode(path)]
            return out if extract is None else [extract(f) for f in out]

    server = FakeServer(published_until=60)
    store = FieldStore(tmp_path)
    src = EcmwfSource(IFS_ENS, server, PositionDecoder(), store, region=Region(48, 3, 52, 8),
                      clock=lambda: RUN + timedelta(hours=8), grid=GRID)
    src.samples(hourly(RUN + timedelta(hours=24), 1), ["t2m"])
    stored = store.get(src.name, RUN, "2t", 24)
    # One grid step of margin around the region: latitudes 53…47, longitudes 2…9.
    rows, cols = np.nonzero((LAT >= 47) & (LAT <= 53) & (LON >= 2) & (LON <= 9))
    expected = np.ravel_multi_index((rows, cols), LAT.shape)
    assert stored.shape == (MEMBERS, expected.size) and expected.size < LAT.size
    assert (stored == expected).all()


class Throttled:
    """Wraps a fake server; requests to `down` fail like an exhausted HTTP 429 retry."""

    def __init__(self, server, down):
        self.server = server
        self.down = down
        self.urls = []

    def get(self, url):
        self.urls.append(url)
        if url.startswith(self.down):
            raise SourceError(f"{url}: HTTP 429")
        return self.server.get(url)

    def download_many(self, requests):
        self.urls += [r.url for r in requests]
        if any(r.url.startswith(self.down) for r in requests):
            raise SourceError("HTTP 429")
        return self.server.download_many(requests)


def test_throttled_origin_falls_back_to_mirror(tmp_path):
    fetcher = Throttled(FakeServer(published_until=60), "https://origin.test/")
    src = EcmwfSource(IFS_ENS, fetcher, FakeDecoder(), FieldStore(tmp_path),
                      hosts=Mirrors("https://origin.test", ["https://mirror.test"]),
                      region=Region(45, 0, 55, 15), clock=lambda: RUN + timedelta(hours=8), grid=GRID)
    result = src.samples(hourly(RUN + timedelta(hours=24), 6), ["precip"])
    assert result.samples.variables["precip"][:, 0] == pytest.approx(np.arange(1, MEMBERS + 1), rel=1e-5)
    # The origin is tried once, then skipped while it cools down.
    assert [u for u in fetcher.urls if u.startswith("https://origin.test/")] == [fetcher.urls[0]]
    assert any(u.startswith("https://mirror.test/") and u.endswith(".grib2") for u in fetcher.urls)


def test_fallback_download_counts_against_the_budget(tmp_path):
    class Budget:
        reserved = []
        settled = []

        def reserve(self, nbytes):
            self.reserved.append(nbytes)
            return len(self.reserved) - 1

        def settle(self, reservation, actual):
            self.settled.append((reservation, actual))

    budget = Budget()
    fetcher = Throttled(FakeServer(published_until=60), "https://origin.test/")
    # Indexes come from the origin; only its GRIB downloads fail.
    fetcher.get = fetcher.server.get
    src = EcmwfSource(IFS_ENS, fetcher, FakeDecoder(), FieldStore(tmp_path),
                      hosts=Mirrors("https://origin.test", ["https://mirror.test"]), budget=budget,
                      region=Region(45, 0, 55, 15), clock=lambda: RUN + timedelta(hours=8), grid=GRID)
    result = src.samples(hourly(RUN + timedelta(hours=24), 6), ["precip"])
    assert budget.reserved == [result.bytes_downloaded] * 2
    # The throttled attempt is released before the repeat; the repeat counts what it downloaded.
    assert budget.settled == [(0, 0), (1, result.bytes_downloaded)]


def test_status_reports_how_far_the_cache_reaches(source):
    source.samples(hourly(RUN + timedelta(hours=8), 12), ["precip", "t2m"])
    s = source.status()
    assert s["model"] == "ecmwf-ens" and s["members"] == MEMBERS and "cape" in s["variables"]
    assert s["domain"] == "global, served for 45–55°N, 0°E–15°E"
    [run] = s["runs"]
    assert run["run"] == RUN and run["reaches"] == RUN + timedelta(hours=360)
    # From the step covering now (+6 h at +8 h) to the last one the query needed, without a gap.
    assert run["cached"]["t2m"] == run["cached"]["precip"] == RUN + timedelta(hours=21)
    assert run["cached"]["wind"] is None
    assert run["cached"]["rh"] is None  # needs dew point too
