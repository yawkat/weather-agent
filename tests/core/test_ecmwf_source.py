"""ECMWF adapter against a fake server: synthetic index files and fields with known values."""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from weather_core.grid import Region, RegularGrid
from weather_core.sources.base import Query, SourceError
from weather_core.sources.ecmwf import IFS_ENS, EcmwfSource
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
    def decode(self, path):
        data = open(path, "rb").read()
        out = []
        for i in range(0, len(data), 16):
            p, m, step, _ = data[i:i + 16].decode().split("|")
            out.append(field(p.strip(), int(step), int(m)))
        return out


@pytest.fixture
def source(tmp_path):
    server = FakeServer(published_until=60)
    src = EcmwfSource(IFS_ENS, server, FakeDecoder(), FieldStore(tmp_path),
                      region=Region(45, 0, 55, 15), clock=lambda: RUN + timedelta(hours=8), grid=GRID)
    src.server = server
    return src


def test_point_window(source):
    start = RUN + timedelta(hours=24)
    query = Query(np.array([50.1]), np.array([7.3]), window=(start, start + timedelta(hours=6)))
    result = source.samples(query, ["precip", "t2m", "gust", "wind"])
    s = result.samples
    assert result.info.run == RUN and result.info.members == MEMBERS
    # Two 3-hourly intervals, each split in halves; member m has m mm/h.
    assert s.dt_hours.tolist() == [1.5, 1.5, 1.5, 1.5]
    assert s.variables["precip"][:, 0] == pytest.approx(np.arange(1, MEMBERS + 1), rel=1e-5)
    # Instant temperature at the steps and window edges (24, 27, 27, 30 h): 10 °C + 0.1 K/h.
    assert s.variables["t2m"][0] == pytest.approx([12.4, 12.7, 12.7, 13.0], abs=1e-3)
    assert s.variables["gust"][0, 0] == pytest.approx(36.0)
    assert s.variables["wind"][0, 0] == pytest.approx(np.hypot(18, 18))
    assert result.attribution["run"] == "2026-10-07T00:00Z"
    assert result.bytes_downloaded > 0


def test_cached_steps_are_not_downloaded_again(source):
    start = RUN + timedelta(hours=24)
    query = Query(np.array([50.1]), np.array([7.3]), window=(start, start + timedelta(hours=6)))
    source.samples(query, ["precip"])
    count = source.server.downloads
    again = source.samples(query, ["precip"])
    assert source.server.downloads == count and again.bytes_downloaded == 0


def test_unpublished_steps_are_reported(source):
    start = RUN + timedelta(hours=100)
    query = Query(np.array([50.1]), np.array([7.3]), window=(start, start + timedelta(hours=3)))
    with pytest.raises(SourceError):
        source.samples(query, ["precip"])


def test_one_hour_windows_use_exact_bounds(source):
    from weather_agent.sql_engine import DuckDbEngine
    from weather_core.forecast import Forecaster

    forecaster = Forecaster([source], DuckDbEngine(512, 2, 10000, 500), default_tz="UTC",
                            clock=lambda: RUN + timedelta(hours=8))
    # Member m rains m mm/h, so a 1-hour window has m mm: P(rain <= 10.5) = 10/50 for every window,
    # including windows that don't contain a 3-hourly model step.
    result = forecaster.forecast("""
        SELECT window_start, prob(rain <= 10.5) AS p FROM per_member GROUP BY ALL ORDER BY window_start""",
        "2026-10-08T00:00+00:00", "2026-10-08T06:00+00:00", lat=50.1, lon=7.3, window_hours=1)["result"]
    assert len(result["rows"]) == 6, result
    assert all(p == pytest.approx(0.2) for _, p in result["rows"]), result


def test_temp_files_are_removed_when_planning_fails(source, tmp_path):
    start = RUN + timedelta(hours=57)
    # +57h…+63h: step 60 is published, step 63 isn't, so planning fails after the first temp file exists.
    query = Query(np.array([50.1]), np.array([7.3]), window=(start, start + timedelta(hours=6)))
    with pytest.raises(SourceError):
        source.samples(query, ["precip"])
    assert not list(tmp_path.glob("*.grib2"))


def test_missing_parameter_makes_only_its_variables_unavailable(source):
    start = RUN + timedelta(hours=42)
    query = Query(np.array([50.1]), np.array([7.3]), window=(start, start + timedelta(hours=6)))
    result = source.samples(query, ["precip", "gust"])
    assert "gust" in result.samples.unavailable
    assert result.samples.variables["precip"][:, 0] == pytest.approx(np.arange(1, MEMBERS + 1), rel=1e-5)


def test_download_budget_is_enforced(tmp_path):
    from weather_core.budget import DownloadBudget

    budget = DownloadBudget(per_request_bytes=10_000, per_hour_bytes=10**9)
    src = EcmwfSource(IFS_ENS, FakeServer(published_until=60), FakeDecoder(), FieldStore(tmp_path),
                      region=Region(45, 0, 55, 15), clock=lambda: RUN + timedelta(hours=8), grid=GRID, budget=budget)
    start = RUN + timedelta(hours=24)
    query = Query(np.array([50.1]), np.array([7.3]), window=(start, start + timedelta(hours=24)))
    with pytest.raises(SourceError, match="per-query limit"):
        src.samples(query, ["precip", "t2m", "wind", "gust"])


def test_route_samples_use_their_own_point(source):
    # Instant temperature is uniform in space here, so the route value must equal the point value at that time.
    t0 = RUN + timedelta(hours=24)
    times = [t0, t0 + timedelta(hours=1), t0 + timedelta(hours=2)]
    query = Query(np.array([50.0, 50.1, 50.2]), np.array([7.0, 7.0, 7.0]), times=times,
                  dt_hours=np.array([0.5, 1.0, 0.5]), bearing=np.zeros(3))
    s = source.samples(query, ["t2m"]).samples
    assert s.variables["t2m"].shape == (MEMBERS, 3)
    assert s.variables["t2m"][0] == pytest.approx([12.4, 12.5, 12.6], abs=1e-3)
