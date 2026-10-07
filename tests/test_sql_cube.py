"""SqlCube: DuckDB loading speed, macros, and the sandbox (no file, network or extension access)."""

import json
import os
import tempfile
import time

import numpy as np
import pytest
from at.yawk.weatheragent import SqlCube

VARIABLES = ["t2m", "precip", "gust", "wind", "cloud", "feels_like"]

SETUP = [
    "CREATE TABLE models (model_id INTEGER, model VARCHAR, run TIMESTAMP, members INTEGER, note VARCHAR)",
    "CREATE TABLE times (model_id INTEGER, t INTEGER, epoch_min INTEGER, local_epoch_min INTEGER, dt_hours FLOAT)",
    "CREATE TABLE points (p INTEGER, lat FLOAT, lon FLOAT, distance_km FLOAT)",
    "CREATE TABLE raw (model_id INTEGER, member INTEGER, t INTEGER, p INTEGER, "
    + ", ".join(f"{v} FLOAT" for v in VARIABLES) + ")",
    """CREATE VIEW samples AS
       SELECT m.model, r.member, make_timestamp(tm.epoch_min::BIGINT * 60000000) AS time,
              make_timestamp(tm.local_epoch_min::BIGINT * 60000000) AS local_time, tm.dt_hours,
              r.p AS point, pt.lat, pt.lon, pt.distance_km, """ + ", ".join(f"r.{v}" for v in VARIABLES) + """
       FROM raw r JOIN models m USING (model_id) JOIN times tm USING (model_id, t) JOIN points pt USING (p)""",
    """CREATE VIEW per_member AS
       SELECT model, member, sum(precip * dt_hours) AS rain, max(t2m) AS tmax, min(t2m) AS tmin,
              max(gust) AS gust_max, min(feels_like) AS feels_min
       FROM samples GROUP BY model, member""",
    "CREATE MACRO prob(c) AS avg(CASE WHEN c THEN 1.0 ELSE 0.0 END)",
    "CREATE MACRO share(m) AS count(*) / sum(count(*)) OVER (PARTITION BY m)",
]


def build(models=2, members=50, times=30, points=1, seed=0, memory_mb=512):
    rng = np.random.default_rng(seed)
    cube = SqlCube(memory_mb, 2)
    for sql in SETUP:
        cube.execute(sql)
    cube.execute("INSERT INTO models VALUES (0, 'ecmwf-ens', TIMESTAMP '2026-10-07 00:00', 50, 'IFS'), "
                 "(1, 'ecmwf-aifs-ens', TIMESTAMP '2026-10-07 00:00', 50, 'AIFS')")
    with tempfile.TemporaryDirectory() as d:
        def load(table, ints, floats):
            ip, fp = os.path.join(d, "i"), os.path.join(d, "f")
            with open(ip, "wb") as f:
                f.write(np.ascontiguousarray(ints, dtype="<i4").tobytes())
            with open(fp, "wb") as f:
                f.write(np.ascontiguousarray(floats, dtype="<f4").tobytes())
            cube.append(table, ints.shape[0], ints.shape[1], ip, floats.shape[1], fp)

        base = 29_350_000  # minutes since epoch, ~2025
        t_ids = np.array([[m, t, base + 60 * t, base + 60 * t + 120] for m in range(models) for t in range(times)])
        load("times", t_ids, np.ones((len(t_ids), 1)))
        load("points", np.arange(points)[:, None], np.column_stack([50 + rng.random(points), 7 + rng.random(points),
                                                                     np.arange(points) * 2.0]))
        grid = np.array(np.meshgrid(range(models), range(members), range(times), range(points), indexing="ij"))
        ints = grid.reshape(4, -1).T
        n = ints.shape[0]
        t2m = 15 + 5 * rng.standard_normal(n) + 0.3 * ints[:, 2]
        precip = np.maximum(rng.gamma(0.3, 1.0, n) - 0.2, 0)
        floats = np.column_stack([t2m, precip, 20 + 10 * rng.random(n), 10 + 5 * rng.random(n),
                                  100 * rng.random(n), t2m - 2])
        started = time.perf_counter()
        load("raw", ints, floats)
        load_seconds = time.perf_counter() - started
    cube.lock()
    return cube, n, load_seconds


def q(cube, sql, max_rows=1000, timeout_ms=5000):
    return json.loads(cube.query(sql, max_rows, timeout_ms))


def test_load_speed_and_example_queries():
    report = []
    for label, kwargs in [("point", dict(times=30)), ("route", dict(times=2000)),
                          ("area", dict(times=30, points=1000))]:
        cube, rows, seconds = build(**kwargs)
        started = time.perf_counter()
        joint = q(cube, "SELECT model, prob(rain < 0.5 AND tmax <= 26) AS p FROM per_member GROUP BY model "
                        "ORDER BY model")
        outcomes = q(cube, """
            SELECT model, CASE WHEN rain >= 0.5 THEN 'wet' WHEN tmax > 26 THEN 'too warm' ELSE 'ok' END AS outcome,
                   share(model) AS p, quantile_cont(tmax, 0.9) AS tmax_p90, max(tmax) AS tmax_max
            FROM per_member GROUP BY model, outcome ORDER BY model, outcome""")
        hourly = q(cube, """
            SELECT model, local_time, quantile_cont(t2m, [0.1, 0.5, 0.9]) AS t2m, prob(precip > 0.1) AS p_rain
            FROM samples GROUP BY ALL ORDER BY model, local_time""", max_rows=50)
        query_seconds = time.perf_counter() - started
        cube.close()
        report.append(f"{label}: {rows} rows, load {seconds:.2f}s, 3 queries {query_seconds:.2f}s")
        assert len(joint["rows"]) == 2
        assert {r[1] for r in outcomes["rows"]} <= {"ok", "wet", "too warm"}
        assert hourly["truncated"] and len(hourly["rows"]) == 50
        assert len(hourly["rows"][0][2]) == 3
    print("\n".join(report))
    print(json.dumps(outcomes))


@pytest.mark.parametrize("sql", [
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM read_text('/etc/passwd')",
    "SELECT * FROM glob('/etc/*')",
    "COPY (SELECT 1) TO '/tmp/claude-1000/duck-escape.csv'",
    "INSTALL httpfs",
    "LOAD httpfs",
    "ATTACH '/tmp/claude-1000/duck-escape.db'",
    "SET enable_external_access = true",
    "SET lock_configuration = false",
    "SELECT * FROM read_parquet('https://example.org/x.parquet')",
    "SELECT * FROM read_blob('/etc/hostname')",
    "SELECT getenv('HOME')",
    "SELECT * FROM duckdb_extensions()",  # harmless, but it wants to read the extension directory
])
def test_sandbox_blocks(sql):
    cube, _, _ = build(members=2, times=2)
    try:
        with pytest.raises(BaseException) as error:
            q(cube, sql)
        print(sql, "->", " | ".join(str(error.value).splitlines())[:400])
    finally:
        cube.close()
    assert not os.path.exists("/tmp/claude-1000/duck-escape.csv")
    assert not os.path.exists("/tmp/claude-1000/duck-escape.db")


def test_timeout_and_memory_limit():
    cube, _, _ = build(members=2, times=2, memory_mb=256)
    try:
        started = time.perf_counter()
        with pytest.raises(BaseException) as slow:
            q(cube, "SELECT count(*) FROM range(100000000) a, range(100000) b WHERE a.range + b.range = 7",
              timeout_ms=2000)
        elapsed = time.perf_counter() - started
        print("timeout ->", str(slow.value)[:160], f"after {elapsed:.1f}s")
        assert elapsed < 10
        with pytest.raises(BaseException) as big:
            q(cube, "SELECT list(range) FROM range(400000000)", timeout_ms=60000)
        print("memory ->", str(big.value)[:160])
        # The cube still works afterwards.
        assert q(cube, "SELECT count(*) FROM samples")["rows"][0][0] == 2 * 2 * 2
    finally:
        cube.close()

