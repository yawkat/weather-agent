"""DuckDbEngine: query speed, macros, and the sandbox (no file, network or extension access; memory, time and
output limits), through the real bwrap worker."""

import json
import os
import time

import numpy as np
import pytest
from weather_agent.sql_engine import DuckDbEngine
from weather_core.cube import QueryError, Table

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


def build(models=2, members=50, times=30, points=1, seed=0):
    """Setup SQL and tables for a synthetic cube, and its row count."""
    rng = np.random.default_rng(seed)
    setup = SETUP + ["INSERT INTO models VALUES (0, 'ecmwf-ens', TIMESTAMP '2026-10-07 00:00', 50, 'IFS'), "
                     "(1, 'ecmwf-aifs-ens', TIMESTAMP '2026-10-07 00:00', 50, 'AIFS')"]
    base = 29_350_000  # minutes since epoch, ~2025
    t_ids = np.array([[m, t, base + 60 * t, base + 60 * t + 120] for m in range(models) for t in range(times)])
    tables = [
        Table("times", ["model_id", "t", "epoch_min", "local_epoch_min"], ["dt_hours"], t_ids,
              np.ones((len(t_ids), 1))),
        Table("points", ["p"], ["lat", "lon", "distance_km"], np.arange(points)[:, None],
              np.column_stack([50 + rng.random(points), 7 + rng.random(points), np.arange(points) * 2.0])),
    ]
    grid = np.array(np.meshgrid(range(models), range(members), range(times), range(points), indexing="ij"))
    ints = grid.reshape(4, -1).T
    n = ints.shape[0]
    t2m = 15 + 5 * rng.standard_normal(n) + 0.3 * ints[:, 2]
    precip = np.maximum(rng.gamma(0.3, 1.0, n) - 0.2, 0)
    floats = np.column_stack([t2m, precip, 20 + 10 * rng.random(n), 10 + 5 * rng.random(n),
                              100 * rng.random(n), t2m - 2])
    tables.append(Table("raw", ["model_id", "member", "t", "p"], VARIABLES, ints, floats))
    return setup, tables, n


def engine(memory_mb=512, timeout_ms=5000, max_rows=1000, max_chars=1_000_000):
    return DuckDbEngine(memory_mb, 2, timeout_ms, max_rows, max_chars=max_chars)


def q(cube, sql, **kwargs):
    setup, tables, _ = cube
    return engine(**kwargs).run(setup, tables, sql)


def test_load_speed_and_example_queries():
    report = []
    for label, kwargs in [("point", dict(times=30)), ("route", dict(times=2000)),
                          ("area", dict(times=30, points=1000))]:
        cube = build(**kwargs)
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
        seconds = time.perf_counter() - started
        report.append(f"{label}: {cube[2]} rows, 3 queries (each a fresh worker) {seconds:.2f}s")
        assert len(joint["rows"]) == 2
        assert {r[1] for r in outcomes["rows"]} <= {"ok", "wet", "too warm"}
        assert hourly["truncated"] and len(hourly["rows"]) == 50
        assert len(hourly["rows"][0][2]) == 3
    print("\n".join(report))
    print(json.dumps(outcomes))


def test_values():
    result = q(build(members=1, times=1), """
        SELECT 42 AS i, 1.23456 AS f, 'nan'::DOUBLE AS nan, TIMESTAMP '2026-10-07 12:00' AS t,
               TIMESTAMP '2026-10-07 12:00:30' AS ts, DATE '2026-10-07' AS d, [0.5, NULL] AS l, {'a': 1} AS s,
               1.5::DECIMAL(4, 1) AS dec, true AS b, NULL AS n, 'ä"\\' AS str""")
    assert result["columns"] == ["i", "f", "nan", "t", "ts", "d", "l", "s", "dec", "b", "n", "str"]
    assert result["rows"] == [[42, 1.2346, None, "2026-10-07T12:00", "2026-10-07T12:00:30", "2026-10-07",
                               [0.5, None], {"a": 1}, 1.5, True, None, 'ä"\\']]
    # NaN floats in the loaded tables are NULL.
    setup, tables, _ = build(members=1, times=1)
    tables[-1].floats[0, 0] = np.nan
    sql = "SELECT count(*) FILTER (WHERE t2m IS NULL), count(*) FROM raw"
    assert engine().run(setup, tables, sql)["rows"] == [[1, 2]]


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
    "DROP TABLE raw; SELECT 1",
    "SELECT 1; SELECT 2",
    "CREATE TABLE x AS SELECT 1",
])
def test_sandbox_blocks(sql):
    with pytest.raises(QueryError) as error:
        q(build(members=2, times=2), sql)
    print(sql, "->", " | ".join(str(error.value).splitlines())[:400])
    assert not os.path.exists("/tmp/claude-1000/duck-escape.csv")
    assert not os.path.exists("/tmp/claude-1000/duck-escape.db")


def test_nothing_of_the_host_in_settings():
    rows = q(build(members=1, times=1), "SELECT name, value FROM duckdb_settings() WHERE value LIKE '%/%'")["rows"]
    assert [name for name, value in rows if value.startswith("/")] == []


def test_timeout():
    started = time.perf_counter()
    with pytest.raises(QueryError) as slow:
        q(build(members=2, times=2),
          "SELECT count(*) FROM range(100000000) a, range(100000) b WHERE a.range + b.range = 7", timeout_ms=2000)
    elapsed = time.perf_counter() - started
    print("timeout ->", str(slow.value)[:160], f"after {elapsed:.1f}s")
    assert "timed out" in str(slow.value)
    assert elapsed < 10


@pytest.mark.parametrize("sql", [
    "SELECT list(range) FROM range(400000000)",
    # Beyond DuckDB's memory_limit, which doesn't track these: only the worker's address space limit stops them.
    "SELECT repeat('x', 200000000) FROM range(500)",
    "SELECT len(range(0, 1000000000))",
    "SELECT list_sort(range(0, 300000000))[1]",
    "SELECT length(string_agg(repeat('x', 1000), ',')) FROM range(3000000)",
])
def test_memory_limit(sql):
    with pytest.raises(QueryError) as big:
        q(build(members=2, times=2), sql, timeout_ms=60000)
    print(sql, "->", str(big.value)[:160])


def test_output_limit():
    result = q(build(members=1, times=1), "SELECT repeat('x', 10000) FROM range(500)", max_chars=100_000)
    assert result["truncated"] and 0 < len(result["rows"]) < 10
