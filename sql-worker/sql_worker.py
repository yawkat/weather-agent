"""Runs one client query on its own DuckDB, in the sandbox that weather_agent/sql_engine.py sets up.

Runs on CPython (with the duckdb and numpy packages), not GraalPy: `python3 -I sql_worker.py REQUEST.json`.
The request names the setup SQL, the tables to load (.npy files next to it) and the query. The result JSON
({"columns", "rows", "truncated"}) goes to stdout.

Exit codes: 0 result on stdout; 3 an error with the client's SQL, message on stdout; anything else a crash.
"""

import datetime
import decimal
import json
import math
import os
import resource
import sys
import threading

import duckdb
import numpy as np

EXIT_QUERY_ERROR = 3


def main() -> None:
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        request = json.load(f)
    directory = os.path.dirname(os.path.abspath(path))
    data_mb = sum(os.path.getsize(os.path.join(directory, t[k])) for t in request["tables"]
                  for k in ("ints", "floats")) / 2**20
    # DuckDB's memory_limit doesn't cover everything (large lists and strings), so bound the whole process.
    # Loading holds the arrays, their column copies and the table at once.
    _limit_address_space(int(request["memory_mb"]) + 3 * data_mb + int(request["overhead_mb"]))
    # If memory runs out anyway (host or service cgroup), the kernel should pick the worker, not the server.
    with open("/proc/self/oom_score_adj", "w") as f:
        f.write("1000")
    con = duckdb.connect(":memory:", config={
        "memory_limit": f"{int(request['memory_mb'])}MB",
        "threads": int(request["threads"]),
        # Spilling to disk would bypass the memory limit (and there is nowhere writable anyway).
        "temp_directory": "",
    })
    for sql in request["setup"]:
        con.execute(sql)
    for table in request["tables"]:
        _load(con, table, directory)
    _lock(con)
    try:
        result = _query(con, request["query"], int(request["max_rows"]), int(request["timeout_ms"]),
                        int(request["max_chars"]))
    except (duckdb.Error, MemoryError, ValueError) as e:
        _exit(EXIT_QUERY_ERROR, _message(e))
    _exit(0, result)


def _limit_address_space(mb: float) -> None:
    """Allow `mb` more address space than mapped now.

    Relative because importing numpy maps a 3 GB OpenBLAS buffer pool that stays untouched (DuckDB doesn't use
    BLAS); an absolute limit would have to include it. Set before any client SQL runs, and SQL can't raise it.
    """
    with open("/proc/self/status") as f:
        size_kb = next(int(line.split()[1]) for line in f if line.startswith("VmSize:"))
    limit = size_kb * 1024 + int(mb * 2**20)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def _load(con, table: dict, directory: str) -> None:
    """Append the rows of table["ints"] / table["floats"] (.npy, [rows, columns]); NaN floats become NULL."""
    ints = np.load(os.path.join(directory, table["ints"]), allow_pickle=False)
    floats = np.load(os.path.join(directory, table["floats"]), allow_pickle=False)
    columns = {}
    for i in range(ints.shape[1]):
        columns[f"i{i}"] = np.ascontiguousarray(ints[:, i])
    for i in range(floats.shape[1]):
        columns[f"f{i}"] = np.ascontiguousarray(floats[:, i])
    if not columns or ints.shape[0] == 0:
        return
    select = [f"i{i}" for i in range(ints.shape[1])]
    select += [f"CASE WHEN isnan(f{i}) THEN NULL ELSE f{i} END" for i in range(floats.shape[1])]
    con.register("_load", columns)
    try:
        con.execute(f'INSERT INTO "{table["name"]}" SELECT {", ".join(select)} FROM _load')
    finally:
        con.unregister("_load")


def _lock(con) -> None:
    """Disable file, network and extension access and freeze the configuration. Irreversible."""
    con.execute("SET enable_external_access = false")
    con.execute("SET autoinstall_known_extensions = false")
    con.execute("SET autoload_known_extensions = false")
    con.execute("SET allow_community_extensions = false")
    # Nothing of the host in duckdb_settings() (the default secret directory is under $HOME).
    con.execute("SET secret_directory = ''")
    con.execute("SET lock_configuration = true")


def _query(con, sql: str, max_rows: int, timeout_ms: int, max_chars: int) -> str:
    statements = duckdb.extract_statements(sql)
    if len(statements) != 1:
        raise ValueError(f"expected one SELECT statement, got {len(statements)}")
    if statements[0].type != duckdb.StatementType.SELECT:
        raise ValueError(f"only SELECT statements are allowed, not {statements[0].type.name}")
    # A clean error before the server's hard kill (which comes a little later).
    timer = threading.Timer(timeout_ms / 1000, con.interrupt)
    timer.daemon = True
    timer.start()
    try:
        cursor = con.execute(sql)
        columns = [d[0] for d in cursor.description]
        rows = []
        size = 0
        truncated = False
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            if len(rows) == max_rows:
                truncated = True
                break
            encoded = json.dumps([_value(v) for v in row], ensure_ascii=False, allow_nan=False)
            size += len(encoded) + 1
            if size > max_chars:
                truncated = True
                break
            rows.append(encoded)
    except duckdb.InterruptException:
        raise ValueError(f"query timed out after {timeout_ms / 1000:g} s") from None
    finally:
        timer.cancel()
    return ('{"columns":' + json.dumps(columns, ensure_ascii=False) + ',"rows":[' + ",".join(rows)
            + '],"truncated":' + ("true" if truncated else "false") + "}")


def _value(v):
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        # Forecast values don't need more than 4 significant decimals.
        return math.floor(v * 10000.0 + 0.5) / 10000.0
    if isinstance(v, decimal.Decimal):
        return _value(float(v))
    if isinstance(v, datetime.datetime):
        if v.tzinfo is not None:
            return v.isoformat()
        # TIMESTAMP columns hold local wall-clock time; ISO without seconds when they're zero.
        return v.isoformat(timespec="minutes" if v.second == 0 and v.microsecond == 0 else "auto")
    if isinstance(v, (datetime.date, datetime.time)):
        return v.isoformat()
    if isinstance(v, (list, tuple)):
        return [_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _value(x) for k, x in v.items()}
    return str(v)


def _message(e: BaseException) -> str:
    if isinstance(e, MemoryError):
        return "out of memory"
    return str(e) or type(e).__name__


def _exit(code: int, text: str) -> None:
    sys.stdout.buffer.write(text.encode("utf-8"))
    sys.stdout.buffer.flush()
    # Skip interpreter and DuckDB teardown, which can crash after an out-of-memory error.
    os._exit(code)


if __name__ == "__main__":
    main()
