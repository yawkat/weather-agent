"""weather_core QueryEngine: each query runs on its own DuckDB in a sandboxed worker process.

The worker (sql-worker/sql_worker.py, on CPython from Nix) runs in bwrap: no network, its own PID/IPC/UTS/user
namespaces, read-only binds of only its own store closure and the request directory, nothing writable. It caps
its own address space before running client SQL, since DuckDB's memory_limit misses large lists and strings.
prlimit bounds output size, open files and CPU time; the server kills it at the deadline. A crash or memory
blowup takes down only the worker. See docs/security.md.
"""

import json
import logging
import os
import tempfile
from typing import Annotated

import numpy as np
from at.yawk.weatheragent import WorkerProcess
from jakarta.inject import Singleton
from micronaut.context.annotation import Value
from weather_core.cube import QueryError, Table

log = logging.getLogger(__name__)

EXIT_QUERY_ERROR = 3  # sql_worker.EXIT_QUERY_ERROR
# Startup and loading the cube; the worker times the query itself, the server kills it this much later.
STARTUP_GRACE_MS = 15000


@Singleton
class DuckDbEngine:
    def __init__(self,
                 memory_mb: Annotated[int, Value("${weather.query.memory-mb:1024}")],
                 threads: Annotated[int, Value("${weather.query.threads:2}")],
                 timeout_ms: Annotated[int, Value("${weather.query.timeout-ms:15000}")],
                 max_rows: Annotated[int, Value("${weather.query.max-rows:500}")],
                 max_chars: Annotated[int, Value("${weather.query.max-chars:1000000}")] = 1000000,
                 overhead_mb: Annotated[int, Value("${weather.query.overhead-mb:1024}")] = 1024,
                 max_concurrent: Annotated[int, Value("${weather.query.max-concurrent:2}")] = 2,
                 worker_python: Annotated[str, Value("${weather.query.worker-python:}")] = "",
                 worker_script: Annotated[str, Value("${weather.query.worker-script:}")] = "",
                 worker_closure: Annotated[str, Value("${weather.query.worker-closure:}")] = ""):
        self.memory_mb = memory_mb
        self.threads = threads
        self.timeout_ms = timeout_ms
        self.max_rows = max_rows
        self.max_chars = max_chars
        self.overhead_mb = overhead_mb
        # Tests construct the engine directly; nix develop exports the worker's location.
        self.worker_python = worker_python or os.environ.get("WEATHER_QUERY_WORKER_PYTHON", "")
        self.worker_script = worker_script or os.environ.get("WEATHER_QUERY_WORKER_SCRIPT", "")
        self.worker_closure = worker_closure or os.environ.get("WEATHER_QUERY_WORKER_CLOSURE", "")
        self._process = WorkerProcess(max_concurrent)
        self._binds = None

    def run(self, setup: list[str], tables: list[Table], sql: str) -> dict:
        with tempfile.TemporaryDirectory(prefix="weather-sql-") as directory:
            request = {
                "memory_mb": self.memory_mb,
                "overhead_mb": self.overhead_mb,
                "threads": self.threads,
                "max_rows": self.max_rows,
                "max_chars": self.max_chars,
                "timeout_ms": self.timeout_ms,
                "setup": setup,
                "tables": [self._save(table, directory) for table in tables],
                "query": sql,
            }
            with open(os.path.join(directory, "request.json"), "w", encoding="utf-8") as f:
                json.dump(request, f, ensure_ascii=False)
            stdout = os.path.join(directory, "stdout")
            stderr = os.path.join(directory, "stderr")
            code = int(self._process.run(self._command(directory), stdout, stderr,
                                         self.timeout_ms + STARTUP_GRACE_MS))
            if code == WorkerProcess.BUSY:
                raise QueryError("too many queries are running; try again in a moment")
            if code == WorkerProcess.TIMEOUT:
                raise QueryError(f"query timed out after {self.timeout_ms / 1000:g} s")
            with open(stdout, "rb") as f:
                output = f.read().decode("utf-8", errors="replace")
            if code == 0:
                return json.loads(output)
            if code == EXIT_QUERY_ERROR:
                raise QueryError(output.strip() or "query failed")
            with open(stderr, "rb") as f:
                f.seek(max(0, os.path.getsize(stderr) - 4000))
                detail = f.read().decode("utf-8", errors="replace")
            log.warning("SQL worker exited with %d: %s", code, detail)
            if code > 128:
                raise QueryError("query crashed the database (most likely out of memory); "
                                 "try a smaller query")
            raise RuntimeError(f"SQL worker failed with exit code {code}")

    @staticmethod
    def _save(table: Table, directory: str) -> dict:
        ints = f"{table.name}.ints.npy"
        floats = f"{table.name}.floats.npy"
        _save_npy(os.path.join(directory, ints), np.ascontiguousarray(table.ints, dtype="<i4"))
        _save_npy(os.path.join(directory, floats), np.ascontiguousarray(table.floats, dtype="<f4"))
        return {"name": table.name, "ints": ints, "floats": floats}

    def _command(self, directory: str) -> list[str]:
        if not (self.worker_python and self.worker_script and self.worker_closure):
            raise RuntimeError("SQL worker not configured: set weather.query.worker-python, -script and -closure "
                               "(nix develop and the weather-agent package do)")
        if self._binds is None:
            with open(self.worker_closure, encoding="utf-8") as f:
                paths = [line.strip() for line in f if line.strip()]
            self._binds = [arg for path in paths for arg in ("--ro-bind", path, path)]
        timeout_s = (self.timeout_ms + STARTUP_GRACE_MS) // 1000 + 1
        return [
            "prlimit",
            # Result (and error message) size; a bigger write kills the worker with SIGXFSZ.
            f"--fsize={4 * self.max_chars + 65536}",
            "--nofile=64",
            "--core=0",
            # Backstop for the server's deadline: all threads together.
            f"--cpu={timeout_s * (self.threads + 1)}",
            "--",
            "bwrap",
            "--unshare-all", "--unshare-user", "--disable-userns",
            "--die-with-parent", "--new-session",
            "--clearenv", "--setenv", "HOME", "/nonexistent",
            # One BLAS thread (DuckDB doesn't use BLAS); few malloc arenas, whose 64 MB reservations per thread
            # would count against the address space limit.
            "--setenv", "OPENBLAS_NUM_THREADS", "1", "--setenv", "MALLOC_ARENA_MAX", "2",
            *self._binds,
            "--ro-bind", self.worker_script, "/worker/sql_worker.py",
            "--ro-bind", directory, "/work",
            "--proc", "/proc",
            # Not --dev, whose tmpfs (with /dev/shm) is writable.
            "--dev-bind", "/dev/null", "/dev/null", "--dev-bind", "/dev/urandom", "/dev/urandom",
            # The root is bwrap's tmpfs: read-only, so nothing in the sandbox is writable.
            "--remount-ro", "/",
            "--chdir", "/work",
            "--",
            self.worker_python, "-I", "-B", "/worker/sql_worker.py", "/work/request.json",
        ]


def _save_npy(path: str, array: np.ndarray) -> None:
    # Not np.save: it writes through a file descriptor, which GraalPy's Java posix backend doesn't have.
    with open(path, "wb") as f:
        np.lib.format.write_array_header_1_0(f, np.lib.format.header_data_from_array_1_0(array))
        f.write(array.tobytes())
