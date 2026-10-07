"""weather_core QueryEngine on DuckDB (through at.yawk.weatheragent.SqlCube)."""

import json
import os
import tempfile
from typing import Annotated

import numpy as np
from at.yawk.weatheragent import SqlCube
from jakarta.inject import Singleton
from micronaut.context.annotation import Value
from weather_core.cube import QueryError, Table

# The JDBC driver puts this generic line in front of the useful message (e.g. "Binder Error: ... LINE 1: ...").
_NOISE = "Attempting to execute an unsuccessful or closed pending query result"


@Singleton
class DuckDbEngine:
    def __init__(self,
                 memory_mb: Annotated[int, Value("${weather.query.memory-mb:1024}")],
                 threads: Annotated[int, Value("${weather.query.threads:2}")],
                 timeout_ms: Annotated[int, Value("${weather.query.timeout-ms:15000}")],
                 max_rows: Annotated[int, Value("${weather.query.max-rows:500}")]):
        self.memory_mb = memory_mb
        self.threads = threads
        self.timeout_ms = timeout_ms
        self.max_rows = max_rows

    def run(self, setup: list[str], tables: list[Table], sql: str) -> dict:
        cube = SqlCube(self.memory_mb, self.threads)
        try:
            for statement in setup:
                cube.execute(statement)
            with tempfile.TemporaryDirectory(prefix="weather-cube-") as directory:
                for table in tables:
                    self._load(cube, table, directory)
            cube.lock()
            try:
                text = cube.query(sql, self.max_rows, self.timeout_ms)
            except BaseException as e:  # SQLException from Java is a foreign exception
                if isinstance(e, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                    raise
                raise QueryError(_clean(e)) from None
            return json.loads(str(text))
        finally:
            cube.close()

    @staticmethod
    def _load(cube, table: Table, directory: str) -> None:
        ints_path = os.path.join(directory, f"{table.name}.i32")
        floats_path = os.path.join(directory, f"{table.name}.f32")
        with open(ints_path, "wb") as f:
            f.write(np.ascontiguousarray(table.ints, dtype="<i4").tobytes())
        with open(floats_path, "wb") as f:
            f.write(np.ascontiguousarray(table.floats, dtype="<f4").tobytes())
        cube.append(table.name, table.rows, len(table.int_columns), ints_path, len(table.float_columns), floats_path)


def _clean(error) -> str:
    try:
        message = str(error.getMessage())
    except BaseException:
        message = str(error)
    lines = [line for line in message.splitlines() if _NOISE not in line]
    text = "\n".join(lines).strip()
    if text.startswith("Error: "):
        text = text[len("Error: "):]
    return text or message
