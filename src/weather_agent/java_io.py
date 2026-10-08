"""GRIB Decoder for weather_core, backed by netCDF-Java."""

import os
import tempfile
from collections.abc import Callable

import numpy as np
from at.yawk.weatheragent import GribDecoder


class JavaDecoder:
    def decode(self, path: str, extract: Callable[[np.ndarray], np.ndarray] | None = None) -> list[np.ndarray]:
        return self.decode_files([path], extract)

    def decode_files(self, paths: list[str],
                     extract: Callable[[np.ndarray], np.ndarray] | None = None) -> list[np.ndarray]:
        fd, out = tempfile.mkstemp(suffix=".f32", dir=os.path.dirname(paths[0]))
        os.close(fd)
        try:
            sizes = [int(n) for n in GribDecoder.decodeFilesToFile(list(paths), out)]
            fields = []
            # One field at a time: a whole ECMWF ENS step is over a gigabyte, too much for one read.
            # np.fromfile needs real file descriptors, which GraalPy's "java" posix backend doesn't provide.
            with open(out, "rb") as f:
                for n in sizes:
                    data = f.read(n * 4)
                    if len(data) != n * 4:
                        raise OSError(f"{out}: decoded output is truncated")
                    values = np.frombuffer(data, dtype="<f4")
                    if extract is not None:
                        # A view would keep the whole field alive.
                        values = np.array(extract(values), copy=True)
                    fields.append(values)
        finally:
            os.unlink(out)
        return fields
