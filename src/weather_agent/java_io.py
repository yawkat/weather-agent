"""GRIB Decoder for weather_core, backed by netCDF-Java."""

import os
import tempfile

import numpy as np
from at.yawk.weatheragent import GribDecoder


class JavaDecoder:
    def decode(self, path: str) -> list[np.ndarray]:
        fd, out = tempfile.mkstemp(suffix=".f32", dir=os.path.dirname(path))
        os.close(fd)
        try:
            sizes = [int(n) for n in GribDecoder.decodeToFile(path, out)]
            # np.fromfile needs real file descriptors, which GraalPy's "java" posix backend doesn't provide.
            with open(out, "rb") as f:
                values = np.frombuffer(f.read(), dtype="<f4")
        finally:
            os.unlink(out)
        fields, offset = [], 0
        for n in sizes:
            fields.append(values[offset:offset + n])
            offset += n
        return fields
