"""JavaDecoder's reading of the decoded output, with GribDecoder stubbed out."""

from types import SimpleNamespace

import numpy as np
import pytest

from weather_agent import java_io


def FakeGribDecoder(fields, truncate=0):
    """Writes the given fields as consecutive float32, like GribDecoder, optionally cut short.

    Not a class: Pyronaut would generate Java for it and import this module twice (pyronaut#331).
    """

    def decode_files_to_file(paths, out):
        data = b"".join(np.asarray(f, dtype="<f4").tobytes() for f in fields)
        with open(out, "wb") as f:
            f.write(data[:len(data) - truncate])
        return [len(f) for f in fields]

    return SimpleNamespace(decodeFilesToFile=decode_files_to_file)


@pytest.fixture
def grib(tmp_path):
    path = tmp_path / "in.grib2"
    path.write_bytes(b"")
    return str(path)


def test_fields_are_split(monkeypatch, grib):
    monkeypatch.setattr(java_io, "GribDecoder", FakeGribDecoder([[1, 2, 3], [4, 5], [6, 7, 8, 9]]))
    fields = java_io.JavaDecoder().decode(grib)
    assert [f.tolist() for f in fields] == [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    assert all(f.dtype == np.float32 for f in fields)


def test_extract_applies_to_each_field_and_copies(monkeypatch, grib):
    monkeypatch.setattr(java_io, "GribDecoder", FakeGribDecoder([np.arange(6), np.arange(6) + 10]))
    fields = java_io.JavaDecoder().decode_files([grib], lambda f: f[1:3])
    assert [f.tolist() for f in fields] == [[1, 2], [11, 12]]
    # A view would keep the whole decoded field alive.
    assert all(f.base is None for f in fields)


def test_truncated_output_raises(monkeypatch, grib, tmp_path):
    monkeypatch.setattr(java_io, "GribDecoder", FakeGribDecoder([[1, 2], [3, 4]], truncate=2))
    with pytest.raises(OSError, match="truncated"):
        java_io.JavaDecoder().decode(grib)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["in.grib2"]
