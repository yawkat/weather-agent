"""On-disk cache of decoded, cropped fields for the latest run of each source.

Layout: ``<root>/<source>/<run>/<variable>/<step>.f32`` holding little-endian float32 ``[member, point]``.
Files are written to a temporary name and renamed, so readers never see partial data. Each source keeps its
newest `keep_runs` runs and deletes older ones. Two, not one: a short run (ECMWF 06z/18z ends at +144h) must
not evict the previous long run that queries beyond its range still use.
"""

import json
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path

import numpy as np


def atomic_replace(src: Path, dst: Path) -> None:
    """os.replace, falling back to Java NIO on GraalPy, whose "java" posix backend can't rename atomically."""
    try:
        os.replace(src, dst)
    except OSError as e:
        if e.errno != 5:
            raise
        import java  # GraalPy only

        files = java.type("java.nio.file.Files")
        paths = java.type("java.nio.file.Path")
        option = java.type("java.nio.file.StandardCopyOption")
        files.move(paths.of(str(src)), paths.of(str(dst)), option.ATOMIC_MOVE, option.REPLACE_EXISTING)


def run_key(run: datetime) -> str:
    return run.strftime("%Y%m%dT%H%MZ")


class FieldStore:
    def __init__(self, root: str | os.PathLike, keep_runs: int = 2):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep_runs = keep_runs
        self._lock = threading.Lock()
        self._meta: dict[Path, dict] = {}

    def _path(self, source: str, run: datetime, variable: str, step: int) -> Path:
        return self.root / source / run_key(run) / variable / f"{step:04d}.f32"

    def has(self, source: str, run: datetime, variable: str, step: int) -> bool:
        return self._path(source, run, variable, step).exists()

    def put(self, source: str, run: datetime, variable: str, step: int, members_by_points: np.ndarray) -> None:
        path = self._path(source, run, variable, step)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = path.parent.parent / "meta.json"
        tmp = path.with_suffix(f".tmp{threading.get_ident()}")
        data = np.ascontiguousarray(members_by_points, dtype="<f4")
        try:
            with open(tmp, "wb") as f:
                f.write(data.tobytes())
            atomic_replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
        with self._lock:
            if not meta.exists():
                meta.write_text(json.dumps({"members": int(data.shape[0]), "points": int(data.shape[1])}))
        self._evict_older(source, run_key(run))

    def get(self, source: str, run: datetime, variable: str, step: int) -> np.ndarray:
        path = self._path(source, run, variable, step)
        meta_path = path.parent.parent / "meta.json"
        meta = self._meta.get(meta_path)
        if meta is None:
            meta = self._meta[meta_path] = json.loads(meta_path.read_text())
        # np.fromfile needs real file descriptors, which GraalPy's "java" posix backend doesn't provide.
        with open(path, "rb") as f:
            return np.frombuffer(f.read(), dtype="<f4").reshape(meta["members"], meta["points"])

    def runs(self, source: str) -> list[str]:
        directory = self.root / source
        return sorted(p.name for p in directory.iterdir()) if directory.exists() else []

    def _evict_older(self, source: str, writing: str) -> None:
        # Never evict the run being written: a query may need an older run (window starting before the newest
        # runs), and evicting it under its own writes would make every repeat download everything again.
        with self._lock:
            for name in self.runs(source)[:-self.keep_runs]:
                if name == writing:
                    continue
                directory = self.root / source / name
                shutil.rmtree(directory, ignore_errors=True)
                self._meta.pop(directory / "meta.json", None)
