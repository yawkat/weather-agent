"""Model grids: cropping fields to the storage region and interpolating to arbitrary points."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Region:
    south: float
    west: float
    north: float
    east: float

    def contains(self, lat, lon):
        return (lat >= self.south) & (lat <= self.north) & (lon >= self.west) & (lon <= self.east)


EUROPE = Region(south=30.0, west=-30.0, north=72.0, east=45.0)


@dataclass(frozen=True)
class RegularGrid:
    """Regular lat/lon grid, rows from lat0 by dlat (negative = north to south), columns from lon0 by dlon."""
    lat0: float
    dlat: float
    nlat: int
    lon0: float
    dlon: float
    nlon: int

    def crop(self, region: Region) -> "Crop":
        rows = np.nonzero((self.lats >= region.south - abs(self.dlat)) & (self.lats <= region.north + abs(self.dlat)))[0]
        lons = self.lons
        cols = np.nonzero((lons >= region.west - self.dlon) & (lons <= region.east + self.dlon))[0]
        if rows.size == 0 or cols.size == 0:
            raise ValueError("region doesn't overlap the grid")
        return Crop(self, int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1)

    @property
    def lats(self) -> np.ndarray:
        return self.lat0 + self.dlat * np.arange(self.nlat)

    @property
    def lons(self) -> np.ndarray:
        """Normalised to [-180, 180)."""
        return (self.lon0 + self.dlon * np.arange(self.nlon) + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class Crop:
    """A contiguous row/column window of a regular grid, stored as [rows, cols] flattened."""
    grid: RegularGrid
    row0: int
    row1: int
    col0: int
    col1: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.row1 - self.row0, self.col1 - self.col0

    @property
    def size(self) -> int:
        return self.shape[0] * self.shape[1]

    def extract(self, field: np.ndarray) -> np.ndarray:
        """Crop a full field (flattened, rows × columns) to this window, flattened."""
        return field.reshape(self.grid.nlat, self.grid.nlon)[self.row0:self.row1, self.col0:self.col1].ravel()

    def interpolation(self, lat: np.ndarray, lon: np.ndarray) -> "Interpolation":
        """Bilinear weights of the four surrounding crop points for each query point."""
        lat = np.asarray(lat, dtype=np.float64)
        lon = (np.asarray(lon, dtype=np.float64) + 180.0) % 360.0 - 180.0
        g = self.grid
        lats = g.lats[self.row0:self.row1]
        lon_start = g.lons[self.col0]
        fr = (lat - lats[0]) / g.dlat
        fc = ((lon - lon_start) % 360.0) / g.dlon
        rows, cols = self.shape
        if np.any((fr < 0) | (fr > rows - 1) | (fc < 0) | (fc > cols - 1)):
            raise OutsideRegion("point outside the stored region")
        r0 = np.minimum(np.floor(fr).astype(np.int64), rows - 2)
        c0 = np.minimum(np.floor(fc).astype(np.int64), cols - 2)
        wr, wc = fr - r0, fc - c0
        index = np.stack([r0 * cols + c0, r0 * cols + c0 + 1, (r0 + 1) * cols + c0, (r0 + 1) * cols + c0 + 1])
        weight = np.stack([(1 - wr) * (1 - wc), (1 - wr) * wc, wr * (1 - wc), wr * wc])
        return Interpolation(index, weight)


@dataclass(frozen=True)
class Interpolation:
    index: np.ndarray  # [k, points] into a flattened stored field
    weight: np.ndarray  # [k, points], sums to 1 over k

    def apply(self, stored: np.ndarray) -> np.ndarray:
        """stored: [..., stored_points] → [..., points]."""
        return np.sum(stored[..., self.index] * self.weight, axis=-2)


class OutsideRegion(ValueError):
    pass


class UnstructuredGrid:
    """Grid given by cell-centre coordinates (ICON). Nearest-cell lookup via a coarse bucket index.

    Points whose nearest cell is farther than `max_km` lie outside a regional model's domain.
    """

    def __init__(self, lat: np.ndarray, lon: np.ndarray, region: Region | None = None, bucket_deg: float = 0.25,
                 max_km: float | None = None):
        lat = np.asarray(lat, dtype=np.float64)
        lon = (np.asarray(lon, dtype=np.float64) + 180.0) % 360.0 - 180.0
        if region is not None:
            margin = bucket_deg
            keep = np.nonzero((lat >= region.south - margin) & (lat <= region.north + margin)
                              & (lon >= region.west - margin) & (lon <= region.east + margin))[0]
        else:
            keep = np.arange(lat.size)
        self.keep = keep  # indices into the full field, in stored order
        self.lat = lat[keep]
        self.lon = lon[keep]
        self.bucket_deg = bucket_deg
        self.max_km = max_km
        keys = self._key(self.lat, self.lon)
        order = np.argsort(keys, kind="stable")
        self._sorted_keys = keys[order]
        self._order = order

    def _key(self, lat, lon):
        return (np.floor((lat + 90.0) / self.bucket_deg).astype(np.int64) * 100000
                + np.floor((lon + 180.0) / self.bucket_deg).astype(np.int64))

    @property
    def size(self) -> int:
        return self.keep.size

    def extract(self, field: np.ndarray) -> np.ndarray:
        return field[self.keep]

    def interpolation(self, lat: np.ndarray, lon: np.ndarray) -> Interpolation:
        lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
        lon = (np.atleast_1d(np.asarray(lon, dtype=np.float64)) + 180.0) % 360.0 - 180.0
        nearest = np.empty(lat.size, dtype=np.int64)
        for i, (la, lo) in enumerate(zip(lat, lon)):
            candidates = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    key = self._key(la + dr * self.bucket_deg, lo + dc * self.bucket_deg)
                    lo_i = np.searchsorted(self._sorted_keys, key, side="left")
                    hi_i = np.searchsorted(self._sorted_keys, key, side="right")
                    candidates.append(self._order[lo_i:hi_i])
            candidates = np.concatenate(candidates)
            if candidates.size == 0:
                raise OutsideRegion("point outside the model domain")
            d2 = (self.lat[candidates] - la) ** 2 + ((self.lon[candidates] - lo) * np.cos(np.radians(la))) ** 2
            best = int(np.argmin(d2))
            if self.max_km is not None and np.sqrt(d2[best]) * 111.2 > self.max_km:
                raise OutsideRegion("point outside the model domain")
            nearest[i] = candidates[best]
        return Interpolation(nearest[None, :], np.ones((1, lat.size)))
