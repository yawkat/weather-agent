"""Build weather_core/data/basemap.npz, the map under forecast maps.

Inputs, all in one directory (`python3 scripts/make_basemap.py var/naturalearth`; no dependencies):

- Natural Earth 1:10m (public domain) as GeoJSON, from
  https://github.com/nvkelso/natural-earth-vector/tree/master/geojson: ne_10m_land, ne_10m_lakes,
  ne_10m_lakes_europe, ne_10m_coastline, ne_10m_admin_0_boundary_lines_land, ne_10m_rivers_lake_centerlines,
  ne_10m_rivers_europe, ne_10m_roads (each `<name>.geojson`).
- GeoNames cities15000.zip (CC BY 4.0), places with at least 15,000 inhabitants, from
  https://download.geonames.org/export/dump/.

Everything is clipped to the forecast region (weather_core.grid.EUROPE plus a margin). Per layer, the .npz holds
the points of all its lines (or area rings) as int16 hundredths of a degree (`<layer>.lat`, `.lon`), each with
its Douglas-Peucker significance (`.sig`: the largest simplification tolerance, in hundredths of a degree, that
keeps the point; 255 = always), so the server simplifies a map to its scale by dropping points
(weather_core.chart.basemap). Per feature: its first point (`.start`, plus the end), Natural Earth's min_zoom × 10
(`.zoom`: the web map zoom level from which it's worth showing) and bounding box (`.box`: s, w, n, e). Areas
(land, lakes) are cut into TILE-degree cells, so a map only reads the parts it overlaps; their outlines, where
drawn, come from line layers (`lake_shore`), which have no such cuts. Places: `places.lat`,
`.lon`, `.pop`, and the names as one UTF-8 `\\n`-separated byte string (`places.names`), largest first.
"""

import io
import json
import math
import struct
import sys
import zipfile
from array import array
from pathlib import Path

SOUTH, WEST, NORTH, EAST = 30.0 - 3, -30.0 - 3, 72.0 + 3, 45.0 + 3
TILE = 5
OUT = Path(__file__).parent.parent / "packages/weather_core/src/weather_core/data/basemap.npz"
# GeoNames feature codes of places that aren't towns of their own: districts, abandoned and historical places.
NOT_TOWNS = {"PPLX", "PPLH", "PPLQ", "PPLW", "PPLCH"}
MAX_ZOOM = 7.5  # Natural Earth's finest features (min_zoom up to 10) are more than a forecast map needs


def features(directory: Path, name: str):
    for feature in json.load(open(directory / f"{name}.geojson"))["features"]:
        if feature["geometry"]:
            yield {k.lower(): v for k, v in feature["properties"].items()}, feature["geometry"]


def parts(geometry) -> list[list[list[float]]]:
    """Lines of a line geometry, or rings of an area."""
    kind, coords = geometry["type"], geometry["coordinates"]
    if kind == "LineString":
        return [coords]
    if kind in ("MultiLineString", "Polygon"):
        return coords
    if kind == "MultiPolygon":
        return [ring for polygon in coords for ring in polygon]
    raise ValueError(kind)


def quantize(points) -> list[tuple[int, int]]:
    out = []
    for lon, lat, *_ in points:
        point = (round(lat * 100), round(lon * 100))
        if not out or out[-1] != point:
            out.append(point)
    return out


def inside(lat: int, lon: int) -> bool:
    return SOUTH * 100 <= lat <= NORTH * 100 and WEST * 100 <= lon <= EAST * 100


def clip_line(points: list[tuple[int, int]]):
    """Runs inside the region, plus one point beyond each end."""
    flags = [inside(*p) for p in points]
    i = 0
    while i < len(points):
        if not flags[i]:
            i += 1
            continue
        j = i
        while j < len(points) and flags[j]:
            j += 1
        run = points[max(i - 1, 0):j + 1]
        if len(run) > 1:
            yield run
        i = j


def clip_ring(ring, s, w, n, e):
    """Sutherland-Hodgman against a box; the result may run along the box edge, which fills fine."""
    for axis, value, sign in ((0, s, 1), (0, n, -1), (1, w, 1), (1, e, -1)):
        if not ring:
            break
        out = []
        for k, cur in enumerate(ring):
            prev = ring[k - 1]
            keep_cur, keep_prev = (cur[axis] - value) * sign >= 0, (prev[axis] - value) * sign >= 0
            if keep_cur != keep_prev:
                t = (value - prev[axis]) / (cur[axis] - prev[axis])
                point = [round(prev[0] + (cur[0] - prev[0]) * t), round(prev[1] + (cur[1] - prev[1]) * t)]
                point[axis] = value
                out.append(tuple(point))
            if keep_cur:
                out.append(cur)
        ring = out
    return [p for k, p in enumerate(ring) if p != ring[k - 1]]


def significance(points: list[tuple[int, int]]) -> list[int]:
    """Per point, the largest Douglas-Peucker tolerance that keeps it (never more than its parent's)."""
    k = math.cos(math.radians(sum(p[0] for p in points) / len(points) / 100))
    sig = [255] * len(points)
    stack = [(0, len(points) - 1, 255)]
    while stack:
        a, b, cap = stack.pop()
        if b - a < 2:
            continue
        (ay, ax), (by, bx) = points[a], points[b]
        ax, bx = ax * k, bx * k
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        best, best_i = -1.0, a + 1
        for i in range(a + 1, b):
            py, px = points[i][0], points[i][1] * k
            d = abs(dx * (ay - py) - dy * (ax - px)) / length if length else math.hypot(px - ax, py - ay)
            if d > best:
                best, best_i = d, i
        level = min(cap, 254, math.ceil(best))
        sig[best_i] = level
        stack.append((a, best_i, level))
        stack.append((best_i, b, level))
    return sig


class Layer:
    def __init__(self):
        self.lat, self.lon, self.sig = array("h"), array("h"), array("B")
        self.start, self.zoom, self.box = array("i", [0]), array("B"), array("h")

    def add(self, min_zoom: float, points: list[tuple[int, int]]) -> None:
        self.lat.extend(p[0] for p in points)
        self.lon.extend(p[1] for p in points)
        self.sig.extend(significance(points))
        self.start.append(len(self.lat))
        self.zoom.append(round(min_zoom * 10))
        lats, lons = [p[0] for p in points], [p[1] for p in points]
        self.box.extend((min(lats), min(lons), max(lats), max(lons)))

    def arrays(self, name: str) -> dict:
        return {f"{name}.{k}": getattr(self, k) for k in ("lat", "lon", "sig", "start", "zoom", "box")}


def zoom(props) -> float | None:
    value = props.get("min_zoom") or 0.0
    return None if value > MAX_ZOOM else value


def lines(directory: Path, names: list[str], select) -> Layer:
    out = Layer()
    for name in names:
        for props, geometry in features(directory, name):
            min_zoom = select(props)
            if min_zoom is None:
                continue
            for part in parts(geometry):
                for run in clip_line(quantize(part)):
                    out.add(min_zoom, run)
    return out


def areas(directory: Path, names: list[str], select) -> Layer:
    out = Layer()
    for name in names:
        for props, geometry in features(directory, name):
            min_zoom = select(props)
            if min_zoom is None:
                continue
            for part in parts(geometry):
                ring = quantize(part)
                if ring and ring[0] == ring[-1]:
                    ring.pop()
                if len(ring) < 3:
                    continue
                lats, lons = [p[0] for p in ring], [p[1] for p in ring]
                for ts in range(int(SOUTH), int(NORTH), TILE):
                    for tw in range(int(WEST), int(EAST), TILE):
                        s, w, n, e = ts * 100, tw * 100, (ts + TILE) * 100, (tw + TILE) * 100
                        if max(lats) < s or min(lats) > n or max(lons) < w or min(lons) > e:
                            continue
                        piece = clip_ring(ring, s, w, n, e)
                        if len(piece) >= 3:
                            # Closed, so simplification keeps both ends of the ring.
                            out.add(min_zoom, piece + [piece[0]])
    return out


def places(path: Path) -> dict:
    rows = []
    for row in zipfile.ZipFile(path).read(path.stem + ".txt").decode().splitlines():
        cols = row.split("\t")
        lat, lon, code, population = float(cols[4]), float(cols[5]), cols[7], int(cols[14] or 0)
        if code not in NOT_TOWNS and SOUTH <= lat <= NORTH and WEST <= lon <= EAST:
            rows.append((round(lat * 100), round(lon * 100), population, cols[1]))
    rows.sort(key=lambda r: -r[2])
    return {
        "places.lat": array("h", [r[0] for r in rows]),
        "places.lon": array("h", [r[1] for r in rows]),
        "places.pop": array("i", [r[2] for r in rows]),
        "places.names": array("B", "\n".join(r[3] for r in rows).encode()),
    }


def is_motorway(p) -> bool:
    return p.get("expressway") == 1 or p["type"] == "Major Highway"


def npy(values: array) -> bytes:
    """An .npy file of a 1-d array (format version 1.0)."""
    descr = {"h": "<i2", "B": "|u1", "i": "<i4"}[values.typecode]
    header = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': ({len(values)},), }}"
    header += " " * (63 - (10 + len(header)) % 64) + "\n"
    assert sys.byteorder == "little"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode() + values.tobytes()


def main(directory: str) -> None:
    d = Path(directory)
    rivers = lambda p: None if p.get("featurecla") == "Lake Centerline" else zoom(p)
    world_rivers = "ne_10m_rivers_lake_centerlines"
    layers = {
        "land": areas(d, ["ne_10m_land"], lambda p: 0.0),
        "lake": areas(d, ["ne_10m_lakes", "ne_10m_lakes_europe"], zoom),
        # Lake shores as lines: the outlines of the lake areas include the cuts between tiles and at the map edge.
        "lake_shore": lines(d, ["ne_10m_lakes", "ne_10m_lakes_europe"], zoom),
        "coastline": lines(d, ["ne_10m_coastline"], lambda p: 0.0),
        "border": lines(d, ["ne_10m_admin_0_boundary_lines_land"], zoom),
        # Big rivers (Natural Earth's world layer, scalerank ≤ 6) are drawn wider.
        "river_major": lines(d, [world_rivers], lambda p: rivers(p) if p["scalerank"] <= 6 else None),
        "river": lines(d, [world_rivers, "ne_10m_rivers_europe"],
                       lambda p: rivers(p) if p["scalerank"] > 6 else None),
        "motorway": lines(d, ["ne_10m_roads"], lambda p: zoom(p) if p["featurecla"] == "Road" and is_motorway(p)
                          else None),
        "road": lines(d, ["ne_10m_roads"], lambda p: zoom(p) if p["featurecla"] == "Road" and not is_motorway(p)
                      else None),
    }
    arrays = {k: v for name, layer in layers.items() for k, v in layer.arrays(name).items()}
    arrays |= places(d / "cities15000.zip")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, values in arrays.items():
            # Fixed timestamps, so the same inputs give the same file.
            z.writestr(zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0)), npy(values), zipfile.ZIP_DEFLATED)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_bytes(buffer.getvalue())


if __name__ == "__main__":
    main(*sys.argv[1:])
