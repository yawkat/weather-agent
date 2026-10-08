"""Points, routes and areas, and how a query samples them in space and time."""

import math
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

EARTH_RADIUS_KM = 6371.0088

# Input limits. Tool arguments come from an LLM or an unauthenticated client, so every size is bounded before
# any work proportional to it happens.
MAX_POLYLINE_CHARS = 100_000
MAX_GPX_CHARS = 5_000_000
MAX_ROUTE_POINTS = 20_000  # decoded input points, before resampling
MAX_ROUTE_SAMPLES = 2_000  # resampled points; spacing grows for very long routes
MAX_AREA_SPAN_DEG = 30.0


@dataclass(frozen=True)
class LatLon:
    lat: float
    lon: float


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance; works elementwise on numpy arrays."""
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing from point 1 to point 2, degrees clockwise from north."""
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    y = np.sin(lon2 - lon1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return np.degrees(np.arctan2(y, x)) % 360.0


# ---------------------------------------------------------------------------------------------------------------
# Encoded polylines (Google's format, precision 5): compact enough for an LLM to pass routes between tool calls.

def encode_polyline(points: list[LatLon], precision: int = 5) -> str:
    factor = 10 ** precision
    out = []
    prev_lat = prev_lon = 0
    for p in points:
        lat, lon = round(p.lat * factor), round(p.lon * factor)
        for delta in (lat - prev_lat, lon - prev_lon):
            value = ~(delta << 1) if delta < 0 else delta << 1
            while value >= 0x20:
                out.append(chr((0x20 | (value & 0x1F)) + 63))
                value >>= 5
            out.append(chr(value + 63))
        prev_lat, prev_lon = lat, lon
    return "".join(out)


def decode_polyline(encoded: str, precision: int = 5) -> list[LatLon]:
    if len(encoded) > MAX_POLYLINE_CHARS:
        raise ValueError(f"polyline longer than {MAX_POLYLINE_CHARS} characters; simplify the route")
    factor = 10 ** precision
    points = []
    index = lat = lon = 0
    while index < len(encoded):
        deltas = []
        for _ in range(2):
            result = shift = 0
            while True:
                if index >= len(encoded):
                    raise ValueError("truncated polyline")
                b = ord(encoded[index]) - 63
                if not 0 <= b < 64:
                    raise ValueError(f"invalid polyline character at position {index}")
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
                # A coordinate delta fits in 7 chunks; unbounded chunks would make decoding quadratic.
                if shift >= 35:
                    raise ValueError(f"invalid polyline value near position {index}")
            deltas.append(~(result >> 1) if result & 1 else result >> 1)
        lat += deltas[0]
        lon += deltas[1]
        if len(points) >= MAX_ROUTE_POINTS:
            raise ValueError(f"route has more than {MAX_ROUTE_POINTS} points; simplify it first")
        point = LatLon(lat / factor, lon / factor)
        if not (-90 <= point.lat <= 90 and -180 <= point.lon <= 180):
            raise ValueError("polyline decodes to coordinates outside the globe; is it precision 5?")
        points.append(point)
    return points


# ---------------------------------------------------------------------------------------------------------------
# GPX

@dataclass(frozen=True)
class Track:
    points: list[LatLon]
    times: list[datetime] | None  # from GPX timestamps, if every point has one


def parse_gpx(text: str) -> Track:
    """Track points (or route points) of the first track/route; timestamps are kept if complete."""
    if len(text) > MAX_GPX_CHARS:
        raise ValueError(f"GPX larger than {MAX_GPX_CHARS} characters")
    # GPX never needs a DTD; refusing it rules out entity-expansion tricks whatever XML backend is in use.
    if "<!DOCTYPE" in text or "<!ENTITY" in text:
        raise ValueError("GPX with a DOCTYPE or entity declarations is not accepted")
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as e:
        raise ValueError(f"invalid GPX: {e}") from None
    points: list[LatLon] = []
    times: list[datetime | None] = []
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag in ("trkpt", "rtept"):
            if len(points) >= MAX_ROUTE_POINTS:
                raise ValueError(f"GPX has more than {MAX_ROUTE_POINTS} points; simplify it first")
            try:
                point = LatLon(float(element.attrib["lat"]), float(element.attrib["lon"]))
            except (KeyError, ValueError):
                raise ValueError("GPX point without valid lat/lon attributes") from None
            if not (-90 <= point.lat <= 90 and -180 <= point.lon <= 180):
                raise ValueError("GPX point outside the globe")
            points.append(point)
            time_text = next((child.text for child in element if child.tag.rsplit("}", 1)[-1] == "time"), None)
            try:
                times.append(datetime.fromisoformat(time_text.replace("Z", "+00:00")) if time_text else None)
            except ValueError:
                times.append(None)
    if len(points) < 2:
        raise ValueError("GPX contains fewer than two track or route points")
    complete = all(t is not None for t in times)
    return Track(points, times if complete else None)


def simplify(points: list[LatLon], tolerance_km: float = 0.05) -> list[LatLon]:
    """Douglas-Peucker on a local equirectangular projection; keeps the route shape at weather-grid scale."""
    if len(points) < 3:
        return list(points)
    lat0 = math.radians(sum(p.lat for p in points) / len(points))
    xy = np.array([[math.radians(p.lon) * math.cos(lat0), math.radians(p.lat)] for p in points]) * EARTH_RADIUS_KM
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        if end - start < 2:
            continue
        segment = xy[end] - xy[start]
        length = np.hypot(*segment)
        offsets = xy[start + 1:end] - xy[start]
        if length == 0:
            distances = np.hypot(offsets[:, 0], offsets[:, 1])
        else:
            distances = np.abs(segment[0] * offsets[:, 1] - segment[1] * offsets[:, 0]) / length
        worst = int(np.argmax(distances))
        if distances[worst] > tolerance_km:
            middle = start + 1 + worst
            keep[middle] = True
            stack.append((start, middle))
            stack.append((middle, end))
    return [p for p, k in zip(points, keep) if k]


# ---------------------------------------------------------------------------------------------------------------
# Route sampling

@dataclass(frozen=True)
class RouteSamples:
    """Positions along a route, each with the time the rider is there and the direction of travel."""
    lat: np.ndarray
    lon: np.ndarray
    times: list[datetime]
    bearing: np.ndarray  # direction of travel, degrees from north
    distance_km: np.ndarray  # from the start
    dt_hours: np.ndarray  # riding time each sample stands for (sums to the total duration)

    @property
    def total_km(self) -> float:
        return float(self.distance_km[-1])

    @property
    def duration(self) -> timedelta:
        return self.times[-1] - self.times[0]


def sample_route(points: list[LatLon], start: datetime, speed_kmh: float | None = None,
                 track_times: list[datetime] | None = None, spacing_km: float = 2.0) -> RouteSamples:
    """Resample a route every ~spacing_km and time each sample.

    Timing comes from a constant average speed, or from GPX timestamps shifted so the track starts at `start`.
    """
    if (speed_kmh is None) == (track_times is None):
        raise ValueError("give either speed_kmh or track_times")
    if speed_kmh is not None and not 0 < speed_kmh <= 200:
        raise ValueError("speed must be between 0 and 200 km/h")
    if track_times is not None and any(t.tzinfo is None for t in track_times):
        track_times = [t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t for t in track_times]
    lat = np.array([p.lat for p in points])
    lon = np.array([p.lon for p in points])
    segment_km = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    cumulative = np.concatenate([[0.0], np.cumsum(segment_km)])
    total = float(cumulative[-1])
    if total == 0:
        raise ValueError("route has zero length")
    count = min(MAX_ROUTE_SAMPLES, max(2, int(math.ceil(total / spacing_km)) + 1))
    at = np.linspace(0.0, total, count)
    sample_lat = np.interp(at, cumulative, lat)
    sample_lon = np.interp(at, cumulative, lon)
    # Bearing of the original segment each sample lies on.
    segment_index = np.clip(np.searchsorted(cumulative, at, side="right") - 1, 0, len(segment_km) - 1)
    segment_bearing = bearing_deg(lat[:-1], lon[:-1], lat[1:], lon[1:])
    bearing = segment_bearing[segment_index]
    if speed_kmh is not None:
        offsets_h = at / speed_kmh
    else:
        track_h = np.array([(t - track_times[0]).total_seconds() / 3600 for t in track_times])
        offsets_h = np.interp(at, cumulative, track_h)
    times = [start + timedelta(hours=float(h)) for h in offsets_h]
    # Each sample stands for half the time to its neighbours.
    edges = np.concatenate([[offsets_h[0]], (offsets_h[:-1] + offsets_h[1:]) / 2, [offsets_h[-1]]])
    dt_hours = np.diff(edges)
    return RouteSamples(sample_lat, sample_lon, times, bearing, at, dt_hours)


def headwind(wind_speed, wind_from_deg, travel_bearing_deg):
    """Wind component against the direction of travel; negative values are tailwind.

    `wind_from_deg` is the meteorological direction the wind blows *from*.
    """
    return wind_speed * np.cos(np.radians(wind_from_deg - travel_bearing_deg))


def crosswind(wind_speed, wind_from_deg, travel_bearing_deg):
    """Magnitude of the wind component across the direction of travel."""
    return np.abs(wind_speed * np.sin(np.radians(wind_from_deg - travel_bearing_deg)))


# ---------------------------------------------------------------------------------------------------------------
# Areas

def _check_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    south, west, north, east = bbox
    if not all(math.isfinite(v) for v in bbox):
        raise ValueError("bbox values must be finite numbers")
    if not (-90 <= south < north <= 90 and -180 <= west < east <= 180):
        raise ValueError("bbox must be (south, west, north, east) within ±90° / ±180°")
    if north - south > MAX_AREA_SPAN_DEG or east - west > MAX_AREA_SPAN_DEG:
        raise ValueError(f"area spans more than {MAX_AREA_SPAN_DEG:.0f}°; choose a smaller one")
    return south, west, north, east


def area_grid(bbox: tuple[float, float, float, float], spacing_km: float = 10.0,
              max_points: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Latitude and longitude axes of a regular grid over a bbox (south, west, north, east), with at most
    max_points points together; the spacing grows for large areas."""
    bbox = _check_bbox(bbox)
    spacing_km = _initial_spacing(bbox, spacing_km, max_points)
    while True:
        lats, lons = _grid_axes(bbox, spacing_km)
        if len(lats) * len(lons) <= max_points:
            return lats, lons
        spacing_km *= 1.05


def _km_per_lon(bbox: tuple[float, float, float, float]) -> float:
    return 111.2 * max(math.cos(math.radians((bbox[0] + bbox[2]) / 2)), 1e-3)


def _initial_spacing(bbox: tuple[float, float, float, float], spacing_km: float, max_points: int) -> float:
    """Spacing for which the bbox grid has about max_points points at most, chosen up front so the first grid
    is never oversized."""
    south, west, north, east = bbox
    estimate = ((north - south) * 111.2 / spacing_km + 1) * ((east - west) * _km_per_lon(bbox) / spacing_km + 1)
    return spacing_km * math.sqrt(estimate / max_points) * 1.05 if estimate > max_points else spacing_km


def _grid_axes(bbox: tuple[float, float, float, float], spacing_km: float) -> tuple[np.ndarray, np.ndarray]:
    south, west, north, east = bbox
    return (np.arange(south, north + 1e-9, spacing_km / 111.2),
            np.arange(west, east + 1e-9, spacing_km / _km_per_lon(bbox)))
