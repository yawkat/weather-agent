from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from weather_core.geometry import (
    MAX_ROUTE_SAMPLES, LatLon, area_grid, bearing_deg, crosswind, decode_polyline, encode_polyline, haversine_km, headwind,
    parse_gpx, sample_route, simplify,
)

START = datetime(2026, 10, 10, 10, 0, tzinfo=timezone.utc)


def test_haversine_and_bearing():
    # One degree of latitude is ~111.2 km; due north is bearing 0, due east ~90.
    assert haversine_km(50.0, 7.0, 51.0, 7.0) == pytest.approx(111.2, abs=0.1)
    assert bearing_deg(50.0, 7.0, 51.0, 7.0) == pytest.approx(0.0, abs=1e-6)
    assert bearing_deg(50.0, 7.0, 50.0, 8.0) == pytest.approx(90.0, abs=0.5)


def test_polyline_round_trip_matches_reference():
    # Reference example from Google's polyline documentation.
    points = [LatLon(38.5, -120.2), LatLon(40.7, -120.95), LatLon(43.252, -126.453)]
    assert encode_polyline(points) == "_p~iF~ps|U_ulLnnqC_mqNvxq`@"
    decoded = decode_polyline("_p~iF~ps|U_ulLnnqC_mqNvxq`@")
    assert [(p.lat, p.lon) for p in decoded] == [(38.5, -120.2), (40.7, -120.95), (43.252, -126.453)]


def test_parse_gpx_keeps_complete_timestamps():
    gpx = """<?xml version="1.0"?>
    <gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>
      <trkpt lat="50.0" lon="7.0"><time>2026-06-01T08:00:00Z</time></trkpt>
      <trkpt lat="50.1" lon="7.0"><time>2026-06-01T08:30:00Z</time></trkpt>
    </trkseg></trk></gpx>"""
    track = parse_gpx(gpx)
    assert track.points == [LatLon(50.0, 7.0), LatLon(50.1, 7.0)]
    assert track.times[1] - track.times[0] == timedelta(minutes=30)


def test_parse_gpx_without_times():
    gpx = '<gpx><rte><rtept lat="50" lon="7"/><rtept lat="50.2" lon="7.1"/></rte></gpx>'
    assert parse_gpx(gpx).times is None


def test_simplify_drops_collinear_points():
    line = [LatLon(50.0, 7.0 + i * 0.01) for i in range(50)]
    assert simplify(line) == [line[0], line[-1]]
    bent = line + [LatLon(50.5, 7.49)]
    assert len(simplify(bent)) == 3


def test_route_timing_at_constant_speed():
    # ~22.2 km due north at 22.2 km/h: one hour.
    route = sample_route([LatLon(50.0, 7.0), LatLon(50.2, 7.0)], START, speed_kmh=22.24)
    assert route.total_km == pytest.approx(22.24, abs=0.05)
    assert route.duration.total_seconds() == pytest.approx(3600, abs=10)
    assert route.dt_hours.sum() == pytest.approx(1.0, abs=0.01)
    assert np.allclose(route.bearing, 0.0, atol=1e-6)
    assert route.times[0] == START


def test_route_timing_from_gpx_times_is_shifted_to_start():
    t0 = datetime(2026, 6, 1, 8, 0, tzinfo=timezone.utc)
    route = sample_route([LatLon(50.0, 7.0), LatLon(50.1, 7.0), LatLon(50.2, 7.0)], START,
                         track_times=[t0, t0 + timedelta(minutes=10), t0 + timedelta(minutes=70)])
    assert route.times[0] == START
    assert route.duration == timedelta(minutes=70)


def test_headwind_sign():
    # Riding north (bearing 0) into a north wind (blowing from 0°) is a full headwind.
    assert headwind(20.0, 0.0, 0.0) == pytest.approx(20.0)
    # A south wind pushes a northbound rider: tailwind, negative.
    assert headwind(20.0, 180.0, 0.0) == pytest.approx(-20.0)
    # An east wind is pure crosswind for a northbound rider.
    assert headwind(20.0, 90.0, 0.0) == pytest.approx(0.0, abs=1e-9)
    assert crosswind(20.0, 90.0, 0.0) == pytest.approx(20.0)


def test_area_grid_covers_the_box_at_about_10_km():
    lats, lons = area_grid((50.0, 7.0, 50.5, 7.5), spacing_km=10)
    assert lats[0] == 50.0 and lats[-1] <= 50.5 and lons[0] == 7.0 and lons[-1] <= 7.5
    assert np.diff(lats).mean() * 111.2 == pytest.approx(10, rel=0.01)
    assert np.diff(lons).mean() * 111.2 * np.cos(np.radians(50.25)) == pytest.approx(10, rel=0.01)


def test_area_grid_is_capped():
    lats, lons = area_grid((45.0, 0.0, 55.0, 20.0), spacing_km=1, max_points=500)
    assert len(lats) * len(lons) <= 500


def test_tiny_area_grid_has_a_point():
    lats, lons = area_grid((50.0, 7.0, 50.01, 7.01))
    assert len(lats) == len(lons) == 1


def test_crafted_polyline_is_rejected_fast():
    import time

    started = time.perf_counter()
    for crafted in ["~" * 99_000, "~" * 200_000, "_p~iF~ps|U\x00"]:
        with pytest.raises(ValueError):
            decode_polyline(crafted)
    assert time.perf_counter() - started < 1.0


def test_gpx_with_doctype_or_missing_attributes_is_rejected():
    entity = '<?xml version="1.0"?><!DOCTYPE gpx [<!ENTITY a "aaaa">]><gpx><trk><trkseg>' \
             '<trkpt lat="50" lon="7"/><trkpt lat="51" lon="7"/></trkseg></trk></gpx>'
    with pytest.raises(ValueError, match="DOCTYPE"):
        parse_gpx(entity)
    with pytest.raises(ValueError, match="lat/lon"):
        parse_gpx('<gpx><trk><trkseg><trkpt lon="1"/><trkpt lat="2" lon="1"/></trkseg></trk></gpx>')


def test_long_routes_are_capped():
    route = sample_route([LatLon(35.0, -25.0), LatLon(70.0, 40.0)], START, speed_kmh=200)
    assert len(route.times) <= MAX_ROUTE_SAMPLES


def test_oversized_or_invalid_areas_are_rejected():
    for bbox in [(-90.0, -180.0, 90.0, 180.0), (float("nan"), 0.0, 10.0, 10.0), (50.0, 7.0, 49.0, 8.0)]:
        with pytest.raises(ValueError):
            area_grid(bbox)
