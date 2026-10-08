"""Queries in the server log keep their shape but not their locations."""

from zoneinfo import ZoneInfo

import pytest

from weather_core.expr.axes import ExprError
from weather_core.expr.language import Env, compile_query, parse
from weather_core.expr.redact import Redacted


def test_point_keeps_shape_hides_coordinates():
    r = Redacted("(forecast().interp(lat=50.94, lon=6.96).sel(time=slice('2026-10-10T10:00', '2026-10-10T16:00'))"
                 ".precip.sum('time') < 0.5).mean('member')")
    assert "50.94" not in r.text and "6.96" not in r.text
    assert "lat=..., lon=..." in r.text and "time=slice(" in r.text
    assert "'2026-10-10T10:00'" in r.text and "precip.sum('time') < 0.5" in r.text


def test_places_and_bindings():
    r = Redacted("home = (50.9, 6.9)\nkoeln = forecast().interp(places(Köln=(50.94, 6.96), Bonn=(50.73, 7.1)))\n"
                 "koeln.sel(point='Bonn', model='ifs-ens').t2m.max('time')", keep=["ifs-ens"])
    for secret in ("50.9", "6.9", "Köln", "Bonn", "koeln", "home", "7.1"):
        assert secret not in r.text, secret
    assert "places(p1=(..., ...), p2=(..., ...))" in r.text
    assert "'ifs-ens'" in r.text and ".t2m.max('time')" in r.text


def test_place_strings_and_polylines():
    r = Redacted("forecast().interp(places('Köln@50.94,6.96; Bonn@50.73,7.10')).sel(time='2026-10-10').t2m"
                 ".mean('member') + forecast().interp(route(polyline='_p~iF~ps|U', start='2026-10-10T09:00', "
                 "speed_kmh=20)).wind.max()")
    assert "Köln" not in r.text and "_p~iF" not in r.text and "50.94" not in r.text
    assert "speed_kmh=20" in r.text and "start='2026-10-10T09:00'" in r.text


def test_areas_and_coordinate_comparisons():
    r = Redacted("fc = forecast().sel(lat=slice(50.5, 51.2), lon=slice(6.5, 7.5))\n(fc.lat > 50.8) & (fc.t2m > 25)")
    for secret in ("50.5", "51.2", "6.5", "7.5", "50.8"):
        assert secret not in r.text, secret
    assert "t2m > 25" in r.text


def _error(query: str) -> ExprError:
    env = Env(("ifs-ens",), {"ifs-ens": {"t2m", "precip"}}, ZoneInfo("Europe/Berlin"))
    with pytest.raises(ExprError) as e:
        compile_query(parse(query), env)
    return e.value


def test_error_location_is_redacted():
    query = "fc = forecast().interp(places(Köln=(50.94, 6.96)))\nfc.koeln.sel(point='Köln').max('time')"
    r = Redacted(query, keep=["ifs-ens"])
    assert r.at(_error(query).span) == "v1.a1"
    assert "koeln" not in r.text and "Köln" not in r.text


def test_error_without_span():
    r = Redacted("forecast().t2m")
    assert r.at(None) is None and Redacted("(").at((1, 0, 1, 1)) is None


def test_unknown_keywords_are_aliased():
    r = Redacted("forecast().interp(lat=50.94, lon=6.96).t2m.rolling(time=3, koeln=1).mean('member')")
    assert "rolling(time=3, k1=1)" in r.text


def test_unparseable():
    r = Redacted("forecast().interp(lat=50.94, lon=6.96")
    assert str(r) == "<unparseable or too large, 37 characters>"


def test_too_large_is_not_walked():
    query = "forecast().interp(lat=50.94, lon=6.96).t2m" + " + 1" * 30_000
    assert str(Redacted(query)) == f"<unparseable or too large, {len(query)} characters>"


def test_vocabulary_bindings():
    r = Redacted("precip = forecast().interp(lat=50.94, lon=6.96).precip\nprecip.sel(point='Bonn').sum('time')")
    assert r.text.startswith("precip = ")


def test_place_labels_that_look_like_vocabulary():
    r = Redacted("forecast().interp(places(time=(50.9, 6.9), **{'ifs-ens': (51.0, 7.0)})).t2m", keep=["ifs-ens"])
    assert "time" not in r.text and "ifs-ens" not in r.text

