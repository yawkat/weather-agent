"""Answers as charts (Forecaster.visualize, weather_core.chart) over a synthetic source."""

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from weather_core.chart import basemap, encode_chart
from weather_core.evaluate import SourceInfo
from weather_core.expr import runtime as rt
from weather_core.expr.axes import NUM, ExprError, Type
from weather_core.forecast import Forecaster
from weather_core.sources.base import Prepared

RUN = datetime(2026, 10, 7, 0, tzinfo=timezone.utc)
MEMBERS = 10
STEPS = list(range(0, 73))
POINT = 'forecast().interp(lat=50.0, lon=7.0).sel(time=slice("2026-10-08T10:00", "2026-10-08T13:00"))'
AREA = ('forecast().sel(lat=slice(50.5, 51.0), lon=slice(5.8, 6.3), '
        'time=slice("2026-10-08T10:00", "2026-10-08T13:00"))')


class Ramp:
    """Member m rains m × 0.1 mm/h everywhere; t2m = 10 + 0.5 m + 0.1 × hour."""

    name = "ramp"

    def __init__(self):
        self.prepared = 0

    def provides(self):
        return {"precip", "t2m"}

    def describe(self):
        return {"resolution": "test", "note": "synthetic"}

    def prepare(self, query, variables):
        self.prepared += 1
        m = np.arange(MEMBERS)[:, None] * np.ones((1, len(query.lat)))
        per_step = {}
        if "precip" in variables:
            per_step["precip"] = {i: m * 0.1 for i in range(len(STEPS))}
        if "t2m" in variables:
            per_step["t2m"] = {i: 10 + 0.5 * m + 0.1 * STEPS[i] for i in range(len(STEPS))}
        info = SourceInfo(self.name, "Ramp", "test", RUN, MEMBERS)
        return Prepared(query, RUN, STEPS, per_step, {"precip"}, set(), info, 0,
                        lambda times: {"provider": "test", "model": self.name})


def forecaster(source=None, **kwargs):
    return Forecaster([source or Ramp()], default_tz="UTC", clock=lambda: RUN, **kwargs)


def test_every_member_over_time():
    answer, chart = forecaster().visualize(POINT + ".t2m")
    (field,) = chart["fields"]
    assert field["dims"] == ["model", "member", "time"] and field["unit"] == "°C"
    assert field["coords"]["model"] == {"labels": ["ramp"]}
    times = field["coords"]["time"]
    assert times["step"] == "1h" and len(times["minutes"]) == 3
    assert times["minutes"][1] - times["minutes"][0] == 60
    # Flat, row-major: member 2 at 11:00 (+35 h) = 10 + 1 + 3.5.
    assert field["data"][2 * 3 + 1] == 14.5
    assert "basemap" not in chart  # a single point isn't a map
    assert chart["models"][0]["model"] == "ramp" and chart["attribution"] == answer["attribution"]
    assert chart["timezone"] == "UTC"
    assert answer["result"]["columns"] == ["model", "member", "time", "value"]  # 30 rows fit in a table


def test_record_with_quantiles_and_probabilities():
    _, chart = forecaster().visualize(
        f'{{"t": {POINT}.t2m.quantile([0.1, 0.5, 0.9], "member"), "wet": ({POINT}.precip > 0.25).mean("member")}}')
    t, wet = chart["fields"]
    assert (t["name"], wet["name"]) == ("t", "wet")
    assert t["dims"] == ["model", "time", "quantile"] and t["coords"]["quantile"] == {"values": [0.1, 0.5, 0.9]}
    assert wet["dims"] == ["model", "time"] and wet["unit"] == ""
    assert wet["data"][0] == 0.7  # members 3…9 rain more than 0.25 mm/h


def test_area_is_a_map_with_coastlines_and_borders():
    _, chart = forecaster().visualize(f'({AREA}.precip.sum("time") > 1).mean("member")')
    (field,) = chart["fields"]
    assert field["dims"] == ["model", "lat", "lon"]
    lats, lons = field["coords"]["lat"]["values"], field["coords"]["lon"]["values"]
    assert len(field["data"]) == len(lats) * len(lons) > 4
    south, west, north, east = chart["basemap"]["bounds"]
    assert south < min(lats) and north > max(lats) and west < min(lons) and east > max(lons)
    assert chart["basemap"]["border"]  # the corner of Germany, Belgium and the Netherlands
    assert "Natural Earth" in chart["basemap"]["source"]


@pytest.mark.parametrize("query, message", [
    (f'{AREA}.precip.sum("time")', "reduce member"),
    (f'{POINT}.t2m.quantile([0.1, 0.9], "model")', "either members or quantiles"),
    (f'{POINT}.t2m.median("member").idxmax("time")', "only labels"),
])
def test_unchartable_answers_are_rejected_before_downloading(query, message):
    source = Ramp()
    with pytest.raises(ExprError, match=message):
        forecaster(source).visualize(query)
    assert source.prepared == 0


def test_infinite_values_are_null():
    # Member 0 has no rain: dividing by its total gives infinity, which JSON can't carry.
    _, chart = forecaster().visualize(f'{POINT}.t2m.mean("time") / {POINT}.precip.sum("time")')
    data = chart["fields"][0]["data"]
    assert data[0] is None and all(isinstance(x, float) for x in data[1:])


def test_long_answers_get_a_summary_instead_of_a_table():
    answer, chart = forecaster(max_rows=5).visualize(POINT + ".t2m")
    summary = answer["result"]["too_long_for_a_table"]["value"]
    assert summary["dims"] == {"model": 1, "member": 10, "time": 3}
    assert summary["min"] == 13.4 and summary["max"] == 18.1
    assert len(chart["fields"][0]["data"]) == 30
    with pytest.raises(ExprError, match="rows"):
        forecaster(max_rows=5).forecast(POINT + ".t2m")


def test_value_limit():
    ctx = rt.Context(timezone.utc, 1 << 30, float("inf"))
    big = rt.Arr(np.zeros(11), ("member",), {"member": rt.Coord(rt.labels(range(11)))})
    with pytest.raises(ExprError, match="11 values"):
        encode_chart(big, Type(NUM, ("member",)), ctx, max_values=10)
    assert encode_chart(big, Type(NUM, ("member",)), ctx, max_values=11)["fields"][0]["data"] == [0.0] * 11


def test_basemap_is_clipped_to_the_map():
    m = basemap(49.5, 5.5, 51.5, 7.5)
    assert m["bounds"] == [49.0, 5.0, 52.0, 8.0]
    inside = lambda a, b: 49.0 <= a <= 52.0 and 5.0 <= b <= 8.0
    for name in ("lake_shore", "coastline", "border", "river_major", "river", "motorway", "road"):
        for line in m[name]:
            # At most its ends reach beyond the box, so the line reaches the edge.
            assert len(line) >= 4 and all(inside(a, b) for a, b in zip(line[2:-2:2], line[3:-2:2]))
    for name in ("land", "lake"):
        for ring in m[name]:
            assert len(ring) >= 6 and all(inside(a, b) for a, b in zip(ring[0::2], ring[1::2]))


def test_basemap_shows_rivers_roads_and_towns_at_its_scale():
    region = basemap(50.5, 6.5, 51.5, 7.5)  # around Cologne
    names = [p[2] for p in region["places"]]
    assert names[0] == "Köln" and "Bonn" in names and len(names) > 20  # largest first
    assert region["river_major"] and region["river"] and region["motorway"] and region["road"]
    assert region["land"] and not region["coastline"]
    europe = basemap(36, -10, 66, 20)  # the largest area a query may ask for
    # Zoomed out: big rivers only.
    assert europe["river_major"] and not europe["river"]
    assert len(europe["places"]) == 150 and "Paris" in [p[2] for p in europe["places"][:10]]
    assert len(json.dumps(europe)) < 600_000


def test_lake_shores_have_no_cuts():
    # Lake Peipus crosses the 27°E line between area tiles: its fill comes in two pieces, its shore doesn't run
    # along the cut.
    m = basemap(58.0, 26.5, 59.0, 28.0)
    assert len(m["lake"]) >= 2 and m["lake_shore"]
    for line in m["lake_shore"]:
        lons = line[1::2]
        assert not any(a == b == 27.0 for a, b in zip(lons, lons[1:]))


def test_one_grid_coordinate_left_is_a_profile_not_a_map():
    # Averaging over lon leaves a north-south profile: a graph along lat (members allowed), no basemap.
    _, chart = forecaster().visualize(f'{AREA}.precip.sum("time").mean("lon")')
    (field,) = chart["fields"]
    assert field["dims"] == ["model", "member", "lat"]
    assert "basemap" not in chart


def test_label_fields_are_named_as_omitted():
    _, chart = forecaster().visualize(
        f'{{"tmax": {POINT}.t2m.median("member").max("time"), "when": {POINT}.t2m.median("member").idxmax("time")}}')
    assert [f["name"] for f in chart["fields"]] == ["tmax"]
    assert chart["omitted"] == ["when"]
