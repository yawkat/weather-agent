"""Forecast queries (the xarray-like expression language), end to end over synthetic sources."""

from datetime import datetime, timezone

import numpy as np
import pytest

from weather_core.evaluate import SourceInfo
from weather_core.expr.axes import ExprError
from weather_core.expr.help import describe_language
from weather_core.expr.runtime import quantiles_last
from weather_core.forecast import Forecaster
from weather_core.geometry import LatLon, encode_polyline
from weather_core.sources.base import Prepared

RUN = datetime(2026, 10, 7, 0, tzinfo=timezone.utc)
STEPS = list(range(0, 73))  # hourly to +72 h


class Synthetic:
    """Member m rains m × scale × (1 + point) mm/h; t2m = 10 + 0.5 m + 0.1 × lead hour; gust = 20 + 5 m."""

    def __init__(self, name="synthetic", members=10, rain_scale=0.1, provides=("precip", "t2m"), point_scale=1.0):
        self.name = name
        self.point_scale = point_scale
        self.members = members
        self.rain_scale = rain_scale
        self._provides = set(provides)
        self.requested = None
        self.query = None

    def provides(self):
        return self._provides

    def describe(self):
        return {"resolution": "test", "note": "synthetic"}

    def prepare(self, query, variables):
        self.requested = sorted(variables)
        self.query = query
        points = len(query.lat)
        m = np.arange(self.members)[:, None] * np.ones((1, points))
        p = np.arange(points)[None, :]
        per_step = {}
        if "precip" in variables:
            per_step["precip"] = {i: m * self.rain_scale * (1 + self.point_scale * p) for i in range(len(STEPS))}
        if "t2m" in variables:
            per_step["t2m"] = {i: 10 + 0.5 * m + 0.1 * STEPS[i] + 0 * p for i in range(len(STEPS))}
        if "gust" in variables and "gust" in self._provides:
            per_step["gust"] = {i: 20 + 5 * m + 0 * p for i in range(len(STEPS))}
        info = SourceInfo(self.name, self.name.title(), "test", RUN, self.members)
        unavailable = {v for v in variables if v not in self._provides}
        return Prepared(query, RUN, STEPS, per_step, {"precip", "gust"}, unavailable, info, 0,
                        lambda times: {"provider": "test", "model": self.name})


def forecaster(*sources, tz="UTC", **kwargs):
    return Forecaster(list(sources) or [Synthetic()], default_tz=tz, clock=lambda: RUN, **kwargs)


WX = 'wx = forecast().interp(lat=50, lon=7).sel(time=slice("2026-10-08T10:00", "2026-10-08T12:00"))\n'  # +34 h … +36 h


def ask(query, *sources, **kwargs):
    return forecaster(*sources, **kwargs).forecast(query)


def rows(result):
    return [dict(zip(result["columns"], row)) for row in result["rows"]]


def by_model(result):
    return {r["model"]: r["value"] for r in rows(result)}


def test_probability_per_model_ignores_padded_members():
    # Two hours: member m of "a" gets 0.2 m mm, so the total is < 0.5 for m = 0, 1, 2. "b" has only 5 members (the
    # member dimension is padded to 10) and every one of them stays dry.
    answer = ask(WX + '(wx.precip.sum("time") < 0.5).mean("member")',
                 Synthetic("a"), Synthetic("b", members=5, rain_scale=0.05))
    assert by_model(answer["result"]) == {"a": 0.3, "b": 1.0}
    assert answer["units"] == ""
    assert [m["model"] for m in answer["models"]] == ["a", "b"]
    assert answer["models"][0]["lead_hours"] == [34, 35]
    assert answer["attribution"] == [{"provider": "test", "model": "a"}, {"provider": "test", "model": "b"}]
    assert answer["warnings"] == []


def test_dict_answers_share_one_table_with_units():
    answer = ask(WX + """rain = wx.precip.sum("time")
{"wet": (rain >= 0.5).mean("member"), "rain": rain.quantile([0.5, 0.9], "member"), "tmax": wx.t2m.max("time").max("member")}""")
    assert rows(answer["result"]) == [{"model": "synthetic", "wet": 0.7, "rain_p50": pytest.approx(0.9),
                                       "rain_p90": pytest.approx(1.62), "tmax": pytest.approx(10 + 4.5 + 3.5)}]
    assert answer["units"] == {"wet": "", "rain": "mm", "tmax": "°C"}


def test_time_slices_exclude_their_end_and_are_local():
    f = forecaster(tz="Europe/Berlin")
    answer = f.forecast('wx = forecast().interp(lat=50, lon=7).sel(time=slice("2026-10-08T10:00", "2026-10-08T12:00"))\n'
                        'wx.t2m.median("member")')
    # 10:00 CEST = 08:00 UTC = +32 h.
    assert rows(answer["result"]) == [
        {"model": "synthetic", "time": "2026-10-08T10:00", "value": pytest.approx(12.25 + 3.2)},
        {"model": "synthetic", "time": "2026-10-08T11:00", "value": pytest.approx(12.25 + 3.3)},
    ]
    day = f.forecast('forecast().interp(lat=50, lon=7).sel(time="2026-10-08").precip.count("time").max(["model", "member"])')
    assert day["result"] == 24.0
    days = f.forecast('forecast().interp(lat=50, lon=7).sel(time=slice("2026-10-08", "2026-10-10")).precip.count("time")'
                      '.max(["model", "member"])')
    assert days["result"] == 48.0


def test_sums_over_time_are_integrals_in_hours():
    answer = ask(WX + '{"rain": wx.precip.sum("time").median("member"), '
                      '"wet_hours": (wx.precip > 0.3).sum("time").median("member"), '
                      '"mean_t": wx.t2m.mean("time").median("member")}')
    assert rows(answer["result"]) == [{"model": "synthetic", "rain": pytest.approx(0.9), "wet_hours": 2.0,
                                       "mean_t": pytest.approx(12.25 + 3.45)}]
    assert answer["units"] == {"rain": "mm", "wet_hours": "h", "mean_t": "°C"}
    assert answer["warnings"] == []


def test_resample_by_local_day():
    f = forecaster(tz="Europe/Berlin")
    answer = f.forecast('forecast().interp(lat=50, lon=7).sel(time=slice("2026-10-08", "2026-10-10")).t2m.resample(time="1D").max()'
                        '.median("member")')
    result = rows(answer["result"])
    assert [r["time"] for r in result] == ["2026-10-08", "2026-10-09"]
    # The last hour of 8 October (local) starts at 21:00 UTC = +45 h.
    assert result[0]["value"] == pytest.approx(12.25 + 4.5)


def test_rolling_windows_and_their_extra_hours():
    source = Synthetic()
    answer = ask('wx = forecast().interp(lat=50, lon=7)\n'
                 'wx.precip.rolling(time=3).sum().sel(time=slice("2026-10-08T10:00", "2026-10-08T12:00"))'
                 '.median("member")', source)
    # Selecting after rolling fetches the two hours before 10:00 too, so the first window is complete.
    assert source.query.hours[0] == datetime(2026, 10, 8, 8, tzinfo=timezone.utc)
    assert [r["value"] for r in rows(answer["result"])] == [pytest.approx(1.35), pytest.approx(1.35)]
    inside = ask(WX + 'wx.precip.rolling(time=2).sum().median("member")')
    # Selecting first leaves the first window incomplete: missing, as in xarray.
    assert [r["value"] for r in rows(inside["result"])] == [None, pytest.approx(0.9)]


def test_top_ranks_one_dimension():
    query = ('day = forecast().interp(lat=50, lon=7).sel(time=slice("2026-10-08T06:00", "2026-10-08T18:00"))\n'
             'p = (day.t2m.rolling(time=2).max() > 15.5).mean("member").min("model")\n')
    answer = ask(query + 'top(p, 3, "time")')
    # Later windows are warmer; ties keep chronological order.
    assert rows(answer["result"]) == [{"time": "2026-10-08T17:00", "value": 0.7},
                                      {"time": "2026-10-08T12:00", "value": 0.6},
                                      {"time": "2026-10-08T13:00", "value": 0.6}]
    worst = ask(query + 'bottom({"p": p, "t": day.t2m.max("member").min("model")}, 1, by="t")')
    assert rows(worst["result"]) == [{"time": "2026-10-08T06:00", "p": None, "t": pytest.approx(17.5)}]
    with pytest.raises(ExprError, match="exactly one remaining dimension"):
        ask(query + 'top(day.t2m.mean("member"), 3, "time")', Synthetic("a"), Synthetic("b"))


def test_models_are_selected_in_the_query():
    a, b = Synthetic("a"), Synthetic("b")
    answer = ask(WX + '(wx.precip.sum("time") < 0.5).mean("member").sel(model="b")', a, b)
    assert answer["result"] == 0.3
    assert a.requested is None and b.requested == ["precip"]
    both = ask(WX + 'wx.sel(model=["a", "b"]).precip.sum("time").mean("member")', Synthetic("a"), Synthetic("b"))
    assert [r["model"] for r in rows(both["result"])] == ["a", "b"]
    with pytest.raises(ExprError, match="unknown model 'nope'"):
        ask(WX + 'wx.precip.sel(model="nope").sum("time")')


def test_places_and_points():
    answer = ask('wx = forecast().interp(places(Köln=(50.94, 6.96), Bonn=(50.73, 7.10))).sel(time=slice("2026-10-08T10:00", '
                 '"2026-10-08T12:00"))\n(wx.precip.sum("time") < 0.5).mean("member").sel(model="synthetic")')
    # Bonn is the second point and rains twice as much.
    assert rows(answer["result"]) == [{"place": "Köln", "value": 0.3}, {"place": "Bonn", "value": 0.2}]
    one = ask('wx = forecast().interp(places("Köln@50.94,6.96; Bonn@50.73,7.10")).sel(time=slice("2026-10-08T10:00", '
              '"2026-10-08T12:00"))\nwx.sel(point="Bonn").precip.sum("time").max(["model", "member"])')
    assert one["result"] == pytest.approx(3.6)


def test_area_anywhere_and_per_point():
    query = 'storm = forecast().sel(lat=slice(49.8, 50.2), lon=slice(6.7, 7.3)).sel(time=slice("2026-10-08T10:00", "2026-10-08T12:00"))\n'
    per_point = ask(query + '(storm.precip.sum("time") < 0.5).mean("member").sel(model="synthetic")')
    table = rows(per_point["result"])
    grid = per_point["locations"][0]["area"]
    n = grid["lat_points"] * grid["lon_points"]
    assert len(table) == n > 1 and set(table[0]) == {"lat", "lon", "value"} and table[0]["value"] == 0.3
    anywhere = ask(query + '(storm.precip.sum("time").max(["lat", "lon"]) < 0.5).mean("member")')
    assert by_model(anywhere["result"]) == {"synthetic": pytest.approx(np.mean([0.2 * m * n < 0.5
                                                                                 for m in range(10)]))}


def test_two_points_combine():
    answer = ask('x = forecast().interp(lat=50, lon=7).t2m - forecast().interp(lat=47, lon=11).t2m\n'
                 'x.sel(time=slice("2026-10-08T10:00", "2026-10-08T12:00")).mean(["member", "time"])')
    assert by_model(answer["result"]) == {"synthetic": 0.0}


def test_route_bins_by_distance():
    line = encode_polyline([LatLon(50.0, 7.0), LatLon(50.27, 7.0)])  # ~30 km north
    answer = ask(f'r = forecast().interp(route(polyline="{line}", start="2026-10-08T10:00", speed_kmh=20))\n'
                 '{"wet": r.precip.sum("time").median("member"), '
                 '"by_km": r.t2m.groupby_bins("distance_km", bins=[0, 10, 20, 30]).max().median("member")}',
                 Synthetic(point_scale=0))
    result = answer["result"]
    length = answer["locations"][0]["route"]["length_km"]
    assert length == pytest.approx(30.0, abs=0.5)
    # The median member rains 0.45 mm/h for the whole ride at 20 km/h: the sum over time is weighted by duration.
    assert by_model(result["wet"]) == {"synthetic": pytest.approx(0.45 * length / 20, rel=0.01)}
    assert [r["distance_km_bins"] for r in rows(result["by_km"])] == ["(0, 10]", "(10, 20]", "(20, 30]"]
    with pytest.raises(ExprError, match="only available on routes"):
        ask(WX + 'wx.headwind.max("time")')
    with pytest.raises(ExprError, match="isn't available on routes"):
        ask(f'forecast().interp(route(polyline="{line}", start="2026-10-08T10:00", speed_kmh=20)).t2m.rolling(time=2).max()')


def test_missing_variables_and_three_valued_logic():
    gusty = Synthetic("gusty", provides=("precip", "t2m", "gust"))
    answer = ask(WX + 'g = wx.gust.max("time") > 40\nrain = wx.precip.sum("time")\n'
                      '{"p": g.mean("member"), "p_or": (g | (rain >= 0)).mean("member"), '
                      '"p_and": (g & (rain < 0)).mean("member")}', Synthetic("calm"), gusty)
    # Unknown or true = true; unknown and false = false, as in SQL.
    assert rows(answer["result"]) == [{"model": "calm", "p": None, "p_or": 1.0, "p_and": 0.0},
                                      {"model": "gusty", "p": 0.5, "p_or": 1.0, "p_and": 0.0}]
    assert answer["models"][0]["missing_variables"] == ["gust"]


def test_partial_coverage_is_missing_with_a_warning():
    answer = ask('wx = forecast().interp(lat=50, lon=7).sel(time=slice("2026-10-09T22:00", "2026-10-10T02:00"))\n'
                 'wx.t2m.count("time").max("member")')
    assert by_model(answer["result"]) == {"synthetic": 2.0}  # the run reaches +72 h = 10 October 00:00
    assert answer["warnings"] == ["synthetic covers only 2026-10-09T22:00 to 2026-10-10T00:00 of the selected "
                                  "time at the point (50.0, 7.0)"]


@pytest.mark.parametrize("query, warning", [
    (WX + 'wx.t2m.sum("time").mean("member")', "summing a state variable"),
    (WX + 'wx.t2m.mean(["model", "member"])', "pooling members across models"),
    (WX + 'wx.t2m.mean("model").max("member")', "reducing model while members remain"),
    (WX + '(wx.t2m + wx.precip).max(["member", "time"])', "combining values in °C and mm/h"),
])
def test_likely_mistakes_are_warned_about(query, warning):
    assert warning in " ".join(ask(query)["warnings"])


def test_meaningful_time_integrals_are_not_warned_about():
    answer = ask(WX + '{"hdd": (18 - wx.t2m).clip(min=0).sum("time").median("member"), '
                      '"frost": (wx.t2m < 0).sum("time").median("member"), '
                      '"rain": wx.precip.sum("time").median("member"), '
                      '"p": (wx.precip.sum("time") > 1).mean("member").min("model")}')
    assert answer["warnings"] == []
    assert answer["units"] == {"hdd": "°C·h", "frost": "h", "rain": "mm", "p": ""}


class Windy(Synthetic):
    """Member m blows from 340 + 15 m degrees at 10 km/h: 340, 355, 10, 25 straddle north."""

    def __init__(self, members=4):
        super().__init__("windy", members=members, provides=("wind", "wind_dir"))

    def prepare(self, query, variables):
        prepared = super().prepare(query, variables)
        direction = np.radians(340 + 15 * np.arange(self.members))[:, None] * np.ones((1, len(query.lat)))
        prepared.base_per_step["wind_u"] = {i: -10 / 3.6 * np.sin(direction) for i in range(len(STEPS))}
        prepared.base_per_step["wind_v"] = {i: -10 / 3.6 * np.cos(direction) for i in range(len(STEPS))}
        return prepared


@pytest.mark.parametrize("reduction", [
    'd.mean("member")', 'd.median("member")', 'd.quantile([0.1, 0.9], "member")', 'd.max("member")',
    'd.std("member")', 'd.sel(member=[0, 1]).mean("time")', 'd.isel(member=0).resample(time="1D").mean()',
    'd.rolling(time=2).mean()', 'd.where(d > 0).mean("member")', 'd.round().mean("member")',
    'd.median("member").mean("time")',
])
def test_reducing_directions_is_warned_about(reduction):
    answer = ask(WX + 'd = wx.wind_dir\n' + reduction, Windy())
    assert any("treats directions as numbers on a line" in w for w in answer["warnings"])


def test_directions_reduce_as_in_xarray_and_circularly_on_request():
    # Linear, as in xarray: 340, 355, 10 and 25 average to 182.5 (south), with a warning.
    linear = ask(WX + 'wx.wind_dir.isel(time=0).mean("member")', Windy())
    assert by_model(linear["result"]) == {"windy": pytest.approx(182.5)}
    # The vector mean the warning suggests: north, no warnings.
    answer = ask(WX + 'r = np.deg2rad(wx.wind_dir.isel(time=0))\n'
                      'm = np.rad2deg(np.arctan2(np.sin(r).mean("member"), np.cos(r).mean("member"))) % 360\n'
                      'dev = (wx.wind_dir.isel(time=0) - m + 180) % 360 - 180\n'
                      '{"mean": m, "spread": dev.quantile([0, 1], "member"), '
                      '"north": ((wx.wind_dir >= 315) | (wx.wind_dir < 45)).mean(["member", "time"]), '
                      '"members": wx.wind_dir.count("member").max("time")}', Windy())
    assert answer["warnings"] == []
    assert answer["units"] == {"mean": "°", "spread": "°", "north": "", "members": ""}
    row, = rows(answer["result"])
    assert row["mean"] == pytest.approx(2.5)
    assert [row["spread_p0"], row["spread_p100"]] == pytest.approx([-22.5, 22.5])
    assert row["north"] == 1.0


def test_trigonometry_takes_radians():
    answer = ask(WX + 'np.sin(wx.wind_dir).mean(["member", "time"])', Windy())
    assert any("np.sin takes radians" in w for w in answer["warnings"])
    assert ask(WX + 'np.arctan2(wx.wind, wx.wind).max(["member", "time"])', Windy())["units"] == "rad"


def test_labels_of_extremes():
    answer = ask('forecast().interp(lat=50, lon=7).sel(time=slice("2026-10-08T00:00", "2026-10-09T00:00")).t2m.median("member")'
                 '.idxmax("time")')
    assert by_model(answer["result"]) == {"synthetic": "2026-10-08T23:00"}
    assert answer["units"] is None


def test_groupby_hour():
    answer = ask('forecast().interp(lat=50, lon=7).sel(time=slice("2026-10-08", "2026-10-10")).t2m.groupby("time.hour").mean()'
                 '.median("member").sel(model="synthetic").sel(hour=[0, 12])')
    # Hour 0 is +24 h and +48 h; hour 12 is +36 h and +60 h.
    assert rows(answer["result"]) == [{"hour": 0, "value": pytest.approx(12.25 + 3.6)},
                                      {"hour": 12, "value": pytest.approx(12.25 + 4.8)}]


@pytest.mark.parametrize("query, fragment", [
    ('forecast().interp(lat=50, lon=7).t2m.max("time")', "select a time range"),
    (WX + 'wx.t2m.max()', "nothing is reduced implicitly"),
    (WX + '(wx.t2m > 1) and (wx.precip < 1)', "`and`/`or` don't work"),
    (WX + 'not (wx.t2m > 1)', "use ~"),
    (WX + 'wx.t2m > 1 & wx.precip < 1', "chained comparisons"),
    (WX + '(wx.t2m > 1) & wx.precip', "put each comparison in parentheses"),
    (WX + 'wx.t2m if wx.t2m > 1 else 0', "x.where"),
    (WX + 'wx.t2m.__class__', "isn't supported"),
    (WX + 'wx.t2m.to_netcdf("/tmp/x")', "isn't supported"),
    (WX + 'wx.t2m.pipe(print)', "isn't supported"),
    ('import os\nforecast()', "only `name = expression`"),
    (WX + '[x for x in wx.t2m]', "ListComp isn't supported"),
    (WX + '(lambda: 1)()', "only functions and methods"),
    (WX + 'wx.t2m[0]', "indexing isn't supported"),
    (WX + 'max(wx.t2m)', "Python's max() doesn't work on arrays"),
    (WX + 'wx.rain.sum("time")', "unknown variable 'rain'"),
    ('t2m.max("time")', "variables belong to the dataset"),
    ('forecast = 1\nforecast', "built-in name"),
    ('forecast().sel(time=slice("2026-10-08T10:00", "2026-10-08T12:00")).t2m.max("time").mean("member")',
     "without a location"),
    (WX + 'wx.t2m.max("time").interp(lat=51, lon=7)', "already has a location"),
    ('forecast().sel(lat=slice(49, 50)).t2m', "an area is .sel(lat=slice"),
    ('forecast().interp(lat=("place", [50, 51]), lon=("place", [7, 7])).t2m', "must be called"),
    ('places(A=(50, 7))', "use them as forecast().interp"),
    (WX + 'wx', "this is a dataset"),
    (WX + 'wx.t2m.rolling(time=3)', "needs an aggregation"),
    (WX + 'wx.t2m.max("point")', "no point dimension here"),
    (WX + 'wx.t2m.max("level")', "unknown dimension"),
    (WX + 'wx.t2m.resample(time="5h").max()', "resample frequencies"),
    (WX + 'wx.sel(time=slice("2026-10-08T12:00", "2026-10-08T10:00")).t2m.max("time")', "must be after its start"),
    ('forecast().interp(lat=50, lon=7).sel(time=slice("2026-12-08", "2026-12-09")).t2m.max("time")', "days ahead"),
    ('forecast().interp(route(gpx, start="2026-10-08T10:00", speed_kmh=20)).t2m.max("time")', "wasn't given"),
    ('max(' * 45 + '1' + ')' * 45, "nested too deeply"),
    ('1 +', "syntax error"),
    ('1 + 1', "uses no weather variable"),
    (WX + 'wx.t2m.max("time").mean("member") + forecast().sel(lat=slice(49.8, 50.2), lon=slice(6.7, 7.3)).lat.mean("lat")',
     "uses only coordinates of this location"),
])
def test_invalid_queries_are_explained(query, fragment):
    with pytest.raises(ValueError, match=fragment.replace("(", r"\(").replace(")", r"\)").replace("[", r"\[")):
        ask(query)


def test_coordinates_do_not_widen_the_models_fetched():
    a, b = Synthetic("a"), Synthetic("b")
    answer = ask(WX + '((wx.sel(model="a").t2m > 10) & (wx.time.dt.hour >= 8)).mean(["member", "time"])', a, b)
    assert answer["result"] == 1.0
    assert a.requested == ["t2m"] and b.requested is None


def test_resample_after_reordering_time():
    answer = ask('x = forecast().interp(lat=50, lon=7).sel(time=slice("2026-10-08", "2026-10-10")).t2m.median("member").sel(model="synthetic")\n'
                 'x.sortby(x, ascending=False).resample(time="1D").max()')
    assert [r["time"] for r in rows(answer["result"])] == ["2026-10-08", "2026-10-09"]


def test_place_names_are_kept_as_given():
    answer = ask('wx = forecast().interp(places({"Köln; Altstadt": (50.94, 6.96), "Bonn@Rhein": (50.73, 7.10)})).sel(time=slice('
                 '"2026-10-08T10:00", "2026-10-08T12:00"))\nwx.precip.sum("time").max("member").sel(model="synthetic")')
    assert [r["place"] for r in rows(answer["result"])] == ["Köln; Altstadt", "Bonn@Rhein"]


def test_runs_covering_every_hour_come_first():
    from weather_core.sources.base import runs_by_coverage
    from weather_core.timeaxis import sample_hours

    hours = [datetime(2026, 10, 7, 22, tzinfo=timezone.utc), datetime(2026, 10, 8, 1, tzinfo=timezone.utc)]
    newer, older = datetime(2026, 10, 8, 0, tzinfo=timezone.utc), datetime(2026, 10, 7, 12, tzinfo=timezone.utc)

    def sampling(run):
        try:
            return sample_hours(run, STEPS, hours)
        except ValueError:
            return None
    # The newer run starts after 22:00, so it covers only part; the older one covers both hours.
    assert [run for run, _ in runs_by_coverage([newer, older], sampling)] == [older, newer]


FC = 'fc = forecast().sel(time=slice("2026-10-08T10:00", "2026-10-08T12:00"))\n'


def test_location_can_be_chosen_after_computing():
    # Each .interp compiles the expression again for its location; a binding is reused for both.
    answer = ask(FC + 'x = fc.t2m.max("time").mean("member")\nx.interp(lat=50, lon=7) - x.interp(lat=47, lon=11)')
    assert by_model(answer["result"]) == {"synthetic": 0.0}
    assert answer["warnings"] == []
    # The route's time samples replace the hourly axis, so time operations are checked for the route.
    line = encode_polyline([LatLon(50.0, 7.0), LatLon(50.27, 7.0)])
    with pytest.raises(ExprError, match="isn't available on routes"):
        ask(f'forecast().t2m.rolling(time=2).max().interp(route(polyline="{line}", start="2026-10-08T10:00", '
            f'speed_kmh=20))')


def test_several_points_and_areas():
    points = ask(FC + 'fc.interp(lat=("point", [50, 51]), lon=("point", [7, 7.5])).precip.sum("time")'
                      '.max("member")')
    assert [(r["lat"], r["lon"]) for r in rows(points["result"])] == [(50.0, 7.0), (51.0, 7.5)]
    area = 'storm = fc.sel(lat=slice(49.8, 50.2), lon=slice(6.7, 7.3))\n'
    full = ask(FC + area + 'storm.precip.count(["lat", "lon"]).max(["model", "member", "time"])')["result"]
    north = ask(FC + area + 'storm.precip.sel(lat=slice(50.0, 50.2)).count(["lat", "lon"])'
                            '.max(["model", "member", "time"])')["result"]
    circle = ask(FC + area + 'storm.precip.where(distance_from(storm, 50, 7) <= 15).count(["lat", "lon"])'
                             '.max(["model", "member", "time"])')["result"]
    assert full > north > 0 and full > circle > 0


def test_area_with_time_in_one_selection():
    answer = ask('s = forecast().sel(lat=slice(49.8, 50.2), lon=slice(6.7, 7.3), time=slice("2026-10-08T10:00", '
                 '"2026-10-08T12:00"))\ns.precip.count("time").max(["model", "member", "lat", "lon"])')
    assert answer["result"] == 2.0


def test_selections_the_location_lacks_are_errors():
    with pytest.raises(ExprError, match="has no lat dimension"):
        ask(FC + 'fc.isel(lat=0).t2m.max("time").mean("member").interp(lat=50, lon=7)')
    with pytest.raises(ExprError, match="the point \\(50.0, 7.0\\) has no lat dimension"):
        ask(FC + 'fc.interp(lat=50, lon=7).sel(lat=slice(49, 51), lon=slice(6, 8)).t2m.max("time")')


def test_repeated_points_stay_apart():
    answer = ask(FC + 'fc.interp(lat=("point", [50, 50, 51]), lon=("point", [7, 7, 7])).precip.sum("time")'
                      '.max("member")')
    assert [(r["lat"], r["value"]) for r in rows(answer["result"])] == [
        (50.0, pytest.approx(1.8)), (50.0, pytest.approx(3.6)), (51.0, pytest.approx(5.4))]


def test_distance_from_a_located_array():
    answer = ask(FC + 'x = fc.t2m.max("time").mean("member").interp(lat=50, lon=7)\n'
                      'x.where(distance_from(x, 50.1, 7) < 20)')
    assert by_model(answer["result"])["synthetic"] is not None
    far = ask(FC + 'x = fc.t2m.max("time").mean("member").interp(lat=50, lon=7)\n'
                   'x.where(distance_from(x, 51, 7) < 20)')
    assert by_model(far["result"]) == {"synthetic": None}


def test_review_edge_cases():
    # A huge range() for bins is refused without being materialised.
    line = encode_polyline([LatLon(50.0, 7.0), LatLon(50.27, 7.0)])
    with pytest.raises(ExprError, match="bins"):
        ask(f'forecast().interp(route(polyline="{line}", start="2026-10-08T10:00", speed_kmh=20)).t2m'
            '.groupby_bins("distance_km", bins=range(0, 1000000000000)).max()')
    # where()'s value branches are unit-checked like np.where's.
    mixed = ask(WX + 'wx.t2m.where(wx.precip > 0.1, wx.precip).max(["member", "time"])')
    assert any("°C and mm/h" in w for w in mixed["warnings"])
    # idxmax over several dimensions is refused instead of reducing only the first.
    with pytest.raises(ExprError, match="exactly one dimension"):
        ask(WX + 'wx.t2m.idxmax(["time", "member"])')
    # Empty selections give missing values, not internal errors.
    empty = ask(WX + '{"q": wx.t2m.isel(member=[]).quantile(0.5, "member").max("time"), '
                     '"i": wx.t2m.isel(time=slice(0, 0)).max("member").idxmax("time")}')
    assert rows(empty["result"]) == [{"model": "synthetic", "q": None, "i": None}]


def test_attribution_covers_only_the_hours_a_run_has():
    class Recording(Synthetic):
        def prepare(self, query, variables):
            p = super().prepare(query, variables)
            p.attribution = lambda times: {"model": self.name, "until": times[-1].isoformat()}
            return p
    answer = ask('wx = forecast().interp(lat=50, lon=7).sel(time=slice("2026-10-09T22:00", "2026-10-10T02:00"))\n'
                 'wx.t2m.count("time").max("member")', Recording())
    assert answer["attribution"] == [{"model": "synthetic", "until": "2026-10-09T23:00:00+00:00"}]


def test_huge_numbers_do_not_hang():
    assert by_model(ask(WX + '(wx.t2m.min("time") ** 10 ** 10 ** 10 > 1).mean("member")')["result"]) == \
        {"synthetic": 1.0}
    with pytest.raises(ExprError, match="out of range"):
        ask(WX + 'wx.t2m.max("time") > 1' + "0" * 400)


def test_large_results_are_refused():
    with pytest.raises(ExprError, match="rows"):
        ask('forecast().interp(lat=50, lon=7).sel(time=slice("2026-10-08", "2026-10-11")).t2m')  # 10 members × 72 hours


def test_memory_estimate_rejects_before_downloading():
    source = Synthetic()
    with pytest.raises(ExprError, match="would need about"):
        ask(WX + '(wx.precip.sum("time") < 0.5).mean("member")', source, eval_bytes=100)
    assert source.requested is None


def test_evaluation_stops_at_the_deadline():
    with pytest.raises(ExprError, match="took too long"):
        ask(WX + '(wx.precip.sum("time") < 0.5).mean("member")', eval_timeout_s=-1)


def test_quantiles_match_numpy_on_padded_members():
    rng = np.random.default_rng(1)
    x = rng.random((3, 7, 50))
    x[:, :, 20:] = np.nan
    x[1, 2] = np.nan
    with pytest.warns(RuntimeWarning):  # numpy complains about the all-missing slice; we return NaN quietly
        expected = np.moveaxis(np.nanquantile(x, [0.1, 0.5, 0.95], axis=-1), 0, -1)
    np.testing.assert_allclose(quantiles_last(x, [0.1, 0.5, 0.95]), expected)


def test_help_mentions_the_essentials():
    text = describe_language()
    for name in ("forecast()", ".interp(lat=", "places(", "route(", ".sel(lat=slice(", ".sel(time=slice(", ".quantile(", ".rolling(",
                 ".resample(", "groupby_bins", "top(", "end excluded", ".mean(\"member\")"):
        assert name in text
