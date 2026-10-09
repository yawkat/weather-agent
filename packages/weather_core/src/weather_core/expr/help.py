"""Reference text for the expression language, returned by the help tool."""

from ..variables import CATALOG
from .language import RESAMPLE


def describe_language() -> str:
    variables = "\n".join(f"  {v.name} [{v.unit}]: {v.description}" + (" (routes only)" if v.route_only else "")
                          for v in CATALOG.values())
    return f"""Forecast queries are a subset of xarray (Python syntax, xarray meaning) over ensemble forecasts. Every
ensemble member is one possible weather outcome; a model's members together show how likely outcomes are.

A query is optional `name = expression` lines and a final expression, the answer. forecast() is the dataset;
its variables (fc.t2m, fc.precip, …) are arrays over the dimensions
  model     every model ("ecmwf-ens", "ecmwf-aifs-ens", "icon-eu-eps", "icon-d2-eps", "icon-d2-ruc-eps")
  member    ensemble members; padded with missing values for models with fewer members
  time      local hours (a sample stands for the hour starting at its label); routes: the rider's positions
  lat, lon  anywhere; every query picks a location, on the dataset or on any expression:
    .interp(lat=50.94, lon=6.96)                                   one point (no spatial dimension)
    .interp(lat=("point", [50.94, 50.73]), lon=("point", [6.96, 7.10]))   several points (point dimension)
    .interp(places(Köln=(50.94, 6.96), Bonn=(50.73, 7.10)))        named points; or places("Köln@50.94,6.96; …")
    .sel(lat=slice(50.5, 51.2), lon=slice(6.5, 7.5))               an area: a ~10 km lat/lon grid, ≤500 points
    .interp(route(polyline="…", start="2026-10-10T09:00", speed_kmh=22))   time = where the rider is when;
                                                     or route(gpx, start=…, use_gpx_times=True) with the gpx argument
  A located grid can be narrowed with .sel(lat=slice(…), lon=slice(…)) (inclusive); for a circle, use
  x.where(distance_from(x, 50.94, 6.96) <= 30) (km).
Coordinates: x.time.dt.hour, x.time.dt.dayofweek (0 = Monday), x.lat, x.lon, x.distance_km (routes).

Select with .sel on the dataset or any array; the query downloads only what it selects:
  fc.sel(time=slice("2026-10-10T10:00", "2026-10-10T16:00"))   end excluded: the hours 10:00 … 15:00
  fc.sel(time=slice("2026-10-10", "2026-10-12"))               two whole local days
  fc.sel(time="2026-10-10")   one day;   .sel(time="2026-10-10T12:00") one hour (drops the dimension)
  .sel(model="ecmwf-ens") (drops the dimension), .sel(model=[…]), .sel(point="Köln"), .isel(time=0)
Times are local (Europe/Berlin) unless they carry an offset. Every time dimension must be limited by a selection
(routes take their times from the route).

Nothing reduces implicitly: every reduction names its dimension(s), one or a list.
  .sum(dim) .mean(dim) .median(dim) .min(dim) .max(dim) .std(dim) .count(dim) .any(dim) .all(dim)
  .quantile(q, dim) with q a number or a list (adds a quantile dimension)
  Over time, sum and mean weigh samples by their duration: .sum("time") is the time integral in hours
  (precip mm/h → mm; a condition → hours for which it holds), .mean("time") the time average.
  .rolling(time=n).sum()/.mean()/.max()/.min()   n samples, labelled at the window's last hour, as in xarray
  .resample(time="3h" or "1D").sum()/.mean()/.max()/.min()   local time; {", ".join(RESAMPLE)}
  .groupby("time.hour").mean()…   .groupby_bins("distance_km", bins=[0, 10, 20]).max()… (routes)
  .idxmax(dim) / .idxmin(dim): the label of the largest/smallest value
Elementwise: + - * / // % **, comparisons, & | ~ (conditions; parenthesise comparisons: (a < 1) & (b > 2)),
  .where(cond, other), .clip(min=…, max=…), .round(n), abs(x), np.maximum/minimum/where/sqrt/hypot/abs,
  np.sin/cos/arctan2 (radians), np.deg2rad/rad2deg.
  Arrays align by dimension name and label, as in xarray.
Directions (wind_dir) reduce linearly, as in xarray: the mean of 350° and 10° is 180°. Average unit vectors:
  r = np.deg2rad(d); np.rad2deg(np.arctan2(np.sin(r).mean("member"), np.cos(r).mean("member"))) % 360
  Spread: quantiles of (d - mean + 180) % 360 - 180. Sectors: ((d >= 315) | (d < 45)).mean("member").
Ranking: top(x, n, "time") / bottom(…) keep the n largest/smallest entries along x's one remaining dimension
  (for a dict of such arrays: by="key"); x.sortby(key, ascending=False).
Answer several values at once with a dict: {{"p_dry": …, "rain_p90": …}}.

Variables:
{variables}

Missing values (a variable a model lacks, padded members, hours a model doesn't reach, points outside its domain)
are skipped by reductions; comparisons with them are missing, not false. So a condition's .mean("member") is
each model's probability.

Combining models is up to you; the server never pools them. Reduce member first, then combine models: the worst
model's probability .mean("member").min("model"), agreement (p > 0.7).sum("model"). Pooling members across models
(.mean(["model", "member"])) weighs models by their member count (IFS 50, ICON-D2 20); you get a warning.

Results: a number or boolean; a table with a column per remaining dimension (at most 500 rows); or, for a dict,
one table if its arrays share their dimensions. "units" gives each value's unit; "warnings" flags likely mistakes
(e.g. summing temperature over time).

Examples:
- Chance of a dry, mild ride, per model:
    ride = forecast().interp(lat=50.94, lon=6.96).sel(time=slice("2026-10-10T10:00", "2026-10-10T16:00"))
    ok = (ride.precip.sum("time") < 0.5) & (ride.t2m.max("time") <= 26) & (ride.gust.max("time") < 40)
    ok.mean("member")
- What goes wrong:
    rain = ride.precip.sum("time")
    {{"wet": (rain >= 0.5).mean("member"), "too_warm": ((rain < 0.5) & (ride.t2m.max("time") > 26)).mean("member"),
     "rain_p90": rain.quantile(0.9, "member")}}
- Hour by hour:  {{"t2m": ride.t2m.quantile([0.1, 0.5, 0.9], "member"), "p_rain": (ride.precip > 0.1).mean("member")}}
- Daily overview:
    week = forecast().interp(lat=50.94, lon=6.96).sel(time=slice("2026-10-10", "2026-10-17"))
    {{"tmax": week.t2m.resample(time="1D").max().median("member"),
     "p_rain": (week.precip.resample(time="1D").sum() > 1).mean("member")}}
- Best 3-hour slots, cautious across models (labels are each window's last hour):
    day = forecast().interp(lat=50.94, lon=6.96).sel(time=slice("2026-10-10T08:00", "2026-10-10T20:00"))
    ok = (day.precip.rolling(time=3).sum() < 0.5) & (day.gust.rolling(time=3).max() < 40)
    top(ok.mean("member").min("model"), 5, "time")
- How many models agree:  (ok.mean("member") > 0.7).sum("model")
- Compare places (compute first, pick the location afterwards):
    dry = (forecast().sel(time=slice("2026-10-10T10:00", "2026-10-10T16:00")).precip.sum("time") < 0.5).mean("member")
    dry.interp(places(Köln=(50.94, 6.96), Bonn=(50.73, 7.10)))
- Area, anywhere:  storm = forecast().sel(lat=slice(50.5, 51.3), lon=slice(6.3, 7.6), time=slice(…))
    (storm.gust.max(["time", "lat", "lon"]) > 70).mean("member")
- Area, as a map:  (storm.precip.sum("time") > 10).mean("member").sel(model="ecmwf-ens")
- Along a route:  r = forecast().interp(route(polyline="…", start="2026-10-10T09:00", speed_kmh=22))
    (r.precip > 0.2).groupby_bins("distance_km", bins=range(0, 110, 10)).max().mean("member")
- Degree-days, sunshine:  ((18 - week.t2m).clip(min=0).sum("time") / 24).median("member"),
    (week.radiation > 120).sum("time").median("member")
- Valley vs mountain:  t = forecast().sel(time=…).t2m
    (t.interp(lat=47.42, lon=10.98) - t.interp(lat=47.21, lon=11.0)).mean(["member", "time"])

Reading the results: raw ensembles are somewhat overconfident (treat 0%/100% as unlikely/likely). Check lead_hours in
the model table: beyond ~7 days details get unreliable, beyond ~10 days use broad tendencies only.
"""
