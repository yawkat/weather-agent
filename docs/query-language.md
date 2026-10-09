# Forecast query language

Status: implemented as the `forecast` tool (`weather_core/expr/`); it replaced the DuckDB SQL tool. The
evaluation is our own, on numpy. xarray itself was measured under GraalPy and not used: it adds about 0.3 s to small
queries, its comparisons treat missing as false, and its quantiles take seconds on padded members, so both would
need overriding anyway.

## Principles

- **Python syntax with Python semantics, by following xarray.** Values are labelled multidimensional arrays.
  Reductions name the dimension they remove; nothing reduces implicitly. If a query means something in xarray, it
  means the same here. We implement a subset of xarray, not something that merely looks like it.
- **The query states what it is about.** Location, time range and models are part of the expression (like SQL's
  `FROM … WHERE`), not tool parameters or settings lines. They are still read before anything is downloaded.
- **Few restrictions, visible pitfalls.** Agents may combine members and models however they like. Answers carry
  units, and likely mistakes produce warnings, not errors.
- **The remaining dimensions are the shape of the answer.** A scalar is a yes/no or a number; an array over
  `time` is a graph; an array over `point` is a map. Parameterized evaluation for visualization needs no extra
  mechanism.

## Data model

A query works on one **dataset**, `forecast()`. Its variables are arrays (`fc.precip`, `fc.t2m`, …) over these
dimensions:

| Dimension | Labels | Notes |
|---|---|---|
| `model` | `"ecmwf-ens"`, `"icon-d2-eps"`, … | every model that covers the location and time |
| `member` | 0, 1, … | padded with missing values: IFS has 50 members, ICON-D2 has 20 |
| `time` | local hours | a regular hourly axis; for routes, the rider's samples (see below) |
| `lat`, `lon` | degrees | the whole world, until the query picks a location |

**Locations are selected like any other dimension.** Models have no common grid (ECMWF 0.25°, ICON triangles), so
`lat`/`lon` are continuous: the query either interpolates to points or selects an area, on a grid we lay out.
This is xarray's `interp` and `sel`:

```python
fc = forecast()
fc.interp(lat=50.94, lon=6.96)                                         # a point: no spatial dimension
fc.interp(lat=("point", [50.94, 50.73]), lon=("point", [6.96, 7.10]))   # several points: a point dimension
fc.interp(places(Köln=(50.94, 6.96), Bonn=(50.73, 7.10)))              # the same, with names
fc.sel(lat=slice(50.5, 51.2), lon=slice(6.5, 7.5))                     # an area: lat and lon dimensions
fc.interp(route(polyline="…", start="2026-10-10T09:00", speed_kmh=22))  # a route: see below
fc.interp(route(gpx, start="2026-10-10T09:00", use_gpx_times=True))    # gpx: the tool's attachment argument
```

- **Areas** are a regular grid of about 10 km spacing (coarser for big areas, at most 500 points), with real
  `lat` and `lon` dimensions, so an answer over both is a map. A located grid can be narrowed with
  `.sel(lat=slice(…))` (inclusive, as xarray's label slices). A circle is a mask:
  `x.where(distance_from(x, 50.94, 6.96) <= 30)`.
- **A route** is xarray's vectorised interpolation along its samples: each time sample is where the rider is at
  that time. `route(…)` is shorthand for those sample coordinates; `places(…)` for named points.
- **Like every selection, a location can come anywhere in the expression**, also after computing:
  `x = fc.t2m.max("time"); x.interp(lat=…, lon=…) - x.interp(lat=…, lon=…)`. The compiler evaluates `x` once per
  location, checking it again for that location (e.g. `rolling` isn't available once a route makes time
  irregular).
- Every variable in the answer needs a location; arguments are literals, so locations are known before
  downloading.

Coordinates: `x.time.dt.hour`, `x.time.dt.dayofweek`, `x.lat`, `x.lon` (per point, grid axis or route sample),
and `x.distance_km` for routes.

**Hourly samples.** Each sample stands for the hour starting at its label, as in pandas' default resampling:
- rates (`precip`, `snow`, `radiation`) are the mean over that hour, in mm/h or W/m²
- instants (`t2m`, `wind`, …) are the value at the label
- interval maxima (`gust`) are the maximum within the hour

So `.sum("time")` of a rate is an amount, `(cond).sum("time")` is a number of hours, and `rolling(time=3)` means
3 hours. Models with 3- or 6-hourly steps are interpolated to hours, and their step values stay on the axis.

**Routes.** Samples are where the rider is at each time, at irregular intervals. The time axis carries `dt`
(hours per sample), and `.sum("time")` on a route is weighted by it, so `.sum("time")` is the time integral in
hours on every dataset. Count- and position-based operations (`rolling`, `isel`, `diff`) are not available on
routes.

**Missing values.** A variable a model doesn't provide, a member beyond a model's member count, a time a model
doesn't reach, a point outside its domain: all are missing (NaN). Reductions skip them, as xarray does
(`skipna=True`). Comparisons with a missing value give missing, not false. So `.mean("member")` of a condition is
a model's probability even though members are padded.

## Selection is part of the query

```python
day = wx.sel(time=slice("2026-10-10T08:00", "2026-10-10T20:00"))
ifs = wx.sel(model="ecmwf-ens")                        # a single label drops the dimension
two = wx.sel(model=["ecmwf-ens", "icon-d2-eps"])
cologne = wx.sel(point="Köln")
```

**Time slices exclude their end**, unlike xarray's label slices: `slice("10:00", "16:00")` is the six hours
10:00–16:00, i.e. the samples labelled 10:00 … 15:00. Rates then add up to exactly that range, and consecutive
slices don't overlap. Instants at 16:00 are not included; end at 17:00 to include them. Dates mean whole local
days: `slice("2026-10-10", "2026-10-12")` is two days.

`.sel` works on the dataset and on arrays, anywhere in the expression. Its arguments must be literals, so the compiler
can work out, for every variable used, which models, times and location it needs (`rolling` widens the time
range by its length). A variable whose time range is never limited is an error ("select a time range with
.sel(time=slice(…))"), except on routes, whose times come from the route. (The SQL tool needed
`start`/`end`/`sources`/`window_hours` arguments and guessed variables with a regex.)

Arrays at different single points combine like any arrays:

```python
t = forecast().sel(time=…).t2m
(t.interp(lat=47.42, lon=10.98) - t.interp(lat=47.21, lon=11.0)).mean("member")   # valley vs mountain
```

## Operations

xarray subset (all reductions take explicit dimensions, one name or a list):

- arithmetic `+ - * / // % **`, `abs()`, comparisons, `&`, `|`, `~`
- `.sum`, `.mean`, `.median`, `.min`, `.max`, `.std`, `.any`, `.all`, `.count`
- `.quantile(q, dim)` (adds a `quantile` dimension for a list of q)
- `.where(cond, other)`, `.clip(min=…, max=…)`, `np.maximum`, `np.minimum`, `np.hypot`, `np.sqrt`
- `np.sin`, `np.cos`, `np.arctan2` (radians), `np.deg2rad`, `np.rad2deg`: for averaging directions (below)
- `.sel`, `.isel`
- `.rolling(time=n).sum()/.mean()/.max()/.min()`: labelled at the window's end, as in xarray
- `.resample(time="3h" | "1D").sum()/.mean()/.max()/.min()`: local time
- `.groupby("time.hour")` / `.groupby_bins("distance_km", bins=[…])` with the same reductions
- `.idxmax(dim)`, `.idxmin(dim)`, `.sortby(x, ascending=False)`
- helpers that aren't xarray: `top(x, n, dim)` / `bottom(x, n, dim)` keep the n largest/smallest entries along
  `dim` (x has exactly that one dimension left, or is a dict of such arrays, ranked by `by="key"`)
- a result may be a dict of arrays: `{"p_dry": …, "rain_p90": …}`

Not supported: `and`/`or`/`not` on arrays (as in xarray they are errors; the message suggests `&`, `|`, `~` and
parentheses), loops, comprehensions, lambdas, attributes other than the listed ones, imports.

Python's precedence makes `a < 1 & b < 2` parse as `a < (1 & b) < 2`. The type checker rejects `1 & b` with "put
comparisons in parentheses: (a < 1) & (b < 2)".

## Units and warnings

Every array has a unit, derived through the operations: `precip.sum("time")` is mm, `(t2m < 0).sum("time")` is h,
a condition's `.mean(…)` is a fraction. The result reports units per value.

Warnings come back with the result. They never block it:

| Pattern | Warning |
|---|---|
| Summing a state variable over time without a threshold (`t2m.sum("time")`, `cloud.sum("time")`) | "°C·h from 0 °C is rarely meaningful; did you mean .mean("time"), a duration ((t2m > 25).sum("time")) or degree-hours ((t2m - 18).clip(min=0).sum("time"))?" |
| Reducing `member` and `model` together, or `member` after `model` | "members are pooled across models: models with more members weigh more (IFS 50, ICON-D2 20)" |
| Adding values with different units (`t2m + precip`) | "adding °C and mm/h" |
| Reducing a direction (`wind_dir.mean("member")`, `.median`, `.quantile`, `.min`/`.max`, `.std`, `.sum`, also in `rolling`/`resample`/`groupby`) | "treats directions as numbers on a line, where 350° and 10° are 340° apart …", with the vector mean below |
| `np.sin`/`np.cos` of degrees | "np.sin takes radians; convert degrees with np.deg2rad(…)" |
| A probability from a model that reaches only part of the selected time | "icon-d2-eps covers only 10:00–12:00 of the selection" |

Hours of a condition, degree-hours, rates turned into amounts and durations are the meaningful time integrals. For
state variables the warning points to the mean or a threshold.

### Directions

`wind_dir` is a compass direction (0 = north, 90 = east). Reductions treat it as a number, as xarray does: the mean
of 350° and 10° is 180°, the opposite of both. This is only right while the values stay on one side of north, so
reducing a direction warns (`count` doesn't). The query stays xarray: we don't switch to a circular mean on our
own. A direction stays one through selections, `where`, `round` and `median`/`min`/`max`/`quantile`. Arithmetic
returns a plain number in °, so the expressions below don't warn.

```python
d = wx.wind_dir
r = np.deg2rad(d)
m = np.rad2deg(np.arctan2(np.sin(r).mean("member"), np.cos(r).mean("member"))) % 360  # mean of unit vectors
spread = ((d - m + 180) % 360 - 180).quantile([0.1, 0.9], "member")   # deviation from m, in -180…180
northerly = ((d >= 315) | (d < 45)).mean("member")                   # probability of a sector
```

Members whose directions differ widely give a mean near nowhere (unit vectors cancel out); look at the spread or a
sector probability then.

## Results

- A scalar: a number or boolean.
- An array: a table with one column per remaining dimension (labels) and one value column, at most 500 rows.
- A dict of arrays: one table if they share dimensions, else one entry per key.
- Every answer carries the model table (run, lead hours, members), units, warnings and attribution.

## Charts

The `show_forecast` tool takes the same query and shows its answer to the user as an interactive chart (an MCP
Apps view, `config/mcp-apps/forecast.html`), next to the answer the agent gets. The answer is sent as dense arrays
(`weather_core/chart.py`, at most 400,000 values) and the dimensions left in each value pick the drawing:

| Dimensions left | Drawing |
|---|---|
| `time`, `hour` or `distance_km_bins` | a graph along that axis; on a route, also the route on a map |
| `lat` × `lon` | a map of grid cells; with `time`, a time slider |
| `point` without an x axis | markers on a map |
| none of these | dots and ranges per model, or a single number |

Within those, `model` is colour (or one map per model), `member` the 10–90 % and 25–75 % bands and median the
view draws for orientation (one thin line per member when one model is shown), `quantile` bands, and `point` with
an x axis one graph per place. Maps can't show every member: the query must reduce `member` first. A grid's map is
the selected area, filled to its edges (select a larger area for more around it); maps of places and routes get
room around them. Under maps lies a basemap: land, lakes, coastlines, borders,
rivers and main roads from Natural Earth 1:10m (public domain), and town names from GeoNames (CC BY 4.0, towns of
15,000 or more), with detail to the map's scale like a web map's zoom levels (`scripts/make_basemap.py`,
`weather_core.chart.basemap`). The agent writes the query; the view only picks colours.

## Examples

Chance of a dry, mild ride, per model:
```python
ride = forecast().interp(lat=50.94, lon=6.96).sel(time=slice("2026-10-10T10:00", "2026-10-10T16:00"))
ok = (ride.precip.sum("time") < 0.5) & (ride.t2m.max("time") <= 26) & (ride.gust.max("time") < 40)
ok.mean("member")
```

What goes wrong, and how badly:
```python
rain = ride.precip.sum("time")
tmax = ride.t2m.max("time")
{"wet": (rain >= 0.5).mean("member"),
 "too_warm": ((rain < 0.5) & (tmax > 26)).mean("member"),
 "rain_p90": rain.quantile(0.9, "member")}
```

Hour by hour:
```python
{"t2m": ride.t2m.quantile([0.1, 0.5, 0.9], "member"),
 "p_rain": (ride.precip > 0.1).mean("member"),
 "gust_max": ride.gust.max("member")}
```

Daily overview for a week:
```python
week = forecast().interp(lat=50.94, lon=6.96).sel(time=slice("2026-10-10", "2026-10-17"))
{"tmax": week.t2m.resample(time="1D").max().median("member"),
 "p_rain": (week.precip.resample(time="1D").sum() > 1).mean("member")}
```

Best 3-hour slots, cautious across models:
```python
day = forecast().interp(lat=50.94, lon=6.96).sel(time=slice("2026-10-10T08:00", "2026-10-10T20:00"))
ok = (day.precip.rolling(time=3).sum() < 0.5) & (day.gust.rolling(time=3).max() < 40)
p = ok.mean("member").min("model")        # the worst model's probability, per window end
top(p, 5, "time")
```

How many models agree:
```python
(ok.mean("member") > 0.7).sum("model")
```

Compare places:
```python
dry = (forecast().sel(time=slice("2026-10-10T10:00", "2026-10-10T16:00")).precip.sum("time") < 0.5).mean("member")
dry.interp(places(Köln=(50.94, 6.96), Bonn=(50.73, 7.10)))      # per model and place
```

Area, "anywhere", and a map:
```python
storm = forecast().sel(lat=slice(50.5, 51.4), lon=slice(6.2, 7.7), time=slice("2026-10-10", "2026-10-11"))
anywhere = (storm.gust.max(["time", "lat", "lon"]) > 70).mean("member")
heavy_rain_map = (storm.precip.sum("time") > 10).mean("member").sel(model="ecmwf-ens")    # over lat × lon
```

Along a route:
```python
r = forecast().interp(route(polyline="…", start="2026-10-10T09:00", speed_kmh=22))
{"wet_mm": r.precip.sum("time").quantile([0.5, 0.9], "member"),
 "rain_by_10km": (r.precip > 0.2).groupby_bins("distance_km", bins=list(range(0, 110, 10))).max().mean("member")}
```

Degree-days and sunshine:
```python
{"heating_degree_days": ((18 - week.t2m).clip(min=0).sum("time") / 24).median("member"),
 "sunny_hours": (week.radiation > 120).sum("time").median("member")}
```

Pooling, if the agent really wants it (allowed, with a warning):
```python
ok.mean(["model", "member"])
```

## Checking before downloading

1. Parse with `ast`; accept only the syntax and names above.
2. Type-check: dimensions, units, kinds; literal arguments for `sel`, `rolling`, `resample`, bins, locations.
3. Demand: for every variable, the models, time range and location it needs.
4. Estimate the download per source (the existing budget estimate) and the evaluation memory per operation from
   the dimension sizes. Reject before downloading if either is over its limit.
5. Download, build the arrays, evaluate with metering (memory, deadline), encode, attach units and warnings.

## Open questions

1. **GPX.** GPX documents are too big for the query; they stay a tool argument that the query refers to by name
   (`route(gpx, …)`).
2. **Time strings.** Local time (Europe/Berlin unless an offset is given) as today; relative forms like
   `"tomorrow 10:00"` would be convenient but are a second date language.
3. **How strict.** Should anything be an error rather than a warning (e.g. unit mismatches)?
4. **Output size.** 500 rows is fine for an agent; charts (`show_forecast`) take up to 400,000 values, and the
   agent then gets a per-value summary instead of the table.
