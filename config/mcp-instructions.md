Ensemble weather forecasts (ECMWF IFS/AIFS ENS; DWD ICON-EU-EPS, ICON-D2-EPS, ICON-D2-RUC-EPS) for Europe, queried
in a subset of xarray.

- Call weather_query_help before the first query. It explains the query language, and its last section says which
  models reach how far, what is cached and how long downloading the rest takes. Plan queries with it: ask only for
  models that cover the hours, and keep uncached time ranges no longer than needed. Choose models for quality, not
  for what is cached: the cache only decides how long a query takes. An uncached model is often worth the wait,
  e.g. ICON-D2-RUC-EPS (the newest runs, best for the next hours) or ICON-D2-EPS for showers and storms; tell the
  user when a query will take a while.
- Show the user forecasts with show_forecast, not as text alone. It draws the query's answer as an interactive chart
  and returns the same answer to you, so one call does both. Meteograms (every member over the hours) for a place,
  probability maps for an area, one line per place to compare places. Use forecast for your own working: checks,
  single numbers, finding the right hours. A plain yes/no or one number needs no chart.
- Answers are per model. Combine models only explicitly, and say which models and runs an answer rests on. Credit
  the attribution each answer carries.
- resolve_place turns place names into coordinates; check its label is the place meant.
