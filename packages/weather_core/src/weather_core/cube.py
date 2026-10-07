"""Forecast samples as SQL tables, for client queries.

Every query is answered from a small in-memory database:

- ``samples``: one row per model, member, window, time sample and point, with the weather variables
- ``per_member``: one row per model, member and window, with common aggregates (``rain``, ``tmax``, ...)
- ``models``: one row per model that contributed (run, members, resolution, note)

There is no cross-model weighting: each model's members are its own sample of possible weather, and the client
decides how to read and combine models.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, tzinfo
from typing import Protocol

import numpy as np

from .sources.base import SourceSamples
from .variables import CATALOG


class QueryError(ValueError):
    """The client's SQL failed (syntax, unknown column, timeout, memory limit)."""


@dataclass
class Table:
    name: str
    int_columns: list[str]
    float_columns: list[str]
    ints: np.ndarray  # [rows, len(int_columns)]
    floats: np.ndarray  # [rows, len(float_columns)]

    @property
    def rows(self) -> int:
        return self.ints.shape[0]


class QueryEngine(Protocol):
    def run(self, setup: list[str], tables: list[Table], sql: str) -> dict:
        """Create tables (setup), load data, run `sql` read-only; {"columns", "rows", "truncated"}.

        Raises QueryError for problems with the client's SQL.
        """


# per_member aggregates: (column, base variable, SQL). Time-weighted where a mean or total is meant.
PER_MEMBER = [
    ("rain", "precip", "sum(precip * dt_hours)"),
    ("precip_max", "precip", "max(precip)"),
    ("wet_hours", "precip", "sum(CASE WHEN precip > 0.1 THEN dt_hours ELSE 0 END)"),
    ("snow_total", "snow", "sum(snow * dt_hours)"),
    ("tmax", "t2m", "max(t2m)"),
    ("tmin", "t2m", "min(t2m)"),
    ("tmean", "t2m", "sum(t2m * dt_hours) / sum(dt_hours)"),
    ("feels_min", "feels_like", "min(feels_like)"),
    ("feels_max", "feels_like", "max(feels_like)"),
    ("gust_max", "gust", "max(gust)"),
    ("wind_max", "wind", "max(wind)"),
    ("wind_mean", "wind", "sum(wind * dt_hours) / sum(dt_hours)"),
    ("cloud_mean", "cloud", "sum(cloud * dt_hours) / sum(dt_hours)"),
    ("radiation_mean", "radiation", "sum(radiation * dt_hours) / sum(dt_hours)"),
    ("cape_max", "cape", "max(cape)"),
    ("rh_mean", "rh", "sum(rh * dt_hours) / sum(dt_hours)"),
    ("headwind_mean", "headwind", "sum(headwind * dt_hours) / sum(dt_hours)"),
    ("crosswind_max", "crosswind", "max(crosswind)"),
]

# Fetched when a query names no variable at all (e.g. SELECT * FROM samples).
DEFAULT_VARIABLES = ["t2m", "precip", "wind", "gust", "cloud"]

MACROS = [
    # Fraction of rows (members, after GROUP BY model) for which a condition holds. Rows where it's NULL (a
    # variable the model doesn't provide) are left out; all-NULL gives NULL, not a misleading 0.
    "CREATE MACRO prob(c) AS avg(CASE WHEN c THEN 1.0 WHEN NOT c THEN 0.0 END)",
    # Fraction of a model's members in the current group, e.g. GROUP BY model, outcome.
    "CREATE MACRO share(m) AS count(*) / sum(count(*)) OVER (PARTITION BY m)",
]

_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]*")


def referenced_variables(sql: str, available: set[str]) -> set[str]:
    """Catalogue variables a query needs, so only those are fetched.

    Matches identifiers in the SQL against variable names and per_member column names. Over-matching (a name
    inside a string literal) only costs an extra download.
    """
    tokens = set(_IDENTIFIER.findall(sql.lower()))
    names = {v for v in available if v in tokens}
    names |= {base for column, base, _ in PER_MEMBER if column in tokens and base in available}
    return names or (set(DEFAULT_VARIABLES) & available)


@dataclass
class ModelEntry:
    """One source's samples for one window."""
    model_id: int
    window: int
    samples: SourceSamples


@dataclass
class CubeData:
    variables: list[str]
    windows: list[tuple[datetime, datetime]]
    lat: np.ndarray
    lon: np.ndarray
    distance_km: np.ndarray | None  # routes only
    kind: str  # "point", "places", "area" or "route"
    tz: tzinfo
    models: list[dict]  # per model_id: model, provider, run, members, resolution, note
    entries: list[ModelEntry] = field(default_factory=list)
    place_names: list[str] | None = None  # per point, for "places" queries

    def setup(self) -> list[str]:
        variables = self.variables
        columns = ", ".join(f"r.{v}" for v in variables)
        aggregates = ",\n".join(f"  {sql} AS {column}" for column, base, sql in PER_MEMBER if base in variables)
        # Per member *and point* (except routes, whose points are the rider's positions over time): for areas,
        # "anywhere" is then a max over points rather than a sum across them; for places, one row per place.
        point_columns = "" if self.kind == "route" else ", point, place, lat, lon"
        statements = [
            "CREATE TABLE models (model_id INTEGER, model VARCHAR, provider VARCHAR, run TIMESTAMP, "
            "members INTEGER, resolution VARCHAR, note VARCHAR)",
            "CREATE TABLE windows (w INTEGER, start_local_min INTEGER, end_local_min INTEGER)",
            "CREATE TABLE times (model_id INTEGER, w INTEGER, t INTEGER, epoch_min INTEGER, "
            "local_epoch_min INTEGER, dt_hours FLOAT)",
            "CREATE TABLE points (p INTEGER, lat FLOAT, lon FLOAT, distance_km FLOAT)",
            "CREATE TABLE place_names (p INTEGER, place VARCHAR)",
            "CREATE TABLE raw (model_id INTEGER, member INTEGER, w INTEGER, t INTEGER, p INTEGER"
            + "".join(f", {v} FLOAT" for v in variables) + ")",
            f"""CREATE VIEW samples AS
SELECT m.model, r.member,
  make_timestamp(tm.local_epoch_min::BIGINT * 60000000) AS time,
  make_timestamp(tm.epoch_min::BIGINT * 60000000) AS time_utc,
  tm.dt_hours,
  make_timestamp(wi.start_local_min::BIGINT * 60000000) AS window_start,
  make_timestamp(wi.end_local_min::BIGINT * 60000000) AS window_end,
  r.p AS point, pn.place, pt.lat, pt.lon, pt.distance_km{", " + columns if columns else ""}
FROM raw r JOIN models m USING (model_id) JOIN times tm USING (model_id, w, t)
  JOIN windows wi USING (w) JOIN points pt USING (p) LEFT JOIN place_names pn USING (p)""",
            f"""CREATE VIEW per_member AS
SELECT model, member, window_start, window_end{point_columns}{"," if aggregates else ""}
{aggregates}
FROM samples GROUP BY model, member, window_start, window_end{point_columns}""",
            *MACROS,
        ]
        for model_id, m in enumerate(self.models):
            statements.append(
                "INSERT INTO models VALUES ("
                f"{model_id}, {_literal(m['model'])}, {_literal(m['provider'])}, "
                f"TIMESTAMP {_literal(m['run'].strftime('%Y-%m-%d %H:%M:%S'))}, {int(m['members'])}, "
                f"{_literal(m['resolution'])}, {_literal(m['note'])})")
        for p, name in enumerate(self.place_names or []):
            statements.append(f"INSERT INTO place_names VALUES ({p}, {_literal(name)})")
        return statements

    def tables(self) -> list[Table]:
        windows = np.array([[w, _local_minutes(a, self.tz), _local_minutes(b, self.tz)]
                            for w, (a, b) in enumerate(self.windows)], dtype=np.int64)
        distance = self.distance_km if self.distance_km is not None else np.full(len(self.lat), np.nan)
        points = Table("points", ["p"], ["lat", "lon", "distance_km"],
                       np.arange(len(self.lat))[:, None], np.column_stack([self.lat, self.lon, distance]))
        time_ints, time_floats, raw_ints, raw_floats = [], [], [], []
        for entry in self.entries:
            s = entry.samples
            n_t = len(s.times)
            time_ints.append(np.column_stack([
                np.full(n_t, entry.model_id), np.full(n_t, entry.window), np.arange(n_t),
                [_utc_minutes(t) for t in s.times], [_local_minutes(t, self.tz) for t in s.times]]))
            time_floats.append(np.asarray(s.samples.dt_hours, dtype=np.float64)[:, None])
            ints, floats = self._raw(entry, n_t)
            raw_ints.append(ints)
            raw_floats.append(floats)
        n_vars = len(self.variables)
        return [
            Table("windows", ["w", "start_local_min", "end_local_min"], [], windows, np.zeros((len(windows), 0))),
            points,
            Table("times", ["model_id", "w", "t", "epoch_min", "local_epoch_min"], ["dt_hours"],
                  _stack(time_ints, 5), _stack(time_floats, 1)),
            Table("raw", ["model_id", "member", "w", "t", "p"], list(self.variables),
                  _stack(raw_ints, 5), _stack(raw_floats, n_vars)),
        ]

    def _raw(self, entry: ModelEntry, n_t: int) -> tuple[np.ndarray, np.ndarray]:
        samples = entry.samples.samples
        # From the run info: a model may provide none of the requested variables (all NULL), so no array to ask.
        members = entry.samples.info.members
        n_p = len(self.lat) if samples.has_space else 1
        values = []
        for name in self.variables:
            v = samples.variables.get(name)
            if v is None:
                v = np.full((members, n_t, n_p), np.nan)
            elif not samples.has_space:
                v = v[:, :, None]  # [member, time, 1]
            values.append(np.asarray(v, dtype=np.float32).reshape(-1))
        member, t, p = (a.reshape(-1) for a in np.meshgrid(np.arange(members), np.arange(n_t), np.arange(n_p),
                                                           indexing="ij"))
        if self.kind == "route":
            p = t  # each route sample is at its own point
        ints = np.column_stack([np.full(member.size, entry.model_id), member, np.full(member.size, entry.window), t, p])
        floats = np.column_stack(values) if values else np.zeros((member.size, 0))
        return ints, floats


def _stack(parts: list[np.ndarray], columns: int) -> np.ndarray:
    return np.concatenate(parts) if parts else np.zeros((0, columns))


def _utc_minutes(t: datetime) -> int:
    return int(t.timestamp() // 60)


def _local_minutes(t: datetime, tz: tzinfo) -> int:
    """Minutes since the epoch of the local wall-clock time (so the SQL TIMESTAMP shows local time)."""
    local = t.astimezone(tz)
    return int(local.replace(tzinfo=timezone.utc).timestamp() // 60)


def _literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def describe_schema() -> str:
    """Schema and usage reference for the help tool."""
    variables = "\n".join(f"  {v.name} [{v.unit}]: {v.description}" + (" (routes only)" if v.route_only else "")
                          for v in CATALOG.values())
    per_member = "\n".join(f"  {column}: {sql}" for column, _, sql in PER_MEMBER)
    return f"""Forecast queries are DuckDB SQL over the ensemble members of each model. Every ensemble member is one
possible weather outcome; a model's members together show how likely outcomes are. There is no cross-model
weighting: compare models yourself (they often disagree, which is itself useful information).

Views:
- samples: one row per model, member, window, time sample and point.
  model, member, time (local), time_utc, dt_hours (duration the sample stands for), window_start, window_end,
  point, place (name, for places queries), lat, lon, distance_km (routes: distance from the start), plus
  variables:
{variables}
  Rows are half-intervals: sum(x * dt_hours) integrates over time (precip in mm/h → mm). Instant values sit at
  model steps and window edges, so min()/max() see every model value.
  Routes: each sample is where the rider is at that time. Areas: one row per grid point and time.
- per_member: one row per model, member, window and point (point, place, lat, lon), aggregated over time.
  Routes: one row per model, member, aggregated over the whole ride.
{per_member}
- models: model, provider, run, members, resolution, note.

Macros:
- prob(condition): fraction of rows where the condition holds, e.g. per model: SELECT model, prob(rain < 0.5)
  FROM per_member GROUP BY model.
- share(model): a model's fraction of members in the current group, e.g. GROUP BY model, outcome.

Only variables the query mentions (directly or through per_member columns) are fetched; mention what you need.

Examples:
- Chance of a dry, mild ride per model:
  SELECT model, prob(rain < 0.5 AND tmax <= 26 AND gust_max < 40) AS p FROM per_member GROUP BY model
- What goes wrong, and how badly:
  SELECT model, CASE WHEN rain >= 0.5 THEN 'wet' WHEN tmax > 26 THEN 'too warm' ELSE 'ok' END AS outcome,
         share(model) AS p, quantile_cont(rain, 0.9) AS rain_p90, max(tmax) AS tmax_max
  FROM per_member GROUP BY model, outcome ORDER BY model, outcome
- Hour-by-hour overview:
  SELECT model, date_trunc('hour', time) AS hour, quantile_cont(t2m, [0.1, 0.5, 0.9]) AS t2m,
         prob(precip > 0.1) AS p_rain, max(gust) AS gust_max
  FROM samples GROUP BY ALL ORDER BY model, hour
- Where along a route it rains:
  SELECT model, round(distance_km / 10) * 10 AS km, prob(precip > 0.2) AS p_rain FROM samples
  GROUP BY ALL ORDER BY model, km
- Compare places (places='Cologne@50.94,6.96; Bonn@50.73,7.10'):
  SELECT place, model, prob(rain < 0.5) AS p_dry FROM per_member GROUP BY ALL ORDER BY place, model
- Best 3-hour slot (window_hours=3, start..end = search range), cautious across models:
  SELECT window_start, window_end, min(p) AS p_worst_model FROM (
    SELECT window_start, window_end, model, prob(rain < 0.5 AND gust_max < 40) AS p FROM per_member GROUP BY ALL)
  GROUP BY ALL ORDER BY p_worst_model DESC LIMIT 10
- Area, "anywhere": aggregate over points per member first:
  SELECT model, prob(gust > 70) AS p_gust70_anywhere, prob(wettest > 10) AS p_10mm_somewhere FROM (
    SELECT model, member, max(gust_max) AS gust, max(rain) AS wettest FROM per_member GROUP BY model, member)
  GROUP BY model

Missing data: a variable a model doesn't provide (see missing_variables in the model table, e.g. AIFS has no
gusts) is NULL. prob() ignores NULL rows and returns NULL if all are NULL; CASE conditions on NULL are not true,
so check missing_variables before reading per-model breakdowns.

Reading the results:
- Each model's members are a sample of possible weather. Raw ensembles are somewhat overconfident: treat 0%/100%
  as unlikely/likely, not impossible/certain.
- Check lead_hours in the model table: beyond ~7 days details get unreliable, beyond ~10 days use broad
  tendencies only.
- When models disagree, say so; that is real uncertainty.
"""
