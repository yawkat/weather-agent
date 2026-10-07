"""Variable catalogue: canonical names and units, and variables derived from what sources provide."""

from dataclasses import dataclass

import numpy as np

from .geometry import crosswind, headwind


@dataclass(frozen=True)
class Variable:
    name: str
    unit: str
    description: str
    route_only: bool = False


CATALOG: dict[str, Variable] = {v.name: v for v in [
    Variable("precip", "mm/h", "precipitation rate (rain, snow as water); sum(precip) is the total in mm"),
    Variable("snow", "mm/h", "snowfall rate as water equivalent"),
    Variable("t2m", "°C", "air temperature at 2 m"),
    Variable("td2m", "°C", "dew point at 2 m"),
    Variable("rh", "%", "relative humidity at 2 m"),
    Variable("feels_like", "°C", "apparent temperature (wind chill when cold, heat index when hot)"),
    Variable("wind", "km/h", "mean wind speed at 10 m"),
    Variable("wind_dir", "°", "direction the wind blows from (0 = north, 90 = east)"),
    Variable("gust", "km/h", "wind gusts at 10 m"),
    Variable("cloud", "%", "total cloud cover"),
    Variable("radiation", "W/m²", "incoming solar radiation at the surface (high = sunny)"),
    Variable("cape", "J/kg", "convective available potential energy (thunderstorm potential, >1000 notable)"),
    Variable("headwind", "km/h", "wind against the direction of travel; negative = tailwind", route_only=True),
    Variable("crosswind", "km/h", "wind across the direction of travel", route_only=True),
]}

# Variables a source may deliver directly. Everything else in CATALOG is derived.
BASE = {"precip", "snow", "t2m", "td2m", "wind_u", "wind_v", "gust", "cloud", "radiation", "cape"}


def derive(base: dict[str, np.ndarray], bearing: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """All catalogue variables computable from `base` (arrays with the time axis at position 1).

    `bearing` is the direction of travel per time sample, for routes.
    """
    out = {name: values for name, values in base.items() if name in CATALOG}
    if "wind_u" in base and "wind_v" in base:
        u, v = base["wind_u"], base["wind_v"]
        out["wind"] = np.hypot(u, v)
        # u/v give where the wind blows *to*; meteorological direction is where it comes *from*.
        out["wind_dir"] = np.degrees(np.arctan2(-u, -v)) % 360.0
    if "t2m" in base and "td2m" in base:
        out["rh"] = relative_humidity(base["t2m"], base["td2m"])
    if "t2m" in out and "wind" in out:
        out["feels_like"] = apparent_temperature(out["t2m"], out["wind"], out.get("rh"))
    if bearing is not None and "wind" in out:
        shape = [1] * out["wind"].ndim
        shape[1] = -1
        b = np.asarray(bearing).reshape(shape)
        out["headwind"] = headwind(out["wind"], out["wind_dir"], b)
        out["crosswind"] = crosswind(out["wind"], out["wind_dir"], b)
    return out


def available(base_names: set[str], route: bool) -> set[str]:
    """Catalogue variables derivable from a set of base variable names."""
    names = {n for n in base_names if n in CATALOG}
    if {"wind_u", "wind_v"} <= base_names:
        names |= {"wind", "wind_dir"}
    if {"t2m", "td2m"} <= base_names:
        names.add("rh")
    if "t2m" in names and "wind" in names:
        names.add("feels_like")
    if route and "wind" in names:
        names |= {"headwind", "crosswind"}
    return names


def relative_humidity(t_c, td_c):
    """Magnus formula, %."""
    def saturation(t):
        return np.exp(17.625 * t / (243.04 + t))
    return np.clip(100.0 * saturation(td_c) / saturation(t_c), 0.0, 100.0)


def apparent_temperature(t_c, wind_kmh, rh=None):
    """Wind chill (Environment Canada) at ≤10 °C and wind >4.8 km/h; heat index (NWS) at ≥27 °C; else air temp."""
    t_c = np.asarray(t_c, dtype=np.float64)
    wind_kmh = np.broadcast_to(np.asarray(wind_kmh, dtype=np.float64), t_c.shape)
    result = t_c.copy()
    v16 = np.power(np.maximum(wind_kmh, 0.0), 0.16)
    chill = 13.12 + 0.6215 * t_c - 11.37 * v16 + 0.3965 * t_c * v16
    cold = (t_c <= 10.0) & (wind_kmh > 4.8)
    result[cold] = chill[cold]
    if rh is not None:
        rh = np.broadcast_to(np.asarray(rh, dtype=np.float64), t_c.shape)
        t_f = t_c * 9 / 5 + 32
        hi_f = (-42.379 + 2.04901523 * t_f + 10.14333127 * rh - 0.22475541 * t_f * rh
                - 6.83783e-3 * t_f ** 2 - 5.481717e-2 * rh ** 2 + 1.22874e-3 * t_f ** 2 * rh
                + 8.5282e-4 * t_f * rh ** 2 - 1.99e-6 * t_f ** 2 * rh ** 2)
        hot = (t_c >= 27.0) & (rh >= 40.0)
        result[hot] = ((hi_f - 32) * 5 / 9)[hot]
    return result


def describe() -> str:
    """Catalogue as text, for the MCP resource the LLM reads."""
    lines = []
    for v in CATALOG.values():
        suffix = " (routes only)" if v.route_only else ""
        lines.append(f"- {v.name} [{v.unit}]: {v.description}{suffix}")
    return "\n".join(lines)
