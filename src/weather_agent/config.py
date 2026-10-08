"""Application settings (`weather.*`). Like any Micronaut property, each can also be set through the environment,
e.g. `WEATHER_CACHE_DIR` for `weather.cache-dir`. The MCP access settings the Java filters use are in
`McpAccessConfig.java`."""

from dataclasses import dataclass

from micronaut.context.annotation import ConfigurationProperties


@ConfigurationProperties("weather")
@dataclass
class WeatherConfig:
    cache_dir: str = "var/cache"
    # Time zone for query times without one.
    timezone: str = "Europe/Berlin"
    user_agent: str = "weather-agent/0.1 (personal use)"
    # Concurrent requests per upstream (ECMWF, DWD).
    max_connections: int = 8


@ConfigurationProperties("weather.ecmwf")
@dataclass
class EcmwfConfig:
    base_url: str = "https://data.ecmwf.int/forecasts"
    # Copies of `base_url` with the same layout, tried in order while it fails (weather_core.sources.mirrors).
    mirrors: list[str] | None = None


@ConfigurationProperties("weather.dwd")
@dataclass
class DwdConfig:
    base_url: str = "https://opendata.dwd.de/weather/nwp/v1/m"


@ConfigurationProperties("weather.download")
@dataclass
class DownloadConfig:
    max_mb_per_query: int = 4000
    max_mb_per_hour: int = 20000


@ConfigurationProperties("weather.query")
@dataclass
class QueryConfig:
    memory_mb: int = 1024
    timeout_ms: int = 15000
    max_rows: int = 500
    max_concurrent: int = 4


@ConfigurationProperties("weather.geocoder")
@dataclass
class GeocoderConfig:
    # Public Nominatim: ≤1 request/s, identifying User-Agent, results cached in memory. Point this at a
    # self-hosted Nominatim if usage grows.
    url: str = "https://nominatim.openstreetmap.org"
    # Nominatim's policy asks for a User-Agent that identifies the application.
    user_agent: str = "weather-agent/0.1 (+https://github.com/yawkat/weather-agent)"
    accept_language: str = "en"
    timeout_seconds: float = 10.0


@ConfigurationProperties("weather.prefetch")
@dataclass
class PrefetchConfig:
    enabled: bool = True
    # Read by PrefetchJob's @Scheduled (as placeholders with the same defaults); declared here for the schema.
    interval: str = "10m"
    initial_delay: str = "2m"
    models: list[str] | None = None
    max_entries: int = 64
    ttl_hours: int = 48
    keep_free_mb: int = 10000
    full_models: list[str] | None = None
    full_variables: list[str] | None = None


def names(values: list[str] | None) -> list[str]:
    """A list setting cleaned up: from the environment, Micronaut splits on commas as is ("a, b", empty values).
    Unset is None: Micronaut doesn't apply a dataclass default_factory, so list defaults live in application.toml."""
    return [value.strip() for value in values or [] if value.strip()]
