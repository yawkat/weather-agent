from typing import Annotated

from jakarta.inject import Singleton
from micronaut.context.annotation import Value
from weather_core.geocode import Geocoder

from .http_fetcher import MicronautFetcher


@Singleton
class GeocodeService:
    """Builds the Nominatim geocoder from configuration; tools call into `geocoder`."""

    def __init__(self,
                 fetcher: MicronautFetcher,
                 base_url: Annotated[str, Value("${weather.geocoder.url:`https://nominatim.openstreetmap.org`}")],
                 # Nominatim's policy asks for a User-Agent that identifies the application.
                 user_agent: Annotated[str, Value("${weather.geocoder.user-agent:`weather-agent/0.1 (+https://github.com/yawkat/weather-agent)`}")],
                 accept_language: Annotated[str, Value("${weather.geocoder.accept-language:en}")],
                 timeout_s: Annotated[float, Value("${weather.geocoder.timeout-seconds:10}")]):
        # Requests hold the geocoder's one-at-a-time lock, so fail fast instead of using the client's read
        # timeout (minutes, for GRIB files): at most two attempts of timeout_s each, plus 1 s backoff.
        self.geocoder = Geocoder(
            lambda url: fetcher.get(url, user_agent=user_agent, timeout_s=timeout_s, attempts=2), base_url,
            accept_language=accept_language)
