from jakarta.inject import Singleton
from weather_core.geocode import Geocoder

from .config import GeocoderConfig
from .http_fetcher import Fetchers


@Singleton
class GeocodeService:
    """Builds the Nominatim geocoder from configuration; tools call into `geocoder`."""

    def __init__(self, fetchers: Fetchers, config: GeocoderConfig):
        # Requests hold the geocoder's one-at-a-time lock, so fail fast: at most two attempts of `timeout_seconds`
        # each, plus 1 s backoff.
        self.geocoder = Geocoder(
            lambda url: fetchers.nominatim.get(url, user_agent=config.user_agent, timeout_s=config.timeout_seconds,
                                               attempts=2),
            config.url, accept_language=config.accept_language)
