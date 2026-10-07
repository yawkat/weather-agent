import logging
from datetime import datetime
from typing import Annotated

from jakarta.inject import Singleton
from micronaut.context.annotation import Value
from weather_core.budget import DownloadBudget
from weather_core.forecast import Forecaster
from weather_core.sources.ecmwf import AIFS_ENS, IFS_ENS, EcmwfSource
from weather_core.store import FieldStore

from .http_fetcher import MicronautFetcher
from .java_io import JavaDecoder
from .sql_engine import DuckDbEngine

log = logging.getLogger(__name__)


@Singleton
class ForecastService:
    """Builds the forecast engine from configuration; tools call into `forecaster`."""

    def __init__(self,
                 fetcher: MicronautFetcher,
                 engine: DuckDbEngine,
                 cache_dir: Annotated[str, Value("${weather.cache-dir:var/cache}")],
                 default_tz: Annotated[str, Value("${weather.timezone:Europe/Berlin}")],
                 ecmwf_base_url: Annotated[str, Value("${weather.ecmwf.base-url:`https://data.ecmwf.int/forecasts`}")],
                 max_mb_per_query: Annotated[int, Value("${weather.download.max-mb-per-query:4000}")],
                 max_mb_per_hour: Annotated[int, Value("${weather.download.max-mb-per-hour:20000}")]):
        decoder = JavaDecoder()
        store = FieldStore(cache_dir)
        budget = DownloadBudget(max_mb_per_query * 1_000_000, max_mb_per_hour * 1_000_000)

        def log_download(source: str, run: datetime, size: int) -> None:
            log.info("downloaded %s run %s: %.1f MB", source, run.strftime("%Y-%m-%d %H:%MZ"), size / 1e6)

        sources = [
            EcmwfSource(IFS_ENS, fetcher, decoder, store, base_url=ecmwf_base_url, on_download=log_download,
                        budget=budget),
            EcmwfSource(AIFS_ENS, fetcher, decoder, store, base_url=ecmwf_base_url, on_download=log_download,
                        budget=budget),
        ]
        self.forecaster = Forecaster(sources, engine, default_tz=default_tz)
