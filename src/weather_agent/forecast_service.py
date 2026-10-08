import logging
from datetime import datetime, timedelta

from jakarta.inject import Singleton
from weather_core.budget import DownloadBudget
from weather_core.forecast import Forecaster
from weather_core.prefetch import Prefetcher, WarmSet
from weather_core.sources.dwd import ICON_D2_EPS, ICON_D2_RUC_EPS, ICON_EU_EPS, DwdIconSource
from weather_core.sources.ecmwf import AIFS_ENS, IFS_ENS, EcmwfSource
from weather_core.sources.mirrors import Mirrors
from weather_core.store import FieldStore

from .config import DownloadConfig, DwdConfig, EcmwfConfig, PrefetchConfig, QueryConfig, WeatherConfig, names
from .http_fetcher import Fetchers
from .java_io import JavaDecoder

log = logging.getLogger(__name__)


@Singleton
class ForecastService:
    """Builds the forecast engine from configuration; tools call into `forecaster`."""

    def __init__(self, fetchers: Fetchers, config: WeatherConfig, ecmwf: EcmwfConfig, dwd: DwdConfig,
                 download: DownloadConfig, query: QueryConfig, prefetch: PrefetchConfig):
        decoder = JavaDecoder()
        store = FieldStore(config.cache_dir)
        budget = DownloadBudget(download.max_mb_per_query * 1_000_000, download.max_mb_per_hour * 1_000_000)

        def log_download(source: str, run: datetime, size: int) -> None:
            log.info("downloaded %s run %s: %.1f MB", source, run.strftime("%Y-%m-%d %H:%MZ"), size / 1e6)

        ecmwf_hosts = Mirrors(ecmwf.base_url, names(ecmwf.mirrors))
        sources = [
            *(EcmwfSource(model, fetchers.ecmwf, decoder, store, hosts=ecmwf_hosts, on_download=log_download,
                          budget=budget)
              for model in (IFS_ENS, AIFS_ENS)),
            *(DwdIconSource(model, fetchers.dwd, decoder, store, base_url=dwd.base_url, on_download=log_download,
                            budget=budget)
              for model in (ICON_D2_RUC_EPS, ICON_D2_EPS, ICON_EU_EPS)),
        ]

        warm = WarmSet(names(prefetch.models), max_entries=prefetch.max_entries,
                       ttl=timedelta(hours=prefetch.ttl_hours))
        self.forecaster = Forecaster(sources, default_tz=config.timezone, budget=budget,
                                     max_concurrent=query.max_concurrent, eval_bytes=query.memory_mb * 2**20,
                                     eval_timeout_s=query.timeout_ms / 1000, max_rows=query.max_rows, warm=warm)
        self.prefetcher = Prefetcher(sources, warm, budget, keep_free_bytes=prefetch.keep_free_mb * 1_000_000,
                                     full_models=names(prefetch.full_models),
                                     full_variables=names(prefetch.full_variables))
