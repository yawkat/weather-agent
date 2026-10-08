import logging
from datetime import datetime, timedelta
from typing import Annotated

from jakarta.inject import Singleton
from micronaut.context.annotation import Value
from weather_core.budget import DownloadBudget
from weather_core.forecast import Forecaster
from weather_core.prefetch import Prefetcher, WarmSet
from weather_core.sources.dwd import ICON_D2_EPS, ICON_D2_RUC_EPS, ICON_EU_EPS, DwdIconSource
from weather_core.sources.ecmwf import AIFS_ENS, IFS_ENS, EcmwfSource
from weather_core.sources.mirrors import Mirrors
from weather_core.store import FieldStore

from .http_fetcher import Fetchers
from .java_io import JavaDecoder

log = logging.getLogger(__name__)


@Singleton
class ForecastService:
    """Builds the forecast engine from configuration; tools call into `forecaster`."""

    def __init__(self,
                 fetchers: Fetchers,
                 cache_dir: Annotated[str, Value("${weather.cache-dir:var/cache}")],
                 default_tz: Annotated[str, Value("${weather.timezone:Europe/Berlin}")],
                 ecmwf_base_url: Annotated[str, Value("${weather.ecmwf.base-url:`https://data.ecmwf.int/forecasts`}")],
                 ecmwf_mirrors: Annotated[str, Value("${weather.ecmwf.mirrors:}")],
                 dwd_base_url: Annotated[str, Value("${weather.dwd.base-url:`https://opendata.dwd.de/weather/nwp/v1/m`}")],
                 max_mb_per_query: Annotated[int, Value("${weather.download.max-mb-per-query:4000}")],
                 max_mb_per_hour: Annotated[int, Value("${weather.download.max-mb-per-hour:20000}")],
                 memory_mb: Annotated[int, Value("${weather.query.memory-mb:1024}")],
                 timeout_ms: Annotated[int, Value("${weather.query.timeout-ms:15000}")],
                 max_rows: Annotated[int, Value("${weather.query.max-rows:500}")],
                 max_concurrent: Annotated[int, Value("${weather.query.max-concurrent:4}")],
                 # RUC-EPS isn't kept warm: a new run every hour, and only worth it for the next few hours.
                 prefetch_models: Annotated[str, Value("${weather.prefetch.models:ecmwf-ens,ecmwf-aifs-ens,icon-d2-eps,icon-eu-eps}")],
                 prefetch_max_entries: Annotated[int, Value("${weather.prefetch.max-entries:64}")],
                 prefetch_ttl_hours: Annotated[int, Value("${weather.prefetch.ttl-hours:48}")],
                 prefetch_keep_free_mb: Annotated[int, Value("${weather.prefetch.keep-free-mb:10000}")]):
        decoder = JavaDecoder()
        store = FieldStore(cache_dir)
        budget = DownloadBudget(max_mb_per_query * 1_000_000, max_mb_per_hour * 1_000_000)

        def log_download(source: str, run: datetime, size: int) -> None:
            log.info("downloaded %s run %s: %.1f MB", source, run.strftime("%Y-%m-%d %H:%MZ"), size / 1e6)

        ecmwf_hosts = Mirrors(ecmwf_base_url, [url.strip() for url in ecmwf_mirrors.split(",") if url.strip()])
        sources = [
            *(EcmwfSource(model, fetchers.ecmwf, decoder, store, hosts=ecmwf_hosts, on_download=log_download,
                          budget=budget)
              for model in (IFS_ENS, AIFS_ENS)),
            *(DwdIconSource(model, fetchers.dwd, decoder, store, base_url=dwd_base_url, on_download=log_download,
                            budget=budget)
              for model in (ICON_D2_RUC_EPS, ICON_D2_EPS, ICON_EU_EPS)),
        ]
        warm = WarmSet([m.strip() for m in prefetch_models.split(",") if m.strip()], max_entries=prefetch_max_entries,
                       ttl=timedelta(hours=prefetch_ttl_hours))
        self.forecaster = Forecaster(sources, default_tz=default_tz, budget=budget, max_concurrent=max_concurrent,
                                     eval_bytes=memory_mb * 2**20, eval_timeout_s=timeout_ms / 1000,
                                     max_rows=max_rows, warm=warm)
        self.prefetcher = Prefetcher(sources, warm, budget, keep_free_bytes=prefetch_keep_free_mb * 1_000_000)
