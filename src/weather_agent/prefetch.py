import logging
import traceback
from typing import Annotated

from micronaut.context.annotation import Context, Value
from micronaut.scheduling.annotation import Scheduled

from .forecast_service import ForecastService

log = logging.getLogger(__name__)


@Context  # created at startup, so a budget that leaves prefetch nothing fails it right away
class PrefetchJob:
    """Keeps recently asked forecasts warm (weather_core.prefetch). Python can't start threads, so Micronaut's
    scheduler runs the passes."""

    def __init__(self, service: ForecastService,
                 enabled: Annotated[bool, Value("${weather.prefetch.enabled:true}")]):
        self.prefetcher = service.prefetcher
        self.enabled = enabled
        budget = self.prefetcher.budget
        if enabled and budget is not None and self.prefetcher.keep_free_bytes >= budget.per_hour_bytes:
            # Every pass would be refused by the budget, and refusals are only logged at debug level.
            raise ValueError(f"weather.prefetch.keep-free-mb ({self.prefetcher.keep_free_bytes // 1_000_000}) must "
                             f"be below weather.download.max-mb-per-hour ({budget.per_hour_bytes // 1_000_000}), or "
                             f"prefetch can never download; lower it or disable prefetch")

    @Scheduled(fixedDelay="${weather.prefetch.interval:10m}", initialDelay="${weather.prefetch.initial-delay:2m}")
    def refresh(self) -> None:
        if not self.enabled:
            return
        try:
            self.prefetcher.refresh()
        except BaseException as e:  # Java exceptions surface as foreign exceptions, not Python Exceptions
            if isinstance(e, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                raise
            log.error("prefetch failed: %s: %s\n%s", type(e).__name__, e, traceback.format_exc())
