import logging
import traceback

from jakarta.inject import Singleton
from micronaut.scheduling.annotation import Scheduled

from .config import PrefetchConfig
from .forecast_service import ForecastService

log = logging.getLogger(__name__)


@Singleton
class PrefetchJob:
    """Keeps the newest runs warm (weather_core.prefetch). Python can't start threads, so Micronaut's scheduler runs
    the passes."""

    def __init__(self, service: ForecastService, config: PrefetchConfig):
        self.prefetcher = service.prefetcher
        self.enabled = config.enabled

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
