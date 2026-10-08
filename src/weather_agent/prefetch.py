import logging
import traceback
from typing import Annotated

from jakarta.inject import Singleton
from micronaut.context.annotation import Value
from micronaut.scheduling.annotation import Scheduled

from .forecast_service import ForecastService

log = logging.getLogger(__name__)


@Singleton
class PrefetchJob:
    """Keeps recently asked forecasts warm (weather_core.prefetch). Python can't start threads, so Micronaut's
    scheduler runs the passes."""

    def __init__(self, service: ForecastService,
                 enabled: Annotated[bool, Value("${weather.prefetch.enabled:true}")]):
        self.prefetcher = service.prefetcher
        self.enabled = enabled

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
