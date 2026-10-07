"""weather_core Fetchers on Micronaut's HTTP client, one client ("service") per upstream.

Each upstream gets its own `micronaut.http.services.<id>` configuration (pool, timeouts, TLS), so one host's
limits and quirks don't leak into the others.

Python can't start threads here, so parallelism comes from the client itself: a batch of requests is started
through the async API, then each result is awaited in turn. Batches bound both concurrency and the number of
response bodies held in memory. Bodies stay Java byte arrays and are written to the destination file through
NIO, so they never get copied into Python objects.
"""

import logging
import time
from typing import Annotated

import java
from jakarta.inject import Singleton
from java.nio import ByteBuffer
from java.nio.channels import FileChannel
from java.nio.file import Path, StandardOpenOption
from java.util.concurrent import TimeUnit
from micronaut.context.annotation import Value
from micronaut.http import HttpRequest
from micronaut.http.client import HttpClient
from micronaut.http.client.annotation import Client
from weather_core.sources.base import Download, SourceError

log = logging.getLogger(__name__)

BYTES = None  # default body type: Micronaut ByteBuffer (array classes can't pass the host-class filter)
STRING = java.type("java.lang.String")
ATTEMPTS = 3


class UpstreamError(SourceError):
    """Download failure; reported to the client like any other source problem."""


@Singleton
class Fetchers:
    """One Fetcher per upstream service."""

    def __init__(self,
                 ecmwf: Annotated[HttpClient, Client("ecmwf")],
                 dwd: Annotated[HttpClient, Client("dwd")],
                 nominatim: Annotated[HttpClient, Client("nominatim")],
                 user_agent: Annotated[str, Value("${weather.user-agent:weather-agent/0.1 (personal use)}")],
                 max_connections: Annotated[int, Value("${weather.max-connections:8}")]):
        self.ecmwf = MicronautFetcher(ecmwf, user_agent, max_connections)
        self.dwd = MicronautFetcher(dwd, user_agent, max_connections)
        self.nominatim = MicronautFetcher(nominatim, user_agent, 1)  # one request at a time anyway


class MicronautFetcher:
    def __init__(self, client: HttpClient, user_agent: str, max_connections: int):
        self.client = client.toAsync()
        self.user_agent = user_agent
        self.batch = max_connections

    def get(self, url: str, user_agent: str | None = None, timeout_s: float | None = None,
            attempts: int = ATTEMPTS) -> bytes | None:
        """Body of a small text resource, or None on 404. `timeout_s` bounds the wait for each attempt, below the
        client's read timeout (sized for GRIB files)."""
        status, body = self._exchange([self._request(url, None, user_agent)], STRING, timeout_s, attempts)[0]
        if status == 404:
            return None
        if status != 200:
            raise UpstreamError(f"{url}: HTTP {status}")
        return str(body).encode()

    def download_many(self, requests: list[Download]) -> list[int]:
        # One HTTP request per byte range, written at its offset in the destination, so each file ends up as
        # the ranges concatenated in order.
        parts = []  # (request index, range, file offset)
        for index, request in enumerate(requests):
            if request.ranges is None:
                parts.append((index, None, 0))
                continue
            offset = 0
            for start, end in request.ranges:
                parts.append((index, (start, end), offset))
                offset += end - start + 1
        total = sum(sum(b - a + 1 for a, b in r.ranges) for r in requests if r.ranges is not None)
        started = time.monotonic()
        log.info("downloading %d file(s), %d request(s), %.1f MB", len(requests), len(parts), total / 1e6)
        channels = [FileChannel.open(Path.of(r.dest), StandardOpenOption.CREATE, StandardOpenOption.WRITE,
                                     StandardOpenOption.TRUNCATE_EXISTING) for r in requests]
        written = [0] * len(requests)
        try:
            for i in range(0, len(parts), self.batch):
                batch = parts[i:i + self.batch]
                results = self._exchange([self._request(requests[idx].url, rng) for idx, rng, _ in batch], BYTES)
                for (idx, rng, offset), (status, body) in zip(batch, results):
                    url = requests[idx].url
                    if status not in (200, 206):
                        raise UpstreamError(f"{url}: HTTP {status}")
                    if rng is not None and len(body) != rng[1] - rng[0] + 1:
                        raise UpstreamError(f"{url}: expected {rng[1] - rng[0] + 1} bytes, got {len(body)}")
                    buffer = ByteBuffer.wrap(body)
                    position = offset
                    while buffer.hasRemaining():
                        position += channels[idx].write(buffer, position)
                    written[idx] += len(body)
        finally:
            for channel in channels:
                channel.close()
        elapsed = time.monotonic() - started
        log.info("downloaded %.1f MB in %.1f s (%.1f MB/s)", sum(written) / 1e6, elapsed,
                 sum(written) / 1e6 / max(elapsed, 1e-3))
        return written

    def _request(self, url: str, byte_range: tuple[int, int] | None, user_agent: str | None = None):
        request = HttpRequest.GET(url).header("User-Agent", user_agent or self.user_agent)
        if byte_range is not None:
            request = request.header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
        return request

    def _exchange(self, requests: list, body_type, timeout_s: float | None = None,
                  attempts: int = ATTEMPTS) -> list[tuple[int, object]]:
        """Run requests concurrently; (status, body) each. Retries 429, 5xx, network errors and timeouts with
        backoff. `timeout_s` bounds each wait for a response; the abandoned request is cancelled."""
        results: list[tuple[int, object] | None] = [None] * len(requests)
        pending = list(range(len(requests)))
        for attempt in range(attempts):
            if attempt:
                time.sleep(2 ** (attempt - 1))
            futures = {i: (self.client.exchange(requests[i]) if body_type is None
                           else self.client.exchange(requests[i], body_type)).toCompletableFuture()
                       for i in pending}
            retry = []
            for i, future in futures.items():
                try:
                    response = (future.get() if timeout_s is None
                                else future.get(int(timeout_s * 1000), TimeUnit.MILLISECONDS))
                    results[i] = (response.code(), _body(response, body_type))
                except BaseException as e:  # Java exceptions are foreign, not Python Exceptions
                    if isinstance(e, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                        raise
                    status = _status_of(e)
                    if _is_timeout(e):
                        future.cancel(True)
                        problem = f"no response within {timeout_s:g} s"
                    else:
                        problem = _describe(e, status)
                    if status is not None and status < 500 and status != 429:
                        results[i] = (status, None)  # definitive, e.g. 404
                    elif attempt == attempts - 1:
                        log.error("giving up on %s after %d attempts: %s", requests[i].getUri(), attempts, problem)
                        raise UpstreamError(f"{requests[i].getUri()}: {problem}") from None
                    else:
                        log.warning("retrying %s (attempt %d of %d): %s", requests[i].getUri(), attempt + 2,
                                    attempts, problem)
                        retry.append(i)
            pending = retry
            if not pending:
                break
        return results


def _body(response, body_type):
    body = response.body()
    if body_type is not None or body is None:
        return body
    try:
        return body.toByteArray()  # copy out of the (possibly pooled) buffer…
    finally:
        try:
            body.release()  # …and give it back
        except BaseException:
            pass


def _is_timeout(e) -> bool:
    try:
        return str(e.getClass().getName()) == "java.util.concurrent.TimeoutException"
    except BaseException:
        return False


def _status_of(e) -> int | None:
    """HTTP status of an HttpClientResponseException (possibly wrapped in an ExecutionException)."""
    cause = e
    for _ in range(3):
        try:
            return int(cause.getStatus().getCode())
        except BaseException:
            pass
        try:
            cause = cause.getCause()
        except BaseException:
            return None
        if cause is None:
            return None
    return None


def _describe(e, status: int | None) -> str:
    if status is not None:
        return f"HTTP {status}"
    try:
        cause = e.getCause() or e
        return f"{cause.getClass().getSimpleName()}: {cause.getMessage()}"
    except BaseException:
        return type(e).__name__
