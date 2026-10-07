"""weather_core Fetcher on Micronaut's HTTP client.

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
from micronaut.context.annotation import Value
from micronaut.http import HttpRequest
from micronaut.http.client import HttpClient
from weather_core.sources.base import Download, SourceError

log = logging.getLogger(__name__)

BYTES = None  # default body type: Micronaut ByteBuffer (array classes can't pass the host-class filter)
STRING = java.type("java.lang.String")
ATTEMPTS = 3


class UpstreamError(SourceError):
    """Download failure; reported to the client like any other source problem."""


@Singleton
class MicronautFetcher:
    def __init__(self, client: HttpClient,
                 user_agent: Annotated[str, Value("${weather.user-agent:weather-agent/0.1 (personal use)}")],
                 max_connections: Annotated[int, Value("${weather.max-connections:8}")]):
        self.client = client.toAsync()
        self.user_agent = user_agent
        self.batch = max_connections

    def get(self, url: str) -> bytes | None:
        """Body of a small text resource, or None on 404."""
        status, body = self._exchange([self._request(url, None)], STRING)[0]
        if status == 404:
            return None
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

    def _request(self, url: str, byte_range: tuple[int, int] | None):
        request = HttpRequest.GET(url).header("User-Agent", self.user_agent)
        if byte_range is not None:
            request = request.header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
        return request

    def _exchange(self, requests: list, body_type) -> list[tuple[int, object]]:
        """Run requests concurrently; (status, body) each. Retries 429, 5xx and network errors with backoff."""
        results: list[tuple[int, object] | None] = [None] * len(requests)
        pending = list(range(len(requests)))
        for attempt in range(ATTEMPTS):
            if attempt:
                time.sleep(2 ** (attempt - 1))
            futures = {i: (self.client.exchange(requests[i]) if body_type is None
                           else self.client.exchange(requests[i], body_type)).toCompletableFuture()
                       for i in pending}
            retry = []
            for i, future in futures.items():
                try:
                    response = future.get()
                    results[i] = (response.code(), _body(response, body_type))
                except BaseException as e:  # Java exceptions are foreign, not Python Exceptions
                    if isinstance(e, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                        raise
                    status = _status_of(e)
                    if status is not None and status < 500 and status != 429:
                        results[i] = (status, None)  # definitive, e.g. 404
                    elif attempt == ATTEMPTS - 1:
                        log.error("giving up on %s after %d attempts: %s", requests[i].getUri(), ATTEMPTS,
                                  _describe(e, status))
                        raise UpstreamError(f"{requests[i].getUri()}: {_describe(e, status)}") from None
                    else:
                        log.warning("retrying %s (attempt %d of %d): %s", requests[i].getUri(), attempt + 2,
                                    ATTEMPTS, _describe(e, status))
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
