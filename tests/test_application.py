import pytest

from pyronaut.test import *

@pytest.fixture
def my_context(request):
    fixture = micronaut_test_fixture(request, MicronautTest(environments=["test"], transactional=False))
    yield fixture
    fixture.stop()

def test_context(my_context):
    assert my_context.isRunning()


def test_geocoder_is_wired(my_context):
    """Nominatim is stubbed out; checks the bean builds and the tool path runs on GraalPy."""
    import json

    from weather_agent.geocoding import GeocodeService

    geocoder = my_context.getBean(GeocodeService).geocoder
    requests = []

    def get(url):
        requests.append(url)
        return json.dumps([{"lat": "50.73", "lon": "7.1", "display_name": "Bonn, Deutschland"}]).encode()

    geocoder._get = get
    geocoder._min_interval = 0
    assert geocoder.resolve_many("  Bonn ")["results"][0]["label"] == "Bonn, Deutschland"
    assert geocoder.resolve("BONN").lat == 50.73
    assert len(requests) == 1


def test_fetcher_reports_error_status(my_context):
    """A non-200, non-404 answer (here GET /mcp on our own server) is an UpstreamError naming the status."""
    from micronaut.runtime.server import EmbeddedServer

    from weather_agent.http_fetcher import MicronautFetcher, UpstreamError

    server = my_context.getBean(EmbeddedServer)
    if not server.isRunning():
        server.start()
    with pytest.raises(UpstreamError, match=r"HTTP [45]\d\d"):
        my_context.getBean(MicronautFetcher).get(f"http://localhost:{server.getPort()}/mcp")


def test_fetcher_timeout(my_context):
    """A server that never answers: the kernel completes the connection, nobody accepts or replies."""
    import time

    import java

    from weather_agent.http_fetcher import MicronautFetcher, UpstreamError

    InetAddress = java.type("java.net.InetAddress")
    socket = java.type("java.net.ServerSocket")(0, 50, InetAddress.getLoopbackAddress())
    try:
        started = time.monotonic()
        with pytest.raises(UpstreamError, match=r"no response within 0\.5 s"):
            my_context.getBean(MicronautFetcher).get(f"http://127.0.0.1:{socket.getLocalPort()}/search",
                                                     timeout_s=0.5, attempts=2)
        assert time.monotonic() - started < 5  # two 0.5 s waits plus 1 s backoff
    finally:
        socket.close()
