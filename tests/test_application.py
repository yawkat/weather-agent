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

    from weather_agent.http_fetcher import Fetchers, UpstreamError

    server = my_context.getBean(EmbeddedServer)
    if not server.isRunning():
        server.start()
    with pytest.raises(UpstreamError, match=r"HTTP [45]\d\d"):
        my_context.getBean(Fetchers).nominatim.get(f"http://localhost:{server.getPort()}/mcp")


def test_fetcher_timeout(my_context):
    """A server that never answers: the kernel completes the connection, nobody accepts or replies."""
    import time

    import java

    from weather_agent.http_fetcher import Fetchers, UpstreamError

    InetAddress = java.type("java.net.InetAddress")
    socket = java.type("java.net.ServerSocket")(0, 50, InetAddress.getLoopbackAddress())
    try:
        started = time.monotonic()
        with pytest.raises(UpstreamError, match=r"no response within 0\.5 s"):
            my_context.getBean(Fetchers).nominatim.get(f"http://127.0.0.1:{socket.getLocalPort()}/search",
                                                       timeout_s=0.5, attempts=2)
        assert time.monotonic() - started < 5  # two 0.5 s waits plus 1 s backoff
    finally:
        socket.close()


def _mcp(context, method, params):
    """One JSON-RPC request to the running server's /mcp endpoint."""
    import json

    import java
    from micronaut.runtime.server import EmbeddedServer

    server = context.getBean(EmbeddedServer)
    if not server.isRunning():
        server.start()
    HttpClient = java.type("java.net.http.HttpClient")
    HttpRequest = java.type("java.net.http.HttpRequest")
    BodyHandlers = java.type("java.net.http.HttpResponse$BodyHandlers")
    URI = java.type("java.net.URI")
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    request = (HttpRequest.newBuilder(URI.create(f"http://localhost:{server.getPort()}/mcp"))
               .header("Content-Type", "application/json")
               .header("Accept", "application/json, text/event-stream")
               .POST(HttpRequest.BodyPublishers.ofString(body)).build())
    response = HttpClient.newHttpClient().send(request, BodyHandlers.ofString())
    assert response.statusCode() == 200, response.body()
    return json.loads(str(response.body()))["result"]


def test_meteogram_tool_links_its_view(my_context):
    tools = {t["name"]: t for t in _mcp(my_context, "tools/list", {})["tools"]}
    meteogram = tools["show_meteogram"]
    uri = meteogram["_meta"]["ui"]["resourceUri"]
    assert uri == "ui://weather-agent/meteogram.html"
    assert meteogram["inputSchema"]["required"] == ["lat", "lon"]
    assert "forecast" in tools  # annotated tools are still there

    contents = _mcp(my_context, "resources/read", {"uri": uri})["contents"][0]
    assert contents["mimeType"] == "text/html;profile=mcp-app"
    assert "ui/initialize" in contents["text"]


def test_meteogram_tool_returns_summary_and_chart(my_context):
    import json

    from weather_agent.forecast_service import ForecastService

    forecaster = my_context.getBean(ForecastService).forecaster
    calls = []

    def meteogram(lat, lon, **kwargs):
        calls.append((lat, lon, kwargs))
        return {"models": [{"model": "m", "daily": []}]}, {"models": [{"model": "m", "times": [0, 3600]}]}

    forecaster.meteogram = meteogram
    try:
        result = _mcp(my_context, "tools/call", {"name": "show_meteogram",
                                                 "arguments": {"lat": 50.7, "lon": 7.1, "label": "Bonn"}})
        assert not result["isError"]
        assert json.loads(result["content"][0]["text"])["models"][0]["model"] == "m"
        assert result["structuredContent"]["models"][0]["times"] == [0, 3600]
        assert calls == [(50.7, 7.1, {"start": None, "end": None, "variables": None, "sources": None,
                                      "label": "Bonn"})]

        def failing(lat, lon, **kwargs):
            raise ValueError("end must be after start")

        forecaster.meteogram = failing
        error = _mcp(my_context, "tools/call", {"name": "show_meteogram", "arguments": {"lat": 50.7, "lon": 7.1}})
        assert error["isError"] and error["content"][0]["text"] == "end must be after start"
        # The SDK checks arguments against the input schema before the tool runs.
        invalid = _mcp(my_context, "tools/call", {"name": "show_meteogram", "arguments": {"lat": "x", "lon": 7.1}})
        assert invalid["isError"]
    finally:
        del forecaster.meteogram
