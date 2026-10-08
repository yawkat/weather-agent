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



def test_ecmwf_mirrors_are_configured(my_context):
    from weather_agent.forecast_service import ForecastService

    sources = [s for s in my_context.getBean(ForecastService).forecaster.sources if s.name.startswith("ecmwf")]
    assert len(sources) == 2 and sources[0].hosts is sources[1].hosts
    assert sources[0].hosts.bases == ["https://data.ecmwf.int/forecasts",
                                      "https://storage.googleapis.com/ecmwf-open-data",
                                      "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"]

def test_fetcher_reports_error_status(my_context):
    """A non-200, non-404 answer (here GET /mcp on our own server) is an UpstreamError naming the status."""
    from micronaut.runtime.server import EmbeddedServer

    from weather_agent.http_fetcher import Fetchers, UpstreamError

    server = my_context.getBean(EmbeddedServer)
    if not server.isRunning():
        server.start()
    with pytest.raises(UpstreamError, match=r"HTTP [45]\d\d"):
        my_context.getBean(Fetchers).nominatim.get(f"http://localhost:{server.getPort()}/mcp")



def test_fetcher_download_404_is_not_published(my_context, tmp_path):
    """Mirrors tells a missing file (NotPublished, final from the origin) from a failing host by this type."""
    from micronaut.runtime.server import EmbeddedServer
    from weather_core.sources.base import Download, NotPublished

    from weather_agent.http_fetcher import Fetchers

    server = my_context.getBean(EmbeddedServer)
    if not server.isRunning():
        server.start()
    url = f"http://localhost:{server.getPort()}/no-such-file.grib2"
    with pytest.raises(NotPublished, match="HTTP 404"):
        my_context.getBean(Fetchers).ecmwf.download_many([Download(url, [(0, 9)], str(tmp_path / "x"))])

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


def _post(context, path, body):
    """POST a JSON body to the running server; the java.net.http response."""
    import java

    from micronaut.runtime.server import EmbeddedServer

    server = context.getBean(EmbeddedServer)
    if not server.isRunning():
        server.start()
    HttpRequest = java.type("java.net.http.HttpRequest")
    request = (HttpRequest.newBuilder(java.type("java.net.URI").create(f"http://localhost:{server.getPort()}{path}"))
               .header("Content-Type", "application/json")
               .header("Accept", "application/json, text/event-stream")
               .POST(HttpRequest.BodyPublishers.ofString(body)).build())
    return java.type("java.net.http.HttpClient").newHttpClient().send(
        request, java.type("java.net.http.HttpResponse").BodyHandlers.ofString())


def _mcp(context, method, params):
    """One JSON-RPC request to /mcp; its result."""
    import json

    response = _post(context, "/mcp", json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}))
    assert response.statusCode() == 200, response.body()
    return json.loads(str(response.body()))["result"]


def test_mcp_path_without_key(my_context):
    """No key in the test config: MCP answers at /mcp (through the /mcp{/key} route) and nothing below it."""
    body = '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
    response = _post(my_context, "/mcp", body)
    assert response.statusCode() == 200, response.body()
    assert "forecast" in response.body()
    assert "weather_query_help" in response.body()
    assert _post(my_context, "/mcp/" + "k" * 40, body).statusCode() == 404


def test_bad_access_key_stops_startup():
    """McpKeyFilter is created at startup, so a bad key fails the context without any request."""
    import java

    ApplicationContext = java.type("io.micronaut.context.ApplicationContext")
    properties = java.type("java.util.HashMap")()
    properties.put("weather.access-key", "too-short")
    with pytest.raises(BaseException, match="access key must be"):
        ApplicationContext.builder().environments("test").properties(properties).start().close()


def _start(**settings):
    import java

    ApplicationContext = java.type("io.micronaut.context.ApplicationContext")
    properties = java.type("java.util.HashMap")()
    for key, value in settings.items():
        properties.put(key, value)
    ApplicationContext.builder().environments("test").properties(properties).start().close()


def test_prefetch_without_budget_stops_startup():
    with pytest.raises(BaseException, match="keep-free-mb .* must be below"):
        _start(**{"weather.download.max-mb-per-hour": "8000"})
    _start(**{"weather.download.max-mb-per-hour": "8000", "weather.prefetch.enabled": "false"})


def test_settings_from_environment(tmp_path):
    """Deployments set `weather.*` through environment variables; lists come as comma-separated values."""
    import java

    from at.yawk.weatheragent import McpAccessConfig
    from weather_agent.config import EcmwfConfig, GeocoderConfig, PrefetchConfig, WeatherConfig
    from weather_agent.forecast_service import ForecastService

    PropertySource = java.type("io.micronaut.context.env.PropertySource")
    env = java.type("java.util.HashMap")()
    env.put("WEATHER_ALLOWED_HOSTS", "localhost,weather.test")
    env.put("WEATHER_CACHE_DIR", str(tmp_path))
    env.put("WEATHER_ECMWF_MIRRORS", "https://mirror.test, https://other.test,")
    env.put("WEATHER_PREFETCH_FULL_MODELS", "icon-d2-eps")
    env.put("WEATHER_GEOCODER_TIMEOUT_SECONDS", "2.5")
    source = PropertySource.of("test-env", env, PropertySource.PropertyConvention.ENVIRONMENT_VARIABLE,
                               PropertySource.Origin.of("test env"))
    context = java.type("io.micronaut.context.ApplicationContext").builder().environments("test") \
        .propertySources(source).start()
    try:
        assert list(context.getBean(McpAccessConfig).allowedHosts()) == ["localhost", "weather.test"]
        assert context.getBean(WeatherConfig).cache_dir == str(tmp_path)
        assert len(context.getBean(EcmwfConfig).mirrors) == 3  # split as is; ForecastService cleans up
        ecmwf = next(s for s in context.getBean(ForecastService).forecaster.sources if s.name.startswith("ecmwf"))
        assert ecmwf.hosts.bases == ["https://data.ecmwf.int/forecasts", "https://mirror.test", "https://other.test"]
        assert list(context.getBean(PrefetchConfig).full_models) == ["icon-d2-eps"]
        assert list(context.getBean(PrefetchConfig).models) == ["ecmwf-ens", "ecmwf-aifs-ens", "icon-d2-eps",
                                                                "icon-eu-eps"]
        assert context.getBean(GeocoderConfig).timeout_seconds == 2.5
    finally:
        context.close()


def test_settings_defaults(my_context):
    from at.yawk.weatheragent import McpAccessConfig

    access = my_context.getBean(McpAccessConfig)
    assert list(access.allowedHosts()) == ["localhost", "127.0.0.1", "::1"]
    assert access.accessKey() is None and access.accessKeyFile() is None


def test_chart_tool_links_its_view(my_context):
    tools = {t["name"]: t for t in _mcp(my_context, "tools/list", {})["tools"]}
    uri = tools["show_forecast"]["_meta"]["ui"]["resourceUri"]
    assert uri == "ui://weather-agent/forecast.html"
    assert tools["show_forecast"]["inputSchema"]["required"] == ["query"]
    assert "forecast" in tools  # annotated tools are still there

    contents = _mcp(my_context, "resources/read", {"uri": uri})["contents"][0]
    assert contents["mimeType"] == "text/html;profile=mcp-app"
    assert "ui/initialize" in contents["text"]


def test_chart_tool_returns_answer_and_chart(my_context):
    import json

    from weather_agent.forecast_service import ForecastService

    forecaster = my_context.getBean(ForecastService).forecaster
    calls = []

    def visualize(query, gpx=None):
        calls.append((query, gpx))
        return {"result": 0.5, "models": []}, {"fields": [{"name": None, "dims": [], "data": [0.5]}]}

    forecaster.visualize = visualize
    try:
        result = _mcp(my_context, "tools/call", {"name": "show_forecast",
                                                 "arguments": {"query": "q", "title": "  Rain \n risk "}})
        assert not result["isError"]
        assert json.loads(result["content"][0]["text"])["result"] == 0.5
        assert result["structuredContent"]["fields"][0]["data"] == [0.5]
        assert result["structuredContent"]["title"] == "Rain risk"
        assert result["structuredContent"]["query"] == "q"
        assert calls == [("q", None)]

        def failing(query, gpx=None):
            raise ValueError("select a time range")

        forecaster.visualize = failing
        error = _mcp(my_context, "tools/call", {"name": "show_forecast", "arguments": {"query": "q"}})
        assert error["isError"] and error["content"][0]["text"] == "select a time range"
    finally:
        del forecaster.visualize
