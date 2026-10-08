"""Tools with an interactive view in the client (MCP Apps); the protocol side is at.yawk.weatheragent.McpApps."""

import json

from at.yawk.weatheragent import McpApps, McpAppTool
from jakarta.inject import Singleton
from micronaut.mcp.annotations import Resource
from weather_core.expr.redact import Redacted

from .forecast_service import ForecastService
from .tools import _forecast_summary, _run

VIEW_URI = "ui://weather-agent/forecast.html"

DESCRIPTION = (
    "Show the user a forecast query's answer as an interactive chart, next to the answer you get. The query is the "
    "same as for the forecast tool; the dimensions left in the answer pick the chart: time (or hour, distance "
    "bins) gives a graph, lat × lon (an area from .sel(lat=slice(…), lon=slice(…))) a map, over time a map with "
    "a time slider, point (places) markers on a map or one line per place. model becomes colours, member thin "
    "lines (one per member), quantile bands. Keep what the user should see: e.g. "
    "forecast().interp(lat=50.94, lon=6.96).sel(time=slice('2026-10-10', '2026-10-13')).t2m for every member "
    "of every model (a meteogram), or (….precip.sum('time') > 10).mean('member') over an area for a map of the "
    "chance of heavy rain. Maps can't show every member: reduce member first. Up to 400,000 values; the "
    "table you get is replaced by a per-value summary when it would be longer than 500 rows. Read "
    "weather_query_help first.")

SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The query; see weather_query_help"},
        "title": {"type": "string", "description": "Chart title for the user, e.g. 'Rain risk around Cologne, "
                                                   "Saturday'"},
        "gpx": {"type": "string", "description": "Optional GPX document; the query refers to it as route(gpx, …)"},
    },
    "required": ["query"],
})

MAX_TITLE = 120


@Singleton
class ForecastView(McpAppTool):
    def __init__(self, service: ForecastService):
        self.forecaster = service.forecaster

    def name(self) -> str:
        return "show_forecast"

    def title(self) -> str:
        return "Forecast chart"

    def description(self) -> str:
        return DESCRIPTION

    def inputSchema(self) -> str:
        return SCHEMA

    def viewUri(self) -> str:
        return VIEW_URI

    def call(self, arguments: str) -> str:
        try:
            args = json.loads(arguments)
        except ValueError:
            args = None
        query, title, gpx = (args.get(k) for k in ("query", "title", "gpx")) if isinstance(args, dict) \
            else (None, None, None)
        valid = isinstance(query, str) and isinstance(title, (str, type(None))) and isinstance(gpx, (str, type(None)))
        redacted = Redacted(query, keep=(s.name for s in self.forecaster.sources)) if valid else None
        call = f"show_forecast {redacted}" + (f" (gpx: {len(gpx)} characters)" if gpx is not None else "") \
            if valid else "show_forecast (invalid arguments)"

        def run() -> dict:
            if not valid:
                raise ValueError("query, title and gpx must be strings")
            answer, chart = self.forecaster.visualize(query, gpx=gpx)
            chart["title"] = " ".join((title or "").split())[:MAX_TITLE]
            chart["query"] = query
            answer = {"display": "The user sees this answer as an interactive chart.", **answer}
            return {"text": json.dumps(answer, ensure_ascii=False), "structuredContent": chart}
        return _run(run, call, redacted, lambda r: _forecast_summary(json.loads(r["text"])))

    @Resource(uri="ui://weather-agent/forecast.html", name="forecast-chart", title="Forecast chart",
              mimeType="text/html;profile=mcp-app")
    def view(self) -> str:
        return McpApps.view("mcp-apps/forecast.html")
