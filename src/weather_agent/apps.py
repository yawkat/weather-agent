"""Tools with an interactive view in the client (MCP Apps); the protocol side is at.yawk.weatheragent.McpApps."""

import json
import math

from at.yawk.weatheragent import McpApps, McpAppTool
from jakarta.inject import Singleton
from micronaut.mcp.annotations import Resource

from .forecast_service import ForecastService
from .tools import _run, _sources

METEOGRAM_URI = "ui://weather-agent/meteogram.html"

METEOGRAM_DESCRIPTION = (
    "Show the user an interactive meteogram for one point: every ensemble member's time series per model, with "
    "percentile bands, for temperature, precipitation, wind and more. Use it when the user wants to see the "
    "forecast rather than only hear about it. The chart is displayed to the user; you get a daily summary per "
    "model (member percentiles) to talk about. Models end where their run ends, or earlier when the download "
    "budget doesn't stretch further (see until/until_reason); long windows are expensive for the DWD models. Use "
    "resolve_place to get coordinates.")

METEOGRAM_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "lat": {"type": "number", "description": "Latitude"},
        "lon": {"type": "number", "description": "Longitude"},
        "label": {"type": "string", "description": "Place name for the chart title, e.g. 'Bonn'"},
        "start": {"type": "string", "description": "Start, ISO 8601 local time (Europe/Berlin unless an offset "
                                                   "is given), e.g. 2026-10-10T06:00; default: now"},
        "end": {"type": "string", "description": "End, ISO 8601 local time; default: start + 3 days"},
        "variables": {"type": "string", "description": "Comma-separated, at most 6 (default: t2m,precip,wind,"
                                                       "gust,cloud); see weather_query_help for the list"},
        "sources": {"type": "string", "description": "Comma-separated model names to restrict to (default: all)"},
    },
    "required": ["lat", "lon"],
})


@Singleton
class MeteogramApp(McpAppTool):
    def __init__(self, service: ForecastService):
        self.forecaster = service.forecaster

    def name(self) -> str:
        return "show_meteogram"

    def title(self) -> str:
        return "Meteogram"

    def description(self) -> str:
        return METEOGRAM_DESCRIPTION

    def inputSchema(self) -> str:
        return METEOGRAM_SCHEMA

    def viewUri(self) -> str:
        return METEOGRAM_URI

    def call(self, arguments: str) -> str:
        def run() -> dict:
            args = json.loads(arguments)
            summary, chart = self.forecaster.meteogram(
                _number(args, "lat"), _number(args, "lon"), start=_text(args, "start"), end=_text(args, "end"),
                variables=_text(args, "variables"), sources=_sources(_text(args, "sources")),
                label=_text(args, "label"))
            summary = {"display": "The user sees an interactive chart of every member; this is its summary.",
                       **summary}
            return {"text": json.dumps(summary, ensure_ascii=False), "structuredContent": chart}
        return _run(run)

    @Resource(uri="ui://weather-agent/meteogram.html", name="meteogram", title="Meteogram",
              mimeType="text/html;profile=mcp-app")
    def meteogram_view(self) -> str:
        return McpApps.view("mcp-apps/meteogram.html")


def _number(args: dict, name: str) -> float:
    value = args.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _text(args: dict, name: str) -> str | None:
    value = args.get(name)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value or None
