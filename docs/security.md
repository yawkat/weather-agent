# Security notes

The MCP endpoint has no authentication in the prototype (GitHub OAuth comes later). These measures limit what a
caller (an unauthenticated client, or an LLM steered by untrusted content) can do.

## Network exposure

- The server binds to `127.0.0.1` by default. Set `WEATHER_BIND_HOST` to a VPN/LAN address only when needed; a
  reverse proxy in front is preferred.
- `/mcp` only accepts requests whose `Host` (and `Origin`, if present) is listed in `WEATHER_ALLOWED_HOSTS`
  (default `localhost,127.0.0.1,::1`), which blocks DNS rebinding from browsers (`McpHostFilter`). Add the
  proxy's or server's public name when deploying.
- Request bodies are capped at 6 MiB.

## Resource limits (per call)

| Input | Limit | Where |
|---|---|---|
| Times | 2 days back to 16 days ahead; windows up to 16 days | `forecast.py` |
| Windows | `window_hours`: every hourly window within start..end (≤ 16 days); places ≤ 20 | `forecast.py` |
| SQL query | 8,000 characters; locked-down DuckDB per query (no file/network/extension access, configuration frozen), 1 GB memory, 2 threads, 15 s timeout, 500 result rows | `forecast.py`, `SqlCube.java`, `sql_engine.py` |
| Query table size | ≤ 4M rows (members × windows × samples × points), estimated before downloading; areas ≤ 500 points | `forecast.py` |
| Polyline | 100k characters, ≤ 7 characters per encoded value, polyline alphabet only | `geometry.py` |
| GPX | 5M characters, no DOCTYPE/entities | `geometry.py` |
| Route | 20k input points, 2,000 samples (spacing grows), speed ≤ 200 km/h | `geometry.py` |
| Area | radius ≤ 1,000 km or bbox span ≤ 30°, ≤ 500 points at 10 km spacing (spacing chosen before allocating) | `geometry.py`, `forecast.py` |
| Place names | ≤ 10 per call, ≤ 200 characters each; Nominatim requests serialised and ≥ 1 s apart, results (hits and misses) cached in memory, LRU of 10,000 names | `geocode.py` |
| Downloads | `weather.download.max-mb-per-query` (4,000) and `…-per-hour` (20,000), checked before downloading | `budget.py` |

Route samples read only their own grid point, so memory is linear in route length.

The sandbox is covered by `tests/test_sql_cube.py` (file reads, globbing, `COPY TO`, URLs, `INSTALL`/`LOAD`,
`ATTACH`, unlocking the configuration, timeouts, memory limit).

## Errors

Unexpected errors are logged with a traceback; clients only get `internal error (reference <id>)`. Source
failures are reported by exception type only.

## Known gaps

- No authentication or per-client rate limit yet; the hourly download budget is global.
- `uv.lock` has no hashes for wheels from the GraalPy index (it doesn't publish them), and Maven dependencies have
  no checksum lock.
