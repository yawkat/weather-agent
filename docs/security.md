# Security notes

The MCP endpoint has no real authentication yet (GitHub OAuth comes later), only an optional secret path. These
measures limit what a caller (anyone who has the URL, or an LLM steered by untrusted content) can do.

## Secret path

With an access key configured (`WEATHER_ACCESS_KEY_FILE`, or `WEATHER_ACCESS_KEY` for development), MCP is served
only at `/mcp/<key>`; `/mcp` and wrong keys get a 404 (`McpKeyFilter`). Without a key, MCP is at `/mcp` and the
server logs a warning. A configured key file that is empty or holds an invalid key stops startup. Set a key on
any instance reachable from the internet.

- Generate it with `openssl rand -base64 32 | tr '+/' '-_' | tr -d '='` (32–256 base64url characters are
  accepted). Pass the file as a systemd credential (`LoadCredential=`, `WEATHER_ACCESS_KEY_FILE=%d/…`), not via
  `Environment=`, which any local user can read with `systemctl show`.
- The route is the template `/mcp{/key}`, so routing never compares the key; the filter compares it in constant
  time.
- This is a bearer secret in the URL, chosen because claude.ai connectors accept a URL but no fixed token, and it
  works for MCP App views, which the client fetches over the same connection. The URL ends up in client
  configuration and in any access log on the way: turn off or redact path logging in the reverse proxy. Rotate the
  key by replacing the file and restarting.
- It identifies no one: everyone with the URL shares the hourly download budget.

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
| Downloads | `weather.download.max-mb-per-query` (4,000, all models of a query together) and `…-per-hour` (20,000), checked against estimates before downloading and settled to actual sizes | `budget.py`, `forecast.py` |

Route samples read only their own grid point, so memory is linear in route length.

The sandbox is covered by `tests/test_sql_cube.py` (file reads, globbing, `COPY TO`, URLs, `INSTALL`/`LOAD`,
`ATTACH`, unlocking the configuration, timeouts, memory limit).

## Errors

Unexpected errors are logged with a traceback; clients only get `internal error (reference <id>)`. Source
failures are reported by exception type only.

## Known gaps

- No per-user authentication or rate limit yet (the secret path is one shared key); the hourly download budget
  is global.
- Concurrent queries aren't capped. Each DuckDB query may take 1 GB of native memory, outside the JVM heap limit.
- `uv.lock` has no hashes for wheels from the GraalPy index (it doesn't publish them); the Nix venv pins them in
  `nix/venv.nix`. Maven dependencies have no checksum lock; the Nix build pins the hash of all downloads together
  (`nix/deps.sha256`).
