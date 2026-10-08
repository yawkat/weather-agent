# Security notes

The MCP endpoint has no real authentication yet (GitHub OAuth comes later), only an optional secret path. These
measures limit what a caller (anyone who has the URL, or an LLM steered by untrusted content) can do.

## Secret path

With an access key configured (`WEATHER_ACCESS_KEY_FILE`, or `WEATHER_ACCESS_KEY` for development), MCP is served
only at `/mcp/<key>`; `/mcp` and wrong keys get a 404 (`McpKeyFilter`). Without a key, MCP is at `/mcp` and the
server logs a warning. Only empty settings mean "no key": a whitespace-only setting, an empty key file or an
invalid key stops startup (the filter is created eagerly). Set a key on any instance reachable from the internet.

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
| Times | 2 days back to 16 days ahead; a location's selected range up to 16 days | `forecast.py` |
| Query | 120,000 characters (polylines are long), 500 syntax nodes, nesting depth 40, 10 locations (each ≤ 500 points: interpolated points or an area grid; ≤ 20 named places; routes below); parsed with `ast` and interpreted (never `eval`), only whitelisted syntax, methods and names; data needs (models, hours, places) and evaluation memory derived from the query and checked before downloading (≤ 4M values per variable and location, ≤ 1 GB), memory then metered per operation; 15 s deadline; 500 result rows (`weather.query.memory-mb`, `…timeout-ms`, `…max-rows`) | `expr/language.py`, `expr/runtime.py`, `forecast.py` |
| Evaluations | at most 4 at once (`weather.query.max-concurrent`); others wait up to 30 s, then fail as busy | `forecast.py` |
| Polyline | 100k characters, ≤ 7 characters per encoded value, polyline alphabet only | `geometry.py` |
| GPX | 5M characters, no DOCTYPE/entities | `geometry.py` |
| Route | 20k input points, 2,000 samples (spacing grows), speed ≤ 200 km/h | `geometry.py` |
| Area | span ≤ 30° in latitude and longitude, ≤ 500 grid points at 10 km spacing (spacing chosen before allocating) | `geometry.py`, `forecast.py` |
| Place names | ≤ 10 per call, ≤ 200 characters each; Nominatim requests serialised and ≥ 1 s apart, results (hits and misses) cached in memory, LRU of 10,000 names | `geocode.py` |
| Downloads | `weather.download.max-mb-per-query` (4,000, all models of a query together) and `…-per-hour` (20,000), checked against estimates before downloading and settled to actual sizes | `budget.py`, `forecast.py` |

Route samples read only their own grid point, so memory is linear in route length.

There is no query engine to sandbox: queries are syntax trees checked against a whitelist and evaluated by our own
numpy code, which has no file, network or Python object access. Limits and rejected syntax (imports, attributes,
comprehensions, lambdas, subscripts, huge numbers, deep nesting) are covered by `tests/core/test_expr.py`.

## Errors

Unexpected errors are logged with a traceback; clients only get `internal error (reference <id>)`. Source
failures are reported by exception type only.

## Query log

Every tool call is logged at INFO with its outcome and duration, without what locates the caller
(`weather_core/expr/redact.py`): forecast queries keep their shape, times, models and thresholds, but coordinates
become `...`, place names, polylines and other unknown strings `'s1'`, and binding names `v1`. Error and warning
messages are scrubbed of the query's hidden strings, quoted strings and decimals. Place lookups and routes log
only counts and sizes. Tracebacks of unexpected errors are not redacted.

## Known gaps

- No per-user authentication or rate limit yet (the secret path is one shared key); the hourly download budget
  is global.
- Downloads of concurrent queries aren't capped beyond the download budget; only evaluations are. Each evaluation
  may take 1 GB of numpy (native) memory, outside the JVM heap limit.
- `uv.lock` has no hashes for wheels from the GraalPy index (it doesn't publish them); the Nix venv pins them in
  `nix/venv.nix`. Maven dependencies have no checksum lock; the Nix build pins the hash of all downloads together
  (`nix/deps.sha256`).
