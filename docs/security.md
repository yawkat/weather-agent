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
| SQL query | 8,000 characters; one `SELECT`; its own DuckDB in a sandboxed worker process (below): 1 GB DuckDB memory, 2 threads, 15 s timeout, 500 result rows and 1M characters of result; ≤ 2 queries at once | `forecast.py`, `sql_engine.py`, `sql_worker.py` |
| Query table size | ≤ 4M rows (members × windows × samples × points), estimated before downloading; areas ≤ 500 points | `forecast.py` |
| Polyline | 100k characters, ≤ 7 characters per encoded value, polyline alphabet only | `geometry.py` |
| GPX | 5M characters, no DOCTYPE/entities | `geometry.py` |
| Route | 20k input points, 2,000 samples (spacing grows), speed ≤ 200 km/h | `geometry.py` |
| Area | radius ≤ 1,000 km or bbox span ≤ 30°, ≤ 500 points at 10 km spacing (spacing chosen before allocating) | `geometry.py`, `forecast.py` |
| Place names | ≤ 10 per call, ≤ 200 characters each; Nominatim requests serialised and ≥ 1 s apart, results (hits and misses) cached in memory, LRU of 10,000 names | `geocode.py` |
| Downloads | `weather.download.max-mb-per-query` (4,000, all models of a query together) and `…-per-hour` (20,000), checked against estimates before downloading and settled to actual sizes | `budget.py`, `forecast.py` |

Route samples read only their own grid point, so memory is linear in route length.

## SQL sandbox

Client SQL never runs in the server process. Each query gets a worker (`sql-worker/sql_worker.py`, CPython with
duckdb and numpy from Nix) that loads the cube from `.npy` files, runs the query and prints JSON. Layers, outside in:

- **Server** (`sql_engine.py`, `WorkerProcess.java`): at most `weather.query.max-concurrent` (2) workers at once;
  kills a worker 15 s after its query timeout. A crash or memory blowup takes down only the worker.
- **prlimit**: output file size (4 × the result character limit; more is SIGXFSZ), 64 open files, no core dumps,
  CPU time as a backstop.
- **bwrap**: new user (no further user namespaces), PID, network (loopback only), IPC, UTS and cgroup namespaces;
  empty environment; read-only binds of only the worker's store closure (`closureInfo`), the script and the
  request directory; fresh `/proc`, only `/dev/null` and `/dev/urandom`; read-only root, so nothing is writable;
  dies with the server.
- **Worker**: caps its own address space (`RLIMIT_AS`) at its current size plus DuckDB's memory limit, three times
  the cube's data and 1 GB, before any client SQL runs. DuckDB's `memory_limit` alone doesn't cover large lists
  and strings: `SELECT len(range(0, 100000000))` used more than 8 GB with a 1 GB limit. It sets
  `oom_score_adj=1000`, so if memory runs out anyway the kernel kills the worker, not the server.
- **DuckDB**: after loading, external access (files, network) off, extension install/load off, secret directory
  cleared, configuration locked. Only a single `SELECT` statement is accepted.

The limit is relative because importing numpy maps a 3 GB OpenBLAS buffer pool that is never touched.
`MALLOC_ARENA_MAX=2` keeps glibc from reserving 64 MB per thread against it; without that, DuckDB segfaulted
on its first join at 512 MB. Allocation failures sometimes crash DuckDB instead of raising an error. That
stays inside the worker, and the client gets "query crashed the database (most likely out of memory)".

The service unit must allow bwrap's namespaces: no `RestrictNamespaces=`, `PrivateUsers=` or
`SystemCallFilter=@system-service` (bwrap mounts), and `RestrictAddressFamilies=` needs `AF_NETLINK`.
`ProtectKernelTunables=`, `ProtectKernelLogs=` and `ProtectHostname=` cover parts of `/proc`, after which the
kernel refuses a fresh `/proc` mount. Use `OOMPolicy=continue`, so an OOM-killed worker doesn't stop the service.
`nix/test.nix` has a unit that works and checks the sandbox under it. A `MemoryMax=` on the unit also bounds
the workers, which run in its cgroup.

Covered by `tests/test_sql_engine.py` (file reads, globbing, `COPY TO`, URLs, `INSTALL`/`LOAD`, `ATTACH`,
`SET`, several statements, settings leaks, timeouts, memory blowups, result size) and the VM test (no host files,
nothing writable, no network, under the service's settings).

## Errors

Unexpected errors are logged with a traceback; clients only get `internal error (reference <id>)`. Source
failures are reported by exception type only.

## Known gaps

- No authentication or per-client rate limit yet; the hourly download budget is global.
- The SQL worker has no seccomp filter; bwrap and the namespaces are the boundary. A DuckDB memory-safety bug
  would give code execution inside the sandbox, without host files or network.
- `uv.lock` has no hashes for wheels from the GraalPy index (it doesn't publish them); the Nix venv pins them in
  `nix/venv.nix`. Maven dependencies have no checksum lock; the Nix build pins the hash of all downloads together
  (`nix/deps.sha256`).
