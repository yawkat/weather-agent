# weather-agent

MCP server answering weather questions from open ensemble forecasts (ECMWF IFS/AIFS ENS; DWD ICON-EU-EPS, ICON-D2-EPS, ICON-D2-RUC-EPS).
Agents query per-member samples through one `forecast` tool, in a subset of xarray that the server parses and
interprets itself (`weather_core/expr/`, documented in `docs/query-language.md`).

## Layout

- `packages/weather_core/`: pure-Python engine (sources, sampling, geometry, query language). It's a uv workspace
  member installed editable into the venv, because the Pyronaut processor would otherwise generate Java for every
  class (pyronaut#331).
- `src/weather_agent/`: Micronaut/MCP glue (tools, fetcher on Micronaut's HTTP client). `apps.py`: `show_forecast`,
  which also shows the answer as an interactive chart (MCP Apps; view in `config/mcp-apps/forecast.html`, protocol
  side in `src-java/…/McpApps.java`).
- `src-java/at/yawk/weatheragent/`: Java helpers where performance or interop needs them (GRIB decoding,
  Host/Origin and secret-path filters, GraalPy native access).
- `nix/`: pure Nix build (venv from `uv.lock`, downloads as a fixed-output derivation, offline JAR build,
  `weather-agent` wrapper) and a VM
  test (`checks.x86_64-linux.vm`). Hosts (goliath) define the systemd unit themselves; `nix/test.nix` shows one.
- `docs/spike-notes.md`: Pyronaut/GraalPy quirks and workarounds. Read it before changing framework-facing code.
- `docs/security.md`: input limits and sandboxing.

## Working on it

- Run everything inside `nix develop`. Tests: `pyronaut test` (pytest on GraalPy). On
  `VFS.initEntries: could not find resource`, run `pyronaut clean` (pyronaut#330).
- After changing `pyproject.toml` (or bumping Pyronaut/GraalVM in Nix), `nix build .#deps` fails with a hash
  mismatch: copy the `got:` hash into `nix/deps.sha256`. Code changes need no hash update. New wheels from the
  GraalPy index need a hash in `nix/venv.nix`.
- Never run several `nix build` / `nix flake check` commands concurrently.
- Don't run `pyronaut` in this directory while a `pyronaut dev` is running here: they share `__pyronaut__/`.
  Use a copy of the project instead.
- Python code can't start threads. Java exceptions need `except BaseException` (pyronaut#335).
- Prefer Python over Java unless performance or interop requires Java.
- Bulky local data lives under `var/` (excluded from backups via `var/.nobackup`).

## Design rules

- The server never weights or pools models on its own: what it returns is per model unless the query asks
  otherwise. Agents may combine models and members explicitly (worst model, agreement, chosen weights); they
  decide which models to trust. Make per-model answers the easy path, and document the pitfalls of pooling
  (member counts differ, so pooling silently weights models).
- Download lazily, only the fields a query needs, within the download budget. Keep only the latest runs.
- Every output carries attribution down to model and run (DWD: "Datenbasis: Deutscher Wetterdienst, eigene
  Bearbeitung"; ECMWF: see `sources/ecmwf.py`).
- Tool inputs come from LLMs or unauthenticated clients: bound every size before doing work proportional to it.
