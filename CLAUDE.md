# weather-agent

MCP server answering weather questions from open ensemble forecasts (ECMWF IFS/AIFS ENS; DWD ICON planned).
Agents query per-member samples with DuckDB SQL through one `forecast` tool.

## Layout

- `packages/weather_core/`: pure-Python engine (sources, sampling, geometry, cube building). It's a uv workspace
  member installed editable into the venv, because the Pyronaut processor would otherwise generate Java for every
  class (pyronaut#331).
- `src/weather_agent/`: Micronaut/MCP glue (tools, fetcher on Micronaut's HTTP client, DuckDB engine).
- `src-java/at/yawk/weatheragent/`: Java helpers where performance or interop needs them (GRIB decoding, DuckDB
  loading, Host/Origin filter, GraalPy native access).
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

- Results are per model; no cross-model weighting or pooled probabilities. The agent decides which models to
  trust.
- Download lazily, only the fields a query needs, within the download budget. Keep only the latest runs.
- Every output carries attribution down to model and run (DWD: "Datenbasis: Deutscher Wetterdienst, eigene
  Bearbeitung"; ECMWF: see `sources/ecmwf.py`).
- Tool inputs come from LLMs or unauthenticated clients: bound every size before doing work proportional to it.
