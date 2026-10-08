# Phase 0 spike notes (2026-10-07)

## Toolchain (spike 0.1): works

- Dev shell: nixpkgs `graalvm-ce` 25.4.4.1.1 (JDK 25.0.4) + `uv`.
  - nixpkgs `graalvm-oracle` is JDK 25.0.3. Pyronaut's GraalPy runtime (polyglot 25.4.4.1.1) rejects it and
    falls back to interpreter-only mode, which is slow. Use CE until an Oracle 25.0.4 build is packaged.
- uv-managed GraalPy (`graalpy-3.13` = 25.4.4) runs under nix-ld, but only with `LD_LIBRARY_PATH` set:
  GraalPy `dlopen()`s `libpythonvm.so`.
- `uv venv --seed -p graalpy-3.13 && uv sync` installs numpy 2.4.4 from the GraalPy wheel index.
  `pyronaut install` accepts the existing `.venv`.
- numpy inside the app needs `graalpy.context.allow-native-access = true`.
- Java packages beyond JDK/Jakarta/Micronaut must be allowed via `graalpy.context.host-class-lookup`.
- `pyronaut test` (pytest engine) works; about 19 s per run, including startup.
- micronaut-mcp HTTP transport, Micronaut Data JDBC (Postgres) and Flyway all work from Python.

## Pyronaut quirks found (candidates for upstream issues)

1. **Stale generated files after incremental processing.** After editing a source, the next `pyronaut run`
   sometimes fails with `VFS.initEntries: could not find resource
   .../__micronaut_java_imports_<hash>.py`. Workaround: `pyronaut clean && pyronaut process`.
2. **Flyway can't find classpath migrations in the fat JAR** ("Schema has version 1, but no migration could
   be resolved"). The SQL file is in the JAR (`PYRONAUT-INF/app/resources/...`). Workaround:
   `WEATHER_MIGRATIONS=filesystem:/path/to/db/migration`. (Moot since the database was removed.)
3. **JVM crash** (`Py_REFCNT` in `libpython-native.so`) when calling `np.frombuffer(memoryview(java_bytebuffer))`.
   Real GraalPy bug. Workaround: hand data over via a file (Java writes, Python reads bytes).
4. `np.fromfile` / `np.memmap` fail: Pyronaut uses GraalPy's `java` posix backend (no real file
   descriptors). Use `np.frombuffer(f.read())`.
5. `np.asarray(java_float_array)` hits `RecursionError`.
6. A custom `__init__.py` in the source tree is rejected; Pyronaut generates package `__init__.py` files.
7. Repository finder names must use the Python (snake_case) property names; `findByDisplayName` fails.
8. The processor generates Java for every annotated Python class under `src/`, and fails on e.g. methods
   returning a Python exception type. Pure-Python code therefore lives in `packages/weather_core`, a uv
   workspace member installed editable into the venv.
9. `graalpy.context.allow-native-access` is not applied to the pytest collection context. A
   `GraalPyContextCustomizer` registered via `META-INF/services` (see `src-java/`) works everywhere.
10. Python code can't start threads ("Creating threads is not allowed"). Parallel downloads use the Micronaut
    HTTP client's async API from Python (`src/weather_agent/http_fetcher.py`): start a batch, then await each
    future. No Python callbacks run on Netty threads.
11. `os.replace`/`os.rename` fail with "Atomic move not supported" (java posix backend);
    `weather_core.store.atomic_replace` falls back to `java.nio.file.Files.move(ATOMIC_MOVE)`.
12. `zoneinfo` finds no system tz database; the `tzdata` package fixes it.
13. Java exceptions raised into Python are not caught by `except Exception`, only by `except BaseException`.
14. `@Tool`/`@ToolArg` arguments must be literals; string concatenation is rejected.
15. Not Pyronaut: the JDK truststore lacks the HARICA root that signs data.ecmwf.int, so the HTTP client trusts
    the system bundle (`micronaut.certificate.file.system-ca` + `ssl.trust-name`). Micronaut's `PemParser`
    rejects the label lines between certificates in distro bundles, so the dev shell writes a cleaned copy.
16. `java.type("byte[]")` is rejected by the host-class filter (array names don't match package prefixes, and
    listing `byte[]` doesn't help); use the default `ByteBuffer` body and `toByteArray()`.
17. Pyronaut maps Python `datetime` to naive Java types and rejects aware values ("Aware datetime.datetime values
    cannot be converted to a naive Java type"); timestamps are stored as naive UTC.
18. Processing never finishes (100 % CPU in `IncrementalCompilation.matchesAnyType`) when a Java source declares an
    anonymous class: its empty type name loops forever. Avoid anonymous classes in `src-java/` (e.g. no
    `new TypeRef<>() {}`). https://github.com/micronaut-projects/micronaut-core/issues/13799
19. MCP Apps tools (`McpApps.java`): micronaut-mcp's `@Tool` can't set the `_meta` that links a tool to its view.
    Python beans implement the Java interface `McpAppTool`, and a listener adds them to the server once it exists.
    Declaring SDK specification beans directly failed once in processing (`NoClassDefFoundError:
    SerdeConfig$SerIgnored` from the serde processor); it didn't reproduce from a clean state, so the cause is open.

## GRIB decoding (spike 0.2): works, via netCDF-Java `edu.ucar:grib` 5.11.0

- All products use CCSDS packing (template 5.42). It needs native `libaec`, found through JNA
  (`LD_LIBRARY_PATH` in dev, `-Djna.library.path` in production).
- Results match ecCodes exactly, except masked points: NaN in Java, 9999 fill in ecCodes.
- Decode time per field: ICON-EU-EPS ~5 ms, ECMWF global 0.25° ~15 ms, RUC-EPS ~120 ms (first call, JIT warm-up).
  Hand-off to numpy adds ~15–35 ms.

## ICON grids (spike 0.3): no grid files needed

- RUC-EPS / D2 use grid 47 and EU-EPS uses grid 63 (GRIB template 3.101, unstructured).
- DWD publishes `clat`/`clon` (and `hsurf`, `fr_land`) as time-invariant GRIB fields per product:
  - old layout: `icon-eu-eps/grib/00/clat/`
  - v1 layout: `nwp/v1/m/icon-d2-ruc-eps/p/CLAT/`
- Decode them with the same code path. `hsurf` gives model orography for elevation adjustment.

## Packaging (spike 0.4): done, pure Nix build (`nix/`)

- The fat JAR (~207 MB) runs on plain Nix `graalvm-ce` `java -jar` with a clean environment.
- **No GraalPy interpreter is needed at runtime.** `VIRTUAL_ENV` only needs:
  - `lib/python3.13/site-packages/`
  - `pyvenv.cfg`
  - any executable `bin/python`; it's never run, but GraalPy uses it to find the venv
- Python packages outside `src/` (`weather_core`, numpy, tzdata) aren't in the JAR; they come from the venv.
- Native wheels need their ELF RPATHs patched (`libstdc++.so.6`). In Nix, use `autoPatchelfHook`.
- Pyronaut's build tools (`pyronaut-processor`, `pyronaut-jar-build`, …) are JVM shell scripts, not native
  binaries. Only the `pyronaut-dev`/`pyronaut-run*` launchers are native. `pyronaut setup` still downloads them
  (~1.6 GB) to read their classpath descriptors, but the JAR build never runs them.
- **Build split:** a fixed-output derivation (`nix/deps.nix`) runs `pyronaut setup` and `pyronaut install` with
  network access and keeps `~/.pyronaut` and `~/.m2`. The JAR (`nix/jar.nix`) is then built offline with
  `pyronaut install --offline` and `pyronaut build --jar --offline`, so code changes need no hash update. The JAR
  is reproducible as is: every entry is dated 1980-01-01.
  - Pyronaut's Java tools take `~/.pyronaut` and `~/.m2` from `user.home`, which comes from passwd, not `$HOME`.
    Set `JAVA_TOOL_OPTIONS=-Duser.home=$HOME`, or they write to the real home (or fail on `/homeless-shelter`).
  - Setup accepts a GraalPy already in `~/.pyronaut/sdks/graalpy3.13-25.4.4/`. The upstream tarball needs
    `autoPatchelfHook` to run in the sandbox (`nix/graalpy.nix`); nixpkgs' `graalpy` is older (25.2).
  - Setup fetches the native launchers through the GitHub API by default, whose unauthenticated rate limit (60/h)
    fails builds. `[native-images] base-url = "https://github.com/micronaut-projects/pyronaut/releases/download/v0.1.0/"`
    in `~/.pyronaut/settings.toml` downloads them directly.
  - `pyronaut install` creates or uses `.venv` and runs `pip install` for the declared requirements, which fails
    (no pip, `weather-core` isn't on PyPI, no network). It skips pip when `.venv/.pyronaut-requirements.json`
    lists the same requirements. The build creates `.venv` with GraalPy, copies in the Nix venv's packages and
    writes that marker with Pyronaut's own functions.
  - Making the downloads reproducible (`nix/normalize-deps.py`): drop lock files, resolver bookkeeping, the
    `#<date>` lines of `.properties` files and the editor stubs (`~/.pyronaut/ide-stubs`, whose overload
    parameter names vary between runs); rename the tools directory (`<hash>-<random UUID>`); copy the JARs Pyronaut
    symlinks from the CLI's store path; replace `$HOME` and the JDK path with placeholders. The ~1.6 GB of native
    launchers are only checked for existence and their classpath descriptors, so they become stubs.
- **Runtime:** GraalPy and Truffle extract native libraries (`libpython-native.so`, `libtruffleattach.so`) to
  `~/.cache/org.graalvm.polyglot` and `dlopen` them, so the service's home must be writable and not `noexec`. With
  systemd `DynamicUser=`, the `StateDirectory` is an idmapped `noexec` mount; that needs `ExecPaths=` for it (seen
  in the VM test), so a plain system user is simpler.
- **Runtime venv** (`nix/venv.nix`): built from `uv.lock` by unpacking wheels, without running Python. The GraalPy
  wheel index publishes no hashes, so their hashes are pinned in `nix/venv.nix`.

## DuckDB as the query engine (spike, 2026-10-07): works; removed 2026-10-08

Replaced by our own query language (`docs/query-language.md`): reloading every query's data into a fresh database,
SQL's limits for per-member questions, and an engine not built for untrusted input. Notes kept for reference.

`org.duckdb:duckdb_jdbc:1.5.6.0` (MIT), used from Java (`src-java/at/yawk/weatheragent/SqlCube.java`) and called
from Python. Tests: `tests/test_duckdb_spike.py`.

- **Native library:** the JDBC jar (85 MB, bundles `libduckdb_java.so` for linux_amd64/arm64) loads under
  Pyronaut on NixOS with the dev shell's library path. It's extracted to a temp file and removed afterwards;
  nothing is written to `~/.duckdb`. The Nix service needs `libstdc++` on its library path.
- **Loading:** Python writes int32/float32 columns with numpy, and Java bulk-appends them through
  `DuckDBAppender`. Measured (2 models × 50 members, 6 variables):

  | Shape | Rows | Load | Example queries |
  |---|---|---|---|
  | point, 30 times | 3,000 | 0.01 s | 0.03 s |
  | route, 2,000 samples | 200,000 | 0.16 s | 0.07 s |
  | area, 30 times × 1,000 points | 3,000,000 | 2.3 s | 0.31 s |

- **Schema in the spike:**
  - `models`, `times`, `points` dimension tables plus a `raw` member table
  - a `samples` view joining them, and a `per_member` view of common aggregates
  - macros `prob(cond)` and `share(group_col)`
- **Macro quirks:**
  - a macro body can't refer to columns that aren't parameters, so it's `share(model)`, not `share()`
  - `GROUP BY ALL` misclassifies the window-over-aggregate in `share()`, so group explicitly
- **Lockdown after loading:** `enable_external_access=false`, extension auto-install/auto-load off, community
  extensions off, `lock_configuration=true`, plus `memory_limit`, `threads` and `temp_directory=''` (no spilling).
  - Verified blocked, all with "Permission Error … file system operations are disabled" or "configuration has
    been locked":
    - reading files: `read_csv`, `read_text`, `read_blob`, `glob`
    - writing: `COPY … TO`
    - remote access: `read_parquet('https://…')`
    - extensions and databases: `INSTALL`/`LOAD httpfs`, `ATTACH`
    - unlocking: `SET enable_external_access`, `SET lock_configuration`
  - `getenv` doesn't exist in this build. `duckdb_extensions()` is blocked too (it reads a directory), which is
    harmless.
- **Limits:**
  - a timeout via `Statement.cancel()` from a scheduler stopped a long query after exactly 2 s
  - a 256 MB memory limit failed an oversized query with "Out of Memory"
  - the cube stayed usable afterwards in both cases
- **Error messages:** the JDBC driver prefixes most errors with "Invalid Input Error: Attempting to execute an
  unsuccessful or closed pending query result". The useful part (e.g. "Binder Error: … LINE 1: …") follows on
  the next lines, so strip that prefix before showing errors to the LLM.

## Upstream issues filed (2026-10-07)

- GraalPy crash on `np.frombuffer(memoryview(ByteBuffer))`: https://github.com/oracle/graalpython/issues/1198
- Micronaut `PemParser` rejects distro CA bundles: https://github.com/micronaut-projects/micronaut-core/issues/13782
- Micronaut HTTP/2 client advertises no flow-control window, so downloads from opendata.dwd.de (h2) run at
  ~1.5 MB/s; the DWD client is pinned to HTTP/1.1: https://github.com/micronaut-projects/micronaut-core/issues/13798
- Pyronaut:
  - Flyway classpath migrations in the fat JAR: https://github.com/micronaut-projects/pyronaut/issues/329
  - stale `__micronaut_java_imports_*.py` after incremental processing: https://github.com/micronaut-projects/pyronaut/issues/330
  - processor generates Java for plain helper classes: https://github.com/micronaut-projects/pyronaut/issues/331
  - `graalpy.context` settings not applied to pytest collection: https://github.com/micronaut-projects/pyronaut/issues/332
  - `java` posix backend (rename, `np.fromfile`, `zoneinfo`): https://github.com/micronaut-projects/pyronaut/issues/333
  - aware datetimes can't map to Java: https://github.com/micronaut-projects/pyronaut/issues/334
  - Java exceptions not caught by `except Exception`: https://github.com/micronaut-projects/pyronaut/issues/335
  - `host-class-lookup` can't allow `byte[]`: https://github.com/micronaut-projects/pyronaut/issues/336
  - validate-config rejects `'6MB'` for `@ReadableBytes`: https://github.com/micronaut-projects/pyronaut/issues/337
- Pyronaut, from the Nix packaging (2026-10-07):
  - dependency lock file with checksums: https://github.com/micronaut-projects/pyronaut/issues/340
  - setup downloads native launchers for JVM-only projects: https://github.com/micronaut-projects/pyronaut/issues/341
  - `install` can't use a uv-managed venv: https://github.com/micronaut-projects/pyronaut/issues/342
  - Java tools use `user.home` instead of `$HOME`: https://github.com/micronaut-projects/pyronaut/issues/343
  - setup/install output isn't reproducible: https://github.com/micronaut-projects/pyronaut/issues/344
  - setup state hardcodes absolute paths and symlinks: https://github.com/micronaut-projects/pyronaut/issues/345
  - configurable GraalPy executable: https://github.com/micronaut-projects/pyronaut/issues/346
  - GitHub API rate limit during setup: https://github.com/micronaut-projects/pyronaut/issues/347
  - bundle the venv into the fat JAR: https://github.com/micronaut-projects/pyronaut/issues/348
  - `--progress off` doesn't silence the nested installer: https://github.com/micronaut-projects/pyronaut/issues/349
