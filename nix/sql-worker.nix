# The SQL worker that runs client queries inside bwrap (src/weather_agent/sql_engine.py): CPython with duckdb and
# numpy, and sql-worker/sql_worker.py. `closure` lists the store paths the sandbox has to bind.
{
  python3,
  closureInfo,
  runCommand,
}:
let
  python = python3.withPackages (ps: [
    ps.duckdb
    ps.numpy
  ]);
  worker = runCommand "weather-sql-worker" { passthru = { inherit python closure; }; } ''
    mkdir -p $out
    cp ${../sql-worker/sql_worker.py} $out/sql_worker.py
    ln -s ${python}/bin/python3 $out/python3
  '';
  closure = closureInfo { rootPaths = [ worker ]; };
in
worker
