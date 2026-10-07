# `weather-agent`: the fat JAR on GraalVM CE with the GraalPy venv and the native libraries it loads.
{
  lib,
  stdenv,
  runCommand,
  makeWrapper,
  cacert,
  graalvm-ce,
  libaec,
  bubblewrap,
  util-linux,
  jar,
  sql-worker,
  venv,
  version,
}:
let
  # Micronaut's PEM parser rejects the label lines between certificates in distro bundles (micronaut-core#13782),
  # so hand it only the BEGIN/END blocks.
  caBundle = runCommand "weather-agent-ca-bundle.pem" { } ''
    sed -n '/-----BEGIN CERTIFICATE-----/,/-----END CERTIFICATE-----/p' ${cacert}/etc/ssl/certs/ca-bundle.crt > $out
  '';
in
stdenv.mkDerivation {
  pname = "weather-agent";
  inherit version;

  dontUnpack = true;
  nativeBuildInputs = [ makeWrapper ];

  # JAVA_OPTS (e.g. -Xmx) is expanded at run time. Native libraries get libstdc++; netCDF-Java loads
  # libaec (CCSDS, GRIB2 template 5.42) through JNA. Client SQL runs in the sandboxed worker (nix/sql-worker.nix)
  # under bwrap and prlimit.
  installPhase = ''
    runHook preInstall
    makeWrapper ${graalvm-ce}/bin/java $out/bin/weather-agent \
      --set VIRTUAL_ENV ${venv} \
      --prefix LD_LIBRARY_PATH : ${lib.makeLibraryPath [ stdenv.cc.cc.lib ]} \
      --set-default WEATHER_CA_BUNDLE ${caBundle} \
      --prefix PATH : ${
        lib.makeBinPath [
          bubblewrap
          util-linux
        ]
      } \
      --set-default WEATHER_QUERY_WORKER_PYTHON ${sql-worker}/python3 \
      --set-default WEATHER_QUERY_WORKER_SCRIPT ${sql-worker}/sql_worker.py \
      --set-default WEATHER_QUERY_WORKER_CLOSURE ${sql-worker.closure}/store-paths \
      --add-flags "-Djna.library.path=${lib.makeLibraryPath [ libaec ]} \$JAVA_OPTS -jar ${jar}"
    runHook postInstall
  '';

  passthru = {
    inherit
      jar
      venv
      caBundle
      sql-worker
      ;
  };

  meta = {
    description = "MCP server answering weather questions from open ensemble forecasts";
    license = lib.licenses.asl20;
    mainProgram = "weather-agent";
    platforms = [ "x86_64-linux" ];
  };
}
