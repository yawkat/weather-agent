# `weather-agent`: the fat JAR on GraalVM CE with the GraalPy venv and the native libraries it loads.
{
  lib,
  stdenv,
  runCommand,
  makeWrapper,
  cacert,
  graalvm-ce,
  libaec,
  jar,
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

  # JAVA_OPTS (e.g. -Xmx) is expanded at run time. netCDF-Java loads libaec (CCSDS, GRIB2 template 5.42) through
  # JNA.
  installPhase = ''
    runHook preInstall
    makeWrapper ${graalvm-ce}/bin/java $out/bin/weather-agent \
      --set VIRTUAL_ENV ${venv} \
      --set-default WEATHER_CA_BUNDLE ${caBundle} \
      --add-flags "-Djna.library.path=${lib.makeLibraryPath [ libaec ]} \$JAVA_OPTS -jar ${jar}"
    runHook postInstall
  '';

  passthru = {
    inherit jar venv caBundle;
  };

  meta = {
    description = "MCP server answering weather questions from open ensemble forecasts";
    license = lib.licenses.asl20;
    mainProgram = "weather-agent";
    platforms = [ "x86_64-linux" ];
  };
}
