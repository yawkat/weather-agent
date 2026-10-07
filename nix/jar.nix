# The application fat JAR, built by Pyronaut offline from the prefetched dependencies (nix/deps.nix).
{
  stdenvNoCC,
  graalvm-ce,
  pyronaut-cli,
  python3,
  deps,
  src,
  version,
}:
stdenvNoCC.mkDerivation {
  name = "weather-agent-${version}.jar";
  inherit src;

  nativeBuildInputs = [
    graalvm-ce
    pyronaut-cli
    python3
  ];

  buildPhase = ''
    runHook preBuild
    ${deps.prepareBuild}
    ${deps.restore deps}
    pyronaut install --offline --progress off
    pyronaut build --jar --offline --progress off
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    cp dist/weather_agent-${version}.jar $out
    runHook postInstall
  '';

  dontFixup = true;
}
