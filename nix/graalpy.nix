# GraalPy standalone at the exact version Pyronaut pins (nixpkgs lags behind). Only the JAR build needs it: Pyronaut
# checks for it during setup and uv creates the build venv with it. The runtime uses the GraalPy embedded in the JAR.
{
  lib,
  stdenv,
  fetchurl,
  autoPatchelfHook,
  zlib,
  libxcrypt-legacy,
  openssl,
  bzip2,
  xz,
  libffi,
  sqlite,
}:
stdenv.mkDerivation (finalAttrs: {
  pname = "graalpy";
  version = "25.4.4";

  src = fetchurl {
    url = "https://github.com/oracle/graalpython/releases/download/graal-${finalAttrs.version}/graalpy3.13-${finalAttrs.version}-linux-amd64.tar.gz";
    hash = "sha256-i3Lm5RPQaXblyOBRIo1p1UH9520F1fkH6u9Lsx24OvQ=";
  };

  nativeBuildInputs = [ autoPatchelfHook ];
  buildInputs = [
    stdenv.cc.cc.lib
    zlib
    libxcrypt-legacy
    openssl
    bzip2
    xz
    libffi
    sqlite
  ];

  dontConfigure = true;
  dontBuild = true;
  dontStrip = true;

  installPhase = ''
    runHook preInstall
    mkdir -p $out
    cp -r . $out/
    runHook postInstall
  '';

  meta = {
    description = "GraalPy ${finalAttrs.version} standalone (build-time only)";
    homepage = "https://github.com/oracle/graalpython";
    license = lib.licenses.upl;
    platforms = [ "x86_64-linux" ];
    sourceProvenance = [ lib.sourceTypes.binaryNativeCode ];
  };
})
