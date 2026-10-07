# Runtime virtualenv for the embedded GraalPy, built from uv.lock without running any Python: wheels are unpacked
# into site-packages and their native libraries patched. GraalPy only needs pyvenv.cfg and some bin/python to
# recognise the venv; the interpreter itself comes from the JAR (docs/spike-notes.md, "Packaging").
{
  lib,
  stdenv,
  fetchurl,
  autoPatchelfHook,
  unzip,
  zlib,
  src,
}:
let
  lock = builtins.fromTOML (builtins.readFile (src + "/uv.lock"));
  packages = lib.listToAttrs (map (p: lib.nameValuePair p.name p) lock.package);

  # The GraalPy wheel index publishes no hashes, so uv.lock has none for its wheels. Pin them here by file name.
  extraHashes = {
    "numpy-2.4.4-graalpy313-graalpy253_313_native-manylinux_2_27_x86_64.whl" =
      "sha256-lBbzL4XC5igoLzSKlNs9PVKB0ptgTNW+r+9p6Q8acQY=";
  };

  # Runtime closure of the root project: [project] dependencies only, not dependency groups (pytest).
  closure =
    let
      go =
        seen: name:
        if seen ? ${name} then
          seen
        else
          let
            pkg = packages.${name};
            deps = map (
              d:
              if d ? marker then
                throw "uv.lock: dependency ${d.name} of ${name} has a marker (${d.marker}); teach nix/venv.nix to evaluate it"
              else
                d.name
            ) (pkg.dependencies or [ ]);
          in
          lib.foldl' go (seen // { ${name} = pkg; }) deps;
    in
    lib.attrValues (removeAttrs (go { } "weather-agent") [ "weather-agent" ]);

  fileName = url: lib.last (lib.splitString "/" url);

  pickWheel =
    pkg:
    let
      wheels = pkg.wheels or [ ];
      native = lib.filter (w: lib.hasSuffix "manylinux_2_27_x86_64.whl" (fileName w.url)) wheels;
      pure = lib.filter (w: lib.hasSuffix "-none-any.whl" (fileName w.url)) wheels;
      wheel =
        if native != [ ] then
          lib.head native
        else if pure != [ ] then
          lib.head pure
        else
          throw "uv.lock: no linux x86_64 wheel for ${pkg.name}";
      name = fileName wheel.url;
    in
    fetchurl {
      inherit (wheel) url;
      hash =
        if wheel ? hash then
          wheel.hash
        else
          extraHashes.${name} or (throw "uv.lock: no hash for ${name}; add it to extraHashes in nix/venv.nix");
    };

  registryPackages = lib.filter (p: p.source ? registry) closure;
  workspacePackages = lib.filter (p: p.source ? editable) closure;
  wheels = map pickWheel registryPackages;

  sitePackages = "lib/python3.13/site-packages";
in
assert lib.all (p: p.source ? registry || p.source ? editable) closure;
stdenv.mkDerivation {
  pname = "weather-agent-venv";
  version = packages.weather-agent.version;

  dontUnpack = true;
  nativeBuildInputs = [
    autoPatchelfHook
    unzip
  ];
  # numpy's bundled OpenBLAS/gfortran need libgcc_s, libstdc++ and libz.
  buildInputs = [
    stdenv.cc.cc.lib
    zlib
  ];

  installPhase = ''
    runHook preInstall
    site=$out/${sitePackages}
    mkdir -p $site $out/bin

    for whl in ${lib.escapeShellArgs wheels}; do
      unzip -q -o "$whl" -d $site
    done
    # Wheel .data directories: keep library code, drop console scripts and headers.
    for data in $site/*.data; do
      [ -e "$data" ] || continue
      for lib in purelib platlib; do
        [ -d "$data/$lib" ] && cp -r "$data/$lib/." $site/
      done
      rm -rf "$data"
    done

    ${lib.concatMapStringsSep "\n" (p: ''
      cp -r ${src + "/${p.source.editable}/src"}/. $site/
    '') workspacePackages}

    cat > $out/pyvenv.cfg <<EOF
    home = $out/bin
    implementation = GraalVM
    version_info = 3.13.14
    include-system-site-packages = false
    EOF
    # GraalPy locates the venv through bin/python but never runs it.
    cat > $out/bin/python <<EOF
    #!/bin/sh
    echo "weather-agent venv: run the application JAR, not this interpreter" >&2
    exit 1
    EOF
    chmod +x $out/bin/python

    runHook postInstall
  '';

  passthru.packages = map (p: "${p.name}==${p.version}") closure;

  meta.platforms = [ "x86_64-linux" ];
}
