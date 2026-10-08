# Everything the JAR build downloads, as a fixed-output derivation: Pyronaut's SDK state (~/.pyronaut) and the
# Maven repository (~/.m2) after `pyronaut setup` and `pyronaut install`. The JAR itself is then built offline
# (nix/jar.nix), so only dependency changes need a new hash here.
#
# The name carries a digest of what the output depends on (pyproject.toml, Pyronaut, GraalVM, the setup and
# normalization steps), so changing any of them forces a rebuild (and a hash update in nix/deps.<system>.sha256 if
# the output changes) instead of silently reusing old dependencies.
# After such a change, `nix build .#deps` fails with "hash mismatch ... got: sha256-..."; put that hash into
# nix/deps.<system>.sha256. The downloads include native launchers, so each system has its own hash.
{
  lib,
  stdenvNoCC,
  cacert,
  graalvm-ce,
  graalpy,
  pyronaut-cli,
  python3,
  venv,
  src,
}:
let
  graalpyName = "graalpy3.13-${graalpy.version}";
  # Shared with nix/jar.nix: a writable $HOME for Pyronaut, its build venv in ./.venv, and the environment both
  # builds run Pyronaut in.
  prepareBuild = ''
    export HOME=$TMPDIR/home
    # Pyronaut's Java tools read ~/.pyronaut and ~/.m2 via user.home, which comes from passwd, not $HOME.
    export JAVA_TOOL_OPTIONS="-Duser.home=$HOME"
    export JAVA_HOME=${graalvm-ce}

    # Pyronaut accepts a GraalPy found in its SDK directory; the upstream binary wouldn't run in the sandbox.
    mkdir -p $HOME/.pyronaut/sdks
    cp -r ${graalpy} $HOME/.pyronaut/sdks/${graalpyName}
    chmod -R u+w $HOME/.pyronaut/sdks
    # Download the native launchers directly instead of through the GitHub API, whose unauthenticated rate limit
    # (60 requests/hour) fails builds.
    cat > $HOME/.pyronaut/settings.toml <<'EOF'
    [native-images]
    base-url = "https://github.com/micronaut-projects/pyronaut/releases/download/v${pyronaut-cli.version}/"
    EOF

    # A GraalPy venv holding the runtime venv's packages. The marker tells `pyronaut install` that the declared
    # requirements are installed, so it skips its own pip run (pip can't resolve weather-core, and there's no
    # network in the JAR build).
    $HOME/.pyronaut/sdks/${graalpyName}/bin/graalpy -m venv --without-pip .venv
    cp -r ${venv}/lib/python3.13/site-packages/. .venv/lib/python3.13/site-packages/
    chmod -R u+w .venv
    PYTHONPATH=${pyronaut-cli}/${python3.sitePackages} python3 - <<'EOF'
    import json
    from pathlib import Path
    from pyronaut_cli_v2 import cli
    requirements, files = cli._project_python_requirements(Path("."))
    state = cli._project_venv_state(Path(".venv/bin/python"), requirements, files)
    Path(".venv", cli._PROJECT_VENV_STATE).write_text(json.dumps(state))
    EOF
  '';

  # Everything that shapes the output besides the network: changing any of it must change the name, or a store
  # that already has the old output would keep using it while fresh builds fail on the hash.
  inputsDigest = builtins.substring 0 12 (
    builtins.hashString "sha256" (
      lib.concatStringsSep "\n" [
        (builtins.hashFile "sha256" (src + "/pyproject.toml"))
        (builtins.hashFile "sha256" ./normalize-deps.py)
        # Without the venv's path: it changes with weather_core's code, which the downloads don't depend on.
        (builtins.hashString "sha256" (
          builtins.unsafeDiscardStringContext (builtins.replaceStrings [ "${venv}" ] [ "@VENV@" ] prepareBuild)
        ))
        pyronaut-cli.version
        graalvm-ce.version
        graalpy.version
      ]
    )
  );
in
stdenvNoCC.mkDerivation {
  name = "weather-agent-deps-${inputsDigest}";
  inherit src;

  nativeBuildInputs = [
    cacert
    graalvm-ce
    pyronaut-cli
    python3
  ];

  outputHashMode = "recursive";
  outputHashAlgo = "sha256";
  outputHash = lib.fileContents (./. + "/deps.${stdenvNoCC.hostPlatform.system}.sha256");

  buildPhase = ''
    runHook preBuild
    ${prepareBuild}
    export SSL_CERT_FILE=${cacert}/etc/ssl/certs/ca-bundle.crt
    pyronaut setup --progress off
    pyronaut install --progress off
    runHook postBuild
  '';

  # Make the downloads reproducible and free of store paths, which a fixed-output derivation may not contain.
  installPhase = ''
    runHook preInstall
    rm -rf $HOME/.pyronaut/sdks
    python3 ${./normalize-deps.py} $HOME ${graalvm-ce}
    mkdir -p $out
    cp -r $HOME/.pyronaut $HOME/.m2 $out/
    if grep -rl /nix/store $out || find $out -lname '/nix/store/*' | grep .; then
      echo "store paths left in the dependencies" >&2
      exit 1
    fi
    runHook postInstall
  '';

  dontFixup = true;

  passthru = {
    inherit prepareBuild;
    # Restores the dependencies into a writable $HOME (after prepareBuild).
    restore = deps: ''
      cp -r ${deps}/.pyronaut ${deps}/.m2 $HOME/
      chmod -R u+w $HOME/.pyronaut $HOME/.m2
      grep -rlZ -e @HOME@ -e @JAVA_HOME@ $HOME/.pyronaut $HOME/.m2 \
        | xargs -0 -r sed -i -e "s|@HOME@|$HOME|g" -e "s|@JAVA_HOME@|${graalvm-ce}|g"
    '';
  };
}
