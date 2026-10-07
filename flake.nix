{
  description = "weather-agent: ensemble weather probabilities from open DWD/ECMWF data, served over MCP";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f (pkgsFor system));
      pkgsFor =
        system:
        import nixpkgs {
          inherit system;
        };
    in
    {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = [
            pkgs.uv
            pkgs.graalvmPackages.graalvm-ce
            pkgs.libaec
          ];
          shellHook = ''
            export JAVA_HOME=${pkgs.graalvmPackages.graalvm-ce}
            # uv-managed GraalPy and Pyronaut's native launchers are generic Linux binaries; nix-ld runs them.
            export NIX_LD_LIBRARY_PATH=${
              pkgs.lib.makeLibraryPath [
                pkgs.zlib
                pkgs.stdenv.cc.cc
                pkgs.libxcrypt-legacy
                pkgs.openssl
                pkgs.bzip2
                pkgs.xz
                pkgs.libffi
                pkgs.libaec # CCSDS (GRIB2 template 5.42) decoding via JNA
              ]
            }
            # GraalPy dlopen()s libpythonvm.so, whose deps nix-ld alone does not resolve.
            export LD_LIBRARY_PATH="$NIX_LD_LIBRARY_PATH''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            export PATH="$HOME/.local/bin:$PATH"
            # Micronaut's PEM parser rejects the label lines between certificates in system bundles, so hand it
            # a copy with only the BEGIN/END blocks.
            mkdir -p var
            sed -n '/-----BEGIN CERTIFICATE-----/,/-----END CERTIFICATE-----/p' \
              ${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt > var/ca-certificates.pem
            export WEATHER_CA_BUNDLE="$PWD/var/ca-certificates.pem"
            # Regenerable state: keep it out of restic backups too.
            for d in .venv __pyronaut__; do
              [ -d "$d" ] && touch "$d/.nobackup"
            done
          '';
        };
      });
    };
}
