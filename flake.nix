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
      version = (builtins.fromTOML (builtins.readFile ./pyproject.toml)).project.version;
      # Only what the JAR build reads, so docs and Nix edits don't change its inputs.
      fs = nixpkgs.lib.fileset;
      src = fs.toSource {
        root = ./.;
        fileset = fs.unions [
          ./pyproject.toml
          ./uv.lock
          ./micronaut-cli.yml
          ./config
          ./packages
          ./src
          ./src-java
        ];
      };
      # The venv only depends on the lock file and the workspace packages it installs.
      venvSrc = fs.toSource {
        root = ./.;
        fileset = fs.unions [
          ./uv.lock
          ./packages
        ];
      };
    in
    {
      packages = forAllSystems (
        pkgs:
        let
          graalvm-ce = pkgs.graalvmPackages.graalvm-ce;
          graalpy = pkgs.callPackage ./nix/graalpy.nix { };
          pyronaut-cli = pkgs.callPackage ./nix/pyronaut-cli.nix { };
          venv = pkgs.callPackage ./nix/venv.nix { src = venvSrc; };
          deps = pkgs.callPackage ./nix/deps.nix {
            inherit
              src
              graalvm-ce
              graalpy
              pyronaut-cli
              venv
              ;
          };
          jar = pkgs.callPackage ./nix/jar.nix {
            inherit
              src
              version
              graalvm-ce
              pyronaut-cli
              deps
              ;
          };
          weather-agent = pkgs.callPackage ./nix/package.nix {
            inherit
              version
              graalvm-ce
              jar
              venv
              ;
          };
        in
        {
          inherit
            graalpy
            pyronaut-cli
            deps
            jar
            venv
            weather-agent
            ;
          default = weather-agent;
        }
      );

      checks = forAllSystems (pkgs: {
        vm = import ./nix/test.nix {
          inherit pkgs;
          weather-agent = self.packages.${pkgs.stdenv.hostPlatform.system}.weather-agent;
        };
      });

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
