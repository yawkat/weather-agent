# The Pyronaut CLI (pure Python; its JVM tools are shell scripts inside the wheel). Build-time only.
{
  lib,
  python3Packages,
  fetchurl,
}:
python3Packages.buildPythonApplication rec {
  pname = "pyronaut";
  version = "0.1.0";
  format = "wheel";

  src = fetchurl {
    url = "https://files.pythonhosted.org/packages/4b/78/b024a267f5f26f5bbee0ab2712bf9a9484b730d7a17da8c5f5d0d3063f7f/pyronaut-${version}-py3-none-any.whl";
    hash = "sha256-zHMCmu28XCYSWL5wLp2/bKSpIAJDTopq9JUwMke+kEY=";
  };

  # Keep the bundled tool scripts as shipped (`#!/bin/sh` exists in the build sandbox).
  dontPatchShebangs = true;

  meta = {
    description = "Polyglot Python/Java runtime on Micronaut (CLI)";
    homepage = "https://pyronaut.io/";
    license = lib.licenses.asl20;
    mainProgram = "pyronaut";
  };
}
