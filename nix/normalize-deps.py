"""Make ~/.pyronaut and ~/.m2 after `pyronaut setup`/`install` reproducible (see nix/deps.nix).

Usage: normalize-deps.py HOME JAVA_HOME
"""
import os
import re
import shutil
import sys
from pathlib import Path

home = Path(sys.argv[1])
java_home = sys.argv[2]
pyronaut = home / ".pyronaut"
m2 = home / ".m2"

# Lock files and Maven resolver bookkeeping (update timestamps, failed lookups in other repositories).
for path in list(pyronaut.rglob("*.lock")) + list(m2.rglob("*.lastUpdated")) + list(
        m2.rglob("resolver-status.properties")):
    path.unlink()

# Maven Central's prefix lists for the resolver's repository filter. Central republishes them whenever a new groupId
# appears, so they'd break the hash within days. Offline, the resolver doesn't filter without them.
shutil.rmtree(m2 / "repository" / ".meta", ignore_errors=True)

# Editor stubs aren't needed for building, and their generation isn't deterministic (parameter names of overloads
# vary between runs).
shutil.rmtree(pyronaut / "ide-stubs", ignore_errors=True)

# The tool runtime directory is named <descriptor hash>-<random UUID>; give it a fixed name.
renames = {}
for versions in (pyronaut / "tools").iterdir():
    current = versions / "current"
    target = os.readlink(current)
    fixed = target.rsplit("-", 5)[0] + "-nix"
    (versions / target).rename(versions / fixed)
    current.unlink()
    current.symlink_to(fixed)
    renames[target] = fixed

# Pyronaut links its bundled JARs from the CLI's installation (a store path) into the tool cache; copy them.
for root in (pyronaut, m2):
    for path in list(root.rglob("*")):
        if path.is_symlink() and os.path.isabs(os.readlink(path)):
            target = path.resolve()
            path.unlink()
            shutil.copy2(target, path)

# Native launchers (~1.6 GB): the JAR build never runs them, but Pyronaut checks that they exist and reads their
# classpath descriptors (resources/<launcher>/*.txt). Keep the metadata, stub the executables, drop the rest.
for platform_dir in (pyronaut / "bin").glob("*/*"):
    launchers = [p.stem for p in platform_dir.glob("*.json")]
    for path in platform_dir.iterdir():
        if path.suffix in (".json", ".txt"):
            continue
        if path.name == "resources":
            for resource in path.iterdir():
                if resource.name not in launchers:
                    shutil.rmtree(resource)
            continue
        if path.name in launchers:
            path.write_text("#!/bin/sh\necho 'native Pyronaut launchers are not available in the Nix build' >&2\nexit 1\n")
            path.chmod(0o755)
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

# Absolute paths become placeholders that nix/deps.nix `restore` substitutes back; drop `#<date>` header lines of
# Java properties files.
date_line = re.compile(r"^#(Mon|Tue|Wed|Thu|Fri|Sat|Sun) .*\n", re.MULTILINE)
for root in (pyronaut, m2):
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink() or path.suffix not in (".json", ".properties", ".repositories", ".txt"):
            continue
        text = path.read_text(encoding="utf-8")
        new = date_line.sub("", text).replace(str(home), "@HOME@").replace(java_home, "@JAVA_HOME@")
        for old, fixed in renames.items():
            new = new.replace(old, fixed)
        if new != text:
            path.write_text(new, encoding="utf-8")
