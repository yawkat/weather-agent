# VM test: the service configured the way a host config (e.g. goliath) would, starting, serving MCP and running
# Python tool code (weather_core, numpy). The VM has no network, so forecasts (downloads, GRIB decoding) aren't
# covered here; the SQL sandbox is checked on its own, under the service's systemd settings.
{ pkgs, weather-agent }:
let
  inherit (weather-agent) sql-worker;
  # Hardening that still lets the SQL worker's bwrap create its namespaces: no RestrictNamespaces= or
  # PrivateUsers=, and no SystemCallFilter= (bwrap mounts, and @system-service excludes @mount). No
  # ProtectKernelTunables=, ProtectKernelLogs= or ProtectHostname= either: they cover parts of /proc, and then the
  # kernel refuses bwrap a fresh /proc for its PID namespace. bwrap brings up loopback in its network namespace over netlink.
  hardening = {
    NoNewPrivileges = true;
    ProtectSystem = "strict";
    ProtectHome = true;
    PrivateTmp = true;
    PrivateDevices = true;
    ProtectKernelModules = true;
    ProtectControlGroups = true;
    ProtectClock = true;
    RestrictAddressFamilies = "AF_INET AF_INET6 AF_UNIX AF_NETLINK";
    RestrictRealtime = true;
    RestrictSUIDSGID = true;
    LockPersonality = true;
    SystemCallArchitectures = "native";
    # The SQL worker asks to be the OOM killer's first choice; losing it mustn't stop the service.
    OOMPolicy = "continue";
  };
  # The worker sandbox as src/weather_agent/sql_engine.py (DuckDbEngine._command) starts it; keep in sync.
  sandboxCheck = pkgs.writeShellScript "sql-sandbox-check" ''
    set -euo pipefail
    export PATH=${
      pkgs.lib.makeBinPath [
        pkgs.bubblewrap
        pkgs.util-linux
        pkgs.coreutils
        pkgs.gnugrep
      ]
    }
    py=${sql-worker}/python3
    dir=$(mktemp -d)
    binds=()
    while read -r path; do binds+=(--ro-bind "$path" "$path"); done < ${sql-worker.closure}/store-paths
    sandbox() {
      prlimit --fsize=4065536 --nofile=64 --core=0 --cpu=93 -- \
        bwrap --unshare-all --unshare-user --disable-userns --die-with-parent --new-session \
        --clearenv --setenv HOME /nonexistent --setenv OPENBLAS_NUM_THREADS 1 --setenv MALLOC_ARENA_MAX 2 \
        "''${binds[@]}" --ro-bind ${sql-worker}/sql_worker.py /worker/sql_worker.py --ro-bind "$dir" /work \
        --proc /proc --dev-bind /dev/null /dev/null --dev-bind /dev/urandom /dev/urandom --remount-ro / \
        --chdir /work -- "$@"
    }
    request() {
      $py -c 'import json, sys; json.dump({"memory_mb": 512, "overhead_mb": 1024, "threads": 2, "max_rows": 500,
        "max_chars": 1000000, "timeout_ms": 5000, "setup": ["CREATE TABLE t (x INTEGER)", "INSERT INTO t VALUES (41)"],
        "tables": [], "query": sys.argv[1]}, open(sys.argv[2], "w"))' "$1" "$dir/request.json"
    }

    request "SELECT x + 1 AS y FROM t"
    sandbox $py -I -B /worker/sql_worker.py /work/request.json | grep -F '"rows":[[42]]'

    request "SELECT len(range(0, 1000000000))"
    status=0; sandbox $py -I -B /worker/sql_worker.py /work/request.json > "$dir/out" || status=$?
    echo "memory blowup: exit $status, $(head -c 200 "$dir/out")"
    test "$status" -ne 0

    # The sandbox itself: no host files, nothing writable, no network.
    sandbox $py -I -c 'if True:
      import os, socket
      for path in ["/etc/passwd", "/var/lib/weather-agent", "/run", "/home"]:
          assert not os.path.exists(path), path
      for path in ["/work/x", "/tmp/x", "/x", "/dev/x", "/dev/shm/x", "/proc/x"]:
          try:
              open(path, "w")
              raise AssertionError(path)
          except OSError:
              pass
      try:
          socket.create_connection(("10.0.2.2", 80), timeout=2)
          raise AssertionError("network")
      except OSError:
          pass
      print("sandbox ok")
    '
  '';
in
pkgs.testers.runNixOSTest {
  name = "weather-agent";

  nodes.machine = {
    virtualisation.memorySize = 3072;
    virtualisation.cores = 2;

    users.users.weather-agent = {
      isSystemUser = true;
      group = "weather-agent";
      home = "/var/lib/weather-agent";
    };
    users.groups.weather-agent = { };

    systemd.services.weather-agent = {
      wantedBy = [ "multi-user.target" ];
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];
      environment = {
        WEATHER_BIND_HOST = "127.0.0.1";
        MICRONAUT_SERVER_PORT = "8080";
        WEATHER_ALLOWED_HOSTS = "localhost,weather.test";
        WEATHER_CACHE_DIR = "/var/lib/weather-agent/cache";
        # Without a cap the JVM takes a quarter of RAM; about 9.6 GB was seen on a dev machine.
        JAVA_OPTS = "-Xmx1g";
      };
      serviceConfig = {
        ExecStart = "${weather-agent}/bin/weather-agent";
        StateDirectory = "weather-agent";
        WorkingDirectory = "/var/lib/weather-agent";
        User = "weather-agent";
        Group = "weather-agent";
        Restart = "on-failure";
        SuccessExitStatus = 143;
      }
      // hardening;
    };
  };

  testScript = ''
    import json
    import shlex

    def mcp(method, params, session=None, host="localhost", request_id=1):
        body = json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        headers = f"-H 'Host: {host}' -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream'"
        if session:
            headers += f" -H 'Mcp-Session-Id: {session}'"
        out = machine.succeed(
            f"curl -sS --fail-with-body -D /tmp/headers {headers} --data-binary @- http://127.0.0.1:8080/mcp <<'EOF'\n{body}\nEOF"
        )
        # Streamable HTTP answers either with JSON or with an SSE stream of JSON-RPC messages.
        lines = [l[len("data:"):] for l in out.splitlines() if l.startswith("data:")] or [out]
        response = json.loads(lines[-1])
        assert "error" not in response, response
        session_id = None
        for line in machine.succeed("cat /tmp/headers").splitlines():
            name, _, value = line.partition(":")
            if name.strip().lower() == "mcp-session-id":
                session_id = value.strip()
        return response["result"], session_id

    machine.wait_for_unit("weather-agent.service")
    machine.wait_for_open_port(8080, timeout=300)

    with subtest("initialize"):
        result, session = mcp("initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "nixos-test", "version": "1"},
        })
        assert result["serverInfo"]["name"] == "weather-agent", result
        machine.succeed(
            "curl -sS --fail -H 'Host: localhost' -H 'Content-Type: application/json' "
            "-H 'Accept: application/json, text/event-stream' "
            + (f"-H 'Mcp-Session-Id: {session}' " if session else "")
            + """--data '{"jsonrpc":"2.0","method":"notifications/initialized"}' http://127.0.0.1:8080/mcp"""
        )

    with subtest("tools are listed"):
        result, _ = mcp("tools/list", {}, session, request_id=2)
        names = {tool["name"] for tool in result["tools"]}
        assert {"forecast", "weather_query_help", "describe_route"} <= names, names

    with subtest("Python tool code runs"):
        result, _ = mcp("tools/call", {
            "name": "describe_route",
            "arguments": {"polyline": "_p~iF~ps|U_ulLnnqC_mqNvxq`@", "speed_kmh": 20},
        }, session, request_id=3)
        route = json.loads(result["content"][0]["text"])
        assert "error" not in route, route
        print(route)

    with subtest("configured host names only"):
        mcp("tools/list", {}, session, host="weather.test", request_id=4)
        status = machine.succeed(
            "curl -s -o /dev/null -w '%{http_code}' -H 'Host: evil.example' -H 'Content-Type: application/json' "
            """--data '{"jsonrpc":"2.0","id":5,"method":"tools/list"}' http://127.0.0.1:8080/mcp"""
        )
        assert status == "403", status

    with subtest("heap cap, state directory, native libraries"):
        machine.succeed("tr '\\0' ' ' < /proc/$(systemctl show -p MainPID --value weather-agent)/cmdline | grep -- -Xmx1g")
        machine.succeed("test -d /var/lib/weather-agent")
        # GraalPy extracts its native libraries under ~/.cache and must be able to dlopen them.
        machine.fail("journalctl -u weather-agent | grep 'failed to map segment'")

    with subtest("SQL worker sandbox under the service's settings"):
        properties = " ".join(shlex.quote(f"--property={k}={'yes' if v is True else v}") for k, v in json.loads(${pkgs.lib.escapeShellArg (builtins.toJSON hardening)}).items())
        print(machine.succeed(
            f"systemd-run --wait --pipe --collect -p User=weather-agent -p Group=weather-agent {properties} "
            "${sandboxCheck}"
        ))
  '';
}
