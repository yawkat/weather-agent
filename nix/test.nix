# VM test: the service configured the way a host config (e.g. goliath) would, starting, serving MCP and running
# Python tool code (weather_core, numpy). The VM has no network, so forecasts (downloads, GRIB decoding)
# aren't covered here.
{ pkgs, weather-agent }:
let
  # A real host keeps the key out of the store, e.g. with agenix or sops-nix.
  accessKey = "test-key-0123456789abcdefghijklmnopqrstuvwxyz";
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

    environment.etc."weather-agent/access-key".text = accessKey;

    systemd.services.weather-agent = {
      wantedBy = [ "multi-user.target" ];
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];
      environment = {
        WEATHER_BIND_HOST = "127.0.0.1";
        MICRONAUT_SERVER_PORT = "8080";
        WEATHER_ALLOWED_HOSTS = "localhost,weather.test";
        WEATHER_CACHE_DIR = "/var/lib/weather-agent/cache";
        # MCP only at /mcp/<key>. A credential, so the key isn't in the unit's environment.
        WEATHER_ACCESS_KEY_FILE = "%d/access-key";
        # Without a cap the JVM takes a quarter of RAM; about 9.6 GB was seen on a dev machine.
        JAVA_OPTS = "-Xmx1g";
      };
      serviceConfig = {
        ExecStart = "${weather-agent}/bin/weather-agent";
        LoadCredential = "access-key:/etc/weather-agent/access-key";
        StateDirectory = "weather-agent";
        WorkingDirectory = "/var/lib/weather-agent";
        User = "weather-agent";
        Group = "weather-agent";
        Restart = "on-failure";
        SuccessExitStatus = 143;
      };
    };
  };

  testScript = ''
    import json

    MCP_URL = "http://127.0.0.1:8080/mcp/${accessKey}"

    def mcp(method, params, session=None, host="localhost", request_id=1):
        body = json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        headers = f"-H 'Host: {host}' -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream'"
        if session:
            headers += f" -H 'Mcp-Session-Id: {session}'"
        out = machine.succeed(
            f"curl -sS --fail-with-body -D /tmp/headers {headers} --data-binary @- {MCP_URL} <<'EOF'\n{body}\nEOF"
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
            + """--data '{"jsonrpc":"2.0","method":"notifications/initialized"}' """ + MCP_URL
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
            """--data '{"jsonrpc":"2.0","id":5,"method":"tools/list"}' """ + MCP_URL
        )
        assert status == "403", status

    with subtest("MCP only at the secret path"):
        for path in ["/mcp", "/mcp/", "/mcp/wrong-key-0123456789abcdefghijklmnopqrstuvwxyz", "/mcp/${accessKey}x"]:
            status = machine.succeed(
                "curl -s -o /dev/null -w '%{http_code}' -H 'Host: localhost' -H 'Content-Type: application/json' "
                """--data '{"jsonrpc":"2.0","id":6,"method":"tools/list"}' http://127.0.0.1:8080""" + path
            )
            assert status == "404", (path, status)
        # The key comes from the credential, not the environment.
        machine.fail("systemctl show -p Environment weather-agent | grep -F ${accessKey}")

    with subtest("heap cap, state directory, native libraries"):
        machine.succeed("tr '\\0' ' ' < /proc/$(systemctl show -p MainPID --value weather-agent)/cmdline | grep -- -Xmx1g")
        machine.succeed("test -d /var/lib/weather-agent")
        # GraalPy extracts its native libraries under ~/.cache and must be able to dlopen them.
        machine.fail("journalctl -u weather-agent | grep 'failed to map segment'")
  '';
}
