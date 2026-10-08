package at.yawk.weatheragent;

import io.micronaut.core.annotation.Nullable;
import io.micronaut.http.HttpRequest;
import io.micronaut.http.HttpResponse;
import io.micronaut.http.HttpStatus;
import io.micronaut.http.annotation.RequestFilter;
import io.micronaut.http.annotation.ServerFilter;
import java.net.URI;
import java.util.Locale;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Host/Origin validation for the MCP endpoint, as the MCP Streamable HTTP spec requires. Without it, a web page
 * could reach a local or LAN instance through DNS rebinding: the browser then sends the attacker's host name,
 * which Micronaut's built-in localhost guard doesn't cover.
 */
@ServerFilter({"/mcp", "/mcp/**"})
public final class McpHostFilter {
    private final Set<String> allowed;

    public McpHostFilter(McpAccessConfig config) {
        this.allowed = config.allowedHosts().stream()
            .map(String::trim)
            .filter(s -> !s.isEmpty())
            .map(s -> s.toLowerCase(Locale.ROOT))
            .collect(Collectors.toUnmodifiableSet());
    }

    @RequestFilter
    @Nullable
    public HttpResponse<?> check(HttpRequest<?> request) {
        String host = hostName(request.getHeaders().get("Host"));
        if (host == null || !allowed.contains(host)) {
            return HttpResponse.status(HttpStatus.FORBIDDEN).body("host not allowed");
        }
        String origin = request.getHeaders().get("Origin");
        if (origin != null && !origin.equals("null")) {
            String originHost;
            try {
                originHost = URI.create(origin).getHost();
            } catch (IllegalArgumentException e) {
                originHost = null;
            }
            if (originHost == null || !allowed.contains(stripBrackets(originHost.toLowerCase(Locale.ROOT)))) {
                return HttpResponse.status(HttpStatus.FORBIDDEN).body("origin not allowed");
            }
        } else if (origin != null) {
            return HttpResponse.status(HttpStatus.FORBIDDEN).body("origin not allowed");
        }
        return null;
    }

    /** "example.org:8080" → "example.org", "[::1]:8080" → "::1". */
    static @Nullable String hostName(@Nullable String header) {
        if (header == null || header.isBlank()) {
            return null;
        }
        String h = header.trim().toLowerCase(Locale.ROOT);
        if (h.startsWith("[")) {
            int end = h.indexOf(']');
            return end < 0 ? null : h.substring(1, end);
        }
        int colon = h.indexOf(':');
        return colon < 0 ? h : h.substring(0, colon);
    }

    private static String stripBrackets(String host) {
        return host.startsWith("[") && host.endsWith("]") ? host.substring(1, host.length() - 1) : host;
    }
}
