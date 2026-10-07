package at.yawk.weatheragent;

import io.micronaut.context.annotation.Context;
import io.micronaut.context.annotation.Value;
import io.micronaut.core.annotation.Nullable;
import io.micronaut.http.HttpRequest;
import io.micronaut.http.HttpResponse;
import io.micronaut.http.annotation.RequestFilter;
import io.micronaut.http.annotation.ServerFilter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.regex.Pattern;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Secret-path access control: with an access key configured, MCP is served only at {@code /mcp/<key>}. This stands
 * in for real auth until OAuth: claude.ai connectors can't send a fixed bearer token, but they keep the URL.
 * <p>
 * The route is the template {@code /mcp{/key}}, so routing never compares against the key; this filter does, in
 * constant time. Wrong or missing keys get a 404, like any unknown path.
 * <p>
 * {@code @Context}: filters are created on the first request otherwise, so a bad key would only show up then.
 */
@Context
@ServerFilter({"/mcp", "/mcp/**"})
public final class McpKeyFilter {
    private static final Logger LOG = LoggerFactory.getLogger(McpKeyFilter.class);
    private static final String PREFIX = "/mcp";
    /** ≥ 192 bits of base64url, e.g. {@code openssl rand -base64 32 | tr '+/' '-_' | tr -d '='}. */
    private static final Pattern KEY = Pattern.compile("[A-Za-z0-9_-]{32,256}");

    private final byte @Nullable [] key;

    public McpKeyFilter(@Value("${weather.access-key:}") String key,
                        @Value("${weather.access-key-file:}") String keyFile) {
        this.key = load(key, keyFile);
        if (this.key == null) {
            LOG.warn("No access key configured: MCP is open to anyone who can reach {}", PREFIX);
        } else {
            LOG.info("MCP requires the access key in the path ({}/<key>)", PREFIX);
        }
    }

    /**
     * The key, or null when both settings are empty (the defaults). Any other value must yield a valid key: a
     * whitespace-only setting or an empty key file (placeholder, failed decryption) fails instead of opening MCP.
     */
    static byte @Nullable [] load(String key, String keyFile) {
        if (key.isEmpty() && keyFile.isEmpty()) {
            return null;
        }
        if (!key.isEmpty() && !keyFile.isEmpty()) {
            throw new IllegalStateException("set weather.access-key or weather.access-key-file, not both");
        }
        if (!keyFile.isEmpty()) {
            try {
                key = Files.readString(Path.of(keyFile), StandardCharsets.UTF_8);
            } catch (IOException | InvalidPathException e) {
                throw new IllegalStateException("cannot read weather.access-key-file", e);
            }
        }
        key = key.strip();
        if (!KEY.matcher(key).matches()) {
            // Don't echo the key: it may be a real secret with a typo.
            throw new IllegalStateException(
                "access key must be 32 to 256 characters of A-Z, a-z, 0-9, '-' and '_' (got " + key.length() + ")");
        }
        return key.getBytes(StandardCharsets.US_ASCII);
    }

    @RequestFilter
    @Nullable
    public HttpResponse<?> check(HttpRequest<?> request) {
        return allowed(request.getPath()) ? null : HttpResponse.notFound();
    }

    public boolean allowed(String path) {
        if (key == null) {
            return path.equals(PREFIX);
        }
        if (!path.startsWith(PREFIX + "/")) {
            return false;
        }
        byte[] given = path.substring(PREFIX.length() + 1).getBytes(StandardCharsets.UTF_8);
        // isEqual is constant-time for equal lengths; the length itself isn't secret.
        return MessageDigest.isEqual(given, key);
    }
}
