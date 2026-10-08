package at.yawk.weatheragent;

import io.micronaut.context.annotation.ConfigurationProperties;
import io.micronaut.core.annotation.Nullable;
import io.micronaut.core.bind.annotation.Bindable;
import java.util.List;

/**
 * Who may reach the MCP endpoint ({@code weather.*}; the other settings are in {@code weather_agent/config.py}).
 *
 * @param allowedHosts  Host names (and Origin hosts) /mcp accepts; protects against DNS rebinding from browsers
 *                      ({@link McpHostFilter})
 * @param accessKey     With a key set, MCP is only at {@code /mcp/<key>} ({@link McpKeyFilter}). For development;
 *                      deployments use {@code accessKeyFile}
 * @param accessKeyFile File holding the access key, e.g. a systemd credential
 */
@ConfigurationProperties("weather")
public record McpAccessConfig(
    @Bindable(defaultValue = "localhost,127.0.0.1,::1") List<String> allowedHosts,
    @Nullable String accessKey,
    @Nullable String accessKeyFile
) {
}
