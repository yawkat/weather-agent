package at.yawk.weatheragent;

import io.micronaut.context.event.BeanCreatedEvent;
import io.micronaut.context.event.BeanCreatedEventListener;
import io.modelcontextprotocol.server.McpServer;
import jakarta.inject.Singleton;

/**
 * Server instructions in the {@code initialize} result, which clients add to the agent's context. micronaut-mcp has
 * no setting for them, so they are set on the SDK's server specification. The text is {@code mcp-instructions.md}
 * in the config directory.
 */
@Singleton
public final class McpInstructions implements BeanCreatedEventListener<McpServer.StatelessSyncSpecification> {
    @Override
    public McpServer.StatelessSyncSpecification onCreated(
            BeanCreatedEvent<McpServer.StatelessSyncSpecification> event) {
        return event.getBean().instructions(McpApps.view("mcp-instructions.md").strip());
    }
}
