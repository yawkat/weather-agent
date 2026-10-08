package at.yawk.weatheragent;

import io.micronaut.context.event.BeanCreatedEvent;
import io.micronaut.context.event.BeanCreatedEventListener;
import io.modelcontextprotocol.json.McpJsonMapper;
import io.modelcontextprotocol.server.McpStatelessServerFeatures;
import io.modelcontextprotocol.server.McpStatelessSyncServer;
import io.modelcontextprotocol.spec.McpSchema;
import jakarta.inject.Singleton;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

/**
 * MCP Apps: tools whose results the client also shows in an interactive view, an HTML page served as a
 * {@code ui://} resource (declared with {@code @Resource} in Python, content from {@link #view}).
 * <p>
 * micronaut-mcp's {@code @Tool} can't set the {@code _meta} that links a tool to its view, so {@link McpAppTool}
 * beans are added to the server here once it exists. They aren't registered as SDK specification beans because
 * Micronaut's annotation processing of the SDK's Jackson-annotated types fails under Pyronaut (missing
 * micronaut-serde classes on the processor path).
 */
@Singleton
public final class McpApps implements BeanCreatedEventListener<McpStatelessSyncServer> {
    public static final String MIME_TYPE = "text/html;profile=mcp-app";

    private final McpJsonMapper json;
    private final List<McpAppTool> tools;

    public McpApps(McpJsonMapper json, List<McpAppTool> tools) {
        this.json = json;
        this.tools = tools;
    }

    @Override
    public McpStatelessSyncServer onCreated(BeanCreatedEvent<McpStatelessSyncServer> event) {
        McpStatelessSyncServer server = event.getBean();
        for (McpAppTool tool : tools) {
            server.addTool(specification(tool));
        }
        return server;
    }

    /** HTML of a view, from the classpath. */
    public static String view(String classpathResource) {
        try (InputStream in = McpApps.class.getClassLoader().getResourceAsStream(classpathResource)) {
            if (in == null) {
                throw new IllegalStateException("missing resource " + classpathResource);
            }
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private McpStatelessServerFeatures.SyncToolSpecification specification(McpAppTool app) {
        String uri = app.viewUri();
        McpSchema.Tool tool = McpSchema.Tool.builder(app.name(), json, app.inputSchema())
            .title(app.title())
            .description(app.description())
            .annotations(new McpSchema.ToolAnnotations(app.title(), true, false, true, true, false))
            // "ui/resourceUri" is the key from before the extension was finalised; some hosts still read it.
            .meta(Map.of("ui", Map.of("resourceUri", uri), "ui/resourceUri", uri))
            .build();
        return McpStatelessServerFeatures.SyncToolSpecification.builder()
            .tool(tool)
            .callHandler((context, request) -> call(app, request))
            .build();
    }

    private McpSchema.CallToolResult call(McpAppTool app, McpSchema.CallToolRequest request) {
        Map<?, ?> result;
        try {
            String arguments = json.writeValueAsString(request.arguments() == null ? Map.of() : request.arguments());
            // Not a TypeRef: Pyronaut's incremental compilation hangs on anonymous classes (micronaut-core#13799).
            result = json.readValue(app.call(arguments), Map.class);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        if (result.get("error") instanceof String error) {
            return McpSchema.CallToolResult.builder().addTextContent(error).isError(true).build();
        }
        return McpSchema.CallToolResult.builder()
            .addTextContent((String) result.get("text"))
            .structuredContent(result.get("structuredContent"))
            .isError(false)
            .build();
    }
}
