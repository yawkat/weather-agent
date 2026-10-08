package at.yawk.weatheragent;

/**
 * A tool whose results the client also shows in an interactive view (MCP Apps), registered by {@link McpApps}.
 * Implemented in Python; JSON strings keep that side free of Java collection conversions.
 */
public interface McpAppTool {
    String name();

    String title();

    String description();

    /** JSON schema of the arguments. */
    String inputSchema();

    /** The {@code ui://} resource of the view. */
    String viewUri();

    /**
     * Run the tool. Gets the arguments as a JSON object and returns a JSON object: {@code {"text": "...",
     * "structuredContent": {...}}}, or {@code {"error": "..."}}. The model reads the text, the view gets the
     * structured content.
     */
    String call(String arguments);
}
