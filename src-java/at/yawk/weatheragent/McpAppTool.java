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

    /**
     * The {@code ui://} resource of the view. Tools link to a copy with the content's hash in its name (see
     * {@link McpApps}), so clients that cache views by URI pick up a changed view.
     */
    String viewUri();

    /** HTML of the view. */
    String view();

    /**
     * Run the tool. Gets the arguments as a JSON object and returns a JSON object: {@code {"text": "...",
     * "structuredContent": {...}}}, or {@code {"error": "..."}}. The model reads the text, the view gets the
     * structured content.
     */
    String call(String arguments);
}
