package at.yawk.weatheragent;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.sql.Array;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.sql.Statement;
import java.time.temporal.TemporalAccessor;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import org.duckdb.DuckDBAppender;
import org.duckdb.DuckDBConnection;

/**
 * One in-memory DuckDB database holding the samples of a single forecast query, which the client then
 * analyses with read-only SQL.
 *
 * Python prepares columns with numpy and writes them to files; this class bulk-loads them (row-wise appends
 * are fast in Java but would be millions of interop calls from Python). After loading, {@link #lock()} cuts
 * off file, network and extension access and freezes the configuration, so client SQL can only read the
 * tables.
 */
public final class SqlCube implements AutoCloseable {
    private static final ScheduledExecutorService CANCELLER = Executors.newSingleThreadScheduledExecutor(r -> {
        Thread t = new Thread(r, "sql-cube-cancel");
        t.setDaemon(true);
        return t;
    });

    private final DuckDBConnection connection;
    private boolean locked;

    public SqlCube(int memoryLimitMb, int threads) throws SQLException {
        connection = (DuckDBConnection) DriverManager.getConnection("jdbc:duckdb:");
        try (Statement s = connection.createStatement()) {
            s.execute("SET memory_limit = '" + memoryLimitMb + "MB'");
            s.execute("SET threads = " + threads);
            // Spilling to disk would bypass the memory limit; queries that don't fit just fail.
            s.execute("SET temp_directory = ''");
        }
    }

    /** Setup SQL (tables, views, macros); only before {@link #lock()}. */
    public void execute(String sql) throws SQLException {
        if (locked) {
            throw new IllegalStateException("locked");
        }
        try (Statement s = connection.createStatement()) {
            s.execute(sql);
        }
    }

    /**
     * Append rows to {@code table}, whose columns are {@code intColumns} INTEGER columns followed by
     * {@code floatColumns} FLOAT columns. The files hold little-endian int32 / float32 values, row-major.
     * NaN floats become NULL.
     */
    public void append(String table, int rows, int intColumns, String intPath, int floatColumns, String floatPath)
        throws SQLException, IOException {
        if (locked) {
            throw new IllegalStateException("locked");
        }
        ByteBuffer ints = read(intPath, (long) rows * intColumns * 4);
        ByteBuffer floats = read(floatPath, (long) rows * floatColumns * 4);
        try (DuckDBAppender appender = connection.createAppender(DuckDBConnection.DEFAULT_SCHEMA, table)) {
            for (int r = 0; r < rows; r++) {
                appender.beginRow();
                for (int c = 0; c < intColumns; c++) {
                    appender.append(ints.getInt());
                }
                for (int c = 0; c < floatColumns; c++) {
                    float v = floats.getFloat();
                    if (Float.isNaN(v)) {
                        appender.appendNull();
                    } else {
                        appender.append(v);
                    }
                }
                appender.endRow();
            }
        }
    }

    private static ByteBuffer read(String path, long expected) throws IOException {
        if (expected == 0) {
            return ByteBuffer.allocate(0);
        }
        try (FileChannel channel = FileChannel.open(Path.of(path), StandardOpenOption.READ)) {
            if (channel.size() != expected) {
                throw new IOException(path + ": expected " + expected + " bytes, found " + channel.size());
            }
            ByteBuffer buffer = ByteBuffer.allocate(Math.toIntExact(expected)).order(ByteOrder.LITTLE_ENDIAN);
            while (buffer.hasRemaining()) {
                if (channel.read(buffer) < 0) {
                    throw new IOException(path + ": unexpected end of file");
                }
            }
            return buffer.flip();
        }
    }

    /** Disable file, network and extension access and freeze the configuration. Irreversible. */
    public void lock() throws SQLException {
        try (Statement s = connection.createStatement()) {
            s.execute("SET enable_external_access = false");
            s.execute("SET autoinstall_known_extensions = false");
            s.execute("SET autoload_known_extensions = false");
            s.execute("SET allow_community_extensions = false");
            s.execute("SET lock_configuration = true");
        }
        locked = true;
    }

    /**
     * Run one client query and return {"columns": [...], "rows": [[...], ...], "truncated": bool} as JSON.
     * Only allowed after {@link #lock()}.
     */
    public String query(String sql, int maxRows, int timeoutMillis) throws SQLException {
        if (!locked) {
            throw new IllegalStateException("lock() before running client SQL");
        }
        try (Statement statement = connection.createStatement()) {
            ScheduledFuture<?> cancel = CANCELLER.schedule(() -> {
                try {
                    statement.cancel();
                } catch (SQLException ignored) {
                    // already finished
                }
            }, timeoutMillis, TimeUnit.MILLISECONDS);
            try (ResultSet rs = statement.executeQuery(sql)) {
                ResultSetMetaData meta = rs.getMetaData();
                int n = meta.getColumnCount();
                StringBuilder json = new StringBuilder("{\"columns\":[");
                for (int c = 1; c <= n; c++) {
                    if (c > 1) {
                        json.append(',');
                    }
                    string(json, meta.getColumnLabel(c));
                }
                json.append("],\"rows\":[");
                int count = 0;
                boolean truncated = false;
                while (rs.next()) {
                    if (count == maxRows) {
                        truncated = true;
                        break;
                    }
                    if (count++ > 0) {
                        json.append(',');
                    }
                    json.append('[');
                    for (int c = 1; c <= n; c++) {
                        if (c > 1) {
                            json.append(',');
                        }
                        value(json, rs.getObject(c));
                    }
                    json.append(']');
                }
                return json.append("],\"truncated\":").append(truncated).append('}').toString();
            } finally {
                cancel.cancel(false);
            }
        }
    }

    private static void value(StringBuilder json, Object v) throws SQLException {
        if (v == null) {
            json.append("null");
        } else if (v instanceof Float f) {
            number(json, f.doubleValue());
        } else if (v instanceof Double d) {
            number(json, d);
        } else if (v instanceof Number || v instanceof BigDecimal || v instanceof BigInteger) {
            json.append(v);
        } else if (v instanceof Boolean b) {
            json.append(b);
        } else if (v instanceof Array a) {
            json.append('[');
            Object[] items = (Object[]) a.getArray();
            for (int i = 0; i < items.length; i++) {
                if (i > 0) {
                    json.append(',');
                }
                value(json, items[i]);
            }
            json.append(']');
        } else if (v instanceof java.sql.Timestamp ts) {
            // TIMESTAMP columns hold local wall-clock time; ISO without seconds when they're zero.
            string(json, ts.toLocalDateTime().toString());
        } else if (v instanceof java.sql.Date d) {
            string(json, d.toLocalDate().toString());
        } else if (v instanceof TemporalAccessor || v instanceof java.util.Date) {
            string(json, v.toString());
        } else {
            string(json, v.toString());
        }
    }

    private static void number(StringBuilder json, double d) {
        if (Double.isNaN(d) || Double.isInfinite(d)) {
            json.append("null");
        } else {
            // Forecast values don't need more than 4 significant decimals.
            json.append(Math.round(d * 10000.0) / 10000.0);
        }
    }

    private static void string(StringBuilder json, String s) {
        json.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> json.append("\\\"");
                case '\\' -> json.append("\\\\");
                case '\n' -> json.append("\\n");
                case '\r' -> json.append("\\r");
                case '\t' -> json.append("\\t");
                default -> {
                    if (c < 0x20) {
                        json.append(String.format("\\u%04x", (int) c));
                    } else {
                        json.append(c);
                    }
                }
            }
        }
        json.append('"');
    }

    @Override
    public void close() throws SQLException {
        connection.close();
    }
}
