package at.yawk.weatheragent;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

/**
 * Runs the sandboxed SQL worker (src/weather_agent/sql_engine.py builds the command) with a deadline and a cap
 * on how many run at once. Output goes to files, so nothing has to drain pipes while we wait.
 */
public final class WorkerProcess {
    /** {@link #run} result: killed at the deadline. */
    public static final int TIMEOUT = -1;
    /** {@link #run} result: no slot became free within the timeout. */
    public static final int BUSY = -2;

    private final Semaphore slots;

    public WorkerProcess(int maxConcurrent) {
        slots = new Semaphore(maxConcurrent, true);
    }

    /** The exit code (128 + signal for a signal), {@link #TIMEOUT} or {@link #BUSY}. */
    public int run(List<String> command, String stdout, String stderr, long timeoutMillis)
        throws IOException, InterruptedException {
        if (!slots.tryAcquire(timeoutMillis, TimeUnit.MILLISECONDS)) {
            return BUSY;
        }
        try {
            Process process = new ProcessBuilder(command)
                .redirectInput(ProcessBuilder.Redirect.from(new File("/dev/null")))
                .redirectOutput(new File(stdout))
                .redirectError(new File(stderr))
                .start();
            try {
                if (!process.waitFor(timeoutMillis, TimeUnit.MILLISECONDS)) {
                    return TIMEOUT;
                }
                return process.exitValue();
            } finally {
                // bwrap --die-with-parent takes the sandboxed processes with it.
                if (process.isAlive()) {
                    process.destroyForcibly();
                    process.waitFor();
                }
            }
        } finally {
            slots.release();
        }
    }
}
