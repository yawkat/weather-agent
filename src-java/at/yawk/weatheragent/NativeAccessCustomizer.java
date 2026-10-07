package at.yawk.weatheragent;

import io.micronaut.context.python.GraalPyContextCustomizer;
import org.graalvm.polyglot.Context;

/**
 * Lets GraalPy load native extensions (numpy). Registered through META-INF/services so it also applies to
 * GraalPy contexts that don't read application configuration, such as the one pytest collection runs in.
 */
public final class NativeAccessCustomizer implements GraalPyContextCustomizer {
    @Override
    public void customize(Context.Builder builder) {
        builder.allowNativeAccess(true);
    }
}
