package at.yawk.weatheragent;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import ucar.nc2.grib.grib2.Grib2Record;
import ucar.nc2.grib.grib2.Grib2RecordScanner;
import ucar.unidata.io.RandomAccessFile;

/**
 * Decodes every GRIB2 message of one or more files with netCDF-Java and writes the values as consecutive
 * little-endian float32 arrays to one output file. Python reads that file in one go: handing Java arrays to numpy
 * directly is slow (element-wise) or crashes GraalPy (buffer protocol on ByteBuffers).
 *
 * <p>A header scan gives every message's size, hence its offset in the output, which is memory-mapped; messages
 * are then decoded in parallel (Python can't start threads) straight into their slots. Only one field per worker
 * is in memory at a time, whether the input is an ECMWF step file with hundreds of global fields or one small
 * file per DWD ensemble member.
 */
public final class GribDecoder {
    private static final ValueLayout.OfFloat LITTLE_ENDIAN_FLOAT =
        ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private GribDecoder() {
    }

    /** @return number of values of each message, in file order */
    public static int[] decodeToFile(String gribPath, String outputPath) throws IOException {
        return decodeFilesToFile(List.of(gribPath), outputPath);
    }

    /** @return number of values of each message, files in the given order, messages in file order */
    public static int[] decodeFilesToFile(List<String> gribPaths, String outputPath) throws IOException {
        // Copy out of the (possibly Python-backed) list before any worker thread touches it.
        List<String> paths = List.copyOf(gribPaths);
        List<Message> messages = new ArrayList<>();
        long offset = 0;
        for (String path : paths) {
            try (RandomAccessFile raf = new RandomAccessFile(path, "r")) {
                Grib2RecordScanner scanner = new Grib2RecordScanner(raf);
                while (scanner.hasNext()) {
                    Grib2Record record = scanner.next();
                    int points = record.getGDSsection().getNumberPoints();
                    messages.add(new Message(path, record, points, offset));
                    offset += (long) points * Float.BYTES;
                }
            }
        }
        try (FileChannel out = FileChannel.open(Path.of(outputPath), StandardOpenOption.CREATE,
                 StandardOpenOption.READ, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING);
             Arena arena = Arena.ofShared()) {
            if (offset > 0) {
                MemorySegment output = out.map(FileChannel.MapMode.READ_WRITE, 0, offset, arena);
                try {
                    messages.parallelStream().forEach(message -> message.decodeInto(output));
                } catch (UncheckedIOException e) {
                    throw e.getCause();
                }
            }
        }
        return messages.stream().mapToInt(Message::points).toArray();
    }

    private record Message(String path, Grib2Record record, int points, long offset) {
        void decodeInto(MemorySegment output) {
            float[] values;
            try (RandomAccessFile raf = new RandomAccessFile(path, "r")) {
                values = record.readData(raf);
            } catch (IOException e) {
                throw new UncheckedIOException(new IOException(path + ": " + e.getMessage(), e));
            }
            if (values.length != points) {
                // e.g. a quasi-regular grid, expanded on decoding; none of our products use one
                throw new UncheckedIOException(new IOException(
                    path + ": decoded " + values.length + " values, header says " + points));
            }
            MemorySegment.copy(values, 0, output, LITTLE_ENDIAN_FLOAT, offset, values.length);
        }
    }
}
