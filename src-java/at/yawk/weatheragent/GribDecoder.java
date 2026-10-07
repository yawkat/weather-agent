package at.yawk.weatheragent;

import java.io.IOException;
import java.nio.ByteBuffer;
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
 * Decodes every GRIB2 message of a file with netCDF-Java and writes the values as consecutive little-endian
 * float32 arrays to one output file. Python reads that file in one go: handing Java arrays to numpy directly is
 * slow (element-wise) or crashes GraalPy (buffer protocol on ByteBuffers).
 */
public final class GribDecoder {
    private GribDecoder() {
    }

    /** @return number of values of each message, in file order */
    public static int[] decodeToFile(String gribPath, String outputPath) throws IOException {
        List<Integer> sizes = new ArrayList<>();
        try (RandomAccessFile raf = new RandomAccessFile(gribPath, "r");
             FileChannel out = FileChannel.open(Path.of(outputPath), StandardOpenOption.CREATE,
                 StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
            Grib2RecordScanner scanner = new Grib2RecordScanner(raf);
            while (scanner.hasNext()) {
                Grib2Record record = scanner.next();
                float[] values = record.readData(raf);
                ByteBuffer buffer = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
                buffer.asFloatBuffer().put(values);
                while (buffer.hasRemaining()) {
                    out.write(buffer);
                }
                sizes.add(values.length);
            }
        }
        return sizes.stream().mapToInt(Integer::intValue).toArray();
    }
}
