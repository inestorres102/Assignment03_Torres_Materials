
import java.io.PrintWriter;
import java.util.Locale;

public class CsvExporter {

    // Write CSV header
    public static void writeHeader(PrintWriter writer) {
        writer.println(
                "method,n,threads,repetition,time_seconds,"
                + "estimated_matrix_memory_mb"
        );
    }

    // Write one benchmark result
    public static void writeResult(
            PrintWriter writer,
            String method,
            int n,
            int threads,
            int repetition,
            double timeSeconds,
            double matrixMemoryMB) {

        writer.printf(
                Locale.US,
                "%s,%d,%d,%d,%.9f,%.6f%n",
                method,
                n,
                threads,
                repetition,
                timeSeconds,
                matrixMemoryMB
        );

        writer.flush();
    }
}
