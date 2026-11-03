
import java.io.FileWriter;
import java.io.PrintWriter;

public class BenchmarkRunner {

    public static void main(String[] args) {

        String csvFile = "results/results_parallel_multi_n.csv";

        int[] matrixSizes = {250, 500, 750, 1000};
        int[] threadCounts = {1, 2, 4, 8, 12, 16};
        int[] configurations = {0, 1, 2, 4, 8, 12, 16};

        int repetitions = 10;
        int warmupRuns = 3;

        try {
            // Correctness validation
            if (!MatrixOperations.checkCorrectness(50)) {
                System.err.println(
                        "Error: Random matrix validation failed.");
                return;
            }

            System.out.println(
                    "Random matrix validation successful.");

            if (!MatrixOperations.checkSpecialCases(50)) {
                System.err.println(
                        "Error: Identity or zero matrix validation failed.");
                return;
            }

            System.out.println(
                    "Identity and zero matrix validation successful.");

            System.out.println(
                    "Available processors: "
                    + Runtime.getRuntime().availableProcessors());

            try (PrintWriter writer =
                    new PrintWriter(new FileWriter(csvFile, false))) {

                CsvExporter.writeHeader(writer);

                // Run the experiment for every matrix size
                for (int n : matrixSizes) {

                    double matrixMemoryMB =
                            (3.0 * n * n * Double.BYTES)
                            / (1024.0 * 1024.0);

                    System.out.printf(
                            "\n--- Matrix size: %d x %d ---\n", n, n);

                    // Generate warm-up inputs
                    double[][] Aw =
                            MatrixOperations.generateMatrix(n, 1L);

                    double[][] Bw =
                            MatrixOperations.generateMatrix(n, 2L);

                    // Sequential warm-up
                    for (int w = 0; w < warmupRuns; w++) {
                        double[][] Cw =
                                MatrixOperations.multiplySequential(
                                        Aw, Bw, n);

                        MatrixOperations.consumeResult(Cw, n);
                    }

                    // Parallel warm-up
                    for (int threads : threadCounts) {
                        for (int w = 0; w < warmupRuns; w++) {
                            double[][] Cw =
                                    MatrixOperations.multiplyParallel(
                                            Aw, Bw, n, threads);

                            MatrixOperations.consumeResult(Cw, n);
                        }
                    }

                    // Measured repetitions
                    for (int rep = 1; rep <= repetitions; rep++) {

                        long seedA = 1000L * n + rep;
                        long seedB = 2000L * n + rep;

                        double[][] A =
                                MatrixOperations.generateMatrix(n, seedA);

                        double[][] B =
                                MatrixOperations.generateMatrix(n, seedB);

                        // Rotate configuration order
                        int offset =
                                (rep - 1) % configurations.length;

                        for (int index = 0;
                                index < configurations.length;
                                index++) {

                            int config = configurations[
                                    (index + offset) % configurations.length
                            ];

                            String method;
                            int threads;
                            double[][] C;

                            long start = System.nanoTime();

                            if (config == 0) {
                                method = "Sequential";
                                threads = 1;

                                C = MatrixOperations.multiplySequential(
                                        A, B, n);
                            } else {
                                method = "Parallel";
                                threads = config;

                                C = MatrixOperations.multiplyParallel(
                                        A, B, n, threads);
                            }

                            long duration = System.nanoTime() - start;

                            // Consume result outside timed region
                            MatrixOperations.consumeResult(C, n);

                            double timeSec = duration / 1.0e9;

                            CsvExporter.writeResult(
                                    writer,
                                    method,
                                    n,
                                    threads,
                                    rep,
                                    timeSec,
                                    matrixMemoryMB
                            );

                            System.out.printf(
                                    "n: %4d | Method: %-10s | "
                                    + "Threads: %2d | Rep: %2d | "
                                    + "Time: %.6f s%n",
                                    n,
                                    method,
                                    threads,
                                    rep,
                                    timeSec
                            );
                        }
                    }
                }
            }

            System.out.println(
                    "\n[Java] Multi-size benchmark completed successfully.");

            System.out.println("Data saved to " + csvFile);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
