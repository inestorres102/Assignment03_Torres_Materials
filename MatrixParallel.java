import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;


public class MatrixParallel {

    // Prevent the result from being ignored by the JVM
    private static volatile double resultSink = 0.0;

    // 1. Deterministic Matrix Generator
    public static double[][] generateMatrix(int n, long seed) {
        Random rand = new Random(seed);
        double[][] matrix = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                matrix[i][j] = rand.nextDouble();
            }
        }
        return matrix;
    }

    // 2. Sequential Baseline
    public static double[][] multiplySequential(double[][] A, double[][] B, int n) {
        double[][] C = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                double a = A[i][k];
                for (int j = 0; j < n; j++) {
                    C[i][j] += a * B[k][j];
                }
            }
        }
        return C;
    }

    // 3. Parallel Implementation (Row Distribution)
    public static double[][] multiplyParallel(double[][] A, double[][] B, int n, int numThreads)
            throws InterruptedException {

        double[][] C = new double[n][n];

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);

        Future<?>[] futures = new Future<?>[n];

        for (int i = 0; i < n; i++) {
            final int row = i;

            futures[i] = executor.submit(() -> {
                for (int k = 0; k < n; k++) {
                    double a = A[row][k];

                    for (int j = 0; j < n; j++) {
                        C[row][j] += a * B[k][j];
                    }
                }
            });
        }

        executor.shutdown();

        try {
            for (Future<?> future : futures) {
                future.get();
            }

            if (!executor.awaitTermination(60, TimeUnit.SECONDS)) {
                executor.shutdownNow();
                throw new InterruptedException("Parallel execution timed out.");
            }

        } catch (java.util.concurrent.ExecutionException e) {
            executor.shutdownNow();
            throw new RuntimeException("A parallel task failed.", e);
        }

        return C;
    }


    // Consume the result outside the timed region
    public static void consumeResult(double[][] C, int n) {
        double checksum = 0.0;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                checksum += C[i][j];
            }
        }

        resultSink = checksum;
    }

    // Compare two matrices using a numerical tolerance
    public static boolean matricesEqual(
            double[][] A,
            double[][] B,
            int n,
            double absTol,
            double relTol) {

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {

                double a = A[i][j];
                double b = B[i][j];

                if (!Double.isFinite(a) || !Double.isFinite(b)) {
                    return false;
                }

                double difference = Math.abs(a - b);
                double tolerance = absTol + relTol * Math.abs(a);

                if (difference > tolerance) {
                    return false;
                }
            }
        }

        return true;
    }

    // 4. Correctness Check
    public static boolean checkCorrectness(int n) throws InterruptedException {
        double[][] A = generateMatrix(n, 42L);
        double[][] B = generateMatrix(n, 84L);

        double[][] cSeq = multiplySequential(A, B, n);
        double[][] cPar = multiplyParallel(A, B, n, 2);

        double absTol = 1e-9;
        double relTol = 1e-9;

        return matricesEqual(cSeq, cPar, n, absTol, relTol);
    }

    // Check multiplication with identity and zero matrices
    public static boolean checkSpecialCases(int n) throws InterruptedException {
        double[][] A = generateMatrix(n, 123L);

        double[][] identity = new double[n][n];
        double[][] zeros = new double[n][n];

        // Build identity matrix
        for (int i = 0; i < n; i++) {
            identity[i][i] = 1.0;
        }

        // Sequential results
        double[][] identitySeq = multiplySequential(A, identity, n);
        double[][] zerosSeq = multiplySequential(A, zeros, n);

        // Parallel results
        double[][] identityPar = multiplyParallel(A, identity, n, 2);
        double[][] zerosPar = multiplyParallel(A, zeros, n, 2);

        double absTol = 1e-9;
        double relTol = 1e-9;

        boolean identityCorrect =
                matricesEqual(A, identitySeq, n, absTol, relTol)
                && matricesEqual(A, identityPar, n, absTol, relTol);

        boolean zerosCorrect =
                matricesEqual(zeros, zerosSeq, n, absTol, relTol)
                && matricesEqual(zeros, zerosPar, n, absTol, relTol);

        return identityCorrect && zerosCorrect;
    }

    // 5. Benchmark Harness
    public static void main(String[] args) {
        String csvFile = "results_parallel_multi_n.csv";

        int[] matrixSizes = {250, 500, 750, 1000};
        int[] threadCounts = {1, 2, 4, 8, 12, 16};
        int[] configurations = {0, 1, 2, 4, 8, 12, 16};

        int repetitions = 10;
        int warmupRuns = 3;

        try {
            // Correctness validation
            if (!checkCorrectness(50)) {
                System.err.println("Error: Random matrix validation failed.");
                return;
            }
            System.out.println("Random matrix validation successful.");

            if (!checkSpecialCases(50)) {
                System.err.println("Error: Identity or zero matrix validation failed.");
                return;
            }
            System.out.println("Identity and zero matrix validation successful.");

            System.out.println("Available processors: "
                    + Runtime.getRuntime().availableProcessors());

            try (PrintWriter writer =
                        new PrintWriter(new FileWriter(csvFile, false))) {

                writer.println(
                        "method,n,threads,repetition,time_seconds,"
                        + "estimated_matrix_memory_mb"
                );

                // Run the experiment for every matrix size
                for (int n : matrixSizes) {

                    double matrixMemoryMB =
                            (3.0 * n * n * Double.BYTES)
                            / (1024.0 * 1024.0);

                    System.out.printf(
                            "\n--- Matrix size: %d x %d ---\n", n, n
                    );

                    // Generate warm-up inputs
                    double[][] Aw = generateMatrix(n, 1L);
                    double[][] Bw = generateMatrix(n, 2L);

                    // Sequential warm-up
                    for (int w = 0; w < warmupRuns; w++) {
                        double[][] Cw =
                                multiplySequential(Aw, Bw, n);
                        consumeResult(Cw, n);
                    }

                    // Parallel warm-up for every thread count
                    for (int threads : threadCounts) {
                        for (int w = 0; w < warmupRuns; w++) {
                            double[][] Cw =
                                    multiplyParallel(Aw, Bw, n, threads);
                            consumeResult(Cw, n);
                        }
                    }

                    // Measured repetitions
                    for (int rep = 1; rep <= repetitions; rep++) {

                        // Same input matrices for all configurations
                        // within this repetition
                        long seedA = 1000L * n + rep;
                        long seedB = 2000L * n + rep;

                        double[][] A = generateMatrix(n, seedA);
                        double[][] B = generateMatrix(n, seedB);

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

                                C = multiplySequential(A, B, n);
                            } else {
                                method = "Parallel";
                                threads = config;

                                C = multiplyParallel(A, B, n, threads);
                            }

                            long duration = System.nanoTime() - start;

                            // Consume result outside timed region
                            consumeResult(C, n);

                            double timeSec = duration / 1.0e9;

                            writer.printf(
                                    Locale.US,
                                    "%s,%d,%d,%d,%.9f,%.6f%n",
                                    method,
                                    n,
                                    threads,
                                    rep,
                                    timeSec,
                                    matrixMemoryMB
                            );

                            writer.flush();

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
                    "\n[Java] Multi-size benchmark completed successfully."
            );
            System.out.println("Data saved to " + csvFile);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}