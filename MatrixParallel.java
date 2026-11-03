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
    public static boolean matricesEqual(double[][] A, double[][] B, int n, double epsilon) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (Math.abs(A[i][j] - B[i][j]) > epsilon) {
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

        double epsilon = 1e-9;

        return matricesEqual(cSeq, cPar, n, epsilon);
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

        double epsilon = 1e-9;

        // Expected results: A * I = A and A * 0 = 0
        boolean identityCorrect = matricesEqual(A, identitySeq, n, epsilon)
                && matricesEqual(A, identityPar, n, epsilon);

        boolean zerosCorrect = matricesEqual(zeros, zerosSeq, n, epsilon)
                && matricesEqual(zeros, zerosPar, n, epsilon);

        return identityCorrect && zerosCorrect;
    }

    // 5. Benchmark Harness
    public static void main(String[] args) {
        try {
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

            int n = 500; // Fixed feasible matrix size for thread scaling comparison

            double matrixMemoryMB =
                    (3.0 * n * n * Double.BYTES) / (1024.0 * 1024.0);

            int[] threadCounts = {1, 2, 4, 8, 12, 16}; // Adjust to your CPU's physical/logical cores
            int repetitions = 5;
            String csvFile = "results_parallel.csv";

            PrintWriter writer = new PrintWriter(new FileWriter(csvFile, false));
            writer.println("method,n,threads,repetition,time_seconds,estimated_matrix_memory_mb");

            System.out.printf("\n--- Running Parallel Matrix Multiplication (n = %d) ---\n", n);

            // Discarded warm-up runs
            double[][] Aw = generateMatrix(n, 1L);
            double[][] Bw = generateMatrix(n, 2L);

            // Warm-up sequential implementation
            for (int w = 0; w < 3; w++) {
                double[][] Cw = multiplySequential(Aw, Bw, n);
                consumeResult(Cw, n);
            }

            // Warm-up parallel implementation for every thread count
            for (int threads : threadCounts) {
                for (int w = 0; w < 3; w++) {
                    double[][] Cw = multiplyParallel(Aw, Bw, n, threads);
                    consumeResult(Cw, n);
                }
            }

            // Test Parallel configurations
            for (int threads : threadCounts) {
                for (int rep = 1; rep <= repetitions; rep++) {
                    double[][] A = generateMatrix(n, 100L + rep);
                    double[][] B = generateMatrix(n, 200L + rep);

                    long start = System.nanoTime();
                    double[][] C = multiplyParallel(A, B, n, threads);
                    long duration = System.nanoTime() - start;

                    consumeResult(C, n);

                    double timeSec = duration / 1.0e9;
                    double memMb = matrixMemoryMB;

                    writer.printf(Locale.US, "Parallel,%d,%d,%d,%.6f,%.4f\n", n, threads, rep, timeSec, memMb);
                    writer.flush();
                    
                    System.out.printf("Method: Parallel   | Threads: %2d | Rep: %d | Time: %.4f s\n", threads, rep, timeSec);
                }
            }

            // Test Sequential baseline
            System.out.println("\n--- Running Sequential Baseline ---");
            for (int rep = 1; rep <= repetitions; rep++) {
                double[][] A = generateMatrix(n, 100L + rep);
                double[][] B = generateMatrix(n, 200L + rep);

                long start = System.nanoTime();
                double[][] C = multiplySequential(A, B, n);
                long duration = System.nanoTime() - start;

                consumeResult(C, n);

                double timeSec = duration / 1.0e9;
                double memMb = matrixMemoryMB;

                writer.printf(Locale.US, "Sequential,%d,1,%d,%.6f,%.4f\n", n, rep, timeSec, memMb);
                writer.flush();
                
                System.out.printf("Method: Sequential | Threads:  1 | Rep: %d | Time: %.4f s\n", rep, timeSec);
            }

            writer.close();
            System.out.println("\n[Java] Parallel benchmark completed successfully. Data saved to " + csvFile);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}