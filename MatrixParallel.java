import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
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
    public static double[][] multiplyParallel(double[][] A, double[][] B, int n, int numThreads) throws InterruptedException {
        double[][] C = new double[n][n];
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);

        for (int i = 0; i < n; i++) {
            final int row = i;
            executor.submit(() -> {
                for (int k = 0; k < n; k++) {
                    double a = A[row][k];
                    for (int j = 0; j < n; j++) {
                        C[row][j] += a * B[k][j];
                    }
                }
            });
        }

        executor.shutdown();
        executor.awaitTermination(60, TimeUnit.SECONDS);
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


    // 4. Correctness Check
    public static boolean checkCorrectness(int n) throws InterruptedException {
        double[][] A = generateMatrix(n, 42L);
        double[][] B = generateMatrix(n, 84L);

        double[][] cSeq = multiplySequential(A, B, n);
        double[][] cPar = multiplyParallel(A, B, n, 2);

        double epsilon = 1e-9;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (Math.abs(cSeq[i][j] - cPar[i][j]) > epsilon) {
                    return false;
                }
            }
        }

        return true;
    }


    // 5. Benchmark Harness
    public static void main(String[] args) {
        try {
            if (!checkCorrectness(50)) {
                System.err.println("Error: Correctness validation failed. Tolerance exceeded.");
                return;
            }
            System.out.println("Validation successful: Sequential and Parallel results match.");

            int n = 500; // Fixed feasible matrix size for thread scaling comparison
            int[] threadCounts = {1, 2, 4, 8, 12, 16}; // Adjust to your CPU's physical/logical cores
            int repetitions = 5;
            String csvFile = "results_parallel.csv";

            PrintWriter writer = new PrintWriter(new FileWriter(csvFile, false));
            writer.println("method,n,threads,repetition,time_seconds,memory_mb");

            Runtime runtime = Runtime.getRuntime();

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

                    System.gc(); // Request garbage collection before memory baseline
                    long memBefore = runtime.totalMemory() - runtime.freeMemory();
                    
                    long start = System.nanoTime();
                    double[][] C = multiplyParallel(A, B, n, threads);
                    long duration = System.nanoTime() - start;
                    
                    // Consume result outside the timed region
                    consumeResult(C, n);

                    long memAfter = runtime.totalMemory() - runtime.freeMemory();
                    
                    double timeSec = duration / 1.0e9;
                    double memMb = Math.max(0, memAfter - memBefore) / (1024.0 * 1024.0);
                    if (memMb <= 0.01) {
                        memMb = (double) (n * n * 8L) / (1024.0 * 1024.0); // Fallback estimate
                    }

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

                System.gc();
                long memBefore = runtime.totalMemory() - runtime.freeMemory();
                
                long start = System.nanoTime();
                double[][] C = multiplySequential(A, B, n);
                long duration = System.nanoTime() - start;
                
                // Consume result outside the timed region
                consumeResult(C, n);

                long memAfter = runtime.totalMemory() - runtime.freeMemory();
                
                double timeSec = duration / 1.0e9;
                double memMb = Math.max(0, memAfter - memBefore) / (1024.0 * 1024.0);
                if (memMb <= 0.01) {
                    memMb = (double) (n * n * 8L) / (1024.0 * 1024.0);
                }

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