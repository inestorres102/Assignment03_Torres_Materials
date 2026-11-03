
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

public class MatrixOperations {

    private static volatile double resultSink = 0.0;

    // Generate a deterministic matrix
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

    // Sequential matrix multiplication
    public static double[][] multiplySequential(
            double[][] A, double[][] B, int n) {

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

    // Parallel matrix multiplication using row distribution
    public static double[][] multiplyParallel(
            double[][] A, double[][] B, int n, int numThreads)
            throws InterruptedException {

        double[][] C = new double[n][n];

        ExecutorService executor =
                Executors.newFixedThreadPool(numThreads);

        Future<?>[] futures = new Future<?>[n];

        try {
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

            for (Future<?> future : futures) {
                future.get();
            }

            if (!executor.awaitTermination(60, TimeUnit.SECONDS)) {
                executor.shutdownNow();
                throw new InterruptedException(
                        "Parallel execution timed out.");
            }

        } catch (InterruptedException e) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
            throw e;

        } catch (ExecutionException e) {
            executor.shutdownNow();
            throw new RuntimeException(
                    "A parallel task failed.", e);
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

    // Compare two matrices using numerical tolerances
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

    // Validate sequential and parallel implementations
    public static boolean checkCorrectness(int n)
            throws InterruptedException {

        double[][] A = generateMatrix(n, 42L);
        double[][] B = generateMatrix(n, 84L);

        double[][] cSeq = multiplySequential(A, B, n);
        double[][] cPar = multiplyParallel(A, B, n, 2);

        double absTol = 1e-9;
        double relTol = 1e-9;

        return matricesEqual(cSeq, cPar, n, absTol, relTol);
    }

    // Check identity and zero matrix cases
    public static boolean checkSpecialCases(int n)
            throws InterruptedException {

        double[][] A = generateMatrix(n, 123L);

        double[][] identity = new double[n][n];
        double[][] zeros = new double[n][n];

        for (int i = 0; i < n; i++) {
            identity[i][i] = 1.0;
        }

        double[][] identitySeq =
                multiplySequential(A, identity, n);

        double[][] zerosSeq =
                multiplySequential(A, zeros, n);

        double[][] identityPar =
                multiplyParallel(A, identity, n, 2);

        double[][] zerosPar =
                multiplyParallel(A, zeros, n, 2);

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
}
