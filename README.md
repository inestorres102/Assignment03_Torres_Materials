# Assignment 3: Parallel Matrix Multiplication and Scalability Analysis

Empirical study evaluating shared-memory parallelization using Java's `ExecutorService`, focusing on multithreaded Speedup \(S(p)\) and Efficiency \(E(p)\) across varying thread counts.

## 1. System and Hardware Environment

- **CPU:** 12th Gen Intel(R) Core(TM) i7-12650H (10 physical cores: 6 Performance + 4 Efficiency cores, 16 logical processors).
- **RAM:** 16 GB DDR4.
- **Operating System:** Windows 11 Home (64-bit).
- **Java Compiler:** `javac 27.0.1.0.3`
- **Java Runtime:** `java 27.0.1.0.3` (Oracle HotSpot 64-Bit Server VM).
- **Available Processors:** 16, as reported by `Runtime.getRuntime().availableProcessors()`.
- **Python Environment:** Python 3.13 (Pandas, Matplotlib and NumPy).

## 2. Experimental Protocol

- **Algorithmic Workload:** Equivalent dense \(O(n^3)\) matrix multiplication algorithm applied in both sequential and parallel implementations, using double-precision floating-point values.
- **Parallel Strategy:** Row-wise work distribution using Java's `ExecutorService` and a fixed thread pool. Each task computes one output row, avoiding concurrent writes to the same row.
- **Thread Scaling (\(p\)):** Tested with 1, 2, 4, 8, 12 and 16 threads, using a fixed matrix dimension of \(n = 500\). A sequential implementation is also used as the baseline.
- **Input Generation:** Matrices are generated using Java's pseudorandom number generator. For each measured repetition, seeds are set to `100L + repetition` and `200L + repetition`. The same input matrices are used by all configurations within each repetition.
- **Validation:** Random matrices, identity matrices and zero matrices are tested. Results are checked using absolute and relative numerical tolerances of \(10^{-9}\).
- **Timing Boundary:** Execution time is measured using Java's monotonic high-resolution timer (`System.nanoTime()`). Input generation is excluded from the timed section. The sequential measurement includes output matrix allocation and multiplication. The parallel measurement additionally includes output allocation, executor creation, task submission, synchronization and executor termination. Result consumption is performed after timing.
- **Warm-up and Repetitions:** Three unrecorded warm-up executions are performed for the sequential implementation and for each parallel thread-count configuration. Each configuration is then measured over 10 repetitions. The execution order is rotated between repetitions to reduce potential ordering effects.
- **Performance Metrics:** Median execution time and interquartile range (IQR) are reported. Speedup is calculated as \(S(p) = T_{seq}/T_p\), and efficiency as \(E(p) = S(p)/p\).
- **Memory Metric:** Estimated matrix storage for three \(n \times n\) double-precision matrices. This is an array-payload estimate, not a measurement of peak process memory or total JVM memory usage. It excludes Java object overhead, executor structures and other runtime allocations.

## 3. Experimental Results

The following results were obtained for matrices of dimension \(500 \times 500\), using 10 measured repetitions per configuration.

| Method | Threads | Median Time (s) | Speedup | Efficiency |
|---|---:|---:|---:|---:|
| Sequential | 1 | 0.019156 | 1.000 | 1.000 |
| Parallel | 1 | 0.020815 | 0.920 | 0.920 |
| Parallel | 2 | 0.012115 | 1.581 | 0.791 |
| Parallel | 4 | 0.007042 | 2.720 | 0.680 |
| Parallel | 8 | 0.005295 | 3.618 | 0.452 |
| Parallel | 12 | 0.005044 | 3.797 | 0.316 |
| Parallel | 16 | 0.005365 | 3.571 | 0.223 |

The best measured speedup was obtained with 12 threads, reaching approximately 3.80. Increasing the thread count to 16 did not improve performance further.

The generated plots illustrate the relationship between thread count, speedup and parallel efficiency.

## 4. Build and Execution Instructions

### Compilation

```bash
javac MatrixParallel.java
```

### Execution

Run the Java benchmark to generate raw measurements (`results_parallel.csv`):

```bash
java MatrixParallel
```

Run the Python analysis script to calculate summary statistics, Speedup and Efficiency, and generate plots:

```bash
python plot_assignment3.py
```

## 5. Repository File Structure

```text
Assignment3/
├── .gitignore
├── README.md
├── MatrixParallel.java
├── plot_assignment3.py
├── results_parallel.csv
├── summary_parallel.csv
├── plot_speedup.png
└── plot_efficiency.png
```