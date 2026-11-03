# Assignment 3: Parallel Matrix Multiplication and Scalability Analysis

Empirical study evaluating shared-memory parallelization using Java's `ExecutorService`, focusing on multithreaded Speedup \(S(p)\) and Efficiency \(E(p)\) across varying thread counts.

## 1. System and Hardware Environment

- **CPU:** 12th Gen Intel(R) Core(TM) i7-12650H (10 physical cores: 6 Performance + 4 Efficiency cores, 16 logical processors).
- **RAM:** 16 GB DDR4.
- **Operating System:** Windows 11 Home (64-bit).
- **Java Runtime:** Java 27.0.1.0.3, Oracle HotSpot 64-Bit Server VM.
- **Available Processors:** 16, as reported by `Runtime.getRuntime().availableProcessors()`.
- **Python Environment:** Python 3.13 (Pandas, Matplotlib and NumPy).

## 2. Experimental Protocol

- **Algorithmic Workload:** Equivalent dense \(O(n^3)\) matrix multiplication algorithm applied in both sequential and parallel implementations, using double-precision floating-point values.
- **Parallel Strategy:** Row-wise work distribution using Java's `ExecutorService` and a fixed thread pool. Each task computes one output row, avoiding concurrent writes to the same row.
- **Thread Scaling (\(p\)):** Tested with 1, 2, 4, 8, 12 and 16 threads, using a fixed matrix dimension of \(n = 500\).
- **Input Generation:** Matrices are generated using Java's pseudorandom number generator with fixed seeds. The same input matrices are used by the sequential and parallel implementations for each measured repetition.
- **Validation:** Random matrices, identity matrices and zero matrices are tested. Sequential and parallel results are compared using an absolute numerical tolerance of \(\varepsilon = 10^{-9}\).
- **Timing Boundary:** Execution time is measured using Java's monotonic high-resolution timer (`System.nanoTime()`). Input generation is excluded from the timed section. The sequential measurement includes output matrix allocation and multiplication. The parallel measurement additionally includes output allocation, executor creation, task submission, synchronization and executor termination. Result consumption is performed after timing.
- **Warm-up and Repetitions:** Three unrecorded warm-up executions per configuration, followed by five recorded repetitions per configuration.
- **Performance Metrics:** Median execution time and interquartile range (IQR) are reported. Speedup is calculated as \(S(p) = T_{seq}/T_p\), and efficiency as \(E(p) = S(p)/p\).
- **Memory Metric:** Estimated matrix storage for three \(n \times n\) double-precision matrices. This is an array-payload estimate, not a measurement of peak process memory or total JVM memory usage. It excludes Java object overhead, executor structures and other runtime allocations.

## 3. Build and Execution Instructions

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

## 4. Repository File Structure

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