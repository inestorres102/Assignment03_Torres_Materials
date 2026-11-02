# Assignment 3: Parallel Matrix Multiplication and Scalability Analysis

Empirical study evaluating shared-memory parallelization using Java's ExecutorService, focusing on multithreaded Speedup ($S(p)$) and Efficiency ($E(p)$) across varying thread counts.

## 1. System and Hardware Environment
- **CPU:** Intel Core i7-12650H (10 Physical Cores: 6P+4E, 16 Logical Threads)
- **RAM:** 16.0 GB DDR4
- **Operating System:** Windows 11 Home (64-bit)
- **Runtime Environment:** Java 21 LTS (Oracle HotSpot 64-Bit Server VM)
- **Python Environment:** Python 3.13.14 (Pandas, Matplotlib, NumPy)

## 2. Experimental Protocol
- **Algorithmic Workload:** Equivalent dense $O(n^3)$ matrix multiplication algorithm applied in both sequential and parallel implementations.
- **Parallel Strategy:** Row-wise distribution utilizing Java's `ExecutorService` and `FixedThreadPool`.
- **Thread Scaling ($p$):** Tested across 1, 2, 4, 8, 12, and 16 threads for a fixed matrix dimension of $n = 500$.
- **Validation:** Both sequential and parallel kernels verified against identity matrices ($A \times I = A$) with a numerical tolerance of $\varepsilon = 10^{-9}$.
- **Timing Boundary:** High-resolution monotonic timers isolating only the parallel/sequential execution kernels, excluding memory allocation and thread pool initialization where possible.
- **Repetitions:** 5 recorded repetitions per configuration following unrecorded JVM warm-up cycles.

## 3. Build and Execution Instructions

### Compilation
```bash
javac MatrixParallel.java
```

### Execution
Run the Java benchmark to generate raw measurements (results_parallel.csv):

```bash
java MatrixParallel
```
Run the Python analysis script to compute Speedup/Efficiency and generate plots:

```bash
python plot_assignment3.py
```

## 4. Repository File Structure

Assignment3/
├── .gitignore               # Excludes compiled .class files and caches
├── README.md                # System specs and execution guide
├── MatrixParallel.java      # Java benchmark (Sequential baseline + Multithreaded kernel)
├── plot_assignment3.py      # Python analysis script
├── results_parallel.csv     # Raw measured data (Time and Memory per thread/repetition)
├── summary_parallel.csv     # Computed statistics (Medians, Speedup, Efficiency)
├── plot_speedup.png         # S(p) scaling plot
└── plot_efficiency.png      # E(p) scaling plot