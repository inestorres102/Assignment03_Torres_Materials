# Parallel Matrix Multiplication and Scalability Analysis

**Assignment 3 – BIG DATA – Academic Year 2026–2027**  
Inés Torres Sánchez

## 1. Project Overview

This project investigates the effect of the number of worker threads on the execution time of dense square matrix multiplication.

A sequential implementation is compared with a parallel implementation using Java's `ExecutorService` and a fixed thread pool.

The main objective is to analyse how parallel execution affects runtime, speedup and parallel efficiency for different matrix sizes and thread configurations.

## 2. Hardware and Software Environment

### Hardware
- **Processor:** 12th Gen Intel Core i7-12650H
- **Physical cores:** 10 (6 Performance cores + 4 Efficient cores)
- **Logical processors:** 16
- **RAM:** 16 GB DDR4

### Software
- **Operating System:** Windows 11 Home 64-bit
- **Programming Language:** Java
- **Java compiler/runtime:** JDK 27.0.1.0.3
- **Data analysis:** Python 3.13
- **Python libraries:** NumPy, pandas and Matplotlib

## 3. Implementations

The project contains three Java source files:

- **`MatrixOperations.java`**: Matrix generation, sequential multiplication, parallel multiplication and correctness checks.
- **`CsvExporter.java`**: CSV header and measurement export functionality.
- **`BenchmarkRunner.java`**: Benchmark configuration, execution, timing and result collection.

Both implementations use:
- Dense square matrices stored as Java `double[][]` arrays.
- Double-precision floating-point values.
- The same conventional triple-loop multiplication algorithm (`i-k-j`).
- The same input matrices for sequential and parallel comparisons within each repetition.

### Parallel implementation

The parallel implementation distributes output matrix rows among worker threads using a fixed-size `ExecutorService` thread pool.

Each task computes one complete output row, ensuring that different tasks write to separate rows of the result matrix.

## 4. Experimental Methodology

### Matrix sizes

The benchmark evaluates four square matrix dimensions:

- 250 × 250
- 500 × 500
- 750 × 750
- 1000 × 1000

### Thread configurations

The sequential implementation is used as the baseline. The parallel implementation is tested with the following worker counts:

- 1
- 2
- 4
- 8
- 12
- 16

### Measurement protocol

- **Warm-up runs:** 3 per configuration.
- **Recorded repetitions:** 10 per configuration.
- **Timer:** `System.nanoTime()`.
- **Execution order:** Rotated between repetitions to reduce systematic ordering effects.
- **Input generation:** Deterministic pseudorandom values using fixed seeds.
- **Input consistency:** Sequential and parallel implementations use the same input matrices for each repetition.

The measured execution time includes result matrix allocation, thread pool creation, task submission, computation and task completion. Result consumption and CSV writing are performed outside the timed region.

The benchmark records individual measurements in CSV format. The analysis script calculates the median and interquartile range (IQR) for each configuration.

### Metrics

The following metrics are analysed:

- **Execution time:** Time required to complete matrix multiplication.
- **Speedup:** Ratio between sequential execution time and parallel execution time.

  `Speedup(p) = T_seq / T_parallel(p)`

- **Parallel efficiency:** Speedup divided by the number of worker threads.

  `Efficiency(p) = Speedup(p) / p`

- **Estimated matrix memory:** Approximate storage required for three dense matrices of double-precision values.

  `Memory = 3 × n² × 8 bytes`

This memory value is an estimate of matrix element storage, not a measurement of total process memory or JVM heap consumption. It excludes Java array-object overhead and other runtime allocations.

## 5. Correctness Validation

The implementation includes correctness checks comparing sequential and parallel multiplication results.

The validation includes:
- Comparisons using deterministic input matrices.
- Absolute and relative numerical tolerance of `1e-9`.
- Identity matrix tests.
- Zero matrix tests.

The parallel implementation also ensures that each task writes to a distinct output row.

## 6. Repository Structure

```text
Assignment3/
│
├── src/
│   ├── MatrixOperations.java
│   ├── CsvExporter.java
│   └── BenchmarkRunner.java
│
├── results/
│   ├── results_parallel_multi_n.csv
│   ├── summary_parallel_multi_n.csv
│   └── initial_n500/
│       ├── results_parallel.csv
│       └── summary_parallel.csv
│
├── plots/
|   ├── speedup_all_sizes.png
|   ├── runtime_all_sizes.png
|   ├── efficiency_all_sizes.png
|   ├── n_250/
|   │   ├── speedup.png
|   │   ├── runtime.png
|   │   └── efficiency.png
|   ├── n_500/
|   │   ├── speedup.png
|   │   ├── runtime.png
|   │   └── efficiency.png
|   ├── n_750/
|   │   ├── speedup.png
|   │   ├── runtime.png
|   │   └── efficiency.png
|   └── n_1000/
|       ├── speedup.png
|       ├── runtime.png
|       └── efficiency.png
│
├── plot_assignment3.py
├── README.md
└── .gitignore
```

The `initial_n500` directory preserves the original single-size experiment. The multi-size CSV files contain the updated benchmark results and are used for the main analysis.

Compiled Java `.class` files are generated locally in the `out/` directory. This directory is excluded from version control and is not required in the repository.

## 7. How to Build and Run

From the project root directory, create the output directory if it does not already exist:

```powershell
New-Item -ItemType Directory -Force out
```

Compile the Java source files:

```bash
javac -d out src/*.java
```

Run the benchmark:

```bash
java -cp out BenchmarkRunner
```

The benchmark generates the raw CSV file in the `results/` directory.

To regenerate the statistical summaries and figures, run:

```bash
python plot_assignment3.py
```

Python dependencies:

```bash
pip install numpy pandas matplotlib
```

## 8. Results and Reproducibility

The repository includes:

- Raw benchmark measurements.
- Statistical summaries containing median runtime, IQR, speedup and parallel efficiency.
- Figures generated from the experimental data.
- Source code and instructions to reproduce the benchmark and analysis.

The multi-size experiment contains 280 recorded measurements: 4 matrix sizes × 7 execution configurations × 10 repetitions.

All measurements and generated figures should be interpreted within the scope of the tested hardware, software and implementation.