
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
})

csv_file = "results_parallel_multi_n.csv"
output_dir = "plots"

if not os.path.exists(csv_file):
    print(f"Error: '{csv_file}' not found.")
    exit(1)

# Create output directory
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_file)

# --------------------------------------------------
# 2. Calculate statistics
# --------------------------------------------------

summary = df.groupby(["method", "n", "threads"]).agg(
    time_median=("time_seconds", "median"),
    time_q1=("time_seconds", lambda x: np.percentile(x, 25)),
    time_q3=("time_seconds", lambda x: np.percentile(x, 75)),
    time_iqr=("time_seconds", lambda x:
              np.percentile(x, 75) - np.percentile(x, 25)),
    mem_median=("estimated_matrix_memory_mb", "median")
).reset_index()

# Obtain the sequential baseline for each matrix size
seq_data = summary[summary["method"] == "Sequential"][
    ["n", "time_median"]
].rename(columns={"time_median": "time_sequential"})

summary = summary.merge(seq_data, on="n", how="left")

if summary["time_sequential"].isna().any():
    print("Error: Missing sequential baseline.")
    exit(1)

# Calculate speedup and efficiency
summary["speedup"] = (
    summary["time_sequential"] / summary["time_median"]
)

summary["efficiency"] = (
    summary["speedup"] / summary["threads"]
)

# Export summary
summary.to_csv("summary_parallel_multi_n.csv", index=False)

print("\n=== SUMMARY STATISTICS ===")
print(summary[
    ["method", "n", "threads", "time_median",
     "time_iqr", "speedup", "efficiency"]
].to_string(index=False))

# --------------------------------------------------
# 3. Generate plots for each matrix size
# --------------------------------------------------

matrix_sizes = sorted(summary["n"].unique())
par_data = summary[summary["method"] == "Parallel"].copy()

for n in matrix_sizes:

    # Create folder for this matrix size
    size_dir = os.path.join(output_dir, f"n_{n}")
    os.makedirs(size_dir, exist_ok=True)

    data = par_data[par_data["n"] == n].sort_values("threads")

    threads = data["threads"].values
    times = data["time_median"].values
    speedups = data["speedup"].values
    efficiencies = data["efficiency"].values

    seq_time = summary[
        (summary["method"] == "Sequential") &
        (summary["n"] == n)
    ]["time_median"].iloc[0]

    # ----------------------------------------------
    # PLOT 1: Runtime with IQR error bars
    # ----------------------------------------------

    errors = np.array([
        times - data["time_q1"].values,
        data["time_q3"].values - times
    ])

    plt.figure(figsize=(8, 5))

    plt.errorbar(
        threads,
        times,
        yerr=errors,
        marker='o',
        capsize=4,
        linewidth=2,
        label="Parallel execution"
    )

    plt.axhline(
        y=seq_time,
        linestyle='--',
        color='gray',
        label="Sequential baseline"
    )

    plt.xlabel("Number of Threads (p)", fontweight="bold")
    plt.ylabel("Execution Time (seconds)", fontweight="bold")
    plt.title(f"Execution Time (n = {n})", fontweight="bold")
    plt.xticks(threads)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(size_dir, "runtime.png"), dpi=300)
    plt.close()

    # ----------------------------------------------
    # PLOT 2: Speedup
    # ----------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        threads,
        speedups,
        marker='o',
        linewidth=2,
        label="Measured speedup"
    )

    plt.plot(
        threads,
        threads,
        linestyle='--',
        color='gray',
        label="Ideal speedup"
    )

    plt.xlabel("Number of Threads (p)", fontweight="bold")
    plt.ylabel("Speedup", fontweight="bold")
    plt.title(f"Parallel Speedup (n = {n})", fontweight="bold")
    plt.xticks(threads)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(size_dir, "speedup.png"), dpi=300)
    plt.close()

    # ----------------------------------------------
    # PLOT 3: Efficiency
    # ----------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        threads,
        efficiencies,
        marker='s',
        linewidth=2,
        label="Measured efficiency"
    )

    plt.axhline(
        y=1.0,
        color='gray',
        linestyle='--',
        label="Ideal efficiency"
    )

    plt.xlabel("Number of Threads (p)", fontweight="bold")
    plt.ylabel("Efficiency", fontweight="bold")
    plt.title(f"Parallel Efficiency (n = {n})", fontweight="bold")
    plt.xticks(threads)
    plt.ylim(bottom=0)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(size_dir, "efficiency.png"), dpi=300)
    plt.close()

    print(f"Generated plots for n = {n}")

# --------------------------------------------------
# 4. Memory plot
# --------------------------------------------------

memory_data = (
    summary[["n", "mem_median"]]
    .drop_duplicates()
    .sort_values("n")
)

plt.figure(figsize=(8, 5))

plt.plot(
    memory_data["n"],
    memory_data["mem_median"],
    marker='o',
    linewidth=2
)

plt.xlabel("Matrix Dimension (n)", fontweight="bold")
plt.ylabel("Estimated Memory (MB)", fontweight="bold")
plt.title("Estimated Matrix Memory vs. Matrix Size", fontweight="bold")
plt.xticks(matrix_sizes)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "memory.png"), dpi=300)
plt.close()



# --------------------------------------------------
# 4. Comparative plots: all matrix sizes
# --------------------------------------------------

# Runtime comparison
plt.figure(figsize=(9, 6))

for n in matrix_sizes:
    data = par_data[par_data["n"] == n].sort_values("threads")

    plt.plot(
        data["threads"],
        data["time_median"],
        marker='o',
        linewidth=2,
        label=f"n = {n}"
    )

plt.xlabel("Number of Threads (p)", fontweight="bold")
plt.ylabel("Execution Time (seconds)", fontweight="bold")
plt.title("Execution Time Comparison", fontweight="bold")
plt.xticks(sorted(par_data["threads"].unique()))
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="Matrix size")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "runtime_comparison.png"), dpi=300)
plt.close()

# Speedup comparison
plt.figure(figsize=(9, 6))

for n in matrix_sizes:
    data = par_data[par_data["n"] == n].sort_values("threads")

    plt.plot(
        data["threads"],
        data["speedup"],
        marker='o',
        linewidth=2,
        label=f"n = {n}"
    )

plt.plot(
    sorted(par_data["threads"].unique()),
    sorted(par_data["threads"].unique()),
    linestyle='--',
    color='gray',
    label="Ideal speedup"
)

plt.xlabel("Number of Threads (p)", fontweight="bold")
plt.ylabel("Speedup", fontweight="bold")
plt.title("Speedup Comparison", fontweight="bold")
plt.xticks(sorted(par_data["threads"].unique()))
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "speedup_comparison.png"), dpi=300)
plt.close()

# Efficiency comparison
plt.figure(figsize=(9, 6))

for n in matrix_sizes:
    data = par_data[par_data["n"] == n].sort_values("threads")

    plt.plot(
        data["threads"],
        data["efficiency"],
        marker='s',
        linewidth=2,
        label=f"n = {n}"
    )

plt.axhline(
    y=1.0,
    color='gray',
    linestyle='--',
    label="Ideal efficiency"
)

plt.xlabel("Number of Threads (p)", fontweight="bold")
plt.ylabel("Efficiency", fontweight="bold")
plt.title("Efficiency Comparison", fontweight="bold")
plt.xticks(sorted(par_data["threads"].unique()))
plt.ylim(bottom=0)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "efficiency_comparison.png"), dpi=300)
plt.close()

print("Generated comparative plots for all matrix sizes")



# --------------------------------------------------
# 5. Finish
# --------------------------------------------------

print("\nAll plots generated successfully!")
print(f"Output directory: '{output_dir}'")
print("Summary exported: 'summary_parallel_multi_n.csv'")
