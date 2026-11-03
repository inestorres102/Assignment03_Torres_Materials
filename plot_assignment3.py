import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set publication-quality plot style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
})

csv_file = "results_parallel.csv"
if not os.path.exists(csv_file):
    print(f"Error: '{csv_file}' not found.")
    exit(1)

df = pd.read_csv(csv_file)

# 1. Aggregate Statistics (Median and IQR for time, Median for memory)
summary = df.groupby(["method", "n", "threads"]).agg(
    time_median=("time_seconds", "median"),
    time_iqr=("time_seconds", lambda x: np.percentile(x, 75) - np.percentile(x, 25)),
    mem_median=("estimated_matrix_memory_mb", "median")
).reset_index()

# 2. Isolate Sequential Baseline (T_seq)
seq_data = summary[summary["method"] == "Sequential"]
if seq_data.empty:
    print("Error: No 'Sequential' baseline found in data.")
    exit(1)
    
t_seq = seq_data["time_median"].values[0]
n_val = seq_data["n"].values[0]

# 3. Calculate Speedup S(p) and Efficiency E(p)
# S(p) = T_seq / T_p
# E(p) = S(p) / p
summary["speedup"] = t_seq / summary["time_median"]
summary["efficiency"] = summary["speedup"] / summary["threads"]

print(f"\n=== SUMMARY STATISTICS: PARALLEL MULTIPLICATION (n = {n_val}) ===")
print(summary[["method", "threads", "time_median", "time_iqr", "speedup", "efficiency"]].to_string(index=False))
summary.to_csv("summary_parallel.csv", index=False)

# Filter parallel data for plotting
par_data = summary[summary["method"] == "Parallel"].sort_values("threads")
threads = par_data["threads"].values
speedups = par_data["speedup"].values
efficiencies = par_data["efficiency"].values

# 4. PLOT 1: Speedup S(p)
plt.figure(figsize=(7, 5))
plt.plot(threads, speedups, marker='o', color='#1f77b4', linewidth=2, label="Measured Speedup $S(p)$")
plt.plot(threads, threads, linestyle='--', color='gray', label="Ideal Speedup ($S(p) = p$)")

plt.xlabel("Number of Threads ($p$)", fontweight="bold")
plt.ylabel("Speedup", fontweight="bold")
plt.title(f"Parallel Speedup vs. Thread Count ($n={n_val}$)", fontweight="bold")
plt.xticks(threads)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig("plot_speedup.png", dpi=300)
plt.close()

# 5. PLOT 2: Efficiency E(p)
plt.figure(figsize=(7, 5))
plt.plot(threads, efficiencies, marker='s', color='#2ca02c', linewidth=2, label="Measured Efficiency $E(p)$")
plt.axhline(y=1.0, color='gray', linestyle='--', label="Ideal Efficiency ($E(p) = 1.0$)")

plt.xlabel("Number of Threads ($p$)", fontweight="bold")
plt.ylabel("Efficiency", fontweight="bold")
plt.title(f"Parallel Efficiency vs. Thread Count ($n={n_val}$)", fontweight="bold")
plt.xticks(threads)
plt.ylim(0, 1.1) # Efficiency max is ideally 1.0
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig("plot_efficiency.png", dpi=300)
plt.close()

print("\nExported: 'summary_parallel.csv'")
print("Exported: 'plot_speedup.png'")
print("Exported: 'plot_efficiency.png'")