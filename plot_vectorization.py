import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

csv_file = "results_vectorized.csv"
if not os.path.exists(csv_file):
    print(f"Error: '{csv_file}' not found.")
    exit(1)

df = pd.read_csv(csv_file)

summary = df.groupby(["method", "n"]).agg(
    time_median=("time_seconds", "median"),
    time_iqr=("time_seconds", lambda x: np.percentile(x, 75) - np.percentile(x, 25))
).reset_index()

scalar_time = summary[summary["method"] == "Scalar"]["time_median"].values[0]
vec_time = summary[summary["method"] == "Vectorized"]["time_median"].values[0]
simd_speedup = scalar_time / vec_time

print("\n=== SIMD VECTORIZATION EXTENSION ===")
print(summary.to_string(index=False))
print(f"\n--> SIMD Speedup: {simd_speedup:.2f}x")

# Plotting
plt.figure(figsize=(6, 5))
methods = summary["method"].values
times = summary["time_median"].values
errors = summary["time_iqr"].values / 2

bars = plt.bar(methods, times, yerr=errors, capsize=5, color=['#ff7f0e', '#1f77b4'], alpha=0.8)
plt.ylabel("Execution Time (Seconds)", fontweight="bold")
plt.title("Single-Threaded Performance:\nScalar vs Explicit AVX2 Vectorization (n=500)\n", fontweight="bold")
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.002, f"{yval:.4f}s", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("plot_vectorization.png", dpi=300)
print("--> Exported: 'plot_vectorization.png'")