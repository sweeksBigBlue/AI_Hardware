import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv("saxpy_profiled_log.csv")

# Bar parameters
bar_width = 0.35
x = np.arange(len(df["log2_N"]))

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))

# Bars
ax.bar(x - bar_width/2, df["KernelTime_ms"], width=bar_width, label="Kernel Time (ms)", color="skyblue")
ax.bar(x + bar_width/2, df["TotalTime_ms"], width=bar_width, label="Total Time (ms)", color="lightcoral")

# Labels and formatting
ax.set_xlabel("log₂(N)")
ax.set_ylabel("Time (ms) (log scale)")
ax.set_title("SAXPY: Kernel vs Total Execution Time (Log Y-Axis)")
ax.set_xticks(x)
ax.set_xticklabels(df["log2_N"])
ax.legend()

# Set log scale on Y-axis
ax.set_yscale('log')
ax.grid(True, axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("saxpy_kernel_vs_total_log.png", dpi=300)
plt.show()
