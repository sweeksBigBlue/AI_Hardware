import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV
df = pd.read_csv("saxpy_profiled_log.csv")

# Set bar width and positions
bar_width = 0.35
x = np.arange(len(df["log2_N"]))

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Bars for kernel time
bars1 = ax.bar(x - bar_width/2, df["KernelTime_ms"], width=bar_width, label="Kernel Time (ms)", color="skyblue")

# Bars for total time
bars2 = ax.bar(x + bar_width/2, df["TotalTime_ms"], width=bar_width, label="Total Time (ms)", color="lightcoral")

# Labels and titles
ax.set_xlabel("log₂(N)")
ax.set_ylabel("Time (ms)")
ax.set_title("SAXPY: Kernel vs Total Execution Time")
ax.set_xticks(x)
ax.set_xticklabels(df["log2_N"])
ax.legend()

plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("saxpy_kernel_vs_total.png", dpi=300)
plt.show()
