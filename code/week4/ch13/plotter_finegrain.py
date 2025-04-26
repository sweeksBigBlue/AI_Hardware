import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the fine-grained CSV
df = pd.read_csv("saxpy_microbench_log.csv")

# Extract timing components
components = ["MallocTime_ms", "CudaMallocTime_ms", "MemcpyH2DTime_ms", "KernelTime_ms", "MemcpyD2HTime_ms", "FreeTime_ms"]
colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#c2c2f0", "#ffb3e6"]

# Stack values
x = np.arange(len(df["log2_N"]))
bottom = np.zeros(len(df))

# Create the figure
fig, ax = plt.subplots(figsize=(14, 7))

# Stack each timing component
for comp, color in zip(components, colors):
    ax.bar(x, df[comp], bottom=bottom, label=comp.replace("_", " ").replace("ms", ""), color=color)
    bottom += df[comp]

# Labels and formatting
ax.set_xlabel("log₂(N)")
ax.set_ylabel("Time (ms) (log scale)")
ax.set_title("SAXPY Fine-Grained Timing Breakdown (Stacked, Log Y-Axis)")
ax.set_xticks(x)
ax.set_xticklabels(df["log2_N"])
ax.set_yscale('log')
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("saxpy_finegrained_stacked_log.png", dpi=300)
plt.show()
