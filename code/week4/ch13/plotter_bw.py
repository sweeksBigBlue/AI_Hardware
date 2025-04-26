import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("saxpy_log.csv")

# Compute bandwidth (in GB/s)
df["Bandwidth_GBps"] = (3 * df["N"] * 4) / (df["ExecutionTime_ms"] * 1e6)

# Create dual-axis plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Bar chart: execution time
bars = ax1.bar(df["log2_N"], df["ExecutionTime_ms"], width=0.5, label="Execution Time (ms)", color="skyblue")
ax1.set_xlabel("log₂(N)")
ax1.set_ylabel("Execution Time (ms)", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

# Line chart: bandwidth
ax2 = ax1.twinx()
ax2.plot(df["log2_N"], df["Bandwidth_GBps"], color="red", marker="o", label="Bandwidth (GB/s)")
ax2.set_ylabel("Effective Bandwidth (GB/s)", color="red")
ax2.tick_params(axis="y", labelcolor="red")

# Title and layout
plt.title("SAXPY Performance: Execution Time & Bandwidth vs log₂(N)")
fig.tight_layout()
plt.grid(True, axis="y", linestyle="--", alpha=0.6)

plt.savefig("saxpy_dual_axis.png", dpi=300)
plt.show()
