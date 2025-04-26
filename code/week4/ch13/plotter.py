import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("saxpy_log.csv")

# Plot bar chart
plt.figure(figsize=(12, 6))
plt.bar(df["log2_N"], df["ExecutionTime_ms"], width=0.5)

plt.xlabel("log₂(N)")
plt.ylabel("Execution Time (ms)")
plt.title("SAXPY Kernel Execution Time vs Input Size (log scale)")
plt.xticks(df["log2_N"])  # show every log2(N) tick
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("saxpy_execution_time.png", dpi=300)
plt.show()
