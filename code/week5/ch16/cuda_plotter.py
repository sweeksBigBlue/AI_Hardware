import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load timing log
df = pd.read_csv("mlp_timing_log.csv")

# Columns to stack (excluding total)
timing_cols = ['Malloc', 'H2D', 'Kernel', 'D2H', 'Free']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Create base figure
fig, ax = plt.subplots(figsize=(10, 6))

batch_labels = df['Batch'].astype(str).tolist()
bottom = np.zeros(len(df))

# Plot each timing component as a layer
for i, col in enumerate(timing_cols):
    ax.bar(batch_labels, df[col], label=col, bottom=bottom, color=colors[i])
    bottom += df[col]

# Format
ax.set_yscale('log')
ax.set_xlabel("Batch Size")
ax.set_ylabel("Time (ms, log scale)")
ax.set_title("MLP CUDA Timing Breakdown by Batch Size (Stacked Log Bar)")
ax.legend(title="Component", loc="upper left")
plt.tight_layout()
plt.savefig("mlp_batch_timing_plot.png")
plt.show()
