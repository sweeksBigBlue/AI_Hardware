import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load PyTorch CSV log
df = pd.read_csv("mlp_timing_log_pytorch.csv")

# Components to visualize
components = ['Init', 'H2D', 'Forward', 'D2H']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

batch_labels = df['Batch'].astype(str).tolist()
bottom = np.zeros(len(df))

fig, ax = plt.subplots(figsize=(10, 6))

for i, comp in enumerate(components):
    ax.bar(batch_labels, df[comp], label=comp, bottom=bottom, color=colors[i])
    bottom += df[comp]

ax.set_yscale('log')
ax.set_xlabel("Batch Size")
ax.set_ylabel("Time (ms, log scale)")
ax.set_title("PyTorch MLP Timing Breakdown (Stacked Log Bar)")
ax.legend(title="Component", loc="upper left")
plt.tight_layout()
plt.savefig("pytorch_stacked_timing.png")
plt.show()
