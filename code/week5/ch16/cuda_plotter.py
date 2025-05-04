import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv("mlp_timing_log.csv")

# Extract data
components = ['Malloc', 'H2D', 'Kernel', 'D2H', 'Free']
times = df.loc[0, components].values.astype(float)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))

bars = ax.bar(["MLP"], [sum(times)], label='Total', color='lightgray')
bottom = 0
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i, (comp, time) in enumerate(zip(components, times)):
    ax.bar(["MLP"], [time], bottom=bottom, label=comp, color=colors[i])
    bottom += time

# Log scale on y-axis
ax.set_yscale('log')
ax.set_ylabel("Time (ms, log scale)")
ax.set_title("MLP CUDA Timing Breakdown (Stacked Bar)")
ax.legend()
plt.tight_layout()
plt.savefig("mlp_timing_plot.png")
plt.show()
