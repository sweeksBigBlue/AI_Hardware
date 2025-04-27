# fib_plotter.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read CSV
csv_file = 'fib_timing_results.csv'
data = pd.read_csv(csv_file)

# Extract data
N = data['N']
host_to_device = data['HostToDevice (ms)']
kernel = data['Kernel (ms)']
device_to_host = data['DeviceToHost (ms)']
total = data['Total (ms)']

# Settings
bar_width = 0.25
index = np.arange(len(N))  # indexes for grouped bars

fig, ax = plt.subplots(figsize=(12, 8))

# Plot grouped bars
ax.bar(index - bar_width, host_to_device, width=bar_width, label='Host to Device', alpha=0.8)
ax.bar(index, kernel, width=bar_width, label='Kernel Execution', alpha=0.8)
ax.bar(index + bar_width, device_to_host, width=bar_width, label='Device to Host', alpha=0.8)

# Plot total time as dashed line
ax.plot(index, total, 'k--o', label='Total Time', markersize=6)

# Log scale for both axes
ax.set_yscale('log')
ax.set_xscale('log')

# X-ticks as powers of 2 (N values)
ax.set_xticks(index)
ax.set_xticklabels([f"$2^{{{int(np.log2(n))}}}$" for n in N], rotation=45)

# Labels and title
ax.set_xlabel('Input Size (N)', fontsize=16)
ax.set_ylabel('Time (ms)', fontsize=16)
ax.set_title('CUDA Fibonacci Fine-Grained Timing (Log-Log Scale)', fontsize=18)
ax.legend(fontsize=12)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Adjust layout
plt.tight_layout()

# Save high-res image
plt.savefig('fib_timing_grouped_plot.png', dpi=300)
plt.show()
