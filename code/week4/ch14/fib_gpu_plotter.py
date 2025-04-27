# fib_plotter_linearx_stacked.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read CSV
csv_file = 'fib_timing_results.csv'
data = pd.read_csv(csv_file)

# Extract data
N = data['N'].to_numpy()
host_to_device = data['HostToDevice (ms)'].to_numpy()
kernel = data['Kernel (ms)'].to_numpy()
device_to_host = data['DeviceToHost (ms)'].to_numpy()
total = data['Total (ms)'].to_numpy()

# Map N to sequential indices (0,1,2,...)
x_pos = np.arange(len(N))

# Settings
fig, ax = plt.subplots(figsize=(14, 8))
bar_width = 0.6  # Wider bars now that X is linear

# Plot stacked bars
ax.bar(x_pos, host_to_device, width=bar_width, label='Host to Device', alpha=0.9, edgecolor='black')
ax.bar(x_pos, kernel, width=bar_width, bottom=host_to_device, label='Kernel Execution', alpha=0.9, edgecolor='black')
ax.bar(x_pos, device_to_host, width=bar_width, bottom=host_to_device+kernel, label='Device to Host', alpha=0.9, edgecolor='black')

# Plot total time
ax.plot(x_pos, total, 'k--o', label='Total Time', markersize=6)

# Axes
ax.set_yscale('log')
# X-axis now categorical
ax.set_xticks(x_pos)
ax.set_xticklabels([f"$2^{{{int(np.log2(n))}}}$" for n in N], rotation=45)

# Labels and title
ax.set_xlabel('Input Size (N)', fontsize=16)
ax.set_ylabel('Time (ms)', fontsize=16)
ax.set_title('CUDA Fibonacci Fine-Grained Timing (Linear X, Log Y, Stacked Bars)', fontsize=18)
ax.legend(fontsize=12)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Layout
plt.tight_layout()
plt.savefig('fib_timing_linearx_stacked.png', dpi=300)
plt.show()
