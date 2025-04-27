# fixed_fib_plotter.py
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

# Settings
bar_width_fraction = 0.2  # fraction of distance between points
log_N = np.log2(N)
dx = 2 ** (log_N - 0.15)  # left offset for first bar
bar_widths = 2 ** (log_N - 0.7)  # dynamic bar width proportional to x-spacing

fig, ax = plt.subplots(figsize=(12, 8))

# Plot each group: slight shift around real N values
ax.bar(N * 0.8, host_to_device, width=bar_widths, label='Host to Device', alpha=0.8)
ax.bar(N, kernel, width=bar_widths, label='Kernel Execution', alpha=0.8)
ax.bar(N * 1.2, device_to_host, width=bar_widths, label='Device to Host', alpha=0.8)

# Plot total time
ax.plot(N, total, 'k--o', label='Total Time', markersize=6)

# Set log-log scale
ax.set_xscale('log')
ax.set_yscale('log')

# X-ticks at exact N values
ax.set_xticks(N)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xticklabels([f"$2^{{{int(np.log2(n))}}}$" for n in N], rotation=45)

# Labels and title
ax.set_xlabel('Input Size (N)', fontsize=16)
ax.set_ylabel('Time (ms)', fontsize=16)
ax.set_title('CUDA Fibonacci Fine-Grained Timing (Log-Log Scale)', fontsize=18)
ax.legend(fontsize=12)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Layout and save
plt.tight_layout()
plt.savefig('fib_timing_grouped_fixed.png', dpi=300)
plt.show()
