# fib_plotter_stacked_fixed.py
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
fig, ax = plt.subplots(figsize=(12, 8))

# Dynamic width for bars based on log spacing
log_N = np.log2(N)
widths = np.diff(np.concatenate(([log_N[0] - (log_N[1] - log_N[0])], log_N)))  # approximate
bar_widths = 2 ** (widths * 0.8)  # slightly compressed for visual spacing

# Plot stacked bars (now dynamically sized)
ax.bar(N, host_to_device, width=bar_widths, align='center', label='Host to Device', alpha=0.9, edgecolor='black')
ax.bar(N, kernel, width=bar_widths, bottom=host_to_device, align='center', label='Kernel Execution', alpha=0.9, edgecolor='black')
ax.bar(N, device_to_host, width=bar_widths, bottom=host_to_device+kernel, align='center', label='Device to Host', alpha=0.9, edgecolor='black')

# Plot total time
ax.plot(N, total, 'k--o', label='Total Time', markersize=6)

# Log scales
ax.set_xscale('log')
ax.set_yscale('log')

# X-ticks
ax.set_xticks(N)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xticklabels([f"$2^{{{int(np.log2(n))}}}$" for n in N], rotation=45)

# Labels and title
ax.set_xlabel('Input Size (N)', fontsize=16)
ax.set_ylabel('Time (ms)', fontsize=16)
ax.set_title('CUDA Fibonacci Fine-Grained Timing (Log-Log, Stacked Bars)', fontsize=18)
ax.legend(fontsize=12)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Layout
plt.tight_layout()
plt.savefig('fib_timing_stacked_fixed.png', dpi=300)
plt.show()
