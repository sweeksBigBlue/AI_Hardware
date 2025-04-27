# fib_plotter_cpu_vs_cuda.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read merged CSV
csv_file = 'fib_timing_combined.csv'
data = pd.read_csv(csv_file)

# Extract data
N = data['N'].to_numpy()
host_to_device = data['HostToDevice (ms)'].to_numpy()
kernel = data['Kernel (ms)'].to_numpy()
device_to_host = data['DeviceToHost (ms)'].to_numpy()
total_cuda = data['Total (ms)'].to_numpy()
cpu_time = data['CPU Time (ms)'].to_numpy()

# Map N to exponents (x-axis)
x_pos = np.log2(N).astype(int)  # Exponents 3-20

# Settings
fig, ax = plt.subplots(figsize=(14, 8))
bar_width = 0.6

# Plot stacked bars for CUDA breakdown
ax.bar(x_pos, host_to_device, width=bar_width, label='Host to Device', alpha=0.9, edgecolor='black')
ax.bar(x_pos, kernel, width=bar_width, bottom=host_to_device, label='Kernel Execution', alpha=0.9, edgecolor='black')
ax.bar(x_pos, device_to_host, width=bar_width, bottom=host_to_device+kernel, label='Device to Host', alpha=0.9, edgecolor='black')

# Plot total CUDA time
ax.plot(x_pos, total_cuda, 'k--o', label='CUDA Total Time', markersize=6)

# Plot CPU time
ax.plot(x_pos, cpu_time, 'r--^', label='CPU Sequential Time', markersize=6)

# Log scale for y
ax.set_yscale('log')

# X-ticks and labels
ax.set_xticks(x_pos)
ax.set_xticklabels([f"$2^{{{exp}}}$" for exp in x_pos], rotation=45)

# Labels and title
ax.set_xlabel('Input Size (N)', fontsize=16)
ax.set_ylabel('Time (ms)', fontsize=16)
ax.set_title('CUDA vs CPU Fibonacci Timing (Log Y, Linear X)', fontsize=18)
ax.legend(fontsize=12)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Layout and save
plt.tight_layout()
plt.savefig('fib_cpu_vs_cuda_timing.png', dpi=300)
plt.show()
