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

# Plot settings
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each component stacked
bar_width = 0.4

# Because you want separate bars for each timing component, stacked style
ax.bar(N, host_to_device, width=bar_width, label='Host to Device')
ax.bar(N, kernel, width=bar_width, bottom=host_to_device, label='Kernel Execution')
ax.bar(N, device_to_host, width=bar_width, bottom=host_to_device+kernel, label='Device to Host')

# Plot total as a separate marker
ax.plot(N, total, 'ko--', label='Total Time', markersize=8)

# Set log scale for both axes
ax.set_xscale('log')
ax.set_yscale('log')

# Labels and title
ax.set_xlabel('N (input size)', fontsize=14)
ax.set_ylabel('Time (ms)', fontsize=14)
ax.set_title('CUDA Fibonacci Timing Breakdown', fontsize=16)
ax.legend()

# Grid for better readability
ax.grid(True, which="both", ls="--", linewidth=0.5)

# Save figure
plt.tight_layout()
plt.savefig('fib_timing_plot.png', dpi=300)
plt.show()
