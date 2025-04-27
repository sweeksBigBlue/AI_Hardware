# merge_fib_results.py
import pandas as pd

# Load both CSVs
cuda_df = pd.read_csv('fib_timing_results.csv')
cpu_df = pd.read_csv('fib_timing_results_cpu.csv')

# Merge them on 'N'
merged_df = pd.merge(cuda_df, cpu_df, on='N')

# Save merged CSV
merged_df.to_csv('fib_timing_combined.csv', index=False)

print("Merged CSV saved as fib_timing_combined.csv")
