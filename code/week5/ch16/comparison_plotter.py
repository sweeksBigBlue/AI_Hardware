import matplotlib.pyplot as plt
import pandas as pd

# Load both timing logs
df_cuda = pd.read_csv("mlp_timing_log.csv")
df_torch = pd.read_csv("mlp_timing_log_pytorch.csv")

# Ensure batch sizes match
df_cuda = df_cuda.sort_values("Batch")
df_torch = df_torch.sort_values("Batch")
assert list(df_cuda["Batch"]) == list(df_torch["Batch"]), "Batch sizes do not match."

batch_sizes = df_cuda["Batch"].tolist()
total_cuda = df_cuda["Total"].tolist()
total_torch = df_torch["Total"].tolist()

# Plot total runtimes
plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, total_cuda, marker='o', linestyle='-', color='blue', label="CUDA Total Time")
plt.plot(batch_sizes, total_torch, marker='s', linestyle='--', color='orange', label="PyTorch Total Time")

# Labels and legend
plt.xlabel("Batch Size")
plt.ylabel("Total Time (ms)")
plt.title("Total Inference Time vs Batch Size (CUDA vs PyTorch)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cuda_vs_pytorch_total_runtime.png")
plt.show()
