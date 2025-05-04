import matplotlib.pyplot as plt
import pandas as pd

# Load timing data
df_cuda = pd.read_csv("mlp_timing_log.csv")
df_torch = pd.read_csv("mlp_timing_log_pytorch.csv")

# Ensure both are sorted and aligned
df_cuda = df_cuda.sort_values("Batch")
df_torch = df_torch.sort_values("Batch")

assert list(df_cuda["Batch"]) == list(df_torch["Batch"]), "Batch size mismatch between CUDA and PyTorch logs."

batch_sizes = df_cuda["Batch"].tolist()
delta = df_cuda["Total"] - df_torch["Total"]

# Plot difference
plt.figure(figsize=(10, 5))
plt.plot(batch_sizes, delta, marker='o', linestyle='-', color='purple', label="CUDA - PyTorch (Total Time)")
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)

# Annotate whether CUDA or PyTorch was faster
for i, b in enumerate(batch_sizes):
    time_diff = delta[i]
    label = "CUDA" if time_diff < 0 else "PyTorch"
    color = "blue" if time_diff < 0 else "orange"
    plt.text(b, time_diff + 0.01 * (-1 if time_diff < 0 else 1), f"{label}", 
             ha='center', va='bottom' if time_diff > 0 else 'top', fontsize=9, color=color)

plt.xlabel("Batch Size")
plt.ylabel("Δ Time (ms)")
plt.title("CUDA vs PyTorch: Difference in Total Inference Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("cuda_vs_pytorch_delta_annotated.png")
plt.show()
