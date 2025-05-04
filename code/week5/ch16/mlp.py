import torch
import torch.nn as nn
import time
import csv
import os

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MLP structure
INPUT_NODES = 4
HIDDEN_NODES = 5
OUTPUT_NODES = 1

# Batch sizes to test
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]

# Output CSV
CSV_FILE = "mlp_timing_log_pytorch.csv"
write_header = not os.path.exists(CSV_FILE)

# MLP Model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(INPUT_NODES, HIDDEN_NODES)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_NODES, OUTPUT_NODES)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Run benchmarks
with open(CSV_FILE, 'a', newline='') as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["Batch", "Init", "H2D", "Forward", "D2H", "Total"])

    for batch_size in BATCH_SIZES:
        print(f"Running for batch size {batch_size}...")

        # --- Initialization time (CPU)
        t0 = time.time()
        model = MLP().to(device)
        x_cpu = torch.rand(batch_size, INPUT_NODES)
        t1 = time.time()

        # --- Host to Device
        x_gpu = x_cpu.to(device)
        t2 = time.time()

        # --- Forward pass (GPU event timing)
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        with torch.no_grad():
            y_gpu = model(x_gpu)
        ender.record()
        torch.cuda.synchronize()
        t_fwd = starter.elapsed_time(ender)  # ms

        # --- Device to Host
        t3 = time.time()
        y_cpu = y_gpu.cpu()
        t4 = time.time()

        # --- Host-side timings
        t_init = (t1 - t0) * 1000
        t_h2d  = (t2 - t1) * 1000
        t_d2h  = (t4 - t3) * 1000
        t_total = (t4 - t0) * 1000

        # --- CSV logging
        writer.writerow([batch_size, t_init, t_h2d, t_fwd, t_d2h, t_total])

        # --- Console output
        print(f"Output [first 5]: {y_cpu[:5].squeeze().tolist()}")
        print(f"Init: {t_init:.4f} ms, H2D: {t_h2d:.4f} ms, Forward: {t_fwd:.4f} ms, D2H: {t_d2h:.4f} ms, Total: {t_total:.4f} ms\n")
