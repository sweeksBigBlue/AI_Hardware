
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Configuration Parameters
# -------------------------------
V_IN = 1.0  # Input voltage applied to rows (V)
R_LINE_PER_CELL = 5.0  # Wire resistance per cell segment (Ohms)
G_ON = 1 / 24e3        # Conductance of ON state
G_OFF = 1 / (24e3 * 100)  # Conductance of OFF state
PULSE_WIDTH_NS = 2.0
V_DROP_TOLERANCE = 0.05  # Max allowable IR drop before failing spec (as fraction of V_IN)

# Define network layers (e.g., NeRF MLP)
LAYER_DIMS = [(84, 256), (256, 256), (256, 4)]

# -------------------------------
# IR Drop Model for One Crossbar
# -------------------------------
def simulate_crossbar_ir_drop(N, M, g_on, g_off, r_line):
    # Assume 50% of devices ON, 50% OFF randomly distributed
    conductances = np.random.choice([g_on, g_off], size=(N, M))

    v_actual = np.zeros(N)
    total_current = 0.0
    total_energy = 0.0
    v_drops = []

    for i in range(N):
        # IR drop from row wire resistance
        r_row = i * r_line
        v_drop = r_row * np.sum(conductances[i]) * V_IN
        v_applied = V_IN - v_drop
        v_actual[i] = max(v_applied, 0)
        v_drops.append(v_drop)

        row_current = np.sum(conductances[i]) * v_actual[i]
        total_current += row_current

        # Energy = V * I * t (t in seconds)
        total_energy += v_actual[i] * row_current * (PULSE_WIDTH_NS * 1e-9)

    avg_v_drop = np.mean(v_drops)
    max_v_drop = np.max(v_drops)
    ir_drop_ok = (max_v_drop / V_IN) <= V_DROP_TOLERANCE

    return avg_v_drop, max_v_drop, total_current, total_energy * 1e12, ir_drop_ok

# -------------------------------
# Run Simulation for All Layers
# -------------------------------
layer_results = []
total_energy = 0
layer_names = []

for idx, (N, M) in enumerate(LAYER_DIMS):
    avg_vdrop, max_vdrop, total_current, energy, ok = simulate_crossbar_ir_drop(N, M, G_ON, G_OFF, R_LINE_PER_CELL)
    layer_results.append((avg_vdrop, max_vdrop, total_current, energy, ok))
    total_energy += energy
    layer_names.append(f"Layer {idx+1} ({N}x{M})")

# -------------------------------
# Plotting Per-Crossbar IR Drop
# -------------------------------
layer_results = np.array(layer_results)
plt.figure()
plt.plot(layer_names, layer_results[:, 0], marker='o', label='Avg Vdrop (V)')
plt.plot(layer_names, layer_results[:, 1], marker='s', label='Max Vdrop (V)')
plt.ylabel("Voltage Drop (V)")
plt.title("IR Drop per Crossbar Layer")
plt.grid(True)
plt.xticks(rotation=30)
plt.legend()
plt.tight_layout()
plt.savefig("ir_drop_per_layer.png")

# -------------------------------
# Plotting Total Energy
# -------------------------------
plt.figure()
plt.bar(layer_names, layer_results[:, 3], color='orange')
plt.title("Energy Consumption per Crossbar Layer")
plt.ylabel("Energy (pJ)")
plt.xticks(rotation=30)
plt.grid(True)
plt.tight_layout()
plt.savefig("energy_per_layer_ir.png")

print("✅ IR drop simulation complete.")
print("Total energy across all layers: {:.2f} pJ".format(total_energy))
