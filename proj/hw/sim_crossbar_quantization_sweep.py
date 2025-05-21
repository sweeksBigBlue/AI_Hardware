
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Configuration Parameters
# -------------------------------
V_RANGE = 1.0  # DAC/ADC voltage swing (V)
CROSSBAR_LATENCY_NS = 2.0
CROSSBAR_PULSE_WIDTH_NS = 2.0
CONDUCTANCE_ON = 1 / 24e3
CONDUCTANCE_OFF = 1 / (24e3 * 100)
BATCH_SIZE = 1024

# NeRF-like architecture
LAYER_DIMS = [(84, 256), (256, 256), (256, 4)]  # NeRF MLP: 84→256→256→4

# -------------------------------
# Latency Models
# -------------------------------
def nonlinear_dac_latency(bits):
    return bits ** 1.05 * 0.4

def nonlinear_adc_latency(bits):
    if bits <= 6:
        return bits * 1.0
    else:
        return 6 + (bits - 6) ** 1.5

# -------------------------------
# Quantization Noise
# -------------------------------
def quantization_noise_std(bits, v_range=V_RANGE):
    v_lsb = v_range / (2 ** bits)
    return v_lsb / np.sqrt(12)  # std of uniform error

# -------------------------------
# Simulation Function
# -------------------------------
def simulate_crossbar_system(layer_dims, batch_size, dac_bits, adc_bits):
    total_latency = 0
    total_energy = 0
    for (N, M) in layer_dims:
        # DAC/ADC Latency
        dac_latency = nonlinear_dac_latency(dac_bits)
        adc_latency = nonlinear_adc_latency(adc_bits)
        crossbar_latency = CROSSBAR_LATENCY_NS
        layer_latency = dac_latency + crossbar_latency + adc_latency
        total_latency += layer_latency

        # Noise-induced energy variability
        dac_noise_std = quantization_noise_std(dac_bits)
        adc_noise_std = quantization_noise_std(adc_bits)
        effective_noise_factor = 1 + np.random.normal(0, (dac_noise_std + adc_noise_std))

        avg_conductance = (CONDUCTANCE_ON + CONDUCTANCE_OFF) / 2
        layer_energy = (batch_size * N * M *
                        (V_RANGE ** 2) *
                        avg_conductance *
                        (CROSSBAR_PULSE_WIDTH_NS * 1e-9)) * 1e12 * effective_noise_factor
        total_energy += layer_energy

    return total_latency, total_energy

# -------------------------------
# Sweep Bit Precisions
# -------------------------------
bit_range = np.arange(2, 13)
latencies = []
energies = []

for bits in bit_range:
    latency, energy = simulate_crossbar_system(LAYER_DIMS, BATCH_SIZE, bits, bits)
    latencies.append(latency)
    energies.append(energy)

# -------------------------------
# Plotting Results
# -------------------------------
plt.figure()
plt.plot(bit_range, latencies, marker='o', label="Total Latency (ns)")
plt.title("Total Latency vs Bit Precision (Crossbar System)")
plt.xlabel("Bit Precision (DAC = ADC)")
plt.ylabel("Latency (ns)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("total_latency_vs_bits.png")

plt.figure()
plt.plot(bit_range, energies, marker='s', label="Total Energy (pJ)", color='orange')
plt.title("Total Energy vs Bit Precision (Crossbar System)")
plt.xlabel("Bit Precision (DAC = ADC)")
plt.ylabel("Energy (pJ)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("total_energy_vs_bits.png")
