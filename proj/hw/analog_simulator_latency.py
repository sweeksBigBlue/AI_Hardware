import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Non-linear ADC/DAC latency parameters
DAC_LATENCY_PER_BIT_NS = 0.4
DAC_ENERGY_PER_BIT_PJ = 0.01
ADC_LATENCY_PER_BIT_NS = 1.0
ADC_ENERGY_PER_BIT_PJ = 0.05
CROSSBAR_LATENCY_NS = 2.0
CONDUCTANCE_ON = 1 / 24e3
CONDUCTANCE_OFF = 1 / (24e3 * 100)
CROSSBAR_VOLTAGE = 0.5
CROSSBAR_PULSE_WIDTH_NS = 2.0

input_dim = 84

weight_matrices = [
    np.random.rand(84, 256),
    np.random.rand(256, 256),
    np.random.rand(256, 4)
]

def nonlinear_dac_latency(bits):
    return DAC_LATENCY_PER_BIT_NS * bits ** 1.05

def nonlinear_adc_latency(bits):
    if bits <= 6:
        return ADC_LATENCY_PER_BIT_NS * bits
    else:
        return 6 * ADC_LATENCY_PER_BIT_NS + ((bits - 6) ** 1.5)

def simulate_batch(batch_size, dac_bits=10, adc_bits=10):
    input_batch = np.random.rand(batch_size, input_dim)
    total_latency = 0
    total_energy = 0
    activations = input_batch

    for weights in weight_matrices:
        N, M = weights.shape
        dac_latency = nonlinear_dac_latency(dac_bits)
        dac_energy = batch_size * N * dac_bits * DAC_ENERGY_PER_BIT_PJ
        crossbar_latency = CROSSBAR_LATENCY_NS
        avg_conductance = (CONDUCTANCE_ON + CONDUCTANCE_OFF) / 2
        crossbar_energy = (batch_size * N * M * (CROSSBAR_VOLTAGE ** 2) * avg_conductance * (CROSSBAR_PULSE_WIDTH_NS * 1e-9)) * 1e12
        adc_latency = nonlinear_adc_latency(adc_bits)
        adc_energy = batch_size * M * adc_bits * ADC_ENERGY_PER_BIT_PJ

        total_latency += dac_latency + crossbar_latency + adc_latency
        total_energy += dac_energy + crossbar_energy + adc_energy

        activations = np.dot(activations, weights)

    return total_latency, total_energy

def main():
    batch_sizes = [128, 512, 1024, 2048]
    results = []

    for batch in batch_sizes:
        latency, energy = simulate_batch(batch)
        results.append({
            'Batch Size': batch,
            'Total Latency (ns)': latency,
            'Total Energy (pJ)': energy
        })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    df.plot(x='Batch Size', y='Total Latency (ns)', marker='o', title='Latency vs Batch Size (Non-linear DAC/ADC)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("batch_scaling_summary.png")
    plt.close()

    df.plot(x='Batch Size', y='Total Energy (pJ)', marker='o', title='Energy vs Batch Size')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("batch_energy_summary.png")
    plt.close()

if __name__ == "__main__":
    main()