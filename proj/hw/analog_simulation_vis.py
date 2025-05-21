import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants for realistic analog hardware
DAC_LATENCY_PER_BIT_NS = 0.4
DAC_ENERGY_PER_BIT_PJ = 0.01
ADC_LATENCY_PER_BIT_NS = 1.0
ADC_ENERGY_PER_BIT_PJ = 0.05
CROSSBAR_LATENCY_NS = 2.0
CONDUCTANCE_ON = 1 / 24e3
CONDUCTANCE_OFF = 1 / (24e3 * 100)
CROSSBAR_VOLTAGE = 0.5
CROSSBAR_PULSE_WIDTH_NS = 2.0

def simulate_crossbar_mlp(input_batch, weight_matrices, dac_bits=10, adc_bits=10):
    batch_size = input_batch.shape[0]
    total_latency = 0
    total_energy = 0
    activations = input_batch
    layer_breakdown = []

    for idx, weights in enumerate(weight_matrices):
        N, M = weights.shape
        dac_latency = dac_bits * DAC_LATENCY_PER_BIT_NS
        dac_energy = batch_size * N * dac_bits * DAC_ENERGY_PER_BIT_PJ
        crossbar_latency = CROSSBAR_LATENCY_NS
        avg_conductance = (CONDUCTANCE_ON + CONDUCTANCE_OFF) / 2
        crossbar_energy = (batch_size * N * M * (CROSSBAR_VOLTAGE ** 2) * avg_conductance * (CROSSBAR_PULSE_WIDTH_NS * 1e-9)) * 1e12
        adc_latency = adc_bits * ADC_LATENCY_PER_BIT_NS
        adc_energy = batch_size * M * adc_bits * ADC_ENERGY_PER_BIT_PJ

        layer_latency = dac_latency + crossbar_latency + adc_latency
        layer_energy = dac_energy + crossbar_energy + adc_energy

        total_latency += layer_latency
        total_energy += layer_energy

        activations = np.dot(activations, weights)

        layer_breakdown.append({
            'Layer': f'Layer {idx + 1}',
            'DAC Latency (ns)': dac_latency,
            'Crossbar Latency (ns)': crossbar_latency,
            'ADC Latency (ns)': adc_latency,
            'Total Latency (ns)': layer_latency,
            'DAC Energy (pJ)': dac_energy,
            'Crossbar Energy (pJ)': crossbar_energy,
            'ADC Energy (pJ)': adc_energy,
            'Total Energy (pJ)': layer_energy
        })

    return total_latency, total_energy, pd.DataFrame(layer_breakdown)

def visualize_breakdown(df, prefix="batch_8"):
    df.set_index('Layer', inplace=True)
    latency_df = df[['DAC Latency (ns)', 'Crossbar Latency (ns)', 'ADC Latency (ns)']]
    energy_df = df[['DAC Energy (pJ)', 'Crossbar Energy (pJ)', 'ADC Energy (pJ)']]

    latency_df.plot(kind='bar', stacked=True, title='Latency Breakdown per Layer')
    plt.ylabel('Latency (ns)')
    plt.tight_layout()
    plt.savefig(f"{prefix}_latency.png")
    plt.close()

    energy_df.plot(kind='bar', stacked=True, title='Energy Breakdown per Layer')
    plt.ylabel('Energy (pJ)')
    plt.tight_layout()
    plt.savefig(f"{prefix}_energy.png")
    plt.close()

def main():
    np.random.seed(0)
    input_dim = 84
    batch_sizes = [128, 512, 1024, 2048]
    all_results = []

    for batch_size in batch_sizes:
        input_batch = np.random.rand(batch_size, input_dim)
        weight_matrices = [
            np.random.rand(84, 256),
            np.random.rand(256, 256),
            np.random.rand(256, 4)
        ]

        total_latency, total_energy, breakdown_df = simulate_crossbar_mlp(input_batch, weight_matrices)
        all_results.append({
            'Batch Size': batch_size,
            'Total Latency (ns)': total_latency,
            'Total Energy (pJ)': total_energy
        })

        print(f"\n=== Batch Size: {batch_size} ===")
        print(breakdown_df.to_string(index=False))

        if batch_size == 1024:
            visualize_breakdown(breakdown_df, prefix="batch_8")

    summary_df = pd.DataFrame(all_results)
    print("\n=== Summary Across Batch Sizes ===")
    print(summary_df.to_string(index=False))

    summary_df.plot(x='Batch Size', y='Total Latency (ns)', marker='o')
    plt.title('Latency vs Batch Size')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("batch_scaling_summary.png")
    plt.close()

if __name__ == "__main__":
    main()