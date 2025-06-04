import numpy as np
import pandas as pd

# System constants for realistic modeling (in nanoseconds and picojoules)
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
        crossbar_energy = (batch_size * N * M *
                           (CROSSBAR_VOLTAGE ** 2) *
                           avg_conductance *
                           (CROSSBAR_PULSE_WIDTH_NS * 1e-9)) * 1e12

        adc_latency = adc_bits * ADC_LATENCY_PER_BIT_NS
        adc_energy = batch_size * M * adc_bits * ADC_ENERGY_PER_BIT_PJ

        layer_latency = dac_latency + crossbar_latency + adc_latency
        layer_energy = dac_energy + crossbar_energy + adc_energy

        total_latency += layer_latency
        total_energy += layer_energy

        activations = np.dot(activations, weights)

        layer_breakdown.append({
            'layer': idx + 1,
            'input_dim': N,
            'output_dim': M,
            'batch_size': batch_size,
            'dac_latency_ns': dac_latency,
            'crossbar_latency_ns': crossbar_latency,
            'adc_latency_ns': adc_latency,
            'total_latency_ns': layer_latency,
            'dac_energy_pj': dac_energy,
            'crossbar_energy_pj': crossbar_energy,
            'adc_energy_pj': adc_energy,
            'total_energy_pj': layer_energy
        })

    return total_latency, total_energy, layer_breakdown

def main():
    np.random.seed(0)
    input_dim = 84
    batch_size = 8
    input_batch = np.random.rand(batch_size, input_dim)

    weight_matrices = [
        np.random.rand(84, 256),
        np.random.rand(256, 256),
        np.random.rand(256, 4)
    ]

    total_latency, total_energy, breakdown = simulate_crossbar_mlp(input_batch, weight_matrices)
    df = pd.DataFrame(breakdown)
    print(df.to_string(index=False))
    print(f"\nTotal Latency: {total_latency:.2f} ns")
    print(f"Total Energy: {total_energy:.2f} pJ")

if __name__ == "__main__":
    main()
