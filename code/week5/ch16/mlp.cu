#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>

#define INPUT_NODES 4
#define HIDDEN_NODES 5
#define OUTPUT_NODES 1

__device__ float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

__global__ void forward_pass_kernel(
    float *input,
    float *weights_input_hidden,
    float *bias_hidden,
    float *weights_hidden_output,
    float *bias_output,
    float *output
) {
    __shared__ float hidden_activations[HIDDEN_NODES];
    int tid = threadIdx.x;

    if (tid < HIDDEN_NODES) {
        float sum = 0.0f;
        for (int i = 0; i < INPUT_NODES; ++i) {
            sum += input[i] * weights_input_hidden[tid * INPUT_NODES + i];
        }
        sum += bias_hidden[tid];
        hidden_activations[tid] = relu(sum);
    }

    __syncthreads();

    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < HIDDEN_NODES; ++i) {
            sum += hidden_activations[i] * weights_hidden_output[i];
        }
        sum += *bias_output;
        *output = relu(sum);
    }
}

int main() {
    using clock = std::chrono::high_resolution_clock;
    auto total_start = clock::now();

    float h_input[INPUT_NODES] = {0.5f, 0.1f, 0.3f, 0.9f};
    float h_weights_input_hidden[INPUT_NODES * HIDDEN_NODES];
    float h_weights_hidden_output[HIDDEN_NODES];
    float h_bias_hidden[HIDDEN_NODES];
    float h_bias_output = 0.1f;
    float h_output;

    // Random weights/biases
    for (int i = 0; i < INPUT_NODES * HIDDEN_NODES; ++i)
        h_weights_input_hidden[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    for (int i = 0; i < HIDDEN_NODES; ++i) {
        h_weights_hidden_output[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        h_bias_hidden[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }

    cudaEvent_t start, stop;
    float malloc_ms = 0, h2d_ms = 0, kernel_ms = 0, d2h_ms = 0, free_ms = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *d_input, *d_weights_input_hidden, *d_weights_hidden_output;
    float *d_bias_hidden, *d_bias_output, *d_output;

    // ------------------ MALLOC ------------------
    cudaEventRecord(start);
    cudaMalloc(&d_input, sizeof(float) * INPUT_NODES);
    cudaMalloc(&d_weights_input_hidden, sizeof(float) * INPUT_NODES * HIDDEN_NODES);
    cudaMalloc(&d_weights_hidden_output, sizeof(float) * HIDDEN_NODES);
    cudaMalloc(&d_bias_hidden, sizeof(float) * HIDDEN_NODES);
    cudaMalloc(&d_bias_output, sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&malloc_ms, start, stop);

    // ------------------ H2D COPY ------------------
    cudaEventRecord(start);
    cudaMemcpy(d_input, h_input, sizeof(float) * INPUT_NODES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_input_hidden, h_weights_input_hidden, sizeof(float) * INPUT_NODES * HIDDEN_NODES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_hidden_output, h_weights_hidden_output, sizeof(float) * HIDDEN_NODES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias_hidden, h_bias_hidden, sizeof(float) * HIDDEN_NODES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias_output, &h_bias_output, sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2d_ms, start, stop);

    // ------------------ KERNEL ------------------
    cudaEventRecord(start);
    forward_pass_kernel<<<1, HIDDEN_NODES>>>(d_input, d_weights_input_hidden, d_bias_hidden,
                                             d_weights_hidden_output, d_bias_output, d_output);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernel_ms, start, stop);

    // ------------------ D2H COPY ------------------
    cudaEventRecord(start);
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2h_ms, start, stop);

    // ------------------ FREE ------------------
    cudaEventRecord(start);
    cudaFree(d_input);
    cudaFree(d_weights_input_hidden);
    cudaFree(d_weights_hidden_output);
    cudaFree(d_bias_hidden);
    cudaFree(d_bias_output);
    cudaFree(d_output);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&free_ms, start, stop);

    auto total_end = clock::now();
    double total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    // ------------------ OUTPUT ------------------
    std::cout << "MLP Output: " << h_output << "\n\n";
    std::cout << "==== Timing Breakdown (ms) ====\n";
    std::cout << "Memory Allocation     : " << malloc_ms << " ms\n";
    std::cout << "Host to Device Copy   : " << h2d_ms << " ms\n";
    std::cout << "Kernel Execution      : " << kernel_ms << " ms\n";
    std::cout << "Device to Host Copy   : " << d2h_ms << " ms\n";
    std::cout << "Device Memory Free    : " << free_ms << " ms\n";
    std::cout << "------------------------------\n";
    std::cout << "Total Execution Time  : " << total_time << " ms\n";

    // ------------------ CSV LOG ------------------
    std::ofstream csv("mlp_timing_log.csv", std::ios::out);
    csv << "Malloc,H2D,Kernel,D2H,Free,Total\n";
    csv << std::fixed << std::setprecision(6)
        << malloc_ms << ","
        << h2d_ms << ","
        << kernel_ms << ","
        << d2h_ms << ","
        << free_ms << ","
        << total_time << "\n";
    csv.close();

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
