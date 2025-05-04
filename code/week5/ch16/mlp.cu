#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <sys/stat.h>

#define INPUT_NODES 4
#define HIDDEN_NODES 5
#define OUTPUT_NODES 1
#define BATCH_SIZE 16  // <== change this value for each run

__device__ float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

__global__ void forward_pass_kernel(
    float *inputs,   // [BATCH_SIZE][INPUT_NODES]
    float *weights_input_hidden,
    float *bias_hidden,
    float *weights_hidden_output,
    float *bias_output,
    float *outputs   // [BATCH_SIZE]
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float hidden_activations[HIDDEN_NODES];

    float *input = &inputs[batch_idx * INPUT_NODES];
    float *output = &outputs[batch_idx];

    if (tid < HIDDEN_NODES) {
        float sum = 0.0f;
        for (int i = 0; i < INPUT_NODES; ++i)
            sum += input[i] * weights_input_hidden[tid * INPUT_NODES + i];
        sum += bias_hidden[tid];
        hidden_activations[tid] = relu(sum);
    }

    __syncthreads();

    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < HIDDEN_NODES; ++i)
            sum += hidden_activations[i] * weights_hidden_output[i];
        sum += *bias_output;
        *output = relu(sum);
    }
}

bool file_exists(const std::string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

int main() {
    using clock = std::chrono::high_resolution_clock;
    auto total_start = clock::now();

    float h_inputs[BATCH_SIZE * INPUT_NODES];
    float h_outputs[BATCH_SIZE];
    float h_weights_input_hidden[INPUT_NODES * HIDDEN_NODES];
    float h_weights_hidden_output[HIDDEN_NODES];
    float h_bias_hidden[HIDDEN_NODES];
    float h_bias_output = 0.1f;

    for (int i = 0; i < BATCH_SIZE * INPUT_NODES; ++i)
        h_inputs[i] = static_cast<float>(rand()) / RAND_MAX;

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

    float *d_inputs, *d_outputs;
    float *d_weights_input_hidden, *d_weights_hidden_output;
    float *d_bias_hidden, *d_bias_output;

    // -------- MALLOC ----------
    cudaEventRecord(start);
    cudaMalloc(&d_inputs, sizeof(float) * INPUT_NODES * BATCH_SIZE);
    cudaMalloc(&d_outputs, sizeof(float) * BATCH_SIZE);
    cudaMalloc(&d_weights_input_hidden, sizeof(float) * INPUT_NODES * HIDDEN_NODES);
    cudaMalloc(&d_weights_hidden_output, sizeof(float) * HIDDEN_NODES);
    cudaMalloc(&d_bias_hidden, sizeof(float) * HIDDEN_NODES);
    cudaMalloc(&d_bias_output, sizeof(float));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&malloc_ms, start, stop);

    // -------- HOST TO DEVICE ----------
    cudaEventRecord(start);
    cudaMemcpy(d_inputs, h_inputs, sizeof(float) * INPUT_NODES * BATCH_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_input_hidden, h_weights_input_hidden, sizeof(float) * INPUT_NODES * HIDDEN_NODES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_hidden_output, h_weights_hidden_output, sizeof(float) * HIDDEN_NODES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias_hidden, h_bias_hidden, sizeof(float) * HIDDEN_NODES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias_output, &h_bias_output, sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2d_ms, start, stop);

    // -------- KERNEL ----------
    cudaEventRecord(start);
    forward_pass_kernel<<<BATCH_SIZE, HIDDEN_NODES>>>(
        d_inputs, d_weights_input_hidden, d_bias_hidden,
        d_weights_hidden_output, d_bias_output, d_outputs);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernel_ms, start, stop);

    // -------- DEVICE TO HOST ----------
    cudaEventRecord(start);
    cudaMemcpy(h_outputs, d_outputs, sizeof(float) * BATCH_SIZE, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2h_ms, start, stop);

    // -------- FREE ----------
    cudaEventRecord(start);
    cudaFree(d_inputs);
    cudaFree(d_outputs);
    cudaFree(d_weights_input_hidden);
    cudaFree(d_weights_hidden_output);
    cudaFree(d_bias_hidden);
    cudaFree(d_bias_output);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&free_ms, start, stop);

    auto total_end = clock::now();
    double total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    std::cout << "MLP Output [first 5 values]: ";
    for (int i = 0; i < std::min(5, BATCH_SIZE); ++i)
        std::cout << h_outputs[i] << " ";
    std::cout << "\n\n==== Timing Breakdown (ms) ====\n";
    std::cout << "Memory Allocation     : " << malloc_ms << " ms\n";
    std::cout << "Host to Device Copy   : " << h2d_ms << " ms\n";
    std::cout << "Kernel Execution      : " << kernel_ms << " ms\n";
    std::cout << "Device to Host Copy   : " << d2h_ms << " ms\n";
    std::cout << "Device Memory Free    : " << free_ms << " ms\n";
    std::cout << "------------------------------\n";
    std::cout << "Total Execution Time  : " << total_time << " ms\n";

    // -------- CSV APPEND ----------
    std::string filename = "mlp_timing_log.csv";
    bool exists = file_exists(filename);

    std::ofstream csv(filename, std::ios::app);
    if (!exists) {
        csv << "Batch,Malloc,H2D,Kernel,D2H,Free,Total\n";
    }
    csv << BATCH_SIZE << ","
        << std::fixed << std::setprecision(6)
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
