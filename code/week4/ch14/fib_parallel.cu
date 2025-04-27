// fib_benchmark.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CSV_FILENAME "fib_timing_results.csv"

__device__ unsigned long long fibonacci(int n) {
    if (n <= 1) return n;
    unsigned long long a = 0, b = 1, c;
    for (int i = 2; i <= n; ++i) {
        c = a + b;
        a = b;
        b = c;
    }
    return b;
}

__global__ void fibonacci_kernel(int *input, unsigned long long *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fibonacci(input[idx]);
    }
}

void run_fibonacci(int n, FILE* fp) {
    int *h_input = nullptr;
    unsigned long long *h_output = nullptr;
    int *d_input = nullptr;
    unsigned long long *d_output = nullptr;

    size_t input_size = n * sizeof(int);
    size_t output_size = n * sizeof(unsigned long long);

    h_input = (int *)malloc(input_size);
    h_output = (unsigned long long *)malloc(output_size);

    if (!h_input || !h_output) {
        printf("Host memory allocation failed for N = %d\n", n);
        return;
    }

    // Initialize input
    for (int i = 0; i < n; ++i) {
        h_input[i] = i % 50;  // Limit Fibonacci input to prevent overflow
    }

    cudaMalloc((void **)&d_input, input_size);
    cudaMalloc((void **)&d_output, output_size);

    if (!d_input || !d_output) {
        printf("Device memory allocation failed for N = %d\n", n);
        free(h_input);
        free(h_output);
        return;
    }

    // Timing events
    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_h2d, stop_h2d;
    cudaEvent_t start_kernel, stop_kernel;
    cudaEvent_t start_d2h, stop_d2h;

    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventCreate(&start_h2d);
    cudaEventCreate(&stop_h2d);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventCreate(&start_d2h);
    cudaEventCreate(&stop_d2h);

    // Start total timer
    cudaEventRecord(start_total);

    // Host to Device
    cudaEventRecord(start_h2d);
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop_h2d);

    // Launch kernel
    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    cudaEventRecord(start_kernel);
    fibonacci_kernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, n);
    cudaEventRecord(stop_kernel);

    // Device to Host
    cudaEventRecord(start_d2h);
    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_d2h);

    // Stop total timer
    cudaEventRecord(stop_total);

    // Wait for all events to complete
    cudaEventSynchronize(stop_total);

    // Measure elapsed times
    float time_total = 0, time_h2d = 0, time_kernel = 0, time_d2h = 0;
    cudaEventElapsedTime(&time_total, start_total, stop_total);
    cudaEventElapsedTime(&time_h2d, start_h2d, stop_h2d);
    cudaEventElapsedTime(&time_kernel, start_kernel, stop_kernel);
    cudaEventElapsedTime(&time_d2h, start_d2h, stop_d2h);

    // Log timing to console
    printf("N = %d: Host2Device = %.6f ms, Kernel = %.6f ms, Device2Host = %.6f ms, Total = %.6f ms\n",
        n, time_h2d, time_kernel, time_d2h, time_total);

    // Write to CSV
    if (fp != NULL) {
        fprintf(fp, "%d,%.6f,%.6f,%.6f,%.6f\n", n, time_h2d, time_kernel, time_d2h, time_total);
    }

    // Cleanup
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_h2d);
    cudaEventDestroy(stop_h2d);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
    cudaEventDestroy(start_d2h);
    cudaEventDestroy(stop_d2h);
}

int main() {
    printf("Starting Fibonacci CUDA Benchmark...\n");

    // Open CSV file
    FILE *fp = fopen(CSV_FILENAME, "w");
    if (fp == NULL) {
        printf("Failed to open %s for writing.\n", CSV_FILENAME);
        return -1;
    }

    // Write CSV header
    fprintf(fp, "N,HostToDevice (ms),Kernel (ms),DeviceToHost (ms),Total (ms)\n");

    // Sweep N from 2^3 to 2^20
    for (int exp = 3; exp <= 20; ++exp) {
        int n = 1 << exp;
        run_fibonacci(n, fp);
    }

    fclose(fp);

    printf("Benchmark complete. Timing data saved to %s\n", CSV_FILENAME);
    return 0;
}
