// fib_cuda.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1 << 20)  // 2^20 = 1048576 elements
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

int main() {
    int *h_input;
    unsigned long long *h_output;
    int *d_input;
    unsigned long long *d_output;

    size_t input_size = N * sizeof(int);
    size_t output_size = N * sizeof(unsigned long long);

    // Allocate host memory
    h_input = (int *)malloc(input_size);
    h_output = (unsigned long long *)malloc(output_size);

    // Initialize input (limit to avoid overflow)
    for (int i = 0; i < N; ++i) {
        h_input[i] = i % 50;
    }

    // Allocate device memory
    cudaMalloc((void **)&d_input, input_size);
    cudaMalloc((void **)&d_output, output_size);

    // Create timing events
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

    cudaEventRecord(start_total);

    // Host to Device Copy
    cudaEventRecord(start_h2d);
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop_h2d);

    // Kernel Launch
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    cudaEventRecord(start_kernel);
    fibonacci_kernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, N);
    cudaEventRecord(stop_kernel);

    // Device to Host Copy
    cudaEventRecord(start_d2h);
    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_d2h);

    cudaEventRecord(stop_total);

    // Synchronize
    cudaEventSynchronize(stop_total);

    // Measure times
    float time_total = 0, time_h2d = 0, time_kernel = 0, time_d2h = 0;
    cudaEventElapsedTime(&time_total, start_total, stop_total);
    cudaEventElapsedTime(&time_h2d, start_h2d, stop_h2d);
    cudaEventElapsedTime(&time_kernel, start_kernel, stop_kernel);
    cudaEventElapsedTime(&time_d2h, start_d2h, stop_d2h);

    // Print timing results
    printf("=== Timing results (N = %d) ===\n", N);
    printf("Total time          : %.4f ms\n", time_total);
    printf("Host to Device time : %.4f ms\n", time_h2d);
    printf("Kernel execution    : %.4f ms\n", time_kernel);
    printf("Device to Host time : %.4f ms\n", time_d2h);

    // Save to CSV
    FILE *fp = fopen(CSV_FILENAME, "w");
    if (fp != NULL) {
        fprintf(fp, "N,HostToDevice (ms),Kernel (ms),DeviceToHost (ms),Total (ms)\n");
        fprintf(fp, "%d,%.6f,%.6f,%.6f,%.6f\n", N, time_h2d, time_kernel, time_d2h, time_total);
        fclose(fp);
        printf("Timing data saved to %s\n", CSV_FILENAME);
    } else {
        printf("Failed to open file %s for writing.\n", CSV_FILENAME);
    }

    // Optionally check some results
    printf("Sample output:\n");
    for (int i = 0; i < 10; ++i) {
        printf("Fib(%d) = %llu\n", h_input[i], h_output[i]);
    }

    // Clean up
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

    return 0;
}
