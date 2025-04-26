#include <stdio.h>
#include <math.h>
#include <fstream>
#include <chrono>

__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

int main(void) {
    std::ofstream csv("saxpy_profiled_log.csv");
    csv << "log2_N,N,KernelTime_ms,TotalTime_ms,MaxError\n";

    for (int exp = 16; exp <= 29; exp++) {
        auto total_start = std::chrono::high_resolution_clock::now();

        unsigned long long N = 1ULL << exp;
        float *x = nullptr, *y = nullptr, *d_x = nullptr, *d_y = nullptr;

        x = (float *)malloc(N * sizeof(float));
        y = (float *)malloc(N * sizeof(float));
        if (!x || !y) {
            printf("Skipping N = 2^%d: malloc failed\n", exp);
            continue;
        }

        if (cudaMalloc(&d_x, N * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&d_y, N * sizeof(float)) != cudaSuccess) {
            printf("Skipping N = 2^%d: cudaMalloc failed\n", exp);
            free(x); free(y);
            continue;
        }

        for (unsigned long long i = 0; i < N; i++) {
            x[i] = 1.0f;
            y[i] = 2.0f;
        }

        cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

        // --- Measure kernel time only ---
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        saxpy<<<blocksPerGrid, threadsPerBlock>>>(N, 2.0f, d_x, d_y);

        cudaError_t errSync = cudaDeviceSynchronize();
        cudaError_t errAsync = cudaGetLastError();
        if (errSync != cudaSuccess) printf("Sync error at N=2^%d: %s\n", exp, cudaGetErrorString(errSync));
        if (errAsync != cudaSuccess) printf("Async error at N=2^%d: %s\n", exp, cudaGetErrorString(errAsync));

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float kernel_ms = 0.0f;
        cudaEventElapsedTime(&kernel_ms, start, stop);

        cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

        float maxError = 0.0f;
        for (unsigned long long i = 0; i < N; i++) {
            maxError = fmaxf(maxError, fabsf(y[i] - 4.0f));
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> total_ms = total_end - total_start;

        printf("N = 2^%d (%llu): Kernel = %.3f ms, Total = %.3f ms, Max error = %f\n",
               exp, N, kernel_ms, total_ms.count(), maxError);

        csv << exp << "," << N << "," << kernel_ms << "," << total_ms.count() << "," << maxError << "\n";

        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_x);
        cudaFree(d_y);
        free(x);
        free(y);
    }

    csv.close();
    return 0;
}
