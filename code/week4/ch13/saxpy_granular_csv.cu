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
    std::ofstream csv("saxpy_microbench_log.csv");
    csv << "log2_N,N,MallocTime_ms,CudaMallocTime_ms,MemcpyH2DTime_ms,KernelTime_ms,MemcpyD2HTime_ms,FreeTime_ms,MaxError\n";

    for (int exp = 16; exp <= 29; exp++) {
        unsigned long long N = 1ULL << exp;

        auto malloc_start = std::chrono::high_resolution_clock::now();
        float *x = (float *)malloc(N * sizeof(float));
        float *y = (float *)malloc(N * sizeof(float));
        auto malloc_end = std::chrono::high_resolution_clock::now();
        if (!x || !y) {
            printf("Skipping N = 2^%d: malloc failed\n", exp);
            continue;
        }
        std::chrono::duration<double, std::milli> malloc_time = malloc_end - malloc_start;

        // CUDA events
        cudaEvent_t startMalloc, stopMalloc;
        cudaEvent_t startH2D, stopH2D;
        cudaEvent_t startKernel, stopKernel;
        cudaEvent_t startD2H, stopD2H;

        cudaEventCreate(&startMalloc);
        cudaEventCreate(&stopMalloc);
        cudaEventCreate(&startH2D);
        cudaEventCreate(&stopH2D);
        cudaEventCreate(&startKernel);
        cudaEventCreate(&stopKernel);
        cudaEventCreate(&startD2H);
        cudaEventCreate(&stopD2H);

        // cudaMalloc
        cudaEventRecord(startMalloc);
        float *d_x = nullptr, *d_y = nullptr;
        cudaMalloc(&d_x, N * sizeof(float));
        cudaMalloc(&d_y, N * sizeof(float));
        cudaEventRecord(stopMalloc);
        cudaEventSynchronize(stopMalloc);

        // Initialize x and y
        for (unsigned long long i = 0; i < N; i++) {
            x[i] = 1.0f;
            y[i] = 2.0f;
        }

        // memcpy host-to-device
        cudaEventRecord(startH2D);
        cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaEventRecord(stopH2D);
        cudaEventSynchronize(stopH2D);

        // Kernel
        cudaEventRecord(startKernel);
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        saxpy<<<blocksPerGrid, threadsPerBlock>>>(N, 2.0f, d_x, d_y);
        cudaDeviceSynchronize();
        cudaEventRecord(stopKernel);
        cudaEventSynchronize(stopKernel);

        // memcpy device-to-host
        cudaEventRecord(startD2H);
        cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaEventRecord(stopD2H);
        cudaEventSynchronize(stopD2H);

        // Compute error
        float maxError = 0.0f;
        for (unsigned long long i = 0; i < N; i++) {
            maxError = fmaxf(maxError, fabsf(y[i] - 4.0f));
        }

        // Free memory
        auto free_start = std::chrono::high_resolution_clock::now();
        cudaFree(d_x);
        cudaFree(d_y);
        free(x);
        free(y);
        auto free_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> free_time = free_end - free_start;

        // Measure all CUDA times
        float cudaMalloc_ms, h2d_ms, kernel_ms, d2h_ms;
        cudaEventElapsedTime(&cudaMalloc_ms, startMalloc, stopMalloc);
        cudaEventElapsedTime(&h2d_ms, startH2D, stopH2D);
        cudaEventElapsedTime(&kernel_ms, startKernel, stopKernel);
        cudaEventElapsedTime(&d2h_ms, startD2H, stopD2H);

        printf("N = 2^%d: malloc=%.3fms cudaMalloc=%.3fms h2d=%.3fms kernel=%.3fms d2h=%.3fms free=%.3fms maxError=%f\n",
               exp, malloc_time.count(), cudaMalloc_ms, h2d_ms, kernel_ms, d2h_ms, free_time.count(), maxError);

        csv << exp << "," << N << ","
            << malloc_time.count() << ","
            << cudaMalloc_ms << ","
            << h2d_ms << ","
            << kernel_ms << ","
            << d2h_ms << ","
            << free_time.count() << ","
            << maxError << "\n";

        cudaEventDestroy(startMalloc);
        cudaEventDestroy(stopMalloc);
        cudaEventDestroy(startH2D);
        cudaEventDestroy(stopH2D);
        cudaEventDestroy(startKernel);
        cudaEventDestroy(stopKernel);
        cudaEventDestroy(startD2H);
        cudaEventDestroy(stopD2H);
    }

    csv.close();
    return 0;
}
