#include <stdio.h>
#include <math.h>
#include <fstream>

__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

int main(void) {
    std::ofstream csv("saxpy_log.csv");
    csv << "log2_N,N,ExecutionTime_ms,MaxError\n";

    for (int exp = 16; exp <= 29; exp++) {
        unsigned long long N = 1ULL << exp;
        float *x = nullptr, *y = nullptr, *d_x = nullptr, *d_y = nullptr;

        x = (float *)malloc(N * sizeof(float));
        y = (float *)malloc(N * sizeof(float));
        if (x == nullptr || y == nullptr) {
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

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        saxpy<<<blocksPerGrid, threadsPerBlock>>>(N, 2.0f, d_x, d_y);

        cudaError_t errSync = cudaDeviceSynchronize();
        cudaError_t errAsync = cudaGetLastError();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

        float maxError = 0.0f;
        for (unsigned long long i = 0; i < N; i++) {
            maxError = fmaxf(maxError, fabsf(y[i] - 4.0f));
        }

        printf("N = 2^%d (%llu elements): Execution time = %.3f ms, Max error = %f\n",
               exp, N, ms, maxError);

        csv << exp << "," << N << "," << ms << "," << maxError << "\n";

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
