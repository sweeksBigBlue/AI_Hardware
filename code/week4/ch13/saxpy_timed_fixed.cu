#include <stdio.h>
#include <math.h>  // for fabsf()

__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

int main(void) {
    // Sweep N from 2^16 to 2^30
    for (int exp = 16; exp <= 29; exp++) {
        unsigned long long N = 1ULL << exp;
        float *x = nullptr, *y = nullptr, *d_x = nullptr, *d_y = nullptr;
        
        // Allocate host memory
        x = (float *)malloc(N * sizeof(float));
        y = (float *)malloc(N * sizeof(float));
        
        if (x == nullptr || y == nullptr) {
            printf("Skipping N = 2^%d (%llu elements): Host malloc failed (too large)\n", exp, N);
            continue;
        }

        // Allocate device memory
        if (cudaMalloc(&d_x, N * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&d_y, N * sizeof(float)) != cudaSuccess) {
            printf("Skipping N = 2^%d (%llu elements): Device cudaMalloc failed (too large)\n", exp, N);
            free(x);
            free(y);
            continue;
        }

        // Initialize x and y arrays
        for (unsigned long long i = 0; i < N; i++) {
            x[i] = 1.0f;
            y[i] = 2.0f;
        }
        
        cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Record start event
        cudaEventRecord(start);

        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        saxpy<<<blocksPerGrid, threadsPerBlock>>>(N, 2.0f, d_x, d_y);

        // Synchronize to make sure kernel actually finishes
        cudaError_t errSync  = cudaDeviceSynchronize();
        cudaError_t errAsync = cudaGetLastError();
        if (errSync != cudaSuccess) {
            printf("Sync kernel error at N=2^%d: %s\n", exp, cudaGetErrorString(errSync));
        }
        if (errAsync != cudaSuccess) {
            printf("Async kernel error at N=2^%d: %s\n", exp, cudaGetErrorString(errAsync));
        }

        // Record stop event
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Calculate elapsed time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Copy result back to host
        cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

        // Compute maximum error
        float maxError = 0.0f;
        for (unsigned long long i = 0; i < N; i++) {
            maxError = fmaxf(maxError, fabsf(y[i] - 4.0f));
        }

        // Output results
        printf("N = 2^%d (%llu elements): Execution time = %.3f ms, Max error = %f\n",
               exp, N, milliseconds, maxError);

        // Clean up
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_x);
        cudaFree(d_y);
        free(x);
        free(y);
    }

    return 0;
}
