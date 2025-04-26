#include <stdio.h>
#include <math.h>  // for fabsf()

__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

int main(void) {
    // Sweep N from 2^16 to 2^25
    for (int exp = 16; exp <= 25; exp++) {
        int N = 1 << exp;
        float *x, *y, *d_x, *d_y;
        
        // Allocate host memory
        x = (float *)malloc(N * sizeof(float));
        y = (float *)malloc(N * sizeof(float));
        
        // Allocate device memory
        cudaMalloc(&d_x, N * sizeof(float));
        cudaMalloc(&d_y, N * sizeof(float));
        
        // Initialize x and y arrays
        for (int i = 0; i < N; i++) {
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

        // Record stop event
        cudaEventRecord(stop);

        // Wait for the event to complete
        cudaEventSynchronize(stop);

        // Calculate elapsed time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Copy result back to host
        cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

        // Compute maximum error
        float maxError = 0.0f;
        for (int i = 0; i < N; i++)
            maxError = fmaxf(maxError, fabsf(y[i] - 4.0f));

        // Output results
        printf("N = 2^%d (%d elements): Execution time = %.3f ms, Max error = %f\n",
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
