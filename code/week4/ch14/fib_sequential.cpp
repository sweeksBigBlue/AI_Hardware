// fib_benchmark_cpu.cpp
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <vector>

#define CSV_FILENAME "fib_timing_results_cpu.csv"

unsigned long long fibonacci(int n) {
    if (n <= 1) return n;
    unsigned long long a = 0, b = 1, c;
    for (int i = 2; i <= n; ++i) {
        c = a + b;
        a = b;
        b = c;
    }
    return b;
}

void run_fibonacci_cpu(int n, FILE* fp) {
    std::vector<int> input(n);
    std::vector<unsigned long long> output(n);

    // Initialize input
    for (int i = 0; i < n; ++i) {
        input[i] = i % 50;  // Keep numbers small to avoid huge Fibonacci results
    }

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Sequential computation
    for (int i = 0; i < n; ++i) {
        output[i] = fibonacci(input[i]);
    }

    // Stop timing
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_ms = stop - start;

    // Print and write timing
    printf("CPU N = %d: Time = %.6f ms\n", n, duration_ms.count());

    if (fp != NULL) {
        fprintf(fp, "%d,%.6f\n", n, duration_ms.count());
    }
}

int main() {
    printf("Starting CPU Fibonacci Benchmark...\n");

    // Open CSV
    FILE *fp = fopen(CSV_FILENAME, "w");
    if (fp == NULL) {
        printf("Failed to open %s for writing.\n", CSV_FILENAME);
        return -1;
    }

    // Write header
    fprintf(fp, "N,CPU Time (ms)\n");

    // Sweep 2^3 to 2^20
    for (int exp = 3; exp <= 20; ++exp) {
        int n = 1 << exp;
        run_fibonacci_cpu(n, fp);
    }

    fclose(fp);

    printf("Benchmark complete. Timing data saved to %s\n", CSV_FILENAME);
    return 0;
}
