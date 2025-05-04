#!/bin/bash

# Batch sizes to test
BATCH_SIZES=(1 2 4 8 16 32 64)

# CUDA source and output files
SRC="mlp_cuda_batched.cu"
EXE="mlp_cuda_batched"
CSV="mlp_timing_log.csv"

# Clean CSV file
rm -f "$CSV"

# Loop through batch sizes
for B in "${BATCH_SIZES[@]}"; do
    echo "Running for BATCH_SIZE=$B..."

    # Replace BATCH_SIZE in the CUDA file using sed
    sed "s/^#define BATCH_SIZE .*/#define BATCH_SIZE $B/" "$SRC" > tmp_$SRC

    # Compile with target GPU architecture (e.g., Turing sm_75)
    nvcc -o "$EXE" "tmp_$SRC" -gencode arch=compute_75,code=sm_75

    # Run the compiled binary
    ./"$EXE"
done

# Clean up
rm -f tmp_$SRC "$EXE"

echo "✅ Benchmarking complete. Results in $CSV"
