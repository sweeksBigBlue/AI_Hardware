#!/bin/bash

# Batch sizes to test
BATCH_SIZES=(1 2 4 8 16 32 64)

# Path to source file
BASE_SRC="mlp_cuda_batched.cu"
TMP_SRC="mlp_tmp.cu"
EXE="mlp_cuda_batched"

# Clean previous CSV output
CSV="mlp_timing_log.csv"
rm -f "$CSV"

# Extract code sections before and after the #define BATCH_SIZE
PRE_DEFINE=$(awk '!/#define BATCH_SIZE/{print}' "$BASE_SRC")
POST_DEFINE=$(awk '/#define BATCH_SIZE/{getline; while(getline) print}' "$BASE_SRC")

# Loop over all batch sizes
for B in "${BATCH_SIZES[@]}"; do
    echo "Running for BATCH_SIZE=$B..."

    # Create modified temp source file
    echo "$PRE_DEFINE" > "$TMP_SRC"
    echo "#define BATCH_SIZE $B" >> "$TMP_SRC"
    echo "$POST_DEFINE" >> "$TMP_SRC"

    # Compile for architecture sm_75
    nvcc -o "$EXE" "$TMP_SRC" -gencode arch=compute_75,code=sm_75

    # Run the executable
    ./"$EXE"
done

# Cleanup
rm -f "$TMP_SRC" "$EXE"

echo "✅ Benchmarking complete. Results saved in $CSV"
