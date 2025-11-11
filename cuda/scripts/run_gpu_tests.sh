#!/bin/bash
set -euo pipefail

# Navigate to CUDA directory
cd ~/hpc_project_cpp/cuda

# Compile GPU kernels if not already compiled
echo "🔧 Building GPU kernels..."
NVCC_ARCH=${NVCC_ARCH:-sm_86}
# Use gencode for better control; derive compute from arch (sm_86 -> compute_86)
compute_cap=${NVCC_ARCH#sm_}
gencode_flag="-gencode=arch=compute_${compute_cap},code=sm_${compute_cap}"
cd src
if [ ! -f gpu_tiled_matmul ] || [ "${FORCE_REBUILD:-0}" = "1" ]; then
  echo "  - compiling gpu_tiled_matmul (arch=${NVCC_ARCH})..."
  nvcc -O3 $gencode_flag gpu_tiled_matmul.cu -o gpu_tiled_matmul 2>/dev/null || {
    echo "⚠️  Failed to compile gpu_tiled_matmul with ${gencode_flag}, trying -arch=${NVCC_ARCH}..."
    nvcc -O3 -arch=${NVCC_ARCH} gpu_tiled_matmul.cu -o gpu_tiled_matmul 2>/dev/null || echo "⚠️  Fallback compile failed for gpu_tiled_matmul"
  }
fi

if [ ! -f gpu_cublas_matmul ] || [ "${FORCE_REBUILD:-0}" = "1" ]; then
  echo "  - compiling gpu_cublas_matmul (arch=${NVCC_ARCH})..."
  nvcc -O3 $gencode_flag gpu_cublas_matmul.cu -lcublas -o gpu_cublas_matmul 2>/dev/null || {
    echo "⚠️  Failed to compile gpu_cublas_matmul with ${gencode_flag}, trying -arch=${NVCC_ARCH}..."
    nvcc -O3 -arch=${NVCC_ARCH} gpu_cublas_matmul.cu -lcublas -o gpu_cublas_matmul 2>/dev/null || echo "⚠️  Fallback compile failed for gpu_cublas_matmul"
  }
fi
cd ..

# Create results directory
mkdir -p results

# Output file
OUT=results/gpu_repeats.csv
echo "framework,impl,n,run,kernel_ms,checksum" > "$OUT"

# Test parameters
ns=(500 1000 2000 3000 4000)
# include both tiled kernel and cuBLAS baseline
impls=("gpu_tiled_matmul" "gpu_cublas_matmul")
repeats=5

echo "🚀 Starting GPU benchmark tests..."
echo "Configurations: ${#ns[@]} matrix sizes × ${#impls[@]} implementation × $repeats repeats"
echo ""

# Main benchmarking loop
total_runs=$((${#ns[@]} * ${#impls[@]} * $repeats))
current_run=0

for n in "${ns[@]}"; do
  for impl in "${impls[@]}"; do
    for r in $(seq 1 $repeats); do
      current_run=$((current_run + 1))
      echo "Running: n=$n, impl=$impl, run=$r/$repeats [$current_run/$total_runs]"
      
      # Run GPU kernel (tiled)
      # Tiled kernel prints: n=2000 kernel_ms=0.123 checksum=...
      line=$(./src/$impl $n 2>/dev/null || echo "n=$n kernel_ms=0 checksum=0")
      kernel=$(echo "$line" | grep -oP 'kernel_ms=\K[0-9.eE+-]+' || echo "0")
      cs=$(echo "$line" | grep -oP 'checksum=\K[0-9.eE+-]+' || echo "0")
      
      # Write to CSV
      echo "GPU,$impl,$n,$r,$kernel,$cs" >> "$OUT"
      
      sleep 0.5
    done
  done
done

echo ""
echo "✅ Benchmarking complete!"
echo "📊 Results saved to: $OUT"
