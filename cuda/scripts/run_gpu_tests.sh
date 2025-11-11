#!/bin/bash
set -euo pipefail

# Navigate to CUDA directory
cd ~/hpc_project_cpp/cuda

# Compile GPU kernels if not already compiled
echo "🔧 Building GPU kernels..."
if [ ! -f src/gpu_tiled_matmul ]; then
    cd src
    echo "  - compiling gpu_tiled_matmul..."
    nvcc -O3 -arch=sm_70 gpu_tiled_matmul.cu -o gpu_tiled_matmul 2>/dev/null || echo "⚠️  Failed to compile gpu_tiled_matmul"
    cd ..
fi

# Create results directory
mkdir -p results

# Output file
OUT=results/gpu_repeats.csv
echo "framework,impl,n,run,kernel_ms,checksum" > "$OUT"

# Test parameters
ns=(500 1000 2000 3000 4000)
impls=("gpu_tiled_matmul")
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
