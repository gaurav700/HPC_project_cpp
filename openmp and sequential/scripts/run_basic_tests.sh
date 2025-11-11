#!/bin/bash
set -euo pipefail

# Navigate to the sequential/openmp directory
cd ~/hpc_project_cpp/openmp\ and\ sequential

# Create directories if needed
mkdir -p results

# Compile programs
echo "🔨 Compiling Sequential program..."
g++ -O3 -march=native src/sequential_matmul.cpp -o seq_mat
if [ ! -f ./seq_mat ]; then
    echo "❌ Sequential compilation failed"
    exit 1
fi
echo "✅ Sequential compilation successful"

echo "🔨 Compiling OpenMP program..."
g++ -O3 -march=native -fopenmp src/openmp_matmul.cpp -o omp_mat
if [ ! -f ./omp_mat ]; then
    echo "❌ OpenMP compilation failed"
    exit 1
fi
echo "✅ OpenMP compilation successful"

# Output file
OUT=results/sequential_openmp_repeats.csv
echo "framework,operation,n,threads,processes,run,time_s,checksum" > "$OUT"

# Test parameters
ns=(500 1000 2000 3000 4000)
threads=(1 2 4 6)
repeats=5

echo ""
echo "🚀 Starting Sequential & OpenMP benchmark tests..."
echo "Configurations: ${#ns[@]} matrix sizes × (1 sequential + ${#threads[@]} OpenMP configs) × $repeats repeats"
echo ""

export OMP_DYNAMIC=false

# Calculate total runs
total_seq=$((${#ns[@]} * $repeats))
total_omp=$((${#ns[@]} * ${#threads[@]} * $repeats))
total_runs=$((total_seq + total_omp))
current_run=0

# Sequential benchmarks
echo "📊 Running Sequential benchmarks..."
for n in "${ns[@]}"; do
  for r in $(seq 1 $repeats); do
    current_run=$((current_run + 1))
    echo "  n=$n, run=$r/$repeats [$current_run/$total_runs]"
    
    line=$(./seq_mat "$n")
    time=$(echo "$line" | grep -oP 'time\(s\)=\K[0-9.eE+-]+' || echo "0")
    cs=$(echo "$line" | grep -oP 'checksum=\K[0-9.eE+-]+' || echo "0")
    echo "sequential,matrix,$n,1,1,$r,$time,$cs" >> "$OUT"
  done
done

# OpenMP benchmarks
echo ""
echo "📊 Running OpenMP benchmarks..."
for n in "${ns[@]}"; do
  for th in "${threads[@]}"; do
    export OMP_NUM_THREADS=$th
    for r in $(seq 1 $repeats); do
      current_run=$((current_run + 1))
      echo "  n=$n, threads=$th, run=$r/$repeats [$current_run/$total_runs]"
      
      line=$(./omp_mat "$n")
      time=$(echo "$line" | grep -oP 'time\(s\)=\K[0-9.eE+-]+' || echo "0")
      cs=$(echo "$line" | grep -oP 'checksum=\K[0-9.eE+-]+' || echo "0")
      echo "OpenMP,matrix,$n,$th,1,$r,$time,$cs" >> "$OUT"
    done
  done
done

echo ""
echo "✅ Benchmarking complete!"
echo "📊 Results saved to: $OUT"
echo ""
