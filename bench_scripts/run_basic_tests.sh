cd ~/hpc_project_cpp

# Sequential
g++ -O3 -march=native sequential_matmul.cpp -o seq_mat

# OpenMP
g++ -O3 -march=native -fopenmp openmp_matmul.cpp -o omp_mat

# MPI hello
mpicxx -O3 mpi_hello.cpp -o mpi_hello

#!/bin/bash
# ===========================================
# Benchmark Script: Sequential + OpenMP Tests
# Author: Gaurav
# Purpose: Collects runtime and checksum data for HPC experiment
# ===========================================

set -euo pipefail

# Output CSV
OUT=results/day1_basic_fixed.csv
mkdir -p results
echo "framework,operation,n,threads,processes,run,time_s,checksum" > "$OUT"

# ---------- Helper Functions ----------
# Extract time(s) and checksum from program output line:
# Expected format: n=500 threads=4 time(s)=0.113466 checksum=3.13075e+07
extract_time() {
  echo "$1" | grep -oP 'time\(s\)=\K[0-9.eE+-]+' || echo ""
}
extract_checksum() {
  echo "$1" | grep -oP 'checksum=\K[0-9.eE+-]+' || echo ""
}

# ---------- Sequential Tests ----------
echo "Running Sequential benchmarks..."
for n in 500 1000 2000; do
  echo "Sequential test for n=$n"
  line=$(./seq_mat "$n")
  time=$(extract_time "$line")
  cs=$(extract_checksum "$line")
  echo "sequential,matrix,$n,1,1,1,$time,$cs" >> "$OUT"
done

# ---------- OpenMP Tests ----------
echo "Running OpenMP benchmarks..."
for n in 500 1000 2000; do
  for th in 1 2 4 6 12; do
    export OMP_NUM_THREADS=$th
    echo "OpenMP test for n=$n threads=$th"
    line=$(./omp_mat "$n" "$th")
    time=$(extract_time "$line")
    cs=$(extract_checksum "$line")
    echo "OpenMP,matrix,$n,$th,1,1,1,$time,$cs" >> "$OUT"
  done
done

# ---------- MPI Hello (Verification only) ----------
echo "Running MPI hello test..."
mpirun -np 4 ./mpi_hello

echo "==========================================="
echo "✅  Done: Results saved to $OUT"
echo "==========================================="
