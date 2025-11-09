cd ~/hpc_project_cpp

# Sequential
g++ -O3 -march=native sequential_matmul.cpp -o seq_mat

# OpenMP
g++ -O3 -march=native -fopenmp openmp_matmul.cpp -o omp_mat

# MPI hello
mpicxx -O3 mpi_hello.cpp -o mpi_hello

#!/bin/bash
set -euo pipefail
OUT=results/day1_repeats.csv
mkdir -p results
echo "framework,operation,n,threads,processes,run,time_s,checksum" > "$OUT"

ns=(500 1000 2000)
threads=(1 2 4 6 12)
repeats=5

# Warm-up
./seq_mat 500 >/dev/null
export OMP_DYNAMIC=false
for n in "${ns[@]}"; do
  ./omp_mat $n 2 >/dev/null || true
done

for n in "${ns[@]}"; do
  # sequential repeats
  for r in $(seq 1 $repeats); do
    line=$(./seq_mat "$n")
    time=$(echo "$line" | grep -oP 'time\(s\)=\K[0-9.eE+-]+')
    cs=$(echo "$line" | grep -oP 'checksum=\K[0-9.eE+-]+')
    echo "sequential,matrix,$n,1,1,$r,$time,$cs" >> "$OUT"
  done

  # openmp repeats
  for th in "${threads[@]}"; do
    export OMP_NUM_THREADS=$th
    for r in $(seq 1 $repeats); do
      line=$(./omp_mat "$n" "$th")
      time=$(echo "$line" | grep -oP 'time\(s\)=\K[0-9.eE+-]+')
      cs=$(echo "$line" | grep -oP 'checksum=\K[0-9.eE+-]+')
      echo "OpenMP,matrix,$n,$th,1,$r,$time,$cs" >> "$OUT"
    done
  done
done

echo "Done: results -> $OUT"
