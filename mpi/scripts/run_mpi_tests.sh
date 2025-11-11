#!/bin/bash
set -euo pipefail

# Navigate to the MPI directory
cd ~/hpc_project_cpp/mpi

# Create directories if needed
mkdir -p results

# Output file
OUT=results/mpi_repeats.csv
echo "framework,operation,n,threads,processes,run,time_s,checksum" > "$OUT"

# Test parameters
procs=(1 2 4 6)
ns=(500 1000 2000 3000 4000)
repeats=5

echo "🚀 Starting MPI benchmark tests..."
echo "Configurations: ${#ns[@]} matrix sizes × ${#procs[@]} process counts × $repeats repeats"
echo ""

# Main benchmarking loop
total_runs=$((${#ns[@]} * ${#procs[@]} * $repeats))
current_run=0

for n in "${ns[@]}"; do
  for p in "${procs[@]}"; do
    for r in $(seq 1 $repeats); do
      current_run=$((current_run + 1))
      echo "Running: n=$n, processes=$p, run=$r/$repeats [$current_run/$total_runs]"
      
      # Run MPI program
      line=$(mpirun --bind-to core -np $p ./mpi_mat $n 2>/dev/null)
      
      # Extract time and checksum
      time=$(echo "$line" | grep -oP 'time\(s\)=\K[0-9.eE+-]+' || echo "0")
      cs=$(echo "$line" | grep -oP 'checksum=\K[0-9.eE+-]+' || echo "0")
      
      # Write to CSV
      echo "MPI,matrix,$n,1,$p,$r,$time,$cs" >> "$OUT"
      
      sleep 0.5
    done
  done
done

echo ""
echo "✅ Benchmarking complete!"
echo "📊 Results saved to: $OUT"
