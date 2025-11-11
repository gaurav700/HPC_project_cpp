#!/bin/bash
set -euo pipefail

# Navigate to MPI directory
cd ~/hpc_project_cpp/mpi

# Compile MPI implementations if needed
echo "🔧 Building MPI implementations..."
cd src

if [ ! -f mpi_matmul ] || [ "${FORCE_REBUILD:-0}" = "1" ]; then
    echo "  - compiling mpi_matmul (naive broadcast)..."
    mpicxx -O3 mpi_matmul.cpp -o mpi_matmul 2>/dev/null || echo "⚠️  Failed to compile mpi_matmul"
fi

if [ ! -f mpi_summa ] || [ "${FORCE_REBUILD:-0}" = "1" ]; then
    echo "  - compiling mpi_summa (SUMMA algorithm)..."
    mpicxx -O3 mpi_summa.cpp -o mpi_summa 2>/dev/null || echo "⚠️  Failed to compile mpi_summa"
fi

cd ..

# Create results directory
mkdir -p results

# Output file
OUT=results/mpi_repeats.csv
echo "framework,impl,n,processes,run,time_s,checksum" > "$OUT"

# Test parameters
ns=(500 1000 2000)
impls=("mpi_matmul" "mpi_summa")
process_counts=(4)  # default: 4 processes (2x2 grid). Override with MPI_PROCESSES env var
if [ -n "${MPI_PROCESSES:-}" ]; then
    process_counts=($MPI_PROCESSES)
fi
repeats=3

echo "🚀 Starting MPI benchmark tests..."
echo "Configurations: ${#ns[@]} matrix sizes × ${#impls[@]} implementations × ${#process_counts[@]} process counts × $repeats repeats"
echo ""

# Check for SLURM environment
USE_SLURM=0
if [ -n "${SLURM_PROCID:-}" ]; then
    USE_SLURM=1
    echo "✓ Detected SLURM environment; will use srun"
fi

total_runs=$((${#ns[@]} * ${#impls[@]} * ${#process_counts[@]} * $repeats))
current_run=0

for n in "${ns[@]}"; do
  for impl in "${impls[@]}"; do
    for np in "${process_counts[@]}"; do
      # Validate that np is a perfect square for SUMMA
      sqrt_np=$(echo "sqrt($np)" | bc)
      if [ "$impl" = "mpi_summa" ] && [ $((sqrt_np * sqrt_np)) -ne $np ]; then
        echo "⚠️  Skipping mpi_summa with $np processes (not a perfect square)"
        continue
      fi
      
      for r in $(seq 1 $repeats); do
        current_run=$((current_run + 1))
        echo "Running: n=$n, impl=$impl, processes=$np, run=$r/$repeats [$current_run/$total_runs]"
        
        # Run MPI benchmark
        if [ "$USE_SLURM" = "1" ]; then
            # Use srun under SLURM
            line=$(srun -n $np ./src/$impl $n 2>/dev/null || echo "n=$n processes=$np time(s)=0 checksum=0")
        else
            # Use mpirun locally
            line=$(mpirun -np $np ./src/$impl $n 2>/dev/null || echo "n=$n processes=$np time(s)=0 checksum=0")
        fi
        
        time_s=$(echo "$line" | grep -oP 'time\(s\)=\K[0-9.eE+-]+' || echo "0")
        cs=$(echo "$line" | grep -oP 'checksum=\K[0-9.eE+-]+' || echo "0")
        
        # Write to CSV
        echo "MPI,$impl,$n,$np,$r,$time_s,$cs" >> "$OUT"
        
        sleep 0.5
      done
    done
  done
done

echo ""
echo "✅ Benchmarking complete!"
echo "📊 Results saved to: $OUT"
echo ""
echo "To run with different configurations:"
echo "  MPI_PROCESSES='4 9 16' FORCE_REBUILD=1 ./mpi/scripts/run_mpi_tests.sh"
echo "To use SLURM (on a cluster):"
echo "  sbatch -n 16 -c 1 --wrap='cd ~/hpc_project_cpp && ./mpi/scripts/run_mpi_tests.sh'"
