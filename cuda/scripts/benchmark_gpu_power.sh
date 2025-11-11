#!/bin/bash
# GPU Energy and Power Benchmarking Script
# Runs GPU benchmarks with concurrent power sampling

set -euo pipefail

cd ~/hpc_project_cpp/cuda

echo "================================================================================================"
echo "🔋 GPU ENERGY & POWER BENCHMARKING"
echo "================================================================================================"
echo ""

mkdir -p results/energy_measurements

ENERGY_LOG="results/energy_measurements/gpu_power_measurements.csv"
echo "test_name,matrix_size,impl,run,duration_s,avg_power_w,min_power_w,max_power_w,stdev_power_w,total_energy_j" > "$ENERGY_LOG"

# Test parameters
ns=(1000 2000 3000 4000)
repeats=3

echo "📊 Running GPU benchmarks with power sampling..."
echo ""

for n in "${ns[@]}"; do
  for r in $(seq 1 $repeats); do
    test_name="GPU_tiled_n${n}_run${r}"
    
    echo "Running: $test_name"
    
    # Start power sampling in background
    python3 scripts/measure_gpu_power.py --duration 15 --interval 0.05 > /tmp/gpu_power_$$.txt 2>&1 &
    POWER_PID=$!
    
    # Give sampler time to start
    sleep 0.5
    
    # Run the GPU benchmark
    start_time=$(date +%s%N)
    ./src/gpu_tiled_matmul $n > /tmp/gpu_bench_$$.txt 2>&1
    end_time=$(date +%s%N)
    
    # Calculate duration
    duration_ns=$((end_time - start_time))
    duration_s=$(echo "scale=3; $duration_ns / 1000000000" | bc)
    
    # Wait for power sampler to finish
    wait $POWER_PID 2>/dev/null || true
    
    # Parse power measurements
    if [ -f /tmp/gpu_power_$$.txt ]; then
      power_data=$(grep "avg=" /tmp/gpu_power_$$.txt | tail -1)
      
      if [[ $power_data =~ avg=([0-9.]+)\ W\ min=([0-9.]+)\ W\ max=([0-9.]+)\ W\ stdev=([0-9.]+) ]]; then
        avg_power="${BASH_REMATCH[1]}"
        min_power="${BASH_REMATCH[2]}"
        max_power="${BASH_REMATCH[3]}"
        stdev_power="${BASH_REMATCH[4]}"
        
        # Estimate total energy (average power * duration)
        total_energy=$(echo "scale=3; $avg_power * $duration_s" | bc)
        
        echo "gpu_tiled_n${n}_run${r},${n},gpu_tiled_matmul,${r},${duration_s},${avg_power},${min_power},${max_power},${stdev_power},${total_energy}" >> "$ENERGY_LOG"
        
        echo "  ✅ Avg Power: ${avg_power} W | Energy: ${total_energy} J"
      fi
    fi
    
    # Cleanup
    rm -f /tmp/gpu_power_$$.txt /tmp/gpu_bench_$$.txt
    sleep 1
  done
done

echo ""
echo "================================================================================================"
echo "✅ GPU Energy measurements saved to: $ENERGY_LOG"
echo "================================================================================================"
echo ""

# Display summary
echo "📈 GPU POWER SUMMARY:"
echo ""
cat "$ENERGY_LOG" | tail -n +2 | awk -F',' '{
  n=$2; avg=$6; energy=$10
  if(n in size_count) {
    size_count[n]++
    size_total_power[n] += avg
    size_total_energy[n] += energy
  } else {
    size_count[n]=1
    size_total_power[n]=avg
    size_total_energy[n]=energy
  }
}
END {
  printf "%-15s %-15s %-15s\n", "Matrix Size", "Avg Power (W)", "Total Energy (J)"
  printf "%-15s %-15s %-15s\n", "==============", "==============", "==============="
  for(n in size_count) {
    printf "%-15s %-15.2f %-15.2f\n", n, size_total_power[n]/size_count[n], size_total_energy[n]/size_count[n]
  }
}'

echo ""
