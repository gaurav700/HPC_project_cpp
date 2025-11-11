#!/bin/bash
set -euo pipefail

# validation_tests.sh
# Quick validation of all implementations and measurements

echo "======================================================================"
echo "🧪 HPC Project Quick Validation Tests"
echo "======================================================================"
echo ""

cd ~/hpc_project_cpp

passed=0
failed=0

# Test 1: CUDA tiled matmul
echo "[1/8] Testing CUDA tiled kernel..."
if cd cuda/src && ./gpu_tiled_matmul 512 >/dev/null 2>&1; then
    echo "  ✅ CUDA tiled kernel works"
    ((passed++))
else
    echo "  ❌ CUDA tiled kernel failed"
    ((failed++))
fi
cd ~/hpc_project_cpp

# Test 2: CUDA cuBLAS
echo "[2/8] Testing CUDA cuBLAS implementation..."
if cd cuda/src && ./gpu_cublas_matmul 512 >/dev/null 2>&1; then
    echo "  ✅ cuBLAS implementation works"
    ((passed++))
else
    echo "  ❌ cuBLAS implementation failed (CUDA/cuBLAS issue)"
    ((failed++))
fi
cd ~/hpc_project_cpp

# Test 3: MPI matmul (naive)
echo "[3/8] Testing MPI naive broadcast..."
if cd mpi/src && mpirun -np 2 ./mpi_matmul 512 >/dev/null 2>&1; then
    echo "  ✅ MPI naive broadcast works"
    ((passed++))
else
    echo "  ❌ MPI naive broadcast failed"
    ((failed++))
fi
cd ~/hpc_project_cpp

# Test 4: MPI SUMMA
echo "[4/8] Testing MPI SUMMA algorithm..."
if cd mpi/src && mpirun -np 4 ./mpi_summa 512 >/dev/null 2>&1; then
    echo "  ✅ MPI SUMMA works"
    ((passed++))
else
    echo "  ❌ MPI SUMMA failed (requires 4 processes for 2x2 grid)"
    ((failed++))
fi
cd ~/hpc_project_cpp

# Test 5: GPU power measurement script
echo "[5/8] Testing GPU power measurement script..."
if python3 cuda/scripts/measure_gpu_power.py --duration 0.5 --interval 0.1 >/dev/null 2>&1; then
    echo "  ✅ GPU power measurement works"
    ((passed++))
else
    echo "  ⚠️  GPU power measurement unavailable (pynvml or nvidia-smi)"
    ((failed++))
fi

# Test 6: CPU energy measurement script
echo "[6/8] Testing CPU energy measurement script..."
if python3 cuda/scripts/measure_cpu_energy.py --duration 0.5 --interval 0.1 >/dev/null 2>&1; then
    echo "  ✅ CPU energy measurement works"
    ((passed++))
else
    echo "  ⚠️  CPU energy measurement unavailable (RAPL not accessible)"
    ((failed++))
fi

# Test 7: GPU summary script
echo "[7/8] Testing GPU summary generation..."
if python3 cuda/src/gpu_summary.py >/dev/null 2>&1; then
    echo "  ✅ GPU summary generation works"
    ((passed++))
else
    echo "  ❌ GPU summary generation failed"
    ((failed++))
fi

# Test 8: GPU plots script
echo "[8/8] Testing GPU plots generation..."
if python3 cuda/src/gpu_plots.py >/dev/null 2>&1; then
    echo "  ✅ GPU plots generation works"
    ((passed++))
else
    echo "  ❌ GPU plots generation failed"
    ((failed++))
fi

echo ""
echo "======================================================================"
echo "📊 Test Summary"
echo "======================================================================"
echo "Passed: $passed / 8"
echo "Failed: $failed / 8"
if [ $failed -eq 0 ]; then
    echo "✅ All tests passed!"
    exit 0
else
    echo "⚠️  Some tests failed or skipped (may be environment-dependent)"
    exit 1
fi
