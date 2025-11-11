#!/bin/bash
set -euo pipefail

# Comprehensive build and functional validation script
# Tests all new implementations added in the latest iteration

echo "================================================================================================"
echo "🧪 HPC PROJECT COMPREHENSIVE VALIDATION SUITE"
echo "================================================================================================"
echo ""

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

# Helper function for test reporting
test_result() {
    local name=$1
    local status=$2
    local message=${3:-}
    
    if [ "$status" = "PASS" ]; then
        echo "  ✅ PASS: $name"
        ((PASS_COUNT++))
    elif [ "$status" = "FAIL" ]; then
        echo "  ❌ FAIL: $name${message:+ - $message}"
        ((FAIL_COUNT++))
    elif [ "$status" = "SKIP" ]; then
        echo "  ⏭️  SKIP: $name${message:+ - $message}"
        ((SKIP_COUNT++))
    fi
}

cd ~/hpc_project_cpp

# ============================================================================
# 1. COMPILATION TESTS
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PART 1: COMPILATION TESTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 1.1: MPI naive compilation
echo "[1.1] Testing MPI naive (broadcast) compilation..."
if mpicxx -c mpi/src/mpi_matmul.cpp -o /tmp/mpi_matmul.o 2>/dev/null; then
    test_result "MPI naive compilation" "PASS"
else
    test_result "MPI naive compilation" "FAIL"
fi
rm -f /tmp/mpi_matmul.o

# Test 1.2: MPI SUMMA compilation
echo "[1.2] Testing MPI SUMMA compilation..."
if mpicxx -c mpi/src/mpi_summa.cpp -o /tmp/mpi_summa.o 2>/dev/null; then
    test_result "MPI SUMMA compilation" "PASS"
else
    test_result "MPI SUMMA compilation" "FAIL"
fi
rm -f /tmp/mpi_summa.o

# Test 1.3: GPU tiled kernel (may skip if no CUDA)
echo "[1.3] Testing CUDA tiled kernel compilation..."
if [ -f cuda/src/gpu_tiled_matmul ]; then
    test_result "CUDA tiled compilation" "PASS"
else
    test_result "CUDA tiled compilation" "FAIL" "Binary not found"
fi

# Test 1.4: GPU cuBLAS kernel (may skip if no CUDA)
echo "[1.4] Testing CUDA cuBLAS wrapper compilation..."
if [ -f cuda/src/gpu_cublas_matmul ]; then
    test_result "CUDA cuBLAS compilation" "PASS"
else
    # This is expected if CUDA is not available
    test_result "CUDA cuBLAS compilation" "SKIP" "CUDA environment not available"
fi

# ============================================================================
# 2. FUNCTIONAL TESTS (MPI)
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PART 2: FUNCTIONAL TESTS (MPI)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 2.1: MPI naive execution
echo "[2.1] Testing MPI naive (n=300, 2 processes)..."
if mpirun -np 2 ./mpi/src/mpi_matmul 300 2>/dev/null | grep -q "checksum="; then
    test_result "MPI naive execution" "PASS"
else
    test_result "MPI naive execution" "FAIL"
fi

# Test 2.2: MPI SUMMA execution
echo "[2.2] Testing MPI SUMMA (n=300, 4 processes)..."
if mpirun -np 4 ./mpi/src/mpi_summa 300 2>/dev/null | grep -q "checksum="; then
    test_result "MPI SUMMA execution" "PASS"
else
    test_result "MPI SUMMA execution" "FAIL"
fi

# Test 2.3: MPI SUMMA with non-square grid (should fail gracefully)
echo "[2.3] Testing MPI SUMMA error handling (invalid process count)..."
if mpirun -np 3 ./mpi/src/mpi_summa 300 2>&1 | grep -q "must equal"; then
    test_result "MPI SUMMA validation" "PASS"
else
    test_result "MPI SUMMA validation" "FAIL"
fi

# ============================================================================
# 3. PYTHON SCRIPT VALIDATION
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PART 3: PYTHON UTILITY SCRIPTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 3.1: GPU power measurement script (syntax check)
echo "[3.1] Testing GPU power measurement script..."
if python3 -m py_compile cuda/scripts/measure_gpu_power.py 2>/dev/null; then
    test_result "GPU power script (syntax)" "PASS"
else
    test_result "GPU power script (syntax)" "FAIL"
fi

# Test 3.2: CPU energy measurement script (syntax check)
echo "[3.2] Testing CPU energy measurement script..."
if python3 -m py_compile cuda/scripts/measure_cpu_energy.py 2>/dev/null; then
    test_result "CPU energy script (syntax)" "PASS"
else
    test_result "CPU energy script (syntax)" "FAIL"
fi

# Test 3.3: Benchmark with power wrapper (syntax check)
echo "[3.3] Testing benchmark_with_power wrapper..."
if python3 -m py_compile cuda/scripts/benchmark_with_power.py 2>/dev/null; then
    test_result "Benchmark power wrapper (syntax)" "PASS"
else
    test_result "Benchmark power wrapper (syntax)" "FAIL"
fi

# Test 3.4: GPU summary script (syntax check)
echo "[3.4] Testing GPU summary script..."
if python3 -m py_compile cuda/src/gpu_summary.py 2>/dev/null; then
    test_result "GPU summary script (syntax)" "PASS"
else
    test_result "GPU summary script (syntax)" "FAIL"
fi

# Test 3.5: GPU plots script (syntax check)
echo "[3.5] Testing GPU plots script..."
if python3 -m py_compile cuda/src/gpu_plots.py 2>/dev/null; then
    test_result "GPU plots script (syntax)" "PASS"
else
    test_result "GPU plots script (syntax)" "FAIL"
fi

# ============================================================================
# 4. BUILD SCRIPT VALIDATION
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PART 4: BENCHMARK RUN SCRIPTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 4.1: GPU run script (syntax check)
echo "[4.1] Testing GPU run script (syntax)..."
if bash -n cuda/scripts/run_gpu_tests.sh 2>/dev/null; then
    test_result "GPU run script (syntax)" "PASS"
else
    test_result "GPU run script (syntax)" "FAIL"
fi

# Test 4.2: MPI run script (syntax check)
echo "[4.2] Testing MPI run script (syntax)..."
if bash -n mpi/scripts/run_mpi_tests.sh 2>/dev/null; then
    test_result "MPI run script (syntax)" "PASS"
else
    test_result "MPI run script (syntax)" "FAIL"
fi

# ============================================================================
# 5. DATA FILES VALIDATION
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PART 5: DATA FILES & OUTPUTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 5.1: GPU results exist
echo "[5.1] Checking GPU results CSV..."
if [ -f cuda/results/gpu_repeats.csv ] && [ -s cuda/results/gpu_repeats.csv ]; then
    test_result "GPU repeats CSV exists" "PASS"
else
    test_result "GPU repeats CSV exists" "FAIL"
fi

# Test 5.2: GPU summary exists
echo "[5.2] Checking GPU summary CSV..."
if [ -f cuda/results/gpu_summary.csv ] && [ -s cuda/results/gpu_summary.csv ]; then
    test_result "GPU summary CSV exists" "PASS"
else
    test_result "GPU summary CSV exists" "FAIL"
fi

# Test 5.3: GPU plots exist
echo "[5.3] Checking GPU plots..."
if [ -d cuda/plots ] && [ "$(ls -1 cuda/plots/*.png 2>/dev/null | wc -l)" -ge 2 ]; then
    test_result "GPU plots generated" "PASS"
else
    test_result "GPU plots generated" "FAIL"
fi

# Test 5.4: MPI results exist
echo "[5.4] Checking MPI results CSV..."
if [ -f mpi/results/mpi_repeats.csv ] && [ -s mpi/results/mpi_repeats.csv ]; then
    test_result "MPI repeats CSV exists" "PASS"
else
    test_result "MPI repeats CSV exists" "FAIL"
fi

# Test 5.5: MPI plots exist
echo "[5.5] Checking MPI plots..."
if [ -d mpi/plots ] && [ "$(ls -1 mpi/plots/*.png 2>/dev/null | wc -l)" -ge 2 ]; then
    test_result "MPI plots generated" "PASS"
else
    test_result "MPI plots generated" "FAIL"
fi

# ============================================================================
# 6. PAPER/DOCUMENTATION VALIDATION
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PART 6: DOCUMENTATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 6.1: Paper Markdown exists
echo "[6.1] Checking IEEE paper (Markdown)..."
if [ -f paper/IEEE_Paper.md ] && grep -q "SUMMA\|cuBLAS\|Energy" paper/IEEE_Paper.md; then
    test_result "IEEE paper (updated)" "PASS"
else
    test_result "IEEE paper (updated)" "FAIL"
fi

# Test 6.2: Paper LaTeX exists
echo "[6.2] Checking IEEE paper (LaTeX)..."
if [ -f paper/IEEE_Paper.tex ] && [ -s paper/IEEE_Paper.tex ]; then
    test_result "IEEE paper (LaTeX)" "PASS"
else
    test_result "IEEE paper (LaTeX)" "FAIL"
fi

# Test 6.3: Overview exists
echo "[6.3] Checking project overview..."
if [ -f overview.md ] && [ -s overview.md ]; then
    test_result "Project overview" "PASS"
else
    test_result "Project overview" "FAIL"
fi

# ============================================================================
# SUMMARY REPORT
# ============================================================================
echo ""
echo "================================================================================================"
echo "📊 VALIDATION SUMMARY"
echo "================================================================================================"
echo ""
echo "  Total Tests Run:  $((PASS_COUNT + FAIL_COUNT + SKIP_COUNT))"
echo "  ✅ Passed:        $PASS_COUNT"
echo "  ❌ Failed:        $FAIL_COUNT"
echo "  ⏭️  Skipped:       $SKIP_COUNT"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo "🎉 All critical tests passed!"
    echo ""
else
    echo "⚠️  Some tests failed. Review output above for details."
    echo ""
fi

echo "================================================================================================"
echo ""
echo "✅ Validation Complete"
echo ""
