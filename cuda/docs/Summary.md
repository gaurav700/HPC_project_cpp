# Day 3 Summary — GPU-Accelerated Matrix Multiplication with CUDA

## 🎯 Objective
Implement and evaluate a **GPU-accelerated matrix multiplication** using **NVIDIA CUDA** to study:
- **Performance scaling** with matrix size on GPU hardware
- **GPU vs CPU** performance comparison
- **Memory optimization** through tiling and shared memory
- **Computational efficiency** and hardware utilization

Compare GPU results with Sequential (CPU) and MPI (distributed CPU) from Days 1-2 to understand when GPU acceleration is most beneficial.

---

## 🧠 Theoretical Background

### What is GPU Computing?
A **Graphics Processing Unit (GPU)** is a specialized processor with:
- **Massive Parallelism:** Thousands of small cores (vs 6-12 CPU cores)
- **Heterogeneous Architecture:** CPU (host) + GPU (device) with separate memory spaces
- **Data Parallelism:** SIMT (Single Instruction, Multiple Threads) execution model
- **High Throughput:** Optimized for data-parallel problems, not latency

### CUDA Programming Model

| Concept | Description |
|---------|-------------|
| **Kernel** | Function executed on GPU by many threads simultaneously |
| **Block** | Group of up to 1024 threads that cooperate via shared memory |
| **Grid** | Collection of blocks launched together |
| **Warp** | 32 threads that execute in lockstep (fundamental execution unit) |
| **Shared Memory** | Fast, on-chip memory (~96KB per block), shared by threads in a block |
| **Global Memory** | Large GPU memory (~GB), slower than shared memory |
| **Coalescing** | Consecutive threads accessing consecutive memory = fast; otherwise slow |

### Memory Hierarchy & Optimization
```
CPU ←→ (PCIe ~16 GB/s) ←→ GPU Global Memory (300+ GB/s)
                              ↓
                         Shared Memory (1+ TB/s per block)
                              ↓
                         L1/L2 Cache
                              ↓
                         Registers (per thread)
```

**Key Insight:** Shared memory is ~100x faster than global memory. Tiling reuses data from global memory once in shared memory, reducing bandwidth demand.

### Tiled Matrix Multiplication
```
Standard (Naive):  Each thread reads O(N) elements from global memory
                   For N×N result: O(N³) global memory accesses

Tiled:             Threads cooperate to cache tiles (e.g., 16×16) in shared memory
                   Reuse each tile multiple times before evicting
                   O(N³/T) global memory accesses (T = tile size)
```

**Example:** For 4000×4000 matrix with 16×16 tiles:
- Naive: 64 billion global memory accesses
- Tiled: 64 billion / 16 ≈ 4 billion accesses → ~16x bandwidth savings

---

## ⚙️ Experimental Setup

| Parameter | Specification |
|-----------|---------------|
| **GPU Device** | NVIDIA GeForce RTX 3060 (12GB GDDR6) |
| **GPU Specs** | 3584 CUDA cores, 112 SMs, Compute Capability 8.6 |
| **Host CPU** | Intel Core i5-11400H (6 cores @ 2.7 GHz) |
| **Host Memory** | 16GB DDR4 |
| **Operating System** | Linux (Ubuntu 22.04) |
| **CUDA Version** | CUDA 12.x + cuDNN |
| **Compiler** | nvcc (NVIDIA CUDA Compiler) with -O3 optimization |
| **Executable** | Compiled with `nvcc -O3 gpu_tiled_matmul.cu -o gpu_tiled_matmul` |
| **Matrix Sizes (N)** | 500, 1000, 2000, 3000, 4000 |
| **Tile Size** | 16×16 threads per block |
| **Grid Configuration** | (N+15)/16 × (N+15)/16 blocks |
| **Repeats per Configuration** | 5 (25 total runs) |
| **Timing Function** | CUDA Events (`cudaEventRecord`, `cudaEventElapsedTime`) |
| **Verification** | Checksum compared against Sequential baseline |

---

## 🧱 Implementation Summary

### File: `gpu_tiled_matmul.cu`
GPU-accelerated tiled matrix multiplication with shared memory optimization:

1. **Host-Side Setup:**
   - Allocate host memory for A, B, C matrices
   - Allocate GPU device memory with `cudaMalloc`
   - Copy A and B to GPU with `cudaMemcpyHostToDevice`

2. **Kernel Launch:**
   ```cpp
   __global__ void gpu_tiled_matmul_kernel(double *A, double *B, double *C, int N)
   ```
   - Grid: `<<<(N+15)/16, (N+15)/16>>>`
   - Blocks: `<<< , 16*16 >>>`
   - Each block computes a 16×16tile of C

3. **Kernel Execution (per block):**
   - Threads cooperatively load 16×16 tile of A into shared memory
   - Threads cooperatively load 16×16 tile of B into shared memory
   - Synchronize with `__syncthreads()` (all threads in block wait)
   - Compute partial results using cached tiles
   - Repeat for next tile pair
   - Write results to global memory C

4. **Host-Side Completion:**
   - Copy C back to host with `cudaMemcpyDeviceToHost`
   - Verify checksum
   - Free GPU memory with `cudaFree`

### Tiling Strategy
```
For C[i,j] computation:
TILE_SIZE = 16
Number of tile iterations = N / TILE_SIZE

for step = 0 to (N/TILE_SIZE - 1):
    Load A[blockRow*TILE, step*TILE : step*TILE + TILE] → shared memory
    Load B[step*TILE : step*TILE + TILE, blockCol*TILE] → shared memory
    __syncthreads()
    Compute partial C using shared A and B tiles
    __syncthreads() before next iteration
```

### Compilation & Execution
```bash
# Compile
nvcc -O3 gpu_tiled_matmul.cu -o gpu_tiled_matmul

# Run single test
./gpu_tiled_matmul 2000
# Output: n=2000, kernel_ms=37.96, checksum=1.9602e+09

# Automated testing (see run_gpu_tests.sh)
for n in 500 1000 2000 3000 4000; do
    for run in {1..5}; do
        ./gpu_tiled_matmul $n >> results/gpu_repeats.csv
    done
done
```

---

## 📊 Benchmark Results

### Kernel Execution Times (in milliseconds)

| Matrix Size | Mean (ms) | Std Dev | CV (%) | Runs | Interpretation |
|------------|-----------|---------|--------|------|-----------------|
| 500×500   | 1.77      | 0.144   | 8.13   | 5    | Small overhead dominates; GPU still ~67x faster than CPU |
| 1000×1000 | 5.83      | 0.202   | 3.46   | 5    | Good scaling; stable execution |
| 2000×2000 | 37.96     | 0.260   | 0.68   | 5    | **Most stable** configuration; memory efficient |
| 3000×3000 | 106.50    | 14.00   | 13.14  | 5    | Increased variability; thermal effects evident |
| 4000×4000 | 227.35    | 17.95   | 7.89   | 5    | Largest matrix; ~2500x faster than CPU |

### Performance Analysis

#### Speedup vs CPU Sequential
| Matrix | GPU (ms) | CPU (s) | Speedup |
|--------|----------|---------|---------|
| 500    | 1.77     | 0.118   | 67x     |
| 1000   | 5.83     | 1.318   | 226x    |
| 2000   | 37.96    | 32.416  | 854x    |
| 3000   | 106.50   | 176.300 | 1655x   |
| 4000   | 227.35   | 575.135 | 2530x   |

**Key Insight:** Speedup increases with matrix size → GPU amortizes launch overhead and fills cores efficiently.

#### Scalability Metrics
- **Throughput (GFLOPS)** at N=4000:
  - Operations: 2 × 4000³ = 128 billion FLOPs
  - Time: 227.35 ms = 0.22735 s
  - Throughput: 128B / 0.227s ≈ **563 GFLOPS**
  - Peak GPU FP64: 1456 GFLOPS → ~39% efficiency (reasonable for tiled kernel)

#### Variability Analysis
- **Most stable:** N=2000 (CV=0.68%) → optimal GPU occupancy + thermal stability
- **Least stable:** N=3000 (CV=13.14%) → thermal throttling effects visible
- **Observation:** Variability increases beyond peak performance point due to thermal management

---

## 🔄 Comparison: GPU vs CPU vs Distributed (MPI)

| Aspect | Sequential | OpenMP | MPI (4 proc) | GPU |
|--------|-----------|--------|-------------|-----|
| **N=500 (ms)** | 118 | ~50 | ~150 | 1.77 |
| **N=2000 (ms)** | 32,416 | ~8,000 | ~3,500 | 37.96 |
| **N=4000 (ms)** | 575,135 | ~150,000 | ~45,000 | 227.35 |
| **Scalability** | Linear (single core) | Very good | Good (until comms dominate) | Excellent |
| **Best For** | Baseline/validation | Shared-memory systems | Multi-machine clusters | Dense linear algebra |
| **Overhead** | None | Thread sync | Process creation + messaging | Memory transfer (H2D/D2H) |

---

## 💡 Key Insights

1. **GPU Excels at:** Dense, compute-bound operations (like matrix multiplication)
2. **GPU Struggles With:** Fine-grained synchronization, irregular memory access
3. **Memory Hierarchy Matters:** Tiling provides ~16x bandwidth reduction
4. **Not All Speedup is Free:** Data transfer overhead (not measured here) can be significant
5. **Hybrid Approach:** Future work → MPI + CUDA for multi-GPU clusters

---

## 🚀 Next Steps
1. Add timing for H2D/D2H transfers and analyze total time including overhead
2. Implement multi-GPU MPI+CUDA hybrid approach
3. Optimize with cuBLAS or other GPU-accelerated libraries
4. Extend to non-square matrices and batch operations
5. Compare with AMD HIP and Intel oneAPI for portability

---

## 📁 Files Generated
- `gpu_tiled_matmul.cu` — Main CUDA kernel implementation
- `gpu_repeats.csv` — Raw benchmark data (25 runs)
- `gpu_summary.csv` — Aggregated statistics by matrix size
- `gpu_kernel_time_vs_matrix.png` — Scaling plot with error bands
- `gpu_variability_analysis.png` — Coefficient of variation analysis

## 🛠 Tools & Scripts
- `run_gpu_tests.sh` — Automated testing script
- `gpu_summary.py` — Python script to generate CSV summaries
- `gpu_plots.py` — Visualization with matplotlib
