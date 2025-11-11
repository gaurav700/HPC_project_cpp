# HPC Project Overview — Complete Performance Analysis

## 📋 Executive Summary

This project implements matrix multiplication using **four different parallel paradigms** on a single workstation:
1. **Sequential** (baseline, single-threaded CPU)
2. **OpenMP** (shared-memory parallelism, 1-6 threads)
3. **MPI** (distributed-memory parallelism, 1-6 processes)
4. **CUDA** (GPU acceleration, NVIDIA RTX 3060)

**Key Finding:** GPU acceleration provides **2500x speedup** on large matrices, but overhead makes small matrices faster on CPU.

---

## 🖥️ Hardware & Software Stack

### Hardware Configuration
| Component | Specification |
|-----------|---------------|
| **CPU** | Intel Core i5-11400H (6 cores / 12 threads @ 2.7 GHz) |
| **GPU** | NVIDIA GeForce RTX 3060 (3584 CUDA cores, 12GB GDDR6) |
| **RAM** | 16GB DDR4 |
| **System** | Linux (Ubuntu 22.04 on WSL2) |

### Software Stack
| Tool | Version | Purpose |
|------|---------|---------|
| **g++** | 13.3.0 | C++ compiler for sequential/OpenMP |
| **Open MPI** | 4.1.6 | Distributed memory parallelism |
| **CUDA Toolkit** | 12.x | GPU programming |
| **Python** | 3.10+ | Data analysis & visualization |
| **Pandas** | Latest | CSV data handling |
| **Matplotlib** | Latest | Plot generation |

---

## 🧮 Algorithm: Matrix Multiplication

### Standard O(N³) Implementation
```cpp
for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
        for (int k = 0; k < N; ++k)
            C[i][j] += A[i][k] * B[k][j];
```

**Computational Complexity:** O(N³) FLOPs  
**Memory Complexity:** O(N²) for three N×N matrices

### Matrix Sizes Tested
| Size | Dimension | FLOPs | Matrix Memory |
|------|-----------|-------|---------------|
| **500** | 500×500 | 250 million | 6 MB |
| **1000** | 1000×1000 | 2 billion | 24 MB |
| **2000** | 2000×2000 | 16 billion | 96 MB |
| **3000** | 3000×3000 | 54 billion | 216 MB |
| **4000** | 4000×4000 | 128 billion | 384 MB |

---

## 📊 Complete Benchmark Results

### 1. Sequential Baseline (Single-Threaded CPU)

**Execution Times (in milliseconds)**

| Matrix Size | Time (ms) | Time (s) | Speedup (1x) |
|------------|-----------|----------|-------------|
| 500×500 | 117.8 | 0.118 | 1.0x |
| 1000×1000 | 1,318.0 | 1.318 | 1.0x |
| 2000×2000 | 32,415.7 | 32.416 | 1.0x |
| 3000×3000 | 176,300.0 | 176.300 | 1.0x |
| 4000×4000 | 575,135.0 | 575.135 | 1.0x |

**Observations:**
- Single-threaded CPU is the baseline
- Time grows approximately O(N³)
- Largest matrix (4000×4000) takes **9.6 minutes**

---

### 2. OpenMP Parallelization (Shared-Memory, Multi-Core)

**Best Performance (Optimal Thread Count) in milliseconds**

| Matrix Size | 1 Thread | Best Time | Best Threads | Speedup |
|------------|----------|-----------|--------------|---------|
| 500×500 | 31.1 | 30.7 | 4 | 3.8x |
| 1000×1000 | 471.9 | 593.9 | 4 | 2.2x |
| 2000×2000 | 11,163.7 | 12,205 | 4 | 2.7x |
| 3000×3000 | 59,992.2 | 62,572 | 6 | 2.8x |
| 4000×4000 | 160,612.8 | 163,475 | 4 | 3.5x |

**Key Insights:**
- **Moderate speedup** (2.2x - 3.8x) despite 6 cores available
- **Thread count paradox:** More threads don't always help
- **Reason:** Cache contention and thread synchronization overhead dominate
- **Non-scaling regime:** Cache effects limit effectiveness

---

### 3. MPI Parallelization (Distributed-Memory, Multi-Process)

**Execution Times with Varying Process Counts (in milliseconds)**

#### N = 2000×2000 (Summary)
| Processes | Time (ms) | Speedup | Efficiency |
|-----------|-----------|---------|-----------|
| 1 | 28,872.1 | 1.0x | 100% |
| 2 | 19,140.5 | 1.5x | 75% |
| 4 | 13,732.0 | 2.1x | 53% |
| 6 | 11,095.6 | 2.6x | 43% |

#### N = 4000×4000 (Summary)
| Processes | Time (ms) | Speedup | Efficiency |
|-----------|-----------|---------|-----------|
| 1 | 557,380.8 | 1.0x | 100% |
| 2 | 394,978.8 | 1.4x | 70% |
| 4 | 185,266.0 | 3.0x | 75% |
| 6 | 155,649.6 | 3.6x | 60% |

**Key Insights:**
- **Good scaling on large matrices:** 3-3.6x speedup on 4000×4000 with 6 processes
- **Poor scaling on small matrices:** 1.5x speedup at 500×500
- **Communication overhead dominates:** MPI_Bcast and MPI_Gatherv become bottleneck for small N
- **Amdahl's Law in action:** Serial broadcast/gather phases limit parallelism

---

### 4. GPU Acceleration (CUDA Tiled Kernel)

**Kernel Execution Times (in milliseconds)**

| Matrix Size | Mean Time | Std Dev | CV (%) | Speedup vs Seq |
|------------|-----------|---------|--------|-----------------|
| 500×500 | 1.77 | 0.144 | 8.13 | **67x** |
| 1000×1000 | 5.83 | 0.202 | 3.46 | **226x** |
| 2000×2000 | 37.96 | 0.260 | 0.68 | **854x** |
| 3000×3000 | 106.50 | 14.00 | 13.14 | **1,655x** |
| 4000×4000 | 227.35 | 17.95 | 7.89 | **2,530x** |

**Performance Metrics at N=4000×4000:**
- **Total FLOPs:** 128 billion
- **Execution Time:** 227.35 ms
- **Throughput:** 563 GFLOPS
- **Peak GPU FP64:** 1,456 GFLOPS
- **Efficiency:** ~39% (reasonable for tiled kernel)

---

## 📈 Cross-Implementation Comparison

### Speedup Comparison at Key Matrix Sizes

#### N = 500×500
| Implementation | Time (ms) | Speedup |
|----------------|-----------|---------|
| Sequential | 117.8 | 1.0x (baseline) |
| OpenMP (best) | 30.7 | **3.8x** |
| MPI (4 procs) | 38.8 | **3.0x** |
| GPU (CUDA) | 1.77 | **67x** |

#### N = 2000×2000
| Implementation | Time (ms) | Speedup |
|----------------|-----------|---------|
| Sequential | 32,415.7 | 1.0x (baseline) |
| OpenMP (best) | 12,205 | **2.7x** |
| MPI (4 procs) | 13,732 | **2.4x** |
| GPU (CUDA) | 37.96 | **854x** |

#### N = 4000×4000
| Implementation | Time (ms) | Speedup |
|----------------|-----------|---------|
| Sequential | 575,135 | 1.0x (baseline) |
| OpenMP (best) | 163,475 | **3.5x** |
| MPI (4 procs) | 185,266 | **3.1x** |
| GPU (CUDA) | 227.35 | **2,530x** |

### Performance Efficiency Analysis

| Aspect | Sequential | OpenMP | MPI | GPU |
|--------|-----------|--------|-----|-----|
| **Scalability with N** | O(N³) | O(N³) | O(N³/P) | O(N³/thousands) |
| **Best for** | Baseline/small N | Shared memory systems | Multi-node clusters | Dense linear algebra |
| **Overhead** | None | Thread creation/sync | Process creation/messages | Memory transfer H2D/D2H |
| **Speedup Range** | 1x | 2-4x (CPU-limited) | 1.5-3.6x | 67-2530x |
| **Parallelism** | Single core | Limited by cache | Limited by network | Massive (3584 cores) |

---

## 🔍 Detailed Analysis

### Why GPU Dominates

1. **Massive Parallelism:**
   - GPU: 3584 cores vs CPU: 6 cores
   - 600x more parallel workers

2. **Memory Bandwidth:**
   - GPU Global: 300+ GB/s
   - CPU L3: ~50 GB/s
   - 6x higher bandwidth

3. **Tiling Optimization:**
   - Reduced global memory accesses by ~16x
   - Leveraged 96KB shared memory per block
   - Coalesced memory access patterns

### Why OpenMP/MPI Show Modest Speedup

1. **Cache Contention:**
   - 6 cores compete for limited L3 cache
   - Context switching overhead
   - False sharing on shared data

2. **Synchronization Overhead:**
   - OpenMP: Barrier and lock contention
   - MPI: Message passing latency dominates on small matrices

3. **Amdahl's Law:**
   - Sequential portions (I/O, setup) limit speedup
   - Cannot exceed 1/(serial fraction)

### When to Use Each Approach

| Implementation | When to Use |
|----------------|------------|
| **Sequential** | Baseline measurement, validation, single-machine workloads |
| **OpenMP** | Shared-memory systems (same machine), I/O-bound workloads |
| **MPI** | Multi-node clusters, extreme-scale computing |
| **GPU (CUDA)** | Dense linear algebra, embarrassingly parallel problems |

---

## 📊 Visualization Summary

### Generated Plots

The `comparison_analysis/plots/` directory contains:

1. **comparison_all_implementations.png** — All implementations on log-log scale
2. **comparison_speedup_vs_sequential.png** — Relative speedup vs baseline
3. **comparison_by_matrix_size.png** — Bar charts at N=500, 1000, 2000, 4000
4. **comparison_efficiency.png** — Scaling efficiency analysis
5. **comparison_linear_scale.png** — Linear scale view of small matrices

### Individual Implementation Plots

- **Sequential/OpenMP:** `openmp and sequential/plots/`
- **MPI:** `mpi/plots/`
- **GPU (CUDA):** `cuda/plots/`
  - `gpu_kernel_time_vs_matrix.png` — Scaling with error bands
  - `gpu_variability_analysis.png` — Coefficient of variation

---

## 🧠 Learning Outcomes

### Days 1-4 Journey

**Day 1: Sequential & OpenMP**
- Established CPU baseline
- Learned shared-memory parallelism limits
- Cache effects dominate at 6 cores

**Day 2: MPI Parallelism**
- Implemented distributed-memory approach
- Studied communication overhead
- Recognized when communication > computation

**Day 3: GPU Acceleration**
- Moved to heterogeneous computing
- Implemented tiled kernel for memory efficiency
- Achieved massive speedup on compute-bound workload

**Day 4: Cross-Implementation Analysis**
- Compared all four approaches statistically
- Understood when to use each paradigm
- Gained intuition for HPC design decisions

### Key Insights
1. **Parallelism ≠ Speedup:** Overhead matters more than core count
2. **Algorithm > Hardware:** Tiling on GPU matters more than having many cores
3. **Problem-specific:** GPU best for dense compute; MPI for distributed systems
4. **Scalability is hard:** Speedup plateaus even with perfect implementations

---

## 📁 Project Structure

```
hpc_project_cpp/
├── overview.md                          (This file - COMPLETE ANALYSIS)
├── openmp and sequential/
│   ├── docs/
│   │   ├── learning_summary.md
│   │   └── Summary.md
│   ├── results/
│   │   ├── day1_sequential_openmp_summary.csv
│   │   └── sequential_openmp_repeats.csv
│   ├── plots/
│   │   └── (visualization files)
│   └── src/
│       ├── sequential_matmul.cpp
│       ├── openmp_matmul.cpp
│       └── summary scripts
├── mpi/
│   ├── docs/
│   │   ├── learning_summary.md
│   │   └── Summary.md
│   ├── results/
│   │   ├── mpi_summary.csv
│   │   └── mpi_repeats.csv
│   ├── plots/
│   │   └── (visualization files)
│   └── src/
│       ├── mpi_matmul.cpp
│       └── summary scripts
├── cuda/
│   ├── docs/
│   │   ├── learning_summary.md
│   │   └── Summary.md
│   ├── results/
│   │   ├── gpu_summary.csv
│   │   └── gpu_repeats.csv
│   ├── plots/
│   │   └── (visualization files)
│   └── src/
│       ├── gpu_tiled_matmul.cu
│       └── summary scripts
└── comparison_analysis/                 (NEW)
    ├── scripts/
    │   └── generate_comparison_plots.py
    └── plots/
        ├── comparison_all_implementations.png
        ├── comparison_speedup_vs_sequential.png
        ├── comparison_by_matrix_size.png
        ├── comparison_efficiency.png
        └── comparison_linear_scale.png
```

---

## 🚀 How to Regenerate Results

### 1. Run All Benchmarks
```bash
# Sequential & OpenMP
bash "openmp and sequential/scripts/run_basic_tests.sh"

# MPI
bash mpi/scripts/run_mpi_tests.sh

# GPU
bash cuda/scripts/run_gpu_tests.sh
```

### 2. Generate Individual Summaries
```bash
python3 "openmp and sequential/src/sequential_openmp_summary.py"
python3 mpi/src/mpi_summary.py
python3 cuda/src/gpu_summary.py
```

### 3. Generate Comparison Plots
```bash
python3 comparison_analysis/scripts/generate_comparison_plots.py
```

---

## ✅ Conclusion

This project demonstrates the complete spectrum of parallelism:
- **Sequential baseline** for validation
- **Shared-memory (OpenMP)** for single-machine multi-core
- **Distributed-memory (MPI)** for clusters
- **GPU (CUDA)** for compute-intensive workloads

**The GPU achieves 2530x speedup on large matrices**, but the key lesson is that **speedup depends on problem structure and overhead management**, not just raw parallelism.

For dense linear algebra on modern systems, **GPU acceleration is the clear winner**. For irregular workloads, distributed-memory systems (MPI) provide better flexibility. CPU parallelism (OpenMP) serves as a practical middle ground for shared-memory systems.

**Future directions:** Multi-GPU clusters (MPI + CUDA), optimized libraries (cuBLAS), and mixed-precision computing for extreme-scale problems.

---

*Generated: November 11, 2025*  
*HPC Project - Complete Performance Analysis & Comparison*

---

## 🧱 Architecture of the Study

### **Phase 1 – Day 1: Sequential & OpenMP (Shared-Memory Parallelism)**
**Goal:** Establish CPU-based baseline and measure intra-node scaling.

- Implemented `sequential_matmul.cpp` and `openmp_matmul.cpp`.
- Benchmarked with 1, 2, 4, 6, 12 threads for N = 500–2000.
- Used `#pragma omp parallel for` for loop parallelization.
- Recorded results → `day1_basic.csv`, summarized → `day1_summary_stats.csv`.

**Key Findings**
| Metric | Observation |
|---------|--------------|
| Speedup | Near-linear up to 6 threads (≈ physical cores) |
| Efficiency | ≈ 90 % up to 6 threads, falls to ~70 % at 12 threads |
| Overhead | Thread creation + scheduling costs visible for small N |
| Verification | Checksums consistent with sequential baseline |

**Conclusion:**  
OpenMP achieves strong scaling within one CPU socket, but saturates once shared cache and memory bandwidth become limiting.

---

### **Phase 2 – Day 2: MPI (Distributed-Memory Parallelism)**
**Goal:** Break shared-memory limits and analyze process-level scaling.

- Implemented `mpi_matmul.cpp` using **MPI 4.1.6**.  
- Partitioned matrix A across processes, broadcast B (`MPI_Bcast`), gathered C (`MPI_Gatherv`).  
- Benchmarked N = 500–4000, processes = 1, 2, 4, 6 × 5 repeats.  
- Added process binding (`--bind-to core`) for stable timings.  
- Automated experiments via `run_day2_mpi_tests.sh`.

**Key Findings**
| Matrix N | Processes | Mean Time (s) | Speedup | Efficiency (%) |
|-----------:|------------:|--------------:|---------:|----------------:|
| 500 | 1→6 | 0.110→0.038 | 2.9× | 80–90 |
| 1000 | 1→6 | 1.44→1.37 | 1.1× | 20–40 |
| 2000 | 1→6 | 27.3→11.3 | 2.4× | 40–55 |

**Interpretation**
- Small N: Communication > Computation → weak scaling.  
- Large N: Computation dominates → ≈ 2.5× speedup at 6 processes.  
- Amdahl’s fit f ≈ 0.9 (90 % parallel fraction).  
- Efficiency drops after 4 processes due to memory and sync overhead.

**Conclusion:**  
MPI scales beyond OpenMP for larger workloads but introduces measurable message-passing cost.  
It prepares the ground for hybrid (MPI + OpenMP) and GPU-based distributed experiments.

---

### **Phase 3 – Day 3 (Upcoming): GPU Acceleration (CUDA/cuBLAS)**
**Goal:** Explore device-level parallelism using NVIDIA RTX 3050 GPU (4 GB VRAM, CUDA 12.7).

**Planned Work**
1. Implement `gpu_matmul.cu` using **CUDA kernels** and **cuBLAS DGEMM**.  
2. Benchmark vs CPU baselines (Sequential / OpenMP / MPI).  
3. Collect metrics: GPU-utilization, kernel latency, PCIe transfer time.  
4. Analyze **compute vs memory bound regions** and **energy efficiency**.

**Expected Outcome**
- GPU expected to outperform CPU > 10× for N ≥ 2000.  
- Performance limited by VRAM capacity and host-device transfer overhead.  
- Completes the 3-tier scalability curve:  
  `CPU (Threads) → Processes → GPU Kernels`.

---

## ⚙️ Benchmarking and Analysis Methodology
| Step | Technique | Purpose |
|------|------------|----------|
| **Multiple Repeats** | 5 runs per config | Reduce noise & average out variance |
| **Core Binding** | `--bind-to core` | Ensure process/thread affinity |
| **Timing** | `std::chrono` / `MPI_Wtime()` / CUDA Events | Accurate wall-clock measurement |
| **Validation** | Checksum comparison | Functional correctness |
| **Statistical Summary** | Mean ± Std Dev | Measurement stability |
| **Visualization** | Matplotlib | Trends (Speedup, Efficiency, Scaling) |

---

## 🧮 Key Takeaways So Far
| Domain | Lesson |
|---------|---------|
| **Performance Modeling** | Learned Amdahl’s Law and efficiency drop with communication. |
| **Resource Monitoring** | Used `htop`, `nvidia-smi`, `free -h` for system-level profiling. |
| **Experimental Rigor** | Built repeatable scripts + CSV logging framework. |
| **Scaling Behavior** | CPU (OpenMP) good up to cores; MPI good for larger data; GPU next for massive parallelism. |

---

## 🧾 Deliverables So Far
| File | Description |
|------|--------------|
| `openmp/src/openmp_matmul.cpp` | Shared-memory implementation |
| `openmp/results/day1_summary_stats.csv` | Baseline results |
| `mpi/src/mpi_matmul.cpp` | Distributed-memory implementation |
| `mpi/scripts/run_day2_mpi_tests.sh` | MPI benchmark script |
| `mpi/results/day2_mpi_summary.csv` | Aggregated MPI data |
| `docs/day1_summary.md` | OpenMP report |
| `docs/day2_summary.md` | MPI report |
| `docs/overview.md` | This master overview document |

---

## 🚀 Research Roadmap
| Stage | Focus | Deliverable |
|--------|--------|-------------|
| ✅ Day 1 | Sequential & OpenMP benchmarking | `day1_summary.md` |
| ✅ Day 2 | MPI scaling and distributed analysis | `day2_summary.md` |
| 🔜 Day 3 | GPU (CUDA/cuBLAS) acceleration | `gpu_summary.md` |
| 🔜 Day 4 (optional) | Hybrid (MPI + OpenMP + CUDA) | `hybrid_summary.md` |
| 📄 Final | Paper + arXiv submission | `paper_final.pdf` |

---

## 🧠 Grand Understanding
> **Sequential → OpenMP → MPI → CUDA**  
> mirrors the real-world evolution of parallel computing — from single core to clusters to accelerators.  
> Each layer teaches how computation, communication, and hardware architecture shape performance scaling.

---

*Prepared by Gaurav — MS CS, NJIT (2025)*  
*High-Performance Computing Research Project — Parallel Systems & Scalability Analysis*