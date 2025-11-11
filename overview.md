# HPC Project Overview — Comparative Study of Parallel Paradigms

## 🎯 Project Aim
Design, implement, and analyze high-performance matrix-multiplication algorithms across **three major parallel architectures** —  
**Shared-Memory (OpenMP)**, **Distributed-Memory (MPI)**, and **GPU (CUDA)** — to understand how computation, communication, and memory design impact scalability and efficiency.

---

## 🧩 Motivation
Modern scientific and data-intensive applications rely on scalable computing.  
Each HPC model — OpenMP, MPI, and CUDA — tackles parallelism differently:

| Model | Memory Type | Communication | Typical Scale |
|--------|--------------|----------------|----------------|
| **Sequential** | Single core | None | Baseline |
| **OpenMP** | Shared memory | Implicit | Multi-core CPU |
| **MPI** | Distributed memory | Explicit (message passing) | Multi-node clusters |
| **CUDA / GPU** | Device global/shared memory | Host–device | Thousands of GPU threads |

This project builds layer-by-layer understanding of these paradigms, measuring **speedup, efficiency, and scalability** at each stage.

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