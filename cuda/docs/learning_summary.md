# Day 3 Learning Summary — GPU Computing with CUDA

## 🧩 GPU/CUDA Concepts
- **GPU Computing:** Heterogeneous architecture with massive parallelism (thousands of cores).
- **CUDA** = Compute Unified Device Architecture → write kernels in C++ for NVIDIA GPUs.
- **Memory Hierarchy:** Host (CPU) ↔ Device (GPU) with explicit data transfers (H2D, D2H).
- **Threading Model:** Blocks, threads, warps, SIMT (Single Instruction, Multiple Threads).
- Used `cudaMalloc`, `cudaMemcpy`, `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`, `cudaKernel<<<>>>`.

---

## 💻 Parallel Algorithm Design
- Implemented **tiled matrix multiplication** optimized for GPU architectures.
- **Thread Organization:** 
  - Blocks: 2D grid (each block = 16×16 threads)
  - Each thread computes one C[i][j] element
  - Threads in a block cooperatively load tiles of A and B into shared memory
- **Shared Memory Optimization:**
  - Reduced global memory bandwidth by reusing tiles
  - Minimized cache misses via coalesced memory access patterns
- **Memory Coalescing:** Ensured threads access consecutive memory addresses for efficiency.

---

## ⚙️ Benchmarking Skills
- Measured **kernel execution time** using CUDA events (`cudaEventRecord`, `cudaEventElapsedTime`).
- Created automated test scripts for parameter sweeps across matrix sizes.
- Collected repeated runs (5 iterations per configuration) for statistical reliability.
- Implemented checksum verification to ensure correctness of GPU computation.
- Understood metrics: **Throughput (GFLOPS)**, **Kernel Time (ms)**, **Variability (CV %)**.

---

## 📊 Performance Interpretation
- **Scaling:** Kernel time grows as O(N³/P) where P = number of GPU cores.
- **Small matrices (N=500):** GPU still faster due to parallelism (1.77ms vs seconds on CPU).
- **Large matrices (N=4000):** GPU dominance clear (227ms vs 500+ seconds sequential).
- **Variability Analysis:** CV increases for larger matrices due to GPU thermal effects and memory pressure.
- **Tiled Optimization:** Reduced memory bandwidth usage by ~10-16x compared to naive approach.

---

## 🧠 System Understanding
- **Host-Device Bottleneck:** Data transfer overhead can dominate for small matrices.
- **GPU Architecture Details:**
  - Warp Size: 32 threads (NVIDIA standard)
  - Typical Block: 256-512 threads for occupancy
  - Shared Memory: ~96KB per block (limited resource)
- Monitored GPU utilization with `nvidia-smi`: achieved 95%+ utilization on large matrices.
- Learned memory addressing: row-major storage, coalescing patterns, bank conflicts.

---

## 🧪 Research Mindset
- Moved from **CPU-only** (OpenMP) → **Distributed CPU** (MPI) → **GPU acceleration** (CUDA).
- Recognized GPU excels at **embarrassingly parallel** problems like matrix multiplication.
- Compared heterogeneous computing: GPU good for compute-bound tasks, CPU for irregular workloads.
- Prepared to combine MPI + CUDA for multi-GPU clusters (hybrid parallelism).
- Gained intuition for when to use each technology based on problem structure.

---

