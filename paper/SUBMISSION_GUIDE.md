# IEEE Paper Submission Package

## Files Included

This package contains two versions of the IEEE paper ready for submission:

### 1. **IEEE_Paper.tex** (Recommended for IEEE Submission)
- **Format:** LaTeX with IEEE conference template (`IEEEtran` class)
- **Purpose:** Direct submission to IEEE Access, IEEE Transactions, or Computer Society conferences
- **Compilation:** 
  ```bash
  pdflatex IEEE_Paper.tex
  bibtex IEEE_Paper
  pdflatex IEEE_Paper.tex
  pdflatex IEEE_Paper.tex
  ```
- **Output:** IEEE_Paper.pdf (conference-compliant format)

### 2. **IEEE_Paper.md** (Markdown Version)
- **Format:** Markdown with IEEE structure
- **Purpose:** Human-readable version, can be converted to PDF via Pandoc
- **Conversion:**
  ```bash
  pandoc IEEE_Paper.md -o IEEE_Paper_MD.pdf \
    -V geometry:margin=1in \
    -V papersize:letter
  ```

---

## Paper Overview

**Title:** Performance Comparison of Parallel Matrix Multiplication Across CPU, MPI, and GPU Architectures

**Keywords:** Parallel computing, matrix multiplication, GPU acceleration, CUDA, MPI, OpenMP, performance analysis

**Scope:** Comprehensive benchmarking and performance analysis of four parallel paradigms (Sequential, OpenMP, MPI, CUDA) for matrix multiplication on heterogeneous hardware.

**Key Contributions:**
1. Empirical comparison of 4 parallelization strategies on identical hardware
2. Detailed performance scaling analysis with theoretical validation (Amdahl's Law)
3. Practical algorithm selection guidelines
4. Cross-implementation benchmark data (open-sourced)

---

## Paper Structure

### Section Breakdown:

| Section | Key Content |
|---------|------------|
| **I. Introduction** | Problem statement, contributions, paper organization |
| **II. Related Work** | Strassen's algorithm, GPU optimization, MPI, scaling theory |
| **III. Methodology** | Hardware specs, algorithm, test configuration, measurement protocol |
| **IV. Results** | 6 detailed performance tables + cross-implementation comparison |
| **V. Analysis** | Why GPU dominates, CPU/MPI limitations, algorithm selection matrix |
| **VI. Validation** | Correctness, reproducibility, statistical analysis |
| **VII. Limitations & Future Work** | Scope limitations, multi-GPU clusters, cuBLAS comparison |
| **VIII. Conclusion** | Key findings and guidance for practitioners |
| **IX. References** | 10 peer-reviewed sources |

---

## Performance Summary (from Paper)

| Matrix Size | Sequential | OpenMP (best) | MPI (4 proc) | GPU |
|------------|-----------|-----------|-----------|-----|
| 500×500 | 117.8 ms | 30.7 ms (3.8×) | 38.8 ms (3.0×) | 1.77 ms (**67×**) |
| 1000×1000 | 1,318 ms | 593.9 ms (2.2×) | 1,043 ms (1.3×) | 5.83 ms (**226×**) |
| 2000×2000 | 32,415.7 ms | 12,205 ms (2.7×) | 13,732 ms (2.4×) | 37.96 ms (**854×**) |
| 3000×3000 | 176,300 ms | 62,572 ms (2.8×) | 54,931 ms (3.3×) | 106.50 ms (**1,655×**) |
| 4000×4000 | 575,135 ms | 163,475 ms (3.5×) | 155,649 ms (3.6×) | 227.35 ms (**2,530×**) |

**Key Finding:** GPU acceleration dominates with 2,530× speedup on 4000×4000 matrices.

---

## Preparation for IEEE Submission

### Step 1: Generate PDF from LaTeX

```bash
cd paper/
pdflatex IEEE_Paper.tex
bibtex IEEE_Paper
pdflatex IEEE_Paper.tex
pdflatex IEEE_Paper.tex
```

This produces: **IEEE_Paper.pdf**

### Step 2: Check Submission Requirements

Different IEEE venues have slightly different requirements:

**IEEE Access** (Open Access, Rapid Publication):
- Single-column format
- More flexible page limits (10-12 pages typical)
- Faster review cycle (2-3 months)
- No publication fees (2025)

**IEEE Transactions on Parallel and Distributed Systems:**
- Rigorous peer review
- Strong publication record
- Longer publication timeline

**IEEE Computer Society:**
- Multiple venues (IPDPS, SC, HPCA)
- Conference format (8-10 pages)
- Competitive acceptance rates

### Step 3: Format Adjustments (if needed)

If your target venue requires different formatting:

**For longer papers (12+ pages):**
- Paper already supports extended discussion
- Expand Section V (Analysis) with more detailed breakdowns

**For shorter format (8 pages):**
- Condense Section II (Related Work) to 1 page
- Combine Tables I-V into summary statistics
- Move detailed appendices online

### Step 4: Figure References

Paper references 5 comparison plots. Include these in submission:

```
comparison_analysis/plots/
├── comparison_all_implementations.png
├── comparison_speedup_vs_sequential.png
├── comparison_by_matrix_size.png
├── comparison_efficiency.png
└── comparison_linear_scale.png
```

**Figure Instructions for LaTeX:**
```latex
\begin{figure}[h]
\includegraphics[width=0.9\linewidth]{comparison_all_implementations.png}
\caption{Performance comparison of all implementations on log-log scale.}
\label{fig:comparison}
\end{figure}
```

---

## Submission Checklist

- [ ] LaTeX file compiles without errors
- [ ] PDF generated with proper formatting
- [ ] All references cite real papers (10 references included)
- [ ] Tables are IEEE-compliant
- [ ] Figures are high-quality PNG (150 DPI+)
- [ ] Author information complete
- [ ] Keywords present (8 keywords)
- [ ] Abstract is 150-250 words (✓ ~200 words)
- [ ] No plagiarism (all original experimental work)
- [ ] Reproducibility: Scripts and data available

---

## IEEE LaTeX Template Notes

The `.tex` file uses:
- **Document class:** `IEEEtran` (v1.8b)
- **Style:** `compsocconf` for Computer Society conferences
- **Packages:** cite, graphicx, amsmath, amssymb, booktabs, hyperref
- **Bibliography style:** IEEE default (author/year)

### Installing IEEEtran (if needed):

**Ubuntu/Linux:**
```bash
sudo apt install texlive-fonts-recommended texlive-latex-extra
```

**macOS (with Homebrew):**
```bash
brew install texlive
```

**Manual:** Download from [CTAN](https://ctan.org/pkg/ieeetran)

---

## Submission Venues & Guidelines

### 1. IEEE Access
- **Website:** https://ieeeaccess.ieee.org/
- **Scope:** Open-access multidisciplinary journal
- **Timeline:** 2-3 months
- **Format:** Single column, max 10-12 pages
- **Best for:** Quick publication

### 2. IEEE Transactions on Parallel and Distributed Systems
- **Website:** https://www.computer.org/csdl/journal/tp
- **Scope:** Peer-reviewed transactions in parallel/distributed computing
- **Timeline:** 4-6 months
- **Format:** Two columns, 8-10 pages typical
- **Best for:** High-impact venue

### 3. IEEE Computer Society Conferences
- **IPDPS:** International Parallel & Distributed Processing Symposium
- **SC:** International Conference for High Performance Computing
- **HPCA:** High-Performance Computer Architecture
- **Format:** 8-10 pages
- **Timeline:** 4-6 months review + conference

---

## Citation Information

If you cite this paper in future work, use:

```bibtex
@article{kumar2025performance,
    author = {Kumar, Gaurav},
    title = {Performance Comparison of Parallel Matrix Multiplication Across 
             CPU, MPI, and GPU Architectures},
    journal = {IEEE Access},
    year = {2025},
    volume = {TBD},
    pages = {TBD},
    doi = {TBD}
}
```

---

## Data & Reproducibility

All experimental data, scripts, and results are available at:
- **GitHub Repository:** https://github.com/gaurav700/HPC_project_cpp
- **Raw Data:** CSV files in each framework's `results/` directory
- **Benchmarking Scripts:** Bash scripts in each framework's `scripts/` directory
- **Analysis Code:** Python scripts in `comparison_analysis/scripts/`

To reproduce results:
```bash
bash "openmp and sequential/scripts/run_basic_tests.sh"
bash mpi/scripts/run_mpi_tests.sh
bash cuda/scripts/run_gpu_tests.sh
python3 comparison_analysis/scripts/generate_comparison_plots.py
```

---

## FAQ

**Q: Can I submit this to multiple venues?**
A: No, most IEEE venues require exclusive submission. Choose the best fit and submit there. If rejected, you can submit elsewhere.

**Q: How do I handle the author information?**
A: Update the `\author{}` section with your actual affiliation. For blind review, use placeholder information.

**Q: Should I include all the appendices?**
A: Appendices are optional for IEEE submissions. Core paper (8-10 pages) is sufficient. You can reference data on GitHub.

**Q: What if my venue uses IEEE 802.11 or similar templates?**
A: The IEEEtran template is standard across all IEEE publications. Your venue will accept this format.

**Q: How do I handle the "Submitted to" notice?**
A: LaTeX version is ready to submit. For blind review, add:
```latex
\thispagestyle{empty}
\centerline{\large \bf Submitted to IEEE Access}
```

---

## Additional Resources

- **IEEE Author Center:** https://authors.ieee.org/
- **LaTeX Help:** https://www.overleaf.com/ (free online LaTeX editor)
- **Performance Analysis References:** 
  - Adve et al. "Parallel Computing" (2006)
  - Blaise Barney's OpenMP Tutorial
  - NVIDIA CUDA Best Practices Guide

---

## Contact & Support

For questions about the paper:
- Email: gk@njit.edu
- GitHub: https://github.com/gaurav700/HPC_project_cpp

---

**Paper Version:** 1.0  
**Last Updated:** November 11, 2025  
**Status:** Ready for Submission
