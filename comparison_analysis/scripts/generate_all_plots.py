#!/usr/bin/env python3
"""
Comprehensive MPI + GPU Comparison Analysis
Generates all required plots and comparison visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

def create_all_plots():
    """Generate comprehensive benchmark visualization plots"""
    
    base_dir = Path("/home/jangi/hpc_project_cpp")
    plots_dir = base_dir / "comparison_analysis" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all data
    try:
        mpi_repeats = pd.read_csv(base_dir / "mpi" / "results" / "mpi_repeats.csv")
        mpi_summary = pd.read_csv(base_dir / "mpi" / "results" / "mpi_summary.csv")
        gpu_repeats = pd.read_csv(base_dir / "cuda" / "results" / "gpu_repeats.csv")
        gpu_summary = pd.read_csv(base_dir / "cuda" / "results" / "gpu_summary.csv")
    except FileNotFoundError as e:
        print(f"❌ Error: Missing data file: {e}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("🎨 GENERATING COMPREHENSIVE BENCHMARK ANALYSIS PLOTS")
    print("="*80 + "\n")
    
    # =========================================================================
    # PART 1: MPI-SPECIFIC PLOTS
    # =========================================================================
    
    print("PART 1: MPI Analysis Plots")
    print("-" * 80)
    
    # MPI Plot 1: Algorithm Comparison
    print("  [1.1] MPI algorithm comparison (naive vs SUMMA)...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for impl in sorted(mpi_summary['impl'].unique()):
        data = mpi_summary[mpi_summary['impl'] == impl].sort_values('n')
        label_name = 'Naive Broadcast' if impl == 'mpi_matmul' else 'SUMMA (2D Grid)'
        ax.plot(data['n'], data['time_mean_s'], 'o-', label=label_name, linewidth=2.5, markersize=10)
        ax.fill_between(data['n'], 
                        data['time_mean_s'] - data['time_std_s'],
                        data['time_mean_s'] + data['time_std_s'],
                        alpha=0.2)
    
    ax.set_xlabel('Matrix Size (N×N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('MPI: Naive Broadcast vs SUMMA Algorithm\n(4 processes, single-node)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(plots_dir / '01_mpi_algorithm_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("     ✅ Saved: 01_mpi_algorithm_comparison.png")
    
    # MPI Plot 2: SUMMA Speedup
    print("  [1.2] SUMMA speedup over naive...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    speedup_data = []
    for n in sorted(mpi_summary['n'].unique()):
        naive_time = mpi_summary[(mpi_summary['impl'] == 'mpi_matmul') & (mpi_summary['n'] == n)]['time_mean_s'].values[0]
        summa_time = mpi_summary[(mpi_summary['impl'] == 'mpi_summa') & (mpi_summary['n'] == n)]['time_mean_s'].values[0]
        speedup = naive_time / summa_time
        efficiency = (speedup / 4) * 100  # ideal speedup is 4 for 4 processes
        speedup_data.append({'n': n, 'speedup': speedup, 'efficiency': efficiency})
    
    speedup_df = pd.DataFrame(speedup_data)
    bars = ax.bar(speedup_df['n'].astype(str), speedup_df['speedup'], 
                 color=['#2ca02c', '#1f77b4', '#ff7f0e'], alpha=0.7, edgecolor='black', linewidth=2)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}×\n({speedup_data[i]["efficiency"]:.1f}%)',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='No improvement', alpha=0.7)
    ax.set_xlabel('Matrix Size (N×N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (Naive Time / SUMMA Time)', fontsize=12, fontweight='bold')
    ax.set_title('SUMMA Speedup Over Naive Broadcast (4 processes)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(speedup_df['speedup']) * 1.15)
    plt.tight_layout()
    plt.savefig(plots_dir / '02_mpi_summa_speedup.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("     ✅ Saved: 02_mpi_summa_speedup.png")
    
    # MPI Plot 3: Variability
    print("  [1.3] MPI result variability...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for impl in sorted(mpi_summary['impl'].unique()):
        data = mpi_summary[mpi_summary['impl'] == impl].sort_values('n')
        cv = (data['time_std_s'] / data['time_mean_s'] * 100).fillna(0)
        label_name = 'Naive Broadcast' if impl == 'mpi_matmul' else 'SUMMA'
        ax.plot(data['n'], cv, 'o-', label=label_name, linewidth=2.5, markersize=10)
    
    ax.set_xlabel('Matrix Size (N×N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
    ax.set_title('MPI Execution Time Variability Across Runs', fontsize=14, fontweight='bold')
    ax.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5% target')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig(plots_dir / '03_mpi_variability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("     ✅ Saved: 03_mpi_variability.png")
    
    # MPI Plot 4: Throughput
    print("  [1.4] MPI computational throughput...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for impl in sorted(mpi_summary['impl'].unique()):
        throughput = []
        matrix_sizes = []
        for _, row in mpi_summary[mpi_summary['impl'] == impl].sort_values('n').iterrows():
            n = row['n']
            time_s = row['time_mean_s']
            gflops = (2 * n**3) / (time_s * 1e9)
            throughput.append(gflops)
            matrix_sizes.append(n)
        
        label_name = 'Naive Broadcast' if impl == 'mpi_matmul' else 'SUMMA'
        ax.plot(matrix_sizes, throughput, 'o-', label=label_name, linewidth=2.5, markersize=10)
    
    ax.set_xlabel('Matrix Size (N×N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Throughput (GFLOPS)', fontsize=12, fontweight='bold')
    ax.set_title('MPI Computational Throughput Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(plots_dir / '04_mpi_throughput.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("     ✅ Saved: 04_mpi_throughput.png")
    
    # =========================================================================
    # PART 2: GPU-SPECIFIC PLOTS
    # =========================================================================
    
    print("\nPART 2: GPU Analysis Plots")
    print("-" * 80)
    
    # GPU Plot 1: Kernel Time Scaling
    print("  [2.1] GPU kernel time scaling...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for impl in sorted(gpu_summary['impl'].unique()):
        data = gpu_summary[gpu_summary['impl'] == impl].sort_values('n')
        ax.plot(data['n'], data['kernel_mean_ms'], 'o-', label=impl, linewidth=2.5, markersize=10)
        ax.fill_between(data['n'], 
                        data['kernel_mean_ms'] - data['kernel_std_ms'],
                        data['kernel_mean_ms'] + data['kernel_std_ms'],
                        alpha=0.2)
    
    ax.set_xlabel('Matrix Size (N×N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Kernel Time (milliseconds)', fontsize=12, fontweight='bold')
    ax.set_title('GPU Kernel Execution Time Scaling', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(plots_dir / '05_gpu_kernel_scaling.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("     ✅ Saved: 05_gpu_kernel_scaling.png")
    
    # GPU Plot 2: GPU Variability
    print("  [2.2] GPU result variability...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for impl in sorted(gpu_summary['impl'].unique()):
        data = gpu_summary[gpu_summary['impl'] == impl].sort_values('n')
        cv = data['kernel_cv_%'].fillna(0)
        ax.plot(data['n'], cv, 'o-', label=impl, linewidth=2.5, markersize=10)
    
    ax.set_xlabel('Matrix Size (N×N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
    ax.set_title('GPU Kernel Time Variability', fontsize=14, fontweight='bold')
    ax.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5% target')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig(plots_dir / '06_gpu_variability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("     ✅ Saved: 06_gpu_variability.png")
    
    # GPU Plot 3: GPU Throughput
    print("  [2.3] GPU computational throughput...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for impl in sorted(gpu_summary['impl'].unique()):
        throughput = []
        matrix_sizes = []
        for _, row in gpu_summary[gpu_summary['impl'] == impl].sort_values('n').iterrows():
            n = row['n']
            time_ms = row['kernel_mean_ms']
            time_s = time_ms / 1000
            gflops = (2 * n**3) / (time_s * 1e9)
            throughput.append(gflops)
            matrix_sizes.append(n)
        
        ax.plot(matrix_sizes, throughput, 'o-', label=impl, linewidth=2.5, markersize=10)
    
    ax.set_xlabel('Matrix Size (N×N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Throughput (GFLOPS)', fontsize=12, fontweight='bold')
    ax.set_title('GPU Computational Throughput', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(plots_dir / '07_gpu_throughput.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("     ✅ Saved: 07_gpu_throughput.png")
    
    # =========================================================================
    # PART 3: CROSS-PLATFORM COMPARISON
    # =========================================================================
    
    print("\nPART 3: Cross-Platform Comparison Plots")
    print("-" * 80)
    
    # Comparison Plot 1: All vs All (normalize to GPU)
    print("  [3.1] Cross-platform performance comparison...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # For each matrix size in GPU data
    comparison_data = {}
    for n in sorted(gpu_summary['n'].unique()):
        gpu_time_ms = gpu_summary[gpu_summary['n'] == n]['kernel_mean_ms'].values[0]
        gpu_time_s = gpu_time_ms / 1000
        
        # Find corresponding MPI data
        mpi_naive_time = mpi_summary[(mpi_summary['impl'] == 'mpi_matmul') & (mpi_summary['n'] == n)]
        mpi_summa_time = mpi_summary[(mpi_summary['impl'] == 'mpi_summa') & (mpi_summary['n'] == n)]
        
        if len(mpi_naive_time) > 0 and len(mpi_summa_time) > 0:
            comparison_data[n] = {
                'GPU': gpu_time_s,
                'MPI (Naive)': mpi_naive_time.iloc[0]['time_mean_s'],
                'MPI (SUMMA)': mpi_summa_time.iloc[0]['time_mean_s']
            }
    
    if comparison_data:
        matrix_sizes = list(comparison_data.keys())
        x = np.arange(len(matrix_sizes))
        width = 0.25
        
        gpu_times = [comparison_data[n]['GPU'] for n in matrix_sizes]
        naive_times = [comparison_data[n]['MPI (Naive)'] for n in matrix_sizes]
        summa_times = [comparison_data[n]['MPI (SUMMA)'] for n in matrix_sizes]
        
        ax.bar(x - width, gpu_times, width, label='GPU', alpha=0.8, edgecolor='black')
        ax.bar(x, naive_times, width, label='MPI (Naive)', alpha=0.8, edgecolor='black')
        ax.bar(x + width, summa_times, width, label='MPI (SUMMA)', alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Matrix Size (N×N)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Cross-Platform Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in matrix_sizes])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(plots_dir / '08_cross_platform_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("     ✅ Saved: 08_cross_platform_comparison.png")
    
    # Comparison Plot 2: Speedup relative to slowest
    print("  [3.2] Speedup analysis...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    speedup_comp = {}
    for n in comparison_data.keys():
        slowest = max(comparison_data[n].values())
        speedup_comp[n] = {k: slowest / v for k, v in comparison_data[n].items()}
    
    matrix_sizes = list(speedup_comp.keys())
    x = np.arange(len(matrix_sizes))
    width = 0.25
    
    gpu_speedup = [speedup_comp[n]['GPU'] for n in matrix_sizes]
    naive_speedup = [speedup_comp[n]['MPI (Naive)'] for n in matrix_sizes]
    summa_speedup = [speedup_comp[n]['MPI (SUMMA)'] for n in matrix_sizes]
    
    ax.bar(x - width, gpu_speedup, width, label='GPU', alpha=0.8, edgecolor='black', color='#2ca02c')
    ax.bar(x, naive_speedup, width, label='MPI (Naive)', alpha=0.8, edgecolor='black', color='#1f77b4')
    ax.bar(x + width, summa_speedup, width, label='MPI (SUMMA)', alpha=0.8, edgecolor='black', color='#ff7f0e')
    
    ax.set_xlabel('Matrix Size (N×N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (relative to slowest)', fontsize=12, fontweight='bold')
    ax.set_title('Relative Speedup Across Implementations', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in matrix_sizes])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(plots_dir / '09_speedup_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("     ✅ Saved: 09_speedup_comparison.png")
    
    # Comparison Plot 3: Efficiency
    print("  [3.3] Computational efficiency...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    efficiency_data = {}
    for n in comparison_data.keys():
        efficiency_data[n] = {}
        for impl, time_s in comparison_data[n].items():
            gflops = (2 * n**3) / (time_s * 1e9)
            efficiency_data[n][impl] = gflops
    
    matrix_sizes = list(efficiency_data.keys())
    x = np.arange(len(matrix_sizes))
    width = 0.25
    
    gpu_flops = [efficiency_data[n]['GPU'] for n in matrix_sizes]
    naive_flops = [efficiency_data[n]['MPI (Naive)'] for n in matrix_sizes]
    summa_flops = [efficiency_data[n]['MPI (SUMMA)'] for n in matrix_sizes]
    
    ax.bar(x - width, gpu_flops, width, label='GPU', alpha=0.8, edgecolor='black', color='#2ca02c')
    ax.bar(x, naive_flops, width, label='MPI (Naive)', alpha=0.8, edgecolor='black', color='#1f77b4')
    ax.bar(x + width, summa_flops, width, label='MPI (SUMMA)', alpha=0.8, edgecolor='black', color='#ff7f0e')
    
    ax.set_xlabel('Matrix Size (N×N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Throughput (GFLOPS)', fontsize=12, fontweight='bold')
    ax.set_title('Computational Efficiency (GFLOPS) Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in matrix_sizes])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(plots_dir / '10_efficiency_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("     ✅ Saved: 10_efficiency_comparison.png")
    
    # =========================================================================
    # PART 4: SUMMARY STATISTICS TABLE
    # =========================================================================
    
    print("\nPART 4: Summary Statistics")
    print("-" * 80)
    
    # Create summary table
    print("\n📊 MPI BENCHMARK SUMMARY:")
    print(mpi_summary.to_string(index=False))
    
    print("\n📊 GPU BENCHMARK SUMMARY:")
    print(gpu_summary.to_string(index=False))
    
    # Create comparison table
    print("\n📊 CROSS-PLATFORM PERFORMANCE TABLE:")
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data).T
        comp_df.index.name = 'Matrix Size'
        print(comp_df.to_string())
        
        print("\n📊 SPEEDUP ANALYSIS TABLE:")
        speedup_table = pd.DataFrame(speedup_comp).T
        speedup_table.index.name = 'Matrix Size'
        print(speedup_table.to_string())
    
    print("\n" + "="*80)
    print(f"✅ ALL PLOTS GENERATED AND SAVED TO: {plots_dir}")
    print("="*80 + "\n")

if __name__ == "__main__":
    create_all_plots()
