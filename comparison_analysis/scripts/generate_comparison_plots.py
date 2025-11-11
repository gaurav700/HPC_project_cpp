#!/usr/bin/env python3
"""
Cross-Implementation Performance Comparison
Compares Sequential, OpenMP, MPI, and CUDA implementations
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

def create_comparison_plots():
    """Generate comprehensive comparison plots"""
    
    # Setup paths
    root = Path(__file__).parent.parent.parent
    plots_dir = Path(__file__).parent.parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Read data
    print("📊 Reading benchmark data from all implementations...")
    
    gpu_df = pd.read_csv(root / "cuda/results/gpu_summary.csv")
    mpi_df = pd.read_csv(root / "mpi/results/mpi_summary.csv")
    openmp_seq_df = pd.read_csv(root / "openmp and sequential/results/day1_sequential_openmp_summary.csv")
    
    # Extract sequential data
    seq_df = openmp_seq_df[openmp_seq_df['framework'] == 'sequential'][['n', 'mean']].copy()
    seq_df.columns = ['n', 'time_ms']
    seq_df['implementation'] = 'Sequential'
    seq_df['time_ms'] = seq_df['time_ms'] * 1000  # Convert to ms
    
    # Extract OpenMP data (1 thread baseline for comparison)
    omp_df = openmp_seq_df[openmp_seq_df['framework'] == 'OpenMP']
    omp_best = omp_df.groupby('n')['mean'].min().reset_index()
    omp_best.columns = ['n', 'time_ms']
    omp_best['implementation'] = 'OpenMP (best)'
    omp_best['time_ms'] = omp_best['time_ms'] * 1000  # Convert to ms
    
    # Extract MPI data (1 process - single core)
    mpi_single = mpi_df[mpi_df['processes'] == 1][['n', 'mean']].copy()
    mpi_single.columns = ['n', 'time_ms']
    mpi_single['implementation'] = 'MPI (1 proc)'
    mpi_single['time_ms'] = mpi_single['time_ms'] * 1000  # Convert to ms
    
    # Extract GPU data
    gpu_data = gpu_df[['n', 'kernel_mean_ms']].copy()
    gpu_data.columns = ['n', 'time_ms']
    gpu_data['implementation'] = 'GPU (CUDA)'
    
    # Combine all data
    all_data = pd.concat([seq_df, omp_best, mpi_single, gpu_data], ignore_index=True)
    
    print("✅ Data loaded successfully!")
    print(f"   Matrix sizes: {sorted(all_data['n'].unique())}")
    print(f"   Implementations: {sorted(all_data['implementation'].unique())}\n")
    
    # ====== PLOT 1: Overall Comparison ======
    print("📈 Plot 1: Overall execution time comparison (log scale)...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = {'Sequential': '#FF6B6B', 'OpenMP (best)': '#4ECDC4', 
              'MPI (1 proc)': '#95E1D3', 'GPU (CUDA)': '#FFE66D'}
    
    for impl in sorted(all_data['implementation'].unique()):
        data = all_data[all_data['implementation'] == impl].sort_values('n')
        ax.plot(data['n'], data['time_ms'], 'o-', label=impl, 
                linewidth=2.5, markersize=10, color=colors.get(impl, '#000'))
    
    ax.set_xlabel('Matrix Size (N)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Execution Time (ms)', fontsize=13, fontweight='bold')
    ax.set_title('Performance Comparison: All Implementations\n(Lower is Better)', 
                 fontsize=15, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(plots_dir / 'comparison_all_implementations.png', dpi=150)
    print(f"   ✅ Saved: comparison_all_implementations.png")
    plt.close()
    
    # ====== PLOT 2: Speedup vs Sequential ======
    print("📈 Plot 2: Speedup relative to sequential baseline...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    seq_times = seq_df.set_index('n')['time_ms']
    
    for impl in ['OpenMP (best)', 'MPI (1 proc)', 'GPU (CUDA)']:
        data = all_data[all_data['implementation'] == impl].sort_values('n')
        speedups = []
        ns = []
        for n, time_ms in zip(data['n'], data['time_ms']):
            if n in seq_times.index:
                speedup = seq_times[n] / time_ms
                speedups.append(speedup)
                ns.append(n)
        
        if speedups:
            ax.plot(ns, speedups, 'o-', label=impl, linewidth=2.5, markersize=10)
    
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Baseline (Sequential)')
    ax.set_xlabel('Matrix Size (N)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Speedup Factor', fontsize=13, fontweight='bold')
    ax.set_title('Speedup vs Sequential Baseline\n(Higher is Better)', 
                 fontsize=15, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(plots_dir / 'comparison_speedup_vs_sequential.png', dpi=150)
    print(f"   ✅ Saved: comparison_speedup_vs_sequential.png")
    plt.close()
    
    # ====== PLOT 3: Bar chart comparison at key sizes ======
    print("📈 Plot 3: Direct comparison at selected matrix sizes...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Detailed Comparison at Key Matrix Sizes\n(Lower is Better)', 
                 fontsize=16, fontweight='bold')
    
    matrix_sizes = [500, 1000, 2000, 4000]
    
    for idx, n in enumerate(matrix_sizes):
        ax = axes[idx // 2, idx % 2]
        
        data_at_n = all_data[all_data['n'] == n].sort_values('time_ms')
        
        bars = ax.bar(range(len(data_at_n)), data_at_n['time_ms'].values, 
                      color=['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFE66D'][:len(data_at_n)])
        
        ax.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
        ax.set_title(f'N = {n}×{n}', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(data_at_n)))
        ax.set_xticklabels(data_at_n['implementation'].values, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, data_at_n['time_ms'].values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}ms', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'comparison_by_matrix_size.png', dpi=150)
    print(f"   ✅ Saved: comparison_by_matrix_size.png")
    plt.close()
    
    # ====== PLOT 4: Efficiency (speedup per unit overhead) ======
    print("📈 Plot 4: Efficiency analysis...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Calculate efficiency for each implementation
    for impl in ['OpenMP (best)', 'MPI (1 proc)', 'GPU (CUDA)']:
        data = all_data[all_data['implementation'] == impl].sort_values('n')
        speedups = []
        ns = []
        for n, time_ms in zip(data['n'], data['time_ms']):
            if n in seq_times.index:
                speedup = seq_times[n] / time_ms
                speedups.append(speedup)
                ns.append(n)
        
        if speedups:
            ax.plot(ns, speedups, 'o-', label=impl, linewidth=2.5, markersize=10)
    
    ax.set_xlabel('Matrix Size (N)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Speedup (log scale)', fontsize=13, fontweight='bold')
    ax.set_title('Scaling Efficiency\n(Speedup Growth with Problem Size)', 
                 fontsize=15, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(plots_dir / 'comparison_efficiency.png', dpi=150)
    print(f"   ✅ Saved: comparison_efficiency.png")
    plt.close()
    
    # ====== PLOT 5: Linear scale comparison (smaller matrices) ======
    print("📈 Plot 5: Linear scale comparison (focus on small matrices)...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    small_data = all_data[all_data['n'] <= 2000]
    
    for impl in sorted(small_data['implementation'].unique()):
        data = small_data[small_data['implementation'] == impl].sort_values('n')
        ax.plot(data['n'], data['time_ms'], 'o-', label=impl, 
                linewidth=2.5, markersize=10, color=colors.get(impl, '#000'))
    
    ax.set_xlabel('Matrix Size (N)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Execution Time (ms)', fontsize=13, fontweight='bold')
    ax.set_title('Performance Comparison (Linear Scale, Small Matrices)', 
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'comparison_linear_scale.png', dpi=150)
    print(f"   ✅ Saved: comparison_linear_scale.png")
    plt.close()
    
    print("\n" + "="*70)
    print(f"✅ All comparison plots saved to: {plots_dir}")
    print("="*70 + "\n")

if __name__ == "__main__":
    create_comparison_plots()
