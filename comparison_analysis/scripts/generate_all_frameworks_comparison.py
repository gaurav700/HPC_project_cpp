#!/usr/bin/env python3
"""
Comprehensive All-Framework Comparison Plots
Compares Sequential, OpenMP, MPI Naive, MPI SUMMA, and GPU CUDA
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

def load_all_data():
    """Load data from all framework implementations"""
    root = Path(__file__).parent.parent.parent
    
    print("📊 Loading data from all frameworks...")
    
    # Load GPU data
    gpu_df = pd.read_csv(root / "cuda/results/gpu_summary.csv")
    
    # Load MPI data
    mpi_df = pd.read_csv(root / "mpi/results/mpi_summary.csv")
    
    # Load OpenMP and Sequential data
    openmp_seq_df = pd.read_csv(root / "openmp and sequential/results/day1_sequential_openmp_summary.csv")
    
    return gpu_df, mpi_df, openmp_seq_df

def prepare_comparison_data(gpu_df, mpi_df, openmp_seq_df):
    """Prepare unified comparison dataset"""
    
    # Sequential data
    seq_data = openmp_seq_df[openmp_seq_df['framework'] == 'sequential'][['n', 'mean']].copy()
    seq_data['framework'] = 'Sequential'
    seq_data['time_s'] = seq_data['mean']
    
    # OpenMP best performance (max threads)
    omp_data = openmp_seq_df[openmp_seq_df['framework'] == 'OpenMP']
    omp_best = omp_data.loc[omp_data.groupby('n')['mean'].idxmin()][['n', 'mean']].copy()
    omp_best['framework'] = 'OpenMP'
    omp_best['time_s'] = omp_best['mean']
    
    # MPI data - separate naive (mpi_matmul) and SUMMA (mpi_summa)
    # MPI Naive (mpi_matmul - best performance per size)
    mpi_naive = mpi_df[mpi_df['impl'] == 'mpi_matmul']
    mpi_naive_best = mpi_naive.loc[mpi_naive.groupby('n')['time_mean_s'].idxmin()][['n', 'time_mean_s']].copy()
    mpi_naive_best['framework'] = 'MPI Naive'
    mpi_naive_best['time_s'] = mpi_naive_best['time_mean_s']
    
    # MPI SUMMA (mpi_summa - best performance per size)
    mpi_summa = mpi_df[mpi_df['impl'] == 'mpi_summa']
    if not mpi_summa.empty:
        mpi_summa_best = mpi_summa.loc[mpi_summa.groupby('n')['time_mean_s'].idxmin()][['n', 'time_mean_s']].copy()
        mpi_summa_best['framework'] = 'MPI SUMMA'
        mpi_summa_best['time_s'] = mpi_summa_best['time_mean_s']
    else:
        # Create empty dataframe if no SUMMA data
        mpi_summa_best = pd.DataFrame(columns=['n', 'framework', 'time_s'])
    
    # GPU data - need to check if we have total_mean_ms column
    if 'total_mean_ms' in gpu_df.columns:
        gpu_data = gpu_df[['n', 'total_mean_ms']].copy()
        gpu_data['time_s'] = gpu_data['total_mean_ms'] / 1000.0
    else:
        # Only kernel time available
        gpu_data = gpu_df[['n', 'kernel_mean_ms']].copy()
        gpu_data['time_s'] = gpu_data['kernel_mean_ms'] / 1000.0
    
    gpu_data['framework'] = 'GPU (CUDA)'
    gpu_data['mean'] = gpu_data['time_s']
    
    # Combine all
    all_data = pd.concat([
        seq_data[['n', 'framework', 'time_s']],
        omp_best[['n', 'framework', 'time_s']],
        mpi_naive_best[['n', 'framework', 'time_s']],
        mpi_summa_best[['n', 'framework', 'time_s']] if not mpi_summa_best.empty else pd.DataFrame(columns=['n', 'framework', 'time_s']),
        gpu_data[['n', 'framework', 'time_s']]
    ], ignore_index=True)
    
    return all_data

def plot_execution_time_comparison(data, output_dir):
    """Plot execution time for all frameworks"""
    plt.figure(figsize=(12, 7))
    
    frameworks = ['Sequential', 'OpenMP', 'MPI Naive', 'MPI SUMMA', 'GPU (CUDA)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    for framework, color, marker in zip(frameworks, colors, markers):
        fw_data = data[data['framework'] == framework].sort_values('n')
        plt.plot(fw_data['n'], fw_data['time_s'], 
                marker=marker, linewidth=2.5, markersize=8,
                label=framework, color=color)
    
    plt.xlabel('Matrix Size (N)', fontsize=12, fontweight='bold')
    plt.ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    plt.title('Execution Time Comparison: All Frameworks', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    
    output_path = output_dir / "all_frameworks_execution_time.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_speedup_comparison(data, output_dir):
    """Plot speedup vs sequential for all parallel frameworks"""
    plt.figure(figsize=(12, 7))
    
    # Get sequential baseline
    seq_baseline = data[data['framework'] == 'Sequential'].set_index('n')['time_s']
    
    frameworks = ['OpenMP', 'MPI Naive', 'MPI SUMMA', 'GPU (CUDA)']
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['s', '^', 'D', 'v']
    
    for framework, color, marker in zip(frameworks, colors, markers):
        fw_data = data[data['framework'] == framework].sort_values('n')
        speedup = []
        matrix_sizes = []
        
        for _, row in fw_data.iterrows():
            n = row['n']
            if n in seq_baseline.index:
                speedup.append(seq_baseline[n] / row['time_s'])
                matrix_sizes.append(n)
        
        plt.plot(matrix_sizes, speedup, 
                marker=marker, linewidth=2.5, markersize=8,
                label=framework, color=color)
    
    plt.xlabel('Matrix Size (N)', fontsize=12, fontweight='bold')
    plt.ylabel('Speedup vs Sequential', fontsize=12, fontweight='bold')
    plt.title('Speedup Comparison: All Parallel Frameworks', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Sequential Baseline')
    plt.tight_layout()
    
    output_path = output_dir / "all_frameworks_speedup.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_relative_performance_bars(data, output_dir):
    """Bar chart showing relative performance at each matrix size"""
    matrix_sizes = sorted(data['n'].unique())
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    frameworks = ['Sequential', 'OpenMP', 'MPI Naive', 'MPI SUMMA', 'GPU (CUDA)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, n in enumerate(matrix_sizes):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        n_data = data[data['n'] == n].sort_values('time_s', ascending=False)
        
        times = []
        labels = []
        bar_colors = []
        
        for framework in frameworks:
            fw_row = n_data[n_data['framework'] == framework]
            if not fw_row.empty:
                times.append(fw_row['time_s'].values[0])
                labels.append(framework)
                bar_colors.append(colors[frameworks.index(framework)])
        
        bars = ax.barh(labels, times, color=bar_colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Execution Time (s)', fontsize=10, fontweight='bold')
        ax.set_title(f'N = {n}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xscale('log')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            if width > 0:
                label = f'{width:.3f}s' if width >= 0.001 else f'{width*1000:.2f}ms'
                ax.text(width * 1.1, bar.get_y() + bar.get_height()/2, 
                       label, ha='left', va='center', fontsize=8)
    
    # Remove empty subplot if odd number
    if len(matrix_sizes) < len(axes):
        for idx in range(len(matrix_sizes), len(axes)):
            fig.delaxes(axes[idx])
    
    plt.suptitle('Execution Time Comparison by Matrix Size', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / "all_frameworks_bar_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_efficiency_comparison(data, output_dir):
    """Plot parallel efficiency for all frameworks"""
    plt.figure(figsize=(12, 7))
    
    # Assume max cores/processes used
    # OpenMP: 12 threads, MPI: 6 processes, GPU: 2048 cores
    efficiency_map = {
        'OpenMP': 12,
        'MPI Naive': 6,
        'MPI SUMMA': 6,
        'GPU (CUDA)': 2048
    }
    
    seq_baseline = data[data['framework'] == 'Sequential'].set_index('n')['time_s']
    
    frameworks = ['OpenMP', 'MPI Naive', 'MPI SUMMA', 'GPU (CUDA)']
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['s', '^', 'D', 'v']
    
    for framework, color, marker in zip(frameworks, colors, markers):
        fw_data = data[data['framework'] == framework].sort_values('n')
        efficiency = []
        matrix_sizes = []
        
        P = efficiency_map[framework]
        
        for _, row in fw_data.iterrows():
            n = row['n']
            if n in seq_baseline.index:
                speedup = seq_baseline[n] / row['time_s']
                eff = (speedup / P) * 100  # Percentage
                efficiency.append(eff)
                matrix_sizes.append(n)
        
        plt.plot(matrix_sizes, efficiency, 
                marker=marker, linewidth=2.5, markersize=8,
                label=framework, color=color)
    
    plt.xlabel('Matrix Size (N)', fontsize=12, fontweight='bold')
    plt.ylabel('Parallel Efficiency (%)', fontsize=12, fontweight='bold')
    plt.title('Parallel Efficiency Comparison: All Frameworks', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Ideal (100%)')
    plt.ylim(0, 110)
    plt.tight_layout()
    
    output_path = output_dir / "all_frameworks_efficiency.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_throughput_comparison(data, output_dir):
    """Plot computational throughput (GFLOPS) for all frameworks"""
    plt.figure(figsize=(12, 7))
    
    frameworks = ['Sequential', 'OpenMP', 'MPI Naive', 'MPI SUMMA', 'GPU (CUDA)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    for framework, color, marker in zip(frameworks, colors, markers):
        fw_data = data[data['framework'] == framework].sort_values('n')
        gflops = []
        matrix_sizes = []
        
        for _, row in fw_data.iterrows():
            n = row['n']
            flops = 2 * n**3  # Matrix multiplication FLOPS
            time_s = row['time_s']
            gflops_val = (flops / time_s) / 1e9
            gflops.append(gflops_val)
            matrix_sizes.append(n)
        
        plt.plot(matrix_sizes, gflops, 
                marker=marker, linewidth=2.5, markersize=8,
                label=framework, color=color)
    
    plt.xlabel('Matrix Size (N)', fontsize=12, fontweight='bold')
    plt.ylabel('Throughput (GFLOPS)', fontsize=12, fontweight='bold')
    plt.title('Computational Throughput Comparison: All Frameworks', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    
    output_path = output_dir / "all_frameworks_throughput.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_performance_matrix(data, output_dir):
    """Heatmap showing relative performance across frameworks and sizes"""
    pivot_data = data.pivot(index='framework', columns='n', values='time_s')
    
    # Normalize by sequential (show as percentage of sequential time)
    seq_times = pivot_data.loc['Sequential']
    normalized = (pivot_data / seq_times) * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(normalized.values, cmap='RdYlGn_r', aspect='auto', 
                   norm=plt.matplotlib.colors.LogNorm())
    
    # Set ticks
    ax.set_xticks(np.arange(len(normalized.columns)))
    ax.set_yticks(np.arange(len(normalized.index)))
    ax.set_xticklabels(normalized.columns)
    ax.set_yticklabels(normalized.index)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('% of Sequential Time', fontsize=11, fontweight='bold')
    
    # Add text annotations
    for i in range(len(normalized.index)):
        for j in range(len(normalized.columns)):
            val = normalized.values[i, j]
            text_color = 'white' if val < 10 or val > 200 else 'black'
            text = ax.text(j, i, f'{val:.1f}%',
                          ha="center", va="center", color=text_color, fontsize=9)
    
    ax.set_xlabel('Matrix Size (N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Framework', fontsize=12, fontweight='bold')
    ax.set_title('Relative Performance Matrix (% of Sequential Time)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / "all_frameworks_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_gpu_kernel_scaling(gpu_df, output_dir):
    """Plot GPU kernel performance scaling characteristics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Kernel time scaling
    ax1 = axes[0, 0]
    ax1.plot(gpu_df['n'], gpu_df['kernel_mean_ms'], 'o-', linewidth=2.5, markersize=8, color='#9467bd')
    ax1.set_xlabel('Matrix Size (N)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Kernel Time (ms)', fontsize=11, fontweight='bold')
    ax1.set_title('GPU Kernel Execution Time Scaling', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Kernel time with error bars (if std available)
    ax2 = axes[0, 1]
    if 'kernel_std_ms' in gpu_df.columns:
        ax2.errorbar(gpu_df['n'], gpu_df['kernel_mean_ms'], yerr=gpu_df['kernel_std_ms'],
                    fmt='s-', linewidth=2.5, markersize=8, color='#9467bd', 
                    capsize=5, label='Kernel Time ± Std')
    else:
        ax2.plot(gpu_df['n'], gpu_df['kernel_mean_ms'], 's-', linewidth=2.5, markersize=8, color='#9467bd')
    
    ax2.set_xlabel('Matrix Size (N)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
    ax2.set_title('GPU Kernel Time with Variance', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Computational intensity (GFLOPS)
    ax3 = axes[1, 0]
    gflops = [(2 * n**3) / (t * 1e6) for n, t in zip(gpu_df['n'], gpu_df['kernel_mean_ms'])]
    ax3.plot(gpu_df['n'], gflops, 's-', linewidth=2.5, markersize=8, color='#2ca02c')
    ax3.set_xlabel('Matrix Size (N)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Throughput (GFLOPS)', fontsize=11, fontweight='bold')
    ax3.set_title('GPU Kernel Throughput', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Efficiency (% of peak theoretical)
    ax4 = axes[1, 1]
    # Assuming RTX 3060 with ~12 TFLOPS peak
    peak_tflops = 12.0
    efficiency = [(g / (peak_tflops * 1000)) * 100 for g in gflops]
    ax4.plot(gpu_df['n'], efficiency, 'D-', linewidth=2.5, markersize=8, color='#d62728')
    ax4.set_xlabel('Matrix Size (N)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Peak Performance Utilization (%)', fontsize=11, fontweight='bold')
    ax4.set_title('GPU Efficiency vs Peak Performance', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_ylim(0, max(efficiency) * 1.2)
    
    plt.tight_layout()
    output_path = output_dir / "gpu_kernel_scaling_characteristics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def plot_cross_platform_overview(data, output_dir):
    """Comprehensive cross-platform performance overview dashboard"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Main execution time comparison (large)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    frameworks = ['Sequential', 'OpenMP', 'MPI Naive', 'MPI SUMMA', 'GPU (CUDA)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    for framework, color, marker in zip(frameworks, colors, markers):
        fw_data = data[data['framework'] == framework].sort_values('n')
        ax1.plot(fw_data['n'], fw_data['time_s'], 
                marker=marker, linewidth=2.5, markersize=8,
                label=framework, color=color)
    
    ax1.set_xlabel('Matrix Size (N)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
    ax1.set_title('Cross-Platform Execution Time', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Speedup comparison (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    seq_baseline = data[data['framework'] == 'Sequential'].set_index('n')['time_s']
    max_n = data['n'].max()
    
    speedups_at_max = []
    labels_at_max = []
    colors_at_max = []
    
    for framework, color in zip(frameworks[1:], colors[1:]):  # Skip Sequential
        fw_data = data[(data['framework'] == framework) & (data['n'] == max_n)]
        if not fw_data.empty and max_n in seq_baseline.index:
            speedup = seq_baseline[max_n] / fw_data['time_s'].values[0]
            speedups_at_max.append(speedup)
            labels_at_max.append(framework.replace(' ', '\n'))
            colors_at_max.append(color)
    
    bars = ax2.bar(range(len(speedups_at_max)), speedups_at_max, color=colors_at_max, alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(labels_at_max)))
    ax2.set_xticklabels(labels_at_max, fontsize=8)
    ax2.set_ylabel('Speedup', fontsize=10, fontweight='bold')
    ax2.set_title(f'Speedup at N={max_n}', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, speedup in zip(bars, speedups_at_max):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.1f}×', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. Efficiency comparison (middle right)
    ax3 = fig.add_subplot(gs[1, 2])
    efficiency_map = {'OpenMP': 12, 'MPI Naive': 6, 'MPI SUMMA': 6, 'GPU (CUDA)': 2048}
    
    efficiencies_at_max = []
    for framework in frameworks[1:]:
        fw_data = data[(data['framework'] == framework) & (data['n'] == max_n)]
        if not fw_data.empty and max_n in seq_baseline.index:
            speedup = seq_baseline[max_n] / fw_data['time_s'].values[0]
            P = efficiency_map[framework]
            eff = (speedup / P) * 100
            efficiencies_at_max.append(eff)
    
    bars = ax3.bar(range(len(efficiencies_at_max)), efficiencies_at_max, color=colors_at_max, alpha=0.7, edgecolor='black')
    ax3.set_xticks(range(len(labels_at_max)))
    ax3.set_xticklabels(labels_at_max, fontsize=8)
    ax3.set_ylabel('Efficiency (%)', fontsize=10, fontweight='bold')
    ax3.set_title(f'Parallel Efficiency at N={max_n}', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    for bar, eff in zip(bars, efficiencies_at_max):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{eff:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 4. Throughput comparison (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    
    for framework, color, marker in zip(frameworks, colors, markers):
        fw_data = data[data['framework'] == framework].sort_values('n')
        gflops = [(2 * n**3) / (t * 1e9) for n, t in zip(fw_data['n'], fw_data['time_s'])]
        ax4.plot(fw_data['n'], gflops, marker=marker, linewidth=2, markersize=6,
                label=framework, color=color)
    
    ax4.set_xlabel('Matrix Size (N)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('GFLOPS', fontsize=10, fontweight='bold')
    ax4.set_title('Computational Throughput', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=7, loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # 5. Relative performance at largest size (bottom middle)
    ax5 = fig.add_subplot(gs[2, 1])
    times_at_max = []
    all_labels = []
    all_colors = []
    
    for framework, color in zip(frameworks, colors):
        fw_data = data[(data['framework'] == framework) & (data['n'] == max_n)]
        if not fw_data.empty:
            times_at_max.append(fw_data['time_s'].values[0])
            all_labels.append(framework.replace(' ', '\n'))
            all_colors.append(color)
    
    bars = ax5.barh(range(len(times_at_max)), times_at_max, color=all_colors, alpha=0.7, edgecolor='black')
    ax5.set_yticks(range(len(all_labels)))
    ax5.set_yticklabels(all_labels, fontsize=8)
    ax5.set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
    ax5.set_title(f'Absolute Time at N={max_n}', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')
    ax5.set_xscale('log')
    
    for bar, time_val in zip(bars, times_at_max):
        width = bar.get_width()
        label = f'{time_val:.3f}s' if time_val >= 0.001 else f'{time_val*1000:.2f}ms'
        ax5.text(width * 1.1, bar.get_y() + bar.get_height()/2, 
                label, ha='left', va='center', fontsize=8, fontweight='bold')
    
    # 6. Performance ranking table (bottom right)
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    # Create ranking data
    ranking_data = []
    for framework in frameworks:
        fw_data = data[data['framework'] == framework]
        avg_time = fw_data['time_s'].mean()
        ranking_data.append([framework, f'{avg_time:.3f}s'])
    
    ranking_data.sort(key=lambda x: float(x[1].replace('s', '')))
    
    table_text = "Performance Ranking\n(Avg across all sizes)\n" + "="*25 + "\n"
    for rank, (fw, time) in enumerate(ranking_data, 1):
        table_text += f"{rank}. {fw}: {time}\n"
    
    ax6.text(0.1, 0.5, table_text, fontsize=9, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Cross-Platform Performance Overview Dashboard', fontsize=16, fontweight='bold', y=0.98)
    
    output_path = output_dir / "cross_platform_overview_dashboard.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()

def generate_summary_table(data, output_dir):
    """Generate and save summary comparison table"""
    pivot_data = data.pivot(index='framework', columns='n', values='time_s')
    
    # Calculate speedups
    seq_times = pivot_data.loc['Sequential']
    speedup_data = seq_times / pivot_data
    
    # Save to CSV
    summary_path = output_dir.parent / "all_frameworks_summary.csv"
    pivot_data.to_csv(summary_path)
    print(f"✅ Saved summary table: {summary_path}")
    
    speedup_path = output_dir.parent / "all_frameworks_speedup_table.csv"
    speedup_data.to_csv(speedup_path)
    print(f"✅ Saved speedup table: {speedup_path}")
    
    return pivot_data, speedup_data

def main():
    """Main execution"""
    print("=" * 70)
    print("Comprehensive All-Framework Comparison Analysis")
    print("=" * 70)
    
    # Load data
    gpu_df, mpi_df, openmp_seq_df = load_all_data()
    
    # Prepare unified dataset
    comparison_data = prepare_comparison_data(gpu_df, mpi_df, openmp_seq_df)
    
    # Setup output directory
    output_dir = Path(__file__).parent.parent / "plots"
    output_dir.mkdir(exist_ok=True)
    
    print("\n📈 Generating comprehensive comparison plots...")
    
    # Generate all plots
    plot_execution_time_comparison(comparison_data, output_dir)
    plot_speedup_comparison(comparison_data, output_dir)
    plot_relative_performance_bars(comparison_data, output_dir)
    plot_efficiency_comparison(comparison_data, output_dir)
    plot_throughput_comparison(comparison_data, output_dir)
    plot_performance_matrix(comparison_data, output_dir)
    
    # Additional specialized plots
    print("\n📊 Generating GPU kernel scaling analysis...")
    plot_gpu_kernel_scaling(gpu_df, output_dir)
    
    print("\n📊 Generating cross-platform overview dashboard...")
    plot_cross_platform_overview(comparison_data, output_dir)
    
    # Generate summary tables
    print("\n📊 Generating summary tables...")
    generate_summary_table(comparison_data, output_dir)
    
    print("\n" + "=" * 70)
    print("✅ All comprehensive comparison plots generated successfully!")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
