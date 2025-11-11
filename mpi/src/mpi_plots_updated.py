#!/usr/bin/env python3
"""
MPI Benchmark Visualization
Generates plots from mpi_repeats.csv comparing naive broadcast vs SUMMA
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

def create_plots():
    """Generate visualization plots from MPI benchmark data"""
    
    # Use absolute paths
    script_dir = Path(__file__).parent
    mpi_root = script_dir.parent
    repeats_file = mpi_root / "results" / "mpi_repeats.csv"
    plots_dir = mpi_root / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    if not repeats_file.exists():
        print(f"❌ Error: {repeats_file} not found")
        sys.exit(1)
    
    try:
        df = pd.read_csv(repeats_file)
        summary = df.groupby(['impl', 'n', 'processes']).agg({
            'time_s': ['mean', 'std']
        }).reset_index()
        summary.columns = ['impl', 'n', 'processes', 'time_mean', 'time_std']
        
        print("\n🎨 Generating MPI benchmark visualizations...\n")
        
        # Plot 1: Execution Time Comparison (naive vs SUMMA)
        print("📊 Plot 1: Execution time comparison...")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for impl in sorted(summary['impl'].unique()):
            data = summary[summary['impl'] == impl].sort_values('n')
            ax.plot(data['n'], data['time_mean'], 'o-', label=impl, linewidth=2.5, markersize=10)
            ax.fill_between(data['n'], 
                            data['time_mean'] - data['time_std'],
                            data['time_mean'] + data['time_std'],
                            alpha=0.2)
        
        ax.set_xlabel('Matrix Size (NxN)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('MPI Matrix Multiplication: Naive Broadcast vs SUMMA\n(4 processes)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(plots_dir / 'mpi_algorithm_comparison.png', dpi=150)
        print(f"   ✅ Saved: mpi_algorithm_comparison.png")
        plt.close()
        
        # Plot 2: Speedup of SUMMA over naive
        print("📊 Plot 2: SUMMA speedup analysis...")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        speedup_data = []
        for n in sorted(df['n'].unique()):
            naive_time = summary[(summary['impl'] == 'mpi_matmul') & (summary['n'] == n)]['time_mean'].values[0]
            summa_time = summary[(summary['impl'] == 'mpi_summa') & (summary['n'] == n)]['time_mean'].values[0]
            speedup = naive_time / summa_time
            speedup_data.append({'n': n, 'speedup': speedup})
        
        speedup_df = pd.DataFrame(speedup_data)
        bars = ax.bar(speedup_df['n'].astype(str), speedup_df['speedup'], color='#2ca02c', alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}×',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='No speedup', alpha=0.7)
        ax.set_xlabel('Matrix Size (NxN)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Speedup Factor (Naive / SUMMA)', fontsize=12, fontweight='bold')
        ax.set_title('SUMMA Algorithm Speedup Over Naive Broadcast\n(4 processes, single-node)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(plots_dir / 'mpi_summa_speedup.png', dpi=150)
        print(f"   ✅ Saved: mpi_summa_speedup.png")
        plt.close()
        
        # Plot 3: Variability Analysis
        print("📊 Plot 3: Result variability...")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for impl in sorted(summary['impl'].unique()):
            data = summary[summary['impl'] == impl].sort_values('n')
            cv = (data['time_std'] / data['time_mean'] * 100).fillna(0)
            ax.plot(data['n'], cv, 'o-', label=impl, linewidth=2.5, markersize=10)
        
        ax.set_xlabel('Matrix Size (NxN)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
        ax.set_title('Execution Time Variability Across Runs\n(4 processes)', 
                    fontsize=14, fontweight='bold')
        ax.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='5% threshold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        plt.tight_layout()
        plt.savefig(plots_dir / 'mpi_variability_analysis.png', dpi=150)
        print(f"   ✅ Saved: mpi_variability_analysis.png")
        plt.close()
        
        # Plot 4: Performance Scaling (Flops per second estimate)
        print("📊 Plot 4: Estimated computational throughput...")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        flops_data = []
        for impl in sorted(summary['impl'].unique()):
            for n in sorted(df['n'].unique()):
                time_s = summary[(summary['impl'] == impl) & (summary['n'] == n)]['time_mean'].values[0]
                # FLOPS = 2*N^3 / time (matrix multiplication is 2N^3 floating-point operations)
                flops = (2 * n**3) / (time_s * 1e9)  # convert to GFLOPS
                flops_data.append({'impl': impl, 'n': n, 'gflops': flops})
        
        flops_df = pd.DataFrame(flops_data)
        for impl in sorted(flops_df['impl'].unique()):
            data = flops_df[flops_df['impl'] == impl].sort_values('n')
            ax.plot(data['n'], data['gflops'], 'o-', label=impl, linewidth=2.5, markersize=10)
        
        ax.set_xlabel('Matrix Size (NxN)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Throughput (GFLOPS)', fontsize=12, fontweight='bold')
        ax.set_title('Estimated Computational Throughput\n(4 processes)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(plots_dir / 'mpi_throughput.png', dpi=150)
        print(f"   ✅ Saved: mpi_throughput.png")
        plt.close()
        
        print("\n" + "="*60)
        print(f"✅ All plots saved to: {plots_dir}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    create_plots()
