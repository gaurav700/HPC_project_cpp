#!/usr/bin/env python3
"""
GPU Benchmark Visualization
Generates plots from gpu_repeats.csv with detailed analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

def create_plots():
    """Generate visualization plots from GPU benchmark data"""
    
    # Use absolute paths based on script location
    script_dir = Path(__file__).parent
    cuda_root = script_dir.parent
    repeats_file = cuda_root / "results" / "gpu_repeats.csv"
    plots_dir = cuda_root / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    if not repeats_file.exists():
        print(f"❌ Error: {repeats_file} not found")
        print("   Run: bash ../scripts/run_gpu_tests.sh")
        sys.exit(1)
    
    try:
        df = pd.read_csv(repeats_file)
        summary = df.groupby(['impl', 'n']).agg({
            'kernel_ms': ['mean', 'std']
        }).reset_index()
        summary.columns = ['impl', 'n', 'kernel_mean', 'kernel_std']
        
        print("\n🎨 Generating GPU benchmark visualizations...\n")
        
        # Plot 1: Kernel Execution Time vs Matrix Size
        print("📊 Plot 1: Kernel time by matrix size...")
        fig, ax = plt.subplots(figsize=(10, 6))
        for impl in summary['impl'].unique():
            data = summary[summary['impl'] == impl].sort_values('n')
            ax.plot(data['n'], data['kernel_mean'], 'o-', label=impl, linewidth=2, markersize=8)
            ax.fill_between(data['n'], 
                            data['kernel_mean'] - data['kernel_std'],
                            data['kernel_mean'] + data['kernel_std'],
                            alpha=0.2)
        ax.set_xlabel('Matrix Size (NxN)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Kernel Time (ms)', fontsize=12, fontweight='bold')
        ax.set_title('GPU Kernel Execution Time vs Matrix Size', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(plots_dir / 'gpu_kernel_time_vs_matrix.png', dpi=150)
        print(f"   ✅ Saved: gpu_kernel_time_vs_matrix.png")
        plt.close()
        
        # Plot 2: Variability Analysis (Standard Deviation)
        print("📊 Plot 2: Result variability analysis...")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for impl in summary['impl'].unique():
            data = summary[summary['impl'] == impl].sort_values('n')
            cv = (data['kernel_std'] / data['kernel_mean'] * 100).fillna(0)
            ax.plot(data['n'], cv, 'o-', label=impl, linewidth=2, markersize=8)
        
        ax.set_xlabel('Matrix Size (NxN)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
        ax.set_title('Kernel Time Variability Across Runs', fontsize=14, fontweight='bold')
        ax.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='5% threshold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        plt.tight_layout()
        plt.savefig(plots_dir / 'gpu_variability_analysis.png', dpi=150)
        print(f"   ✅ Saved: gpu_variability_analysis.png")
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
