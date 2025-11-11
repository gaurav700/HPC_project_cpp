#!/usr/bin/env python3
"""
GPU Energy Efficiency Analysis
Combines performance data with power measurements to calculate GFLOPS/Watt
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Read data
base_dir = Path(__file__).parent.parent
gpu_repeats = pd.read_csv(base_dir / "results" / "gpu_repeats.csv")
gpu_power = pd.read_csv(base_dir / "results" / "energy_measurements" / "gpu_power_measurements.csv")

# Calculate GFLOPS for each run
gpu_repeats['gflops'] = (2 * gpu_repeats['n']**3) / (gpu_repeats['kernel_ms'] / 1000 * 1e9)

# Merge with power data
merged = pd.merge(
    gpu_repeats,
    gpu_power[['matrix_size', 'run', 'avg_power_w', 'total_energy_j']],
    left_on=['n', 'run'],
    right_on=['matrix_size', 'run'],
    how='left'
)

# Calculate energy efficiency (GFLOPS per Watt)
merged['gflops_per_watt'] = merged['gflops'] / merged['avg_power_w']

# Calculate energy per operation (Joules per GFLOP)
merged['joules_per_gflop'] = (merged['avg_power_w'] * merged['kernel_ms'] / 1000) / merged['gflops']

# Group by matrix size
summary = merged.groupby('n').agg({
    'gflops': 'mean',
    'avg_power_w': 'mean',
    'gflops_per_watt': 'mean',
    'joules_per_gflop': 'mean',
    'total_energy_j': 'mean'
}).round(3)

print("\n" + "="*80)
print("📊 GPU ENERGY EFFICIENCY ANALYSIS")
print("="*80)
print(summary.to_string())
print("="*80 + "\n")

# Save summary
summary.to_csv(base_dir / "results" / "energy_measurements" / "gpu_energy_efficiency.csv")
print(f"✅ Saved to: results/energy_measurements/gpu_energy_efficiency.csv\n")

# Generate energy efficiency plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: GFLOPS/Watt vs Matrix Size
ax1.plot(summary.index, summary['gflops_per_watt'], 'o-', linewidth=2.5, markersize=10, color='green')
ax1.set_xlabel('Matrix Size (N×N)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Energy Efficiency (GFLOPS/Watt)', fontsize=12, fontweight='bold')
ax1.set_title('GPU Energy Efficiency', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

# Plot 2: Power Draw vs Matrix Size
ax2.plot(summary.index, summary['avg_power_w'], 'o-', linewidth=2.5, markersize=10, color='red')
ax2.set_xlabel('Matrix Size (N×N)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Average Power Draw (Watts)', fontsize=12, fontweight='bold')
ax2.set_title('GPU Power Consumption', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

plt.tight_layout()
plt.savefig(base_dir / "plots" / "gpu_energy_efficiency.png", dpi=150, bbox_inches='tight')
print(f"✅ Plot saved to: plots/gpu_energy_efficiency.png\n")
plt.close()

# Additional plot: Energy vs Performance
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(summary['gflops'], summary['avg_power_w'], s=200, alpha=0.7, c=range(len(summary)), cmap='viridis')
for i, (idx, row) in enumerate(summary.iterrows()):
    ax.annotate(f'N={idx}', (row['gflops'], row['avg_power_w']), 
                xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')

ax.set_xlabel('Computational Throughput (GFLOPS)', fontsize=12, fontweight='bold')
ax.set_ylabel('Power Draw (Watts)', fontsize=12, fontweight='bold')
ax.set_title('GPU Performance vs Power Consumption', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(base_dir / "plots" / "gpu_performance_vs_power.png", dpi=150, bbox_inches='tight')
print(f"✅ Plot saved to: plots/gpu_performance_vs_power.png\n")

print("="*80)
print("✅ Energy efficiency analysis complete!")
print("="*80 + "\n")
