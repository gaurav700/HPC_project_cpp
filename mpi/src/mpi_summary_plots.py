import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the summary CSV file
df = pd.read_csv("../results/mpi_summary.csv")

# Create output directory if needed
import os
os.makedirs("../plots", exist_ok=True)

# 1. Execution Time vs Number of Processes (for each matrix size)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('MPI Matrix Multiplication: Mean Execution Time vs Processes', fontsize=16, fontweight='bold')

matrix_sizes = sorted(df['n'].unique())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for idx, n in enumerate(matrix_sizes):
    ax = axes[idx // 3, idx % 3]
    data = df[df['n'] == n].sort_values('processes')
    
    ax.errorbar(data['processes'], data['mean'], yerr=data['std'], 
                marker='o', capsize=5, capthick=2, linewidth=2.5, markersize=10,
                color=colors[idx % len(colors)], ecolor='gray', alpha=0.8)
    ax.set_xlabel('Number of Processes', fontsize=11)
    ax.set_ylabel('Time (seconds)', fontsize=11)
    ax.set_title(f'Matrix Size: {n}×{n}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(data['processes'].unique())

# Hide the last subplot if odd number of matrices
if len(matrix_sizes) % 3 != 0:
    axes[-1, -1].axis('off')

plt.tight_layout()
plt.savefig("../plots/summary_time_vs_processes.png", dpi=150, bbox_inches='tight')
plt.close()
print('✅ Plot 1 saved: summary_time_vs_processes.png')

# 2. Speedup Analysis
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('MPI Matrix Multiplication: Speedup vs Processes', fontsize=16, fontweight='bold')

for idx, n in enumerate(matrix_sizes):
    ax = axes[idx // 3, idx % 3]
    data = df[df['n'] == n].sort_values('processes')
    
    # Calculate speedup relative to single process
    single_proc_time = data[data['processes'] == 1]['mean'].values[0]
    speedup = single_proc_time / data['mean']
    
    ax.plot(data['processes'], speedup, 'o-', linewidth=2.5, markersize=10, 
            label='Actual Speedup', color=colors[idx % len(colors)])
    ax.plot(data['processes'], data['processes'], 'k--', linewidth=2, label='Ideal Speedup', alpha=0.7)
    ax.set_xlabel('Number of Processes', fontsize=11)
    ax.set_ylabel('Speedup', fontsize=11)
    ax.set_title(f'Matrix Size: {n}×{n}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xticks(data['processes'].unique())

if len(matrix_sizes) % 3 != 0:
    axes[-1, -1].axis('off')

plt.tight_layout()
plt.savefig("../plots/summary_speedup_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print('✅ Plot 2 saved: summary_speedup_analysis.png')

# 3. All data together - Execution Time vs Matrix Size, colored by processes
fig, ax = plt.subplots(figsize=(12, 7))

for processes in sorted(df['processes'].unique()):
    data = df[df['processes'] == processes].sort_values('n')
    ax.plot(data['n'], data['mean'], 'o-', linewidth=2.5, 
            markersize=10, label=f'{processes} Process(es)')

ax.set_xlabel('Matrix Size (n)', fontsize=12)
ax.set_ylabel('Execution Time (seconds)', fontsize=12)
ax.set_title('MPI Matrix Multiplication: Time vs Matrix Size', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=11, loc='upper left')
plt.tight_layout()
plt.savefig("../plots/summary_time_vs_matrix_size.png", dpi=150, bbox_inches='tight')
plt.close()
print('✅ Plot 3 saved: summary_time_vs_matrix_size.png')

# 4. Efficiency Analysis
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('MPI Matrix Multiplication: Parallel Efficiency', fontsize=16, fontweight='bold')

for idx, n in enumerate(matrix_sizes):
    ax = axes[idx // 3, idx % 3]
    data = df[df['n'] == n].sort_values('processes')
    
    # Calculate efficiency: speedup / processes
    single_proc_time = data[data['processes'] == 1]['mean'].values[0]
    speedup = single_proc_time / data['mean']
    efficiency = speedup / data['processes'] * 100
    
    bars = ax.bar(data['processes'].astype(str), efficiency, alpha=0.7, width=0.6,
                   color=colors[idx % len(colors)])
    ax.axhline(y=100, color='r', linestyle='--', linewidth=2, label='100% Efficiency', alpha=0.7)
    ax.set_xlabel('Number of Processes', fontsize=11)
    ax.set_ylabel('Efficiency (%)', fontsize=11)
    ax.set_title(f'Matrix Size: {n}×{n}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

if len(matrix_sizes) % 3 != 0:
    axes[-1, -1].axis('off')

plt.tight_layout()
plt.savefig("../plots/summary_efficiency_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print('✅ Plot 4 saved: summary_efficiency_analysis.png')

# 5. Standard Deviation Analysis (Error margins)
fig, ax = plt.subplots(figsize=(12, 7))

for n in matrix_sizes:
    data = df[df['n'] == n].sort_values('processes')
    ax.errorbar(data['processes'], data['mean'], yerr=data['std'],
                marker='o', capsize=5, capthick=2, linewidth=2.5, markersize=10,
                label=f'n={n}', alpha=0.8)

ax.set_xlabel('Number of Processes', fontsize=12)
ax.set_ylabel('Execution Time (seconds)', fontsize=12)
ax.set_title('MPI Performance: Mean Execution Time with Standard Deviation', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("../plots/summary_std_deviation_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print('✅ Plot 5 saved: summary_std_deviation_analysis.png')

# 6. Coefficient of Variation (stability metric)
fig, ax = plt.subplots(figsize=(12, 7))

df['cv'] = (df['std'] / df['mean']) * 100  # Coefficient of variation

for n in matrix_sizes:
    data = df[df['n'] == n].sort_values('processes')
    ax.plot(data['processes'], data['cv'], 'o-', linewidth=2.5, markersize=10, label=f'n={n}')

ax.set_xlabel('Number of Processes', fontsize=12)
ax.set_ylabel('Coefficient of Variation (%)', fontsize=12)
ax.set_title('MPI Performance Stability: Variability Across Runs', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("../plots/summary_cv_stability.png", dpi=150, bbox_inches='tight')
plt.close()
print('✅ Plot 6 saved: summary_cv_stability.png')

# Print summary statistics
print('\n' + '='*60)
print('📊 SUMMARY STATISTICS')
print('='*60)
print(f'Matrix sizes analyzed: {sorted(df["n"].unique())}')
print(f'Process counts: {sorted(df["processes"].unique())}')
print(f'Total configurations: {len(df)}')
print('\nBest Speedup (actual vs ideal ratio):')
for n in sorted(df['n'].unique()):
    data = df[df['n'] == n].sort_values('processes')
    single_proc = data[data['processes'] == 1]['mean'].values[0]
    for _, row in data.iterrows():
        speedup = single_proc / row['mean']
        efficiency = (speedup / row['processes']) * 100
        if row['processes'] == 6:  # Show best performance
            print(f'  n={n}, p={row["processes"]}: Speedup={speedup:.2f}x, Efficiency={efficiency:.1f}%')
print('='*60)

print('\n🎉 All plots generated successfully!')
