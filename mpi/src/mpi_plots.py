import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv("../results/mpi_repeats.csv")

# Calculate average time for each configuration
avg_df = df.groupby(['n', 'processes']).agg({
    'time_s': ['mean', 'std'],
    'checksum': 'first'
}).reset_index()
avg_df.columns = ['n', 'processes', 'time_mean', 'time_std', 'checksum']

# Create output directory if needed
import os
os.makedirs("../plots", exist_ok=True)

# 1. Execution Time vs Number of Processes (for each matrix size)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('MPI Matrix Multiplication: Execution Time vs Processes', fontsize=16)

matrix_sizes = sorted(df['n'].unique())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for idx, n in enumerate(matrix_sizes):
    ax = axes[idx // 3, idx % 3]
    data = avg_df[avg_df['n'] == n].sort_values('processes')
    
    ax.errorbar(data['processes'], data['time_mean'], yerr=data['time_std'], 
                marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
    ax.set_xlabel('Number of Processes')
    ax.set_ylabel('Time (seconds)')
    ax.set_title(f'Matrix Size: {n}x{n}')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(data['processes'].unique())

# Hide the last subplot if odd number of matrices
if len(matrix_sizes) % 3 != 0:
    axes[-1, -1].axis('off')

plt.tight_layout()
plt.savefig("../plots/mpi_time_vs_processes.png", dpi=150, bbox_inches='tight')
plt.close()
print('✅ Plot 1 saved: mpi_time_vs_processes.png')

# 2. Speedup Analysis
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('MPI Matrix Multiplication: Speedup vs Processes', fontsize=16)

for idx, n in enumerate(matrix_sizes):
    ax = axes[idx // 3, idx % 3]
    data = avg_df[avg_df['n'] == n].sort_values('processes')
    
    # Calculate speedup relative to single process
    single_proc_time = data[data['processes'] == 1]['time_mean'].values[0]
    speedup = single_proc_time / data['time_mean']
    
    ax.plot(data['processes'], speedup, 'o-', linewidth=2, markersize=8, label='Actual Speedup')
    ax.plot(data['processes'], data['processes'], 'k--', linewidth=2, label='Ideal Speedup')
    ax.set_xlabel('Number of Processes')
    ax.set_ylabel('Speedup')
    ax.set_title(f'Matrix Size: {n}x{n}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks(data['processes'].unique())

if len(matrix_sizes) % 3 != 0:
    axes[-1, -1].axis('off')

plt.tight_layout()
plt.savefig("../plots/mpi_speedup_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print('✅ Plot 2 saved: mpi_speedup_analysis.png')

# 3. All data together - Execution Time vs Matrix Size, colored by processes
fig, ax = plt.subplots(figsize=(12, 7))

for processes in sorted(df['processes'].unique()):
    data = avg_df[avg_df['processes'] == processes].sort_values('n')
    ax.plot(data['n'], data['time_mean'], 'o-', linewidth=2.5, 
            markersize=10, label=f'{processes} Process(es)')

ax.set_xlabel('Matrix Size (n)', fontsize=12)
ax.set_ylabel('Execution Time (seconds)', fontsize=12)
ax.set_title('MPI Matrix Multiplication: Time vs Matrix Size', fontsize=14)
ax.set_yscale('log')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("../plots/mpi_time_vs_matrix_size.png", dpi=150, bbox_inches='tight')
plt.close()
print('✅ Plot 3 saved: mpi_time_vs_matrix_size.png')

# 4. Efficiency Analysis
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('MPI Matrix Multiplication: Parallel Efficiency', fontsize=16)

for idx, n in enumerate(matrix_sizes):
    ax = axes[idx // 3, idx % 3]
    data = avg_df[avg_df['n'] == n].sort_values('processes')
    
    # Calculate efficiency: speedup / processes
    single_proc_time = data[data['processes'] == 1]['time_mean'].values[0]
    speedup = single_proc_time / data['time_mean']
    efficiency = speedup / data['processes'] * 100
    
    ax.bar(data['processes'], efficiency, alpha=0.7, width=0.6)
    ax.axhline(y=100, color='r', linestyle='--', linewidth=2, label='100% Efficiency')
    ax.set_xlabel('Number of Processes')
    ax.set_ylabel('Efficiency (%)')
    ax.set_title(f'Matrix Size: {n}x{n}')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(data['processes'].unique())
    ax.legend()

if len(matrix_sizes) % 3 != 0:
    axes[-1, -1].axis('off')

plt.tight_layout()
plt.savefig("../plots/mpi_efficiency_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print('✅ Plot 4 saved: mpi_efficiency_analysis.png')

# 5. Variability Analysis (error bars across runs)
fig, ax = plt.subplots(figsize=(12, 7))

for n in matrix_sizes:
    data = avg_df[avg_df['n'] == n].sort_values('processes')
    ax.errorbar(data['processes'], data['time_mean'], yerr=data['time_std'],
                marker='o', capsize=5, capthick=2, linewidth=2, markersize=8,
                label=f'n={n}')

ax.set_xlabel('Number of Processes', fontsize=12)
ax.set_ylabel('Execution Time (seconds)', fontsize=12)
ax.set_title('MPI Performance: Variability Across Runs', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("../plots/mpi_variability_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print('✅ Plot 5 saved: mpi_variability_analysis.png')

print('\n🎉 All plots generated successfully!')
print(f'Total runs analyzed: {len(df)}')
print(f'Matrix sizes: {sorted(df["n"].unique())}')
print(f'Process counts: {sorted(df["processes"].unique())}')
