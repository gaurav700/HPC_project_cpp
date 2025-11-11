import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv("../results/sequential_openmp_repeats.csv")

# Filter out any empty rows
df = df.dropna()

if len(df) == 0:
    print("❌ No data found in CSV file. Please run the benchmarks first.")
    exit(1)

# Create output directory if needed
import os
os.makedirs("../plots", exist_ok=True)

# Get unique values
frameworks = df['framework'].unique()
matrix_sizes = sorted(df['n'].unique())
threads_list = sorted(df[df['framework'] == 'OpenMP']['threads'].unique())

print(f"Frameworks: {frameworks}")
print(f"Matrix sizes: {matrix_sizes}")
print(f"Thread counts: {threads_list}")

# 1. Execution Time vs Threads (for each matrix size)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Sequential vs OpenMP: Execution Time vs Threads', fontsize=16, fontweight='bold')

for idx, n in enumerate(matrix_sizes):
    ax = axes[idx // 3, idx % 3]
    
    # Sequential (single point)
    seq_data = df[(df['n'] == n) & (df['framework'] == 'sequential')]
    if len(seq_data) > 0:
        seq_mean = seq_data['time_s'].mean()
        seq_std = seq_data['time_s'].std()
        ax.errorbar(1, seq_mean, yerr=seq_std, marker='s', capsize=5, capthick=2, 
                   linewidth=2.5, markersize=10, label='Sequential', color='red', alpha=0.8)
    
    # OpenMP (multiple threads)
    omp_data = df[(df['n'] == n) & (df['framework'] == 'OpenMP')].sort_values('threads')
    if len(omp_data) > 0:
        omp_grouped = omp_data.groupby('threads').agg({'time_s': ['mean', 'std']})
        omp_grouped.columns = ['mean', 'std']
        ax.errorbar(omp_grouped.index, omp_grouped['mean'], yerr=omp_grouped['std'],
                   marker='o', capsize=5, capthick=2, linewidth=2.5, markersize=10,
                   label='OpenMP', color='blue', alpha=0.8)
    
    ax.set_xlabel('Threads', fontsize=11)
    ax.set_ylabel('Time (seconds)', fontsize=11)
    ax.set_title(f'Matrix Size: {n}×{n}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    if len(omp_data) > 0:
        ax.set_xticks([1] + list(omp_grouped.index))

# Hide the last subplot if odd number of matrices
if len(matrix_sizes) % 3 != 0:
    axes[-1, -1].axis('off')

plt.tight_layout()
plt.savefig("../plots/sequential_openmp_time_vs_threads.png", dpi=150, bbox_inches='tight')
plt.close()
print('✅ Plot 1 saved: sequential_openmp_time_vs_threads.png')

# 2. Speedup Analysis (OpenMP only)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('OpenMP: Speedup vs Threads', fontsize=16, fontweight='bold')

for idx, n in enumerate(matrix_sizes):
    ax = axes[idx // 3, idx % 3]
    
    omp_data = df[(df['n'] == n) & (df['framework'] == 'OpenMP')].sort_values('threads')
    seq_data = df[(df['n'] == n) & (df['framework'] == 'sequential')]
    
    if len(omp_data) > 0 and len(seq_data) > 0:
        seq_time = seq_data['time_s'].mean()
        omp_grouped = omp_data.groupby('threads')['time_s'].agg(['mean', 'std'])
        
        speedup = seq_time / omp_grouped['mean']
        speedup_std = speedup * (omp_grouped['std'] / omp_grouped['mean'])
        
        ax.errorbar(omp_grouped.index, speedup, yerr=speedup_std, 
                   marker='o', capsize=5, capthick=2, linewidth=2.5, markersize=10,
                   label='Actual Speedup', color='#1f77b4', alpha=0.8)
        ax.plot(omp_grouped.index, omp_grouped.index, 'k--', linewidth=2, 
               label='Ideal Speedup', alpha=0.7)
        
        ax.set_xlabel('Threads', fontsize=11)
        ax.set_ylabel('Speedup', fontsize=11)
        ax.set_title(f'Matrix Size: {n}×{n}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_xticks(omp_grouped.index)

if len(matrix_sizes) % 3 != 0:
    axes[-1, -1].axis('off')

plt.tight_layout()
plt.savefig("../plots/sequential_openmp_speedup.png", dpi=150, bbox_inches='tight')
plt.close()
print('✅ Plot 2 saved: sequential_openmp_speedup.png')

# 3. Parallel Efficiency (OpenMP)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('OpenMP: Parallel Efficiency', fontsize=16, fontweight='bold')

for idx, n in enumerate(matrix_sizes):
    ax = axes[idx // 3, idx % 3]
    
    omp_data = df[(df['n'] == n) & (df['framework'] == 'OpenMP')].sort_values('threads')
    seq_data = df[(df['n'] == n) & (df['framework'] == 'sequential')]
    
    if len(omp_data) > 0 and len(seq_data) > 0:
        seq_time = seq_data['time_s'].mean()
        omp_grouped = omp_data.groupby('threads')['time_s'].mean()
        
        speedup = seq_time / omp_grouped
        efficiency = (speedup / omp_grouped.index) * 100
        
        bars = ax.bar(omp_grouped.index.astype(str), efficiency, alpha=0.7, width=0.6, color='#2ca02c')
        ax.axhline(y=100, color='r', linestyle='--', linewidth=2, label='100% Efficiency', alpha=0.7)
        ax.set_xlabel('Threads', fontsize=11)
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
plt.savefig("../plots/sequential_openmp_efficiency.png", dpi=150, bbox_inches='tight')
plt.close()
print('✅ Plot 3 saved: sequential_openmp_efficiency.png')

# 4. Time vs Matrix Size (all frameworks and threads)
fig, ax = plt.subplots(figsize=(12, 7))

# Sequential line
seq_data = df[df['framework'] == 'sequential'].sort_values('n')
seq_grouped = seq_data.groupby('n')['time_s'].agg(['mean', 'std'])
ax.errorbar(seq_grouped.index, seq_grouped['mean'], yerr=seq_grouped['std'],
           marker='s', capsize=5, capthick=2, linewidth=2.5, markersize=10,
           label='Sequential', color='red', alpha=0.8)

# OpenMP threads
omp_data = df[df['framework'] == 'OpenMP']
for threads in sorted(omp_data['threads'].unique()):
    thread_data = omp_data[omp_data['threads'] == threads].sort_values('n')
    thread_grouped = thread_data.groupby('n')['time_s'].agg(['mean', 'std'])
    ax.errorbar(thread_grouped.index, thread_grouped['mean'], yerr=thread_grouped['std'],
               marker='o', capsize=5, capthick=2, linewidth=2.5, markersize=10,
               label=f'OpenMP ({threads}T)', alpha=0.8)

ax.set_xlabel('Matrix Size (n)', fontsize=12)
ax.set_ylabel('Execution Time (seconds)', fontsize=12)
ax.set_title('Sequential vs OpenMP: Time vs Matrix Size', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=11, loc='upper left')
plt.tight_layout()
plt.savefig("../plots/sequential_openmp_time_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print('✅ Plot 4 saved: sequential_openmp_time_comparison.png')

# 5. Variability Analysis
fig, ax = plt.subplots(figsize=(12, 7))

for threads in sorted(df[df['framework'] == 'OpenMP']['threads'].unique()):
    thread_data = df[(df['framework'] == 'OpenMP') & (df['threads'] == threads)].sort_values('n')
    thread_grouped = thread_data.groupby('n')['time_s'].agg(['mean', 'std'])
    cv = (thread_grouped['std'] / thread_grouped['mean']) * 100
    ax.plot(cv.index, cv.values, 'o-', linewidth=2.5, markersize=10, label=f'{threads} threads')

seq_data = df[df['framework'] == 'sequential'].sort_values('n')
seq_grouped = seq_data.groupby('n')['time_s'].agg(['mean', 'std'])
cv_seq = (seq_grouped['std'] / seq_grouped['mean']) * 100
ax.plot(cv_seq.index, cv_seq.values, 's-', linewidth=2.5, markersize=10, label='Sequential', color='red')

ax.set_xlabel('Matrix Size (n)', fontsize=12)
ax.set_ylabel('Coefficient of Variation (%)', fontsize=12)
ax.set_title('Performance Stability: Variability Across Runs', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("../plots/sequential_openmp_variability.png", dpi=150, bbox_inches='tight')
plt.close()
print('✅ Plot 5 saved: sequential_openmp_variability.png')

print('\n🎉 All plots generated successfully!')
print(f'Total data points: {len(df)}')
print(f'Frameworks: {list(df["framework"].unique())}')
