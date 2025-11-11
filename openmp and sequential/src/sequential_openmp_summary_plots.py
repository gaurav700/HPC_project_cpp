import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file and calculate summaries
df = pd.read_csv("../results/sequential_openmp_repeats.csv")

# Filter out any empty rows
df = df.dropna()

if len(df) == 0:
    print("❌ No data found in CSV file. Please run the benchmarks first.")
    exit(1)

# Calculate summary statistics
summary_data = []

# Sequential summary
seq_data = df[df['framework'] == 'sequential']
if len(seq_data) > 0:
    for n in sorted(seq_data['n'].unique()):
        n_data = seq_data[seq_data['n'] == n]
        summary_data.append({
            'framework': 'sequential',
            'n': n,
            'threads': 1,
            'mean': n_data['time_s'].mean(),
            'std': n_data['time_s'].std(),
            'count': len(n_data)
        })

# OpenMP summary
omp_data = df[df['framework'] == 'OpenMP']
if len(omp_data) > 0:
    for n in sorted(omp_data['n'].unique()):
        for threads in sorted(omp_data[omp_data['n'] == n]['threads'].unique()):
            thread_n_data = omp_data[(omp_data['n'] == n) & (omp_data['threads'] == threads)]
            summary_data.append({
                'framework': 'OpenMP',
                'n': n,
                'threads': threads,
                'mean': thread_n_data['time_s'].mean(),
                'std': thread_n_data['time_s'].std(),
                'count': len(thread_n_data)
            })

summary_df = pd.DataFrame(summary_data)

# Create output directory if needed
import os
os.makedirs("../plots", exist_ok=True)

# Get unique values
matrix_sizes = sorted(summary_df['n'].unique())
threads_list = sorted(summary_df[summary_df['framework'] == 'OpenMP']['threads'].unique())

print(f"Matrix sizes: {matrix_sizes}")
print(f"Thread counts: {threads_list}")

# 1. Summary: Execution Time vs Threads (for each matrix size)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Summary: Sequential vs OpenMP - Execution Time vs Threads', fontsize=16, fontweight='bold')

for idx, n in enumerate(matrix_sizes):
    ax = axes[idx // 3, idx % 3]
    
    # Sequential
    seq_summary = summary_df[(summary_df['n'] == n) & (summary_df['framework'] == 'sequential')]
    if len(seq_summary) > 0:
        ax.errorbar(1, seq_summary['mean'].values[0], yerr=seq_summary['std'].values[0],
                   marker='s', capsize=5, capthick=2, linewidth=2.5, markersize=10,
                   label='Sequential', color='red', alpha=0.8)
    
    # OpenMP
    omp_summary = summary_df[(summary_df['n'] == n) & (summary_df['framework'] == 'OpenMP')].sort_values('threads')
    if len(omp_summary) > 0:
        ax.errorbar(omp_summary['threads'], omp_summary['mean'], yerr=omp_summary['std'],
                   marker='o', capsize=5, capthick=2, linewidth=2.5, markersize=10,
                   label='OpenMP', color='blue', alpha=0.8)
    
    ax.set_xlabel('Threads', fontsize=11)
    ax.set_ylabel('Time (seconds)', fontsize=11)
    ax.set_title(f'Matrix Size: {n}×{n}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

if len(matrix_sizes) % 3 != 0:
    axes[-1, -1].axis('off')

plt.tight_layout()
plt.savefig("../plots/summary_time_vs_threads.png", dpi=150, bbox_inches='tight')
plt.close()
print('✅ Summary Plot 1 saved: summary_time_vs_threads.png')

# 2. Summary: Speedup Analysis
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Summary: OpenMP Speedup vs Threads', fontsize=16, fontweight='bold')

for idx, n in enumerate(matrix_sizes):
    ax = axes[idx // 3, idx % 3]
    
    omp_summary = summary_df[(summary_df['n'] == n) & (summary_df['framework'] == 'OpenMP')].sort_values('threads')
    seq_summary = summary_df[(summary_df['n'] == n) & (summary_df['framework'] == 'sequential')]
    
    if len(omp_summary) > 0 and len(seq_summary) > 0:
        seq_time = seq_summary['mean'].values[0]
        speedup = seq_time / omp_summary['mean']
        
        ax.plot(omp_summary['threads'], speedup, 'o-', linewidth=2.5, markersize=10,
               label='Actual Speedup', color='#1f77b4', alpha=0.8)
        ax.plot(omp_summary['threads'], omp_summary['threads'], 'k--', linewidth=2,
               label='Ideal Speedup', alpha=0.7)
        
        ax.set_xlabel('Threads', fontsize=11)
        ax.set_ylabel('Speedup', fontsize=11)
        ax.set_title(f'Matrix Size: {n}×{n}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_xticks(omp_summary['threads'].unique())

if len(matrix_sizes) % 3 != 0:
    axes[-1, -1].axis('off')

plt.tight_layout()
plt.savefig("../plots/summary_speedup_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print('✅ Summary Plot 2 saved: summary_speedup_analysis.png')

# 3. Summary: Parallel Efficiency
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Summary: OpenMP Parallel Efficiency', fontsize=16, fontweight='bold')

for idx, n in enumerate(matrix_sizes):
    ax = axes[idx // 3, idx % 3]
    
    omp_summary = summary_df[(summary_df['n'] == n) & (summary_df['framework'] == 'OpenMP')].sort_values('threads')
    seq_summary = summary_df[(summary_df['n'] == n) & (summary_df['framework'] == 'sequential')]
    
    if len(omp_summary) > 0 and len(seq_summary) > 0:
        seq_time = seq_summary['mean'].values[0]
        speedup = seq_time / omp_summary['mean']
        efficiency = (speedup / omp_summary['threads']) * 100
        
        bars = ax.bar(omp_summary['threads'].astype(str), efficiency, alpha=0.7, width=0.6, color='#2ca02c')
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
plt.savefig("../plots/summary_efficiency_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print('✅ Summary Plot 3 saved: summary_efficiency_analysis.png')

# 4. Summary: Time vs Matrix Size
fig, ax = plt.subplots(figsize=(12, 7))

# Sequential
seq_summary = summary_df[summary_df['framework'] == 'sequential'].sort_values('n')
ax.errorbar(seq_summary['n'], seq_summary['mean'], yerr=seq_summary['std'],
           marker='s', capsize=5, capthick=2, linewidth=2.5, markersize=10,
           label='Sequential', color='red', alpha=0.8)

# OpenMP
omp_summary = summary_df[summary_df['framework'] == 'OpenMP']
for threads in sorted(omp_summary['threads'].unique()):
    thread_data = omp_summary[omp_summary['threads'] == threads].sort_values('n')
    ax.errorbar(thread_data['n'], thread_data['mean'], yerr=thread_data['std'],
               marker='o', capsize=5, capthick=2, linewidth=2.5, markersize=10,
               label=f'OpenMP ({threads}T)', alpha=0.8)

ax.set_xlabel('Matrix Size (n)', fontsize=12)
ax.set_ylabel('Execution Time (seconds)', fontsize=12)
ax.set_title('Summary: Sequential vs OpenMP - Time vs Matrix Size', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=11, loc='upper left')
plt.tight_layout()
plt.savefig("../plots/summary_time_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print('✅ Summary Plot 4 saved: summary_time_comparison.png')

# 5. Summary: Coefficient of Variation (Stability)
fig, ax = plt.subplots(figsize=(12, 7))

# OpenMP
omp_summary = summary_df[summary_df['framework'] == 'OpenMP']
for threads in sorted(omp_summary['threads'].unique()):
    thread_data = omp_summary[omp_summary['threads'] == threads].sort_values('n')
    cv = (thread_data['std'] / thread_data['mean']) * 100
    ax.plot(thread_data['n'], cv.values, 'o-', linewidth=2.5, markersize=10, label=f'{threads} threads')

# Sequential
seq_summary = summary_df[summary_df['framework'] == 'sequential'].sort_values('n')
cv_seq = (seq_summary['std'] / seq_summary['mean']) * 100
ax.plot(seq_summary['n'], cv_seq.values, 's-', linewidth=2.5, markersize=10, label='Sequential', color='red')

ax.set_xlabel('Matrix Size (n)', fontsize=12)
ax.set_ylabel('Coefficient of Variation (%)', fontsize=12)
ax.set_title('Summary: Performance Stability - Variability Across Runs', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("../plots/summary_cv_stability.png", dpi=150, bbox_inches='tight')
plt.close()
print('✅ Summary Plot 5 saved: summary_cv_stability.png')

# Print summary statistics
print('\n' + '='*70)
print('📊 SUMMARY STATISTICS - Sequential vs OpenMP')
print('='*70)
print(f'Matrix sizes: {sorted(summary_df["n"].unique())}')
print(f'OpenMP thread counts: {sorted(summary_df[summary_df["framework"] == "OpenMP"]["threads"].unique())}')
print(f'\n{"Matrix Size":<15} {"Framework":<15} {"Threads":<10} {"Mean Time (s)":<20} {"Std Dev (s)":<15}')
print('-'*70)

for n in sorted(summary_df['n'].unique()):
    # Sequential
    seq_data = summary_df[(summary_df['n'] == n) & (summary_df['framework'] == 'sequential')]
    if len(seq_data) > 0:
        row = seq_data.iloc[0]
        print(f"{n:<15} {'Sequential':<15} {1:<10} {row['mean']:<20.6f} {row['std']:<15.6f}")
    
    # OpenMP
    omp_data = summary_df[(summary_df['n'] == n) & (summary_df['framework'] == 'OpenMP')].sort_values('threads')
    for _, row in omp_data.iterrows():
        print(f"{'':<15} {'OpenMP':<15} {int(row['threads']):<10} {row['mean']:<20.6f} {row['std']:<15.6f}")

print('='*70)
print('\n🎉 All summary plots generated successfully!')
