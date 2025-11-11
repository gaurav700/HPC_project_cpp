python3 << 'PY'
import pandas as pd

df = pd.read_csv("results/sequential_openmp_repeats.csv")

# Create summary by framework, n, and threads
summary = df.groupby(['framework', 'n', 'threads']).agg({
    'time_s': ['mean', 'std'],
    'checksum': 'count'
}).reset_index()

summary.columns = ['framework', 'n', 'threads', 'mean', 'std', 'count']

# Save summary
summary.to_csv("results/day1_sequential_openmp_summary.csv", index=False)

print("📊 Sequential & OpenMP Summary Statistics")
print("=" * 80)
print(f"{'Framework':<15} {'Matrix Size':<15} {'Threads':<12} {'Mean (s)':<15} {'Std (s)':<15} {'Runs':<10}")
print("-" * 80)

for _, row in summary.iterrows():
    print(f"{row['framework']:<15} {row['n']:<15} {int(row['threads']):<12} {row['mean']:<15.6f} {row['std']:<15.6f} {int(row['count']):<10}")

print("=" * 80)
print("\n✅ Summary saved to: results/day1_sequential_openmp_summary.csv")
PY