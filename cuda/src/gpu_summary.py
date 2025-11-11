#!/usr/bin/env python3
"""
GPU Benchmark Summary Statistics Generator
Aggregates gpu_repeats.csv with mean, std, and count statistics
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def generate_summary():
    """Generate summary statistics from GPU repeats data"""
    
    # Use absolute paths based on script location
    script_dir = Path(__file__).parent
    cuda_root = script_dir.parent
    repeats_file = cuda_root / "results" / "gpu_repeats.csv"
    summary_file = cuda_root / "results" / "gpu_summary.csv"
    
    if not repeats_file.exists():
        print(f"❌ Error: {repeats_file} not found")
        print(f"   Run: bash {cuda_root}/scripts/run_gpu_tests.sh")
        sys.exit(1)
    
    try:
        # Read raw benchmark data
        df = pd.read_csv(repeats_file)
        
        # Aggregate by implementation and matrix size
        summary = df.groupby(['impl', 'n']).agg({
            'kernel_ms': ['mean', 'std', 'count']
        }).round(6)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()
        
        # Rename for clarity
        summary.rename(columns={
            'kernel_ms_mean': 'kernel_mean_ms',
            'kernel_ms_std': 'kernel_std_ms',
            'kernel_ms_count': 'runs'
        }, inplace=True)
        
        # Calculate coefficient of variation (std/mean * 100)
        summary['kernel_cv_%'] = (summary['kernel_std_ms'] / summary['kernel_mean_ms'] * 100).round(2)
        
        # Save summary
        summary.to_csv(summary_file, index=False)
        
        # Print formatted output
        print("\n" + "="*100)
        print("📊 GPU BENCHMARK SUMMARY")
        print("="*100)
        print(f"Total runs: {len(df)}")
        print(f"Matrix sizes: {sorted(df['n'].unique())}")
        print(f"Implementations: {sorted(df['impl'].unique())}")
        print("="*100 + "\n")
        
        # Display summary table
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(summary.to_string(index=False))
        
        print("\n" + "="*100)
        print(f"✅ Summary saved to: {summary_file}")
        print("="*100 + "\n")
        
        # Additional statistics (with NaN handling)
        print("📈 DETAILED STATISTICS BY IMPLEMENTATION:\n")
        for impl in sorted(df['impl'].unique()):
            impl_data = summary[summary['impl'] == impl]
            print(f"\n{impl}:")
            print(f"  Matrix sizes tested: {impl_data['n'].tolist()}")
            print(f"  Kernel time range: {impl_data['kernel_mean_ms'].min():.6f}ms - {impl_data['kernel_mean_ms'].max():.6f}ms")
            
            # Handle NaN values in CV
            cv_values = impl_data['kernel_cv_%'].dropna()
            if len(cv_values) > 0:
                print(f"  Avg variability (CV): {cv_values.mean():.2f}%")
                if len(cv_values) > 0:
                    min_cv_idx = cv_values.idxmin()
                    max_cv_idx = cv_values.idxmax()
                    print(f"  Most stable run (lowest CV): n={impl_data.loc[min_cv_idx, 'n']}, CV={cv_values.min():.2f}%")
                    print(f"  Least stable run (highest CV): n={impl_data.loc[max_cv_idx, 'n']}, CV={cv_values.max():.2f}%")
            else:
                print(f"  Avg variability (CV): N/A (no valid data)")
        
        return summary
        
    except Exception as e:
        print(f"⚠️  Warning: Error in additional statistics: {e}")
        print(f"✅ Summary CSV still saved to: {summary_file}")
        sys.exit(0)

if __name__ == "__main__":
    generate_summary()
