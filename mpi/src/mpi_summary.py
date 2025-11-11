#!/usr/bin/env python3
"""
MPI Benchmark Summary Statistics Generator
Aggregates mpi_repeats.csv with mean, std, and count statistics
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def generate_summary():
    """Generate summary statistics from MPI repeats data"""
    
    # Use absolute paths based on script location
    script_dir = Path(__file__).parent
    mpi_root = script_dir.parent
    repeats_file = mpi_root / "results" / "mpi_repeats.csv"
    summary_file = mpi_root / "results" / "mpi_summary.csv"
    
    if not repeats_file.exists():
        print(f"❌ Error: {repeats_file} not found")
        print(f"   Run: bash {mpi_root}/scripts/run_mpi_tests.sh")
        sys.exit(1)
    
    try:
        # Read raw benchmark data
        df = pd.read_csv(repeats_file)
        
        # Aggregate by implementation, matrix size, and process count
        summary = df.groupby(['impl', 'n', 'processes']).agg({
            'time_s': ['mean', 'std', 'count']
        }).round(6)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()
        
        # Rename for clarity
        summary.rename(columns={
            'time_s_mean': 'time_mean_s',
            'time_s_std': 'time_std_s',
            'time_s_count': 'runs'
        }, inplace=True)
        
        # Calculate coefficient of variation (std/mean * 100)
        summary['time_cv_%'] = (summary['time_std_s'] / summary['time_mean_s'] * 100).round(2)
        
        # Save summary
        summary.to_csv(summary_file, index=False)
        
        # Print formatted output
        print("\n" + "="*100)
        print("📊 MPI BENCHMARK SUMMARY")
        print("="*100)
        print(f"Total runs: {len(df)}")
        print(f"Matrix sizes: {sorted(df['n'].unique())}")
        print(f"Implementations: {sorted(df['impl'].unique())}")
        print(f"Process counts: {sorted(df['processes'].unique())}")
        print("="*100 + "\n")
        
        # Display summary table
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(summary.to_string(index=False))
        
        print("\n" + "="*100)
        print(f"✅ Summary saved to: {summary_file}")
        print("="*100 + "\n")
        
        # Additional statistics by implementation
        print("📈 DETAILED STATISTICS BY IMPLEMENTATION:\n")
        for impl in sorted(df['impl'].unique()):
            impl_data = summary[summary['impl'] == impl]
            print(f"\n{impl}:")
            print(f"  Matrix sizes tested: {impl_data['n'].tolist()}")
            print(f"  Process counts: {impl_data['processes'].tolist()}")
            print(f"  Execution time range: {impl_data['time_mean_s'].min():.6f}s - {impl_data['time_mean_s'].max():.6f}s")
            
            # Handle NaN values in CV
            cv_values = impl_data['time_cv_%'].dropna()
            if len(cv_values) > 0:
                print(f"  Avg variability (CV): {cv_values.mean():.2f}%")
                if len(cv_values) > 0:
                    min_cv_idx = cv_values.idxmin()
                    max_cv_idx = cv_values.idxmax()
                    print(f"  Most stable run (lowest CV): n={impl_data.loc[min_cv_idx, 'n']}, CV={cv_values.min():.2f}%")
                    print(f"  Least stable run (highest CV): n={impl_data.loc[max_cv_idx, 'n']}, CV={cv_values.max():.2f}%")
        
        return summary
        
    except Exception as e:
        print(f"⚠️  Error generating summary: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    generate_summary()
