#!/usr/bin/env python3
"""
benchmark_with_power.py
Wrapper to run a benchmark command while sampling GPU power and CPU energy.

This is a simple reference implementation. For production use, consider:
  - In-kernel measurement (lower latency/better precision)
  - Synchronized sampling with kernel launch events
  - Low-noise isolated runs with thermal stability

Usage:
    python3 benchmark_with_power.py --command "mpirun -np 4 ./src/mpi_matmul 1000" --duration 10

Outputs:
    - avg/min/max power (Watts) for GPU
    - avg/min/max power (Watts) for CPU
    - command output and exit code
"""
import argparse
import subprocess
import threading
import time
import sys

def sample_gpu_power(duration, interval=0.1):
    """Sample GPU power in background thread"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        samples = []
        start = time.time()
        while time.time() - start < duration:
            mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            samples.append(mw / 1000.0)
            time.sleep(interval)
        pynvml.nvmlShutdown()
        return samples
    except Exception as e:
        print(f"⚠️  GPU power sampling failed: {e}", file=sys.stderr)
        return []

def sample_cpu_energy(duration, interval=0.1, socket_idx=0):
    """Sample CPU energy in background thread via powercap"""
    import os
    samples = []
    prev_energy = None
    prev_time = None
    
    try:
        rapl_path = f'/sys/class/powercap/intel-rapl/intel-rapl:{socket_idx}/energy_uj'
        if not os.path.exists(rapl_path):
            return []
        
        with open(rapl_path, 'r') as f:
            prev_energy = float(f.read().strip()) / 1e6
        
        start = time.time()
        prev_time = start
        
        while time.time() - start < duration:
            time.sleep(interval)
            curr_time = time.time()
            with open(rapl_path, 'r') as f:
                curr_energy = float(f.read().strip()) / 1e6
            
            energy_diff = curr_energy - prev_energy
            time_diff = curr_time - prev_time
            if time_diff > 0 and energy_diff >= 0:
                power = energy_diff / time_diff
                samples.append(power)
            
            prev_energy = curr_energy
            prev_time = curr_time
        
        return samples
    except Exception as e:
        print(f"⚠️  CPU energy sampling failed: {e}", file=sys.stderr)
        return []

def main():
    parser = argparse.ArgumentParser(description='Run benchmark with power monitoring')
    parser.add_argument('--command', type=str, required=True, help='Command to execute')
    parser.add_argument('--duration', type=float, default=10.0, help='Max sampling duration (s)')
    parser.add_argument('--interval', type=float, default=0.1, help='Sample interval (s)')
    args = parser.parse_args()
    
    print(f"🔥 Running: {args.command}")
    print(f"   Sampling for up to {args.duration}s...\n")
    
    # Start power sampling threads
    gpu_samples = []
    cpu_samples = []
    
    gpu_thread = threading.Thread(target=lambda: gpu_samples.extend(
        sample_gpu_power(args.duration + 1, args.interval)
    ), daemon=True)
    cpu_thread = threading.Thread(target=lambda: cpu_samples.extend(
        sample_cpu_energy(args.duration + 1, args.interval)
    ), daemon=True)
    
    gpu_thread.start()
    cpu_thread.start()
    
    # Run the benchmark
    start_time = time.time()
    result = subprocess.run(args.command, shell=True)
    elapsed = time.time() - start_time
    
    # Wait for sampling threads to finish
    gpu_thread.join(timeout=5)
    cpu_thread.join(timeout=5)
    
    print(f"\n" + "="*60)
    print(f"⏱️  Elapsed time: {elapsed:.3f}s")
    print(f"Exit code: {result.returncode}")
    print("="*60)
    
    if gpu_samples:
        import statistics
        print(f"\n📊 GPU Power (Watts):")
        print(f"   avg={statistics.mean(gpu_samples):.3f}, min={min(gpu_samples):.3f}, max={max(gpu_samples):.3f}")
    
    if cpu_samples:
        import statistics
        print(f"\n⚡ CPU Energy (Watts):")
        print(f"   avg={statistics.mean(cpu_samples):.3f}, min={min(cpu_samples):.3f}, max={max(cpu_samples):.3f}")
    
    sys.exit(result.returncode)

if __name__ == '__main__':
    main()
