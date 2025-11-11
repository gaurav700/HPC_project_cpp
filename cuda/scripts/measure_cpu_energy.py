#!/usr/bin/env python3
"""
measure_cpu_energy.py
Simple CPU energy sampling helper using RAPL (Running Average Power Limit).

Supports:
  - Reading from Linux perf event interface (preferred)
  - Reading from sysfs /sys/class/powercap/intel-rapl (fallback)

Note: Requires:
  - Linux kernel >= 3.13
  - RAPL module loaded (usually automatic)
  - Permissions to read MSRs or powercap interface (may require CAP_SYS_ADMIN or group membership)

Usage:
    python3 measure_cpu_energy.py --duration 5 --interval 0.05 [--socket 0]

Outputs energy reading for specified socket to stdout.
"""
import time
import argparse
import subprocess
import os
import sys

def read_energy_via_perf():
    """Try reading RAPL via perf stat (requires perf tool)"""
    try:
        # perf stat reports RAPL energy as 'power/energy-pkg/' or similar
        cmd = [
            'perf', 'stat', '-e', 'power/energy-pkg/',
            '-e', 'power/energy-ram/',
            'sleep', '0.1'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        return result.stdout + result.stderr
    except Exception:
        return None

def read_energy_via_powercap(socket_idx=0):
    """Read RAPL energy from sysfs /sys/class/powercap/intel-rapl"""
    try:
        # Try to find energy reading
        rapl_path = f'/sys/class/powercap/intel-rapl/intel-rapl:{socket_idx}/energy_uj'
        if os.path.exists(rapl_path):
            with open(rapl_path, 'r') as f:
                return float(f.read().strip()) / 1e6  # convert microjoules to joules
    except Exception:
        pass
    return None

def sample_energy_via_powercap(duration=5.0, interval=0.1, socket_idx=0):
    """Sample CPU energy readings via powercap over duration"""
    samples = []
    prev_energy = read_energy_via_powercap(socket_idx)
    if prev_energy is None:
        return None
    
    start = time.time()
    prev_time = start
    
    while time.time() - start < duration:
        time.sleep(interval)
        curr_time = time.time()
        curr_energy = read_energy_via_powercap(socket_idx)
        
        if curr_energy is not None:
            # Calculate power: (energy_diff / time_diff) = watts
            energy_diff = curr_energy - prev_energy
            time_diff = curr_time - prev_time
            if time_diff > 0 and energy_diff > 0:
                power = energy_diff / time_diff
                samples.append(power)
            prev_energy = curr_energy
            prev_time = curr_time
    
    return samples

def main():
    parser = argparse.ArgumentParser(description='Sample CPU energy using RAPL')
    parser.add_argument('--duration', type=float, default=5.0, help='seconds to sample')
    parser.add_argument('--interval', type=float, default=0.1, help='seconds between samples')
    parser.add_argument('--socket', type=int, default=0, help='RAPL socket index')
    args = parser.parse_args()
    
    print(f"📊 Sampling CPU energy (socket {args.socket}) for {args.duration}s...")
    
    samples = sample_energy_via_powercap(args.duration, args.interval, args.socket)
    
    if samples is None or len(samples) == 0:
        print("⚠️  Could not read RAPL energy. Requirements:")
        print("   - Linux kernel >= 3.13 with RAPL support")
        print("   - Permissions to read /sys/class/powercap/intel-rapl")
        print("   - Try: 'grep -r . /sys/class/powercap/intel-rapl/energy_uj'")
        sys.exit(1)
    
    try:
        import statistics
        print(f"samples={len(samples)} interval={args.interval:.3f}s duration={args.duration:.3f}s")
        print(f"avg={statistics.mean(samples):.3f} W min={min(samples):.3f} W max={max(samples):.3f} W stdev={statistics.pstdev(samples):.3f} W")
    except Exception as e:
        print(f"Warning: Could not compute statistics: {e}")
        print(f"Raw samples: {samples}")

if __name__ == '__main__':
    main()
