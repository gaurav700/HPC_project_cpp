#!/usr/bin/env python3
"""
measure_gpu_power.py
Simple GPU power sampling helper.
- Tries to use pynvml if available (recommended).
- Falls back to polling `nvidia-smi --query-gpu=power.draw`.

Usage:
    python3 measure_gpu_power.py --duration 5 --interval 0.05

Outputs average, min, max power (Watts) to stdout.
"""
import time
import argparse
import subprocess

try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlShutdown
    HAVE_PYNVML = True
except Exception:
    HAVE_PYNVML = False

parser = argparse.ArgumentParser()
parser.add_argument('--duration', type=float, default=5.0, help='seconds to sample')
parser.add_argument('--interval', type=float, default=0.1, help='seconds between samples')
parser.add_argument('--gpu', type=int, default=0, help='GPU index')
args = parser.parse_args()

samples = []
start = time.time()
if HAVE_PYNVML:
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(args.gpu)
        while time.time() - start < args.duration:
            # power in milliwatts
            mw = nvmlDeviceGetPowerUsage(handle)
            samples.append(mw / 1000.0)
            time.sleep(args.interval)
    finally:
        nvmlShutdown()
else:
    # fallback: use nvidia-smi
    cmd = ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits', '-i', str(args.gpu)]
    while time.time() - start < args.duration:
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
            samples.append(float(out))
        except Exception:
            # couldn't sample; append 0
            samples.append(0.0)
        time.sleep(args.interval)

if len(samples) == 0:
    print('No samples collected')
else:
    import statistics
    print('samples=%d interval=%.3fs duration=%.3fs' % (len(samples), args.interval, args.duration))
    print('avg=%.3f W min=%.3f W max=%.3f W stdev=%.3f W' % (statistics.mean(samples), min(samples), max(samples), statistics.pstdev(samples)))
