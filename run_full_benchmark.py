#!/usr/bin/env python3
import subprocess
import time
import json
import re
import os
import numpy as np
from PIL import Image

def compute_metrics(img_path, ref_path):
    img = np.array(Image.open(img_path).convert('RGB')).astype(np.float64)
    ref = np.array(Image.open(ref_path).convert('RGB')).astype(np.float64)

    diff = np.abs(img - ref)
    max_diff = int(np.max(diff))
    mean_abs_diff = float(np.mean(diff))
    mse = float(np.mean((img - ref)**2))
    psnr = float(20 * np.log10(255.0 / np.sqrt(mse))) if mse > 0 else float('inf')

    # SSIM
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    mu1 = np.mean(img)
    mu2 = np.mean(ref)
    sigma1_sq = np.var(img)
    sigma2_sq = np.var(ref)
    sigma12 = np.mean((img - mu1) * (ref - mu2))
    ssim = float(((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)))

    # Byte identity
    exact_match = (max_diff == 0)

    return {
        "exact_match": exact_match,
        "max_diff": max_diff,
        "mean_abs_diff": mean_abs_diff,
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim
    }

def get_peak_rss_kb(cmd):
    time_cmd = ["/usr/bin/time", "-v"] + cmd
    res = subprocess.run(time_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in res.stderr.splitlines():
        if "Maximum resident set size" in line:
            parts = line.split(":")
            return int(parts[1].strip())
    return 0

def measure_cli_latencies(cmd, runs=10):
    # Warmup
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return {
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
        "p50": float(np.median(times)),
        "p95": float(np.percentile(times, 95)),
        "runs": times
    }

def run_warm_benchmarks():
    results = {}

    # 1. Standalone C Engine
    print("Benchmarking Standalone C Engine (rife_v46)...")
    c_out = subprocess.run(
        ["./rife_v46", "flownet-v46.bin", "frame0.png", "frame1.png", "0.5", "/tmp/bench_c_out.png", "15"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
    ).stdout
    m = re.search(r"Warm Inference \(15 runs\): Mean = ([\d.]+) ms, Std = ([\d.]+) ms, Min = ([\d.]+) ms, Max = ([\d.]+) ms", c_out)
    if m:
        results["rife_v46"] = {
            "mean": float(m.group(1)),
            "std": float(m.group(2)),
            "min": float(m.group(3)),
            "max": float(m.group(4))
        }

    # 2. RIFE-NCNN CPU
    print("Benchmarking RIFE-NCNN CPU...")
    ncnn_out = subprocess.run(
        ["/home/user/repos/playground/bench_ncnn", "15", "4"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
    ).stdout
    m = re.search(r"RIFE-NCNN CPU \(4 threads, 15 runs\): Mean = ([\d.]+) ms, Std = ([\d.]+) ms, Min = ([\d.]+) ms, Max = ([\d.]+) ms", ncnn_out)
    if m:
        results["rife_ncnn"] = {
            "mean": float(m.group(1)),
            "std": float(m.group(2)),
            "min": float(m.group(3)),
            "max": float(m.group(4))
        }

    # 3. Practical-RIFE PyTorch CPU
    print("Benchmarking Practical-RIFE PyTorch CPU...")
    pt_out = subprocess.run(
        ["python3", "-c", """
import time, math, torch, numpy as np
from bench_pytorch import IFNet, warp
from PIL import Image

torch.set_num_threads(4)
torch.set_grad_enabled(False)
model = IFNet()
state_dict = torch.load('/tmp/flownet_v4.6.pkl', map_location='cpu')
clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(clean_state_dict)
model.eval()

im0 = np.array(Image.open('/home/user/repos/playground/frame0.png').convert('RGB')).astype(np.float32) / 255.0
im1 = np.array(Image.open('/home/user/repos/playground/frame1.png').convert('RGB')).astype(np.float32) / 255.0

t0 = torch.from_numpy(im0).permute(2, 0, 1).unsqueeze(0)
t1 = torch.from_numpy(im1).permute(2, 0, 1).unsqueeze(0)
H, W = t0.shape[2], t0.shape[3]
timestep = torch.full((1, 1, H, W), 0.5)

pw, ph = W, H
tenFlow_div = torch.tensor([(pw - 1.0) / 2.0, (ph - 1.0) / 2.0], dtype=torch.float)
tenHorizontal = torch.linspace(-1.0, 1.0, pw, dtype=torch.float).view(1, 1, 1, pw).expand(-1, -1, ph, -1)
tenVertical = torch.linspace(-1.0, 1.0, ph, dtype=torch.float).view(1, 1, ph, 1).expand(-1, -1, -1, pw)
backwarp_tenGrid = torch.cat([tenHorizontal, tenVertical], 1)

# Warmup 5 runs
for _ in range(5):
    _ = model(t0, t1, timestep, tenFlow_div, backwarp_tenGrid)

times = []
for _ in range(15):
    t_start = time.perf_counter()
    out = model(t0, t1, timestep, tenFlow_div, backwarp_tenGrid)
    t_end = time.perf_counter()
    times.append((t_end - t_start) * 1000.0)

print(f'PyTorch CPU: Mean = {np.mean(times):.2f} ms, Std = {np.std(times):.2f} ms, Min = {np.min(times):.2f} ms, Max = {np.max(times):.2f} ms')
out_np = (out.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)
Image.fromarray(out_np).save('/tmp/bench_pt_out.png')
"""],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
    ).stdout
    m = re.search(r"PyTorch CPU: Mean = ([\d.]+) ms, Std = ([\d.]+) ms, Min = ([\d.]+) ms, Max = ([\d.]+) ms", pt_out)
    if m:
        results["practical_rife"] = {
            "mean": float(m.group(1)),
            "std": float(m.group(2)),
            "min": float(m.group(3)),
            "max": float(m.group(4))
        }

    return results

def main():
    print("=" * 60)
    print("RIFE v4.6 CPU IMPLEMENTATION BENCHMARK SUITE")
    print("AMD Ryzen 7 3700X (4 vCPUs) | Linux x86_64")
    print("=" * 60)

    # 1. Warm Inference Latency
    warm_results = run_warm_benchmarks()

    # 2. Cold CLI Execution Latency
    print("\nMeasuring Cold CLI Execution Latencies (10 runs)...")
    cmd_c = ["./rife_v46", "flownet-v46.bin", "frame0.png", "frame1.png", "0.5", "/tmp/out_c.png"]
    cmd_ncnn = ["/home/user/repos/rife-ncnn-vulkan/build/rife-ncnn-vulkan", "-m", "/home/user/repos/rife-ncnn-vulkan/models/rife-v4.6", "-0", "frame0.png", "-1", "frame1.png", "-o", "/tmp/out_ncnn.png", "-g", "-1", "-j", "1:4:1"]
    cmd_pt = ["python3", "/home/user/repos/playground/inference_cli_pytorch.py", "frame0.png", "frame1.png", "0.5", "/tmp/out_pt.png"]

    cli_c = measure_cli_latencies(cmd_c, runs=10)
    cli_ncnn = measure_cli_latencies(cmd_ncnn, runs=10)
    cli_pt = measure_cli_latencies(cmd_pt, runs=10)

    # 3. Peak RSS
    print("\nMeasuring Memory Footprint (Peak RSS)...")
    rss_c = get_peak_rss_kb(cmd_c) / 1024.0
    rss_ncnn = get_peak_rss_kb(cmd_ncnn) / 1024.0
    rss_pt = get_peak_rss_kb(cmd_pt) / 1024.0

    # 4. Output Fidelity
    print("\nMeasuring Numerical Output Fidelity against ref_out.png...")
    ref_path = "/home/user/repos/playground/ref_out.png"
    fid_c = compute_metrics("/tmp/out_c.png", ref_path)
    fid_ncnn = compute_metrics("/tmp/out_ncnn.png", ref_path)
    fid_pt = compute_metrics("/tmp/out_pt.png", ref_path)

    # 5. Footprint
    size_c_kb = os.path.getsize("/home/user/repos/playground/rife_v46") / 1024.0
    size_ncnn_kb = os.path.getsize("/home/user/repos/rife-ncnn-vulkan/build/rife-ncnn-vulkan") / 1024.0
    # PyTorch package size
    import torch
    torch_dir = os.path.dirname(torch.__file__)
    du_out = subprocess.run(["du", "-sm", torch_dir], stdout=subprocess.PIPE, text=True).stdout
    size_pt_mb = float(du_out.split()[0])

    benchmark_summary = {
        "warm_inference": warm_results,
        "cold_cli": {
            "rife_v46": cli_c,
            "rife_ncnn": cli_ncnn,
            "practical_rife": cli_pt
        },
        "peak_rss_mb": {
            "rife_v46": rss_c,
            "rife_ncnn": rss_ncnn,
            "practical_rife": rss_pt
        },
        "fidelity": {
            "rife_v46": fid_c,
            "rife_ncnn": fid_ncnn,
            "practical_rife": fid_pt
        },
        "binary_size": {
            "rife_v46_kb": size_c_kb,
            "rife_ncnn_kb": size_ncnn_kb,
            "practical_rife_torch_mb": size_pt_mb
        }
    }

    with open("/home/user/repos/playground/benchmark_results.json", "w") as f:
        json.dump(benchmark_summary, f, indent=2)

    print("\nBenchmark successfully completed and written to benchmark_results.json!")

if __name__ == '__main__':
    main()
