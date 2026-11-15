"""Benchmark rife-ncnn-vulkan (rife-v4.6) per-interpolation time, excluding startup.
Runs directory mode with two sequence lengths and takes the slope per actual model call."""
import os, re, subprocess, sys, time, tempfile, numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN_DIR = os.path.join(ROOT, "ext/ncnn-bin/rife-ncnn-vulkan-20221029-ubuntu")
SRC = "/usr/share/wallpapers/openSUSEdefault/contents/images/1920x1200.png"

def make_frames(d, n, w, h):
  im = Image.open(SRC).convert("RGB").resize((w + 4 * n, h + 2 * n))
  for i in range(n): im.crop((4 * i, 2 * i, 4 * i + w, 2 * i + h)).save(os.path.join(d, f"{i:04d}.png"), compress_level=1)

def run(n, w, h, extra):
  with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
    i, o = os.path.join(d, "in"), os.path.join(d, "out"); os.makedirs(i); os.makedirs(o)
    make_frames(i, n, w, h)
    cmd = [os.path.join(BIN_DIR, "rife-ncnn-vulkan"), "-i", i, "-o", o, "-m", os.path.join(BIN_DIR, "rife-v4.6"), "-g", "0", "-v", *extra]
    t0 = time.perf_counter(); p = subprocess.run(cmd, capture_output=True, text=True); dt = time.perf_counter() - t0
    if p.returncode: sys.exit(p.stderr[-2000:])
    calls = sum(1 for ts in re.findall(r" ([0-9.]+) -> ", p.stderr) if 0 < float(ts) < 1)
    return dt, calls, p.stderr

if __name__ == "__main__":
  w, h = map(int, sys.argv[1].split("x"))
  extra = sys.argv[2:]   # e.g. -j 1:2:2
  n1, n2, reps = (int(os.environ.get(k, d)) for k, d in (("N1", 4), ("N2", 16), ("REPS", 3)))
  run(2, w, h, extra)    # warm the driver shader cache
  pers = []
  for _ in range(reps):
    t1, c1, _ = run(n1, w, h, extra)
    t2, c2, log = run(n2, w, h, extra)
    pers.append((t2 - t1) / (c2 - c1))
  per = float(np.median(pers))
  print(f"ncnn-vulkan {w}x{h} {' '.join(extra)}: {per*1e3:.1f} ms/interp ({1/per:.2f} fps)  "
        f"[median of {reps}: {', '.join(f'{x*1e3:.1f}' for x in pers)} ms; {c1}->{c2} calls]")
