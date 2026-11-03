"""2x frame interpolation of a PNG sequence with the OpenGL RIFE 4.6.
Usage: python run_gl_dir.py inputs/ outputs/
Output: in[0], mid(0,1), in[1], mid(1,2), ..., in[N-1]  -> 2N-1 frames, numbered %08d.png from 1."""
import argparse, glob, os, shutil, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import gl_rife

ap = argparse.ArgumentParser()
ap.add_argument("in_dir"); ap.add_argument("out_dir")
ap.add_argument("--threads", type=int, default=6)
args = ap.parse_args()

src = sorted(glob.glob(os.path.join(args.in_dir, "*.png")))
assert len(src) >= 2, "need at least 2 frames"
os.makedirs(args.out_dir, exist_ok=True)
w, h = Image.open(src[0]).size
name = lambda k: os.path.join(args.out_dir, f"{k + 1:08d}.png")
load = lambda p: np.asarray(Image.open(p).convert("RGB"))
def save(img, k): Image.fromarray(img).save(name(k), compress_level=1)

g = gl_rife.GLRife(h, w)
pool = ThreadPoolExecutor(args.threads)
pending = []
gpu_s = 0.0
t0 = time.perf_counter()
nxt = pool.submit(load, src[0])
for i in range(len(src) - 1):
  a = nxt.result() if i == 0 else b
  b_f = pool.submit(load, src[i + 1])
  pending.append(pool.submit(shutil.copyfile, src[i], name(2 * i)))
  b = b_f.result()
  assert a.shape == b.shape, f"size mismatch at {src[i + 1]}"
  tg = time.perf_counter()
  mid = g(a, b, 0.5)
  gpu_s += time.perf_counter() - tg
  pending.append(pool.submit(save, mid, 2 * i + 1))
  if (i + 1) % 50 == 0:
    el = time.perf_counter() - t0
    print(f"  {i + 1}/{len(src) - 1} interpolated, {el:.1f}s elapsed, {(i + 1) / el:.2f} interp/s", flush=True)
pending.append(pool.submit(shutil.copyfile, src[-1], name(2 * (len(src) - 1))))
for p in pending: p.result()
el = time.perf_counter() - t0
n = len(src) - 1
print(f"done: {len(src)} -> {2 * len(src) - 1} frames ({w}x{h}) in {el:.1f}s | "
      f"{n / el:.2f} interp/s overall, RIFE calls {gpu_s / n * 1e3:.0f} ms avg (upload+GPU+download)")
