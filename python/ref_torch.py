"""Independent PyTorch reference of RIFE v4.6, built from stock torch ops. Optional: nothing in the C build uses it.

  python python/ref_torch.py a.png b.png ref.png [-t 0.5] [-m model_dir] [-b N] [--threads N]
  build/pngdiff ref.png out.png 1     # a C backend's output must be within 1 uint8 level of the reference

-b N also times N runs (the PyTorch baseline of the CPU backend). Needs torch, numpy and pillow.
"""
import argparse, os, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(ROOT, "ext/rife-ncnn-vulkan/models/rife-v4.6")   # from fetch_model.sh
SCALES, WIDTH, CIN = (8, 4, 2, 1), (192, 128, 96, 64), (7, 12, 12, 12)       # the four IFBlocks

def load(model_dir):
  """flownet.bin in export order, per IFBlock: 2 stride-2 convs, 8 residual convs, 1 deconv, as (weight, bias)."""
  blob, off, blocks = open(os.path.join(model_dir, "flownet.bin"), "rb").read(), 0, []
  for c, cin in zip(WIDTH, CIN):
    layers = []
    for o, i, k in [(c // 2, cin, 3), (c, c // 2, 3)] + [(c, c, 3)] * 8 + [(24, c, 4)]:
      n = o * i * k * k
      assert blob[off:off + 4] == b"\x47\x6b\x30\x01", "expected fp16 weights"
      w = np.frombuffer(blob, np.float16, n, off + 4).astype(np.float32).reshape(o, i, k, k)
      off += 4 + (n * 2 + 3) // 4 * 4
      layers.append((torch.from_numpy(w), torch.from_numpy(np.frombuffer(blob, np.float32, o, off).copy())))
      off += o * 4
    blocks.append(layers)
  assert off == len(blob), "not a RIFE v4.6 flownet.bin"
  return blocks

def interp(x, f): return x if f == 1 else F.interpolate(x, scale_factor=f, mode="bilinear", align_corners=False)

def warp(img, flow):
  N, _, H, W = img.shape
  gx = torch.linspace(-1, 1, W).view(1, 1, 1, W).expand(N, -1, H, -1)
  gy = torch.linspace(-1, 1, H).view(1, 1, H, 1).expand(N, -1, -1, W)
  f = torch.cat([flow[:, 0:1] / ((W - 1) / 2), flow[:, 1:2] / ((H - 1) / 2)], 1)
  g = (torch.cat([gx, gy], 1) + f).permute(0, 2, 3, 1)
  return F.grid_sample(img, g, mode="bilinear", padding_mode="border", align_corners=True)

def block(L, x):
  for w, b in L[0:2]: x = F.leaky_relu(F.conv2d(x, w, b, stride=2, padding=1), 0.2)
  for w, b in L[2:10]: x = F.leaky_relu(F.conv2d(x, w, b, padding=1) + x, 0.2)
  w, b = L[10]
  return F.pixel_shuffle(F.conv_transpose2d(x, w.transpose(0, 1), b, stride=2, padding=1), 2)

@torch.inference_mode()
def rife(blocks, img0, img1, t):
  """img0, img1: 1x3xHxW in [0, 1], H and W multiples of 32."""
  tmap = torch.full_like(img0[:, :1], t)
  flow = mask = None
  for L, s in zip(blocks, SCALES):
    if flow is None: x = interp(torch.cat([img0, img1, tmap], 1), 1 / s)
    else:
      x = torch.cat([warp(img0, flow[:, :2]), warp(img1, flow[:, 2:4]), tmap, mask], 1)
      x = torch.cat([interp(x, 1 / s), interp(flow, 1 / s) / s], 1)
    tmp = interp(block(L, x), s)
    flow = tmp[:, :4] * s if flow is None else flow + tmp[:, :4] * s
    mask = tmp[:, 4:5] if mask is None else mask + tmp[:, 4:5]
  m = torch.sigmoid(mask)
  return warp(img0, flow[:, :2]) * m + warp(img1, flow[:, 2:4]) * (1 - m)

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("out")
  ap.add_argument("-t", type=float, default=0.5); ap.add_argument("-m", default=DEFAULT_MODEL)
  ap.add_argument("-b", dest="bench", type=int, default=0); ap.add_argument("--threads", type=int)
  args = ap.parse_args()
  if args.threads: torch.set_num_threads(args.threads)
  a, b = (np.array(Image.open(p).convert("RGB")) for p in (args.a, args.b))
  h, w = a.shape[:2]
  pad = lambda im: F.pad(torch.from_numpy(im).permute(2, 0, 1)[None] / 255, (0, -w % 32, 0, -h % 32))   # zeros, like the C code
  blocks, x0, x1 = load(args.m), pad(a), pad(b)
  out = rife(blocks, x0, x1, args.t)
  Image.fromarray((out[0, :, :h, :w].permute(1, 2, 0) * 255).round().clamp(0, 255).byte().numpy()).save(args.out)
  if args.bench:
    ms = []
    for _ in range(args.bench):
      t0 = time.perf_counter(); rife(blocks, x0, x1, args.t); ms.append((time.perf_counter() - t0) * 1e3)
    print(f"torch {torch.__version__}, {torch.get_num_threads()} threads, {w}x{h}: median {np.median(ms):.1f} ms, min {min(ms):.1f} ms")
