"""Validate the OpenGL implementation against the torch reference."""
import numpy as np, torch
from PIL import Image
import gl_rife, ref_torch

IMG = "/usr/share/plasma/avatars/photos/Owl.png"  # any photo >= 480x420 works; pass a path as argv[1]

def psnr(a, b): return 10 * np.log10(1 / np.mean((a - b) ** 2))

def main(img=IMG, w=448, h=384, dx=12, dy=6):
  im = np.asarray(Image.open(img).convert("RGB"))
  crop = lambda ox, oy: im[16+oy:16+oy+h, 16+ox:16+ox+w]
  a, gt, b = crop(0, 0), crop(dx//2, dy//2), crop(dx, dy)
  g = gl_rife.GLRife(h, w)
  out = g(a, b, 0.5)
  flow_gl = np.frombuffer(g.flow.read(), np.float32).reshape(g.H, g.W, 4).transpose(2, 0, 1)
  tt = lambda x: torch.from_numpy(x.transpose(2, 0, 1)[None].astype(np.float32) / 255)
  ref, flow_ref = ref_torch.RefRIFE46()(tt(a), tt(b), 0.5, return_flow=True)
  ref_u8 = (ref[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).round().astype(np.uint8)
  ferr = np.abs(flow_gl - flow_ref[0].numpy()).max()
  d = np.abs(out.astype(int) - ref_u8.astype(int))
  print(f"flow max|diff| vs torch: {ferr:.2e}")
  print(f"output uint8 diff vs torch: max {d.max()}, pixels differing {np.mean(d > 0)*100:.3f}%")
  inner = (slice(16, -16), slice(16, -16))
  f = lambda x: x.astype(np.float32) / 255
  print(f"PSNR vs ground truth: GL {psnr(f(out)[inner], f(gt)[inner]):.2f} dB, torch {psnr(f(ref_u8)[inner], f(gt)[inner]):.2f} dB")
  assert ferr < 1e-2 and d.max() <= 1, "mismatch"
  print("PASS")

if __name__ == "__main__":
  import sys
  main(*sys.argv[1:2])
