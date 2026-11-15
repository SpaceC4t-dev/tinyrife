"""Output correctness: GL and ncnn-vulkan vs the fp32 torch reference and vs ground truth.
Frames are translated crops of a photo, so the true middle frame is known exactly."""
import os, subprocess, sys, tempfile, numpy as np, torch
from PIL import Image
import gl_rife, ref_torch
from bench_ncnn import BIN_DIR

PHOTO_S = "/usr/share/plasma/avatars/photos/Owl.png"
PHOTO_L = "mosaic"  # 4x3 grid of the 512x512 plasma avatar photos -> 2048x1536 natural-image texture
CASES = [  # (photo, w, h, dx, dy)
  (PHOTO_S, 448, 384, 12, 6),
  (PHOTO_L, 1280, 720, 24, 10),
  (PHOTO_L, 1280, 720, 64, 0),
  (PHOTO_L, 1280, 720, 120, 40),
]

def load(photo):
  if photo != "mosaic": return np.asarray(Image.open(photo).convert("RGB"))
  import glob
  tiles = [np.asarray(Image.open(p).convert("RGB").resize((512, 512))) for p in sorted(glob.glob("/usr/share/plasma/avatars/photos/*.png"))[:12]]
  return np.concatenate([np.concatenate(tiles[r*4:(r+1)*4], 1) for r in range(3)], 0)

def frames(photo, w, h, dx, dy):
  im = load(photo)
  ox, oy = 8, 8
  crop = lambda x, y: im[oy+y:oy+y+h, ox+x:ox+x+w].copy()
  return crop(0, 0), crop(dx // 2, dy // 2), crop(dx, dy)

def torch_ref(a, b):
  tt = lambda x: torch.from_numpy(x.transpose(2, 0, 1)[None].astype(np.float32) / 255)
  h, w = a.shape[:2]
  ph, pw = (h + 31) // 32 * 32, (w + 31) // 32 * 32
  pad = lambda x: torch.nn.functional.pad(tt(x), (0, pw - w, 0, ph - h))  # zero pad, like ncnn / GL
  r = ref_torch.RefRIFE46()(pad(a), pad(b), 0.5)[:, :, :h, :w]
  return (r[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).round().astype(np.uint8)

def ncnn(a, b):
  with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
    pa, pb, po = (os.path.join(d, n) for n in ("a.png", "b.png", "o.png"))
    Image.fromarray(a).save(pa); Image.fromarray(b).save(pb)
    subprocess.run([os.path.join(BIN_DIR, "rife-ncnn-vulkan"), "-0", pa, "-1", pb, "-o", po,
                    "-m", os.path.join(BIN_DIR, "rife-v4.6"), "-g", "0"], check=True, capture_output=True)
    return np.asarray(Image.open(po).convert("RGB"))

def psnr(x, y, m=16):
  x, y = x[m:-m, m:-m].astype(np.float64), y[m:-m, m:-m].astype(np.float64)
  return 10 * np.log10(255 ** 2 / np.mean((x - y) ** 2))

def stats(name, out, ref, gt):
  d = np.abs(out.astype(int) - ref.astype(int))
  print(f"  {name:10s} vs torch: max {d.max():3d}  mean {d.mean():.4f}  >1: {np.mean(d > 1)*100:6.3f}%  >8: {np.mean(d > 8)*100:6.3f}%"
        f"  | PSNR vs GT {psnr(out, gt):.2f} dB")

if __name__ == "__main__":
  for photo, w, h, dx, dy in CASES:
    a, gt, b = frames(photo, w, h, dx, dy)
    print(f"{os.path.basename(photo)} {w}x{h}, motion ({dx},{dy}) px; blend baseline PSNR {psnr(((a.astype(int)+b)//2).astype(np.uint8), gt):.2f} dB")
    ref = torch_ref(a, b)
    print(f"  {'torch fp32':10s} (reference)                                          | PSNR vs GT {psnr(ref, gt):.2f} dB")
    stats("GL fp32", gl_rife.GLRife(h, w)(a, b, 0.5), ref, gt)
    stats("ncnn-vk", ncnn(a, b), ref, gt)
