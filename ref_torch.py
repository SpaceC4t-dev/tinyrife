"""Independent PyTorch reference of the rife-v4.6 graph (stock torch ops), used to validate the OpenGL port."""
import os, torch, torch.nn.functional as F
from ncnn_weights import load_ncnn
from gl_rife import DEFAULT_MODEL, SCALES

def tw(a): return torch.from_numpy(a)

def interp(x, f): return x if f == 1 else F.interpolate(x, scale_factor=f, mode="bilinear", align_corners=False)

def warp(img, flow):
  N, _, H, W = img.shape
  gx = torch.linspace(-1, 1, W).view(1, 1, 1, W).expand(N, -1, H, -1)
  gy = torch.linspace(-1, 1, H).view(1, 1, H, 1).expand(N, -1, -1, W)
  grid = torch.cat([gx, gy], 1)
  f = torch.cat([flow[:, 0:1] / ((W - 1) / 2), flow[:, 1:2] / ((H - 1) / 2)], 1)
  g = (grid + f).permute(0, 2, 3, 1)
  return F.grid_sample(img, g, mode="bilinear", padding_mode="border", align_corners=True)

class RefRIFE46:
  def __init__(self, model_dir=DEFAULT_MODEL):
    self.L = load_ncnn(os.path.join(model_dir, "flownet.param"), os.path.join(model_dir, "flownet.bin"))

  def block(self, i, x):
    L = self.L[i*11:(i+1)*11]
    for l in L[0:2]: x = F.leaky_relu(F.conv2d(x, tw(l["weight"]), tw(l["bias"]), stride=2, padding=1), 0.2)
    for l in L[2:10]: x = F.leaky_relu(F.conv2d(x, tw(l["weight"]), tw(l["bias"]), padding=1) + x, 0.2)
    d = L[10]
    return F.pixel_shuffle(F.conv_transpose2d(x, tw(d["weight"]).transpose(0, 1), tw(d["bias"]), stride=2, padding=1), 2)

  @torch.no_grad()
  def __call__(self, img0, img1, t: float, return_flow=False):
    N, _, H, W = img0.shape
    tmap = torch.full((N, 1, H, W), t)
    flow = mask = None
    for i, s in enumerate(SCALES):
      if flow is None: x = interp(torch.cat([img0, img1, tmap], 1), 1 / s)
      else:
        x = torch.cat([warp(img0, flow[:, :2]), warp(img1, flow[:, 2:4]), tmap, mask], 1)
        x = torch.cat([interp(x, 1 / s), interp(flow, 1 / s) / s], 1)
      tmp = interp(self.block(i, x), s)
      flow = tmp[:, :4] * s if flow is None else flow + tmp[:, :4] * s
      mask = tmp[:, 4:5] if mask is None else mask + tmp[:, 4:5]
    m = torch.sigmoid(mask)
    out = warp(img0, flow[:, :2]) * m + warp(img1, flow[:, 2:4]) * (1 - m)
    return (out, flow) if return_flow else out
