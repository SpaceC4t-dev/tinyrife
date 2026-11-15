"""RIFE v4.6 with OpenGL 4.3 core compute shaders, fp32 only, no extensions.

Tensor layout ("pack4"): SSBO of vec4, index ((c4 * H + y) * W + x), 4 channels per vec4.
Images: SSBO of uint (RGBA8, unpackUnorm4x8), zero padded to a multiple of 32.
"""
from __future__ import annotations
import os, numpy as np, moderngl
from ncnn_weights import load_ncnn, DEFAULT_MODEL, SCALES, ROOT

HEADER = "#version 430\n"

# ---------------------------------------------------------------- shaders
# The GLSL lives in shaders/ (shared with the C host); this module only loads it.

def _glsl(name): return open(os.path.join(ROOT, "shaders", name)).read()

COMMON = _glsl("common.glsl")
CONV_SRC, DECONV_SRC = _glsl("conv.comp"), _glsl("deconv.comp")
PREP_SRC, UPDATE_SRC, FINAL_SRC = (COMMON + _glsl(n) for n in ("prep.comp", "update.comp", "final.comp"))
UNPACK_SRC, PACK_SRC = _glsl("unpack.comp"), _glsl("pack.comp")

def cdiv(a, b): return (a + b - 1) // b

# ---------------------------------------------------------------- weight packing

def pack_conv(w: np.ndarray, b: np.ndarray, ic_pad: int):
  oc, ic = w.shape[:2]
  if ic_pad > ic: w = np.concatenate([w, np.zeros((oc, ic_pad - ic, 3, 3), w.dtype)], 1)
  oc4, ic4 = oc // 4, ic_pad // 4
  # wt[og][k][tap] = mat4 with column i (input ch k*4+i), row j (output ch og*4+j); std430 mat4 is column major
  wt = w.reshape(oc4, 4, ic4, 4, 9).transpose(0, 2, 4, 3, 1)
  return np.ascontiguousarray(wt, np.float32), np.ascontiguousarray(b, np.float32)

def pack_deconv(w: np.ndarray, b: np.ndarray):
  oc, ic = w.shape[:2]           # ncnn: (24, IC, 4, 4), out[oc] += in[ic] * w[oc][ic][ky][kx]
  ic4 = ic // 4
  wt = w.reshape(oc, ic4, 4, 4, 4).transpose(3, 4, 1, 0, 2)  # [ky][kx][ic4][oc][4]
  return np.ascontiguousarray(wt, np.float32), np.ascontiguousarray(b, np.float32)

# ---------------------------------------------------------------- runtime

class GLRife:
  def __init__(self, h: int, w: int, model_dir: str = DEFAULT_MODEL, ctx: moderngl.Context | None = None,
               px: int = 1, ot: int = 4, lx: int = 16, ly: int = 4):
    self.h, self.w = h, w
    self.H, self.W = cdiv(h, 32) * 32, cdiv(w, 32) * 32
    self.ctx = ctx or moderngl.create_standalone_context(backend="egl", require=430)
    self.cfg = dict(PX=px, OT=ot, LX=lx, LY=ly)
    self._progs: dict = {}
    layers = load_ncnn(os.path.join(model_dir, "flownet.param"), os.path.join(model_dir, "flownet.bin"))
    H, W, ctx = self.H, self.W, self.ctx
    buf = lambda nbytes: ctx.buffer(reserve=max(16, nbytes))
    self.img0, self.img1, self.out = buf(H * W * 4), buf(H * W * 4), buf(H * W * 4)
    self.nbytes = h * w * 3
    self.raw_in, self.raw_out = buf(cdiv(self.nbytes, 4) * 4), buf(cdiv(h * w, 4) * 12)
    self._unpack = self._prog(UNPACK_SRC, IW=w, IH=h)
    self._pack = self._prog(PACK_SRC, IW=w, IH=h)
    self._prev, self._last, self._count = self.img0, self.img1, 0
    self.flow, self.mask = buf(H * W * 16), buf(H * W * 4)
    self.steps = []   # list of (program, {binding: buffer}, groups, uniforms)
    for bi, s in enumerate(SCALES):
      L = layers[bi * 11:(bi + 1) * 11]
      first = bi == 0
      hs, ws = H // s, W // s                        # block input res
      cin4 = 2 if first else 3
      xin = buf(cin4 * hs * ws * 16)
      prep = self._prog(PREP_SRC, S=s, FIRST=int(first), WO=ws, HO=hs)
      self.steps.append((prep, {3: xin}, (cdiv(ws, 8), cdiv(hs, 8), 1), True))
      # conv0: two stride-2 convs
      cur, ch, cw, c4 = xin, hs, ws, cin4
      for li in range(2):
        wl = L[li]
        oc4 = wl["weight"].shape[0] // 4
        wt, bs = pack_conv(wl["weight"], wl["bias"], c4 * 4)
        oh, ow = ch // 2, cw // 2
        o = buf(oc4 * oh * ow * 16)
        self.steps.append(self._conv(cur, ctx.buffer(wt.tobytes()), ctx.buffer(bs.tobytes()), o, c4, oc4, 2, 0, ch, cw, oh, ow))
        cur, ch, cw, c4 = o, oh, ow, oc4
      # 8 residual convs, ping-pong
      pp = [cur, buf(c4 * ch * cw * 16)]
      for k in range(8):
        wt, bs = pack_conv(L[2 + k]["weight"], L[2 + k]["bias"], c4 * 4)
        self.steps.append(self._conv(pp[k % 2], ctx.buffer(wt.tobytes()), ctx.buffer(bs.tobytes()), pp[(k + 1) % 2], c4, c4, 1, 1, ch, cw, ch, cw))
      cur = pp[0]  # after 8 (even) steps result lands back in pp[0]
      # deconv + pixelshuffle -> 6ch at block input res
      wt, bs = pack_deconv(L[10]["weight"], L[10]["bias"])
      tmp = buf(2 * hs * ws * 16)
      dec = self._prog(DECONV_SRC, IC4=c4, HI=ch, WI=cw, HO=hs, WO=ws)
      self.steps.append((dec, {0: cur, 1: ctx.buffer(wt.tobytes()), 2: ctx.buffer(bs.tobytes()), 3: tmp}, (cdiv(cw, 8), cdiv(ch, 8), 1), False))
      upd = self._prog(UPDATE_SRC, S=s, FIRST=int(first))
      self.steps.append((upd, {0: tmp}, (cdiv(W, 8), cdiv(H, 8), 1), False))
    fin = self._prog(FINAL_SRC)
    self.steps.append((fin, {3: self.out}, (cdiv(W, 8), cdiv(H, 8), 1), False))
    self.shared = {4: self.img0, 5: self.img1, 6: self.flow, 7: self.mask}

  def _prog(self, src: str, **defs):
    defs = {"W": self.W, "H": self.H, **defs}
    key = (src, tuple(sorted(defs.items())))
    if key not in self._progs:
      head = HEADER + "".join(f"#define {k} {v}\n" for k, v in defs.items())
      self._progs[key] = self.ctx.compute_shader(head + src)
    return self._progs[key]

  def _conv(self, inp, wt, bs, out, ic4, oc4, stride, resid, hi, wi, ho, wo):
    c = self.cfg
    ot = c["OT"]
    while oc4 % ot: ot //= 2
    p = self._prog(CONV_SRC, IC4=ic4, S=stride, RESID=resid, HI=hi, WI=wi, HO=ho, WO=wo, PX=c["PX"], OT=ot, LX=c["LX"], LY=c["LY"])
    groups = (cdiv(cdiv(wo, c["PX"]), c["LX"]), cdiv(ho, c["LY"]), oc4 // ot)
    return (p, {0: inp, 1: wt, 2: bs, 3: out}, groups, False)

  def push(self, frame):
    """Upload the next frame (HxWx3 uint8 array or raw RGB24 bytes). Interpolation runs between the last two pushed."""
    data = frame if isinstance(frame, (bytes, bytearray, memoryview)) else np.ascontiguousarray(frame, np.uint8)
    assert len(memoryview(data).cast("B")) == self.nbytes, "frame size mismatch"
    dst = self._prev                      # the older slot is free; it becomes the newest frame
    self.raw_in.write(data)
    self.raw_in.bind_to_storage_buffer(0); dst.bind_to_storage_buffer(3)
    self._unpack.run(cdiv(self.W, 8), cdiv(self.H, 8), 1)
    self.ctx.memory_barrier()
    self._prev, self._last = self._last, dst
    self._count += 1

  def interpolate_bytes(self, timestep: float = 0.5) -> bytes:
    """Frame at `timestep` between the last two pushed frames, as tight RGB24 bytes."""
    assert self._count >= 2, "push two frames first"
    self.run(timestep)
    self.out.bind_to_storage_buffer(0); self.raw_out.bind_to_storage_buffer(3)
    self._pack.run(cdiv(cdiv(self.h * self.w, 4), 64), 1, 1)
    self.ctx.memory_barrier()
    return self.raw_out.read(size=self.nbytes)

  def interpolate(self, timestep: float = 0.5) -> np.ndarray:
    return np.frombuffer(self.interpolate_bytes(timestep), np.uint8).reshape(self.h, self.w, 3)

  def run(self, timestep: float = 0.5):
    self.shared[4], self.shared[5] = self._prev, self._last
    for prog, binds, groups, has_t in self.steps:
      for k, b in self.shared.items(): b.bind_to_storage_buffer(k)
      for k, b in binds.items(): b.bind_to_storage_buffer(k)
      if has_t: prog["timestep"] = timestep
      prog.run(*groups)
      self.ctx.memory_barrier()

  def __call__(self, a: np.ndarray, b: np.ndarray, timestep: float = 0.5) -> np.ndarray:
    self.push(a); self.push(b); return self.interpolate(timestep)

if __name__ == "__main__":
  import argparse
  from PIL import Image
  ap = argparse.ArgumentParser()
  ap.add_argument("img0"); ap.add_argument("img1"); ap.add_argument("-o", "--out", default="out.png")
  ap.add_argument("-t", "--timestep", type=float, default=0.5)
  args = ap.parse_args()
  a, b = (np.asarray(Image.open(p).convert("RGB")) for p in (args.img0, args.img1))
  Image.fromarray(GLRife(a.shape[0], a.shape[1])(a, b, args.timestep)).save(args.out)
  print("wrote", args.out)
