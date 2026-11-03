"""RIFE v4.6 with OpenGL 4.3 core compute shaders, fp32 only, no extensions.

Tensor layout ("pack4"): SSBO of vec4, index ((c4 * H + y) * W + x), 4 channels per vec4.
Images: SSBO of uint (RGBA8, unpackUnorm4x8), zero padded to a multiple of 32.
"""
from __future__ import annotations
import os, numpy as np, moderngl
from ncnn_weights import load_ncnn

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "ext/rife-ncnn-vulkan/models/rife-v4.6")
SCALES = (8, 4, 2, 1)

HEADER = "#version 430\n"

# ---------------------------------------------------------------- shaders

CONV_SRC = """
layout(local_size_x = LX, local_size_y = LY, local_size_z = 1) in;
layout(std430, binding = 0) readonly buffer In  { vec4 inp[]; };
layout(std430, binding = 1) readonly buffer Wt  { mat4 wt[]; };    // [oc4][ic4][9], column i = input ch i
layout(std430, binding = 2) readonly buffer Bs  { vec4 bias[]; };  // [oc4]
layout(std430, binding = 3) writeonly buffer Out { vec4 outp[]; };

void main() {
  int ox0 = int(gl_GlobalInvocationID.x) * PX;
  int oy = int(gl_GlobalInvocationID.y);
  int og0 = int(gl_WorkGroupID.z) * OT;          // workgroup-uniform -> scalar weight loads
  if (oy >= HO || ox0 >= WO) return;
  vec4 acc[OT][PX];
  for (int t = 0; t < OT; t++) for (int p = 0; p < PX; p++) acc[t][p] = bias[og0 + t];
  for (int ic = 0; ic < IC4; ic++) {
    for (int ky = 0; ky < 3; ky++) {
      int iy = oy * S - 1 + ky;
      if (iy < 0 || iy >= HI) continue;
      int rowbase = (ic * HI + iy) * WI;
      for (int kx = 0; kx < 3; kx++) {
        vec4 v[PX];
        for (int p = 0; p < PX; p++) {
          int ix = (ox0 + p) * S - 1 + kx;
          v[p] = (ix >= 0 && ix < WI) ? inp[rowbase + ix] : vec4(0.0);
        }
        for (int t = 0; t < OT; t++) {
          mat4 w = wt[((og0 + t) * IC4 + ic) * 9 + ky * 3 + kx];
          for (int p = 0; p < PX; p++) acc[t][p] += w * v[p];
        }
      }
    }
  }
  for (int t = 0; t < OT; t++) for (int p = 0; p < PX; p++) {
    int ox = ox0 + p;
    if (ox >= WO) break;
    int o = ((og0 + t) * HO + oy) * WO + ox;
    vec4 r = acc[t][p];
#if RESID
    r += inp[o];
#endif
    outp[o] = max(r, 0.2 * r);
  }
}
"""

# deconv(4x4, s2, p1) -> 24ch at 2x, then PixelShuffle(2) -> 6ch at 4x; we compute the 6ch result directly
# One thread per input pixel (i,j): it owns deconv outputs (2i+py, 2j+px), py,px in {0,1}, which only touch
# inputs i-1..i+1 / j-1..j+1, so every tap and weight index is compile-time constant (uniform loads).
# Deconv output (dy,dx) channel c*4+sub -> final pixel (2dy + sub/2, 2dx + sub%2) channel c. Only c<5 is used.
DECONV_SRC = """
layout(local_size_x = 8, local_size_y = 8) in;
layout(std430, binding = 0) readonly buffer In  { vec4 inp[]; };   // IC4 x HI x WI
layout(std430, binding = 1) readonly buffer Wt  { vec4 wt[]; };    // [ky][kx][ic4][oc24] vec4 over ic
layout(std430, binding = 2) readonly buffer Bs  { float bias[]; }; // [24]
layout(std430, binding = 3) writeonly buffer Out { vec4 outp[]; }; // 2 x HO x WO : (flow4) (mask, 0, 0, 0)
void main() {
  int j = int(gl_GlobalInvocationID.x), i = int(gl_GlobalInvocationID.y);
  if (j >= WI || i >= HI) return;
  float acc[2][2][20];
  for (int py = 0; py < 2; py++) for (int px = 0; px < 2; px++) for (int o = 0; o < 20; o++) acc[py][px][o] = bias[o];
  for (int ic = 0; ic < IC4; ic++) {
    vec4 v[3][3];
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) {
      int iy = i - 1 + a, ix = j - 1 + b;
      v[a][b] = (iy >= 0 && iy < HI && ix >= 0 && ix < WI) ? inp[(ic * HI + iy) * WI + ix] : vec4(0.0);
    }
    for (int py = 0; py < 2; py++) for (int t = 0; t < 2; t++) {
      int ky = (1 - py) + 2 * t;              // ky = py+1 (mod 2)
      int a = ((py + 1 - ky) >> 1) + 1;       // iy = i + (py+1-ky)/2
      for (int px = 0; px < 2; px++) for (int u = 0; u < 2; u++) {
        int kx = (1 - px) + 2 * u;
        int b = ((px + 1 - kx) >> 1) + 1;
        int wb = ((ky * 4 + kx) * IC4 + ic) * 24;
        for (int o = 0; o < 20; o++) acc[py][px][o] += dot(v[a][b], wt[wb + o]);
      }
    }
  }
  for (int py = 0; py < 2; py++) for (int px = 0; px < 2; px++) for (int sub = 0; sub < 4; sub++) {
    int Y = 2 * (2 * i + py) + (sub >> 1), X = 2 * (2 * j + px) + (sub & 1);
    outp[Y * WO + X] = vec4(acc[py][px][sub], acc[py][px][4 + sub], acc[py][px][8 + sub], acc[py][px][12 + sub]);
    outp[(HO + Y) * WO + X] = vec4(acc[py][px][16 + sub], 0.0, 0.0, 0.0);
  }
}
"""

COMMON = """
layout(std430, binding = 4) readonly buffer Img0 { uint img0[]; };
layout(std430, binding = 5) readonly buffer Img1 { uint img1[]; };
layout(std430, binding = 6) buffer Flow { vec4 flow[]; };   // full res: (f0x, f0y, f1x, f1y)
layout(std430, binding = 7) buffer Mask { float mask[]; };  // full res
vec4 px0(int x, int y) { return unpackUnorm4x8(img0[y * W + x]); }
vec4 px1(int x, int y) { return unpackUnorm4x8(img1[y * W + x]); }
// backward warp == grid_sample(bilinear, border, align_corners=True), flow in pixels
#define WARP(PX, fx, fy) { \\
  float gx = clamp(float(x) + fx, 0.0, float(W - 1)), gy = clamp(float(y) + fy, 0.0, float(H - 1)); \\
  int x0 = int(floor(gx)), y0 = int(floor(gy)); int x1 = min(x0 + 1, W - 1), y1 = min(y0 + 1, H - 1); \\
  float ax = gx - float(x0), ay = gy - float(y0); \\
  res = mix(mix(PX(x0, y0), PX(x1, y0), ax), mix(PX(x0, y1), PX(x1, y1), ax), ay); }
vec3 warp0(int x, int y, vec2 f) { vec4 res; WARP(px0, f.x, f.y) return res.rgb; }
vec3 warp1(int x, int y, vec2 f) { vec4 res; WARP(px1, f.x, f.y) return res.rgb; }
"""

# builds block input at 1/S res: downscale(cat(...)) as exact align_corners=False bilinear (mean of 2x2 middle taps)
PREP_SRC = COMMON + """
layout(local_size_x = 8, local_size_y = 8) in;
layout(std430, binding = 3) writeonly buffer Out { vec4 outp[]; };
uniform float timestep;
void main() {
  int X = int(gl_GlobalInvocationID.x), Y = int(gl_GlobalInvocationID.y);
  if (X >= WO || Y >= HO) return;
  vec4 a = vec4(0.0), b = vec4(0.0), c = vec4(0.0);
#if S == 1
  const int NT = 1, OFF = 0;
#else
  const int NT = 2, OFF = S / 2 - 1;
#endif
  for (int ty = 0; ty < NT; ty++) for (int tx = 0; tx < NT; tx++) {
    int x = X * S + OFF + tx, y = Y * S + OFF + ty;
#if FIRST
    vec4 p0 = px0(x, y), p1 = px1(x, y);
    a += vec4(p0.rgb, p1.r); b += vec4(p1.gb, timestep, 0.0);
#else
    vec4 f = flow[y * W + x];
    vec3 w0 = warp0(x, y, f.xy), w1 = warp1(x, y, f.zw);
    a += vec4(w0, w1.r); b += vec4(w1.gb, timestep, mask[y * W + x]); c += f;
#endif
  }
  float inv = 1.0 / float(NT * NT);
  int n = HO * WO, o = Y * WO + X;
  outp[o] = a * inv; outp[n + o] = b * inv;
#if !FIRST
  outp[2 * n + o] = c * (inv / float(S));
#endif
}
"""

# upsample block output (6ch @ 1/S) to full res, accumulate flow (*S) and mask
UPDATE_SRC = COMMON + """
layout(local_size_x = 8, local_size_y = 8) in;
layout(std430, binding = 0) readonly buffer T { vec4 tmp[]; };
void main() {
  int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y);
  if (x >= W || y >= H) return;
  vec4 f, m;
#if S == 1
  f = tmp[y * W + x]; m = tmp[(H + y) * W + x];
#else
  const int WL = W / S, HL = H / S;
  float sx = max((float(x) + 0.5) / float(S) - 0.5, 0.0), sy = max((float(y) + 0.5) / float(S) - 0.5, 0.0);
  int x0 = int(sx), y0 = int(sy); int x1 = min(x0 + 1, WL - 1), y1 = min(y0 + 1, HL - 1);
  float ax = sx - float(x0), ay = sy - float(y0);
  f = mix(mix(tmp[y0 * WL + x0], tmp[y0 * WL + x1], ax), mix(tmp[y1 * WL + x0], tmp[y1 * WL + x1], ax), ay);
  int n = HL * WL;
  m = mix(mix(tmp[n + y0 * WL + x0], tmp[n + y0 * WL + x1], ax), mix(tmp[n + y1 * WL + x0], tmp[n + y1 * WL + x1], ax), ay);
#endif
  int o = y * W + x;
#if FIRST
  flow[o] = f * float(S); mask[o] = m.x;
#else
  flow[o] += f * float(S); mask[o] += m.x;
#endif
}
"""

FINAL_SRC = COMMON + """
layout(local_size_x = 8, local_size_y = 8) in;
layout(std430, binding = 3) writeonly buffer Out { uint outp[]; };
void main() {
  int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y);
  if (x >= W || y >= H) return;
  int o = y * W + x;
  vec4 f = flow[o];
  float m = 1.0 / (1.0 + exp(-mask[o]));
  vec3 r = warp0(x, y, f.xy) * m + warp1(x, y, f.zw) * (1.0 - m);
  outp[o] = packUnorm4x8(vec4(r, 1.0));
}
"""

# tight RGB24 (IW x IH, as bytes in a uint SSBO) -> padded RGBA8 (W x H), zero outside the image
UNPACK_SRC = """
layout(local_size_x = 8, local_size_y = 8) in;
layout(std430, binding = 0) readonly buffer Raw { uint raw[]; };
layout(std430, binding = 3) writeonly buffer Out { uint outp[]; };
uint byte_at(int i) { return (raw[i >> 2] >> uint((i & 3) * 8)) & 0xffu; }
void main() {
  int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y);
  if (x >= W || y >= H) return;
  uint v = 0u;
  if (x < IW && y < IH) {
    int i = (y * IW + x) * 3;
    v = byte_at(i) | (byte_at(i + 1) << 8) | (byte_at(i + 2) << 16);
  }
  outp[y * W + x] = v;
}
"""

# padded RGBA8 (W x H) -> tight RGB24 (IW x IH); each thread packs 4 pixels into 3 uints (no byte-write races)
PACK_SRC = """
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Src { uint src[]; };
layout(std430, binding = 3) writeonly buffer Dst { uint dst[]; };
void main() {
  int t = int(gl_GlobalInvocationID.x), p0 = t * 4, n = IW * IH;
  if (p0 >= n) return;
  uint b[12];
  for (int k = 0; k < 4; k++) {
    int p = p0 + k;
    uint v = p < n ? src[(p / IW) * W + p % IW] : 0u;
    b[3 * k] = v & 0xffu; b[3 * k + 1] = (v >> 8) & 0xffu; b[3 * k + 2] = (v >> 16) & 0xffu;
  }
  for (int j = 0; j < 3; j++) dst[t * 3 + j] = b[4 * j] | (b[4 * j + 1] << 8) | (b[4 * j + 2] << 16) | (b[4 * j + 3] << 24);
}
"""

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
