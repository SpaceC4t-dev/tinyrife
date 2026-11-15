"""Minimal reader for the ncnn .param/.bin pair of rife-v4.6 (Convolution / Deconvolution layers only)."""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (this file is in python/)
DEFAULT_MODEL = os.path.join(ROOT, "ext/rife-ncnn-vulkan/models/rife-v4.6")   # from fetch_model.sh
SCALES = (8, 4, 2, 1)   # input downscale of the four IFBlocks

FP16_TAG, FP32_TAG = 0x01306B47, 0x00000000

def _parse_params(tokens):
  p = {}
  for t in tokens:
    k, v = t.split("=", 1)
    p[int(k)] = v
  return p

def load_ncnn(param_path: str, bin_path: str) -> list[dict]:
  """Returns weight-bearing layers in file order: {name, type, weight (out,in,kh,kw) float32, bias, stride, pad}."""
  lines = open(param_path).read().split("\n")
  assert lines[0].strip() == "7767517", "not an ncnn param file"
  blob = open(bin_path, "rb").read()
  off, layers = 0, []
  for line in lines[2:]:
    parts = line.split()
    if not parts or parts[0] not in ("Convolution", "Deconvolution"): continue
    ltype, name, nin, nout = parts[0], parts[1], int(parts[2]), int(parts[3])
    p = _parse_params(parts[4 + nin + nout:])
    outc, k, stride, pad = int(p[0]), int(p[1]), int(p.get(3, 1)), int(p.get(4, 0))
    has_bias, wsize = int(p.get(5, 0)), int(p[6])
    tag = int.from_bytes(blob[off:off+4], "little"); off += 4
    if tag == FP16_TAG:
      w = np.frombuffer(blob, np.float16, wsize, off).astype(np.float32); off += (wsize * 2 + 3) // 4 * 4
    elif tag == FP32_TAG:
      w = np.frombuffer(blob, np.float32, wsize, off).copy(); off += wsize * 4
    else: raise ValueError(f"{name}: unsupported weight tag {tag:#x}")
    b = None
    if has_bias:
      b = np.frombuffer(blob, np.float32, outc, off).copy(); off += outc * 4
    inc = wsize // (outc * k * k)
    layers.append(dict(name=name, type=ltype, weight=w.reshape(outc, inc, k, k), bias=b, stride=stride, pad=pad))
  assert off == len(blob), f"consumed {off} of {len(blob)} bytes -- layout mismatch"
  return layers

if __name__ == "__main__":
  import sys
  for l in load_ncnn(sys.argv[1], sys.argv[2]):
    print(f"{l['type']:14s} {l['name']:12s} w{tuple(l['weight'].shape)} s{l['stride']} p{l['pad']} |w|max={np.abs(l['weight']).max():.3f}")
