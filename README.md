# tinyrife

RIFE v4.6 frame interpolation ([Practical-RIFE](https://github.com/hzwer/Practical-RIFE) weights, via the
[rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan) model files) implemented with **OpenGL 4.3 core compute
shaders**, fp32 only, no extensions.

## Setup

```sh
pip install moderngl numpy pillow        # torch only needed for tests/compare
./fetch_model.sh                         # model -> ext/; add --ncnn-bin for the comparison binary
```

Runs headless through EGL (`moderngl.create_standalone_context(backend="egl")`).

## Usage

```sh
python gl_rife.py frame0.png frame1.png -o mid.png [-t 0.5]   # single pair
python run_gl_dir.py inputs/ outputs/                          # 2x a PNG sequence (2N-1 frames)
```

```python
from gl_rife import GLRife
g = GLRife(h, w)            # fixed resolution, padded internally to a multiple of 32
mid = g(img0, img1, 0.5)    # uint8 HxWx3 in, uint8 HxWx3 out
```

## Implementation

Tensors live in SSBOs in a pack4 layout (`vec4[C/4][H][W]`); conv weights are pre-packed as `mat4` so each 4-in/4-out
step is one `mat4 * vec4`. Five shaders:

| shader | role |
|---|---|
| prep | warp + concat + bilinear downscale (align_corners=False) into the block input, fused |
| conv3x3 | stride 1/2, bias, LeakyReLU(0.2), optional residual add; output-channel group per workgroup (uniform weight loads) |
| deconv | ConvTranspose 4x4/s2 fused with PixelShuffle(2); one thread per input pixel, compile-time taps |
| update | bilinear upsample of block output, accumulate flow (x scale) and mask |
| final | sigmoid mask, two warps, blend, pack RGBA8 |

Warp matches `grid_sample(bilinear, padding="border", align_corners=True)`.

## Correctness

`test_gl.py` checks against an independent PyTorch reference built from stock ops (`ref_torch.py`):
flow max |diff| 1.2e-4, output within 1 uint8 level. `compare_outputs.py` compares GL and rife-ncnn-vulkan against the
reference and against ground truth (translated photo crops):

| motion | GL vs torch (max diff) | ncnn vs torch (max diff) | PSNR vs GT torch / GL / ncnn |
|---|---|---|---|
| 12,6 px (448x384) | 1 | 2 | 39.48 / 39.48 / 39.46 dB |
| 24,10 px (720p) | 1 | 6 | 36.85 / 36.85 / 36.84 dB |
| 64,0 px (720p) | 1 | 26 | 35.34 / 35.34 / 35.30 dB |
| 120,40 px (720p) | 1 | 94 | 25.04 / 25.04 / 25.05 dB |

GL reproduces the fp32 model exactly; ncnn (fp16, different edge handling in its warp) differs from it but is equally
accurate against ground truth.

## Performance

AMD Renoir iGPU (Radeon Vega, Mesa 26.2 radeonsi/RADV), `bench_gl.py` / `bench_ncnn.py` (rife-ncnn-vulkan 20221029, fp16):

| resolution | GL GPU time | GL incl. upload/download | rife-ncnn-vulkan |
|---|---|---|---|
| 128x128 | 8.1 ms | 10.2 ms | **5.2 ms** |
| 256x256 | **12.1 ms** | 16.5 ms | 13.8 ms |
| 640x360 | **32.4 ms** | 43.1 ms | 45.5 ms |
| 960x540 | **66.5 ms** | 88.0 ms | 99.9 ms |
| 1280x720 | **117.5 ms** | 152.1 ms | 184.1 ms |
| 1920x1080 | **268.6 ms** | 336.4 ms | 462.1 ms |

`python bench_gl.py 1280x720 --breakdown` prints per-layer GPU times (GL timer queries).
