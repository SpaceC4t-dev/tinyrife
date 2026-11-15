# tinyrife

RIFE v4.6 frame interpolation ([Practical-RIFE](https://github.com/hzwer/Practical-RIFE) weights, via the
[rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan) model files) implemented with **OpenGL 4.3 core compute
shaders**, fp32 only, no extensions. There are three implementations of the same network: two C programs
(GPU: `gl` for a frame pair, `gl_dir` for a PNG sequence and `gl_video` for a video, built on the host library `gl_host.c`; CPU: `cpu` with
AVX2 + OpenMP, ~3.5x faster than PyTorch) and a Python GL host.

## Layout

```
src/          the C code: cpu.c; gl_host.{c,h} (GL host library), gl.c (one pair), gl_dir.c (PNG sequence),
              gl_video.c (video via ffmpeg), util.h (queue and helpers shared by the last two)
shaders/      GLSL compute shaders (*.comp) shared by both GL hosts; embedded into the C build by embed.sh
third_party/  stb_image / stb_image_write (vendored)
python/       optional helpers: GL host, torch reference, tests, benchmarks
data/         sample frames
tests/        C unit tests, end-to-end golden tests and their golden images (no Python needed)
Makefile      builds build/cpu, build/gl, build/gl_dir and build/gl_video; needs only a C compiler, make and sh (ffmpeg only to run gl_video)
```

Nothing in `src/` or the build depends on Python; `python/` only uses what the C side defines (the shaders, the model).

## Build and run (C)

```sh
./fetch_model.sh                       # model -> ext/ (needed by both programs; add --ncnn-bin for the comparison binary)
make                                   # build/cpu, build/gl, build/gl_dir, build/gl_video

build/gl     [-m model_dir] [-t 0.5] [-b N] a.png b.png out.png   # GPU, one pair; -b N also times N runs
build/gl_dir [-m model_dir] [-j 4] in_dir out_dir                 # GPU, 2x a PNG sequence (2N-1 frames)
build/gl_video [-m model_dir] [-x 2] [-t SECONDS] [--vcodec "libsvtav1 -preset 8 -crf 28"] in.webm out.mkv   # GPU, video -> video
build/cpu    [--pad 32|64] ext/rife-ncnn-vulkan/models/rife-v4.6/flownet.bin a.png b.png 0.5 out.png [iterations]
```

## Tests

```sh
make test       # kernel unit tests + end-to-end golden tests (model needed for the latter: ./fetch_model.sh)
```

- `tests/test_kernels.c` checks the CPU kernels (resampling, warp, the fused warp+downscale vs the unfused pipeline it
  replaces, final merge, elementwise ops, fp16 decode) against naive references; no model or files needed.
- `tests/run.sh` runs both programs on the frames in `data/` against the images in `tests/golden/` (tolerance 1 uint8 level,
  the project's pass criterion), checks that the CPU and GL outputs agree, and that truncated, padded or wrong model files
  and bad arguments are rejected. The GL cases are skipped, not failed, when there is no OpenGL 4.3 context.
- `gl_dir` and `gl_video` are also built against a fake GL host (`tests/fake_gl_host.c`, "interpolation" = the linear blend of
  the two frames), so their handling, error cases and threading are tested exactly and without a GPU. `gl_dir`'s output is
  verified by `tests/check_dir.c`; on the GPU its frames are compared with `gl`'s for each pair. `gl_video` is run against fake
  `ffmpeg`/`ffprobe` scripts (`tests/fake_ffmpeg.sh`) on raw RGB24 files: `tests/check_video.c` checks the exact frame
  sequence for several multipliers, a cut-off last frame and a single frame, and the tests check the ffmpeg command lines and
  the failing-ffprobe, decoder and encoder cases. With a real ffmpeg on `PATH` one more test runs a real video (and, with a GL
  context, the real GPU path) and checks the frame count, rate and that the audio is copied bit for bit.
- `python/test_gl.py` is the independent check against the PyTorch reference (needs torch and a GPU).

### Checking tools

```sh
make sanitize   # unit + end-to-end tests under AddressSanitizer + UBSan + leak check (gcc only)
make lint       # -Wall -Wextra -Wshadow -Wformat=2 ... as errors
make analyze    # gcc -fanalyzer (slow)
make tidy       # clang-tidy, if installed (config in .clang-tidy; `pip install clang-tidy` works without root)
```

`make sanitize` and `make lint` are clean. `make analyze` reports one location-less `-Wanalyzer-use-of-uninitialized-value`
that I could not trace to a source line. ThreadSanitizer is not usable with libgomp (it reports the fork/join of every
parallel region as a race); instead the CPU output is byte-identical at 1, 2, 3, 6 and 12 threads and across repeated runs.
It is also unusable against the real GL driver (it reports Mesa's own threads), so the threading of `gl_dir` and `gl_video` is checked
with TSan against the fake host, where it found (and `make sanitize` now guards against) a use-after-free in the reader thread.

`gl_dir` turns `in_dir/*.png` (sorted) into `out_dir/%08d.png`: input frame 1, the interpolated frame, input frame 2, ...
(2N-1 frames). The inputs are copied byte for byte, the in-between frames use fast PNG compression, and every frame is
decoded once and uploaded once. A reader thread decodes ahead and `-j` writer threads encode and copy behind, so PNG I/O
overlaps the GPU; all GL calls stay on the main thread. On 24 frames of 1080p it takes 6.4 s and 131 MB peak,
against 7.0 s and 264 MB for the Python tool it replaces, with identical output (RIFE calls dominate at about 270 ms each).

`gl_video` decodes `in` with ffmpeg to raw RGB24, interpolates, and pipes the frames into a second ffmpeg that encodes (and copies
the audio): the output has `-x` times the frame rate, each source frame followed by x-1 interpolated ones (the last frame is
repeated x times), `-t` only processes the first N seconds. A reader and a writer thread keep the pipes busy while all GL calls
stay on the main thread. It needs `ffmpeg` and `ffprobe` on `PATH` (or named by `FFMPEG` / `FFPROBE`). On 24 frames of 1080p
(x2, lossless codec) its decoded output is byte-identical to the Python tool it replaces and it runs about 4% faster (6.5 s against
6.8 s; GPU-bound). Peak memory is the same, about 660 MB, almost all of it the two ffmpeg processes. Note `-x` is the multiplier
(the Python tool used `-m`, which is the model directory in every C tool).

`gl` links nothing: libEGL is `dlopen`ed and GL entry points come from `eglGetProcAddress`. `cpu` is hardcoded
to this network (`net_forward()` is the whole pipeline) and checks the `.bin` against the architecture.
`RIFE_PROFILE=1` prints a per-phase time breakdown. `--pad 64` matches Practical-RIFE's padding.

## Python helpers

```sh
pip install moderngl numpy pillow        # torch only needed for tests/compare/bench_cpu_torch
./fetch_model.sh                         # model -> ext/; add --ncnn-bin for the comparison binary
```

Runs headless through EGL (`moderngl.create_standalone_context(backend="egl")`).

## Usage

```sh
python python/gl_rife.py frame0.png frame1.png -o mid.png [-t 0.5]   # single pair
```

```python
from gl_rife import GLRife            # run from python/ or put it on PYTHONPATH
g = GLRife(h, w)            # fixed resolution, padded internally to a multiple of 32
mid = g(img0, img1, 0.5)    # uint8 HxWx3 in, uint8 HxWx3 out

# streaming: each frame is uploaded once (HxWx3 uint8 or raw RGB24 bytes)
g.push(f0)
for f in frames:
    g.push(f)
    mid = g.interpolate_bytes(0.5)   # tight RGB24 bytes, ready for an ffmpeg rawvideo pipe
```

## Implementation

Tensors live in SSBOs in a pack4 layout (`vec4[C/4][H][W]`); conv weights are pre-packed as `mat4` so each 4-in/4-out
step is one `mat4 * vec4`. Shaders:

| shader | role |
|---|---|
| prep | warp + concat + bilinear downscale (align_corners=False) into the block input, fused |
| conv3x3 | stride 1/2, bias, LeakyReLU(0.2), optional residual add; output-channel group per workgroup (uniform weight loads) |
| deconv | ConvTranspose 4x4/s2 fused with PixelShuffle(2); one thread per input pixel, compile-time taps |
| update | bilinear upsample of block output, accumulate flow (x scale) and mask |
| final | sigmoid mask, two warps, blend, pack RGBA8 |
| unpack / pack | tight RGB24 <-> padded RGBA8 on the GPU, so the host does no per-pixel work |

Warp matches `grid_sample(bilinear, padding="border", align_corners=True)`.

## Correctness

`python/test_gl.py` checks against an independent PyTorch reference built from stock ops (`ref_torch.py`):
flow max |diff| 1.2e-4, output within 1 uint8 level. `python/compare_outputs.py` compares GL and rife-ncnn-vulkan against the
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

AMD Renoir iGPU (Radeon Vega, Mesa 26.2 radeonsi/RADV), `python/bench_gl.py` / `python/bench_ncnn.py` (rife-ncnn-vulkan 20221029, fp16):

| resolution | GL GPU time | rife-ncnn-vulkan |
|---|---|---|
| 128x128 | 8.1 ms | **5.2 ms** |
| 256x256 | **12.1 ms** | 13.8 ms |
| 640x360 | **32.4 ms** | 45.5 ms |
| 960x540 | **66.5 ms** | 99.9 ms |
| 1280x720 | **110.8 ms** | 184.1 ms |
| 1920x1080 | **249.7 ms** | 462.1 ms |

Streaming end-to-end (upload + interpolate + download, `push` + `interpolate_bytes`) is within ~4 ms of GPU time:
115 ms at 720p, 254 ms at 1080p. Python dispatch overhead is ~3-7 us per dispatch (53 per frame).
A 10 s 1080p30 clip -> 60 fps with SVT-AV1 encoding in parallel took 84.8 s (3.55 interp/s), measured with the Python video tool
(now `gl_video`, which is GPU-bound in the same way).

`python python/bench_gl.py 1280x720 --breakdown` prints per-layer GPU times (GL timer queries).

## CPU implementation

`src/cpu.c` is the same network on the CPU: 4 IFBlocks (192/128/96/64 channels at 1/8..1/1 scale) refine a flow
field and a blend mask, then a fused warp + blend produces the frame. The residual 3x3 convs run as Winograd
F(4x4)/F(6x6) in a packed-8 channel layout, the strided convs and the deconv(+PixelShuffle) are direct AVX2/FMA
kernels, and warp + concat + downscale and the final warp + sigmoid blend are fused.

Benchmark vs. the Practical-RIFE v4.6 graph in stock PyTorch (`python/ref_torch.py`, same weights; torch 2.15.0+cpu from
pip with oneDNN; `inference_mode`). Warm inference, mean ms per frame, Ryzen 5 4650U (6C/12T, Zen 2); timings are noisy
from clock/thermal swings. Output matches the torch reference within 1 uint8 level.

1280x704:

| threads | PyTorch | C impl | speedup |
|--------:|--------:|-------:|--------:|
| 1       | 1079    | 472    | 2.3x    |
| 4       | 523     | 179    | 2.9x    |
| 6       | 516     | 148    | 3.5x    |
| 12      | 607     | 151    | 4.0x    |

256x256:

| threads | PyTorch | C impl | speedup |
|--------:|--------:|-------:|--------:|
| 4       | 38.8    | 16.2   | 2.4x    |
| 12      | 39.2    | 13.7   | 2.9x    |

PyTorch stops scaling past ~4-6 threads and gets slower at 12 (SMT oversubscription), so the 6-thread row is the fair
headline. The distro `python313-torch` (no oneDNN) is slower still; these numbers use the PyPI wheel.
`python python/bench_cpu_torch.py [--threads N] [--ref out.png]` runs the torch side on `data/frame*.png`.
