# tinyrife

RIFE v4.6 frame interpolation ([Practical-RIFE](https://github.com/hzwer/Practical-RIFE) weights, via the
[rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan) model files) in plain C, with two backends behind one
small API (`src/rife.h`):

- **gl**: OpenGL 4.3 core compute shaders, fp32, no extensions; links nothing (libEGL is `dlopen`ed).
- **cpu**: AVX2/FMA + OpenMP, about 3.5x faster than PyTorch.

Both are hardcoded to this one network and match the PyTorch reference within 1 uint8 level.

## Build and run

Needs a C compiler, make and sh (and ffmpeg/ffprobe on `PATH` to run the video tool).

```sh
./fetch_model.sh        # model -> ext/
make                    # build/{cpu,gl}, build/{cpu,gl}_dir, build/{cpu,gl}_video
make test               # unit + end-to-end tests
```

Each tool exists once per backend, with the same options (`-m model_dir` defaults to the fetched model):

```sh
build/gl       [-m model_dir] [-t 0.5] [-b N] a.png b.png out.png    # one pair; -b N also times N runs
build/gl_dir   [-m model_dir] [-j 4] in_dir out_dir                  # 2x a PNG sequence
build/gl_video [-m model_dir] [-x 2] [-t SECONDS] [--vcodec "libsvtav1 -preset 8 -crf 28"] in.webm out.mkv
build/cpu ...  build/cpu_dir ...  build/cpu_video ...                # the same on the CPU
```

- **pair** writes the frame at `-t` between `a` and `b`. With `-b N`, `RIFE_PROFILE=1` adds the time per stage.
- **dir** turns `in_dir/*.png` (sorted) into `out_dir/%08d.png`: input 1, in-between, input 2, ... (2N-1 frames). The
  inputs are copied byte for byte; a reader thread and `-j` writer threads keep PNG I/O off the interpolation thread.
- **video** pipes raw RGB24 from one ffmpeg into another, with no intermediate files: `-x` times the frame rate, each
  source frame followed by x-1 interpolated ones (the last frame is repeated), audio copied. `-t` processes only the
  first N seconds. `FFMPEG` / `FFPROBE` name other binaries.

## Layout

```
src/rife.h      the API: rife_new, rife_push, rife_interpolate, rife_free (+ timing)
src/gl.c        GL backend          shaders/   its GLSL, embedded into the build by embed.sh
src/cpu.c       CPU backend         (net_forward() is the whole pipeline: read it first)
src/pair.c dir.c video.c   the tools, each built against both backends; util.h: queue and helpers for dir/video
tests/          kernel unit tests, end-to-end tests, golden images
python/         ref_torch.py, the optional PyTorch reference; nothing in the build uses it
third_party/    stb_image, stb_image_write
```

## Tests

```sh
make test       # needs the model for the end-to-end part; GL cases are skipped without an OpenGL 4.3 context
make sanitize   # the same suite under AddressSanitizer + UBSan, and the threaded tools under ThreadSanitizer
make lint       # warnings as errors
```

- `tests/test_kernels.c` checks the CPU kernels against naive references (no model needed).
- `tests/run.sh` runs every tool of both backends: golden images (tolerance 1 uint8 level), CPU against GL, `dir`
  against `pair`, and rejection of damaged models and bad arguments.
- `dir` and `video` are also built against a fake backend (`tests/fake.c`: the "interpolated" frame is a linear blend),
  and `video` against fake `ffmpeg`/`ffprobe` scripts, so their file handling, error paths and threading are checked
  exactly, without a model, GPU or codec.
- `python python/ref_torch.py a.png b.png ref.png`, then `build/pngdiff ref.png out.png 1`, checks a backend's output
  against PyTorch (needs torch, numpy, pillow).

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

Warp matches `grid_sample(bilinear, padding="border", align_corners=True)`.

The CPU backend runs the same network: 4 IFBlocks (192/128/96/64 channels at 1/8..1/1 scale) refine a flow field and a
blend mask, then a fused warp + blend produces the frame. The residual 3x3 convs run as Winograd F(4x4)/F(6x6) in a
packed-8 channel layout (the tile grouping is sized from the detected caches; `RIFE_CACHE=L2,L3slice,L3total`
overrides), the strided convs and the deconv(+PixelShuffle) are direct AVX2/FMA kernels, and warp + concat + downscale
and the final warp + sigmoid blend are fused.

## Measurements

Accuracy against the PyTorch reference and against ground truth (translated photo crops), with rife-ncnn-vulkan
20221029 (fp16) for comparison:

| motion | GL vs torch (max diff) | ncnn vs torch (max diff) | PSNR vs GT torch / GL / ncnn |
|---|---|---|---|
| 12,6 px (448x384) | 1 | 2 | 39.48 / 39.48 / 39.46 dB |
| 24,10 px (720p) | 1 | 6 | 36.85 / 36.85 / 36.84 dB |
| 64,0 px (720p) | 1 | 26 | 35.34 / 35.34 / 35.30 dB |
| 120,40 px (720p) | 1 | 94 | 25.04 / 25.04 / 25.05 dB |

GL time per frame, AMD Renoir iGPU (Radeon Vega, Mesa 26.2):

| resolution | GL GPU time | rife-ncnn-vulkan |
|---|---|---|
| 128x128 | 8.1 ms | **5.2 ms** |
| 256x256 | **12.1 ms** | 13.8 ms |
| 640x360 | **32.4 ms** | 45.5 ms |
| 960x540 | **66.5 ms** | 99.9 ms |
| 1280x720 | **110.8 ms** | 184.1 ms |
| 1920x1080 | **249.7 ms** | 462.1 ms |

CPU time per frame against the same graph in stock PyTorch (same weights; torch 2.15.0+cpu from pip, with oneDNN),
mean ms, Ryzen 5 4650U (6C/12T, Zen 2). `python/ref_torch.py ... -b N --threads N` gives the PyTorch side:

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
