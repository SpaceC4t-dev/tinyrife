# Roadmap

Current state: the OpenGL 4.3 / fp32 port matches the PyTorch reference within 1 uint8 level and is GPU-bound end to end
(1080p: 250 ms GPU, 254 ms streaming end to end on an AMD Renoir iGPU; 1.4-1.7x faster than rife-ncnn-vulkan at 540p+).
Main convs run at 74-80% of fp32 peak, so the remaining work is mostly features, polish and portability.

Priority order:

## 1. Scene-cut detection (quality, small)

On hard cuts RIFE produces a morph / cross-fade (e.g. a colour change plus text appearing between two frames).
Practical-RIFE compares the two frames at low resolution (SSIM threshold) and duplicates a frame instead of interpolating.

- Small GPU reduction over a downscaled frame pair (or reuse block-0 input), threshold configurable.
- Expose in `run_gl_video.py` / `run_gl_dir.py` (`--scene-thresh`, off = always interpolate).
- ~30 lines.

## 2. `scale` option for 4K / large motion (speed + quality, small)

Practical-RIFE's `scale` runs the whole flow network at reduced resolution (0.5 is standard for 4K); only the final warp
happens at full resolution.

- Scale list becomes `[8, 4, 2, 1] / scale`, plus a downscale of the inputs and an upscale of flow/mask before `final`.
- 4K on this iGPU: ~1 s/frame at scale 1 vs ~0.25-0.3 s at scale 0.5 (estimate). Also more robust to very large motion.
- Validate against `ref_torch.py` extended with the same scale logic.

## 3. Shader tuning (speed, medium)

Expected total: roughly 111 -> 100 ms at 720p.

- **deconv**: 128 VGPRs -> 2 waves/SIMD (34% of peak). Split the 20 output channels across workgroups (z) to cut
  registers; ~5 ms.
- **block 0**: 920-pixel grid cannot fill 6 CUs (57% of peak). Split input channels or use smaller tiles for more
  workgroups.
- **conv0a** (12 input channels, 37% of peak): load-bound; increase pixels per thread for this shape only.
- **fuse `update` into the next block's `prep`**: saves a full-res read/write per block; a few ms.

Method: `bench_gl.py --breakdown` for per-dispatch GPU time, convert to % of peak
(6 CU x 64 x 2 x 1.5 GHz = 1.15 TFLOP/s), and check `MESA_SHADER_CACHE_DISABLE=true AMD_DEBUG=cs,stats,asm` for
VGPRs / waves / spills / instruction mix. Lock clocks
(`echo high > /sys/class/drm/card1/device/power_dpm_force_performance_level`) for stable numbers.

## 4. Portability (medium)

GL 4.3 core + fp32 was chosen for compatibility but has only been verified on one AMD GPU.

- Run `test_gl.py` and `bench_gl.py` on Intel and NVIDIA (Linux and Windows).
- OpenGL ES 3.1 variant (same GLSL plus precision qualifiers) for Raspberry Pi 5 / Android.

## 5. Newer RIFE models (larger)

RIFE 4.7-4.26 add a small feature encoder and wider blocks; noticeably better on hard motion than 4.6.

- Same layer types (3x3 conv, LeakyReLU, deconv + PixelShuffle, warp), so the shaders mostly carry over.
- Needs a weight loader for those models and an updated reference.

## 6. Packaging / integration (larger)

- pip package with a CLI entry point.
- Thin C/Rust host (~300 lines, same GLSL) for a VapourSynth plugin or an ffmpeg filter, no Python dependency.
