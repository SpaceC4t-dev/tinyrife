# tinyrife (CPU)

Single-file C port of RIFE v4.6 (`rife_v46.c`) for CPU inference: fused IFBlocks, Winograd F(4x4)/F(6x6) convolutions, OpenMP.

## Build and run

```
gcc -O3 -march=native -fopenmp rife_v46.c -o rife_v46 -lm
./rife_v46 [--pad 32|64] flownet-v46.bin frame0.png frame1.png 0.5 out.png [iterations]
```

## Benchmark vs. PyTorch

Reference: Practical-RIFE v4.6 graph in stock PyTorch ops (`ref_torch.py` in the repo root, same weights),
torch 2.15.0+cpu from pip (oneDNN enabled), `inference_mode`. Warm inference, mean ms per frame.
Machine: Ryzen 5 4650U (6C/12T, Zen 2). Timings are noisy from clock/thermal swings.

1280x704:

| Threads | PyTorch | C impl | Speedup |
|--------:|--------:|-------:|--------:|
| 1       | 1079    | 472    | 2.3x    |
| 4       | 523     | 179    | 2.9x    |
| 6       | 516     | 148    | 3.5x    |
| 12      | 607     | 151    | 4.0x    |

256x256:

| Threads | PyTorch | C impl | Speedup |
|--------:|--------:|-------:|--------:|
| 4       | 38.8    | 16.2   | 2.4x    |
| 12      | 39.2    | 13.7   | 2.9x    |

Notes:
- PyTorch stops scaling past ~4-6 threads and gets slower at 12 (SMT oversubscription), so the 6-thread row
  is the fair headline comparison (3.5x). The C impl gains slightly from SMT.
- Output matches the torch reference within 1 uint8 level (max diff 1 at 1280x704).
- The distro `python313-torch` (no oneDNN) is slower still; these numbers use the PyPI wheel.
