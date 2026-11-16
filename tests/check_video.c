/* Helper for the gl_video tests, which work on raw RGB24 files (see fake_ffmpeg.sh).
 *   check_video gen OUT.raw W H N SEED      N deterministic pseudo-random frames
 *   check_video verify IN.raw OUT.raw W H MULT
 *       OUT must be exactly what gl_video produces with the fake GL host for IN: each frame, then MULT-1 blends at
 *       t = k / MULT, and the last frame MULT times. A partial trailing frame in IN is ignored, like gl_video does. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint8_t *slurp(const char *path, long *len) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END); *len = ftell(f); fseek(f, 0, SEEK_SET);
  uint8_t *b = malloc((size_t)*len + 1);
  if (fread(b, 1, (size_t)*len, f) != (size_t)*len) { fclose(f); free(b); return NULL; }
  fclose(f);
  return b;
}

int main(int argc, char **argv) {
  if (argc == 7 && !strcmp(argv[1], "gen")) {
    long w = atol(argv[3]), h = atol(argv[4]), n = atol(argv[5]);
    uint32_t s = (uint32_t)atol(argv[6]);
    FILE *f = fopen(argv[2], "wb");
    if (!f) return 2;
    for (long i = 0; i < w * h * 3 * n; i++) { s = s * 1664525u + 1013904223u; fputc((int)(s >> 24), f); }
    return fclose(f) != 0;
  }
  if (argc != 7 || strcmp(argv[1], "verify")) { fprintf(stderr, "usage: see the comment at the top of check_video.c\n"); return 2; }
  long w = atol(argv[4]), h = atol(argv[5]), mult = atol(argv[6]), li, lo;
  size_t fb = (size_t)(w * h * 3);
  uint8_t *in = slurp(argv[2], &li), *out = slurp(argv[3], &lo);
  if (!in || !out) { printf("cannot read the files\n"); return 1; }
  long n = li / (long)fb;
  long want = n * mult;
  if (lo != (long)(want * (long)fb)) { printf("output has %ld bytes = %.2f frames, expected %ld frames\n", lo, (double)lo / (double)fb, want); return 1; }
  long k_out = 0, bad = 0;
  for (long i = 0; i < n; i++) {
    const uint8_t *a = in + (size_t)i * fb;
    if (i == n - 1) {
      for (long r = 0; r < mult; r++, k_out++) if (memcmp(out + (size_t)k_out * fb, a, fb)) { printf("frame %ld (repeat of the last frame) differs\n", k_out); bad++; }
      break;
    }
    const uint8_t *b = in + (size_t)(i + 1) * fb;
    if (memcmp(out + (size_t)k_out * fb, a, fb)) { printf("frame %ld (copy of input %ld) differs\n", k_out, i); bad++; }
    k_out++;
    for (long k = 1; k < mult; k++, k_out++) {
      double t = (double)(float)((double)k / (double)mult);   /* the timestep reaches the host as a float */
      const uint8_t *o = out + (size_t)k_out * fb;
      for (size_t j = 0; j < fb; j++)
        if (o[j] != (uint8_t)((double)a[j] * (1.0 - t) + (double)b[j] * t + 0.5)) { printf("frame %ld (blend %ld/%ld of inputs %ld,%ld) differs at byte %zu\n", k_out, k, mult, i, i + 1, j); bad++; break; }
    }
  }
  printf("%s: %ld frames in, %ld out, %ld problems\n", bad ? "FAIL" : "ok", n, want, bad);
  return bad != 0;
}
