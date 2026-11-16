/* RIFE v4.6 frame interpolation: the API both backends implement (gl.c: OpenGL 4.3 compute shaders, cpu.c: AVX2 +
 * OpenMP). One Rife per process; not thread-safe: keep every call on the thread that called rife_new().
 * Frames are tight RGB24 (h * w * 3 bytes). */
#ifndef RIFE_H
#define RIFE_H

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define RIFE_DEFAULT_MODEL "ext/rife-ncnn-vulkan/models/rife-v4.6"   /* from fetch_model.sh */

typedef struct Rife Rife;

/* Print a message to stderr and exit(1). */
__attribute__((noreturn, format(printf, 1, 2))) static inline void rife_die(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
  exit(1);
}

/* A pipeline for h x w frames; model_dir holds flownet.bin. Exits with a message if it cannot be set up. */
Rife *rife_new(int h, int w, const char *model_dir);
void rife_free(Rife *r);
const char *rife_backend(const Rife *r);   /* what is running it, for logs */

/* Hand over the next frame. Interpolation is between the last two pushed frames, so each frame is converted once. */
void rife_push(Rife *r, const uint8_t *rgb);
/* The frame at `timestep` (0..1) between the last two pushed frames. */
void rife_interpolate(Rife *r, float timestep, uint8_t *rgb_out);

/* For benchmarking: the time in ms of one interpolation without the conversion back to RGB24 (GL: GPU time from timer
 * queries), and the mean time per stage over the rife_time_ms() calls so far, printed to stdout. */
double rife_time_ms(Rife *r, float timestep);
void rife_print_profile(const Rife *r);

#endif
