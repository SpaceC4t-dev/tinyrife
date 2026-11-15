/* A stand-in for gl_host.c with the same API (gl_host.h): no GPU, and the "interpolated" frame is the linear blend of
 * the two pushed frames at the requested timestep, so the output of gl_dir and gl_video can be predicted exactly. Used to
 * test their handling and threading without a GL context, and to run them under ThreadSanitizer without the GL driver's
 * own threads in the way. */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gl_host.h"

struct Rife {
  int h, w, count;
  size_t nbytes;
  uint8_t *prev, *last;
};

void rife_die(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
  exit(1);
}

void gl_init(void) {}
void gl_shutdown(void) {}
const char *gl_renderer(void) { return "fake"; }

Rife *rife_new(int h, int w, const char *model_dir) {
  (void)model_dir;
  Rife *r = calloc(1, sizeof *r);
  if (!r) rife_die("out of memory");
  r->h = h, r->w = w, r->nbytes = (size_t)h * w * 3;
  r->prev = malloc(r->nbytes), r->last = malloc(r->nbytes);
  if (!r->prev || !r->last) rife_die("out of memory");
  return r;
}

void rife_free(Rife *r) { free(r->prev); free(r->last); free(r); }
size_t rife_frame_bytes(const Rife *r) { return r->nbytes; }
void rife_padded_size(const Rife *r, int *W, int *H) { *W = (r->w + 31) / 32 * 32, *H = (r->h + 31) / 32 * 32; }

void rife_push(Rife *r, const uint8_t *rgb) {
  uint8_t *t = r->prev; r->prev = r->last; r->last = t;   /* the older slot becomes the newest frame */
  memcpy(r->last, rgb, r->nbytes);
  r->count++;
}

/* prev * (1 - t) + last * t, rounded; tests/check_video.c computes the same expression. At t = 0.5 this is (a + b + 1) / 2. */
void rife_interpolate(Rife *r, float timestep, uint8_t *out) {
  if (r->count < 2) rife_die("push two frames first");
  double t = (double)timestep;
  for (size_t i = 0; i < r->nbytes; i++) out[i] = (uint8_t)((double)r->prev[i] * (1.0 - t) + (double)r->last[i] * t + 0.5);
}

double rife_gpu_ms(Rife *r, float timestep) { (void)r; (void)timestep; return 0; }
