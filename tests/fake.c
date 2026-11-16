/* A stand-in backend for rife.h: no model, no GPU, and the "interpolated" frame is the linear blend of the two pushed
 * frames at the requested timestep, so the output of the dir and video tools can be predicted exactly. Used to test their
 * file handling, error paths and threading (also under ThreadSanitizer, without a GL driver's own threads in the way). */
#include <string.h>
#include "rife.h"

struct Rife {
  int count;
  size_t nbytes;
  uint8_t *prev, *last;
};

Rife *rife_new(int h, int w, const char *model_dir) {
  (void)model_dir;
  Rife *r = calloc(1, sizeof *r);
  if (!r) rife_die("out of memory");
  r->nbytes = (size_t)h * w * 3;
  r->prev = malloc(r->nbytes), r->last = malloc(r->nbytes);
  if (!r->prev || !r->last) rife_die("out of memory");
  return r;
}

void rife_free(Rife *r) { free(r->prev); free(r->last); free(r); }
const char *rife_backend(const Rife *r) { (void)r; return "fake"; }

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

double rife_time_ms(Rife *r, float timestep) { (void)r; (void)timestep; return 0; }
void rife_print_profile(const Rife *r) { (void)r; }
