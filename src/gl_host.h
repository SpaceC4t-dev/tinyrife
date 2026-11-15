/* RIFE v4.6 on OpenGL 4.3 compute shaders (host library). One GL context per process; not thread-safe:
 * keep every call on the thread that called gl_init(). Frames are tight RGB24 (h * w * 3 bytes). */
#ifndef GL_HOST_H
#define GL_HOST_H

#include <stddef.h>
#include <stdint.h>

#define RIFE_DEFAULT_MODEL "ext/rife-ncnn-vulkan/models/rife-v4.6"   /* from fetch_model.sh */

typedef struct Rife Rife;

/* Print a message to stderr and exit(1). */
void rife_die(const char *fmt, ...) __attribute__((noreturn, format(printf, 1, 2)));

/* Create / destroy the (surfaceless EGL) OpenGL 4.3 core context. gl_init() exits with a message if there is none. */
void gl_init(void);
void gl_shutdown(void);
const char *gl_renderer(void);

/* A pipeline for h x w frames (padded internally to a multiple of 32); model_dir holds flownet.param/.bin. */
Rife *rife_new(int h, int w, const char *model_dir);
void rife_free(Rife *r);   /* host memory only; GL objects go away with gl_shutdown() */

size_t rife_frame_bytes(const Rife *r);
void rife_padded_size(const Rife *r, int *W, int *H);

/* Upload the next frame. Interpolation is between the last two pushed frames, so each frame is uploaded once. */
void rife_push(Rife *r, const uint8_t *rgb);
/* The frame at `timestep` (0..1) between the last two pushed frames. */
void rife_interpolate(Rife *r, float timestep, uint8_t *rgb_out);
/* GPU time in ms of one interpolation (no readback), for benchmarking. */
double rife_gpu_ms(Rife *r, float timestep);

#endif
