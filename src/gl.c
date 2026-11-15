/* RIFE v4.6 on OpenGL 4.3 compute shaders: interpolate one frame pair (host library: gl_host.c).
 *
 * Build:  make build/gl
 * Usage:  build/gl [-m model_dir] [-t timestep] [-b N] frame0.png frame1.png out.png
 *         (model_dir holds flownet.param/.bin; -b N additionally times N GPU runs)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "gl_host.h"

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static int cmp_d(const void *a, const void *b) { return (*(const double *)a > *(const double *)b) - (*(const double *)a < *(const double *)b); }

int main(int argc, char **argv) {
  const char *model = RIFE_DEFAULT_MODEL, *pos[3];
  float timestep = 0.5f;
  int bench = 0, np = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-m") && i + 1 < argc) model = argv[++i];
    else if (!strcmp(argv[i], "-t") && i + 1 < argc) timestep = (float)atof(argv[++i]);
    else if (!strcmp(argv[i], "-b") && i + 1 < argc) bench = atoi(argv[++i]);
    else if (np < 3) pos[np++] = argv[i];
    else np = 4;
  }
  if (np != 3) rife_die("usage: %s [-m model_dir] [-t timestep] [-b N] frame0.png frame1.png out.png", argv[0]);

  int w, h, w1, h1, c;
  uint8_t *a = stbi_load(pos[0], &w, &h, &c, 3), *b = stbi_load(pos[1], &w1, &h1, &c, 3);
  if (!a || !b) rife_die("cannot read input images");
  if (w != w1 || h != h1) rife_die("frame sizes differ: %dx%d vs %dx%d", w, h, w1, h1);

  gl_init();
  fprintf(stderr, "GL renderer: %s\n", gl_renderer());
  Rife *r = rife_new(h, w, model);
  uint8_t *out = malloc(rife_frame_bytes(r));
  if (!out) rife_die("out of memory");
  rife_push(r, a);
  rife_push(r, b);
  rife_interpolate(r, timestep, out);
  if (!stbi_write_png(pos[2], w, h, 3, out, w * 3)) rife_die("cannot write %s", pos[2]);
  printf("wrote %s\n", pos[2]);

  if (bench > 0) {
    double *gpu = malloc((size_t)bench * sizeof(double));
    if (!gpu) rife_die("out of memory");
    for (int i = 0; i < bench; i++) gpu[i] = rife_gpu_ms(r, timestep);
    qsort(gpu, (size_t)bench, sizeof(double), cmp_d);
    double t0 = now_ms();
    for (int i = 0; i < bench; i++) {   /* streaming: one upload + one interpolation per frame */
      rife_push(r, i & 1 ? a : b);
      rife_interpolate(r, timestep, out);
    }
    double e2e = (now_ms() - t0) / bench;
    int W, H;
    rife_padded_size(r, &W, &H);
    printf("GL(C) %dx%d (padded %dx%d): gpu %.1f ms (%.2f fps) | end-to-end %.1f ms (%.2f fps)\n", w, h, W, H,
           gpu[bench / 2], 1e3 / gpu[bench / 2], e2e, 1e3 / e2e);
    free(gpu);
  }
  free(out);
  rife_free(r);
  gl_shutdown();
  stbi_image_free(a);
  stbi_image_free(b);
  return 0;
}
