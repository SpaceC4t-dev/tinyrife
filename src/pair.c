/* Interpolate one frame pair (build/gl on the GPU, build/cpu on the CPU; see rife.h).
 *
 * Usage:  build/gl [-m model_dir] [-t timestep] [-b N] frame0.png frame1.png out.png
 *         (model_dir holds flownet.bin; -b N additionally times N runs, RIFE_PROFILE=1 adds the time per stage)
 */
#include <string.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "rife.h"

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
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
  if (!a || !b) rife_die("cannot read %s", a ? pos[1] : pos[0]);
  if (w != w1 || h != h1) rife_die("frame sizes differ: %dx%d vs %dx%d", w, h, w1, h1);

  Rife *r = rife_new(h, w, model);
  fprintf(stderr, "backend: %s\n", rife_backend(r));
  uint8_t *out = malloc((size_t)h * w * 3);
  if (!out) rife_die("out of memory");
  rife_push(r, a);
  rife_push(r, b);
  rife_interpolate(r, timestep, out);
  if (!stbi_write_png(pos[2], w, h, 3, out, w * 3)) rife_die("cannot write %s", pos[2]);
  printf("wrote %s\n", pos[2]);

  if (bench > 0) {
    double *ms = malloc((size_t)bench * sizeof(double));
    if (!ms) rife_die("out of memory");
    for (int i = 0; i < bench; i++) ms[i] = rife_time_ms(r, timestep);
    qsort(ms, (size_t)bench, sizeof(double), cmp_d);
    double t0 = now_s();
    for (int i = 0; i < bench; i++) {   /* streaming: one push + one interpolation per frame */
      rife_push(r, i & 1 ? a : b);
      rife_interpolate(r, timestep, out);
    }
    double e2e = (now_s() - t0) * 1e3 / bench;
    printf("%dx%d: interpolation %.1f ms median, %.1f ms min (%.2f fps) | end-to-end %.1f ms (%.2f fps)\n", w, h,
           ms[bench / 2], ms[0], 1e3 / ms[bench / 2], e2e, 1e3 / e2e);
    if (getenv("RIFE_PROFILE")) rife_print_profile(r);
    free(ms);
  }
  free(out);
  rife_free(r);
  stbi_image_free(a);
  stbi_image_free(b);
  return 0;
}
