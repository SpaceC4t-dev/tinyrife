/* check_dir in_dir out_dir [avg]: verify the output of gl_dir for a sequence.
 * Always: 2N-1 files named 00000001.png.., the even-numbered ones (1st, 3rd, ...) byte-identical to the inputs in order.
 * With "avg" (fake GL host): every in-between frame is exactly (a + b + 1) / 2 of its neighbours. */
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static int cmp_str(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

static char **list(const char *dir, int *n) {
  DIR *d = opendir(dir);
  if (!d) { fprintf(stderr, "cannot open %s\n", dir); exit(2); }
  char **v = NULL; int cnt = 0;
  for (struct dirent *e; (e = readdir(d)) != NULL;) {
    size_t l = strlen(e->d_name);
    if (e->d_name[0] == '.' || l < 5 || strcmp(e->d_name + l - 4, ".png") != 0) continue;
    v = realloc(v, (size_t)(cnt + 1) * sizeof *v);
    v[cnt] = malloc(strlen(dir) + l + 2);
    sprintf(v[cnt++], "%s/%s", dir, e->d_name);
  }
  closedir(d);
  if (cnt > 1) qsort(v, (size_t)cnt, sizeof *v, cmp_str);
  *n = cnt;
  return v;
}

static unsigned char *slurp(const char *path, long *len) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END); *len = ftell(f); fseek(f, 0, SEEK_SET);
  unsigned char *b = malloc((size_t)*len + 1);
  if (fread(b, 1, (size_t)*len, f) != (size_t)*len) { fclose(f); free(b); return NULL; }
  fclose(f);
  return b;
}

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s in_dir out_dir [avg]\n", argv[0]); return 2; }
  int avg = argc > 3 && !strcmp(argv[3], "avg"), n_in, n_out, bad = 0;
  char **in = list(argv[1], &n_in), **out = list(argv[2], &n_out);
  if (n_out != 2 * n_in - 1) { printf("expected %d output frames, found %d\n", 2 * n_in - 1, n_out); return 1; }
  for (int k = 0; k < n_out; k++) {
    char want[16];
    snprintf(want, sizeof want, "%08d.png", k + 1);
    if (strcmp(strrchr(out[k], '/') + 1, want) != 0) { printf("output %d is %s, expected %s\n", k, out[k], want); bad++; }
  }
  for (int k = 0; k < n_out; k += 2) {
    long la, lb; unsigned char *a = slurp(in[k / 2], &la), *b = slurp(out[k], &lb);
    if (!a || !b || la != lb || memcmp(a, b, (size_t)la) != 0) { printf("frame %d is not a byte-identical copy of %s\n", k + 1, in[k / 2]); bad++; }
    free(a); free(b);
  }
  if (avg) {
    for (int k = 1; k < n_out; k += 2) {
      int w, h, c, w2, h2, w3, h3;
      unsigned char *a = stbi_load(out[k - 1], &w, &h, &c, 3), *m = stbi_load(out[k], &w2, &h2, &c, 3), *b = stbi_load(out[k + 1], &w3, &h3, &c, 3);
      if (!a || !m || !b || w != w2 || w != w3 || h != h2 || h != h3) { printf("cannot compare frames around %d\n", k + 1); bad++; continue; }
      long wrong = 0;
      for (long i = 0; i < (long)w * h * 3; i++) wrong += m[i] != ((a[i] + b[i] + 1) >> 1);
      if (wrong) { printf("in-between frame %d: %ld values are not the average of its neighbours\n", k + 1, wrong); bad++; }
      stbi_image_free(a); stbi_image_free(m); stbi_image_free(b);
    }
  }
  printf("%s: %d frames in, %d out, %d problems\n", bad ? "FAIL" : "ok", n_in, n_out, bad);
  return bad != 0;
}
