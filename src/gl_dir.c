/* 2x frame interpolation of a PNG sequence with the OpenGL RIFE 4.6 (host library: gl_host.c).
 *
 * Build:  make build/gl_dir
 * Usage:  build/gl_dir [-m model_dir] [-j writer_threads] in_dir out_dir
 * Output: in[0], mid(0,1), in[1], mid(1,2), ..., in[N-1]  ->  2N-1 frames, numbered %08d.png from 1.
 *         The input frames are copied byte for byte; the in-between frames are written with fast PNG compression.
 *
 * Each frame is decoded once and uploaded once; a reader thread decodes ahead and a pool of writer threads encodes and
 * copies behind, so PNG I/O overlaps the GPU. All GL calls stay on the main thread.
 */
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "gl_host.h"
#include "util.h"

#define AHEAD 4            /* frames decoded ahead of the GPU */
#define DEFAULT_WRITERS 4

/* ------------------------------------------------------------------ reader: decode frames in order */

typedef struct {
  uint8_t *rgb;   /* NULL on failure */
  char *err;
} Frame;

typedef struct {
  char **paths;
  int n, w, h;
  Queue *q;
} ReaderArgs;

static void *reader_main(void *p) {
  ReaderArgs *a = p;
  for (int i = 0; i < a->n; i++) {
    Frame *f = calloc(1, sizeof *f);
    if (!f) rife_die("out of memory");
    int w, h, c;
    uint8_t *d = stbi_load(a->paths[i], &w, &h, &c, 3);
    if (!d) {
      f->err = fmt_alloc("cannot read %s: %s", a->paths[i], stbi_failure_reason());
    } else if (w != a->w || h != a->h) {
      f->err = fmt_alloc("size mismatch at %s: %dx%d, expected %dx%d", a->paths[i], w, h, a->w, a->h);
      stbi_image_free(d);
    } else {
      f->rgb = d;
    }
    int failed = f->err != NULL;   /* read before the hand-off: the consumer frees f as soon as it pops it */
    q_push(a->q, f);
    if (failed) break;
  }
  q_close(a->q);
  return NULL;
}

/* ------------------------------------------------------------------ writers: copy an input frame or encode a PNG */

typedef struct {
  char *src;      /* copy src -> dst, or */
  char *dst;
  uint8_t *rgb;   /* encode rgb as a PNG at dst */
} Job;

typedef struct {
  Queue *q;
  int w, h;
} WriterArgs;

static int copy_file(const char *src, const char *dst) {
  int in = open(src, O_RDONLY);
  if (in < 0) return -1;
  int out = open(dst, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (out < 0) { close(in); return -1; }
  static const size_t BUF = 1 << 16;
  char *buf = malloc(BUF);
  int rc = buf ? 0 : -1;
  for (ssize_t n; rc == 0 && (n = read(in, buf, BUF)) != 0;) {
    if (n < 0) { rc = -1; break; }
    for (ssize_t off = 0; off < n;) {
      ssize_t k = write(out, buf + off, (size_t)(n - off));
      if (k < 0) { rc = -1; break; }
      off += k;
    }
  }
  free(buf);
  if (close(out) != 0) rc = -1;
  close(in);
  return rc;
}

static void *writer_main(void *p) {
  WriterArgs *a = p;
  Job *j;
  while ((j = q_pop(a->q)) != NULL) {
    if (j->src) {
      if (copy_file(j->src, j->dst) != 0) set_error(fmt_alloc("cannot copy %s to %s: %s", j->src, j->dst, strerror(errno)));
    } else {
      if (!stbi_write_png(j->dst, a->w, a->h, 3, j->rgb, a->w * 3)) set_error(fmt_alloc("cannot write %s", j->dst));
    }
    free(j->src); free(j->dst); free(j->rgb); free(j);
  }
  return NULL;
}

static void submit(Queue *q, char *src, char *dst, uint8_t *rgb) {
  Job *j = calloc(1, sizeof *j);
  if (!j) rife_die("out of memory");
  j->src = src, j->dst = dst, j->rgb = rgb;
  q_push(q, j);
}

/* ------------------------------------------------------------------ directory helpers */

static int cmp_str(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

/* Sorted paths of the *.png files directly in dir (dotfiles skipped, like a shell glob). */
static char **list_pngs(const char *dir, int *count) {
  DIR *d = opendir(dir);
  if (!d) rife_die("cannot open directory %s: %s", dir, strerror(errno));
  char **paths = NULL;
  int n = 0, cap = 0;
  for (struct dirent *e; (e = readdir(d)) != NULL;) {
    size_t len = strlen(e->d_name);
    if (e->d_name[0] == '.' || len < 5 || strcmp(e->d_name + len - 4, ".png") != 0) continue;
    if (n == cap) {
      cap = cap ? cap * 2 : 64;
      char **grown = (char **)realloc((void *)paths, (size_t)cap * sizeof(char *));
      if (!grown) rife_die("out of memory");
      paths = grown;
    }
    paths[n++] = fmt_alloc("%s/%s", dir, e->d_name);
  }
  closedir(d);
  if (n > 1) qsort((void *)paths, (size_t)n, sizeof(char *), cmp_str);
  *count = n;
  return paths;
}

static void mkdirs(const char *path) {
  char *p = fmt_alloc("%s", path);
  for (char *s = p + 1; *s; s++) {
    if (*s != '/') continue;
    *s = 0;
    if (mkdir(p, 0777) != 0 && errno != EEXIST) rife_die("cannot create %s: %s", p, strerror(errno));
    *s = '/';
  }
  struct stat st;
  if (mkdir(p, 0777) != 0 && errno != EEXIST) rife_die("cannot create %s: %s", p, strerror(errno));
  if (stat(p, &st) != 0 || !S_ISDIR(st.st_mode)) rife_die("%s is not a directory", p);
  free(p);
}

/* ------------------------------------------------------------------ main */

int main(int argc, char **argv) {
  const char *model = RIFE_DEFAULT_MODEL, *pos[2];
  int writers = DEFAULT_WRITERS, np = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-m") && i + 1 < argc) model = argv[++i];
    else if (!strcmp(argv[i], "-j") && i + 1 < argc) writers = atoi(argv[++i]);
    else if (np < 2) pos[np++] = argv[i];
    else np = 3;
  }
  if (np != 2 || writers < 1 || writers > 64)
    rife_die("usage: %s [-m model_dir] [-j writer_threads (1..64)] in_dir out_dir", argv[0]);

  int n;
  char **src = list_pngs(pos[0], &n);
  if (n < 2) rife_die("need at least 2 .png frames in %s (found %d)", pos[0], n);
  int w, h, c;
  if (!stbi_info(src[0], &w, &h, &c)) rife_die("cannot read %s: %s", src[0], stbi_failure_reason());

  char *out_dir = fmt_alloc("%s", pos[1]);
  for (size_t l = strlen(out_dir); l > 1 && out_dir[l - 1] == '/'; l--) out_dir[l - 1] = 0;
  mkdirs(out_dir);
  stbi_write_png_compression_level = 1;   /* the in-between frames are a stream, not an archive */

  gl_init();
  fprintf(stderr, "GL renderer: %s\n", gl_renderer());
  Rife *r = rife_new(h, w, model);
  size_t nbytes = rife_frame_bytes(r);

  Queue q_in, q_out;
  q_init(&q_in, AHEAD);
  q_init(&q_out, 2 * writers + 2);
  ReaderArgs ra = { src, n, w, h, &q_in };
  WriterArgs wa = { &q_out, w, h };
  pthread_t reader, *pool = malloc((size_t)writers * sizeof(pthread_t));
  if (!pool) rife_die("out of memory");
  if (pthread_create(&reader, NULL, reader_main, &ra) != 0) rife_die("cannot start the reader thread");
  for (int i = 0; i < writers; i++)
    if (pthread_create(&pool[i], NULL, writer_main, &wa) != 0) rife_die("cannot start a writer thread");

#define OUT_NAME(k) fmt_alloc("%s/%08d.png", out_dir, (k) + 1)
#define NEXT_FRAME(f)                                                                                                  \
  do {                                                                                                                 \
    (f) = q_pop(&q_in);                                                                                                \
    if (!(f)) rife_die("frame reader stopped early");                                                                  \
    if ((f)->err) rife_die("%s", (f)->err);                                                                            \
  } while (0)

  double t0 = now_s(), gpu_s = 0;
  Frame *f;
  NEXT_FRAME(f);
  rife_push(r, f->rgb);
  stbi_image_free(f->rgb); free(f);
  for (int i = 0; i < n - 1; i++) {
    submit(&q_out, fmt_alloc("%s", src[i]), OUT_NAME(2 * i), NULL);
    NEXT_FRAME(f);
    double tg = now_s();
    rife_push(r, f->rgb);                       /* each frame is uploaded once and reused for the next pair */
    stbi_image_free(f->rgb); free(f);
    uint8_t *mid = malloc(nbytes);
    if (!mid) rife_die("out of memory");
    rife_interpolate(r, 0.5f, mid);
    gpu_s += now_s() - tg;
    submit(&q_out, NULL, OUT_NAME(2 * i + 1), mid);
    char *err = get_error();
    if (err) rife_die("%s", err);
    if ((i + 1) % 50 == 0) {
      double el = now_s() - t0;
      printf("  %d/%d interpolated, %.1fs elapsed, %.2f interp/s\n", i + 1, n - 1, el, (i + 1) / el);
      fflush(stdout);
    }
  }
  submit(&q_out, fmt_alloc("%s", src[n - 1]), OUT_NAME(2 * (n - 1)), NULL);
  q_close(&q_out);
  for (int i = 0; i < writers; i++) pthread_join(pool[i], NULL);
  pthread_join(reader, NULL);
  char *err = get_error();
  if (err) rife_die("%s", err);

  double el = now_s() - t0;
  printf("done: %d -> %d frames (%dx%d) in %.1fs | %.2f interp/s overall, RIFE calls %.0f ms avg (upload+GPU+download)\n",
         n, 2 * n - 1, w, h, el, (n - 1) / el, gpu_s / (n - 1) * 1e3);

  q_destroy(&q_in);
  q_destroy(&q_out);
  free(pool);
  rife_free(r);
  gl_shutdown();
  for (int i = 0; i < n; i++) free(src[i]);
  free((void *)src);
  free(out_dir);
  return 0;
}
