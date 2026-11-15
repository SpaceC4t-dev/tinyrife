/* Video -> video frame interpolation with the OpenGL RIFE 4.6 (host library: gl_host.c), piping raw RGB24 through ffmpeg
 * (no intermediate files).
 *
 * Build:  make build/gl_video
 * Usage:  build/gl_video [-m model_dir] [-x multiplier] [-t seconds] [--vcodec "libsvtav1 -preset 8 -crf 28"] in out
 *
 * The output has x times the frame rate: each source frame is followed by x-1 interpolated frames (the last frame is
 * repeated x times). -t only processes the first N seconds. Audio is copied when present. Needs ffmpeg and ffprobe on
 * PATH (or named by the FFMPEG / FFPROBE environment variables).
 *
 * A reader thread feeds decoded frames, a writer thread feeds the encoder, and all GL calls stay on the main thread.
 */
// NOLINTNEXTLINE(cert-dcl51-cpp) the standard feature-test macro, needed for pipe2 and F_SETPIPE_SZ
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "gl_host.h"
#include "util.h"

#define QUEUE_FRAMES 8
#define PIPE_BYTES (1 << 20)

/* ------------------------------------------------------------------ child processes */

typedef struct {
  pid_t pid;
  int fd;   /* our end of the pipe: the child's stdout (read) or stdin (write) */
} Proc;

enum { CHILD_STDOUT, CHILD_STDIN };

/* Run argv with its stdout (CHILD_STDOUT) or stdin (CHILD_STDIN) connected to a pipe. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wanalyzer-fd-leak" /* the dup2() result is the stdin/stdout of the program exec'd below */
static Proc spawn(char *const argv[], int which) {
  int p[2];
  if (pipe2(p, O_CLOEXEC) != 0) rife_die("pipe: %s", strerror(errno));
  pid_t pid = fork();
  if (pid < 0) rife_die("fork: %s", strerror(errno));
  if (pid == 0) {
    /* dup2 clears close-on-exec on the new descriptor; every other pipe end closes at exec */
    if (dup2(which == CHILD_STDOUT ? p[1] : p[0], which == CHILD_STDOUT ? 1 : 0) < 0) _exit(127);
    /* no shell is involved: file names and codec options reach ffmpeg as separate arguments, which is the point of the tool */
    // NOLINTNEXTLINE(clang-analyzer-optin.taint.GenericTaint)
    execvp(argv[0], argv);
    fprintf(stderr, "cannot run %s: %s\n", argv[0], strerror(errno));
    _exit(127);
  }
  Proc proc = { pid, which == CHILD_STDOUT ? p[0] : p[1] };
  close(which == CHILD_STDOUT ? p[1] : p[0]);
  fcntl(proc.fd, F_SETPIPE_SZ, PIPE_BYTES);   /* bigger pipe, fewer wake-ups; fine if refused */
  return proc;
}
#pragma GCC diagnostic pop

/* Wait for the child; its exit status, or 128 + signal. */
static int finish(Proc p) {
  int st = 0;
  while (waitpid(p.pid, &st, 0) < 0 && errno == EINTR) {}
  return WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
}

/* Exit status of a child that should already have exited, waiting up to ~2 s; -1 if it is still running. */
static int wait_brief(Proc p) {
  for (int i = 0; i < 200; i++) {
    int st = 0;
    pid_t r = waitpid(p.pid, &st, WNOHANG);
    if (r == p.pid) return WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
    if (r < 0 && errno != EINTR) return -1;
    usleep(10000);
  }
  return -1;
}

/* A write to the encoder failed: report why the encoder is gone (its own message is on stderr above). */
static void die_encoder(Proc enc, const char *err) {
  int rc = wait_brief(enc);
  if (rc > 0) rife_die("the encoder failed (ffmpeg exit status %d; see its message above)", rc);
  rife_die("%s", err);
}

static const char *prog(const char *env, const char *dflt) {
  const char *v = getenv(env);
  return v && *v ? v : dflt;
}

/* Width, height and frame rate (num/den) of the first video stream. */
static void probe(const char *input, int *w, int *h, long *num, long *den) {
  char *argv[] = { (char *)prog("FFPROBE", "ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                   "stream=width,height,r_frame_rate", "-of", "default=noprint_wrappers=1", (char *)input, NULL };
  Proc p = spawn(argv, CHILD_STDOUT);
  char buf[1024];
  size_t n = 0;
  for (ssize_t k; n < sizeof buf - 1 && (k = read(p.fd, buf + n, sizeof buf - 1 - n)) != 0;) {
    if (k < 0) { if (errno == EINTR) continue; break; }
    n += (size_t)k;
  }
  buf[n] = 0;
  close(p.fd);
  int rc = finish(p);
  *w = *h = 0, *num = *den = 0;
  for (char *line = strtok(buf, "\n"); line; line = strtok(NULL, "\n")) {
    sscanf(line, "width=%d", w);
    sscanf(line, "height=%d", h);
    sscanf(line, "r_frame_rate=%ld/%ld", num, den);
  }
  if (rc != 0 || *w <= 0 || *h <= 0 || *num <= 0 || *den <= 0)
    rife_die("cannot read the video stream of %s (ffprobe %s)", input, rc ? "failed" : "gave no usable size / frame rate");
}

/* ------------------------------------------------------------------ reader and writer threads */

typedef struct {
  int fd;
  size_t nbytes;
  Queue *q;
} ReaderArgs;

/* Exactly n bytes, or fewer at end of file. */
static size_t read_full(int fd, uint8_t *buf, size_t n) {
  size_t got = 0;
  while (got < n) {
    ssize_t k = read(fd, buf + got, n - got);
    if (k < 0) { if (errno == EINTR) continue; break; }
    if (k == 0) break;
    got += (size_t)k;
  }
  return got;
}

static void *reader_main(void *p) {
  ReaderArgs *a = p;
  for (;;) {
    uint8_t *f = malloc(a->nbytes);
    if (!f) { set_error(fmt_alloc("out of memory")); break; }
    if (read_full(a->fd, f, a->nbytes) != a->nbytes) { free(f); break; }   /* end of video (a partial frame is dropped) */
    q_push(a->q, f);
  }
  q_close(a->q);
  return NULL;
}

typedef struct {
  int fd;
  size_t nbytes;
  Queue *q;
} WriterArgs;

static void *writer_main(void *p) {
  WriterArgs *a = p;
  int failed = 0;
  uint8_t *f;
  while ((f = q_pop(a->q)) != NULL) {
    for (size_t off = 0; !failed && off < a->nbytes;) {
      ssize_t k = write(a->fd, f + off, a->nbytes - off);
      if (k < 0) {
        if (errno == EINTR) continue;
        set_error(fmt_alloc("cannot write to the encoder: %s", strerror(errno)));
        failed = 1;   /* keep draining so the producer never blocks on a full queue */
        break;
      }
      off += (size_t)k;
    }
    free(f);
  }
  close(a->fd);   /* the encoder sees end of input */
  return NULL;
}

/* ------------------------------------------------------------------ main */

/* Split a command-line style string on whitespace into args[]; returns the count. */
static int split_words(char *s, char **args, int max) {
  int n = 0;
  for (char *tok = strtok(s, " \t"); tok && n < max; tok = strtok(NULL, " \t")) args[n++] = tok;
  return n;
}

int main(int argc, char **argv) {
  const char *model = RIFE_DEFAULT_MODEL, *pos[2], *vcodec = "libsvtav1 -preset 8 -crf 28";
  double duration = 0;
  int mult = 2, np = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-m") && i + 1 < argc) model = argv[++i];
    else if (!strcmp(argv[i], "-x") && i + 1 < argc) mult = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-t") && i + 1 < argc) duration = atof(argv[++i]);
    else if (!strcmp(argv[i], "--vcodec") && i + 1 < argc) vcodec = argv[++i];
    else if (np < 2) pos[np++] = argv[i];
    else np = 3;
  }
  if (np != 2 || mult < 1 || mult > 16 || duration < 0)
    rife_die("usage: %s [-m model_dir] [-x multiplier (1..16)] [-t seconds] [--vcodec \"codec options\"] in out", argv[0]);
  signal(SIGPIPE, SIG_IGN);   /* a dead encoder shows up as a write error, not a signal */

  int w, h;
  long num, den;
  probe(pos[0], &w, &h, &num, &den);
  size_t nbytes = (size_t)w * h * 3;

  char tbuf[32], rate[64], size[32];
  snprintf(tbuf, sizeof tbuf, "%g", duration);
  snprintf(rate, sizeof rate, "%ld/%ld", num * mult, den);
  snprintf(size, sizeof size, "%dx%d", w, h);
  const char *ffmpeg = prog("FFMPEG", "ffmpeg");

  char *dec_argv[16]; int d = 0;
  dec_argv[d++] = (char *)ffmpeg; dec_argv[d++] = "-v"; dec_argv[d++] = "error";
  if (duration > 0) { dec_argv[d++] = "-t"; dec_argv[d++] = tbuf; }
  dec_argv[d++] = "-i"; dec_argv[d++] = (char *)pos[0];
  dec_argv[d++] = "-f"; dec_argv[d++] = "rawvideo"; dec_argv[d++] = "-pix_fmt"; dec_argv[d++] = "rgb24"; dec_argv[d++] = "-";
  dec_argv[d] = NULL;

  char *codec = fmt_alloc("%s", vcodec), *codec_args[32];
  int nc = split_words(codec, codec_args, 32);
  char *enc_argv[64]; int e = 0;
  enc_argv[e++] = (char *)ffmpeg; enc_argv[e++] = "-v"; enc_argv[e++] = "error"; enc_argv[e++] = "-y";
  enc_argv[e++] = "-f"; enc_argv[e++] = "rawvideo"; enc_argv[e++] = "-pix_fmt"; enc_argv[e++] = "rgb24";
  enc_argv[e++] = "-s"; enc_argv[e++] = size; enc_argv[e++] = "-r"; enc_argv[e++] = rate; enc_argv[e++] = "-i"; enc_argv[e++] = "-";
  if (duration > 0) { enc_argv[e++] = "-t"; enc_argv[e++] = tbuf; }
  enc_argv[e++] = "-i"; enc_argv[e++] = (char *)pos[0];
  enc_argv[e++] = "-map"; enc_argv[e++] = "0:v:0"; enc_argv[e++] = "-map"; enc_argv[e++] = "1:a?"; enc_argv[e++] = "-c:a"; enc_argv[e++] = "copy";
  enc_argv[e++] = "-c:v";
  for (int i = 0; i < nc; i++) enc_argv[e++] = codec_args[i];
  enc_argv[e++] = "-pix_fmt"; enc_argv[e++] = "yuv420p"; enc_argv[e++] = (char *)pos[1];
  enc_argv[e] = NULL;

  gl_init();
  fprintf(stderr, "GL renderer: %s\n", gl_renderer());
  Rife *r = rife_new(h, w, model);

  Proc dec = spawn(dec_argv, CHILD_STDOUT), enc = spawn(enc_argv, CHILD_STDIN);
  Queue q_in, q_out;
  q_init(&q_in, QUEUE_FRAMES);
  q_init(&q_out, QUEUE_FRAMES);
  ReaderArgs ra = { dec.fd, nbytes, &q_in };
  WriterArgs wa = { enc.fd, nbytes, &q_out };
  pthread_t reader, writer;
  if (pthread_create(&reader, NULL, reader_main, &ra) != 0 || pthread_create(&writer, NULL, writer_main, &wa) != 0)
    rife_die("cannot start the I/O threads");

  double t0 = now_s(), gpu_s = 0;
  long n_in = 0, n_interp = 0;
  uint8_t *prev = NULL, *frame;
  while ((frame = q_pop(&q_in)) != NULL) {
    rife_push(r, frame);
    n_in++;
    if (prev) {
      q_push(&q_out, prev);   /* the writer owns it now */
      for (int k = 1; k < mult; k++) {
        uint8_t *mid = malloc(nbytes);
        if (!mid) rife_die("out of memory");
        double tg = now_s();
        rife_interpolate(r, (float)((double)k / mult), mid);
        gpu_s += now_s() - tg;
        q_push(&q_out, mid);
        n_interp++;
      }
      char *err = get_error();
      if (err) die_encoder(enc, err);
      if (n_in % 50 == 0) {
        double el = now_s() - t0;
        printf("  %ld frames in, %ld interpolated, %.2f interp/s\n", n_in, n_interp, n_interp / el);
        fflush(stdout);
      }
    }
    prev = frame;
  }
  if (!prev) rife_die("no frames could be decoded from %s", pos[0]);
  for (int i = 0; i < mult; i++) {   /* the last frame has nothing to interpolate towards: repeat it */
    uint8_t *f = prev;
    if (i < mult - 1) {
      f = malloc(nbytes);
      if (!f) rife_die("out of memory");
      memcpy(f, prev, nbytes);
    }
    q_push(&q_out, f);
  }
  q_close(&q_out);
  pthread_join(writer, NULL);
  pthread_join(reader, NULL);
  close(dec.fd);
  char *err = get_error();
  if (err) die_encoder(enc, err);
  int dec_rc = finish(dec), enc_rc = finish(enc);

  double el = now_s() - t0;
  printf("done: %ld -> %ld frames (%dx%d, %g -> %g fps) in %.1fs | %.2f interp/s overall, %.0f ms per interpolation\n",
         n_in, n_in * mult, w, h, (double)num / den, (double)(num * mult) / den, el, n_interp / el,
         gpu_s / (n_interp > 0 ? (double)n_interp : 1.0) * 1e3);
  if (dec_rc != 0) fprintf(stderr, "warning: the decoder exited with status %d; the output may be cut short\n", dec_rc);
  if (enc_rc != 0) rife_die("the encoder failed (ffmpeg exit status %d; see its message above)", enc_rc);

  q_destroy(&q_in);
  q_destroy(&q_out);
  free(codec);
  rife_free(r);
  gl_shutdown();
  return 0;
}
