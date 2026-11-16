/* RIFE v4.6 on OpenGL 4.3 compute shaders: the GL backend of rife.h.
 *
 * The GLSL lives in shaders/ (embedded at build time as shaders.h). No GL/EGL headers or link-time dependencies:
 * libEGL is dlopen()ed and every GL entry point comes from eglGetProcAddress.
 */
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rife.h"
#include "shaders.h"

/* ------------------------------------------------------------------ minimal EGL / GL declarations */

typedef void *EGLDisplay, *EGLConfig, *EGLContext, *EGLSurface;
typedef int32_t EGLint;
typedef unsigned EGLBoolean, EGLenum;
typedef unsigned GLuint, GLenum, GLbitfield;
typedef int GLint, GLsizei;
typedef intptr_t GLsizeiptr, GLintptr;
typedef char GLchar;
typedef uint64_t GLuint64;

#define EGL_NONE 0x3038
#define EGL_SURFACE_TYPE 0x3033
#define EGL_PBUFFER_BIT 0x0001
#define EGL_RENDERABLE_TYPE 0x3040
#define EGL_OPENGL_BIT 0x0008
#define EGL_OPENGL_API 0x30A2
#define EGL_CONTEXT_MAJOR_VERSION 0x3098
#define EGL_CONTEXT_MINOR_VERSION 0x30FB
#define EGL_CONTEXT_OPENGL_PROFILE_MASK 0x30FD
#define EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT 0x0001

#define GL_COMPUTE_SHADER 0x91B9
#define GL_COMPILE_STATUS 0x8B81
#define GL_LINK_STATUS 0x8B82
#define GL_INFO_LOG_LENGTH 0x8B84
#define GL_SHADER_STORAGE_BUFFER 0x90D2
#define GL_DYNAMIC_DRAW 0x88E8
#define GL_ALL_BARRIER_BITS 0xFFFFFFFFu
#define GL_TIME_ELAPSED 0x88BF
#define GL_QUERY_RESULT 0x8866
#define GL_RENDERER 0x1F01

#define GL_FUNCS(X)                                                                                                    \
  X(GLuint, CreateShader, (GLenum)) X(void, ShaderSource, (GLuint, GLsizei, const GLchar *const *, const GLint *))     \
  X(void, CompileShader, (GLuint)) X(void, GetShaderiv, (GLuint, GLenum, GLint *))                                     \
  X(void, GetShaderInfoLog, (GLuint, GLsizei, GLsizei *, GLchar *)) X(GLuint, CreateProgram, (void))                   \
  X(void, AttachShader, (GLuint, GLuint)) X(void, LinkProgram, (GLuint))                                               \
  X(void, GetProgramiv, (GLuint, GLenum, GLint *)) X(void, GetProgramInfoLog, (GLuint, GLsizei, GLsizei *, GLchar *))  \
  X(void, DeleteShader, (GLuint)) X(void, UseProgram, (GLuint)) X(GLint, GetUniformLocation, (GLuint, const GLchar *)) \
  X(void, Uniform1f, (GLint, float)) X(void, GenBuffers, (GLsizei, GLuint *))                                          \
  X(void, BindBuffer, (GLenum, GLuint)) X(void, BindBufferBase, (GLenum, GLuint, GLuint))                              \
  X(void, BufferData, (GLenum, GLsizeiptr, const void *, GLenum))                                                      \
  X(void, BufferSubData, (GLenum, GLintptr, GLsizeiptr, const void *))                                                 \
  X(void, GetBufferSubData, (GLenum, GLintptr, GLsizeiptr, void *))                                                    \
  X(void, DispatchCompute, (GLuint, GLuint, GLuint)) X(void, MemoryBarrier, (GLbitfield)) X(void, Finish, (void))      \
  X(void, GenQueries, (GLsizei, GLuint *)) X(void, BeginQuery, (GLenum, GLuint)) X(void, EndQuery, (GLenum))           \
  X(void, GetQueryObjectui64v, (GLuint, GLenum, GLuint64 *)) X(const unsigned char *, GetString, (GLenum))

#define X(ret, name, args) static ret(*gl##name) args;
GL_FUNCS(X)
#undef X

static struct { EGLDisplay dpy; EGLContext ctx; EGLBoolean (*make_current)(EGLDisplay, EGLSurface, EGLSurface, EGLContext);
                EGLBoolean (*destroy_context)(EGLDisplay, EGLContext); EGLBoolean (*terminate)(EGLDisplay); } g_egl;

static void gl_init(void) {
  void *lib = dlopen("libEGL.so.1", RTLD_NOW | RTLD_GLOBAL);
  if (!lib) rife_die("cannot load libEGL.so.1");
  EGLDisplay (*getDisplay)(void *) = dlsym(lib, "eglGetDisplay");
  EGLBoolean (*initialize)(EGLDisplay, EGLint *, EGLint *) = dlsym(lib, "eglInitialize");
  EGLBoolean (*bindAPI)(EGLenum) = dlsym(lib, "eglBindAPI");
  EGLBoolean (*chooseConfig)(EGLDisplay, const EGLint *, EGLConfig *, EGLint, EGLint *) = dlsym(lib, "eglChooseConfig");
  EGLContext (*createContext)(EGLDisplay, EGLConfig, EGLContext, const EGLint *) = dlsym(lib, "eglCreateContext");
  EGLBoolean (*makeCurrent)(EGLDisplay, EGLSurface, EGLSurface, EGLContext) = dlsym(lib, "eglMakeCurrent");
  void *(*getProc)(const char *) = dlsym(lib, "eglGetProcAddress");
  EGLBoolean (*destroyContext)(EGLDisplay, EGLContext) = dlsym(lib, "eglDestroyContext");
  EGLBoolean (*terminate)(EGLDisplay) = dlsym(lib, "eglTerminate");
  if (!getDisplay || !initialize || !bindAPI || !chooseConfig || !createContext || !makeCurrent || !getProc)
    rife_die("libEGL is missing required entry points");

  EGLDisplay dpy = getDisplay(NULL);
  EGLint maj, min;
  if (!dpy || !initialize(dpy, &maj, &min)) rife_die("eglInitialize failed");
  if (!bindAPI(EGL_OPENGL_API)) rife_die("eglBindAPI(OpenGL) failed");
  const EGLint cfg_attr[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE};
  EGLConfig cfg;
  EGLint n = 0;
  if (!chooseConfig(dpy, cfg_attr, &cfg, 1, &n) || n < 1) rife_die("no EGL config");
  const EGLint ctx_attr[] = {EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 3,
                             EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL_NONE};
  EGLContext ctx = createContext(dpy, cfg, NULL, ctx_attr);
  if (!ctx) rife_die("cannot create an OpenGL 4.3 core context");
  if (!makeCurrent(dpy, NULL, NULL, ctx)) rife_die("eglMakeCurrent failed");
  g_egl.dpy = dpy, g_egl.ctx = ctx, g_egl.make_current = makeCurrent, g_egl.destroy_context = destroyContext, g_egl.terminate = terminate;

#define X(ret, name, args)                                                                                             \
  if (!(gl##name = (ret(*) args)getProc("gl" #name))) rife_die("missing GL function gl" #name);
  GL_FUNCS(X)
#undef X
}

/* Release the context so the driver frees its own state (all GL objects go with it). */
static void gl_shutdown(void) {
  if (!g_egl.dpy) return;
  g_egl.make_current(g_egl.dpy, NULL, NULL, NULL);
  if (g_egl.destroy_context) g_egl.destroy_context(g_egl.dpy, g_egl.ctx);
  if (g_egl.terminate) g_egl.terminate(g_egl.dpy);
  g_egl.dpy = NULL;
}

/* ------------------------------------------------------------------ weights */

/* The network is fixed, so the layer shapes are hardcoded and flownet.bin is read in export order (the same way
 * cpu.c does): 4 IFBlocks x (2 stride-2 convs, 8 residual convs, 1 deconv). Each layer is an fp16 tag, the weights as
 * halves (padded to 4 bytes) and an fp32 bias. */
typedef struct {
  int outc, inc, k;
  float *w, *b;
} Layer;

enum { N_BLOCKS = 4, BLOCK_LAYERS = 11, N_LAYERS = N_BLOCKS * BLOCK_LAYERS };
static const int BLOCK_C[N_BLOCKS] = {192, 128, 96, 64};   /* IFBlock width */
static const int BLOCK_CIN[N_BLOCKS] = {7, 12, 12, 12};    /* input channels */
static const int SCALES[N_BLOCKS] = {8, 4, 2, 1};          /* input downscale */

static float half_to_float(uint16_t h) {
  int e = (h >> 10) & 31, m = h & 1023;
  float f = e == 0 ? ldexpf((float)m, -24) : e == 31 ? (m ? NAN : INFINITY) : ldexpf((float)(m | 1024), e - 25);
  return (h >> 15) ? -f : f;
}

static void read_layer(FILE *f, const char *path, Layer *l, int outc, int inc, int k) {
  size_t n = (size_t)outc * inc * k * k, hbytes = (n * 2 + 3) / 4 * 4;
  uint16_t *h = malloc(hbytes);
  l->outc = outc, l->inc = inc, l->k = k;
  l->w = malloc(n * sizeof(float)), l->b = malloc((size_t)outc * sizeof(float));
  if (!h || !l->w || !l->b) rife_die("out of memory");
  uint32_t tag = 0;
  if (fread(&tag, 4, 1, f) != 1 || tag != 0x01306B47u || fread(h, 1, hbytes, f) != hbytes ||
      fread(l->b, sizeof(float), (size_t)outc, f) != (size_t)outc)
    rife_die("%s: truncated weights (is this a RIFE v4.6 flownet.bin?)", path);
  for (size_t i = 0; i < n; i++) l->w[i] = half_to_float(h[i]);
  free(h);
}

static void load_model(const char *dir, Layer *L) {
  char path[1024];
  snprintf(path, sizeof path, "%s/flownet.bin", dir);
  FILE *f = fopen(path, "rb");
  if (!f) rife_die("cannot open %s", path);
  for (int bi = 0; bi < N_BLOCKS; bi++) {
    Layer *B = &L[bi * BLOCK_LAYERS];
    int c = BLOCK_C[bi];
    read_layer(f, path, &B[0], c / 2, BLOCK_CIN[bi], 3);
    read_layer(f, path, &B[1], c, c / 2, 3);
    for (int i = 0; i < 8; i++) read_layer(f, path, &B[2 + i], c, c, 3);
    read_layer(f, path, &B[10], 24, c, 4);   /* deconv: 6 channels x PixelShuffle(2) */
  }
  if (fgetc(f) != EOF) rife_die("%s: unexpected trailing data (is this a RIFE v4.6 flownet.bin?)", path);
  fclose(f);
}

/* ------------------------------------------------------------------ GL helpers */

static int cdiv(int a, int b) { return (a + b - 1) / b; }

static GLuint make_buffer(size_t bytes, const void *data) {
  GLuint b;
  glGenBuffers(1, &b);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, b);
  glBufferData(GL_SHADER_STORAGE_BUFFER, bytes < 16 ? 16 : (GLsizeiptr)bytes, data, GL_DYNAMIC_DRAW);
  return b;
}

typedef struct {
  GLuint prog;
  GLuint buf[4]; /* SSBO bound at binding 0..3 (0 = leave as is) */
  GLuint groups[3];
  GLint timestep_loc; /* -1 if the shader has no timestep uniform */
  int block;          /* IFBlock index and stage name, for the profile */
  const char *name;
  GLuint query;       /* timer query, created by the first rife_time_ms() */
  double ms;          /* GPU time summed over the rife_time_ms() calls */
} Step;

#define MAX_STEPS 256
#define MAX_PROGS 128

struct Rife {
  int h, w, H, W; /* image and padded size */
  size_t nbytes;
  GLuint img[3], out, raw_in, raw_out, flow, mask; /* img[]: 2 live frames + 1 spare, rotated by push() */
  int prev, last, count;
  Step steps[MAX_STEPS];
  int nsteps;
  GLuint unpack, pack;
  struct { char *src; GLuint prog; } progs[MAX_PROGS]; /* compiled programs, keyed by their full source */
  int nprogs;
  int block;  /* IFBlock being built (-1: none), stamped on each added step */
  int timed;  /* number of rife_time_ms() calls */
};

/* Compile `body` with "#define NAME value" lines for each name/value pair (NULL-terminated). */
static GLuint get_prog(Rife *r, const char *body, const char *const *defs, const int *vals) {
  size_t blen = strlen(body), cap = blen + 64 * 32, used = 0;
  char *src = malloc(cap);
  if (!src) rife_die("out of memory");
  used += (size_t)snprintf(src, cap, "#version 430\n#define W %d\n#define H %d\n", r->W, r->H);
  for (int i = 0; defs[i] && used < cap; i++) used += (size_t)snprintf(src + used, cap - used, "#define %s %d\n", defs[i], vals[i]);
  if (used + blen + 1 > cap) rife_die("shader source buffer too small");
  memcpy(src + used, body, blen + 1);
  for (int i = 0; i < r->nprogs; i++)
    if (!strcmp(r->progs[i].src, src)) { free(src); return r->progs[i].prog; }
  GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
  glShaderSource(sh, 1, (const GLchar *const *)&src, NULL);
  glCompileShader(sh);
  GLint ok;
  glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[4096];
    glGetShaderInfoLog(sh, sizeof log, NULL, log);
    rife_die("shader compile failed:\n%s", log);
  }
  GLuint prog = glCreateProgram();
  glAttachShader(prog, sh);
  glLinkProgram(prog);
  glGetProgramiv(prog, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[4096];
    glGetProgramInfoLog(prog, sizeof log, NULL, log);
    rife_die("shader link failed:\n%s", log);
  }
  glDeleteShader(sh);
  if (r->nprogs >= MAX_PROGS) rife_die("program cache full");
  r->progs[r->nprogs].src = src;
  r->progs[r->nprogs++].prog = prog;
  return prog;
}

#define DEFS(...) ((const char *const[]){__VA_ARGS__, NULL})
#define VALS(...) ((const int[]){__VA_ARGS__})

static Step *add_step(Rife *r, const char *name, GLuint prog, unsigned gx, unsigned gy, unsigned gz) {
  if (r->nsteps >= MAX_STEPS) rife_die("too many steps");
  Step *s = &r->steps[r->nsteps++];
  memset(s, 0, sizeof *s);
  s->prog = prog, s->groups[0] = gx, s->groups[1] = gy, s->groups[2] = gz;
  s->block = r->block, s->name = name;
  s->timestep_loc = glGetUniformLocation(prog, "timestep");
  return s;
}

/* wt[oc4][ic4][tap] = mat4, column i = input channel, row j = output channel (std430: column major). */
static float *pack_conv(const Layer *l, int ic_pad) {
  int oc4 = l->outc / 4, ic4 = ic_pad / 4;
  float *wt = calloc((size_t)oc4 * ic4 * 9 * 16, sizeof(float));
  for (int og = 0; og < oc4; og++)
    for (int k = 0; k < ic4; k++)
      for (int tap = 0; tap < 9; tap++)
        for (int i = 0; i < 4; i++)
          for (int j = 0; j < 4; j++) {
            int ic = k * 4 + i;
            if (ic < l->inc) wt[((((size_t)og * ic4 + k) * 9 + tap) * 4 + i) * 4 + j] = l->w[(((size_t)(og * 4 + j) * l->inc + ic) * 9) + tap];
          }
  return wt;
}

/* wt[ky][kx][ic4][oc] = vec4 over 4 input channels. */
static float *pack_deconv(const Layer *l) {
  int ic4 = l->inc / 4;
  float *wt = malloc((size_t)16 * ic4 * l->outc * 4 * sizeof(float));
  for (int ky = 0; ky < 4; ky++)
    for (int kx = 0; kx < 4; kx++)
      for (int k = 0; k < ic4; k++)
        for (int oc = 0; oc < l->outc; oc++)
          for (int i = 0; i < 4; i++)
            wt[((((size_t)ky * 4 + kx) * ic4 + k) * l->outc + oc) * 4 + i] =
                l->w[(((size_t)oc * l->inc + k * 4 + i) * 4 + ky) * 4 + kx];
  return wt;
}

/* ------------------------------------------------------------------ pipeline construction */

enum { PX = 1, OT = 4, LX = 16, LY = 4 }; /* conv tiling: pixels per thread, out-channel groups per workgroup, local size */

static void add_conv(Rife *r, const char *name, GLuint in, const Layer *l, GLuint out, int ic4, int stride, int resid, int hi, int wi, int ho, int wo) {
  int oc4 = l->outc / 4, ot = OT;
  while (oc4 % ot) ot /= 2;
  float *wt = pack_conv(l, ic4 * 4);
  GLuint wbuf = make_buffer((size_t)oc4 * ic4 * 9 * 64, wt), bbuf = make_buffer(l->outc * 4, l->b);
  free(wt);
  GLuint prog = get_prog(r, SRC_CONV, DEFS("IC4", "S", "RESID", "HI", "WI", "HO", "WO", "PX", "OT", "LX", "LY"),
                         VALS(ic4, stride, resid, hi, wi, ho, wo, PX, ot, LX, LY));
  Step *s = add_step(r, name, prog, cdiv(cdiv(wo, PX), LX), cdiv(ho, LY), oc4 / ot);
  s->buf[0] = in, s->buf[1] = wbuf, s->buf[2] = bbuf, s->buf[3] = out;
}

Rife *rife_new(int h, int w, const char *model_dir) {
  gl_init();
  Rife *r = calloc(1, sizeof *r);
  if (!r) rife_die("out of memory");
  r->h = h, r->w = w, r->H = cdiv(h, 32) * 32, r->W = cdiv(w, 32) * 32;
  int H = r->H, W = r->W;
  Layer *L = calloc(N_LAYERS, sizeof(Layer));
  if (!L) rife_die("out of memory");
  load_model(model_dir, L);

  for (int i = 0; i < 3; i++) r->img[i] = make_buffer((size_t)H * W * 4, NULL);
  r->out = make_buffer((size_t)H * W * 4, NULL);
  r->nbytes = (size_t)h * w * 3;
  r->raw_in = make_buffer(cdiv((int)r->nbytes, 4) * 4, NULL);
  r->raw_out = make_buffer((size_t)cdiv(h * w, 4) * 12, NULL);
  r->unpack = get_prog(r, SRC_UNPACK, DEFS("IW", "IH"), VALS(w, h));
  r->pack = get_prog(r, SRC_PACK, DEFS("IW", "IH"), VALS(w, h));
  r->prev = 0, r->last = 1;
  r->flow = make_buffer((size_t)H * W * 16, NULL);
  r->mask = make_buffer((size_t)H * W * 4, NULL);

  for (int bi = 0; bi < N_BLOCKS; bi++) {
    const Layer *B = &L[bi * BLOCK_LAYERS];
    r->block = bi;
    int s = SCALES[bi], first = bi == 0, hs = H / s, ws = W / s, cin4 = first ? 2 : 3;
    GLuint xin = make_buffer((size_t)cin4 * hs * ws * 16, NULL);
    GLuint prep = get_prog(r, SRC_PREP, DEFS("S", "FIRST", "WO", "HO"), VALS(s, first, ws, hs));
    add_step(r, "prep", prep, cdiv(ws, 8), cdiv(hs, 8), 1)->buf[3] = xin;

    GLuint cur = xin;
    int ch = hs, cw = ws, c4 = cin4;
    for (int li = 0; li < 2; li++) { /* conv0: two stride-2 convs */
      int oc4 = B[li].outc / 4, oh = ch / 2, ow = cw / 2;
      GLuint o = make_buffer((size_t)oc4 * oh * ow * 16, NULL);
      add_conv(r, li ? "conv0b" : "conv0a", cur, &B[li], o, c4, 2, 0, ch, cw, oh, ow);
      cur = o, ch = oh, cw = ow, c4 = oc4;
    }
    GLuint pp[2] = {cur, make_buffer((size_t)c4 * ch * cw * 16, NULL)};
    for (int k = 0; k < 8; k++) /* residual convs ping-pong; 8 is even so the result lands back in pp[0] */
      add_conv(r, "res x8", pp[k % 2], &B[2 + k], pp[(k + 1) % 2], c4, 1, 1, ch, cw, ch, cw);

    /* deconv + PixelShuffle -> (flow4, mask) at block input resolution */
    float *dwt = pack_deconv(&B[10]);
    GLuint tmp = make_buffer((size_t)2 * hs * ws * 16, NULL);
    GLuint dprog = get_prog(r, SRC_DECONV, DEFS("IC4", "HI", "WI", "HO", "WO"), VALS(c4, ch, cw, hs, ws));
    Step *ds = add_step(r, "deconv", dprog, cdiv(cw, 8), cdiv(ch, 8), 1);
    ds->buf[0] = pp[0];
    ds->buf[1] = make_buffer((size_t)16 * c4 * B[10].outc * 16, dwt);
    ds->buf[2] = make_buffer(B[10].outc * 4, B[10].b);
    ds->buf[3] = tmp;
    free(dwt);

    GLuint uprog = get_prog(r, SRC_UPDATE, DEFS("S", "FIRST"), VALS(s, first));
    add_step(r, "update", uprog, cdiv(W, 8), cdiv(H, 8), 1)->buf[0] = tmp;
  }
  r->block = -1;
  GLuint fprog = get_prog(r, SRC_FINAL, (const char *const[]){NULL}, (const int[]){0});
  add_step(r, "final", fprog, cdiv(W, 8), cdiv(H, 8), 1)->buf[3] = r->out;

  for (int i = 0; i < N_LAYERS; i++) free(L[i].w), free(L[i].b);
  free(L);
  return r;
}

void rife_free(Rife *r) {
  for (int i = 0; i < r->nprogs; i++) free(r->progs[i].src);
  free(r);
  gl_shutdown();   /* the GL buffers and programs go away with the context */
}

const char *rife_backend(const Rife *r) {
  (void)r;
  return (const char *)glGetString(GL_RENDERER);
}

/* ------------------------------------------------------------------ runtime */

/* Upload the next frame (tight RGB24). Interpolation runs between the last two pushed frames. */
void rife_push(Rife *r, const uint8_t *rgb) {
  int dst = 3 - r->prev - r->last; /* the spare slot becomes the newest frame */
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, r->raw_in);
  glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)r->nbytes, rgb);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, r->raw_in);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, r->img[dst]);
  glUseProgram(r->unpack);
  glDispatchCompute(cdiv(r->W, 8), cdiv(r->H, 8), 1);
  glMemoryBarrier(GL_ALL_BARRIER_BITS);
  r->prev = r->last, r->last = dst;
  r->count++;
}

static void rife_run(Rife *r, float timestep, int timed) {
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, r->img[r->prev]);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, r->img[r->last]);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, r->flow);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, r->mask);
  for (int i = 0; i < r->nsteps; i++) {
    Step *s = &r->steps[i];
    for (int b = 0; b < 4; b++)
      if (s->buf[b]) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, b, s->buf[b]);
    glUseProgram(s->prog);
    if (s->timestep_loc >= 0) glUniform1f(s->timestep_loc, timestep);
    if (timed) {
      if (!s->query) glGenQueries(1, &s->query);
      glBeginQuery(GL_TIME_ELAPSED, s->query);
    }
    glDispatchCompute(s->groups[0], s->groups[1], s->groups[2]);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    if (timed) glEndQuery(GL_TIME_ELAPSED);
  }
}

/* Interpolated frame between the last two pushed frames, as tight RGB24. */
void rife_interpolate(Rife *r, float timestep, uint8_t *rgb_out) {
  if (r->count < 2) rife_die("push two frames first");
  rife_run(r, timestep, 0);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, r->out);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, r->raw_out);
  glUseProgram(r->pack);
  glDispatchCompute(cdiv(cdiv(r->h * r->w, 4), 64), 1, 1);
  glMemoryBarrier(GL_ALL_BARRIER_BITS);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, r->raw_out);
  glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)r->nbytes, rgb_out);
}

/* GPU time of one interpolation between the last two pushed frames: one timer query per step, read back after the
 * last dispatch so that reading them does not stall the pipeline. */
double rife_time_ms(Rife *r, float timestep) {
  if (r->count < 2) rife_die("push two frames first");
  rife_run(r, timestep, 1);
  double total = 0;
  for (int i = 0; i < r->nsteps; i++) {
    GLuint64 ns;
    glGetQueryObjectui64v(r->steps[i].query, GL_QUERY_RESULT, &ns);
    r->steps[i].ms += (double)ns / 1e6;
    total += (double)ns / 1e6;
  }
  r->timed++;
  return total;
}

void rife_print_profile(const Rife *r) {
  for (int i = 0; i < r->nsteps && r->timed;) {
    const Step *s = &r->steps[i];
    double ms = 0;
    for (; i < r->nsteps && r->steps[i].block == s->block && r->steps[i].name == s->name; i++) ms += r->steps[i].ms;
    if (s->block >= 0) printf("  b%d %-8s %7.2f ms\n", s->block, s->name, ms / r->timed);
    else printf("  %-11s %7.2f ms\n", s->name, ms / r->timed);
  }
}
