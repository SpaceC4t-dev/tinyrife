/*
 * RIFE v4.6 - single-file C99/C11 CPU inference
 *   Input:  two RGB images (HxWx3, uint8) + timestep in [0,1]
 *   Output: interpolated RGB image (HxWx3, uint8)
 *
 * Build:  gcc -O3 -march=native -fopenmp rife_v46.c -o rife_v46 -lm
 * Run:    ./rife_v46 [--pad 32|64] flownet-v46.bin frame0.png frame1.png 0.5 out.png [iterations]
 *
 * --pad sets the multiple the frame is zero-padded to (default 32, as in
 * rife-ncnn-vulkan). --pad 64 matches Practical-RIFE's inference scripts.
 *
 * The network is fixed, so it is hardcoded rather than interpreted from the
 * ncnn .param file: 4 IFBlocks (channels 192/128/96/64 at 1/8, 1/4, 1/2, 1/1
 * scale) refine a flow field and a blend mask, and a final merge warps both
 * frames and blends them. The .bin is read straight into the layouts the
 * kernels want (fp16 weights, layer order as written by the ncnn export), and
 * its size is checked against the architecture.
 *
 *   net_forward()          the whole pipeline, one function (read this first)
 *   ifblock_*()            one IFBlock: 2 strided convs, 8 residual convs
 *                          (Winograd), deconv+PixelShuffle, in a packed layout
 *   warpdown / merge       fused warp+downscale and fused final warp+blend
 *
 * AVX2/FMA throughout, persistent workspaces, pre-reordered weights, no
 * external dependencies.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>

#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_get_thread_num(void) { return 0; }
static inline int omp_get_max_threads(void) { return 1; }
#endif

#define RIFE_SLOPE 0.2f   /* LeakyReLU slope of every activation in v4.6 */

/* =====================  TENSORS  ===================== */
/* Planar CHW float tensors. A padded tensor carries a 1-pixel zero border per
 * channel. A "view" is a run of consecutive channels of another tensor. */

typedef struct {
    int w, h, c;
    float* data;
    int is_alias;
    int is_padded;
} Tensor;

static Tensor tensor_make(int w, int h, int c) {
    Tensor t;
    t.w = w; t.h = h; t.c = c;
    t.is_alias = 0;
    t.is_padded = 0;
    size_t sz = (size_t)w * h * c * sizeof(float);
    size_t alloc_sz = (sz + 63) & ~63;
    t.data = (float*)aligned_alloc(64, alloc_sz ? alloc_sz : 64);
    assert(t.data != NULL);
    memset(t.data, 0, sz);
    return t;
}

static Tensor tensor_make_padded(int w, int h, int c) {
    Tensor t;
    t.w = w; t.h = h; t.c = c;
    t.is_alias = 0;
    t.is_padded = 1;
    size_t sz = (size_t)c * (h + 2) * (w + 2) * sizeof(float);
    size_t alloc_sz = (sz + 63) & ~63;
    t.data = (float*)aligned_alloc(64, alloc_sz ? alloc_sz : 64);
    assert(t.data != NULL);
    memset(t.data, 0, sz);
    return t;
}

static void tensor_free(Tensor* t) {
    if (t->data && !t->is_alias) {
        free(t->data);
        t->data = NULL;
    }
}

/* Channels [c0, c0 + nc) of t, sharing its memory. */
static Tensor tensor_view(const Tensor* t, int c0, int nc) {
    size_t plane = t->is_padded ? (size_t)(t->h + 2) * (t->w + 2) : (size_t)t->w * t->h;
    Tensor v = *t;
    v.data = t->data + (size_t)c0 * plane;
    v.c = nc;
    v.is_alias = 1;
    return v;
}

/* =====================  ELEMENTWISE / RESAMPLING KERNELS  ===================== */

/* Deconv (K=4, S=2, P=1, 24 outputs) from a channels-last input, written through a fused
 * PixelShuffle(2) as 6 channels at 2x the deconv output size. */
static void deconv_core(const float* wt_reord, const float* B, const float* in_hwc, int W, int H, int C, Tensor* out) {
    int OW = out->w / 2, OH = out->h / 2;

    __m256 b0 = _mm256_loadu_ps(B + 0);
    __m256 b1 = _mm256_loadu_ps(B + 8);
    __m256 b2 = _mm256_loadu_ps(B + 16);

    // 2-OW unrolled + 4-channel SIMD deconvolution
    #pragma omp parallel for schedule(static)
    for (int oh = 0; oh < OH; oh++) {
        int kh_start = (oh + 1) & 1;
        float row_buf[OW * 24];

        for (int parity = 0; parity < 2; parity++) {
            int ow_start = parity;
            int kw_start = (ow_start + 1) & 1;
            int ow = ow_start;

            for (; ow + 2 < OW; ow += 4) {
                int ow0 = ow;
                int ow1 = ow + 2;

                __m256 s0_0 = b0, s1_0 = b1, s2_0 = b2;
                __m256 s0_1 = b0, s1_1 = b1, s2_1 = b2;

                for (int kh = kh_start; kh < 4; kh += 2) {
                    int ih = (oh + 1 - kh) >> 1;
                    if ((unsigned)ih >= (unsigned)H) continue;
                    const float* in_row = in_hwc + (size_t)ih * W * C;

                    for (int kw = kw_start; kw < 4; kw += 2) {
                        int iw0 = (ow0 + 1 - kw) >> 1;
                        int iw1 = (ow1 + 1 - kw) >> 1;

                        const float* wt_p = wt_reord + ((size_t)(kh * 4 + kw) * C) * 24;
                        const float* in0 = ((unsigned)iw0 < (unsigned)W) ? (in_row + (size_t)iw0 * C) : NULL;
                        const float* in1 = ((unsigned)iw1 < (unsigned)W) ? (in_row + (size_t)iw1 * C) : NULL;

                        if (in0 && in1) {
                            for (int c = 0; c < C; c += 4) {
                                __m256 w0 = _mm256_load_ps(wt_p + 0);
                                __m256 w1 = _mm256_load_ps(wt_p + 8);
                                __m256 w2 = _mm256_load_ps(wt_p + 16);
                                __m256 v0_0 = _mm256_set1_ps(in0[c + 0]);
                                __m256 v0_1 = _mm256_set1_ps(in1[c + 0]);
                                s0_0 = _mm256_fmadd_ps(v0_0, w0, s0_0);
                                s1_0 = _mm256_fmadd_ps(v0_0, w1, s1_0);
                                s2_0 = _mm256_fmadd_ps(v0_0, w2, s2_0);
                                s0_1 = _mm256_fmadd_ps(v0_1, w0, s0_1);
                                s1_1 = _mm256_fmadd_ps(v0_1, w1, s1_1);
                                s2_1 = _mm256_fmadd_ps(v0_1, w2, s2_1);

                                __m256 w3 = _mm256_load_ps(wt_p + 24);
                                __m256 w4 = _mm256_load_ps(wt_p + 32);
                                __m256 w5 = _mm256_load_ps(wt_p + 40);
                                __m256 v1_0 = _mm256_set1_ps(in0[c + 1]);
                                __m256 v1_1 = _mm256_set1_ps(in1[c + 1]);
                                s0_0 = _mm256_fmadd_ps(v1_0, w3, s0_0);
                                s1_0 = _mm256_fmadd_ps(v1_0, w4, s1_0);
                                s2_0 = _mm256_fmadd_ps(v1_0, w5, s2_0);
                                s0_1 = _mm256_fmadd_ps(v1_1, w3, s0_1);
                                s1_1 = _mm256_fmadd_ps(v1_1, w4, s1_1);
                                s2_1 = _mm256_fmadd_ps(v1_1, w5, s2_1);

                                __m256 w6 = _mm256_load_ps(wt_p + 48);
                                __m256 w7 = _mm256_load_ps(wt_p + 56);
                                __m256 w8 = _mm256_load_ps(wt_p + 64);
                                __m256 v2_0 = _mm256_set1_ps(in0[c + 2]);
                                __m256 v2_1 = _mm256_set1_ps(in1[c + 2]);
                                s0_0 = _mm256_fmadd_ps(v2_0, w6, s0_0);
                                s1_0 = _mm256_fmadd_ps(v2_0, w7, s1_0);
                                s2_0 = _mm256_fmadd_ps(v2_0, w8, s2_0);
                                s0_1 = _mm256_fmadd_ps(v2_1, w6, s0_1);
                                s1_1 = _mm256_fmadd_ps(v2_1, w7, s1_1);
                                s2_1 = _mm256_fmadd_ps(v2_1, w8, s2_1);

                                __m256 w9 = _mm256_load_ps(wt_p + 72);
                                __m256 w10 = _mm256_load_ps(wt_p + 80);
                                __m256 w11 = _mm256_load_ps(wt_p + 88);
                                __m256 v3_0 = _mm256_set1_ps(in0[c + 3]);
                                __m256 v3_1 = _mm256_set1_ps(in1[c + 3]);
                                s0_0 = _mm256_fmadd_ps(v3_0, w9, s0_0);
                                s1_0 = _mm256_fmadd_ps(v3_0, w10, s1_0);
                                s2_0 = _mm256_fmadd_ps(v3_0, w11, s2_0);
                                s0_1 = _mm256_fmadd_ps(v3_1, w9, s0_1);
                                s1_1 = _mm256_fmadd_ps(v3_1, w10, s1_1);
                                s2_1 = _mm256_fmadd_ps(v3_1, w11, s2_1);

                                wt_p += 96;
                            }
                        } else if (in0) {
                            for (int c = 0; c < C; c += 2) {
                                __m256 w0 = _mm256_load_ps(wt_p + 0);
                                __m256 w1 = _mm256_load_ps(wt_p + 8);
                                __m256 w2 = _mm256_load_ps(wt_p + 16);
                                __m256 v0_0 = _mm256_set1_ps(in0[c + 0]);

                                s0_0 = _mm256_fmadd_ps(v0_0, w0, s0_0);
                                s1_0 = _mm256_fmadd_ps(v0_0, w1, s1_0);
                                s2_0 = _mm256_fmadd_ps(v0_0, w2, s2_0);

                                __m256 w3 = _mm256_load_ps(wt_p + 24);
                                __m256 w4 = _mm256_load_ps(wt_p + 32);
                                __m256 w5 = _mm256_load_ps(wt_p + 40);
                                __m256 v1_0 = _mm256_set1_ps(in0[c + 1]);

                                s0_0 = _mm256_fmadd_ps(v1_0, w3, s0_0);
                                s1_0 = _mm256_fmadd_ps(v1_0, w4, s1_0);
                                s2_0 = _mm256_fmadd_ps(v1_0, w5, s2_0);

                                wt_p += 48;
                            }
                        } else if (in1) {
                            for (int c = 0; c < C; c += 2) {
                                __m256 w0 = _mm256_load_ps(wt_p + 0);
                                __m256 w1 = _mm256_load_ps(wt_p + 8);
                                __m256 w2 = _mm256_load_ps(wt_p + 16);
                                __m256 v0_1 = _mm256_set1_ps(in1[c + 0]);

                                s0_1 = _mm256_fmadd_ps(v0_1, w0, s0_1);
                                s1_1 = _mm256_fmadd_ps(v0_1, w1, s1_1);
                                s2_1 = _mm256_fmadd_ps(v0_1, w2, s2_1);

                                __m256 w3 = _mm256_load_ps(wt_p + 24);
                                __m256 w4 = _mm256_load_ps(wt_p + 32);
                                __m256 w5 = _mm256_load_ps(wt_p + 40);
                                __m256 v1_1 = _mm256_set1_ps(in1[c + 1]);

                                s0_1 = _mm256_fmadd_ps(v1_1, w3, s0_1);
                                s1_1 = _mm256_fmadd_ps(v1_1, w4, s1_1);
                                s2_1 = _mm256_fmadd_ps(v1_1, w5, s2_1);

                                wt_p += 48;
                            }
                        }
                    }
                }
                _mm256_storeu_ps(row_buf + ow0 * 24 + 0,  s0_0);
                _mm256_storeu_ps(row_buf + ow0 * 24 + 8,  s1_0);
                _mm256_storeu_ps(row_buf + ow0 * 24 + 16, s2_0);

                _mm256_storeu_ps(row_buf + ow1 * 24 + 0,  s0_1);
                _mm256_storeu_ps(row_buf + ow1 * 24 + 8,  s1_1);
                _mm256_storeu_ps(row_buf + ow1 * 24 + 16, s2_1);
            }

            // Remainder for this parity
            for (; ow < OW; ow += 2) {
                __m256 s0 = b0, s1 = b1, s2 = b2;
                for (int kh = kh_start; kh < 4; kh += 2) {
                    int ih = (oh + 1 - kh) >> 1;
                    if ((unsigned)ih >= (unsigned)H) continue;
                    const float* in_row = in_hwc + (size_t)ih * W * C;

                    for (int kw = kw_start; kw < 4; kw += 2) {
                        int iw = (ow + 1 - kw) >> 1;
                        if ((unsigned)iw >= (unsigned)W) continue;

                        const float* wt_p = wt_reord + ((size_t)(kh * 4 + kw) * C) * 24;
                        const float* in_p = in_row + (size_t)iw * C;
                        for (int c = 0; c < C; c += 2) {
                            __m256 v0 = _mm256_set1_ps(in_p[c + 0]);
                            __m256 v1 = _mm256_set1_ps(in_p[c + 1]);

                            s0 = _mm256_fmadd_ps(v0, _mm256_load_ps(wt_p + 0),  s0);
                            s1 = _mm256_fmadd_ps(v0, _mm256_load_ps(wt_p + 8),  s1);
                            s2 = _mm256_fmadd_ps(v0, _mm256_load_ps(wt_p + 16), s2);

                            s0 = _mm256_fmadd_ps(v1, _mm256_load_ps(wt_p + 24), s0);
                            s1 = _mm256_fmadd_ps(v1, _mm256_load_ps(wt_p + 32), s1);
                            s2 = _mm256_fmadd_ps(v1, _mm256_load_ps(wt_p + 40), s2);

                            wt_p += 48;
                        }
                    }
                }
                _mm256_storeu_ps(row_buf + ow * 24 + 0,  s0);
                _mm256_storeu_ps(row_buf + ow * 24 + 8,  s1);
                _mm256_storeu_ps(row_buf + ow * 24 + 16, s2);
            }
        }

        int OW2 = OW * 2;
        size_t HW2 = (size_t)(OH * 2) * OW2;
        for (int oc_ps = 0; oc_ps < 6; oc_ps++) {
            float* row0 = out->data + (size_t)oc_ps * HW2 + (size_t)(oh * 2 + 0) * OW2;
            float* row1 = out->data + (size_t)oc_ps * HW2 + (size_t)(oh * 2 + 1) * OW2;
            int ic_base = oc_ps * 4;
            for (int ow = 0; ow < OW; ow++) {
                row0[ow * 2 + 0] = row_buf[ow * 24 + (ic_base + 0)];
                row0[ow * 2 + 1] = row_buf[ow * 24 + (ic_base + 1)];
                row1[ow * 2 + 0] = row_buf[ow * 24 + (ic_base + 2)];
                row1[ow * 2 + 1] = row_buf[ow * 24 + (ic_base + 3)];
            }
        }
    }
}

/* Bilinear resize (align_corners=False), channel by channel; `scale` is out/in. */
static void interp_forward(const Tensor* in, Tensor* out, float scale) {
    int W = in->w, H = in->h, C = in->c;
    int OW = out->w, OH = out->h;

    double ws = 1.0 / (double)scale;
    double hs = ws;

    int* xofs = (int*)alloca(OW * sizeof(int));
    float* alpha = (float*)alloca(OW * 2 * sizeof(float));
    int* yofs = (int*)alloca(OH * sizeof(int));
    float* beta = (float*)alloca(OH * 2 * sizeof(float));

    for (int dx = 0; dx < OW; dx++) {
        float fx = (float)((dx + 0.5) * ws - 0.5);
        int sx = (int)floorf(fx);
        fx -= (float)sx;
        if (sx < 0) { sx = 0; fx = 0.0f; }
        if (sx >= W - 1) { sx = W - 2; fx = 1.0f; }
        xofs[dx] = sx;
        alpha[dx * 2 + 0] = 1.0f - fx;
        alpha[dx * 2 + 1] = fx;
    }
    for (int dy = 0; dy < OH; dy++) {
        float fy = (float)((dy + 0.5) * hs - 0.5);
        int sy = (int)floorf(fy);
        fy -= (float)sy;
        if (sy < 0) { sy = 0; fy = 0.0f; }
        if (sy >= H - 1) { sy = H - 2; fy = 1.0f; }
        yofs[dy] = sy;
        beta[dy * 2 + 0] = 1.0f - fy;
        beta[dy * 2 + 1] = fy;
    }

    int in_pw = in->is_padded ? (W + 2) : W;
    size_t in_ch_sz = in->is_padded ? (size_t)(H + 2) * in_pw : (size_t)H * W;
    int in_col_base = in->is_padded ? 1 : 0;

    int out_pw = out->is_padded ? (OW + 2) : OW;
    size_t out_ch_sz = out->is_padded ? (size_t)(OH + 2) * out_pw : (size_t)OH * OW;
    int out_col_base = out->is_padded ? 1 : 0;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int c = 0; c < C; c++) {
        for (int dy = 0; dy < OH; dy++) {
            int sy = yofs[dy];
            float b0 = beta[dy * 2 + 0];
            float b1 = beta[dy * 2 + 1];
            size_t in_row0_off = (size_t)c * in_ch_sz + (size_t)(in->is_padded ? (sy + 1) : sy) * in_pw + in_col_base;
            size_t in_row1_off = (size_t)c * in_ch_sz + (size_t)(in->is_padded ? (sy + 2) : sy + 1) * in_pw + in_col_base;
            const float* row0 = in->data + in_row0_off;
            const float* row1 = in->data + in_row1_off;
            float* out_row = out->data + (size_t)c * out_ch_sz + (size_t)(out->is_padded ? (dy + 1) : dy) * out_pw + out_col_base;
            int dx = 0;
            for (; dx <= OW - 4; dx += 4) {
                int sx0 = xofs[dx + 0], sx1 = xofs[dx + 1], sx2 = xofs[dx + 2], sx3 = xofs[dx + 3];
                float a0_0 = alpha[(dx + 0) * 2 + 0], a1_0 = alpha[(dx + 0) * 2 + 1];
                float a0_1 = alpha[(dx + 1) * 2 + 0], a1_1 = alpha[(dx + 1) * 2 + 1];
                float a0_2 = alpha[(dx + 2) * 2 + 0], a1_2 = alpha[(dx + 2) * 2 + 1];
                float a0_3 = alpha[(dx + 3) * 2 + 0], a1_3 = alpha[(dx + 3) * 2 + 1];

                float v0_0 = row0[sx0] * a0_0 + row0[sx0 + 1] * a1_0;
                float v1_0 = row1[sx0] * a0_0 + row1[sx0 + 1] * a1_0;
                out_row[dx + 0] = v0_0 * b0 + v1_0 * b1;

                float v0_1 = row0[sx1] * a0_1 + row0[sx1 + 1] * a1_1;
                float v1_1 = row1[sx1] * a0_1 + row1[sx1 + 1] * a1_1;
                out_row[dx + 1] = v0_1 * b0 + v1_1 * b1;

                float v0_2 = row0[sx2] * a0_2 + row0[sx2 + 1] * a1_2;
                float v1_2 = row1[sx2] * a0_2 + row1[sx2 + 1] * a1_2;
                out_row[dx + 2] = v0_2 * b0 + v1_2 * b1;

                float v0_3 = row0[sx3] * a0_3 + row0[sx3 + 1] * a1_3;
                float v1_3 = row1[sx3] * a0_3 + row1[sx3 + 1] * a1_3;
                out_row[dx + 3] = v0_3 * b0 + v1_3 * b1;
            }
            for (; dx < OW; dx++) {
                int sx = xofs[dx];
                float a0 = alpha[dx * 2 + 0];
                float a1 = alpha[dx * 2 + 1];
                float v0 = row0[sx] * a0 + row0[sx + 1] * a1;
                float v1 = row1[sx] * a0 + row1[sx + 1] * a1;
                out_row[dx] = v0 * b0 + v1 * b1;
            }
        }
    }
}

/* out = a * s, over all channels incl. any zero border (in place is fine) */
static void scale_forward(const Tensor* a, Tensor* out, float s) {
    assert(a->is_padded == out->is_padded && a->c == out->c);
    int N = (a->is_padded ? (a->w + 2) * (a->h + 2) : a->w * a->h) * a->c;
    __m256 vs = _mm256_set1_ps(s);
    #pragma omp parallel for if(N >= 65536) schedule(static)
    for (int i = 0; i <= N - 32; i += 32) {
        _mm256_storeu_ps(out->data + i +  0, _mm256_mul_ps(_mm256_loadu_ps(a->data + i +  0), vs));
        _mm256_storeu_ps(out->data + i +  8, _mm256_mul_ps(_mm256_loadu_ps(a->data + i +  8), vs));
        _mm256_storeu_ps(out->data + i + 16, _mm256_mul_ps(_mm256_loadu_ps(a->data + i + 16), vs));
        _mm256_storeu_ps(out->data + i + 24, _mm256_mul_ps(_mm256_loadu_ps(a->data + i + 24), vs));
    }
    int start = (N / 32) * 32;
    for (int i = start; i < N; i++) out->data[i] = a->data[i] * s;
}

/* out = a + b */
static void add_forward(const Tensor* a, const Tensor* b, Tensor* out) {
    int N = out->c * out->h * out->w;
    #pragma omp parallel for if(N >= 65536) schedule(static)
    for (int i = 0; i <= N - 32; i += 32) {
        _mm256_storeu_ps(out->data + i +  0, _mm256_add_ps(_mm256_loadu_ps(a->data + i +  0), _mm256_loadu_ps(b->data + i +  0)));
        _mm256_storeu_ps(out->data + i +  8, _mm256_add_ps(_mm256_loadu_ps(a->data + i +  8), _mm256_loadu_ps(b->data + i +  8)));
        _mm256_storeu_ps(out->data + i + 16, _mm256_add_ps(_mm256_loadu_ps(a->data + i + 16), _mm256_loadu_ps(b->data + i + 16)));
        _mm256_storeu_ps(out->data + i + 24, _mm256_add_ps(_mm256_loadu_ps(a->data + i + 24), _mm256_loadu_ps(b->data + i + 24)));
    }
    int start = (N / 32) * 32;
    for (int i = start; i < N; i++) out->data[i] = a->data[i] + b->data[i];
}

/* out = a * c0 + b * c1 */
static void axpby_forward(const Tensor* a, const Tensor* b, Tensor* out, float c0, float c1) {
    int N = a->w * a->h * a->c;
    __m256 vc0 = _mm256_set1_ps(c0);
    __m256 vc1 = _mm256_set1_ps(c1);

    #pragma omp parallel for if(N >= 65536) schedule(static)
    for (int i = 0; i <= N - 32; i += 32) {
        __m256 va0 = _mm256_loadu_ps(a->data + i +  0);
        __m256 va1 = _mm256_loadu_ps(a->data + i +  8);
        __m256 va2 = _mm256_loadu_ps(a->data + i + 16);
        __m256 va3 = _mm256_loadu_ps(a->data + i + 24);

        __m256 vb0 = _mm256_loadu_ps(b->data + i +  0);
        __m256 vb1 = _mm256_loadu_ps(b->data + i +  8);
        __m256 vb2 = _mm256_loadu_ps(b->data + i + 16);
        __m256 vb3 = _mm256_loadu_ps(b->data + i + 24);

        _mm256_storeu_ps(out->data + i +  0, _mm256_fmadd_ps(va0, vc0, _mm256_mul_ps(vb0, vc1)));
        _mm256_storeu_ps(out->data + i +  8, _mm256_fmadd_ps(va1, vc0, _mm256_mul_ps(vb1, vc1)));
        _mm256_storeu_ps(out->data + i + 16, _mm256_fmadd_ps(va2, vc0, _mm256_mul_ps(vb2, vc1)));
        _mm256_storeu_ps(out->data + i + 24, _mm256_fmadd_ps(va3, vc0, _mm256_mul_ps(vb3, vc1)));
    }
    int start = (N / 32) * 32;
    for (int i = start; i < N; i++) {
        out->data[i] = a->data[i] * c0 + b->data[i] * c1;
    }
}

static void warp_forward(const Tensor* img, const Tensor* flow, Tensor* out) {
    int C = img->c, H = img->h, W = img->w;
    const float* fxptr = flow->data;
    const float* fyptr = flow->data + (size_t)H * W;
    size_t HW = (size_t)H * W;
    const float* in0 = img->data;
    const float* in1 = img->data + HW;
    const float* in2 = img->data + HW * 2;
    /* The output may be a (padded) channel slice of a concat buffer. */
    assert(!img->is_padded && !flow->is_padded);
    int opw = out->is_padded ? W + 2 : W;
    size_t ohw = out->is_padded ? (size_t)(H + 2) * opw : HW;
    float* out0 = out->data + (out->is_padded ? opw + 1 : 0);
    float* out1 = out0 + ohw;
    float* out2 = out0 + ohw * 2;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; y++) {
        const float* fx_row = fxptr + y * W;
        const float* fy_row = fyptr + y * W;
        size_t row_dst = (size_t)y * opw;

        for (int x = 0; x < W; x++) {
            float sample_x = (float)x + fx_row[x];
            float sample_y = (float)y + fy_row[x];

            float flx = _mm_cvtss_f32(_mm_floor_ss(_mm_setzero_ps(), _mm_set_ss(sample_x)));
            float fly = _mm_cvtss_f32(_mm_floor_ss(_mm_setzero_ps(), _mm_set_ss(sample_y)));
            int x0 = (int)flx;
            int y0 = (int)fly;
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            if (x0 < 0) x0 = 0; else if (x0 > W - 1) x0 = W - 1;
            if (y0 < 0) y0 = 0; else if (y0 > H - 1) y0 = H - 1;
            if (x1 < 0) x1 = 0; else if (x1 > W - 1) x1 = W - 1;
            if (y1 < 0) y1 = 0; else if (y1 > H - 1) y1 = H - 1;

            float alpha = sample_x - flx;
            float beta  = sample_y - fly;
            float inv_alpha = 1.0f - alpha;
            float inv_beta  = 1.0f - beta;

            size_t idx00 = (size_t)y0 * W + x0;
            size_t idx01 = (size_t)y0 * W + x1;
            size_t idx10 = (size_t)y1 * W + x0;
            size_t idx11 = (size_t)y1 * W + x1;
            size_t dst_idx = row_dst + x;

            if (C == 3) {
                float v4_0 = in0[idx00] * inv_alpha + in0[idx01] * alpha;
                float v5_0 = in0[idx10] * inv_alpha + in0[idx11] * alpha;
                out0[dst_idx] = v4_0 * inv_beta + v5_0 * beta;

                float v4_1 = in1[idx00] * inv_alpha + in1[idx01] * alpha;
                float v5_1 = in1[idx10] * inv_alpha + in1[idx11] * alpha;
                out1[dst_idx] = v4_1 * inv_beta + v5_1 * beta;

                float v4_2 = in2[idx00] * inv_alpha + in2[idx01] * alpha;
                float v5_2 = in2[idx10] * inv_alpha + in2[idx11] * alpha;
                out2[dst_idx] = v4_2 * inv_beta + v5_2 * beta;
            } else {
                for (int c = 0; c < C; c++) {
                    const float* in_ch = img->data + (size_t)c * HW;
                    float v0 = in_ch[idx00];
                    float v1 = in_ch[idx01];
                    float v2 = in_ch[idx10];
                    float v3 = in_ch[idx11];

                    float v4 = v0 * inv_alpha + v1 * alpha;
                    float v5 = v2 * inv_alpha + v3 * alpha;
                    out0[(size_t)c * ohw + dst_idx] = v4 * inv_beta + v5 * beta;
                }
            }
        }
    }
}

/* Copy a plain tensor into a same-sized zero-bordered one (e.g. a channel view of an IFBlock input). */
static void copy_into_padded(const Tensor* src, Tensor* dst) {
    assert(dst->is_padded && !src->is_padded && src->c == dst->c && src->w == dst->w && src->h == dst->h);
    int W = src->w, H = src->h, PW = W + 2;
    size_t dst_ch = (size_t)(H + 2) * PW;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int c = 0; c < src->c; c++)
        for (int y = 0; y < H; y++)
            memcpy(dst->data + (size_t)c * dst_ch + (size_t)(y + 1) * PW + 1,
                   src->data + ((size_t)c * H + y) * W, W * sizeof(float));
}

/* =====================  CACHE SIZES  ===================== */
/*
 * The Winograd scheduling heuristics are expressed in terms of the cache
 * hierarchy, read once at startup: Linux sysfs first (it also says how many
 * CPUs share an L3, so the total L3 of multi-CCX parts is right), then glibc
 * sysconf, then defaults matching the Zen 2 part the heuristics were tuned on.
 */
typedef struct {
    size_t l2;         /* per core */
    size_t l3_slice;   /* one L3 instance (e.g. one CCX on AMD) */
    size_t l3_total;   /* all L3 instances */
} CacheInfo;

static CacheInfo g_cache = { 512u << 10, 4u << 20, 8u << 20 };

static int read_line(const char* path, char* buf, size_t n) {
    FILE* f = fopen(path, "r");
    if (!f) return 0;
    int ok = fgets(buf, (int)n, f) != NULL;
    fclose(f);
    return ok;
}

/* "512K", "4096K", "32M" -> bytes */
static size_t parse_size(const char* s) {
    char* end;
    double v = strtod(s, &end);
    if (*end == 'K' || *end == 'k') v *= 1024;
    else if (*end == 'M' || *end == 'm') v *= 1024 * 1024;
    else if (*end == 'G' || *end == 'g') v *= 1024.0 * 1024 * 1024;
    return v > 0 ? (size_t)v : 0;
}

/* "0-2,6-8" -> 6 */
static int count_cpu_list(const char* s) {
    int n = 0;
    while (*s && *s != '\n') {
        char* end;
        long a = strtol(s, &end, 10), b = a;
        if (end == s) break;
        s = end;
        if (*s == '-') { b = strtol(s + 1, &end, 10); s = end; }
        n += (int)(b - a + 1);
        if (*s == ',') s++;
    }
    return n;
}

static void detect_caches(void) {
    size_t l2 = 0, l3 = 0;
    int l3_cpus = 0;
#ifdef __linux__
    for (int i = 0; i < 16; i++) {
        char path[128], buf[256];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu0/cache/index%d/level", i);
        if (!read_line(path, buf, sizeof(buf))) break;
        int level = atoi(buf);
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu0/cache/index%d/type", i);
        if (!read_line(path, buf, sizeof(buf)) || !strncmp(buf, "Instruction", 11)) continue;
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu0/cache/index%d/size", i);
        if (!read_line(path, buf, sizeof(buf))) continue;
        size_t sz = parse_size(buf);
        if (level == 2) l2 = sz;
        if (level == 3) {
            l3 = sz;
            snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu0/cache/index%d/shared_cpu_list", i);
            if (read_line(path, buf, sizeof(buf))) l3_cpus = count_cpu_list(buf);
        }
    }
#endif
#ifdef _SC_LEVEL2_CACHE_SIZE
    if (!l2) { long v = sysconf(_SC_LEVEL2_CACHE_SIZE); if (v > 0) l2 = (size_t)v; }
#endif
#ifdef _SC_LEVEL3_CACHE_SIZE
    if (!l3) { long v = sysconf(_SC_LEVEL3_CACHE_SIZE); if (v > 0) l3 = (size_t)v; }
#endif
    if (l2) g_cache.l2 = l2;
    if (l3) {
        g_cache.l3_slice = l3;
        long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
        int slices = (l3_cpus > 0 && ncpu > l3_cpus) ? (int)(ncpu / l3_cpus) : 1;
        g_cache.l3_total = l3 * slices;
    } else if (l2) {
        /* no L3 reported (some small cores): budget against L2 alone */
        g_cache.l3_slice = g_cache.l3_total = l2;
    }
    /* RIFE_CACHE=L2,L3slice,L3total (e.g. 1M,32M,32M) overrides detection */
    const char* ov = getenv("RIFE_CACHE");
    if (ov) {
        char buf[128];
        strncpy(buf, ov, sizeof(buf) - 1); buf[sizeof(buf) - 1] = 0;
        char* a = strtok(buf, ","); char* b = a ? strtok(NULL, ",") : NULL; char* c = b ? strtok(NULL, ",") : NULL;
        if (a && b && c && parse_size(a) && parse_size(b) && parse_size(c) >= parse_size(b)) {
            g_cache.l2 = parse_size(a); g_cache.l3_slice = parse_size(b); g_cache.l3_total = parse_size(c);
        } else {
            fprintf(stderr, "RIFE_CACHE must be L2,L3slice,L3total (e.g. 1M,32M,32M); ignored\n");
        }
    }
    printf("Caches: L2 %zu KB/core, L3 %zu KB x %zu (%zu KB total)\n", g_cache.l2 >> 10,
           g_cache.l3_slice >> 10, g_cache.l3_total / g_cache.l3_slice, g_cache.l3_total >> 10);
}


/* =====================  FUSED IFBLOCK (packed-8 layout)  ===================== */
/*
 * One RIFE IFBlock = conv3x3s2+lrelu, conv3x3s2+lrelu, 8 x (conv3x3 + x, lrelu),
 * deconv4x4s2 + PixelShuffle(2). Inside the block activations live in a packed
 * layout [C/8][PH][PW][8] with a zero border (1 px top/left, enough bottom/right
 * to cover whole 4x4 Winograd tiles), so every kernel reads 8 channels per load.
 *
 *  - stride-2 convs: direct, 16 OC x 6 px register tile (12 accumulators).
 *  - stride-1 convs: Winograd F(4x4, 3x3). Each task takes a group of tiles,
 *    transforms them into an L2-resident buffer, runs 36 small GEMMs with a
 *    6 tile x 16 OC microkernel, and applies the output transform with bias,
 *    residual and LeakyReLU fused.
 */

typedef struct {
    int C, H, W;   /* logical size */
    int PH, PW;    /* padded size */
    float* data;
} P8;

static void p8_alloc(P8* t, int C, int H, int W) {
    t->C = C; t->H = H; t->W = W;
    t->PH = ((H + 11) / 12) * 12 + 2;   /* whole 4x4 or 6x6 Winograd tiles + border */
    t->PW = ((W + 11) / 12) * 12 + 2;
    size_t n = (size_t)C * t->PH * t->PW;
    t->data = (float*)aligned_alloc(64, (n * sizeof(float) + 63) & ~(size_t)63);
    assert(t->data);
    memset(t->data, 0, n * sizeof(float));
}

static inline size_t p8_plane(const P8* t) { return (size_t)t->PH * t->PW * 8; }

/* channel block cb, padded coordinates (y, x) */
static inline float* p8_at(const P8* t, int cb, int y, int x) {
    return t->data + (size_t)cb * p8_plane(t) + ((size_t)y * t->PW + x) * 8;
}

typedef struct IFBlock {
    int cin, c0, c;        /* input channels; channels after conv0a / conv0b */
    float *w0a, *b0a;      /* conv0a: [c0/16][cin][9][16], bias [c0] */
    float *w0b, *b0b;      /* conv0b: [c/16][c0][9][16], bias [c] */
    float *res_w[8];       /* residual convs, raw [c][c][9] until ifblock_prepare builds U */
    float *res_b[8];       /* biases [c] */
    float *dw, *db;        /* deconv: [4][4][c][3][8], bias [24] */
    float* U[8];           /* Winograd weights [K][c/16][c][16] */
    int m, K;              /* Winograd output tile (4 or 6), K = (m+2)^2 */
    int nt_out;            /* streaming stores for the res-conv outputs */
    int in_w, in_h;        /* input size the buffers were built for */
    P8 a, x, y;            /* conv0a out, ping-pong for the res convs */
    int G, S;              /* tile groups per task, OC splits */
    int global;            /* 3-phase Winograd (few tiles, big weights) */
    float *gV, *gM;        /* its buffers */
} IFBlock;

/* Per-thread Winograd workspace */
static float* g_wino_ws = NULL;
static size_t g_wino_ws_per_thread = 0;

static inline __m256 lrelu8(__m256 v, __m256 slope) {
    return _mm256_max_ps(v, _mm256_mul_ps(v, slope));   /* valid for 0 <= slope <= 1 */
}

/* ---------- stride-2 direct conv, output packed into a P8 ---------- */

/* in: planar (pack=1, channel c at in + c*in_plane) or packed-8 (pack=8, block
 * c/8 at in + (c/8)*in_plane, lane c%8). Pixel (y, x) in padded coordinates is at
 * + (y*in_PW + x)*pack. wt: [OC/16][C][9][16]. */
static void conv3x3s2_p8(const float* in, int pack, size_t in_plane, int in_PW, int C,
                         const float* wt, const float* bias, float slope, P8* out) {
    int OC = out->C, OH = out->H, OW = out->W;
    int nob = OC / 16;
    __m256 vs = _mm256_set1_ps(slope);

    #pragma omp parallel for collapse(2) schedule(dynamic, 4)
    for (int ob = 0; ob < nob; ob++) {
        for (int oy = 0; oy < OH; oy++) {
            const float* wob = wt + (size_t)ob * C * 9 * 16;
            __m256 b0 = _mm256_loadu_ps(bias + ob * 16), b1 = _mm256_loadu_ps(bias + ob * 16 + 8);
            float* o0 = p8_at(out, ob * 2, oy + 1, 1);
            float* o1 = p8_at(out, ob * 2 + 1, oy + 1, 1);
            int ox = 0;
            for (; ox + 6 <= OW; ox += 6) {
                __m256 c00 = b0, c01 = b1, c10 = b0, c11 = b1, c20 = b0, c21 = b1;
                __m256 c30 = b0, c31 = b1, c40 = b0, c41 = b1, c50 = b0, c51 = b1;
                const float* w = wob;
                for (int c = 0; c < C; c++) {
                    const float* ic = in + (size_t)(c / pack) * in_plane + (c % pack);
                    for (int ky = 0; ky < 3; ky++) {
                        const float* row = ic + ((size_t)(2 * oy + ky) * in_PW + 2 * ox) * pack;
                        for (int kx = 0; kx < 3; kx++) {
                            const float* p = row + kx * pack;
                            __m256 w0 = _mm256_load_ps(w), w1 = _mm256_load_ps(w + 8);
                            __m256 v;
                            w += 16;
                            v = _mm256_broadcast_ss(p + 0 * pack); c00 = _mm256_fmadd_ps(v, w0, c00); c01 = _mm256_fmadd_ps(v, w1, c01);
                            v = _mm256_broadcast_ss(p + 2 * pack); c10 = _mm256_fmadd_ps(v, w0, c10); c11 = _mm256_fmadd_ps(v, w1, c11);
                            v = _mm256_broadcast_ss(p + 4 * pack); c20 = _mm256_fmadd_ps(v, w0, c20); c21 = _mm256_fmadd_ps(v, w1, c21);
                            v = _mm256_broadcast_ss(p + 6 * pack); c30 = _mm256_fmadd_ps(v, w0, c30); c31 = _mm256_fmadd_ps(v, w1, c31);
                            v = _mm256_broadcast_ss(p + 8 * pack); c40 = _mm256_fmadd_ps(v, w0, c40); c41 = _mm256_fmadd_ps(v, w1, c41);
                            v = _mm256_broadcast_ss(p + 10 * pack); c50 = _mm256_fmadd_ps(v, w0, c50); c51 = _mm256_fmadd_ps(v, w1, c51);
                        }
                    }
                }
                _mm256_storeu_ps(o0 + (ox + 0) * 8, lrelu8(c00, vs)); _mm256_storeu_ps(o1 + (ox + 0) * 8, lrelu8(c01, vs));
                _mm256_storeu_ps(o0 + (ox + 1) * 8, lrelu8(c10, vs)); _mm256_storeu_ps(o1 + (ox + 1) * 8, lrelu8(c11, vs));
                _mm256_storeu_ps(o0 + (ox + 2) * 8, lrelu8(c20, vs)); _mm256_storeu_ps(o1 + (ox + 2) * 8, lrelu8(c21, vs));
                _mm256_storeu_ps(o0 + (ox + 3) * 8, lrelu8(c30, vs)); _mm256_storeu_ps(o1 + (ox + 3) * 8, lrelu8(c31, vs));
                _mm256_storeu_ps(o0 + (ox + 4) * 8, lrelu8(c40, vs)); _mm256_storeu_ps(o1 + (ox + 4) * 8, lrelu8(c41, vs));
                _mm256_storeu_ps(o0 + (ox + 5) * 8, lrelu8(c50, vs)); _mm256_storeu_ps(o1 + (ox + 5) * 8, lrelu8(c51, vs));
            }
            for (; ox < OW; ox++) {
                __m256 c00 = b0, c01 = b1;
                const float* w = wob;
                for (int c = 0; c < C; c++) {
                    const float* ic = in + (size_t)(c / pack) * in_plane + (c % pack);
                    for (int ky = 0; ky < 3; ky++) {
                        const float* row = ic + ((size_t)(2 * oy + ky) * in_PW + 2 * ox) * pack;
                        for (int kx = 0; kx < 3; kx++) {
                            __m256 v = _mm256_broadcast_ss(row + kx * pack);
                            c00 = _mm256_fmadd_ps(v, _mm256_load_ps(w), c00);
                            c01 = _mm256_fmadd_ps(v, _mm256_load_ps(w + 8), c01);
                            w += 16;
                        }
                    }
                }
                _mm256_storeu_ps(o0 + ox * 8, lrelu8(c00, vs));
                _mm256_storeu_ps(o1 + ox * 8, lrelu8(c01, vs));
            }
        }
    }
}

/* ---------- Winograd F(4x4, 3x3) ---------- */

/* 1-D B^T (6 -> 6) */
#define WINO_BT(d0, d1, d2, d3, d4, d5, r0, r1, r2, r3, r4, r5) do { \
    __m256 _4 = _mm256_set1_ps(4.0f), _5 = _mm256_set1_ps(5.0f), _2 = _mm256_set1_ps(2.0f); \
    __m256 _t0 = _mm256_fnmadd_ps(_4, d2, d4);          /* d4 - 4 d2 */ \
    __m256 _t1 = _mm256_fnmadd_ps(_4, d1, d3);          /* d3 - 4 d1 */ \
    __m256 _t2 = _mm256_sub_ps(d4, d2);                 /* d4 - d2   */ \
    __m256 _t3 = _mm256_mul_ps(_2, _mm256_sub_ps(d3, d1)); /* 2(d3 - d1) */ \
    r0 = _mm256_fnmadd_ps(_5, d2, _mm256_fmadd_ps(_4, d0, d4)); \
    r1 = _mm256_add_ps(_t0, _t1); \
    r2 = _mm256_sub_ps(_t0, _t1); \
    r3 = _mm256_add_ps(_t2, _t3); \
    r4 = _mm256_sub_ps(_t2, _t3); \
    r5 = _mm256_fnmadd_ps(_5, d3, _mm256_fmadd_ps(_4, d1, d5)); \
} while (0)

/* 1-D A^T (6 -> 4) */
#define WINO_AT(m0, m1, m2, m3, m4, m5, o0, o1, o2, o3) do { \
    __m256 _a = _mm256_add_ps(m1, m2), _b = _mm256_sub_ps(m1, m2); \
    __m256 _c = _mm256_add_ps(m3, m4), _d = _mm256_sub_ps(m3, m4); \
    o0 = _mm256_add_ps(_mm256_add_ps(m0, _a), _c); \
    o1 = _mm256_fmadd_ps(_mm256_set1_ps(2.0f), _d, _b); \
    o2 = _mm256_fmadd_ps(_mm256_set1_ps(4.0f), _c, _a); \
    o3 = _mm256_add_ps(_mm256_fmadd_ps(_mm256_set1_ps(8.0f), _d, _b), m5); \
} while (0)

/* raw [OC][IC][3][3] -> U [K][OC/16][IC][16], U = G g G^T, for F(m x m, 3x3) */
static float* wino_weights(const float* raw, int OC, int IC, int m) {
    static const double G4[6][3] = {
        { 1.0 / 4,  0,         0       },
        {-1.0 / 6, -1.0 / 6,  -1.0 / 6 },
        {-1.0 / 6,  1.0 / 6,  -1.0 / 6 },
        { 1.0 / 24, 1.0 / 12,  1.0 / 6 },
        { 1.0 / 24, -1.0 / 12, 1.0 / 6 },
        { 0,        0,         1       },
    };
    static const double G6[8][3] = {
        { 1,          0,          0         },
        {-2.0 / 9,   -2.0 / 9,   -2.0 / 9   },
        {-2.0 / 9,    2.0 / 9,   -2.0 / 9   },
        { 1.0 / 90,   1.0 / 45,   2.0 / 45  },
        { 1.0 / 90,  -1.0 / 45,   2.0 / 45  },
        { 32.0 / 45,  16.0 / 45,  8.0 / 45  },
        { 32.0 / 45, -16.0 / 45,  8.0 / 45  },
        { 0,          0,          1         },
    };
    const double (*G)[3] = m == 6 ? G6 : G4;
    int T = m + 2, K = T * T;
    int nob = OC / 16;
    size_t n = (size_t)K * OC * IC;
    float* U = (float*)aligned_alloc(64, (n * sizeof(float) + 63) & ~(size_t)63);
    assert(U);
    for (int oc = 0; oc < OC; oc++) {
        for (int ic = 0; ic < IC; ic++) {
            const float* g = raw + ((size_t)oc * IC + ic) * 9;
            double t[8][3];
            for (int i = 0; i < T; i++)
                for (int j = 0; j < 3; j++)
                    t[i][j] = G[i][0] * g[0 * 3 + j] + G[i][1] * g[1 * 3 + j] + G[i][2] * g[2 * 3 + j];
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < T; j++) {
                    double u = t[i][0] * G[j][0] + t[i][1] * G[j][1] + t[i][2] * G[j][2];
                    int k = i * T + j;
                    U[(((size_t)k * nob + oc / 16) * IC + ic) * 16 + oc % 16] = (float)u;
                }
            }
        }
    }
    return U;
}

/* Input transform of one 6x6 tile (top-left padded corner 4ty, 4tx) for channel
 * block cb. Writes 36 vectors, k-th at dst + k*ks. */
static inline void wino_in_tile(const P8* in, int ty, int tx, int cb, float* dst, size_t ks) {
    const float* src = in->data + (size_t)cb * p8_plane(in) + ((size_t)(4 * ty) * in->PW + 4 * tx) * 8;
    size_t rs = (size_t)in->PW * 8;
    __m256 r[6][6];
    for (int c = 0; c < 6; c++) {
        const float* s = src + c * 8;
        __m256 d0 = _mm256_loadu_ps(s), d1 = _mm256_loadu_ps(s + rs), d2 = _mm256_loadu_ps(s + 2 * rs);
        __m256 d3 = _mm256_loadu_ps(s + 3 * rs), d4 = _mm256_loadu_ps(s + 4 * rs), d5 = _mm256_loadu_ps(s + 5 * rs);
        WINO_BT(d0, d1, d2, d3, d4, d5, r[0][c], r[1][c], r[2][c], r[3][c], r[4][c], r[5][c]);
    }
    for (int i = 0; i < 6; i++) {
        __m256 v0, v1, v2, v3, v4, v5;
        WINO_BT(r[i][0], r[i][1], r[i][2], r[i][3], r[i][4], r[i][5], v0, v1, v2, v3, v4, v5);
        _mm256_storeu_ps(dst + (i * 6 + 0) * ks, v0);
        _mm256_storeu_ps(dst + (i * 6 + 1) * ks, v1);
        _mm256_storeu_ps(dst + (i * 6 + 2) * ks, v2);
        _mm256_storeu_ps(dst + (i * 6 + 3) * ks, v3);
        _mm256_storeu_ps(dst + (i * 6 + 4) * ks, v4);
        _mm256_storeu_ps(dst + (i * 6 + 5) * ks, v5);
    }
}

/* 6 tiles x 16 OC microkernel over IC = 8*ICb. v: [ICb][6][8] (tile-major within
 * a channel block), u: [IC][16]. Writes tile j's 16 outputs to m + j*ts. */
static inline void wino_gemm_6x16(const float* v, const float* u, int ICb, float* m, size_t ts) {
    __m256 c00 = _mm256_setzero_ps(), c01 = c00, c10 = c00, c11 = c00, c20 = c00, c21 = c00;
    __m256 c30 = c00, c31 = c00, c40 = c00, c41 = c00, c50 = c00, c51 = c00;
    for (int cb = 0; cb < ICb; cb++, v += 48) {
        for (int l = 0; l < 8; l++) {
            __m256 w0 = _mm256_load_ps(u), w1 = _mm256_load_ps(u + 8), b;
            u += 16;
            b = _mm256_broadcast_ss(v + l + 0);  c00 = _mm256_fmadd_ps(b, w0, c00); c01 = _mm256_fmadd_ps(b, w1, c01);
            b = _mm256_broadcast_ss(v + l + 8);  c10 = _mm256_fmadd_ps(b, w0, c10); c11 = _mm256_fmadd_ps(b, w1, c11);
            b = _mm256_broadcast_ss(v + l + 16); c20 = _mm256_fmadd_ps(b, w0, c20); c21 = _mm256_fmadd_ps(b, w1, c21);
            b = _mm256_broadcast_ss(v + l + 24); c30 = _mm256_fmadd_ps(b, w0, c30); c31 = _mm256_fmadd_ps(b, w1, c31);
            b = _mm256_broadcast_ss(v + l + 32); c40 = _mm256_fmadd_ps(b, w0, c40); c41 = _mm256_fmadd_ps(b, w1, c41);
            b = _mm256_broadcast_ss(v + l + 40); c50 = _mm256_fmadd_ps(b, w0, c50); c51 = _mm256_fmadd_ps(b, w1, c51);
        }
    }
    _mm256_storeu_ps(m + 0 * ts, c00); _mm256_storeu_ps(m + 0 * ts + 8, c01);
    _mm256_storeu_ps(m + 1 * ts, c10); _mm256_storeu_ps(m + 1 * ts + 8, c11);
    _mm256_storeu_ps(m + 2 * ts, c20); _mm256_storeu_ps(m + 2 * ts + 8, c21);
    _mm256_storeu_ps(m + 3 * ts, c30); _mm256_storeu_ps(m + 3 * ts + 8, c31);
    _mm256_storeu_ps(m + 4 * ts, c40); _mm256_storeu_ps(m + 4 * ts + 8, c41);
    _mm256_storeu_ps(m + 5 * ts, c50); _mm256_storeu_ps(m + 5 * ts + 8, c51);
}

/* Output transform of tile t for channel block ocb, + bias + residual, lrelu.
 * m: 36 vectors, k-th at m + k*ks. */
static inline void wino_out_tile(const float* m, size_t ks, const float* bias, __m256 vs,
                                 const P8* in, P8* out, int t, int tw, int ocb, int nt) {
    int H = in->H, W = in->W;
    int ty = t / tw, tx = t % tw;
    int ny = H - 4 * ty < 4 ? H - 4 * ty : 4;
    int nx = W - 4 * tx < 4 ? W - 4 * tx : 4;
    __m256 s[4][6];
    for (int j = 0; j < 6; j++) {
        WINO_AT(_mm256_loadu_ps(m + (0 * 6 + j) * ks), _mm256_loadu_ps(m + (1 * 6 + j) * ks),
                _mm256_loadu_ps(m + (2 * 6 + j) * ks), _mm256_loadu_ps(m + (3 * 6 + j) * ks),
                _mm256_loadu_ps(m + (4 * 6 + j) * ks), _mm256_loadu_ps(m + (5 * 6 + j) * ks),
                s[0][j], s[1][j], s[2][j], s[3][j]);
    }
    __m256 vb = _mm256_loadu_ps(bias + ocb * 8);
    for (int r = 0; r < ny; r++) {
        __m256 o[4];
        WINO_AT(s[r][0], s[r][1], s[r][2], s[r][3], s[r][4], s[r][5], o[0], o[1], o[2], o[3]);
        const float* res = p8_at(in, ocb, 4 * ty + r + 1, 4 * tx + 1);
        float* dst = p8_at(out, ocb, 4 * ty + r + 1, 4 * tx + 1);
        for (int c = 0; c < nx; c++) {
            __m256 v = lrelu8(_mm256_add_ps(_mm256_add_ps(o[c], vb), _mm256_loadu_ps(res + c * 8)), vs);
            if (nt) _mm256_stream_ps(dst + c * 8, v); else _mm256_storeu_ps(dst + c * 8, v);
        }
    }
}

/* ---------- Winograd F(6x6, 3x3): 8x8 input tiles, 64 products ---------- */

/* 1-D B^T (8 -> 8), points 0, +-1, +-2, +-1/2 */
#define WINO6_BT(d, r) do { \
    __m256 _t1, _t2; \
    r[0] = _mm256_fmadd_ps(_mm256_sub_ps(d[4], d[2]), _mm256_set1_ps(5.25f), _mm256_sub_ps(d[0], d[6])); \
    r[7] = _mm256_fmadd_ps(_mm256_sub_ps(d[3], d[5]), _mm256_set1_ps(5.25f), _mm256_sub_ps(d[7], d[1])); \
    _t1 = _mm256_fnmadd_ps(d[4], _mm256_set1_ps(4.25f), _mm256_add_ps(d[2], d[6])); \
    _t2 = _mm256_fnmadd_ps(d[3], _mm256_set1_ps(4.25f), _mm256_add_ps(d[1], d[5])); \
    r[1] = _mm256_add_ps(_t1, _t2); r[2] = _mm256_sub_ps(_t1, _t2); \
    _t1 = _mm256_fnmadd_ps(d[4], _mm256_set1_ps(1.25f), _mm256_fmadd_ps(d[2], _mm256_set1_ps(0.25f), d[6])); \
    _t2 = _mm256_fmadd_ps(d[5], _mm256_set1_ps(2.0f), _mm256_fnmadd_ps(d[3], _mm256_set1_ps(2.5f), _mm256_mul_ps(d[1], _mm256_set1_ps(0.5f)))); \
    r[3] = _mm256_add_ps(_t1, _t2); r[4] = _mm256_sub_ps(_t1, _t2); \
    _t1 = _mm256_fmadd_ps(_mm256_fnmadd_ps(d[4], _mm256_set1_ps(1.25f), d[2]), _mm256_set1_ps(4.0f), d[6]); \
    _t2 = _mm256_fmadd_ps(d[5], _mm256_set1_ps(0.5f), _mm256_fnmadd_ps(d[3], _mm256_set1_ps(2.5f), _mm256_mul_ps(d[1], _mm256_set1_ps(2.0f)))); \
    r[5] = _mm256_add_ps(_t1, _t2); r[6] = _mm256_sub_ps(_t1, _t2); \
} while (0)

/* 1-D A^T (8 -> 6) */
#define WINO6_AT(m, o) do { \
    __m256 _a1 = _mm256_add_ps(m[1], m[2]), _b1 = _mm256_sub_ps(m[1], m[2]); \
    __m256 _a2 = _mm256_add_ps(m[3], m[4]), _b2 = _mm256_sub_ps(m[3], m[4]); \
    __m256 _a3 = _mm256_add_ps(m[5], m[6]), _b3 = _mm256_sub_ps(m[5], m[6]); \
    o[0] = _mm256_add_ps(_mm256_add_ps(m[0], _a1), _mm256_add_ps(_a2, _a3)); \
    o[1] = _mm256_fmadd_ps(_b3, _mm256_set1_ps(0.5f), _mm256_fmadd_ps(_b2, _mm256_set1_ps(2.0f), _b1)); \
    o[2] = _mm256_fmadd_ps(_a3, _mm256_set1_ps(0.25f), _mm256_fmadd_ps(_a2, _mm256_set1_ps(4.0f), _a1)); \
    o[3] = _mm256_fmadd_ps(_b3, _mm256_set1_ps(0.125f), _mm256_fmadd_ps(_b2, _mm256_set1_ps(8.0f), _b1)); \
    o[4] = _mm256_fmadd_ps(_a3, _mm256_set1_ps(0.0625f), _mm256_fmadd_ps(_a2, _mm256_set1_ps(16.0f), _a1)); \
    o[5] = _mm256_add_ps(m[7], _mm256_fmadd_ps(_b3, _mm256_set1_ps(0.03125f), _mm256_fmadd_ps(_b2, _mm256_set1_ps(32.0f), _b1))); \
} while (0)

static inline void wino6_in_tile(const P8* in, int ty, int tx, int cb, float* dst, size_t ks) {
    const float* src = in->data + (size_t)cb * p8_plane(in) + ((size_t)(6 * ty) * in->PW + 6 * tx) * 8;
    size_t rs = (size_t)in->PW * 8;
    __m256 t[8][8];
    for (int c = 0; c < 8; c++) {
        __m256 d[8], r[8];
        for (int k = 0; k < 8; k++) d[k] = _mm256_loadu_ps(src + c * 8 + k * rs);
        WINO6_BT(d, r);
        for (int k = 0; k < 8; k++) t[k][c] = r[k];
    }
    for (int i = 0; i < 8; i++) {
        __m256 r[8];
        WINO6_BT(t[i], r);
        for (int j = 0; j < 8; j++) _mm256_storeu_ps(dst + (i * 8 + j) * ks, r[j]);
    }
}

static inline void wino6_out_tile(const float* m, size_t ks, const float* bias, __m256 vs,
                                  const P8* in, P8* out, int t, int tw, int ocb, int nt) {
    int H = in->H, W = in->W;
    int ty = t / tw, tx = t % tw;
    int ny = H - 6 * ty < 6 ? H - 6 * ty : 6;
    int nx = W - 6 * tx < 6 ? W - 6 * tx : 6;
    __m256 s[6][8];
    for (int j = 0; j < 8; j++) {
        __m256 col[8], o[6];
        for (int i = 0; i < 8; i++) col[i] = _mm256_loadu_ps(m + (i * 8 + j) * ks);
        WINO6_AT(col, o);
        for (int r = 0; r < 6; r++) s[r][j] = o[r];
    }
    __m256 vb = _mm256_loadu_ps(bias + ocb * 8);
    for (int r = 0; r < ny; r++) {
        __m256 o[6];
        WINO6_AT(s[r], o);
        const float* res = p8_at(in, ocb, 6 * ty + r + 1, 6 * tx + 1);
        float* dst = p8_at(out, ocb, 6 * ty + r + 1, 6 * tx + 1);
        for (int c = 0; c < nx; c++) {
            __m256 v = lrelu8(_mm256_add_ps(_mm256_add_ps(o[c], vb), _mm256_loadu_ps(res + c * 8)), vs);
            if (nt) _mm256_stream_ps(dst + c * 8, v); else _mm256_storeu_ps(dst + c * 8, v);
        }
    }
}

/* out = lrelu(conv3x3(in) + bias + in), in/out packed P8 with C channels. */
static void wino_conv_res(const IFBlock* B, const float* U, const float* bias, float slope,
                          const P8* in, P8* out) {
    int C = in->C, H = in->H, W = in->W;
    int ICb = C / 8, nob = C / 16;
    int m = B->m, K = B->K;
    int th = (H + m - 1) / m, tw = (W + m - 1) / m, T = th * tw;
    int NG = (T + 5) / 6;                 /* groups of 6 tiles */
    __m256 vs = _mm256_set1_ps(slope);
    /* Outputs much larger than L3 are written with streaming stores: they skip
     * the read-for-ownership of each line, which roughly halves the bandwidth
     * left for loads on this memory-bound path. */
    int nt = B->nt_out;

    if (B->global) {
        /* Few tiles, large weights: transform everything once, then run the GEMMs
         * split over (k, OC block) so each weight slice is read exactly once. */
        float* V = B->gV;                  /* [K][NG][ICb][6][8] */
        float* M = B->gM;                  /* [NG*6][K][C]       */
        size_t ks = (size_t)NG * ICb * 48 + 16;   /* +1 cache line: spread the K slices over L1 sets */
        #pragma omp parallel
        {
            #pragma omp for collapse(2) schedule(static)
            for (int t = 0; t < NG * 6; t++) {
                for (int cb = 0; cb < ICb; cb++) {
                    float* dst = V + ((size_t)(t / 6) * ICb + cb) * 48 + (t % 6) * 8;
                    if (t < T) {
                        (m == 6 ? wino6_in_tile : wino_in_tile)(in, t / tw, t % tw, cb, dst, ks);
                    } else {
                        for (int k = 0; k < K; k++) _mm256_storeu_ps(dst + k * ks, _mm256_setzero_ps());
                    }
                }
            }
            #pragma omp for collapse(2) schedule(static)
            for (int k = 0; k < K; k++) {
                for (int ob = 0; ob < nob; ob++) {
                    const float* u = U + ((size_t)k * nob + ob) * C * 16;
                    for (int gi = 0; gi < NG; gi++)
                        wino_gemm_6x16(V + k * ks + (size_t)gi * ICb * 48, u, ICb,
                                       M + ((size_t)(gi * 6) * K + k) * C + ob * 16, (size_t)K * C);
                }
            }
            #pragma omp for collapse(2) schedule(static)
            for (int t = 0; t < T; t++) {
                for (int ocb = 0; ocb < C / 8; ocb++)
                    (m == 6 ? wino6_out_tile : wino_out_tile)(M + (size_t)t * K * C + ocb * 8, C, bias, vs, in, out, t, tw, ocb, nt);
            }
            if (nt) _mm_sfence();
        }
        return;
    }

    int G = B->G, S = B->S;
    int ntg = (NG + G - 1) / G;           /* tile tasks */
    int obs = (nob + S - 1) / S;          /* OC blocks per split */

    #pragma omp parallel
    {
        float* ws = g_wino_ws + (size_t)omp_get_thread_num() * g_wino_ws_per_thread;
        float* V = ws;                                      /* [K][G][ICb][6][8] */
        size_t ks = (size_t)G * ICb * 48 + 16;              /* +1 cache line: spread the K slices over L1 sets */
        float* M = ws + (size_t)K * ks;                     /* [G*6][K][OCt]     */

        #pragma omp for schedule(dynamic, 1)
        for (int task = 0; task < ntg * S; task++) {
            int sp = task / ntg, tg = task % ntg;
            int g0 = tg * G, gn = (NG - g0) < G ? (NG - g0) : G;
            int ob0 = sp * obs, ob1 = ob0 + obs < nob ? ob0 + obs : nob;
            if (ob0 >= ob1) continue;
            int OCt = (ob1 - ob0) * 16;

            for (int cb = 0; cb < ICb; cb++) {
                for (int gi = 0; gi < gn; gi++) {
                    for (int jt = 0; jt < 6; jt++) {
                        int t = (g0 + gi) * 6 + jt;
                        float* dst = V + ((size_t)gi * ICb + cb) * 48 + jt * 8;
                        if (t < T) {
                            (m == 6 ? wino6_in_tile : wino_in_tile)(in, t / tw, t % tw, cb, dst, ks);
                        } else {
                            for (int k = 0; k < K; k++) _mm256_storeu_ps(dst + k * ks, _mm256_setzero_ps());
                        }
                    }
                }
            }
            for (int k = 0; k < K; k++) {
                for (int ob = ob0; ob < ob1; ob++) {
                    const float* u = U + ((size_t)k * nob + ob) * C * 16;
                    for (int gi = 0; gi < gn; gi++)
                        wino_gemm_6x16(V + k * ks + (size_t)gi * ICb * 48, u, ICb,
                                       M + ((size_t)(gi * 6) * K + k) * OCt + (ob - ob0) * 16, (size_t)K * OCt);
                }
            }
            for (int ocb = ob0 * 2; ocb < ob1 * 2; ocb++) {
                for (int tl = 0; tl < gn * 6; tl++) {
                    int t = g0 * 6 + tl;
                    if (t >= T) break;
                    (m == 6 ? wino6_out_tile : wino_out_tile)(M + (size_t)tl * K * OCt + (ocb - ob0 * 2) * 8, OCt, bias, vs, in, out, t, tw, ocb, nt);
                }
            }
        }
        if (nt) _mm_sfence();
    }
}

/* ---------- weight loading ---------- */

#define TAG_FP16  0x01306B47

static float fp16_to_f32(uint16_t h) {
    int s = (h >> 15) & 1;
    int e = (h >> 10) & 0x1F;
    int m = h & 0x3FF;
    if (e == 0) return s ? -0.0f : 0.0f;
    if (e == 31) return s ? -INFINITY : INFINITY;
    e = e - 15 + 127;
    return (float)((s ? -1 : 1) * (1.0 + m / 1024.0) * ldexp(1.0, e - 127));
}

static void* alloc64(size_t bytes) {
    void* p = aligned_alloc(64, (bytes + 63) & ~(size_t)63);
    if (!p) { fprintf(stderr, "out of memory (%zu bytes)\n", bytes); exit(1); }
    return p;
}

static void die_model(const char* what, const char* path) {
    fprintf(stderr, "%s: %s (is this a RIFE v4.6 flownet.bin?)\n", path, what);
    exit(1);
}

/* One ncnn weight blob: fp16 tag, n halves (padded to 4 bytes), converted to float. */
static float* read_weights(FILE* f, size_t n, const char* path) {
    uint32_t tag = 0;
    if (fread(&tag, sizeof tag, 1, f) != 1 || tag != TAG_FP16) die_model("bad weight tag or truncated file", path);
    size_t bytes = (n * sizeof(uint16_t) + 3) & ~(size_t)3;
    uint16_t* h = (uint16_t*)malloc(bytes);
    if (fread(h, 1, bytes, f) != bytes) die_model("truncated file", path);
    float* w = (float*)alloc64(n * sizeof(float));
    for (size_t k = 0; k < n; k++) w[k] = fp16_to_f32(h[k]);
    free(h);
    return w;
}

static float* read_bias(FILE* f, size_t n, const char* path) {
    float* b = (float*)alloc64(n * sizeof(float));
    if (fread(b, sizeof(float), n, f) != n) die_model("truncated file", path);
    return b;
}

/* Stride-2 conv weights: raw [OC][C][9] -> [OC/16][C][9][16] */
static float* pack_s2(const float* raw, int OC, int C) {
    float* w = (float*)alloc64((size_t)OC * C * 9 * sizeof(float));
    for (int ob = 0; ob < OC / 16; ob++)
        for (int c = 0; c < C; c++)
            for (int k = 0; k < 9; k++)
                for (int s = 0; s < 16; s++)
                    w[(((size_t)ob * C + c) * 9 + k) * 16 + s] = raw[(((size_t)ob * 16 + s) * C + c) * 9 + k];
    return w;
}

/* Deconv weights (K=4, 24 outputs): raw [24][C][4][4] -> [kh][kw][C][3][8] */
static float* pack_deconv(const float* raw, int C) {
    float* w = (float*)alloc64((size_t)16 * C * 24 * sizeof(float));
    for (int kh = 0; kh < 4; kh++)
        for (int kw = 0; kw < 4; kw++)
            for (int c = 0; c < C; c++)
                for (int ob = 0; ob < 3; ob++)
                    for (int s = 0; s < 8; s++)
                        w[(((size_t)(kh * 4 + kw) * C + c) * 3 + ob) * 8 + s] =
                            raw[(((size_t)(ob * 8 + s) * C + c) * 4 + kh) * 4 + kw];
    return w;
}

/* Read one IFBlock (c channels, cin input channels) from the weight file, in export order. */
static IFBlock* ifblock_load(FILE* f, int cin, int c, const char* path) {
    IFBlock* B = (IFBlock*)calloc(1, sizeof(IFBlock));
    B->cin = cin; B->c0 = c / 2; B->c = c;
    assert(B->c0 % 16 == 0 && B->c % 16 == 0);
    float* raw;
    raw = read_weights(f, (size_t)B->c0 * cin * 9, path); B->w0a = pack_s2(raw, B->c0, cin); free(raw);
    B->b0a = read_bias(f, B->c0, path);
    raw = read_weights(f, (size_t)c * B->c0 * 9, path);   B->w0b = pack_s2(raw, c, B->c0); free(raw);
    B->b0b = read_bias(f, c, path);
    for (int i = 0; i < 8; i++) {
        B->res_w[i] = read_weights(f, (size_t)c * c * 9, path);
        B->res_b[i] = read_bias(f, c, path);
    }
    raw = read_weights(f, (size_t)24 * c * 16, path);     B->dw = pack_deconv(raw, c); free(raw);
    B->db = read_bias(f, 24, path);
    return B;
}

/* ---------- block setup / execution ---------- */

/* Workspace for the deconv's channels-last input, shared by all blocks. */
static float* g_hwc = NULL;
static size_t g_hwc_size = 0;

static void ifblock_prepare(IFBlock* B, int in_w, int in_h) {
    B->in_w = in_w; B->in_h = in_h;
    int W1 = (in_w - 1) / 2 + 1, H1 = (in_h - 1) / 2 + 1;
    int W2 = (W1 - 1) / 2 + 1, H2 = (H1 - 1) / 2 + 1;
    p8_alloc(&B->a, B->c0, H1, W1);
    p8_alloc(&B->x, B->c, H2, W2);
    p8_alloc(&B->y, B->c, H2, W2);

    int C = B->c, nob = C / 16;
    /* F(6x6) needs 21% fewer FLOPs but 64/36 = 1.8x the weight memory of
     * F(4x4). Use it while its weights stay well inside one L3 slice; on Zen 2
     * (4 MB slices) this picks F(6x6) for <= 96 ch, which measured faster, and
     * F(4x4) for 128/192 ch, where F(6x6) measured slower. */
    size_t u6 = (size_t)64 * C * C * sizeof(float);
    B->m = u6 * 10 <= g_cache.l3_slice * 6 ? 6 : 4;
    const char* tenv = getenv("RIFE_WINO_TILE");
    if (tenv) B->m = atoi(tenv) == 4 ? 4 : 6;
    B->K = (B->m + 2) * (B->m + 2);
    int m = B->m, K = B->K;
    for (int i = 0; i < 8; i++) {
        B->U[i] = wino_weights(B->res_w[i], C, C, m);
        free(B->res_w[i]);
        B->res_w[i] = NULL;
    }
    int T = ((H2 + m - 1) / m) * ((W2 + m - 1) / m), NG = (T + 5) / 6;
    size_t act_bytes = (size_t)C * B->x.PH * B->x.PW * sizeof(float);
    B->nt_out = act_bytes > g_cache.l3_total;   /* can't stay cached anyway: skip the RFO */
    const char* nenv = getenv("RIFE_NT");
    if (nenv) B->nt_out = atoi(nenv);
    int nt = omp_get_max_threads();
    /* keep V + M for one task within about half of L2 */
    size_t per_group = (size_t)K * 6 * (C + C) * sizeof(float);
    int G = (int)((g_cache.l2 / 2) / per_group);
    if (G < 1) G = 1;
    if (G > 4) G = 4;
    while (G > 1 && (NG + G - 1) / G < 3 * nt) G--;
    /* Split OC so each split's weights (about half an L3 slice) stay in L3
     * while all threads work through it (tasks are ordered split-major), and
     * for parallelism. */
    size_t u_all = (size_t)K * C * C * sizeof(float);
    size_t split = g_cache.l3_slice / 2;
    int S = (int)((u_all + split - 1) / split);
    if (S > nob) S = nob;
    while (S < nob && ((NG + G - 1) / G) * S < 3 * nt) S++;
    B->G = G; B->S = S;

    /* Weights larger than what one task should stream, and few enough tiles
     * that transforming them all at once stays cache-friendly. */
    size_t u_bytes = (size_t)K * C * C * sizeof(float);
    size_t v_bytes = (size_t)K * (NG * 6 * C + 16) * sizeof(float);
    B->global = u_bytes > 2 * v_bytes;
    const char* env = getenv("RIFE_WINO_GLOBAL");
    if (env) B->global = atoi(env);
    if (B->global) {
        B->gV = (float*)aligned_alloc(64, (v_bytes + 63) & ~(size_t)63);
        B->gM = (float*)aligned_alloc(64, (v_bytes + 63) & ~(size_t)63);
        assert(B->gV && B->gM);
    }

    size_t need = (size_t)K * ((size_t)G * 6 * C + 16) + (size_t)K * G * 6 * C;   /* V (padded), M */
    if (need > g_wino_ws_per_thread) {
        free(g_wino_ws);
        g_wino_ws_per_thread = need;
        g_wino_ws = (float*)aligned_alloc(64, (need * nt * sizeof(float) + 63) & ~(size_t)63);
        assert(g_wino_ws);
    }
    size_t hwc_need = (size_t)B->x.C * B->x.H * B->x.W;   /* deconv input, channels-last */
    if (hwc_need > g_hwc_size) {
        free(g_hwc);
        g_hwc = (float*)alloc64(hwc_need * sizeof(float));
        g_hwc_size = hwc_need;
    }
    if (getenv("RIFE_VERBOSE"))
        printf("IFBlock c=%d %dx%d: F(%dx%d,3x3) G=%d S=%d %s%s\n", C, W2, H2, B->m, B->m, B->G, B->S,
               B->global ? "3-phase" : "tiled", B->nt_out ? " streaming-stores" : "");
}

static void ifblock_forward(IFBlock* B, const Tensor* in, Tensor* out) {
    assert(in->is_padded && in->w == B->in_w && in->h == B->in_h);
    /* conv0a: planar padded input -> a */
    conv3x3s2_p8(in->data, 1, (size_t)(in->h + 2) * (in->w + 2), in->w + 2, in->c,
                 B->w0a, B->b0a, RIFE_SLOPE, &B->a);
    /* conv0b: a -> x */
    conv3x3s2_p8(B->a.data, 8, p8_plane(&B->a), B->a.PW, B->c0,
                 B->w0b, B->b0b, RIFE_SLOPE, &B->x);
    /* 8 residual convs, ping-pong x <-> y */
    P8* cur = &B->x;
    P8* nxt = &B->y;
    for (int i = 0; i < 8; i++) {
        wino_conv_res(B, B->U[i], B->res_b[i], RIFE_SLOPE, cur, nxt);
        P8* t = cur; cur = nxt; nxt = t;
    }
    /* deconv: packed -> HWC scratch -> deconv core (+ pixelshuffle) */
    int C = cur->C, H = cur->H, W = cur->W;
    assert((size_t)C * H * W <= g_hwc_size);
    float* hwc = g_hwc;
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            float* d = hwc + ((size_t)h * W + w) * C;
            for (int cb = 0; cb < C / 8; cb++)
                _mm256_storeu_ps(d + cb * 8, _mm256_loadu_ps(p8_at(cur, cb, h + 1, w + 1)));
        }
    }
    deconv_core(B->dw, B->db, hwc, W, H, C, out);
}

/* =====================  FUSED FINAL MERGE  ===================== */
/*
 * out = warp(imgA, flowA) * m + warp(imgB, flowB) * (1 - m), m = sigmoid(logit):
 * the two warps, the sigmoid and the blend in a single full-res pass.
 */

/* Bilinear border-clamped sample of a 3-channel planar image, same arithmetic
 * as warp_forward. */
static inline void warp_sample3(const float* img, size_t HW, int W, int H, float sx, float sy, float o[3]) {
    float flx = floorf(sx), fly = floorf(sy);
    int x0 = (int)flx, y0 = (int)fly, x1 = x0 + 1, y1 = y0 + 1;
    if (x0 < 0) x0 = 0; else if (x0 > W - 1) x0 = W - 1;
    if (y0 < 0) y0 = 0; else if (y0 > H - 1) y0 = H - 1;
    if (x1 < 0) x1 = 0; else if (x1 > W - 1) x1 = W - 1;
    if (y1 < 0) y1 = 0; else if (y1 > H - 1) y1 = H - 1;
    float a = sx - flx, b = sy - fly, ia = 1.0f - a, ib = 1.0f - b;
    size_t i00 = (size_t)y0 * W + x0, i01 = (size_t)y0 * W + x1;
    size_t i10 = (size_t)y1 * W + x0, i11 = (size_t)y1 * W + x1;
    for (int c = 0; c < 3; c++) {
        const float* p = img + c * HW;
        float t0 = p[i00] * ia + p[i01] * a;
        float t1 = p[i10] * ia + p[i11] * a;
        o[c] = t0 * ib + t1 * b;
    }
}

static void merge_forward(const Tensor* imgA, const Tensor* flowA, const Tensor* imgB, const Tensor* flowB,
                          const Tensor* logit, Tensor* out) {
    assert(!imgA->is_padded && !imgB->is_padded && !flowA->is_padded && !flowB->is_padded &&
           !logit->is_padded && !out->is_padded);
    assert(imgA->c == 3 && imgB->c == 3 && out->c == 3);
    int W = out->w, H = out->h;
    size_t HW = (size_t)W * H;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            size_t i = (size_t)y * W + x;
            float v = logit->data[i];
            v = fmaxf(-88.0f, fminf(v, 88.0f));
            float m = 1.0f / (1.0f + expf(-v));
            float wA = m, wB = 1.0f - m;
            float a[3], b[3];
            warp_sample3(imgA->data, HW, W, H, (float)x + flowA->data[i], (float)y + flowA->data[HW + i], a);
            warp_sample3(imgB->data, HW, W, H, (float)x + flowB->data[i], (float)y + flowB->data[HW + i], b);
            for (int c = 0; c < 3; c++)
                out->data[c * HW + i] = a[c] * wA + b[c] * wB;
        }
    }
}

/* =====================  FUSED WARP + CONCAT + DOWNSCALE  ===================== */
/*
 * Interp(Concat(warp(img0, f0), warp(img1, f1), t, mask), 1/s) would compute
 * every warp at full resolution, write the full-res concat buffer, and then
 * read back only the bilinear taps of the downscale (for s = 4, 4 of every 16
 * pixels). warpdown_forward evaluates the warps only at those taps and writes
 * the low-res result directly, channels in source order.
 *
 * A source is a warp (img warped by flow) or, with flow == NULL, a plain
 * tensor of at most 8 channels that is only downscaled.
 */
typedef struct {
    const Tensor* a;      /* image to warp, or the plain tensor */
    const Tensor* flow;   /* 2-channel flow, or NULL */
} WarpSrc;

static void warpdown_forward(const WarpSrc* src, int n_src, float scale, Tensor* out) {
    const Tensor* first = src[0].a;
    int W = first->w, H = first->h;
    int OW = out->w, OH = out->h;
    size_t HW = (size_t)W * H;
    assert(W >= 2 && H >= 2);

    /* coordinate / weight tables, identical to interp_forward */
    double ws = 1.0 / (double)scale, hs = ws;
    int* xofs = (int*)malloc(OW * sizeof(int));
    float* alpha = (float*)malloc(OW * 2 * sizeof(float));
    int* yofs = (int*)malloc(OH * sizeof(int));
    float* beta = (float*)malloc(OH * 2 * sizeof(float));
    for (int dx = 0; dx < OW; dx++) {
        float fx = (float)((dx + 0.5) * ws - 0.5);
        int sx = (int)floorf(fx);
        fx -= (float)sx;
        if (sx < 0) { sx = 0; fx = 0.0f; }
        if (sx >= W - 1) { sx = W - 2; fx = 1.0f; }
        xofs[dx] = sx; alpha[dx * 2] = 1.0f - fx; alpha[dx * 2 + 1] = fx;
    }
    for (int dy = 0; dy < OH; dy++) {
        float fy = (float)((dy + 0.5) * hs - 0.5);
        int sy = (int)floorf(fy);
        fy -= (float)sy;
        if (sy < 0) { sy = 0; fy = 0.0f; }
        if (sy >= H - 1) { sy = H - 2; fy = 1.0f; }
        yofs[dy] = sy; beta[dy * 2] = 1.0f - fy; beta[dy * 2 + 1] = fy;
    }

    int out_pw = out->is_padded ? OW + 2 : OW;
    size_t out_ch = out->is_padded ? (size_t)(OH + 2) * out_pw : (size_t)OH * OW;
    float* out_base = out->data + (out->is_padded ? out_pw + 1 : 0);

    #pragma omp parallel for schedule(static)
    for (int dy = 0; dy < OH; dy++) {
        int sy = yofs[dy];
        float b0 = beta[dy * 2], b1 = beta[dy * 2 + 1];
        for (int dx = 0; dx < OW; dx++) {
            int sx = xofs[dx];
            float a0 = alpha[dx * 2], a1 = alpha[dx * 2 + 1];
            /* tap (r, c) at full-res (sy + r, sx + c) */
            size_t ti[4] = { (size_t)sy * W + sx, (size_t)sy * W + sx + 1,
                             (size_t)(sy + 1) * W + sx, (size_t)(sy + 1) * W + sx + 1 };
            int tx[4] = { sx, sx + 1, sx, sx + 1 }, ty[4] = { sy, sy, sy + 1, sy + 1 };
            int oc = 0;
            for (int k = 0; k < n_src; k++) {
                const Tensor* A = src[k].a;
                int C = A->c;
                assert(C <= 8 && !A->is_padded);
                float v[4][8];
                if (src[k].flow) {
                    const Tensor* F = src[k].flow;
                    for (int t = 0; t < 4; t++)
                        warp_sample3(A->data, HW, W, H, (float)tx[t] + F->data[ti[t]],
                                     (float)ty[t] + F->data[HW + ti[t]], v[t]);
                } else {
                    for (int t = 0; t < 4; t++)
                        for (int c = 0; c < C; c++) v[t][c] = A->data[c * HW + ti[t]];
                }
                for (int c = 0; c < C; c++) {
                    float r0 = v[0][c] * a0 + v[1][c] * a1;
                    float r1 = v[2][c] * a0 + v[3][c] * a1;
                    out_base[(size_t)(oc + c) * out_ch + (size_t)dy * out_pw + dx] = r0 * b0 + r1 * b1;
                }
                oc += C;
            }
        }
    }
    free(xofs); free(alpha); free(yofs); free(beta);
}

/* =====================  THE NETWORK  ===================== */

enum { NBLK = 4 };
static const int BLK_C[NBLK]     = { 192, 128, 96, 64 };   /* IFBlock width */
static const int BLK_CIN[NBLK]   = {   7,  12, 12, 12 };   /* input channels */
static const int BLK_SCALE[NBLK] = {   8,   4,  2,  1 };   /* input downscale */

typedef struct {
    int w, h;                 /* padded frame size */
    Tensor img0, img1, tmap;  /* 3, 3, 1 channels at w x h; tmap holds the timestep */
    Tensor x[NBLK];           /* IFBlock inputs (padded tensors): 7 ch @1/8, then 12 ch @1/4, 1/2, 1/1 */
    Tensor tmp[NBLK];         /* IFBlock outputs: 6 ch = flow (4) + mask (1) + unused, at the input size */
    Tensor up[NBLK - 1];      /* tmp[0..2] resampled to full size */
    Tensor flow[NBLK];        /* accumulated flow after each block: (f0x, f0y, f1x, f1y) at full size */
    Tensor mask[NBLK];        /* accumulated mask logit after each block (mask[0] is a view of up[0]) */
    Tensor out;               /* 3 ch */
    IFBlock* blk[NBLK];
} Net;

static Net* net_create(const char* bin_path, int w, int h, float timestep) {
    Net* n = (Net*)calloc(1, sizeof(Net));
    n->w = w; n->h = h;
    FILE* f = fopen(bin_path, "rb");
    if (!f) { fprintf(stderr, "Failed to open model: %s\n", bin_path); exit(1); }
    for (int b = 0; b < NBLK; b++) n->blk[b] = ifblock_load(f, BLK_CIN[b], BLK_C[b], bin_path);
    if (fgetc(f) != EOF) die_model("unexpected trailing data", bin_path);
    fclose(f);

    detect_caches();
    n->img0 = tensor_make(w, h, 3);
    n->img1 = tensor_make(w, h, 3);
    n->tmap = tensor_make(w, h, 1);
    for (int i = 0; i < w * h; i++) n->tmap.data[i] = timestep;
    for (int b = 0; b < NBLK; b++) {
        int s = BLK_SCALE[b];
        n->x[b] = tensor_make_padded(w / s, h / s, BLK_CIN[b]);
        n->tmp[b] = tensor_make(w / s, h / s, 6);
        n->flow[b] = tensor_make(w, h, 4);
        if (b < NBLK - 1) n->up[b] = tensor_make(w, h, 6);
        if (b > 0) n->mask[b] = tensor_make(w, h, 1);
        ifblock_prepare(n->blk[b], w / s, h / s);
    }
    n->mask[0] = tensor_view(&n->up[0], 4, 1);
    n->out = tensor_make(w, h, 3);
    return n;
}

static void net_free(Net* n) {
    tensor_free(&n->img0); tensor_free(&n->img1); tensor_free(&n->tmap); tensor_free(&n->out);
    for (int b = 0; b < NBLK; b++) {
        tensor_free(&n->x[b]); tensor_free(&n->tmp[b]); tensor_free(&n->flow[b]);
        if (b < NBLK - 1) tensor_free(&n->up[b]);
        if (b > 0) tensor_free(&n->mask[b]);
    }
    free(n);
}

/* Optional per-phase timing (RIFE_PROFILE=1). */
enum { P_PREP, P_IFBLOCK, P_UPDATE, P_MERGE, P_N };
static const char* const P_NAMES[P_N] = { "prep (resample/warp)", "ifblocks", "update (flow, mask)", "merge" };
static double g_prof[P_N];

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec * 1e-6;
}

static double lap(int phase, double since) {
    double t = now_ms();
    g_prof[phase] += t - since;
    return t;
}

/*
 * Coarse-to-fine flow estimation. Every block sees the two frames warped by the
 * flow so far, the timestep, the mask and the flow, downscaled by its scale s,
 * and predicts a residual: flow += s * up(res.flow), mask += up(res.mask).
 *
 *   block 0 (s=8)  input: img0, img1, t                       flow = 8 * up(res.flow)
 *   block 1 (s=4)  input: warp(img0), warp(img1), t, mask, flow/4
 *   block 2 (s=2)  same, at 1/2 scale
 *   block 3 (s=1)  same, at full scale; its output is added without resampling
 *
 * Finally out = warp(img0, flow[0:2]) * sigmoid(mask) + warp(img1, flow[2:4]) * (1 - sigmoid(mask)).
 */
static void net_forward(Net* n) {
    double t = now_ms();
    for (int b = 0; b < NBLK; b++) {
        int s = BLK_SCALE[b];
        Tensor* x = &n->x[b];

        /* ---- assemble the block input ---- */
        if (b == 0) {
            Tensor a = tensor_view(x, 0, 3), c = tensor_view(x, 3, 3), tt = tensor_view(x, 6, 1);
            interp_forward(&n->img0, &a, 1.0f / s);
            interp_forward(&n->img1, &c, 1.0f / s);
            interp_forward(&n->tmap, &tt, 1.0f / s);
        } else if (b < NBLK - 1) {
            Tensor f0 = tensor_view(&n->flow[b - 1], 0, 2), f1 = tensor_view(&n->flow[b - 1], 2, 2);
            WarpSrc src[4] = { { &n->img0, &f0 }, { &n->img1, &f1 }, { &n->tmap, NULL }, { &n->mask[b - 1], NULL } };
            Tensor warped = tensor_view(x, 0, 8), flow = tensor_view(x, 8, 4);
            warpdown_forward(src, 4, 1.0f / s, &warped);
            interp_forward(&n->flow[b - 1], &flow, 1.0f / s);
            scale_forward(&flow, &flow, 1.0f / s);
        } else {   /* full scale: nothing to downscale, so warp straight into the input */
            Tensor f0 = tensor_view(&n->flow[b - 1], 0, 2), f1 = tensor_view(&n->flow[b - 1], 2, 2);
            Tensor w0 = tensor_view(x, 0, 3), w1 = tensor_view(x, 3, 3);
            Tensor tt = tensor_view(x, 6, 1), mk = tensor_view(x, 7, 1), fl = tensor_view(x, 8, 4);
            warp_forward(&n->img0, &f0, &w0);
            warp_forward(&n->img1, &f1, &w1);
            copy_into_padded(&n->tmap, &tt);
            copy_into_padded(&n->mask[b - 1], &mk);
            copy_into_padded(&n->flow[b - 1], &fl);
        }
        t = lap(P_PREP, t);

        ifblock_forward(n->blk[b], x, &n->tmp[b]);
        t = lap(P_IFBLOCK, t);

        /* ---- accumulate the block's residual flow and mask ---- */
        Tensor res_flow = tensor_view(&n->tmp[b], 0, 4), res_mask = tensor_view(&n->tmp[b], 4, 1);
        if (b < NBLK - 1) {
            interp_forward(&n->tmp[b], &n->up[b], (float)s);
            res_flow = tensor_view(&n->up[b], 0, 4);
            res_mask = tensor_view(&n->up[b], 4, 1);
        }
        if (b == 0) {
            scale_forward(&res_flow, &n->flow[0], (float)s);
        } else {
            if (b < NBLK - 1) axpby_forward(&n->flow[b - 1], &res_flow, &n->flow[b], 1.0f, (float)s);
            else              add_forward(&n->flow[b - 1], &res_flow, &n->flow[b]);
            add_forward(&n->mask[b - 1], &res_mask, &n->mask[b]);
        }
        t = lap(P_UPDATE, t);
    }

    Tensor fa = tensor_view(&n->flow[NBLK - 1], 0, 2), fb = tensor_view(&n->flow[NBLK - 1], 2, 2);
    merge_forward(&n->img0, &fa, &n->img1, &fb, &n->mask[NBLK - 1], &n->out);
    lap(P_MERGE, t);
}

/* =====================  IMAGE I/O (stb)  ===================== */

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

static Tensor load_image_rgb(const char* path, int* w, int* h) {
    int c;
    unsigned char* data = stbi_load(path, w, h, &c, 3);
    if (!data) {
        fprintf(stderr, "Failed to load image: %s\n", path);
        exit(1);
    }
    Tensor t = tensor_make(*w, *h, 3);
    for (int y = 0; y < *h; y++) {
        for (int x = 0; x < *w; x++) {
            for (int ch = 0; ch < 3; ch++) {
                t.data[(size_t)(ch * *h + y) * *w + x] = data[(y * *w + x) * 3 + ch] / 255.0f;
            }
        }
    }
    stbi_image_free(data);
    return t;
}

static void save_image_rgb(const char* path, const Tensor* t) {
    int w = t->w, h = t->h;
    size_t sz = (size_t)w * h * 3;
    unsigned char* buf = (unsigned char*)malloc(sz);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int ch = 0; ch < 3; ch++) {
                float v = t->data[(size_t)(ch * h + y) * w + x] * 255.0f;
                int iv = (int)roundf(v);
                buf[(y * w + x) * 3 + ch] = (unsigned char)(iv < 0 ? 0 : iv > 255 ? 255 : iv);
            }
        }
    }
    stbi_write_png(path, w, h, 3, buf, w * 3);
    free(buf);
}

/* =====================  MAIN  ===================== */

int main(int argc, char** argv) {
    /* Options (may appear anywhere): --pad N. Remaining args are positional. */
    int pad = 32;
    int nargs = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--pad") && i + 1 < argc) {
            pad = atoi(argv[++i]);
        } else if (!strncmp(argv[i], "--pad=", 6)) {
            pad = atoi(argv[i] + 6);
        } else {
            argv[nargs++] = argv[i];
        }
    }
    argc = nargs;
    if (pad <= 0 || pad % 32 != 0) {
        fprintf(stderr, "--pad must be a positive multiple of 32 (got %d)\n", pad);
        return 1;
    }
    if (argc < 6) {
        fprintf(stderr, "usage: %s [--pad 32|64] model.bin frame0.png frame1.png timestep out.png [iterations]\n", argv[0]);
        return 1;
    }

    const char* bin_path = argv[1];
    const char* img0_path = argv[2];
    const char* img1_path = argv[3];
    float timestep = (float)atof(argv[4]);
    const char* out_path = argv[5];
    int iters = (argc > 6) ? atoi(argv[6]) : 1;

    int w, h, w1, h1;
    Tensor img0 = load_image_rgb(img0_path, &w, &h);
    Tensor img1 = load_image_rgb(img1_path, &w1, &h1);
    if (w != w1 || h != h1) {
        fprintf(stderr, "frame sizes differ: %dx%d vs %dx%d\n", w, h, w1, h1);
        return 1;
    }
    int wp = (w + pad - 1) / pad * pad;
    int hp = (h + pad - 1) / pad * pad;

    Net* net = net_create(bin_path, wp, hp, timestep);

    /* zero-padded inputs */
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < h; y++) {
            memcpy(net->img0.data + (size_t)(c * hp + y) * wp, img0.data + (size_t)(c * h + y) * w, w * sizeof(float));
            memcpy(net->img1.data + (size_t)(c * hp + y) * wp, img1.data + (size_t)(c * h + y) * w, w * sizeof(float));
        }
    }

    if (iters > 1) {
        net_forward(net);   /* warmup */
        double total_ms = 0.0, min_ms = 1e9, max_ms = 0.0;
        double* times = (double*)malloc(iters * sizeof(double));
        memset(g_prof, 0, sizeof g_prof);
        for (int it = 0; it < iters; it++) {
            double t0 = now_ms();
            net_forward(net);
            double ms = now_ms() - t0;
            times[it] = ms;
            total_ms += ms;
            if (ms < min_ms) min_ms = ms;
            if (ms > max_ms) max_ms = ms;
        }
        double mean_ms = total_ms / iters;
        double variance = 0.0;
        for (int it = 0; it < iters; it++) variance += (times[it] - mean_ms) * (times[it] - mean_ms);
        printf("Warm Inference (%d runs): Mean = %.2f ms, Std = %.2f ms, Min = %.2f ms, Max = %.2f ms\n",
               iters, mean_ms, sqrt(variance / iters), min_ms, max_ms);
        if (getenv("RIFE_PROFILE"))
            for (int p = 0; p < P_N; p++) printf("  %-22s: %6.2f ms\n", P_NAMES[p], g_prof[p] / iters);
        free(times);
    } else {
        double t0 = now_ms();
        net_forward(net);
        printf("Inference time: %.2f ms\n", now_ms() - t0);
        if (getenv("RIFE_PROFILE"))
            for (int p = 0; p < P_N; p++) printf("  %-22s: %6.2f ms\n", P_NAMES[p], g_prof[p]);
    }

    /* crop the padding off */
    Tensor out = tensor_make(w, h, 3);
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < h; y++)
            memcpy(out.data + (size_t)(c * h + y) * w, net->out.data + (size_t)(c * hp + y) * wp, w * sizeof(float));
    save_image_rgb(out_path, &out);

    tensor_free(&img0);
    tensor_free(&img1);
    tensor_free(&out);
    net_free(net);
    return 0;
}
