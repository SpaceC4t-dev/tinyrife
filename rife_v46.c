/*
 * RIFE v4.6 — High-Performance Single-File C99/C11 CPU Inference
 *   Input:  two RGB images (HxWx3, uint8) + timestep in [0,1]
 *   Output: interpolated RGB image (HxWx3, uint8)
 *
 * Build:  gcc -O3 -march=native -fopenmp rife_v46.c -o rife_v46 -lm
 * Run:    ./rife_v46 flownet-v46.bin frame0.png frame1.png 0.5 out.png
 *
 * Fully vectorized AVX2/FMA execution with persistent workspaces,
 * zero-copy splits/crops, and pre-reordered weights. Zero external deps.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include <time.h>

#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* =====================  BASIC TYPES & MEMORY  ===================== */

typedef struct {
    int w, h, c;
    float* data;
    int is_alias;
    int is_padded;
} Tensor;

static inline size_t tensor_bytes(const Tensor* t) {
    if (t->is_padded) {
        return (size_t)t->c * (t->h + 2) * (t->w + 2) * sizeof(float);
    }
    return (size_t)t->w * t->h * t->c * sizeof(float);
}

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

/* Persistent workspace shared by conv input padding and deconv HWC reordering.
 * Sized in main() from the inferred blob shapes (in floats). */
static float* g_pin_scratch = NULL;
static size_t g_pin_scratch_size = 0;

static void scratch_check(size_t need, const char* who) {
    if (need > g_pin_scratch_size) {
        fprintf(stderr, "%s: scratch too small (%zu > %zu floats)\n", who, need, g_pin_scratch_size);
        abort();
    }
}
double s1_time = 0, s2_time = 0, pad_time = 0;
double s1_8_time = 0, s1_16_time = 0, s1_32_time = 0, s1_64_time = 0;

/* =====================  GRAPH & LAYER DEFINITIONS  ===================== */

#define MAX_LAYERS 512
#define MAX_BLOBS  1024
#define MAX_NAME   64

typedef enum {
    LT_INPUT,
    LT_SPLIT,
    LT_CONCAT,
    LT_INTERP,
    LT_CONV,
    LT_DECONV,
    LT_RELU,
    LT_BINARYOP,
    LT_PIXELSHUFFLE,
    LT_CROP,
    LT_WARP,
    LT_ELTWISE,
    LT_SIGMOID,
    LT_NOP
} LayerType;

typedef struct {
    LayerType type;
    char name[MAX_NAME];
    int in_count, out_count;
    int in_idx[8], out_idx[8];
    int res_blob_idx; /* For fused ResBlock (conv + res_add + leakyrelu) */

    /* Conv / Deconv */
    int num_output, kernel, stride, pad, dilation, bias_term;
    int act_type;
    float act_params[4];
    int weight_data_size;
    float* weights;
    float* bias;

    /* Interp */
    int resize_type;
    float scale_w, scale_h;
    int out_w, out_h;
    int align_corner;

    /* Crop (channel slice) */
    int crop_start, crop_end;

    /* BinaryOp */
    int op_type;
    int with_scalar;
    float b_scalar;

    /* PixelShuffle */
    int upscale;
    int ps_mode;

    /* Eltwise */
    int eltwise_op;
    float eltwise_coeffs[8];
    int num_coeffs;

    /* ReLU */
    float relu_slope;
} Layer;

typedef struct {
    Layer layers[MAX_LAYERS];
    int n_layers;
    char blob_names[MAX_BLOBS][MAX_NAME];
    int n_blobs;
} Graph;

static int blob_idx(Graph* g, const char* name) {
    for (int i = 0; i < g->n_blobs; i++) {
        if (strcmp(g->blob_names[i], name) == 0) return i;
    }
    assert(g->n_blobs < MAX_BLOBS);
    strncpy(g->blob_names[g->n_blobs], name, MAX_NAME - 1);
    g->blob_names[g->n_blobs][MAX_NAME - 1] = '\0';
    return g->n_blobs++;
}

static LayerType layer_type_from_name(const char* s) {
    if (!strcmp(s, "Input")) return LT_INPUT;
    if (!strcmp(s, "Split")) return LT_SPLIT;
    if (!strcmp(s, "Concat")) return LT_CONCAT;
    if (!strcmp(s, "Interp")) return LT_INTERP;
    if (!strcmp(s, "Convolution")) return LT_CONV;
    if (!strcmp(s, "Deconvolution")) return LT_DECONV;
    if (!strcmp(s, "ReLU")) return LT_RELU;
    if (!strcmp(s, "BinaryOp")) return LT_BINARYOP;
    if (!strcmp(s, "PixelShuffle")) return LT_PIXELSHUFFLE;
    if (!strcmp(s, "Crop")) return LT_CROP;
    if (!strcmp(s, "rife.Warp")) return LT_WARP;
    if (!strcmp(s, "Eltwise")) return LT_ELTWISE;
    if (!strcmp(s, "Sigmoid")) return LT_SIGMOID;
    return -1;
}

/* =====================  PARAM PARSING  ===================== */

static int parse_param_line(Graph* g, char* line) {
    char* tokens[128];
    int ntokens = 0;
    char* p = line;
    while (*p && ntokens < 128) {
        while (*p && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) *p++ = '\0';
        if (!*p) break;
        tokens[ntokens++] = p;
        while (*p && *p != ' ' && *p != '\t' && *p != '\r' && *p != '\n') p++;
    }

    if (ntokens < 4) return -1;

    LayerType ltype = layer_type_from_name(tokens[0]);
    if ((int)ltype < 0) return -1;

    assert(g->n_layers < MAX_LAYERS);
    Layer* L = &g->layers[g->n_layers++];
    memset(L, 0, sizeof(Layer));
    L->type = ltype;
    strncpy(L->name, tokens[1], MAX_NAME - 1);
    L->in_count = atoi(tokens[2]);
    L->out_count = atoi(tokens[3]);

    int tok_idx = 4;
    for (int i = 0; i < L->in_count; i++) {
        L->in_idx[i] = blob_idx(g, tokens[tok_idx++]);
    }
    for (int i = 0; i < L->out_count; i++) {
        L->out_idx[i] = blob_idx(g, tokens[tok_idx++]);
    }

    L->kernel = 1;
    L->stride = 1;
    L->dilation = 1;
    L->scale_w = 1.0f;
    L->scale_h = 1.0f;
    L->upscale = 1;
    L->res_blob_idx = -1;

    while (tok_idx < ntokens) {
        char* token = tokens[tok_idx++];
        char* eq = strchr(token, '=');
        if (!eq) continue;
        *eq = '\0';
        char* key_str = token;
        char* val_str = eq + 1;

        int raw_key = atoi(key_str);
        if (raw_key <= -23300) {
            int param_id = -(raw_key + 23300);
            char* val_ptr = val_str;
            int arr_len = atoi(val_ptr);
            char* next_comma = strchr(val_ptr, ',');
            float arr_vals[8] = {0};
            for (int k = 0; k < arr_len && next_comma; k++) {
                val_ptr = next_comma + 1;
                arr_vals[k] = (float)atof(val_ptr);
                next_comma = strchr(val_ptr, ',');
            }

            if (L->type == LT_CONV || L->type == LT_DECONV) {
                if (param_id == 10 && arr_len > 0) {
                    L->act_params[0] = arr_vals[0];
                    if (arr_len > 1) L->act_params[1] = arr_vals[1];
                }
            } else if (L->type == LT_CROP) {
                if (param_id == 9 && arr_len > 0) {
                    L->crop_start = (int)arr_vals[0];
                } else if (param_id == 10 && arr_len > 0) {
                    L->crop_end = (int)arr_vals[0];
                }
            } else if (L->type == LT_ELTWISE) {
                if (param_id == 1) {
                    L->num_coeffs = arr_len;
                    for (int k = 0; k < arr_len && k < 8; k++) {
                        L->eltwise_coeffs[k] = arr_vals[k];
                    }
                }
            }
        } else {
            int key = raw_key;
            if (L->type == LT_CONV || L->type == LT_DECONV) {
                if (key == 0) L->num_output = atoi(val_str);
                else if (key == 1) L->kernel = atoi(val_str);
                else if (key == 2) L->dilation = atoi(val_str);
                else if (key == 3) L->stride = atoi(val_str);
                else if (key == 4) L->pad = atoi(val_str);
                else if (key == 5) L->bias_term = atoi(val_str);
                else if (key == 6) L->weight_data_size = atoi(val_str);
                else if (key == 9) L->act_type = atoi(val_str);
            } else if (L->type == LT_RELU) {
                if (key == 0) L->relu_slope = (float)atof(val_str);
            } else if (L->type == LT_INTERP) {
                if (key == 0) L->resize_type = atoi(val_str);
                else if (key == 1) L->scale_h = (float)atof(val_str);
                else if (key == 2) L->scale_w = (float)atof(val_str);
                else if (key == 3) L->out_h = atoi(val_str);
                else if (key == 4) L->out_w = atoi(val_str);
                else if (key == 6) L->align_corner = atoi(val_str);
            } else if (L->type == LT_BINARYOP) {
                if (key == 0) L->op_type = atoi(val_str);
                else if (key == 1) L->with_scalar = atoi(val_str);
                else if (key == 2) L->b_scalar = (float)atof(val_str);
            } else if (L->type == LT_PIXELSHUFFLE) {
                if (key == 0) L->upscale = atoi(val_str);
                else if (key == 1) L->ps_mode = atoi(val_str);
            } else if (L->type == LT_ELTWISE) {
                if (key == 0) L->eltwise_op = atoi(val_str);
            }
        }
    }

    return 0;
}

static int graph_load_param(Graph* g, const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    char line[4096];
    if (!fgets(line, sizeof(line), f)) { fclose(f); return -1; }
    if (!fgets(line, sizeof(line), f)) { fclose(f); return -1; }
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        parse_param_line(g, line);
    }
    fclose(f);
    return 0;
}

/* =====================  BIN WEIGHT LOADING  ===================== */

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

static int graph_load_bin(Graph* g, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    for (int i = 0; i < g->n_layers; i++) {
        Layer* L = &g->layers[i];
        if (L->type != LT_CONV && L->type != LT_DECONV) continue;

        uint32_t tag = 0;
        if (fread(&tag, sizeof(uint32_t), 1, f) != 1 || tag != TAG_FP16) {
            fprintf(stderr, "Invalid tag 0x%08x at layer %s\n", tag, L->name);
            fclose(f);
            return -1;
        }

        int wsize = L->weight_data_size;
        size_t align_bytes = (wsize * sizeof(uint16_t) + 3) & ~3;
        uint16_t* halfs = (uint16_t*)malloc(align_bytes);
        if (fread(halfs, 1, align_bytes, f) != align_bytes) {
            fprintf(stderr, "Failed reading weights for %s\n", L->name);
            free(halfs);
            fclose(f);
            return -1;
        }

        float* raw_weights = (float*)malloc(wsize * sizeof(float));
        for (int k = 0; k < wsize; k++) {
            raw_weights[k] = fp16_to_f32(halfs[k]);
        }
        free(halfs);

        if (L->type == LT_DECONV) {
            /* Deconv in RIFE: K=4, S=2, P=1, OC=24
             * Reorder to: [kh][kw][c][oc_b][8] where oc_b in 0..2 */
            int OC = L->num_output;
            assert(OC == 24);
            int C = wsize / (OC * 16);
            size_t reord_sz = (size_t)16 * C * 24 * sizeof(float);
            L->weights = (float*)aligned_alloc(64, (reord_sz + 63) & ~63);

            for (int kh = 0; kh < 4; kh++) {
                for (int kw = 0; kw < 4; kw++) {
                    for (int c = 0; c < C; c++) {
                        for (int oc_b = 0; oc_b < 3; oc_b++) {
                            for (int s = 0; s < 8; s++) {
                                int oc = oc_b * 8 + s;
                                L->weights[(((kh * 4 + kw) * C + c) * 3 + oc_b) * 8 + s] =
                                    raw_weights[((oc * C + c) * 4 + kh) * 4 + kw];
                            }
                        }
                    }
                }
            }
            free(raw_weights);
        } else if (L->type == LT_CONV && L->stride == 2) {
            /* Conv S=2: K=3, P=1, OC is multiple of 8
             * Reorder to: [OC/8][C][9][8] */
            int OC = L->num_output;
            int C = wsize / (OC * 9);
            size_t reord_sz = (size_t)OC * C * 9 * sizeof(float);
            L->weights = (float*)aligned_alloc(64, (reord_sz + 63) & ~63);

            for (int oc_b = 0; oc_b < OC / 8; oc_b++) {
                for (int c = 0; c < C; c++) {
                    for (int k = 0; k < 9; k++) {
                        for (int s = 0; s < 8; s++) {
                            int oc = oc_b * 8 + s;
                            L->weights[((oc_b * C + c) * 9 + k) * 8 + s] =
                                raw_weights[(oc * C + c) * 9 + k];
                        }
                    }
                }
            }
            free(raw_weights);
        } else {
            /* Conv S=1: NCHW [OC][C][9] */
            size_t w_sz = (size_t)wsize * sizeof(float);
            L->weights = (float*)aligned_alloc(64, (w_sz + 63) & ~63);
            memcpy(L->weights, raw_weights, w_sz);
            free(raw_weights);
        }

        if (L->bias_term) {
            size_t b_sz = (size_t)L->num_output * sizeof(float);
            L->bias = (float*)aligned_alloc(64, (b_sz + 63) & ~63);
            if (fread(L->bias, sizeof(float), L->num_output, f) != (size_t)L->num_output) {
                fprintf(stderr, "Failed reading bias for %s\n", L->name);
                fclose(f);
                return -1;
            }
        }
    }

    fclose(f);
    return 0;
}

/* =====================  HIGH-PERFORMANCE LAYER OPERATORS  ===================== */

static inline __m256 apply_act(__m256 acc, int act_type, __m256 vslope, __m256 vzero) {
    if (act_type == 2) {
        return _mm256_blendv_ps(_mm256_mul_ps(acc, vslope), acc, _mm256_cmp_ps(acc, vzero, _CMP_GT_OQ));
    } else if (act_type == 1) {
        return _mm256_max_ps(acc, vzero);
    }
    return acc;
}

/* Generic 3x3 stride-1 conv over output channels [oc_begin, oc_end) and output
 * columns [ow_begin, OW). Handles any width (masked 8-wide tails) and any channel
 * count (2 OC at a time, odd last channel on its own). Used for the edges the
 * register-tiled paths below don't cover. */
static void conv_s1_tail(const Layer* L, const float* pin, const Tensor* res, Tensor* out,
                         int W, int H, int C, int oc_begin, int oc_end, int ow_begin) {
    int OW = out->w, OH = out->h;
    if (oc_begin >= oc_end || ow_begin >= OW) return;
    int PW = W + 2, PH = H + 2;
    const float* wt = L->weights;
    const float* B = L->bias;
    int act_type = L->act_type;
    __m256 vslope = _mm256_set1_ps(L->act_params[0]);
    __m256 vzero = _mm256_setzero_ps();
    const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

    int out_pw = out->is_padded ? (OW + 2) : OW;
    size_t out_ch_sz = (size_t)out_pw * (out->is_padded ? (OH + 2) : OH);
    int out_col_base = out->is_padded ? 1 : 0;

    int res_pw = (res && res->is_padded) ? (OW + 2) : OW;
    size_t res_ch_sz = (res && res->is_padded) ? (size_t)res_pw * (OH + 2) : (size_t)OW * OH;
    int res_col_base = (res && res->is_padded) ? 1 : 0;

    int n_pairs = (oc_end - oc_begin + 1) / 2;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int p = 0; p < n_pairs; p++) {
        for (int oh = 0; oh < OH; oh++) {
            int oc0 = oc_begin + 2 * p;
            int has1 = (oc0 + 1) < oc_end;
            int oc1 = has1 ? oc0 + 1 : oc0;
            const float* wt0 = wt + (size_t)oc0 * C * 9;
            const float* wt1 = wt + (size_t)oc1 * C * 9;
            __m256 vbias0 = _mm256_set1_ps(B ? B[oc0] : 0.0f);
            __m256 vbias1 = _mm256_set1_ps(B ? B[oc1] : 0.0f);
            size_t out_row_off = (size_t)(out->is_padded ? (oh + 1) : oh) * out_pw + out_col_base;
            size_t res_row_off = (size_t)((res && res->is_padded) ? (oh + 1) : oh) * res_pw + res_col_base;
            float* out_row0 = out->data + (size_t)oc0 * out_ch_sz + out_row_off;
            float* out_row1 = out->data + (size_t)oc1 * out_ch_sz + out_row_off;

            for (int ow = ow_begin; ow < OW; ow += 8) {
                int n = OW - ow < 8 ? OW - ow : 8;
                __m256i mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(n), lane);
                __m256 a0 = vbias0, a1 = vbias1;

                for (int c = 0; c < C; c++) {
                    const float* pin_c = pin + (size_t)c * PH * PW + ow;
                    const float* w0 = wt0 + c * 9;
                    const float* w1 = wt1 + c * 9;
                    for (int ky = 0; ky < 3; ky++) {
                        const float* r = pin_c + (size_t)(oh + ky) * PW;
                        for (int kx = 0; kx < 3; kx++) {
                            __m256 in = _mm256_maskload_ps(r + kx, mask);
                            a0 = _mm256_fmadd_ps(in, _mm256_set1_ps(w0[ky * 3 + kx]), a0);
                            a1 = _mm256_fmadd_ps(in, _mm256_set1_ps(w1[ky * 3 + kx]), a1);
                        }
                    }
                }
                if (res) {
                    a0 = _mm256_add_ps(a0, _mm256_maskload_ps(res->data + (size_t)oc0 * res_ch_sz + res_row_off + ow, mask));
                    a1 = _mm256_add_ps(a1, _mm256_maskload_ps(res->data + (size_t)oc1 * res_ch_sz + res_row_off + ow, mask));
                }
                _mm256_maskstore_ps(out_row0 + ow, mask, apply_act(a0, act_type, vslope, vzero));
                if (has1) _mm256_maskstore_ps(out_row1 + ow, mask, apply_act(a1, act_type, vslope, vzero));
            }
        }
    }
}

static void conv_s1_forward(const Layer* L, const float* pin, const Tensor* res, Tensor* out, int in_w, int in_h, int in_c) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int W = in_w, H = in_h, C = in_c;
    int OW = out->w, OH = out->h, OC = out->c;
    int PW = W + 2, PH = H + 2;
    const float* wt = L->weights;
    const float* B = L->bias;
    int act_type = L->act_type;
    float act_param = L->act_params[0];
    __m256 vslope = _mm256_set1_ps(act_param);
    __m256 vzero = _mm256_setzero_ps();

    int out_pw = out->is_padded ? (OW + 2) : OW;
    size_t out_ch_sz = (size_t)out_pw * (out->is_padded ? (OH + 2) : OH);
    int out_col_base = out->is_padded ? 1 : 0;

    int res_pw = (res && res->is_padded) ? (OW + 2) : OW;
    size_t res_ch_sz = (res && res->is_padded) ? (size_t)res_pw * (OH + 2) : (size_t)OW * OH;
    int res_col_base = (res && res->is_padded) ? 1 : 0;

    if (OW == 8) {
        #pragma omp parallel for schedule(static)
        for (int oc = 0; oc < OC - 3; oc += 4) {
            int oc0 = oc, oc1 = oc + 1, oc2 = oc + 2, oc3 = oc + 3;
            __m256 vbias0 = _mm256_set1_ps(B ? B[oc0] : 0.0f);
            __m256 vbias1 = _mm256_set1_ps(B ? B[oc1] : 0.0f);
            __m256 vbias2 = _mm256_set1_ps(B ? B[oc2] : 0.0f);
            __m256 vbias3 = _mm256_set1_ps(B ? B[oc3] : 0.0f);

            const float* wt0 = wt + (size_t)oc0 * C * 9;
            const float* wt1 = wt + (size_t)oc1 * C * 9;
            const float* wt2 = wt + (size_t)oc2 * C * 9;
            const float* wt3 = wt + (size_t)oc3 * C * 9;

            float* out0 = out->data + (size_t)oc0 * out_ch_sz;
            float* out1 = out->data + (size_t)oc1 * out_ch_sz;
            float* out2 = out->data + (size_t)oc2 * out_ch_sz;
            float* out3 = out->data + (size_t)oc3 * out_ch_sz;
            const float* res0 = res ? res->data + (size_t)oc0 * res_ch_sz : NULL;
            const float* res1 = res ? res->data + (size_t)oc1 * res_ch_sz : NULL;
            const float* res2 = res ? res->data + (size_t)oc2 * res_ch_sz : NULL;
            const float* res3 = res ? res->data + (size_t)oc3 * res_ch_sz : NULL;

            for (int oh = 0; oh < OH; oh++) {
                size_t out_row_off = (size_t)(out->is_padded ? (oh + 1) : oh) * out_pw + out_col_base;
                size_t res_row_off = (size_t)((res && res->is_padded) ? (oh + 1) : oh) * res_pw + res_col_base;

                __m256 a0 = vbias0, a1 = vbias1, a2 = vbias2, a3 = vbias3;
                for (int c = 0; c < C; c++) {
                    const float* pin_c = pin + (size_t)c * PH * PW;
                    const float* r0 = pin_c + (oh + 0) * PW;
                    const float* r1 = pin_c + (oh + 1) * PW;
                    const float* r2 = pin_c + (oh + 2) * PW;
                    const float* w0 = wt0 + c * 9;
                    const float* w1 = wt1 + c * 9;
                    const float* w2 = wt2 + c * 9;
                    const float* w3 = wt3 + c * 9;

                    #define FMA_4OC_1OW(rk, off, k) do { \
                        __m256 kw0 = _mm256_set1_ps(w0[k]); \
                        __m256 kw1 = _mm256_set1_ps(w1[k]); \
                        __m256 kw2 = _mm256_set1_ps(w2[k]); \
                        __m256 kw3 = _mm256_set1_ps(w3[k]); \
                        __m256 in0 = _mm256_loadu_ps((rk) + (off)); \
                        a0 = _mm256_fmadd_ps(in0, kw0, a0); \
                        a1 = _mm256_fmadd_ps(in0, kw1, a1); \
                        a2 = _mm256_fmadd_ps(in0, kw2, a2); \
                        a3 = _mm256_fmadd_ps(in0, kw3, a3); \
                    } while(0)

                    FMA_4OC_1OW(r0, 0, 0); FMA_4OC_1OW(r0, 1, 1); FMA_4OC_1OW(r0, 2, 2);
                    FMA_4OC_1OW(r1, 0, 3); FMA_4OC_1OW(r1, 1, 4); FMA_4OC_1OW(r1, 2, 5);
                    FMA_4OC_1OW(r2, 0, 6); FMA_4OC_1OW(r2, 1, 7); FMA_4OC_1OW(r2, 2, 8);
                    #undef FMA_4OC_1OW
                }
                if (res) {
                    a0 = _mm256_add_ps(a0, _mm256_loadu_ps(res0 + res_row_off));
                    a1 = _mm256_add_ps(a1, _mm256_loadu_ps(res1 + res_row_off));
                    a2 = _mm256_add_ps(a2, _mm256_loadu_ps(res2 + res_row_off));
                    a3 = _mm256_add_ps(a3, _mm256_loadu_ps(res3 + res_row_off));
                }
                a0 = apply_act(a0, act_type, vslope, vzero);
                a1 = apply_act(a1, act_type, vslope, vzero);
                a2 = apply_act(a2, act_type, vslope, vzero);
                a3 = apply_act(a3, act_type, vslope, vzero);

                _mm256_storeu_ps(out0 + out_row_off, a0);
                _mm256_storeu_ps(out1 + out_row_off, a1);
                _mm256_storeu_ps(out2 + out_row_off, a2);
                _mm256_storeu_ps(out3 + out_row_off, a3);
            }
        }
        conv_s1_tail(L, pin, res, out, W, H, C, OC - OC % 4, OC, 0);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        s1_8_time += (t1.tv_sec - t0.tv_sec)*1000.0 + (t1.tv_nsec - t0.tv_nsec)*1e-6;
        return;
    }

    if (OW == 16) {
        #pragma omp parallel for schedule(static)
        for (int oc = 0; oc < OC - 3; oc += 4) {
            int oc0 = oc, oc1 = oc + 1, oc2 = oc + 2, oc3 = oc + 3;
            __m256 vbias0 = _mm256_set1_ps(B ? B[oc0] : 0.0f);
            __m256 vbias1 = _mm256_set1_ps(B ? B[oc1] : 0.0f);
            __m256 vbias2 = _mm256_set1_ps(B ? B[oc2] : 0.0f);
            __m256 vbias3 = _mm256_set1_ps(B ? B[oc3] : 0.0f);

            const float* wt0 = wt + (size_t)oc0 * C * 9;
            const float* wt1 = wt + (size_t)oc1 * C * 9;
            const float* wt2 = wt + (size_t)oc2 * C * 9;
            const float* wt3 = wt + (size_t)oc3 * C * 9;

            float* out0 = out->data + (size_t)oc0 * out_ch_sz;
            float* out1 = out->data + (size_t)oc1 * out_ch_sz;
            float* out2 = out->data + (size_t)oc2 * out_ch_sz;
            float* out3 = out->data + (size_t)oc3 * out_ch_sz;
            const float* res0 = res ? res->data + (size_t)oc0 * res_ch_sz : NULL;
            const float* res1 = res ? res->data + (size_t)oc1 * res_ch_sz : NULL;
            const float* res2 = res ? res->data + (size_t)oc2 * res_ch_sz : NULL;
            const float* res3 = res ? res->data + (size_t)oc3 * res_ch_sz : NULL;

            for (int oh = 0; oh < OH; oh++) {
                size_t out_row_off = (size_t)(out->is_padded ? (oh + 1) : oh) * out_pw + out_col_base;
                size_t res_row_off = (size_t)((res && res->is_padded) ? (oh + 1) : oh) * res_pw + res_col_base;

                __m256 a0_0 = vbias0, a1_0 = vbias0;
                __m256 a0_1 = vbias1, a1_1 = vbias1;
                __m256 a0_2 = vbias2, a1_2 = vbias2;
                __m256 a0_3 = vbias3, a1_3 = vbias3;

                for (int c = 0; c < C; c++) {
                    const float* pin_c = pin + (size_t)c * PH * PW;
                    const float* r0 = pin_c + (oh + 0) * PW;
                    const float* r1 = pin_c + (oh + 1) * PW;
                    const float* r2 = pin_c + (oh + 2) * PW;
                    const float* w0 = wt0 + c * 9;
                    const float* w1 = wt1 + c * 9;
                    const float* w2 = wt2 + c * 9;
                    const float* w3 = wt3 + c * 9;

                    #define FMA_4OC_2OW(rk, off, k) do { \
                        __m256 kw0 = _mm256_set1_ps(w0[k]); \
                        __m256 kw1 = _mm256_set1_ps(w1[k]); \
                        __m256 kw2 = _mm256_set1_ps(w2[k]); \
                        __m256 kw3 = _mm256_set1_ps(w3[k]); \
                        __m256 in0 = _mm256_loadu_ps((rk) + (off) + 0); \
                        __m256 in1 = _mm256_loadu_ps((rk) + (off) + 8); \
                        a0_0 = _mm256_fmadd_ps(in0, kw0, a0_0); \
                        a1_0 = _mm256_fmadd_ps(in1, kw0, a1_0); \
                        a0_1 = _mm256_fmadd_ps(in0, kw1, a0_1); \
                        a1_1 = _mm256_fmadd_ps(in1, kw1, a1_1); \
                        a0_2 = _mm256_fmadd_ps(in0, kw2, a0_2); \
                        a1_2 = _mm256_fmadd_ps(in1, kw2, a1_2); \
                        a0_3 = _mm256_fmadd_ps(in0, kw3, a0_3); \
                        a1_3 = _mm256_fmadd_ps(in1, kw3, a1_3); \
                    } while(0)

                    FMA_4OC_2OW(r0, 0, 0); FMA_4OC_2OW(r0, 1, 1); FMA_4OC_2OW(r0, 2, 2);
                    FMA_4OC_2OW(r1, 0, 3); FMA_4OC_2OW(r1, 1, 4); FMA_4OC_2OW(r1, 2, 5);
                    FMA_4OC_2OW(r2, 0, 6); FMA_4OC_2OW(r2, 1, 7); FMA_4OC_2OW(r2, 2, 8);
                    #undef FMA_4OC_2OW
                }
                if (res) {
                    a0_0 = _mm256_add_ps(a0_0, _mm256_loadu_ps(res0 + res_row_off + 0));
                    a1_0 = _mm256_add_ps(a1_0, _mm256_loadu_ps(res0 + res_row_off + 8));
                    a0_1 = _mm256_add_ps(a0_1, _mm256_loadu_ps(res1 + res_row_off + 0));
                    a1_1 = _mm256_add_ps(a1_1, _mm256_loadu_ps(res1 + res_row_off + 8));
                    a0_2 = _mm256_add_ps(a0_2, _mm256_loadu_ps(res2 + res_row_off + 0));
                    a1_2 = _mm256_add_ps(a1_2, _mm256_loadu_ps(res2 + res_row_off + 8));
                    a0_3 = _mm256_add_ps(a0_3, _mm256_loadu_ps(res3 + res_row_off + 0));
                    a1_3 = _mm256_add_ps(a1_3, _mm256_loadu_ps(res3 + res_row_off + 8));
                }

                a0_0 = apply_act(a0_0, act_type, vslope, vzero);
                a1_0 = apply_act(a1_0, act_type, vslope, vzero);
                a0_1 = apply_act(a0_1, act_type, vslope, vzero);
                a1_1 = apply_act(a1_1, act_type, vslope, vzero);
                a0_2 = apply_act(a0_2, act_type, vslope, vzero);
                a1_2 = apply_act(a1_2, act_type, vslope, vzero);
                a0_3 = apply_act(a0_3, act_type, vslope, vzero);
                a1_3 = apply_act(a1_3, act_type, vslope, vzero);

                _mm256_storeu_ps(out0 + out_row_off + 0, a0_0); _mm256_storeu_ps(out0 + out_row_off + 8, a1_0);
                _mm256_storeu_ps(out1 + out_row_off + 0, a0_1); _mm256_storeu_ps(out1 + out_row_off + 8, a1_1);
                _mm256_storeu_ps(out2 + out_row_off + 0, a0_2); _mm256_storeu_ps(out2 + out_row_off + 8, a1_2);
                _mm256_storeu_ps(out3 + out_row_off + 0, a0_3); _mm256_storeu_ps(out3 + out_row_off + 8, a1_3);
            }
        }
        conv_s1_tail(L, pin, res, out, W, H, C, OC - OC % 4, OC, 0);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        s1_16_time += (t1.tv_sec - t0.tv_sec)*1000.0 + (t1.tv_nsec - t0.tv_nsec)*1e-6;
        return;
    }

    if (OW == 32) {
        #pragma omp parallel for schedule(static)
        for (int oc = 0; oc < OC - 2; oc += 3) {
            int oc0 = oc, oc1 = oc + 1, oc2 = oc + 2;
            __m256 vbias0 = _mm256_set1_ps(B ? B[oc0] : 0.0f);
            __m256 vbias1 = _mm256_set1_ps(B ? B[oc1] : 0.0f);
            __m256 vbias2 = _mm256_set1_ps(B ? B[oc2] : 0.0f);

            const float* wt0 = wt + (size_t)oc0 * C * 9;
            const float* wt1 = wt + (size_t)oc1 * C * 9;
            const float* wt2 = wt + (size_t)oc2 * C * 9;

            float* out0 = out->data + (size_t)oc0 * out_ch_sz;
            float* out1 = out->data + (size_t)oc1 * out_ch_sz;
            float* out2 = out->data + (size_t)oc2 * out_ch_sz;
            const float* res0 = res ? res->data + (size_t)oc0 * res_ch_sz : NULL;
            const float* res1 = res ? res->data + (size_t)oc1 * res_ch_sz : NULL;
            const float* res2 = res ? res->data + (size_t)oc2 * res_ch_sz : NULL;

            for (int oh = 0; oh < OH; oh++) {
                size_t out_row_off = (size_t)(out->is_padded ? (oh + 1) : oh) * out_pw + out_col_base;
                size_t res_row_off = (size_t)((res && res->is_padded) ? (oh + 1) : oh) * res_pw + res_col_base;
                float* out_row0 = out0 + out_row_off;
                float* out_row1 = out1 + out_row_off;
                float* out_row2 = out2 + out_row_off;

                __m256 a0_0 = vbias0, a1_0 = vbias0, a2_0 = vbias0, a3_0 = vbias0;
                __m256 a0_1 = vbias1, a1_1 = vbias1, a2_1 = vbias1, a3_1 = vbias1;
                __m256 a0_2 = vbias2, a1_2 = vbias2, a2_2 = vbias2, a3_2 = vbias2;

                for (int c = 0; c < C; c++) {
                    const float* pin_c = pin + (size_t)c * PH * PW;
                    const float* r0 = pin_c + (oh + 0) * PW;
                    const float* r1 = pin_c + (oh + 1) * PW;
                    const float* r2 = pin_c + (oh + 2) * PW;
                    const float* w0 = wt0 + c * 9;
                    const float* w1 = wt1 + c * 9;
                    const float* w2 = wt2 + c * 9;

                    #define FMA_3OC_4OW(rk, off, kw_idx) do { \
                        __m256 kw0 = _mm256_set1_ps(w0[kw_idx]); \
                        __m256 kw1 = _mm256_set1_ps(w1[kw_idx]); \
                        __m256 kw2 = _mm256_set1_ps(w2[kw_idx]); \
                        __m256 in0 = _mm256_loadu_ps((rk) + (off) + 0); \
                        a0_0 = _mm256_fmadd_ps(in0, kw0, a0_0); \
                        a0_1 = _mm256_fmadd_ps(in0, kw1, a0_1); \
                        a0_2 = _mm256_fmadd_ps(in0, kw2, a0_2); \
                        __m256 in1 = _mm256_loadu_ps((rk) + (off) + 8); \
                        a1_0 = _mm256_fmadd_ps(in1, kw0, a1_0); \
                        a1_1 = _mm256_fmadd_ps(in1, kw1, a1_1); \
                        a1_2 = _mm256_fmadd_ps(in1, kw2, a1_2); \
                        __m256 in2 = _mm256_loadu_ps((rk) + (off) + 16); \
                        a2_0 = _mm256_fmadd_ps(in2, kw0, a2_0); \
                        a2_1 = _mm256_fmadd_ps(in2, kw1, a2_1); \
                        a2_2 = _mm256_fmadd_ps(in2, kw2, a2_2); \
                        __m256 in3 = _mm256_loadu_ps((rk) + (off) + 24); \
                        a3_0 = _mm256_fmadd_ps(in3, kw0, a3_0); \
                        a3_1 = _mm256_fmadd_ps(in3, kw1, a3_1); \
                        a3_2 = _mm256_fmadd_ps(in3, kw2, a3_2); \
                    } while(0)

                    FMA_3OC_4OW(r0, 0, 0); FMA_3OC_4OW(r0, 1, 1); FMA_3OC_4OW(r0, 2, 2);
                    FMA_3OC_4OW(r1, 0, 3); FMA_3OC_4OW(r1, 1, 4); FMA_3OC_4OW(r1, 2, 5);
                    FMA_3OC_4OW(r2, 0, 6); FMA_3OC_4OW(r2, 1, 7); FMA_3OC_4OW(r2, 2, 8);
                    #undef FMA_3OC_4OW
                }

                if (res) {
                    const float* r0 = res0 + res_row_off;
                    const float* r1 = res1 + res_row_off;
                    const float* r2 = res2 + res_row_off;
                    a0_0 = _mm256_add_ps(a0_0, _mm256_loadu_ps(r0 + 0));
                    a1_0 = _mm256_add_ps(a1_0, _mm256_loadu_ps(r0 + 8));
                    a2_0 = _mm256_add_ps(a2_0, _mm256_loadu_ps(r0 + 16));
                    a3_0 = _mm256_add_ps(a3_0, _mm256_loadu_ps(r0 + 24));

                    a0_1 = _mm256_add_ps(a0_1, _mm256_loadu_ps(r1 + 0));
                    a1_1 = _mm256_add_ps(a1_1, _mm256_loadu_ps(r1 + 8));
                    a2_1 = _mm256_add_ps(a2_1, _mm256_loadu_ps(r1 + 16));
                    a3_1 = _mm256_add_ps(a3_1, _mm256_loadu_ps(r1 + 24));

                    a0_2 = _mm256_add_ps(a0_2, _mm256_loadu_ps(r2 + 0));
                    a1_2 = _mm256_add_ps(a1_2, _mm256_loadu_ps(r2 + 8));
                    a2_2 = _mm256_add_ps(a2_2, _mm256_loadu_ps(r2 + 16));
                    a3_2 = _mm256_add_ps(a3_2, _mm256_loadu_ps(r2 + 24));
                }

                a0_0 = apply_act(a0_0, act_type, vslope, vzero);
                a1_0 = apply_act(a1_0, act_type, vslope, vzero);
                a2_0 = apply_act(a2_0, act_type, vslope, vzero);
                a3_0 = apply_act(a3_0, act_type, vslope, vzero);
                a0_1 = apply_act(a0_1, act_type, vslope, vzero);
                a1_1 = apply_act(a1_1, act_type, vslope, vzero);
                a2_1 = apply_act(a2_1, act_type, vslope, vzero);
                a3_1 = apply_act(a3_1, act_type, vslope, vzero);
                a0_2 = apply_act(a0_2, act_type, vslope, vzero);
                a1_2 = apply_act(a1_2, act_type, vslope, vzero);
                a2_2 = apply_act(a2_2, act_type, vslope, vzero);
                a3_2 = apply_act(a3_2, act_type, vslope, vzero);

                _mm256_storeu_ps(out_row0 + 0,  a0_0);
                _mm256_storeu_ps(out_row0 + 8,  a1_0);
                _mm256_storeu_ps(out_row0 + 16, a2_0);
                _mm256_storeu_ps(out_row0 + 24, a3_0);

                _mm256_storeu_ps(out_row1 + 0,  a0_1);
                _mm256_storeu_ps(out_row1 + 8,  a1_1);
                _mm256_storeu_ps(out_row1 + 16, a2_1);
                _mm256_storeu_ps(out_row1 + 24, a3_1);

                _mm256_storeu_ps(out_row2 + 0,  a0_2);
                _mm256_storeu_ps(out_row2 + 8,  a1_2);
                _mm256_storeu_ps(out_row2 + 16, a2_2);
                _mm256_storeu_ps(out_row2 + 24, a3_2);
            }
        }
        conv_s1_tail(L, pin, res, out, W, H, C, OC - OC % 3, OC, 0);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        s1_32_time += (t1.tv_sec - t0.tv_sec)*1000.0 + (t1.tv_nsec - t0.tv_nsec)*1e-6;
        return;
    }

    if (OW == 64) {
        #pragma omp parallel for schedule(static)
        for (int oc = 0; oc < OC - 3; oc += 4) {
            int oc0 = oc, oc1 = oc + 1, oc2 = oc + 2, oc3 = oc + 3;
            __m256 vbias0 = _mm256_set1_ps(B ? B[oc0] : 0.0f);
            __m256 vbias1 = _mm256_set1_ps(B ? B[oc1] : 0.0f);
            __m256 vbias2 = _mm256_set1_ps(B ? B[oc2] : 0.0f);
            __m256 vbias3 = _mm256_set1_ps(B ? B[oc3] : 0.0f);

            const float* wt0 = wt + (size_t)oc0 * C * 9;
            const float* wt1 = wt + (size_t)oc1 * C * 9;
            const float* wt2 = wt + (size_t)oc2 * C * 9;
            const float* wt3 = wt + (size_t)oc3 * C * 9;

            float* out0 = out->data + (size_t)oc0 * out_ch_sz;
            float* out1 = out->data + (size_t)oc1 * out_ch_sz;
            float* out2 = out->data + (size_t)oc2 * out_ch_sz;
            float* out3 = out->data + (size_t)oc3 * out_ch_sz;
            const float* res0 = res ? res->data + (size_t)oc0 * res_ch_sz : NULL;
            const float* res1 = res ? res->data + (size_t)oc1 * res_ch_sz : NULL;
            const float* res2 = res ? res->data + (size_t)oc2 * res_ch_sz : NULL;
            const float* res3 = res ? res->data + (size_t)oc3 * res_ch_sz : NULL;

            for (int oh = 0; oh < OH; oh++) {
                size_t out_row_off = (size_t)(out->is_padded ? (oh + 1) : oh) * out_pw + out_col_base;
                size_t res_row_off = (size_t)((res && res->is_padded) ? (oh + 1) : oh) * res_pw + res_col_base;
                float* out_row0 = out0 + out_row_off;
                float* out_row1 = out1 + out_row_off;
                float* out_row2 = out2 + out_row_off;
                float* out_row3 = out3 + out_row_off;

                for (int ow = 0; ow + 16 <= OW; ow += 16) {
                    __m256 a0_0 = vbias0, a1_0 = vbias0;
                    __m256 a0_1 = vbias1, a1_1 = vbias1;
                    __m256 a0_2 = vbias2, a1_2 = vbias2;
                    __m256 a0_3 = vbias3, a1_3 = vbias3;

                    for (int c = 0; c < C; c++) {
                        const float* pin_c = pin + (size_t)c * PH * PW;
                        const float* r0 = pin_c + (oh + 0) * PW + ow;
                        const float* r1 = pin_c + (oh + 1) * PW + ow;
                        const float* r2 = pin_c + (oh + 2) * PW + ow;
                        const float* w0 = wt0 + c * 9;
                        const float* w1 = wt1 + c * 9;
                        const float* w2 = wt2 + c * 9;
                        const float* w3 = wt3 + c * 9;

                        #define FMA_4OC_2OW(rk, off, k) do { \
                            __m256 kw0 = _mm256_set1_ps(w0[k]); \
                            __m256 kw1 = _mm256_set1_ps(w1[k]); \
                            __m256 kw2 = _mm256_set1_ps(w2[k]); \
                            __m256 kw3 = _mm256_set1_ps(w3[k]); \
                            __m256 in0 = _mm256_loadu_ps((rk) + (off) + 0); \
                            __m256 in1 = _mm256_loadu_ps((rk) + (off) + 8); \
                            a0_0 = _mm256_fmadd_ps(in0, kw0, a0_0); \
                            a1_0 = _mm256_fmadd_ps(in1, kw0, a1_0); \
                            a0_1 = _mm256_fmadd_ps(in0, kw1, a0_1); \
                            a1_1 = _mm256_fmadd_ps(in1, kw1, a1_1); \
                            a0_2 = _mm256_fmadd_ps(in0, kw2, a0_2); \
                            a1_2 = _mm256_fmadd_ps(in1, kw2, a1_2); \
                            a0_3 = _mm256_fmadd_ps(in0, kw3, a0_3); \
                            a1_3 = _mm256_fmadd_ps(in1, kw3, a1_3); \
                        } while(0)

                        FMA_4OC_2OW(r0, 0, 0); FMA_4OC_2OW(r0, 1, 1); FMA_4OC_2OW(r0, 2, 2);
                        FMA_4OC_2OW(r1, 0, 3); FMA_4OC_2OW(r1, 1, 4); FMA_4OC_2OW(r1, 2, 5);
                        FMA_4OC_2OW(r2, 0, 6); FMA_4OC_2OW(r2, 1, 7); FMA_4OC_2OW(r2, 2, 8);
                        #undef FMA_4OC_2OW
                    }
                    if (res) {
                        const float* r0 = res0 + res_row_off + ow;
                        const float* r1 = res1 + res_row_off + ow;
                        const float* r2 = res2 + res_row_off + ow;
                        const float* r3 = res3 + res_row_off + ow;
                        a0_0 = _mm256_add_ps(a0_0, _mm256_loadu_ps(r0 + 0));
                        a1_0 = _mm256_add_ps(a1_0, _mm256_loadu_ps(r0 + 8));
                        a0_1 = _mm256_add_ps(a0_1, _mm256_loadu_ps(r1 + 0));
                        a1_1 = _mm256_add_ps(a1_1, _mm256_loadu_ps(r1 + 8));
                        a0_2 = _mm256_add_ps(a0_2, _mm256_loadu_ps(r2 + 0));
                        a1_2 = _mm256_add_ps(a1_2, _mm256_loadu_ps(r2 + 8));
                        a0_3 = _mm256_add_ps(a0_3, _mm256_loadu_ps(r3 + 0));
                        a1_3 = _mm256_add_ps(a1_3, _mm256_loadu_ps(r3 + 8));
                    }

                    a0_0 = apply_act(a0_0, act_type, vslope, vzero);
                    a1_0 = apply_act(a1_0, act_type, vslope, vzero);
                    a0_1 = apply_act(a0_1, act_type, vslope, vzero);
                    a1_1 = apply_act(a1_1, act_type, vslope, vzero);
                    a0_2 = apply_act(a0_2, act_type, vslope, vzero);
                    a1_2 = apply_act(a1_2, act_type, vslope, vzero);
                    a0_3 = apply_act(a0_3, act_type, vslope, vzero);
                    a1_3 = apply_act(a1_3, act_type, vslope, vzero);

                    _mm256_storeu_ps(out_row0 + ow + 0, a0_0); _mm256_storeu_ps(out_row0 + ow + 8, a1_0);
                    _mm256_storeu_ps(out_row1 + ow + 0, a0_1); _mm256_storeu_ps(out_row1 + ow + 8, a1_1);
                    _mm256_storeu_ps(out_row2 + ow + 0, a0_2); _mm256_storeu_ps(out_row2 + ow + 8, a1_2);
                    _mm256_storeu_ps(out_row3 + ow + 0, a0_3); _mm256_storeu_ps(out_row3 + ow + 8, a1_3);
                }
            }
        }
        conv_s1_tail(L, pin, res, out, W, H, C, OC - OC % 4, OC, 0);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        s1_64_time += (t1.tv_sec - t0.tv_sec)*1000.0 + (t1.tv_nsec - t0.tv_nsec)*1e-6;
        return;
    }

    // Default fallback for any other OW
    #pragma omp parallel for schedule(static)
    for (int oc = 0; oc < OC - 1; oc += 2) {
        int oc0 = oc, oc1 = oc + 1;
        __m256 vbias0 = _mm256_set1_ps(B ? B[oc0] : 0.0f);
        __m256 vbias1 = _mm256_set1_ps(B ? B[oc1] : 0.0f);
        const float* wt_oc0 = wt + (size_t)oc0 * C * 9;
        const float* wt_oc1 = wt + (size_t)oc1 * C * 9;
        float* out_ch0 = out->data + (size_t)oc0 * out_ch_sz;
        float* out_ch1 = out->data + (size_t)oc1 * out_ch_sz;
        const float* res_ch0 = res ? res->data + (size_t)oc0 * res_ch_sz : NULL;
        const float* res_ch1 = res ? res->data + (size_t)oc1 * res_ch_sz : NULL;

        for (int oh = 0; oh < OH; oh++) {
            size_t out_row_off = (size_t)(out->is_padded ? (oh + 1) : oh) * out_pw + out_col_base;
            size_t res_row_off = (size_t)((res && res->is_padded) ? (oh + 1) : oh) * res_pw + res_col_base;
            float* out_row0 = out_ch0 + out_row_off;
            float* out_row1 = out_ch1 + out_row_off;
            int ow = 0;

            for (; ow + 32 <= OW; ow += 32) {
                __m256 a0_0 = vbias0, a1_0 = vbias0, a2_0 = vbias0, a3_0 = vbias0;
                __m256 a0_1 = vbias1, a1_1 = vbias1, a2_1 = vbias1, a3_1 = vbias1;

                for (int c = 0; c < C; c++) {
                    const float* pin_c = pin + (size_t)c * PH * PW;
                    const float* r0 = pin_c + (oh + 0) * PW + ow;
                    const float* r1 = pin_c + (oh + 1) * PW + ow;
                    const float* r2 = pin_c + (oh + 2) * PW + ow;
                    const float* w0 = wt_oc0 + c * 9;
                    const float* w1 = wt_oc1 + c * 9;

                    #define FMA_2OC_4OW(rk, off, kw_idx) do { \
                        __m256 kw0 = _mm256_set1_ps(w0[kw_idx]); \
                        __m256 kw1 = _mm256_set1_ps(w1[kw_idx]); \
                        __m256 in0 = _mm256_loadu_ps((rk) + (off) + 0); \
                        __m256 in1 = _mm256_loadu_ps((rk) + (off) + 8); \
                        __m256 in2 = _mm256_loadu_ps((rk) + (off) + 16); \
                        __m256 in3 = _mm256_loadu_ps((rk) + (off) + 24); \
                        a0_0 = _mm256_fmadd_ps(in0, kw0, a0_0); \
                        a0_1 = _mm256_fmadd_ps(in0, kw1, a0_1); \
                        a1_0 = _mm256_fmadd_ps(in1, kw0, a1_0); \
                        a1_1 = _mm256_fmadd_ps(in1, kw1, a1_1); \
                        a2_0 = _mm256_fmadd_ps(in2, kw0, a2_0); \
                        a2_1 = _mm256_fmadd_ps(in2, kw1, a2_1); \
                        a3_0 = _mm256_fmadd_ps(in3, kw0, a3_0); \
                        a3_1 = _mm256_fmadd_ps(in3, kw1, a3_1); \
                    } while(0)

                    FMA_2OC_4OW(r0, 0, 0); FMA_2OC_4OW(r0, 1, 1); FMA_2OC_4OW(r0, 2, 2);
                    FMA_2OC_4OW(r1, 0, 3); FMA_2OC_4OW(r1, 1, 4); FMA_2OC_4OW(r1, 2, 5);
                    FMA_2OC_4OW(r2, 0, 6); FMA_2OC_4OW(r2, 1, 7); FMA_2OC_4OW(r2, 2, 8);
                    #undef FMA_2OC_4OW
                }

                if (res) {
                    const float* r0 = res_ch0 + res_row_off + ow;
                    const float* r1 = res_ch1 + res_row_off + ow;
                    a0_0 = _mm256_add_ps(a0_0, _mm256_loadu_ps(r0 + 0));
                    a1_0 = _mm256_add_ps(a1_0, _mm256_loadu_ps(r0 + 8));
                    a2_0 = _mm256_add_ps(a2_0, _mm256_loadu_ps(r0 + 16));
                    a3_0 = _mm256_add_ps(a3_0, _mm256_loadu_ps(r0 + 24));
                    a0_1 = _mm256_add_ps(a0_1, _mm256_loadu_ps(r1 + 0));
                    a1_1 = _mm256_add_ps(a1_1, _mm256_loadu_ps(r1 + 8));
                    a2_1 = _mm256_add_ps(a2_1, _mm256_loadu_ps(r1 + 16));
                    a3_1 = _mm256_add_ps(a3_1, _mm256_loadu_ps(r1 + 24));
                }

                a0_0 = apply_act(a0_0, act_type, vslope, vzero);
                a1_0 = apply_act(a1_0, act_type, vslope, vzero);
                a2_0 = apply_act(a2_0, act_type, vslope, vzero);
                a3_0 = apply_act(a3_0, act_type, vslope, vzero);

                a0_1 = apply_act(a0_1, act_type, vslope, vzero);
                a1_1 = apply_act(a1_1, act_type, vslope, vzero);
                a2_1 = apply_act(a2_1, act_type, vslope, vzero);
                a3_1 = apply_act(a3_1, act_type, vslope, vzero);

                _mm256_storeu_ps(out_row0 + ow + 0,  a0_0);
                _mm256_storeu_ps(out_row0 + ow + 8,  a1_0);
                _mm256_storeu_ps(out_row0 + ow + 16, a2_0);
                _mm256_storeu_ps(out_row0 + ow + 24, a3_0);

                _mm256_storeu_ps(out_row1 + ow + 0,  a0_1);
                _mm256_storeu_ps(out_row1 + ow + 8,  a1_1);
                _mm256_storeu_ps(out_row1 + ow + 16, a2_1);
                _mm256_storeu_ps(out_row1 + ow + 24, a3_1);
            }
        }
    }
    /* Columns past the last full 32-wide tile, for the paired channels */
    conv_s1_tail(L, pin, res, out, W, H, C, 0, OC - OC % 2, OW - OW % 32);
    /* Odd last channel, all columns */
    conv_s1_tail(L, pin, res, out, W, H, C, OC - OC % 2, OC, 0);
}

static void conv_s2_forward(const Layer* L, const float* pin, Tensor* out, int in_w, int in_h, int in_c) {
    int W = in_w, H = in_h, C = in_c;
    int OW = out->w, OH = out->h, OC = out->c;
    int PW = W + 2, PH = H + 2;
    const float* wt_reord = L->weights;
    const float* B = L->bias;
    int act_type = L->act_type;
    float act_param = L->act_params[0];
    __m256 vslope = _mm256_set1_ps(act_param);
    __m256 vzero = _mm256_setzero_ps();

    int out_pw = out->is_padded ? (OW + 2) : OW;
    size_t out_ch_sz = (size_t)out_pw * (out->is_padded ? (OH + 2) : OH);
    int out_col_base = out->is_padded ? 1 : 0;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int oc_b = 0; oc_b < OC / 8; oc_b++) {
        for (int oh = 0; oh < OH; oh++) {
            __m256 vbias = B ? _mm256_loadu_ps(B + oc_b * 8) : _mm256_setzero_ps();
            int sy = oh * 2;
            const float* wt_block = wt_reord + (size_t)oc_b * C * 9 * 8;
            size_t out_row_off = (size_t)(out->is_padded ? (oh + 1) : oh) * out_pw + out_col_base;
            int ow = 0;

            // 4-way unroll across OW: ow0, ow1, ow2, ow3
            for (; ow + 4 <= OW; ow += 4) {
                int sx0 = (ow + 0) * 2;
                int sx1 = (ow + 1) * 2;
                int sx2 = (ow + 2) * 2;
                int sx3 = (ow + 3) * 2;

                __m256 acc0 = vbias;
                __m256 acc1 = vbias;
                __m256 acc2 = vbias;
                __m256 acc3 = vbias;

                const float* w_ptr = wt_block;
                for (int c = 0; c < C; c++) {
                    const float* pin_c = pin + (size_t)c * PH * PW;
                    const float* r0 = pin_c + (sy + 0) * PW;
                    const float* r1 = pin_c + (sy + 1) * PW;
                    const float* r2 = pin_c + (sy + 2) * PW;

                    #define FMA_S2_4OW(r, k_off, w_off) do { \
                        __m256 w_vec = _mm256_load_ps(w_ptr + (w_off)); \
                        acc0 = _mm256_fmadd_ps(_mm256_set1_ps((r)[sx0 + (k_off)]), w_vec, acc0); \
                        acc1 = _mm256_fmadd_ps(_mm256_set1_ps((r)[sx1 + (k_off)]), w_vec, acc1); \
                        acc2 = _mm256_fmadd_ps(_mm256_set1_ps((r)[sx2 + (k_off)]), w_vec, acc2); \
                        acc3 = _mm256_fmadd_ps(_mm256_set1_ps((r)[sx3 + (k_off)]), w_vec, acc3); \
                    } while(0)

                    FMA_S2_4OW(r0, 0, 0);
                    FMA_S2_4OW(r0, 1, 8);
                    FMA_S2_4OW(r0, 2, 16);
                    FMA_S2_4OW(r1, 0, 24);
                    FMA_S2_4OW(r1, 1, 32);
                    FMA_S2_4OW(r1, 2, 40);
                    FMA_S2_4OW(r2, 0, 48);
                    FMA_S2_4OW(r2, 1, 56);
                    FMA_S2_4OW(r2, 2, 64);
                    #undef FMA_S2_4OW

                    w_ptr += 72;
                }

                acc0 = apply_act(acc0, act_type, vslope, vzero);
                acc1 = apply_act(acc1, act_type, vslope, vzero);
                acc2 = apply_act(acc2, act_type, vslope, vzero);
                acc3 = apply_act(acc3, act_type, vslope, vzero);

                float tmp0[8], tmp1[8], tmp2[8], tmp3[8];
                _mm256_storeu_ps(tmp0, acc0);
                _mm256_storeu_ps(tmp1, acc1);
                _mm256_storeu_ps(tmp2, acc2);
                _mm256_storeu_ps(tmp3, acc3);

                for (int s = 0; s < 8; s++) {
                    size_t ch_off = (size_t)(oc_b * 8 + s) * out_ch_sz + out_row_off;
                    out->data[ch_off + ow + 0] = tmp0[s];
                    out->data[ch_off + ow + 1] = tmp1[s];
                    out->data[ch_off + ow + 2] = tmp2[s];
                    out->data[ch_off + ow + 3] = tmp3[s];
                }
            }

            // Remainder for OW (if any)
            for (; ow < OW; ow++) {
                int sx = ow * 2;
                __m256 acc = vbias;
                const float* w_ptr = wt_block;
                for (int c = 0; c < C; c++) {
                    const float* pin_c = pin + (size_t)c * PH * PW;
                    const float* r0 = pin_c + (sy + 0) * PW + sx;
                    const float* r1 = pin_c + (sy + 1) * PW + sx;
                    const float* r2 = pin_c + (sy + 2) * PW + sx;

                    acc = _mm256_fmadd_ps(_mm256_set1_ps(r0[0]), _mm256_load_ps(w_ptr + 0),  acc);
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(r0[1]), _mm256_load_ps(w_ptr + 8),  acc);
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(r0[2]), _mm256_load_ps(w_ptr + 16), acc);
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(r1[0]), _mm256_load_ps(w_ptr + 24), acc);
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(r1[1]), _mm256_load_ps(w_ptr + 32), acc);
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(r1[2]), _mm256_load_ps(w_ptr + 40), acc);
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(r2[0]), _mm256_load_ps(w_ptr + 48), acc);
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(r2[1]), _mm256_load_ps(w_ptr + 56), acc);
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(r2[2]), _mm256_load_ps(w_ptr + 64), acc);
                    w_ptr += 72;
                }
                acc = apply_act(acc, act_type, vslope, vzero);
                float tmp[8];
                _mm256_storeu_ps(tmp, acc);
                for (int s = 0; s < 8; s++) {
                    size_t ch_off = (size_t)(oc_b * 8 + s) * out_ch_sz + out_row_off;
                    out->data[ch_off + ow] = tmp[s];
                }
            }
        }
    }
}

static void conv_forward(const Layer* L, const Tensor* in, const Tensor* res, Tensor* out) {
    int W = in->w, H = in->h, C = in->c;
    int PW = W + 2, PH = H + 2;
    const float* pin = NULL;

    struct timespec tp0, tp1;
    clock_gettime(CLOCK_MONOTONIC, &tp0);

    if (in->is_padded) {
        pin = in->data;
    } else {
        size_t total_pin = (size_t)C * PH * PW;
        scratch_check(total_pin, L->name);
        pin = g_pin_scratch;

        #pragma omp parallel for collapse(2) schedule(static)
        for (int c = 0; c < C; c++) {
            for (int y = 0; y < H; y++) {
                float* pin_c = g_pin_scratch + (size_t)c * PH * PW;
                const float* src_c = in->data + (size_t)c * H * W;
                if (y == 0) memset(pin_c, 0, PW * sizeof(float));
                float* dst_row = pin_c + (y + 1) * PW;
                const float* src_row = src_c + y * W;
                dst_row[0] = 0.0f;
                int x = 0;
                for (; x <= W - 16; x += 16) {
                    __m256 v0 = _mm256_loadu_ps(src_row + x + 0);
                    __m256 v1 = _mm256_loadu_ps(src_row + x + 8);
                    _mm256_storeu_ps(dst_row + 1 + x + 0, v0);
                    _mm256_storeu_ps(dst_row + 1 + x + 8, v1);
                }
                for (; x <= W - 8; x += 8) {
                    _mm256_storeu_ps(dst_row + 1 + x, _mm256_loadu_ps(src_row + x));
                }
                for (; x < W; x++) dst_row[1 + x] = src_row[x];
                dst_row[W + 1] = 0.0f;
                if (y == H - 1) memset(pin_c + (size_t)(PH - 1) * PW, 0, PW * sizeof(float));
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &tp1);
    pad_time += (tp1.tv_sec - tp0.tv_sec)*1000.0 + (tp1.tv_nsec - tp0.tv_nsec)*1e-6;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (L->stride == 1) {
        conv_s1_forward(L, pin, res, out, W, H, C);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        s1_time += (t1.tv_sec - t0.tv_sec)*1000.0 + (t1.tv_nsec - t0.tv_nsec)*1e-6;
    } else if (L->stride == 2) {
        conv_s2_forward(L, pin, out, W, H, C);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        s2_time += (t1.tv_sec - t0.tv_sec)*1000.0 + (t1.tv_nsec - t0.tv_nsec)*1e-6;
    }
}

static void deconv_forward(const Layer* L, const Tensor* in, Tensor* out) {
    int W = in->w, H = in->h, C = in->c;
    int OW = out->w, OH = out->h;
    if (L->upscale == 2) {
        OW /= 2;
        OH /= 2;
    }
    const float* wt_reord = L->weights;
    const float* B = L->bias;
    scratch_check((size_t)H * W * C, L->name);
    float* in_hwc = g_pin_scratch;

    if (in->is_padded) {
        int PW = W + 2;
        size_t in_ch_sz = (size_t)(H + 2) * PW;
        #pragma omp parallel for collapse(2) schedule(static)
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                float* dst = in_hwc + ((size_t)h * W + w) * C;
                size_t src_offset = (size_t)(h + 1) * PW + (w + 1);
                for (int c = 0; c < C; c += 8) {
                    dst[c + 0] = in->data[(size_t)(c + 0) * in_ch_sz + src_offset];
                    dst[c + 1] = in->data[(size_t)(c + 1) * in_ch_sz + src_offset];
                    dst[c + 2] = in->data[(size_t)(c + 2) * in_ch_sz + src_offset];
                    dst[c + 3] = in->data[(size_t)(c + 3) * in_ch_sz + src_offset];
                    dst[c + 4] = in->data[(size_t)(c + 4) * in_ch_sz + src_offset];
                    dst[c + 5] = in->data[(size_t)(c + 5) * in_ch_sz + src_offset];
                    dst[c + 6] = in->data[(size_t)(c + 6) * in_ch_sz + src_offset];
                    dst[c + 7] = in->data[(size_t)(c + 7) * in_ch_sz + src_offset];
                }
            }
        }
    } else {
        int HW = H * W;
        #pragma omp parallel for schedule(static)
        for (int hw = 0; hw < HW; hw++) {
            float* dst = in_hwc + (size_t)hw * C;
            for (int c = 0; c < C; c += 8) {
                dst[c + 0] = in->data[(size_t)(c + 0) * HW + hw];
                dst[c + 1] = in->data[(size_t)(c + 1) * HW + hw];
                dst[c + 2] = in->data[(size_t)(c + 2) * HW + hw];
                dst[c + 3] = in->data[(size_t)(c + 3) * HW + hw];
                dst[c + 4] = in->data[(size_t)(c + 4) * HW + hw];
                dst[c + 5] = in->data[(size_t)(c + 5) * HW + hw];
                dst[c + 6] = in->data[(size_t)(c + 6) * HW + hw];
                dst[c + 7] = in->data[(size_t)(c + 7) * HW + hw];
            }
        }
    }

    __m256 b0 = B ? _mm256_loadu_ps(B + 0) : _mm256_setzero_ps();
    __m256 b1 = B ? _mm256_loadu_ps(B + 8) : _mm256_setzero_ps();
    __m256 b2 = B ? _mm256_loadu_ps(B + 16) : _mm256_setzero_ps();

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

        if (L->upscale == 2) {
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
        } else {
            for (int oc = 0; oc < 24; oc++) {
                float* dst = out->data + (size_t)oc * OH * OW + oh * OW;
                for (int ow = 0; ow < OW; ow++) {
                    dst[ow] = row_buf[ow * 24 + oc];
                }
            }
        }
    }
}

static void relu_forward(const Layer* L, const Tensor* in, Tensor* out) {
    float slope = L->relu_slope;
    int N = in->w * in->h * in->c;
    __m256 vslope = _mm256_set1_ps(slope);
    __m256 vzero = _mm256_setzero_ps();

    #pragma omp parallel for if(N >= 65536) schedule(static)
    for (int i = 0; i <= N - 32; i += 32) {
        __m256 v0 = _mm256_loadu_ps(in->data + i + 0);
        __m256 v1 = _mm256_loadu_ps(in->data + i + 8);
        __m256 v2 = _mm256_loadu_ps(in->data + i + 16);
        __m256 v3 = _mm256_loadu_ps(in->data + i + 24);

        __m256 m0 = _mm256_cmp_ps(v0, vzero, _CMP_GT_OQ);
        __m256 m1 = _mm256_cmp_ps(v1, vzero, _CMP_GT_OQ);
        __m256 m2 = _mm256_cmp_ps(v2, vzero, _CMP_GT_OQ);
        __m256 m3 = _mm256_cmp_ps(v3, vzero, _CMP_GT_OQ);

        _mm256_storeu_ps(out->data + i + 0, _mm256_blendv_ps(_mm256_mul_ps(v0, vslope), v0, m0));
        _mm256_storeu_ps(out->data + i + 8, _mm256_blendv_ps(_mm256_mul_ps(v1, vslope), v1, m1));
        _mm256_storeu_ps(out->data + i + 16, _mm256_blendv_ps(_mm256_mul_ps(v2, vslope), v2, m2));
        _mm256_storeu_ps(out->data + i + 24, _mm256_blendv_ps(_mm256_mul_ps(v3, vslope), v3, m3));
    }
    int start = (N / 32) * 32;
    for (int i = start; i < N; i++) {
        float v = in->data[i];
        out->data[i] = v > 0.0f ? v : v * slope;
    }
}

static void sigmoid_forward(const Tensor* in, Tensor* out) {
    int N = in->w * in->h * in->c;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        float v = in->data[i];
        v = fmaxf(-88.0f, fminf(v, 88.0f));
        out->data[i] = 1.0f / (1.0f + expf(-v));
    }
}

static void interp_forward(const Layer* L, const Tensor* in, Tensor* out) {
    int W = in->w, H = in->h, C = in->c;
    int OW = out->w, OH = out->h;

    double ws = 1.0 / (double)L->scale_w;
    double hs = 1.0 / (double)L->scale_h;

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

static void pixelshuffle_forward(const Layer* L, const Tensor* in, Tensor* out) {
    int r = L->upscale;
    int H = in->h, W = in->w;
    int OH = out->h, OW = out->w, OC = out->c;

    if (r == 2) {
        #pragma omp parallel for schedule(static)
        for (int oc = 0; oc < OC; oc++) {
            for (int oh = 0; oh < OH; oh++) {
                int sh = oh & 1;
                int ih = oh >> 1;
                int ic0 = oc * 4 + sh * 2 + 0;
                int ic1 = oc * 4 + sh * 2 + 1;
                const float* r0 = in->data + (size_t)(ic0 * H + ih) * W;
                const float* r1 = in->data + (size_t)(ic1 * H + ih) * W;
                float* out_row = out->data + (size_t)(oc * OH + oh) * OW;
                for (int iw = 0; iw < W; iw++) {
                    out_row[iw * 2 + 0] = r0[iw];
                    out_row[iw * 2 + 1] = r1[iw];
                }
            }
        }
    } else {
        int r2 = r * r;
        #pragma omp parallel for schedule(static)
        for (int oc = 0; oc < OC; oc++) {
            for (int oh = 0; oh < OH; oh++) {
                int sh = oh % r;
                int ih = oh / r;
                float* out_row = out->data + (size_t)(oc * OH + oh) * OW;
                for (int ow = 0; ow < OW; ow++) {
                    int sw = ow % r;
                    int iw = ow / r;
                    int ic = oc * r2 + sh * r + sw;
                    out_row[ow] = in->data[(size_t)(ic * H + ih) * W + iw];
                }
            }
        }
    }
}

static void binaryop_forward(const Layer* L, const Tensor* a, const Tensor* b, Tensor* out) {
    if (L->with_scalar) {
        float s = L->b_scalar;
        int N = a->w * a->h * a->c;
        if (L->op_type == 7) {
            __m256 vs = _mm256_set1_ps(s);
            #pragma omp parallel for if(N >= 65536) schedule(static)
            for (int i = 0; i <= N - 32; i += 32) {
                _mm256_storeu_ps(out->data + i +  0, _mm256_sub_ps(vs, _mm256_loadu_ps(a->data + i +  0)));
                _mm256_storeu_ps(out->data + i +  8, _mm256_sub_ps(vs, _mm256_loadu_ps(a->data + i +  8)));
                _mm256_storeu_ps(out->data + i + 16, _mm256_sub_ps(vs, _mm256_loadu_ps(a->data + i + 16)));
                _mm256_storeu_ps(out->data + i + 24, _mm256_sub_ps(vs, _mm256_loadu_ps(a->data + i + 24)));
            }
            int start = (N / 32) * 32;
            for (int i = start; i < N; i++) out->data[i] = s - a->data[i];
        } else if (L->op_type == 2) {
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
        } else {
            #pragma omp parallel for if(N >= 65536) schedule(static)
            for (int i = 0; i < N; i++) {
                float av = a->data[i];
                switch (L->op_type) {
                    case 0: out->data[i] = av + s; break;
                    case 1: out->data[i] = av - s; break;
                    case 3: out->data[i] = av / s; break;
                    default: out->data[i] = av; break;
                }
            }
        }
    } else {
        int W = out->w, H = out->h, C = out->c;
        int HW = H * W;
        if (a->c == b->c) {
            int N = C * HW;
            if (L->op_type == 0) {
                #pragma omp parallel for if(N >= 65536) schedule(static)
                for (int i = 0; i <= N - 32; i += 32) {
                    _mm256_storeu_ps(out->data + i +  0, _mm256_add_ps(_mm256_loadu_ps(a->data + i +  0), _mm256_loadu_ps(b->data + i +  0)));
                    _mm256_storeu_ps(out->data + i +  8, _mm256_add_ps(_mm256_loadu_ps(a->data + i +  8), _mm256_loadu_ps(b->data + i +  8)));
                    _mm256_storeu_ps(out->data + i + 16, _mm256_add_ps(_mm256_loadu_ps(a->data + i + 16), _mm256_loadu_ps(b->data + i + 16)));
                    _mm256_storeu_ps(out->data + i + 24, _mm256_add_ps(_mm256_loadu_ps(a->data + i + 24), _mm256_loadu_ps(b->data + i + 24)));
                }
                int start = (N / 32) * 32;
                for (int i = start; i < N; i++) out->data[i] = a->data[i] + b->data[i];
            } else if (L->op_type == 1) {
                #pragma omp parallel for if(N >= 65536) schedule(static)
                for (int i = 0; i <= N - 32; i += 32) {
                    _mm256_storeu_ps(out->data + i +  0, _mm256_sub_ps(_mm256_loadu_ps(a->data + i +  0), _mm256_loadu_ps(b->data + i +  0)));
                    _mm256_storeu_ps(out->data + i +  8, _mm256_sub_ps(_mm256_loadu_ps(a->data + i +  8), _mm256_loadu_ps(b->data + i +  8)));
                    _mm256_storeu_ps(out->data + i + 16, _mm256_sub_ps(_mm256_loadu_ps(a->data + i + 16), _mm256_loadu_ps(b->data + i + 16)));
                    _mm256_storeu_ps(out->data + i + 24, _mm256_sub_ps(_mm256_loadu_ps(a->data + i + 24), _mm256_loadu_ps(b->data + i + 24)));
                }
                int start = (N / 32) * 32;
                for (int i = start; i < N; i++) out->data[i] = a->data[i] - b->data[i];
            } else if (L->op_type == 2) {
                #pragma omp parallel for if(N >= 65536) schedule(static)
                for (int i = 0; i <= N - 32; i += 32) {
                    _mm256_storeu_ps(out->data + i +  0, _mm256_mul_ps(_mm256_loadu_ps(a->data + i +  0), _mm256_loadu_ps(b->data + i +  0)));
                    _mm256_storeu_ps(out->data + i +  8, _mm256_mul_ps(_mm256_loadu_ps(a->data + i +  8), _mm256_loadu_ps(b->data + i +  8)));
                    _mm256_storeu_ps(out->data + i + 16, _mm256_mul_ps(_mm256_loadu_ps(a->data + i + 16), _mm256_loadu_ps(b->data + i + 16)));
                    _mm256_storeu_ps(out->data + i + 24, _mm256_mul_ps(_mm256_loadu_ps(a->data + i + 24), _mm256_loadu_ps(b->data + i + 24)));
                }
                int start = (N / 32) * 32;
                for (int i = start; i < N; i++) out->data[i] = a->data[i] * b->data[i];
            }
        } else {
            /* Channel broadcasting: single outer parallel region over c */
            #pragma omp parallel for schedule(static)
            for (int c = 0; c < C; c++) {
                int ac = (c < a->c) ? c : 0;
                int bc = (c < b->c) ? c : 0;
                const float* a_ch = a->data + (size_t)ac * HW;
                const float* b_ch = b->data + (size_t)bc * HW;
                float* out_ch = out->data + (size_t)c * HW;
                if (L->op_type == 2) {
                    for (int i = 0; i <= HW - 32; i += 32) {
                        _mm256_storeu_ps(out_ch + i +  0, _mm256_mul_ps(_mm256_loadu_ps(a_ch + i +  0), _mm256_loadu_ps(b_ch + i +  0)));
                        _mm256_storeu_ps(out_ch + i +  8, _mm256_mul_ps(_mm256_loadu_ps(a_ch + i +  8), _mm256_loadu_ps(b_ch + i +  8)));
                        _mm256_storeu_ps(out_ch + i + 16, _mm256_mul_ps(_mm256_loadu_ps(a_ch + i + 16), _mm256_loadu_ps(b_ch + i + 16)));
                        _mm256_storeu_ps(out_ch + i + 24, _mm256_mul_ps(_mm256_loadu_ps(a_ch + i + 24), _mm256_loadu_ps(b_ch + i + 24)));
                    }
                    int start = (HW / 32) * 32;
                    for (int i = start; i < HW; i++) out_ch[i] = a_ch[i] * b_ch[i];
                } else if (L->op_type == 0) {
                    for (int i = 0; i <= HW - 32; i += 32) {
                        _mm256_storeu_ps(out_ch + i +  0, _mm256_add_ps(_mm256_loadu_ps(a_ch + i +  0), _mm256_loadu_ps(b_ch + i +  0)));
                        _mm256_storeu_ps(out_ch + i +  8, _mm256_add_ps(_mm256_loadu_ps(a_ch + i +  8), _mm256_loadu_ps(b_ch + i +  8)));
                        _mm256_storeu_ps(out_ch + i + 16, _mm256_add_ps(_mm256_loadu_ps(a_ch + i + 16), _mm256_loadu_ps(b_ch + i + 16)));
                        _mm256_storeu_ps(out_ch + i + 24, _mm256_add_ps(_mm256_loadu_ps(a_ch + i + 24), _mm256_loadu_ps(b_ch + i + 24)));
                    }
                    int start = (HW / 32) * 32;
                    for (int i = start; i < HW; i++) out_ch[i] = a_ch[i] + b_ch[i];
                } else {
                    for (int i = 0; i < HW; i++) {
                        float av = a_ch[i], bv = b_ch[i];
                        switch (L->op_type) {
                            case 1: out_ch[i] = av - bv; break;
                            case 3: out_ch[i] = av / bv; break;
                            case 7: out_ch[i] = bv - av; break;
                            default: out_ch[i] = av + bv; break;
                        }
                    }
                }
            }
        }
    }
}

static void concat_forward(const Layer* L, const Tensor* blobs, Tensor* out) {
    int H = out->h, W = out->w;
    if (out->is_padded) {
        int PW = W + 2;
        size_t out_ch_sz = (size_t)(H + 2) * PW;
        size_t out_ch = 0;
        for (int i = 0; i < L->in_count; i++) {
            const Tensor* in = &blobs[L->in_idx[i]];
            int in_c = in->c;
            int in_pw = in->is_padded ? (in->w + 2) : in->w;
            size_t in_ch_sz = in->is_padded ? (size_t)(in->h + 2) * in_pw : (size_t)in->h * in->w;
            int in_col_base = in->is_padded ? 1 : 0;
            #pragma omp parallel for collapse(2) schedule(static)
            for (int c = 0; c < in_c; c++) {
                for (int y = 0; y < H; y++) {
                    const float* src_row = in->data + (size_t)c * in_ch_sz + (size_t)(in->is_padded ? (y + 1) : y) * in_pw + in_col_base;
                    float* dst_row = out->data + (size_t)(out_ch + c) * out_ch_sz + (size_t)(y + 1) * PW + 1;
                    memcpy(dst_row, src_row, W * sizeof(float));
                }
            }
            out_ch += in_c;
        }
    } else {
        int HW = H * W;
        size_t out_ch = 0;
        for (int i = 0; i < L->in_count; i++) {
            const Tensor* in = &blobs[L->in_idx[i]];
            int in_c = in->c;
            const float* src = in->data;
            float* dst = out->data + out_ch * HW;
            size_t N = (size_t)in_c * HW;
            memcpy(dst, src, N * sizeof(float));
            out_ch += in_c;
        }
    }
}

static void eltwise_forward(const Layer* L, const Tensor* a, const Tensor* b, Tensor* out) {
    float c0 = L->num_coeffs > 0 ? L->eltwise_coeffs[0] : 1.0f;
    float c1 = L->num_coeffs > 1 ? L->eltwise_coeffs[1] : 1.0f;
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
    float* out0 = out->data;
    float* out1 = out->data + HW;
    float* out2 = out->data + HW * 2;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; y++) {
        const float* fx_row = fxptr + y * W;
        const float* fy_row = fyptr + y * W;
        size_t row_dst = (size_t)y * W;

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
                    out->data[(size_t)c * HW + dst_idx] = v4 * inv_beta + v5 * beta;
                }
            }
        }
    }
}

/* =====================  GRAPH OPTIMIZATION PASSES  ===================== */

static void fuse_resblocks(Graph* g) {
    int fused_count = 0;
    for (int i = 0; i < g->n_layers - 2; i++) {
        Layer* L_conv = &g->layers[i];
        if (L_conv->type != LT_CONV || L_conv->stride != 1 || L_conv->kernel != 3 || L_conv->act_type != 0 || L_conv->out_count != 1) {
            continue;
        }
        int conv_out_blob = L_conv->out_idx[0];

        Layer* L_add = &g->layers[i + 1];
        if (L_add->type != LT_BINARYOP || L_add->op_type != 0 || L_add->with_scalar || L_add->in_count != 2 || L_add->out_count != 1) {
            continue;
        }
        if (L_add->in_idx[0] != conv_out_blob && L_add->in_idx[1] != conv_out_blob) {
            continue;
        }
        int res_blob = (L_add->in_idx[0] == conv_out_blob) ? L_add->in_idx[1] : L_add->in_idx[0];
        int add_out_blob = L_add->out_idx[0];

        Layer* L_relu = &g->layers[i + 2];
        if (L_relu->type != LT_RELU || L_relu->in_count != 1 || L_relu->out_count != 1 || L_relu->in_idx[0] != add_out_blob) {
            continue;
        }
        int relu_out_blob = L_relu->out_idx[0];

        // Fuse ResBlock: Conv computes (conv(x) + res) followed by activation
        L_conv->res_blob_idx = res_blob;
        L_conv->act_type = (L_relu->relu_slope > 0.0f) ? 2 : 1;
        L_conv->act_params[0] = L_relu->relu_slope;
        L_conv->out_idx[0] = relu_out_blob;

        L_add->type = LT_NOP;
        L_relu->type = LT_NOP;

        fused_count++;
        i += 2; // Skip past add and relu
    }
    printf("Fused %d ResBlocks (conv + res_add + leakyrelu)\n", fused_count);
}

static void fuse_deconv_pixelshuffle(Graph* g) {
    int fused_count = 0;
    for (int i = 0; i < g->n_layers - 1; i++) {
        Layer* L_deconv = &g->layers[i];
        if (L_deconv->type != LT_DECONV) continue;

        Layer* L_ps = &g->layers[i + 1];
        if (L_ps->type != LT_PIXELSHUFFLE || L_ps->upscale != 2) continue;
        if (L_ps->in_idx[0] != L_deconv->out_idx[0]) continue;

        // Fuse! Deconv writes directly to pixelshuffle output tensor
        L_deconv->out_idx[0] = L_ps->out_idx[0];
        L_deconv->upscale = 2;
        L_ps->type = LT_NOP;

        fused_count++;
    }
    printf("Fused %d Deconv+PixelShuffle blocks\n", fused_count);
}

/* =====================  GRAPH EXECUTION  ===================== */

static void graph_execute(const Graph* g, Tensor* blobs) {
    s1_time = 0;
    s2_time = 0;
    pad_time = 0;
    s1_8_time = 0; s1_16_time = 0; s1_32_time = 0; s1_64_time = 0;
    double type_times[16] = {0};
    for (int i = 0; i < g->n_layers; i++) {
        const Layer* L = &g->layers[i];
        if (L->type == LT_NOP) continue;
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        switch (L->type) {
            case LT_INPUT:
            case LT_NOP:
                break;
            case LT_SPLIT:
                /* Zero-copy: output tensors already alias input data pointer */
                break;
            case LT_CONCAT:
                concat_forward(L, blobs, &blobs[L->out_idx[0]]);
                break;
            case LT_INTERP:
                interp_forward(L, &blobs[L->in_idx[0]], &blobs[L->out_idx[0]]);
                break;
            case LT_CONV: {
                const Tensor* res = (L->res_blob_idx >= 0) ? &blobs[L->res_blob_idx] : NULL;
                conv_forward(L, &blobs[L->in_idx[0]], res, &blobs[L->out_idx[0]]);
                break;
            }
            case LT_DECONV:
                deconv_forward(L, &blobs[L->in_idx[0]], &blobs[L->out_idx[0]]);
                break;
            case LT_RELU:
                relu_forward(L, &blobs[L->in_idx[0]], &blobs[L->out_idx[0]]);
                break;
            case LT_BINARYOP:
                binaryop_forward(L, &blobs[L->in_idx[0]],
                                 L->with_scalar ? NULL : &blobs[L->in_idx[1]],
                                 &blobs[L->out_idx[0]]);
                break;
            case LT_PIXELSHUFFLE:
                pixelshuffle_forward(L, &blobs[L->in_idx[0]], &blobs[L->out_idx[0]]);
                break;
            case LT_CROP:
                /* Zero-copy channel slice */
                break;
            case LT_WARP:
                warp_forward(&blobs[L->in_idx[0]], &blobs[L->in_idx[1]], &blobs[L->out_idx[0]]);
                break;
            case LT_ELTWISE:
                eltwise_forward(L, &blobs[L->in_idx[0]], &blobs[L->in_idx[1]], &blobs[L->out_idx[0]]);
                break;
            case LT_SIGMOID:
                sigmoid_forward(&blobs[L->in_idx[0]], &blobs[L->out_idx[0]]);
                break;
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        type_times[L->type] += (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) * 1e-6;
    }
    const char* type_names[] = {
        "Input", "Split", "Concat", "Interp", "Conv", "Deconv", "ReLU",
        "BinaryOp", "PixelShuffle", "Crop", "Warp", "Eltwise", "Sigmoid"
    };
    for (int t = 0; t < 13; t++) {
        if (type_times[t] > 0.01) {
            printf("  %-14s: %6.2f ms\n", type_names[t], type_times[t]);
        }
    }
    printf("    -> Conv S=1: %6.2f ms, Conv S=2: %6.2f ms, Conv Pad: %6.2f ms\n",
           s1_time, s2_time, pad_time);
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
    if (argc < 6) {
        fprintf(stderr, "usage: %s model.bin frame0.png frame1.png timestep out.png [iterations]\n", argv[0]);
        return 1;
    }

    const char* bin_path = argv[1];
    const char* img0_path = argv[2];
    const char* img1_path = argv[3];
    float timestep = (float)atof(argv[4]);
    const char* out_path = argv[5];

    char param_path[1024];
    strncpy(param_path, bin_path, sizeof(param_path) - 1);
    param_path[sizeof(param_path) - 1] = '\0';
    char* dot = strrchr(param_path, '.');
    if (dot) strcpy(dot, ".param");
    else strncat(param_path, ".param", sizeof(param_path) - strlen(param_path) - 1);

    Graph g;
    memset(&g, 0, sizeof(Graph));
    if (graph_load_param(&g, param_path) < 0) {
        fprintf(stderr, "Failed to load param: %s\n", param_path);
        return 1;
    }
    if (graph_load_bin(&g, bin_path) < 0) {
        fprintf(stderr, "Failed to load bin: %s\n", bin_path);
        return 1;
    }

    fuse_resblocks(&g);
    fuse_deconv_pixelshuffle(&g);

    int w, h;
    Tensor img0 = load_image_rgb(img0_path, &w, &h);
    Tensor img1 = load_image_rgb(img1_path, &w, &h);

    int wp = (w + 31) / 32 * 32;
    int hp = (h + 31) / 32 * 32;

    // Shape inference
    int shapes[MAX_BLOBS][3];
    memset(shapes, 0, sizeof(shapes));

    int in0_idx = blob_idx(&g, "in0");
    int in1_idx = blob_idx(&g, "in1");
    int in2_idx = blob_idx(&g, "in2");

    shapes[in0_idx][0] = wp; shapes[in0_idx][1] = hp; shapes[in0_idx][2] = 3;
    shapes[in1_idx][0] = wp; shapes[in1_idx][1] = hp; shapes[in1_idx][2] = 3;
    shapes[in2_idx][0] = wp; shapes[in2_idx][1] = hp; shapes[in2_idx][2] = 1;

    // Alias tracking: alias_of[blob_id] = root_blob_id (-1 if owns memory)
    int alias_of[MAX_BLOBS];
    int alias_offset[MAX_BLOBS];
    for (int i = 0; i < MAX_BLOBS; i++) {
        alias_of[i] = -1;
        alias_offset[i] = 0;
    }

    for (int i = 0; i < g.n_layers; i++) {
        Layer* L = &g.layers[i];
        if (L->type == LT_INPUT || L->type == LT_NOP) continue;

        int iw = shapes[L->in_idx[0]][0];
        int ih = shapes[L->in_idx[0]][1];
        int ic = shapes[L->in_idx[0]][2];

        int ow = iw, oh = ih, oc = ic;

        switch (L->type) {
            case LT_SPLIT:
                for (int o = 0; o < L->out_count; o++) {
                    int idx = L->out_idx[o];
                    shapes[idx][0] = iw; shapes[idx][1] = ih; shapes[idx][2] = ic;
                    alias_of[idx] = L->in_idx[0];
                    alias_offset[idx] = 0;
                }
                continue;
            case LT_CONCAT:
                oc = 0;
                for (int j = 0; j < L->in_count; j++) {
                    oc += shapes[L->in_idx[j]][2];
                }
                ow = iw; oh = ih;
                break;
            case LT_INTERP:
                ow = L->out_w ? L->out_w : (int)(iw * L->scale_w);
                oh = L->out_h ? L->out_h : (int)(ih * L->scale_h);
                oc = ic;
                break;
            case LT_CONV:
                ow = (iw + 2 * L->pad - L->kernel) / L->stride + 1;
                oh = (ih + 2 * L->pad - L->kernel) / L->stride + 1;
                oc = L->num_output;
                break;
            case LT_DECONV:
                ow = (iw - 1) * L->stride - 2 * L->pad + L->kernel;
                oh = (ih - 1) * L->stride - 2 * L->pad + L->kernel;
                oc = L->num_output;
                if (L->upscale == 2) {
                    ow *= 2;
                    oh *= 2;
                    oc /= 4;
                }
                break;
            case LT_RELU:
            case LT_SIGMOID:
            case LT_WARP:
            case LT_ELTWISE:
                ow = iw; oh = ih; oc = ic;
                break;
            case LT_PIXELSHUFFLE: {
                int r = L->upscale;
                ow = iw * r;
                oh = ih * r;
                oc = ic / (r * r);
                break;
            }
            case LT_CROP:
                ow = iw;
                oh = ih;
                oc = L->crop_end - L->crop_start;
                alias_of[L->out_idx[0]] = L->in_idx[0];
                alias_offset[L->out_idx[0]] = L->crop_start * ih * iw;
                break;
            case LT_BINARYOP:
                if (L->with_scalar) {
                    ow = iw; oh = ih; oc = ic;
                } else {
                    int iw1 = shapes[L->in_idx[1]][0];
                    int ih1 = shapes[L->in_idx[1]][1];
                    int ic1 = shapes[L->in_idx[1]][2];
                    ow = iw > iw1 ? iw : iw1;
                    oh = ih > ih1 ? ih : ih1;
                    oc = ic > ic1 ? ic : ic1;
                }
                break;
            default:
                break;
        }

        for (int o = 0; o < L->out_count; o++) {
            int idx = L->out_idx[o];
            shapes[idx][0] = ow;
            shapes[idx][1] = oh;
            shapes[idx][2] = oc;
        }
    }

    // Determine which blobs should have zero-padded border
    int blob_is_padded[MAX_BLOBS] = {0};
    for (int i = 0; i < g.n_layers; i++) {
        const Layer* L = &g.layers[i];
        if (L->type == LT_CONV) {
            blob_is_padded[L->out_idx[0]] = 1;
            blob_is_padded[L->in_idx[0]] = 1;
        }
    }

    // Allocate blob tensors: non-aliased get fresh memory; aliased share data
    Tensor* blobs = (Tensor*)calloc(g.n_blobs, sizeof(Tensor));
    for (int i = 0; i < g.n_blobs; i++) {
        if (shapes[i][0] > 0 && shapes[i][1] > 0 && shapes[i][2] > 0) {
            if (alias_of[i] == -1) {
                if (blob_is_padded[i]) {
                    blobs[i] = tensor_make_padded(shapes[i][0], shapes[i][1], shapes[i][2]);
                } else {
                    blobs[i] = tensor_make(shapes[i][0], shapes[i][1], shapes[i][2]);
                }
            } else {
                blobs[i].w = shapes[i][0];
                blobs[i].h = shapes[i][1];
                blobs[i].c = shapes[i][2];
                blobs[i].is_alias = 1;
                // data pointer resolved before execution
            }
        }
    }

    // Wire up aliases
    for (int i = 0; i < g.n_blobs; i++) {
        if (alias_of[i] != -1) {
            int root = alias_of[i];
            int total_off = alias_offset[i];
            while (alias_of[root] != -1) {
                total_off += alias_offset[root];
                root = alias_of[root];
            }
            blobs[i].data = blobs[root].data + total_off;
            blobs[i].is_padded = blobs[root].is_padded;
        }
    }

    // Size the shared workspace for the largest user: an unpadded conv input
    // copied into a zero-bordered buffer, or a deconv input reordered to HWC.
    size_t scratch_need = 0;
    for (int i = 0; i < g.n_layers; i++) {
        const Layer* L = &g.layers[i];
        const Tensor* in = &blobs[L->in_idx[0]];
        size_t need = 0;
        if (L->type == LT_CONV && !in->is_padded) {
            need = (size_t)in->c * (in->h + 2) * (in->w + 2);
        } else if (L->type == LT_DECONV) {
            need = (size_t)in->c * in->h * in->w;
        }
        if (need > scratch_need) scratch_need = need;
    }
    g_pin_scratch_size = scratch_need;
    size_t scratch_bytes = (scratch_need * sizeof(float) + 63) & ~(size_t)63;
    g_pin_scratch = (float*)aligned_alloc(64, scratch_bytes ? scratch_bytes : 64);
    if (!g_pin_scratch) {
        fprintf(stderr, "Failed to allocate %zu-byte workspace\n", scratch_bytes);
        return 1;
    }

    // Copy padded inputs
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < h; y++) {
            memcpy(blobs[in0_idx].data + (size_t)(c * hp + y) * wp,
                   img0.data + (size_t)(c * h + y) * w,
                   w * sizeof(float));
            memcpy(blobs[in1_idx].data + (size_t)(c * hp + y) * wp,
                   img1.data + (size_t)(c * h + y) * w,
                   w * sizeof(float));
        }
    }
    // Timestep
    for (int i = 0; i < wp * hp; i++) {
        blobs[in2_idx].data[i] = timestep;
    }

    int iters = (argc > 6) ? atoi(argv[6]) : 1;

    if (iters > 1) {
        // Warmup run
        graph_execute(&g, blobs);

        double total_ms = 0.0;
        double min_ms = 1e9, max_ms = 0.0;
        double* times = (double*)malloc(iters * sizeof(double));

        for (int it = 0; it < iters; it++) {
            struct timespec ts0, ts1;
            clock_gettime(CLOCK_MONOTONIC, &ts0);
            graph_execute(&g, blobs);
            clock_gettime(CLOCK_MONOTONIC, &ts1);
            double ms = (ts1.tv_sec - ts0.tv_sec) * 1000.0 + (ts1.tv_nsec - ts0.tv_nsec) * 1e-6;
            times[it] = ms;
            total_ms += ms;
            if (ms < min_ms) min_ms = ms;
            if (ms > max_ms) max_ms = ms;
        }
        double mean_ms = total_ms / iters;
        double variance = 0.0;
        for (int it = 0; it < iters; it++) {
            variance += (times[it] - mean_ms) * (times[it] - mean_ms);
        }
        double std_ms = sqrt(variance / iters);
        printf("Warm Inference (%d runs): Mean = %.2f ms, Std = %.2f ms, Min = %.2f ms, Max = %.2f ms\n",
               iters, mean_ms, std_ms, min_ms, max_ms);
        free(times);
    } else {
        // Single run
        struct timespec ts0, ts1;
        clock_gettime(CLOCK_MONOTONIC, &ts0);

        graph_execute(&g, blobs);

        clock_gettime(CLOCK_MONOTONIC, &ts1);
        double elapsed_ms = (ts1.tv_sec - ts0.tv_sec) * 1000.0 + (ts1.tv_nsec - ts0.tv_nsec) * 1e-6;
        printf("Inference time: %.2f ms\n", elapsed_ms);
    }

    // Extract output
    int out_idx = blob_idx(&g, "out0");
    Tensor* out_padded = &blobs[out_idx];

    Tensor out = tensor_make(w, h, 3);
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < h; y++) {
            memcpy(out.data + (size_t)(c * h + y) * w,
                   out_padded->data + (size_t)(c * hp + y) * wp,
                   w * sizeof(float));
        }
    }

    save_image_rgb(out_path, &out);

    // Cleanup
    tensor_free(&img0);
    tensor_free(&img1);
    tensor_free(&out);
    for (int i = 0; i < g.n_blobs; i++) {
        tensor_free(&blobs[i]);
    }
    free(blobs);
    for (int i = 0; i < g.n_layers; i++) {
        if (g.layers[i].weights) free(g.layers[i].weights);
        if (g.layers[i].bias) free(g.layers[i].bias);
    }
    if (g_pin_scratch) free(g_pin_scratch);

    return 0;
}
