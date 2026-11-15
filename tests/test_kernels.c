/* Unit tests for the CPU kernels in src/cpu.c against naive references. No model or image files needed.
 * Build/run: make test   (or: gcc -O2 -march=native -fopenmp -Ithird_party tests/test_kernels.c -lm) */
#define main cpu_main
#include "../src/cpu.c"
#undef main

static int g_checks = 0, g_fail = 0;
#define CHECK(cond, ...)                                                                                              \
    do {                                                                                                              \
        g_checks++;                                                                                                   \
        if (!(cond)) {                                                                                                \
            g_fail++;                                                                                                 \
            printf("  FAIL %s:%d: ", __FILE__, __LINE__);                                                             \
            printf(__VA_ARGS__);                                                                                      \
            printf("\n");                                                                                             \
        }                                                                                                             \
    } while (0)

static uint32_t g_rng = 12345;
static float frand(float lo, float hi) {
    g_rng = g_rng * 1664525u + 1013904223u;
    return lo + (hi - lo) * ((g_rng >> 8) * (1.0f / 16777216.0f));
}
static void fill(Tensor* t, float lo, float hi) {   /* plain tensors only */
    for (size_t i = 0; i < (size_t)t->w * t->h * t->c; i++) t->data[i] = frand(lo, hi);
}

/* Border-clamped bilinear sample of one planar channel (grid_sample border + align_corners=True semantics). */
static double sample(const float* p, int W, int H, double x, double y) {
    x = x < 0 ? 0 : x > W - 1 ? W - 1 : x;
    y = y < 0 ? 0 : y > H - 1 ? H - 1 : y;
    int x0 = (int)floor(x), y0 = (int)floor(y);
    int x1 = x0 + 1 > W - 1 ? W - 1 : x0 + 1, y1 = y0 + 1 > H - 1 ? H - 1 : y0 + 1;
    double ax = x - x0, ay = y - y0;
    return (p[y0 * W + x0] * (1 - ax) + p[y0 * W + x1] * ax) * (1 - ay) + (p[y1 * W + x0] * (1 - ax) + p[y1 * W + x1] * ax) * ay;
}

/* value of channel c at (x, y), for plain and zero-bordered tensors alike */
static float at(const Tensor* t, int c, int x, int y) {
    if (t->is_padded) return t->data[((size_t)c * (t->h + 2) + y + 1) * (t->w + 2) + x + 1];
    return t->data[((size_t)c * t->h + y) * t->w + x];
}

static double max_abs_diff(const Tensor* a, const Tensor* b) {
    double m = 0;
    for (int c = 0; c < a->c; c++)
        for (int y = 0; y < a->h; y++)
            for (int x = 0; x < a->w; x++) {
                double d = fabs((double)at(a, c, x, y) - at(b, c, x, y));
                if (d > m) m = d;
            }
    return m;
}

/* the zero border of a padded tensor must stay zero */
static int border_is_zero(const Tensor* t) {
    int PW = t->w + 2, PH = t->h + 2;
    for (int c = 0; c < t->c; c++)
        for (int y = 0; y < PH; y++)
            for (int x = 0; x < PW; x++)
                if ((y == 0 || y == PH - 1 || x == 0 || x == PW - 1) && t->data[((size_t)c * PH + y) * PW + x] != 0.0f) return 0;
    return 1;
}

static void test_fp16(void) {
    printf("fp16_to_f32\n");
    CHECK(fp16_to_f32(0x0000) == 0.0f, "zero");
    CHECK(fp16_to_f32(0x3C00) == 1.0f, "one");
    CHECK(fp16_to_f32(0xC000) == -2.0f, "minus two");
    CHECK(fp16_to_f32(0x3800) == 0.5f, "half");
    CHECK(fabsf(fp16_to_f32(0x3555) - 0.33325195f) < 1e-7f, "1/3 rounded");
    CHECK(fp16_to_f32(0x7BFF) == 65504.0f, "max normal");
    CHECK(isinf(fp16_to_f32(0x7C00)) && fp16_to_f32(0x7C00) > 0, "+inf");
    CHECK(isinf(fp16_to_f32(0xFC00)) && fp16_to_f32(0xFC00) < 0, "-inf");
}

static void test_interp(void) {
    printf("interp_forward (bilinear, align_corners=False)\n");
    struct { int iw, ih, ow, oh; } cases[] = { {64, 48, 8, 6}, {64, 48, 16, 12}, {64, 48, 32, 24}, {8, 6, 64, 48}, {16, 12, 64, 48}, {32, 24, 64, 48} };
    for (int padded = 0; padded < 2; padded++) {
        for (unsigned k = 0; k < sizeof cases / sizeof *cases; k++) {
            int iw = cases[k].iw, ih = cases[k].ih, ow = cases[k].ow, oh = cases[k].oh;
            Tensor in = tensor_make(iw, ih, 3);
            fill(&in, -2, 2);
            Tensor out = padded ? tensor_make_padded(ow, oh, 3) : tensor_make(ow, oh, 3);
            float scale = (float)ow / iw;
            interp_forward(&in, &out, scale);
            double worst = 0;
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < oh; y++)
                    for (int x = 0; x < ow; x++) {
                        double ref = sample(in.data + (size_t)c * iw * ih, iw, ih, (x + 0.5) / scale - 0.5, (y + 0.5) / scale - 0.5);
                        double d = fabs(ref - at(&out, c, x, y));
                        if (d > worst) worst = d;
                    }
            CHECK(worst < 1e-5, "%dx%d -> %dx%d%s: max err %g", iw, ih, ow, oh, padded ? " (padded out)" : "", worst);
            if (padded) CHECK(border_is_zero(&out), "%dx%d -> %dx%d: border written", iw, ih, ow, oh);
            tensor_free(&in); tensor_free(&out);
        }
    }
}

static void test_warp(void) {
    printf("warp_forward (border-clamped bilinear)\n");
    int W = 40, H = 32;
    for (int padded = 0; padded < 2; padded++) {
        Tensor img = tensor_make(W, H, 3), flow = tensor_make(W, H, 2);
        fill(&img, 0, 1);
        fill(&flow, -5, 5);
        flow.data[7] = 100.0f; flow.data[W * H + 7] = -100.0f;           /* far outside: must clamp to the border */
        flow.data[W + 3] = -0.0f; flow.data[W * H + W + 3] = 0.0f;       /* exactly on a pixel */
        Tensor out = padded ? tensor_make_padded(W, H, 3) : tensor_make(W, H, 3);
        warp_forward(&img, &flow, &out);
        double worst = 0;
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++) {
                    double ref = sample(img.data + (size_t)c * W * H, W, H, x + flow.data[y * W + x], y + flow.data[W * H + y * W + x]);
                    double d = fabs(ref - at(&out, c, x, y));
                    if (d > worst) worst = d;
                }
        CHECK(worst < 1e-5, "%s out: max err %g", padded ? "padded" : "plain", worst);
        if (padded) CHECK(border_is_zero(&out), "border written");
        tensor_free(&img); tensor_free(&flow); tensor_free(&out);
    }
}

/* warpdown must equal the unfused pipeline it replaces: warps at full size, concat, bilinear downscale */
static void test_warpdown(void) {
    printf("warpdown_forward == warp + concat + interp\n");
    int W = 64, H = 48;
    float scales[] = { 0.5f, 0.25f };
    for (int k = 0; k < 2; k++) {
        float s = scales[k];
        int ow = (int)(W * s), oh = (int)(H * s);
        Tensor img0 = tensor_make(W, H, 3), img1 = tensor_make(W, H, 3), t = tensor_make(W, H, 1), mask = tensor_make(W, H, 1);
        Tensor f0 = tensor_make(W, H, 2), f1 = tensor_make(W, H, 2);
        fill(&img0, 0, 1); fill(&img1, 0, 1); fill(&t, 0, 1); fill(&mask, -3, 3); fill(&f0, -6, 6); fill(&f1, -6, 6);
        WarpSrc src[4] = { { &img0, &f0 }, { &img1, &f1 }, { &t, NULL }, { &mask, NULL } };
        Tensor got = tensor_make_padded(ow, oh, 8);
        warpdown_forward(src, 4, s, &got);

        Tensor full = tensor_make(W, H, 8), ref = tensor_make(ow, oh, 8);
        Tensor w0 = tensor_view(&full, 0, 3), w1 = tensor_view(&full, 3, 3);
        warp_forward(&img0, &f0, &w0);
        warp_forward(&img1, &f1, &w1);
        memcpy(full.data + (size_t)6 * W * H, t.data, (size_t)W * H * sizeof(float));
        memcpy(full.data + (size_t)7 * W * H, mask.data, (size_t)W * H * sizeof(float));
        interp_forward(&full, &ref, s);
        double d = max_abs_diff(&got, &ref);
        CHECK(d < 1e-5, "scale %g: max diff vs unfused %g", s, d);
        CHECK(border_is_zero(&got), "scale %g: border written", s);
        Tensor* all[] = { &img0, &img1, &t, &mask, &f0, &f1, &got, &full, &ref };
        for (unsigned i = 0; i < sizeof all / sizeof *all; i++) tensor_free(all[i]);
    }
}

static void test_merge(void) {
    printf("merge_forward (warp*sigmoid + warp*(1-sigmoid))\n");
    int W = 48, H = 40;
    Tensor a = tensor_make(W, H, 3), b = tensor_make(W, H, 3), fa = tensor_make(W, H, 2), fb = tensor_make(W, H, 2);
    Tensor logit = tensor_make(W, H, 1), out = tensor_make(W, H, 3);
    fill(&a, 0, 1); fill(&b, 0, 1); fill(&fa, -4, 4); fill(&fb, -4, 4); fill(&logit, -8, 8);
    logit.data[3] = 500.0f; logit.data[4] = -500.0f;   /* beyond the clamp: sigmoid saturates, no NaN/inf */
    merge_forward(&a, &fa, &b, &fb, &logit, &out);
    double worst = 0;
    int finite = 1;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            int i = y * W + x;
            double m = 1.0 / (1.0 + exp(-fmax(-88.0, fmin(88.0, (double)logit.data[i]))));
            for (int c = 0; c < 3; c++) {
                double ref = sample(a.data + (size_t)c * W * H, W, H, x + fa.data[i], y + fa.data[W * H + i]) * m +
                             sample(b.data + (size_t)c * W * H, W, H, x + fb.data[i], y + fb.data[W * H + i]) * (1 - m);
                double v = out.data[(size_t)c * W * H + i];
                finite &= isfinite(v);
                if (fabs(ref - v) > worst) worst = fabs(ref - v);
            }
        }
    CHECK(finite, "non-finite output");
    CHECK(worst < 1e-5, "max err %g", worst);
    Tensor* all[] = { &a, &b, &fa, &fb, &logit, &out };
    for (unsigned i = 0; i < sizeof all / sizeof *all; i++) tensor_free(all[i]);
}

static void test_elementwise(void) {
    printf("scale / add / axpby (incl. non-multiple-of-32 sizes)\n");
    int dims[][3] = { {7, 5, 1}, {33, 3, 4}, {64, 64, 5} };   /* 35, 396 and 20480 elements: all hit the vector and the tail loops */
    for (int k = 0; k < 3; k++) {
        int w = dims[k][0], h = dims[k][1], c = dims[k][2];
        size_t n = (size_t)w * h * c;
        Tensor a = tensor_make(w, h, c), b = tensor_make(w, h, c), o = tensor_make(w, h, c);
        fill(&a, -3, 3); fill(&b, -3, 3);
        scale_forward(&a, &o, 0.25f);
        int ok = 1;
        for (size_t i = 0; i < n; i++) ok &= o.data[i] == a.data[i] * 0.25f;
        CHECK(ok, "scale %dx%dx%d", w, h, c);
        add_forward(&a, &b, &o);
        ok = 1;
        for (size_t i = 0; i < n; i++) ok &= o.data[i] == a.data[i] + b.data[i];
        CHECK(ok, "add %dx%dx%d", w, h, c);
        axpby_forward(&a, &b, &o, 1.0f, 4.0f);
        ok = 1;
        for (size_t i = 0; i < n; i++) ok &= fabsf(o.data[i] - (a.data[i] + 4.0f * b.data[i])) < 1e-5f;
        CHECK(ok, "axpby %dx%dx%d", w, h, c);
        tensor_free(&a); tensor_free(&b); tensor_free(&o);
    }
}

/* regression: scaling a zero-bordered view in place must cover every channel plane in full (the planes are
 * (w+2)*(h+2) apart, so sizing the loop as w*h*c left the tail of the last channel unscaled) */
static void test_scale_padded_view(void) {
    printf("scale_forward on a padded channel view\n");
    int w = 16, h = 12;
    Tensor x = tensor_make_padded(w, h, 12);
    for (int c = 0; c < 12; c++)
        for (int y = 0; y < h; y++)
            for (int xx = 0; xx < w; xx++) x.data[((size_t)c * (h + 2) + y + 1) * (w + 2) + xx + 1] = 1.0f;
    Tensor v = tensor_view(&x, 8, 4);
    scale_forward(&v, &v, 0.25f);
    int ok = 1, untouched = 1;
    for (int c = 0; c < 12; c++)
        for (int y = 0; y < h; y++)
            for (int xx = 0; xx < w; xx++) {
                float want = c >= 8 ? 0.25f : 1.0f;
                float got = at(&x, c, xx, y);
                if (c >= 8) ok &= got == want; else untouched &= got == want;
            }
    CHECK(ok, "a pixel of the scaled view was not scaled");
    CHECK(untouched, "channels outside the view were modified");
    CHECK(border_is_zero(&x), "border modified");
    tensor_free(&x);
}

static void test_copy_into_padded(void) {
    printf("copy_into_padded\n");
    Tensor src = tensor_make(20, 14, 3), dst = tensor_make_padded(20, 14, 3);
    fill(&src, -1, 1);
    copy_into_padded(&src, &dst);
    CHECK(max_abs_diff(&src, &dst) == 0.0, "interior differs");
    CHECK(border_is_zero(&dst), "border written");
    tensor_free(&src); tensor_free(&dst);
}

int main(void) {
    test_fp16();
    test_interp();
    test_warp();
    test_warpdown();
    test_merge();
    test_elementwise();
    test_scale_padded_view();
    test_copy_into_padded();
    printf("\nkernel tests: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail != 0;
}
