#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <vector>
#include <string>

#include "/home/user/repos/rife-ncnn-vulkan/src/rife.h"
#include "net.h"
#include "mat.h"

// stb_image
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec * 1e-6;
}

int main(int argc, char** argv) {
    int iters = (argc > 1) ? atoi(argv[1]) : 10;
    int num_threads = (argc > 2) ? atoi(argv[2]) : 4;

    const char* img0_path = "/home/user/repos/playground/frame0.png";
    const char* img1_path = "/home/user/repos/playground/frame1.png";
    const char* model_dir = "/home/user/repos/rife-ncnn-vulkan/models/rife-v4.6";

    int w, h, c;
    unsigned char* data0 = stbi_load(img0_path, &w, &h, &c, 3);
    unsigned char* data1 = stbi_load(img1_path, &w, &h, &c, 3);
    if (!data0 || !data1) {
        fprintf(stderr, "Failed to load input images\n");
        return 1;
    }

    ncnn::Mat in0 = ncnn::Mat(w, h, (void*)data0, (size_t)3, 1);
    ncnn::Mat in1 = ncnn::Mat(w, h, (void*)data1, (size_t)3, 1);
    ncnn::Mat out(w, h, (size_t)3, 1);

    // Initialize RIFE CPU: gpuid = -1, tta = false, tta_temporal = false, uhd = false, num_threads = 4, rife_v2 = false, rife_v4 = true
    RIFE rife(-1, false, false, false, num_threads, false, true);
    if (rife.load(model_dir) != 0) {
        fprintf(stderr, "Failed to load RIFE model\n");
        return 1;
    }

    // Warmup run
    rife.process_v4_cpu(in0, in1, 0.5f, out);

    double total_ms = 0.0;
    double min_ms = 1e9, max_ms = 0.0;
    std::vector<double> times(iters);

    for (int it = 0; it < iters; it++) {
        double t0 = get_time_ms();
        rife.process_v4_cpu(in0, in1, 0.5f, out);
        double t1 = get_time_ms();
        double ms = t1 - t0;
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

    printf("RIFE-NCNN CPU (%d threads, %d runs): Mean = %.2f ms, Std = %.2f ms, Min = %.2f ms, Max = %.2f ms\n",
           num_threads, iters, mean_ms, std_ms, min_ms, max_ms);

    stbi_write_png("/tmp/bench_ncnn_out.png", w, h, 3, out.data, w * 3);

    stbi_image_free(data0);
    stbi_image_free(data1);
    return 0;
}
