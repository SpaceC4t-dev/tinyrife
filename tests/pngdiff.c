/* pngdiff a.png b.png [tol]: exit 0 if both images have the same size and max |a - b| <= tol (default 1 uint8 level). */
#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s a.png b.png [tol]\n", argv[0]); return 2; }
    int tol = argc > 3 ? atoi(argv[3]) : 1;
    int wa, ha, wb, hb, c;
    unsigned char* a = stbi_load(argv[1], &wa, &ha, &c, 3);
    unsigned char* b = stbi_load(argv[2], &wb, &hb, &c, 3);
    if (!a || !b) { fprintf(stderr, "cannot read %s\n", a ? argv[2] : argv[1]); return 2; }
    if (wa != wb || ha != hb) { printf("size differs: %dx%d vs %dx%d\n", wa, ha, wb, hb); return 1; }
    int maxd = 0;
    long bad = 0, n = (long)wa * ha * 3;
    for (long i = 0; i < n; i++) {
        int d = abs(a[i] - b[i]);
        if (d > maxd) maxd = d;
        bad += d > tol;
    }
    printf("max diff %d (tolerance %d), %ld of %ld values over\n", maxd, tol, bad, n);
    return maxd > tol;
}
