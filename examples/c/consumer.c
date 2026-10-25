/* Minimal pure-C consumer of the VortexPak native core: proves the ABI in
 * vortexpak.h is usable from C without any Python in the process. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vortexpak.h"

int main(void) {
    /* 1. Opaque-alpha detection on one RGBA pixel */
    unsigned char rgba[4] = {10, 20, 30, 255};
    if (!vxtx_is_all_opaque(rgba, 4)) {
        fprintf(stderr, "FAIL: vxtx_is_all_opaque\n");
        return 1;
    }

    /* 2. Alpha strip pack/unpack roundtrip */
    unsigned char rgb[3] = {0, 0, 0};
    if (vxtx_alpha_strip_pack(rgba, 4, rgb) != 0) {
        fprintf(stderr, "FAIL: vxtx_alpha_strip_pack\n");
        return 1;
    }
    if (rgb[0] != 10 || rgb[1] != 20 || rgb[2] != 30) {
        fprintf(stderr, "FAIL: alpha strip payload\n");
        return 1;
    }
    unsigned char rgba_back[4] = {0, 0, 0, 0};
    if (vxtx_alpha_strip_unpack(rgb, 3, rgba_back) != 0) {
        fprintf(stderr, "FAIL: vxtx_alpha_strip_unpack\n");
        return 1;
    }
    if (rgba_back[0] != 10 || rgba_back[1] != 20 || rgba_back[2] != 30 ||
        rgba_back[3] != 255) {
        fprintf(stderr, "FAIL: alpha strip roundtrip\n");
        return 1;
    }

    /* 3. Proprietary VXLZ compress/decompress roundtrip */
    unsigned char src[64];
    for (int i = 0; i < 64; i++) {
        src[i] = (unsigned char)(i % 7);
    }
    unsigned char comp[256];
    long comp_len = vxlz_compress_c(src, sizeof(src), comp, sizeof(comp));
    if (comp_len <= 0) {
        fprintf(stderr, "FAIL: vxlz_compress_c (%ld)\n", comp_len);
        return 1;
    }
    unsigned char dec[64];
    long dec_len = vxlz_decompress_c(comp, (size_t)comp_len, dec, sizeof(dec));
    if (dec_len != 64 || memcmp(src, dec, sizeof(src)) != 0) {
        fprintf(stderr, "FAIL: vxlz roundtrip (%ld)\n", dec_len);
        return 1;
    }

    printf("VortexPak C-ABI consumer OK\n");
    return 0;
}
