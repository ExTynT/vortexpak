/* VortexPak native C-ABI — generated from c_api.rs with cbindgen. DO NOT EDIT BY HAND. */

#ifndef VORTEXPAK_NATIVE_H
#define VORTEXPAK_NATIVE_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#define BCN_BC1 1

#define BCN_BC4 4

#define BCN_BC5 5

#define QUALITY_FAST 0

#define QUALITY_STANDARD 1

#define QUALITY_MAX 2

#define BCN_BC7 7

#define STRAT_RAW_BYPASS 0

#define STRAT_LZ4_FAST 1

#define STRAT_ZSTD_3 2

#define STRAT_ZSTD_6 3

#define STRAT_ZSTD_12 4

#define STRAT_ZSTD_19 5

#define STRAT_TOKEN_FOLD_ZSTD6 6

#define STRAT_TOKEN_FOLD_ZSTD19 7

#define STRAT_VXLZ 8

#define TX_RAW_DENSE 0

#define TX_ALPHA_STRIP (1 << 0)

#define TX_ROW_RLE (1 << 1)

#define TX_PLANAR_CHW (1 << 2)

typedef struct VortexVfs_File VortexVfs_File;

typedef struct CMicroBlockDesc {
  uint32_t block_id;
  uint32_t strategy_id;
  uintptr_t raw_size;
  const uint8_t *comp_ptr;
  uintptr_t comp_len;
  uint8_t *dst_ptr;
  uintptr_t dst_cap;
  const uint8_t *pat_ptr;
  uintptr_t pat_len;
  uint8_t tok_byte;
  uintptr_t out_len;
  int32_t status;
} CMicroBlockDesc;

typedef struct CTextureDecodeDesc {
  const uint8_t *src_ptr;
  uintptr_t src_len;
  uint8_t *dst_ptr;
  uintptr_t dst_len;
  uint32_t width;
  uint32_t height;
  uint32_t orig_channels;
  uint32_t tx_flags;
  uint32_t rle_unique_rows;
  uint32_t materialize_rgba;
  int32_t status;
} CTextureDecodeDesc;

typedef struct CVxlzCompressDesc {
  const uint8_t *src_ptr;
  uintptr_t src_len;
  uint8_t *dst_ptr;
  uintptr_t dst_cap;
  uintptr_t out_len;
  uint32_t out_strategy;
  uint32_t allow_bypass;
  int32_t status;
} CVxlzCompressDesc;

int32_t vxtx_is_all_opaque(const uint8_t *rgba_ptr, uintptr_t len);

int32_t vxtx_alpha_strip_pack(const uint8_t *rgba_ptr, uintptr_t rgba_len, uint8_t *rgb_dst_ptr);

int32_t vxtx_alpha_strip_unpack(const uint8_t *rgb_ptr, uintptr_t rgb_len, uint8_t *rgba_dst_ptr);

int32_t vxtx_row_rle_decode(const uint16_t *counts_ptr,
                            uintptr_t counts_len,
                            const uint8_t *unique_ptr,
                            uintptr_t row_bytes,
                            uint8_t *dst_ptr,
                            uintptr_t dst_len);

int32_t vxtx_row_rle_encode(const uint8_t *data_ptr,
                            uintptr_t height,
                            uintptr_t row_bytes,
                            uint16_t *counts_dst,
                            uint8_t *unique_dst,
                            uintptr_t max_unique);

int32_t vxtx_planar_decode(const uint8_t *src_ptr,
                           uint8_t *dst_ptr,
                           uintptr_t width,
                           uintptr_t height,
                           uintptr_t channels);

int32_t vxtx_planar_encode(const uint8_t *src_ptr,
                           uint8_t *dst_ptr,
                           uintptr_t width,
                           uintptr_t height,
                           uintptr_t channels);

int32_t rdo_quantize_image(const uint8_t *src_ptr,
                           uintptr_t width,
                           uintptr_t height,
                           uintptr_t channels,
                           int32_t level,
                           uint8_t *dst_ptr);

int32_t rdo_quantize_floats(float *floats_ptr, uintptr_t count, uint32_t drop_bits);

double calculate_image_psnr_c(const uint8_t *orig_ptr, const uint8_t *quant_ptr, uintptr_t len);

double calculate_float_psnr_c(const float *orig_ptr,
                              const float *quant_ptr,
                              uintptr_t count,
                              float range);

int64_t vpak_decompress_block_c(uint32_t strategy_id,
                                const uint8_t *comp_ptr,
                                uintptr_t comp_len,
                                uintptr_t raw_size,
                                uint8_t *dst_ptr,
                                const uint8_t *pat_ptr,
                                uintptr_t pat_len,
                                uint8_t tok_byte);

int32_t vpak_decompress_blocks_parallel_c(struct CMicroBlockDesc *blocks_ptr, uintptr_t num_blocks);

int32_t vxtx_decode_textures_batch_parallel_c(struct CTextureDecodeDesc *descs_ptr,
                                              uintptr_t num_textures);

int64_t rdo_quantize_glb_c(const uint8_t *glb_ptr,
                           uintptr_t glb_len,
                           uint32_t drop_bits,
                           uint8_t *dst_ptr,
                           uintptr_t dst_cap);

struct VortexVfs_File *vfs_open_c(const char *path_ptr);

void vfs_close_c(struct VortexVfs_File *handle);

int64_t vfs_get_asset_size_c(struct VortexVfs_File *handle, const char *path_ptr);

int64_t vfs_read_asset_c(struct VortexVfs_File *handle,
                         const char *path_ptr,
                         uint8_t *dst_ptr,
                         uintptr_t max_len,
                         int32_t materialize_rgba);

uintptr_t vfs_get_assets_count_c(struct VortexVfs_File *handle);

int64_t vxlz_compress_c(const uint8_t *src_ptr,
                        uintptr_t src_len,
                        uint8_t *dst_ptr,
                        uintptr_t dst_cap);

int64_t vxlz_decompress_c(const uint8_t *src_ptr,
                          uintptr_t src_len,
                          uint8_t *dst_ptr,
                          uintptr_t dst_len);

int32_t vxlz_compress_blocks_parallel_c(struct CVxlzCompressDesc *descs_ptr, uintptr_t num_blocks);

int64_t bcn_encode_c(int32_t fmt,
                     const uint8_t *src_ptr,
                     uintptr_t src_len,
                     uint32_t width,
                     uint32_t height,
                     uint8_t *dst_ptr,
                     uintptr_t dst_cap);

int64_t bcn_decode_c(int32_t fmt,
                     const uint8_t *src_ptr,
                     uintptr_t src_len,
                     uint32_t width,
                     uint32_t height,
                     uint8_t *dst_ptr,
                     uintptr_t dst_cap);

int64_t bcn_encode_ex_c(int32_t fmt,
                        uint32_t quality,
                        const uint8_t *src_ptr,
                        uintptr_t src_len,
                        uint32_t width,
                        uint32_t height,
                        uint8_t *dst_ptr,
                        uintptr_t dst_cap);

int64_t bcn_encode_bc7_c(const uint8_t *src_ptr,
                         uintptr_t src_len,
                         uint32_t width,
                         uint32_t height,
                         uint8_t *dst_ptr,
                         uintptr_t dst_cap);

int64_t bcn_decode_bc7_c(const uint8_t *src_ptr,
                         uintptr_t src_len,
                         uint32_t width,
                         uint32_t height,
                         uint8_t *dst_ptr,
                         uintptr_t dst_cap);

int64_t bcn_encode_bc1_rdo_c(uint32_t quality,
                             uint32_t strength_x100,
                             uint32_t iterations,
                             const uint8_t *src_ptr,
                             uintptr_t src_len,
                             uint32_t width,
                             uint32_t height,
                             uint8_t *dst_ptr,
                             uintptr_t dst_cap);

#endif  /* VORTEXPAK_NATIVE_H */
