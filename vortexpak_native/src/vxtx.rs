//! VortexPak Texture Tensor (VXTX) Transforms
//! Hardware-accelerated with AVX2 SIMD intrinsics and scalar fallback.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use rayon::prelude::*;

pub const TX_RAW_DENSE: u32 = 0;
pub const TX_ALPHA_STRIP: u32 = 1 << 0;
pub const TX_ROW_RLE: u32 = 1 << 1;
pub const TX_PLANAR_CHW: u32 = 1 << 2;

/// Checks if all alpha values in an RGBA buffer are 255 (0xFF).
/// Uses AVX2 SIMD if available.
pub fn is_all_opaque(rgba: &[u8]) -> bool {
    if rgba.len() % 4 != 0 {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { is_all_opaque_avx2(rgba) };
        }
    }
    is_all_opaque_scalar(rgba)
}

fn is_all_opaque_scalar(rgba: &[u8]) -> bool {
    let mut i = 3;
    while i < rgba.len() {
        if rgba[i] != 255 {
            return false;
        }
        i += 4;
    }
    true
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn is_all_opaque_avx2(rgba: &[u8]) -> bool {
    let total_pixels = rgba.len() / 4;
    let chunks = total_pixels / 8; // 8 pixels = 32 bytes
    let alpha_mask = _mm256_set1_epi32(0xFF000000_u32 as i32);

    let ptr = rgba.as_ptr() as *const __m256i;
    for i in 0..chunks {
        let v = _mm256_loadu_si256(ptr.add(i));
        let masked = _mm256_and_si256(v, alpha_mask);
        let cmp = _mm256_cmpeq_epi32(masked, alpha_mask);
        let mask = _mm256_movemask_epi8(cmp);
        if mask != -1 {
            return false;
        }
    }

    // Handle remaining pixels (less than 8)
    let rem_start = chunks * 8 * 4;
    let mut i = rem_start + 3;
    while i < rgba.len() {
        if *rgba.get_unchecked(i) != 255 {
            return false;
        }
        i += 4;
    }
    true
}

/// Strip constant alpha channel (RGBA -> RGB).
pub fn alpha_strip_pack(rgba: &[u8], rgb_dst: &mut [u8]) {
    assert_eq!(rgba.len() / 4 * 3, rgb_dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                alpha_strip_pack_avx2(rgba, rgb_dst);
                return;
            }
        }
    }
    alpha_strip_pack_scalar(rgba, rgb_dst);
}

fn alpha_strip_pack_scalar(rgba: &[u8], rgb: &mut [u8]) {
    let mut src_idx = 0;
    let mut dst_idx = 0;
    while src_idx + 4 <= rgba.len() {
        rgb[dst_idx] = rgba[src_idx];
        rgb[dst_idx + 1] = rgba[src_idx + 1];
        rgb[dst_idx + 2] = rgba[src_idx + 2];
        src_idx += 4;
        dst_idx += 3;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn alpha_strip_pack_avx2(rgba: &[u8], rgb: &mut [u8]) {
    let total_pixels = rgba.len() / 4;
    let chunks8 = total_pixels / 8; // 8 pixels = 32 bytes in (256-bit AVX2), 24 bytes out

    // 256-bit shuffle mask (per 128-bit lane) to pack [R0 G0 B0 A0 ... R3 G3 B3 A3] -> 12 bytes per lane
    let shuf_mask256 = _mm256_set_epi8(
        -1, -1, -1, -1, 14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0, -1, -1, -1, -1, 14, 13, 12, 10, 9,
        8, 6, 5, 4, 2, 1, 0,
    );

    let mut src_ptr = rgba.as_ptr();
    let mut dst_ptr = rgb.as_mut_ptr();

    for _ in 0..chunks8 {
        let v32 = _mm256_loadu_si256(src_ptr as *const __m256i);
        let packed256 = _mm256_shuffle_epi8(v32, shuf_mask256);

        let lane0 = _mm256_castsi256_si128(packed256);
        let lane1 = _mm256_extracti128_si256(packed256, 1);

        // Store 12 bytes from lane 0
        let lo64_0 = _mm_cvtsi128_si64(lane0);
        std::ptr::copy_nonoverlapping(&lo64_0 as *const i64 as *const u8, dst_ptr, 8);
        let hi32_0 = _mm_extract_epi32(lane0, 2);
        std::ptr::copy_nonoverlapping(&hi32_0 as *const i32 as *const u8, dst_ptr.add(8), 4);

        // Store 12 bytes from lane 1
        let lo64_1 = _mm_cvtsi128_si64(lane1);
        std::ptr::copy_nonoverlapping(&lo64_1 as *const i64 as *const u8, dst_ptr.add(12), 8);
        let hi32_1 = _mm_extract_epi32(lane1, 2);
        std::ptr::copy_nonoverlapping(&hi32_1 as *const i32 as *const u8, dst_ptr.add(20), 4);

        src_ptr = src_ptr.add(32);
        dst_ptr = dst_ptr.add(24);
    }

    // Remainder pixels
    let rem_pixels = total_pixels % 8;
    for _ in 0..rem_pixels {
        *dst_ptr = *src_ptr;
        *dst_ptr.add(1) = *src_ptr.add(1);
        *dst_ptr.add(2) = *src_ptr.add(2);
        src_ptr = src_ptr.add(4);
        dst_ptr = dst_ptr.add(3);
    }
}

/// Unpack stripped alpha (RGB -> RGBA with A=255).
pub fn alpha_strip_unpack(rgb: &[u8], rgba_dst: &mut [u8]) {
    assert_eq!(rgb.len() / 3 * 4, rgba_dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                alpha_strip_unpack_avx2(rgb, rgba_dst);
                return;
            }
        }
    }
    alpha_strip_unpack_scalar(rgb, rgba_dst);
}

fn alpha_strip_unpack_scalar(rgb: &[u8], rgba: &mut [u8]) {
    let mut src_idx = 0;
    let mut dst_idx = 0;
    while src_idx + 3 <= rgb.len() {
        rgba[dst_idx] = rgb[src_idx];
        rgba[dst_idx + 1] = rgb[src_idx + 1];
        rgba[dst_idx + 2] = rgb[src_idx + 2];
        rgba[dst_idx + 3] = 255;
        src_idx += 3;
        dst_idx += 4;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn alpha_strip_unpack_avx2(rgb: &[u8], rgba: &mut [u8]) {
    let total_pixels = rgb.len() / 3;
    // Process 8 pixels (24 bytes in -> 32 bytes out via 256-bit AVX2) while at least 28 bytes remain in `rgb`
    // so that reading 16 bytes at `src_ptr + 12` stays safely within bounds.
    let safe_chunks8 = if rgb.len() >= 28 {
        (rgb.len() - 4) / 24
    } else {
        0
    };

    let shuf_mask256 = _mm256_set_epi8(
        -1, 11, 10, 9, -1, 8, 7, 6, -1, 5, 4, 3, -1, 2, 1, 0, -1, 11, 10, 9, -1, 8, 7, 6, -1, 5, 4,
        3, -1, 2, 1, 0,
    );
    let alpha_mask256 = _mm256_set1_epi32(0xFF000000_u32 as i32);

    let mut src_ptr = rgb.as_ptr();
    let mut dst_ptr = rgba.as_mut_ptr();

    for _ in 0..safe_chunks8 {
        let lane0 = _mm_loadu_si128(src_ptr as *const __m128i);
        let lane1 = _mm_loadu_si128(src_ptr.add(12) as *const __m128i);
        let in24 = _mm256_set_m128i(lane1, lane0);
        let shuffled = _mm256_shuffle_epi8(in24, shuf_mask256);
        let out32 = _mm256_or_si256(shuffled, alpha_mask256);
        _mm256_storeu_si256(dst_ptr as *mut __m256i, out32);

        src_ptr = src_ptr.add(24);
        dst_ptr = dst_ptr.add(32);
    }

    let rem_pixels = total_pixels - safe_chunks8 * 8;
    for _ in 0..rem_pixels {
        *dst_ptr = *src_ptr;
        *dst_ptr.add(1) = *src_ptr.add(1);
        *dst_ptr.add(2) = *src_ptr.add(2);
        *dst_ptr.add(3) = 255;
        src_ptr = src_ptr.add(3);
        dst_ptr = dst_ptr.add(4);
    }
}

/// Fast scanline row comparison using AVX2.
pub fn rows_equal(row_a: &[u8], row_b: &[u8]) -> bool {
    debug_assert_eq!(row_a.len(), row_b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { rows_equal_avx2(row_a, row_b) };
        }
    }
    row_a == row_b
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rows_equal_avx2(a: &[u8], b: &[u8]) -> bool {
    let len = a.len();
    let chunks = len / 32;
    let ptr_a = a.as_ptr() as *const __m256i;
    let ptr_b = b.as_ptr() as *const __m256i;

    for i in 0..chunks {
        let va = _mm256_loadu_si256(ptr_a.add(i));
        let vb = _mm256_loadu_si256(ptr_b.add(i));
        let cmp = _mm256_cmpeq_epi8(va, vb);
        if _mm256_movemask_epi8(cmp) != -1 {
            return false;
        }
    }

    let rem = len % 32;
    if rem > 0 {
        let start = chunks * 32;
        return a[start..] == b[start..];
    }
    true
}

/// Row-RLE encode: deduplicate consecutive identical scanlines.
pub fn row_rle_encode(data: &[u8], height: usize, row_bytes: usize) -> (Vec<u16>, Vec<u8>) {
    assert_eq!(data.len(), height * row_bytes);
    if height == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut counts = Vec::new();
    let mut unique_rows = Vec::new();

    let mut prev_row_idx = 0;
    let mut cur_count: u16 = 1;

    for r in 1..height {
        let prev_slice = &data[prev_row_idx * row_bytes..(prev_row_idx + 1) * row_bytes];
        let cur_slice = &data[r * row_bytes..(r + 1) * row_bytes];

        if rows_equal(prev_slice, cur_slice) && cur_count < u16::MAX {
            cur_count += 1;
        } else {
            counts.push(cur_count);
            unique_rows.extend_from_slice(prev_slice);
            prev_row_idx = r;
            cur_count = 1;
        }
    }

    counts.push(cur_count);
    unique_rows.extend_from_slice(&data[prev_row_idx * row_bytes..(prev_row_idx + 1) * row_bytes]);

    (counts, unique_rows)
}

/// Row-RLE decode: reconstruct full scanlines from run counts and unique rows.
pub fn row_rle_decode(counts: &[u16], unique_rows: &[u8], row_bytes: usize, dst: &mut [u8]) {
    let mut u_offset = 0;
    let mut dst_offset = 0;

    for &cnt in counts {
        let src_row = &unique_rows[u_offset..u_offset + row_bytes];
        for _ in 0..cnt {
            dst[dst_offset..dst_offset + row_bytes].copy_from_slice(src_row);
            dst_offset += row_bytes;
        }
        u_offset += row_bytes;
    }
    debug_assert_eq!(u_offset, unique_rows.len());
    debug_assert_eq!(dst_offset, dst.len());
}

/// Planar channel separation: (H, W, C) interleaved -> (C, H, W) planar.
/// Accelerated via Rayon parallelization across channel planes with unrolled gather loops.
pub fn planar_encode(src: &[u8], width: usize, height: usize, channels: usize, dst: &mut [u8]) {
    assert_eq!(src.len(), width * height * channels);
    assert_eq!(dst.len(), src.len());

    let plane_size = width * height;
    if plane_size == 0 || channels == 0 {
        return;
    }

    if plane_size < 16384 {
        for c in 0..channels {
            let plane_dst = &mut dst[c * plane_size..(c + 1) * plane_size];
            for idx in 0..plane_size {
                plane_dst[idx] = src[idx * channels + c];
            }
        }
        return;
    }

    // Parallelize across channel planes using Rayon
    dst.par_chunks_mut(plane_size)
        .enumerate()
        .for_each(|(c, plane_dst)| {
            let mut idx = 0;
            // 8-wide unrolling for cache-friendly strided gathering
            while idx + 8 <= plane_size {
                let s = idx * channels + c;
                plane_dst[idx] = src[s];
                plane_dst[idx + 1] = src[s + channels];
                plane_dst[idx + 2] = src[s + channels * 2];
                plane_dst[idx + 3] = src[s + channels * 3];
                plane_dst[idx + 4] = src[s + channels * 4];
                plane_dst[idx + 5] = src[s + channels * 5];
                plane_dst[idx + 6] = src[s + channels * 6];
                plane_dst[idx + 7] = src[s + channels * 7];
                idx += 8;
            }
            while idx < plane_size {
                plane_dst[idx] = src[idx * channels + c];
                idx += 1;
            }
        });
}

/// Planar channel interleave: (C, H, W) planar -> (H, W, C) interleaved.
/// Accelerated with AVX2 SIMD intrinsics and Rayon multi-threading across pixel chunks.
pub fn planar_decode(src: &[u8], dst: &mut [u8], width: usize, height: usize, channels: usize) {
    assert_eq!(src.len(), width * height * channels);
    assert_eq!(dst.len(), src.len());

    let plane_size = width * height;
    if plane_size == 0 || channels == 0 {
        return;
    }

    if plane_size < 16384 {
        planar_decode_scalar(src, dst, plane_size, channels);
        return;
    }

    let chunk_pixels = 8192;
    dst.par_chunks_mut(chunk_pixels * channels)
        .enumerate()
        .for_each(|(chunk_idx, dst_chunk)| {
            let pixel_start = chunk_idx * chunk_pixels;
            let current_chunk_pixels = dst_chunk.len() / channels;
            let pixel_end = pixel_start + current_chunk_pixels;

            if channels == 4 {
                #[cfg(target_arch = "x86_64")]
                {
                    if is_x86_feature_detected!("avx2") {
                        unsafe {
                            planar_decode_rgba_simd(
                                src,
                                plane_size,
                                pixel_start,
                                pixel_end,
                                dst_chunk,
                            );
                            return;
                        }
                    }
                }
                planar_decode_rgba_unrolled(src, plane_size, pixel_start, pixel_end, dst_chunk);
            } else if channels == 3 {
                planar_decode_rgb_unrolled(src, plane_size, pixel_start, pixel_end, dst_chunk);
            } else {
                for (p_rel, p_abs) in (pixel_start..pixel_end).enumerate() {
                    for c in 0..channels {
                        dst_chunk[p_rel * channels + c] = src[c * plane_size + p_abs];
                    }
                }
            }
        });
}

fn planar_decode_scalar(src: &[u8], dst: &mut [u8], plane_size: usize, channels: usize) {
    for c in 0..channels {
        let plane_src = &src[c * plane_size..(c + 1) * plane_size];
        for idx in 0..plane_size {
            dst[idx * channels + c] = plane_src[idx];
        }
    }
}

fn planar_decode_rgb_unrolled(
    src: &[u8],
    plane_size: usize,
    pixel_start: usize,
    pixel_end: usize,
    dst_chunk: &mut [u8],
) {
    let r_plane = &src[0..plane_size];
    let g_plane = &src[plane_size..2 * plane_size];
    let b_plane = &src[2 * plane_size..3 * plane_size];

    let mut p_abs = pixel_start;
    let mut d_idx;

    unsafe {
        let mut r_ptr = r_plane.as_ptr().add(p_abs);
        let mut g_ptr = g_plane.as_ptr().add(p_abs);
        let mut b_ptr = b_plane.as_ptr().add(p_abs);
        let mut dst_ptr = dst_chunk.as_mut_ptr();

        while p_abs + 8 <= pixel_end {
            *dst_ptr = *r_ptr;
            *dst_ptr.add(1) = *g_ptr;
            *dst_ptr.add(2) = *b_ptr;

            *dst_ptr.add(3) = *r_ptr.add(1);
            *dst_ptr.add(4) = *g_ptr.add(1);
            *dst_ptr.add(5) = *b_ptr.add(1);

            *dst_ptr.add(6) = *r_ptr.add(2);
            *dst_ptr.add(7) = *g_ptr.add(2);
            *dst_ptr.add(8) = *b_ptr.add(2);

            *dst_ptr.add(9) = *r_ptr.add(3);
            *dst_ptr.add(10) = *g_ptr.add(3);
            *dst_ptr.add(11) = *b_ptr.add(3);

            *dst_ptr.add(12) = *r_ptr.add(4);
            *dst_ptr.add(13) = *g_ptr.add(4);
            *dst_ptr.add(14) = *b_ptr.add(4);

            *dst_ptr.add(15) = *r_ptr.add(5);
            *dst_ptr.add(16) = *g_ptr.add(5);
            *dst_ptr.add(17) = *b_ptr.add(5);

            *dst_ptr.add(18) = *r_ptr.add(6);
            *dst_ptr.add(19) = *g_ptr.add(6);
            *dst_ptr.add(20) = *b_ptr.add(6);

            *dst_ptr.add(21) = *r_ptr.add(7);
            *dst_ptr.add(22) = *g_ptr.add(7);
            *dst_ptr.add(23) = *b_ptr.add(7);

            r_ptr = r_ptr.add(8);
            g_ptr = g_ptr.add(8);
            b_ptr = b_ptr.add(8);
            dst_ptr = dst_ptr.add(24);
            p_abs += 8;
        }

        d_idx = (p_abs - pixel_start) * 3;
    }

    while p_abs < pixel_end {
        dst_chunk[d_idx] = r_plane[p_abs];
        dst_chunk[d_idx + 1] = g_plane[p_abs];
        dst_chunk[d_idx + 2] = b_plane[p_abs];
        d_idx += 3;
        p_abs += 1;
    }
}

fn planar_decode_rgba_unrolled(
    src: &[u8],
    plane_size: usize,
    pixel_start: usize,
    pixel_end: usize,
    dst_chunk: &mut [u8],
) {
    let r_plane = &src[0..plane_size];
    let g_plane = &src[plane_size..2 * plane_size];
    let b_plane = &src[2 * plane_size..3 * plane_size];
    let a_plane = &src[3 * plane_size..4 * plane_size];

    let mut p_abs = pixel_start;
    let mut d_idx;

    unsafe {
        let mut r_ptr = r_plane.as_ptr().add(p_abs);
        let mut g_ptr = g_plane.as_ptr().add(p_abs);
        let mut b_ptr = b_plane.as_ptr().add(p_abs);
        let mut a_ptr = a_plane.as_ptr().add(p_abs);
        let mut dst_ptr = dst_chunk.as_mut_ptr();

        while p_abs + 4 <= pixel_end {
            *dst_ptr = *r_ptr;
            *dst_ptr.add(1) = *g_ptr;
            *dst_ptr.add(2) = *b_ptr;
            *dst_ptr.add(3) = *a_ptr;

            *dst_ptr.add(4) = *r_ptr.add(1);
            *dst_ptr.add(5) = *g_ptr.add(1);
            *dst_ptr.add(6) = *b_ptr.add(1);
            *dst_ptr.add(7) = *a_ptr.add(1);

            *dst_ptr.add(8) = *r_ptr.add(2);
            *dst_ptr.add(9) = *g_ptr.add(2);
            *dst_ptr.add(10) = *b_ptr.add(2);
            *dst_ptr.add(11) = *a_ptr.add(2);

            *dst_ptr.add(12) = *r_ptr.add(3);
            *dst_ptr.add(13) = *g_ptr.add(3);
            *dst_ptr.add(14) = *b_ptr.add(3);
            *dst_ptr.add(15) = *a_ptr.add(3);

            r_ptr = r_ptr.add(4);
            g_ptr = g_ptr.add(4);
            b_ptr = b_ptr.add(4);
            a_ptr = a_ptr.add(4);
            dst_ptr = dst_ptr.add(16);
            p_abs += 4;
        }

        d_idx = (p_abs - pixel_start) * 4;
    }

    while p_abs < pixel_end {
        dst_chunk[d_idx] = r_plane[p_abs];
        dst_chunk[d_idx + 1] = g_plane[p_abs];
        dst_chunk[d_idx + 2] = b_plane[p_abs];
        dst_chunk[d_idx + 3] = a_plane[p_abs];
        d_idx += 4;
        p_abs += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn planar_decode_rgba_simd(
    src: &[u8],
    plane_size: usize,
    pixel_start: usize,
    pixel_end: usize,
    dst_chunk: &mut [u8],
) {
    let r_plane = &src[0..plane_size];
    let g_plane = &src[plane_size..2 * plane_size];
    let b_plane = &src[2 * plane_size..3 * plane_size];
    let a_plane = &src[3 * plane_size..4 * plane_size];

    let mut p_abs = pixel_start;
    let mut r_ptr = r_plane.as_ptr().add(p_abs);
    let mut g_ptr = g_plane.as_ptr().add(p_abs);
    let mut b_ptr = b_plane.as_ptr().add(p_abs);
    let mut a_ptr = a_plane.as_ptr().add(p_abs);
    let mut dst_ptr = dst_chunk.as_mut_ptr();

    // True 256-bit AVX2 loop: 32 pixels (128 RGBA bytes) per iteration
    while p_abs + 32 <= pixel_end {
        let r = _mm256_loadu_si256(r_ptr as *const __m256i);
        let g = _mm256_loadu_si256(g_ptr as *const __m256i);
        let b = _mm256_loadu_si256(b_ptr as *const __m256i);
        let a = _mm256_loadu_si256(a_ptr as *const __m256i);

        let rg_lo = _mm256_unpacklo_epi8(r, g);
        let rg_hi = _mm256_unpackhi_epi8(r, g);
        let ba_lo = _mm256_unpacklo_epi8(b, a);
        let ba_hi = _mm256_unpackhi_epi8(b, a);

        let rgba_0_16 = _mm256_unpacklo_epi16(rg_lo, ba_lo);
        let rgba_4_20 = _mm256_unpackhi_epi16(rg_lo, ba_lo);
        let rgba_8_24 = _mm256_unpacklo_epi16(rg_hi, ba_hi);
        let rgba_12_28 = _mm256_unpackhi_epi16(rg_hi, ba_hi);

        let out0 = _mm256_permute2x128_si256(rgba_0_16, rgba_4_20, 0x20);
        let out1 = _mm256_permute2x128_si256(rgba_8_24, rgba_12_28, 0x20);
        let out2 = _mm256_permute2x128_si256(rgba_0_16, rgba_4_20, 0x31);
        let out3 = _mm256_permute2x128_si256(rgba_8_24, rgba_12_28, 0x31);

        _mm256_storeu_si256(dst_ptr as *mut __m256i, out0);
        _mm256_storeu_si256(dst_ptr.add(32) as *mut __m256i, out1);
        _mm256_storeu_si256(dst_ptr.add(64) as *mut __m256i, out2);
        _mm256_storeu_si256(dst_ptr.add(96) as *mut __m256i, out3);

        r_ptr = r_ptr.add(32);
        g_ptr = g_ptr.add(32);
        b_ptr = b_ptr.add(32);
        a_ptr = a_ptr.add(32);
        dst_ptr = dst_ptr.add(128);
        p_abs += 32;
    }

    let mut d_idx = (p_abs - pixel_start) * 4;
    while p_abs < pixel_end {
        dst_chunk[d_idx] = r_plane[p_abs];
        dst_chunk[d_idx + 1] = g_plane[p_abs];
        dst_chunk[d_idx + 2] = b_plane[p_abs];
        dst_chunk[d_idx + 3] = a_plane[p_abs];
        d_idx += 4;
        p_abs += 1;
    }
}

/// Fused Planar RGB (3 channels) -> Interleaved RGBA (4 channels with A=255) decode.
/// Eliminates intermediate 24-bit RGB buffer allocation and halves memory bandwidth on 4K opaque textures.
pub fn planar_decode_rgb_to_rgba(src: &[u8], dst: &mut [u8], width: usize, height: usize) {
    let plane_size = width * height;
    assert_eq!(src.len(), plane_size * 3);
    assert_eq!(dst.len(), plane_size * 4);
    if plane_size == 0 {
        return;
    }

    let chunk_pixels = 16384;
    if plane_size < chunk_pixels {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    planar_decode_rgb_to_rgba_simd(src, plane_size, 0, plane_size, dst);
                    return;
                }
            }
        }
        planar_decode_rgb_to_rgba_scalar(src, plane_size, 0, plane_size, dst);
        return;
    }

    dst.par_chunks_mut(chunk_pixels * 4)
        .enumerate()
        .for_each(|(chunk_idx, dst_chunk)| {
            let pixel_start = chunk_idx * chunk_pixels;
            let current_chunk_pixels = dst_chunk.len() / 4;
            let pixel_end = pixel_start + current_chunk_pixels;
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    unsafe {
                        planar_decode_rgb_to_rgba_simd(
                            src,
                            plane_size,
                            pixel_start,
                            pixel_end,
                            dst_chunk,
                        );
                        return;
                    }
                }
            }
            planar_decode_rgb_to_rgba_scalar(src, plane_size, pixel_start, pixel_end, dst_chunk);
        });
}

fn planar_decode_rgb_to_rgba_scalar(
    src: &[u8],
    plane_size: usize,
    pixel_start: usize,
    pixel_end: usize,
    dst_chunk: &mut [u8],
) {
    let r_plane = &src[0..plane_size];
    let g_plane = &src[plane_size..2 * plane_size];
    let b_plane = &src[2 * plane_size..3 * plane_size];
    for (p_rel, p_abs) in (pixel_start..pixel_end).enumerate() {
        let d = p_rel * 4;
        dst_chunk[d] = r_plane[p_abs];
        dst_chunk[d + 1] = g_plane[p_abs];
        dst_chunk[d + 2] = b_plane[p_abs];
        dst_chunk[d + 3] = 255;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn planar_decode_rgb_to_rgba_simd(
    src: &[u8],
    plane_size: usize,
    pixel_start: usize,
    pixel_end: usize,
    dst_chunk: &mut [u8],
) {
    let r_plane = &src[0..plane_size];
    let g_plane = &src[plane_size..2 * plane_size];
    let b_plane = &src[2 * plane_size..3 * plane_size];

    let mut p_abs = pixel_start;
    let mut r_ptr = r_plane.as_ptr().add(p_abs);
    let mut g_ptr = g_plane.as_ptr().add(p_abs);
    let mut b_ptr = b_plane.as_ptr().add(p_abs);
    let mut dst_ptr = dst_chunk.as_mut_ptr();
    let a = _mm256_set1_epi8(-1); // 0xFF = 255 across all 32 bytes

    // True 256-bit AVX2 loop: 32 pixels (128 RGBA bytes) per iteration
    while p_abs + 32 <= pixel_end {
        let r = _mm256_loadu_si256(r_ptr as *const __m256i);
        let g = _mm256_loadu_si256(g_ptr as *const __m256i);
        let b = _mm256_loadu_si256(b_ptr as *const __m256i);

        let rg_lo = _mm256_unpacklo_epi8(r, g);
        let rg_hi = _mm256_unpackhi_epi8(r, g);
        let ba_lo = _mm256_unpacklo_epi8(b, a);
        let ba_hi = _mm256_unpackhi_epi8(b, a);

        let rgba_0_16 = _mm256_unpacklo_epi16(rg_lo, ba_lo);
        let rgba_4_20 = _mm256_unpackhi_epi16(rg_lo, ba_lo);
        let rgba_8_24 = _mm256_unpacklo_epi16(rg_hi, ba_hi);
        let rgba_12_28 = _mm256_unpackhi_epi16(rg_hi, ba_hi);

        let out0 = _mm256_permute2x128_si256(rgba_0_16, rgba_4_20, 0x20);
        let out1 = _mm256_permute2x128_si256(rgba_8_24, rgba_12_28, 0x20);
        let out2 = _mm256_permute2x128_si256(rgba_0_16, rgba_4_20, 0x31);
        let out3 = _mm256_permute2x128_si256(rgba_8_24, rgba_12_28, 0x31);

        _mm256_storeu_si256(dst_ptr as *mut __m256i, out0);
        _mm256_storeu_si256(dst_ptr.add(32) as *mut __m256i, out1);
        _mm256_storeu_si256(dst_ptr.add(64) as *mut __m256i, out2);
        _mm256_storeu_si256(dst_ptr.add(96) as *mut __m256i, out3);

        r_ptr = r_ptr.add(32);
        g_ptr = g_ptr.add(32);
        b_ptr = b_ptr.add(32);
        dst_ptr = dst_ptr.add(128);
        p_abs += 32;
    }

    let mut d_idx = (p_abs - pixel_start) * 4;
    while p_abs < pixel_end {
        dst_chunk[d_idx] = r_plane[p_abs];
        dst_chunk[d_idx + 1] = g_plane[p_abs];
        dst_chunk[d_idx + 2] = b_plane[p_abs];
        dst_chunk[d_idx + 3] = 255;
        d_idx += 4;
        p_abs += 1;
    }
}

/// Fused zero-allocation VXTX texture decoder directly into destination slice `dst`.
/// Handles any combination of TX_ROW_RLE, TX_PLANAR_CHW, and TX_ALPHA_STRIP without temporary heap buffers.
pub fn decode_texture_vxtx_into(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    orig_channels: usize,
    tx_flags: u32,
    rle_unique_rows: usize,
    materialize_rgba: bool,
) -> Result<(), String> {
    let stripped = (tx_flags & TX_ALPHA_STRIP) != 0;
    let work_c = if stripped { 3 } else { orig_channels };
    let out_c = if stripped && materialize_rgba {
        4
    } else {
        work_c
    };

    if dst.len() < width * height * out_c {
        return Err(format!(
            "Destination buffer too small for VXTX texture: {} < {}",
            dst.len(),
            width * height * out_c
        ));
    }

    if (tx_flags & TX_ROW_RLE) != 0 {
        let counts_bytes = rle_unique_rows * 2;
        if src.len() < counts_bytes {
            return Err("Truncated Row-RLE header".into());
        }
        let counts_raw = &src[..counts_bytes];
        let unique_rows = &src[counts_bytes..];
        let src_row_bytes = width * work_c;
        let dst_row_bytes = width * out_c;

        if unique_rows.len() < rle_unique_rows * src_row_bytes {
            return Err("Truncated Row-RLE unique rows".into());
        }

        // Parallelize Row-RLE across 8 disjoint destination slices for large textures
        if width * height >= 131_072 && rle_unique_rows >= 32 {
            let num_chunks = 8.min(rle_unique_rows);
            let u_per_chunk = (rle_unique_rows + num_chunks - 1) / num_chunks;
            let mut work_chunks: Vec<(usize, usize, &mut [u8])> = Vec::with_capacity(num_chunks);
            let mut rem_dst = &mut dst[..width * height * out_c];

            for c_idx in 0..num_chunks {
                let u_start = c_idx * u_per_chunk;
                if u_start >= rle_unique_rows {
                    break;
                }
                let u_end = ((c_idx + 1) * u_per_chunk).min(rle_unique_rows);
                let mut rows_in_chunk = 0usize;
                for i in u_start..u_end {
                    rows_in_chunk +=
                        u16::from_le_bytes([counts_raw[i * 2], counts_raw[i * 2 + 1]]) as usize;
                }
                let chunk_bytes = rows_in_chunk * dst_row_bytes;
                if chunk_bytes <= rem_dst.len() {
                    let (cur_slice, rest) = rem_dst.split_at_mut(chunk_bytes);
                    rem_dst = rest;
                    work_chunks.push((u_start, u_end, cur_slice));
                }
            }

            work_chunks
                .into_par_iter()
                .for_each(|(u_start, u_end, chunk_dst)| {
                    let mut dst_offset = 0;
                    for i in u_start..u_end {
                        let cnt =
                            u16::from_le_bytes([counts_raw[i * 2], counts_raw[i * 2 + 1]]) as usize;
                        if cnt == 0 {
                            continue;
                        }
                        let u_offset = i * src_row_bytes;
                        let src_row = &unique_rows[u_offset..u_offset + src_row_bytes];
                        if stripped && materialize_rgba {
                            let first_dst_end = dst_offset + dst_row_bytes;
                            alpha_strip_unpack(src_row, &mut chunk_dst[dst_offset..first_dst_end]);
                            for k in 1..cnt {
                                let target_start = dst_offset + k * dst_row_bytes;
                                chunk_dst.copy_within(dst_offset..first_dst_end, target_start);
                            }
                            dst_offset += cnt * dst_row_bytes;
                        } else {
                            for _ in 0..cnt {
                                chunk_dst[dst_offset..dst_offset + dst_row_bytes]
                                    .copy_from_slice(src_row);
                                dst_offset += dst_row_bytes;
                            }
                        }
                    }
                });
            return Ok(());
        }

        let mut u_offset = 0;
        let mut dst_offset = 0;

        for i in 0..rle_unique_rows {
            let cnt = u16::from_le_bytes([counts_raw[i * 2], counts_raw[i * 2 + 1]]) as usize;
            if cnt == 0 {
                u_offset += src_row_bytes;
                continue;
            }
            let src_row = &unique_rows[u_offset..u_offset + src_row_bytes];
            if stripped && materialize_rgba {
                // Unpack RGB -> RGBA once into the first destination row
                let first_dst_end = dst_offset + dst_row_bytes;
                alpha_strip_unpack(src_row, &mut dst[dst_offset..first_dst_end]);
                // Duplicate already-unpacked RGBA row for remaining cnt - 1 rows
                for k in 1..cnt {
                    let target_start = dst_offset + k * dst_row_bytes;
                    dst.copy_within(dst_offset..first_dst_end, target_start);
                }
                dst_offset += cnt * dst_row_bytes;
            } else {
                for _ in 0..cnt {
                    dst[dst_offset..dst_offset + dst_row_bytes].copy_from_slice(src_row);
                    dst_offset += dst_row_bytes;
                }
            }
            u_offset += src_row_bytes;
        }
        Ok(())
    } else if (tx_flags & TX_PLANAR_CHW) != 0 {
        if stripped && materialize_rgba {
            planar_decode_rgb_to_rgba(src, &mut dst[..width * height * 4], width, height);
        } else {
            planar_decode(
                src,
                &mut dst[..width * height * out_c],
                width,
                height,
                out_c,
            );
        }
        Ok(())
    } else {
        if stripped && materialize_rgba {
            alpha_strip_unpack(src, &mut dst[..width * height * 4]);
        } else {
            let n = width * height * out_c;
            dst[..n].copy_from_slice(&src[..n]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_strip_roundtrip() {
        let mut rgba = vec![0u8; 16 * 16 * 4];
        for (i, b) in rgba.iter_mut().enumerate() {
            if i % 4 == 3 {
                *b = 255;
            } else {
                *b = (i % 256) as u8;
            }
        }
        assert!(is_all_opaque(&rgba));

        let mut rgb = vec![0u8; 16 * 16 * 3];
        alpha_strip_pack(&rgba, &mut rgb);

        let mut recovered_rgba = vec![0u8; 16 * 16 * 4];
        alpha_strip_unpack(&rgb, &mut recovered_rgba);

        assert_eq!(rgba, recovered_rgba);
    }

    #[test]
    fn test_row_rle_roundtrip() {
        let width = 64;
        let height = 32;
        let channels = 3;
        let row_bytes = width * channels;
        let mut data = vec![0u8; height * row_bytes];

        // Fill with repeating bands of rows
        for r in 0..height {
            let val = (r / 4) as u8 * 30;
            for c in 0..row_bytes {
                data[r * row_bytes + c] = val;
            }
        }

        let (counts, unique) = row_rle_encode(&data, height, row_bytes);
        assert_eq!(counts.len(), 8); // 32 / 4 = 8 unique bands
        for &cnt in &counts {
            assert_eq!(cnt, 4);
        }

        let mut decoded = vec![0u8; height * row_bytes];
        row_rle_decode(&counts, &unique, row_bytes, &mut decoded);
        assert_eq!(data, decoded);
    }

    #[test]
    fn test_planar_roundtrip() {
        let w = 32;
        let h = 32;
        let c = 3;
        let mut src = vec![0u8; w * h * c];
        for (i, b) in src.iter_mut().enumerate() {
            *b = (i % 255) as u8;
        }

        let mut planar = vec![0u8; src.len()];
        planar_encode(&src, w, h, c, &mut planar);

        let mut out = vec![0u8; src.len()];
        planar_decode(&planar, &mut out, w, h, c);

        assert_eq!(src, out);
    }
}
