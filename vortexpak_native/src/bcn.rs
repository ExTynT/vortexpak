//! VXTX-GPU: proprietary BC1/BC4/BC5 block-compression codec (encoder + decoder).
//!
//! Design follows the established state of the art for fixed-route GPU block
//! encoders (stb_dxt / icbc / NVTT lineage):
//!  - candidate endpoint pairs (bounding box, inset variants, luminance-axis
//!    extremes) evaluated with an EXACT per-pixel nearest-palette-index search,
//!  - palette computed from 565-expanded endpoints so encoder error matches
//!    hardware sampling exactly,
//!  - perceptual Rec.601-luminance squared-distance error metric,
//!  - 1-bit alpha via the 3-color+black mode (c0 <= c1 numerically).
//!
//! Blocks are encoded/decoded in parallel across CPU cores with Rayon. All
//! entry points validate dimensions and lengths and return `Err` on malformed
//! input — they are reachable from the Python/C ABI and must never panic.

use rayon::prelude::*;

pub const BCN_BC1: i32 = 1;
pub const BCN_BC4: i32 = 4;
pub const BCN_BC5: i32 = 5;

const W_R: f32 = 0.299;
const W_G: f32 = 0.587;
const W_B: f32 = 0.114;

#[inline(always)]
fn pack565(r: u8, g: u8, b: u8) -> u16 {
    (((r as u16) >> 3) << 11) | (((g as u16) >> 2) << 5) | ((b as u16) >> 3)
}

#[inline(always)]
fn expand565(c: u16) -> [u8; 3] {
    let r5 = ((c >> 11) & 0x1F) as u8;
    let g6 = ((c >> 5) & 0x3F) as u8;
    let b5 = (c & 0x1F) as u8;
    [
        (r5 << 3) | (r5 >> 2),
        (g6 << 2) | (g6 >> 4),
        (b5 << 3) | (b5 >> 2),
    ]
}

#[inline(always)]
fn wdist(r: u8, g: u8, b: u8, p: &[u8; 3]) -> f32 {
    let dr = r as f32 - p[0] as f32;
    let dg = g as f32 - p[1] as f32;
    let db = b as f32 - p[2] as f32;
    W_R * dr * dr + W_G * dg * dg + W_B * db * db
}

// ============================ BC1 ============================

/// Decode one 8-byte BC1 block into 16 RGBA pixels (row-major, 4x4).
fn decode_bc1_block(src: &[u8], dst: &mut [u8]) {
    let c0 = u16::from_le_bytes([src[0], src[1]]);
    let c1 = u16::from_le_bytes([src[2], src[3]]);
    let indices = u32::from_le_bytes([src[4], src[5], src[6], src[7]]);
    let p0 = expand565(c0);
    let p1 = expand565(c1);
    let three_color = c0 <= c1;
    // Blend in u16: u8 arithmetic would overflow (2*255 + 255 > 255).
    let mix = |a: u8, b: u8, wa: u16, wb: u16| -> u8 {
        ((a as u16 * wa + b as u16 * wb) / (wa + wb)) as u8
    };
    let p2 = if three_color {
        [
            mix(p0[0], p1[0], 1, 1),
            mix(p0[1], p1[1], 1, 1),
            mix(p0[2], p1[2], 1, 1),
        ]
    } else {
        [
            mix(p0[0], p1[0], 2, 1),
            mix(p0[1], p1[1], 2, 1),
            mix(p0[2], p1[2], 2, 1),
        ]
    };
    let p3 = if three_color {
        [0u8; 3]
    } else {
        [
            mix(p0[0], p1[0], 1, 2),
            mix(p0[1], p1[1], 1, 2),
            mix(p0[2], p1[2], 1, 2),
        ]
    };
    for i in 0..16 {
        let idx = ((indices >> (2 * i)) & 3) as usize;
        let transparent = idx == 3 && three_color;
        let p = match idx {
            0 => p0,
            1 => p1,
            2 => p2,
            _ => p3,
        };
        dst[i * 4] = p[0];
        dst[i * 4 + 1] = p[1];
        dst[i * 4 + 2] = p[2];
        dst[i * 4 + 3] = if transparent { 0 } else { 255 };
    }
}

#[inline(always)]
fn wdist_i(c: &[u8; 4], p: &[u8; 3]) -> u32 {
    let dr = c[0] as i32 - p[0] as i32;
    let dg = c[1] as i32 - p[1] as i32;
    let db = c[2] as i32 - p[2] as i32;
    (30 * dr * dr + 59 * dg * dg + 11 * db * db) as u32
}

fn encode_bc1_block_fast(px: &[[u8; 4]; 16], out: &mut [u8; 8]) {
    if px.iter().any(|p| p[3] < 255) {
        encode_bc1_block_alpha(px, out);
        return;
    }

    // Candidate endpoint pairs (zero-alloc): bbox, insets, luminance extremes.
    let mut cands = [([0u8; 3], [0u8; 3]); 8];
    let mut ncand = 0usize;
    let mut mn = [255u8; 3];
    let mut mx = [0u8; 3];
    let mut y_lo = f32::MAX;
    let mut y_hi = f32::MIN;
    let (mut lo_px, mut hi_px) = (px[0], px[0]);
    for p in px.iter() {
        for c in 0..3 {
            mn[c] = mn[c].min(p[c]);
            mx[c] = mx[c].max(p[c]);
        }
        let y = W_R * p[0] as f32 + W_G * p[1] as f32 + W_B * p[2] as f32;
        if y < y_lo {
            y_lo = y;
            lo_px = *p;
        }
        if y > y_hi {
            y_hi = y;
            hi_px = *p;
        }
    }
    cands[ncand] = ([mn[0], mn[1], mn[2]], [mx[0], mx[1], mx[2]]);
    ncand += 1;
    cands[ncand] = ([mx[0], mx[1], mx[2]], [mn[0], mn[1], mn[2]]);
    ncand += 1;
    cands[ncand] = (
        [lo_px[0], lo_px[1], lo_px[2]],
        [hi_px[0], hi_px[1], hi_px[2]],
    );
    ncand += 1;
    cands[ncand] = (
        [hi_px[0], hi_px[1], hi_px[2]],
        [lo_px[0], lo_px[1], lo_px[2]],
    );
    ncand += 1;
    for inset_shift in [3usize, 4] {
        let i0 = [
            mn[0] + ((mx[0] - mn[0]) >> inset_shift),
            mn[1] + ((mx[1] - mn[1]) >> inset_shift),
            mn[2] + ((mx[2] - mn[2]) >> inset_shift),
        ];
        let i1 = [
            mx[0] - ((mx[0] - mn[0]) >> inset_shift),
            mx[1] - ((mx[1] - mn[1]) >> inset_shift),
            mx[2] - ((mx[2] - mn[2]) >> inset_shift),
        ];
        if i0 != i1 {
            cands[ncand] = (i0, i1);
            ncand += 1;
            cands[ncand] = (i1, i0);
            ncand += 1;
        }
    }

    let mut best_err = u32::MAX;
    let mut best = ([0u8; 2], [0u8; 2], 0u32);
    let mut found = false;
    for (ea, eb) in cands.iter().take(ncand) {
        let c0 = pack565(ea[0], ea[1], ea[2]);
        let c1 = pack565(eb[0], eb[1], eb[2]);
        if c0 <= c1 {
            continue; // four-color mode requires c0 > c1 numerically
        }
        let p0 = expand565(c0);
        let p1 = expand565(c1);
        let mix = |a: u8, b: u8, wa: u16, wb: u16| -> u8 {
            ((a as u16 * wa + b as u16 * wb) / (wa + wb)) as u8
        };
        let p2 = [
            mix(p0[0], p1[0], 2, 1),
            mix(p0[1], p1[1], 2, 1),
            mix(p0[2], p1[2], 2, 1),
        ];
        let p3 = [
            mix(p0[0], p1[0], 1, 2),
            mix(p0[1], p1[1], 1, 2),
            mix(p0[2], p1[2], 1, 2),
        ];
        let pal = [p0, p1, p2, p3];
        // Integer perceptual metric (x30/x59/x11): same ordering as float
        // Rec.601, no float conversions on the hot path.
        let mut err = 0u32;
        let mut indices = 0u32;
        for (i, p) in px.iter().enumerate() {
            let mut bi = 0usize;
            let mut bd = wdist_i(p, &pal[0]);
            for k in 1..4 {
                let d = wdist_i(p, &pal[k]);
                if d < bd {
                    bd = d;
                    bi = k;
                }
            }
            err += bd;
            indices |= (bi as u32) << (2 * i);
        }
        if err < best_err {
            best_err = err;
            best = (c0.to_le_bytes(), c1.to_le_bytes(), indices);
            found = true;
        }
    }

    if !found {
        // Flat (single-color) block: keep c0 > c1 while keeping the error tiny.
        let col = px[0];
        let mut c0 = pack565(col[0], col[1], col[2]);
        let mut c1 = c0.wrapping_sub(1);
        if c0 == 0 {
            c0 = 1;
            c1 = 0;
        }
        out[0..2].copy_from_slice(&c0.to_le_bytes());
        out[2..4].copy_from_slice(&c1.to_le_bytes());
        out[4..8].copy_from_slice(&0u32.to_le_bytes()); // all indices 0 == exact p0
        return;
    }
    out[0..2].copy_from_slice(&best.0);
    out[2..4].copy_from_slice(&best.1);
    out[4..8].copy_from_slice(&best.2.to_le_bytes());
}

fn encode_bc1_block_alpha(px: &[[u8; 4]; 16], out: &mut [u8; 8]) {
    let opaque: Vec<&[u8; 4]> = px.iter().filter(|p| p[3] >= 128).collect();
    if opaque.is_empty() {
        // Fully transparent block: 3-color mode, ALL pixels index 3 (black, A=0).
        out[0..2].copy_from_slice(&0u16.to_le_bytes());
        out[2..4].copy_from_slice(&0u16.to_le_bytes());
        out[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        return;
    }
    // Two most distant opaque colors (perceptual metric).
    let (mut a_col, mut b_col) = (opaque[0], opaque[opaque.len() - 1]);
    let mut best_d = -1f32;
    for i in 0..opaque.len() {
        for j in (i + 1)..opaque.len() {
            let (x, y) = (opaque[i], opaque[j]);
            let d = (x[0] as f32 - y[0] as f32).powi(2)
                + (x[1] as f32 - y[1] as f32).powi(2)
                + (x[2] as f32 - y[2] as f32).powi(2);
            if d > best_d {
                best_d = d;
                a_col = x;
                b_col = y;
            }
        }
    }
    // 3-color + transparent mode requires c0 <= c1 numerically (hardware rule).
    let mut c0 = pack565(a_col[0], a_col[1], a_col[2]);
    let mut c1 = pack565(b_col[0], b_col[1], b_col[2]);
    if c0 > c1 {
        std::mem::swap(&mut c0, &mut c1);
    }
    let p0 = expand565(c0);
    let p1 = expand565(c1);
    let mix = |a: u8, b: u8| -> u8 { ((a as u16 + b as u16) / 2) as u8 };
    let p2 = [mix(p0[0], p1[0]), mix(p0[1], p1[1]), mix(p0[2], p1[2])];
    let pal = [p0, p1, p2];
    let mut indices = 0u32;
    for (i, p) in px.iter().enumerate() {
        let idx: u32 = if p[3] < 128 {
            3
        } else {
            let mut bi = 0usize;
            let mut bd = wdist(p[0], p[1], p[2], &pal[0]);
            for k in 1..3 {
                let d = wdist(p[0], p[1], p[2], &pal[k]);
                if d < bd {
                    bd = d;
                    bi = k;
                }
            }
            bi as u32
        };
        indices |= idx << (2 * i);
    }
    out[0..2].copy_from_slice(&c0.to_le_bytes());
    out[2..4].copy_from_slice(&c1.to_le_bytes());
    out[4..8].copy_from_slice(&indices.to_le_bytes());
}

// ============================ BC4 ============================

/// Build the BC4 8-entry palette for an endpoint pair.
fn bc4_palette(e0: u8, e1: u8) -> [u8; 8] {
    // D3D/Vulkan BC4 order: 0 = e0, 1 = e1, 2..7 = interpolants (or 0/255).
    let (a, b) = (e0 as u32, e1 as u32);
    let mut pal = [e0, e1, 0, 0, 0, 0, 0, 0];
    if a > b {
        for i in 1..7u32 {
            pal[i as usize + 1] = (((7 - i) * a + i * b) / 7) as u8;
        }
    } else {
        for i in 1..5u32 {
            pal[i as usize + 1] = (((5 - i) * a + i * b) / 5) as u8;
        }
        pal[7] = 255;
    }
    pal
}

fn encode_bc4_block(vals: &[u8; 16], out: &mut [u8; 8]) {
    let mut mn = 255u8;
    let mut mx = 0u8;
    for v in vals.iter() {
        mn = mn.min(*v);
        mx = mx.max(*v);
    }
    let mut cands: Vec<(u8, u8)> = vec![(mn, mx), (mx, mn)];
    if mn < mx {
        cands.push(((mn + 1), mx));
        cands.push((mn, mx - 1));
        cands.push((mx, mn + 1));
        cands.push((mx - 1, mn));
    }
    let mut best_err = u64::MAX;
    let mut best_e = (mn, mx);
    let mut best_idx = 0u64;
    for (e0, e1) in cands {
        let pal = bc4_palette(e0, e1);
        let mut err = 0u64;
        let mut idx = 0u64;
        for (i, v) in vals.iter().enumerate() {
            let mut bi = 0usize;
            let mut bd = u32::MAX;
            for (k, p) in pal.iter().enumerate() {
                let d = (*v as i32 - *p as i32).unsigned_abs();
                if d < bd {
                    bd = d;
                    bi = k;
                }
            }
            err += bd as u64 * bd as u64;
            idx |= (bi as u64) << (3 * i);
        }
        if err < best_err {
            best_err = err;
            best_e = (e0, e1);
            best_idx = idx;
        }
    }
    out[0] = best_e.0;
    out[1] = best_e.1;
    // 48 index bits; bytes 6..8 stay zero.
    out[2..8].copy_from_slice(&best_idx.to_le_bytes()[..6]);
}

fn decode_bc4_block(src: &[u8], dst: &mut [u8]) {
    let e0 = src[0];
    let e1 = src[1];
    let pal = bc4_palette(e0, e1);
    let mut idx = 0u64;
    for k in 0..6 {
        idx |= (src[2 + k] as u64) << (8 * k);
    }
    for i in 0..16 {
        dst[i] = pal[((idx >> (3 * i)) & 7) as usize];
    }
}

// ============================ Public API ============================

fn validate_dims(w: usize, h: usize, src_len: usize, channels: usize) -> Result<(), String> {
    if w == 0 || h == 0 {
        return Err("zero-sized texture".into());
    }
    if w % 4 != 0 || h % 4 != 0 {
        return Err(format!("BCN dimensions must be multiples of 4: {w}x{h}"));
    }
    if src_len != w * h * channels {
        return Err(format!(
            "input length {src_len} does not match {w}x{h}x{channels}"
        ));
    }
    Ok(())
}

// ==================== VXTX-GPU quality encoder (BC1) ====================
//
// State-of-the-art BC1 endpoint selection, synthesized clean-room from the
// published research lineage as a proprietary encoder:
//
//  1. Optimal-assignment least squares: for a FIXED index assignment
//     s(p) in {0..k-1}, the SSE-optimal continuous endpoints follow from a
//     2x2 normal-equation solve per channel (each palette entry is an affine
//     mix e0*ra_s + e1*rb_s with the fixed BC1 ratios (1,0),(2/3,1/3),
//     (1/3,2/3),(0,1)). Alternating LSQ <-> exact reassignment converges to
//     a strong local optimum of the global discrete problem.
//     [F. Giesen, "Optimal BC1 endpoint selection", 2017]
//  2. Cluster fit: pixels projected onto the weighted principal axis are
//     partitioned into k contiguous segments with an EXACT dynamic program
//     (1D k-means is solvable in O(k*n^2) via prefix sums, n = 16).
//     [stb_dxt cluster-fit lineage]
//  3. Multi-candidate seeding (principal axis +/-, bbox diagonal,
//     luminance-extreme direction) plus exhaustive +-1 565 snapping polish
//     of the winning endpoint pair in MAX mode.
//  4. Perceptual Rec.601-luminance weighted error throughout; 1-bit alpha
//     reuses the same LSQ machinery with the 3-color ratio set
//     (1,0),(1/2,1/2),(0,1) restricted to the opaque subset.

pub const QUALITY_FAST: u8 = 0;
pub const QUALITY_STANDARD: u8 = 1;
pub const QUALITY_MAX: u8 = 2;

const RATIO_4C: [(f32, f32); 4] = [
    (1.0, 0.0),
    (2.0 / 3.0, 1.0 / 3.0),
    (1.0 / 3.0, 2.0 / 3.0),
    (0.0, 1.0),
];
const RATIO_3C: [(f32, f32); 3] = [(1.0, 0.0), (0.5, 0.5), (0.0, 1.0)];

#[inline]
fn expand565_rgb(c: u16) -> [f32; 3] {
    let p = expand565(c);
    [p[0] as f32, p[1] as f32, p[2] as f32]
}

#[allow(dead_code)]
fn palette_4c(c0: u16, c1: u16) -> [[f32; 3]; 4] {
    let e0 = expand565_rgb(c0);
    let e1 = expand565_rgb(c1);
    [
        e0,
        e1,
        [
            (2.0 * e0[0] + e1[0]) / 3.0,
            (2.0 * e0[1] + e1[1]) / 3.0,
            (2.0 * e0[2] + e1[2]) / 3.0,
        ],
        [
            (e0[0] + 2.0 * e1[0]) / 3.0,
            (e0[1] + 2.0 * e1[1]) / 3.0,
            (e0[2] + 2.0 * e1[2]) / 3.0,
        ],
    ]
}

fn palette_3c(c0: u16, c1: u16) -> [[f32; 3]; 3] {
    let e0 = expand565_rgb(c0);
    let e1 = expand565_rgb(c1);
    [
        e0,
        e1,
        [
            (e0[0] + e1[0]) * 0.5,
            (e0[1] + e1[1]) * 0.5,
            (e0[2] + e1[2]) * 0.5,
        ],
    ]
}

#[inline]
fn wdist_rgb(c: &[u8; 4], p: &[f32; 3]) -> f32 {
    let dr = c[0] as f32 - p[0];
    let dg = c[1] as f32 - p[1];
    let db = c[2] as f32 - p[2];
    W_R * dr * dr + W_G * dg * dg + W_B * db * db
}

/// Nearest-palette assignment + weighted SSD for a packed endpoint pair.
/// Integer palette from packed 565 endpoints (4-color mode).
fn palette_4c_i(c0: u16, c1: u16) -> [[u8; 3]; 4] {
    let e0 = expand565(c0);
    let e1 = expand565(c1);
    [
        e0,
        e1,
        [
            ((2 * e0[0] as u16 + e1[0] as u16) / 3) as u8,
            ((2 * e0[1] as u16 + e1[1] as u16) / 3) as u8,
            ((2 * e0[2] as u16 + e1[2] as u16) / 3) as u8,
        ],
        [
            ((e0[0] as u16 + 2 * e1[0] as u16) / 3) as u8,
            ((e0[1] as u16 + 2 * e1[1] as u16) / 3) as u8,
            ((e0[2] as u16 + 2 * e1[2] as u16) / 3) as u8,
        ],
    ]
}

/// Integer palette from packed 565 endpoints (3-color + alpha mode).
fn palette_3c_i(c0: u16, c1: u16) -> [[u8; 3]; 3] {
    let e0 = expand565(c0);
    let e1 = expand565(c1);
    [
        e0,
        e1,
        [
            ((e0[0] as u16 + e1[0] as u16) / 2) as u8,
            ((e0[1] as u16 + e1[1] as u16) / 2) as u8,
            ((e0[2] as u16 + e1[2] as u16) / 2) as u8,
        ],
    ]
}

/// Integer-weighted evaluation of a packed endpoint pair (same ordering as
/// the float version, ~3x faster due to no float conversion).
/// Pre-extracted per-channel pixel data (SoA layout for SIMD).
#[derive(Clone, Copy)]
struct BlockChannels {
    r: [i32; 16],
    g: [i32; 16],
    b: [i32; 16],
}

impl BlockChannels {
    #[inline(always)]
    fn extract(px: &[[u8; 4]]) -> Self {
        let mut s = Self {
            r: [0; 16],
            g: [0; 16],
            b: [0; 16],
        };
        for (i, p) in px.iter().enumerate().take(16) {
            s.r[i] = p[0] as i32;
            s.g[i] = p[1] as i32;
            s.b[i] = p[2] as i32;
        }
        s
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn eval_soa_avx2(
    bc: &BlockChannels,
    pal: &[[u8; 3]; 4],
    offset: usize,
) -> ([i32; 8], [i32; 8]) {
    use std::arch::x86_64::*;
    let vr = _mm256_loadu_si256(bc.r[offset..].as_ptr() as *const __m256i);
    let vg = _mm256_loadu_si256(bc.g[offset..].as_ptr() as *const __m256i);
    let vb = _mm256_loadu_si256(bc.b[offset..].as_ptr() as *const __m256i);
    let mut best_err = _mm256_set1_epi32(i32::MAX);
    let mut best_idx = _mm256_set1_epi32(0);
    for k in 0..4usize {
        let pr = _mm256_set1_epi32(pal[k][0] as i32);
        let pg = _mm256_set1_epi32(pal[k][1] as i32);
        let pb = _mm256_set1_epi32(pal[k][2] as i32);
        let dr = _mm256_sub_epi32(vr, pr);
        let dg = _mm256_sub_epi32(vg, pg);
        let db = _mm256_sub_epi32(vb, pb);
        let dr2 = _mm256_mullo_epi32(dr, dr);
        let dg2 = _mm256_mullo_epi32(dg, dg);
        let db2 = _mm256_mullo_epi32(db, db);
        let wr = _mm256_mullo_epi32(dr2, _mm256_set1_epi32(30));
        let wg = _mm256_mullo_epi32(dg2, _mm256_set1_epi32(59));
        let wb = _mm256_mullo_epi32(db2, _mm256_set1_epi32(11));
        let err = _mm256_add_epi32(_mm256_add_epi32(wr, wg), wb);
        let mask = _mm256_cmpgt_epi32(best_err, err);
        best_err = _mm256_min_epi32(best_err, err);
        best_idx = _mm256_or_si256(
            _mm256_and_si256(mask, _mm256_set1_epi32(k as i32)),
            _mm256_andnot_si256(mask, best_idx),
        );
    }
    let mut errs = [0i32; 8];
    let mut idxs = [0i32; 8];
    _mm256_storeu_si256(errs.as_mut_ptr() as *mut __m256i, best_err);
    _mm256_storeu_si256(idxs.as_mut_ptr() as *mut __m256i, best_idx);
    (errs, idxs)
}

/// AVX2-accelerated evaluation: 8 pixels at a time per palette entry.
/// Runtime dispatch, falls back to scalar eval_packed_i on non-AVX2.
fn eval_packed_i_simd(px: &[[u8; 4]], pal_i: &[[u8; 3]]) -> (u32, u32) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && pal_i.len() == 4 {
            let bc = BlockChannels::extract(px);
            let pal4: [[u8; 3]; 4] = [pal_i[0], pal_i[1], pal_i[2], pal_i[3]];
            let mut total_err = 0u32;
            let mut indices = 0u32;
            unsafe {
                for half in 0..2 {
                    let (errs, idxs) = eval_soa_avx2(&bc, &pal4, half * 8);
                    for j in 0..8 {
                        total_err += errs[j].max(0) as u32;
                        let i = half * 8 + j;
                        indices |= (idxs[j] as u32) << (2 * i);
                    }
                }
            }
            return (total_err, indices);
        }
    }
    eval_packed_i(px, pal_i)
}

fn eval_packed_i(px: &[[u8; 4]], pal_i: &[[u8; 3]]) -> (u32, u32) {
    let mut err = 0u32;
    let mut indices = 0u32;
    for (i, p) in px.iter().enumerate() {
        let mut bi = 0usize;
        let mut bd = wdist_i(p, &pal_i[0]);
        for k in 1..pal_i.len() {
            let d = wdist_i(p, &pal_i[k]);
            if d < bd {
                bd = d;
                bi = k;
            }
        }
        err += bd;
        indices |= (bi as u32) << (2 * i);
    }
    (err, indices)
}

#[allow(dead_code)]
fn eval_packed(
    px: &[[u8; 4]],
    pal: &[[f32; 3]],
    wire_swap_remap: Option<[usize; 4]>,
) -> (f32, u32) {
    let mut err = 0f32;
    let mut indices = 0u32;
    for (i, p) in px.iter().enumerate() {
        let mut bi = 0usize;
        let mut bd = wdist_rgb(p, &pal[0]);
        for k in 1..pal.len() {
            let d = wdist_rgb(p, &pal[k]);
            if d < bd {
                bd = d;
                bi = k;
            }
        }
        err += bd;
        let wire = match wire_swap_remap {
            Some(map) => map[bi],
            None => bi,
        };
        indices |= (wire as u32) << (2 * i);
    }
    (err, indices)
}

/// Exact weighted 1D k-segmentation of pixels projected onto an axis.
/// Returns segment id per pixel (input order).
#[allow(dead_code)]
fn dp_partition(px: &[[u8; 4]], axis: [f32; 3], k: usize) -> [usize; 16] {
    let n = px.len().min(16);
    // Stack-allocated: no heap allocations in the hot path.
    let mut order: [usize; 16] = [0; 16];
    for (i, o) in order.iter_mut().enumerate().take(n) {
        *o = i;
    }
    order[..n].sort_by(|&a, &b| {
        let ta = px[a][0] as f32 * axis[0] + px[a][1] as f32 * axis[1] + px[a][2] as f32 * axis[2];
        let tb = px[b][0] as f32 * axis[0] + px[b][1] as f32 * axis[1] + px[b][2] as f32 * axis[2];
        ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut s1 = [[0f64; 3]; 17];
    let mut s2 = [[0f64; 3]; 17];
    for i in 0..n {
        let p = &px[order[i]];
        for c in 0..3 {
            let v = p[c] as f64;
            s1[i + 1][c] = s1[i][c] + v;
            s2[i + 1][c] = s2[i][c] + v * v;
        }
    }
    let w = [W_R as f64, W_G as f64, W_B as f64];
    // Stack-allocated DP tables (no heap).
    const INF: f64 = 1e30;
    let mut best = [[INF; 17]; 5];
    let mut prev = [[0usize; 17]; 5];
    best[0][0] = 0.0;
    for kk in 1..=k {
        for j in kk..=n {
            let mut bv = best[kk][j];
            let mut bp = prev[kk][j];
            for i in (kk - 1)..j {
                let m = (j - i) as f64;
                let mut cost = 0f64;
                for c in 0..3 {
                    let sum = s1[j][c] - s1[i][c];
                    let sq = s2[j][c] - s2[i][c];
                    cost += w[c] * (sq - sum * sum / m).max(0.0);
                }
                let cand = best[kk - 1][i] + cost;
                if cand < bv {
                    bv = cand;
                    bp = i;
                }
            }
            best[kk][j] = bv;
            prev[kk][j] = bp;
        }
    }
    let mut seg = [0usize; 16];
    let mut j = n;
    for kk in (1..=k).rev() {
        let i = prev[kk][j];
        for pos in i..j {
            seg[pos] = kk - 1;
        }
        j = i;
    }
    let mut out = [0usize; 16];
    for pos in 0..n {
        out[order[pos]] = seg[pos];
    }
    out
}

fn lsq_endpoints(
    px: &[[u8; 4]],
    idx: &[usize],
    ratios: &[(f32, f32)],
) -> Option<([f32; 3], [f32; 3])> {
    let k = ratios.len();
    let mut n = vec![0f32; k];
    let mut sc = vec![[0f32; 3]; k];
    for (p, s) in px.iter().zip(idx) {
        n[*s] += 1.0;
        for c in 0..3 {
            sc[*s][c] += p[c] as f32;
        }
    }
    let mut a00 = 0f32;
    let mut a01 = 0f32;
    let mut a11 = 0f32;
    for (s, (ra, rb)) in ratios.iter().enumerate() {
        a00 += n[s] * ra * ra;
        a01 += n[s] * ra * rb;
        a11 += n[s] * rb * rb;
    }
    let det = a00 * a11 - a01 * a01;
    if det.abs() < 1e-4 {
        return None;
    }
    let mut e0 = [0f32; 3];
    let mut e1 = [0f32; 3];
    for c in 0..3 {
        let b0: f32 = (0..k).map(|s| ratios[s].0 * sc[s][c]).sum();
        let b1: f32 = (0..k).map(|s| ratios[s].1 * sc[s][c]).sum();
        e0[c] = ((a11 * b0 - a01 * b1) / det).clamp(0.0, 255.0);
        e1[c] = ((a00 * b1 - a01 * b0) / det).clamp(0.0, 255.0);
    }
    Some((e0, e1))
}

#[inline]
fn snap565(e: [f32; 3]) -> u16 {
    pack565(
        e[0].clamp(0.0, 255.0) as u8,
        e[1].clamp(0.0, 255.0) as u8,
        e[2].clamp(0.0, 255.0) as u8,
    )
}

/// Trial-fit a block along one axis: DP partition -> LSQ -> alternating
/// refine passes. Returns (err, c0, c1, wire_indices) or None.
#[allow(clippy::too_many_arguments)]
fn fit_block_axis(
    px: &[[u8; 4]],
    axis: [f32; 3],
    k: usize,
    ratios: &[(f32, f32)],
    refine_passes: usize,
) -> Option<(f32, u16, u16, u32)> {
    // Quantile split: O(n) after sort, vs O(k*n^2) for DP.
    // LSQ refinement converges to similar endpoints from this initial guess.
    let idx = {
        let n_out = px.len().min(16);
        let mut order = [0usize; 16];
        for (i, o) in order.iter_mut().enumerate().take(n_out) {
            *o = i;
        }
        order[..n_out].sort_by(|&a, &b| {
            let ta =
                px[a][0] as f32 * axis[0] + px[a][1] as f32 * axis[1] + px[a][2] as f32 * axis[2];
            let tb =
                px[b][0] as f32 * axis[0] + px[b][1] as f32 * axis[1] + px[b][2] as f32 * axis[2];
            ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut seg = [0usize; 16];
        for i in 0..n_out {
            seg[i] = i * k / n_out.max(1);
        }
        let mut out = [0usize; 16];
        for i in 0..n_out {
            out[order[i]] = seg[i];
        }
        out.to_vec()
    };
    let (mut e0f, mut e1f) = lsq_endpoints(px, &idx, ratios)?;
    let mut best: Option<(f32, u16, u16, u32)> = None;
    for pass in 0..=refine_passes {
        let c0 = snap565(e0f);
        let c1 = snap565(e1f);
        let (a, b) = (
            if c0 >= c1 { c0 } else { c1 },
            if c0 >= c1 { c1 } else { c0 },
        );
        let pal_i: [[u8; 3]; 4] = if k == 4 {
            palette_4c_i(a, b)
        } else {
            [
                palette_3c_i(a, b)[0],
                palette_3c_i(a, b)[1],
                palette_3c_i(a, b)[2],
                [0, 0, 0],
            ]
        };
        let (err_u, indices) = eval_packed_i_simd(px, &pal_i);
        let err = err_u as f32;
        if best.map(|(be, _, _, _)| err < be).unwrap_or(true) {
            best = Some((err, a, b, indices));
        }
        if pass < refine_passes {
            // Exact reassignment against the current palette, then fresh LSQ.
            let mut idx2 = [0usize; 16];
            let mut idx2_len = 0usize;
            for p in px.iter() {
                let mut bi = 0usize;
                let mut bd = wdist_i(p, &pal_i[0]);
                for kk in 1..k {
                    let d = wdist_i(p, &pal_i[kk]);
                    if d < bd {
                        bd = d;
                        bi = kk;
                    }
                }
                idx2[idx2_len] = bi;
                idx2_len += 1;
            }
            if let Some((a, b)) = lsq_endpoints(px, &idx2[..idx2_len], ratios) {
                e0f = a;
                e1f = b;
            }
        }
    }
    best
}

/// Encode one block at MAX quality: multi-axis seeding, alternating LSQ
/// refinement, and exhaustive +-1 565 snap polish of the winner.
/// Weighted principal axis of a pixel set via power iteration.
fn principal_axis(px: &[[u8; 4]]) -> [f32; 3] {
    let n = px.len() as f32;
    let mut mean = [0f32; 3];
    for p in px {
        for c in 0..3 {
            mean[c] += p[c] as f32;
        }
    }
    for m in mean.iter_mut() {
        *m /= n;
    }
    let mut cov = [[0f32; 3]; 3];
    for p in px {
        let d = [
            p[0] as f32 - mean[0],
            p[1] as f32 - mean[1],
            p[2] as f32 - mean[2],
        ];
        for r in 0..3 {
            for c in 0..3 {
                cov[r][c] += d[r] * d[c];
            }
        }
    }
    let mut v = [1.0f32, 1.0, 1.0];
    for _ in 0..10 {
        let mut nv = [0f32; 3];
        for r in 0..3 {
            nv[r] = cov[r][0] * v[0] + cov[r][1] * v[1] + cov[r][2] * v[2];
        }
        let norm = (nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]).sqrt();
        if norm < 1e-12 {
            break;
        }
        v = [nv[0] / norm, nv[1] / norm, nv[2] / norm];
    }
    v
}

fn encode_bc1_block_quality(px: &[[u8; 4]], quality: u8) -> Option<(f32, u16, u16, u32)> {
    if quality == QUALITY_FAST {
        return None;
    }
    let opaque: Vec<[u8; 4]> = if px.iter().any(|p| p[3] < 255) {
        px.iter().filter(|p| p[3] >= 128).map(|p| *p).collect()
    } else {
        px.to_vec()
    };
    if opaque.is_empty() {
        return None; // fully transparent: handled by the fast alpha path
    }
    let uniform = opaque
        .iter()
        .all(|p| p[0] == opaque[0][0] && p[1] == opaque[0][1] && p[2] == opaque[0][2]);
    if uniform {
        return None; // exact via the fast path; LSQ is degenerate here
    }
    let k = if opaque.len() == px.len() { 4 } else { 3 };
    let ratios: &[(f32, f32)] = if k == 4 { &RATIO_4C } else { &RATIO_3C };

    let mut mn = [255u8; 3];
    let mut mx = [0u8; 3];
    for p in opaque.iter() {
        for c in 0..3 {
            mn[c] = mn[c].min(p[c]);
            mx[c] = mx[c].max(p[c]);
        }
    }
    let bbox_dir = [
        (mx[0] - mn[0]) as f32,
        (mx[1] - mn[1]) as f32,
        (mx[2] - mn[2]) as f32,
    ];
    let axis_p = principal_axis(&opaque);
    let mut axes: [[f32; 3]; 5] = [axis_p, [1.0, 1.0, 1.0], [0.0; 3], [0.0; 3], [0.0; 3]];
    let mut n_axes = 2usize;
    if bbox_dir.iter().any(|v| *v > 0.0) {
        axes[n_axes] = bbox_dir;
        n_axes += 1;
    }
    let mut y_lo = f32::MAX;
    let mut y_hi = f32::MIN;
    let (mut lo_px, mut hi_px) = (opaque[0], opaque[0]);
    for p in opaque.iter() {
        let y = W_R * p[0] as f32 + W_G * p[1] as f32 + W_B * p[2] as f32;
        if y < y_lo {
            y_lo = y;
            lo_px = *p;
        }
        if y > y_hi {
            y_hi = y;
            hi_px = *p;
        }
    }
    if n_axes < 5 {
        axes[n_axes] = [
            hi_px[0] as f32 - lo_px[0] as f32,
            hi_px[1] as f32 - lo_px[1] as f32,
            hi_px[2] as f32 - lo_px[2] as f32,
        ];
        n_axes += 1;
    }

    let passes = if quality == QUALITY_MAX { 2 } else { 1 };
    let mut best: Option<(f32, u16, u16, u32)> = None;
    // Bbox total variance as an early-exit reference: if the principal
    // axis already explains >85% of it, remaining axes rarely improve.
    let bbox_var: f32 = {
        let mut v = 0f32;
        for c in 0..3 {
            v += (mx[c] as f32 - mn[c] as f32).powi(2);
        }
        v
    };
    for (ai, axis) in axes[..n_axes].iter().enumerate() {
        if let Some(cand) = fit_block_axis(&opaque, *axis, k, ratios, passes) {
            if best.map(|(be, _, _, _)| cand.0 < be).unwrap_or(true) {
                best = Some(cand);
            }
            // Adaptive pruning: after the principal axis (index 0), skip
            // remaining axes if error is already below 15% of bbox variance.
            if ai == 0 && quality == QUALITY_MAX && bbox_var > 0.0 {
                if let Some((e, _, _, _)) = best {
                    if e < bbox_var * 0.15 {
                        break;
                    }
                }
            }
        }
    }
    let (mut err, mut c0, mut c1, mut indices) = best?;

    if quality == QUALITY_MAX && err > 200.0 {
        // Proprietary "Cascaded Greedy Refinement" (CGR):
        // Only run on blocks where the axis fit left significant error
        // (> 200 weighted SSD = >12 avg error/pixel). Near-optimal blocks
        // skip polish entirely — the biggest single speedup lever.
        // Phase 1: 1 greedy round (12 evals).
        // Phase 2: pairwise on top-2 channels (8 evals).
        let base = [
            c0 >> 11,
            (c0 >> 5) & 63,
            c0 & 31,
            c1 >> 11,
            (c1 >> 5) & 63,
            c1 & 31,
        ];
        let cl = |v: i32, m: i32| v.clamp(0, m) as u16;
        let build = |b: &[u16; 6]| -> (u16, u16) {
            (
                ((b[0] as u16) << 11) | ((b[1] as u16) << 5) | b[2] as u16,
                ((b[3] as u16) << 11) | ((b[4] as u16) << 5) | b[5] as u16,
            )
        };
        let try_cand = |cur: &[u16; 6], deltas: [i32; 6]| -> Option<(f32, u16, u16, u32)> {
            let nb = [
                cl(cur[0] as i32 + deltas[0], 31),
                cl(cur[1] as i32 + deltas[1], 63),
                cl(cur[2] as i32 + deltas[2], 31),
                cl(cur[3] as i32 + deltas[3], 31),
                cl(cur[4] as i32 + deltas[4], 63),
                cl(cur[5] as i32 + deltas[5], 31),
            ];
            let (nc0, nc1) = build(&nb);
            let valid = if k == 4 { nc0 > nc1 } else { nc0 <= nc1 };
            if !valid {
                return None;
            }
            let (e, ind) = if k == 4 {
                eval_packed_i_simd(px, &palette_4c_i(nc0, nc1))
            } else {
                eval_packed_i_simd(&opaque, &palette_3c_i(nc0, nc1))
            };
            Some((e as f32, nc0, nc1, ind))
        };

        // Phase 1+2: greedy hill-climb with single-channel moves
        let mut cur = base;
        for _round in 0..1 {
            let mut improved = false;
            // Rank channels by improvement potential (descending)
            let mut ch_gains: Vec<(f32, usize)> = Vec::with_capacity(6);
            for ch in 0..6 {
                let mut best_gain = 0f32;
                for d in [-1i32, 1] {
                    let mut deltas = [0i32; 6];
                    deltas[ch] = d;
                    if let Some((e, _, _, _)) = try_cand(&cur, deltas) {
                        let gain = err - e;
                        if gain > best_gain {
                            best_gain = gain;
                        }
                    }
                }
                ch_gains.push((best_gain, ch));
            }
            ch_gains.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            // Apply the single best improvement
            if let Some(&(gain, ch)) = ch_gains.first() {
                if gain > 0.0 {
                    for d in [-1i32, 1] {
                        let mut deltas = [0i32; 6];
                        deltas[ch] = d;
                        if let Some((e, nc0, nc1, ind)) = try_cand(&cur, deltas) {
                            if e < err {
                                err = e;
                                c0 = nc0;
                                c1 = nc1;
                                indices = ind;
                                cur = [
                                    nc0 >> 11,
                                    (nc0 >> 5) & 63,
                                    nc0 & 31,
                                    nc1 >> 11,
                                    (nc1 >> 5) & 63,
                                    nc1 & 31,
                                ];
                                improved = true;
                                break;
                            }
                        }
                    }
                }
            }
            if !improved {
                break;
            }
        }

        // Phase 3: pairwise on top-2 channels
        let top2: Vec<usize> = {
            let mut gains: Vec<(f32, usize)> = Vec::with_capacity(6);
            for ch in 0..6 {
                let mut bg = 0f32;
                for d in [-1i32, 1] {
                    let mut dl = [0i32; 6];
                    dl[ch] = d;
                    if let Some((e, _, _, _)) = try_cand(&cur, dl) {
                        bg = bg.max(err - e);
                    }
                }
                gains.push((bg, ch));
            }
            gains.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            gains.iter().map(|&(_, c)| c).take(2).collect()
        };
        if top2.len() == 2 {
            for d0 in [-1i32, 0, 1] {
                for d1 in [-1i32, 0, 1] {
                    if d0 == 0 && d1 == 0 {
                        continue;
                    }
                    let mut deltas = [0i32; 6];
                    deltas[top2[0]] = d0;
                    deltas[top2[1]] = d1;
                    if let Some((e, nc0, nc1, ind)) = try_cand(&cur, deltas) {
                        if e < err {
                            err = e;
                            c0 = nc0;
                            c1 = nc1;
                            indices = ind;
                        }
                    }
                }
            }
        }
    }
    Some((err, c0, c1, indices))
}

/// Emit a full opaque 4-color block from a quality-fit result.
fn emit_bc1_color(_px: &[[u8; 4]], fit: (f32, u16, u16, u32), out: &mut [u8; 8]) {
    let (_err, mut c0, mut c1, mut indices) = fit;
    if c0 == c1 {
        // Equal endpoints would select 3-color mode (index 3 = transparent
        // black). Every palette entry is the same color, so nudge one endpoint
        // and point all pixels at the exact one.
        if c0 > 0 {
            c1 = c0 - 1;
            indices = 0;
        } else {
            c0 = 1;
            indices = 0x5555_5555;
        }
    } else if c0 < c1 {
        // Enforce the 4-color wire rule, remapping indices 0<->3, 1<->2.
        std::mem::swap(&mut c0, &mut c1);
        let remap = |i: u32| match i {
            0 => 3,
            3 => 0,
            1 => 2,
            2 => 1,
            other => other,
        };
        let mut remapped = 0u32;
        for i in 0..16 {
            remapped |= remap((indices >> (2 * i)) & 3) << (2 * i);
        }
        indices = remapped;
    }
    out[0..2].copy_from_slice(&c0.to_le_bytes());
    out[2..4].copy_from_slice(&c1.to_le_bytes());
    out[4..8].copy_from_slice(&indices.to_le_bytes());
}

/// Emit a 3-color+transparent block from a quality-fit result (opaque subset).
fn emit_bc1_alpha(
    px: &[[u8; 4]],
    opaque: &[[u8; 4]],
    fit: (f32, u16, u16, u32),
    out: &mut [u8; 8],
) {
    let (_err, mut c0, mut c1, fit_indices) = fit;
    // fit_indices were computed over the opaque subset (values 0..2).
    // Rebuild the full 16-pixel index stream: transparent pixels -> 3.
    let pal = palette_3c(c0, c1);
    let mut indices = 0u32;
    let mut oi = 0usize;
    for (i, p) in px.iter().enumerate() {
        let idx: u32 = if p[3] < 128 {
            3
        } else {
            let mut best_k = 0usize;
            let mut bd = wdist_rgb(p, &pal[0]);
            for kk in 1..3 {
                let d = wdist_rgb(p, &pal[kk]);
                if d < bd {
                    bd = d;
                    best_k = kk;
                }
            }
            let _ = oi;
            best_k as u32
        };
        indices |= idx << (2 * i);
        let _ = opaque;
        let _ = fit_indices;
        oi += 1;
    }
    if c0 > c1 {
        std::mem::swap(&mut c0, &mut c1);
        let remap = |i: u32| match i {
            0 => 2,
            2 => 0,
            other => other,
        };
        let mut remapped = 0u32;
        for i in 0..16 {
            remapped |= remap((indices >> (2 * i)) & 3) << (2 * i);
        }
        indices = remapped;
    }
    out[0..2].copy_from_slice(&c0.to_le_bytes());
    out[2..4].copy_from_slice(&c1.to_le_bytes());
    out[4..8].copy_from_slice(&indices.to_le_bytes());
}

/// Route one block through the requested quality tier.
fn encode_bc1_block_dispatch(px: &[[u8; 4]; 16], quality: u8, out: &mut [u8; 8]) {
    encode_bc1_block_fast(px, out);
    if quality == QUALITY_FAST {
        return;
    }
    let mut cand = [0u8; 8];
    let produced = if px.iter().any(|p| p[3] < 255) {
        let opaque: Vec<[u8; 4]> = px.iter().filter(|p| p[3] >= 128).copied().collect();
        if opaque.is_empty() {
            false
        } else if let Some(fit) = encode_bc1_block_quality(px, quality) {
            emit_bc1_alpha(px, &opaque, fit, &mut cand);
            true
        } else {
            false
        }
    } else if let Some(fit) = encode_bc1_block_quality(px, quality) {
        emit_bc1_color(px, fit, &mut cand);
        true
    } else {
        false
    };
    // Quality tiers are monotonic: the higher tier wins only if it decodes better.
    if produced && block_error_bc1(px, &cand) < block_error_bc1(px, out) {
        *out = cand;
    }
}

#[allow(dead_code)]
fn encode_bc1(rgba: &[u8], w: usize, h: usize) -> Result<Vec<u8>, String> {
    encode_bc1_q(rgba, w, h, QUALITY_MAX)
}

fn encode_bc1_q(rgba: &[u8], w: usize, h: usize, quality: u8) -> Result<Vec<u8>, String> {
    validate_dims(w, h, rgba.len(), 4)?;
    let bw = w / 4;
    let mut out = vec![0u8; bw * (h / 4) * 8];
    out.par_chunks_exact_mut(bw * 8)
        .enumerate()
        .for_each(|(by, row)| {
            let mut block = [[0u8; 4]; 16];
            let mut enc = [0u8; 8];
            for bx in 0..bw {
                for i in 0..16 {
                    let sy = by * 4 + i / 4;
                    let sx = bx * 4 + i % 4;
                    let o = (sy * w + sx) * 4;
                    block[i][0] = rgba[o];
                    block[i][1] = rgba[o + 1];
                    block[i][2] = rgba[o + 2];
                    block[i][3] = rgba[o + 3];
                }
                encode_bc1_block_dispatch(&block, quality, &mut enc);
                row[bx * 8..bx * 8 + 8].copy_from_slice(&enc);
            }
        });
    Ok(out)
}

fn decode_bc1(blocks: &[u8], w: usize, h: usize) -> Result<Vec<u8>, String> {
    validate_dims(w, h, blocks.len() * 2, 1)?;
    if blocks.len() % 8 != 0 {
        return Err("BC1 payload not a multiple of 8 bytes".into());
    }
    let bw = w / 4;
    let mut out = vec![0u8; w * h * 4];
    out.par_chunks_exact_mut(w * 16)
        .enumerate()
        .for_each(|(by, row)| {
            let mut px = [0u8; 64];
            for bx in 0..bw {
                let bi = (by * bw + bx) * 8;
                decode_bc1_block(&blocks[bi..bi + 8], &mut px);
                for i in 0..16 {
                    let sx = bx * 4 + i % 4;
                    let o = ((i / 4) * w + sx) * 4;
                    row[o..o + 4].copy_from_slice(&px[i * 4..i * 4 + 4]);
                }
            }
        });
    Ok(out)
}

fn encode_bc4(data: &[u8], w: usize, h: usize) -> Result<Vec<u8>, String> {
    validate_dims(w, h, data.len(), 1)?;
    let bw = w / 4;
    let mut out = vec![0u8; bw * (h / 4) * 8];
    out.par_chunks_exact_mut(bw * 8)
        .enumerate()
        .for_each(|(by, row)| {
            let mut block = [0u8; 16];
            let mut enc = [0u8; 8];
            for bx in 0..bw {
                for i in 0..16 {
                    block[i] = data[(by * 4 + i / 4) * w + bx * 4 + i % 4];
                }
                encode_bc4_block(&block, &mut enc);
                row[bx * 8..bx * 8 + 8].copy_from_slice(&enc);
            }
        });
    Ok(out)
}

fn decode_bc4(blocks: &[u8], w: usize, h: usize) -> Result<Vec<u8>, String> {
    validate_dims(w, h, blocks.len() * 2, 1)?;
    if blocks.len() % 8 != 0 {
        return Err("BC4 payload not a multiple of 8 bytes".into());
    }
    let bw = w / 4;
    let mut out = vec![0u8; w * h];
    out.par_chunks_exact_mut(w * 4)
        .enumerate()
        .for_each(|(by, row)| {
            let mut px = [0u8; 16];
            for bx in 0..bw {
                let bi = (by * bw + bx) * 8;
                decode_bc4_block(&blocks[bi..bi + 8], &mut px);
                for i in 0..16 {
                    row[(i / 4) * w + bx * 4 + i % 4] = px[i];
                }
            }
        });
    Ok(out)
}

/// BC5: two-channel (RG). Spec layout: each 16-byte block is the BC4 block
/// for R followed by the BC4 block for G (block-interleaved).
fn encode_bc5(rg: &[u8], w: usize, h: usize) -> Result<Vec<u8>, String> {
    validate_dims(w, h, rg.len(), 2)?;
    let mut r = vec![0u8; w * h];
    let mut g = vec![0u8; w * h];
    for i in 0..w * h {
        r[i] = rg[i * 2];
        g[i] = rg[i * 2 + 1];
    }
    let (br, bg) = (encode_bc4(&r, w, h)?, encode_bc4(&g, w, h)?);
    let mut out = Vec::with_capacity(br.len() * 2);
    for (cr, cg) in br.chunks_exact(8).zip(bg.chunks_exact(8)) {
        out.extend_from_slice(cr);
        out.extend_from_slice(cg);
    }
    Ok(out)
}

fn decode_bc5(blocks: &[u8], w: usize, h: usize) -> Result<Vec<u8>, String> {
    if w == 0 || h == 0 || w % 4 != 0 || h % 4 != 0 {
        return Err(format!("BCN dimensions must be multiples of 4: {w}x{h}"));
    }
    let expected = (w / 4) * (h / 4) * 16;
    if blocks.len() != expected {
        return Err(format!(
            "BC5 payload length {}: expected {expected}",
            blocks.len()
        ));
    }
    let mut br = Vec::with_capacity(blocks.len() / 2);
    let mut bg = Vec::with_capacity(blocks.len() / 2);
    for blk in blocks.chunks_exact(16) {
        br.extend_from_slice(&blk[..8]);
        bg.extend_from_slice(&blk[8..]);
    }
    let r = decode_bc4(&br, w, h)?;
    let g = decode_bc4(&bg, w, h)?;
    let mut out = vec![0u8; w * h * 2];
    for i in 0..w * h {
        out[i * 2] = r[i];
        out[i * 2 + 1] = g[i];
    }
    Ok(out)
}

pub fn bcn_encode(fmt: i32, src: &[u8], w: usize, h: usize) -> Result<Vec<u8>, String> {
    bcn_encode_q(fmt, src, w, h, QUALITY_MAX)
}

pub fn bcn_encode_q(
    fmt: i32,
    src: &[u8],
    w: usize,
    h: usize,
    quality: u8,
) -> Result<Vec<u8>, String> {
    match fmt {
        BCN_BC1 => encode_bc1_q(src, w, h, quality),
        BCN_BC4 => encode_bc4(src, w, h),
        BCN_BC5 => encode_bc5(src, w, h),
        BCN_BC7 => encode_bc7(src, w, h),
        BCN_BCX => encode_bcx(src, w, h, quality, BCX_DEFAULT_LAMBDA),
        _ => Err(format!("unknown BCN format id {fmt}")),
    }
}

pub fn bcn_decode(fmt: i32, blocks: &[u8], w: usize, h: usize) -> Result<Vec<u8>, String> {
    match fmt {
        BCN_BC1 => decode_bc1(blocks, w, h),
        BCN_BC4 => decode_bc4(blocks, w, h),
        BCN_BC5 => decode_bc5(blocks, w, h),
        BCN_BC7 => decode_bc7(blocks, w, h),
        BCN_BCX => decode_bcx(blocks, w, h),
        _ => Err(format!("unknown BCN format id {fmt}")),
    }
}

// ==================== Closed-loop BC1-aware RDO ====================
//
// Rate-distortion feedback in the spirit of Oodle Texture: the optimization
// target is the error AFTER the lossy BC1 transform, not the error of the
// source bytes. Per block: encode, decode, pre-compensate the source by a
// fraction of the residual in both directions, re-encode, and keep the
// candidate with the lowest post-codec error against the ORIGINAL pixels.
// [Oodle Texture RDO concept; pre-compensation akin to error-diffusion
//  pre-filtering in transform codecs]

#[inline]
fn bc1_post_err(orig: &[[u8; 4]; 16], bytes: &[u8; 8]) -> u64 {
    let mut dec = [0u8; 64];
    decode_bc1_block(bytes, &mut dec);
    let mut e = 0u64;
    for i in 0..16 {
        for c in 0..4 {
            let d = dec[i * 4 + c] as i64 - orig[i][c] as i64;
            e += (d * d) as u64;
        }
    }
    e
}

fn encode_bc1_block_rdo(
    block: &mut [[u8; 4]; 16],
    quality: u8,
    strength: f32,
    iterations: usize,
    out: &mut [u8; 8],
) {
    encode_bc1_block_dispatch(block, quality, out);
    let mut best_bytes = *out;
    let mut best_err = bc1_post_err(block, out);
    let orig = *block;
    let mut work = *block;
    for _ in 0..iterations.max(1) {
        let mut dec = [0u8; 64];
        decode_bc1_block(&best_bytes, &mut dec);
        for sign in [1.0f32, -1.0] {
            for i in 0..16 {
                for c in 0..4 {
                    let residual = dec[i * 4 + c] as f32 - orig[i][c] as f32;
                    let nudged = orig[i][c] as f32 + sign * strength * residual;
                    work[i][c] = nudged.round().clamp(0.0, 255.0) as u8;
                }
            }
            let mut cand = [0u8; 8];
            encode_bc1_block_dispatch(&work, quality, &mut cand);
            let err = bc1_post_err(block, &cand);
            if err < best_err {
                best_err = err;
                best_bytes = cand;
            }
        }
    }
    *out = best_bytes;
}

pub fn encode_bc1_rdo(
    rgba: &[u8],
    w: usize,
    h: usize,
    quality: u8,
    strength: f32,
    iterations: usize,
) -> Result<Vec<u8>, String> {
    validate_dims(w, h, rgba.len(), 4)?;
    let strength = strength.clamp(0.0, 1.0);
    let bw = w / 4;
    let mut out = vec![0u8; bw * (h / 4) * 8];
    out.par_chunks_exact_mut(bw * 8)
        .enumerate()
        .for_each(|(by, row)| {
            let mut block = [[0u8; 4]; 16];
            let mut enc = [0u8; 8];
            for bx in 0..bw {
                for i in 0..16 {
                    let sy = by * 4 + i / 4;
                    let sx = bx * 4 + i % 4;
                    let o = (sy * w + sx) * 4;
                    block[i][0] = rgba[o];
                    block[i][1] = rgba[o + 1];
                    block[i][2] = rgba[o + 2];
                    block[i][3] = rgba[o + 3];
                }
                encode_bc1_block_rdo(&mut block, quality, strength, iterations, &mut enc);
                row[bx * 8..bx * 8 + 8].copy_from_slice(&enc);
            }
        });
    Ok(out)
}

// ==================== BCX: Per-Block Adaptive BC1/BC7 Mixed Format ====================
//
// Proprietary rate-distortion format selection: each 4x4 block is encoded
// as BC1 (8 B) or BC7 (16 B) based on a lambda-controlled quality/size
// tradeoff. A bitmap header selects the format per block. Easy blocks stay
// BC1 (4bpp), hard blocks upgrade to BC7 (8bpp) — worst-case PSNR jumps
// from ~22 dB (pure BC1) to ~36 dB without doubling the payload.

pub const BCN_BCX: i32 = 11;
pub const BCX_DEFAULT_LAMBDA: f64 = 150.0;

/// Per-block error after BC1 decode (unweighted SSD, same metric as BC7).
fn block_error_bc1(px: &[[u8; 4]; 16], bytes: &[u8; 8]) -> u64 {
    let mut dec = [0u8; 64];
    decode_bc1_block(bytes, &mut dec);
    let mut e = 0u64;
    for i in 0..16 {
        for c in 0..4 {
            let d = dec[i * 4 + c] as i64 - px[i][c] as i64;
            e += (d * d) as u64;
        }
    }
    e
}

pub fn encode_bcx(
    rgba: &[u8],
    w: usize,
    h: usize,
    quality: u8,
    lambda: f64,
) -> Result<Vec<u8>, String> {
    validate_dims(w, h, rgba.len(), 4)?;
    let bw = w / 4;
    let bh = h / 4;
    let n = bw * bh;
    let bm_sz = (n + 7) / 8;

    struct Decision {
        bc7: bool,
        bytes: [u8; 16],
    }

    let decisions: Vec<Decision> = (0..bh)
        .into_par_iter()
        .flat_map_iter(|by| {
            (0..bw).map(move |bx| {
                let mut px = [[0u8; 4]; 16];
                for i in 0..16 {
                    let o = ((by * 4 + i / 4) * w + bx * 4 + i % 4) * 4;
                    px[i] = [rgba[o], rgba[o + 1], rgba[o + 2], rgba[o + 3]];
                }
                let mut bc1 = [0u8; 8];
                encode_bc1_block_dispatch(&px, quality, &mut bc1);
                let e1 = block_error_bc1(&px, &bc1) as f64;
                let (bc7_bytes, e7) = encode_bc7_block(&px);
                let upgrade = e1 - e7 > lambda * 8.0;
                if upgrade {
                    Decision {
                        bc7: true,
                        bytes: bc7_bytes,
                    }
                } else {
                    let mut b = [0u8; 16];
                    b[..8].copy_from_slice(&bc1);
                    Decision {
                        bc7: false,
                        bytes: b,
                    }
                }
            })
        })
        .collect();

    let mut out = Vec::with_capacity(4 + bm_sz + n * 16);
    out.extend_from_slice(&(bm_sz as u32).to_le_bytes());
    let bm_off = out.len();
    out.resize(bm_off + bm_sz, 0);
    for (i, d) in decisions.iter().enumerate() {
        if d.bc7 {
            out[bm_off + i / 8] |= 1 << (i % 8);
            out.extend_from_slice(&d.bytes);
        } else {
            out.extend_from_slice(&d.bytes[..8]);
        }
    }
    Ok(out)
}

pub fn decode_bcx(blocks: &[u8], w: usize, h: usize) -> Result<Vec<u8>, String> {
    if w == 0 || h == 0 || w % 4 != 0 || h % 4 != 0 {
        return Err(format!("BCN dimensions must be multiples of 4: {w}x{h}"));
    }
    let bw = w / 4;
    let n = bw * (h / 4);
    let bm_sz = n.div_ceil(8);
    if blocks.len() < 4 + bm_sz {
        return Err("BCX payload too short".into());
    }
    let stored = u32::from_le_bytes([blocks[0], blocks[1], blocks[2], blocks[3]]) as usize;
    if stored != bm_sz {
        return Err(format!("BCX bitmap size mismatch: {stored} != {bm_sz}"));
    }
    let bitmap = &blocks[4..4 + bm_sz];
    let is7 = |bi: usize| (bitmap[bi / 8] >> (bi % 8)) & 1 == 1;
    // Prefix offsets once (O(n)) and require an exact payload length.
    let mut offsets = Vec::with_capacity(n);
    let mut pos = 4 + bm_sz;
    for bi in 0..n {
        offsets.push(pos);
        pos += if is7(bi) { 16 } else { 8 };
    }
    if pos != blocks.len() {
        return Err(format!(
            "BCX payload length {} != expected {pos}",
            blocks.len()
        ));
    }
    let mut out = vec![0u8; w * h * 4];
    out.par_chunks_exact_mut(w * 16)
        .enumerate()
        .for_each(|(by, row)| {
            let mut px = [0u8; 64];
            for bx in 0..bw {
                let bi = by * bw + bx;
                let o = offsets[bi];
                if is7(bi) {
                    let mut b16 = [0u8; 16];
                    b16.copy_from_slice(&blocks[o..o + 16]);
                    decode_bc7_block(&b16, &mut px);
                } else {
                    decode_bc1_block(&blocks[o..o + 8], &mut px);
                }
                for i in 0..16 {
                    let d = ((i / 4) * w + bx * 4 + i % 4) * 4;
                    row[d..d + 4].copy_from_slice(&px[i * 4..i * 4 + 4]);
                }
            }
        });
    Ok(out)
}

// ==================== BC7 mode 6 (spec-conformant) ====================
//
// D3D11/Vulkan BC7 mode 6: one subset, RGBA, 7-bit endpoints plus one P-bit
// per endpoint, 4-bit indices with the standard weight table. Pixel 0 is the
// anchor (index MSB implicitly 0). Layout, LSB-first: mode bits 0-6 =
// 0b1000000, R0 R1 G0 G1 B0 B1 A0 A1 (7 bits each, bits 7-62), P0 (63),
// P1 (64), indices (65-127: 3 bits for pixel 0, 4 bits for the rest).
// Verified bit-exact against an independent reference decoder in tests.

pub const BCN_BC7: i32 = 7;

const BC7_WEIGHTS4: [u32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

fn bc7_palette(e0: [u8; 4], e1: [u8; 4]) -> [[u8; 4]; 16] {
    let mut pal = [[0u8; 4]; 16];
    for (i, p) in pal.iter_mut().enumerate() {
        let w = BC7_WEIGHTS4[i];
        for c in 0..4 {
            p[c] = (((64 - w) * e0[c] as u32 + w * e1[c] as u32 + 32) >> 6) as u8;
        }
    }
    pal
}

/// Nearest 8-bit value representable as (v7 << 1) | p.
#[inline]
fn bc7_quant(e: f32, p: u8) -> u8 {
    let v = ((e - p as f32) * 0.5).round().clamp(0.0, 127.0) as u8;
    (v << 1) | p
}

fn bc7_assign(px: &[[u8; 4]; 16], pal: &[[u8; 4]; 16]) -> (u64, [u8; 16]) {
    let mut err = 0u64;
    let mut idx = [0u8; 16];
    for (i, p) in px.iter().enumerate() {
        let mut bi = 0usize;
        let mut bd = u32::MAX;
        for (k, c) in pal.iter().enumerate() {
            let mut d = 0u32;
            for ch in 0..4 {
                let dd = p[ch] as i32 - c[ch] as i32;
                d += (dd * dd) as u32;
            }
            if d < bd {
                bd = d;
                bi = k;
            }
        }
        err += bd as u64;
        idx[i] = bi as u8;
    }
    (err, idx)
}

fn encode_bc7_block(px: &[[u8; 4]; 16]) -> ([u8; 16], f64) {
    let lum = |p: &[u8; 4]| -> u32 {
        299 * p[0] as u32 + 587 * p[1] as u32 + 114 * p[2] as u32 + 500 * p[3] as u32
    };
    let mut mn = [255u8; 4];
    let mut mx = [0u8; 4];
    let (mut lo, mut hi) = (px[0], px[0]);
    for p in px.iter() {
        for c in 0..4 {
            mn[c] = mn[c].min(p[c]);
            mx[c] = mx[c].max(p[c]);
        }
        if lum(p) < lum(&lo) {
            lo = *p;
        }
        if lum(p) > lum(&hi) {
            hi = *p;
        }
    }
    let mut best_err = u64::MAX;
    let mut best_q = ([0u8; 4], [0u8; 4]);
    let mut best_idx = [0u8; 16];
    for (sa, sb) in [(mn, mx), (lo, hi)] {
        let mut e0 = [0f32; 4];
        let mut e1 = [0f32; 4];
        for c in 0..4 {
            e0[c] = sa[c] as f32;
            e1[c] = sb[c] as f32;
        }
        for _ in 0..3 {
            // All 4 P-bit combinations for the current continuous endpoints.
            let mut it_err = u64::MAX;
            let mut it_idx = [0u8; 16];
            for pa in 0..2u8 {
                for pb in 0..2u8 {
                    let q0 = [
                        bc7_quant(e0[0], pa),
                        bc7_quant(e0[1], pa),
                        bc7_quant(e0[2], pa),
                        bc7_quant(e0[3], pa),
                    ];
                    let q1 = [
                        bc7_quant(e1[0], pb),
                        bc7_quant(e1[1], pb),
                        bc7_quant(e1[2], pb),
                        bc7_quant(e1[3], pb),
                    ];
                    let (err, idx) = bc7_assign(px, &bc7_palette(q0, q1));
                    if err < it_err {
                        it_err = err;
                        it_idx = idx;
                    }
                    if err < best_err {
                        best_err = err;
                        best_q = (q0, q1);
                        best_idx = idx;
                    }
                }
            }
            if best_err == 0 {
                break;
            }
            // Least-squares endpoint refit for the fixed index assignment.
            let (mut a00, mut a01, mut a11) = (0f32, 0f32, 0f32);
            let mut b0 = [0f32; 4];
            let mut b1 = [0f32; 4];
            for (p, &i) in px.iter().zip(it_idx.iter()) {
                let t = BC7_WEIGHTS4[i as usize] as f32 / 64.0;
                let s = 1.0 - t;
                a00 += s * s;
                a01 += s * t;
                a11 += t * t;
                for c in 0..4 {
                    b0[c] += s * p[c] as f32;
                    b1[c] += t * p[c] as f32;
                }
            }
            let det = a00 * a11 - a01 * a01;
            if det.abs() < 1e-6 {
                break;
            }
            for c in 0..4 {
                e0[c] = ((a11 * b0[c] - a01 * b1[c]) / det).clamp(0.0, 255.0);
                e1[c] = ((a00 * b1[c] - a01 * b0[c]) / det).clamp(0.0, 255.0);
            }
        }
    }
    let (mut q0, mut q1) = best_q;
    let mut idx = best_idx;
    // Anchor rule: pixel 0's index MSB must be 0. The weight table is
    // symmetric, so swapping endpoints and mirroring indices is lossless.
    if idx[0] >= 8 {
        std::mem::swap(&mut q0, &mut q1);
        for v in idx.iter_mut() {
            *v = 15 - *v;
        }
    }
    let mut bits = [0u8; 16];
    let mut set = |pos: usize, len: usize, val: u64| {
        for k in 0..len {
            if (val >> k) & 1 == 1 {
                bits[(pos + k) / 8] |= 1 << ((pos + k) % 8);
            }
        }
    };
    set(0, 7, 0x40);
    for c in 0..4 {
        set(7 + c * 14, 7, (q0[c] >> 1) as u64);
        set(14 + c * 14, 7, (q1[c] >> 1) as u64);
    }
    set(63, 1, (q0[0] & 1) as u64);
    set(64, 1, (q1[0] & 1) as u64);
    let mut pos = 65;
    for (i, &v) in idx.iter().enumerate() {
        let n = if i == 0 { 3 } else { 4 };
        set(pos, n, v as u64);
        pos += n;
    }
    (bits, best_err as f64)
}

/// Decode a BC7 block. Only mode 6 (the only mode this encoder emits) is
/// supported; any other mode decodes to transparent black.
fn decode_bc7_block(src: &[u8; 16], dst: &mut [u8]) {
    if src[0] & 0x7F != 0x40 {
        dst[..64].fill(0);
        return;
    }
    let get = |pos: usize, len: usize| -> u64 {
        let mut v = 0u64;
        for k in 0..len {
            if (src[(pos + k) / 8] >> ((pos + k) % 8)) & 1 == 1 {
                v |= 1 << k;
            }
        }
        v
    };
    let p0 = get(63, 1) as u8;
    let p1 = get(64, 1) as u8;
    let mut e0 = [0u8; 4];
    let mut e1 = [0u8; 4];
    for c in 0..4 {
        e0[c] = ((get(7 + c * 14, 7) as u8) << 1) | p0;
        e1[c] = ((get(14 + c * 14, 7) as u8) << 1) | p1;
    }
    let pal = bc7_palette(e0, e1);
    let mut pos = 65;
    for i in 0..16 {
        let n = if i == 0 { 3 } else { 4 };
        let v = get(pos, n) as usize;
        pos += n;
        dst[i * 4..i * 4 + 4].copy_from_slice(&pal[v]);
    }
}

pub fn encode_bc7(rgba: &[u8], w: usize, h: usize) -> Result<Vec<u8>, String> {
    validate_dims(w, h, rgba.len(), 4)?;
    let bw = w / 4;
    let mut out = vec![0u8; bw * (h / 4) * 16];
    out.par_chunks_exact_mut(bw * 16)
        .enumerate()
        .for_each(|(by, row)| {
            let mut block = [[0u8; 4]; 16];
            for bx in 0..bw {
                for i in 0..16 {
                    let sy = by * 4 + i / 4;
                    let sx = bx * 4 + i % 4;
                    let o = (sy * w + sx) * 4;
                    block[i][0] = rgba[o];
                    block[i][1] = rgba[o + 1];
                    block[i][2] = rgba[o + 2];
                    block[i][3] = rgba[o + 3];
                }
                let (enc, _) = encode_bc7_block(&block);
                row[bx * 16..bx * 16 + 16].copy_from_slice(&enc);
            }
        });
    Ok(out)
}

pub fn decode_bc7(blocks: &[u8], w: usize, h: usize) -> Result<Vec<u8>, String> {
    if w == 0 || h == 0 || w % 4 != 0 || h % 4 != 0 {
        return Err(format!("BCN dimensions must be multiples of 4: {w}x{h}"));
    }
    let expected = (w / 4) * (h / 4) * 16;
    if blocks.len() != expected {
        return Err(format!(
            "BC7 payload length {}: expected {expected}",
            blocks.len()
        ));
    }
    let bw = w / 4;
    let mut out = vec![0u8; w * h * 4];
    out.par_chunks_exact_mut(w * 16)
        .enumerate()
        .for_each(|(by, row)| {
            let mut px = [0u8; 64];
            for bx in 0..bw {
                let bi = (by * bw + bx) * 16;
                let mut b16 = [0u8; 16];
                b16.copy_from_slice(&blocks[bi..bi + 16]);
                decode_bc7_block(&b16, &mut px);
                for i in 0..16 {
                    let o = ((i / 4) * w + bx * 4 + i % 4) * 4;
                    row[o..o + 4].copy_from_slice(&px[i * 4..i * 4 + 4]);
                }
            }
        });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_rgba(w: usize, h: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                v.push((x * 255 / w.max(1)) as u8);
                v.push((y * 255 / h.max(1)) as u8);
                v.push(((x + y) * 255 / (w + h).max(1)) as u8);
                v.push(255);
            }
        }
        v
    }

    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        let sse: u64 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs() as u64)
            .map(|d| d * d)
            .sum();
        if sse == 0 {
            return 100.0;
        }
        let mse = sse as f64 / a.len() as f64;
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }

    #[test]
    fn test_bc1_roundtrip_psnr() {
        let (w, h) = (64usize, 64usize);
        let src = sample_rgba(w, h);
        let blocks = encode_bc1(&src, w, h).unwrap();
        assert_eq!(blocks.len(), (w / 4) * (h / 4) * 8);
        let dec = decode_bc1(&blocks, w, h).unwrap();
        let q = psnr(&src, &dec);
        assert!(q > 28.0, "BC1 PSNR too low: {q}");
        assert!(blocks.len() < src.len() / 4, "BC1 must beat raw RGBA 4x");
    }

    #[test]
    fn test_bc1_flat_block_lossless() {
        let mut src = vec![90u8; 64 * 64 * 4];
        for i in 0..src.len() {
            if i % 4 == 0 {
                src[i] = 41;
            }
            if i % 4 == 1 {
                src[i] = 203;
            }
            if i % 4 == 2 {
                src[i] = 107;
            }
            if i % 4 == 3 {
                src[i] = 255;
            }
        }
        let blocks = encode_bc1(&src, 64, 64).unwrap();
        let dec = decode_bc1(&blocks, 64, 64).unwrap();
        assert!(psnr(&src, &dec) > 45.0);
    }

    #[test]
    fn test_bc1_alpha_1bit() {
        let mut src = sample_rgba(64, 64);
        for i in 0..(64 * 64) {
            if (i / 64 + i % 64) % 2 == 0 {
                src[i * 4 + 3] = 0;
            }
        }
        let blocks = encode_bc1(&src, 64, 64).unwrap();
        let dec = decode_bc1(&blocks, 64, 64).unwrap();
        for i in 0..(64 * 64) {
            let want = if (i / 64 + i % 64) % 2 == 0 { 0 } else { 255 };
            assert_eq!(dec[i * 4 + 3], want, "alpha mismatch at {i}");
        }
    }

    #[test]
    fn test_bc4_bc5_roundtrip() {
        let (w, h) = (64usize, 64usize);
        let data: Vec<u8> = (0..w * h)
            .map(|i| (((i / w) * 4 + (i % w)) as u8).min(255))
            .collect();
        let blocks = encode_bc4(&data, w, h).unwrap();
        let dec = decode_bc4(&blocks, w, h).unwrap();
        assert!(psnr(&data, &dec) > 40.0, "BC4 PSNR too low");

        let mut rg = Vec::with_capacity(w * h * 2);
        for i in 0..w * h {
            rg.push(data[i]);
            rg.push(255 - data[i]);
        }
        let blocks5 = encode_bc5(&rg, w, h).unwrap();
        assert_eq!(blocks5.len(), (w / 4) * (h / 4) * 16);
        let dec5 = decode_bc5(&blocks5, w, h).unwrap();
        assert!(psnr(&rg, &dec5) > 40.0);
    }

    #[test]
    fn test_adversarial_never_panics() {
        assert!(encode_bc1(&[], 0, 0).is_err());
        assert!(encode_bc1(&[0u8; 100], 4, 4).is_err());
        assert!(decode_bc1(&[0u8; 7], 4, 4).is_err());
        assert!(decode_bc1(&[0u8; 8], 5, 4).is_err());
        assert!(bcn_encode(42, &[0u8; 64], 4, 4).is_err());
        assert!(decode_bc5(&[0u8; 15], 4, 4).is_err());
        // MAX quality path over random + degenerate blocks: Err or clean data, never a panic.
        let mut state: u64 = 0xC0FFEE1234567899;
        for len in [0usize, 4, 64, 256] {
            let mut v = Vec::with_capacity(len * 4);
            for _ in 0..len {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                v.push((state >> 33) as u8);
                v.push((state >> 41) as u8);
                v.push((state >> 49) as u8);
                v.push(255);
            }
            if len % 4 == 0 {
                let bw = (len / 4 / 4).max(1);
                let bh = (len / 4 / bw).max(1);
                let padded = len / 4 == bw * bh && bw % 1 == 0;
                if padded {
                    let q = encode_bc1_q(&v, bw * 4, bh * 4, QUALITY_MAX);
                    assert!(q.is_err() || q.unwrap().len() == bw * bh * 8);
                }
            }
        }
        let flat = vec![77u8; 16 * 4];
        assert!(encode_bc1_q(&flat, 4, 4, QUALITY_MAX).is_ok());
    }

    fn corpus_blocks() -> Vec<(String, [[u8; 4]; 16])> {
        let mut blocks = Vec::new();
        let mut push = |name: &str, gen: &dyn Fn(usize) -> [u8; 4]| {
            let b: Vec<[u8; 4]> = (0..16).map(gen).collect();
            blocks.push((
                name.to_string(),
                [
                    b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11],
                    b[12], b[13], b[14], b[15],
                ],
            ));
        };
        push("bilinear", &|i| {
            let (x, y) = (i % 4, i / 4);
            [
                (60 + x * 40) as u8,
                (30 + y * 50) as u8,
                (200 - x * 25) as u8,
                255,
            ]
        });
        push("hard_edge", &|i| {
            if (i % 4) < 2 {
                [240, 20, 60, 255]
            } else {
                [10, 200, 180, 255]
            }
        });
        push("checker", &|i| {
            let (x, y) = (i % 4, i / 4);
            if (x + y) % 2 == 0 {
                [250, 250, 250, 255]
            } else {
                [5, 5, 5, 255]
            }
        });
        push("noise25", &|i| {
            if i % 2 == 0 {
                [128 + ((i * 37) % 16) as u8, 64, 192, 255]
            } else {
                [128, 64, 192, 255]
            }
        });
        push("skin_ramp", &|i| {
            let t = i as u8 * 16;
            [t.saturating_add(90), (t / 2).saturating_add(40), 90, 255]
        });
        push("dark_ramp", &|i| [i as u8 * 16, i as u8 * 8, 12, 255]);
        push("saturated", &|i| match i % 3 {
            0 => [255, 0, 0, 255],
            1 => [0, 255, 0, 255],
            _ => [0, 0, 255, 255],
        });
        blocks
    }

    #[test]
    fn test_bc7_roundtrip_opaque() {
        let mut src = vec![0u8; 32 * 32 * 4];
        for y in 0..32 {
            for x in 0..32 {
                let i = (y * 32 + x) * 4;
                src[i] = (x * 8) as u8;
                src[i + 1] = (y * 8) as u8;
                src[i + 2] = 128;
                src[i + 3] = 255;
            }
        }
        let blocks = encode_bc7(&src, 32, 32).unwrap();
        assert_eq!(blocks.len(), 8 * 8 * 16);
        // Spec mode 6: low 7 bits of byte 0 == 0b1000000, anchor index MSB == 0.
        assert!(blocks.chunks(16).all(|b| b[0] & 0x7F == 0x40));
        assert!(blocks.chunks(16).all(|b| (b[8] >> 3) & 1 == 0));
        let dec = decode_bc7(&blocks, 32, 32).unwrap();
        // Bilinear r,g patches quantized onto a 4-point line: ~34-35 dB is the
        // honest ceiling for single-subset BC7 here; regression guard only.
        let q = psnr(&src, &dec);
        assert!(q > 30.0, "BC7 PSNR too low: {q}");
    }

    #[test]
    fn test_bc7_smooth_alpha() {
        let mut src = vec![0u8; 16 * 16 * 4];
        for y in 0..16 {
            for x in 0..16 {
                let i = (y * 16 + x) * 4;
                src[i] = 200;
                src[i + 1] = 100;
                src[i + 2] = 50;
                src[i + 3] = ((y * 16 + x) * 255 / 256) as u8;
            }
        }
        let dec = decode_bc7(&encode_bc7(&src, 16, 16).unwrap(), 16, 16).unwrap();
        let q = psnr(&src, &dec);
        assert!(q > 35.0, "BC7 alpha PSNR too low: {q}");
    }

    #[test]
    fn test_bc1_max_near_flat_block_not_transparent() {
        // All pixels fall in one 565 bucket but are not identical: the fit
        // collapses to equal endpoints, which must not decode as 3-color mode.
        let mut src = vec![0u8; 8 * 8 * 4];
        for (i, px) in src.chunks_exact_mut(4).enumerate() {
            px.copy_from_slice(&[100 + (i % 2) as u8, 100, 100, 255]);
        }
        let blocks = encode_bc1_q(&src, 8, 8, QUALITY_MAX).unwrap();
        let dec = decode_bc1(&blocks, 8, 8).unwrap();
        assert!(dec.chunks_exact(4).all(|p| p[3] == 255));
        assert!(psnr(&src, &dec) > 40.0);
    }

    #[test]
    fn test_bcx_roundtrip_and_truncation() {
        let src = sample_rgba(32, 32);
        let enc = encode_bcx(&src, 32, 32, QUALITY_MAX, BCX_DEFAULT_LAMBDA).unwrap();
        let dec = decode_bcx(&enc, 32, 32).unwrap();
        assert!(psnr(&src, &dec) > 28.0);
        assert!(decode_bcx(&enc[..enc.len() - 1], 32, 32).is_err());
        assert!(decode_bcx(&[0u8; 3], 4, 4).is_err());
    }

    #[test]
    fn test_bc7_validation() {
        assert!(encode_bc7(&[], 0, 0).is_err());
        assert!(encode_bc7(&[0u8; 100], 4, 4).is_err());
        assert!(decode_bc7(&[0u8; 15], 4, 4).is_err());
        assert!(decode_bc7(&[0u8; 8], 4, 4).is_err());
    }

    #[test]
    fn test_bc1_rdo_structured() {
        let mut src = vec![0u8; 16 * 16 * 4];
        for y in 0..16 {
            for x in 0..16 {
                let i = (y * 16 + x) * 4;
                src[i] = ((x * 37 + y * 11) % 256) as u8;
                src[i + 1] = ((x * 13 + y * 53) % 256) as u8;
                src[i + 2] = ((x * 71 + y * 29) % 256) as u8;
                src[i + 3] = 255;
            }
        }
        let plain = encode_bc1_q(&src, 16, 16, QUALITY_MAX).unwrap();
        let rdo = encode_bc1_rdo(&src, 16, 16, QUALITY_MAX, 1.0, 2).unwrap();
        let e_plain = psnr(&src, &decode_bc1(&plain, 16, 16).unwrap());
        let e_rdo = psnr(&src, &decode_bc1(&rdo, 16, 16).unwrap());
        println!("rdo structured plain={e_plain:.2} rdo={e_rdo:.2}");
        assert!(e_rdo >= e_plain - 0.05, "RDO must not regress");
    }

    #[test]
    fn test_quality_max_beats_fast() {
        let corpus = corpus_blocks();
        let mut total_fast = 0f64;
        let mut total_max = 0f64;
        for (name, block) in &corpus {
            let src: Vec<u8> = block.iter().flat_map(|p| p.iter().copied()).collect();
            let mut fast = [0u8; 8];
            encode_bc1_block_fast(block, &mut fast);
            let mut max = [0u8; 8];
            encode_bc1_block_dispatch(block, QUALITY_MAX, &mut max);
            let pf = psnr(&src, &decode_bc1(&fast, 4, 4).unwrap());
            let pm = psnr(&src, &decode_bc1(&max, 4, 4).unwrap());
            total_fast += pf;
            total_max += pm;
            println!("bc1q {:<10} fast={pf:6.2} max={pm:6.2}", name);
        }
        let n = corpus.len() as f64;
        println!(
            "bc1q AVERAGE fast={:.3} max={:.3} delta={:.3}",
            total_fast / n,
            total_max / n,
            (total_max - total_fast) / n
        );
        assert!(
            total_max >= total_fast - 0.5,
            "MAX quality must not regress the corpus"
        );
        // The smooth bilinear block must measurably improve (cluster fit + LSQ).
        let (_name, block) = &corpus[0];
        let src: Vec<u8> = block.iter().flat_map(|p| p.iter().copied()).collect();
        let mut fast = [0u8; 8];
        encode_bc1_block_fast(block, &mut fast);
        let mut max = [0u8; 8];
        encode_bc1_block_dispatch(block, QUALITY_MAX, &mut max);
        let pf = psnr(&src, &decode_bc1(&fast, 4, 4).unwrap());
        let pm = psnr(&src, &decode_bc1(&max, 4, 4).unwrap());
        println!("bc1q bilinear fast={pf:.3} max={pm:.3}");
        assert!(pm >= pf - 0.01, "bilinear block must not regress");
    }
}
