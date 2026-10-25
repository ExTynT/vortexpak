//! VortexLZ-ANS (`VXLZ` v2.0) — Proprietary Multi-Stream LZ + Chunked 8-Way rANS Codec
//! ====================================================================================
//! 100% original, zero-dependency entropy and dictionary compression engine designed for
//! VortexPak micro-blocks:
//!   1. `VXANS`: 8-Way Interleaved 12-bit table-driven rANS (Range Asymmetric Numeral Systems)
//!      with adaptive 32 KB block-local frequency tables (`SUB_RANS8_CHUNKED`), a single 16 KB
//!      L1-cache lookup table (`sym: 8b | freq: 12b | bias: 12b`), and branchless 16-bit (`u16`)
//!      state renormalization.
//!   2. `VXLZ`: 6-Stream homogeneous Lempel-Ziv engine (`Tokens`, `Literals`, `Off16_Lo`,
//!      `Off16_Hi`, `Off24_Rep2`, `ExtLengths`) with a 3-entry Repeat-Offset Cache
//!      (`rep0, rep1, rep2`), 3-tier hash match finder (8-byte `table8`, 8-way 4-byte `table4`,
//!      2-way 3-byte `table3`), bit-cost-aware lazy matching, and cascaded second-order
//!      sub-stream compression.
//!   3. SIMD Spatial & Color Decorrelation Pre-Filters:
//!      - `BLK_VXLZ_SUBGREEN3` / AVX2 `BLK_VXLZ_SUBGREEN4` (`R -= G, B -= G`)
//!      - SSE2/AVX2 Prefix-Scan `BLK_VXLZ_DELTA1`, `BLK_VXLZ_DELTA3`, `BLK_VXLZ_DELTA4`

use std::borrow::Cow;
use std::mem::MaybeUninit;

const SCALE_BITS: u32 = 12;
const SCALE: usize = 1 << SCALE_BITS; // 4096
const RANS_L: u64 = 1 << 16; // 65536
const NUM_LANES: usize = 8;
const RANS_CHUNK_SIZE: usize = 32768;

const SUB_RAW: u8 = 0;
const SUB_CONST: u8 = 1;
const SUB_RANS8: u8 = 2;
const SUB_VXLZ1: u8 = 3;
const SUB_RANS8_CHUNKED: u8 = 4;

const BLK_RAW: u8 = 0;
const BLK_CONST: u8 = 1;
const BLK_RANS8: u8 = 2;
const BLK_VXLZ: u8 = 3;
const BLK_VXLZ_SUBGREEN3: u8 = 4;
const BLK_VXLZ_SUBGREEN4: u8 = 5;
const BLK_VXLZ_DELTA1: u8 = 6;
const BLK_VXLZ_DELTA3: u8 = 7;
const BLK_VXLZ_DELTA4: u8 = 8;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Fast multi-window 0-th order entropy proxy (lower is more compressible).
/// Samples up to 4 windows across `src` so headers at offset 0 do not skew the estimate.
fn estimate_entropy_proxy(src: &[u8]) -> u64 {
    if src.is_empty() {
        return 0;
    }
    let mut counts = [0u32; 256];
    let n = src.len();
    if n <= 4096 {
        for &b in src {
            counts[b as usize] += 1;
        }
    } else {
        let win = 1024usize;
        for w in 0..4 {
            let start = (w * (n - win)) / 3;
            for &b in &src[start..start + win] {
                counts[b as usize] += 1;
            }
        }
    }
    let mut score = 0u64;
    for &c in &counts {
        if c > 0 {
            let cu = c as u64;
            let lz = c.leading_zeros() as u64;
            score += cu * lz;
        }
    }
    score
}

fn apply_subgreen3_encode(src: &[u8]) -> Vec<u8> {
    let mut out = src.to_vec();
    let mut i = 0usize;
    while i + 3 <= out.len() {
        let g = out[i + 1];
        out[i] = out[i].wrapping_sub(g);
        out[i + 2] = out[i + 2].wrapping_sub(g);
        i += 3;
    }
    out
}

fn apply_subgreen3_decode_in_place(dst: &mut [u8]) {
    let mut i = 0usize;
    let n = dst.len();
    unsafe {
        let ptr = dst.as_mut_ptr();
        while i + 12 <= n {
            let g0 = *ptr.add(i + 1);
            *ptr.add(i) = (*ptr.add(i)).wrapping_add(g0);
            *ptr.add(i + 2) = (*ptr.add(i + 2)).wrapping_add(g0);

            let g1 = *ptr.add(i + 4);
            *ptr.add(i + 3) = (*ptr.add(i + 3)).wrapping_add(g1);
            *ptr.add(i + 5) = (*ptr.add(i + 5)).wrapping_add(g1);

            let g2 = *ptr.add(i + 7);
            *ptr.add(i + 6) = (*ptr.add(i + 6)).wrapping_add(g2);
            *ptr.add(i + 8) = (*ptr.add(i + 8)).wrapping_add(g2);

            let g3 = *ptr.add(i + 10);
            *ptr.add(i + 9) = (*ptr.add(i + 9)).wrapping_add(g3);
            *ptr.add(i + 11) = (*ptr.add(i + 11)).wrapping_add(g3);
            i += 12;
        }
        while i + 3 <= n {
            let g = *ptr.add(i + 1);
            *ptr.add(i) = (*ptr.add(i)).wrapping_add(g);
            *ptr.add(i + 2) = (*ptr.add(i + 2)).wrapping_add(g);
            i += 3;
        }
    }
}

fn apply_subgreen4_encode(src: &[u8]) -> Vec<u8> {
    let mut out = src.to_vec();
    let mut i = 0usize;
    while i + 4 <= out.len() {
        let g = out[i + 1];
        out[i] = out[i].wrapping_sub(g);
        out[i + 2] = out[i + 2].wrapping_sub(g);
        i += 4;
    }
    out
}

fn apply_subgreen4_decode_in_place(dst: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                apply_subgreen4_decode_avx2(dst);
                return;
            }
        }
    }
    let mut i = 0usize;
    while i + 4 <= dst.len() {
        let g = dst[i + 1];
        dst[i] = dst[i].wrapping_add(g);
        dst[i + 2] = dst[i + 2].wrapping_add(g);
        i += 4;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_subgreen4_decode_avx2(dst: &mut [u8]) {
    let n = dst.len();
    let chunks32 = n / 32;
    let g_mask = _mm256_set_epi8(
        -1, 13, -1, 13, -1, 9, -1, 9, -1, 5, -1, 5, -1, 1, -1, 1, -1, 13, -1, 13, -1, 9, -1, 9, -1,
        5, -1, 5, -1, 1, -1, 1,
    );
    let ptr = dst.as_mut_ptr() as *mut __m256i;
    for c in 0..chunks32 {
        let v = _mm256_loadu_si256(ptr.add(c));
        let g_vec = _mm256_shuffle_epi8(v, g_mask);
        let restored = _mm256_add_epi8(v, g_vec);
        _mm256_storeu_si256(ptr.add(c), restored);
    }
    let mut i = chunks32 * 32;
    while i + 4 <= n {
        let g = *dst.get_unchecked(i + 1);
        *dst.get_unchecked_mut(i) = dst.get_unchecked(i).wrapping_add(g);
        *dst.get_unchecked_mut(i + 2) = dst.get_unchecked(i + 2).wrapping_add(g);
        i += 4;
    }
}

fn apply_delta_stride_encode(src: &[u8], stride: usize) -> Vec<u8> {
    let n = src.len();
    let mut out = vec![0u8; n];
    if n <= stride {
        out.copy_from_slice(src);
        return out;
    }
    out[..stride].copy_from_slice(&src[..stride]);
    for i in stride..n {
        out[i] = src[i].wrapping_sub(src[i - stride]);
    }
    out
}

fn apply_delta1_decode_in_place(dst: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                apply_delta1_decode_sse2(dst);
                return;
            }
        }
    }
    for i in 1..dst.len() {
        dst[i] = dst[i].wrapping_add(dst[i - 1]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_delta1_decode_sse2(dst: &mut [u8]) {
    let n = dst.len();
    let chunks16 = n / 16;
    let ptr = dst.as_mut_ptr() as *mut __m128i;
    let mut carry = _mm_setzero_si128();
    let bcast_last = _mm_set1_epi8(15);

    for c in 0..chunks16 {
        let mut v = _mm_loadu_si128(ptr.add(c));
        v = _mm_add_epi8(v, _mm_slli_si128(v, 1));
        v = _mm_add_epi8(v, _mm_slli_si128(v, 2));
        v = _mm_add_epi8(v, _mm_slli_si128(v, 4));
        v = _mm_add_epi8(v, _mm_slli_si128(v, 8));
        v = _mm_add_epi8(v, carry);
        _mm_storeu_si128(ptr.add(c), v);
        carry = _mm_shuffle_epi8(v, bcast_last);
    }

    let start = chunks16 * 16;
    for i in start.max(1)..n {
        *dst.get_unchecked_mut(i) = dst.get_unchecked(i).wrapping_add(*dst.get_unchecked(i - 1));
    }
}

fn apply_delta3_decode_in_place(dst: &mut [u8]) {
    let n = dst.len();
    unsafe {
        let ptr = dst.as_mut_ptr();
        for i in 3..n {
            *ptr.add(i) = (*ptr.add(i)).wrapping_add(*ptr.add(i - 3));
        }
    }
}

fn apply_delta4_decode_in_place(dst: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                apply_delta4_decode_sse2(dst);
                return;
            }
        }
    }
    for i in 4..dst.len() {
        dst[i] = dst[i].wrapping_add(dst[i - 4]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_delta4_decode_sse2(dst: &mut [u8]) {
    let n = dst.len();
    let chunks16 = n / 16;
    let ptr = dst.as_mut_ptr() as *mut __m128i;
    let mut carry = _mm_setzero_si128();
    let bcast_last4 = _mm_set_epi8(
        15, 14, 13, 12, 15, 14, 13, 12, 15, 14, 13, 12, 15, 14, 13, 12,
    );

    for c in 0..chunks16 {
        let mut v = _mm_loadu_si128(ptr.add(c));
        v = _mm_add_epi8(v, _mm_slli_si128(v, 4));
        v = _mm_add_epi8(v, _mm_slli_si128(v, 8));
        v = _mm_add_epi8(v, carry);
        _mm_storeu_si128(ptr.add(c), v);
        carry = _mm_shuffle_epi8(v, bcast_last4);
    }

    let start = chunks16 * 16;
    for i in start.max(4)..n {
        *dst.get_unchecked_mut(i) = dst.get_unchecked(i).wrapping_add(*dst.get_unchecked(i - 4));
    }
}

const HASH8_LOG: usize = 15;
const HASH8_SIZE: usize = 1 << HASH8_LOG; // 32,768 buckets x 4 ways
const HASH4_LOG: usize = 16;
const HASH4_SIZE: usize = 1 << HASH4_LOG; // 65,536 buckets x 8 ways
const HASH3_LOG: usize = 15;
const HASH3_SIZE: usize = 1 << HASH3_LOG; // 32,768 buckets x 2 ways

#[inline(always)]
fn hash8(v: u64) -> usize {
    ((v.wrapping_mul(0xCF1BBCDCBFA533CB)) >> (64 - HASH8_LOG)) as usize
}

#[inline(always)]
fn hash4(v: u32) -> usize {
    ((v.wrapping_mul(0x1E35_A7BD)) >> (32 - HASH4_LOG)) as usize
}

#[inline(always)]
fn hash3(v: u32) -> usize {
    (((v & 0x00FF_FFFF).wrapping_mul(0x1E35_A7BD)) >> (32 - HASH3_LOG)) as usize
}

/// Normalize byte counts into 12-bit frequencies summing to exactly `SCALE` (4096).
/// Every symbol with `count > 0` receives `freq >= 1`.
fn normalize_frequencies(counts: &[u32; 256], total: usize) -> ([u16; 256], [u16; 256], usize) {
    let mut freqs = [0u16; 256];
    let mut cdf = [0u16; 256];
    if total == 0 {
        return (freqs, cdf, 0);
    }

    let mut active_syms = 0usize;
    let mut sum_f = 0usize;
    let mut max_sym = 0usize;
    let mut max_cnt = 0u32;

    for s in 0..256 {
        let c = counts[s];
        if c > 0 {
            active_syms += 1;
            if c > max_cnt {
                max_cnt = c;
                max_sym = s;
            }
            let f = (((c as u64) * (SCALE as u64) + (total as u64 / 2)) / (total as u64)) as usize;
            let f_clamped = f.max(1);
            freqs[s] = f_clamped as u16;
            sum_f += f_clamped;
        }
    }

    if active_syms <= 1 {
        return (freqs, cdf, active_syms);
    }

    if sum_f < SCALE {
        let diff = SCALE - sum_f;
        freqs[max_sym] += diff as u16;
    } else if sum_f > SCALE {
        let mut excess = sum_f - SCALE;
        if (freqs[max_sym] as usize) > excess + 1 {
            freqs[max_sym] -= excess as u16;
            excess = 0;
        } else if freqs[max_sym] > 1 {
            let take = (freqs[max_sym] as usize) - 1;
            freqs[max_sym] = 1;
            excess -= take;
        }
        if excess > 0 {
            let mut order: Vec<usize> = (0..256).filter(|&s| freqs[s] > 1).collect();
            order.sort_unstable_by(|&a, &b| freqs[b].cmp(&freqs[a]));
            while excess > 0 && !order.is_empty() {
                let mut changed = false;
                for &s in &order {
                    if freqs[s] > 1 && excess > 0 {
                        freqs[s] -= 1;
                        excess -= 1;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
        }
    }

    let mut cur = 0u16;
    for s in 0..256 {
        cdf[s] = cur;
        cur = cur.wrapping_add(freqs[s]);
    }
    debug_assert_eq!(cur as usize, SCALE);
    (freqs, cdf, active_syms)
}

/// Compress a byte slice using 8-Way Interleaved 12-bit rANS (`VXANS`).
pub fn vxans_compress(src: &[u8]) -> Option<Vec<u8>> {
    if src.len() < 32 {
        return None;
    }

    let mut counts = [0u32; 256];
    for &b in src {
        counts[b as usize] += 1;
    }

    let (freqs, cdf, active_syms) = normalize_frequencies(&counts, src.len());
    if active_syms <= 1 {
        return None;
    }

    let mut header = Vec::with_capacity(32 + active_syms * 2 + 24);
    let mut bitmap = [0u8; 32];
    for s in 0..256 {
        if freqs[s] > 0 {
            bitmap[s >> 3] |= 1 << (s & 7);
        }
    }
    header.extend_from_slice(&bitmap);
    for s in 0..256 {
        let f = freqs[s];
        if f > 0 {
            if f < 128 {
                header.push(f as u8);
            } else {
                header.push(0x80 | ((f & 0x7F) as u8));
                header.push((f >> 7) as u8);
            }
        }
    }

    let mut states = [RANS_L; NUM_LANES];
    let mut word_stack: Vec<u16> = Vec::with_capacity(src.len() / 2 + 16);

    let n = src.len();
    let rem = n % NUM_LANES;
    let aligned = n - rem;

    for idx in (aligned..n).rev() {
        let lane = idx - aligned;
        let s = src[idx] as usize;
        let f = freqs[s] as u64;
        let c = cdf[s] as u64;
        let x_max = f << 20;
        let mut x = states[lane];
        if x >= x_max {
            word_stack.push((x & 0xFFFF) as u16);
            x >>= 16;
        }
        states[lane] = ((x / f) << SCALE_BITS) + (x % f) + c;
    }

    let mut chunk_start = aligned;
    while chunk_start >= NUM_LANES {
        chunk_start -= NUM_LANES;
        let chunk = &src[chunk_start..chunk_start + NUM_LANES];
        for lane in (0..NUM_LANES).rev() {
            let s = chunk[lane] as usize;
            let f = freqs[s] as u64;
            let c = cdf[s] as u64;
            let x_max = f << 20;
            let mut x = states[lane];
            if x >= x_max {
                word_stack.push((x & 0xFFFF) as u16);
                x >>= 16;
            }
            states[lane] = ((x / f) << SCALE_BITS) + (x % f) + c;
        }
    }

    let total_comp_size = header.len() + NUM_LANES * 4 + word_stack.len() * 2;
    if total_comp_size + 8 >= src.len() {
        return None;
    }

    let mut out = Vec::with_capacity(total_comp_size);
    out.extend_from_slice(&header);
    for lane in 0..NUM_LANES {
        out.extend_from_slice(&(states[lane] as u32).to_le_bytes());
    }
    for &w in word_stack.iter().rev() {
        out.extend_from_slice(&w.to_le_bytes());
    }
    Some(out)
}

/// Adaptive chunked 8-Way rANS compressor: splits large streams into 32 KB windows
/// with independent local frequency tables (or constant/raw bypass per window).
pub fn vxans_compress_chunked(src: &[u8]) -> Option<Vec<u8>> {
    if src.len() <= RANS_CHUNK_SIZE {
        return None;
    }
    let mut out = Vec::with_capacity(src.len() / 2);
    let mut offset = 0usize;
    while offset < src.len() {
        let end = (offset + RANS_CHUNK_SIZE).min(src.len());
        let slice = &src[offset..end];
        let first = slice[0];
        if slice.len() > 4 && slice.iter().all(|&b| b == first) {
            out.push(SUB_CONST);
            out.push(first);
        } else if let Some(c_chunk) = vxans_compress(slice) {
            out.push(SUB_RANS8);
            push_u32_le(&mut out, c_chunk.len() as u32);
            out.extend_from_slice(&c_chunk);
        } else {
            out.push(SUB_RAW);
            out.extend_from_slice(slice);
        }
        offset = end;
    }
    if out.len() + 16 < src.len() {
        Some(out)
    } else {
        None
    }
}

pub fn vxans_decompress_chunked(src: &[u8], dst: &mut [u8]) -> Result<(), String> {
    let mut pos = 0usize;
    let mut out_off = 0usize;
    while out_off < dst.len() {
        if pos >= src.len() {
            return Err("Truncated chunked VXANS stream".into());
        }
        let end_off = (out_off + RANS_CHUNK_SIZE).min(dst.len());
        let chunk_dst = &mut dst[out_off..end_off];
        let mode = src[pos];
        pos += 1;
        match mode {
            SUB_CONST => {
                if pos >= src.len() {
                    return Err("Truncated chunked SUB_CONST".into());
                }
                chunk_dst.fill(src[pos]);
                pos += 1;
            }
            SUB_RAW => {
                if pos + chunk_dst.len() > src.len() {
                    return Err("Truncated chunked SUB_RAW".into());
                }
                chunk_dst.copy_from_slice(&src[pos..pos + chunk_dst.len()]);
                pos += chunk_dst.len();
            }
            SUB_RANS8 => {
                let c_len = read_u32_le(src, &mut pos)? as usize;
                if pos + c_len > src.len() {
                    return Err("Truncated chunked SUB_RANS8".into());
                }
                vxans_decompress(&src[pos..pos + c_len], chunk_dst)?;
                pos += c_len;
            }
            _ => return Err(format!("Invalid chunked VXANS mode: {}", mode)),
        }
        out_off = end_off;
    }
    Ok(())
}

/// Decompress an 8-Way Interleaved 12-bit rANS (`VXANS`) stream into `dst`.
pub fn vxans_decompress(src: &[u8], dst: &mut [u8]) -> Result<(), String> {
    if dst.is_empty() {
        return Ok(());
    }
    if src.len() < 32 + NUM_LANES * 4 {
        return Err("Truncated VXANS header".into());
    }

    let bitmap = &src[0..32];
    let mut pos = 32usize;

    let mut freqs = [0u16; 256];
    let mut sum_f = 0usize;
    for byte_idx in 0..32 {
        let mut bits = bitmap[byte_idx];
        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let s = (byte_idx << 3) | bit;
            if pos >= src.len() {
                return Err("Truncated VXANS frequency table".into());
            }
            let b0 = unsafe { *src.get_unchecked(pos) };
            pos += 1;
            let f = if (b0 & 0x80) == 0 {
                b0 as u16
            } else {
                if pos >= src.len() {
                    return Err("Truncated VXANS 2-byte freq".into());
                }
                let b1 = unsafe { *src.get_unchecked(pos) };
                pos += 1;
                ((b0 & 0x7F) as u16) | ((b1 as u16) << 7)
            };
            freqs[s] = f;
            sum_f += f as usize;
        }
    }

    if sum_f != SCALE {
        return Err(format!(
            "Invalid VXANS frequency sum: {} != {}",
            sum_f, SCALE
        ));
    }

    // Build 16 KB L1-cache lookup table without redundant zero-initialization
    let mut dec_lut_uninit: MaybeUninit<[u32; SCALE]> = MaybeUninit::uninit();
    let lut_ptr = dec_lut_uninit.as_mut_ptr() as *mut u32;
    let mut cur_cdf = 0usize;
    unsafe {
        for byte_idx in 0..32 {
            let mut bits = bitmap[byte_idx];
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let s = (byte_idx << 3) | bit;
                let f = *freqs.get_unchecked(s) as usize;
                let base_entry = (s as u32) | ((f as u32) << 8);
                let out_slice = lut_ptr.add(cur_cdf);
                for bias in 0..f {
                    *out_slice.add(bias) = base_entry | ((bias as u32) << 20);
                }
                cur_cdf += f;
            }
        }
    }

    if pos + NUM_LANES * 4 > src.len() {
        return Err("Truncated VXANS initial states".into());
    }

    let mut states = [0u64; NUM_LANES];
    for lane in 0..NUM_LANES {
        let off = pos + lane * 4;
        states[lane] =
            u32::from_le_bytes([src[off], src[off + 1], src[off + 2], src[off + 3]]) as u64;
    }
    pos += NUM_LANES * 4;

    let words_bytes = &src[pos..];
    let n = dst.len();
    let rem = n % NUM_LANES;
    let aligned = n - rem;

    unsafe {
        let mut word_ptr = words_bytes.as_ptr();
        let word_end = words_bytes.as_ptr().add(words_bytes.len());
        let mut out_ptr = dst.as_mut_ptr();

        let mut s0 = states[0];
        let mut s1 = states[1];
        let mut s2 = states[2];
        let mut s3 = states[3];
        let mut s4 = states[4];
        let mut s5 = states[5];
        let mut s6 = states[6];
        let mut s7 = states[7];

        let chunks = aligned / NUM_LANES;
        for _ in 0..chunks {
            macro_rules! step_lane {
                ($st:ident, $offset:expr) => {{
                    let entry = *lut_ptr.add(($st & 0xFFF) as usize);
                    *out_ptr.add($offset) = entry as u8;
                    $st = ($st >> 12) * (((entry >> 8) & 0xFFF) as u64) + ((entry >> 20) as u64);
                    if $st < RANS_L && word_ptr.add(2) <= word_end {
                        let w = (word_ptr as *const u16).read_unaligned() as u64;
                        $st = ($st << 16) | w;
                        word_ptr = word_ptr.add(2);
                    }
                }};
            }

            step_lane!(s0, 0);
            step_lane!(s1, 1);
            step_lane!(s2, 2);
            step_lane!(s3, 3);
            step_lane!(s4, 4);
            step_lane!(s5, 5);
            step_lane!(s6, 6);
            step_lane!(s7, 7);

            out_ptr = out_ptr.add(NUM_LANES);
        }

        states = [s0, s1, s2, s3, s4, s5, s6, s7];

        for lane in 0..rem {
            let mut st = states[lane];
            let entry = *lut_ptr.add((st & 0xFFF) as usize);
            *out_ptr.add(lane) = entry as u8;
            st = (st >> 12) * (((entry >> 8) & 0xFFF) as u64) + ((entry >> 20) as u64);
            if st < RANS_L && word_ptr.add(2) <= word_end {
                let w = (word_ptr as *const u16).read_unaligned() as u64;
                st = (st << 16) | w;
                word_ptr = word_ptr.add(2);
            }
            states[lane] = st;
        }
    }

    Ok(())
}

#[inline(always)]
fn push_varint(buf: &mut Vec<u8>, mut val: usize) {
    while val >= 255 {
        buf.push(255);
        val -= 255;
    }
    buf.push(val as u8);
}

#[inline(always)]
fn read_varint(buf: &[u8], pos: &mut usize) -> usize {
    let mut sum = 0usize;
    while *pos < buf.len() {
        let b = unsafe { *buf.get_unchecked(*pos) };
        *pos += 1;
        sum += b as usize;
        if b < 255 {
            break;
        }
    }
    sum
}

#[inline(always)]
fn push_u32_le(buf: &mut Vec<u8>, val: u32) {
    buf.extend_from_slice(&val.to_le_bytes());
}

#[inline(always)]
fn read_u32_le(buf: &[u8], pos: &mut usize) -> Result<u32, String> {
    if *pos + 4 > buf.len() {
        return Err("Truncated u32 in VXLZ stream".into());
    }
    let v = u32::from_le_bytes([buf[*pos], buf[*pos + 1], buf[*pos + 2], buf[*pos + 3]]);
    *pos += 4;
    Ok(v)
}

/// Count matching bytes between `src[a..]` and `src[b..]` (`a < b`).
#[inline(always)]
fn count_match(src: &[u8], a: usize, b: usize) -> usize {
    let limit = src.len() - b;
    let mut len = 0usize;
    unsafe {
        let pa = src.as_ptr().add(a);
        let pb = src.as_ptr().add(b);
        while len + 8 <= limit {
            let va = (pa.add(len) as *const u64).read_unaligned();
            let vb = (pb.add(len) as *const u64).read_unaligned();
            let diff = va ^ vb;
            if diff != 0 {
                return len + (diff.trailing_zeros() >> 3) as usize;
            }
            len += 8;
        }
        while len < limit && *pa.add(len) == *pb.add(len) {
            len += 1;
        }
    }
    len
}

/// Allocate an uninitialized `Vec<u8>` of exact length `len` to avoid zero-fill overhead.
#[inline(always)]
fn alloc_uninit_vec(len: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(len);
    unsafe {
        v.set_len(len);
    }
    v
}

/// Compress a sub-stream using the best of `SUB_CONST`, `SUB_RANS8`, `SUB_RANS8_CHUNKED`, `SUB_VXLZ1`, or `SUB_RAW`.
fn compress_substream(data: &[u8], allow_lz: bool) -> Vec<u8> {
    let n = data.len();
    let mut out = Vec::with_capacity(n + 9);
    if n == 0 {
        out.push(SUB_RAW);
        push_u32_le(&mut out, 0);
        return out;
    }

    let first = data[0];
    if n > 2 && data.iter().all(|&b| b == first) {
        out.push(SUB_CONST);
        push_u32_le(&mut out, n as u32);
        out.push(first);
        return out;
    }

    let mut best_mode = SUB_RAW;
    let mut best_payload: Option<Vec<u8>> = None;
    let mut best_len = n;

    if let Some(rans_blob) = vxans_compress(data) {
        if rans_blob.len() < best_len {
            best_len = rans_blob.len();
            best_mode = SUB_RANS8;
            best_payload = Some(rans_blob);
        }
    }

    if n > RANS_CHUNK_SIZE {
        if let Some(chunked_blob) = vxans_compress_chunked(data) {
            if chunked_blob.len() < best_len {
                best_len = chunked_blob.len();
                best_mode = SUB_RANS8_CHUNKED;
                best_payload = Some(chunked_blob);
            }
        }
    }

    if allow_lz && n >= 128 {
        let lz_blob = vxlz_compress_core(data, false);
        if lz_blob.len() + 4 < best_len {
            best_len = lz_blob.len();
            best_mode = SUB_VXLZ1;
            best_payload = Some(lz_blob);
        }
    }

    out.push(best_mode);
    push_u32_le(&mut out, n as u32);
    if let Some(pay) = best_payload {
        push_u32_le(&mut out, pay.len() as u32);
        out.extend_from_slice(&pay);
    } else {
        push_u32_le(&mut out, n as u32);
        out.extend_from_slice(data);
    }
    debug_assert!(best_len <= n);
    out
}

/// Decompress a sub-stream encoded by `compress_substream`.
/// Uses `Cow::Borrowed` for zero-copy `SUB_RAW` and uninitialized `Vec` for decoded streams.
fn decompress_substream<'a>(src: &'a [u8], pos: &mut usize) -> Result<Cow<'a, [u8]>, String> {
    if *pos >= src.len() {
        return Err("Unexpected EOF in VXLZ substream".into());
    }
    let mode = src[*pos];
    *pos += 1;
    let raw_len = read_u32_le(src, pos)? as usize;
    if raw_len == 0 {
        return Ok(Cow::Borrowed(&[]));
    }

    if mode == SUB_CONST {
        if *pos >= src.len() {
            return Err("Truncated SUB_CONST byte".into());
        }
        let val = src[*pos];
        *pos += 1;
        return Ok(Cow::Owned(vec![val; raw_len]));
    }

    let comp_len = read_u32_le(src, pos)? as usize;
    if *pos + comp_len > src.len() {
        return Err("Truncated VXLZ substream payload".into());
    }
    let payload = &src[*pos..*pos + comp_len];
    *pos += comp_len;

    match mode {
        SUB_RAW => {
            if comp_len != raw_len {
                return Err("SUB_RAW size mismatch".into());
            }
            Ok(Cow::Borrowed(payload))
        }
        SUB_RANS8 => {
            let mut out = alloc_uninit_vec(raw_len);
            vxans_decompress(payload, &mut out)?;
            Ok(Cow::Owned(out))
        }
        SUB_RANS8_CHUNKED => {
            let mut out = alloc_uninit_vec(raw_len);
            vxans_decompress_chunked(payload, &mut out)?;
            Ok(Cow::Owned(out))
        }
        SUB_VXLZ1 => {
            let mut out = alloc_uninit_vec(raw_len);
            vxlz_decompress_into(payload, &mut out)?;
            Ok(Cow::Owned(out))
        }
        _ => Err(format!("Unknown VXLZ substream mode: {}", mode)),
    }
}

/// Core Multi-Stream VXLZ encoder with 3-tier hash match finder (`table8`, `table4`, `table3`).
fn vxlz_compress_core(src: &[u8], allow_cascaded_lz: bool) -> Vec<u8> {
    let n = src.len();
    if n == 0 {
        return vec![BLK_RAW];
    }

    let first = src[0];
    if n >= 16 && src.iter().all(|&b| b == first) {
        return vec![BLK_CONST, first];
    }

    if n < 32 {
        let mut raw = Vec::with_capacity(n + 1);
        raw.push(BLK_RAW);
        raw.extend_from_slice(src);
        return raw;
    }

    let use_t8 = allow_cascaded_lz && n >= 16384;
    let mut table8 = if use_t8 {
        vec![[0u32; 4]; HASH8_SIZE]
    } else {
        Vec::new()
    };
    let mut head8 = if use_t8 {
        vec![0u8; HASH8_SIZE]
    } else {
        Vec::new()
    };
    let mut table4 = vec![[0u32; 8]; HASH4_SIZE];
    let mut head4 = vec![0u8; HASH4_SIZE];
    let mut table3 = vec![[0u32; 2]; HASH3_SIZE];
    let mut head3 = vec![0u8; HASH3_SIZE];

    let mut tokens: Vec<u8> = Vec::with_capacity(n / 8);
    let mut lits: Vec<u8> = Vec::with_capacity(n / 4);
    let mut off16_lo: Vec<u8> = Vec::with_capacity(n / 16);
    let mut off16_hi: Vec<u8> = Vec::with_capacity(n / 16);
    let mut off24: Vec<u8> = Vec::new();
    let mut ext_len: Vec<u8> = Vec::new();

    let mut rep0: usize = 1;
    let mut rep1: usize = 4;
    let mut rep2: usize = 8;

    let mut anchor: usize = 0;
    let mut ip: usize = 0;
    let match_limit = n.saturating_sub(8);

    // Closure to find the best bit-cost-weighted match at `pos`
    let find_best_match = |pos: usize,
                           r0: usize,
                           r1: usize,
                           r2: usize,
                           t8: &[[u32; 4]],
                           t4: &[[u32; 8]],
                           t3: &[[u32; 2]]|
     -> (usize, usize, u8, i32) {
        // Returns (match_len, offset, mode_code: 0=REP0, 1=REP1, 2=OFF16, 3=OFF24, 4=REP2, score_qbits)
        let mut best_ml = 0usize;
        let mut best_off = 0usize;
        let mut best_mode = 2u8;
        let mut best_score = 11i32; // Minimum threshold to beat literal emission

        // 1. Check repeat offsets (rep0, rep1, rep2)
        for &(r_off, m_code, bonus) in &[(r0, 0u8, 8i32), (r1, 1u8, 8i32), (r2, 4u8, 4i32)] {
            if r_off > 0 && r_off <= pos {
                let cand = pos - r_off;
                if src[cand] == src[pos]
                    && src[cand + 1] == src[pos + 1]
                    && src[cand + 2] == src[pos + 2]
                {
                    let ml = 3 + count_match(src, cand + 3, pos + 3);
                    let score = (ml as i32) * 4 + bonus;
                    if score > best_score {
                        best_score = score;
                        best_ml = ml;
                        best_off = r_off;
                        best_mode = m_code;
                    }
                }
            }
        }

        if best_ml >= 128 {
            return (best_ml, best_off, best_mode, best_score);
        }

        // 2. Check 8-byte long-match hash table
        let cur8 = unsafe { (src.as_ptr().add(pos) as *const u64).read_unaligned() };
        if !t8.is_empty() {
            let h8 = hash8(cur8);
            for &c_u32 in &t8[h8] {
                let cand = c_u32 as usize;
                if cand < pos && (cand > 0 || src[0..8] == src[pos..pos + 8]) {
                    let off = pos - cand;
                    if off <= 0xFF_FFFF {
                        let cand8 =
                            unsafe { (src.as_ptr().add(cand) as *const u64).read_unaligned() };
                        if cand8 == cur8 {
                            let ml = 8 + count_match(src, cand + 8, pos + 8);
                            let (m_code, penalty) = if off <= 256 {
                                (2u8, -2i32)
                            } else if off <= 65536 {
                                (2u8, 0i32)
                            } else {
                                (3u8, 4i32)
                            };
                            let score = (ml as i32) * 4 - penalty;
                            if score > best_score {
                                best_score = score;
                                best_ml = ml;
                                best_off = off;
                                best_mode = m_code;
                            }
                        }
                    }
                }
            }
        }

        if best_ml >= 64 {
            return (best_ml, best_off, best_mode, best_score);
        }

        // 3. Check 8-way 4-byte hash table
        let cur4 = cur8 as u32;
        let h4 = hash4(cur4);
        for &c_u32 in &t4[h4] {
            let cand = c_u32 as usize;
            if cand < pos && (cand > 0 || src[0..4] == src[pos..pos + 4]) {
                let off = pos - cand;
                if off <= 0xFF_FFFF {
                    let cand4 = unsafe { (src.as_ptr().add(cand) as *const u32).read_unaligned() };
                    if cand4 == cur4 {
                        let ml = 4 + count_match(src, cand + 4, pos + 4);
                        if off > 65536 && ml < 5 {
                            continue;
                        }
                        let (m_code, penalty) = if off <= 256 {
                            (2u8, -2i32)
                        } else if off <= 4096 {
                            (2u8, 0i32)
                        } else if off <= 65536 {
                            (2u8, 1i32)
                        } else {
                            (3u8, 5i32)
                        };
                        let score = (ml as i32) * 4 - penalty;
                        if score > best_score {
                            best_score = score;
                            best_ml = ml;
                            best_off = off;
                            best_mode = m_code;
                        }
                    }
                }
            }
        }

        // 4. Check 2-way 3-byte hash table for short close matches (offset <= 2048)
        if best_ml < 4 {
            let h3 = hash3(cur4);
            for &c_u32 in &t3[h3] {
                let cand = c_u32 as usize;
                if cand < pos && pos - cand <= 2048 {
                    if src[cand] == src[pos]
                        && src[cand + 1] == src[pos + 1]
                        && src[cand + 2] == src[pos + 2]
                    {
                        let off = pos - cand;
                        let ml = 3;
                        let penalty = if off <= 256 { -1i32 } else { 0i32 };
                        let score = 12 - penalty;
                        if score > best_score {
                            best_score = score;
                            best_ml = ml;
                            best_off = off;
                            best_mode = 2u8;
                        }
                    }
                }
            }
        }

        (best_ml, best_off, best_mode, best_score)
    };

    macro_rules! insert_pos {
        ($p:expr) => {{
            let pos_val = $p;
            let v8 = unsafe { (src.as_ptr().add(pos_val) as *const u64).read_unaligned() };
            let v4 = v8 as u32;
            if use_t8 {
                let h8 = hash8(v8);
                let s8 = (head8[h8] & 3) as usize;
                table8[h8][s8] = pos_val as u32;
                head8[h8] = head8[h8].wrapping_add(1);
            }

            let h4 = hash4(v4);
            let s4 = (head4[h4] & 7) as usize;
            table4[h4][s4] = pos_val as u32;
            head4[h4] = head4[h4].wrapping_add(1);

            let h3 = hash3(v4);
            let s3 = (head3[h3] & 1) as usize;
            table3[h3][s3] = pos_val as u32;
            head3[h3] = head3[h3].wrapping_add(1);
        }};
    }

    while ip < match_limit {
        let (mut ml, mut off, mut mode, mut score) =
            find_best_match(ip, rep0, rep1, rep2, &table8, &table4, &table3);

        insert_pos!(ip);

        if ml < 3 {
            ip += 1;
            continue;
        }

        // 2-step lazy match evaluation if current match isn't long
        if ml < 32 && ip + 1 < match_limit {
            let (ml2, off2, mode2, score2) =
                find_best_match(ip + 1, rep0, rep1, rep2, &table8, &table4, &table3);
            if score2 > score + 3 {
                insert_pos!(ip + 1);
                ip += 1;
                ml = ml2;
                off = off2;
                mode = mode2;
                score = score2;

                if ml < 32 && ip + 1 < match_limit {
                    let (ml3, off3, mode3, score3) =
                        find_best_match(ip + 1, rep0, rep1, rep2, &table8, &table4, &table3);
                    if score3 > score + 3 {
                        insert_pos!(ip + 1);
                        ip += 1;
                        ml = ml3;
                        off = off3;
                        mode = mode3;
                    }
                }
            }
        }

        // Emit sequence: literals [anchor..ip], then match (ml, off, mode)
        let lit_len = ip - anchor;
        if lit_len > 0 {
            lits.extend_from_slice(&src[anchor..ip]);
        }

        let lit_code = if lit_len < 7 {
            lit_len as u8
        } else {
            push_varint(&mut ext_len, lit_len - 7);
            7u8
        };

        let ml_adj = ml - 3;
        let ml_code = if ml_adj < 7 {
            ml_adj as u8
        } else {
            push_varint(&mut ext_len, ml_adj - 7);
            7u8
        };

        let wire_mode = match mode {
            0 => 0u8,
            1 => {
                let tmp = rep1;
                rep1 = rep0;
                rep0 = tmp;
                1u8
            }
            4 => {
                let tmp = rep2;
                rep2 = rep1;
                rep1 = rep0;
                rep0 = tmp;
                off24.push(0u8);
                3u8
            }
            2 => {
                let enc_off = (off - 1) as u16;
                off16_lo.push((enc_off & 0xFF) as u8);
                off16_hi.push((enc_off >> 8) as u8);
                rep2 = rep1;
                rep1 = rep0;
                rep0 = off;
                2u8
            }
            _ => {
                off24.push(((off >> 16) & 0xFF) as u8);
                off16_lo.push((off & 0xFF) as u8);
                off16_hi.push(((off >> 8) & 0xFF) as u8);
                rep2 = rep1;
                rep1 = rep0;
                rep0 = off;
                3u8
            }
        };

        let token = (lit_code << 5) | (ml_code << 2) | wire_mode;
        tokens.push(token);

        let next_ip = ip + ml;
        let step = if ml <= 24 {
            1
        } else if ml <= 128 {
            4
        } else {
            32
        };
        let mut p = ip + 1;
        while p < next_ip.min(match_limit) {
            let v8 = unsafe { (src.as_ptr().add(p) as *const u64).read_unaligned() };
            let v4 = v8 as u32;
            if use_t8 {
                let h8 = hash8(v8);
                let s8 = (head8[h8] & 3) as usize;
                table8[h8][s8] = p as u32;
                head8[h8] = head8[h8].wrapping_add(1);
            }

            let h4 = hash4(v4);
            let s4 = (head4[h4] & 7) as usize;
            table4[h4][s4] = p as u32;
            head4[h4] = head4[h4].wrapping_add(1);

            if step == 1 {
                let h3 = hash3(v4);
                let s3 = (head3[h3] & 1) as usize;
                table3[h3][s3] = p as u32;
                head3[h3] = head3[h3].wrapping_add(1);
            }
            p += step;
        }

        ip = next_ip;
        anchor = ip;
    }

    if anchor < n {
        lits.extend_from_slice(&src[anchor..n]);
    }

    let c_tokens = compress_substream(&tokens, allow_cascaded_lz);
    let c_lits = compress_substream(&lits, allow_cascaded_lz);
    let c_off16_lo = compress_substream(&off16_lo, allow_cascaded_lz);
    let c_off16_hi = compress_substream(&off16_hi, allow_cascaded_lz);
    let c_off24 = compress_substream(&off24, false);
    let c_ext_len = compress_substream(&ext_len, allow_cascaded_lz);

    let vxlz_total = 1
        + c_tokens.len()
        + c_lits.len()
        + c_off16_lo.len()
        + c_off16_hi.len()
        + c_off24.len()
        + c_ext_len.len();

    let pure_rans = vxans_compress(src);
    let pure_rans_len = pure_rans
        .as_ref()
        .map(|v| v.len() + 1)
        .unwrap_or(usize::MAX);

    let min_comp = vxlz_total.min(pure_rans_len);
    if min_comp >= n {
        let mut raw = Vec::with_capacity(n + 1);
        raw.push(BLK_RAW);
        raw.extend_from_slice(src);
        return raw;
    }

    if pure_rans_len < vxlz_total {
        let r_vec = pure_rans.unwrap();
        let mut out = Vec::with_capacity(r_vec.len() + 1);
        out.push(BLK_RANS8);
        out.extend_from_slice(&r_vec);
        return out;
    }

    let mut out = Vec::with_capacity(vxlz_total);
    out.push(BLK_VXLZ);
    out.extend_from_slice(&c_tokens);
    out.extend_from_slice(&c_lits);
    out.extend_from_slice(&c_off16_lo);
    out.extend_from_slice(&c_off16_hi);
    out.extend_from_slice(&c_off24);
    out.extend_from_slice(&c_ext_len);
    out
}

/// Public entry point to compress a micro-block with VortexLZ-ANS (`VXLZ`).
/// Evaluates raw `VXLZ` as well as low-entropy pre-conditioned transforms (`SUBGREEN3/4`, `DELTA1/3/4`).
pub fn vxlz_compress(src: &[u8]) -> Vec<u8> {
    let mut best = vxlz_compress_core(src, true);

    if src.len() >= 256 {
        let raw_e = estimate_entropy_proxy(src);

        // 1. Evaluate Sub-Green color decorrelation (R -= G, B -= G)
        if src.len() % 3 == 0 {
            let sg3_full = apply_subgreen3_encode(src);
            if estimate_entropy_proxy(&sg3_full) * 100 < raw_e * 98 {
                let mut c_sg3 = vxlz_compress_core(&sg3_full, true);
                if c_sg3.len() + 1 < best.len() {
                    let mut wrapped = Vec::with_capacity(c_sg3.len() + 1);
                    wrapped.push(BLK_VXLZ_SUBGREEN3);
                    wrapped.append(&mut c_sg3);
                    best = wrapped;
                }
            }
        }

        if src.len() % 4 == 0 {
            let sg4_full = apply_subgreen4_encode(src);
            if estimate_entropy_proxy(&sg4_full) * 100 < raw_e * 98 {
                let mut c_sg4 = vxlz_compress_core(&sg4_full, true);
                if c_sg4.len() + 1 < best.len() {
                    let mut wrapped = Vec::with_capacity(c_sg4.len() + 1);
                    wrapped.push(BLK_VXLZ_SUBGREEN4);
                    wrapped.append(&mut c_sg4);
                    best = wrapped;
                }
            }
        }

        // 2. Evaluate spatial Stride-Delta pre-filters (Delta-1 for planar, Delta-3 for RGB, Delta-4 for RGBA/float32)
        let mut best_delta_e = raw_e;
        let mut best_stride = 0usize;
        let mut best_delta_buf = Vec::new();

        for &(stride, tag) in &[
            (1usize, BLK_VXLZ_DELTA1),
            (3usize, BLK_VXLZ_DELTA3),
            (4usize, BLK_VXLZ_DELTA4),
        ] {
            let d_buf = apply_delta_stride_encode(src, stride);
            let de = estimate_entropy_proxy(&d_buf);
            if de < best_delta_e {
                best_delta_e = de;
                best_stride = tag as usize;
                best_delta_buf = d_buf;
            }
        }

        if best_stride != 0 {
            let mut c_delta = vxlz_compress_core(&best_delta_buf, true);
            if c_delta.len() + 1 < best.len() {
                let mut wrapped = Vec::with_capacity(c_delta.len() + 1);
                wrapped.push(best_stride as u8);
                wrapped.append(&mut c_delta);
                best = wrapped;
            }
        }
    }

    best
}

/// Decompress a `VXLZ` compressed micro-block directly into `dst`.
pub fn vxlz_decompress_into(src: &[u8], dst: &mut [u8]) -> Result<usize, String> {
    let n = dst.len();
    if n == 0 {
        return Ok(0);
    }
    if src.is_empty() {
        return Err("Empty VXLZ input".into());
    }

    let tag = src[0];
    match tag {
        BLK_RAW => {
            if src.len() - 1 < n {
                return Err("Truncated BLK_RAW payload".into());
            }
            dst.copy_from_slice(&src[1..1 + n]);
            Ok(n)
        }
        BLK_CONST => {
            if src.len() < 2 {
                return Err("Truncated BLK_CONST payload".into());
            }
            dst.fill(src[1]);
            Ok(n)
        }
        BLK_RANS8 => {
            vxans_decompress(&src[1..], dst)?;
            Ok(n)
        }
        BLK_VXLZ_SUBGREEN3 => {
            let written = vxlz_decompress_into(&src[1..], dst)?;
            apply_subgreen3_decode_in_place(&mut dst[..written]);
            Ok(written)
        }
        BLK_VXLZ_SUBGREEN4 => {
            let written = vxlz_decompress_into(&src[1..], dst)?;
            apply_subgreen4_decode_in_place(&mut dst[..written]);
            Ok(written)
        }
        BLK_VXLZ_DELTA1 => {
            let written = vxlz_decompress_into(&src[1..], dst)?;
            apply_delta1_decode_in_place(&mut dst[..written]);
            Ok(written)
        }
        BLK_VXLZ_DELTA3 => {
            let written = vxlz_decompress_into(&src[1..], dst)?;
            apply_delta3_decode_in_place(&mut dst[..written]);
            Ok(written)
        }
        BLK_VXLZ_DELTA4 => {
            let written = vxlz_decompress_into(&src[1..], dst)?;
            apply_delta4_decode_in_place(&mut dst[..written]);
            Ok(written)
        }
        BLK_VXLZ => {
            let mut pos = 1usize;
            let tokens = decompress_substream(src, &mut pos)?;
            let lits = decompress_substream(src, &mut pos)?;
            let off16_lo = decompress_substream(src, &mut pos)?;
            let off16_hi = decompress_substream(src, &mut pos)?;
            let off24 = decompress_substream(src, &mut pos)?;
            let ext_len = decompress_substream(src, &mut pos)?;

            let mut lit_idx = 0usize;
            let mut off16_idx = 0usize;
            let mut off24_idx = 0usize;
            let mut ext_idx = 0usize;

            let mut rep0: usize = 1;
            let mut rep1: usize = 4;
            let mut rep2: usize = 8;
            let mut op: usize = 0;

            let lits_len = lits.len();
            let o16lo_len = off16_lo.len();
            let o16hi_len = off16_hi.len();
            let o24_len = off24.len();

            unsafe {
                let dst_ptr = dst.as_mut_ptr();
                let lits_ptr = lits.as_ptr();
                let off16_lo_ptr = off16_lo.as_ptr();
                let off16_hi_ptr = off16_hi.as_ptr();
                let off24_ptr = off24.as_ptr();

                for &tok in tokens.iter() {
                    let lit_code = (tok >> 5) as usize;
                    let lit_len = if lit_code < 7 {
                        lit_code
                    } else {
                        7 + read_varint(&ext_len, &mut ext_idx)
                    };

                    if lit_len > 0 {
                        // Bounds guards: corrupt streams must yield Err, never OOB writes.
                        if lit_len > n - op || lit_idx + lit_len > lits_len {
                            return Err("VXLZ literal run out of bounds".into());
                        }
                        std::ptr::copy_nonoverlapping(
                            lits_ptr.add(lit_idx),
                            dst_ptr.add(op),
                            lit_len,
                        );
                        lit_idx += lit_len;
                        op += lit_len;
                    }

                    let ml_code = ((tok >> 2) & 0x07) as usize;
                    let match_len = if ml_code < 7 {
                        ml_code + 3
                    } else {
                        10 + read_varint(&ext_len, &mut ext_idx)
                    };

                    let wire_mode = tok & 0x03;
                    let offset = match wire_mode {
                        0 => rep0,
                        1 => {
                            let tmp = rep1;
                            rep1 = rep0;
                            rep0 = tmp;
                            tmp
                        }
                        2 => {
                            if off16_idx >= o16lo_len || off16_idx >= o16hi_len {
                                return Err("VXLZ 16-bit offset stream exhausted".into());
                            }
                            let lo = *off16_lo_ptr.add(off16_idx) as usize;
                            let hi = *off16_hi_ptr.add(off16_idx) as usize;
                            off16_idx += 1;
                            let off = (lo | (hi << 8)) + 1;
                            rep2 = rep1;
                            rep1 = rep0;
                            rep0 = off;
                            off
                        }
                        _ => {
                            if off24_idx >= o24_len {
                                return Err("VXLZ 24-bit offset stream exhausted".into());
                            }
                            let b2 = *off24_ptr.add(off24_idx) as usize;
                            off24_idx += 1;
                            if b2 == 0 {
                                let tmp = rep2;
                                rep2 = rep1;
                                rep1 = rep0;
                                rep0 = tmp;
                                tmp
                            } else {
                                if off16_idx >= o16lo_len || off16_idx >= o16hi_len {
                                    return Err("VXLZ 16-bit offset stream exhausted".into());
                                }
                                let lo = *off16_lo_ptr.add(off16_idx) as usize;
                                let hi = *off16_hi_ptr.add(off16_idx) as usize;
                                off16_idx += 1;
                                let off = (b2 << 16) | (hi << 8) | lo;
                                rep2 = rep1;
                                rep1 = rep0;
                                rep0 = off;
                                off
                            }
                        }
                    };

                    if offset == 0 || offset > op || match_len > n - op {
                        return Err(format!(
                            "VXLZ invalid match: offset {} op {} match_len {} dst {}",
                            offset, op, match_len, n
                        ));
                    }

                    let match_src = dst_ptr.add(op - offset);
                    let match_dst = dst_ptr.add(op);
                    if offset >= match_len {
                        std::ptr::copy_nonoverlapping(match_src, match_dst, match_len);
                    } else if offset == 1 {
                        let val = *match_src;
                        std::ptr::write_bytes(match_dst, val, match_len);
                    } else if offset >= 8 {
                        let mut copied = 0usize;
                        while copied + 8 <= match_len {
                            let v = (match_src.add(copied) as *const u64).read_unaligned();
                            (match_dst.add(copied) as *mut u64).write_unaligned(v);
                            copied += 8;
                        }
                        while copied < match_len {
                            *match_dst.add(copied) = *match_src.add(copied);
                            copied += 1;
                        }
                    } else {
                        for i in 0..match_len {
                            *match_dst.add(i) = *match_src.add(i);
                        }
                    }
                    op += match_len;
                }

                let rem_lits = lits_len - lit_idx;
                if rem_lits > n - op {
                    return Err("VXLZ trailing literals out of bounds".into());
                }
                if rem_lits > 0 {
                    std::ptr::copy_nonoverlapping(lits_ptr.add(lit_idx), dst_ptr.add(op), rem_lits);
                    op += rem_lits;
                }
            }

            if op != n {
                return Err(format!("VXLZ decoded length mismatch: {} != {}", op, n));
            }
            Ok(op)
        }
        _ => Err(format!("Unknown VXLZ block tag: {}", tag)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vxans_roundtrip() {
        let mut data = Vec::with_capacity(10000);
        for i in 0..10000 {
            data.push(if i % 3 == 0 {
                0
            } else if i % 5 == 0 {
                1
            } else {
                (i % 64) as u8
            });
        }
        let comp = vxans_compress(&data).expect("VXANS should compress skewed data");
        assert!(comp.len() < (data.len() * 65) / 100);
        let mut dec = vec![0u8; data.len()];
        vxans_decompress(&comp, &mut dec).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn test_vxlz_subgreen_rgba_roundtrip() {
        let mut rgba = Vec::with_capacity(4096 * 4);
        for i in 0..4096 {
            let g = ((i * 3) % 256) as u8;
            rgba.push(g.wrapping_add(2)); // R strongly correlated with G
            rgba.push(g); // G
            rgba.push(g.wrapping_sub(1)); // B strongly correlated with G
            rgba.push(255); // A
        }
        let comp = vxlz_compress(&rgba);
        let mut dec = vec![0u8; rgba.len()];
        let written = vxlz_decompress_into(&comp, &mut dec).unwrap();
        assert_eq!(written, rgba.len());
        assert_eq!(dec, rgba);
    }

    #[test]
    fn test_vxlz_delta_filters_roundtrip() {
        for stride in [1usize, 3, 4] {
            let mut data = Vec::with_capacity(8192 * stride);
            let mut val = 10u8;
            for i in 0..(8192 * stride) {
                if i % stride == 0 {
                    val = val.wrapping_add(1);
                }
                data.push(val.wrapping_add((i % stride) as u8));
            }
            let comp = vxlz_compress(&data);
            let mut dec = vec![0u8; data.len()];
            let written = vxlz_decompress_into(&comp, &mut dec).unwrap();
            assert_eq!(written, data.len());
            assert_eq!(dec, data);
        }
    }

    #[test]
    fn test_vxlz_repetitive_xml_roundtrip() {
        let pattern = b"   <tile gid=\"0\"/>\n   <tile gid=\"12\"/>\n   <tile gid=\"0\"/>\n";
        let mut raw = Vec::new();
        for _ in 0..5000 {
            raw.extend_from_slice(pattern);
        }
        let comp = vxlz_compress(&raw);
        assert!(comp.len() < 400, "Compressed size {} too large", comp.len());
        let mut dec = vec![0u8; raw.len()];
        let n = vxlz_decompress_into(&comp, &mut dec).unwrap();
        assert_eq!(n, raw.len());
        assert_eq!(dec, raw);
    }

    #[test]
    fn test_vxlz_mixed_structured_roundtrip() {
        let mut data = Vec::new();
        for i in 0..2000u32 {
            data.extend_from_slice(b"function update_player_pos(x, y, z) return ");
            data.extend_from_slice(&i.to_le_bytes());
            data.extend_from_slice(b" end\n");
        }
        let comp = vxlz_compress(&data);
        assert!(comp.len() < data.len() / 5);
        let mut dec = vec![0u8; data.len()];
        vxlz_decompress_into(&comp, &mut dec).unwrap();
        assert_eq!(dec, data);
    }

    /// Adversarial fuzz: truncated / bit-flipped / random-tag inputs must be
    /// rejected with Err — a panic here means an attacker-facing crash (the
    /// Python bridge calls this across FFI, where a panic aborts the process).
    #[test]
    fn test_adversarial_inputs_never_panic() {
        let mut data = Vec::with_capacity(16384);
        let mut val = 7u8;
        for i in 0..16384 {
            val = val.wrapping_add((i as u8) | 1);
            data.push(val);
            if i % 97 == 0 {
                data.extend_from_slice(b"repeated-repeated-repeated-");
            }
        }

        let mut corpus: Vec<Vec<u8>> = vec![vxlz_compress(&data)];
        if let Some(rans) = vxans_compress(&data) {
            corpus.push(rans);
        }
        if let Some(chunked) = vxans_compress_chunked(&data) {
            corpus.push(chunked);
        }

        for comp in &corpus {
            // Every truncation point must produce Err or a clean value, never a panic.
            let cuts: Vec<usize> = (1..comp.len()).step_by((comp.len() / 97).max(1)).collect();
            for cut in cuts {
                let mut dec = vec![0u8; data.len()];
                let _ = vxlz_decompress_into(&comp[..cut], &mut dec);
                let _ = vxans_decompress(&comp[..cut], &mut dec);
                let _ = vxans_decompress_chunked(&comp[..cut], &mut dec);
            }
            // Seeded single-bit flips across the whole stream.
            let mut state: u64 = 0x9E3779B97F4A7C15;
            for i in 0..comp.len() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bit = (state >> 33) % 8;
                let mut bad = comp.clone();
                bad[i] ^= 1 << bit;
                let mut dec = vec![0u8; data.len()];
                let _ = vxlz_decompress_into(&bad, &mut dec);
            }
        }

        // Every possible block tag followed by pseudo-random garbage.
        let mut state: u64 = 0xDEADBEEFCAFEBABE;
        for tag in 0u8..=24 {
            for len in [0usize, 1, 8, 64, 600] {
                let mut bad = vec![tag];
                for _ in 0..len {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    bad.push((state >> 33) as u8);
                }
                let mut dec = vec![0u8; 4096];
                let _ = vxlz_decompress_into(&bad, &mut dec);
                let _ = vxans_decompress_chunked(&bad, &mut dec);
            }
        }
    }
}
