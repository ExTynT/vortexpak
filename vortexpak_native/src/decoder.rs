//! VortexPak Parallel Native Decoder
//! Decompresses micro-blocks in parallel across CPU cores using Rayon.

use rayon::prelude::*;
use std::io::Read;

pub const STRAT_RAW_BYPASS: u32 = 0;
pub const STRAT_LZ4_FAST: u32 = 1;
pub const STRAT_ZSTD_3: u32 = 2;
pub const STRAT_ZSTD_6: u32 = 3;
pub const STRAT_ZSTD_12: u32 = 4;
pub const STRAT_ZSTD_19: u32 = 5;
pub const STRAT_TOKEN_FOLD_ZSTD6: u32 = 6;
pub const STRAT_TOKEN_FOLD_ZSTD19: u32 = 7;
pub const STRAT_VXLZ: u32 = 8;

#[derive(Debug, Clone)]
pub struct MicroBlockDesc<'a> {
    pub block_id: u32,
    pub strategy_id: u32,
    pub raw_size: usize,
    pub compressed_payload: &'a [u8],
    pub token_pattern: &'a [u8],
    pub token_byte: u8,
}

/// Decompress a single micro-block directly into an existing buffer slice without heap reallocations.
pub fn decompress_block_into(block: &MicroBlockDesc, dst: &mut [u8]) -> Result<usize, String> {
    match block.strategy_id {
        STRAT_RAW_BYPASS => {
            let len = block.compressed_payload.len();
            if len > dst.len() {
                return Err(format!(
                    "Raw bypass dst buffer too small: dst {} B, needed {} B",
                    dst.len(),
                    len
                ));
            }
            dst[..len].copy_from_slice(block.compressed_payload);
            Ok(len)
        }
        STRAT_LZ4_FAST => {
            if block.compressed_payload.is_empty() {
                return Ok(0);
            }
            if block.compressed_payload.len() >= 4
                && &block.compressed_payload[0..4] == &[0x04, 0x22, 0x4D, 0x18]
            {
                // LZ4 Frame format
                let mut decoder = lz4_flex::frame::FrameDecoder::new(block.compressed_payload);
                let mut written = 0;
                while written < dst.len() {
                    let n = decoder
                        .read(&mut dst[written..])
                        .map_err(|e| format!("LZ4 frame decode error: {e:?}"))?;
                    if n == 0 {
                        break;
                    }
                    written += n;
                }
                Ok(written)
            } else {
                // Raw LZ4 block format
                lz4_flex::decompress_into(block.compressed_payload, dst)
                    .map_err(|e| format!("LZ4 decode error: {e:?}"))
            }
        }
        STRAT_ZSTD_3 | STRAT_ZSTD_6 | STRAT_ZSTD_12 | STRAT_ZSTD_19 => {
            if block.compressed_payload.is_empty() && block.raw_size == 0 {
                return Ok(0);
            }
            zstd::bulk::decompress_to_buffer(block.compressed_payload, dst)
                .map_err(|e| format!("Zstd bulk decode error: {e:?}"))
        }
        STRAT_TOKEN_FOLD_ZSTD6 | STRAT_TOKEN_FOLD_ZSTD19 => {
            if block.compressed_payload.is_empty() && block.raw_size == 0 {
                return Ok(0);
            }
            if block.token_pattern.is_empty() {
                return zstd::bulk::decompress_to_buffer(block.compressed_payload, dst)
                    .map_err(|e| format!("Zstd bulk token-fold decode error: {e:?}"));
            }
            // Decompress folded bytes using bulk decompress
            let folded = zstd::bulk::decompress(block.compressed_payload, block.raw_size)
                .map_err(|e| format!("Zstd token-fold bulk error: {e:?}"))?;

            let mut out_idx = 0;
            let pat = block.token_pattern;
            let tok = block.token_byte;
            for &b in &folded {
                if b == tok {
                    if out_idx + pat.len() > dst.len() {
                        return Err(format!(
                            "Destination buffer overflow during token unfolding: dst {} B, needed > {} B",
                            dst.len(),
                            out_idx + pat.len()
                        ));
                    }
                    dst[out_idx..out_idx + pat.len()].copy_from_slice(pat);
                    out_idx += pat.len();
                } else {
                    if out_idx >= dst.len() {
                        return Err("Destination buffer overflow during token unfolding".into());
                    }
                    dst[out_idx] = b;
                    out_idx += 1;
                }
            }
            Ok(out_idx)
        }
        STRAT_VXLZ => {
            if block.compressed_payload.is_empty() && block.raw_size == 0 {
                return Ok(0);
            }
            crate::vxlz::vxlz_decompress_into(block.compressed_payload, dst)
        }
        unknown => Err(format!("Unknown block strategy: {unknown}")),
    }
}

/// Decompress a single micro-block into a byte vector.
pub fn decompress_block(block: &MicroBlockDesc) -> Result<Vec<u8>, String> {
    let cap = if block.strategy_id == STRAT_RAW_BYPASS {
        block.compressed_payload.len().max(block.raw_size)
    } else {
        block.raw_size
    };
    let mut out = vec![0u8; cap];
    let written = decompress_block_into(block, &mut out)?;
    out.truncate(written);
    Ok(out)
}

/// Decompress multiple micro-blocks in parallel using Rayon thread pool.
pub fn decompress_blocks_parallel(blocks: &[MicroBlockDesc]) -> Result<Vec<Vec<u8>>, String> {
    blocks.par_iter().map(|b| decompress_block(b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bypass_decompress() {
        let raw = b"Hello, VortexPak bypass lane!";
        let desc = MicroBlockDesc {
            block_id: 0,
            strategy_id: STRAT_RAW_BYPASS,
            raw_size: raw.len(),
            compressed_payload: raw,
            token_pattern: b"",
            token_byte: 0,
        };
        let res = decompress_block(&desc).unwrap();
        assert_eq!(res, raw);
    }

    #[test]
    fn test_zstd_parallel_decompress() {
        let raw1 = b"Texture stream block 1 repeated repeated repeated repeated repeated";
        let raw2 = b"Texture stream block 2 different data here different data here";

        let comp1 = zstd::encode_all(&raw1[..], 3).unwrap();
        let comp2 = zstd::encode_all(&raw2[..], 6).unwrap();

        let blocks = vec![
            MicroBlockDesc {
                block_id: 0,
                strategy_id: STRAT_ZSTD_3,
                raw_size: raw1.len(),
                compressed_payload: &comp1,
                token_pattern: b"",
                token_byte: 0,
            },
            MicroBlockDesc {
                block_id: 1,
                strategy_id: STRAT_ZSTD_6,
                raw_size: raw2.len(),
                compressed_payload: &comp2,
                token_pattern: b"",
                token_byte: 0,
            },
        ];

        let results = decompress_blocks_parallel(&blocks).unwrap();
        assert_eq!(results[0], raw1);
        assert_eq!(results[1], raw2);
    }

    #[test]
    fn test_lz4_frame_decompress() {
        let raw = b"LZ4 frame compression test data with repetitive strings repetitive strings";
        let mut frame_comp = Vec::new();
        {
            use std::io::Write;
            let mut enc = lz4_flex::frame::FrameEncoder::new(&mut frame_comp);
            enc.write_all(raw).unwrap();
            enc.finish().unwrap();
        }

        let desc = MicroBlockDesc {
            block_id: 0,
            strategy_id: STRAT_LZ4_FAST,
            raw_size: raw.len(),
            compressed_payload: &frame_comp,
            token_pattern: b"",
            token_byte: 0,
        };
        let res = decompress_block(&desc).unwrap();
        assert_eq!(res, raw);
    }

    /// Corrupt payloads handed to decompress_block must return Err, never panic
    /// (Python calls this across FFI, where a panic aborts the host process).
    #[test]
    fn test_adversarial_blocks_never_panic() {
        let mut state: u64 = 0x0123456789ABCDEF;
        let mut garbage = |n: usize| -> Vec<u8> {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                v.push((state >> 33) as u8);
            }
            v
        };
        for strategy in [
            STRAT_VXLZ,
            STRAT_LZ4_FAST,
            STRAT_ZSTD_3,
            STRAT_ZSTD_19,
            STRAT_TOKEN_FOLD_ZSTD6,
            42u32,
        ] {
            for payload_len in [0usize, 1, 9, 64, 777] {
                let payload = garbage(payload_len);
                let desc = MicroBlockDesc {
                    block_id: 0,
                    strategy_id: strategy,
                    raw_size: 4096,
                    compressed_payload: &payload,
                    token_pattern: b"ab",
                    token_byte: b'a',
                };
                let res = decompress_block(&desc);
                assert!(res.is_err() || res.unwrap().len() <= 4096);
            }
        }
        // Raw bypass larger than dst must Err, not panic.
        let desc = MicroBlockDesc {
            block_id: 0,
            strategy_id: STRAT_RAW_BYPASS,
            raw_size: 8,
            compressed_payload: b"0123456789",
            token_pattern: b"",
            token_byte: 0,
        };
        let mut small = [0u8; 8];
        assert!(decompress_block_into(&desc, &mut small).is_err());
    }
}
