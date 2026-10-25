//! Seekable Virtual File System (VFS) for VortexPak (.vpak) archives.
//! Allows direct random-access reading of single assets without decompressing
//! the rest of the archive or other semantic lanes.

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt};

use crate::decoder::{self, MicroBlockDesc};
use crate::vxtx::{self, TX_ALPHA_STRIP, TX_PLANAR_CHW, TX_ROW_RLE};

pub const VXPK_MAGIC: &[u8; 6] = b"VXPK\x01\x01";

#[derive(Debug, Clone)]
pub struct VfsAssetEntry {
    pub rel_path: String,
    pub raw_size: usize,
    pub is_texture_tensor: bool,
    pub tx_flags: u32,
    pub tx_shape: Vec<usize>,
    pub tx_mode: String,
    pub tx_rle_unique_rows: usize,
    pub block_id: usize,
    pub offset_in_block: usize,
    pub length_in_block: usize,
    pub aux_block_id: i32,
    pub aux_offset: usize,
    pub aux_length: usize,
    pub stripe_spans: Vec<(usize, usize, usize)>, // (block_id, offset, length)
}

#[derive(Debug, Clone)]
pub struct VfsBlockInfo {
    pub block_id: usize,
    pub lane_name: String,
    pub strategy_id: u32,
    pub raw_size: usize,
    pub comp_size: usize,
    pub file_offset: u64,
    pub token_pattern: Vec<u8>,
    pub token_byte: u8,
}

pub struct VortexVfs<R: Read + Seek> {
    reader: R,
    pub mode: String,
    pub total_raw_bytes: usize,
    pub entries: Vec<VfsAssetEntry>,
    pub path_index: HashMap<String, usize>,
    pub blocks: Vec<VfsBlockInfo>,
    // In-memory genuine LRU block cache
    block_cache: HashMap<usize, Vec<u8>>,
    lru_queue: VecDeque<usize>,
    cache_capacity: usize,
}

impl VortexVfs<File> {
    /// Open a .vpak archive from disk in seekable streaming mode.
    pub fn open_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open vpak file: {e}"))?;
        Self::new(file)
    }
}

/// Texture tensor flag: payload is GPU block data (see docs/FORMAT.md).
const TX_BCN: u32 = 1 << 3;

fn decode_bcn_entry(entry: &VfsAssetEntry, payload: &[u8]) -> Result<Vec<u8>, String> {
    let (h, w) = match entry.tx_shape.as_slice() {
        [h, w, ..] => (*h, *w),
        _ => return Err("BCN entry without a (h, w, c) shape".into()),
    };
    let fmt = match entry.tx_mode.as_str() {
        "BC1" => crate::bcn::BCN_BC1,
        "BC4" => crate::bcn::BCN_BC4,
        "BC5" => crate::bcn::BCN_BC5,
        "BC7" => crate::bcn::BCN_BC7,
        "BCX" => crate::bcn::BCN_BCX,
        other => return Err(format!("unknown BCN tx_mode {other:?}")),
    };
    crate::bcn::bcn_decode(fmt, payload, w, h)
}

impl<R: Read + Seek> VortexVfs<R> {
    pub fn new(mut reader: R) -> Result<Self, String> {
        // 1. Read and verify magic
        let mut magic = [0u8; 6];
        reader
            .read_exact(&mut magic)
            .map_err(|e| format!("Failed to read magic: {e}"))?;
        if &magic != VXPK_MAGIC {
            return Err("Invalid VortexPak magic header".into());
        }

        // 2. Read meta length
        let meta_len = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| format!("Failed to read meta_len: {e}"))? as usize;

        // 3. Read compressed metadata
        let mut meta_comp = vec![0u8; meta_len];
        reader
            .read_exact(&mut meta_comp)
            .map_err(|e| format!("Failed to read meta_comp: {e}"))?;

        // 4. Decompress metadata (supports proprietary VXLZ-ANS header or legacy Zstd header)
        let meta_combined = if meta_comp.len() >= 8 && &meta_comp[0..4] == b"VXLZ" {
            let raw_meta_len =
                u32::from_le_bytes([meta_comp[4], meta_comp[5], meta_comp[6], meta_comp[7]])
                    as usize;
            let mut buf = vec![0u8; raw_meta_len];
            crate::vxlz::vxlz_decompress_into(&meta_comp[8..], &mut buf)
                .map_err(|e| format!("VXLZ header decompress failed: {e}"))?;
            buf
        } else {
            let mut buf = Vec::new();
            let mut zdec = zstd::stream::Decoder::new(&meta_comp[..])
                .map_err(|e| format!("Zstd header init failed: {e}"))?;
            zdec.read_to_end(&mut buf)
                .map_err(|e| format!("Zstd header decompress failed: {e}"))?;
            buf
        };

        if meta_combined.len() < 4 {
            return Err("Corrupt header payload".into());
        }

        let json_len = u32::from_le_bytes([
            meta_combined[0],
            meta_combined[1],
            meta_combined[2],
            meta_combined[3],
        ]) as usize;

        let meta_json_bytes = &meta_combined[4..4 + json_len];
        let header_dict: serde_json::Value = serde_json::from_slice(meta_json_bytes)
            .map_err(|e| format!("Failed to parse header JSON: {e}"))?;

        let mode = header_dict["mode"].as_str().unwrap_or("engine").to_string();
        let total_raw_bytes = header_dict["total_raw_bytes"].as_u64().unwrap_or(0) as usize;

        // 5. Parse blocks and compute file offsets
        let mut cur_file_offset: u64 = 10 + (meta_len as u64);
        let mut blocks = Vec::new();

        if let Some(blocks_arr) = header_dict["blocks"].as_array() {
            for b_val in blocks_arr {
                let row = b_val.as_array().ok_or("Invalid block row")?;
                let b_id = row[0].as_u64().unwrap_or(0) as usize;
                let lane_name = row[1].as_str().unwrap_or("").to_string();
                let strat_id = row[2].as_u64().unwrap_or(0) as u32;
                let raw_sz = row[3].as_u64().unwrap_or(0) as usize;
                let comp_sz = row[4].as_u64().unwrap_or(0) as usize;
                let pat_hex = row[5].as_str().unwrap_or("");
                let tok_hex = row[6].as_str().unwrap_or("");

                let pat = if !pat_hex.is_empty() {
                    hex_decode(pat_hex)?
                } else {
                    Vec::new()
                };
                let tok = if !tok_hex.is_empty() {
                    hex_decode(tok_hex)?[0]
                } else {
                    0
                };

                blocks.push(VfsBlockInfo {
                    block_id: b_id,
                    lane_name,
                    strategy_id: strat_id,
                    raw_size: raw_sz,
                    comp_size: comp_sz,
                    file_offset: cur_file_offset,
                    token_pattern: pat,
                    token_byte: tok,
                });

                cur_file_offset += comp_sz as u64;
            }
        }

        // 6. Parse asset entries
        let mut entries = Vec::new();
        let mut path_index = HashMap::new();

        if let Some(entries_arr) = header_dict["entries"].as_array() {
            for (idx, e_val) in entries_arr.iter().enumerate() {
                let row = e_val.as_array().ok_or("Invalid entry row")?;
                let rel_path = row[0].as_str().unwrap_or("").to_string();
                let raw_size = row[1].as_u64().unwrap_or(0) as usize;
                let is_texture = row[2].as_u64().unwrap_or(0) != 0;
                let tx_flags = row[3].as_u64().unwrap_or(0) as u32;
                let tx_shape: Vec<usize> = row[4]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_u64().map(|x| x as usize))
                            .collect()
                    })
                    .unwrap_or_default();
                let tx_mode = row[5].as_str().unwrap_or("").to_string();
                let tx_rle_u = row[6].as_u64().unwrap_or(0) as usize;
                let block_id = row[7].as_u64().unwrap_or(0) as usize;
                let offset_in_block = row[8].as_u64().unwrap_or(0) as usize;
                let length_in_block = row[9].as_u64().unwrap_or(0) as usize;
                let aux_block_id = row[10].as_i64().unwrap_or(-1) as i32;
                let aux_offset = row[11].as_u64().unwrap_or(0) as usize;
                let aux_length = row[12].as_u64().unwrap_or(0) as usize;

                let stripe_spans: Vec<(usize, usize, usize)> = row[13]
                    .as_array()
                    .map(|spans| {
                        spans
                            .iter()
                            .filter_map(|s| {
                                let triple = s.as_array()?;
                                Some((
                                    triple[0].as_u64()? as usize,
                                    triple[1].as_u64()? as usize,
                                    triple[2].as_u64()? as usize,
                                ))
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                path_index.insert(rel_path.clone(), idx);
                entries.push(VfsAssetEntry {
                    rel_path,
                    raw_size,
                    is_texture_tensor: is_texture,
                    tx_flags,
                    tx_shape,
                    tx_mode,
                    tx_rle_unique_rows: tx_rle_u,
                    block_id,
                    offset_in_block,
                    length_in_block,
                    aux_block_id,
                    aux_offset,
                    aux_length,
                    stripe_spans,
                });
            }
        }

        Ok(Self {
            reader,
            mode,
            total_raw_bytes,
            entries,
            path_index,
            blocks,
            block_cache: HashMap::new(),
            lru_queue: VecDeque::new(),
            cache_capacity: 16,
        })
    }

    /// Read only the requested microblock from disk and decompress it.
    fn load_block(&mut self, block_id: usize) -> Result<&[u8], String> {
        if !self.block_cache.contains_key(&block_id) {
            let b_info = self
                .blocks
                .get(block_id)
                .ok_or_else(|| format!("Block id {block_id} out of range"))?
                .clone();

            // Seek directly to block's offset in .vpak file
            self.reader
                .seek(SeekFrom::Start(b_info.file_offset))
                .map_err(|e| format!("Seek failed: {e}"))?;

            let mut comp_buf = vec![0u8; b_info.comp_size];
            self.reader
                .read_exact(&mut comp_buf)
                .map_err(|e| format!("Read block failed: {e}"))?;

            let desc = MicroBlockDesc {
                block_id: b_info.block_id as u32,
                strategy_id: b_info.strategy_id,
                raw_size: b_info.raw_size,
                compressed_payload: &comp_buf,
                token_pattern: &b_info.token_pattern,
                token_byte: b_info.token_byte,
            };

            let decompressed = decoder::decompress_block(&desc)?;

            // Genuine LRU eviction
            while self.block_cache.len() >= self.cache_capacity {
                if let Some(oldest) = self.lru_queue.pop_front() {
                    self.block_cache.remove(&oldest);
                } else {
                    break;
                }
            }

            self.block_cache.insert(block_id, decompressed);
            self.lru_queue.push_back(block_id);
        } else {
            // Re-order hit to back of LRU queue
            if let Some(pos) = self.lru_queue.iter().position(|&id| id == block_id) {
                self.lru_queue.remove(pos);
            }
            self.lru_queue.push_back(block_id);
        }

        Ok(self.block_cache.get(&block_id).unwrap().as_slice())
    }

    /// Check if asset exists in the virtual file system (handles both / and \).
    pub fn exists(&self, rel_path: &str) -> bool {
        let clean = rel_path.replace('\\', "/");
        self.path_index.contains_key(&clean)
    }

    /// Get raw size of an asset without decompressing.
    pub fn get_asset_size(&self, rel_path: &str) -> Option<usize> {
        let clean = rel_path.replace('\\', "/");
        self.path_index
            .get(&clean)
            .map(|&idx| self.entries[idx].raw_size)
    }

    /// Read a single specific asset on-demand by relative path.
    /// Does NOT touch or decompress any other blocks in the archive!
    /// Supports both forward slashes and Windows backslashes seamlessly.
    pub fn read_asset(
        &mut self,
        rel_path: &str,
        materialize_rgba: bool,
    ) -> Result<Vec<u8>, String> {
        let clean_path = rel_path.replace('\\', "/");
        let entry_idx = *self
            .path_index
            .get(&clean_path)
            .ok_or_else(|| format!("Asset not found in archive: {rel_path}"))?;

        let entry = self.entries[entry_idx].clone();

        // Multi-stripe large asset
        if !entry.stripe_spans.is_empty() {
            let mut out = Vec::with_capacity(entry.raw_size);
            for &(b_id, off, len) in &entry.stripe_spans {
                let block_bytes = self.load_block(b_id)?;
                if off + len > block_bytes.len() {
                    return Err(format!(
                        "Stripe span {off}..{off}+{len} exceeds block size {}",
                        block_bytes.len()
                    ));
                }
                out.extend_from_slice(&block_bytes[off..off + len]);
            }
            if materialize_rgba && entry.is_texture_tensor && entry.tx_flags & TX_BCN != 0 {
                return decode_bcn_entry(&entry, &out);
            }
            return Ok(out);
        }

        let block_bytes = self.load_block(entry.block_id)?;
        if entry.offset_in_block + entry.length_in_block > block_bytes.len() {
            return Err(format!(
                "Asset offset {}..{} exceeds block size {}",
                entry.offset_in_block,
                entry.offset_in_block + entry.length_in_block,
                block_bytes.len()
            ));
        }
        let raw_slice =
            &block_bytes[entry.offset_in_block..entry.offset_in_block + entry.length_in_block];

        if entry.is_texture_tensor && entry.tx_flags & TX_BCN != 0 {
            // GPU block payload: stored blocks as-is, or decoded texels.
            return if materialize_rgba {
                decode_bcn_entry(&entry, raw_slice)
            } else {
                Ok(raw_slice.to_vec())
            };
        }

        if entry.is_texture_tensor {
            let h = if !entry.tx_shape.is_empty() {
                entry.tx_shape[0]
            } else {
                0
            };
            let w = if entry.tx_shape.len() > 1 {
                entry.tx_shape[1]
            } else {
                0
            };
            let orig_c = if entry.tx_shape.len() > 2 {
                entry.tx_shape[2]
            } else {
                1
            };

            let cur_c = if entry.tx_flags & TX_ALPHA_STRIP != 0 {
                3
            } else {
                orig_c
            };
            let row_bytes = w * cur_c;

            let mut tensor_buf = if entry.tx_flags & TX_ROW_RLE != 0 {
                let counts_bytes = entry.tx_rle_unique_rows * 2;
                if raw_slice.len() < counts_bytes {
                    return Err("Truncated Row-RLE counts in block".into());
                }
                let counts_raw = &raw_slice[..counts_bytes];
                let unique_raw = &raw_slice[counts_bytes..];

                let mut counts = Vec::with_capacity(entry.tx_rle_unique_rows);
                for i in 0..entry.tx_rle_unique_rows {
                    counts.push(u16::from_le_bytes([
                        counts_raw[i * 2],
                        counts_raw[i * 2 + 1],
                    ]));
                }

                let mut expanded = vec![0u8; h * row_bytes];
                vxtx::row_rle_decode(&counts, unique_raw, row_bytes, &mut expanded);
                expanded
            } else if entry.tx_flags & TX_PLANAR_CHW != 0 {
                let mut interleaved = vec![0u8; h * w * cur_c];
                vxtx::planar_decode(raw_slice, &mut interleaved, w, h, cur_c);
                interleaved
            } else {
                raw_slice.to_vec()
            };

            if materialize_rgba && (entry.tx_flags & TX_ALPHA_STRIP != 0) {
                let mut rgba = vec![0u8; h * w * 4];
                vxtx::alpha_strip_unpack(&tensor_buf, &mut rgba);
                tensor_buf = rgba;
            }

            Ok(tensor_buf)
        } else {
            Ok(raw_slice.to_vec())
        }
    }

    /// List all asset paths in the archive.
    pub fn list_assets(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.rel_path.as_str()).collect()
    }
}

fn hex_decode(s: &str) -> Result<Vec<u8>, String> {
    if s.len() % 2 != 0 {
        return Err("Hex string has odd length".into());
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| e.to_string()))
        .collect()
}
