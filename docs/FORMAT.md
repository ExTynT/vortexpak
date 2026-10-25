# VXPK Container Format Specification

**VXPK (`.vpak`) — VortexPak archive container, format revision `1.1`** (magic `VXPK\x01\x01`).

This document describes the on-disk layout produced by `VortexArchive.to_bytes()`
(`vortexpak/codec.py`) and parsed by `VortexArchive.from_bytes()` / `VortexVFS`.
All integers are **little-endian**. The reference parser rejects any deviation
with `VXPKError`; malformed inputs must never crash the decoder.

## 1. Container layout

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 6 | Magic: ASCII `VXPK` + format bytes `0x01 0x01` |
| 6 | 4 | `meta_comp_len` (u32) — size of the compressed meta section |
| 10 | `meta_comp_len` | `meta_comp` — compressed meta section (see §2) |
| 10 + `meta_comp_len` | rest of file | Micro-block payloads, concatenated in `blocks[]` order (see §4) |

## 2. Meta section (`meta_comp`)

Two variants, distinguished by prefix:

1. **VXLZ meta** (native builds, default): `b"VXLZ"` + u32 `raw_meta_len` +
   VXLZ-ANS stream that inflates to exactly `raw_meta_len` bytes.
2. **zstd meta** (pure-Python fallback): a raw zstd frame.

`meta_comp` inflates to `meta_combined`:

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 4 | `json_len` (u32) |
| 4 | `json_len` | `meta_json` — UTF-8 JSON header (see §3) |
| 4 + `json_len` | 8 × N | XXH3-64 checksum table, one u64 per entry, in `entries[]` order |

## 3. Meta JSON header

```json
{
  "mode": "strict" | "engine",
  "total_raw_bytes": <int>,
  "total_pixel_raw_bytes": <int>,
  "entries":  [ <entry row>, ... ],
  "blocks":   [ <block row>, ... ]
}
```

**Entry row** (exactly 14 fields):

```
[rel_path, raw_size, is_texture_tensor, tx_flags, tx_shape, tx_mode,
 tx_rle_unique_rows, block_id, offset_in_block, length_in_block,
 aux_block_id, aux_offset, aux_length, stripe_spans]
```

* `rel_path` — `/`-separated path inside the archive.
* `block_id` / `offset_in_block` / `length_in_block` — where the asset's
  payload lives after decompressing that block.
* `aux_block_id` — `-1` = none; otherwise points at a companion block
  (strict mode: PNG metadata chunks split from IDAT streams).
* `stripe_spans` — list of `[block_id, offset, length]` triples. Non-empty for
  assets striped across several parallel blocks (e.g. huge tilemap XML).

**Block row** (exactly 7 fields):

```
[block_id, lane_name, strategy_id, raw_size, comp_size,
 token_pattern_hex, token_byte_hex]
```

* `lane_name` — semantic lane label (e.g. `LANE_TEXTURES`, `LANE_MEDIA`).
* `token_pattern_hex` / `token_byte_hex` — hex strings, `""` when unused
  (token-fold unfolding, §5 strategies 6–7).
* `comp_size` — exact payload size in the concatenated block region.

### Integrity

Per-entry XXH3-64 over the raw asset bytes; `codec.verify_archive()` (and
`vortexpak info`) checks 100 % of entries.

## 4. Micro-block payloads

A block decompresses (per `strategy_id`) to exactly `raw_size` bytes.

### Strategy IDs (`strategy_id`)

| ID | Name | Payload format |
| --- | --- | --- |
| 0 | `RAW_BYPASS` | stored verbatim (incompressible media bypass) |
| 1 | `LZ4_FAST` | LZ4 frame (`04 22 4D 18` magic) or raw LZ4 block |
| 2–5 | `ZSTD_3/6/12/19` | raw zstd frame |
| 6–7 | `TOKEN_FOLD_ZSTD6/ZSTD19` | zstd frame of token-folded bytes |
| 8 | `VXLZ` | proprietary VortexLZ-ANS stream (§6) |
| 9 | `ZSTD6_DICT` | zstd frame seeded with the lane's trained dictionary block |
| 10 | `ZSTD19_DICT` | reserved: zstd-19 + dictionary frame (decoder-identical to 9) |

**Lane dictionaries**: block rows carry an 8th field `dict_block_id`
(`-1` = none; 7-field rows from older archives remain valid). It points to a
normal block holding the lane's trained zstd dictionary (stored as a
zstd-19 frame). Blocks with strategy 9/10 decompress with that dictionary;
the dictionary block itself never uses a dictionary strategy (no chaining),
so random access is fully preserved.

### VXTX texture tensor flags (`tx_flags`, engine mode)

| Bit | Name | Meaning |
| --- | --- | --- |
| 0 | `TX_ALPHA_STRIP` | RGBA source was 100 % opaque (`A==255`), stored as RGB |
| 1 | `TX_ROW_RLE` | duplicate scanlines deduplicated (`counts` + unique rows) |
| 2 | `TX_PLANAR_CHW` | interleaved pixels stored as planar (C,H,W) |
| 3 | `TX_BCN` | payload is GPU block data (BC1/BC4/BC5), see below |
| — | `0` | dense raw tensor |

### VXTX-GPU block compression (`TX_BCN`)

When `tx_flags` has bit 3 set, the entry payload is GPU block data:

* `tx_mode` carries the format name. `BC1`, `BC4`, `BC5` and `BC7` are
  bit-exact to the D3D11/Vulkan specification and upload to the GPU as-is
  (`glCompressedTexImage2D` / `vkCmdCopyBufferToImage`). `BCX` is a
  storage-only format (see below) and must be decoded before upload.
* `tx_shape` = (height, width, channels) of the decoded tensor:
  4 for BC1/BC7/BCX, 2 for BC5, 1 for BC4. Dimensions are multiples of 4.
* `xxh64` is the checksum of the DECODED tensor (lossy format), so
  integrity verification works uniformly for all asset classes.
* `materialize_full_rgba=False` reads return the stored payload;
  `True` returns the decoded tensor (BC5 yields an (H, W, 2) array).

Conformance is tested in CI: every format is decoded by an independent
reference decoder (Pillow's BCn decoder via a DDS wrapper) and must match
our decoder bit-for-bit (`tests/test_bcn.py::TestReferenceDecoderConformance`).

* **BC4**: palette index 0 = endpoint 0, 1 = endpoint 1, 2..7 = interpolants
  (or 4 interpolants + 0 + 255 when endpoint 0 <= endpoint 1).
* **BC5**: each 16-byte block is the BC4 block for R followed by the BC4
  block for G.

#### BC7 (mode 6, `tx_mode = "BC7"`)

Spec mode 6: 8bpp RGBA, single subset, 7-bit endpoints with one P-bit per
endpoint, 4-bit indices with the standard weight table
(0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64); pixel 0 is the
anchor (index MSB implicitly 0). The encoder only emits mode 6; the decoder
returns transparent black for any other mode. Use BC7 for alpha or
gradient content (BC1's 1-bit alpha collapses smooth alpha), BC1 for opaque
4bpp.

#### BCX (storage-only, `tx_mode = "BCX"`)

Per-block mix of BC1 (8 B) and BC7 mode-6 (16 B) blocks for disk size:
`u32 bitmap_bytes` + bitmap (1 bit per block, 1 = BC7) + blocks in raster
order. GPUs cannot sample a mixed texture, so BCX is decoded to RGBA (or
transcoded) at load time. `--gpu auto` never selects it; it is opt-in
(`--gpu bcx`).

#### Closed-loop RDO (BC1, cook-time)

`encode_bcn(..., rdo=x)` (native, BC1): per-block residual pre-compensation —
the encoder iteratively nudges source pixels by a fraction of the BC1
reconstruction residual and keeps the candidate that minimizes POST-codec
error against the original. Optimizes the right objective (post-transform
distortion, Oodle-Texture-style) at cook time; decode path is unchanged.

## 5. VXLZ-ANS stream (`strategy_id` 8, meta variant 1)

First byte selects the block type:

| Tag | Name | Payload |
| --- | --- | --- |
| 0 | `BLK_RAW` | verbatim `dst_len` bytes |
| 1 | `BLK_CONST` | one byte, repeated |
| 2 | `BLK_RANS8` | VXANS stream (below) |
| 3 | `BLK_VXLZ` | 6 LZ sub-streams (tokens, literals, off16_lo, off16_hi, off24, ext_len) |
| 4 | `BLK_VXLZ_SUBGREEN3` | nested VXLZ block + R-=G, B-=G inverse filter |
| 5 | `BLK_VXLZ_SUBGREEN4` | nested VXLZ block + RGBA sub-green inverse filter |
| 6 | `BLK_VXLZ_DELTA1` | nested VXLZ block + stride-1 delta inverse filter |
| 7 | `BLK_VXLZ_DELTA3` | nested VXLZ block + stride-3 delta inverse filter |
| 8 | `BLK_VXLZ_DELTA4` | nested VXLZ block + stride-4 delta inverse filter |

Sub-stream framing (used by `BLK_VXLZ` and chunked VXANS):

| Mode | Payload |
| --- | --- |
| 0 `SUB_RAW` | u32 `raw_len` + u32 `comp_len` + verbatim bytes |
| 1 `SUB_CONST` | u32 `raw_len` + one repeated byte |
| 2 `SUB_RANS8` | u32 `raw_len` + u32 `comp_len` + VXANS stream |
| 3 `SUB_VXLZ1` | u32 `raw_len` + u32 `comp_len` + nested VXLZ block |
| 4 `SUB_RANS8_CHUNKED` | u32 `raw_len` + u32 `comp_len` + 32 KiB-chunked VXANS |

### VXANS entropy stream

8-way interleaved 12-bit rANS (`SCALE = 4096`, `RANS_L = 2^16`):

1. 32-byte symbol-present bitmap (one bit per byte value).
2. Per set bit: varint frequency (low 7 bits + continuation bit, little-endian
   order). Frequencies must sum to exactly 4096.
3. 8 × u32 initial rANS states.
4. u16 renewal words, consumed in lane order.

## 6. Validation contract

Parsers (`VortexArchive.from_bytes`, `VortexVFS.open`, Rust decoders) must:

* verify magic, meta framing, JSON shape, row arity and numeric types;
* bound every offset/length against the real file size before use;
* reject unknown `strategy_id`, block tags and sub-stream modes;
* guarantee corrupted input raises `VXPKError` (Python) / returns `Err`
  (Rust) — never panic, hang or write out of bounds (enforced by adversarial
  fuzz tests in `tests/test_robustness.py` and `vxlz.rs::test_adversarial_inputs_never_panic`).

## 7. C-ABI

Native engines consume the same functionality without Python via
`vortexpak_native/vortexpak.h` (generated from `c_api.rs` with cbindgen):
VXTX transforms, parallel micro-block/texture decode, RDO quantization,
VXLZ compress/decompress and a native seekable VFS.
