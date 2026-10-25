//! C ABI interface for VortexPak Native Engine
//! Provides high-throughput zero-copy access from Python (ctypes) and native game engines.

use crate::decoder::{self, MicroBlockDesc};
use crate::rdo::{self, RdoLevel};
use crate::vxlz;
use crate::vxtx;

#[no_mangle]
pub unsafe extern "C" fn vxtx_is_all_opaque(rgba_ptr: *const u8, len: usize) -> i32 {
    if rgba_ptr.is_null() || len == 0 || len % 4 != 0 {
        return 0;
    }
    let slice = std::slice::from_raw_parts(rgba_ptr, len);
    if vxtx::is_all_opaque(slice) {
        1
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn vxtx_alpha_strip_pack(
    rgba_ptr: *const u8,
    rgba_len: usize,
    rgb_dst_ptr: *mut u8,
) -> i32 {
    if rgba_ptr.is_null() || rgb_dst_ptr.is_null() || rgba_len % 4 != 0 {
        return -1;
    }
    let src = std::slice::from_raw_parts(rgba_ptr, rgba_len);
    let dst = std::slice::from_raw_parts_mut(rgb_dst_ptr, rgba_len / 4 * 3);
    vxtx::alpha_strip_pack(src, dst);
    0
}

#[no_mangle]
pub unsafe extern "C" fn vxtx_alpha_strip_unpack(
    rgb_ptr: *const u8,
    rgb_len: usize,
    rgba_dst_ptr: *mut u8,
) -> i32 {
    if rgb_ptr.is_null() || rgba_dst_ptr.is_null() || rgb_len % 3 != 0 {
        return -1;
    }
    let src = std::slice::from_raw_parts(rgb_ptr, rgb_len);
    let dst = std::slice::from_raw_parts_mut(rgba_dst_ptr, rgb_len / 3 * 4);
    vxtx::alpha_strip_unpack(src, dst);
    0
}

#[no_mangle]
pub unsafe extern "C" fn vxtx_row_rle_decode(
    counts_ptr: *const u16,
    counts_len: usize,
    unique_ptr: *const u8,
    row_bytes: usize,
    dst_ptr: *mut u8,
    dst_len: usize,
) -> i32 {
    if counts_ptr.is_null() || unique_ptr.is_null() || dst_ptr.is_null() || row_bytes == 0 {
        return -1;
    }
    let counts = std::slice::from_raw_parts(counts_ptr, counts_len);
    let unique = std::slice::from_raw_parts(unique_ptr, counts_len * row_bytes);
    let dst = std::slice::from_raw_parts_mut(dst_ptr, dst_len);
    vxtx::row_rle_decode(counts, unique, row_bytes, dst);
    0
}

#[no_mangle]
pub unsafe extern "C" fn vxtx_row_rle_encode(
    data_ptr: *const u8,
    height: usize,
    row_bytes: usize,
    counts_dst: *mut u16,
    unique_dst: *mut u8,
    max_unique: usize,
) -> i32 {
    if data_ptr.is_null() || height == 0 || row_bytes == 0 {
        return -1;
    }
    let data = std::slice::from_raw_parts(data_ptr, height * row_bytes);
    let (counts, unique) = vxtx::row_rle_encode(data, height, row_bytes);
    let n_unique = counts.len();
    if counts_dst.is_null() || unique_dst.is_null() {
        return n_unique as i32;
    }
    if n_unique > max_unique {
        return -2;
    }
    std::ptr::copy_nonoverlapping(counts.as_ptr(), counts_dst, n_unique);
    std::ptr::copy_nonoverlapping(unique.as_ptr(), unique_dst, n_unique * row_bytes);
    n_unique as i32
}

#[no_mangle]
pub unsafe extern "C" fn vxtx_planar_decode(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    width: usize,
    height: usize,
    channels: usize,
) -> i32 {
    if src_ptr.is_null() || dst_ptr.is_null() {
        return -1;
    }
    let total_len = width * height * channels;
    let src = std::slice::from_raw_parts(src_ptr, total_len);
    let dst = std::slice::from_raw_parts_mut(dst_ptr, total_len);
    vxtx::planar_decode(src, dst, width, height, channels);
    0
}

#[no_mangle]
pub unsafe extern "C" fn vxtx_planar_encode(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    width: usize,
    height: usize,
    channels: usize,
) -> i32 {
    if src_ptr.is_null() || dst_ptr.is_null() {
        return -1;
    }
    let total_len = width * height * channels;
    let src = std::slice::from_raw_parts(src_ptr, total_len);
    let dst = std::slice::from_raw_parts_mut(dst_ptr, total_len);
    vxtx::planar_encode(src, width, height, channels, dst);
    0
}

#[no_mangle]
pub unsafe extern "C" fn rdo_quantize_image(
    src_ptr: *const u8,
    width: usize,
    height: usize,
    channels: usize,
    level: i32,
    dst_ptr: *mut u8,
) -> i32 {
    if src_ptr.is_null() || dst_ptr.is_null() {
        return -1;
    }
    let total = width * height * channels;
    let src = std::slice::from_raw_parts(src_ptr, total);
    let r_lvl = match level {
        1 => RdoLevel::Subtle,
        2 => RdoLevel::Balanced,
        _ => RdoLevel::None,
    };
    let quant = rdo::quantize_image_rdo(src, width, height, channels, r_lvl);
    std::ptr::copy_nonoverlapping(quant.as_ptr(), dst_ptr, total);
    0
}

#[no_mangle]
pub unsafe extern "C" fn rdo_quantize_floats(
    floats_ptr: *mut f32,
    count: usize,
    drop_bits: u32,
) -> i32 {
    if floats_ptr.is_null() {
        return -1;
    }
    let slice = std::slice::from_raw_parts_mut(floats_ptr, count);
    rdo::quantize_floats_slice(slice, drop_bits);
    0
}

#[no_mangle]
pub unsafe extern "C" fn calculate_image_psnr_c(
    orig_ptr: *const u8,
    quant_ptr: *const u8,
    len: usize,
) -> f64 {
    if orig_ptr.is_null() || quant_ptr.is_null() {
        return 0.0;
    }
    let orig = std::slice::from_raw_parts(orig_ptr, len);
    let quant = std::slice::from_raw_parts(quant_ptr, len);
    rdo::calculate_image_psnr(orig, quant)
}

#[no_mangle]
pub unsafe extern "C" fn calculate_float_psnr_c(
    orig_ptr: *const f32,
    quant_ptr: *const f32,
    count: usize,
    range: f32,
) -> f64 {
    if orig_ptr.is_null() || quant_ptr.is_null() {
        return 0.0;
    }
    let orig = std::slice::from_raw_parts(orig_ptr, count);
    let quant = std::slice::from_raw_parts(quant_ptr, count);
    rdo::calculate_float_psnr(orig, quant, range)
}

#[no_mangle]
pub unsafe extern "C" fn vpak_decompress_block_c(
    strategy_id: u32,
    comp_ptr: *const u8,
    comp_len: usize,
    raw_size: usize,
    dst_ptr: *mut u8,
    pat_ptr: *const u8,
    pat_len: usize,
    tok_byte: u8,
) -> i64 {
    if comp_ptr.is_null() || dst_ptr.is_null() {
        return -1;
    }
    let comp = std::slice::from_raw_parts(comp_ptr, comp_len);
    let pat = if pat_ptr.is_null() || pat_len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(pat_ptr, pat_len)
    };

    let desc = MicroBlockDesc {
        block_id: 0,
        strategy_id,
        raw_size,
        compressed_payload: comp,
        token_pattern: pat,
        token_byte: tok_byte,
    };

    let dst_slice = std::slice::from_raw_parts_mut(dst_ptr, raw_size);
    match decoder::decompress_block_into(&desc, dst_slice) {
        Ok(written) => written as i64,
        Err(_) => -2,
    }
}

#[repr(C)]
pub struct CMicroBlockDesc {
    pub block_id: u32,
    pub strategy_id: u32,
    pub raw_size: usize,
    pub comp_ptr: *const u8,
    pub comp_len: usize,
    pub dst_ptr: *mut u8,
    pub dst_cap: usize,
    pub pat_ptr: *const u8,
    pub pat_len: usize,
    pub tok_byte: u8,
    pub out_len: usize,
    pub status: i32,
}

unsafe impl Send for CMicroBlockDesc {}
unsafe impl Sync for CMicroBlockDesc {}

#[no_mangle]
pub unsafe extern "C" fn vpak_decompress_blocks_parallel_c(
    blocks_ptr: *mut CMicroBlockDesc,
    num_blocks: usize,
) -> i32 {
    if blocks_ptr.is_null() || num_blocks == 0 {
        return -1;
    }
    use rayon::prelude::*;
    let blocks = std::slice::from_raw_parts_mut(blocks_ptr, num_blocks);

    blocks.par_iter_mut().for_each(|b| {
        if b.comp_ptr.is_null() || b.dst_ptr.is_null() {
            b.status = -1;
            return;
        }
        let comp = std::slice::from_raw_parts(b.comp_ptr, b.comp_len);
        let pat = if b.pat_ptr.is_null() || b.pat_len == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(b.pat_ptr, b.pat_len)
        };

        let desc = MicroBlockDesc {
            block_id: b.block_id,
            strategy_id: b.strategy_id,
            raw_size: b.raw_size,
            compressed_payload: comp,
            token_pattern: pat,
            token_byte: b.tok_byte,
        };

        let dst_slice = std::slice::from_raw_parts_mut(b.dst_ptr, b.dst_cap);
        match decoder::decompress_block_into(&desc, dst_slice) {
            Ok(written) => {
                b.out_len = written;
                b.status = 0;
            }
            Err(_) => {
                b.status = -2;
            }
        }
    });

    0
}

#[repr(C)]
pub struct CTextureDecodeDesc {
    pub src_ptr: *const u8,
    pub src_len: usize,
    pub dst_ptr: *mut u8,
    pub dst_len: usize,
    pub width: u32,
    pub height: u32,
    pub orig_channels: u32,
    pub tx_flags: u32,
    pub rle_unique_rows: u32,
    pub materialize_rgba: u32,
    pub status: i32,
}

unsafe impl Send for CTextureDecodeDesc {}
unsafe impl Sync for CTextureDecodeDesc {}

#[no_mangle]
pub unsafe extern "C" fn vxtx_decode_textures_batch_parallel_c(
    descs_ptr: *mut CTextureDecodeDesc,
    num_textures: usize,
) -> i32 {
    if descs_ptr.is_null() || num_textures == 0 {
        return -1;
    }
    use rayon::prelude::*;
    let descs = std::slice::from_raw_parts_mut(descs_ptr, num_textures);

    descs.par_iter_mut().for_each(|d| {
        if d.src_ptr.is_null() || d.dst_ptr.is_null() {
            d.status = -1;
            return;
        }
        let src = std::slice::from_raw_parts(d.src_ptr, d.src_len);
        let dst = std::slice::from_raw_parts_mut(d.dst_ptr, d.dst_len);
        match vxtx::decode_texture_vxtx_into(
            src,
            dst,
            d.width as usize,
            d.height as usize,
            d.orig_channels as usize,
            d.tx_flags,
            d.rle_unique_rows as usize,
            d.materialize_rgba != 0,
        ) {
            Ok(()) => d.status = 0,
            Err(_) => d.status = -2,
        }
    });

    0
}

#[no_mangle]
pub unsafe extern "C" fn rdo_quantize_glb_c(
    glb_ptr: *const u8,
    glb_len: usize,
    drop_bits: u32,
    dst_ptr: *mut u8,
    dst_cap: usize,
) -> i64 {
    if glb_ptr.is_null() || dst_ptr.is_null() || glb_len == 0 {
        return -1;
    }
    let glb_bytes = std::slice::from_raw_parts(glb_ptr, glb_len);
    match rdo::quantize_glb_buffer(glb_bytes, drop_bits) {
        Ok(out) => {
            if out.len() > dst_cap {
                return -2;
            }
            std::ptr::copy_nonoverlapping(out.as_ptr(), dst_ptr, out.len());
            out.len() as i64
        }
        Err(_) => -3,
    }
}

// ================= Seekable VFS C-ABI Interface =================

use crate::vfs::VortexVfs;
use std::ffi::CStr;
use std::os::raw::c_char;

#[no_mangle]
pub unsafe extern "C" fn vfs_open_c(path_ptr: *const c_char) -> *mut VortexVfs<std::fs::File> {
    if path_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let c_str = CStr::from_ptr(path_ptr);
    let path = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    match VortexVfs::open_file(path) {
        Ok(vfs) => Box::into_raw(Box::new(vfs)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn vfs_close_c(handle: *mut VortexVfs<std::fs::File>) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

#[no_mangle]
pub unsafe extern "C" fn vfs_get_asset_size_c(
    handle: *mut VortexVfs<std::fs::File>,
    path_ptr: *const c_char,
) -> i64 {
    if handle.is_null() || path_ptr.is_null() {
        return -1;
    }
    let c_str = CStr::from_ptr(path_ptr);
    let path = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let vfs = &*handle;
    match vfs.get_asset_size(path) {
        Some(sz) => sz as i64,
        None => -1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn vfs_read_asset_c(
    handle: *mut VortexVfs<std::fs::File>,
    path_ptr: *const c_char,
    dst_ptr: *mut u8,
    max_len: usize,
    materialize_rgba: i32,
) -> i64 {
    if handle.is_null() || path_ptr.is_null() || dst_ptr.is_null() {
        return -1;
    }
    let c_str = CStr::from_ptr(path_ptr);
    let path = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let vfs = &mut *handle;
    match vfs.read_asset(path, materialize_rgba != 0) {
        Ok(data) => {
            if data.len() > max_len {
                return -2;
            }
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst_ptr, data.len());
            data.len() as i64
        }
        Err(_) => -3,
    }
}

#[no_mangle]
pub unsafe extern "C" fn vfs_get_assets_count_c(handle: *mut VortexVfs<std::fs::File>) -> usize {
    if handle.is_null() {
        return 0;
    }
    let vfs = &*handle;
    vfs.entries.len()
}

// ================= Proprietary VXLZ-ANS C-ABI Interface =================

#[no_mangle]
pub unsafe extern "C" fn vxlz_compress_c(
    src_ptr: *const u8,
    src_len: usize,
    dst_ptr: *mut u8,
    dst_cap: usize,
) -> i64 {
    if src_ptr.is_null() || dst_ptr.is_null() {
        return -1;
    }
    let src = std::slice::from_raw_parts(src_ptr, src_len);
    let comp = vxlz::vxlz_compress(src);
    if comp.len() > dst_cap {
        return -2;
    }
    std::ptr::copy_nonoverlapping(comp.as_ptr(), dst_ptr, comp.len());
    comp.len() as i64
}

#[no_mangle]
pub unsafe extern "C" fn vxlz_decompress_c(
    src_ptr: *const u8,
    src_len: usize,
    dst_ptr: *mut u8,
    dst_len: usize,
) -> i64 {
    if src_ptr.is_null() || dst_ptr.is_null() {
        return -1;
    }
    let src = std::slice::from_raw_parts(src_ptr, src_len);
    let dst = std::slice::from_raw_parts_mut(dst_ptr, dst_len);
    match vxlz::vxlz_decompress_into(src, dst) {
        Ok(written) => written as i64,
        Err(_) => -2,
    }
}

#[repr(C)]
pub struct CVxlzCompressDesc {
    pub src_ptr: *const u8,
    pub src_len: usize,
    pub dst_ptr: *mut u8,
    pub dst_cap: usize,
    pub out_len: usize,
    pub out_strategy: u32,
    pub allow_bypass: u32,
    pub status: i32,
}

unsafe impl Send for CVxlzCompressDesc {}
unsafe impl Sync for CVxlzCompressDesc {}

#[no_mangle]
pub unsafe extern "C" fn vxlz_compress_blocks_parallel_c(
    descs_ptr: *mut CVxlzCompressDesc,
    num_blocks: usize,
) -> i32 {
    if descs_ptr.is_null() || num_blocks == 0 {
        return -1;
    }
    use rayon::prelude::*;
    let descs = std::slice::from_raw_parts_mut(descs_ptr, num_blocks);

    descs.par_iter_mut().for_each(|d| {
        if d.src_ptr.is_null() || d.dst_ptr.is_null() {
            d.status = -1;
            return;
        }
        let src = std::slice::from_raw_parts(d.src_ptr, d.src_len);
        if src.is_empty() {
            d.out_len = 0;
            d.out_strategy = decoder::STRAT_RAW_BYPASS;
            d.status = 0;
            return;
        }
        let comp = vxlz::vxlz_compress(src);
        if d.allow_bypass != 0 && comp.len() >= (src.len() * 992) / 1000 {
            if src.len() > d.dst_cap {
                d.status = -2;
                return;
            }
            std::ptr::copy_nonoverlapping(src.as_ptr(), d.dst_ptr, src.len());
            d.out_len = src.len();
            d.out_strategy = decoder::STRAT_RAW_BYPASS;
            d.status = 0;
        } else {
            if comp.len() > d.dst_cap {
                d.status = -2;
                return;
            }
            std::ptr::copy_nonoverlapping(comp.as_ptr(), d.dst_ptr, comp.len());
            d.out_len = comp.len();
            d.out_strategy = decoder::STRAT_VXLZ;
            d.status = 0;
        }
    });

    0
}

// ================= VXTX-GPU BCn block-compression C-ABI =================

#[no_mangle]
pub unsafe extern "C" fn bcn_encode_c(
    fmt: i32,
    src_ptr: *const u8,
    src_len: usize,
    width: u32,
    height: u32,
    dst_ptr: *mut u8,
    dst_cap: usize,
) -> i64 {
    if (src_ptr.is_null() && src_len > 0) || dst_ptr.is_null() {
        return -1;
    }
    let src = std::slice::from_raw_parts(src_ptr, src_len);
    let dst = std::slice::from_raw_parts_mut(dst_ptr, dst_cap);
    match crate::bcn::bcn_encode(fmt, src, width as usize, height as usize) {
        Ok(out) => {
            if out.len() > dst_cap {
                return -2;
            }
            dst[..out.len()].copy_from_slice(&out);
            out.len() as i64
        }
        Err(_) => -3,
    }
}

#[no_mangle]
pub unsafe extern "C" fn bcn_decode_c(
    fmt: i32,
    src_ptr: *const u8,
    src_len: usize,
    width: u32,
    height: u32,
    dst_ptr: *mut u8,
    dst_cap: usize,
) -> i64 {
    if (src_ptr.is_null() && src_len > 0) || dst_ptr.is_null() {
        return -1;
    }
    let src = std::slice::from_raw_parts(src_ptr, src_len);
    let dst = std::slice::from_raw_parts_mut(dst_ptr, dst_cap);
    match crate::bcn::bcn_decode(fmt, src, width as usize, height as usize) {
        Ok(out) => {
            if out.len() > dst_cap {
                return -2;
            }
            dst[..out.len()].copy_from_slice(&out);
            out.len() as i64
        }
        Err(_) => -3,
    }
}

#[no_mangle]
pub unsafe extern "C" fn bcn_encode_ex_c(
    fmt: i32,
    quality: u32,
    src_ptr: *const u8,
    src_len: usize,
    width: u32,
    height: u32,
    dst_ptr: *mut u8,
    dst_cap: usize,
) -> i64 {
    if (src_ptr.is_null() && src_len > 0) || dst_ptr.is_null() {
        return -1;
    }
    let src = std::slice::from_raw_parts(src_ptr, src_len);
    let dst = std::slice::from_raw_parts_mut(dst_ptr, dst_cap);
    match crate::bcn::bcn_encode_q(fmt, src, width as usize, height as usize, quality as u8) {
        Ok(out) => {
            if out.len() > dst_cap {
                return -2;
            }
            dst[..out.len()].copy_from_slice(&out);
            out.len() as i64
        }
        Err(_) => -3,
    }
}

#[no_mangle]
pub unsafe extern "C" fn bcn_encode_bc7_c(
    src_ptr: *const u8,
    src_len: usize,
    width: u32,
    height: u32,
    dst_ptr: *mut u8,
    dst_cap: usize,
) -> i64 {
    if (src_ptr.is_null() && src_len > 0) || dst_ptr.is_null() {
        return -1;
    }
    let src = std::slice::from_raw_parts(src_ptr, src_len);
    let dst = std::slice::from_raw_parts_mut(dst_ptr, dst_cap);
    match crate::bcn::encode_bc7(src, width as usize, height as usize) {
        Ok(out) => {
            if out.len() > dst_cap {
                return -2;
            }
            dst[..out.len()].copy_from_slice(&out);
            out.len() as i64
        }
        Err(_) => -3,
    }
}

#[no_mangle]
pub unsafe extern "C" fn bcn_decode_bc7_c(
    src_ptr: *const u8,
    src_len: usize,
    width: u32,
    height: u32,
    dst_ptr: *mut u8,
    dst_cap: usize,
) -> i64 {
    if (src_ptr.is_null() && src_len > 0) || dst_ptr.is_null() {
        return -1;
    }
    let src = std::slice::from_raw_parts(src_ptr, src_len);
    let dst = std::slice::from_raw_parts_mut(dst_ptr, dst_cap);
    match crate::bcn::decode_bc7(src, width as usize, height as usize) {
        Ok(out) => {
            if out.len() > dst_cap {
                return -2;
            }
            dst[..out.len()].copy_from_slice(&out);
            out.len() as i64
        }
        Err(_) => -3,
    }
}

#[no_mangle]
pub unsafe extern "C" fn bcn_encode_bc1_rdo_c(
    quality: u32,
    strength_x100: u32,
    iterations: u32,
    src_ptr: *const u8,
    src_len: usize,
    width: u32,
    height: u32,
    dst_ptr: *mut u8,
    dst_cap: usize,
) -> i64 {
    if (src_ptr.is_null() && src_len > 0) || dst_ptr.is_null() {
        return -1;
    }
    let src = std::slice::from_raw_parts(src_ptr, src_len);
    let dst = std::slice::from_raw_parts_mut(dst_ptr, dst_cap);
    match crate::bcn::encode_bc1_rdo(
        src,
        width as usize,
        height as usize,
        quality as u8,
        strength_x100 as f32 / 100.0,
        iterations as usize,
    ) {
        Ok(out) => {
            if out.len() > dst_cap {
                return -2;
            }
            dst[..out.len()].copy_from_slice(&out);
            out.len() as i64
        }
        Err(_) => -3,
    }
}
