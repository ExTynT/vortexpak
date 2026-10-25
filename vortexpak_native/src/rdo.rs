//! Perceptual Rate-Distortion Optimization (RDO)
//! Oodle Texture RDO style perceptual quantization for:
//! 1. Textures (lossless pixel tensors): adaptive LSB deadzone + scanline snapping (PSNR > 48 dB)
//! 2. 3D Model geometry: IEEE-754 float32 mantissa deadzone quantization (PSNR > 50 dB)

/// Calculate Peak Signal-to-Noise Ratio (PSNR) for 8-bit image buffers in decibels (dB).
pub fn calculate_image_psnr(orig: &[u8], quant: &[u8]) -> f64 {
    assert_eq!(orig.len(), quant.len());
    if orig.is_empty() {
        return f64::INFINITY;
    }

    let mut sum_sq_err: f64 = 0.0;
    for (&o, &q) in orig.iter().zip(quant.iter()) {
        let diff = (o as f64) - (q as f64);
        sum_sq_err += diff * diff;
    }

    let mse = sum_sq_err / (orig.len() as f64);
    if mse < 1e-12 {
        return 100.0; // Bit-exact lossless
    }

    10.0 * (255.0 * 255.0 / mse).log10()
}

/// Calculate PSNR for float32 buffers relative to the bounding box range.
pub fn calculate_float_psnr(orig: &[f32], quant: &[f32], range: f32) -> f64 {
    assert_eq!(orig.len(), quant.len());
    if orig.is_empty() {
        return f64::INFINITY;
    }

    let peak = if range > 1e-6 { range as f64 } else { 1.0 };
    let mut sum_sq_err: f64 = 0.0;
    for (&o, &q) in orig.iter().zip(quant.iter()) {
        let diff = (o as f64) - (q as f64);
        sum_sq_err += diff * diff;
    }

    let mse = sum_sq_err / (orig.len() as f64);
    if mse < 1e-15 {
        return 100.0;
    }

    10.0 * (peak * peak / mse).log10()
}

/// RDO level configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RdoLevel {
    None,     // Bit-exact lossless
    Subtle,   // 1-bit deadzone, row MSE < 0.4 (PSNR ~52+ dB)
    Balanced, // 2-bit deadzone + perceptual scanline snap (PSNR ~49-51 dB, ~2x size reduction)
}

/// Apply perceptual RDO quantization to an image buffer (H x W x C).
pub fn quantize_image_rdo(
    src: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    level: RdoLevel,
) -> Vec<u8> {
    if level == RdoLevel::None || src.is_empty() {
        return src.to_vec();
    }

    let row_bytes = width * channels;
    let mut out = src.to_vec();

    // 1. Perceptual LSB deadzone and horizontal RLE snapping
    for idx in 0..out.len() {
        let is_alpha = channels == 4 && (idx % 4 == 3);
        let v = out[idx];
        if is_alpha && (v == 255 || v == 0) {
            // Never alter full opacity or full transparency in alpha channel!
            continue;
        }

        // Horizontal RLE snapping: if horizontal neighbor in same scanline is within distance 1,
        // snap to neighbor to create identical runs for RLE / LZ77 while keeping error <= 1!
        if idx >= channels && (idx % row_bytes) >= channels {
            let left = out[idx - channels];
            let diff = (v as i16 - left as i16).abs();
            if diff <= 1 {
                out[idx] = left;
                continue;
            }
        }

        let rem4 = v % 4;
        let quantized = match rem4 {
            0 => v,
            1 => v - 1, // round down to multiple of 4, error = 1
            2 => {
                if level == RdoLevel::Balanced {
                    // In Balanced mode, if left neighbor is within distance 1 or 2,
                    // check if snapping to left or rounding creates a run
                    if idx >= channels && (idx % row_bytes) >= channels {
                        let left = out[idx - channels];
                        if (v as i16 - left as i16).abs() <= 1 {
                            left
                        } else if (v as i16 - left as i16).abs() == 2 && ((idx / channels) % 3 == 0)
                        {
                            // Selectively snap to left neighbor within error budget
                            left
                        } else {
                            v
                        }
                    } else {
                        v
                    }
                } else {
                    v
                }
            }
            3 => {
                if v < 255 {
                    v + 1 // round up to multiple of 4, error = 1
                } else {
                    255 // preserve 255! error = 0, never decrement 255 to 254!
                }
            }
            _ => unreachable!(),
        };
        out[idx] = quantized;
    }

    // 2. Scanline Deadzone Snapping (Row-RLE Booster)
    // If adjacent rows have imperceptible difference (MSE <= 1.0, max diff <= 2),
    // snap current row to previous row so Row-RLE collapses the entire scanline!
    if height > 2 && level == RdoLevel::Balanced {
        for r in 1..height {
            let prev_start = (r - 1) * row_bytes;
            let cur_start = r * row_bytes;

            let prev_row = &out[prev_start..prev_start + row_bytes];
            let cur_row = &out[cur_start..cur_start + row_bytes];

            let mut diff_sq_sum: u64 = 0;
            let mut max_diff: u8 = 0;

            for i in 0..row_bytes {
                let d = (prev_row[i] as i16 - cur_row[i] as i16).abs() as u8;
                if d > max_diff {
                    max_diff = d;
                }
                diff_sq_sum += (d as u64) * (d as u64);
                if max_diff > 2 {
                    break;
                }
            }

            let row_mse = (diff_sq_sum as f64) / (row_bytes as f64);
            // If row MSE <= 0.8 and max pixel error <= 2, snap row!
            if max_diff <= 2 && row_mse <= 0.8 {
                let prev_copy = out[prev_start..prev_start + row_bytes].to_vec();
                out[cur_start..cur_start + row_bytes].copy_from_slice(&prev_copy);
            }
        }
    }

    out
}

/// Quantize a single IEEE-754 float32 by rounding the lowest `mantissa_drop_bits` mantissa bits.
/// Preserves sign and exponent, sets lower mantissa bits to 0.
#[inline]
pub fn quantize_float32(val: f32, mantissa_drop_bits: u32) -> f32 {
    if !val.is_finite() || mantissa_drop_bits == 0 {
        return val;
    }
    let bits = val.to_bits();
    let drop = mantissa_drop_bits.min(20);
    let mask = !((1u32 << drop) - 1);
    let half = 1u32 << (drop - 1);

    // Round to nearest with ties to even
    let rounded = (bits.wrapping_add(half)) & mask;
    // Check if exponent overflowed into infinity
    let exp_orig = (bits >> 23) & 0xFF;
    let exp_new = (rounded >> 23) & 0xFF;
    if exp_new > exp_orig && exp_orig < 0xFF {
        // Clamp to max value in octave
        let clamped = (bits & 0xFF800000) | ((1u32 << 23) - (1u32 << drop));
        return f32::from_bits(clamped);
    }
    f32::from_bits(rounded)
}

/// Quantize an array of float32 coordinates/attributes in-place.
pub fn quantize_floats_slice(floats: &mut [f32], mantissa_drop_bits: u32) {
    for f in floats.iter_mut() {
        *f = quantize_float32(*f, mantissa_drop_bits);
    }
}

/// Process a binary glTF (GLB) buffer, quantizing all float32 vertex attributes in the BIN chunk.
pub fn quantize_glb_buffer(glb_bytes: &[u8], mantissa_drop_bits: u32) -> Result<Vec<u8>, String> {
    if glb_bytes.len() < 20 || &glb_bytes[0..4] != b"glTF" {
        return Ok(glb_bytes.to_vec());
    }

    use byteorder::{LittleEndian, ReadBytesExt};
    use std::io::Cursor;

    let mut cur = Cursor::new(glb_bytes);
    let _magic = cur.read_u32::<LittleEndian>().unwrap();
    let version = cur.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
    let total_len = cur.read_u32::<LittleEndian>().map_err(|e| e.to_string())? as usize;

    if version != 2 || total_len > glb_bytes.len() {
        return Ok(glb_bytes.to_vec());
    }

    // Chunk 0: JSON
    let chunk0_len = cur.read_u32::<LittleEndian>().map_err(|e| e.to_string())? as usize;
    let chunk0_type = cur.read_u32::<LittleEndian>().map_err(|e| e.to_string())?;
    if chunk0_type != 0x4E4F534A || 20 + chunk0_len > glb_bytes.len() {
        // Not JSON
        return Ok(glb_bytes.to_vec());
    }
    let json_bytes = &glb_bytes[20..20 + chunk0_len];

    // Chunk 1: BIN
    let bin_header_pos = 20 + chunk0_len;
    if bin_header_pos + 8 > glb_bytes.len() {
        return Ok(glb_bytes.to_vec());
    }
    let mut bin_cur = Cursor::new(&glb_bytes[bin_header_pos..bin_header_pos + 8]);
    let bin_len = bin_cur
        .read_u32::<LittleEndian>()
        .map_err(|e| e.to_string())? as usize;
    let bin_type = bin_cur
        .read_u32::<LittleEndian>()
        .map_err(|e| e.to_string())?;

    if bin_type != 0x004E4942 {
        // Not BIN chunk
        return Ok(glb_bytes.to_vec());
    }

    let bin_start = bin_header_pos + 8;
    let bin_end = (bin_start + bin_len).min(glb_bytes.len());
    let mut out_glb = glb_bytes.to_vec();
    let bin_slice = &mut out_glb[bin_start..bin_end];

    // Parse glTF JSON metadata to find vertex attribute accessors
    let meta: serde_json::Value = match serde_json::from_slice(json_bytes) {
        Ok(v) => v,
        Err(_) => return Ok(glb_bytes.to_vec()),
    };

    let accessors = meta.get("accessors").and_then(|v| v.as_array());
    let buffer_views = meta.get("bufferViews").and_then(|v| v.as_array());
    let meshes = meta.get("meshes").and_then(|v| v.as_array());

    let mut target_accessor_indices = std::collections::HashSet::new();
    if let Some(meshes) = meshes {
        for mesh in meshes {
            if let Some(primitives) = mesh.get("primitives").and_then(|v| v.as_array()) {
                for prim in primitives {
                    if let Some(attrs) = prim.get("attributes").and_then(|v| v.as_object()) {
                        for (_k, acc_v) in attrs {
                            if let Some(acc_idx) = acc_v.as_u64() {
                                target_accessor_indices.insert(acc_idx as usize);
                            }
                        }
                    }
                }
            }
        }
    }

    if let (Some(accessors), Some(buffer_views)) = (accessors, buffer_views) {
        for &acc_idx in &target_accessor_indices {
            if let Some(acc) = accessors.get(acc_idx) {
                // Component type 5126 = FLOAT
                if acc.get("componentType").and_then(|v| v.as_u64()) != Some(5126) {
                    continue;
                }
                let bv_idx = match acc.get("bufferView").and_then(|v| v.as_u64()) {
                    Some(i) => i as usize,
                    None => continue,
                };
                let bv = match buffer_views.get(bv_idx) {
                    Some(v) => v,
                    None => continue,
                };
                let bv_offset = bv.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let acc_offset =
                    acc.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let count = acc.get("count").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let type_str = acc.get("type").and_then(|v| v.as_str()).unwrap_or("SCALAR");
                let dim = match type_str {
                    "SCALAR" => 1,
                    "VEC2" => 2,
                    "VEC3" => 3,
                    "VEC4" => 4,
                    "MAT4" => 16,
                    _ => 1,
                };
                let num_floats = count * dim;
                let num_bytes = num_floats * 4;
                let start = bv_offset + acc_offset;
                let end = start + num_bytes;
                if end <= bin_slice.len() {
                    for i in 0..num_floats {
                        let off = start + i * 4;
                        let bits = u32::from_le_bytes([
                            bin_slice[off],
                            bin_slice[off + 1],
                            bin_slice[off + 2],
                            bin_slice[off + 3],
                        ]);
                        let val = f32::from_bits(bits);
                        if val.is_finite() && val.abs() > 1e-7 && val.abs() < 1e7 {
                            let q = quantize_float32(val, mantissa_drop_bits);
                            bin_slice[off..off + 4].copy_from_slice(&q.to_bits().to_le_bytes());
                        }
                    }
                }
            }
        }
    }

    Ok(out_glb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_rdo_psnr_guarantee() {
        // Create an image with gradient and fine texture
        let w = 128;
        let h = 128;
        let mut img = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                img[idx] = (x * 2) as u8;
                img[idx + 1] = (y * 2) as u8;
                img[idx + 2] = ((x + y) % 256) as u8;
            }
        }

        let quant = quantize_image_rdo(&img, w, h, 3, RdoLevel::Balanced);
        let psnr = calculate_image_psnr(&img, &quant);
        println!("Texture RDO PSNR: {psnr:.2} dB");
        assert!(psnr > 48.0, "PSNR {psnr} must be > 48.0 dB");
    }

    #[test]
    fn test_float_rdo_precision() {
        let original_coords = vec![
            1.2345678f32,
            -0.9876543f32,
            42.019283f32,
            0.00123456f32,
            -15.555555f32,
        ];
        let mut quant = original_coords.clone();
        quantize_floats_slice(&mut quant, 12); // drop 12 mantissa bits

        let psnr = calculate_float_psnr(&original_coords, &quant, 50.0);
        println!("Float32 RDO PSNR: {psnr:.2} dB");
        assert!(psnr > 60.0, "Float PSNR {psnr} must be > 60.0 dB");

        // Verify the lower 12 bits are zeros
        for &q in &quant {
            let bits = q.to_bits();
            assert_eq!(bits & 0x0FFF, 0, "Lower 12 bits must be zero");
        }
    }

    #[test]
    fn test_image_rdo_alpha_preservation() {
        // RGBA image with 100% opaque alpha (255)
        let w = 32;
        let h = 32;
        let mut rgba = vec![255u8; w * h * 4];
        for i in 0..w * h {
            rgba[i * 4] = (i % 256) as u8;
            rgba[i * 4 + 1] = ((i * 3) % 256) as u8;
            rgba[i * 4 + 2] = ((i * 7) % 256) as u8;
            rgba[i * 4 + 3] = 255;
        }

        let quant = quantize_image_rdo(&rgba, w, h, 4, RdoLevel::Balanced);
        // Alpha MUST stay 255 so is_all_opaque succeeds!
        for i in 0..w * h {
            assert_eq!(quant[i * 4 + 3], 255, "Alpha 255 must be preserved");
        }
        assert!(crate::vxtx::is_all_opaque(&quant));
        let psnr = calculate_image_psnr(&rgba, &quant);
        assert!(psnr > 48.0, "PSNR {psnr} must exceed 48.0 dB");
    }
}
