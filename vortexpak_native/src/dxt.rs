fn dxt_kind(fourcc: &[u8]) -> Option<(u8, usize)> {
    match fourcc {
        b"DXT1" => Some((1, 8)),
        b"DXT2" | b"DXT3" => Some((3, 16)),
        b"DXT4" | b"DXT5" => Some((5, 16)),
        _ => None,
    }
}

pub fn dxt_split(dds: &[u8]) -> Option<Vec<u8>> {
    if dds.len() < 128 || &dds[..4] != b"DDS " || dds[4..8] != 124u32.to_le_bytes() {
        return None;
    }
    let (kind, bs) = dxt_kind(&dds[84..88])?;
    let n = (dds.len() - 128) / bs;
    let count = u32::try_from(n).ok()?;
    let tail_len = (dds.len() - 128) % bs;
    let mut rec = vec![0; dds.len().checked_add(10)?];
    rec[..4].copy_from_slice(b"VXD1");
    rec[4] = kind;
    rec[5..9].copy_from_slice(&count.to_le_bytes());
    rec[9] = tail_len as u8;
    rec[10..138].copy_from_slice(&dds[..128]);
    let indices = 138 + 4 * n;
    let alpha = 138 + 8 * n;
    let blocks_end = dds.len() - tail_len;
    for (i, block) in dds[128..blocks_end].chunks_exact(bs).enumerate() {
        let c = 138 + 4 * i;
        let ix = indices + 4 * i;
        rec[c..c + 4].copy_from_slice(&block[bs - 8..bs - 4]);
        rec[ix..ix + 4].copy_from_slice(&block[bs - 4..]);
        match kind {
            3 => {
                let a = alpha + 8 * i;
                rec[a..a + 8].copy_from_slice(&block[..8]);
            }
            5 => {
                let ae = alpha + 2 * i;
                let ai = alpha + 2 * n + 6 * i;
                rec[ae..ae + 2].copy_from_slice(&block[..2]);
                rec[ai..ai + 6].copy_from_slice(&block[2..8]);
            }
            _ => {}
        }
    }
    rec[blocks_end + 10..].copy_from_slice(&dds[blocks_end..]);
    Some(rec)
}

pub fn dxt_merge(rec: &[u8]) -> Result<Vec<u8>, String> {
    if rec.len() < 138 {
        return Err("DXT record is shorter than 138 bytes".into());
    }
    if &rec[..4] != b"VXD1" {
        return Err("Invalid DXT record magic".into());
    }
    let kind = rec[4];
    let bs = match kind {
        1 => 8usize,
        3 | 5 => 16usize,
        _ => return Err("Invalid DXT record kind".into()),
    };
    if dxt_kind(&rec[94..98]).map(|(header_kind, _)| header_kind) != Some(kind) {
        return Err("DXT header fourCC does not match record kind".into());
    }
    let n = u32::from_le_bytes([rec[5], rec[6], rec[7], rec[8]]) as usize;
    let tail_len = rec[9] as usize;
    if tail_len >= bs {
        return Err("Invalid DXT tail length".into());
    }
    let expected_len = n
        .checked_mul(bs)
        .and_then(|size| size.checked_add(138))
        .and_then(|size| size.checked_add(tail_len))
        .ok_or_else(|| "DXT record length overflow".to_string())?;
    if rec.len() != expected_len {
        return Err("DXT record length does not match block count".into());
    }
    let mut dds = vec![0; rec.len() - 10];
    dds[..128].copy_from_slice(&rec[10..138]);
    let indices = 138 + 4 * n;
    let alpha = 138 + 8 * n;
    let blocks_end = dds.len() - tail_len;
    for (i, block) in dds[128..blocks_end].chunks_exact_mut(bs).enumerate() {
        let c = 138 + 4 * i;
        let ix = indices + 4 * i;
        block[bs - 8..bs - 4].copy_from_slice(&rec[c..c + 4]);
        block[bs - 4..].copy_from_slice(&rec[ix..ix + 4]);
        match kind {
            3 => {
                let a = alpha + 8 * i;
                block[..8].copy_from_slice(&rec[a..a + 8]);
            }
            5 => {
                let ae = alpha + 2 * i;
                let ai = alpha + 2 * n + 6 * i;
                block[..2].copy_from_slice(&rec[ae..ae + 2]);
                block[2..8].copy_from_slice(&rec[ai..ai + 6]);
            }
            _ => {}
        }
    }
    dds[blocks_end..].copy_from_slice(&rec[blocks_end + 10..]);
    Ok(dds)
}

#[rustfmt::skip]
#[cfg(test)] #[path = "dxt_tests.rs"] mod tests;
