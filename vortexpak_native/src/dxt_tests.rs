use super::*;

// ---------- deterministic inline LCG (no crates) ----------

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }

    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    fn byte(&mut self) -> u8 {
        self.next_u32() as u8
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.byte()).collect()
    }
}

fn bs_for(fourcc: &[u8; 4]) -> usize {
    if fourcc == b"DXT1" {
        8
    } else {
        16
    }
}

fn make_dds(fourcc: &[u8; 4], n: usize, tail_len: usize, rng: &mut Lcg) -> Vec<u8> {
    let bs = bs_for(fourcc);
    let mut x = rng.bytes(128 + n * bs + tail_len);
    x[0..4].copy_from_slice(b"DDS ");
    x[4..8].copy_from_slice(&124u32.to_le_bytes());
    x[84..88].copy_from_slice(fourcc);
    x
}

fn layout_dds(fourcc: &[u8; 4], payload: &[u8], rng: &mut Lcg) -> Vec<u8> {
    let mut dds = rng.bytes(128);
    dds.extend_from_slice(payload);
    dds[0..4].copy_from_slice(b"DDS ");
    dds[4..8].copy_from_slice(&124u32.to_le_bytes());
    dds[84..88].copy_from_slice(fourcc);
    dds
}

// ---------- 1. roundtrip ----------

#[test]
fn roundtrip_all_fourccs_counts_and_tails() {
    let fourccs: [&[u8; 4]; 5] = [b"DXT1", b"DXT2", b"DXT3", b"DXT4", b"DXT5"];
    let mut rng = Lcg::new(0xC0FFEE);
    for &fcc in fourccs.iter() {
        let bs = bs_for(fcc);
        for &n in [0usize, 1, 7, 64].iter() {
            for tail_len in 0..bs {
                let x = make_dds(fcc, n, tail_len, &mut rng);
                let rec = dxt_split(&x).unwrap_or_else(|| {
                    panic!("dxt_split rejected {:?} n={} tail={}", fcc, n, tail_len)
                });
                let back = dxt_merge(&rec).unwrap_or_else(|e| {
                    panic!(
                        "dxt_merge failed {:?} n={} tail={}: {}",
                        fcc, n, tail_len, e
                    )
                });
                assert_eq!(back, x, "roundtrip {:?} n={} tail={}", fcc, n, tail_len);
            }
        }
    }
}

// ---------- 2. record layout ----------

#[test]
fn layout_dxt5() {
    let mut rng = Lcg::new(77);
    // 2 blocks of known bytes (16 each) + 3-byte tail; payload[k] == k
    let payload: Vec<u8> = (0..35u8).collect();
    let dds = layout_dds(b"DXT5", &payload, &mut rng);
    let rec = dxt_split(&dds).unwrap();
    assert_eq!(&rec[0..4], b"VXD1");
    assert_eq!(rec[4], 5);
    assert_eq!(u32::from_le_bytes([rec[5], rec[6], rec[7], rec[8]]), 2);
    assert_eq!(rec[9], 3);
    assert_eq!(&rec[10..138], &dds[..128]);
    // C = per-block bytes 8..12
    assert_eq!(&rec[138..142], &payload[8..12]);
    assert_eq!(&rec[142..146], &payload[24..28]);
    // I = per-block bytes 12..16
    assert_eq!(&rec[146..150], &payload[12..16]);
    assert_eq!(&rec[150..154], &payload[28..32]);
    // AE = per-block bytes 0..2
    assert_eq!(&rec[154..156], &payload[0..2]);
    assert_eq!(&rec[156..158], &payload[16..18]);
    // AI = per-block bytes 2..8
    assert_eq!(&rec[158..164], &payload[2..8]);
    assert_eq!(&rec[164..170], &payload[18..24]);
    // tail
    assert_eq!(&rec[170..173], &payload[32..35]);
    assert_eq!(rec.len(), 138 + 2 * 16 + 3);
}

#[test]
fn layout_dxt1() {
    let mut rng = Lcg::new(78);
    // 2 blocks (8 bytes each) + 3-byte tail; payload[k] == k
    let payload: Vec<u8> = (0..19u8).collect();
    let dds = layout_dds(b"DXT1", &payload, &mut rng);
    let rec = dxt_split(&dds).unwrap();
    assert_eq!(&rec[0..4], b"VXD1");
    assert_eq!(rec[4], 1);
    assert_eq!(u32::from_le_bytes([rec[5], rec[6], rec[7], rec[8]]), 2);
    assert_eq!(rec[9], 3);
    assert_eq!(&rec[10..138], &dds[..128]);
    // C = per-block bytes 0..4
    assert_eq!(&rec[138..142], &payload[0..4]);
    assert_eq!(&rec[142..146], &payload[8..12]);
    // I = per-block bytes 4..8
    assert_eq!(&rec[146..150], &payload[4..8]);
    assert_eq!(&rec[150..154], &payload[12..16]);
    // tail
    assert_eq!(&rec[154..157], &payload[16..19]);
    assert_eq!(rec.len(), 138 + 2 * 8 + 3);
}

#[test]
fn layout_dxt3() {
    let mut rng = Lcg::new(79);
    // 2 blocks (16 bytes each) + 3-byte tail; payload[k] == k
    let payload: Vec<u8> = (0..35u8).collect();
    let dds = layout_dds(b"DXT3", &payload, &mut rng);
    let rec = dxt_split(&dds).unwrap();
    assert_eq!(&rec[0..4], b"VXD1");
    assert_eq!(rec[4], 3);
    assert_eq!(u32::from_le_bytes([rec[5], rec[6], rec[7], rec[8]]), 2);
    assert_eq!(rec[9], 3);
    assert_eq!(&rec[10..138], &dds[..128]);
    // C = per-block bytes 8..12
    assert_eq!(&rec[138..142], &payload[8..12]);
    assert_eq!(&rec[142..146], &payload[24..28]);
    // I = per-block bytes 12..16
    assert_eq!(&rec[146..150], &payload[12..16]);
    assert_eq!(&rec[150..154], &payload[28..32]);
    // EA = per-block bytes 0..8
    assert_eq!(&rec[154..162], &payload[0..8]);
    assert_eq!(&rec[162..170], &payload[16..24]);
    // tail
    assert_eq!(&rec[170..173], &payload[32..35]);
    assert_eq!(rec.len(), 138 + 2 * 16 + 3);
}

// ---------- 3. split rejects invalid DDS ----------

#[test]
fn split_rejects_invalid_inputs() {
    let mut rng = Lcg::new(0x1234);

    assert!(dxt_split(&[]).is_none(), "empty input");

    let short = rng.bytes(127);
    assert!(dxt_split(&short).is_none(), "127 bytes");

    let four_zero = [0u8; 4];
    assert!(dxt_split(&four_zero).is_none(), "four zero bytes");

    let base = make_dds(b"DXT3", 2, 4, &mut rng);

    let mut bad = base.clone();
    bad[0..4].copy_from_slice(b"DXS ");
    assert!(dxt_split(&bad).is_none(), "wrong magic");

    let mut bad = base.clone();
    bad[4..8].copy_from_slice(&123u32.to_le_bytes());
    assert!(dxt_split(&bad).is_none(), "size field 123");

    let mut bad = base.clone();
    bad[84..88].copy_from_slice(b"DX10");
    assert!(dxt_split(&bad).is_none(), "fourCC DX10");

    let mut bad = base.clone();
    bad[84..88].copy_from_slice(b"ATI2");
    assert!(dxt_split(&bad).is_none(), "fourCC ATI2");

    let mut bad = base.clone();
    bad[84..88].copy_from_slice(&[0, 0, 0, 0]);
    assert!(dxt_split(&bad).is_none(), "zero fourCC");
}

// ---------- 4. merge rejects invalid records ----------

#[test]
fn merge_rejects_invalid_records() {
    let mut rng = Lcg::new(99);
    let dds = make_dds(b"DXT5", 2, 3, &mut rng);
    let rec = dxt_split(&dds).unwrap();

    // every strict prefix
    for l in 0..rec.len() {
        assert!(
            dxt_merge(&rec[..l]).is_err(),
            "strict prefix of len {} must be Err",
            l
        );
    }

    // one extra trailing byte
    let mut ext = rec.clone();
    ext.push(0xAB);
    assert!(dxt_merge(&ext).is_err(), "extra trailing byte");

    // wrong magic
    let mut bad = rec.clone();
    bad[0] = b'X';
    assert!(dxt_merge(&bad).is_err(), "wrong magic");

    // kind not in {1,3,5}
    let mut bad = rec.clone();
    bad[4] = 2;
    assert!(dxt_merge(&bad).is_err(), "kind 2");

    // kind inconsistent with header fourCC (header still says DXT5)
    let mut bad = rec.clone();
    bad[4] = 1;
    assert!(dxt_merge(&bad).is_err(), "kind 1 with DXT5 fourCC");

    // tail_len == bs (bs = 16 for kind 5)
    let mut bad = rec.clone();
    bad[9] = 16;
    assert!(dxt_merge(&bad).is_err(), "tail_len == bs");
}

// ---------- 5. fuzz ----------

#[test]
fn fuzz_mutated_records_never_panic() {
    let mut rng = Lcg::new(0xDEADBEEF);
    let fourccs: [&[u8; 4]; 5] = [b"DXT1", b"DXT2", b"DXT3", b"DXT4", b"DXT5"];
    let mut seeds: Vec<Vec<u8>> = Vec::new();
    for &fcc in fourccs.iter() {
        let bs = bs_for(fcc);
        for &n in [0usize, 1, 3, 17].iter() {
            for &tail in [0usize, 1, 5, bs - 1].iter() {
                let dds = make_dds(fcc, n, tail, &mut rng);
                seeds.push(dxt_split(&dds).unwrap());
            }
        }
    }
    assert_eq!(seeds.len(), 80);

    for _ in 0..3000usize {
        let mut r = seeds[(rng.next_u32() as usize) % seeds.len()].clone();
        if rng.next_u32() % 2 == 0 {
            // 1..=3 flipped bits
            let flips = 1 + (rng.next_u32() % 3) as usize;
            for _ in 0..flips {
                let pos = (rng.next_u32() as usize) % r.len();
                let bit = rng.next_u32() % 8;
                r[pos] ^= 1u8 << bit;
            }
        } else {
            // 1..=4 overwritten bytes
            let writes = 1 + (rng.next_u32() % 4) as usize;
            for _ in 0..writes {
                let pos = (rng.next_u32() as usize) % r.len();
                r[pos] = rng.byte();
            }
        }
        // must return without panicking
        let _ = dxt_merge(&r);
    }
}
