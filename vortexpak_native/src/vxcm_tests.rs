use super::*;

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn byte(&mut self) -> u8 {
        (self.next_u64() >> 32) as u8
    }
    fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() >> 16) as usize) % n
    }
}

fn lcg_bytes(seed: u64, len: usize) -> Vec<u8> {
    let mut rng = Lcg::new(seed);
    (0..len).map(|_| rng.byte()).collect()
}

fn pattern_data(pat: &[u8], len: usize) -> Vec<u8> {
    (0..len).map(|i| pat[i % pat.len()]).collect()
}

fn text_like(seed: u64, len: usize) -> Vec<u8> {
    const WORDS: [&str; 30] = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "vortex",
        "pack",
        "compression",
        "archive",
        "stream",
        "block",
        "level",
        "codec",
        "byte",
        "range",
        "coder",
        "match",
        "pattern",
        "window",
        "hash",
        "chain",
        "literal",
        "copy",
        "length",
        "distance",
        "token",
        "end",
    ];
    let mut rng = Lcg::new(seed);
    let mut out = Vec::with_capacity(len + 16);
    while out.len() < len {
        if !out.is_empty() {
            out.push(b' ');
        }
        out.extend_from_slice(WORDS[rng.below(WORDS.len())].as_bytes());
    }
    out
}

fn records_data(seed: u64, len: usize) -> Vec<u8> {
    let mut rng = Lcg::new(seed);
    let mut base = [0u8; 16];
    for b in base.iter_mut() {
        *b = rng.byte();
    }
    let mut out = Vec::with_capacity(len + 16);
    while out.len() < len {
        let mut rec = base;
        let flips = 1 + rng.below(3);
        for _ in 0..flips {
            let pos = rng.below(16);
            rec[pos] ^= 1u8 << rng.below(3);
        }
        out.extend_from_slice(&rec);
    }
    out.truncate(len);
    out
}

fn dup_half_data(total: usize) -> Vec<u8> {
    let half = lcg_bytes(77, total / 2);
    let mut v = half.clone();
    v.extend_from_slice(&half);
    v
}

fn cycle_data(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i & 0xff) as u8).collect()
}

fn roundtrip_corpus() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("empty", Vec::new()),
        ("one_byte", vec![0xAB]),
        ("two_bytes", vec![0x00, 0xFF]),
        ("zeros_255", vec![0u8; 255]),
        ("rand_100k", lcg_bytes(11, 100_000)),
        ("pattern7_100k", pattern_data(b"VORTEX1", 100_000)),
        ("text_200k", text_like(22, 200_000)),
        ("records_64k", records_data(33, 65_536)),
        ("dup_half_1m", dup_half_data(1_048_576)),
        ("cycle256_64k", cycle_data(65_536)),
    ]
}

fn strict_prefix_lengths(len: usize, cap: usize) -> Vec<usize> {
    if len <= cap {
        (0..len).collect()
    } else {
        let mut lens: Vec<usize> = (0..cap).map(|k| k * (len - 1) / (cap - 1)).collect();
        lens.sort_unstable();
        lens.dedup();
        lens
    }
}

#[test]
fn roundtrip_levels_on_corpus() {
    let corpus = roundtrip_corpus();
    for &level in &[1u8, 5, 6, 9] {
        for (name, data) in &corpus {
            let comp = vxcm_compress(data, level);
            let out = vxcm_decompress(&comp, data.len()).unwrap_or_else(|e| {
                panic!("level {} corpus {}: decompress failed: {}", level, name, e)
            });
            assert_eq!(
                out, *data,
                "roundtrip mismatch at level {} for {}",
                level, name
            );
        }
    }
}

#[test]
fn ratio_sanity() {
    let pattern = pattern_data(b"VORTEX1", 1_048_576);
    let comp = vxcm_compress(&pattern, 5);
    assert!(
        comp.len() * 100 < pattern.len(),
        "1 MB repeated pattern compressed to {} bytes ({:.4}%), expected < 1%",
        comp.len(),
        100.0 * comp.len() as f64 / pattern.len() as f64
    );

    let random = lcg_bytes(44, 100_000);
    let comp = vxcm_compress(&random, 5);
    let bound = random.len() + random.len() / 100 + 64;
    assert!(
        comp.len() <= bound,
        "100 KB random expanded to {} bytes, allowed bound {}",
        comp.len(),
        bound
    );
}

#[test]
fn header_format() {
    let inputs: Vec<Vec<u8>> = vec![Vec::new(), vec![1u8, 2, 3], lcg_bytes(55, 4096)];
    for level in 1u8..=9 {
        for data in &inputs {
            let comp = vxcm_compress(data, level);
            assert!(
                comp.len() >= 6,
                "stream shorter than 6-byte header at level {}",
                level
            );
            assert_eq!(&comp[0..4], b"VXC1", "bad magic at level {}", level);
            assert!(
                comp[4] < 225,
                "props byte {} out of range at level {}",
                comp[4],
                level
            );
        }
    }
}

#[test]
fn level_clamp_roundtrip() {
    let data = text_like(66, 32 * 1024);
    for level in [0u8, 200] {
        let comp = vxcm_compress(&data, level);
        let out = vxcm_decompress(&comp, data.len())
            .unwrap_or_else(|e| panic!("level {} clamp roundtrip failed: {}", level, e));
        assert_eq!(out, data, "level {} clamp roundtrip mismatch", level);
    }
}

#[test]
fn robustness_wrong_raw_len() {
    let corpus = roundtrip_corpus();
    for (name, data) in &corpus {
        if data.is_empty() {
            continue;
        }
        let comp = vxcm_compress(data, 5);
        for raw_len in [data.len() + 1, data.len() - 1] {
            match vxcm_decompress(&comp, raw_len) {
                Ok(out) => assert_ne!(
                    out, *data,
                    "raw_len {} on corpus {} reproduced the original",
                    raw_len, name
                ),
                Err(_) => {}
            }
        }
    }
}

#[test]
fn robustness_strict_prefixes() {
    let data = text_like(88, 50_000);
    let comp = vxcm_compress(&data, 5);
    assert!(comp.len() > 6, "compressed stream suspiciously tiny");
    for plen in strict_prefix_lengths(comp.len(), 300) {
        match vxcm_decompress(&comp[..plen], data.len()) {
            Ok(out) => assert_ne!(
                out, data,
                "prefix of length {} reproduced the original",
                plen
            ),
            Err(_) => {}
        }
    }
}

#[test]
fn robustness_mutated_streams() {
    let entries: Vec<Vec<u8>> = vec![
        vec![0x00, 0xFF],
        vec![0u8; 255],
        pattern_data(b"VORTEX1", 8192),
        text_like(99, 8192),
        records_data(111, 8192),
        lcg_bytes(123, 8192),
        cycle_data(8192),
    ];
    let corpus: Vec<(Vec<u8>, usize)> = entries
        .iter()
        .map(|d| (vxcm_compress(d, 5), d.len()))
        .filter(|(c, _)| c.len() > 6)
        .collect();
    assert!(
        !corpus.is_empty(),
        "no mutable compressed streams available"
    );

    let mut rng = Lcg::new(424242);
    for _ in 0..2000 {
        let idx = rng.below(corpus.len());
        let (stream, raw_len) = &corpus[idx];
        let mut m = stream.clone();
        if rng.below(2) == 0 {
            let flips = 1 + rng.below(3);
            for _ in 0..flips {
                let pos = 6 + rng.below(m.len() - 6);
                m[pos] ^= 1u8 << rng.below(8);
            }
        } else {
            let start = 6 + rng.below(m.len() - 6);
            let n = 1 + rng.below(4);
            for i in 0..n {
                if start + i < m.len() {
                    m[start + i] = rng.byte();
                }
            }
        }
        let _ = vxcm_decompress(&m, *raw_len); // must not panic; any result accepted
    }
}

#[test]
fn robustness_bad_magic_and_props() {
    let data = pattern_data(b"VORTEX1", 4096);
    let comp = vxcm_compress(&data, 5);
    for magic in [&b"VXC2"[..], &b"1CXV"[..], &b"VXCM"[..], &b"\0\0\0\0"[..]] {
        let mut bad = comp.clone();
        bad[0..4].copy_from_slice(magic);
        assert!(
            vxcm_decompress(&bad, data.len()).is_err(),
            "wrong magic {:?} must be rejected",
            magic
        );
    }
    let mut bad = comp.clone();
    bad[4] = 255;
    assert!(
        vxcm_decompress(&bad, data.len()).is_err(),
        "props byte 255 must be rejected"
    );
}

#[test]
fn determinism() {
    let inputs: Vec<Vec<u8>> = vec![text_like(77, 32 * 1024), lcg_bytes(78, 32 * 1024)];
    for level in [1u8, 5, 9] {
        for (i, data) in inputs.iter().enumerate() {
            let a = vxcm_compress(data, level);
            let b = vxcm_compress(data, level);
            assert_eq!(a, b, "input {} level {} not deterministic", i, level);
        }
    }
}
