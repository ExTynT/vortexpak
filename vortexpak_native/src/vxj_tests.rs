use super::*;

// ---------------------------------------------------------------------------
// Deterministic PRNG (inline LCG, no external crates).
// ---------------------------------------------------------------------------
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

    // Random value in 0..n (n > 0).
    fn below(&mut self, n: u32) -> u32 {
        self.next_u32() % n
    }

    fn byte(&mut self) -> u8 {
        self.next_u32() as u8
    }
}

// ---------------------------------------------------------------------------
// Hand-built tiny baseline JPEG: 8x8, 1 component, quantizer steps all 1,
// one block coded as DC category 0 + EOB.
// ---------------------------------------------------------------------------

// JPEG Annex K.3 Table K.3: standard luminance DC BITS.
const DC_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
// Luminance DC values 0..=11.
const DC_VALS: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

// JPEG Annex K.3 Table K.5: standard luminance AC BITS.
const AC_BITS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D];
// Standard luminance AC values in canonical code order (162 entries).
const AC_VALS: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];

fn tiny_jpeg() -> Vec<u8> {
    let mut j: Vec<u8> = Vec::with_capacity(313);

    // SOI
    j.extend_from_slice(&[0xFF, 0xD8]);

    // DQT: len 67, Pq/Tq 0x00, 64 quantizer steps of 1
    j.extend_from_slice(&[0xFF, 0xDB, 0x00, 0x43, 0x00]);
    j.extend_from_slice(&[1u8; 64]);

    // SOF0: len 11, precision 8, height 8, width 8, 1 component (id 1, sampling 0x11, table 0)
    j.extend_from_slice(&[
        0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x08, 0x00, 0x08, 0x01, 0x01, 0x11, 0x00,
    ]);

    // DHT: class 0 (DC), id 0, len 31
    j.extend_from_slice(&[0xFF, 0xC4, 0x00, 0x1F, 0x00]);
    j.extend_from_slice(&DC_BITS);
    j.extend_from_slice(&DC_VALS);

    // DHT: class 1 (AC), id 0, len 181
    j.extend_from_slice(&[0xFF, 0xC4, 0x00, 0xB5, 0x10]);
    j.extend_from_slice(&AC_BITS);
    j.extend_from_slice(&AC_VALS);

    // SOS: len 8, 1 component (id 1, tables 0x00), Ss 0, Se 63, Ah/Al 0
    j.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00]);

    // Scan data: DC category 0 ('00') + AC EOB ('1010') + '11' padding = 0x2B
    j.push(0x2B);

    // EOI
    j.extend_from_slice(&[0xFF, 0xD9]);
    j
}

// 1. Tiny valid JPEG round-trips exactly; trailing garbage after EOI is kept.
#[test]
fn test1_tiny_jpeg_roundtrip() {
    let jpeg = tiny_jpeg();
    let rec = vxj_encode(&jpeg).expect("tiny hand-built JPEG must encode");
    assert!(
        rec.starts_with(b"VXJ1"),
        "record must start with VXJ1 magic"
    );
    let back = vxj_decode(&rec).expect("own record must decode");
    assert_eq!(back, jpeg, "decode must reproduce the JPEG byte-for-byte");

    // Same file with 5 garbage bytes after EOI must also round-trip exactly.
    let mut with_garbage = tiny_jpeg();
    with_garbage.extend_from_slice(&[0x00, 0x42, 0x99, 0xFF, 0xD7]);
    let rec2 = vxj_encode(&with_garbage).expect("JPEG with trailing garbage must encode");
    assert!(rec2.starts_with(b"VXJ1"));
    assert_eq!(
        vxj_decode(&rec2).unwrap(),
        with_garbage,
        "trailing garbage after EOI must be preserved exactly"
    );
}

// 2. Malformed / unsupported inputs return Err without panicking.
#[test]
fn test2_rejects_invalid_inputs() {
    assert!(vxj_encode(&[]).is_err(), "empty input must be Err");
    assert!(vxj_encode(b"hello").is_err(), "non-JPEG input must be Err");
    assert!(vxj_encode(&[0xFF, 0xD8]).is_err(), "SOI only must be Err");

    // Truncated at every length 0..len-1.
    let jpeg = tiny_jpeg();
    for cut in 0..jpeg.len() {
        let r = vxj_encode(&jpeg[..cut]);
        assert!(r.is_err(), "truncation at {} bytes must be Err", cut);
    }

    // SOF0 replaced by SOF2 (progressive) -> not supported.
    let mut progressive = tiny_jpeg();
    let sof0 = progressive
        .windows(2)
        .position(|w| w[0] == 0xFF && w[1] == 0xC0)
        .expect("SOF0 marker present in tiny JPEG");
    progressive[sof0 + 1] = 0xC2;
    assert!(
        vxj_encode(&progressive).is_err(),
        "progressive JPEG (SOF2) must be Err"
    );
}

// 3. Decoder robustness: any input, any result, but never a panic.
#[test]
fn test3_decode_never_panics() {
    let _ = vxj_decode(&[]);
    let _ = vxj_decode(b"VXJ1");

    let record = vxj_encode(&tiny_jpeg()).expect("encode for mutation base");

    let mut rng = Lcg::new(0xC0FF_EE01);

    // 1,000 random byte strings, lengths 0..200.
    for _ in 0..1000 {
        let len = rng.below(200) as usize;
        let buf: Vec<u8> = (0..len).map(|_| rng.byte()).collect();
        let _ = vxj_decode(&buf);
    }

    // 2,000 mutations of the valid record: 1-3 flipped bits or 1-4 overwritten bytes.
    for _ in 0..2000 {
        let mut m = record.clone();
        if rng.below(2) == 0 {
            let flips = 1 + rng.below(3);
            for _ in 0..flips {
                let pos = rng.below(m.len() as u32) as usize;
                m[pos] ^= 1u8 << rng.below(8);
            }
        } else {
            let overwrites = 1 + rng.below(4);
            for _ in 0..overwrites {
                let pos = rng.below(m.len() as u32) as usize;
                m[pos] = rng.byte();
            }
        }
        let _ = vxj_decode(&m);
    }
}

// 4. Encoding is deterministic.
#[test]
fn test4_encode_is_deterministic() {
    let jpeg = tiny_jpeg();
    let a = vxj_encode(&jpeg).expect("first encode must succeed");
    let b = vxj_encode(&jpeg).expect("second encode must succeed");
    assert_eq!(
        a, b,
        "encoding the same JPEG twice must give identical records"
    );
}
