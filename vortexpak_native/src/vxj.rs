use std::cmp::{max, min};

const ZZ: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];
type Block = [i16; 64];
const fn ac_positions<const N: usize>(inner: bool) -> [usize; N] {
    let mut out = [0; N];
    let mut count = 0;
    let mut i = 1;
    while i < 64 {
        let p = ZZ[i];
        if (p / 8 != 0 && p % 8 != 0) == inner {
            out[count] = p;
            count += 1;
        }
        i += 1;
    }
    out
}
const INNER: [usize; 49] = ac_positions(true);
const EDGES: [usize; 14] = ac_positions(false);
const fn zigzag_bits() -> [u64; 64] {
    let mut bits = [0; 64];
    let mut i = 1;
    while i < 64 {
        bits[ZZ[i]] = 1u64 << i;
        i += 1;
    }
    bits
}
const ZIGZAG_BITS: [u64; 64] = zigzag_bits();

fn bad<T>(s: &str) -> Result<T, String> {
    Err(s.to_string())
}
fn be16(b: &[u8]) -> usize {
    ((b[0] as usize) << 8) | b[1] as usize
}
fn zeros<T: Clone>(len: usize, value: T) -> Result<Vec<T>, String> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| "allocation limit".to_string())?;
    v.resize(len, value);
    Ok(v)
}

#[derive(Clone)]
struct Huff {
    code: [u16; 256],
    len: [u8; 256],
    fast: [u16; 1024],
    first: [u32; 17],
    count: [u32; 17],
    offset: [usize; 17],
    symbols: Vec<u8>,
}
impl Huff {
    fn new(counts: &[u8], symbols: &[u8], ac: bool) -> Result<Self, String> {
        if symbols.is_empty() || symbols.len() > 256 {
            return bad("invalid Huffman table");
        }
        let mut t = Self {
            code: [0; 256],
            len: [0; 256],
            fast: [0; 1024],
            first: [0; 17],
            count: [0; 17],
            offset: [0; 17],
            symbols: symbols.to_vec(),
        };
        let mut c = 0u32;
        let mut k = 0;
        for n in 1..=16 {
            let count = counts[n - 1] as u32;
            if c + count >= 1u32 << n && count != 0 {
                return bad("invalid Huffman codes");
            }
            t.first[n] = c;
            t.count[n] = count;
            t.offset[n] = k;
            for _ in 0..count {
                let s = symbols[k] as usize;
                if t.len[s] != 0
                    || (!ac && s > 11)
                    || (ac && ((s & 15) > 10 || ((s & 15) == 0 && s != 0 && s != 240)))
                {
                    return bad("invalid Huffman symbol");
                }
                t.code[s] = c as u16;
                t.len[s] = n as u8;
                if n <= 10 {
                    let begin = (c as usize) << (10 - n);
                    let end = begin + (1 << (10 - n));
                    for v in &mut t.fast[begin..end] {
                        *v = ((n as u16) << 8) | s as u16;
                    }
                }
                c += 1;
                k += 1;
            }
            c <<= 1;
        }
        Ok(t)
    }
    fn read(&self, r: &mut Bits<'_>) -> Result<u8, String> {
        r.fill(10);
        if r.n >= 10 {
            let v = self.fast[((r.bits >> (r.n - 10)) & 1023) as usize];
            if v != 0 {
                r.n -= (v >> 8) as u8;
                return Ok(v as u8);
            }
        }
        let mut code = 0u32;
        for n in 1..=16 {
            code = (code << 1) | r.get(1)?;
            if code >= self.first[n] && code - self.first[n] < self.count[n] {
                return Ok(self.symbols[self.offset[n] + (code - self.first[n]) as usize]);
            }
        }
        bad("invalid Huffman scan")
    }
    fn write(&self, w: &mut BitOut, s: usize) -> Result<(), String> {
        if self.len[s] == 0 {
            return bad("not reproducible");
        }
        w.put(self.code[s] as u32, self.len[s]);
        Ok(())
    }
    #[inline(always)]
    fn write_value(&self, w: &mut BitOut, s: usize, value: i32, n: u8) -> Result<(), String> {
        let len = self.len[s];
        if len == 0 {
            return bad("not reproducible");
        }
        let amplitude = if value < 0 {
            (value + (1 << n) - 1) as u32
        } else {
            value as u32
        };
        w.put(((self.code[s] as u32) << n) | amplitude, len + n);
        Ok(())
    }
}

#[derive(Clone)]
struct Component {
    id: u8,
    h: usize,
    v: usize,
    q: usize,
}
#[derive(Clone)]
struct Frame {
    w: usize,
    h: usize,
    max_h: usize,
    max_v: usize,
    comps: Vec<Component>,
}
#[derive(Clone)]
struct Select {
    ci: usize,
    dc: usize,
    ac: usize,
}
#[derive(Clone)]
struct Scan {
    sels: Vec<Select>,
}
#[derive(Clone)]
struct Header {
    frame: Option<Frame>,
    q: [bool; 4],
    dc: [Option<Huff>; 4],
    ac: [Option<Huff>; 4],
    restart: usize,
}
impl Header {
    fn new() -> Self {
        Self {
            frame: None,
            q: [false; 4],
            dc: std::array::from_fn(|_| None),
            ac: std::array::from_fn(|_| None),
            restart: 0,
        }
    }
    fn segment(&mut self, marker: u8, b: &[u8]) -> Result<(), String> {
        match marker {
            0xc0 | 0xc1 => {
                if self.frame.is_some() || b.len() < 6 || b[0] != 8 {
                    return bad("unsupported frame");
                }
                let h = be16(&b[1..3]);
                let w = be16(&b[3..5]);
                let nc = b[5] as usize;
                if nc == 0 || nc > 4 || b.len() != 6 + 3 * nc || w == 0 || h == 0 {
                    return bad("invalid frame");
                }
                if (w as u64) * (h as u64) * (nc as u64) > 1 << 28 {
                    return bad("image limit");
                }
                let mut comps: Vec<Component> = Vec::new();
                for c in b[6..].chunks_exact(3) {
                    let hc = (c[1] >> 4) as usize;
                    let vc = (c[1] & 15) as usize;
                    if hc == 0
                        || hc > 4
                        || vc == 0
                        || vc > 4
                        || c[2] > 3
                        || comps.iter().any(|v| v.id == c[0])
                    {
                        return bad("invalid components");
                    }
                    comps.push(Component {
                        id: c[0],
                        h: hc,
                        v: vc,
                        q: c[2] as usize,
                    });
                }
                self.frame = Some(Frame {
                    w,
                    h,
                    max_h: comps.iter().map(|c| c.h).max().unwrap_or(1),
                    max_v: comps.iter().map(|c| c.v).max().unwrap_or(1),
                    comps,
                });
            }
            0xdb => {
                let mut p = 0;
                if b.is_empty() {
                    return bad("empty DQT");
                }
                while p < b.len() {
                    let t = b[p];
                    p += 1;
                    let precision = t >> 4;
                    let id = (t & 15) as usize;
                    if precision > 1 || id > 3 {
                        return bad("invalid DQT");
                    }
                    let size = 64 * (precision as usize + 1);
                    if size > b.len() - p {
                        return bad("truncated DQT");
                    }
                    if precision == 0 {
                        if b[p..p + size].contains(&0) {
                            return bad("zero quantizer");
                        }
                    } else if b[p..p + size].chunks_exact(2).any(|v| be16(v) == 0) {
                        return bad("zero quantizer");
                    }
                    p += size;
                    self.q[id] = true;
                }
            }
            0xc4 => {
                let mut p = 0;
                if b.is_empty() {
                    return bad("empty DHT");
                }
                while p < b.len() {
                    if b.len() - p < 17 {
                        return bad("truncated DHT");
                    }
                    let tag = b[p];
                    p += 1;
                    if tag >> 4 > 1 || tag & 15 > 3 {
                        return bad("invalid DHT");
                    }
                    let counts = &b[p..p + 16];
                    p += 16;
                    let n: usize = counts.iter().map(|v| *v as usize).sum();
                    if n > b.len() - p {
                        return bad("truncated DHT");
                    }
                    let h = Huff::new(counts, &b[p..p + n], tag >> 4 == 1)?;
                    p += n;
                    if tag >> 4 == 0 {
                        self.dc[(tag & 15) as usize] = Some(h);
                    } else {
                        self.ac[(tag & 15) as usize] = Some(h);
                    }
                }
            }
            0xdd => {
                if b.len() != 2 {
                    return bad("invalid DRI");
                }
                self.restart = be16(b);
            }
            0xe0..=0xef | 0xfe => {}
            _ => return bad("unsupported JPEG marker"),
        }
        Ok(())
    }
    fn scan(&self, b: &[u8]) -> Result<Scan, String> {
        let f = self.frame.as_ref().ok_or("missing frame")?;
        if b.is_empty() {
            return bad("invalid SOS");
        }
        let n = b[0] as usize;
        if n == 0 || n > f.comps.len() || b.len() != 1 + 2 * n + 3 || b[1 + 2 * n..] != [0, 63, 0] {
            return bad("unsupported scan");
        }
        let mut sels: Vec<Select> = Vec::new();
        for c in b[1..1 + 2 * n].chunks_exact(2) {
            let ci = f
                .comps
                .iter()
                .position(|v| v.id == c[0])
                .ok_or("unknown scan component")?;
            let dc = (c[1] >> 4) as usize;
            let ac = (c[1] & 15) as usize;
            if dc > 3
                || ac > 3
                || self.dc[dc].is_none()
                || self.ac[ac].is_none()
                || !self.q[f.comps[ci].q]
                || sels.iter().any(|s| s.ci == ci)
            {
                return bad("invalid scan tables or components");
            }
            sels.push(Select { ci, dc, ac });
        }
        Ok(Scan { sels })
    }
}

fn marker<'a>(data: &'a [u8], p: &mut usize) -> Result<(u8, &'a [u8]), String> {
    if data.get(*p) != Some(&255) {
        return bad("missing JPEG marker");
    }
    while data.get(*p) == Some(&255) {
        *p += 1;
    }
    let m = *data.get(*p).ok_or("truncated marker")?;
    *p += 1;
    if m == 0 || m == 0xd8 || m == 1 || (0xd0..=0xd7).contains(&m) {
        return bad("unexpected JPEG marker");
    }
    if m == 0xd9 {
        return Ok((m, &[]));
    }
    if data.len().saturating_sub(*p) < 2 {
        return bad("truncated segment");
    }
    let n = be16(&data[*p..*p + 2]);
    if n < 2 || n > data.len() - *p {
        return bad("invalid segment length");
    }
    let b = &data[*p + 2..*p + n];
    *p += n;
    Ok((m, b))
}
fn first_header(data: &[u8]) -> Result<(Header, Scan, usize), String> {
    if !data.starts_with(&[255, 216]) {
        return bad("not JPEG");
    }
    let mut h = Header::new();
    let mut p = 2;
    loop {
        let (m, b) = marker(data, &mut p)?;
        if m == 0xda {
            return Ok((h.clone(), h.scan(b)?, p));
        }
        if m == 0xd9 {
            return bad("missing scan");
        }
        h.segment(m, b)?;
    }
}

struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    bits: u64,
    n: u8,
}
impl<'a> Bits<'a> {
    fn new(data: &'a [u8], pos: usize) -> Self {
        Self {
            data,
            pos,
            bits: 0,
            n: 0,
        }
    }
    fn fill(&mut self, n: u8) {
        while self.n < n {
            let Some(&v) = self.data.get(self.pos) else {
                break;
            };
            if v == 255 {
                if self.data.get(self.pos + 1) != Some(&0) {
                    break;
                }
                self.pos += 1;
            }
            self.pos += 1;
            self.bits = (self.bits << 8) | v as u64;
            self.n += 8;
        }
    }
    fn get(&mut self, n: u8) -> Result<u32, String> {
        self.fill(n);
        if self.n < n {
            return bad("truncated entropy scan");
        }
        self.n -= n;
        Ok(((self.bits >> self.n) & ((1u64 << n) - 1)) as u32)
    }
    fn align(&mut self) -> Result<(), String> {
        if self.n > 7 || self.bits & ((1u64 << self.n) - 1) != (1u64 << self.n) - 1 {
            return bad("not reproducible");
        }
        self.n = 0;
        self.bits = 0;
        Ok(())
    }
}
struct BitOut {
    data: Vec<u8>,
    bits: u64,
    n: u8,
}
impl BitOut {
    fn new() -> Self {
        Self {
            data: Vec::new(),
            bits: 0,
            n: 0,
        }
    }
    fn put(&mut self, v: u32, n: u8) {
        // Each symbol contributes at most 27 bits to a tail shorter than 32 bits.
        self.bits |= (v as u64) << (64 - self.n - n);
        self.n += n;
        if self.n >= 32 {
            let bytes = (self.bits >> 32) as u32;
            let bytes = bytes.to_be_bytes();
            if bytes.iter().all(|&b| b != 255) {
                self.data.extend_from_slice(&bytes);
            } else {
                for b in bytes {
                    self.data.push(b);
                    if b == 255 {
                        self.data.push(0);
                    }
                }
            }
            self.bits <<= 32;
            self.n -= 32;
        }
    }
    fn align(&mut self) {
        let padding = (8 - (self.n & 7)) & 7;
        if padding != 0 {
            self.put((1 << padding) - 1, padding);
        }
        while self.n != 0 {
            let b = (self.bits >> 56) as u8;
            self.bits <<= 8;
            self.n -= 8;
            self.data.push(b);
            if b == 255 {
                self.data.push(0);
            }
        }
    }
}
struct Plane {
    cols: usize,
    rows: usize,
    blocks: Vec<Block>,
    nonzero: Vec<u64>,
}
fn layout(h: &Header, s: &Scan, block_limit: usize) -> Result<(Vec<Plane>, usize, usize), String> {
    let f = h.frame.as_ref().ok_or("missing frame")?;
    let mx = f.w.div_ceil(8 * f.max_h);
    let my = f.h.div_ceil(8 * f.max_v);
    let mut planes = Vec::new();
    let mut total = 0usize;
    for sel in &s.sels {
        let c = &f.comps[sel.ci];
        let (cols, rows) = if s.sels.len() == 1 {
            (
                (f.w * c.h).div_ceil(8 * f.max_h),
                (f.h * c.v).div_ceil(8 * f.max_v),
            )
        } else {
            (mx * c.h, my * c.v)
        };
        let count = cols.checked_mul(rows).ok_or("image limit")?;
        total = total.checked_add(count).ok_or("image limit")?;
        planes.push(Plane {
            cols,
            rows,
            blocks: Vec::new(),
            nonzero: Vec::new(),
        });
    }
    if total > block_limit {
        return bad("truncated coefficient data");
    }
    for p in &mut planes {
        p.blocks = zeros(p.cols * p.rows, [0; 64])?;
    }
    let (x, y) = if planes.len() == 1 {
        (planes[0].cols, planes[0].rows)
    } else {
        (mx, my)
    };
    Ok((planes, x, y))
}
fn extend(v: u32, n: u8) -> i32 {
    if n == 0 {
        0
    } else if v < 1 << (n - 1) {
        v as i32 + 1 - (1 << n)
    } else {
        v as i32
    }
}
#[inline(always)]
fn read_block(
    r: &mut Bits<'_>,
    dc: &Huff,
    ac: &Huff,
    predictor: &mut i32,
) -> Result<Block, String> {
    let mut b = [0i16; 64];
    let n = dc.read(r)?;
    let v = *predictor + extend(r.get(n)?, n);
    b[0] = i16::try_from(v).map_err(|_| "DC out of range")?;
    *predictor = v;
    let mut k = 1;
    while k < 64 {
        let t = ac.read(r)?;
        let n = t & 15;
        let run = (t >> 4) as usize;
        if n == 0 {
            if run == 0 {
                break;
            }
            if run != 15 || k + 16 > 64 {
                return bad("invalid zero run");
            }
            k += 16;
        } else {
            k += run;
            if k >= 64 {
                return bad("AC run out of range");
            }
            b[ZZ[k]] = extend(r.get(n)?, n) as i16;
            k += 1;
        }
    }
    Ok(b)
}
fn category(v: i32) -> u8 {
    (32 - v.unsigned_abs().leading_zeros()) as u8
}
#[inline(always)]
fn write_block(
    w: &mut BitOut,
    b: &Block,
    dc: &Huff,
    ac: &Huff,
    predictor: &mut i32,
    nonzero: Option<u64>,
) -> Result<(), String> {
    let d = b[0] as i32 - *predictor;
    *predictor = b[0] as i32;
    let n = category(d);
    if n > 11 {
        return bad("DC difference out of range");
    }
    dc.write_value(w, n as usize, d, n)?;
    let mut mask = nonzero.unwrap_or_else(|| {
        let mut mask = 0;
        for pos in 1..64 {
            mask |= ZIGZAG_BITS[pos] & 0u64.wrapping_sub((b[pos] != 0) as u64);
        }
        mask
    });
    let mut next = 1;
    while mask != 0 {
        let k = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        let mut run = k - next;
        let v = b[ZZ[k]] as i32;
        while run >= 16 {
            ac.write(w, 240)?;
            run -= 16;
        }
        let n = category(v);
        if n > 10 {
            return bad("AC out of range");
        }
        ac.write_value(w, (run << 4) | n as usize, v, n)?;
        next = k + 1;
    }
    if next != 64 {
        ac.write(w, 0)?;
    }
    Ok(())
}
fn read_scan(
    data: &[u8],
    start: usize,
    h: &Header,
    s: &Scan,
) -> Result<(Vec<Plane>, usize), String> {
    // A JPEG block needs at least one DC bit and one AC bit.
    let (mut planes, nx, ny) = layout(h, s, data.len().saturating_sub(start).saturating_mul(4))?;
    let f = h.frame.as_ref().ok_or("missing frame")?;
    let mut r = Bits::new(data, start);
    let mut pred = [0i32; 4];
    let mut rst = 0;
    let total = nx * ny;
    for m in 0..total {
        for (pi, sel) in s.sels.iter().enumerate() {
            let c = &f.comps[sel.ci];
            let (bh, bv) = if planes.len() == 1 {
                (1, 1)
            } else {
                (c.h, c.v)
            };
            let dc = h.dc[sel.dc].as_ref().ok_or("missing DC table")?;
            let ac = h.ac[sel.ac].as_ref().ok_or("missing AC table")?;
            let p = &mut planes[pi];
            for dy in 0..bv {
                for dx in 0..bh {
                    let index = ((m / nx) * bv + dy) * p.cols + (m % nx) * bh + dx;
                    p.blocks[index] = read_block(&mut r, dc, ac, &mut pred[sel.ci])?;
                }
            }
        }
        if h.restart != 0 && (m + 1) % h.restart == 0 && m + 1 < total {
            r.align()?;
            if data.get(r.pos..r.pos + 2) != Some(&[255, 0xd0 + rst][..]) {
                return bad("invalid restart marker");
            }
            r.pos += 2;
            rst = (rst + 1) & 7;
            pred = [0; 4];
        }
    }
    r.align()?;
    if data.get(r.pos) != Some(&255)
        || data.get(r.pos + 1).is_none()
        || data[r.pos + 1] == 0
        || (0xd0..=0xd7).contains(&data[r.pos + 1])
    {
        return bad("invalid scan ending");
    }
    Ok((planes, r.pos))
}
fn write_scan(h: &Header, s: &Scan, planes: &[Plane]) -> Result<Vec<u8>, String> {
    let f = h.frame.as_ref().ok_or("missing frame")?;
    let (nx, ny) = if planes.len() == 1 {
        (planes[0].cols, planes[0].rows)
    } else {
        (f.w.div_ceil(8 * f.max_h), f.h.div_ceil(8 * f.max_v))
    };
    let mut w = BitOut::new();
    let capacity = planes.iter().map(|p| p.blocks.len()).sum::<usize>() * 16;
    w.data
        .try_reserve_exact(capacity)
        .map_err(|_| "allocation limit")?;
    let mut pred = [0i32; 4];
    let mut rst = 0;
    for m in 0..nx * ny {
        for (pi, sel) in s.sels.iter().enumerate() {
            let c = &f.comps[sel.ci];
            let (bh, bv) = if planes.len() == 1 {
                (1, 1)
            } else {
                (c.h, c.v)
            };
            let dc = h.dc[sel.dc].as_ref().ok_or("missing DC table")?;
            let ac = h.ac[sel.ac].as_ref().ok_or("missing AC table")?;
            let p = &planes[pi];
            for dy in 0..bv {
                for dx in 0..bh {
                    let index = ((m / nx) * bv + dy) * p.cols + (m % nx) * bh + dx;
                    write_block(
                        &mut w,
                        &p.blocks[index],
                        dc,
                        ac,
                        &mut pred[sel.ci],
                        p.nonzero.get(index).copied(),
                    )?;
                }
            }
        }
        if h.restart != 0 && (m + 1) % h.restart == 0 && m + 1 < nx * ny {
            w.align();
            w.data.extend_from_slice(&[255, 0xd0 + rst]);
            rst = (rst + 1) & 7;
            pred = [0; 4];
        }
    }
    w.align();
    Ok(w.data)
}
// Later complete scans stay verbatim in the trailer, but must also be reproducible.
fn validate_tail(data: &[u8], start: usize, mut h: Header, first: &Scan) -> Result<(), String> {
    let mut seen = [false; 4];
    for s in &first.sels {
        seen[s.ci] = true;
    }
    let mut p = start;
    loop {
        let (m, b) = marker(data, &mut p)?;
        if m == 0xd9 {
            let f = h.frame.as_ref().ok_or("missing frame")?;
            if seen[..f.comps.len()].iter().any(|v| !*v) {
                return bad("missing component scan");
            }
            return Ok(());
        }
        if m == 0xda {
            let s = h.scan(b)?;
            for c in &s.sels {
                if seen[c.ci] {
                    return bad("repeated sequential scan");
                }
                seen[c.ci] = true;
            }
            let begin = p;
            let (planes, end) = read_scan(data, begin, &h, &s)?;
            if write_scan(&h, &s, &planes)? != data[begin..end] {
                return bad("not reproducible");
            }
            p = end;
        } else {
            h.segment(m, b)?;
        }
    }
}

trait Binary {
    const READ: bool;
    fn bit<const RATE: u8>(&mut self, p: &mut u16, value: bool) -> bool;
    fn check(&self) -> Result<(), String> {
        Ok(())
    }
}
const fn adaptation_table(rate: u8) -> [[u16; 2]; 4096] {
    let mut table = [[0; 2]; 4096];
    let mut p = 1u16;
    while p < 4096 {
        let up_step = (4096 - p) >> rate;
        let down_step = p >> rate;
        let up = p + if up_step == 0 { 1 } else { up_step };
        let down = p - if down_step == 0 { 1 } else { down_step };
        table[p as usize] = [
            if up > 4095 { 4095 } else { up },
            if down < 1 { 1 } else { down },
        ];
        p += 1;
    }
    table
}
static ADAPT4: [[u16; 2]; 4096] = adaptation_table(4);
static ADAPT6: [[u16; 2]; 4096] = adaptation_table(6);
static ADAPT7: [[u16; 2]; 4096] = adaptation_table(7);
#[inline(always)]
fn transition<const RATE: u8>(p: u16) -> &'static [u16; 2] {
    let table = match RATE {
        4 => &ADAPT4,
        6 => &ADAPT6,
        7 => &ADAPT7,
        _ => unreachable!(),
    };
    &table[p as usize & 4095]
}
#[inline(always)]
fn adapt<const RATE: u8>(p: &mut u16, bit: bool) {
    *p = transition::<RATE>(*p)[bit as usize];
}
struct RangeOut {
    low: u64,
    range: u32,
    cache: u8,
    pending: usize,
    data: Vec<u8>,
}
impl RangeOut {
    fn new() -> Self {
        Self {
            low: 0,
            range: u32::MAX,
            cache: 0,
            pending: 1,
            data: Vec::new(),
        }
    }
    fn shift(&mut self) {
        let lo = self.low as u32;
        let carry = (self.low >> 32) as u8;
        if lo < 0xff00_0000 || carry != 0 {
            self.data.push(self.cache.wrapping_add(carry));
            for _ in 1..self.pending {
                self.data.push(255u8.wrapping_add(carry));
            }
            self.cache = (lo >> 24) as u8;
            self.pending = 0;
        }
        self.pending += 1;
        self.low = ((lo & 0x00ff_ffff) as u64) << 8;
    }
    fn put(&mut self, p: u16, bit: bool) {
        let bound = (self.range >> 12) * p as u32;
        if !bit {
            self.range = bound;
        } else {
            self.low += bound as u64;
            self.range -= bound;
        }
        while self.range < 1 << 24 {
            self.range <<= 8;
            self.shift();
        }
    }
    fn finish(mut self) -> Vec<u8> {
        for _ in 0..5 {
            self.shift();
        }
        self.data
    }
}
impl Binary for RangeOut {
    const READ: bool = false;
    #[inline(always)]
    fn bit<const RATE: u8>(&mut self, p: &mut u16, value: bool) -> bool {
        self.put(*p, value);
        adapt::<RATE>(p, value);
        value
    }
}
struct RangeIn<'a> {
    data: &'a [u8],
    pos: usize,
    range: u32,
    code: u32,
}
impl<'a> RangeIn<'a> {
    fn new(data: &'a [u8]) -> Result<Self, String> {
        if data.len() < 5 || data[0] != 0 {
            return bad("invalid arithmetic body");
        }
        let code = u32::from_be_bytes([data[1], data[2], data[3], data[4]]);
        Ok(Self {
            data,
            pos: 5,
            range: u32::MAX,
            code,
        })
    }
    fn finish(self) -> Result<(), String> {
        if self.pos != self.data.len() || self.code != 0 {
            return bad("noncanonical arithmetic body");
        }
        Ok(())
    }
}
impl Binary for RangeIn<'_> {
    const READ: bool = true;
    #[inline(always)]
    fn bit<const RATE: u8>(&mut self, p: &mut u16, _value: bool) -> bool {
        let next = transition::<RATE>(*p);
        let bound = (self.range >> 12) * *p as u32;
        let bit = self.code >= bound;
        self.range = if bit { self.range - bound } else { bound };
        self.code = self.code.wrapping_sub(if bit { bound } else { 0 });
        while self.range < 1 << 24 {
            let b = self.data.get(self.pos).copied().unwrap_or(0);
            self.pos += 1;
            self.range <<= 8;
            self.code = (self.code << 8) | b as u32;
        }
        *p = next[bit as usize];
        bit
    }
    #[inline(always)]
    fn check(&self) -> Result<(), String> {
        if self.pos > self.data.len() {
            return bad("truncated arithmetic body");
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct MagnitudeModel {
    nonzero: u16,
    exponent: [u16; 9],
    sign: u16,
}
#[derive(Clone)]
struct AcModel {
    groups: [[MagnitudeModel; 7]; 11],
    mantissa: [[[u16; 9]; 10]; 4],
}
impl AcModel {
    fn new() -> Self {
        Self {
            groups: [[MagnitudeModel {
                nonzero: 2048,
                exponent: [2048; 9],
                sign: 2048,
            }; 7]; 11],
            mantissa: [[[2048; 9]; 10]; 4],
        }
    }
}
fn bucket(v: u32) -> usize {
    (32 - v.leading_zeros()) as usize
}
fn count_nz(b: &Block) -> usize {
    (1..8)
        .map(|y| (1..8).filter(|&x| b[y * 8 + x] != 0).count())
        .sum()
}
fn code_count<C: Binary>(c: &mut C, p: &mut [u16; 64], value: usize) -> usize {
    let mut lo = 0;
    let mut hi = 49;
    let mut node = 1;
    while lo < hi {
        let mid = (lo + hi) / 2;
        let bit = c.bit::<4>(&mut p[node & 63], value > mid);
        if bit {
            lo = mid + 1;
        } else {
            hi = mid;
        }
        node = node * 2 + bit as usize;
    }
    lo
}
#[inline(always)]
fn magnitude<C: Binary>(
    c: &mut C,
    p: &mut AcModel,
    value: i16,
    predicted_magnitude: u32,
    predicted_negative: bool,
    rb: usize,
    forced: bool,
) -> i16 {
    let pb = min(10, bucket(predicted_magnitude));
    let group = &mut p.groups[pb][rb.min(6)];
    if !forced && !c.bit::<4>(&mut group.nonzero, value != 0) {
        return 0;
    }
    let input_n = category(value as i32) as usize;
    let mut n = 10;
    for (i, probability) in group.exponent.iter_mut().enumerate() {
        if !c.bit::<4>(probability, input_n > i + 1) {
            n = i + 1;
            break;
        }
    }
    let mut v = 1u32;
    let mantissa = &mut p.mantissa[pb / 3][n - 1];
    for (bit, probability) in mantissa[..n - 1].iter_mut().enumerate().rev() {
        let b = c.bit::<6>(probability, ((value.unsigned_abs() as u32 >> bit) & 1) != 0);
        v = (v << 1) | b as u32;
    }
    let neg = c.bit::<7>(&mut group.sign, (value < 0) ^ predicted_negative) ^ predicted_negative;
    if neg {
        -(v as i16)
    } else {
        v as i16
    }
}
#[inline(always)]
fn dc_value<C: Binary>(
    c: &mut C,
    p: &mut [u16; 64],
    value: i16,
    predicted: i32,
) -> Result<i16, String> {
    let residual = value as i32 - predicted;
    if !c.bit::<4>(&mut p[0], residual != 0) {
        return i16::try_from(predicted).map_err(|_| "invalid DC prediction".to_string());
    }
    let input_n = category(residual) as usize;
    let mut n = 16;
    for (i, probability) in p[1..16].iter_mut().enumerate() {
        if !c.bit::<4>(probability, input_n > i + 1) {
            n = i + 1;
            break;
        }
    }
    let mut v = 1i32;
    for (bit, probability) in p[16..16 + n - 1].iter_mut().enumerate().rev() {
        let b = c.bit::<4>(probability, ((residual.unsigned_abs() >> bit) & 1) != 0);
        v = (v << 1) | b as i32;
    }
    if c.bit::<7>(&mut p[32 + n - 1], residual < 0) {
        v = -v;
    }
    i16::try_from(predicted + v).map_err(|_| "DC out of range".to_string())
}
fn coefficients<C: Binary>(c: &mut C, planes: &mut [Plane]) -> Result<(), String> {
    let zero = [0i16; 64];
    let mut count_models = [[2048u16; 64]; 24];
    let mut ac_models: Box<[AcModel; 64]> = zeros(64, AcModel::new())?
        .into_boxed_slice()
        .try_into()
        .map_err(|_| "invalid model size")?;
    let mut dc_models = [[[2048u16; 64]; 7]; 8];
    for (plane_index, plane) in planes.iter_mut().enumerate() {
        if plane_index != 0 {
            count_models.fill([2048; 64]);
            ac_models.fill(AcModel::new());
            dc_models.fill([[2048; 64]; 7]);
        }
        let mut counts = zeros(plane.blocks.len(), 0u8)?;
        if C::READ {
            plane.nonzero = zeros(plane.blocks.len(), 0u64)?;
        }
        for y in 0..plane.rows {
            for x in 0..plane.cols {
                let index = y * plane.cols + x;
                let left = if x != 0 {
                    &plane.blocks[index - 1]
                } else {
                    &zero
                };
                let above = if y != 0 {
                    &plane.blocks[index - plane.cols]
                } else {
                    &zero
                };
                let diagonal = if x != 0 && y != 0 {
                    &plane.blocks[index - plane.cols - 1]
                } else {
                    &zero
                };
                let mut b = if C::READ { zero } else { plane.blocks[index] };
                let ln = if x != 0 {
                    counts[index - 1] as usize
                } else {
                    0
                };
                let an = if y != 0 {
                    counts[index - plane.cols] as usize
                } else {
                    0
                };
                let predicted_count = (ln + an + 1) / 2;
                let count_context = if predicted_count < 8 {
                    predicted_count
                } else {
                    8 + (predicted_count - 8) / 3
                };
                let actual_count = if C::READ { 0 } else { count_nz(&b) };
                let nz = code_count(c, &mut count_models[count_context.min(23)], actual_count);
                counts[index] = nz as u8;
                let nz_bucket = min(6, bucket(nz as u32));
                let mut nonzero = 0u64;
                let mut remaining = nz;
                let mut positions = 49;
                for &position in &INNER {
                    let pos = position & 63;
                    if remaining == 0 {
                        break;
                    }
                    let a = above[pos] as i32;
                    let l = left[pos] as i32;
                    let d = diagonal[pos] as i32;
                    let mag =
                        (a.unsigned_abs() * 13 + l.unsigned_abs() * 13 + d.unsigned_abs() * 6) >> 5;
                    let negative = mag != 0 && a * 13 + l * 13 + d * 6 < 0;
                    b[pos] = magnitude(
                        c,
                        &mut ac_models[pos],
                        b[pos],
                        mag,
                        negative,
                        min(6, bucket(remaining as u32)),
                        remaining == positions,
                    );
                    if b[pos] != 0 {
                        remaining -= 1;
                        if C::READ {
                            nonzero |= ZIGZAG_BITS[pos];
                        }
                    }
                    positions -= 1;
                }
                if remaining != 0 {
                    return bad("invalid nonzero count");
                }
                for &position in &EDGES {
                    let pos = position & 63;
                    let prediction = if pos / 8 == 0 { above[pos] } else { left[pos] } as i32;
                    b[pos] = magnitude(
                        c,
                        &mut ac_models[pos],
                        b[pos],
                        prediction.unsigned_abs(),
                        prediction < 0,
                        nz_bucket,
                        false,
                    );
                    if C::READ {
                        nonzero |= ZIGZAG_BITS[pos] & 0u64.wrapping_sub((b[pos] != 0) as u64);
                    }
                }
                let l = left[0] as i32;
                let a = above[0] as i32;
                let d = diagonal[0] as i32;
                let prediction = if x == 0 {
                    a
                } else if y == 0 {
                    l
                } else {
                    let grad = l + a - d;
                    max(min(l, a), min(max(l, a), grad))
                };
                let dc_context = min(7, bucket((l - a).unsigned_abs()));
                b[0] = dc_value(c, &mut dc_models[dc_context][nz_bucket], b[0], prediction)?;
                c.check()?;
                if C::READ {
                    plane.nonzero[index] = nonzero;
                }
                plane.blocks[index] = b;
            }
        }
    }
    Ok(())
}

pub fn vxj_encode(jpeg: &[u8]) -> Result<Vec<u8>, String> {
    let (h, scan, start) = first_header(jpeg)?;
    let (mut planes, end) = read_scan(jpeg, start, &h, &scan)?;
    if write_scan(&h, &scan, &planes)? != jpeg[start..end] {
        return bad("not reproducible");
    }
    validate_tail(jpeg, end, h, &scan)?;
    let header_len = u32::try_from(start).map_err(|_| "header limit")?;
    let trailer_len = u32::try_from(jpeg.len() - end).map_err(|_| "trailer limit")?;
    let mut coder = RangeOut::new();
    coefficients(&mut coder, &mut planes)?;
    let body = coder.finish();
    let mut rec = Vec::new();
    rec.try_reserve_exact(12 + start + jpeg.len() - end + body.len())
        .map_err(|_| "allocation limit")?;
    rec.extend_from_slice(b"VXJ1");
    rec.extend_from_slice(&header_len.to_le_bytes());
    rec.extend_from_slice(&trailer_len.to_le_bytes());
    rec.extend_from_slice(&jpeg[..start]);
    rec.extend_from_slice(&jpeg[end..]);
    rec.extend_from_slice(&body);
    Ok(rec)
}
pub fn vxj_decode(rec: &[u8]) -> Result<Vec<u8>, String> {
    if rec.len() < 12 || &rec[..4] != b"VXJ1" {
        return bad("invalid VXJ record");
    }
    let hl = u32::from_le_bytes([rec[4], rec[5], rec[6], rec[7]]) as usize;
    let tl = u32::from_le_bytes([rec[8], rec[9], rec[10], rec[11]]) as usize;
    let hp = 12usize.checked_add(hl).ok_or("record limit")?;
    let tp = hp.checked_add(tl).ok_or("record limit")?;
    if tp > rec.len() || rec.len() - tp < 5 {
        return bad("truncated VXJ record");
    }
    let header = &rec[12..hp];
    let trailer = &rec[hp..tp];
    let (h, scan, end) = first_header(header)?;
    if end != header.len() {
        return bad("invalid VXJ header");
    }
    validate_tail(trailer, 0, h.clone(), &scan)?;
    let mut coder = RangeIn::new(&rec[tp..])?;
    // Every block needs at least 20 decisions with probability at most 4095/4096.
    let (mut planes, _, _) = layout(&h, &scan, (rec.len() - tp).saturating_mul(2048))?;
    coefficients(&mut coder, &mut planes)?;
    coder.finish()?;
    let entropy = write_scan(&h, &scan, &planes)?;
    let mut jpeg = Vec::new();
    jpeg.try_reserve_exact(header.len() + entropy.len() + trailer.len())
        .map_err(|_| "allocation limit")?;
    jpeg.extend_from_slice(header);
    jpeg.extend_from_slice(&entropy);
    jpeg.extend_from_slice(trailer);
    Ok(jpeg)
}
#[cfg(test)]
mod coverage {
    use super::*;

    fn segment(out: &mut Vec<u8>, tag: u8, body: &[u8]) {
        out.extend_from_slice(&[255, tag]);
        out.extend_from_slice(&((body.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(body);
    }
    fn synthetic(sampling: &[(u8, u8)], restart: u16, split: bool) -> Vec<u8> {
        let mut jpeg = vec![255, 216];
        let mut q = vec![1; 65];
        q[0] = 0;
        segment(&mut jpeg, 0xdb, &q);
        let mut frame = vec![8, 0, 45, 0, 53, sampling.len() as u8];
        for (i, &(h, v)) in sampling.iter().enumerate() {
            frame.extend_from_slice(&[i as u8 + 1, h * 16 + v, 0]);
        }
        segment(&mut jpeg, 0xc1, &frame);
        let mut tables = vec![0; 17];
        tables[4] = 12;
        tables.extend(0..12);
        tables.push(16);
        let begin = tables.len();
        tables.resize(begin + 16, 0);
        tables[begin + 7] = 162;
        tables.extend_from_slice(&[0, 240]);
        for run in 0..16 {
            for n in 1..11 {
                tables.push(run * 16 + n);
            }
        }
        segment(&mut jpeg, 0xc4, &tables);
        segment(&mut jpeg, 0xdd, &restart.to_be_bytes());
        segment(&mut jpeg, 0xe1, b"verbatim metadata\x00\xff");
        let mut h = Header::new();
        let mut at = 2;
        while at < jpeg.len() {
            let (tag, body) = marker(&jpeg, &mut at).unwrap();
            h.segment(tag, body).unwrap();
        }
        let scans = if split { sampling.len() } else { 1 };
        let mut random = 0x1729u32;
        for scan_index in 0..scans {
            let ids: Vec<_> = if split {
                vec![scan_index]
            } else {
                (0..sampling.len()).collect()
            };
            let mut sos = vec![ids.len() as u8];
            for &id in &ids {
                sos.extend_from_slice(&[id as u8 + 1, 0]);
            }
            sos.extend_from_slice(&[0, 63, 0]);
            segment(&mut jpeg, 0xda, &sos);
            let s = h.scan(&sos).unwrap();
            let (mut planes, _, _) = layout(&h, &s, usize::MAX).unwrap();
            for (ci, plane) in planes.iter_mut().enumerate() {
                for (index, block) in plane.blocks.iter_mut().enumerate() {
                    block[0] = ((index * 7 + ci * 31) % 700) as i16 - 350;
                    for value in &mut block[1..] {
                        random ^= random << 13;
                        random ^= random >> 17;
                        random ^= random << 5;
                        if random & 7 == 0 {
                            *value = ((random >> 8) % 1023) as i16 - 511;
                        }
                    }
                }
            }
            jpeg.extend(write_scan(&h, &s, &planes).unwrap());
        }
        jpeg.extend_from_slice(&[255, 217, 0, 255, 216, 127]);
        jpeg
    }
    #[test]
    fn sampling_restart_and_sequential_scans() {
        for samples in [
            &[(1, 1)][..],
            &[(2, 2), (1, 1), (1, 1)][..],
            &[(3, 1), (1, 2), (1, 1), (1, 1)][..],
            &[(4, 1), (1, 1)][..],
        ] {
            for restart in [0, 1, 3, 17] {
                for split in [false, true] {
                    let jpeg = synthetic(samples, restart, split);
                    let rec = vxj_encode(&jpeg).unwrap();
                    assert_eq!(vxj_decode(&rec).unwrap(), jpeg);
                }
            }
        }
    }
    #[test]
    fn adaptation_matches_original() {
        fn check<const RATE: u8>() {
            for old in 1u16..4096 {
                for bit in [false, true] {
                    let expected = if bit {
                        max(1, old - max(1, old >> RATE))
                    } else {
                        min(4095, old + max(1, (4096 - old) >> RATE))
                    };
                    let mut actual = old;
                    adapt::<RATE>(&mut actual, bit);
                    assert_eq!(actual, expected);
                }
            }
        }
        check::<4>();
        check::<6>();
        check::<7>();
    }
    #[test]
    fn arithmetic_body_is_stable() {
        let mut planes = vec![Plane {
            cols: 2,
            rows: 2,
            blocks: vec![[0; 64]; 4],
            nonzero: Vec::new(),
        }];
        for (i, b) in planes[0].blocks.iter_mut().enumerate() {
            b[0] = [-512, -1, 511, 24][i];
            b[1] = [0, 1, -7, 1023][i];
            b[8] = [-1023, 0, 7, -1][i];
            b[9] = [1, -3, 0, 0][i];
            b[63] = [0, 0, -1, 5][i];
        }
        let mut coder = RangeOut::new();
        coefficients(&mut coder, &mut planes).unwrap();
        let expected = [
            0, 6, 63, 247, 252, 0, 63, 240, 2, 14, 227, 90, 62, 152, 103, 30, 225, 208, 0, 0, 0,
            37, 160, 118, 46, 99, 228, 153, 122, 245, 0, 0, 0, 181, 85, 219, 152, 95, 20, 249, 179,
            85, 248,
        ];
        let original = planes[0].blocks.clone();
        assert_eq!(coder.finish(), expected);
        planes[0].blocks.fill([0; 64]);
        let mut input = RangeIn::new(&expected).unwrap();
        coefficients(&mut input, &mut planes).unwrap();
        input.finish().unwrap();
        assert_eq!(planes[0].blocks, original);
        for at in 1..expected.len() {
            for bit in 0..8 {
                let mut damaged = expected;
                damaged[at] ^= 1 << bit;
                planes[0].blocks.fill([0; 64]);
                let mut input = RangeIn::new(&damaged).unwrap();
                let _ = coefficients(&mut input, &mut planes);
                let _ = input.finish();
            }
        }
        for cut in 5..expected.len() {
            planes[0].blocks.fill([0; 64]);
            let mut input = RangeIn::new(&expected[..cut]).unwrap();
            assert!(coefficients(&mut input, &mut planes).is_err() || input.finish().is_err());
        }
    }
    #[test]
    fn coefficient_coder_extremes() {
        let mut planes = vec![Plane {
            cols: 40,
            rows: 30,
            blocks: vec![[0; 64]; 1200],
            nonzero: Vec::new(),
        }];
        for (i, block) in planes[0].blocks.iter_mut().enumerate() {
            if i % 137 == 0 {
                block[0] = if i % 2 == 0 { i16::MIN } else { i16::MAX };
                for (pos, v) in block.iter_mut().enumerate().skip(1) {
                    *v = if pos % 2 == 0 { -1023 } else { 1023 };
                }
            }
        }
        let expected = planes[0].blocks.clone();
        let mut out = RangeOut::new();
        coefficients(&mut out, &mut planes).unwrap();
        let data = out.finish();
        let mut input = RangeIn::new(&data).unwrap();
        planes[0].blocks.fill([0; 64]);
        coefficients(&mut input, &mut planes).unwrap();
        input.finish().unwrap();
        assert_eq!(planes[0].blocks, expected);
        for cut in [0, 1, 4, data.len() - 1] {
            assert!(vxj_decode(&data[..cut]).is_err());
        }
    }
    #[test]
    fn buffered_huffman_output_matches_byte_writer() {
        let mut output = BitOut::new();
        let mut expected = Vec::new();
        let mut bits = 0u64;
        let mut pending = 0u8;
        let mut random = 0x1729u32;
        for i in 0..8192 {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            let n = (i % 27 + 1) as u8;
            let value = if i & 1 == 0 { random } else { u32::MAX } & ((1 << n) - 1);
            output.put(value, n);
            bits = (bits << n) | value as u64;
            pending += n;
            while pending >= 8 {
                pending -= 8;
                let byte = (bits >> pending) as u8;
                expected.push(byte);
                if byte == 255 {
                    expected.push(0);
                }
            }
            if i % 31 == 0 {
                output.align();
                if pending != 0 {
                    let padding = 8 - pending;
                    let byte = ((bits << padding) | ((1 << padding) - 1)) as u8;
                    expected.push(byte);
                    if byte == 255 {
                        expected.push(0);
                    }
                    pending = 0;
                }
                assert_eq!(output.data, expected);
            }
        }
        output.align();
        if pending != 0 {
            let padding = 8 - pending;
            let byte = ((bits << padding) | ((1 << padding) - 1)) as u8;
            expected.push(byte);
            if byte == 255 {
                expected.push(0);
            }
        }
        assert_eq!(output.data, expected);
    }
    #[test]
    fn range_decoder_extreme_probabilities() {
        let mut random = 0x1729u32;
        let mut decisions = Vec::new();
        let mut output = RangeOut::new();
        for i in 0..32768 {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            let probability = match i & 3 {
                0 => 1,
                1 => 4095,
                2 => 2048,
                _ => (random % 4095 + 1) as u16,
            };
            let bit = random & 0x8000_0000 != 0;
            decisions.push((probability, bit));
            output.put(probability, bit);
        }
        let data = output.finish();
        let mut input = RangeIn::new(&data).unwrap();
        for (i, (mut probability, bit)) in decisions.into_iter().enumerate() {
            let decoded = match i % 3 {
                0 => input.bit::<4>(&mut probability, false),
                1 => input.bit::<6>(&mut probability, false),
                _ => input.bit::<7>(&mut probability, false),
            };
            assert_eq!(decoded, bit);
            input.check().unwrap();
        }
        input.finish().unwrap();
    }
    #[test]
    fn rejects_noncanonical_scan_and_oversized_frame() {
        let mut jpeg = synthetic(&[(1, 1)], 0, false);
        let (h, _, start) = first_header(&jpeg).unwrap();
        let pos = jpeg.windows(2).position(|b| b == [255, 193]).unwrap();
        let mut header = jpeg[..start].to_vec();
        header[pos + 5..pos + 9].copy_from_slice(&[0, 8, 0, 8]);
        let dc = h.dc[0].as_ref().unwrap();
        let ac = h.ac[0].as_ref().unwrap();
        let mut bits = BitOut::new();
        dc.write(&mut bits, 0).unwrap();
        ac.write(&mut bits, 0).unwrap();
        bits.align();
        let mut canonical = header.clone();
        canonical.extend(bits.data);
        canonical.extend_from_slice(&[255, 217]);
        let record = vxj_encode(&canonical).unwrap();
        assert_eq!(vxj_decode(&record).unwrap(), canonical);
        for cut in 0..record.len() {
            assert!(vxj_decode(&record[..cut]).is_err());
        }
        let mut bad_padding = canonical;
        let last = bad_padding.len() - 3;
        bad_padding[last] &= 254;
        assert_eq!(vxj_encode(&bad_padding).unwrap_err(), "not reproducible");
        let mut bits = BitOut::new();
        dc.write(&mut bits, 0).unwrap();
        ac.write(&mut bits, 240).unwrap();
        ac.write(&mut bits, 0).unwrap();
        bits.align();
        let mut redundant = header;
        redundant.extend(bits.data);
        redundant.extend_from_slice(&[255, 217]);
        assert_eq!(vxj_encode(&redundant).unwrap_err(), "not reproducible");
        jpeg[pos + 5..pos + 9].fill(255);
        assert!(vxj_encode(&jpeg).is_err());
        let mut oversized = record;
        oversized[12 + pos + 5..12 + pos + 9].copy_from_slice(&[3, 232, 234, 96]);
        assert_eq!(
            vxj_decode(&oversized).unwrap_err(),
            "truncated coefficient data"
        );
    }
}
#[rustfmt::skip]
#[cfg(test)] #[path = "vxj_tests.rs"] mod tests;
