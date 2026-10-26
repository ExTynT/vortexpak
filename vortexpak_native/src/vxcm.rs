const TOP: u32 = 1 << 24;
const TOTAL: u32 = 1 << 11;
const MAX_LEN: usize = 273;
const INF: u32 = u32::MAX / 4;
const NONE: u32 = u32::MAX;

#[derive(Clone, Copy)]
struct Props {
    lc: usize,
    lp: usize,
    pb: usize,
}

impl Props {
    fn byte(self) -> u8 {
        (self.lc + 9 * (self.lp + 5 * self.pb)) as u8
    }

    fn context(self, pos: usize, prev: u8) -> usize {
        ((pos & ((1 << self.lp) - 1)) << self.lc) | ((prev as usize) >> (8 - self.lc))
    }
}

struct RangeEncoder {
    low: u64,
    range: u32,
    cache: u8,
    cache_size: usize,
    out: Vec<u8>,
}

impl RangeEncoder {
    fn new(header: &[u8]) -> Self {
        Self {
            low: 0,
            range: u32::MAX,
            cache: 0,
            cache_size: 1,
            out: header.to_vec(),
        }
    }

    fn shift_low(&mut self) {
        let low = self.low as u32;
        let carry = (self.low >> 32) as u8;
        if low < 0xff00_0000 || carry != 0 {
            let mut byte = self.cache;
            loop {
                self.out.push(byte.wrapping_add(carry));
                byte = 0xff;
                self.cache_size -= 1;
                if self.cache_size == 0 {
                    break;
                }
            }
            self.cache = (low >> 24) as u8;
        }
        self.cache_size += 1;
        self.low = ((low & 0x00ff_ffff) as u64) << 8;
    }

    #[inline]
    fn bit(&mut self, probability: &mut u16, bit: u32) {
        let p = *probability as u32;
        let bound = (self.range >> 11) * p;
        if bit == 0 {
            self.range = bound;
            *probability = (p + ((TOTAL - p) >> 5)) as u16;
        } else {
            self.low += bound as u64;
            self.range -= bound;
            *probability = (p - (p >> 5)) as u16;
        }
        if self.range < TOP {
            self.range <<= 8;
            self.shift_low();
        }
    }

    fn tree(&mut self, probs: &mut [u16], bits: usize, value: u32) {
        let mut node = 1;
        for shift in (0..bits).rev() {
            let bit = (value >> shift) & 1;
            self.bit(&mut probs[node], bit);
            node = (node << 1) | bit as usize;
        }
    }

    fn reverse(&mut self, probs: &mut [u16], offset: isize, bits: usize, value: u32) {
        let mut node = 1;
        for shift in 0..bits {
            let bit = (value >> shift) & 1;
            self.bit(&mut probs[(offset + node as isize) as usize], bit);
            node = (node << 1) | bit as usize;
        }
    }

    fn direct(&mut self, value: u32, bits: usize) {
        for shift in (0..bits).rev() {
            self.range >>= 1;
            if (value >> shift) & 1 != 0 {
                self.low += self.range as u64;
            }
            if self.range < TOP {
                self.range <<= 8;
                self.shift_low();
            }
        }
    }

    fn finish(mut self) -> Vec<u8> {
        for _ in 0..5 {
            self.shift_low();
        }
        self.out
    }
}

struct RangeDecoder<'a> {
    input: &'a [u8],
    pos: usize,
    code: u32,
    range: u32,
}

impl<'a> RangeDecoder<'a> {
    fn new(input: &'a [u8]) -> Result<Self, String> {
        if input.len() < 5 {
            return Err("truncated range stream".into());
        }
        if input[0] != 0 {
            return Err("invalid range prefix".into());
        }
        let code = u32::from_be_bytes([input[1], input[2], input[3], input[4]]);
        if code == u32::MAX {
            return Err("invalid range code".into());
        }
        Ok(Self {
            input,
            pos: 5,
            code,
            range: u32::MAX,
        })
    }

    #[inline]
    fn normalize(&mut self) -> Result<(), String> {
        if self.range < TOP {
            let byte = *self.input.get(self.pos).ok_or("truncated range stream")?;
            self.pos += 1;
            self.range <<= 8;
            self.code = (self.code << 8) | byte as u32;
        }
        Ok(())
    }

    #[inline]
    fn bit(&mut self, probability: &mut u16) -> Result<u32, String> {
        let p = *probability as u32;
        let bound = (self.range >> 11) * p;
        let bit;
        if self.code < bound {
            self.range = bound;
            *probability = (p + ((TOTAL - p) >> 5)) as u16;
            bit = 0;
        } else {
            self.range -= bound;
            self.code -= bound;
            *probability = (p - (p >> 5)) as u16;
            bit = 1;
        }
        self.normalize()?;
        Ok(bit)
    }

    fn tree(&mut self, probs: &mut [u16], bits: usize) -> Result<u32, String> {
        let mut node = 1;
        for _ in 0..bits {
            node = (node << 1) | self.bit(&mut probs[node])? as usize;
        }
        Ok((node - (1 << bits)) as u32)
    }

    fn reverse(&mut self, probs: &mut [u16], offset: isize, bits: usize) -> Result<u32, String> {
        let mut node = 1;
        let mut value = 0;
        for shift in 0..bits {
            let bit = self.bit(&mut probs[(offset + node as isize) as usize])?;
            node = (node << 1) | bit as usize;
            value |= bit << shift;
        }
        Ok(value)
    }

    fn direct(&mut self, bits: usize) -> Result<u32, String> {
        let mut value = 0;
        for _ in 0..bits {
            self.range >>= 1;
            let bit = u32::from(self.code >= self.range);
            if bit != 0 {
                self.code -= self.range;
            }
            value = (value << 1) | bit;
            self.normalize()?;
        }
        Ok(value)
    }
}

struct LengthModel {
    choice: [u16; 2],
    low: [[u16; 8]; 16],
    mid: [[u16; 8]; 16],
    high: [u16; 256],
}

impl LengthModel {
    fn new() -> Self {
        Self {
            choice: [1024; 2],
            low: [[1024; 8]; 16],
            mid: [[1024; 8]; 16],
            high: [1024; 256],
        }
    }

    fn encode(&mut self, rc: &mut RangeEncoder, pos: usize, len: usize) {
        let value = len - 2;
        if value < 8 {
            rc.bit(&mut self.choice[0], 0);
            rc.tree(&mut self.low[pos], 3, value as u32);
        } else {
            rc.bit(&mut self.choice[0], 1);
            if value < 16 {
                rc.bit(&mut self.choice[1], 0);
                rc.tree(&mut self.mid[pos], 3, (value - 8) as u32);
            } else {
                rc.bit(&mut self.choice[1], 1);
                rc.tree(&mut self.high, 8, (value - 16) as u32);
            }
        }
    }

    fn decode(&mut self, rc: &mut RangeDecoder<'_>, pos: usize) -> Result<usize, String> {
        if rc.bit(&mut self.choice[0])? == 0 {
            Ok(rc.tree(&mut self.low[pos], 3)? as usize + 2)
        } else if rc.bit(&mut self.choice[1])? == 0 {
            Ok(rc.tree(&mut self.mid[pos], 3)? as usize + 10)
        } else {
            Ok(rc.tree(&mut self.high, 8)? as usize + 18)
        }
    }

    fn fill_prices(&self, prices: &[u16; 2048], pos_states: usize, out: &mut [[u32; 274]; 16]) {
        let low = bit_price(prices, self.choice[0], 0);
        let upper = bit_price(prices, self.choice[0], 1);
        let mid = upper + bit_price(prices, self.choice[1], 0);
        let mut high = [0u32; 512];
        high[1] = upper + bit_price(prices, self.choice[1], 1);
        for node in 1..256 {
            high[node * 2] = high[node] + bit_price(prices, self.high[node], 0);
            high[node * 2 + 1] = high[node] + bit_price(prices, self.high[node], 1);
        }
        for (pos, row) in out[..pos_states].iter_mut().enumerate() {
            row[18..].copy_from_slice(&high[256..]);
            for value in 0..8 {
                row[value + 2] = low + tree_price(prices, &self.low[pos], 3, value as u32);
                row[value + 10] = mid + tree_price(prices, &self.mid[pos], 3, value as u32);
            }
        }
    }

    #[cfg(test)]
    fn price(&self, prices: &[u16; 2048], pos: usize, len: usize) -> u32 {
        let value = len - 2;
        if value < 8 {
            bit_price(prices, self.choice[0], 0)
                + tree_price(prices, &self.low[pos], 3, value as u32)
        } else if value < 16 {
            bit_price(prices, self.choice[0], 1)
                + bit_price(prices, self.choice[1], 0)
                + tree_price(prices, &self.mid[pos], 3, (value - 8) as u32)
        } else {
            bit_price(prices, self.choice[0], 1)
                + bit_price(prices, self.choice[1], 1)
                + tree_price(prices, &self.high, 8, (value - 16) as u32)
        }
    }
}

struct Models<L> {
    literal: L,
    is_match: [[u16; 16]; 12],
    is_rep: [u16; 12],
    rep_g0: [u16; 12],
    rep_g1: [u16; 12],
    rep_g2: [u16; 12],
    rep0_long: [[u16; 16]; 12],
    length: LengthModel,
    rep_length: LengthModel,
    slots: [[u16; 64]; 4],
    special: [u16; 114],
    align: [u16; 16],
}

impl<L: AsRef<[u16]> + AsMut<[u16]>> Models<L> {
    fn new(literal: L) -> Self {
        Self {
            literal,
            is_match: [[1024; 16]; 12],
            is_rep: [1024; 12],
            rep_g0: [1024; 12],
            rep_g1: [1024; 12],
            rep_g2: [1024; 12],
            rep0_long: [[1024; 16]; 12],
            length: LengthModel::new(),
            rep_length: LengthModel::new(),
            slots: [[1024; 64]; 4],
            special: [1024; 114],
            align: [1024; 16],
        }
    }

    fn literal_encode(
        &mut self,
        rc: &mut RangeEncoder,
        context: usize,
        byte: u8,
        matched: Option<u8>,
    ) {
        let probs = &mut self.literal.as_mut()[context * 0x300..(context + 1) * 0x300];
        let mut node = 1;
        let mut matching = matched.is_some();
        let match_byte = matched.unwrap_or(0);
        for shift in (0..8).rev() {
            let bit = ((byte >> shift) & 1) as u32;
            let match_bit = ((match_byte >> shift) & 1) as usize;
            let index = if matching {
                ((1 + match_bit) << 8) + node
            } else {
                node
            };
            rc.bit(&mut probs[index], bit);
            node = (node << 1) | bit as usize;
            matching &= bit as usize == match_bit;
        }
    }

    fn literal_decode(
        &mut self,
        rc: &mut RangeDecoder<'_>,
        context: usize,
        matched: Option<u8>,
    ) -> Result<u8, String> {
        let probs = &mut self.literal.as_mut()[context * 0x300..(context + 1) * 0x300];
        let mut node = 1;
        let mut matching = matched.is_some();
        let match_byte = matched.unwrap_or(0);
        for shift in (0..8).rev() {
            let match_bit = ((match_byte >> shift) & 1) as usize;
            let index = if matching {
                ((1 + match_bit) << 8) + node
            } else {
                node
            };
            let bit = rc.bit(&mut probs[index])? as usize;
            node = (node << 1) | bit;
            matching &= bit == match_bit;
        }
        Ok(node as u8)
    }

    fn literal_price(
        &self,
        prices: &[u16; 2048],
        context: usize,
        byte: u8,
        matched: Option<u8>,
    ) -> u32 {
        let probs = &self.literal.as_ref()[context * 0x300..(context + 1) * 0x300];
        let mut node = 1;
        let mut matching = matched.is_some();
        let match_byte = matched.unwrap_or(0);
        let mut price = 0;
        for shift in (0..8).rev() {
            let bit = ((byte >> shift) & 1) as u32;
            let match_bit = ((match_byte >> shift) & 1) as usize;
            let index = if matching {
                ((1 + match_bit) << 8) + node
            } else {
                node
            };
            price += bit_price(prices, probs[index], bit);
            node = (node << 1) | bit as usize;
            matching &= bit as usize == match_bit;
        }
        price
    }

    fn distance_encode(&mut self, rc: &mut RangeEncoder, len: usize, distance: usize) {
        let value = (distance - 1) as u32;
        let slot = distance_slot(value);
        rc.tree(&mut self.slots[(len - 2).min(3)], 6, slot);
        if slot >= 4 {
            let bits = (slot as usize >> 1) - 1;
            let base = (2 | (slot & 1)) << bits;
            let reduced = value - base;
            if slot < 14 {
                rc.reverse(
                    &mut self.special,
                    base as isize - slot as isize - 1,
                    bits,
                    reduced,
                );
            } else {
                rc.direct(reduced >> 4, bits - 4);
                rc.reverse(&mut self.align, 0, 4, reduced & 15);
            }
        }
    }

    fn distance_decode(&mut self, rc: &mut RangeDecoder<'_>, len: usize) -> Result<usize, String> {
        let slot = rc.tree(&mut self.slots[(len - 2).min(3)], 6)?;
        let value;
        if slot < 4 {
            value = slot;
        } else {
            let bits = (slot as usize >> 1) - 1;
            let base = (2 | (slot & 1)) << bits;
            let reduced = if slot < 14 {
                rc.reverse(&mut self.special, base as isize - slot as isize - 1, bits)?
            } else {
                (rc.direct(bits - 4)? << 4) | rc.reverse(&mut self.align, 0, 4)?
            };
            value = base | reduced;
        }
        if value == u32::MAX {
            return Err("invalid distance".into());
        }
        Ok(value as usize + 1)
    }

    #[cfg(test)]
    fn distance_price(&self, prices: &[u16; 2048], len: usize, distance: usize) -> u32 {
        let value = (distance - 1) as u32;
        let slot = distance_slot(value);
        let mut price = tree_price(prices, &self.slots[(len - 2).min(3)], 6, slot);
        if slot >= 4 {
            let bits = (slot as usize >> 1) - 1;
            let base = (2 | (slot & 1)) << bits;
            let reduced = value - base;
            if slot < 14 {
                price += reverse_price(
                    prices,
                    &self.special,
                    base as isize - slot as isize - 1,
                    bits,
                    reduced,
                );
            } else {
                price += ((bits - 4) * 64) as u32
                    + reverse_price(prices, &self.align, 0, 4, reduced & 15);
            }
        }
        price
    }
}

#[inline]
fn distance_slot(value: u32) -> u32 {
    if value < 4 {
        value
    } else {
        let high = 31 - value.leading_zeros();
        (high << 1) + ((value >> (high - 1)) & 1)
    }
}

#[inline]
fn literal_state(state: usize) -> usize {
    if state < 4 {
        0
    } else if state < 10 {
        state - 3
    } else {
        state - 6
    }
}

fn price_table() -> [u16; 2048] {
    let mut prices = [0; 2048];
    prices[0] = 2048;
    for (i, p) in prices.iter_mut().enumerate().skip(1) {
        *p = (-((i as f64) / 2048.0).log2() * 64.0).round() as u16;
    }
    prices
}

#[inline]
fn bit_price(prices: &[u16; 2048], probability: u16, bit: u32) -> u32 {
    prices[if bit == 0 {
        probability as usize
    } else {
        2048 - probability as usize
    }] as u32
}

fn tree_price(prices: &[u16; 2048], probs: &[u16], bits: usize, value: u32) -> u32 {
    let mut node = 1;
    let mut price = 0;
    for shift in (0..bits).rev() {
        let bit = (value >> shift) & 1;
        price += bit_price(prices, probs[node], bit);
        node = (node << 1) | bit as usize;
    }
    price
}

fn reverse_price(
    prices: &[u16; 2048],
    probs: &[u16],
    offset: isize,
    bits: usize,
    value: u32,
) -> u32 {
    let mut node = 1;
    let mut price = 0;
    for shift in 0..bits {
        let bit = (value >> shift) & 1;
        price += bit_price(prices, probs[(offset + node as isize) as usize], bit);
        node = (node << 1) | bit as usize;
    }
    price
}

struct DistancePrices {
    slots: [[u32; 64]; 4],
    short: [u32; 128],
    align: [u32; 16],
}

impl DistancePrices {
    fn new() -> Self {
        Self {
            slots: [[0; 64]; 4],
            short: [0; 128],
            align: [0; 16],
        }
    }

    fn update<L: AsRef<[u16]> + AsMut<[u16]>>(&mut self, model: &Models<L>, prices: &[u16; 2048]) {
        for (context, row) in self.slots.iter_mut().enumerate() {
            let mut nodes = [0u32; 128];
            for node in 1..64 {
                nodes[node * 2] = nodes[node] + bit_price(prices, model.slots[context][node], 0);
                nodes[node * 2 + 1] =
                    nodes[node] + bit_price(prices, model.slots[context][node], 1);
            }
            row.copy_from_slice(&nodes[64..]);
        }
        for value in 4..128 {
            let slot = distance_slot(value as u32);
            let bits = (slot as usize >> 1) - 1;
            let base = (2 | (slot & 1)) << bits;
            self.short[value] = reverse_price(
                prices,
                &model.special,
                base as isize - slot as isize - 1,
                bits,
                value as u32 - base,
            );
        }
        for value in 0..16 {
            self.align[value] = reverse_price(prices, &model.align, 0, 4, value as u32);
        }
    }

    #[inline]
    fn prices(&self, distance: usize) -> [u32; 4] {
        let value = distance - 1;
        let slot = distance_slot(value as u32) as usize;
        let footer = if value < 128 {
            self.short[value]
        } else {
            ((slot / 2 - 5) * 64) as u32 + self.align[value & 15]
        };
        std::array::from_fn(|context| self.slots[context][slot] + footer)
    }
}

#[derive(Clone, Copy)]
struct Match {
    len: usize,
    distance: usize,
}

struct MatchFinder<'a> {
    src: &'a [u8],
    head2: Vec<u32>,
    head3: Vec<u32>,
    head4: Vec<u32>,
    chain: Vec<u32>,
    mask: usize,
    hash_shift: u32,
    window: usize,
    depth: usize,
    nice: usize,
}

#[inline]
fn match_len(src: &[u8], pos: usize, distance: usize, limit: usize) -> usize {
    let other = pos - distance;
    let mut len = 0;
    while len + 8 <= limit {
        let a = u64::from_le_bytes(src[pos + len..pos + len + 8].try_into().unwrap());
        let b = u64::from_le_bytes(src[other + len..other + len + 8].try_into().unwrap());
        let diff = a ^ b;
        if diff != 0 {
            return len + diff.trailing_zeros() as usize / 8;
        }
        len += 8;
    }
    while len < limit && src[pos + len] == src[other + len] {
        len += 1;
    }
    len
}

impl<'a> MatchFinder<'a> {
    fn new(src: &'a [u8], level: u8, dict_log: u8) -> Self {
        let hash_bits = (usize::BITS - src.len().max(1).leading_zeros()).clamp(16, 20);
        let depths = [4, 8, 16, 32, 64, 128, 256, 384, 768];
        let nice = [16, 24, 32, 48, 64, 96, 128, 192, 273];
        let window = src.len().min(1 << 26);
        Self {
            src,
            head2: vec![NONE; 1 << 16],
            head3: vec![NONE; 1 << 16],
            head4: vec![NONE; 1 << hash_bits],
            chain: vec![NONE; window],
            mask: (1usize << dict_log) - 1,
            hash_shift: 32 - hash_bits,
            window,
            depth: depths[(level - 1) as usize],
            nice: nice[(level - 1) as usize],
        }
    }

    fn insert(&mut self, pos: usize) -> (u32, u32, u32) {
        let src = self.src;
        let mut previous = (NONE, NONE, NONE);
        if pos + 1 < src.len() {
            let h2 = src[pos] as usize | ((src[pos + 1] as usize) << 8);
            previous.0 = self.head2[h2];
            self.head2[h2] = pos as u32;
            if pos + 2 < src.len() {
                let h3 = ((h2 as u32 ^ ((src[pos + 2] as u32) << 16)).wrapping_mul(0x1e35_a7bd)
                    >> 16) as usize;
                previous.1 = self.head3[h3];
                self.head3[h3] = pos as u32;
                if pos + 3 < src.len() {
                    let word = u32::from_le_bytes(src[pos..pos + 4].try_into().unwrap());
                    let h4 = (word.wrapping_mul(0x9e37_79b1) >> self.hash_shift) as usize;
                    previous.2 = self.head4[h4];
                    self.head4[h4] = pos as u32;
                }
            }
        }
        self.chain[pos & self.mask] = previous.2;
        previous
    }

    fn matches(&mut self, pos: usize, out: &mut Vec<Match>) {
        out.clear();
        let previous = self.insert(pos);
        let limit = (self.src.len() - pos).min(MAX_LEN);
        if limit < 2 {
            return;
        }
        let mut best = 1;
        for candidate in [previous.0, previous.1] {
            if candidate == NONE {
                continue;
            }
            let distance = (pos as u32).wrapping_sub(candidate) as usize;
            if distance == 0 || distance > self.window || distance > pos {
                continue;
            }
            if self.src[pos + best] != self.src[pos + best - distance] {
                continue;
            }
            let len = match_len(self.src, pos, distance, limit);
            if len > best {
                out.push(Match { len, distance });
                best = len;
                if best == limit {
                    return;
                }
            }
        }
        let mut candidate = previous.2;
        for _ in 0..self.depth {
            if candidate == NONE {
                break;
            }
            let distance = (pos as u32).wrapping_sub(candidate) as usize;
            if distance == 0 || distance > self.window || distance > pos {
                break;
            }
            let other = pos - distance;
            let next = self.chain[other & self.mask];
            if self.src[pos + best] == self.src[other + best] && self.src[pos] == self.src[other] {
                let len = match_len(self.src, pos, distance, limit);
                if len > best {
                    out.push(Match { len, distance });
                    best = len;
                    if best >= self.nice || best == limit {
                        break;
                    }
                }
            }
            candidate = next;
        }
    }
}

#[derive(Clone, Copy)]
struct Token {
    len: usize,
    kind: usize,
}

#[derive(Clone, Copy)]
struct Optimum {
    price: u32,
    reps: [usize; 4],
    state: usize,
    prev: usize,
    kind: usize,
}

impl Optimum {
    const EMPTY: Self = Self {
        price: INF,
        reps: [1; 4],
        state: 0,
        prev: 0,
        kind: 0,
    };

    #[inline]
    fn offer(&mut self, price: u32, reps: [usize; 4], state: usize, prev: usize, kind: usize) {
        if price < self.price {
            *self = Self {
                price,
                reps,
                state,
                prev,
                kind,
            };
        }
    }
}

struct Encoder<'a> {
    src: &'a [u8],
    props: Props,
    model: Models<Vec<u16>>,
    rc: RangeEncoder,
    state: usize,
    reps: [usize; 4],
    prices: [u16; 2048],
}

impl<'a> Encoder<'a> {
    fn new(src: &'a [u8], props: Props, dict_log: u8) -> Self {
        let header = [b'V', b'X', b'C', b'1', props.byte(), dict_log];
        Self {
            src,
            props,
            model: Models::new(vec![1024; 0x300 << (props.lc + props.lp)]),
            rc: RangeEncoder::new(&header),
            state: 0,
            reps: [1; 4],
            prices: price_table(),
        }
    }

    fn emit(&mut self, pos: usize, token: Token) {
        let state = self.state;
        let ps = pos & ((1 << self.props.pb) - 1);
        let model = &mut self.model;
        if token.kind == 0 {
            self.rc.bit(&mut model.is_match[state][ps], 0);
            let prev = if pos == 0 { 0 } else { self.src[pos - 1] };
            let matched = if state >= 7 {
                Some(self.src[pos - self.reps[0]])
            } else {
                None
            };
            model.literal_encode(
                &mut self.rc,
                self.props.context(pos, prev),
                self.src[pos],
                matched,
            );
            self.state = literal_state(state);
        } else {
            self.rc.bit(&mut model.is_match[state][ps], 1);
            if token.kind <= 5 {
                self.rc.bit(&mut model.is_rep[state], 1);
                let rep = token.kind.saturating_sub(2);
                if rep == 0 {
                    self.rc.bit(&mut model.rep_g0[state], 0);
                    self.rc
                        .bit(&mut model.rep0_long[state][ps], u32::from(token.len != 1));
                } else {
                    self.rc.bit(&mut model.rep_g0[state], 1);
                    if rep == 1 {
                        self.rc.bit(&mut model.rep_g1[state], 0);
                    } else {
                        self.rc.bit(&mut model.rep_g1[state], 1);
                        self.rc.bit(&mut model.rep_g2[state], (rep - 2) as u32);
                    }
                    let distance = self.reps[rep];
                    for r in (1..=rep).rev() {
                        self.reps[r] = self.reps[r - 1];
                    }
                    self.reps[0] = distance;
                }
                if token.len == 1 {
                    self.state = if state < 7 { 9 } else { 11 };
                } else {
                    model.rep_length.encode(&mut self.rc, ps, token.len);
                    self.state = if state < 7 { 8 } else { 11 };
                }
            } else {
                self.rc.bit(&mut model.is_rep[state], 0);
                model.length.encode(&mut self.rc, ps, token.len);
                let distance = token.kind - 5;
                model.distance_encode(&mut self.rc, token.len, distance);
                self.reps = [distance, self.reps[0], self.reps[1], self.reps[2]];
                self.state = if state < 7 { 7 } else { 10 };
            }
        }
    }

    fn best_lazy(&self, pos: usize, matches: &[Match]) -> Token {
        let limit = (self.src.len() - pos).min(MAX_LEN);
        let mut token = Token { len: 1, kind: 0 };
        if let Some(m) = matches.last() {
            if m.len >= 4
                || (m.len == 3 && m.distance < (1 << 15))
                || (m.len == 2 && m.distance < 128)
            {
                token = Token {
                    len: m.len,
                    kind: m.distance + 5,
                };
            }
        }
        for (rep, &distance) in self.reps.iter().enumerate() {
            if distance > pos {
                continue;
            }
            let len = match_len(self.src, pos, distance, limit);
            if len >= 2 && len + 1 >= token.len {
                token = Token { len, kind: rep + 2 };
            }
        }
        if token.len == 1 && self.reps[0] <= pos && self.src[pos] == self.src[pos - self.reps[0]] {
            let s = self.state;
            let ps = pos & ((1 << self.props.pb) - 1);
            let prev = if pos == 0 { 0 } else { self.src[pos - 1] };
            let matched = if s >= 7 {
                Some(self.src[pos - self.reps[0]])
            } else {
                None
            };
            let literal = bit_price(&self.prices, self.model.is_match[s][ps], 0)
                + self.model.literal_price(
                    &self.prices,
                    self.props.context(pos, prev),
                    self.src[pos],
                    matched,
                );
            let rep = bit_price(&self.prices, self.model.is_match[s][ps], 1)
                + bit_price(&self.prices, self.model.is_rep[s], 1)
                + bit_price(&self.prices, self.model.rep_g0[s], 0)
                + bit_price(&self.prices, self.model.rep0_long[s][ps], 0);
            if rep <= literal {
                token.kind = 1;
            }
        }
        token
    }

    fn lazy(&mut self, finder: &mut MatchFinder<'_>, level: u8) {
        let mut pos = 0;
        let mut current = Vec::with_capacity(32);
        let mut next = Vec::with_capacity(32);
        let mut prefetched = false;
        while pos < self.src.len() {
            if !prefetched {
                finder.matches(pos, &mut current);
            }
            prefetched = false;
            let mut token = self.best_lazy(pos, &current);
            let mut inserted = pos + 1;
            if level >= 2 && token.len >= 2 && token.len < finder.nice && pos + 1 < self.src.len() {
                finder.matches(pos + 1, &mut next);
                inserted += 1;
                let later = self.best_lazy(pos + 1, &next);
                if later.len > token.len + 1 {
                    token = Token { len: 1, kind: 0 };
                    std::mem::swap(&mut current, &mut next);
                    prefetched = true;
                }
            }
            self.emit(pos, token);
            for p in inserted..pos + token.len {
                finder.insert(p);
            }
            pos += token.len;
        }
    }
}

impl Encoder<'_> {
    fn long_token(
        &self,
        pos: usize,
        reps: [usize; 4],
        matches: &[Match],
        nice: usize,
        rep_lengths: &mut [usize; 4],
    ) -> Option<Token> {
        let limit = (self.src.len() - pos).min(MAX_LEN);
        rep_lengths.fill(0);
        let mut best = matches.last().filter(|m| m.len >= nice).map(|m| Token {
            len: m.len,
            kind: m.distance + 5,
        });
        for (r, distance) in reps.into_iter().enumerate() {
            if distance > pos || (r != 0 && reps[..r].contains(&distance)) {
                continue;
            }
            let len = match_len(self.src, pos, distance, limit);
            rep_lengths[r] = len;
            if len >= nice && best.map_or(true, |old| len + 1 >= old.len) {
                best = Some(Token { len, kind: r + 2 });
            }
        }
        best
    }

    fn optimal(&mut self, finder: &mut MatchFinder<'_>) {
        const BLOCK: usize = 4096;
        let mut nodes = vec![Optimum::EMPTY; BLOCK + 1];
        let mut matches = Vec::with_capacity(32);
        let mut path = Vec::with_capacity(BLOCK);
        let mut len_prices = [[0u32; 274]; 16];
        let mut rep_prices = [[0u32; 274]; 16];
        let mut distance_prices = DistancePrices::new();
        let mut rep_lengths = [0; 4];
        let mut base = 0;
        while base < self.src.len() {
            finder.matches(base, &mut matches);
            if let Some(token) =
                self.long_token(base, self.reps, &matches, finder.nice, &mut rep_lengths)
            {
                self.emit(base, token);
                for p in base + 1..base + token.len {
                    finder.insert(p);
                }
                base += token.len;
                continue;
            }
            let horizon = if base < 128 {
                32
            } else if base < 512 {
                64
            } else if base < 2048 {
                128
            } else if base < 8192 {
                512
            } else if base < 32768 {
                1024
            } else {
                BLOCK
            };
            let n = (self.src.len() - base).min(horizon);
            nodes[..=n].fill(Optimum::EMPTY);
            nodes[0] = Optimum {
                price: 0,
                reps: self.reps,
                state: self.state,
                prev: 0,
                kind: 0,
            };
            self.model
                .length
                .fill_prices(&self.prices, 1 << self.props.pb, &mut len_prices);
            self.model
                .rep_length
                .fill_prices(&self.prices, 1 << self.props.pb, &mut rep_prices);
            distance_prices.update(&self.model, &self.prices);
            let mut stop = n;
            let mut tail = None;
            for i in 0..n {
                let pos = base + i;
                if i != 0 {
                    finder.matches(pos, &mut matches);
                }
                let cur = nodes[i];
                if i != 0 {
                    if let Some(token) =
                        self.long_token(pos, cur.reps, &matches, finder.nice, &mut rep_lengths)
                    {
                        stop = i;
                        tail = Some(token);
                        break;
                    }
                }
                let s = cur.state;
                let ps = pos & ((1 << self.props.pb) - 1);
                let prev = if pos == 0 { 0 } else { self.src[pos - 1] };
                let matched = if s >= 7 {
                    Some(self.src[pos - cur.reps[0]])
                } else {
                    None
                };
                let literal = cur.price
                    + bit_price(&self.prices, self.model.is_match[s][ps], 0)
                    + self.model.literal_price(
                        &self.prices,
                        self.props.context(pos, prev),
                        self.src[pos],
                        matched,
                    );
                nodes[i + 1].offer(literal, cur.reps, literal_state(s), i, 0);
                let match_base = cur.price + bit_price(&self.prices, self.model.is_match[s][ps], 1);
                let rep_base = match_base + bit_price(&self.prices, self.model.is_rep[s], 1);
                let limit = (n - i).min(MAX_LEN);
                for r in 0..4 {
                    let distance = cur.reps[r];
                    if distance > pos || (r != 0 && cur.reps[..r].contains(&distance)) {
                        continue;
                    }
                    let len = rep_lengths[r].min(limit);
                    if len == 0 {
                        continue;
                    }
                    let mut price =
                        rep_base + bit_price(&self.prices, self.model.rep_g0[s], u32::from(r != 0));
                    if r == 0 {
                        nodes[i + 1].offer(
                            price + bit_price(&self.prices, self.model.rep0_long[s][ps], 0),
                            cur.reps,
                            if s < 7 { 9 } else { 11 },
                            i,
                            1,
                        );
                        price += bit_price(&self.prices, self.model.rep0_long[s][ps], 1);
                    } else {
                        price += bit_price(&self.prices, self.model.rep_g1[s], u32::from(r != 1));
                        if r >= 2 {
                            price += bit_price(&self.prices, self.model.rep_g2[s], (r - 2) as u32);
                        }
                    }
                    if len < 2 {
                        continue;
                    }
                    let mut reps = cur.reps;
                    for j in (1..=r).rev() {
                        reps[j] = reps[j - 1];
                    }
                    reps[0] = distance;
                    let state = if s < 7 { 8 } else { 11 };
                    for length in 2..=len {
                        nodes[i + length].offer(
                            price + rep_prices[ps][length],
                            reps,
                            state,
                            i,
                            r + 2,
                        );
                    }
                }
                let normal_base = match_base + bit_price(&self.prices, self.model.is_rep[s], 0);
                for m in &matches {
                    let len = m.len.min(limit);
                    let distance = distance_prices.prices(m.distance);
                    let reps = [m.distance, cur.reps[0], cur.reps[1], cur.reps[2]];
                    let state = if s < 7 { 7 } else { 10 };
                    for length in 2..=len {
                        let price =
                            normal_base + len_prices[ps][length] + distance[(length - 2).min(3)];
                        nodes[i + length].offer(price, reps, state, i, m.distance + 5);
                    }
                }
            }
            path.clear();
            let mut end = stop;
            while end != 0 {
                let node = nodes[end];
                path.push(Token {
                    len: end - node.prev,
                    kind: node.kind,
                });
                end = node.prev;
            }
            let mut pos = base;
            for &token in path.iter().rev() {
                self.emit(pos, token);
                pos += token.len;
            }
            if let Some(token) = tail {
                self.emit(pos, token);
                for p in pos + 1..pos + token.len {
                    finder.insert(p);
                }
                pos += token.len;
            }
            base = pos;
        }
    }
}

fn encode_with(src: &[u8], level: u8, props: Props) -> Vec<u8> {
    let dict_log = if src.len() <= 1 {
        0
    } else {
        (usize::BITS - (src.len() - 1).leading_zeros()).min(26) as u8
    };
    if src.is_empty() {
        return vec![b'V', b'X', b'C', b'1', props.byte(), dict_log];
    }
    let mut encoder = Encoder::new(src, props, dict_log);
    let mut finder = MatchFinder::new(src, level, dict_log);
    if level >= 6 {
        encoder.optimal(&mut finder);
    } else {
        encoder.lazy(&mut finder, level);
    }
    encoder.rc.finish()
}

fn select_props(sample: &[u8]) -> Props {
    const CANDIDATES: [Props; 6] = [
        Props {
            lc: 3,
            lp: 0,
            pb: 2,
        },
        Props {
            lc: 0,
            lp: 2,
            pb: 2,
        },
        Props {
            lc: 1,
            lp: 3,
            pb: 3,
        },
        Props {
            lc: 0,
            lp: 4,
            pb: 4,
        },
        Props {
            lc: 4,
            lp: 0,
            pb: 0,
        },
        Props {
            lc: 8,
            lp: 0,
            pb: 0,
        },
    ];
    let dict_log = if sample.len() <= 1 {
        0
    } else {
        (usize::BITS - (sample.len() - 1).leading_zeros()).min(26) as u8
    };
    let mut encoders = CANDIDATES.map(|props| Encoder::new(sample, props, dict_log));
    let mut finder = MatchFinder::new(sample, 4, dict_log);
    let mut current = Vec::with_capacity(32);
    let mut next = Vec::with_capacity(32);
    let mut prefetched = false;
    let mut pos = 0;
    // Level-four token lengths and rep histories are independent of properties.
    // Only the one-byte literal/short-rep decision needs each candidate's model.
    while pos < sample.len() {
        if !prefetched {
            finder.matches(pos, &mut current);
        }
        prefetched = false;
        let mut token = encoders[0].best_lazy(pos, &current);
        let mut inserted = pos + 1;
        let mut deferred = false;
        if token.len >= 2 && token.len < finder.nice && pos + 1 < sample.len() {
            finder.matches(pos + 1, &mut next);
            inserted += 1;
            let later = encoders[0].best_lazy(pos + 1, &next);
            if later.len > token.len + 1 {
                token = Token { len: 1, kind: 0 };
                std::mem::swap(&mut current, &mut next);
                prefetched = true;
                deferred = true;
            }
        }
        for encoder in &mut encoders {
            let choice = if token.len == 1 && !deferred {
                encoder.best_lazy(pos, &current)
            } else {
                token
            };
            encoder.emit(pos, choice);
        }
        for p in inserted..pos + token.len {
            finder.insert(p);
        }
        pos += token.len;
    }
    let mut best = usize::MAX;
    let mut props = CANDIDATES[0];
    for encoder in encoders {
        let size = encoder.rc.finish().len();
        if size < best {
            best = size;
            props = encoder.props;
        }
    }
    props
}

/// Compress a byte stream using Vortex context modelling and LZ packets.
pub fn vxcm_compress(src: &[u8], level: u8) -> Vec<u8> {
    let level = level.clamp(1, 9);
    let mut props = Props {
        lc: 3,
        lp: 0,
        pb: 2,
    };
    if level >= 5 && !src.is_empty() {
        const SLICE: usize = 256 * 1024;
        let sampled;
        let sample = if src.len() <= 4 * SLICE {
            src
        } else {
            let mut data = Vec::with_capacity(4 * SLICE);
            for i in 0..4 {
                let start = (src.len() - SLICE) * i / 3;
                data.extend_from_slice(&src[start..start + SLICE]);
            }
            sampled = data;
            &sampled
        };
        props = select_props(sample);
    }
    encode_with(src, level, props)
}

#[inline(never)]
fn decode_body<const N: usize>(
    body: &[u8],
    raw_len: usize,
    props: Props,
    dict_log: u8,
) -> Result<Vec<u8>, String> {
    let mut rc = RangeDecoder::new(body)?;
    // Literal tables live on the stack; output is the decoder's only large heap allocation.
    let mut model = Models::new([1024u16; N]);
    let mut out = Vec::new();
    out.try_reserve_exact(raw_len)
        .map_err(|_| "output allocation failed")?;
    let mut state = 0;
    let mut reps = [1usize; 4];
    let pos_mask = (1 << props.pb) - 1;
    let dictionary = 1usize << dict_log;
    while out.len() < raw_len {
        let pos = out.len();
        let ps = pos & pos_mask;
        if rc.bit(&mut model.is_match[state][ps])? == 0 {
            let prev = out.last().copied().unwrap_or(0);
            let matched = if state >= 7 {
                if reps[0] > pos {
                    return Err("literal distance out of range".into());
                }
                Some(out[pos - reps[0]])
            } else {
                None
            };
            let byte = model.literal_decode(&mut rc, props.context(pos, prev), matched)?;
            out.push(byte);
            state = literal_state(state);
        } else {
            let len;
            if rc.bit(&mut model.is_rep[state])? == 0 {
                len = model.length.decode(&mut rc, ps)?;
                let distance = model.distance_decode(&mut rc, len)?;
                reps = [distance, reps[0], reps[1], reps[2]];
                state = if state < 7 { 7 } else { 10 };
            } else {
                if rc.bit(&mut model.rep_g0[state])? == 0 {
                    if rc.bit(&mut model.rep0_long[state][ps])? == 0 {
                        len = 1;
                        state = if state < 7 { 9 } else { 11 };
                    } else {
                        len = model.rep_length.decode(&mut rc, ps)?;
                        state = if state < 7 { 8 } else { 11 };
                    }
                } else {
                    let rep = if rc.bit(&mut model.rep_g1[state])? == 0 {
                        1
                    } else {
                        2 + rc.bit(&mut model.rep_g2[state])? as usize
                    };
                    let distance = reps[rep];
                    for r in (1..=rep).rev() {
                        reps[r] = reps[r - 1];
                    }
                    reps[0] = distance;
                    len = model.rep_length.decode(&mut rc, ps)?;
                    state = if state < 7 { 8 } else { 11 };
                }
            }
            let distance = reps[0];
            if distance == 0 || distance > pos || distance > dictionary {
                return Err("distance out of range".into());
            }
            if len > raw_len - pos {
                return Err("match exceeds output length".into());
            }
            for _ in 0..len {
                let byte = out[out.len() - distance];
                out.push(byte);
            }
        }
    }
    if rc.code != 0 || rc.pos != body.len() {
        return Err("corrupt range termination".into());
    }
    Ok(out)
}

/// Decode exactly `raw_len` bytes, rejecting malformed or incomplete packets.
pub fn vxcm_decompress(comp: &[u8], raw_len: usize) -> Result<Vec<u8>, String> {
    if comp.len() < 6 || &comp[..4] != b"VXC1" {
        return Err("invalid VXCM header".into());
    }
    let value = comp[4] as usize;
    if value >= 9 * 5 * 5 || comp[5] > 26 {
        return Err("invalid VXCM properties".into());
    }
    let props = Props {
        lc: value % 9,
        lp: (value / 9) % 5,
        pb: value / 45,
    };
    if raw_len == 0 {
        return if comp.len() == 6 {
            Ok(Vec::new())
        } else {
            Err("nonempty body for empty output".into())
        };
    }
    let body = &comp[6..];
    if body.len() < 5 {
        return Err("truncated range stream".into());
    }
    let bits = props.lc + props.lp;
    match bits {
        0 => decode_body::<768>(body, raw_len, props, comp[5]),
        1 => decode_body::<1536>(body, raw_len, props, comp[5]),
        2 => decode_body::<3072>(body, raw_len, props, comp[5]),
        3 => decode_body::<6144>(body, raw_len, props, comp[5]),
        4 => decode_body::<12288>(body, raw_len, props, comp[5]),
        5 => decode_body::<24576>(body, raw_len, props, comp[5]),
        6 => decode_body::<49152>(body, raw_len, props, comp[5]),
        7 => decode_body::<98304>(body, raw_len, props, comp[5]),
        8 => decode_body::<196608>(body, raw_len, props, comp[5]),
        _ => std::thread::scope(|scope| {
            std::thread::Builder::new()
                .stack_size(16 * 1024 * 1024)
                .spawn_scoped(scope, || match bits {
                    9 => decode_body::<393216>(body, raw_len, props, comp[5]),
                    10 => decode_body::<786432>(body, raw_len, props, comp[5]),
                    11 => decode_body::<1572864>(body, raw_len, props, comp[5]),
                    _ => decode_body::<3145728>(body, raw_len, props, comp[5]),
                })
                .map_err(|_| "decoder thread allocation failed".to_owned())?
                .join()
                .map_err(|_| "decoder worker failed".to_owned())?
        }),
    }
}

#[cfg(test)]
#[test]
fn cached_distance_prices_match_live_model() {
    let prices = price_table();
    let mut model = Models::new(vec![1024; 768]);
    let mut rc = RangeEncoder::new(&[]);
    let mut cached = DistancePrices::new();
    for i in 0..2048usize {
        let distance = 1 + (i * 7919) % (1 << 26);
        model.distance_encode(&mut rc, 2 + i % 272, distance);
    }
    cached.update(&model, &prices);
    for distance in (1..=4096).chain((0..26).flat_map(|bit| {
        let base = 1usize << bit;
        [base, base + 1, base + base / 2, base * 2 - 1]
    })) {
        for (context, price) in cached.prices(distance).into_iter().enumerate() {
            assert_eq!(price, model.distance_price(&prices, context + 2, distance));
        }
    }
}

#[cfg(test)]
#[test]
fn cached_length_prices_match_live_model() {
    let prices = price_table();
    let mut model = LengthModel::new();
    let mut rc = RangeEncoder::new(&[]);
    for i in 0..2048 {
        model.encode(&mut rc, i & 15, 2 + (i * 37) % 272);
    }
    let mut cached = [[0u32; 274]; 16];
    for pos_states in [1, 4, 8, 16] {
        model.fill_prices(&prices, pos_states, &mut cached);
        for (pos, row) in cached[..pos_states].iter().enumerate() {
            for (len, &price) in row.iter().enumerate().skip(2) {
                assert_eq!(price, model.price(&prices, pos, len));
            }
        }
    }
}

#[cfg(test)]
#[test]
fn shared_property_trials_match_independent_encodes() {
    for len in [1, 255, 16_384] {
        for structured in [false, true] {
            let mut seed = 0x1234_5678u32;
            let sample: Vec<u8> = (0..len)
                .map(|i| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    if structured && i % 16 < 12 {
                        (i % 29) as u8
                    } else {
                        seed as u8
                    }
                })
                .collect();
            let expected = [
                (3, 0, 2),
                (0, 2, 2),
                (1, 3, 3),
                (0, 4, 4),
                (4, 0, 0),
                (8, 0, 0),
            ]
            .into_iter()
            .map(|(lc, lp, pb)| Props { lc, lp, pb })
            .min_by_key(|&props| encode_with(&sample, 4, props).len())
            .unwrap();
            assert_eq!(select_props(&sample).byte(), expected.byte());
        }
    }
}

#[rustfmt::skip]
#[cfg(test)] #[path = "vxcm_tests.rs"] mod tests;
