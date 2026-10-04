//! CRC-32, IEEE 802.3 (the zlib CRC). The Update service's QUERY reports it
//! over every transfer byte accepted so far, so a host can check progress
//! against its own copy without any key.

const POLY: u32 = 0xEDB8_8320;

const fn table() -> [u32; 16] {
    let mut t = [0u32; 16];
    let mut i = 0;
    while i < 16 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 4 {
            c = if c & 1 != 0 { POLY ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
}

const TABLE: [u32; 16] = table();

/// Running CRC-32. `Crc32::new()` then `update` any number of times, then
/// `finish`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Crc32(u32);

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

impl Crc32 {
    /// A fresh running checksum, with no bytes folded in yet.
    pub const fn new() -> Self {
        Crc32(0xFFFF_FFFF)
    }

    /// Fold more bytes into the running checksum.
    pub fn update(&mut self, data: &[u8]) {
        let mut c = self.0;
        for &b in data {
            c ^= b as u32;
            c = TABLE[(c & 0xF) as usize] ^ (c >> 4);
            c = TABLE[(c & 0xF) as usize] ^ (c >> 4);
        }
        self.0 = c;
    }

    /// The checksum of every byte folded in so far.
    pub const fn finish(self) -> u32 {
        self.0 ^ 0xFFFF_FFFF
    }
}

/// One-shot CRC-32 of a slice.
pub fn crc32(data: &[u8]) -> u32 {
    let mut c = Crc32::new();
    c.update(data);
    c.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_value() {
        // The standard check value for CRC-32/ISO-HDLC.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn incremental_equals_one_shot() {
        let data: [u8; 300] = core::array::from_fn(|i| (i * 7 + 3) as u8);
        let mut c = Crc32::new();
        c.update(&data[..100]);
        c.update(&data[100..250]);
        c.update(&data[250..]);
        assert_eq!(c.finish(), crc32(&data));
    }
}
