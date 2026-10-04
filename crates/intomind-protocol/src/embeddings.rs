//! Embeddings (notify): the encoder's output for one window, in two forms.
//!
//! The window embedding is the mean over live channels and over time of the
//! tokens, and is what a head consumes. The tokens are the encoder's
//! output before that pooling, one vector per channel per slice of the
//! window, and are what reconstruction consumes. Both are quantized the way
//! a head receives an embedding (`head::EMBED_SCALE`), so a head trained on
//! what a host receives runs on the device unchanged. New in 1.2.

use crate::Error;

pub const PACKET_TYPE_EMBEDDING: u8 = 0x03;
pub const HEADER_LEN: usize = 28;
/// The `token` byte of a window embedding. A token's own index otherwise,
/// channel-major over all channels: `channel * tokens_per_channel + slice`.
pub const WINDOW_TOKEN: u8 = 0xFF;
/// Values one notification carries: as many as fit
/// [`NOTIFICATION_MAX`](crate::NOTIFICATION_MAX) after the header, so that
/// every part of every vector arrives whole at the smallest MTU the
/// contract allows. A wider vector goes in parts (section 27 of 1.4).
/// Until firmware 1.4.2 a part carried 108 values, 244 bytes, which a host
/// below an MTU of 247 received cut short.
pub const MAX_VALUES_PER_PACKET: usize = (crate::NOTIFICATION_MAX - HEADER_LEN) / 2;

pub mod flags {
    pub const GAP_IN_WINDOW: u8 = 1 << 0;
    pub const DUTY_REDUCED: u8 = 1 << 1;
    pub const LEADOFF_IN_WINDOW: u8 = 1 << 2;
    /// Another notification carries the rest of this vector's values.
    pub const MORE_PARTS: u8 = 1 << 3;
}

/// SET_EMBEDDINGS argument: which form the device sends.
pub mod form {
    pub const OFF: u8 = 0;
    pub const WINDOW: u8 = 1;
    pub const TOKENS: u8 = 2;
    pub const BOTH: u8 = 3;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingHeader {
    pub flags: u8,
    /// Values in the whole vector.
    pub embed_dim: u8,
    /// Index of the first value carried in this notification.
    pub first: u8,
    /// Raw stream index of the window's first sample.
    pub sample_index: u32,
    /// Device time of that sample.
    pub device_time: u64,
    /// Window length in raw samples at the current rate.
    pub window_samples: u16,
    pub encoder_id: [u8; 8],
    /// What the window was taken from, a `pipeline::input_source` value.
    pub input_source: u8,
    /// `WINDOW_TOKEN`, or the token's index.
    pub token: u8,
}

/// The values of one notification, borrowed from the packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Values<'a>(&'a [u8]);

impl<'a> Values<'a> {
    pub fn len(&self) -> usize {
        self.0.len() / 2
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn get(&self, i: usize) -> Option<i16> {
        let b = self.0.get(i * 2..i * 2 + 2)?;
        Some(i16::from_le_bytes([b[0], b[1]]))
    }

    pub fn iter(&self) -> impl Iterator<Item = i16> + 'a {
        let bytes = self.0;
        (0..bytes.len() / 2).map(move |i| i16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]))
    }
}

impl EmbeddingHeader {
    /// Parse a notification. Returns the header and its values.
    pub fn parse(buf: &[u8]) -> Result<(Self, Values<'_>), Error> {
        if buf.len() < HEADER_LEN {
            return Err(Error::Truncated);
        }
        if buf[0] != PACKET_TYPE_EMBEDDING {
            return Err(Error::Invalid);
        }
        let mut encoder_id = [0u8; 8];
        encoder_id.copy_from_slice(&buf[18..26]);
        let h = EmbeddingHeader {
            flags: buf[1],
            embed_dim: buf[2],
            first: buf[3],
            sample_index: u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
            device_time: u64::from_le_bytes([buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15]]),
            window_samples: u16::from_le_bytes([buf[16], buf[17]]),
            encoder_id,
            input_source: buf[26],
            token: buf[27],
        };
        let payload = &buf[HEADER_LEN..];
        if payload.len() % 2 != 0 || payload.is_empty() {
            return Err(Error::Truncated);
        }
        // The values carried must lie inside the vector they belong to.
        let n = payload.len() / 2;
        if h.first as usize + n > h.embed_dim as usize {
            return Err(Error::Invalid);
        }
        Ok((h, Values(payload)))
    }

    /// Encode header plus values. `values` are the vector's values from
    /// `first` on, and must fit inside `embed_dim` and the buffer.
    pub fn encode(&self, values: &[i16], out: &mut [u8]) -> Result<usize, Error> {
        if values.is_empty() || self.first as usize + values.len() > self.embed_dim as usize {
            return Err(Error::Invalid);
        }
        let n = HEADER_LEN + values.len() * 2;
        if out.len() < n {
            return Err(Error::NoRoom);
        }
        out[0] = PACKET_TYPE_EMBEDDING;
        out[1] = self.flags;
        out[2] = self.embed_dim;
        out[3] = self.first;
        out[4..8].copy_from_slice(&self.sample_index.to_le_bytes());
        out[8..16].copy_from_slice(&self.device_time.to_le_bytes());
        out[16..18].copy_from_slice(&self.window_samples.to_le_bytes());
        out[18..26].copy_from_slice(&self.encoder_id);
        out[26] = self.input_source;
        out[27] = self.token;
        for (i, v) in values.iter().enumerate() {
            out[HEADER_LEN + i * 2..HEADER_LEN + i * 2 + 2].copy_from_slice(&v.to_le_bytes());
        }
        Ok(n)
    }

    /// Whether this notification completes its vector.
    pub fn is_last_part(&self) -> bool {
        self.flags & flags::MORE_PARTS == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(first: u8, token: u8, flags: u8) -> EmbeddingHeader {
        EmbeddingHeader {
            flags,
            embed_dim: 96,
            first,
            sample_index: 123_456,
            device_time: 9_876_543_210,
            window_samples: 2000,
            encoder_id: [1, 2, 3, 4, 5, 6, 7, 8],
            input_source: crate::pipeline::input_source::STREAM,
            token,
        }
    }

    #[test]
    fn a_vector_that_fits_round_trips_in_one_notification() {
        let mut h = header(0, WINDOW_TOKEN, flags::LEADOFF_IN_WINDOW);
        h.embed_dim = 64;
        let values: [i16; 64] = core::array::from_fn(|i| (i as i16 - 32) * 100);
        let mut buf = [0u8; crate::NOTIFICATION_MAX];
        let n = h.encode(&values, &mut buf).unwrap();
        assert_eq!(n, crate::NOTIFICATION_MAX);
        let (h2, v) = EmbeddingHeader::parse(&buf[..n]).unwrap();
        assert_eq!(h2, h);
        assert_eq!(v.len(), 64);
        assert_eq!(v.get(0), Some(-3200));
        assert_eq!(v.get(63), Some(3100));
        assert!(v.get(64).is_none());
        assert_eq!(v.iter().count(), 64);
        assert!(h2.is_last_part());
    }

    #[test]
    fn a_wide_vector_goes_in_parts_that_name_their_place() {
        let mut h = header(0, 17, flags::MORE_PARTS);
        h.embed_dim = 128;
        let mut buf = [0u8; crate::NOTIFICATION_MAX];
        let n = h.encode(&[7i16; MAX_VALUES_PER_PACKET], &mut buf).unwrap();
        assert_eq!(n, crate::NOTIFICATION_MAX);
        let (p1, v1) = EmbeddingHeader::parse(&buf[..n]).unwrap();
        assert!(!p1.is_last_part());
        assert_eq!((p1.first, v1.len(), p1.token), (0, 64, 17));
        let h2 = EmbeddingHeader { first: 64, flags: 0, ..h };
        let n = h2.encode(&[9i16; 64], &mut buf).unwrap();
        let (p2, v2) = EmbeddingHeader::parse(&buf[..n]).unwrap();
        assert!(p2.is_last_part());
        assert_eq!((p2.first, v2.len()), (64, 64));
        assert_eq!(p2.first as usize + v2.len(), p2.embed_dim as usize);
    }

    #[test]
    fn values_outside_the_vector_and_odd_payloads_are_refused() {
        let h = header(90, WINDOW_TOKEN, 0);
        let mut buf = [0u8; crate::NOTIFICATION_MAX];
        assert_eq!(h.encode(&[1i16; 7], &mut buf), Err(Error::Invalid));
        assert_eq!(h.encode(&[], &mut buf), Err(Error::Invalid));
        let n = h.encode(&[1i16; 6], &mut buf).unwrap();
        assert_eq!(EmbeddingHeader::parse(&buf[..n - 1]), Err(Error::Truncated));
        assert_eq!(EmbeddingHeader::parse(&buf[..HEADER_LEN]), Err(Error::Truncated));
        buf[3] = 91;
        assert_eq!(EmbeddingHeader::parse(&buf[..n]), Err(Error::Invalid));
        buf[3] = 90;
        buf[0] = 0x02;
        assert_eq!(EmbeddingHeader::parse(&buf[..n]), Err(Error::Invalid));
    }

    /// Every part of every vector the format can carry fits the limit, and
    /// the parts tile the vector exactly. The launch model's 76 values go
    /// in two parts, 64 and 12. With the 108-value parts of firmware 1.4.1
    /// and before, its one part was 180 bytes, more than an MTU of 159
    /// carries.
    #[test]
    fn every_part_of_every_vector_fits_one_notification() {
        assert_eq!(MAX_VALUES_PER_PACKET, 64);
        assert_eq!(HEADER_LEN + 2 * MAX_VALUES_PER_PACKET, crate::NOTIFICATION_MAX);
        for dim in 1..=u8::MAX as usize {
            let mut first = 0;
            let mut parts = 0;
            while first < dim {
                let n = (dim - first).min(MAX_VALUES_PER_PACKET);
                let more = first + n < dim;
                let h = EmbeddingHeader { embed_dim: dim as u8, first: first as u8, ..header(0, 3, if more { flags::MORE_PARTS } else { 0 }) };
                let mut buf = [0u8; crate::NOTIFICATION_MAX];
                let len = h.encode(&[1i16; MAX_VALUES_PER_PACKET][..n], &mut buf).unwrap();
                assert!(len <= crate::NOTIFICATION_MAX);
                let (p, v) = EmbeddingHeader::parse(&buf[..len]).unwrap();
                assert_eq!((p.first as usize, v.len(), p.is_last_part()), (first, n, !more));
                first += n;
                parts += 1;
            }
            assert_eq!(parts, dim.div_ceil(MAX_VALUES_PER_PACKET));
        }
        assert_eq!(76usize.div_ceil(MAX_VALUES_PER_PACKET), 2);
    }
}
