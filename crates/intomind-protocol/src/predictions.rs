//! Predictions (notify): one self-describing window of model outputs.

use crate::Error;

pub const PACKET_TYPE_PREDICTION: u8 = 0x02;
pub const HEADER_LEN: usize = 28;
/// The most outputs one prediction carries: as many four byte values as fit
/// [`NOTIFICATION_MAX`](crate::NOTIFICATION_MAX) after the header, so a
/// prediction always arrives whole. A head may have no more outputs than
/// this, and the device reports it as `head_max_outputs`.
pub const MAX_OUTPUTS: usize = (crate::NOTIFICATION_MAX - HEADER_LEN) / 4;

pub mod flags {
    pub const GAP_IN_WINDOW: u8 = 1 << 0;
    pub const DUTY_REDUCED: u8 = 1 << 1;
    pub const LEADOFF_IN_WINDOW: u8 = 1 << 2;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PredictionHeader {
    pub flags: u8,
    pub head_slot: u8,
    pub n_outputs: u8,
    /// Raw stream index of the window's first sample.
    pub sample_index: u32,
    /// Device time of that sample.
    pub device_time: u64,
    /// Window length in raw samples at the current rate.
    pub window_samples: u16,
    pub head_id: [u8; 8],
    /// 1.1: what the window was taken from, a `pipeline::input_source`
    /// value. A 1.0 device sends 0, the stream.
    pub input_source: u8,
}

/// The outputs of one prediction, borrowed from the packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Outputs<'a>(&'a [u8]);

impl<'a> Outputs<'a> {
    pub fn len(&self) -> usize {
        self.0.len() / 4
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn get(&self, i: usize) -> Option<f32> {
        let b = self.0.get(i * 4..i * 4 + 4)?;
        Some(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn iter(&self) -> impl Iterator<Item = f32> + 'a {
        self.0.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

impl PredictionHeader {
    /// Parse a notification. The payload must be exactly `n_outputs` f32s.
    pub fn parse(buf: &[u8]) -> Result<(Self, Outputs<'_>), Error> {
        if buf.len() < HEADER_LEN {
            return Err(Error::Truncated);
        }
        if buf[0] != PACKET_TYPE_PREDICTION {
            return Err(Error::Invalid);
        }
        let mut head_id = [0u8; 8];
        head_id.copy_from_slice(&buf[18..26]);
        let h = PredictionHeader {
            flags: buf[1],
            head_slot: buf[2],
            n_outputs: buf[3],
            sample_index: u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
            device_time: u64::from_le_bytes([
                buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
            ]),
            window_samples: u16::from_le_bytes([buf[16], buf[17]]),
            head_id,
            input_source: buf[26],
        };
        let payload = &buf[HEADER_LEN..];
        if payload.len() != h.n_outputs as usize * 4 {
            return Err(Error::Truncated);
        }
        Ok((h, Outputs(payload)))
    }

    /// Encode header plus outputs. `outputs.len()` must equal `n_outputs`.
    pub fn encode(&self, outputs: &[f32], out: &mut [u8]) -> Result<usize, Error> {
        if outputs.len() != self.n_outputs as usize {
            return Err(Error::Invalid);
        }
        let n = HEADER_LEN + outputs.len() * 4;
        if out.len() < n {
            return Err(Error::NoRoom);
        }
        out[0] = PACKET_TYPE_PREDICTION;
        out[1] = self.flags;
        out[2] = self.head_slot;
        out[3] = self.n_outputs;
        out[4..8].copy_from_slice(&self.sample_index.to_le_bytes());
        out[8..16].copy_from_slice(&self.device_time.to_le_bytes());
        out[16..18].copy_from_slice(&self.window_samples.to_le_bytes());
        out[18..26].copy_from_slice(&self.head_id);
        out[26] = self.input_source;
        out[27] = 0;
        for (i, v) in outputs.iter().enumerate() {
            out[HEADER_LEN + i * 4..HEADER_LEN + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let h = PredictionHeader {
            flags: flags::DUTY_REDUCED,
            head_slot: 2,
            n_outputs: 3,
            sample_index: 123_456,
            device_time: 9_876_543_210,
            window_samples: 2000,
            head_id: [1, 2, 3, 4, 5, 6, 7, 8],
            input_source: crate::pipeline::input_source::OWN_CHAIN,
        };
        let outs = [0.25f32, -1.5, 1e-3];
        let mut buf = [0u8; HEADER_LEN + 12];
        assert_eq!(h.encode(&outs, &mut buf).unwrap(), 40);
        let (h2, o) = PredictionHeader::parse(&buf).unwrap();
        assert_eq!(h2, h);
        assert_eq!(o.len(), 3);
        assert_eq!(o.get(1), Some(-1.5));
        assert!(o.get(3).is_none());
        let got: [f32; 3] = [o.get(0).unwrap(), o.get(1).unwrap(), o.get(2).unwrap()];
        assert_eq!(got, outs);
        assert_eq!(o.iter().count(), 3);
    }

    #[test]
    fn payload_length_and_type_are_enforced() {
        let h = PredictionHeader {
            flags: 0,
            head_slot: 0,
            n_outputs: 2,
            sample_index: 0,
            device_time: 0,
            window_samples: 1000,
            head_id: [0; 8],
                    input_source: 0,
        };
        let mut buf = [0u8; HEADER_LEN + 8];
        assert_eq!(h.encode(&[1.0, 2.0, 3.0], &mut buf), Err(Error::Invalid));
        h.encode(&[1.0, 2.0], &mut buf).unwrap();
        assert_eq!(PredictionHeader::parse(&buf[..HEADER_LEN + 7]), Err(Error::Truncated));
        buf[0] = 0x01;
        assert_eq!(PredictionHeader::parse(&buf), Err(Error::Invalid));
    }

    #[test]
    fn the_largest_head_fits_one_notification() {
        assert!(HEADER_LEN + 32 * 4 <= 244);
    }
}
