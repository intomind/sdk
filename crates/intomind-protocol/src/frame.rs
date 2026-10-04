//! EEG data packet: the stream itself.
//!
//! Header little-endian. The sample payload is in the converter's own
//! order, big-endian two's-complement int24, MSB first, so the device
//! copies its bytes through without swapping any of them and the host
//! does sign extension and scaling. Layout per the contract, offsets in
//! bytes:
//!
//! ```text
//! 0   u8   packet_type            0x01 = EEG data, 0x04 = synthetic data (1.3)
//! 1   u8   flags                  bit0 discontinuity_before_this_packet
//!                                 bit1 leadoff_active
//!                                 bits2-3 mode (0 normal, 1 test, 2 short)
//!                                 bit4 usb_present
//! 2   u16  samples_lost_before    convenience; sample_index is authoritative
//! 4   u32  sample_index           index of first sample; monotonic; wraps 2^32
//! 8   u64  device_time            device-time the first sample was ready, in ticks
//! 16  u8   n_samples
//! 17  u8   loff_statp             latest lead-off, bit n = channel n+1
//! 18  u8   gain_code              0..6 -> gain 1,2,4,6,8,12,24
//! 19  u8   rate_code              4=1000, 5=500, 6=250 SPS
//! 20  ...  payload = n_samples x n_channels x 3 bytes (int24 BE)
//! ```

use crate::Error;

/// Packet type for a batch of measured samples.
pub const PACKET_TYPE_EEG: u8 = 0x01;
/// 1.3: samples the device generated with its converter off. Exactly the
/// layout of type 0x01, so a host decodes it with the same code and knows
/// every sample in it was generated. A 1.2 host refuses it, and so never
/// stores one as a measurement.
pub const PACKET_TYPE_SYNTHETIC: u8 = 0x04;

/// Whether a packet type carries samples in this layout.
pub fn is_data_packet(packet_type: u8) -> bool {
    packet_type == PACKET_TYPE_EEG || packet_type == PACKET_TYPE_SYNTHETIC
}
/// Bytes in the fixed header, ahead of the sample payload.
pub const HEADER_LEN: usize = 20;
/// The IntoMind One's channel count. The header does not carry it; it is a
/// device capability from Device Info, and this constant exists for the
/// device build and the tests, not as a protocol assumption.
pub const CHANNELS_IM1: usize = 4;

/// Bits of the header's `flags` byte.
pub mod flags {
    /// The timeline broke before this packet: a loss or a re-base. See
    /// [`continuity`](super::continuity).
    pub const DISCONTINUITY: u8 = 1 << 0;
    /// Lead-off detection was enabled for this packet's epoch.
    pub const LEADOFF_ACTIVE: u8 = 1 << 1;
    /// Shift to read the two mode bits out of `flags`.
    pub const MODE_SHIFT: u8 = 2;
    /// Mask for the two mode bits of `flags`, the same values as
    /// `control::mode`.
    pub const MODE_MASK: u8 = 0b11 << 2;
    /// 1.4: USB power was present when this packet was sent.
    pub const USB_PRESENT: u8 = 1 << 4;
}

/// `samples_lost_before` saturates here.
pub const LOST_SATURATED: u16 = 0xFFFF;

/// Gains by gain_code index, per the spec table.
pub const GAIN_BY_CODE: [u8; 7] = [1, 2, 4, 6, 8, 12, 24];

/// SPS by rate_code. Codes outside the table are undefined on the wire.
pub fn sps_of_rate_code(code: u8) -> Option<u16> {
    match code {
        4 => Some(1000),
        5 => Some(500),
        6 => Some(250),
        _ => None,
    }
}

/// The decoded fixed header of an EEG Data or synthetic data packet, ahead
/// of its sample payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataHeader {
    /// See the [`flags`] module.
    pub flags: u8,
    /// Convenience count of samples lost before this packet, saturating.
    /// `sample_index` continuity is the authoritative account.
    pub samples_lost_before: u16,
    /// Index of the first sample in this packet. Monotonic, wraps at 2^32.
    pub sample_index: u32,
    /// Device time of the first sample's data-ready edge, in `time_tick_hz` ticks.
    pub device_time: u64,
    /// Samples carried in this packet.
    pub n_samples: u8,
    /// Lead-off status latched with this packet's last sample, bit n for channel n+1.
    pub loff_statp: u8,
    /// The gain in force, a `GAIN_BY_CODE` index.
    pub gain_code: u8,
    /// The sample rate code in force. `sps_of_rate_code` converts it to samples per second.
    pub rate_code: u8,
}

impl DataHeader {
    /// Parse a notification. Returns the header and the borrowed sample
    /// payload. The payload length must be exactly
    /// `n_samples * channels * 3` for the channel count the caller knows
    /// from Device Info.
    pub fn parse(buf: &[u8], channels: usize) -> Result<(Self, &[u8]), Error> {
        if buf.len() < HEADER_LEN {
            return Err(Error::Truncated);
        }
        if !is_data_packet(buf[0]) {
            return Err(Error::Invalid);
        }
        let h = DataHeader {
            flags: buf[1],
            samples_lost_before: u16::from_le_bytes([buf[2], buf[3]]),
            sample_index: u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
            device_time: u64::from_le_bytes([
                buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
            ]),
            n_samples: buf[16],
            loff_statp: buf[17],
            gain_code: buf[18],
            rate_code: buf[19],
        };
        if h.gain_code as usize >= GAIN_BY_CODE.len() {
            return Err(Error::Invalid);
        }
        let want = h.n_samples as usize * channels * 3;
        let payload = &buf[HEADER_LEN..];
        if payload.len() != want {
            return Err(Error::Truncated);
        }
        Ok((h, payload))
    }

    /// Encode the header into the front of `out`. The device then appends
    /// the raw converter bytes. Returns `HEADER_LEN`.
    pub fn encode(&self, out: &mut [u8]) -> Result<usize, Error> {
        self.encode_as(PACKET_TYPE_EEG, out)
    }

    /// Encode the header as `packet_type`: measured samples or, in 1.3,
    /// synthetic ones.
    pub fn encode_as(&self, packet_type: u8, out: &mut [u8]) -> Result<usize, Error> {
        if out.len() < HEADER_LEN {
            return Err(Error::NoRoom);
        }
        if !is_data_packet(packet_type) {
            return Err(Error::Invalid);
        }
        out[0] = packet_type;
        out[1] = self.flags;
        out[2..4].copy_from_slice(&self.samples_lost_before.to_le_bytes());
        out[4..8].copy_from_slice(&self.sample_index.to_le_bytes());
        out[8..16].copy_from_slice(&self.device_time.to_le_bytes());
        out[16] = self.n_samples;
        out[17] = self.loff_statp;
        out[18] = self.gain_code;
        out[19] = self.rate_code;
        Ok(HEADER_LEN)
    }
}

/// Sign-extend one big-endian int24 sample to i32.
#[inline]
pub fn int24_be(b: [u8; 3]) -> i32 {
    ((i32::from(b[0]) << 24) | (i32::from(b[1]) << 16) | (i32::from(b[2]) << 8)) >> 8
}

/// Host-side scaling: `microvolts = raw * (2 * vref_uv) / (gain * 2^24)`.
/// Kept in f64 because it runs on the host. The device never scales.
pub fn microvolts(raw: i32, vref_uv: f64, gain: u8) -> f64 {
    raw as f64 * (2.0 * vref_uv) / (gain as f64 * (1u32 << 24) as f64)
}

/// What one packet's indexing says happened since the previous packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Continuity {
    /// The expected next index arrived.
    Continuous,
    /// Samples were lost. The count is exact from the index arithmetic.
    Gap {
        /// Samples lost.
        lost: u32,
    },
    /// The timeline re-based (epoch reset or stream start). A break, not a
    /// loss. Per the spec a non-positive step is never read as a count.
    Break,
}

/// Judge continuity from the previous packet's first index and sample count
/// and the new packet's first index and discontinuity flag. The subtraction
/// is done signed so the index's own 2^32 wrap stays exact, per the spec
/// (this arithmetic was a v0.1 firmware defect, fixed 2026-08-31).
pub fn continuity(
    prev_index: u32,
    prev_n_samples: u8,
    new_index: u32,
    new_flags: u8,
) -> Continuity {
    let expected = prev_index.wrapping_add(prev_n_samples as u32);
    let step = new_index.wrapping_sub(expected) as i32;
    let flagged = new_flags & flags::DISCONTINUITY != 0;
    if step == 0 && !flagged {
        return Continuity::Continuous;
    }
    if step > 0 {
        return Continuity::Gap { lost: step as u32 };
    }
    // step <= 0: backwards or same index with the flag. A re-base.
    // step == 0 with the flag set is the documented "break of no meaningful
    // extent".
    let _ = flagged;
    Continuity::Break
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header() -> DataHeader {
        DataHeader {
            flags: flags::LEADOFF_ACTIVE | (1 << flags::MODE_SHIFT),
            samples_lost_before: 0,
            sample_index: 0xDEAD_BEEF,
            device_time: 0x0123_4567_89AB_CDEF,
            n_samples: 2,
            loff_statp: 0b0000_0101,
            gain_code: 6,
            rate_code: 5,
        }
    }

    #[test]
    fn round_trip() {
        let h = header();
        let mut buf = [0u8; HEADER_LEN + 2 * CHANNELS_IM1 * 3];
        h.encode(&mut buf).unwrap();
        // fill payload with recognizable converter bytes
        for (i, b) in buf[HEADER_LEN..].iter_mut().enumerate() {
            *b = i as u8;
        }
        let (h2, payload) = DataHeader::parse(&buf, CHANNELS_IM1).unwrap();
        assert_eq!(h, h2);
        assert_eq!(payload.len(), 2 * CHANNELS_IM1 * 3);
        assert_eq!(payload[0], 0);
    }

    #[test]
    fn payload_length_is_enforced() {
        let h = header();
        let mut buf = [0u8; HEADER_LEN + 2 * CHANNELS_IM1 * 3 - 1];
        h.encode(&mut buf).unwrap();
        assert_eq!(DataHeader::parse(&buf, CHANNELS_IM1), Err(Error::Truncated));
    }

    #[test]
    fn undefined_gain_code_is_rejected() {
        let mut h = header();
        h.gain_code = 7;
        h.n_samples = 0;
        let mut buf = [0u8; HEADER_LEN];
        h.encode(&mut buf).unwrap();
        assert_eq!(DataHeader::parse(&buf, CHANNELS_IM1), Err(Error::Invalid));
    }

    #[test]
    fn int24_sign_extension() {
        assert_eq!(int24_be([0x00, 0x00, 0x01]), 1);
        assert_eq!(int24_be([0xFF, 0xFF, 0xFF]), -1);
        assert_eq!(int24_be([0x80, 0x00, 0x00]), -(1 << 23));
        assert_eq!(int24_be([0x7F, 0xFF, 0xFF]), (1 << 23) - 1);
    }

    #[test]
    fn scaling_matches_the_spec_example() {
        // Spec: gain 24, vref 4.5 V -> ~0.02235 uV/LSB.
        let uv_per_lsb = microvolts(1, 4_500_000.0, 24);
        assert!((uv_per_lsb - 0.02235).abs() < 0.0001, "{uv_per_lsb}");
    }

    #[test]
    fn continuity_continuous_and_gap() {
        assert_eq!(continuity(100, 5, 105, 0), Continuity::Continuous);
        assert_eq!(continuity(100, 5, 108, 0), Continuity::Gap { lost: 3 });
    }

    #[test]
    fn continuity_is_exact_across_the_u32_wrap() {
        // prev packet holds the last 5 samples before wrap, next starts at 3:
        // expected wraps to 1, so 2 samples were lost, not 2^32 - something.
        let prev = u32::MAX - 3; // expected next = prev + 5 wraps to 1
        assert_eq!(continuity(prev, 5, 3, 0), Continuity::Gap { lost: 2 });
    }

    #[test]
    fn rebase_is_a_break_never_a_loss_count() {
        // Backwards index: epoch re-base. The spec's v0.1 defect read this
        // as a saturated loss of 65535. It is a break.
        assert_eq!(continuity(1000, 5, 0, flags::DISCONTINUITY), Continuity::Break);
        // Zero step with the flag set: break of no meaningful extent.
        assert_eq!(continuity(1000, 5, 1005, flags::DISCONTINUITY), Continuity::Break);
    }
}
