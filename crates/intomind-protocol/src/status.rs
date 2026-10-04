//! Status characteristic: read plus notify, twelve bytes.
//!
//! ```text
//! u8   state              0 idle, 1 streaming
//! u8   mode
//! u8   gain_code
//! u8   rate_code
//! u8   charger_state      0 no-input, 1 charging, 2 complete, 3 fault, 4 standby
//! u8   battery_percent    0..100, or 0xFF unknown
//! u8   loff_statp         as latched with the most recent sample
//! u8   flags              bit0 usb_present, bit1 buffer_high_watermark
//! u16  dropped_total      lifetime dropped samples this power-on, saturates
//! u16  buffer_fill        samples currently buffered on the device
//! ```
//!
//! `dropped_total` is telemetry. The authoritative loss record is
//! `sample_index` continuity in the data stream.

use crate::Error;

/// Bytes in the message.
pub const LEN: usize = 12;
/// `battery_percent` when the device has no battery sense circuit.
pub const BATTERY_UNKNOWN: u8 = 0xFF;

/// Bits of the `flags` field.
pub mod flags {
    /// USB power is present.
    pub const USB_PRESENT: u8 = 1 << 0;
    /// `buffer_fill` is at or above three quarters of the device's buffer.
    pub const BUFFER_HIGH_WATERMARK: u8 = 1 << 1;
}

/// A parsed Status read or notification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatusMsg {
    /// 0 idle, 1 streaming.
    pub state: u8,
    /// The mode in force, a `control::mode` value.
    pub mode: u8,
    /// The gain in force, a `frame::GAIN_BY_CODE` index.
    pub gain_code: u8,
    /// The sample rate code in force. `frame::sps_of_rate_code` converts it to samples per second.
    pub rate_code: u8,
    /// 0 no input, 1 charging, 2 complete, 3 fault, 4 standby.
    pub charger_state: u8,
    /// 0 to 100, or `BATTERY_UNKNOWN`.
    pub battery_percent: u8,
    /// Lead-off status latched with the most recent sample acquired.
    pub loff_statp: u8,
    /// See the [`flags`] module.
    pub flags: u8,
    /// Samples acquired but never delivered this power-on, saturating.
    pub dropped_total: u16,
    /// Samples currently buffered on the device.
    pub buffer_fill: u16,
}

impl StatusMsg {
    /// Decode a Status read or notification.
    pub fn parse(buf: &[u8]) -> Result<Self, Error> {
        if buf.len() < LEN {
            return Err(Error::Truncated);
        }
        Ok(StatusMsg {
            state: buf[0],
            mode: buf[1],
            gain_code: buf[2],
            rate_code: buf[3],
            charger_state: buf[4],
            battery_percent: buf[5],
            loff_statp: buf[6],
            flags: buf[7],
            dropped_total: u16::from_le_bytes([buf[8], buf[9]]),
            buffer_fill: u16::from_le_bytes([buf[10], buf[11]]),
        })
    }

    /// Encode a Status message into `out`.
    pub fn encode(&self, out: &mut [u8]) -> Result<usize, Error> {
        if out.len() < LEN {
            return Err(Error::NoRoom);
        }
        out[0] = self.state;
        out[1] = self.mode;
        out[2] = self.gain_code;
        out[3] = self.rate_code;
        out[4] = self.charger_state;
        out[5] = self.battery_percent;
        out[6] = self.loff_statp;
        out[7] = self.flags;
        out[8..10].copy_from_slice(&self.dropped_total.to_le_bytes());
        out[10..12].copy_from_slice(&self.buffer_fill.to_le_bytes());
        Ok(LEN)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let s = StatusMsg {
            state: 1,
            mode: 0,
            gain_code: 6,
            rate_code: 5,
            charger_state: 2,
            battery_percent: 87,
            loff_statp: 0b0000_1001,
            flags: flags::USB_PRESENT,
            dropped_total: 3,
            buffer_fill: 512,
        };
        let mut buf = [0u8; LEN];
        s.encode(&mut buf).unwrap();
        assert_eq!(StatusMsg::parse(&buf).unwrap(), s);
    }

    #[test]
    fn short_buffer_is_truncated_not_a_panic() {
        assert_eq!(StatusMsg::parse(&[0; LEN - 1]), Err(Error::Truncated));
    }
}
