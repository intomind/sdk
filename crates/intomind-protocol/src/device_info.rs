//! Device Info (read): identity, capabilities, capacities.
//!
//! The first 27 bytes are the v0.1 layout, unchanged. Byte 27 is
//! `info_len`, and a 1.0 device reports 76. Hosts read fields up to
//! `info_len` and never past it, so later minors may append.

use crate::Error;

/// Length of the v0.1 prefix.
pub const V01_LEN: usize = 27;
/// Length of the full 1.0 layout.
pub const LEN_1_0: usize = 76;
/// Length of the 1.3 layout: the 1.0 layout with a second capability word
/// appended at offset 76, holding bits 16 and up.
pub const LEN: usize = 80;

/// Bits of the `capabilities` field.
pub mod capability {
    /// `Status::battery_percent` and GET_BATTERY carry measured values.
    pub const BATTERY_VOLTAGE: u16 = 1 << 0;
    /// The hardware has a low battery indicator.
    pub const BATTERY_LOW_FLAG: u16 = 1 << 1;
    /// Lead-off detection is available.
    pub const LEADOFF: u16 = 1 << 2;
    /// The internal test signal mode is available.
    pub const TEST_SIGNAL: u16 = 1 << 3;
    /// Reserved for converter power mode selection.
    pub const DCDC_MODE: u16 = 1 << 4;
    /// Input short mode is available.
    pub const INPUT_SHORT: u16 = 1 << 5;
    /// The Update service is present and accepts application and weights images.
    pub const UPDATE: u16 = 1 << 6;
    /// A model runtime is present in this firmware.
    pub const MODEL: u16 = 1 << 7;
    /// Valid weights were loaded at the time of this read, so predictions can be enabled.
    pub const MODEL_READY: u16 = 1 << 8;
    /// User head slots are present, and head operations are accepted.
    pub const HEADS: u16 = 1 << 9;
    /// 1.1: the device runs a processing chain on its signal and offers
    /// the pipeline operations.
    pub const PIPELINE: u16 = 1 << 10;
    /// 1.1: the device has a bias drive a host may set and read.
    pub const BIAS_DRIVE: u16 = 1 << 11;
    /// 1.2: the device sends its encoder's output for each window on
    /// request, as the window embedding, the tokens, or both.
    pub const EMBEDDINGS: u16 = 1 << 12;
    /// 1.2: the device has a status lamp whose level a host reads and
    /// sets, and that identifies itself on request.
    pub const INDICATOR: u16 = 1 << 13;
    /// 1.2: the device lets a host read its converter's registers.
    pub const CONVERTER_REGISTERS: u16 = 1 << 14;
    /// 1.3: the model runs with no rest of its own and the host sets the
    /// interval between described windows.
    pub const MODEL_CADENCE: u16 = 1 << 15;
}

/// Capability bits 16 and up, in the second word at offset 76. Bit 16 of
/// the contract is bit 0 here.
pub mod capability_high {
    /// 1.3: the device generates a synthetic signal with the converter off.
    pub const SYNTHETIC: u32 = 1 << 0;
    /// 1.3: the device composes its name from a name and an adjective a
    /// host sets, and keeps them.
    pub const DEVICE_NAME: u32 = 1 << 1;
}

/// Bits of `supported_rates`.
pub mod rate_bit {
    /// 250 samples a second.
    pub const SPS_250: u8 = 1 << 0;
    /// 500 samples a second.
    pub const SPS_500: u8 = 1 << 1;
    /// 1000 samples a second.
    pub const SPS_1000: u8 = 1 << 2;
}

/// The 1.0 extension, bytes 28 to 75.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extension {
    /// Hardware revision.
    pub hw_version: (u8, u8, u8),
    /// Identifies the exact firmware image.
    pub fw_build_id: [u8; 8],
    /// 0 when this firmware has no model runtime.
    pub model_embed_dim: u16,
    /// The model's native rate, in samples per second.
    pub model_native_sps: u16,
    /// Samples per prediction window at the native rate.
    pub model_window_samples: u16,
    /// User head slots.
    pub head_slots: u8,
    /// Largest `out_dim` a head may have.
    pub head_max_outputs: u8,
    /// Capacity of one head slot, in bytes.
    pub head_slot_bytes: u16,
    /// Largest Update Data write the device accepts.
    pub update_chunk_max: u16,
    /// Application image capacity, including its header.
    pub app_slot_bytes: u32,
    /// Weights image capacity, including its header.
    pub weights_image_bytes: u32,
}

/// A parsed Device Info read: identity, capabilities, and capacities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Protocol major and minor version the device speaks.
    pub proto_version: (u8, u8),
    /// Firmware major, minor, and patch version.
    pub fw_version: (u8, u8, u8),
    /// Channels. 4 on the IntoMind One.
    pub channel_count: u8,
    /// Converter resolution in bits. 24 on the IntoMind One.
    pub adc_bits: u8,
    /// Device time ticks per second.
    pub time_tick_hz: u32,
    /// The converter's reference voltage, in microvolts.
    pub vref_uv: u32,
    /// Capability bitmask, with its bits in the `capability` module.
    pub capabilities: u16,
    /// Rates this device offers, bits of `rate_bit`.
    pub supported_rates: u8,
    /// Stable per-unit id from the SoC's factory id.
    pub device_id: [u8; 8],
    /// `None` when the device reports only the v0.1 layout.
    pub ext: Option<Extension>,
    /// 1.3: capability bits 16 and up. Zero on a device whose layout ends
    /// at 76 bytes.
    pub capabilities_high: u32,
}

impl DeviceInfo {
    /// Parse a read. Exactly 27 bytes is a v0.x device. Otherwise
    /// `info_len` must be at least 76 and the buffer must hold that much.
    pub fn parse(b: &[u8]) -> Result<Self, Error> {
        if b.len() < V01_LEN {
            return Err(Error::Truncated);
        }
        let mut device_id = [0u8; 8];
        device_id.copy_from_slice(&b[19..27]);
        let mut info = DeviceInfo {
            proto_version: (b[0], b[1]),
            fw_version: (b[2], b[3], b[4]),
            channel_count: b[5],
            adc_bits: b[6],
            time_tick_hz: u32::from_le_bytes([b[7], b[8], b[9], b[10]]),
            vref_uv: u32::from_le_bytes([b[11], b[12], b[13], b[14]]),
            capabilities: u16::from_le_bytes([b[15], b[16]]),
            supported_rates: b[17],
            device_id,
            ext: None,
            capabilities_high: 0,
        };
        if b.len() == V01_LEN {
            return Ok(info);
        }
        let info_len = b[27] as usize;
        if info_len < LEN_1_0 {
            return Err(Error::Invalid);
        }
        if b.len() < LEN_1_0 {
            return Err(Error::Truncated);
        }
        if info_len >= LEN && b.len() >= LEN {
            info.capabilities_high = u32::from_le_bytes([b[76], b[77], b[78], b[79]]);
        }
        let mut fw_build_id = [0u8; 8];
        fw_build_id.copy_from_slice(&b[32..40]);
        info.ext = Some(Extension {
            hw_version: (b[28], b[29], b[30]),
            fw_build_id,
            model_embed_dim: u16::from_le_bytes([b[40], b[41]]),
            model_native_sps: u16::from_le_bytes([b[42], b[43]]),
            model_window_samples: u16::from_le_bytes([b[44], b[45]]),
            head_slots: b[46],
            head_max_outputs: b[47],
            head_slot_bytes: u16::from_le_bytes([b[48], b[49]]),
            update_chunk_max: u16::from_le_bytes([b[50], b[51]]),
            app_slot_bytes: u32::from_le_bytes([b[52], b[53], b[54], b[55]]),
            weights_image_bytes: u32::from_le_bytes([b[56], b[57], b[58], b[59]]),
        });
        Ok(info)
    }

    /// Encode the full 1.0 layout. `ext` must be present on a 1.0 device.
    pub fn encode(&self, out: &mut [u8]) -> Result<usize, Error> {
        let ext = self.ext.ok_or(Error::Invalid)?;
        if out.len() < LEN {
            return Err(Error::NoRoom);
        }
        out[0] = self.proto_version.0;
        out[1] = self.proto_version.1;
        out[2] = self.fw_version.0;
        out[3] = self.fw_version.1;
        out[4] = self.fw_version.2;
        out[5] = self.channel_count;
        out[6] = self.adc_bits;
        out[7..11].copy_from_slice(&self.time_tick_hz.to_le_bytes());
        out[11..15].copy_from_slice(&self.vref_uv.to_le_bytes());
        out[15..17].copy_from_slice(&self.capabilities.to_le_bytes());
        out[17] = self.supported_rates;
        out[18] = 0;
        out[19..27].copy_from_slice(&self.device_id);
        out[27] = LEN as u8;
        out[28] = ext.hw_version.0;
        out[29] = ext.hw_version.1;
        out[30] = ext.hw_version.2;
        out[31] = 0;
        out[32..40].copy_from_slice(&ext.fw_build_id);
        out[40..42].copy_from_slice(&ext.model_embed_dim.to_le_bytes());
        out[42..44].copy_from_slice(&ext.model_native_sps.to_le_bytes());
        out[44..46].copy_from_slice(&ext.model_window_samples.to_le_bytes());
        out[46] = ext.head_slots;
        out[47] = ext.head_max_outputs;
        out[48..50].copy_from_slice(&ext.head_slot_bytes.to_le_bytes());
        out[50..52].copy_from_slice(&ext.update_chunk_max.to_le_bytes());
        out[52..56].copy_from_slice(&ext.app_slot_bytes.to_le_bytes());
        out[56..60].copy_from_slice(&ext.weights_image_bytes.to_le_bytes());
        out[60..76].fill(0);
        out[76..80].copy_from_slice(&self.capabilities_high.to_le_bytes());
        Ok(LEN)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn im1() -> DeviceInfo {
        DeviceInfo {
            proto_version: (1, 0),
            fw_version: (1, 0, 0),
            channel_count: 4,
            adc_bits: 24,
            time_tick_hz: 1_000_000,
            vref_uv: 4_500_000,
            capabilities: capability::BATTERY_VOLTAGE
                | capability::LEADOFF
                | capability::TEST_SIGNAL
                | capability::INPUT_SHORT
                | capability::UPDATE
                | capability::MODEL
                | capability::MODEL_READY
                | capability::HEADS,
            supported_rates: rate_bit::SPS_250 | rate_bit::SPS_500 | rate_bit::SPS_1000,
            device_id: [0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7],
            ext: Some(Extension {
                hw_version: (1, 1, 5),
                fw_build_id: [1, 2, 3, 4, 5, 6, 7, 8],
                model_embed_dim: 96,
                model_native_sps: 500,
                model_window_samples: 2000,
                head_slots: 4,
                head_max_outputs: 32,
                head_slot_bytes: 4096,
                update_chunk_max: 244,
                app_slot_bytes: 237_568,
                weights_image_bytes: 507_904,
            }),
            capabilities_high: capability_high::DEVICE_NAME,
        }
    }

    #[test]
    fn round_trip_full_layout() {
        let d = im1();
        let mut buf = [0u8; LEN];
        assert_eq!(d.encode(&mut buf).unwrap(), 80);
        assert_eq!(buf[27], 80);
        assert_eq!(DeviceInfo::parse(&buf).unwrap(), d);
        // The 1.0 layout is the first 76 bytes, unchanged: a host that reads
        // 76 sees everything it knows and no second capability word.
        let old = DeviceInfo::parse(&buf[..76]).unwrap();
        assert_eq!(old.capabilities_high, 0);
        assert_eq!(DeviceInfo { capabilities_high: 0, ..d }, old);
        // A 1.2 device's 76 byte info reports info_len 76 and parses whole.
        let mut older = buf;
        older[27] = 76;
        assert_eq!(DeviceInfo::parse(&older[..76]).unwrap().capabilities_high, 0);
        assert_eq!(DeviceInfo::parse(&older).unwrap().capabilities_high, 0, "info_len 76 means the second word is not there");
    }

    #[test]
    fn v01_offsets_are_unchanged() {
        // The v0.1 host reads device_id at bytes 19..27 and everything
        // before it with "<BBBBBBBIIHBB". Those offsets are load bearing.
        let d = im1();
        let mut buf = [0u8; LEN];
        d.encode(&mut buf).unwrap();
        assert_eq!(&buf[19..27], &d.device_id);
        assert_eq!(u32::from_le_bytes([buf[7], buf[8], buf[9], buf[10]]), 1_000_000);
        assert_eq!(u32::from_le_bytes([buf[11], buf[12], buf[13], buf[14]]), 4_500_000);
        assert_eq!(buf[17], 0b111);
    }

    #[test]
    fn a_v0_device_parses_without_the_extension() {
        let d = im1();
        let mut buf = [0u8; LEN];
        d.encode(&mut buf).unwrap();
        let old = DeviceInfo::parse(&buf[..V01_LEN]).unwrap();
        assert_eq!(old.ext, None);
        assert_eq!(old.device_id, d.device_id);
    }

    #[test]
    fn lengths_are_policed() {
        let d = im1();
        let mut buf = [0u8; LEN];
        d.encode(&mut buf).unwrap();
        assert_eq!(DeviceInfo::parse(&buf[..26]), Err(Error::Truncated));
        // Claims the extension but does not deliver it.
        assert_eq!(DeviceInfo::parse(&buf[..40]), Err(Error::Truncated));
        // A 1.x device may never report an info_len below 76.
        buf[27] = 60;
        assert_eq!(DeviceInfo::parse(&buf), Err(Error::Invalid));
        // A longer future layout still parses its 1.0 fields.
        buf[27] = 90;
        let mut longer = [0u8; 90];
        longer[..LEN].copy_from_slice(&buf);
        assert_eq!(DeviceInfo::parse(&longer).unwrap().ext, d.ext);
    }
}
