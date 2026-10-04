//! The IntoMind BLE protocol, version 1.4: the wire format as code.
//!
//! This crate is the executable form of the contract in
//! the protocol specification. The wire formats of v0.1 are unchanged in
//! v1.0; 1.1 only adds the processing chain, the model's input, the bias
//! drive, two capability bits, and two bytes that were reserved; 1.2 only
//! adds the status lamp, the read-only converter registers, embeddings
//! over the air, three capability bits, and one byte appended to the model
//! info; 1.3 adds the model's cadence, the device's name, the synthetic
//! signal, the second capability word, five bytes appended to the model
//! info, and the encoder identity a head carries; 1.4 adds one control
//! status, refused while USB power is present. Both the device firmware
//! and the host tooling build this same crate, so an encode on one side and
//! a decode on the other are the same code and cannot drift apart.
//!
//! `no_std`, no dependencies, no allocation. Every message parses from a
//! borrowed byte slice and encodes into a caller-provided buffer. Nothing
//! here panics on wire input.

#![cfg_attr(not(test), no_std)]
#![deny(unsafe_code)]

pub mod control;
pub mod crc32;
pub mod device_info;
pub mod embeddings;
pub mod frame;
pub mod head;
pub mod pipeline;
pub mod predictions;
pub mod status;
pub mod update;

/// Protocol major.minor spoken by this codec.
pub const PROTOCOL_VERSION: (u8, u8) = (1, 4);

/// 128-bit UUID base: `f3a1xxxx-2c4b-4d1e-9a6f-1b2c3d4e5f60`.
/// The sixteen-bit fill for each characteristic.
pub mod uuid_fill {
    pub const SERVICE: u16 = 0x0001;
    pub const DEVICE_INFO: u16 = 0x0002;
    pub const CONTROL: u16 = 0x0003;
    pub const CONTROL_RSP: u16 = 0x0004;
    pub const EEG_DATA: u16 = 0x0005;
    pub const STATUS: u16 = 0x0006;
    /// Held by an earlier development firmware. Never reused.
    pub const RESERVED_0007: u16 = 0x0007;
    pub const UPDATE_CONTROL: u16 = 0x0008;
    pub const UPDATE_DATA: u16 = 0x0009;
    pub const PREDICTIONS: u16 = 0x000A;
    /// 1.2: the encoder's output for each window.
    pub const EMBEDDINGS: u16 = 0x000B;
}

/// Build the full 128-bit UUID for a fill, big-endian byte order as GATT
/// tables want it.
pub fn uuid128(fill: u16) -> [u8; 16] {
    let mut u = [
        0xf3, 0xa1, 0x00, 0x00, 0x2c, 0x4b, 0x4d, 0x1e, 0x9a, 0x6f, 0x1b, 0x2c, 0x3d, 0x4e,
        0x5f, 0x60,
    ];
    u[2] = (fill >> 8) as u8;
    u[3] = fill as u8;
    u
}

/// Errors a decoder can return.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Buffer shorter than the fixed part the message requires.
    Truncated,
    /// A field holds a value the contract does not define, or the length
    /// does not match the fields.
    Invalid,
    /// Caller's output buffer is too small for the encoding.
    NoRoom,
    /// An opcode number the contract reserves. The device answers
    /// Unsupported, never treats it as malformed.
    Reserved,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuid_matches_the_documented_base() {
        // The spec writes the base as f3a1xxxx-2c4b-4d1e-9a6f-1b2c3d4e5f60.
        let u = uuid128(0x0005);
        assert_eq!(u[0], 0xf3);
        assert_eq!(u[1], 0xa1);
        assert_eq!(u[2], 0x00);
        assert_eq!(u[3], 0x05);
        assert_eq!(u[4..], [0x2c, 0x4b, 0x4d, 0x1e, 0x9a, 0x6f, 0x1b, 0x2c, 0x3d, 0x4e, 0x5f, 0x60]);
    }
}
