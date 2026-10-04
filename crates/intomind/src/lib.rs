//! The IntoMind instrument client.
//!
//! This crate does no input and no output. It is handed bytes that
//! arrived and it hands back bytes to send, which means it works over
//! whatever Bluetooth stack you already have, on whatever runtime you
//! already use, and can be tested without either.
//!
//! ```
//! use intomind::{protocol::uuid_fill, Event, Session};
//!
//! fn drive(device_info: &[u8], notification: &[u8]) -> Result<(), intomind::Error> {
//!     let mut session = Session::new();
//!     // Whatever your stack read from the device info characteristic.
//!     session.on_device_info(device_info)?;
//!     // Ask it to stream, and write what comes back.
//!     let command = session.start_stream();
//!     assert_eq!(command.characteristic, uuid_fill::CONTROL);
//!     // Hand every notification straight in.
//!     for event in session.on_notification(uuid_fill::EEG_DATA, notification) {
//!         match event {
//!             Event::Samples(batch) => { let _ = batch.rows(); }
//!             // Written down, never smoothed over.
//!             Event::Gap(gap) => { let _ = gap.samples_lost; }
//!             _ => {}
//!         }
//!     }
//!     Ok(())
//! }
//! ```
//!
//! What a device can do is what the device says it can do. Nothing here
//! holds a table of hardware, and a capability is a bit the device
//! reports rather than a version number this crate compares against.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod embeddings;
mod names;
mod session;
mod timebase;
mod transfer;

pub use intomind_pipeline as pipeline;
pub use intomind_protocol as protocol;
pub use names::display_names;
pub use session::{Batch, Command, Event, Gap, Session};
pub use timebase::{Exchange, Fit, Timebase, MAX_SKEW_PPM, MAX_SKEW_STDERR_PPM};

pub use transfer::{activate, Step, Transfer, TransferError};

pub use intomind_capture as capture;
#[cfg(feature = "model")]
pub use intomind_model as model;

/// What went wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Bytes that are not a message, from the codec.
    Wire(protocol::Error),
    /// The device has not said what it is yet, so nothing can be decoded.
    NoDeviceInfo,
    /// The device does not claim the capability this needs.
    NotCapable(&'static str),
    /// A chain the device would refuse at the rate given, and why.
    Chain(intomind_pipeline::Refusal),
    /// A transfer stopped, and the transfer says why.
    Transfer(transfer::TransferError),
}

impl From<protocol::Error> for Error {
    fn from(e: protocol::Error) -> Error {
        Error::Wire(e)
    }
}

impl From<transfer::TransferError> for Error {
    fn from(e: transfer::TransferError) -> Error {
        Error::Transfer(e)
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::Wire(e) => write!(f, "not a message: {e:?}"),
            Error::NoDeviceInfo => write!(f, "read device info first, so nothing has to be assumed"),
            Error::NotCapable(c) => write!(f, "this device does not claim the {c} capability"),
            Error::Chain(r) => write!(f, "the device would refuse this chain at that rate: {r:?}"),
            Error::Transfer(e) => write!(f, "the transfer stopped: {e}"),
        }
    }
}

impl std::error::Error for Error {}
