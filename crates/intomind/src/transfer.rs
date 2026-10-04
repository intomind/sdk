//! Carrying an image or a head to a device, without doing any input or
//! output.
//!
//! The same three things go over this path: an application image, the
//! model weights, and a head. The first two are ours, signed and
//! encrypted, and the device checks the signature against the key in its
//! own flash. A head is the user's, carries its own hash, and is checked
//! for that hash and for its shape. Nothing here can forge any of them,
//! which is the point: a host carries an update without being able to
//! read or write one.
//!
//! Nothing is activated by a transfer. The image lands in the idle slot
//! and is verified there, and putting it in force is a separate act.
//!
//! This is a state machine, like the rest of the crate. Ask it what to do,
//! do that, and hand back what came of it.
//!
//! ```no_run
//! # use intomind::{Session, Transfer, Step};
//! # fn write(_: &intomind::Command) {}
//! # fn write_no_response(_: u16, _: &[u8]) {}
//! # fn next_answer() -> Vec<u8> { Vec::new() }
//! # fn run(session: &Session, image: &[u8]) -> Result<(), intomind::Error> {
//! let mut transfer = Transfer::app(session, 1, image)?;
//! loop {
//!     match transfer.step(image) {
//!         Step::Send(command) => {
//!             write(&command);
//!             transfer.on_answer(&next_answer(), image)?;
//!         }
//!         Step::Data { characteristic, from, to } => {
//!             write_no_response(characteristic, &image[from..to]);
//!             transfer.sent(to - from);
//!         }
//!         Step::Verified(_result) => break,
//!     }
//! }
//! # Ok(()) }
//! ```

use intomind_protocol as p;
use p::crc32::crc32;
use p::update::{self, Op, Start, Status, Target, VerifyResult};
use p::uuid_fill;

use crate::session::Command;
use crate::{Error, Session};

/// How many writes go out before the device is asked what it has. Often
/// enough that a bad write is caught near where it happened, rarely
/// enough that the check is not the cost of the transfer.
const WRITES_PER_CHECK: u32 = 32;

/// The largest write this contract allows, whatever a device claims.
const CHUNK_CEILING: usize = 244;

/// What a transfer needs done next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    /// Write this and hand the answer to `on_answer`.
    Send(Command),
    /// Write `image[from..to]` to this characteristic without a response,
    /// then call `sent` with how many bytes went out.
    Data {
        /// The sixteen bit fill of the characteristic to write to.
        characteristic: u16,
        /// Where this write starts in the image.
        from: usize,
        /// Where it ends.
        to: usize,
    },
    /// The device has the image and has checked it. Nothing was activated.
    Verified(VerifyResult),
}

/// Why a transfer stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferError {
    /// The device refused a request, with the status it gave.
    Refused {
        /// What was asked.
        op: Op,
        /// What the device said about it.
        status: Status,
    },
    /// The device has a different number of bytes than were sent. The
    /// transfer stops here rather than at the end, so the disagreement is
    /// reported where it happened.
    Offset {
        /// How many bytes went out.
        sent: usize,
        /// How many the device says it holds.
        device: u32,
    },
    /// The device's checksum over what it holds is not ours over what we
    /// sent, so the bytes arrived changed.
    Checksum {
        /// How many bytes the two sides are talking about.
        sent: usize,
        /// Our checksum over them.
        ours: u32,
        /// The device's.
        device: u32,
    },
    /// The device refused the finished image, and said why.
    Rejected(VerifyResult),
    /// An answer arrived that is not an answer to this transfer.
    Unexpected,
}

impl core::fmt::Display for TransferError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TransferError::Refused { op, status } => write!(f, "the device refused {op:?}: {status:?}"),
            TransferError::Offset { sent, device } => {
                write!(f, "the device accepted {device} bytes where {sent} were sent")
            }
            TransferError::Checksum { sent, ours, device } => {
                write!(f, "over {sent} bytes our checksum is {ours:#010x} and the device's is {device:#010x}")
            }
            TransferError::Rejected(r) => write!(f, "the device refused the image: {r:?}"),
            TransferError::Unexpected => write!(f, "an answer arrived that this transfer did not ask for"),
        }
    }
}

impl std::error::Error for TransferError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Starting,
    Sending,
    Checking,
    Finishing,
    Done(VerifyResult),
}

/// One image, on its way to one device.
#[derive(Debug, Clone)]
pub struct Transfer {
    target: Target,
    slot: u8,
    total: usize,
    id: [u8; 8],
    offset: usize,
    chunk: usize,
    since_check: u32,
    phase: Phase,
}

impl Transfer {
    /// An application image, for the slot the device is not running from.
    pub fn app(session: &Session, slot: u8, image: &[u8]) -> Result<Self, Error> {
        Self::new(session, Target::App, slot, image)
    }

    /// The model weights. There is one copy of them, so no slot.
    pub fn weights(session: &Session, image: &[u8]) -> Result<Self, Error> {
        Self::new(session, Target::Weights, update::SLOT_LINK_NONE, image)
    }

    /// A head, into one of the device's head slots.
    ///
    /// The blob is checked here against the width of the embedding this
    /// device's model produces, so a head trained for a different encoder
    /// is refused before any of it goes on the wire.
    pub fn head(session: &Session, slot: u8, blob: &[u8]) -> Result<Self, Error> {
        if !session.can(p::device_info::capability::HEADS) {
            return Err(Error::NotCapable("heads"));
        }
        let info = session.info()?;
        let (embed_dim, max_outputs) = match info.ext {
            Some(e) => (e.model_embed_dim, e.head_max_outputs),
            None => return Err(Error::NotCapable("heads")),
        };
        p::head::layout(blob, embed_dim, max_outputs).map_err(|_| Error::Wire(p::Error::Invalid))?;
        Self::new(session, Target::Head, slot, blob)
    }

    fn new(session: &Session, target: Target, slot: u8, image: &[u8]) -> Result<Self, Error> {
        if !session.can(p::device_info::capability::UPDATE) {
            return Err(Error::NotCapable("update"));
        }
        if image.is_empty() {
            return Err(Error::Wire(p::Error::Invalid));
        }
        // The device's own limit, never more than the contract allows.
        let chunk = session
            .info()?
            .ext
            .map(|e| e.update_chunk_max as usize)
            .filter(|n| *n > 0)
            .unwrap_or(CHUNK_CEILING)
            .min(CHUNK_CEILING);
        Ok(Transfer {
            target,
            slot,
            total: image.len(),
            id: transfer_id(image),
            offset: 0,
            chunk,
            since_check: 0,
            phase: Phase::Starting,
        })
    }

    /// How much the device has taken, of how much there is.
    pub fn progress(&self) -> (usize, usize) {
        (self.offset, self.total)
    }

    /// What to do next.
    pub fn step(&mut self, image: &[u8]) -> Step {
        match self.phase {
            Phase::Starting => {
                let start = Start {
                    target: self.target,
                    slot: self.slot,
                    total_len: self.total as u32,
                    transfer_id: self.id,
                };
                let mut bytes = [0u8; Start::LEN];
                let n = start.encode(&mut bytes).expect("a start fits its own buffer");
                Step::Send(control(bytes[..n].to_vec()))
            }
            Phase::Checking => Step::Send(control(op_bytes(Op::Query))),
            Phase::Finishing => Step::Send(control(op_bytes(Op::Finish))),
            Phase::Done(result) => Step::Verified(result),
            Phase::Sending => {
                let to = (self.offset + self.chunk).min(image.len());
                Step::Data { characteristic: uuid_fill::UPDATE_DATA, from: self.offset, to }
            }
        }
    }

    /// Say how many bytes went out, after writing what `step` asked for.
    pub fn sent(&mut self, n: usize) {
        if self.phase != Phase::Sending {
            return;
        }
        self.offset = (self.offset + n).min(self.total);
        self.since_check += 1;
        if self.since_check >= WRITES_PER_CHECK || self.offset == self.total {
            self.since_check = 0;
            self.phase = Phase::Checking;
        }
    }

    /// Hand in the answer to whatever `step` last asked to be sent.
    pub fn on_answer(&mut self, bytes: &[u8], image: &[u8]) -> Result<(), TransferError> {
        let answer = update::Response::parse(bytes).map_err(|_| TransferError::Unexpected)?;
        match (self.phase, answer.op) {
            (Phase::Starting, Op::Start) => {
                refuse(answer.op, answer.status)?;
                let start = update::StartResponse::parse(answer.payload)
                    .map_err(|_| TransferError::Unexpected)?;
                // The device may already hold part of this transfer, and
                // says so. Believe it, and check the claim at the next
                // checkpoint like any other.
                self.offset = (start.resume_offset as usize).min(self.total);
                if start.chunk_max > 0 {
                    self.chunk = self.chunk.min(start.chunk_max as usize);
                }
                self.since_check = 0;
                self.phase = if self.offset == self.total { Phase::Checking } else { Phase::Sending };
                Ok(())
            }
            (Phase::Checking, Op::Query) => {
                refuse(answer.op, answer.status)?;
                let q = update::QueryResponse::parse(answer.payload)
                    .map_err(|_| TransferError::Unexpected)?;
                if q.offset as usize != self.offset {
                    return Err(TransferError::Offset { sent: self.offset, device: q.offset });
                }
                let ours = crc32(&image[..self.offset]);
                if q.crc32 != ours {
                    return Err(TransferError::Checksum { sent: self.offset, ours, device: q.crc32 });
                }
                self.phase = if self.offset == self.total { Phase::Finishing } else { Phase::Sending };
                Ok(())
            }
            (Phase::Finishing, Op::Finish) => {
                let result = answer
                    .payload
                    .first()
                    .and_then(|b| VerifyResult::from_u8(*b))
                    .unwrap_or(VerifyResult::Verified);
                if answer.status != Status::Ok {
                    return Err(TransferError::Rejected(result));
                }
                self.phase = Phase::Done(result);
                Ok(())
            }
            _ => Err(TransferError::Unexpected),
        }
    }

    /// Give up. The device drops what it has.
    pub fn abort(&self) -> Command {
        control(op_bytes(Op::Abort))
    }
}

/// Put a transferred image in force.
///
/// The device resets, so the link drops and everything a caller knows
/// about it is stale. An application comes back on trial and rolls itself
/// back if it cannot confirm. Weights are verified at every boot, so
/// restarting is what puts new ones in force.
pub fn activate() -> Command {
    control(op_bytes(Op::Activate))
}

fn control(bytes: Vec<u8>) -> Command {
    Command { characteristic: uuid_fill::UPDATE_CONTROL, bytes, with_response: true }
}

fn op_bytes(op: Op) -> Vec<u8> {
    vec![op as u8]
}

fn refuse(op: Op, status: Status) -> Result<(), TransferError> {
    if status == Status::Ok {
        Ok(())
    } else {
        Err(TransferError::Refused { op, status })
    }
}

/// A transfer names itself by what is being sent, so an interrupted one
/// resumes only against the same bytes and never against different ones
/// that happen to be the same length.
fn transfer_id(image: &[u8]) -> [u8; 8] {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(image);
    let digest = h.finalize();
    let mut id = [0u8; 8];
    id.copy_from_slice(&digest[..8]);
    id
}
