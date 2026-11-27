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
//! A write is never larger than the device's own limit or this contract's,
//! 244 bytes, and a link may carry less: a host writes at most three bytes
//! less than the link's size at a time. Whoever drives a transfer tells it
//! the largest write the system reports for the link, where the system
//! reports one above 20, with `limit_writes`; a figure of 20 or less is the
//! size every link starts at, and says nothing. Where the system gives no
//! figure the transfer starts at 244. A data write the system refuses while
//! the link is up is told with `write_refused`. Every progress check compares
//! the device's checksum over the bytes it holds with ours over as many of
//! the file's first bytes, whatever its count, and a count behind what was
//! sent whose checksum matches is the start of the file with writes lost on
//! the way, which the transfer carries on from. When a write is refused, or a
//! progress check shows a checksum that does not match, a count not moving,
//! or a device that failed the transfer at its first 32 bytes with a checksum
//! that is not ours over the file up to the end of the write that carried its
//! 32nd byte, the transfer starts the file again
//! at the next smaller of 101 and 20, each held to the device's limit: it
//! aborts what the device holds, starts again, and sends from the count the
//! device's answer to the start gives, which after an abort the device
//! handled is the first byte, and its progress is that count. A link that
//! goes down during that abort ends the transfer in the error that stopped
//! the larger size, the write's own or, where a progress check found it,
//! `smaller_because`. When 20 fails too, it ends in the error it ends in on
//! any link. A later transfer on the same
//! connection starts at the size the last one finished at, `largest_write`,
//! given to it with `limit_writes`, and a new connection starts again from
//! the system's figure. On a link that carries 244 nothing of this asks or
//! writes anything more.
//!
//! ```no_run
//! # use intomind::{Session, Transfer, Step};
//! # fn write(_: &intomind::Command) {}
//! # fn write_no_response(_: u16, _: &[u8]) -> Result<(), std::io::Error> { Ok(()) }
//! # fn next_answer() -> Option<Vec<u8>> { None }
//! # fn run(session: &Session, image: &[u8], largest: Option<usize>, link_up: impl Fn() -> bool) -> Result<usize, Box<dyn std::error::Error>> {
//! let mut transfer = Transfer::app(session, 0, image)?;
//! // The largest write the system reports for the link, where above 20.
//! if let Some(n) = largest.filter(|n| *n > 20) {
//!     transfer.limit_writes(n);
//! }
//! // A write the system refused, which made the file start again at a
//! // smaller size, until the abort that starts it again is done.
//! let mut refused: Option<std::io::Error> = None;
//! loop {
//!     match transfer.step(image) {
//!         Step::Send(command) => {
//!             // What stopped the larger size, while the abort that starts
//!             // the file again is under way.
//!             let stopped: Option<Box<dyn std::error::Error>> =
//!                 refused.take().map(Into::into).or(transfer.smaller_because().map(Into::into));
//!             write(&command);
//!             // None when the device turned the request away, or did not
//!             // answer it in time.
//!             match next_answer() {
//!                 // A link that went down meanwhile ends the transfer in it.
//!                 _ if stopped.is_some() && !link_up() => return Err(stopped.unwrap()),
//!                 Some(answer) => transfer.on_answer(&answer, image)?,
//!                 None => match transfer.unanswered() {
//!                     Some(pause) => std::thread::sleep(pause),
//!                     None => return Err("the device did not answer".into()),
//!                 },
//!             }
//!         }
//!         Step::Data { characteristic, from, to } => match write_no_response(characteristic, &image[from..to]) {
//!             Ok(()) => transfer.sent(to - from),
//!             // A smaller size remains, and the file starts again at it.
//!             Err(e) if link_up() && transfer.write_refused() => refused = Some(e),
//!             Err(e) => return Err(e.into()),
//!         },
//!         Step::Verified(_result) => break,
//!     }
//! }
//! // What a later transfer on the same connection is held to.
//! Ok(transfer.largest_write())
//! # }
//! ```

use intomind_protocol as p;
use p::crc32::crc32;
use p::update::{self, Op, Start, State, Status, Target, VerifyResult};
use p::uuid_fill;

use crate::session::Command;
use crate::{Error, Session};

/// How many writes go out before the device is asked what it has, as the
/// Python library asks. Writes without a response have no flow control: a
/// device writing a page of its flash holds eight incoming writes and
/// throws away what arrives beyond them, a progress check among them.
const WRITES_PER_CHECK: u32 = 8;

/// How many times a transfer carries on from the device's own count before
/// it gives up, as the library's.
const RESUMES: u32 = 64;

/// How many times a progress check is asked in all, when the device turns
/// it away or does not answer it, as the library asks it.
const CHECK_ATTEMPTS: u32 = 3;

/// The largest write this contract allows, whatever a device claims.
const CHUNK_CEILING: usize = 244;

/// The sizes a transfer starts again at, in turn, when its writes do not
/// arrive whole: 101 fits the link of 104 some computers report, and 20
/// fits every link.
const SMALLER: [usize; 2] = [101, 20];

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
    /// The device has a different number of bytes than were sent, and its
    /// count is not advancing: carrying on from it did not move it, or did
    /// so more times than a transfer allows. The transfer stops here rather
    /// than at the end, so the disagreement is reported where it happened.
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
    /// The device refused the finished transfer, a head or an image, and
    /// said why. A transfer that a progress check says has failed is
    /// finished at once, so the device says why, and it ends here, or as
    /// `Refused` with the device's status when the device's answer to that
    /// finish gives no verdict, a verdict the contract does not define, or a
    /// verdict that says verified. One failed at its first 32 bytes with a
    /// checksum that is not ours over the writes that carried them starts
    /// the file again at the next smaller size instead, where one remains.
    Rejected(VerifyResult),
    /// An answer arrived that is not an answer to this transfer.
    Unexpected,
}

impl core::fmt::Display for TransferError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TransferError::Refused { op, status } => write!(f, "the device refused {}: {}", asked(*op), status_words(*status)),
            TransferError::Offset { sent, device } => {
                write!(f, "the device accepted {device} bytes where {sent} were sent, and the transfer is not advancing")
            }
            TransferError::Checksum { sent, ours, device } => {
                write!(f, "the device's checksum over {sent} bytes is {device:#010x} and ours is {ours:#010x}")
            }
            TransferError::Rejected(r) => write!(f, "the device refused the transfer: {}", verdict_words(*r)),
            TransferError::Unexpected => write!(f, "an answer arrived that this transfer did not ask for"),
        }
    }
}

impl std::error::Error for TransferError {}

/// What a refused request asked for, in the Python library's words for its
/// refusal: to start the transfer, a progress check, and to activate the
/// image. A refused finish names the transfer, as a rejected one does, since
/// this error does not say whether the transfer carried an image or a head.
/// The library never reports a refused abort, which is named as the others
/// are.
fn asked(op: Op) -> &'static str {
    match op {
        Op::Start => "to start the transfer",
        Op::Query => "a progress check",
        Op::Finish => "the transfer",
        Op::Activate => "to activate the image",
        Op::Abort => "to abort the transfer",
    }
}

/// The device's verdict on a finished transfer in the Python library's
/// words, as its `VERIFY_RESULTS` names each.
fn verdict_words(result: VerifyResult) -> &'static str {
    match result {
        VerifyResult::Verified => "verified",
        VerifyResult::Length => "length",
        VerifyResult::Malformed => "malformed",
        VerifyResult::Target => "target",
        VerifyResult::Slot => "slot",
        VerifyResult::Hash => "hash",
        VerifyResult::Signature => "signature",
        VerifyResult::SecurityCounter => "security counter",
        VerifyResult::HeadShape => "head shape",
        VerifyResult::Flash => "flash",
        VerifyResult::KeyId => "key id",
    }
}

/// The device's status in the Python library's words.
fn status_words(status: Status) -> &'static str {
    match status {
        Status::Ok => "ok",
        Status::InvalidArg => "invalid argument",
        Status::Unsupported => "unsupported",
        Status::Busy => "busy",
        Status::NoTransfer => "no transfer",
        Status::Flash => "flash",
        Status::Verify => "verification failed",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Starting,
    Sending,
    Checking,
    Finishing,
    Aborting,
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
    /// Times this transfer carried on from the device's count, and the
    /// count it last carried on from.
    resumes: u32,
    resumed_at: Option<usize>,
    /// Times the progress check under way went unanswered.
    unanswered: u32,
    /// This is true once a progress check has said the transfer failed. The
    /// answer to the finish that follows is then a refusal, whatever its
    /// status.
    failed: bool,
    /// The end of the last write sent at this size that carried the file's
    /// 32nd byte, or `None` before one: the device judges an image's
    /// envelope when its 32 bytes are in, and its checksum then covers every
    /// byte of the writes it took.
    enveloped: Option<usize>,
    /// What a progress check found that made the transfer start the file
    /// again at a smaller size, while that abort is under way.
    smaller_because: Option<TransferError>,
    phase: Phase,
}

impl Transfer {
    /// An application image. The device takes it into the slot it does
    /// not run from, and its answer to the start says which, in
    /// `protocol::update::StartResponse::target_slot`. The image must be
    /// built for that slot: the device's check of what arrived refuses one
    /// built for the other. The start names slot zero, as the contract has
    /// it for everything but a head, so `slot` must be zero, and any other
    /// is refused as invalid before anything is sent.
    pub fn app(session: &Session, slot: u8, image: &[u8]) -> Result<Self, Error> {
        if slot != 0 {
            return Err(Error::Wire(p::Error::Invalid));
        }
        Self::new(session, Target::App, 0, image)
    }

    /// The model weights. There is one copy of them, so the start names
    /// slot zero, as the contract has it for everything but a head.
    pub fn weights(session: &Session, image: &[u8]) -> Result<Self, Error> {
        Self::new(session, Target::Weights, 0, image)
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
            resumes: 0,
            resumed_at: None,
            unanswered: 0,
            failed: false,
            enveloped: None,
            smaller_because: None,
            phase: Phase::Starting,
        })
    }

    /// How much the device has taken, of how much there is. Where the
    /// transfer starts the file again at a smaller size, it is what the
    /// device's answer to the start then gives, which after an abort the
    /// device handled is the start.
    ///
    /// This gives a figure after every answer, and the figures a program
    /// shows as the transfer's progress are those the Python library
    /// reports: after each progress check that finds the device holding
    /// what was sent, where the transfer starts the file again at a
    /// smaller size, after the device's answer to the start, the count the
    /// data is then carried from, and after the first start where its
    /// answer says the device already holds the whole file, before the
    /// verdict is asked for. The figure after any other answer is not
    /// progress to show: after any other first start, a progress check
    /// whose count or checksum does not match what was sent or that says
    /// the transfer failed, an abort, and the finish.
    pub fn progress(&self) -> (usize, usize) {
        (self.offset, self.total)
    }

    /// Hold every write to at most `n` bytes: the largest write the system
    /// reports for this link, where it reports one above 20, or the size the
    /// last transfer on the same connection finished at, its
    /// `largest_write`. Given before the transfer starts. A write is never
    /// larger than the device's own limit or this contract's either, and
    /// where a write does not arrive whole the transfer starts again at the
    /// next smaller of 101 and 20.
    ///
    /// ```no_run
    /// # fn run(session: &intomind::Session, image: &[u8], largest: usize) -> Result<(), intomind::Error> {
    /// let mut transfer = intomind::Transfer::app(session, 0, image)?;
    /// transfer.limit_writes(largest);
    /// # Ok(()) }
    /// ```
    pub fn limit_writes(&mut self, n: usize) {
        self.chunk = self.chunk.min(n.max(1));
    }

    /// The largest write this transfer makes now, which a later transfer on
    /// the same connection is held to with `limit_writes`.
    pub fn largest_write(&self) -> usize {
        self.chunk
    }

    /// Say that the system refused the data write `step` last asked for,
    /// while the link is up, as some systems refuse a write longer than the
    /// link carries. Where a smaller size remains, the transfer starts the
    /// file again at it and this gives true: `step` asks for an abort, then
    /// a start, and the data from the count the device's answer to the
    /// start gives. Where none remains it gives false and nothing changes,
    /// and the transfer ends in the write's own error. A link that goes down
    /// during that abort ends the transfer in the write's own error too.
    pub fn write_refused(&mut self) -> bool {
        self.phase == Phase::Sending && self.smaller(None)
    }

    /// What a progress check found that made the transfer start the file
    /// again at a smaller size, while the abort that starts it again is under
    /// way, and `None` otherwise: a count not advancing or a checksum over
    /// the device's count not matching, as it ends where no smaller size
    /// remains, or, for a device that failed the transfer at its first 32
    /// bytes, a checksum not matching over the file up to the end of the
    /// write that carried its 32nd byte. A link that goes down during that
    /// abort ends the transfer in this error, as the Python library's
    /// transfer ends.
    pub fn smaller_because(&self) -> Option<TransferError> {
        if self.phase == Phase::Aborting {
            self.smaller_because
        } else {
            None
        }
    }

    /// Go to the next smaller size, where one remains: the file starts
    /// again, with an abort, and this gives true, `why` kept as what
    /// stopped the larger size. Where none remains this gives false and
    /// nothing changes.
    fn smaller(&mut self, why: Option<TransferError>) -> bool {
        let Some(next) = SMALLER.into_iter().find(|size| *size < self.chunk) else {
            return false;
        };
        self.chunk = next;
        self.smaller_because = why;
        self.since_check = 0;
        self.resumes = 0;
        self.resumed_at = None;
        self.unanswered = 0;
        self.failed = false;
        self.enveloped = None;
        self.phase = Phase::Aborting;
        true
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
            Phase::Aborting => Step::Send(control(op_bytes(Op::Abort))),
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
        let end = (self.offset + n).min(self.total);
        if self.offset < update::ENVELOPE_LEN && update::ENVELOPE_LEN <= end {
            self.enveloped = Some(end);
        }
        self.offset = end;
        self.since_check += 1;
        if self.since_check >= WRITES_PER_CHECK || self.offset == self.total {
            self.since_check = 0;
            self.unanswered = 0;
            self.phase = Phase::Checking;
        }
    }

    /// Say that the device turned away the request `step` last asked to be
    /// sent, or did not answer it. A progress check is asked again, as the
    /// Python library asks it: three times in all, half a second after the
    /// first goes unanswered and a second after the second. Gives how long
    /// to wait before `step` asks for it again, or `None` when it is not
    /// asked again and the transfer stops. Nothing else is asked again. The
    /// library waits five seconds for the answer to a check. The abort of a
    /// transfer starting the file again at a smaller size is passed over:
    /// this gives no wait, and `step` asks for a start.
    pub fn unanswered(&mut self) -> Option<std::time::Duration> {
        if self.phase == Phase::Aborting {
            self.smaller_because = None;
            self.phase = Phase::Starting;
            return Some(std::time::Duration::ZERO);
        }
        if self.phase != Phase::Checking {
            return None;
        }
        self.unanswered += 1;
        (self.unanswered < CHECK_ATTEMPTS).then(|| std::time::Duration::from_millis(500 * u64::from(self.unanswered)))
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
                self.unanswered = 0;
                self.phase = if self.offset == self.total { Phase::Checking } else { Phase::Sending };
                Ok(())
            }
            (Phase::Checking, Op::Query) => {
                refuse(answer.op, answer.status)?;
                let q = update::QueryResponse::parse(answer.payload)
                    .map_err(|_| TransferError::Unexpected)?;
                if q.state == State::Failed {
                    // The device stopped taking this transfer, as it does at
                    // the first bytes of an image built for its other slot,
                    // and holds its verdict. Failed at its 32nd byte, where
                    // it judges an image's envelope, with a checksum that is
                    // not ours over the writes that carried that byte, it
                    // judged an envelope that is not the one sent: the writes
                    // did not arrive whole, and the file starts again at a
                    // smaller size, where one remains. Otherwise its answer
                    // to a finish gives its verdict, and the transfer stops
                    // there with the device's reason.
                    if let Some(end) = self.enveloped.filter(|_| q.offset as usize == update::ENVELOPE_LEN) {
                        let ours = crc32(&image[..end.min(image.len())]);
                        if q.crc32 != ours && self.smaller(Some(TransferError::Checksum { sent: end, ours, device: q.crc32 })) {
                            return Ok(());
                        }
                    }
                    self.failed = true;
                    self.phase = Phase::Finishing;
                    return Ok(());
                }
                let device = q.offset as usize;
                // The device's checksum is over the bytes it holds, so a
                // count behind what was sent is checked as well: one whose
                // checksum does not match holds bytes that are not the start
                // of the file, as a link that cuts writes leaves it, and the
                // file starts again at a smaller size, where one remains.
                let ours = crc32(&image[..device.min(image.len())]);
                if q.crc32 != ours {
                    let changed = TransferError::Checksum { sent: device, ours, device: q.crc32 };
                    if self.smaller(Some(changed)) {
                        return Ok(());
                    }
                    return Err(changed);
                }
                if device != self.offset {
                    // A device that threw writes away holds fewer bytes
                    // than were sent, and they are the file's first bytes.
                    // The transfer carries on from the device's own count,
                    // which is what a resume is for, as the Python library
                    // does, and gives up when that count stops advancing.
                    self.resumes += 1;
                    if self.resumes > RESUMES || self.resumed_at.is_some_and(|at| device <= at) {
                        // Writes that do not arrive whole on a link smaller
                        // than they are look so: the file starts again at a
                        // smaller size, where one remains.
                        let stalled = TransferError::Offset { sent: self.offset, device: q.offset };
                        if self.smaller(Some(stalled)) {
                            return Ok(());
                        }
                        return Err(stalled);
                    }
                    self.resumed_at = Some(device);
                    self.offset = device;
                    self.phase = if self.offset < self.total { Phase::Sending } else { Phase::Finishing };
                    return Ok(());
                }
                self.phase = if self.offset == self.total { Phase::Finishing } else { Phase::Sending };
                Ok(())
            }
            (Phase::Aborting, Op::Abort) => {
                // Whatever the device answered, a refusal among them, the
                // file starts again, from the count the device's answer to
                // the start gives.
                self.smaller_because = None;
                self.phase = Phase::Starting;
                Ok(())
            }
            (Phase::Finishing, Op::Finish) => {
                let result = answer.payload.first().and_then(|b| VerifyResult::from_u8(*b));
                // A transfer that a progress check said had failed is never
                // verified: the answer to its finish ends it as a refused
                // finish ends it, whatever the answer's status.
                if answer.status != Status::Ok || self.failed {
                    // A refusal that says why the transfer was refused is a
                    // rejected transfer. One that gives no reason, a number the
                    // contract does not define, or a verdict that says
                    // verified is a refused request, with the status the
                    // device gave. A device asked to finish again after a
                    // transfer that verified refuses so: no transfer is left
                    // to finish, and it still holds the verdict.
                    return Err(match result {
                        Some(result) if result != VerifyResult::Verified => TransferError::Rejected(result),
                        _ => TransferError::Refused { op: Op::Finish, status: answer.status },
                    });
                }
                self.phase = Phase::Done(result.unwrap_or(VerifyResult::Verified));
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
