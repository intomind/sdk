//! A transfer, driven end to end against a device that answers the way
//! the contract says a device answers.
//!
//! The device here is a few dozen lines rather than a mock: it holds
//! bytes, counts them, checksums them, and answers. That is enough to
//! test the state machine against.

use intomind::protocol::update::{self, Op, Request, Status, Target, VerifyResult};
use intomind::protocol::{crc32::crc32, device_info, uuid_fill};
use intomind::{Error, Session, Step, Transfer, TransferError};
use std::time::Duration;

/// Bytes in a page of the device's flash, which it writes whole.
const PAGE: usize = 4096;
/// Incoming writes the device holds while it writes a page.
const HOLDS: usize = 8;
/// The transport key a device that judges envelopes holds.
const KEY_ID: u8 = 1;
/// The slot such a device takes an application into, which its answer to
/// the start gives.
const IDLE_SLOT: u8 = 0;

/// The device's verdict on an image's envelope, its first 32 bytes, in the
/// order the firmware checks them, for a transfer of `total` bytes to
/// `target`, on a device that holds the transport key `KEY_ID` and takes an
/// application into `IDLE_SLOT`: `None` when it takes it.
fn judged(envelope: &[u8], target: Target, total: u32) -> Option<VerifyResult> {
    let Ok(e) = update::Envelope::parse(envelope) else {
        return Some(VerifyResult::Malformed);
    };
    if e.target != target {
        return Some(VerifyResult::Target);
    }
    if target == Target::App && e.slot_link != IDLE_SLOT {
        return Some(VerifyResult::Slot);
    }
    if e.key_id != KEY_ID {
        return Some(VerifyResult::KeyId);
    }
    if e.total_len() != total {
        return Some(VerifyResult::Length);
    }
    None
}

/// An application image of `total` bytes whose envelope names `IDLE_SLOT`,
/// the transport key `key_id` and the image's own length, the bytes after
/// it counting through every byte value, as the Python library's tests make
/// one.
fn enveloped(total: usize, key_id: u8) -> Vec<u8> {
    let envelope = update::Envelope {
        target: Target::App,
        slot_link: IDLE_SLOT,
        key_id,
        nonce: core::array::from_fn(|i| i as u8),
        plain_len: (total - update::ENVELOPE_LEN) as u32,
    };
    let mut out = vec![0u8; total];
    envelope.encode(&mut out).unwrap();
    for (i, b) in out.iter_mut().enumerate().skip(update::ENVELOPE_LEN) {
        *b = (i - update::ENVELOPE_LEN) as u8;
    }
    out
}

/// A device that takes an image and says what it has, with a real one's
/// limits: writing a page of its flash takes time, and meanwhile it holds
/// eight incoming writes and throws away what arrives beyond them, a check
/// among them. Time passes only while this side waits for an answer or
/// pauses, so a burst of writes arrives back to back.
///
/// With `judges_envelopes` set, it judges an image's first 32 bytes as its
/// envelope once they are in, from whichever writes carried them, as the
/// firmware does, by `judged`. An envelope it refuses fails the transfer, as
/// an `envelope_verdict` does: its count stays at the 32 bytes, its checksum
/// covers every byte of every write it took, and it throws away every write
/// that follows.
struct Device {
    held: Vec<u8>,
    chunk_max: u16,
    /// Bytes this device already holds when the transfer starts. A real
    /// one holds what it was actually sent before the link dropped.
    already: Vec<u8>,
    /// Answer a query with this instead of the truth, once.
    lie_offset: Option<u32>,
    corrupt_at: Option<usize>,
    verdict: (Status, VerifyResult),
    /// What follows the status in the answer to a finish, in place of the
    /// verdict's own byte: nothing, or a number the contract does not
    /// define.
    reason: Option<Vec<u8>>,
    /// Falls behind at every this many data writes, and throws away every
    /// write until it is next asked what it has, as the library's busy
    /// device does.
    behind_every: usize,
    /// How many checks this device turns away before it answers one.
    turn_away: usize,
    /// The verdict this device gives an image's envelope, its first 32
    /// bytes, when it refuses the image there, as a device does with an
    /// image built for its other slot, or `None` when it takes the envelope.
    envelope_verdict: Option<VerifyResult>,
    /// Whether it judges an image's envelope by what it holds, as the
    /// firmware does.
    judges_envelopes: bool,
    /// What the transfer's start named: its target, and its length.
    target: Target,
    total: u32,
    /// The verdict this device holds once it has stopped taking the
    /// transfer, or `None` while it takes it.
    failed: Option<VerifyResult>,
    /// Every byte its checksum covers once it has stopped: the whole of
    /// each write it took, though it counted no further than the envelope.
    summed: Option<Vec<u8>>,
    /// Whether a finish has verified the transfer. The device holds that
    /// verdict, and takes nothing more of the transfer.
    verified: bool,
    /// Whether it is writing a page, and the writes that arrived meanwhile.
    busy: bool,
    queue: Vec<Vec<u8>>,
    /// Data writes that arrived, checks asked, writes thrown away, and the
    /// most held at once.
    writes: usize,
    checks: usize,
    thrown: usize,
    most_held: usize,
    /// Each pause this side took before asking again.
    waited: Vec<Duration>,
    behind: bool,
}

impl Device {
    fn new() -> Device {
        Device {
            held: Vec::new(),
            chunk_max: 244,
            already: Vec::new(),
            lie_offset: None,
            corrupt_at: None,
            verdict: (Status::Ok, VerifyResult::Verified),
            reason: None,
            behind_every: 0,
            turn_away: 0,
            envelope_verdict: None,
            judges_envelopes: false,
            target: Target::App,
            total: 0,
            failed: None,
            summed: None,
            verified: false,
            busy: false,
            queue: Vec::new(),
            writes: 0,
            checks: 0,
            thrown: 0,
            most_held: 0,
            waited: Vec::new(),
            behind: false,
        }
    }

    /// The answer to one request, or `None` when the device threw the
    /// request away.
    fn control(&mut self, bytes: &[u8]) -> Option<Vec<u8>> {
        if self.busy {
            // A request is one more incoming write. Held, it is answered
            // once what came before it is written, while this side waits.
            if self.queue.len() >= HOLDS {
                self.thrown += 1;
                return None;
            }
            self.catch_up();
        }
        let mut out = [0u8; 32];
        // The device's own parser, so this answers what a device would
        // answer and refuses what a device would refuse.
        let request = Request::parse(bytes).expect("a request this contract defines");
        let n = match request {
            Request::Start(start) => {
                assert!(start.total_len > 0);
                self.target = start.target;
                self.total = start.total_len;
                self.held = self.already.clone();
                self.failed = None;
                self.summed = None;
                self.verified = false;
                let p = update::StartResponse {
                    target_slot: start.slot,
                    chunk_max: self.chunk_max,
                    resume_offset: self.held.len() as u32,
                };
                let mut body = [0u8; update::StartResponse::LEN];
                p.encode(&mut body).unwrap();
                update::encode_response(Op::Start, Status::Ok, &body, &mut out).unwrap()
            }
            Request::Query => {
                self.checks += 1;
                self.behind = false;
                if self.turn_away > 0 {
                    self.turn_away -= 1;
                    return None;
                }
                let offset = self.lie_offset.take().unwrap_or(self.held.len() as u32);
                let q = update::QueryResponse {
                    state: match (self.failed, self.verified) {
                        (Some(_), _) => update::State::Failed,
                        (None, true) => update::State::Complete,
                        (None, false) => update::State::Receiving,
                    },
                    offset,
                    crc32: crc32(self.summed.as_ref().unwrap_or(&self.held)),
                };
                let mut body = [0u8; update::QueryResponse::LEN];
                q.encode(&mut body).unwrap();
                update::encode_response(Op::Query, Status::Ok, &body, &mut out).unwrap()
            }
            Request::Finish => {
                // A transfer that failed, or one that verified, has nothing
                // left to finish: the device answers "no transfer", with the
                // verdict it holds.
                if let Some(held) = self.failed.or(self.verified.then_some(VerifyResult::Verified)) {
                    let n = update::encode_response(Op::Finish, Status::NoTransfer, &[held as u8], &mut out).unwrap();
                    return Some(out[..n].to_vec());
                }
                let (status, result) = self.verdict;
                let reason = self.reason.clone().unwrap_or_else(|| vec![result as u8]);
                self.verified = status == Status::Ok;
                update::encode_response(Op::Finish, status, &reason, &mut out).unwrap()
            }
            Request::Abort => {
                self.held.clear();
                self.failed = None;
                self.summed = None;
                self.verified = false;
                update::encode_response(Op::Abort, Status::Ok, &[], &mut out).unwrap()
            }
            Request::Activate => update::encode_response(Op::Activate, Status::Ok, &[], &mut out).unwrap(),
        };
        Some(out[..n].to_vec())
    }

    /// One write without a response.
    fn data(&mut self, bytes: &[u8]) {
        self.writes += 1;
        if self.behind_every > 0 && self.writes % self.behind_every == 0 {
            self.behind = true;
        }
        if self.behind || (self.busy && self.queue.len() >= HOLDS) {
            self.thrown += 1;
            return;
        }
        if self.busy {
            self.queue.push(bytes.to_vec());
            self.most_held = self.most_held.max(self.queue.len());
            return;
        }
        self.take(bytes);
    }

    /// Time passes: the page is written, and then what arrived meanwhile.
    fn catch_up(&mut self) {
        for bytes in std::mem::take(&mut self.queue) {
            self.take(&bytes);
        }
        self.busy = false;
    }

    fn take(&mut self, bytes: &[u8]) {
        // A device that stopped taking the transfer refuses every write after.
        if self.failed.is_some() || self.verified {
            return;
        }
        let mut bytes = bytes.to_vec();
        if let Some(at) = self.corrupt_at {
            if self.held.len() <= at && at < self.held.len() + bytes.len() {
                let i = at - self.held.len();
                bytes[i] ^= 0x01;
                self.corrupt_at = None;
            }
        }
        if (self.envelope_verdict.is_some() || self.judges_envelopes)
            && self.target != Target::Head
            && self.held.len() < update::ENVELOPE_LEN
            && self.held.len() + bytes.len() >= update::ENVELOPE_LEN
        {
            // The envelope is the 32 bytes held, from whichever writes
            // brought them. One the device refuses stops it there: the
            // write that completes the envelope is checksummed whole, and
            // its envelope bytes are counted.
            let mut summed = self.held.clone();
            summed.extend_from_slice(&bytes);
            let verdict = self.envelope_verdict.or_else(|| judged(&summed[..update::ENVELOPE_LEN], self.target, self.total));
            if verdict.is_some() {
                self.summed = Some(summed);
                let rest = update::ENVELOPE_LEN - self.held.len();
                self.held.extend_from_slice(&bytes[..rest]);
                self.failed = verdict;
                return;
            }
        }
        let before = self.held.len();
        self.held.extend_from_slice(&bytes);
        if before / PAGE != self.held.len() / PAGE {
            self.busy = true;
        }
    }
}

/// A session that has read a device claiming everything this needs.
fn session() -> Session {
    let info = device_info::DeviceInfo {
        proto_version: intomind::protocol::PROTOCOL_VERSION,
        fw_version: (1, 0, 0),
        channel_count: 4,
        adc_bits: 24,
        time_tick_hz: 1_000_000,
        vref_uv: 4_500_000,
        capabilities: device_info::capability::UPDATE
            | device_info::capability::MODEL
            | device_info::capability::HEADS,
        supported_rates: 0b111,
        device_id: [1, 2, 3, 4, 5, 6, 7, 8],
        ext: Some(device_info::Extension {
            hw_version: (1, 1, 5),
            fw_build_id: [0; 8],
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
        capabilities_high: 0,
    };
    let mut buf = [0u8; device_info::LEN];
    info.encode(&mut buf).unwrap();
    let mut s = Session::new();
    s.on_device_info(&buf).unwrap();
    s
}

/// Why a run stopped short.
#[derive(Debug)]
enum Stop {
    /// The transfer said why.
    Transfer(TransferError),
    /// The device did not answer, and the transfer did not ask again.
    Unanswered,
}

/// Drive a transfer to its end against a device, returning what the
/// device said and how many writes it took. A request the device threw
/// away is never answered: the device catches up while this side waits,
/// and the transfer says whether to ask again, and after how long.
fn drive(t: &mut Transfer, image: &[u8], dev: &mut Device) -> Result<(VerifyResult, usize), Stop> {
    let mut writes = 0;
    for _ in 0..100_000 {
        match t.step(image) {
            Step::Send(command) => {
                assert!(command.with_response, "a control write is answered");
                assert_eq!(command.characteristic, uuid_fill::UPDATE_CONTROL);
                match dev.control(&command.bytes) {
                    Some(answer) => t.on_answer(&answer, image).map_err(Stop::Transfer)?,
                    None => {
                        dev.catch_up();
                        dev.waited.push(t.unanswered().ok_or(Stop::Unanswered)?);
                    }
                }
            }
            Step::Data { characteristic, from, to } => {
                assert_eq!(characteristic, uuid_fill::UPDATE_DATA);
                assert!(to > from, "a write carries bytes");
                assert!(to - from <= 244, "no write is larger than the contract allows");
                dev.data(&image[from..to]);
                writes += 1;
                t.sent(to - from);
            }
            Step::Verified(result) => return Ok((result, writes)),
        }
    }
    panic!("the transfer did not finish");
}

/// `drive`, for a device that answers every check in the end.
fn run(t: &mut Transfer, image: &[u8], dev: &mut Device) -> Result<(VerifyResult, usize), TransferError> {
    drive(t, image, dev).map_err(|stop| match stop {
        Stop::Transfer(e) => e,
        Stop::Unanswered => panic!("the device did not answer a check, and the transfer did not ask again"),
    })
}

fn image(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i * 7 + 13) as u8).collect()
}

/// What a stand-in link does with a write longer than it carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Way {
    /// Refuses it, as Linux does and Windows is reported to.
    Refuses,
    /// Drops it with no sign, the write still succeeding, as iOS is
    /// reported to and as a Mac's browser resolves a write whatever happens.
    Drops,
    /// Cuts it to what the link carries with no sign, as Android documents.
    Cuts,
}

const WAYS: [Way; 3] = [Way::Refuses, Way::Drops, Way::Cuts];

/// One thing a stand-in link carried: a request, by its operation, or a
/// data write, by the bytes sent.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Carried {
    Request(Op),
    Write(usize),
}

/// How a drive through a stand-in link ended short.
#[derive(Debug)]
enum Short {
    /// The transfer said why.
    Transfer(TransferError),
    /// The system refused a write, and the transfer did not start again.
    Refused,
}

/// Drive a transfer to its end through a link of `size`, as a system
/// reports its size, which carries a write of at most its size less 3 and
/// treats a longer one its `way`, to `dev`. A data write the link refuses is
/// told to the transfer, and is the end of it where the transfer says it
/// does not start again. Gives the verdict and everything the link carried,
/// in order.
fn over(t: &mut Transfer, image: &[u8], dev: &mut Device, size: usize, way: Way) -> (Result<VerifyResult, Short>, Vec<Carried>) {
    let carries = size - 3;
    let mut trace = Vec::new();
    for _ in 0..100_000 {
        match t.step(image) {
            Step::Send(command) => {
                trace.push(Carried::Request(Op::from_u8(command.bytes[0]).expect("an operation the contract defines")));
                match dev.control(&command.bytes) {
                    Some(answer) => {
                        if let Err(e) = t.on_answer(&answer, image) {
                            return (Err(Short::Transfer(e)), trace);
                        }
                    }
                    None => {
                        dev.catch_up();
                        dev.waited.push(t.unanswered().expect("the device answers in the end"));
                    }
                }
            }
            Step::Data { from, to, .. } => {
                let bytes = &image[from..to];
                trace.push(Carried::Write(bytes.len()));
                if bytes.len() <= carries {
                    dev.data(bytes);
                } else {
                    match way {
                        Way::Refuses => {
                            if t.write_refused() {
                                continue;
                            }
                            return (Err(Short::Refused), trace);
                        }
                        Way::Cuts => dev.data(&bytes[..carries]),
                        Way::Drops => {}
                    }
                }
                t.sent(to - from);
            }
            Step::Verified(result) => return (Ok(result), trace),
        }
    }
    panic!("the transfer did not finish");
}

/// What a transfer of `total` bytes carried before a link's size was asked
/// about, to a device that threw nothing away: a start, writes of 244 bytes
/// with a progress check after every eighth and after the last, and a
/// finish.
fn today(total: usize) -> Vec<Carried> {
    let mut out = vec![Carried::Request(Op::Start)];
    let (mut sent, mut since) = (0, 0);
    while sent < total {
        let n = 244.min(total - sent);
        out.push(Carried::Write(n));
        sent += n;
        since += 1;
        if since == 8 || sent == total {
            out.push(Carried::Request(Op::Query));
            since = 0;
        }
    }
    out.push(Carried::Request(Op::Finish));
    out
}

/// How many aborts `trace` holds.
fn aborts(trace: &[Carried]) -> usize {
    trace.iter().filter(|c| **c == Carried::Request(Op::Abort)).count()
}

/// The writes of `trace` after its last abort, by the bytes sent.
fn writes_after_the_last_abort(trace: &[Carried]) -> Vec<usize> {
    let from = trace.iter().rposition(|c| *c == Carried::Request(Op::Abort)).map_or(0, |i| i + 1);
    trace[from..].iter().filter_map(|c| if let Carried::Write(n) = c { Some(*n) } else { None }).collect()
}

/// The largest write after each start in `trace`, in order.
fn largest_after_each_start(trace: &[Carried]) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for c in trace {
        match c {
            Carried::Request(Op::Start) => out.push(0),
            Carried::Write(n) => {
                let last = out.last_mut().expect("a write after a start");
                *last = (*last).max(*n);
            }
            Carried::Request(_) => {}
        }
    }
    out
}

/// `n` writes of `size` bytes, as a trace lists them.
fn writes_of(n: usize, size: usize) -> Vec<Carried> {
    vec![Carried::Write(size); n]
}

/// What a trace lists for a request of `op`.
fn asked(op: Op) -> Vec<Carried> {
    vec![Carried::Request(op)]
}

#[test]
fn a_link_that_carries_244_bytes_sees_exactly_the_requests_and_writes_a_transfer_made_before_a_links_size_was_asked_about_and_no_wait_whatever_it_does_with_a_longer_write() {
    let s = session();
    // So for a device that judges an image's envelope, too.
    for (img, judges) in [(image(30_000), false), (enveloped(30_000, KEY_ID), true)] {
        for way in WAYS {
            let mut dev = Device::new();
            dev.judges_envelopes = judges;
            let mut t = Transfer::app(&s, 0, &img).unwrap();
            let (result, trace) = over(&mut t, &img, &mut dev, 247, way);
            assert_eq!(result.unwrap(), VerifyResult::Verified, "{way:?}");
            assert_eq!(trace, today(img.len()), "{way:?}");
            assert!(dev.waited.is_empty(), "{way:?}");
            assert_eq!(dev.thrown, 0, "{way:?}");
            assert_eq!(dev.held, img, "{way:?}");
        }
    }
}

#[test]
fn on_a_link_of_104_that_refuses_drops_or_cuts_a_longer_write_a_transfer_starts_the_file_again_at_101_bytes_from_the_first_byte_and_is_verified() {
    let s = session();
    let img = image(30_000);
    for way in WAYS {
        let mut dev = Device::new();
        let mut t = Transfer::app(&s, 0, &img).unwrap();
        let (result, trace) = over(&mut t, &img, &mut dev, 104, way);
        assert_eq!(result.unwrap(), VerifyResult::Verified, "{way:?}");
        assert_eq!(dev.held, img, "{way:?}: the device holds exactly what was sent");
        assert_eq!(trace[1], Carried::Write(244), "{way:?}: the first write is 244 bytes");
        assert_eq!(aborts(&trace), 1, "{way:?}: the file started again once");
        let again = trace.iter().position(|c| *c == Carried::Request(Op::Abort)).unwrap();
        assert_eq!(trace[again + 1], Carried::Request(Op::Start), "{way:?}");
        let after = writes_after_the_last_abort(&trace);
        assert!(after[0] == 101 && after.iter().all(|n| *n <= 101), "{way:?}: then 101 bytes at a time");
        assert_eq!(after.iter().sum::<usize>(), img.len(), "{way:?}: from the first byte, each byte once");
        assert_eq!(t.largest_write(), 101, "{way:?}");
    }
}

#[test]
fn on_a_link_of_23_that_refuses_drops_or_cuts_a_longer_write_a_transfer_starts_the_file_again_at_101_bytes_and_then_at_20_and_is_verified() {
    let s = session();
    let img = image(5000);
    for way in WAYS {
        let mut dev = Device::new();
        let mut t = Transfer::app(&s, 0, &img).unwrap();
        let (result, trace) = over(&mut t, &img, &mut dev, 23, way);
        assert_eq!(result.unwrap(), VerifyResult::Verified, "{way:?}");
        assert_eq!(dev.held, img, "{way:?}");
        assert_eq!(aborts(&trace), 2, "{way:?}: the file started again twice");
        let after = writes_after_the_last_abort(&trace);
        assert!(after[0] == 20 && after.iter().all(|n| *n <= 20), "{way:?}");
        assert_eq!(after.iter().sum::<usize>(), img.len(), "{way:?}");
        assert_eq!(t.largest_write(), 20, "{way:?}");
    }
}

#[test]
fn where_20_bytes_fail_too_a_transfer_ends_in_the_error_and_the_words_it_ends_in_on_any_link() {
    // A link that carries 17 bytes, less than every size a transfer starts
    // again at.
    let s = session();
    let img = image(5000);
    let mut t = Transfer::app(&s, 0, &img).unwrap();
    let (result, trace) = over(&mut t, &img, &mut Device::new(), 20, Way::Refuses);
    assert!(matches!(result, Err(Short::Refused)), "{result:?}");
    assert_eq!((aborts(&trace), t.largest_write(), t.write_refused()), (2, 20, false), "no smaller size remains");
    match over(&mut Transfer::app(&s, 0, &img).unwrap(), &img, &mut Device::new(), 20, Way::Drops).0 {
        Err(Short::Transfer(e @ TransferError::Offset { sent, device })) => {
            assert_eq!(e.to_string(), format!("the device accepted {device} bytes where {sent} were sent, and the transfer is not advancing"));
        }
        other => panic!("{other:?}"),
    }
    match over(&mut Transfer::app(&s, 0, &img).unwrap(), &img, &mut Device::new(), 20, Way::Cuts).0 {
        Err(Short::Transfer(e @ TransferError::Checksum { sent, ours, device })) => {
            assert_ne!(ours, device);
            assert_eq!(e.to_string(), format!("the device's checksum over {sent} bytes is {device:#010x} and ours is {ours:#010x}"));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_transfer_held_to_the_largest_write_a_system_reports_writes_no_more_than_that_and_starts_again_at_the_smaller_of_101_and_20_below_it() {
    let s = session();
    let img = image(5000);
    // A link of 185 whose system says so carries every write.
    let mut t = Transfer::app(&s, 0, &img).unwrap();
    t.limit_writes(182);
    let (result, trace) = over(&mut t, &img, &mut Device::new(), 185, Way::Drops);
    assert_eq!(result.unwrap(), VerifyResult::Verified);
    assert_eq!((trace[1].clone(), aborts(&trace)), (Carried::Write(182), 0), "182 bytes at a time, and nothing started again");
    assert!(trace.iter().all(|c| !matches!(c, Carried::Write(n) if *n > 182)));
    // A size held to that is still too large starts again at 101.
    let mut t = Transfer::app(&s, 0, &img).unwrap();
    t.limit_writes(150);
    let (result, trace) = over(&mut t, &img, &mut Device::new(), 104, Way::Refuses);
    assert_eq!(result.unwrap(), VerifyResult::Verified);
    assert_eq!((trace[1].clone(), aborts(&trace), t.largest_write()), (Carried::Write(150), 1, 101));
    // The size the last transfer finished at, given to the next, starts it
    // there, with only 20 below it.
    let mut t = Transfer::app(&s, 0, &img).unwrap();
    t.limit_writes(101);
    assert_eq!(t.largest_write(), 101);
    let (result, trace) = over(&mut t, &img, &mut Device::new(), 23, Way::Cuts);
    assert_eq!(result.unwrap(), VerifyResult::Verified);
    assert_eq!((trace[1].clone(), aborts(&trace), t.largest_write()), (Carried::Write(101), 1, 20));
    // Never more than the device's own limit or this contract's.
    let mut t = Transfer::app(&s, 0, &img).unwrap();
    t.limit_writes(512);
    assert_eq!(t.largest_write(), 244);
}

#[test]
fn a_transfer_that_starts_the_file_again_carries_the_data_from_the_count_the_devices_answer_to_the_start_gives_its_progress_that_count_and_passes_over_an_abort_turned_away_unanswered_or_refused() {
    let s = session();
    let img = image(5000);
    let to_the_first_write = |t: &mut Transfer, dev: &mut Device| loop {
        match t.step(&img) {
            Step::Send(command) => t.on_answer(&dev.control(&command.bytes).unwrap(), &img).unwrap(),
            Step::Data { .. } => break,
            Step::Verified(result) => panic!("verified before a write: {result:?}"),
        }
    };
    let mut dev = Device::new();
    let mut t = Transfer::app(&s, 0, &img).unwrap();
    to_the_first_write(&mut t, &mut dev);
    dev.data(&img[..244]);
    t.sent(244);
    assert_eq!(t.progress(), (244, 5000));
    assert!(t.write_refused(), "the second write is refused");
    assert_eq!(t.progress(), (244, 5000), "nothing moves until the device answers the start");
    assert_eq!(t.smaller_because(), None, "a refused write is the driver's own error");
    assert!(matches!(t.step(&img), Step::Send(c) if c.bytes == [Op::Abort as u8]), "an abort first");
    assert_eq!(t.unanswered(), Some(Duration::ZERO), "an abort turned away or unanswered is passed over at once");
    let start = match t.step(&img) {
        Step::Send(c) if c.bytes[0] == Op::Start as u8 => c,
        other => panic!("then a start, not {other:?}"),
    };
    // The abort never reached the device, which still holds the first
    // write and says so when asked to start again.
    dev.already = dev.held.clone();
    t.on_answer(&dev.control(&start.bytes).unwrap(), &img).unwrap();
    assert_eq!(t.progress(), (244, 5000), "the count the device's answer to the start gives");
    assert_eq!(t.step(&img), Step::Data { characteristic: uuid_fill::UPDATE_DATA, from: 244, to: 345 }, "the data goes on from there, 101 bytes at a time");
    assert_eq!(run(&mut t, &img, &mut dev).unwrap().0, VerifyResult::Verified);
    assert_eq!(dev.held, img, "each byte once, and none past the file's length");

    // An abort the device refuses, as one with no transfer refuses it.
    let mut u = Transfer::app(&s, 0, &img).unwrap();
    to_the_first_write(&mut u, &mut Device::new());
    assert!(u.write_refused());
    let mut out = [0u8; 8];
    let n = update::encode_response(Op::Abort, Status::NoTransfer, &[], &mut out).unwrap();
    u.on_answer(&out[..n], &img).unwrap();
    let again = match u.step(&img) {
        Step::Send(c) if c.bytes[0] == Op::Start as u8 => c,
        other => panic!("a start after a refused abort, not {other:?}"),
    };
    u.on_answer(&Device::new().control(&again.bytes).unwrap(), &img).unwrap();
    assert_eq!(u.progress(), (0, 5000), "a device that handled the abort starts from the first byte");
    assert_eq!(u.largest_write(), 101);
}

#[test]
fn while_the_abort_that_starts_the_file_again_at_a_smaller_size_is_under_way_the_transfer_gives_what_a_progress_check_found_as_it_ends_where_no_smaller_size_remains() {
    // Every write of 244 bytes is lost on a link of 104, so the device's
    // count stops advancing at the second check.
    let s = session();
    let img = image(5000);
    let mut dev = Device::new();
    let mut t = Transfer::app(&s, 0, &img).unwrap();
    let mut why = None;
    for _ in 0..1000 {
        match t.step(&img) {
            Step::Send(command) => {
                t.on_answer(&dev.control(&command.bytes).unwrap(), &img).unwrap();
                why = t.smaller_because();
                if why.is_some() {
                    break;
                }
            }
            Step::Data { from, to, .. } => t.sent(to - from),
            Step::Verified(result) => panic!("verified: {result:?}"),
        }
    }
    match why {
        Some(e @ TransferError::Offset { sent, device: 0 }) => {
            assert_eq!(e.to_string(), format!("the device accepted 0 bytes where {sent} were sent, and the transfer is not advancing"));
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(t.largest_write(), 101);
    let abort = match t.step(&img) {
        Step::Send(c) if c.bytes == [Op::Abort as u8] => c,
        other => panic!("{other:?}"),
    };
    t.on_answer(&dev.control(&abort.bytes).unwrap(), &img).unwrap();
    assert_eq!(t.smaller_because(), None, "and nothing once the abort is done");
}

#[test]
fn a_link_of_23_that_cuts_writes_ends_in_a_verified_transfer_at_20_bytes_where_the_envelope_the_device_judges_at_its_first_32_bytes_is_made_of_two_writes() {
    // The device gets the first 20 bytes of each write, so the envelope it
    // judges is made of two writes, and it fails the transfer there. Its
    // checksum is then not ours over the first write, so it did not get the
    // bytes that were sent: the file starts again at the next smaller size
    // with no finish, at 101 and then at 20, where the image is verified.
    let s = session();
    let img = enveloped(24 * 256, KEY_ID);
    let mut dev = Device::new();
    dev.judges_envelopes = true;
    let mut t = Transfer::app(&s, 0, &img).unwrap();
    let (result, trace) = over(&mut t, &img, &mut dev, 23, Way::Cuts);
    assert_eq!(result.unwrap(), VerifyResult::Verified);
    assert_eq!(dev.held, img);
    assert_eq!(largest_after_each_start(&trace), [244, 101, 20]);
    let failed = [
        asked(Op::Start),
        writes_of(8, 244),
        asked(Op::Query),
        asked(Op::Abort),
        asked(Op::Start),
        writes_of(8, 101),
        asked(Op::Query),
        asked(Op::Abort),
        asked(Op::Start),
    ]
    .concat();
    assert_eq!(trace[..failed.len()], failed[..]);
    assert_eq!(trace.iter().position(|c| *c == Carried::Request(Op::Finish)), Some(trace.len() - 1), "a finish only at the end");
}

#[test]
fn a_link_of_104_that_cuts_writes_starts_the_file_again_at_101_bytes_at_its_first_progress_check_for_an_image_and_for_a_head() {
    // The device gets the first 101 bytes of each write. At the first check
    // its count is behind what was sent and its checksum is not ours over
    // as many of the file's first bytes, so it holds bytes that are not the
    // start of the file: the file starts again there, rather than carrying
    // on from the device's count. A head has no envelope.
    use intomind::protocol::head;
    let s = session();
    let mut name = [0u8; head::NAME_LEN];
    name[..5].copy_from_slice(b"focus");
    let header = head::HeadHeader { version: head::FORMAT_VERSION_2, kind: head::KIND_LINEAR, in_dim: 96, out_dim: 32, name, encoder_id: [0; 8] };
    let weights: Vec<i8> = (0..96 * 32).map(|i| (i % 7) as i8 - 3).collect();
    let mut blob = vec![0u8; head::blob_len(head::FORMAT_VERSION_2, 96, 32)];
    head::encode_unhashed(&header, &weights, &[0; 32], &[1.0; 32], &mut blob).unwrap();
    assert!(blob.len() > 8 * 244, "the head is longer than eight writes");
    let img = enveloped(24 * 256, KEY_ID);
    for (what, file, mut t) in [
        ("image", &img, Transfer::app(&s, 0, &img).unwrap()),
        ("head", &blob, Transfer::head(&s, 1, &blob).unwrap()),
    ] {
        let mut dev = Device::new();
        dev.judges_envelopes = true;
        let (result, trace) = over(&mut t, file, &mut dev, 104, Way::Cuts);
        assert_eq!(result.unwrap(), VerifyResult::Verified, "{what}");
        assert_eq!(&dev.held, file, "{what}");
        let first = [asked(Op::Start), writes_of(8, 244), asked(Op::Query), asked(Op::Abort), asked(Op::Start)].concat();
        assert_eq!(trace[..first.len()], first[..], "{what}");
        assert_eq!(largest_after_each_start(&trace), [244, 101], "{what}");
    }
}

#[test]
fn an_image_the_device_refuses_at_its_first_32_bytes_on_a_link_that_carries_244_ends_in_its_verdict_through_a_finish_with_no_smaller_size_tried_and_the_next_transfer_starts_at_244() {
    // Its checksum is ours over the first write: the device got the bytes
    // that were sent, and refused the image.
    let s = session();
    let good = enveloped(24 * 256, KEY_ID);
    let mut short = good.clone();
    short[24..28].copy_from_slice(&7u32.to_le_bytes());
    for (bad, verdict) in [(enveloped(24 * 256, 9), VerifyResult::KeyId), (short, VerifyResult::Length)] {
        let mut dev = Device::new();
        dev.judges_envelopes = true;
        let mut t = Transfer::app(&s, 0, &bad).unwrap();
        let (result, trace) = over(&mut t, &bad, &mut dev, 247, Way::Refuses);
        match result {
            Err(Short::Transfer(TransferError::Rejected(given))) => assert_eq!(given, verdict),
            other => panic!("{verdict:?}: {other:?}"),
        }
        assert_eq!(trace, [asked(Op::Start), writes_of(8, 244), asked(Op::Query), asked(Op::Finish)].concat(), "{verdict:?}");
        assert_eq!(t.largest_write(), 244, "{verdict:?}");
        let mut next = Transfer::app(&s, 0, &good).unwrap();
        next.limit_writes(t.largest_write());
        let (then, trace) = over(&mut next, &good, &mut dev, 247, Way::Refuses);
        assert_eq!(then.unwrap(), VerifyResult::Verified, "{verdict:?}");
        assert_eq!(largest_after_each_start(&trace), [244], "{verdict:?}");
    }
}

#[test]
fn at_20_bytes_a_device_holding_bytes_that_are_not_the_start_of_the_file_ends_the_transfer_at_that_check_in_the_words_for_a_checksum_that_does_not_match_over_the_devices_count() {
    // A link of 20 cuts each write to 17 bytes.
    let s = session();
    let img: Vec<u8> = (0..12 * 256).map(|i| i as u8).collect();
    let held: Vec<u8> = (0..8).flat_map(|k| img[20 * k..20 * k + 17].to_vec()).collect();
    let (result, trace) = over(&mut Transfer::app(&s, 0, &img).unwrap(), &img, &mut Device::new(), 20, Way::Cuts);
    match result {
        Err(Short::Transfer(e @ TransferError::Checksum { sent, ours, device })) => {
            assert_eq!((sent, ours, device), (held.len(), crc32(&img[..held.len()]), crc32(&held)));
            assert_eq!(e.to_string(), format!("the device's checksum over {} bytes is {device:#010x} and ours is {ours:#010x}", held.len()));
        }
        other => panic!("{other:?}"),
    }
    let at_20 = trace.iter().rposition(|c| *c == Carried::Request(Op::Start)).unwrap();
    assert_eq!(trace[at_20..], [asked(Op::Start), writes_of(8, 20), asked(Op::Query)].concat()[..]);
}

#[test]
fn an_image_arrives_byte_for_byte() {
    let s = session();
    let img = image(5000);
    let mut dev = Device::new();
    let mut t = Transfer::app(&s, 0, &img).unwrap();
    let (result, writes) = run(&mut t, &img, &mut dev).unwrap();
    assert_eq!(result, VerifyResult::Verified);
    assert_eq!(dev.held, img, "the device holds exactly what was sent");
    assert_eq!(writes, 5000_usize.div_ceil(244));
    assert_eq!(t.progress(), (5000, 5000));
}

#[test]
fn a_device_that_already_has_part_of_it_is_not_sent_that_part_again() {
    let s = session();
    let img = image(5000);
    let mut dev = Device::new();
    dev.already = img[..2000].to_vec();
    let mut t = Transfer::app(&s, 0, &img).unwrap();
    let (_, writes) = run(&mut t, &img, &mut dev).unwrap();
    assert_eq!(dev.held, img);
    assert_eq!(writes, 3000_usize.div_ceil(244), "only what was missing went out");
}

#[test]
fn a_device_claiming_bytes_it_does_not_hold_is_caught_at_the_first_check() {
    // A resume is a claim, and the checksum is what tests it. A device
    // that says it has two thousand bytes of this image, and holds two
    // thousand bytes of something else, must not have the rest written
    // on top and called verified.
    let s = session();
    let img = image(5000);
    let mut dev = Device::new();
    dev.already = vec![0u8; 2000];
    let mut t = Transfer::app(&s, 0, &img).unwrap();
    match run(&mut t, &img, &mut dev) {
        Err(TransferError::Checksum { ours, device, .. }) => assert_ne!(ours, device),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_device_that_says_it_holds_fewer_bytes_than_it_does_is_caught_by_the_checksum_at_that_check_which_is_over_the_count_it_gives() {
    let s = session();
    let img = image(20_000);
    let mut dev = Device::new();
    dev.lie_offset = Some(0);
    let mut t = Transfer::app(&s, 0, &img).unwrap();
    // At the smallest size, where no smaller one remains to start the file
    // again at.
    t.limit_writes(20);
    match run(&mut t, &img, &mut dev) {
        // Our checksum over nothing is 0, and the device's is over the 160
        // bytes it holds.
        Err(TransferError::Checksum { sent, ours, device }) => assert_eq!((sent, ours, device), (0, 0, crc32(&img[..160]))),
        other => panic!("{other:?}"),
    }
    assert_eq!(dev.checks, 1, "it stopped at the check that said so, not later");
}

#[test]
fn a_device_busy_writing_its_flash_takes_a_whole_image_or_weights_and_throws_nothing_away() {
    let s = session();
    let img = image(30_000);
    for mut t in [Transfer::app(&s, 0, &img).unwrap(), Transfer::weights(&s, &img).unwrap()] {
        let mut dev = Device::new();
        let (result, writes) = run(&mut t, &img, &mut dev).unwrap();
        assert_eq!(result, VerifyResult::Verified);
        assert_eq!(dev.held, img, "the device holds exactly what was sent");
        assert_eq!(writes, 30_000_usize.div_ceil(244), "every write went once");
        assert!(dev.most_held > 0, "writes arrived while the device wrote a page");
        assert_eq!(dev.thrown, 0, "the device threw nothing away");
        assert!(dev.waited.is_empty(), "every check was answered");
    }
}

#[test]
fn a_transfer_carries_on_from_the_devices_own_count_past_writes_it_threw_away_and_asks_a_check_it_turned_away_again() {
    // The Python library's busy device: it falls behind at every
    // thirteenth write and throws away what arrives until it is asked where
    // it is, and it turns the first check away.
    let s = session();
    let img: Vec<u8> = (0..5120).map(|i| i as u8).collect();
    let mut dev = Device::new();
    dev.chunk_max = 100;
    dev.behind_every = 13;
    dev.turn_away = 1;
    let mut t = Transfer::app(&s, 0, &img).unwrap();
    let (result, writes) = run(&mut t, &img, &mut dev).unwrap();
    assert_eq!(result, VerifyResult::Verified);
    assert_eq!(dev.held, img, "every byte arrived once, in order");
    assert!(dev.thrown > 0, "the device threw writes away");
    assert_eq!(writes, 5120_usize.div_ceil(100) + dev.thrown, "what was thrown away went again, and nothing else");
    assert!(dev.checks > 5120 / (100 * 8), "progress is asked in short bursts");
}

#[test]
fn a_check_the_device_does_not_answer_is_asked_again_after_half_a_second_and_then_a_second_and_a_third_unanswered_stops_the_transfer() {
    let s = session();
    let img = image(5000);
    let mut dev = Device::new();
    dev.turn_away = 2;
    let (result, _) = run(&mut Transfer::app(&s, 0, &img).unwrap(), &img, &mut dev).unwrap();
    assert_eq!(result, VerifyResult::Verified);
    assert_eq!(dev.held, img);
    assert_eq!(dev.waited, [Duration::from_millis(500), Duration::from_secs(1)]);

    let mut deaf = Device::new();
    deaf.turn_away = 3;
    assert!(matches!(drive(&mut Transfer::app(&s, 0, &img).unwrap(), &img, &mut deaf), Err(Stop::Unanswered)));
    assert_eq!(deaf.waited, [Duration::from_millis(500), Duration::from_secs(1)], "three times in all");

    let mut t = Transfer::app(&s, 0, &img).unwrap();
    assert_eq!(t.unanswered(), None, "a start is not asked again");
}

#[test]
fn a_transfer_whose_count_stops_advancing_is_given_up_in_the_librarys_words() {
    let s = session();
    let img = vec![0u8; 5000];
    let mut dev = Device::new();
    dev.behind_every = 1;
    let mut t = Transfer::app(&s, 0, &img).unwrap();
    match run(&mut t, &img, &mut dev) {
        Err(e @ TransferError::Offset { sent, device }) => {
            assert_eq!(device, 0);
            assert!(sent > 0 && sent < img.len(), "it stopped where the count stopped, not at the end");
            assert_eq!(
                e.to_string(),
                format!("the device accepted 0 bytes where {sent} were sent, and the transfer is not advancing")
            );
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn bytes_that_arrived_changed_are_caught_by_the_checksum() {
    let s = session();
    let img = image(20_000);
    let mut dev = Device::new();
    dev.corrupt_at = Some(100);
    let mut t = Transfer::app(&s, 0, &img).unwrap();
    // At the smallest size, where no smaller one remains to start the file
    // again at.
    t.limit_writes(20);
    match run(&mut t, &img, &mut dev) {
        Err(TransferError::Checksum { ours, device, .. }) => assert_ne!(ours, device),
        other => panic!("{other:?}"),
    }
}

#[test]
fn an_image_the_device_refuses_says_why() {
    let s = session();
    let img = image(500);
    let mut dev = Device::new();
    dev.verdict = (Status::Verify, VerifyResult::Signature);
    let mut t = Transfer::app(&s, 0, &img).unwrap();
    match run(&mut t, &img, &mut dev) {
        Err(TransferError::Rejected(VerifyResult::Signature)) => {}
        other => panic!("{other:?}"),
    }
}

#[test]
fn an_image_the_device_refuses_at_its_first_bytes_is_finished_at_the_first_check_and_the_devices_verdict_says_why() {
    // As a device does with an image built for its other slot: it stops at
    // the envelope's 32 bytes and holds its verdict, and a progress check
    // says the transfer failed.
    let s = session();
    for n in [500, 20_000] {
        let img = image(n);
        let mut dev = Device::new();
        dev.envelope_verdict = Some(VerifyResult::Slot);
        let mut t = Transfer::app(&s, 0, &img).unwrap();
        match run(&mut t, &img, &mut dev) {
            Err(TransferError::Rejected(VerifyResult::Slot)) => {}
            other => panic!("{n} bytes: {other:?}"),
        }
        assert_eq!(dev.checks, 1, "{n} bytes: the device said so at the first check");
        assert_eq!(dev.writes, n.div_ceil(244).min(8), "{n} bytes: nothing was written after it said so");
    }
}

#[test]
fn a_finish_the_device_answers_ok_after_a_progress_check_said_the_transfer_failed_ends_in_the_refusal_a_refused_finish_gives_and_never_in_a_verified_transfer() {
    // No firmware answers so. The device stops at the envelope, a progress
    // check says the transfer failed, and the answer to the finish that
    // follows says ok.
    let s = session();
    let img = image(500);
    let failed_at_finish = || {
        let mut dev = Device::new();
        dev.envelope_verdict = Some(VerifyResult::Slot);
        let mut t = Transfer::app(&s, 0, &img).unwrap();
        loop {
            match t.step(&img) {
                Step::Send(command) if command.bytes == [Op::Finish as u8] => break t,
                Step::Send(command) => t.on_answer(&dev.control(&command.bytes).unwrap(), &img).unwrap(),
                Step::Data { from, to, .. } => {
                    dev.data(&img[from..to]);
                    t.sent(to - from);
                }
                Step::Verified(result) => panic!("the transfer said verified before a finish was sent: {result:?}"),
            }
        }
    };
    let ok = |verdict: &[u8]| {
        let mut out = [0u8; 8];
        let n = update::encode_response(Op::Finish, Status::Ok, verdict, &mut out).unwrap();
        out[..n].to_vec()
    };
    assert_eq!(
        failed_at_finish().on_answer(&ok(&[VerifyResult::Slot as u8]), &img),
        Err(TransferError::Rejected(VerifyResult::Slot)),
        "a verdict that is not verified is the device's reason"
    );
    // A verdict that says verified, none, or a number the contract does not
    // define is a refused finish with the device's status.
    for verdict in [vec![VerifyResult::Verified as u8], vec![], vec![0xEE]] {
        match failed_at_finish().on_answer(&ok(&verdict), &img) {
            Err(e @ TransferError::Refused { op: Op::Finish, status: Status::Ok }) => {
                assert_eq!(e.to_string(), "the device refused the transfer: ok");
            }
            other => panic!("{verdict:?}: {other:?}"),
        }
    }
}

#[test]
fn a_transfer_the_device_refuses_is_called_the_transfer_whether_it_carried_an_image_or_a_head() {
    use intomind::protocol::head;
    let s = session();
    let img = image(500);
    let mut dev = Device::new();
    dev.verdict = (Status::Verify, VerifyResult::Signature);
    match run(&mut Transfer::app(&s, 0, &img).unwrap(), &img, &mut dev) {
        Err(e @ TransferError::Rejected(VerifyResult::Signature)) => {
            assert_eq!(e.to_string(), "the device refused the transfer: signature");
        }
        other => panic!("{other:?}"),
    }

    let mut name = [0u8; head::NAME_LEN];
    name[..5].copy_from_slice(b"focus");
    let header = head::HeadHeader { version: head::FORMAT_VERSION_2, kind: head::KIND_LINEAR, in_dim: 96, out_dim: 2, name, encoder_id: [0; 8] };
    let mut blob = vec![0u8; head::blob_len(head::FORMAT_VERSION_2, 96, 2)];
    head::encode_unhashed(&header, &[1; 192], &[0, 0], &[1.0, 1.0], &mut blob).unwrap();
    let mut dev = Device::new();
    dev.verdict = (Status::Verify, VerifyResult::Hash);
    match run(&mut Transfer::head(&s, 1, &blob).unwrap(), &blob, &mut dev) {
        Err(e @ TransferError::Rejected(VerifyResult::Hash)) => {
            assert_eq!(e.to_string(), "the device refused the transfer: hash");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_finish_refused_with_no_reason_or_one_the_contract_does_not_define_is_a_refused_finish_with_its_status_and_one_with_a_reason_is_a_rejected_image() {
    let s = session();
    let img = image(500);
    for status in [Status::Verify, Status::Flash] {
        for reason in [vec![], vec![0xEE]] {
            let mut dev = Device::new();
            dev.verdict = (status, VerifyResult::Verified);
            dev.reason = Some(reason.clone());
            let mut t = Transfer::app(&s, 0, &img).unwrap();
            match run(&mut t, &img, &mut dev) {
                Err(e @ TransferError::Refused { op: Op::Finish, status: given }) => {
                    assert_eq!(given, status, "{reason:?}");
                    let words = if status == Status::Verify { "verification failed" } else { "flash" };
                    assert_eq!(e.to_string(), format!("the device refused the transfer: {words}"));
                }
                other => panic!("{status:?} {reason:?}: {other:?}"),
            }
        }
        let mut dev = Device::new();
        dev.verdict = (status, VerifyResult::Hash);
        let mut t = Transfer::app(&s, 0, &img).unwrap();
        match run(&mut t, &img, &mut dev) {
            Err(TransferError::Rejected(VerifyResult::Hash)) => {}
            other => panic!("{status:?}: {other:?}"),
        }
    }

    // An accepted finish with nothing after its status is verified, as it was.
    let mut dev = Device::new();
    dev.reason = Some(vec![]);
    let (result, _) = run(&mut Transfer::app(&s, 0, &img).unwrap(), &img, &mut dev).unwrap();
    assert_eq!(result, VerifyResult::Verified);
}

#[test]
fn a_finish_asked_again_after_the_transfer_verified_is_a_refused_finish_and_never_a_transfer_rejected_as_verified() {
    // The device verifies the image, and its answer does not arrive. Asked
    // to finish again, it refuses, as there is no transfer left to finish,
    // and still carries the verdict it holds: verified.
    let s = session();
    let img = image(500);
    let mut dev = Device::new();
    let mut t = Transfer::app(&s, 0, &img).unwrap();
    let finish = loop {
        match t.step(&img) {
            Step::Send(command) if command.bytes == [Op::Finish as u8] => break command,
            Step::Send(command) => t.on_answer(&dev.control(&command.bytes).unwrap(), &img).unwrap(),
            Step::Data { from, to, .. } => {
                dev.data(&img[from..to]);
                t.sent(to - from);
            }
            Step::Verified(result) => panic!("verified before a finish was sent: {result:?}"),
        }
    };
    let lost = dev.control(&finish.bytes).unwrap();
    assert_eq!(lost, [Op::Finish as u8, Status::Ok as u8, VerifyResult::Verified as u8], "the device verified it");
    let again = match t.step(&img) {
        Step::Send(command) => command,
        other => panic!("{other:?}"),
    };
    assert_eq!(again, finish, "the transfer asks to finish again");
    let answer = dev.control(&again.bytes).unwrap();
    assert_eq!(answer, [Op::Finish as u8, Status::NoTransfer as u8, VerifyResult::Verified as u8]);
    match t.on_answer(&answer, &img) {
        Err(e @ TransferError::Refused { op: Op::Finish, status: Status::NoTransfer }) => {
            assert_eq!(e.to_string(), "the device refused the transfer: no transfer");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_rejected_transfer_says_the_devices_verdict_in_the_python_librarys_words_for_each_verdict() {
    let verdicts = [
        (VerifyResult::Verified, "verified"),
        (VerifyResult::Length, "length"),
        (VerifyResult::Malformed, "malformed"),
        (VerifyResult::Target, "target"),
        (VerifyResult::Slot, "slot"),
        (VerifyResult::Hash, "hash"),
        (VerifyResult::Signature, "signature"),
        (VerifyResult::SecurityCounter, "security counter"),
        (VerifyResult::HeadShape, "head shape"),
        (VerifyResult::Flash, "flash"),
        (VerifyResult::KeyId, "key id"),
    ];
    for (result, words) in verdicts {
        assert_eq!(TransferError::Rejected(result).to_string(), format!("the device refused the transfer: {words}"));
    }
}

#[test]
fn a_refused_request_is_said_in_the_python_librarys_words_for_each_operation_and_each_status() {
    let asked = [
        (Op::Start, "to start the transfer"),
        (Op::Query, "a progress check"),
        (Op::Finish, "the transfer"),
        (Op::Activate, "to activate the image"),
        (Op::Abort, "to abort the transfer"),
    ];
    let statuses = [
        (Status::Ok, "ok"),
        (Status::InvalidArg, "invalid argument"),
        (Status::Unsupported, "unsupported"),
        (Status::Busy, "busy"),
        (Status::NoTransfer, "no transfer"),
        (Status::Flash, "flash"),
        (Status::Verify, "verification failed"),
    ];
    for (op, what) in asked {
        for (status, words) in statuses {
            assert_eq!(TransferError::Refused { op, status }.to_string(), format!("the device refused {what}: {words}"));
        }
    }
}

#[test]
fn a_head_is_checked_against_this_device_before_any_of_it_is_sent() {
    use intomind::protocol::head;
    let s = session();

    // A head for an encoder of the wrong width never reaches the wire.
    let mut name = [0u8; head::NAME_LEN];
    name[..5].copy_from_slice(b"focus");
    let header = head::HeadHeader { version: head::FORMAT_VERSION_2, kind: head::KIND_LINEAR, in_dim: 4, out_dim: 2, name, encoder_id: [0; 8] };
    let mut wrong = vec![0u8; head::blob_len(head::FORMAT_VERSION_2, 4, 2)];
    head::encode_unhashed(&header, &[1; 8], &[0, 0], &[1.0, 1.0], &mut wrong).unwrap();
    assert!(Transfer::head(&s, 1, &wrong).is_err(), "this device's model produces 96, not 4");

    // One of the right width goes.
    let header = head::HeadHeader { version: head::FORMAT_VERSION_2, kind: head::KIND_LINEAR, in_dim: 96, out_dim: 2, name, encoder_id: [0; 8] };
    let mut blob = vec![0u8; head::blob_len(head::FORMAT_VERSION_2, 96, 2)];
    head::encode_unhashed(&header, &[1; 192], &[0, 0], &[1.0, 1.0], &mut blob).unwrap();
    let mut dev = Device::new();
    let mut t = Transfer::head(&s, 1, &blob).unwrap();
    let (result, _) = run(&mut t, &blob, &mut dev).unwrap();
    assert_eq!(result, VerifyResult::Verified);
    assert_eq!(dev.held, blob, "the head arrived as it was built");
}

#[test]
fn a_device_that_does_not_claim_the_capability_is_not_asked() {
    let mut s = session();
    // A device with no capabilities at all.
    let mut buf = [0u8; device_info::LEN];
    let mut info = *s.info().unwrap();
    info.capabilities = 0;
    info.encode(&mut buf).unwrap();
    let mut bare = Session::new();
    bare.on_device_info(&buf).unwrap();
    assert!(Transfer::app(&bare, 0, &image(100)).is_err());
    assert!(Transfer::head(&bare, 1, &image(100)).is_err());
    assert!(bare.list_heads().is_err());
    assert!(bare.list_head_encoders().is_err());
    assert!(bare.remove_head(1).is_err());
    // And the capable one is.
    assert!(Transfer::app(&s, 0, &image(100)).is_ok());
    assert!(s.list_heads().is_ok());
    let _ = &mut s;
}

#[test]
fn only_a_1_4_device_is_asked_which_encoder_its_heads_name() {
    let s = session();
    let mut buf = [0u8; device_info::LEN];
    let mut info = *s.info().unwrap();
    for (version, asked) in [((1, 3), false), ((1, 4), true), ((1, 5), true)] {
        info.proto_version = version;
        info.encode(&mut buf).unwrap();
        let mut d = Session::new();
        d.on_device_info(&buf).unwrap();
        match d.list_head_encoders() {
            Ok(c) => {
                assert!(asked, "{version:?}");
                assert_eq!(c.bytes, [0x8A]);
            }
            Err(_) => assert!(!asked, "{version:?}"),
        }
    }
}

#[test]
fn the_smallest_and_largest_images_both_work() {
    let s = session();
    for n in [1usize, 243, 244, 245, 488] {
        let img = image(n);
        let mut dev = Device::new();
        let mut t = Transfer::app(&s, 0, &img).unwrap();
        let (result, _) = run(&mut t, &img, &mut dev).unwrap();
        assert_eq!(result, VerifyResult::Verified, "{n} bytes");
        assert_eq!(dev.held, img, "{n} bytes");
    }
    // Nothing is not an image.
    assert!(Transfer::app(&s, 0, &[]).is_err());
}

#[test]
fn a_device_that_wants_smaller_writes_gets_them() {
    let s = session();
    let img = image(2000);
    let mut dev = Device::new();
    dev.chunk_max = 20;
    let mut t = Transfer::app(&s, 0, &img).unwrap();
    let (_, writes) = run(&mut t, &img, &mut dev).unwrap();
    assert_eq!(dev.held, img);
    assert_eq!(writes, 100, "the device's limit, not ours");
}

/// The start of a transfer, as the crate writes it, in hex.
fn start(t: &mut Transfer, image: &[u8]) -> String {
    match t.step(image) {
        Step::Send(command) => command.bytes.iter().map(|b| format!("{b:02x}")).collect(),
        other => panic!("a transfer starts with a write, not {other:?}"),
    }
}

#[test]
fn a_weights_start_is_the_bytes_the_python_library_writes() {
    // The contract has the start name slot zero for everything but a head.
    // These are the bytes the Python library 1.4.4 gives for this image:
    //   img = bytes(((i * 7 + 13) & 0xFF) for i in range(1000))
    //   encode_update_start("weights", 0, len(img), hashlib.sha256(img).digest()[:8]).hex()
    let s = session();
    let img = image(1000);
    let mut t = Transfer::weights(&s, &img).unwrap();
    assert_eq!(start(&mut t, &img), "010200e8030000c85e29b0cb8af116");
}

#[test]
fn an_applications_start_names_slot_zero_and_another_slot_is_refused_before_anything_is_sent() {
    // encode_update_start("app", 0, len(img), id) for the same image.
    let s = session();
    let img = image(1000);
    let mut t = Transfer::app(&s, 0, &img).unwrap();
    assert_eq!(start(&mut t, &img), "010100e8030000c85e29b0cb8af116");
    assert_eq!(Transfer::app(&s, 1, &img).err(), Some(Error::Wire(intomind::protocol::Error::Invalid)));
}
