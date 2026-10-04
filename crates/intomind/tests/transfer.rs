//! A transfer, driven end to end against a device that answers the way
//! the contract says a device answers.
//!
//! The device here is a few dozen lines rather than a mock: it holds
//! bytes, counts them, checksums them, and answers. That is enough to
//! prove the state machine, and it is the same shape the real one has.

use intomind::protocol::update::{self, Op, Request, Status, VerifyResult};
use intomind::protocol::{crc32::crc32, device_info, uuid_fill};
use intomind::{Session, Step, Transfer, TransferError};

/// A device that takes an image and says what it has.
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
        }
    }

    fn control(&mut self, bytes: &[u8]) -> Vec<u8> {
        let mut out = [0u8; 32];
        // The device's own parser, so this answers what a device would
        // answer and refuses what a device would refuse.
        let request = Request::parse(bytes).expect("a request this contract defines");
        let n = match request {
            Request::Start(start) => {
                assert!(start.total_len > 0);
                self.held = self.already.clone();
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
                let offset = self.lie_offset.take().unwrap_or(self.held.len() as u32);
                let q = update::QueryResponse {
                    state: update::State::Receiving,
                    offset,
                    crc32: crc32(&self.held),
                };
                let mut body = [0u8; update::QueryResponse::LEN];
                q.encode(&mut body).unwrap();
                update::encode_response(Op::Query, Status::Ok, &body, &mut out).unwrap()
            }
            Request::Finish => {
                let (status, result) = self.verdict;
                update::encode_response(Op::Finish, status, &[result as u8], &mut out).unwrap()
            }
            Request::Abort => {
                self.held.clear();
                update::encode_response(Op::Abort, Status::Ok, &[], &mut out).unwrap()
            }
            Request::Activate => update::encode_response(Op::Activate, Status::Ok, &[], &mut out).unwrap(),
        };
        out[..n].to_vec()
    }

    fn data(&mut self, bytes: &[u8]) {
        let mut bytes = bytes.to_vec();
        if let Some(at) = self.corrupt_at {
            if self.held.len() <= at && at < self.held.len() + bytes.len() {
                let i = at - self.held.len();
                bytes[i] ^= 0x01;
                self.corrupt_at = None;
            }
        }
        self.held.extend_from_slice(&bytes);
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

/// Drive a transfer to its end against a device, returning what the
/// device said and how many writes it took.
fn run(t: &mut Transfer, image: &[u8], dev: &mut Device) -> Result<(VerifyResult, usize), TransferError> {
    let mut writes = 0;
    for _ in 0..100_000 {
        match t.step(image) {
            Step::Send(command) => {
                assert!(command.with_response, "a control write is answered");
                assert_eq!(command.characteristic, uuid_fill::UPDATE_CONTROL);
                let answer = dev.control(&command.bytes);
                t.on_answer(&answer, image)?;
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

fn image(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i * 7 + 13) as u8).collect()
}

#[test]
fn an_image_arrives_byte_for_byte() {
    let s = session();
    let img = image(5000);
    let mut dev = Device::new();
    let mut t = Transfer::app(&s, 1, &img).unwrap();
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
    let mut t = Transfer::app(&s, 1, &img).unwrap();
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
    let mut t = Transfer::app(&s, 1, &img).unwrap();
    match run(&mut t, &img, &mut dev) {
        Err(TransferError::Checksum { ours, device, .. }) => assert_ne!(ours, device),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_write_the_device_did_not_get_stops_the_transfer_where_it_happened() {
    let s = session();
    let img = image(20_000);
    let mut dev = Device::new();
    dev.lie_offset = Some(0);
    let mut t = Transfer::app(&s, 1, &img).unwrap();
    match run(&mut t, &img, &mut dev) {
        Err(TransferError::Offset { sent, device }) => {
            assert_eq!(device, 0);
            assert!(sent > 0 && sent < img.len(), "it stopped at the first check, not at the end");
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
    let mut t = Transfer::app(&s, 1, &img).unwrap();
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
    let mut t = Transfer::app(&s, 1, &img).unwrap();
    match run(&mut t, &img, &mut dev) {
        Err(TransferError::Rejected(VerifyResult::Signature)) => {}
        other => panic!("{other:?}"),
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
    assert!(Transfer::app(&bare, 1, &image(100)).is_err());
    assert!(Transfer::head(&bare, 1, &image(100)).is_err());
    assert!(bare.list_heads().is_err());
    assert!(bare.list_head_encoders().is_err());
    assert!(bare.remove_head(1).is_err());
    // And the capable one is.
    assert!(Transfer::app(&s, 1, &image(100)).is_ok());
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
        let mut t = Transfer::app(&s, 1, &img).unwrap();
        let (result, _) = run(&mut t, &img, &mut dev).unwrap();
        assert_eq!(result, VerifyResult::Verified, "{n} bytes");
        assert_eq!(dev.held, img, "{n} bytes");
    }
    // Nothing is not an image.
    assert!(Transfer::app(&s, 1, &[]).is_err());
}

#[test]
fn a_device_that_wants_smaller_writes_gets_them() {
    let s = session();
    let img = image(2000);
    let mut dev = Device::new();
    dev.chunk_max = 20;
    let mut t = Transfer::app(&s, 1, &img).unwrap();
    let (_, writes) = run(&mut t, &img, &mut dev).unwrap();
    assert_eq!(dev.held, img);
    assert_eq!(writes, 100, "the device's limit, not ours");
}
