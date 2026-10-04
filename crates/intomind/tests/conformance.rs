//! The session against the contract's own vectors.
//!
//! `contract/conformance.json` is emitted by the firmware from the codec
//! the device runs. This crate builds that same codec, so the wire is the
//! same by construction. What these tests hold down is everything built
//! on top of it: that a batch carries the samples the vector states, that
//! a break in the timeline is announced before the samples that follow
//! it, and that a loss is counted exactly while a re-base is not counted
//! at all.

use intomind::protocol::pipeline::{input_source, kind, Chain, PredictionInput, Stage};
use intomind::protocol::{self, uuid_fill};
use intomind::{Event, Session};
use serde_json::Value;

fn vectors() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../contract/conformance.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("the contract's vectors")).unwrap()
}

fn bytes_of(hex: &str) -> Vec<u8> {
    (0..hex.len()).step_by(2).map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap()).collect()
}

/// A 64-bit field, which the contract writes as a decimal string. Parsing
/// it as a number would round it, and the rounded value would still agree
/// with a rounded expectation, so the mistake would pass unnoticed.
fn u64_of(v: &Value) -> u64 {
    v.as_str().expect("a 64-bit field is a decimal string").parse().expect("a decimal integer")
}

fn find<'a>(v: &'a Value, list: &str, kind: &str) -> Vec<&'a Value> {
    v[list].as_array().unwrap().iter().filter(|e| e["kind"] == kind).collect()
}

fn session_with_info(v: &Value) -> Session {
    let info = find(v, "decode", "device_info")[0];
    let mut s = Session::new();
    s.on_device_info(&bytes_of(info["bytes"].as_str().unwrap())).expect("device info");
    s
}

#[test]
fn the_vectors_are_for_this_version() {
    let (major, minor) = protocol::PROTOCOL_VERSION;
    assert_eq!(vectors()["protocol"], format!("{major}.{minor}"));
}

#[test]
fn a_device_says_what_it_is_and_nothing_is_assumed() {
    let v = vectors();
    let mut s = Session::new();
    let raw = bytes_of(find(&v, "decode", "device_info")[0]["bytes"].as_str().unwrap());
    let info = s.on_device_info(&raw).unwrap();
    assert_eq!(info.channel_count, 4);
    assert_eq!(info.adc_bits, 24);
    assert_eq!(info.time_tick_hz, 1_000_000);
    assert_eq!(info.vref_uv, 4_500_000);
    let ext = info.ext.expect("a device speaking this version reports its extension");
    assert_eq!(ext.model_embed_dim, 96);
    assert_eq!(ext.head_slots, 4);
    // Capabilities are bits the device reports.
    use intomind::protocol::device_info::capability;
    assert!(s.can(capability::MODEL) && s.can(capability::HEADS) && s.can(capability::UPDATE));
    assert!(!s.can(capability::BATTERY_LOW_FLAG));
    // A device that has not said what it is decodes nothing.
    let mut blank = Session::new();
    assert!(matches!(blank.on_notification(uuid_fill::EEG_DATA, &[0u8; 44])[..], [Event::Undecodable]));
}

#[test]
fn a_batch_carries_the_samples_the_vector_states() {
    let v = vectors();
    let mut s = session_with_info(&v);
    let packet = find(&v, "decode", "eeg_data")[0];
    let events = s.on_notification(uuid_fill::EEG_DATA, &bytes_of(packet["bytes"].as_str().unwrap()));
    let Event::Samples(batch) = &events[0] else { panic!("{events:?}") };
    let want = packet["fields"]["samples"].as_array().unwrap();
    assert_eq!(batch.rows(), want.len());
    assert_eq!(batch.channels, 4);
    assert_eq!(batch.gain, 24);
    assert_eq!(batch.sample_rate_hz, 500);
    for (row, expected) in want.iter().enumerate() {
        for (c, value) in expected.as_array().unwrap().iter().enumerate() {
            assert_eq!(batch.counts[row * 4 + c] as i64, value.as_i64().unwrap(), "row {row} channel {c}");
        }
    }
    // The device time is a full 64-bit value. The contract writes it as a
    // decimal string, because a JSON number cannot hold one exactly, and it
    // is read back as an integer here.
    assert_eq!(batch.device_time, u64_of(&packet["fields"]["device_time"]));
    assert_eq!(batch.device_time, 81_985_529_216_486_895);

    // And one count is worth what the device's own numbers say.
    let info = *s.info().unwrap();
    let uv = batch.microvolts(0, 0, &info);
    assert!((uv - 0.022_351_741_790_771_484).abs() < 1e-12, "{uv}");
}

#[test]
fn a_break_is_announced_before_the_samples_that_follow_it() {
    let v = vectors();
    let mut s = session_with_info(&v);
    let template = bytes_of(find(&v, "decode", "eeg_data")[0]["bytes"].as_str().unwrap());

    // Rewrite the header's index and flags to walk the contract's own
    // continuity cases through a real session.
    let repoint = |index: u32, discontinuity: bool| {
        let mut p = template.clone();
        p[1] = if discontinuity { p[1] | 1 } else { p[1] & !1 };
        p[4..8].copy_from_slice(&index.to_le_bytes());
        p
    };

    let first = s.on_notification(uuid_fill::EEG_DATA, &repoint(100, false));
    assert!(matches!(first[..], [Event::Samples(_)]), "the first packet follows nothing");

    // Three samples never arrived: a loss, counted exactly.
    let events = s.on_notification(uuid_fill::EEG_DATA, &repoint(105, false));
    let Event::Gap(gap) = &events[0] else { panic!("a loss has to be announced first: {events:?}") };
    assert_eq!(gap.samples_lost, Some(3));
    assert_eq!(gap.first_index_after, 105);
    assert!(matches!(events[1], Event::Samples(_)));
    assert_eq!(s.samples_lost, 3);

    // The timeline re-based: a break whose extent is not a number.
    let events = s.on_notification(uuid_fill::EEG_DATA, &repoint(0, true));
    let Event::Gap(gap) = &events[0] else { panic!("{events:?}") };
    assert!(gap.is_rebase());
    assert_eq!(gap.samples_lost, None, "zero missing would be a different claim from unknown");
    assert_eq!(s.samples_lost, 3, "a re-base adds nothing to the loss count");
    assert_eq!(s.rebases, 1);
}

#[test]
fn a_packet_delivered_twice_is_passed_on_once() {
    // Measured on the bench: the system's Bluetooth service sends each
    // notification once per program subscribed, and every subscriber hears
    // all of the copies. A copy is the same packet, first index and device
    // time; it is dropped and counted, and nothing is announced as a break.
    let v = vectors();
    let mut s = session_with_info(&v);
    let packet = bytes_of(find(&v, "decode", "eeg_data")[0]["bytes"].as_str().unwrap());
    let first = s.on_notification(uuid_fill::EEG_DATA, &packet);
    assert!(matches!(first[..], [Event::Samples(_)]));
    let again = s.on_notification(uuid_fill::EEG_DATA, &packet);
    assert!(matches!(again[..], [Event::Duplicate { .. }]), "a copy is not samples and not a break: {again:?}");
    assert_eq!((s.duplicates, s.rebases, s.samples_lost), (1, 0, 0));
    // A restart is never taken for a copy: the device's clock has moved on.
    let mut restart = packet.clone();
    restart[1] |= 1;
    let later = u64::from_le_bytes(restart[8..16].try_into().unwrap()) + 1_000_000;
    restart[8..16].copy_from_slice(&later.to_le_bytes());
    let events = s.on_notification(uuid_fill::EEG_DATA, &restart);
    assert!(matches!(events[0], Event::Gap(_)), "{events:?}");
}

#[test]
fn every_continuity_case_in_the_contract_comes_out_the_same_way() {
    let v = vectors();
    for case in v["continuity"].as_array().unwrap() {
        let mut s = session_with_info(&v);
        let template = bytes_of(find(&v, "decode", "eeg_data")[0]["bytes"].as_str().unwrap());
        let prev_index = case["prev_index"].as_u64().unwrap() as u32;
        let prev_n = case["prev_n_samples"].as_u64().unwrap() as u8;
        let new_index = case["new_index"].as_u64().unwrap() as u32;
        let flagged = case["discontinuity_flag"].as_bool().unwrap();

        let mut first = template.clone();
        first[1] &= !1;
        first[4..8].copy_from_slice(&prev_index.to_le_bytes());
        first[16] = prev_n;
        // The payload must match the sample count the header claims.
        let body = 20 + prev_n as usize * 4 * 3;
        first.resize(body, 0);
        s.on_notification(uuid_fill::EEG_DATA, &first);

        let mut second = template.clone();
        second[1] = if flagged { second[1] | 1 } else { second[1] & !1 };
        second[4..8].copy_from_slice(&new_index.to_le_bytes());
        let events = s.on_notification(uuid_fill::EEG_DATA, &second);
        match case["verdict"].as_str().unwrap() {
            "continuous" => assert!(matches!(events[..], [Event::Samples(_)]), "{case}"),
            "gap" => {
                let Event::Gap(g) = &events[0] else { panic!("{case}: {events:?}") };
                assert_eq!(g.samples_lost, Some(case["lost"].as_u64().unwrap() as u32), "{case}");
            }
            "break" => {
                let Event::Gap(g) = &events[0] else { panic!("{case}: {events:?}") };
                assert!(g.is_rebase(), "{case}");
                // Neither the contract nor the session claims an extent for
                // a break. Zero would be a claim that nothing was lost.
                assert!(case["lost"].is_null(), "{case}: a break has no count");
                assert_eq!(g.samples_lost, None, "{case}");
            }
            other => panic!("unknown verdict {other}"),
        }
    }
}

#[test]
fn the_other_notifications_decode_to_what_they_say() {
    let v = vectors();
    let mut s = session_with_info(&v);

    let st = find(&v, "decode", "status")[0];
    let events = s.on_notification(uuid_fill::STATUS, &bytes_of(st["bytes"].as_str().unwrap()));
    let Event::Status(status) = &events[0] else { panic!("{events:?}") };
    assert_eq!(status.state, 1);
    assert_eq!(status.buffer_fill, st["fields"]["buffer_fill"].as_u64().unwrap() as u16);

    let pr = find(&v, "decode", "prediction")[0];
    let events = s.on_notification(uuid_fill::PREDICTIONS, &bytes_of(pr["bytes"].as_str().unwrap()));
    let Event::Prediction(p) = &events[0] else { panic!("{events:?}") };
    assert_eq!(p.head_slot, 2);
    assert_eq!(p.device_time, u64_of(&pr["fields"]["device_time"]));
    assert!(p.duty_reduced && !p.gap_in_window);
    assert_eq!(p.outputs.len(), 3);
    assert!((p.outputs[1] + 1.5).abs() < 1e-6);

    let answer = find(&v, "decode", "control_response")[0];
    let events = s.on_notification(uuid_fill::CONTROL_RSP, &bytes_of(answer["bytes"].as_str().unwrap()));
    assert!(matches!(events[0], Event::Answer { .. }));

    // Anything that is not a message is counted, never guessed at.
    let events = s.on_notification(uuid_fill::STATUS, &[0u8; 3]);
    assert!(matches!(events[..], [Event::Undecodable]));
    assert_eq!(s.undecodable, 1);
}

#[test]
fn a_command_is_the_bytes_the_contract_states() {
    let v = vectors();
    let s = session_with_info(&v);
    for request in find(&v, "decode", "control_request") {
        let want = bytes_of(request["bytes"].as_str().unwrap());
        let opcode = request["fields"]["opcode"].as_u64().unwrap() as u8;
        let got = match opcode {
            0x01 => Some(s.start_stream()),
            0x10 => Some(s.set_rate(500).unwrap()),
            0x11 => Some(s.set_gain(24).unwrap()),
            0x20 => Some(s.set_mode(request["fields"]["arg"].as_u64().unwrap() as u8).unwrap()),
            0x88 => Some(s.set_model_interval(request["fields"]["interval_s"].as_u64().unwrap() as u16).unwrap()),
            0x89 => Some(s.model_interval().unwrap()),
            0x47 => Some(s.name().unwrap()),
            0x48 => Some(
                s.set_name(request["fields"]["name"].as_str().unwrap(), request["fields"]["adjective"].as_str().unwrap(), "IntoMind One")
                    .unwrap(),
            ),
            0x40 => Some(s.time_sync()),
            0x80 => Some(s.set_predictions(true).unwrap()),
            0x81 => Some(s.select_head(2).unwrap()),
            0x90 => Some(s.pipeline_catalog().unwrap()),
            0x91 => Some(s.pipeline().unwrap()),
            0x92 => {
                let mut chain = Chain::NATURAL;
                chain.push(Stage::one(kind::HIGH_PASS, 10)).unwrap();
                Some(s.set_pipeline(&chain, 500).unwrap())
            }
            0x93 => Some(s.clear_pipeline().unwrap()),
            0x94 => Some(s.restore_pipeline_default().unwrap()),
            0x85 => Some(s.set_prediction_input(&PredictionInput { source: input_source::NATURAL, chain: Chain::NATURAL }, 500).unwrap()),
            0x86 => Some(s.prediction_input().unwrap()),
            0x44 => Some(s.indicator().unwrap()),
            0x45 => Some(s.set_indicator(2).unwrap()),
            0x46 => Some(s.identify(5).unwrap()),
            0x34 => Some(s.converter_registers().unwrap()),
            0x87 => Some(s.set_embeddings(Some(intomind::embeddings::Form::Both)).unwrap()),
            _ => None,
        };
        if let Some(c) = got {
            assert_eq!(c.bytes, want, "{}", request["name"]);
            assert_eq!(c.characteristic, uuid_fill::CONTROL);
        }
    }
    // A rate the contract does not define is refused here rather than on
    // the wire.
    assert!(s.set_rate(333).is_err());
    assert!(s.set_gain(3).is_err());
    assert!(s.set_mode(9).is_err());
    // So is a chain the device would refuse, with the device's reason.
    let mut bad = Chain::NATURAL;
    bad.push(Stage::one(kind::LOW_PASS, 2250)).unwrap();
    assert!(matches!(s.set_pipeline(&bad, 500), Err(intomind::Error::Chain(intomind::pipeline::Refusal::AboveNyquist(_, 2250)))));
    assert!(s.set_pipeline(&bad, 1000).is_ok(), "the same corner runs at a rate that carries it");
    // The vectors' device claims no bias drive, and is not asked for one.
    assert!(matches!(s.set_bias(1), Err(intomind::Error::NotCapable("bias_drive"))));
    // 1.2: a level or an identify the contract does not define is refused here.
    assert!(s.set_indicator(3).is_err());
    assert!(s.identify(31).is_err());
    assert_eq!(s.set_embeddings(None).unwrap().bytes, vec![0x87, 0]);
}

#[test]
fn the_registers_and_the_embeddings_decode_and_a_window_comes_back_whole() {
    use intomind::embeddings::{Assembler, Embedding, Form};
    use protocol::control::ConverterRegisters;
    let v = vectors();
    let regs = find(&v, "decode", "converter_registers")[0];
    let raw = bytes_of(regs["bytes"].as_str().unwrap());
    let r = ConverterRegisters::parse(&raw).unwrap();
    assert_eq!((r.family, r.first, r.values.len()), (1, 0, 24));
    assert_eq!(r.values.to_vec(), bytes_of(regs["fields"]["values"].as_str().unwrap()));

    let mi = find(&v, "decode", "model_info");
    for e in &mi {
        let m = protocol::control::ModelInfo::parse(&bytes_of(e["bytes"].as_str().unwrap())).unwrap();
        assert_eq!(m.tokens_per_channel as u64, e["fields"]["tokens_per_channel"].as_u64().unwrap(), "{}", e["name"]);
    }

    let parts = find(&v, "decode", "embedding");
    let values_of = |e: &serde_json::Value| -> Vec<i16> { e["fields"]["values"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i16).collect() };
    // The launch model's window embedding as a device sends it from 1.4.2:
    // two parts, each within the 156 byte limit (1.4, section 27).
    for i in [0, 1, 3, 4] {
        assert!(bytes_of(parts[i]["bytes"].as_str().unwrap()).len() <= protocol::NOTIFICATION_MAX, "{}", parts[i]["name"]);
    }
    let head = Embedding::parse(&bytes_of(parts[0]["bytes"].as_str().unwrap())).unwrap();
    let tail = Embedding::parse(&bytes_of(parts[1]["bytes"].as_str().unwrap())).unwrap();
    assert_eq!((head.token, head.embed_dim, head.first, head.values.len(), head.more_parts), (None, 76, 0, 64, true));
    assert_eq!((tail.token, tail.first, tail.values.len(), tail.more_parts), (None, 64, 12, false));
    assert!(head.leadoff_in_window);
    assert_eq!(head.values, values_of(parts[0]));
    // The same vector as firmware before 1.4.2 sent it, in one notification,
    // reads the same, and the two parts put back together equal it.
    let whole = Embedding::parse(&bytes_of(parts[2]["bytes"].as_str().unwrap())).unwrap();
    assert_eq!((whole.first, whole.values.len(), whole.more_parts), (0, 76, false));
    let mut joined = Assembler::new(1, 1, Form::Window);
    assert!(joined.feed(head).is_none());
    let w = joined.feed(tail).expect("the second part completes the window embedding");
    assert_eq!(w.embedding, Some(whole.values.clone()));
    assert_eq!(whole.values, values_of(parts[2]));
    let one = Embedding::parse(&bytes_of(parts[3]["bytes"].as_str().unwrap())).unwrap();
    let two = Embedding::parse(&bytes_of(parts[4]["bytes"].as_str().unwrap())).unwrap();
    assert_eq!((one.token, one.first, one.values.len(), one.more_parts), (Some(17), 0, 64, true));
    assert_eq!((two.token, two.first, two.values.len(), two.more_parts), (Some(17), 64, 64, false));

    // A whole window from notifications in any order: one channel, three
    // tokens of four values, and the window embedding, one token in two parts.
    fn packet(token: u8, first: u8, values: &[i16], more: bool) -> Vec<u8> {
        use protocol::embeddings::{flags, EmbeddingHeader};
        let h = EmbeddingHeader {
            flags: if more { flags::MORE_PARTS } else { 0 },
            embed_dim: 4,
            first,
            sample_index: 77,
            device_time: 1000,
            window_samples: 2000,
            encoder_id: [7; 8],
            input_source: 0,
            token,
        };
        let mut b = vec![0u8; 64];
        let n = h.encode(values, &mut b).unwrap();
        b.truncate(n);
        b
    }
    let mut a = Assembler::new(1, 3, Form::Both);
    let fed: Vec<Option<_>> = [
        packet(0, 0, &[1, 2, 3, 4], false),
        packet(2, 0, &[9, 9], true),
        packet(0xFF, 0, &[5, 6, 7, 8], false),
        packet(1, 0, &[-1, -2, -3, -4], false),
        packet(2, 2, &[8, 8], false),
    ]
    .iter()
    .map(|b| a.feed(Embedding::parse(b).unwrap()))
    .collect();
    assert!(fed[..4].iter().all(Option::is_none));
    let w = fed[4].clone().expect("the last piece completes the window");
    assert_eq!(w.embedding, Some(vec![5, 6, 7, 8]));
    assert_eq!(w.tokens, Some(vec![vec![1, 2, 3, 4], vec![-1, -2, -3, -4], vec![9, 9, 8, 8]]));
    assert_eq!(w.embedding_f32().unwrap()[0], 5.0 / 4096.0);
    // The window form alone completes without a token.
    let mut b = Assembler::new(1, 3, Form::Window);
    assert!(b.feed(Embedding::parse(&packet(0xFF, 0, &[1, 1, 1, 1], false)).unwrap()).is_some());
    // A window left behind by newer ones is dropped and counted.
    let mut c = Assembler::new(1, 3, Form::Tokens);
    for index in 1u32..=4 {
        let mut raw = packet(0, 0, &[1, 2, 3, 4], false);
        raw[4..8].copy_from_slice(&index.to_le_bytes());
        c.feed(Embedding::parse(&raw).unwrap());
    }
    assert_eq!(c.incomplete, 1);
    // The session hands an embedding notification on as an event.
    let mut s = session_with_info(&v);
    let ev = s.on_notification(protocol::uuid_fill::EMBEDDINGS, &bytes_of(parts[0]["bytes"].as_str().unwrap()));
    assert!(matches!(ev.as_slice(), [intomind::Event::Embedding(e)] if e.embed_dim == 76 && e.more_parts));
}

#[test]
fn a_capability_the_device_does_not_claim_is_not_asked_for() {
    let v = vectors();
    let info = find(&v, "decode", "device_info")[0];
    let mut raw = bytes_of(info["bytes"].as_str().unwrap());
    // Clear every capability bit: a device that claims nothing.
    raw[15] = 0;
    raw[16] = 0;
    let mut s = Session::new();
    s.on_device_info(&raw).unwrap();
    assert!(s.set_leadoff(true).is_err());
    assert!(s.set_predictions(true).is_err());
    assert!(s.select_head(1).is_err());
    // And the things every device does are still there.
    assert_eq!(s.start_stream().bytes, vec![0x01]);
}

/// The contract's word for why a message was refused.
fn reason_of(e: protocol::Error) -> &'static str {
    match e {
        protocol::Error::Truncated => "truncated",
        protocol::Error::Invalid => "invalid",
        protocol::Error::Reserved => "reserved",
        protocol::Error::NoRoom => "no room",
    }
}

/// A head blob is checked against the device that would receive it, so
/// reading one needs the device's embedding width and output limit. The
/// vectors' head is four by two.
fn refuse_head(blob: &[u8]) -> Result<(), &'static str> {
    use protocol::head;
    // Shorter than a header and a hash is not a head at all, whatever its
    // bytes say. The same rule the other implementations state.
    if blob.len() < head::HEADER_LEN + head::HASH_LEN {
        return Err("truncated");
    }
    let layout = head::layout(blob, 4, 32).map_err(|_| "invalid")?;
    let mut digest = [0u8; head::HASH_LEN];
    digest.copy_from_slice(&intomind_image_sha256(layout.hashed));
    if digest != layout.hash {
        return Err("invalid");
    }
    Ok(())
}

/// The same SHA-256 the device uses, so a head's identity is one number
/// everywhere.
fn intomind_image_sha256(data: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(data);
    h.finalize().into()
}

#[test]
fn the_head_lists_and_their_encoders_decode_to_the_stated_fields() {
    use protocol::control::{self, HeadEncoders, ListHeads};
    let v = vectors();
    for e in find(&v, "decode", "list_heads") {
        let name = e["name"].as_str().unwrap();
        let b = bytes_of(e["bytes"].as_str().unwrap());
        assert!(2 + b.len() <= control::RESPONSE_MAX, "{name}: an answer is at most 156 bytes");
        let l = ListHeads::parse(&b).unwrap();
        assert_eq!(l.active_slot as u64, e["fields"]["active_slot"].as_u64().unwrap(), "{name}");
        let want = e["fields"]["heads"].as_array().unwrap();
        assert_eq!(l.len(), want.len(), "{name}");
        for (got, want) in l.iter().zip(want) {
            assert_eq!(got.slot as u64, want["slot"].as_u64().unwrap(), "{name}");
            assert_eq!(got.state as u64, want["state"].as_u64().unwrap(), "{name}");
            assert_eq!(got.out_dim as u64, want["out_dim"].as_u64().unwrap(), "{name}");
            assert_eq!(got.encoder_id, [0; 8], "{name}: the list carries no encoder ids");
            if let Some(id) = want.get("head_id") {
                assert_eq!(bytes_of(id.as_str().unwrap()), got.head_id, "{name}");
            }
        }
    }
    let e = find(&v, "decode", "list_head_encoders")[0];
    let b = bytes_of(e["bytes"].as_str().unwrap());
    let h = HeadEncoders::parse(&b).unwrap();
    let want = e["fields"]["heads"].as_array().unwrap();
    assert_eq!(h.len(), want.len());
    for (got, want) in h.iter().zip(want) {
        assert_eq!(got.slot as u64, want["slot"].as_u64().unwrap());
        assert_eq!(got.encoder_id.to_vec(), bytes_of(want["encoder_id"].as_str().unwrap()));
    }
}

#[test]
fn every_malformed_message_is_refused_for_its_stated_reason() {
    use protocol::{control, device_info, frame, pipeline, predictions, status, update};
    let v = vectors();
    let mut seen = 0;
    for e in v["refuse"].as_array().unwrap() {
        let kind = e["kind"].as_str().unwrap();
        let name = e["name"].as_str().unwrap();
        let want = e["reason"].as_str().unwrap();
        let b = bytes_of(e["bytes"].as_str().unwrap());
        let got: Result<(), &'static str> = match kind {
            "device_info" => device_info::DeviceInfo::parse(&b).map(|_| ()).map_err(reason_of),
            "eeg_data" => frame::DataHeader::parse(&b, 4).map(|_| ()).map_err(reason_of),
            "control_request" => control::Request::parse(&b).map_err(reason_of).and_then(|r| match r.opcode {
                // A request that carries a payload is judged by the payload's own reader.
                control::Opcode::SetModelInterval => control::interval::parse_seconds(&b[1..]).map(|_| ()).map_err(reason_of),
                control::Opcode::SetName => control::name::parse_parts(&b[1..]).map(|_| ()).map_err(reason_of),
                _ => Ok(()),
            }),
            "control_response" => control::Response::parse(&b).map(|_| ()).map_err(reason_of),
            "status" => status::StatusMsg::parse(&b).map(|_| ()).map_err(reason_of),
            "battery_info" => control::BatteryInfo::parse(&b).map(|_| ()).map_err(reason_of),
            "boot_info" => control::BootInfo::parse(&b).map(|_| ()).map_err(reason_of),
            "model_info" => control::ModelInfo::parse(&b).map(|_| ()).map_err(reason_of),
            "list_heads" => control::ListHeads::parse(&b).map(|_| ()).map_err(reason_of),
            "list_head_encoders" => control::HeadEncoders::parse(&b).map(|_| ()).map_err(reason_of),
            "prediction" => predictions::PredictionHeader::parse(&b).map(|_| ()).map_err(reason_of),
            "update_response" => update::Response::parse(&b).map(|_| ()).map_err(reason_of),
            "envelope" => update::Envelope::parse(&b).map(|_| ()).map_err(reason_of),
            "head" => refuse_head(&b),
            "pipeline_catalog" => pipeline::Catalog::parse(&b).map(|_| ()).map_err(reason_of),
            "chain" => pipeline::Chain::parse(&b).map(|_| ()).map_err(reason_of),
            "pipeline_state" => pipeline::PipelineState::parse(&b).map(|_| ()).map_err(reason_of),
            "prediction_input" => pipeline::PredictionInput::parse(&b).map(|_| ()).map_err(reason_of),
            "bias_diagnostic" => pipeline::BiasDiagnostic::parse(&b).map(|_| ()).map_err(reason_of),
            "converter_registers" => control::ConverterRegisters::parse(&b).map(|_| ()).map_err(reason_of),
            "embedding" => protocol::embeddings::EmbeddingHeader::parse(&b).map(|_| ()).map_err(reason_of),
            other => panic!("{name}: no reader here for a {other}, and every refusal vector needs one"),
        };
        seen += 1;
        match got {
            Ok(()) => panic!("{name}: accepted, and it is not a message"),
            Err(r) => assert_eq!(r, want, "{name}: refused as {r}, and the contract calls it {want}"),
        }
    }
    assert_eq!(seen, v["refuse"].as_array().unwrap().len(), "every refusal vector is read here");
}
