//! The recorder held to the Python library's: every session in
//! `contract/recorder` replayed through a [`Session`], and everything it
//! writes compared with what the library wrote from the same script. The
//! clock record after every exchange, the configuration and the processing
//! read back after a run, the manifest and the events byte for byte, the
//! arrays element for element and in the same layout, the checksums, the
//! session log, and which files are left read-only.
//!
//! The one difference allowed is the software a recording names: a
//! recording made here names this crate where the library names Python.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use intomind::capture::record::{self, Run, Setup};
use intomind::capture::recording::{npz_digest, Collection};
use intomind::protocol::device_info::{capability, capability_high};
use intomind::protocol::pipeline::PipelineState;
use intomind::protocol::status::StatusMsg;
use intomind::protocol::uuid_fill;
use intomind::{HostClocks, Session};
use intomind::capture::json::{loads, Value};

fn fixture() -> Value {
    let text = include_str!("../../../contract/recorder/sessions.json");
    loads(text).expect("the fixture is JSON")
}

fn hex(s: &Value) -> Vec<u8> {
    let s = s.as_str().expect("hex");
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

fn clocks(v: &Value) -> HostClocks {
    HostClocks { monotonic: v[0].as_f64().unwrap(), realtime: v[1].as_f64().unwrap() }
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("intomind-recorder-{}-{name}", std::process::id()));
    if d.exists() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let mut perm = e.metadata().unwrap().permissions();
            #[allow(clippy::permissions_set_readonly_false)]
            perm.set_readonly(false);
            let _ = std::fs::set_permissions(e.path(), perm);
        }
        std::fs::remove_dir_all(&d).unwrap();
    }
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// An answer's payload: what follows its opcode and status.
fn payload(answer: &Value) -> Vec<u8> {
    hex(answer)[2..].to_vec()
}

/// The library's manifest with the software it names put as this crate
/// names itself, so everything else can be compared byte for byte.
fn as_named_here(python: &str) -> String {
    let start = python.find("  \"host_versions\": {").expect("a manifest names its software");
    let end = start + python[start..].find("\n  }").expect("the block closes") + "\n  }".len();
    let ours = format!(
        "  \"host_versions\": {{\n    \"intomind_rust\": \"{}\",\n    \"platform\": \"{}-{}\"\n  }}",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    format!("{}{}{}", &python[..start], ours, &python[end..])
}

#[test]
fn every_session_records_what_the_python_library_records() {
    let f = fixture();
    let sessions = f["sessions"].as_array().unwrap();
    assert!(sessions.len() >= 19, "{}", sessions.len());
    let mut problems: Vec<String> = Vec::new();
    for s in sessions {
        let name = s["name"].as_str().unwrap();
        let want = &s["expected"];
        let dir = scratch(name);
        let collection = Collection::open(&dir);
        let mut session = Session::new();
        // What the library's own Bluetooth adapter names itself, declared
        // here as an application declares its own.
        let mut setup = Setup {
            adapter: Some(Value::obj([
                ("name", Value::str("intomind.ble")),
                ("version", Value::str("1")),
                ("first_party", Value::Bool(true)),
                ("transport", Value::str("bluetooth-le")),
            ])),
            ..Setup::default()
        };
        let mut syncs = 0;
        let mut link_ok = true;
        let mut config_after: Option<Value> = None;
        let mut processing: Option<Value> = None;
        let mut problem = |what: String| problems.push(format!("{name}: {what}"));
        for step in s["script"].as_array().unwrap() {
            match step["op"].as_str().unwrap() {
                "info" => {
                    session.on_device_info(&hex(&step["bytes"])).unwrap();
                }
                "site" => setup.site = Some(step["site"].clone()),
                "sync" => {
                    let ticks = step["ticks"].as_u64().unwrap().to_le_bytes();
                    session.on_time_sync_clocks(&ticks, clocks(&step["before"]), clocks(&step["after"])).unwrap();
                    let mine = session.clock().unwrap().quality();
                    if mine != want["clock"][syncs] {
                        problem(format!("exchange {syncs}: {mine} differs from {}", want["clock"][syncs]));
                    }
                    syncs += 1;
                }
                "start" => {
                    let _ = session.reset_epoch();
                    let _ = session.start_stream();
                    session.on_notification(uuid_fill::CONTROL_RSP, &[0x01, 0x00]);
                }
                "reset_epoch" => {
                    let _ = session.reset_epoch();
                }
                "arm" => session.arm_recorder(),
                "data" => {
                    for p in step["packets"].as_array().unwrap() {
                        session.on_notification(uuid_fill::EEG_DATA, &hex(p));
                    }
                }
                "link_lost" => link_ok = false,
                "stop" => {
                    let _ = session.stop_stream();
                    if link_ok {
                        session.on_notification(uuid_fill::CONTROL_RSP, &[0x02, 0x00]);
                    }
                }
                "disarm" => session.disarm_recorder(),
                "config_after" => {
                    config_after = if !step["link_ok"].as_bool().unwrap() {
                        None
                    } else {
                        let status = StatusMsg::parse(&hex(&step["status"])).unwrap();
                        let pipeline = session
                            .can(capability::PIPELINE)
                            .then(|| PipelineState::parse(&payload(&step["pipeline"])).unwrap());
                        let indicator = session.can(capability::INDICATOR).then(|| payload(&step["indicator"])[0]);
                        let interval = session.can(capability::MODEL_CADENCE).then(|| {
                            let b = payload(&step["model_interval"]);
                            (u16::from_le_bytes([b[0], b[1]]), u16::from_le_bytes([b[2], b[3]]))
                        });
                        let named = session.can_high(capability_high::DEVICE_NAME).then(|| payload(&step["name"]));
                        let parts = named.as_ref().map(|b| intomind::protocol::control::name::parse_parts(b).unwrap());
                        let parts =
                            parts.map(|(n, a)| (std::str::from_utf8(n).unwrap(), std::str::from_utf8(a).unwrap()));
                        Some(record::config_after(
                            &status,
                            session.leadoff_setting(),
                            pipeline.as_ref(),
                            indicator,
                            interval,
                            parts,
                        ))
                    };
                    let theirs = &want["config_after"];
                    if config_after.as_ref().unwrap_or(&Value::Null) != theirs {
                        problem(format!("config read back {config_after:?} differs from {theirs}"));
                    }
                }
                "processing" => {
                    let capable = session.can(capability::PIPELINE);
                    let state = PipelineState::parse(&payload(&step["pipeline"])).ok();
                    let rate = config_after.as_ref().and_then(|c| c["rate_sps"].as_u64()).map(|r| r as u32);
                    processing = record::processing(capable, state.as_ref(), rate);
                    if processing.as_ref().unwrap_or(&Value::Null) != &want["processing"] {
                        problem(format!("processing {processing:?} differs from {}", want["processing"]));
                    }
                }
                "save" => {
                    let label = step["label"].as_str().unwrap();
                    let mut extra = step["extra"].clone();
                    assert!(extra.as_object().is_some(), "{name}: extra is an object");
                    if step["with_processing"].as_bool() == Some(true) {
                        let description = processing.as_ref().map(|p| p["description"].clone());
                        if let Some(p) = &processing {
                            extra.set("processing", p.clone());
                        }
                        extra.set("signal", description.unwrap_or_else(|| Value::str("unknown")));
                    }
                    let run = Run {
                        extra,
                        events: step.get("events").filter(|e| !e.is_null()).cloned(),
                        regs_after: step.get("regs_after").filter(|e| !e.is_null()).cloned(),
                        config_after: match step.get("config_after") {
                            Some(Value::Str(s)) if s == "read" => config_after.clone(),
                            Some(v) if !v.is_null() => Some(v.clone()),
                            _ => None,
                        },
                        aborted: step.get("aborted").and_then(Value::as_str).map(str::to_string),
                        finished_unix: Some(step["now"][1].as_f64().unwrap()),
                    };
                    if let Err(e) = session.save_recording(&collection, label, &setup, &run) {
                        problem(format!("not saved: {e}"));
                        continue;
                    }
                    let read = |suffix: &str| std::fs::read_to_string(dir.join(format!("{label}{suffix}"))).ok();
                    match read(".meta.json") {
                        Some(mine) if mine == as_named_here(want["meta_json"].as_str().unwrap()) => {}
                        Some(mine) => problem(format!(
                            "the manifest differs:\n{}",
                            first_difference(&mine, &as_named_here(want["meta_json"].as_str().unwrap()))
                        )),
                        None => problem("no manifest".into()),
                    }
                    if read(".events.json").as_deref() != want["events_json"].as_str() {
                        problem(format!("events {:?} differ from {:?}", read(".events.json"), want["events_json"]));
                    }
                    if read("SESSIONS.jsonl").is_some() || dir.join("SESSIONS.jsonl").exists() {
                        let log = std::fs::read_to_string(dir.join("SESSIONS.jsonl")).unwrap();
                        if log != want["sessions_jsonl"].as_str().unwrap() {
                            problem(format!("session log {log:?} differs from {}", want["sessions_jsonl"]));
                        }
                    }
                    let names: Option<Vec<String>> =
                        read(".sha256").map(|t| t.lines().map(|l| l.split_once("  ").unwrap().1.to_string()).collect());
                    let theirs: Option<Vec<String>> = want["sha256_names"]
                        .as_array()
                        .map(|a| a.iter().map(|v| v.as_str().unwrap().to_string()).collect());
                    if names != theirs {
                        problem(format!("checksums for {names:?}, the library's for {theirs:?}"));
                    }
                    if let Some(text) = read(".sha256") {
                        for line in text.lines() {
                            let (sum, file) = line.split_once("  ").unwrap();
                            use sha2::{Digest, Sha256};
                            let bytes = std::fs::read(dir.join(file)).unwrap();
                            let have: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
                            if have != sum {
                                problem(format!("{file}'s checksum is not its bytes'"));
                            }
                        }
                    }
                    let mut files: Vec<(String, bool)> = std::fs::read_dir(&dir)
                        .unwrap()
                        .flatten()
                        .map(|e| {
                            (
                                e.file_name().to_string_lossy().to_string(),
                                e.metadata().unwrap().permissions().readonly(),
                            )
                        })
                        .collect();
                    files.sort();
                    let mut theirs: Vec<(String, bool)> = want["files"]
                        .as_object()
                        .unwrap()
                        .iter()
                        .map(|(k, v)| (k.clone(), v["read_only"].as_bool().unwrap()))
                        .collect();
                    theirs.sort();
                    if files != theirs {
                        problem(format!("files {files:?} differ from {theirs:?}"));
                    }
                    let arrays = npz_digest(&std::fs::read(dir.join(format!("{label}.npz"))).unwrap()).unwrap();
                    let mine: Vec<Value> = arrays
                        .iter()
                        .map(|(n, d, s, h, f)| Value::from(vec![Value::from(n), Value::from(d), Value::from(s.clone()), Value::from(h), Value::from(*f)]))
                        .collect();
                    if Value::Array(mine.clone()) != want["arrays"] {
                        problem(format!("arrays {mine:?} differ from {}", want["arrays"]));
                    }
                }
                other => panic!("no step {other}"),
            }
        }
        let _ = scratch(name);
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(problems.is_empty(), "{} problems:\n{}", problems.len(), problems.join("\n"));
}

/// Where two texts first part, with a little of each.
fn first_difference(a: &str, b: &str) -> String {
    let at = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
    let from = at.saturating_sub(120);
    let take = |s: &str| s.chars().skip(from).take(240).collect::<String>();
    format!("written here: …{}…\nthe library's: …{}…", take(a), take(b))
}

/// The device's info as the recorder fixture's first session gives it,
/// with `firmware` in place of its own.
fn info_of(firmware: (u8, u8, u8)) -> Vec<u8> {
    let f = fixture();
    let script = f["sessions"][0]["script"].as_array().unwrap();
    let step = script.iter().find(|s| s["op"].as_str() == Some("info")).unwrap();
    let mut info = hex(&step["bytes"]);
    info[2..5].copy_from_slice(&[firmware.0, firmware.1, firmware.2]);
    info
}

/// One data packet of `n` samples of zeros on four channels at 500 a
/// second, gain code `gain_code`, with lead-off detection on or off.
fn packet(index: u32, device_time: u64, n: u8, gain_code: u8, detecting: bool) -> Vec<u8> {
    use intomind::protocol::frame::{flags, DataHeader};
    let h = DataHeader {
        flags: if detecting { flags::LEADOFF_ACTIVE } else { 0 },
        samples_lost_before: 0,
        sample_index: index,
        device_time,
        n_samples: n,
        loff_statp: 0,
        gain_code,
        rate_code: 5,
    };
    let mut bytes = vec![0u8; 20 + n as usize * 4 * 3];
    h.encode(&mut bytes).unwrap();
    bytes
}

/// One data packet of `n` samples of zeros on four channels at gain code 6
/// and the rate `rate_code` names, flagged as a break or not, with
/// `lost_before` samples lost before it, lead-off detection on or off, and
/// the lead-off bits `leadoff`.
#[allow(clippy::too_many_arguments)]
fn stream_packet(index: u32, device_time: u64, n: u8, rate_code: u8, discontinuity: bool, lost_before: u16, detecting: bool, leadoff: u8) -> Vec<u8> {
    use intomind::protocol::frame::{flags, DataHeader};
    let h = DataHeader {
        flags: (if detecting { flags::LEADOFF_ACTIVE } else { 0 }) | (if discontinuity { flags::DISCONTINUITY } else { 0 }),
        samples_lost_before: lost_before,
        sample_index: index,
        device_time,
        n_samples: n,
        loff_statp: leadoff,
        gain_code: 6,
        rate_code,
    };
    let mut bytes = vec![0u8; 20 + n as usize * 4 * 3];
    h.encode(&mut bytes).unwrap();
    bytes
}

/// A sample as the session heard it.
#[derive(Debug, Clone, Copy)]
struct Heard {
    index: u32,
    /// Whether its lead-off bits are a verdict, as the session's
    /// `leadoff_judged` says of its row.
    judged: bool,
    leadoff: u8,
    leadoff_active: bool,
    /// Whether the session announced a gap before it, which it does before
    /// the first sample of a batch alone.
    gap_before: bool,
}

/// How [`judging`] sends a stream: `n` samples at `rate` a second, `per` a
/// packet, with lead-off detection on or off, the stream started by the
/// session or not, its count restarted before the start or not, every
/// verdict `leadoff`, the samples whose index is in `dropped` lost by the
/// device, the first `skip` packets lost on the way, and the packet that
/// holds each index in `lost`.
struct Stream {
    detecting: bool,
    start: bool,
    new_epoch: bool,
    skip: usize,
    leadoff: u8,
    rate: u16,
    per: usize,
    n: u32,
    dropped: Vec<u32>,
    lost: Vec<u32>,
}

impl Default for Stream {
    fn default() -> Stream {
        Stream {
            detecting: true,
            start: true,
            new_epoch: true,
            skip: 0,
            leadoff: 0,
            rate: 500,
            per: 18,
            n: 720,
            dropped: Vec::new(),
            lost: Vec::new(),
        }
    }
}

/// What the session made of a stream [`judging`] sent.
struct Judged {
    heard: Vec<Heard>,
    /// For each sample heard, whether the bits its packet carries are a
    /// verdict, or with detection off, true.
    verdicts: Vec<bool>,
    /// What the recording the session's recorder made of it holds under
    /// `leadoff_judged`.
    recorded: Option<Vec<bool>>,
}

/// The samples the session hears of a stream sent as `how` says, from a
/// device of `firmware`, sent as the device's firmware sends them, and the
/// recording the session's recorder made of it.
///
/// The device restarts its count at every start, so a stream's first sample
/// is index 1 whether the session restarted the count before it or not. A
/// sample the device loses ends the packet under way before it, and the
/// next is flagged as a break with the loss counted. From firmware 1.4.6,
/// with detection on, the device judges each electrode over a block of
/// round(0.512 * rate) consecutive samples and gives its verdict with the
/// block's last sample. A lost sample starts the block over and leaves the
/// last verdict standing, and until the first verdict every lead-off bit
/// reads 0. On an older firmware, or with detection off, every sample's
/// bits are the verdict. Every packet carries the bits of its last sample,
/// and the first packet is flagged as a break.
fn judging(firmware: (u8, u8, u8), how: &Stream) -> Judged {
    static MADE: AtomicUsize = AtomicUsize::new(0);
    let name = format!("judging-{}", MADE.fetch_add(1, Ordering::Relaxed));
    let dir = scratch(&name);
    let mut session = Session::new();
    session.on_device_info(&info_of(firmware)).unwrap();
    session.arm_recorder();
    if how.start {
        if how.new_epoch {
            let _ = session.reset_epoch();
        }
        let _ = session.start_stream();
    }
    let judges = how.detecting && firmware >= (1, 4, 6);
    let block = (0.512 * how.rate as f64).round() as u32;
    let tick = 1_000_000 / how.rate as u64;
    let rate_code = match how.rate {
        1000 => 4,
        500 => 5,
        250 => 6,
        _ => unreachable!("a rate the device offers"),
    };
    // The samples the device keeps, each with its index, its bits and
    // whether they are a verdict, gathered into packets of consecutive
    // samples.
    let mut packets: Vec<Vec<(u32, u8, bool)>> = Vec::new();
    let (mut taken, mut verdict) = (0, false);
    for index in 1..=how.n {
        if how.dropped.contains(&index) {
            taken = 0;
            continue;
        }
        if judges {
            taken += 1;
            if taken == block {
                taken = 0;
                verdict = true;
            }
        }
        let held = (index, if verdict || !judges { how.leadoff } else { 0 }, if judges { verdict } else { !how.detecting });
        match packets.last_mut() {
            Some(last) if last.len() < how.per && last[last.len() - 1].0 == index - 1 => last.push(held),
            _ => packets.push(vec![held]),
        }
    }
    let mut heard = Vec::new();
    let mut verdicts = Vec::new();
    for (k, pk) in packets.iter().enumerate() {
        let first = pk[0].0;
        let last = pk[pk.len() - 1];
        let lost_before = if k > 0 { first - packets[k - 1][packets[k - 1].len() - 1].0 - 1 } else { 0 };
        if k < how.skip || pk.iter().any(|held| how.lost.contains(&held.0)) {
            continue;
        }
        let bytes = stream_packet(
            first,
            5_000_000 + tick * (first as u64 - 1),
            pk.len() as u8,
            rate_code,
            k == 0 || lost_before > 0,
            lost_before as u16,
            how.detecting,
            last.1,
        );
        let mut gap = false;
        for event in session.on_notification(uuid_fill::EEG_DATA, &bytes) {
            match event {
                intomind::Event::Gap(_) => gap = true,
                intomind::Event::Samples(b) => {
                    for i in 0..b.rows() {
                        heard.push(Heard {
                            index: b.index + i as u32,
                            judged: session.leadoff_judged(&b, i),
                            leadoff: b.leadoff,
                            leadoff_active: b.leadoff_active,
                            gap_before: gap && i == 0,
                        });
                        verdicts.push(last.2);
                    }
                }
                _ => {}
            }
        }
    }
    let collection = Collection::open(&dir);
    let run = Run { finished_unix: Some(1_767_225_600.0), ..Run::default() };
    session.save_recording(&collection, "judging", &Setup::default(), &run).unwrap();
    let recorded = collection.load("judging").unwrap().leadoff_judged();
    let _ = scratch(&name);
    let _ = std::fs::remove_dir_all(&dir);
    Judged { heard, verdicts, recorded }
}

fn judged_of(heard: &[Heard]) -> Vec<bool> {
    heard.iter().map(|h| h.judged).collect()
}

fn not_first(n: usize, total: usize) -> Vec<bool> {
    (0..total).map(|i| i >= n).collect()
}

#[test]
fn the_first_half_second_of_a_stream_started_with_detection_on_carries_no_verdict_from_firmware_1_4_6() {
    // From firmware 1.4.6 the device judges each electrode over a block of
    // 0.512 s of consecutive samples, 256 at 500 a second, its lead-off bits
    // read 0 until its first judgment of a stream, which comes with the
    // block's last sample, and each packet carries the bits of its last
    // sample. So the samples of a stream this session started with
    // detection on are not judged until the packet that holds the 256th, the
    // first 252 at 18 a packet, and every sample from that packet on is,
    // detection running for all of them, whether the count was restarted
    // before the start or not.
    let got = judging((1, 4, 6), &Stream::default());
    assert!(got.heard.len() == 720 && got.heard.iter().all(|h| h.leadoff_active));
    assert_eq!(judged_of(&got.heard), not_first(252, 720), "judged from the packet that holds the 256th sample, round(0.512 * 500)");
    assert_eq!(got.recorded, Some(not_first(252, 720)), "the recording says which samples carry a verdict");
    // A first packet lost on the way starts the count at the first sample
    // heard, since the device may have lost those samples itself and
    // started its block after them.
    let got = judging((1, 4, 6), &Stream { skip: 2, ..Stream::default() });
    assert_eq!([0, 251, 502].map(|i| (got.heard[i].index, got.heard[i].judged)), [(37, false), (288, false), (539, true)]);
    assert_eq!(judged_of(&got.heard), not_first(252, 684));
    let got = judging((1, 5, 0), &Stream { new_epoch: false, ..Stream::default() });
    assert_eq!(got.heard[0].index, 1);
    assert_eq!(judged_of(&got.heard), not_first(252, 720), "a count not restarted before the start");
    // Detection off, and a stream this session did not start: every
    // sample's bits are a verdict, and the recording holds no such flags.
    for (detecting, start) in [(false, true), (true, false)] {
        let got = judging((1, 4, 6), &Stream { detecting, start, ..Stream::default() });
        assert!(got.heard.len() == 720 && got.heard.iter().all(|h| h.judged), "{detecting} {start}");
        assert_eq!(got.recorded, None, "{detecting} {start}");
    }
}

/// Each rate the device offers, and the samples of its block there,
/// round(0.512 * rate).
const BLOCKS: [(u16, u32); 3] = [(250, 128), (500, 256), (1000, 512)];

#[test]
fn the_first_lead_off_verdict_of_a_stream_is_marked_from_the_packet_that_holds_it_at_every_rate_the_device_offers() {
    // The device's first verdict of a stream comes with the last sample of
    // its first block, the 128th, 256th or 512th, index 128, 256 or 512
    // since a stream's first sample is index 1, and every packet carries
    // the bits of its last sample. A stream this session started with
    // detection on is marked judged from the packet that holds that sample
    // and not before, whether its count was restarted before the start or
    // not: at 1 sample a packet the first 127, 255 or 511 samples are not
    // judged, and at 18 a packet the samples of every packet before the one
    // that holds it. Every sample marked judged carries the verdict, and
    // none before it does.
    for (rate, block) in BLOCKS {
        for per in [1, 18] {
            for new_epoch in [true, false] {
                let said = format!("rate {rate}, {per} a packet, new epoch {new_epoch}");
                let total = 3 * block as usize;
                let got = judging((1, 4, 6), &Stream { rate, per, n: 3 * block, new_epoch, leadoff: 0b0100, ..Stream::default() });
                assert!(got.heard.len() == total && got.heard[0].index == 1, "{said}");
                let first = (block as usize - 1) / per * per;
                assert_eq!(got.verdicts, not_first(first, total), "{said}");
                assert_eq!(judged_of(&got.heard), got.verdicts, "{said}");
                assert_eq!(got.recorded, Some(got.verdicts.clone()), "{said}");
                assert!(got.heard[first].index == first as u32 + 1 && (per > 1 || got.heard[first].index == block), "{said}");
                assert!(got.heard[..first].iter().all(|h| h.leadoff == 0), "{said}");
                assert!(got.heard[first..].iter().all(|h| h.leadoff == 0b0100), "{said}");
            }
        }
    }
}

#[test]
fn a_break_before_the_first_lead_off_verdict_of_a_stream_starts_the_count_again_and_one_after_it_takes_nothing_from_what_is_marked_judged() {
    // The device starts its block over at a sample it loses, and gives no
    // verdict until a whole block of consecutive samples has ended, its bits
    // reading 0 meanwhile. At each rate the device offers, a sample the
    // device loses before its first verdict of a stream, the stream's first
    // sample among them, starts the count again at the sample after it: no
    // sample whose bits still read 0 is marked judged, and every sample
    // whose bits are a verdict is. A packet lost on the way before the first
    // verdict is a break a host cannot tell from one the device made, and
    // the count starts again after it too: the samples are marked judged
    // from the packet that holds the block's last sample counted from the
    // first sample after it, and the device's verdict came no later. After
    // the first verdict the device keeps its verdict through a break, and a
    // sample it loses or a packet lost on the way takes nothing from what is
    // marked judged.
    for (rate, block) in BLOCKS {
        let n = 4 * block;
        let step = (block - 1) / 18 * 18;
        for dropped in [vec![1], vec![40], vec![40, 41, 90]] {
            let said = format!("rate {rate}, the device lost {dropped:?}");
            let got = judging((1, 4, 6), &Stream { rate, n, dropped: dropped.clone(), leadoff: 0b0100, ..Stream::default() });
            let judged = judged_of(&got.heard);
            assert_eq!(judged, got.verdicts, "{said}");
            assert_eq!(got.recorded, Some(got.verdicts.clone()), "{said}");
            let first = judged.iter().position(|&j| j).unwrap();
            assert_eq!(got.heard[first].index, dropped[dropped.len() - 1] + 1 + step, "{said}");
            assert!(got.heard[..first].iter().all(|h| h.leadoff == 0), "{said}");
            assert!(got.heard[first..].iter().all(|h| h.leadoff == 0b0100), "{said}");
        }
        let said = format!("rate {rate}, a packet lost on the way");
        let got = judging((1, 4, 6), &Stream { rate, n, lost: vec![40], leadoff: 0b0100, ..Stream::default() });
        let judged = judged_of(&got.heard);
        assert!(got.heard[36].index == 55 && got.heard[36].gap_before, "{said}");
        let first = judged.iter().position(|&j| j).unwrap();
        assert!(got.heard[first].index == 55 + step && judged[first..].iter().all(|&j| j), "{said}");
        assert!(got.verdicts.iter().position(|&v| v).unwrap() <= first, "{said}");
        assert!(got.verdicts.iter().zip(&judged).all(|(&v, &j)| v || !j), "{said}");
        let said = format!("rate {rate}, after the first verdict");
        let got = judging((1, 4, 6), &Stream { rate, n, dropped: vec![2 * block + 7], lost: vec![3 * block], leadoff: 0b0100, ..Stream::default() });
        let judged = judged_of(&got.heard);
        assert_eq!(judged, got.verdicts, "{said}");
        assert_eq!(judged, not_first(step as usize, got.heard.len()), "{said}");
    }
}

#[test]
fn firmware_older_than_1_4_6_gives_no_verdict_with_detection_on_and_a_contact_check_on_it_is_refused() {
    // Every sample sent with detection on carries no verdict, whoever
    // started the stream, and the recording says so of each.
    for (start, new_epoch) in [(true, true), (false, true), (true, false)] {
        let got = judging((1, 4, 5), &Stream { start, new_epoch, ..Stream::default() });
        assert!(got.heard.len() == 720 && got.heard.iter().all(|h| !h.judged), "{start} {new_epoch}");
        assert_eq!(got.recorded, Some(vec![false; 720]), "{start} {new_epoch}");
    }
    // With detection off every sample's bits are a verdict, as on any
    // firmware, and the recording holds no such flags.
    let got = judging((1, 4, 5), &Stream { detecting: false, ..Stream::default() });
    assert!(got.heard.len() == 720 && got.heard.iter().all(|h| h.judged));
    assert_eq!(got.recorded, None);

    // Turning detection on is refused before any command is made, with the
    // error a device without the capability gets; turning it off is not.
    let mut session = Session::new();
    session.on_device_info(&info_of((1, 4, 5))).unwrap();
    assert!(session.can(capability::LEADOFF), "the device claims lead-off");
    let refused = session.set_leadoff(true).unwrap_err();
    assert_eq!(refused, intomind::Error::NotCapable("leadoff"));
    // Nothing was asked of the device: an answer to the opcode changes
    // nothing this session holds.
    session.on_notification(uuid_fill::CONTROL_RSP, &[0x30, 0x00]);
    assert_eq!(session.leadoff_setting(), None, "a refused request is never taken as accepted");
    let off = session.set_leadoff(false).unwrap();
    assert_eq!(off.bytes, vec![0x30, 0x00]);
    session.on_notification(uuid_fill::CONTROL_RSP, &[0x30, 0x00]);
    assert_eq!(session.leadoff_setting(), Some(false));
    // From 1.4.6 the device takes it.
    let mut session = Session::new();
    session.on_device_info(&info_of((1, 4, 6))).unwrap();
    assert_eq!(session.set_leadoff(true).unwrap().bytes, vec![0x30, 0x01]);
}

#[test]
fn a_batch_is_built_from_the_fields_it_was_published_with_and_the_session_says_which_rows_carry_a_verdict() {
    // A program builds a Batch from the ten fields it was published with.
    let made = intomind::Batch {
        index: 0,
        device_time: 5_000_000,
        channels: 4,
        counts: vec![0; 4 * 18],
        leadoff: 0,
        leadoff_active: true,
        gain: 6,
        sample_rate_hz: 500,
        mode: 0,
        synthetic: false,
    };
    assert_eq!(Session::new().judged_from(&made), 0, "a session that has read no device info");
    let batch = |session: &mut Session, bytes: &[u8]| {
        session.on_notification(uuid_fill::EEG_DATA, bytes).into_iter().find_map(|e| match e {
            intomind::Event::Samples(b) => Some(b),
            _ => None,
        })
    };
    // The `k`-th packet of 18 samples of a stream at 500 a second whose
    // first packet's device time is `at`, 2000 ticks a sample.
    let sent = |session: &mut Session, k: u32, at: u64| {
        batch(session, &stream_packet(18 * k + 1, at + 2000 * 18 * k as u64, 18, 5, k == 0, 0, true, 0)).unwrap()
    };
    let mut session = Session::new();
    session.on_device_info(&info_of((1, 4, 6))).unwrap();
    let _ = session.reset_epoch();
    let _ = session.start_stream();
    // The device's first verdict comes with the stream's 256th sample, in
    // its 15th packet: no row before that packet carries a verdict, and
    // every row from it on does.
    let first = sent(&mut session, 0, 5_000_000);
    let mut kept = vec![first.clone()];
    for k in 1..16 {
        kept.push(sent(&mut session, k, 5_000_000));
    }
    let (before, later) = (&kept[13], &kept[14]);
    assert_eq!((before.index, later.index), (235, 253));
    let rows = |session: &Session| kept.iter().map(|b| session.judged_from(b)).collect::<Vec<_>>();
    let mut want = vec![18; 14];
    want.extend([0, 0]);
    assert_eq!(rows(&session), want);
    assert!(!session.leadoff_judged(&first, 17) && !session.leadoff_judged(before, 17) && session.leadoff_judged(later, 0));
    // A second stream: its first samples carry no verdict, and a batch kept
    // from the first is still answered as it was when it arrived.
    let _ = session.reset_epoch();
    let _ = session.start_stream();
    let next = sent(&mut session, 0, 5_000_000 + 2000 * 18 * 100);
    assert_eq!(session.judged_from(&next), 18);
    assert_eq!(rows(&session), want);
}

#[test]
fn a_recording_that_kept_no_packet_carries_the_gain_the_device_last_stated_and_never_a_runs() {
    let dir = scratch("stated-gain");
    let collection = Collection::open(&dir);
    let mut session = Session::new();
    session.on_device_info(&info_of((1, 4, 6))).unwrap();
    assert_eq!(session.gain(), 6, "before the device has stated a gain");
    let ask = session.set_gain(12).unwrap();
    assert_eq!(session.gain(), 6, "a gain asked for and not yet accepted");
    session.on_notification(uuid_fill::CONTROL_RSP, &[ask.bytes[0], 0x00]);
    assert_eq!(session.gain(), 12, "a gain the device accepted");
    let mut status = vec![0u8; 12];
    status[2] = 6; // gain code 6, 24
    session.on_notification(uuid_fill::STATUS, &status);
    assert_eq!(session.gain(), 24, "the gain a Status states");
    session.arm_recorder();
    session.disarm_recorder();
    let mut extra = Value::object();
    extra.set("gain", Value::int(8));
    let run = Run { extra, finished_unix: Some(1_767_225_600.0), ..Run::default() };
    let meta = session.save_recording(&collection, "nothing-kept", &Setup::default(), &run).unwrap();
    assert_eq!(meta["gain"], Value::int(24));
    session.arm_recorder();
    session.on_notification(uuid_fill::EEG_DATA, &packet(0, 5_000_000, 18, 1, false));
    assert_eq!(session.gain(), 2, "the gain a packet carries");
    let meta = session.save_recording(&collection, "one-packet", &Setup::default(), &run).unwrap();
    assert_eq!(meta["gain"], Value::int(2));
    let _ = scratch("stated-gain");
    let _ = std::fs::remove_dir_all(&dir);
}
