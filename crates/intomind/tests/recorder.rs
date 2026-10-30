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
