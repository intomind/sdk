//! The recordings fixture: every export this crate writes, held to what the
//! Python library wrote from the same synthetic recordings, and every number
//! and JSON document held to how Python formats it.
//!
//! `contract/recordings/make.py` emits the fixture from the Python library.
//! bdf, edf, csv and tsv must match it byte for byte. npz must hold the same
//! arrays, element for element. Every summary must carry the same keys and
//! values, and every refusal the same words.

use std::path::{Path, PathBuf};

use crate::export;
use crate::pyfmt;
use crate::pyjson::{self, Value};
use crate::recording::{read_npz, Collection};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../contract/recordings");

fn fixture(name: &str) -> Value {
    let text = std::fs::read_to_string(Path::new(ROOT).join(name)).expect("the fixture is in the repository");
    pyjson::loads(&text).expect("the fixture is JSON")
}

fn f64_of(hex: &Value) -> f64 {
    let s = hex.as_str().expect("a float's bits, in hex");
    let bytes: Vec<u8> = (0..16).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect();
    f64::from_le_bytes(bytes.try_into().unwrap())
}

fn rows(v: &Value, key: &str) -> Vec<Vec<Value>> {
    match v.get(key) {
        Some(Value::Array(a)) => a
            .iter()
            .map(|r| match r {
                Value::Array(x) => x.clone(),
                _ => panic!("a row"),
            })
            .collect(),
        _ => panic!("no {key} in the fixture"),
    }
}

#[test]
fn floats_are_written_as_python_writes_them() {
    let n = fixture("numbers.json");
    let mut checked = 0;
    for r in rows(&n, "repr") {
        assert_eq!(pyfmt::repr(f64_of(&r[0])), r[1].as_str().unwrap(), "repr of {:?}", f64_of(&r[0]));
        checked += 1;
    }
    for r in rows(&n, "fixed") {
        let p = r[1].as_i64().unwrap() as usize;
        assert_eq!(pyfmt::fixed(f64_of(&r[0]), p), r[2].as_str().unwrap(), "{:?} to {p} places", f64_of(&r[0]));
        checked += 1;
    }
    for r in rows(&n, "general") {
        let p = r[1].as_i64().unwrap() as usize;
        assert_eq!(pyfmt::general(f64_of(&r[0]), p), r[2].as_str().unwrap(), "{:?} as .{p}g", f64_of(&r[0]));
        checked += 1;
    }
    for r in rows(&n, "quantize") {
        let (x, p) = (f64_of(&r[0]), r[1].as_i64().unwrap() as u32);
        assert_eq!(pyfmt::quantize(x, p, pyfmt::Toward::Floor).as_deref(), r[2].as_str(), "{x:?} floor at {p}");
        assert_eq!(pyfmt::quantize(x, p, pyfmt::Toward::Ceiling).as_deref(), r[3].as_str(), "{x:?} ceiling at {p}");
        checked += 1;
    }
    assert!(checked > 10_000, "{checked}");
}

#[test]
fn json_reads_and_writes_as_pythons_does() {
    let n = fixture("numbers.json");
    for r in rows(&n, "json") {
        let v = pyjson::loads(r[0].as_str().unwrap()).unwrap();
        assert_eq!(pyjson::dumps_sorted(&v), r[1].as_str().unwrap());
        assert_eq!(pyjson::dumps_indented(&v, 2), r[2].as_str().unwrap());
    }
}

/// A folder of its own for one test's exports.
fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("intomind-capture-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A summary key for key, exactly: integers as integers, floats bit for
/// bit, lists in order.
fn same(want: &Value, got: &serde_json::Value) -> bool {
    use serde_json::Value as J;
    match (want, got) {
        (Value::Null, J::Null) => true,
        (Value::Bool(a), J::Bool(b)) => a == b,
        (Value::Int(a), J::Number(b)) => {
            b.is_i64() && &b.as_i64().unwrap().to_string() == a || b.is_u64() && &b.as_u64().unwrap().to_string() == a
        }
        (Value::Float(a), J::Number(b)) => b.as_f64() == Some(*a) && !(b.is_i64() || b.is_u64()),
        (Value::Str(a), J::String(b)) => a == b,
        (Value::Array(a), J::Array(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same(x, y)),
        (Value::Object(a), J::Object(b)) => {
            a.len() == b.len() && a.iter().all(|(k, v)| b.get(k).is_some_and(|y| same(v, y)))
        }
        _ => false,
    }
}

fn sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn every_export_is_the_python_librarys() {
    let expected = fixture("expected.json");
    let Some(Value::Object(recordings)) = expected.get("recordings") else { panic!("no recordings") };
    let collection = Collection::open(Path::new(ROOT).join("captures"));
    let out = scratch("exports");
    let mut problems = Vec::new();
    let mut checked = 0;
    for (label, by_format) in recordings {
        let Value::Object(by_format) = by_format else { panic!() };
        for (fmt, want) in by_format {
            let path = out.join(format!("{label}.{fmt}"));
            let got = export::export(&collection, label, fmt, Some(&path));
            checked += 1;
            if let Some(refused) = want.get("refused") {
                match got {
                    Ok(_) => {
                        problems.push(format!("{label} {fmt}: exported where Python refused: {}", refused.py_str()))
                    }
                    Err(e) if e.to_string() != refused.py_str() => problems.push(format!(
                        "{label} {fmt}: refused as {:?}, Python said {:?}",
                        e.to_string(),
                        refused.py_str()
                    )),
                    Err(_) => {}
                }
                continue;
            }
            let summary = match got {
                Ok(mut s) => {
                    s.as_object_mut().unwrap().remove("path");
                    s
                }
                Err(e) => {
                    problems.push(format!("{label} {fmt}: refused where Python exported: {e}"));
                    continue;
                }
            };
            if !same(want.field("summary"), &summary) {
                problems.push(format!(
                    "{label} {fmt}: summary {summary} differs from {}",
                    pyjson::dumps_sorted(want.field("summary"))
                ));
            }
            let bytes = std::fs::read(&path).unwrap();
            if let Some(h) = want.get("sha256") {
                if sha256(&bytes) != h.py_str() {
                    problems.push(format!("{label} {fmt}: the file is not the bytes Python wrote"));
                }
            }
            if let Some(Value::Array(arrays)) = want.get("arrays") {
                let mine = read_npz(&bytes).unwrap();
                let digest: Vec<String> = mine
                    .iter()
                    .map(|(k, a)| {
                        // numpy's `tobytes()`: the elements after the header.
                        let raw = crate::npy::write(a);
                        format!("{k} {} {:?} {}", a.data.descr(), a.shape, sha256(&raw[header_len(&raw)..]))
                    })
                    .collect();
                let theirs: Vec<String> = arrays
                    .iter()
                    .map(|r| {
                        let Value::Array(r) = r else { panic!() };
                        let shape: Vec<usize> = match &r[2] {
                            Value::Array(s) => s.iter().map(|d| d.as_i64().unwrap() as usize).collect(),
                            _ => panic!(),
                        };
                        format!("{} {} {:?} {}", r[0].py_str(), r[1].py_str(), shape, r[3].py_str())
                    })
                    .collect();
                if digest != theirs {
                    problems.push(format!("{label} {fmt}: arrays {digest:?} differ from {theirs:?}"));
                }
            }
        }
    }
    let _ = std::fs::remove_dir_all(&out);
    assert!(checked >= 200, "{checked}");
    assert!(problems.is_empty(), "{} of {checked} differ:\n{}", problems.len(), problems.join("\n"));
}

/// Where an `.npy` file's data starts.
fn header_len(npy: &[u8]) -> usize {
    match npy[6] {
        1 => 10 + u16::from_le_bytes([npy[8], npy[9]]) as usize,
        _ => 12 + u32::from_le_bytes([npy[8], npy[9], npy[10], npy[11]]) as usize,
    }
}
