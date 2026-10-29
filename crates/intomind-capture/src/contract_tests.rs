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
use crate::record;
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
            if let Some(m) = want.get("mat") {
                let mine = mat_digest(&bytes);
                let Some(Value::Array(vars)) = m.get("variables") else { panic!("no variables") };
                let theirs = (
                    m.field("header_tail").py_str(),
                    vars.iter()
                        .map(|v| {
                            let Value::Array(v) = v else { panic!() };
                            let shape: Vec<u64> = match &v[2] {
                                Value::Array(s) => s.iter().map(|d| d.as_i64().unwrap() as u64).collect(),
                                _ => panic!(),
                            };
                            let attrs: Vec<String> = match &v[4] {
                                Value::Array(a) => a
                                    .iter()
                                    .map(|a| {
                                        let Value::Array(a) = a else { panic!() };
                                        format!("{} {} {}", a[0].py_str(), a[1].py_str(), a[2].py_str())
                                    })
                                    .collect(),
                                _ => panic!(),
                            };
                            format!("{} {} {:?} {} {:?}", v[0].py_str(), v[1].py_str(), shape, v[3].py_str(), attrs)
                        })
                        .collect::<Vec<_>>(),
                );
                if mine != theirs {
                    problems.push(format!("{label} {fmt}: {mine:?} differs from {theirs:?}"));
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

/// A MATLAB file as h5py reports it: the end of MATLAB's header, then each
/// variable's name, element type, shape, the sha256 of its elements and its
/// attributes, in the order h5py lists them.
///
/// Read by following the HDF5 structures from the superblock, as the HDF5
/// specification lays them out, never from where this crate's writer puts
/// them. It reads what a MATLAB export can hold: a version 0 superblock, a
/// symbol table group, version 1 object headers, and contiguous or compact
/// data. Anything else fails the test rather than being skipped.
fn mat_digest(file: &[u8]) -> (String, Vec<String>) {
    let u16_at = |p: usize| u16::from_le_bytes(file[p..p + 2].try_into().unwrap()) as usize;
    let u32_at = |p: usize| u32::from_le_bytes(file[p..p + 4].try_into().unwrap()) as usize;
    let u64_at = |p: usize| u64::from_le_bytes(file[p..p + 8].try_into().unwrap());
    let tail: String = file[116..128].iter().map(|b| format!("{b:02x}")).collect();

    // HDF5 looks for its superblock at 0, 512, 1024 and so on.
    let sb = (0..)
        .map(|i| 512 * i)
        .take_while(|&p| p + 96 <= file.len())
        .find(|&p| &file[p..p + 8] == b"\x89HDF\r\n\x1a\n")
        .expect("an HDF5 superblock");
    assert_eq!(file[sb + 8], 0, "superblock version");
    assert_eq!((file[sb + 13], file[sb + 14]), (8, 8), "the size of addresses and lengths");
    let base = u64_at(sb + 24) as usize;
    let at = |a: u64| base + a as usize;

    // An object header's messages, as (type, body).
    let messages = |h: usize| -> Vec<(u16, &[u8])> {
        assert_eq!(file[h], 1, "object header version");
        let (n, size) = (u16_at(h + 2), u32_at(h + 8));
        let (mut p, end, mut out) = (h + 16, h + 16 + size, Vec::new());
        while out.len() < n {
            assert!(p + 8 <= end, "a message past the end of its header");
            let (kind, len) = (u16_at(p) as u16, u16_at(p + 2));
            assert_ne!(kind, 0x0010, "a continuation, which a MATLAB export does not need");
            out.push((kind, &file[p + 8..p + 8 + len]));
            p += 8 + len;
        }
        out
    };
    // A datatype as numpy names it, and its size.
    let dtype = |t: &[u8]| -> (String, usize) {
        let size = u32::from_le_bytes(t[4..8].try_into().unwrap()) as usize;
        let order = if size == 1 {
            "|"
        } else if t[1] & 1 == 0 {
            "<"
        } else {
            ">"
        };
        match t[0] & 0x0f {
            0 => (format!("{order}{}{size}", if t[1] & 0x08 != 0 { "i" } else { "u" }), size),
            1 => (format!("{order}f{size}"), size),
            3 => (format!("|S{size}"), size),
            c => panic!("datatype class {c}"),
        }
    };
    // A dataspace's dimensions.
    let dims = |d: &[u8]| -> Vec<u64> {
        assert_eq!(d[0], 1, "dataspace version");
        (0..d[1] as usize).map(|i| u64::from_le_bytes(d[8 + 8 * i..16 + 8 * i].try_into().unwrap())).collect()
    };

    // The root group's B-tree and local heap.
    let root = messages(at(u64_at(sb + 56 + 8)));
    let st = root.iter().find(|m| m.0 == 0x0011).expect("the root group's symbol table").1;
    let (btree, heap) =
        (u64::from_le_bytes(st[0..8].try_into().unwrap()), u64::from_le_bytes(st[8..16].try_into().unwrap()));
    let h = at(heap);
    assert_eq!(&file[h..h + 4], b"HEAP");
    let names = at(u64_at(h + 24));
    let name = |off: u64| -> String {
        let s = &file[names + off as usize..];
        String::from_utf8(s[..s.iter().position(|&c| c == 0).unwrap()].to_vec()).unwrap()
    };
    // Every symbol node under the B-tree, in key order.
    fn leaves(file: &[u8], at: &dyn Fn(u64) -> usize, node: u64, out: &mut Vec<u64>) {
        let p = at(node);
        assert_eq!(&file[p..p + 4], b"TREE");
        assert_eq!(file[p + 4], 0, "a group B-tree");
        let (level, used) = (file[p + 5], u16::from_le_bytes([file[p + 6], file[p + 7]]) as usize);
        for i in 0..used {
            let c = p + 24 + 8 + 16 * i;
            let child = u64::from_le_bytes(file[c..c + 8].try_into().unwrap());
            if level == 0 {
                out.push(child);
            } else {
                leaves(file, at, child, out);
            }
        }
    }
    let mut snods = Vec::new();
    leaves(file, &at, btree, &mut snods);

    let mut vars = Vec::new();
    for s in snods {
        let p = at(s);
        assert_eq!(&file[p..p + 4], b"SNOD");
        for e in 0..u16_at(p + 6) {
            let entry = p + 8 + 40 * e;
            let var = name(u64_at(entry));
            let msgs = messages(at(u64_at(entry + 8)));
            let body = |kind: u16| msgs.iter().find(|m| m.0 == kind).map(|m| m.1).expect("a dataset message");
            let shape = dims(body(0x0001));
            let (dt, _) = dtype(body(0x0003));
            let layout = body(0x0008);
            assert_eq!(layout[0], 3, "layout version");
            let data: &[u8] = match layout[1] {
                0 => &layout[4..4 + u16::from_le_bytes([layout[2], layout[3]]) as usize],
                1 => {
                    let (a, n) = (
                        u64::from_le_bytes(layout[2..10].try_into().unwrap()),
                        u64::from_le_bytes(layout[10..18].try_into().unwrap()),
                    );
                    if a == u64::MAX {
                        &[]
                    } else {
                        &file[at(a)..at(a) + n as usize]
                    }
                }
                c => panic!("layout class {c}"),
            };
            let mut attrs: Vec<String> = msgs
                .iter()
                .filter(|m| m.0 == 0x000C)
                .map(|m| {
                    let a = m.1;
                    assert_eq!(a[0], 1, "attribute version");
                    let (nlen, tlen, slen) = (
                        u16::from_le_bytes([a[2], a[3]]) as usize,
                        u16::from_le_bytes([a[4], a[5]]) as usize,
                        u16::from_le_bytes([a[6], a[7]]) as usize,
                    );
                    let pad = |n: usize| n.div_ceil(8) * 8;
                    let aname = String::from_utf8(a[8..8 + nlen - 1].to_vec()).unwrap();
                    let t = &a[8 + pad(nlen)..8 + pad(nlen) + tlen];
                    let space = &a[8 + pad(nlen) + pad(tlen)..];
                    assert_eq!(space[1], 0, "a scalar attribute");
                    let v = &a[8 + pad(nlen) + pad(tlen) + pad(slen)..];
                    let (adt, size) = dtype(t);
                    let v = &v[..size];
                    let value = match t[0] & 0x0f {
                        // numpy drops a fixed string's trailing nulls.
                        3 => py_bytes(&v[..v.iter().rposition(|&c| c != 0).map_or(0, |i| i + 1)]),
                        0 if t[1] & 0x08 != 0 => {
                            let mut b = [if v[size - 1] & 0x80 != 0 { 0xff } else { 0 }; 8];
                            b[..size].copy_from_slice(v);
                            i64::from_le_bytes(b).to_string()
                        }
                        0 => {
                            let mut b = [0u8; 8];
                            b[..size].copy_from_slice(v);
                            u64::from_le_bytes(b).to_string()
                        }
                        c => panic!("an attribute of datatype class {c}"),
                    };
                    format!("{aname} {adt} {value}")
                })
                .collect();
            attrs.sort();
            vars.push(format!("{var} {dt} {shape:?} {} {attrs:?}", sha256(data)));
        }
    }
    (tail, vars)
}

/// Python's `repr` of a bytes object.
fn py_bytes(b: &[u8]) -> String {
    let q = if b.contains(&b'\'') && !b.contains(&b'"') { '"' } else { '\'' };
    let mut s = format!("b{q}");
    for &c in b {
        match c {
            b'\\' => s.push_str("\\\\"),
            b'\t' => s.push_str("\\t"),
            b'\n' => s.push_str("\\n"),
            b'\r' => s.push_str("\\r"),
            c if c as char == q => {
                s.push('\\');
                s.push(q);
            }
            0x20..=0x7e => s.push(c as char),
            c => s.push_str(&format!("\\x{c:02x}")),
        }
    }
    s.push(q);
    s
}

fn recorder_fixture() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contract/recorder/sessions.json");
    pyjson::loads(&std::fs::read_to_string(path).expect("the recorder fixture is in the repository")).expect("JSON")
}

#[test]
fn where_the_electrodes_went_is_recorded_as_the_library_records_it() {
    let f = recorder_fixture();
    let Some(Value::Array(cases)) = f.get("montage") else { panic!("no montage cases") };
    assert!(cases.len() >= 30, "{}", cases.len());
    let mut problems = Vec::new();
    for c in cases {
        let name = c.field("name").py_str();
        let channels = c.field("channels").as_i64().map(|n| n as u64);
        let got = record::montage_fields(c.field("site"), channels, c.field("layout"), c.field("place"));
        match (got, c.get("result"), c.get("raises")) {
            (Ok(v), Some(want), _) if pyjson::dumps(&v) == pyjson::dumps(want) => {}
            (Ok(v), Some(want), _) => {
                problems.push(format!("{name}: {} differs from {}", pyjson::dumps(&v), pyjson::dumps(want)))
            }
            (Err(e), _, Some(want)) => {
                problems.push(format!("{name}: raised {e:?}, the library raised {}", want.py_str()))
            }
            (got, _, _) => problems.push(format!("{name}: {got:?} where the library did otherwise")),
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

#[test]
fn a_chain_is_described_as_the_library_describes_it() {
    use intomind_protocol::pipeline::{Stage, MAX_PARAMS};
    let f = recorder_fixture();
    let Some(Value::Array(chains)) = f.get("chains") else { panic!("no chains") };
    for c in chains {
        let Value::Array(stages) = c.field("stages") else { panic!() };
        let stages: Vec<Stage> = stages
            .iter()
            .map(|s| {
                let Value::Array(p) = s.field("params") else { panic!() };
                let mut params = [0u16; MAX_PARAMS];
                for (i, x) in p.iter().enumerate() {
                    params[i] = x.as_i64().unwrap() as u16;
                }
                Stage { kind: s.field("kind").as_i64().unwrap() as u8, n_params: p.len() as u8, params }
            })
            .collect();
        let rate = c.field("rate").as_i64().map(|r| r as u32);
        assert_eq!(record::describe_chain(&stages, rate), c.field("description").py_str(), "{}", pyjson::dumps(c));
    }
}
