//! Six ways out of a recording, each written as the Python library writes
//! it, with the same summary.
//!
//! Every summary carries the same six keys, `label`, `path`, `format`,
//! `channels`, `samples` and `sample_rate_hz`, and then what that format
//! cost on the file just written. A recording that fails verification is
//! not exported at all. An export can always be made again, and the
//! recording it came from cannot.
//!
//! `bdf`, `edf`, `csv` and `tsv` are byte for byte the files the Python
//! library writes from the same recording. `npz` and `mat` hold the same
//! arrays and fields, and differ only in what their containers stamp: the
//! time a member was written, and the order a library lays out its records.
//! The fixtures in `contract/recordings` hold both claims.

use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value as J};

use crate::edf::{self, Width};
use crate::hdf5;
use crate::npy::{self, Array, Data};
use crate::pyfmt;
use crate::pyjson::{self, Value};
use crate::recording::{error, Collection, Error, Recording};
use crate::zip;

/// Every format, in the order a program may offer them: the record first,
/// then the two clinical containers, then text, then MATLAB.
pub const FORMATS: [&str; 6] = ["npz", "edf", "bdf", "csv", "tsv", "mat"];

/// Channels the lead-off status covers: one byte per packet, one bit per
/// channel.
const LEADOFF_CHANNELS: usize = 8;

fn suffix(fmt: &str) -> &'static str {
    match fmt {
        "npz" => ".npz",
        "edf" => ".edf",
        "bdf" => ".bdf",
        "csv" => ".csv",
        "tsv" => ".tsv",
        _ => ".mat",
    }
}

/// Export `label` from `collection` as `format`, into `path`: a file, a
/// folder to put it in, or, when None, `exports` beside the recordings
/// folder. The summary is the Python library's, key for key.
pub fn export(collection: &Collection, label: &str, format: &str, path: Option<&Path>) -> Result<J, Error> {
    let name = format.trim().to_lowercase();
    let name = name.trim_start_matches('.');
    if !FORMATS.contains(&name) {
        return Err(error(format!(
            "unknown export format {}. Known formats: {}",
            Value::Str(format.to_string()).py_repr(),
            FORMATS.join(", ")
        )));
    }
    // Refuse, load, and decide where the file goes, in that order.
    let v = collection.verify(label)?;
    if !v.ok {
        return Err(error(format!("refusing to export {label}: {}", v.problems.join("; "))));
    }
    let cap = collection.load(label)?;
    let out = path_for(collection, label, path, name)?;
    let problems = v.problems;
    match name {
        "npz" => export_npz(&cap, &out, &problems),
        "edf" => export_edf_family(collection, &cap, &out, &problems, &edf::EDF),
        "bdf" => export_edf_family(collection, &cap, &out, &problems, &edf::BDF),
        "csv" => export_delimited(&cap, &out, &problems, "csv", ','),
        "tsv" => export_delimited(&cap, &out, &problems, "tsv", '\t'),
        _ => export_mat(&cap, &out, &problems),
    }
}

fn path_for(collection: &Collection, label: &str, path: Option<&Path>, fmt: &str) -> Result<PathBuf, Error> {
    let file = format!("{}{}", crate::recording::check_label(label)?, suffix(fmt));
    let mk = |d: &Path| std::fs::create_dir_all(d).map_err(|e| error(format!("{}: {e}", d.display())));
    match path {
        None => {
            let d = collection.dir().parent().unwrap_or(Path::new(".")).join("exports");
            mk(&d)?;
            Ok(d.join(file))
        }
        Some(p) if p.is_dir() => Ok(p.join(file)),
        Some(p) => {
            if let Some(parent) = p.parent().filter(|d| !d.as_os_str().is_empty()) {
                mk(parent)?;
            }
            Ok(p.to_path_buf())
        }
    }
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    std::fs::write(path, bytes).map_err(|e| error(format!("{}: {e}", path.display())))
}

/// A float as a summary carries it. A summary holds no NaN in practice, and
/// one that did is written as null rather than as invalid JSON.
fn f(x: f64) -> J {
    serde_json::Number::from_f64(x).map(J::Number).unwrap_or(J::Null)
}

/// The six keys every format answers with, and the problems verification
/// reported, then the format's own.
fn summary(cap: &Recording, path: &Path, fmt: &str, samples: usize, problems: &[String], extra: Vec<(&str, J)>) -> J {
    let mut m = Map::new();
    m.insert("label".into(), json!(cap.label()));
    m.insert("path".into(), json!(path.display().to_string()));
    m.insert("format".into(), json!(fmt));
    m.insert("channels".into(), json!(cap.channels()));
    m.insert("samples".into(), json!(samples));
    m.insert("sample_rate_hz".into(), f(cap.fs()));
    m.insert("provenance_problems".into(), json!(problems));
    for (k, v) in extra {
        m.insert(k.into(), v);
    }
    J::Object(m)
}

fn column(cap: &Recording, c: usize) -> String {
    if cap.synthetic() {
        format!("ch{}_synthetic_uv", c + 1)
    } else {
        format!("ch{}_uv", c + 1)
    }
}

/// Every per-sample array except the microvolts, each only where it lines
/// up with `n` samples, and a sentence for each one left out. A recording
/// made before detection was kept has no `leadoff_active`, and `contact`
/// says so instead of a sentence here.
fn per_sample(cap: &Recording, n: usize) -> (Vec<(&'static str, Array)>, Vec<String>) {
    let mut found = Vec::new();
    let mut omitted = Vec::new();
    let all: [(&'static str, Option<&Array>); 6] = [
        ("index", Some(&cap.index)),
        ("device_ticks", cap.device_ticks.as_ref()),
        ("host_time", Some(&cap.host_time)),
        ("leadoff", cap.leadoff.as_ref()),
        ("gap_before", cap.gap_before.as_ref()),
        ("leadoff_active", cap.leadoff_active.as_ref()),
    ];
    for (name, a) in all {
        match a {
            None if name == "leadoff_active" => {}
            None => omitted.push(format!("{name}: the capture does not carry it")),
            Some(a) => {
                let got = if a.shape.is_empty() { 0 } else { a.shape[0] };
                if a.shape.is_empty() || got != n {
                    omitted.push(format!("{name}: {got} values for {n} samples, so it cannot be lined up with them"));
                } else {
                    found.push((name, a.clone()));
                }
            }
        }
    }
    (found, omitted)
}

fn find<'a>(found: &'a [(&'static str, Array)], name: &str) -> Option<&'a Array> {
    found.iter().find(|(n, _)| *n == name).map(|(_, a)| a)
}

/// Whether electrode contact was checked: "checked", "partly checked",
/// "not checked", or "not known" where the recording does not say whether
/// detection was running.
fn contact_state(leadoff: Option<&Array>, active: Option<&Array>) -> &'static str {
    let (Some(_), Some(active)) = (leadoff, active) else { return "not known" };
    let n = active.data.len();
    let on = (0..n).filter(|&i| active.data.bool_at(i)).count();
    if on == 0 {
        "not checked"
    } else if on == n {
        "checked"
    } else {
        "partly checked"
    }
}

/// Decimal places that carry one count with room to spare.
fn decimals(lsb_uv: f64) -> usize {
    if lsb_uv == 0.0 || lsb_uv <= 0.0 || !lsb_uv.is_finite() {
        return 6;
    }
    (-(lsb_uv / 100.0).log10()).ceil().clamp(3.0, 12.0) as usize
}

fn uv_all(cap: &Recording) -> Result<Vec<Vec<f64>>, Error> {
    (0..cap.channels()).map(|c| cap.uv(c)).collect()
}

// ---------------------------------------------------------------- npz

fn export_npz(cap: &Recording, out: &Path, problems: &[String]) -> Result<J, Error> {
    let n = cap.samples();
    let uv = uv_all(cap)?;
    let mut arrays: Vec<(String, Array)> = Vec::new();
    for (c, u) in uv.iter().enumerate() {
        arrays.push((column(cap, c), Array::vector(Data::F32(u.iter().map(|&v| v as f32).collect()))));
    }
    let (found, omitted) = per_sample(cap, n);
    for (name, a) in &found {
        arrays.push((name.to_string(), a.clone()));
    }
    arrays.push(("manifest_json".into(), Array::text(&pyjson::dumps_sorted(&cap.meta))));
    let members: Vec<(String, Vec<u8>)> = arrays.iter().map(|(k, a)| (format!("{k}.npy"), npy::write(a))).collect();
    write_file(out, &zip::write(&members, now()))?;

    let lsb = cap.lsb_uv()?;
    let mut err: f64 = 0.0;
    for (c, u) in uv.iter().enumerate() {
        let Data::F32(stored) = &arrays[c].1.data else { unreachable!("float32 columns") };
        let mut worst = f64::NEG_INFINITY;
        let mut nan = false;
        for (s, &v) in stored.iter().zip(u) {
            let e = (*s as f64 - v).abs();
            if e.is_nan() {
                nan = true;
            } else if e > worst {
                worst = e;
            }
        }
        let worst = if n == 0 {
            0.0
        } else if nan {
            f64::NAN
        } else {
            worst
        };
        if worst > err {
            err = worst;
        }
    }
    let recovered = err < 0.5 * lsb;
    let mut names: Vec<String> = arrays.iter().map(|(k, _)| k.clone()).collect();
    names.sort();
    Ok(summary(
        cap,
        out,
        "npz",
        n,
        problems,
        vec![
            ("lossy", json!(!recovered)),
            ("counts_recovered", json!(recovered)),
            ("max_error_uv", f(err)),
            ("lsb_uv", f(lsb)),
            ("omitted", json!(omitted)),
            ("arrays", json!(names)),
            ("contact", json!(contact_state(find(&found, "leadoff"), find(&found, "leadoff_active")))),
        ],
    ))
}

/// Seconds since 1970, for the time a container stamps on what it holds.
fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

// ---------------------------------------------------------------- csv, tsv

fn export_delimited(cap: &Recording, out: &Path, problems: &[String], fmt: &str, delimiter: char) -> Result<J, Error> {
    let n = cap.samples();
    let nch = cap.channels();
    let uv = uv_all(cap)?;
    let lsb = cap.lsb_uv()?;
    let dec = decimals(lsb);
    let (found, omitted) = per_sample(cap, n);
    let host = find(&found, "host_time");
    let ticks = find(&found, "device_ticks");
    let byte = find(&found, "leadoff");
    let active = find(&found, "leadoff_active");
    let checked = contact_state(byte, active);
    let off = if matches!(checked, "checked" | "partly checked") { nch.min(LEADOFF_CHANNELS) } else { 0 };

    let mut text = String::with_capacity(n * (16 + nch * 14));
    let d = delimiter;
    let mut header: Vec<String> = vec!["host_time_s".into(), "device_time".into()];
    header.extend((0..nch).map(|c| column(cap, c)));
    header.extend((0..off).map(|c| format!("ch{}_leadoff", c + 1)));
    text.push_str(&header.join(&d.to_string()));
    text.push('\n');
    for i in 0..n {
        if let Some(h) = host {
            text.push_str(&pyfmt::fixed(h.data.f64_at(i), 6));
        }
        text.push(d);
        if let Some(t) = ticks {
            text.push_str(&t.data.i64_at(i).to_string());
        }
        for u in &uv {
            text.push(d);
            text.push_str(&pyfmt::fixed(u[i], dec));
        }
        for c in 0..off {
            text.push(d);
            let (b, a) = (byte.expect("checked"), active.expect("checked"));
            if a.data.bool_at(i) {
                text.push_str(&((b.data.i64_at(i) >> c) & 1).to_string());
            }
        }
        text.push('\n');
    }
    write_file(out, text.as_bytes())?;

    let resolution: f64 = format!("1e-{dec}").parse().expect("a power of ten");
    let recovered = resolution / 2.0 < 0.5 * lsb;
    let omitted: Vec<String> =
        omitted.into_iter().filter(|o| o.starts_with("host_time") || o.starts_with("device_ticks")).collect();
    Ok(summary(
        cap,
        out,
        fmt,
        n,
        problems,
        vec![
            ("delimiter", json!(delimiter.to_string())),
            ("decimals", json!(dec)),
            ("resolution_uv", f(resolution)),
            ("lsb_uv", f(lsb)),
            ("counts_recovered", json!(recovered)),
            ("lossy", json!(!recovered)),
            ("omitted", json!(omitted)),
            ("contact", json!(checked)),
        ],
    ))
}

// ---------------------------------------------------------------- edf, bdf

/// What made this recording, in the words its manifest carries.
fn device_of(meta: &Value) -> String {
    for k in ["device_name", "device_id"] {
        let v = meta.field(k);
        if v.truthy() {
            return v.py_str();
        }
    }
    "unknown device".into()
}

fn prefilter_of(meta: &Value) -> String {
    let take80 = |s: String| s.chars().take(80).collect::<String>();
    if let Value::Object(_) = meta.field("processing") {
        let d = meta.field("processing").field("description");
        if d.truthy() {
            return take80(d.py_str());
        }
    }
    let sig = meta.field("signal");
    if sig.is_null() || sig.as_str() == Some("") || sig.as_str() == Some("raw") {
        return "none (natural signal)".into();
    }
    take80(sig.py_str())
}

fn transducer_of(meta: &Value) -> Result<String, Error> {
    let m = meta.field("montage");
    if !m.truthy() {
        return Ok(String::new());
    }
    let Value::Object(_) = m else { return Err(error("the manifest's montage is not an object")) };
    let e = m.field("electrode");
    Ok(if e.truthy() { e.py_str() } else { String::new() })
}

fn export_edf_family(
    collection: &Collection,
    cap: &Recording,
    out: &Path,
    problems: &[String],
    w: &Width,
) -> Result<J, Error> {
    let fmt = if w.sample_bytes == 2 { "edf" } else { "bdf" };
    let label = cap.label();
    let fs = cap.fs();
    if fs <= 0.0 {
        return Err(error(format!(
            "refusing to export {label} as {}: the capture states no sample rate, and this container places every \
             sample by rate",
            fmt.to_uppercase()
        )));
    }
    let meta = &cap.meta;
    let start = edf::start_of(meta.field("started_utc").as_str());

    // A recording that armed before its count was reset begins with
    // samples from before it, and the file starts at the reset.
    let index = cap.index();
    let fs_round = if fs.is_finite() { (fs.round_ties_even() as i64).max(1) as usize } else { 1 };
    let last_back = (1..index.len()).rev().find(|&i| index[i] - index[i - 1] < 1);
    let skip = match last_back {
        Some(b) if b < fs_round => b,
        _ => 0,
    };

    let mut events = Vec::new();
    if let Ok(text) = std::fs::read_to_string(collection.path(label, ".events.json")?) {
        let raw = pyjson::loads(&text).map_err(|e| error(format!("{label}.events.json: {}", e.0)))?;
        let host = cap.host_time();
        let t0 = if host.is_empty() { 0.0 } else { host[skip] };
        if let Value::Array(items) = raw {
            for e in items {
                let Some(ht) = e.get("host_time") else { continue };
                let t = ht.as_f64().ok_or_else(|| error(format!("{label}: an event's host_time is not a number")))?;
                let text = match e.get("cond") {
                    Some(c) => c.py_str(),
                    None => e.get("kind").map(Value::py_str).unwrap_or_default(),
                };
                events.push(edf::Event { t: t - t0, label: text });
            }
        }
    }

    let synthetic = meta.field("synthetic").truthy();
    let label_of = |c: usize| if synthetic { format!("ch{} synthetic", c + 1) } else { format!("ch{}", c + 1) };
    let uv = uv_all(cap)?;
    let signals: Vec<(String, Vec<f64>)> =
        uv.iter().enumerate().map(|(c, u)| (label_of(c), u[skip.min(u.len())..].to_vec())).collect();

    // Lead-off, labeled only where detection was running.
    let (found, _) = per_sample(cap, index.len());
    let byte = find(&found, "leadoff").map(|a| a.data.skip(skip));
    let active = find(&found, "leadoff_active").map(|a| a.data.skip(skip));
    let (byte_a, active_a) = (byte.clone().map(Array::vector), active.clone().map(Array::vector));
    let checked = contact_state(byte_a.as_ref(), active_a.as_ref());
    let mut stretches: Vec<(String, Vec<bool>)> = Vec::new();
    if matches!(checked, "checked" | "partly checked") {
        let (b, a) = (byte.expect("checked"), active.expect("checked"));
        for c in 0..cap.channels().min(LEADOFF_CHANNELS) {
            let flags = (0..a.len()).map(|i| a.bool_at(i) && (b.i64_at(i) >> c) & 1 != 0).collect();
            stretches.push((format!("{} {}", edf::LEADOFF_LABEL, label_of(c)), flags));
        }
        if checked == "partly checked" {
            stretches.push((edf::NOT_CHECKED_LABEL.into(), (0..a.len()).map(|i| !a.bool_at(i)).collect()));
        }
    }
    let rev = meta.get("board_revision").map(Value::py_str).unwrap_or_else(|| "unknown".into());
    let rec_bits = meta.field("adc_bits");
    let bits = if w.sample_bytes == 2 { 16 } else { 24 };
    let note = if rec_bits.truthy() {
        let rb = rec_bits.as_i64().ok_or_else(|| error("adc_bits is not a whole number"))?;
        if bits < rb {
            " lossy"
        } else {
            ""
        }
    } else {
        ""
    };
    let generated = if synthetic { "SYNTHETIC, not measured: " } else { "" };
    let recording = format!("{generated}{} {rev} {bits}-bit{note}; raw record is {label}.npz", device_of(meta));
    let prefilter = prefilter_of(meta);
    let transducer = transducer_of(meta)?;
    let idx = &index[skip.min(index.len())..];
    let input = edf::Input {
        signals: &signals,
        fs,
        start,
        index: idx,
        events: &events,
        patient: "X X X X",
        recording: &recording,
        prefilter: &prefilter,
        transducer: &transducer,
        stretches: &stretches,
    };
    let (bytes, info) = edf::write(&input, w)?;
    write_file(out, &bytes)?;

    let lsb = cap.lsb_uv()?;
    let recovered = info.max_error_uv < 0.5 * lsb;
    let off_stretches: usize =
        info.stretch_counts.iter().filter(|(k, _)| k.starts_with(edf::LEADOFF_LABEL)).map(|(_, v)| v).sum();
    let _ = (info.n_records, info.samples_per_record, info.record_duration_s, info.fs);
    Ok(summary(
        cap,
        out,
        fmt,
        info.samples_written,
        problems,
        vec![
            ("lossy", json!(if bits == 16 { true } else { !recovered })),
            ("counts_recovered", json!(recovered)),
            ("max_error_uv", f(info.max_error_uv)),
            ("quantum_uv", f(info.quantum_uv)),
            ("lsb_uv", f(lsb)),
            ("bits", json!(bits)),
            ("family", json!(w.family)),
            ("continuity", json!(info.continuity)),
            ("samples_dropped", json!(0)),
            ("samples_filled", json!(info.samples_filled)),
            ("gaps", json!(info.gaps)),
            ("tail_filled", json!(info.tail_filled)),
            ("contact", json!(checked)),
            ("electrode_off_stretches", json!(off_stretches)),
            ("samples_before_restart", json!(skip)),
            ("n_annotations", json!(info.n_annotations)),
        ],
    ))
}

// ---------------------------------------------------------------- mat

/// The 128 bytes MATLAB reads before the HDF5 data: a description, the
/// subsystem offset, the version and the byte order mark, as the Python
/// library writes them.
fn mat_header(unix: u64) -> [u8; 128] {
    let days = (unix / 86_400) as i64;
    let secs = unix % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    const DAYS: [&str; 7] = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"];
    const MONTHS: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    let when = format!(
        "{} {} {:02} {:02}:{:02}:{:02} {}",
        DAYS[days.rem_euclid(7) as usize],
        MONTHS[(m - 1) as usize],
        d,
        secs / 3600,
        secs % 3600 / 60,
        secs % 60,
        y
    );
    // Python's `sys.platform` for this system.
    let platform = match std::env::consts::OS {
        "macos" | "ios" => "darwin",
        "windows" => "win32",
        os => os,
    };
    let text = format!("MATLAB 7.3 MAT-file, Platform: {platform}, Created on: {when} HDF5 schema 1.00 .");
    let mut h = [b' '; 128];
    for (i, c) in text.chars().take(116).enumerate() {
        h[i] = if c.is_ascii() { c as u8 } else { b'?' };
    }
    h[116..124].fill(0);
    h[124..128].copy_from_slice(b"\x00\x02IM");
    h
}

/// One MATLAB variable from one array, as the Python library writes it: a
/// boolean as `logical`, a type MATLAB names by its own name, and anything
/// else as `double`.
fn mat_variable(name: &str, dims: Vec<u64>, data: &Data) -> hdf5::Variable {
    use hdf5::Elem;
    let n = data.len();
    let bytes = |f: &dyn Fn(usize, &mut Vec<u8>)| {
        let mut v = Vec::new();
        for i in 0..n {
            f(i, &mut v);
        }
        v
    };
    let (elem, class, decode, raw) = match data {
        Data::Bool(x) => (Elem::U8, "logical", Some(1), x.iter().map(|&b| b as u8).collect()),
        Data::U8(x) => (Elem::U8, "uint8", None, x.clone()),
        Data::U16(x) => (Elem::U16, "uint16", None, x.iter().flat_map(|v| v.to_le_bytes()).collect()),
        Data::U32(x) => (Elem::U32, "uint32", None, x.iter().flat_map(|v| v.to_le_bytes()).collect()),
        Data::U64(x) => (Elem::U64, "uint64", None, x.iter().flat_map(|v| v.to_le_bytes()).collect()),
        Data::I16(x) => (Elem::I16, "int16", None, x.iter().flat_map(|v| v.to_le_bytes()).collect()),
        Data::I32(x) => (Elem::I32, "int32", None, x.iter().flat_map(|v| v.to_le_bytes()).collect()),
        Data::I64(x) => (Elem::I64, "int64", None, x.iter().flat_map(|v| v.to_le_bytes()).collect()),
        Data::F32(x) => (Elem::F32, "single", None, x.iter().flat_map(|v| v.to_le_bytes()).collect()),
        Data::F64(x) => (Elem::F64, "double", None, x.iter().flat_map(|v| v.to_le_bytes()).collect()),
        other => (Elem::F64, "double", None, bytes(&|i, v| v.extend(other.f64_at(i).to_le_bytes()))),
    };
    hdf5::Variable { name: name.into(), dims, elem, data: raw, matlab_class: class.into(), int_decode: decode }
}

fn export_mat(cap: &Recording, out: &Path, problems: &[String]) -> Result<J, Error> {
    let n = cap.samples();
    let nch = cap.channels();
    let uv = uv_all(cap)?;
    let (found, omitted) = per_sample(cap, n);
    let matrix = if cap.synthetic() { "synthetic_uv" } else { "uv" };
    let mut vars = Vec::new();
    // Channels by samples, which MATLAB opens as samples by channels.
    let flat: Vec<f64> = uv.iter().flatten().copied().collect();
    vars.push(mat_variable(matrix, vec![nch as u64, n as u64], &Data::F64(flat)));
    for (name, a) in &found {
        vars.push(mat_variable(name, vec![1, a.data.len() as u64], &a.data));
    }
    // The manifest as MATLAB text: UTF-16 code units.
    let text: Vec<u16> = pyjson::dumps_sorted(&cap.meta).encode_utf16().collect();
    let mut v = mat_variable("manifest_json", vec![1, text.len() as u64], &Data::U16(text));
    v.matlab_class = "char".into();
    v.int_decode = Some(2);
    vars.push(v);
    write_file(out, &hdf5::write(&mat_header(now()), &vars))?;

    let lsb = cap.lsb_uv()?;
    let mut names: Vec<String> = vec![matrix.to_string(), "manifest_json".to_string()];
    names.extend(found.iter().map(|(k, _)| k.to_string()));
    names.sort();
    Ok(summary(
        cap,
        out,
        "mat",
        n,
        problems,
        vec![
            ("lossy", json!(false)),
            ("counts_recovered", json!(true)),
            ("max_error_uv", f(0.0)),
            ("lsb_uv", f(lsb)),
            ("stored_dtype", json!("float64")),
            ("omitted", json!(omitted)),
            ("variables", json!(names)),
            ("contact", json!(contact_state(find(&found, "leadoff"), find(&found, "leadoff_active")))),
        ],
    ))
}
