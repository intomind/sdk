//! A recording written as the Python library's recorder writes one: the
//! samples in `name.npz`, the manifest in `name.meta.json`, the markers in
//! `name.events.json`, and `name.sha256` last, so that a run which dies on
//! the way leaves no checksums and is never taken for a whole recording.
//!
//! [`Take`] holds what a run collected: every sample and every packet that
//! carried them. [`save`] writes them with the manifest the library writes
//! at its current version, built the way it builds one, from what the
//! device said, what the application declares in a [`Setup`], and what the
//! run adds in a [`Run`]. Every recording this writes reads back through
//! [`crate::recording`] and every IntoMind program, and is held to the
//! Python library's own recordings by the fixture in `contract/recorder`.

use std::fs;
use std::path::PathBuf;

use intomind_protocol::frame;
use intomind_protocol::pipeline::{PipelineState, Stage};
use intomind_protocol::status::StatusMsg;

use crate::montage::{self, Fail};
use crate::npy::{self, Array, Data};
use crate::placement;
use crate::pyfmt;
use crate::pyjson::{self, Value};
use crate::recording::{check_label, error, Collection, Error};
use crate::zip;

/// The manifest version this writes. 7 is the version whose npz carries
/// `leadoff_active`, whether lead-off detection ran at each sample.
pub const MANIFEST_VERSION: i64 = 7;

/// What one packet said about the samples it carried, as a manifest reads
/// it.
#[derive(Debug, Clone, PartialEq)]
pub struct Packet {
    /// The device's index for the packet's first sample.
    pub index: i64,
    /// The device's time for the packet's first sample, in its ticks.
    pub device_time: i64,
    /// The gain the samples were taken at.
    pub gain: i64,
    /// The rate the samples were taken at. None for a rate code the
    /// contract does not define.
    pub sample_rate_hz: Option<i64>,
    /// What the samples were: `normal`, `test`, `short` or `synthetic`.
    pub mode: String,
    /// Whether the device generated the samples rather than measuring them.
    pub synthetic: bool,
}

/// What a run collected: every sample, and every packet that carried them.
#[derive(Debug, Clone, Default)]
pub struct Take {
    index: Vec<i64>,
    device_ticks: Vec<i64>,
    host_time: Vec<f64>,
    counts: Vec<Vec<i64>>,
    leadoff: Vec<u8>,
    gap_before: Vec<bool>,
    leadoff_active: Vec<Option<bool>>,
    packets: Vec<Packet>,
}

impl Take {
    /// Nothing collected yet.
    pub fn new() -> Take {
        Take::default()
    }

    /// One packet, as it arrived.
    pub fn packet(&mut self, p: Packet) {
        self.packets.push(p);
    }

    /// One sample: its index and time on the device, its time on the host,
    /// its counts across the channels, the lead-off byte of its packet,
    /// whether a break comes before it, and whether lead-off detection was
    /// running, when the source said.
    #[allow(clippy::too_many_arguments)]
    pub fn sample(
        &mut self,
        index: i64,
        device_ticks: i64,
        host_time: f64,
        counts: &[i64],
        leadoff: u8,
        gap_before: bool,
        leadoff_active: Option<bool>,
    ) {
        self.index.push(index);
        self.device_ticks.push(device_ticks);
        self.host_time.push(host_time);
        self.counts.push(counts.to_vec());
        self.leadoff.push(leadoff);
        self.gap_before.push(gap_before);
        self.leadoff_active.push(leadoff_active);
    }

    /// How many samples were collected.
    pub fn samples(&self) -> usize {
        self.index.len()
    }

    /// The packets, in the order they arrived.
    pub fn packets(&self) -> &[Packet] {
        &self.packets
    }
}

/// What the device said about itself, read when the recording is saved.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceFacts {
    /// The device's id, as lowercase hex.
    pub device_id: String,
    /// Its firmware version, `major.minor.patch`.
    pub firmware: String,
    /// The protocol version it speaks, `major.minor`.
    pub protocol: Option<String>,
    /// Its clock's ticks per second.
    pub tick_hz: u64,
    /// Its converter's reference, in microvolts.
    pub vref_uv: u64,
    /// Its converter's width, in bits.
    pub adc_bits: Option<u64>,
    /// Its channels.
    pub channels: Option<u64>,
    /// The clock record at saving: how far `host_time` can be trusted.
    pub clock: Option<serde_json::Value>,
    /// Breaks the device announced while this run was armed.
    pub gaps_in_run: u64,
}

/// What the application declares about the setup a recording was made on.
/// Nothing here is guessed: what is not declared is written as unknown.
#[derive(Debug, Clone, Default)]
pub struct Setup {
    /// Facts about the hardware no code can discover, as a JSON object: the
    /// keys the Python library's `use_site` takes, `board_revision`,
    /// `mains_hz`, `psu_volts`, `firmware_sha256`, `firmware_artifact`,
    /// `electrode_montage`, `subject_id`, `notes`, and a structured
    /// `montage`.
    pub site: Option<serde_json::Value>,
    /// The source tree of the application doing the recording, whose
    /// commit and whether it has changes are recorded.
    pub repo: Option<PathBuf>,
    /// The adapter that carried the samples: `name`, `version`,
    /// `first_party` and `transport`. Unknown when not declared.
    pub adapter: Option<serde_json::Value>,
    /// The filter the device was set to, when one was: `on`, `hp_hz`,
    /// `lp_hz` and `bands`.
    pub filter: Option<serde_json::Value>,
    /// Where a filter ran, `device` unless the adapter filtered on the host.
    pub filters_in: Option<String>,
}

/// What the run adds to its recording.
#[derive(Debug, Clone, Default)]
pub struct Run {
    /// The run's own fields, as a JSON object. `experiment` names what
    /// was run. `started_utc`, `electrode_layout`, `placement`,
    /// `regs_before` and `config_before` are read as the Python library
    /// reads them, and every other field joins the manifest. A null never
    /// replaces a value the device measured.
    pub extra: serde_json::Map<String, serde_json::Value>,
    /// The markers, written as `name.events.json` when there are any to
    /// keep, an empty list included.
    pub events: Option<serde_json::Value>,
    /// The registers after the run, from an adapter that reads them.
    pub regs_after: Option<serde_json::Value>,
    /// What the device said it was set to after the run.
    pub config_after: Option<serde_json::Value>,
    /// Why the run ended early, `link_lost` for a dropped link. An
    /// aborted recording gets no checksums.
    pub aborted: Option<String>,
    /// When the recording finished, in seconds since 1970. The host clock
    /// now, when not given.
    pub finished_unix: Option<f64>,
}

/// What a recording made by this crate says made it.
pub(crate) fn host_versions() -> Value {
    Value::obj([
        ("intomind_rust", Value::str(env!("CARGO_PKG_VERSION"))),
        ("platform", Value::str(format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH))),
    ])
}

/// Write `take` as the recording `label` in `collection`, and return its
/// manifest.
pub fn save(
    collection: &Collection,
    label: &str,
    take: &Take,
    device: &DeviceFacts,
    setup: &Setup,
    run: &Run,
) -> Result<serde_json::Value, Error> {
    save_with(collection, label, take, device, setup, run, host_versions()).map(|m| m.to_serde())
}

/// [`save`], naming the software that made the recording.
pub(crate) fn save_with(
    collection: &Collection,
    label: &str,
    take: &Take,
    device: &DeviceFacts,
    setup: &Setup,
    run: &Run,
    versions: Value,
) -> Result<Value, Error> {
    check_label(label)?;
    let dir = collection.dir();
    fs::create_dir_all(dir).map_err(|e| error(format!("{}: {e}", dir.display())))?;
    let path = |suffix: &str| collection.path(label, suffix);
    let finished = run.finished_unix.unwrap_or_else(now);

    write(&path(".npz")?, &npz(take, finished)?)?;

    let mut extra = Value::from_serde(&serde_json::Value::Object(run.extra.clone()));
    let protocol = extra.pop("experiment");
    let regs_before = extra.pop("regs_before");
    let params = extra.clone();
    let mut meta = build(
        label,
        device,
        setup,
        &take.packets,
        &regs_before,
        &run.regs_after.as_ref().map(Value::from_serde).unwrap_or(Value::Null),
        &protocol,
        params,
        extra,
        &take.index,
        &take.device_ticks,
        &run.config_after.as_ref().map(Value::from_serde).unwrap_or(Value::Null),
        finished,
        versions,
    )
    .map_err(|f| match f {
        Fail::Montage(s) | Fail::Placement(s) | Fail::Other(s) => error(s),
    })?;
    meta.set("n", Value::int(take.samples() as i64));
    // The names every earlier reader reads.
    for (k, v) in [("fw", Value::str(device.firmware.clone())), ("experiment", protocol), ("regs_before", regs_before)]
    {
        if meta.get(k).is_none() {
            meta.set(k, v);
        }
    }
    if let Some(why) = run.aborted.as_ref().filter(|w| !w.is_empty()) {
        meta.set("aborted", Value::str(why.clone()));
    }
    write(&path(".meta.json")?, pyjson::dumps_indented(&meta, 2).as_bytes())?;
    if let Some(events) = &run.events {
        write(&path(".events.json")?, pyjson::dumps_indented(&Value::from_serde(events), 2).as_bytes())?;
    }
    // Checksums mark a recording finalized. An aborted run never gets them.
    if run.aborted.as_ref().is_none_or(|w| w.is_empty()) {
        write_checksums(collection, label)?;
        freeze(collection, label)?;
    }
    append_session(collection, &meta)?;
    Ok(meta)
}

fn now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

fn write(path: &std::path::Path, bytes: &[u8]) -> Result<(), Error> {
    fs::write(path, bytes).map_err(|e| error(format!("{}: {e}", path.display())))
}

/// The arrays as `np.savez_compressed` writes them for the recorder: the
/// counts channels by samples, stored sample after sample, which numpy
/// marks as Fortran order wherever both dimensions exceed one.
fn npz(take: &Take, finished: f64) -> Result<Vec<u8>, Error> {
    let n = take.samples();
    let channels = take.counts.first().map_or(0, Vec::len);
    if take.counts.iter().any(|row| row.len() != channels) {
        return Err(error("the samples of one recording carry different numbers of channels"));
    }
    let flat: Vec<i64> = take.counts.iter().flatten().copied().collect();
    let counts = if n == 0 {
        Array::vector(Data::I64(Vec::new()))
    } else {
        Array { shape: vec![channels, n], fortran: channels > 1 && n > 1, data: Data::I64(flat) }
    };
    let mut arrays = vec![
        ("counts", counts),
        ("index", Array::vector(Data::I64(take.index.clone()))),
        ("device_ticks", Array::vector(Data::I64(take.device_ticks.clone()))),
        ("host_time", Array::vector(Data::F64(take.host_time.clone()))),
        ("leadoff", Array::vector(Data::U8(take.leadoff.clone()))),
        ("gap_before", Array::vector(Data::Bool(take.gap_before.clone()))),
    ];
    // Kept beside the lead-off byte only when every sample said, because
    // a zero byte means an attached electrode only where detection ran.
    if take.leadoff_active.iter().all(Option::is_some) {
        arrays.push((
            "leadoff_active",
            Array::vector(Data::Bool(take.leadoff_active.iter().map(|a| a.unwrap_or(false)).collect())),
        ));
    }
    let members: Vec<(String, Vec<u8>)> = arrays.iter().map(|(k, a)| (format!("{k}.npy"), npy::write(a))).collect();
    Ok(zip::write(&members, finished.max(0.0) as u64))
}

fn write_checksums(collection: &Collection, label: &str) -> Result<(), Error> {
    use sha2::{Digest, Sha256};
    let mut sums: Vec<(String, String)> = Vec::new();
    for suffix in [".npz", ".meta.json", ".events.json", ".inputs.npz"] {
        let p = collection.path(label, suffix)?;
        if let Ok(bytes) = fs::read(&p) {
            let hex: String = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
            sums.push((format!("{label}{suffix}"), hex));
        }
    }
    sums.sort();
    let text: String = sums.iter().map(|(k, v)| format!("{v}  {k}\n")).collect();
    write(&collection.path(label, ".sha256")?, text.as_bytes())
}

/// Make a finalized recording read-only, as the library does. Best
/// effort, never fatal.
fn freeze(collection: &Collection, label: &str) -> Result<(), Error> {
    for suffix in [".npz", ".meta.json", ".events.json", ".inputs.npz", ".sha256"] {
        let p = collection.path(label, suffix)?;
        if let Ok(m) = fs::metadata(&p) {
            let mut perm = m.permissions();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                perm.set_mode(0o444);
            }
            #[cfg(not(unix))]
            perm.set_readonly(true);
            let _ = fs::set_permissions(&p, perm);
        }
    }
    Ok(())
}

/// One line for the collection's session log, as the library appends it.
fn append_session(collection: &Collection, meta: &Value) -> Result<(), Error> {
    use std::io::Write;
    let mut row = Value::Object(Vec::new());
    for k in [
        "label",
        "started_utc",
        "finished_utc",
        "protocol_id",
        "board_revision",
        "firmware_version",
        "gain",
        "rate_sps",
        "device_config_changed_by_run",
        "git_commit",
    ] {
        row.set(k, meta.field(k).clone());
    }
    row.set("n", meta.field("n").clone());
    row.set("samples_lost", meta.field("quality").field("samples_lost").clone());
    let p = collection.dir().join("SESSIONS.jsonl");
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&p)
        .map_err(|e| error(format!("{}: {e}", p.display())))?;
    f.write_all(format!("{}\n", pyjson::dumps(&row)).as_bytes()).map_err(|e| error(format!("{}: {e}", p.display())))
}

/// The library's `site()`: every fact it records, unknown until declared.
fn site(declared: &Option<serde_json::Value>) -> Value {
    let mut s = Value::obj([
        ("board_revision", Value::str("unknown")),
        ("mains_hz", Value::Null),
        ("psu_volts", Value::Null),
        ("firmware_sha256", Value::Null),
        ("firmware_artifact", Value::Null),
        ("electrode_montage", Value::Null),
        ("subject_id", Value::Null),
        ("notes", Value::str("")),
    ]);
    if let Some(serde_json::Value::Object(o)) = declared {
        for (k, v) in o {
            s.set(k, Value::from_serde(v));
        }
    }
    s
}

/// The structured montage, from where the electrodes were put or from the
/// site, and whether it agrees with the device. A montage that cannot be
/// true is recorded with its error rather than refusing a recording that
/// has already been made.
pub(crate) fn montage_fields(
    site_cfg: &Value,
    dev_channels: Option<u64>,
    layout: &Value,
    place: &Value,
) -> Result<Value, Fail> {
    let ok = |m: &Value| match dev_channels {
        None => Value::Null,
        Some(n) => Value::Bool(montage::channel_count(m) as u64 == n),
    };
    let fields = |m: Value, err: Value, matches: Value, source: Value, layout: Value, placement: Value| {
        Value::obj([
            ("montage", m),
            ("montage_error", err),
            ("montage_matches_device", matches),
            ("montage_source", source),
            ("electrode_layout", layout),
            ("placement", placement),
        ])
    };
    if !layout.is_null() && !place.is_null() {
        let made = (|| {
            let lay = placement::layout_from(layout)?.unwrap_or(&Value::Null);
            let m = placement::to_montage(lay, place)?;
            let r = placement::resolve(lay, place)?;
            Ok((m, r))
        })();
        return match made {
            Ok((m, r)) => Ok(fields(m.clone(), Value::Null, ok(&m), Value::str("placement"), layout.clone(), r)),
            Err(Fail::Montage(e) | Fail::Placement(e)) => Ok(fields(
                Value::Null,
                Value::str(e),
                Value::Bool(false),
                Value::str("placement"),
                layout.clone(),
                Value::Null,
            )),
            Err(other) => Err(other),
        };
    }
    match montage::from_site(site_cfg) {
        Err(Fail::Montage(e)) => {
            Ok(fields(Value::Null, Value::str(e), Value::Bool(false), Value::str("site"), Value::Null, Value::Null))
        }
        Err(other) => Err(other),
        Ok(None) => Ok(fields(Value::Null, Value::Null, Value::Null, Value::Null, Value::Null, Value::Null)),
        Ok(Some(m)) => Ok(fields(m.clone(), Value::Null, ok(&m), Value::str("site"), Value::Null, Value::Null)),
    }
}

/// Where the breaks are, not only how many. A count that steps back is an
/// epoch reset and lost nothing.
pub(crate) fn gap_ledger(index: &[i64]) -> Value {
    let mut gaps = Vec::new();
    let mut resets = 0i64;
    let mut lost = 0i64;
    for w in index.windows(2) {
        let d = w[1] - w[0];
        if d > 1 {
            gaps.push(Value::obj([("after_sample", Value::int(w[0])), ("lost", Value::int(d - 1))]));
            lost += d - 1;
        } else if d < 0 {
            resets += 1;
        }
    }
    let received = index.len() as i64;
    let n_gaps = gaps.len() as i64;
    Value::obj([
        ("gaps", Value::Array(gaps)),
        ("n_gaps", Value::int(n_gaps)),
        ("epoch_resets", Value::int(resets)),
        ("samples_received", Value::int(received)),
        ("samples_expected", Value::int(received + lost)),
        ("samples_lost", Value::int(lost)),
    ])
}

/// The rate from the device's clock: samples over the span of their ticks.
pub(crate) fn fs_from_ticks(ticks: &[i64], tick_hz: u64) -> f64 {
    if ticks.len() < 2 || tick_hz == 0 {
        return 0.0;
    }
    let span = (ticks[ticks.len() - 1] as f64 - ticks[0] as f64) / tick_hz as f64;
    if span > 0.0 {
        (ticks.len() - 1) as f64 / span
    } else {
        0.0
    }
}

/// The rate from the first and last packets' index and device time.
fn fs_from_packets(packets: &[Packet], tick_hz: u64) -> f64 {
    if packets.len() < 2 {
        return 0.0;
    }
    let (p0, p1) = (&packets[0], &packets[packets.len() - 1]);
    let span = (p1.device_time - p0.device_time) as f64 / tick_hz as f64;
    if span > 0.0 {
        (p1.index - p0.index) as f64 / span
    } else {
        0.0
    }
}

/// Which adapter produced the recording, and where it was filtered.
pub(crate) fn signal_chain(setup: &Setup) -> Result<Value, Fail> {
    let adapter = match setup.adapter.as_ref().map(Value::from_serde) {
        Some(a) if a.truthy() => py_dict(&a)?,
        _ => Value::obj([
            ("name", Value::str("unknown")),
            ("version", Value::Null),
            ("first_party", Value::Bool(false)),
            ("transport", Value::Null),
        ]),
    };
    // `getattr(dev, "filter", None) or {}`, read with `.get`, which only a
    // dict has.
    let f = setup.filter.as_ref().map(Value::from_serde).unwrap_or(Value::Null);
    if f.truthy() && !matches!(f, Value::Object(_)) {
        return Err(Fail::Other(format!("AttributeError: {} has no attribute 'get'", f.py_repr())));
    }
    let on = f.field("on").truthy();
    let filters = if on {
        // `[list(b) for b in (f.get("bands") or [])]`
        let bands = f.field("bands");
        let each = if bands.truthy() { placement::iterate(bands)? } else { Vec::new() };
        let bands = each.iter().map(|b| placement::iterate(b).map(Value::Array)).collect::<Result<Vec<_>, _>>()?;
        Value::obj([
            ("hp_hz", f.field("hp_hz").clone()),
            ("lp_hz", f.field("lp_hz").clone()),
            ("bands", Value::Array(bands)),
        ])
    } else {
        Value::Null
    };
    Ok(Value::obj([
        ("adapter", adapter),
        (
            "filtering",
            Value::str(if on { setup.filters_in.clone().unwrap_or_else(|| "device".into()) } else { "none".into() }),
        ),
        ("filters", filters),
    ]))
}

/// What `dict(v)` makes of a value a program declared: an object as itself,
/// and a list of key and value pairs as an object, keys Python holds equal
/// (`1`, `1.0` and `True`) kept once in their first form with the last
/// value, and written as `json.dumps` writes a key that is not text.
fn py_dict(v: &Value) -> Result<Value, Fail> {
    if let Value::Object(_) = v {
        return Ok(v.clone());
    }
    let mut out: Vec<(Value, Value)> = Vec::new();
    for (i, item) in placement::iterate(v)?.into_iter().enumerate() {
        if !matches!(item, Value::Array(_) | Value::Str(_) | Value::Object(_)) {
            return Err(Fail::Other(format!(
                "TypeError: cannot convert dictionary update sequence element #{i} to a sequence"
            )));
        }
        let [key, value] = <[Value; 2]>::try_from(placement::iterate(&item)?).map_err(|p| {
            Fail::Other(format!("ValueError: dictionary update sequence element #{i} has length {}; 2 is required", p.len()))
        })?;
        if key.json_key().is_none() {
            return Err(Fail::Other(format!("TypeError: unhashable type: {}", key.py_repr())));
        }
        match out.iter_mut().find(|(have, _)| have.py_eq(&key)) {
            Some(slot) => slot.1 = value,
            None => out.push((key, value)),
        }
    }
    Ok(Value::Object(out.into_iter().map(|(k, v)| (k.json_key().unwrap_or_default(), v)).collect()))
}

/// What the samples were, off the packets that carried them.
fn signal_source(packets: &[Packet]) -> (Value, Value) {
    let mut modes: Vec<&str> = packets.iter().map(|p| p.mode.as_str()).collect();
    modes.sort();
    modes.dedup();
    match modes.len() {
        0 => (Value::Null, Value::Null),
        1 => (Value::str(modes[0]), Value::Bool(packets.iter().any(|p| p.synthetic))),
        _ => (
            Value::Array(modes.iter().map(|m| Value::str(*m)).collect()),
            Value::Bool(packets.iter().any(|p| p.synthetic)),
        ),
    }
}

/// The commit of the declared source tree and whether it has changes, or
/// nulls when none was declared or git cannot say.
fn git_state(repo: &Option<PathBuf>) -> (Value, Value) {
    let Some(repo) = repo else { return (Value::Null, Value::Null) };
    let git = |args: &[&str]| -> Option<String> {
        let mut child = std::process::Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(args)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .ok()?;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            match child.try_wait() {
                Ok(Some(status)) if status.success() => break,
                Ok(Some(_)) => return None,
                Ok(None) if std::time::Instant::now() < deadline => {
                    std::thread::sleep(std::time::Duration::from_millis(10))
                }
                _ => {
                    let _ = child.kill();
                    return None;
                }
            }
        }
        let mut out = String::new();
        std::io::Read::read_to_string(&mut child.stdout.take()?, &mut out).ok()?;
        Some(out.trim().to_string())
    };
    let commit = git(&["rev-parse", "HEAD"]);
    let status = git(&["status", "--porcelain"]);
    (commit.map_or(Value::Null, Value::Str), status.map_or(Value::Null, |s| Value::Bool(!s.is_empty())))
}

/// UTC as `YYYY-MM-DDTHH:MM:SSZ`, from seconds since 1970.
fn utc(unix: f64) -> String {
    let secs = unix.floor() as i64;
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem / 60 % 60, rem % 60)
}

/// What changed between two readings of the instrument, in key order.
fn diff(before: &Value, after: &Value) -> Value {
    let (Value::Object(a), Value::Object(b)) = (before, after) else { return Value::Object(Vec::new()) };
    let mut keys: Vec<&String> = a.iter().map(|(k, _)| k).chain(b.iter().map(|(k, _)| k)).collect();
    keys.sort();
    keys.dedup();
    let mut out = Vec::new();
    for k in keys {
        let (x, y) = (before.field(k), after.field(k));
        if !x.py_eq(y) {
            out.push((k.clone(), Value::Array(vec![x.clone(), y.clone()])));
        }
    }
    Value::Object(out)
}

/// The manifest, built as the library's `build_manifest` builds it.
#[allow(clippy::too_many_arguments)]
fn build(
    label: &str,
    device: &DeviceFacts,
    setup: &Setup,
    packets: &[Packet],
    regs_before: &Value,
    regs_after: &Value,
    protocol: &Value,
    params: Value,
    mut extra: Value,
    index: &[i64],
    ticks: &[i64],
    config_after: &Value,
    finished: f64,
    versions: Value,
) -> Result<Value, Fail> {
    let s = site(&setup.site);
    let layout = extra.pop("electrode_layout");
    let place = extra.pop("placement");
    let fs =
        if ticks.len() >= 2 { fs_from_ticks(ticks, device.tick_hz) } else { fs_from_packets(packets, device.tick_hz) };
    let config_before = extra.field("config_before").clone();
    let changed = if regs_before.truthy() && regs_after.truthy() {
        diff(regs_before, regs_after)
    } else if config_before.truthy() && config_after.truthy() {
        diff(&config_before, config_after)
    } else {
        Value::Object(Vec::new())
    };
    let started = extra.pop("started_utc");
    let opt = |v: Option<u64>| v.map_or(Value::Null, |x| Value::int(x as i64));
    let mut m = Value::obj([
        ("manifest_version", Value::int(MANIFEST_VERSION)),
        ("label", Value::str(label)),
        ("started_utc", started),
        ("finished_utc", Value::str(utc(finished))),
        ("captured_unix", Value::Float(finished)),
        ("device_id", Value::str(device.device_id.clone())),
        ("firmware_version", Value::str(device.firmware.clone())),
        ("firmware_sha256", s.field("firmware_sha256").clone()),
        ("firmware_artifact", s.field("firmware_artifact").clone()),
        ("tick_hz", Value::int(device.tick_hz as i64)),
        ("vref_uv", Value::int(device.vref_uv as i64)),
        ("adc_bits", opt(device.adc_bits)),
        ("channels", opt(device.channels)),
        ("device_proto", device.protocol.clone().map_or(Value::Null, Value::Str)),
        ("board_revision", s.field("board_revision").clone()),
        ("electrode_montage", s.field("electrode_montage").clone()),
    ]);
    if let Value::Object(fields) = montage_fields(&s, device.channels, &layout, &place)? {
        for (k, v) in fields {
            m.set(&k, v);
        }
    }
    let last = packets.last();
    let (source, synthetic) = signal_source(packets);
    let (commit, dirty) = git_state(&setup.repo);
    for (k, v) in [
        ("mains_hz", s.field("mains_hz").clone()),
        ("psu_volts", s.field("psu_volts").clone()),
        ("subject_id", s.field("subject_id").clone()),
        ("regs_before", regs_before.clone()),
        ("regs_after", regs_after.clone()),
        ("regs_diff", changed.clone()),
        ("config_after", config_after.clone()),
        ("device_config_changed_by_run", Value::Bool(changed.truthy())),
        ("gain", last.map_or(Value::Null, |p| Value::int(p.gain))),
        ("rate_sps", last.and_then(|p| p.sample_rate_hz).map_or(Value::Null, Value::int)),
        ("fs_effective", Value::Float(fs)),
        ("protocol_id", protocol.clone()),
        ("protocol_params", params),
        ("signal_chain", signal_chain(setup)?),
        ("signal_source", source),
        ("synthetic", synthetic),
        ("quality", gap_ledger(index)),
        ("gaps_announced", Value::int(device.gaps_in_run as i64)),
        ("clock", device.clock.as_ref().map(Value::from_serde).unwrap_or(Value::Null)),
        ("git_commit", commit),
        ("git_dirty", dirty),
        ("host_versions", versions),
    ] {
        m.set(k, v);
    }
    // A run's fields may add to the manifest. They may not erase a
    // measurement: a null never wins against a value the device gave.
    if let Value::Object(items) = extra {
        for (k, v) in items {
            if v.is_null() && !m.field(&k).is_null() {
                continue;
            }
            m.set(&k, v);
        }
    }
    Ok(m)
}

/// A stage's kind in words, `kind N` for one the contract does not define.
fn kind_name(kind: u8) -> String {
    match kind {
        1 => "highpass".into(),
        2 => "lowpass".into(),
        3 => "notch".into(),
        k => format!("kind {k}"),
    }
}

/// A processing chain in words, as the Python library says it in a
/// manifest and in a BDF or EDF header: `natural signal` for no stage,
/// corners in hertz, and the automatic low-pass at the rate it follows.
pub fn describe_chain(stages: &[Stage], rate_sps: Option<u32>) -> String {
    if stages.is_empty() {
        return "natural signal".into();
    }
    let hz = |dhz: u16| pyfmt::general(dhz as f64 / 10.0, 6);
    let mut words = Vec::new();
    for s in stages {
        let p = s.params();
        words.push(match (s.kind, p) {
            (1, [corner, ..]) => format!("high-pass {} Hz", hz(*corner)),
            (2, [corner, ..]) if *corner != 0 => format!("low-pass {} Hz", hz(*corner)),
            (2, [_, ..]) => match rate_sps.filter(|r| *r != 0) {
                Some(r) => {
                    format!("low-pass {} Hz (automatic at {r} SPS)", pyfmt::general((r as i64 * 4) as f64 / 10.0, 6))
                }
                None => "low-pass automatic".into(),
            },
            (3, [low, high, ..]) => format!("notch {} to {} Hz", hz(*low), hz(*high)),
            (k, p) => format!("{} [{}]", kind_name(k), p.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")),
        });
    }
    words.join(", ")
}

/// A chain in force as a manifest stores it: whose it is, its stages, and
/// the chain in words at the rate given.
fn chain_value(state: &PipelineState, rate_sps: Option<u32>) -> Value {
    let stages = state.chain.stages();
    Value::obj([
        (
            "origin",
            Value::str(match state.origin {
                0 => "default".to_string(),
                1 => "host".to_string(),
                o => format!("origin {o}"),
            }),
        ),
        (
            "stages",
            Value::Array(
                stages
                    .iter()
                    .map(|s| {
                        Value::obj([
                            ("kind", Value::int(s.kind as i64)),
                            ("name", Value::str(kind_name(s.kind))),
                            ("params", Value::Array(s.params().iter().map(|x| Value::int(*x as i64)).collect())),
                        ])
                    })
                    .collect(),
            ),
        ),
        ("description", Value::str(describe_chain(stages, rate_sps))),
    ])
}

/// What a recording stores as `processing`, the chain the device ran on
/// the signal, as the Python library reads it after a run: the chain in
/// force at the rate the device was set to, read from a device that claims
/// a processing chain, or the natural signal from one that does not. None
/// where a device that claims one did not answer.
pub fn processing(capable: bool, state: Option<&PipelineState>, rate_sps: Option<u32>) -> Option<serde_json::Value> {
    if !capable {
        return Some(
            Value::obj([
                ("origin", Value::str("none")),
                ("stages", Value::Array(Vec::new())),
                ("description", Value::str("natural signal")),
            ])
            .to_serde(),
        );
    }
    state.map(|s| chain_value(s, rate_sps).to_serde())
}

/// What the device said it was set to, read after a run, as the Python
/// library reads it back: the status characteristic, which every device
/// has, the lead-off setting as last known, and then what the device
/// claims: the chain in force, the lamp's level, the model's cadence in
/// seconds, and the name and adjective it composes its name from. Pass
/// None for what the device does not claim.
pub fn config_after(
    status: &StatusMsg,
    leadoff_setting: Option<bool>,
    pipeline: Option<&PipelineState>,
    indicator: Option<u8>,
    model_interval: Option<(u16, u16)>,
    name: Option<(&str, &str)>,
) -> serde_json::Value {
    let rate = frame::sps_of_rate_code(status.rate_code).map(u32::from);
    let mut c = Value::obj([
        ("gain", frame::GAIN_BY_CODE.get(status.gain_code as usize).map_or(Value::Null, |g| Value::int(*g as i64))),
        ("rate_sps", rate.map_or(Value::Null, |r| Value::int(r as i64))),
        (
            "mode",
            Value::str(match status.mode {
                0 => "normal".to_string(),
                1 => "test".to_string(),
                2 => "short".to_string(),
                3 => "synthetic".to_string(),
                m => format!("mode {m}"),
            }),
        ),
        ("leadoff", leadoff_setting.map_or(Value::Null, Value::Bool)),
        ("leadoff_status", Value::int(status.loff_statp as i64)),
    ]);
    if let Some(p) = pipeline {
        c.set("processing", chain_value(p, rate));
    }
    if let Some(level) = indicator {
        let words = match level {
            0 => "silent".to_string(),
            1 => "reserved".to_string(),
            2 => "verbose".to_string(),
            l => format!("level {l}"),
        };
        c.set("indicator", Value::str(words));
    }
    if let Some((every, least)) = model_interval {
        c.set("model_interval_s", Value::int(every as i64));
        c.set("model_interval_min_s", Value::int(least as i64));
    }
    if let Some((n, a)) = name {
        let mut parts = Vec::new();
        if !n.is_empty() {
            parts.push(format!("{n}'s"));
        }
        if !a.is_empty() {
            parts.push(a.to_string());
        }
        parts.push("IntoMind One".to_string());
        c.set("name", Value::str(n));
        c.set("adjective", Value::str(a));
        c.set("composed_name", Value::str(parts.join(" ")));
    }
    c.to_serde()
}
