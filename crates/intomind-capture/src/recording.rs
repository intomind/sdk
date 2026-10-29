//! The recording every IntoMind program writes, read as the Python library
//! reads it.
//!
//! A recording is a set of files that share one name in a recordings
//! folder: `name.npz` with the samples, `name.meta.json` with the manifest,
//! `name.events.json` with the markers, `name.inputs.npz` when a controller
//! was recorded, and `name.sha256`, written last, with a checksum of each.
//! The Command Center, the Python library and this crate all read and
//! write this format, and docs.intomind.com/files.html states its promises.
//!
//! Reading follows the Python library's `analysis.load` exactly, including
//! the two corrections it makes in memory and never on disk: samples
//! delivered twice by two Bluetooth subscriptions, and sample times
//! stretched by an impossible clock fit. Each correction says what it did in
//! [`Recording::repairs`].

use std::fmt;
use std::path::{Path, PathBuf};

use crate::npy::{self, Array, Data};
use crate::pyfmt;
use crate::pyjson::{self, Value};
use crate::zip;

/// Why a recording could not be read or exported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub(crate) String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub(crate) fn error(what: impl Into<String>) -> Error {
    Error(what.into())
}

/// Refuse a label that is not a plain name: letters, digits, dots, dashes
/// and underscores, starting with a letter or a digit, at most 128
/// characters, and never `..`. A label becomes a file name, so this is
/// checked wherever one is joined to a path, as the Python library checks
/// it.
pub fn check_label(label: &str) -> Result<&str, Error> {
    let b = label.as_bytes();
    let ok = !b.is_empty()
        && b.len() <= 128
        && b[0].is_ascii_alphanumeric()
        && b.iter().all(|&c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'-'));
    if !ok {
        return Err(error(format!(
            "'{label}' is not a capture label. A label is letters, digits, dots, dashes and underscores, \
             starts with a letter or a digit, and is at most 128 characters."
        )));
    }
    if label.contains("..") {
        return Err(error(format!("'{label}' is not a capture label: it walks up a directory")));
    }
    Ok(label)
}

/// Whether a recording's bytes are the ones it was finalized with, in the
/// four states the Python library distinguishes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verification {
    /// False when a checksum does not match, a file is missing, or the run
    /// was aborted. Such a recording is never exported.
    pub ok: bool,
    /// What a reader should know, empty for a recording finalized at write
    /// time and intact.
    pub problems: Vec<String>,
}

/// A recordings folder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Collection {
    dir: PathBuf,
}

impl Collection {
    /// The folder at `dir`.
    pub fn open(dir: impl Into<PathBuf>) -> Collection {
        Collection { dir: dir.into() }
    }

    /// The folder the Python library uses when nothing else is said:
    /// `INTOMIND_CAPTURES`, or `~/.local/share/intomind/captures`.
    pub fn default_dir() -> PathBuf {
        if let Some(d) = std::env::var_os("INTOMIND_CAPTURES").filter(|d| !d.is_empty()) {
            return PathBuf::from(d);
        }
        let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).unwrap_or_default();
        PathBuf::from(home).join(".local").join("share").join("intomind").join("captures")
    }

    /// Where this folder is.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The path of one of a recording's files, its label checked.
    pub fn path(&self, label: &str, suffix: &str) -> Result<PathBuf, Error> {
        Ok(self.dir.join(format!("{}{suffix}", check_label(label)?)))
    }

    /// Hash lines checked against the files beside them.
    fn check_lines(&self, text: &str) -> Vec<String> {
        let mut problems = Vec::new();
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let mut parts = line.trim_start().splitn(2, char::is_whitespace);
            let want = parts.next().unwrap_or("");
            let name = parts.next().unwrap_or("").trim();
            // A name is a file in this folder. The Python library joins it to
            // the folder as it is, so a name that would leave the folder is a
            // problem here rather than a read somewhere else.
            let plain = !name.is_empty() && !name.contains('/') && !name.contains('\\') && name != "." && name != "..";
            let p = self.dir.join(name);
            if !plain || !p.is_file() {
                problems.push(format!("missing: {name}"));
            } else if sha256_file(&p).as_deref() != Some(want) {
                problems.push(format!("CHECKSUM MISMATCH: {name}"));
            }
        }
        problems
    }

    /// The legacy ledger's lines for `label`: hashes taken after the fact
    /// for recordings made before checksums existed.
    fn legacy_lines(&self, label: &str) -> Option<String> {
        let text = std::fs::read_to_string(self.dir.join("LEGACY.sha256")).ok()?;
        let mut mine = Vec::new();
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let Some((_, name)) = line.trim_start().split_once(char::is_whitespace) else { continue };
            let name = name.trim();
            let mut l = name.rsplit_once('.').map(|(a, _)| a).unwrap_or(name);
            for suffix in [".meta", ".events"] {
                if let Some(s) = l.strip_suffix(suffix) {
                    l = s;
                }
            }
            if l == label {
                mine.push(line);
            }
        }
        if mine.is_empty() {
            None
        } else {
            Some(mine.join("\n"))
        }
    }

    /// Whether `label`'s bytes are the ones it was finalized with.
    pub fn verify(&self, label: &str) -> Result<Verification, Error> {
        let cp = self.path(label, ".sha256")?;
        if cp.exists() {
            let text = std::fs::read_to_string(&cp).map_err(|e| error(format!("{}: {e}", cp.display())))?;
            let problems = self.check_lines(&text);
            return Ok(Verification { ok: problems.is_empty(), problems });
        }
        if let Some(why) = self.aborted(label)? {
            return Ok(Verification {
                ok: false,
                problems: vec![format!("ABORTED ({why}): never finalized, do not analyze")],
            });
        }
        if let Some(lines) = self.legacy_lines(label) {
            let problems = self.check_lines(&lines);
            if !problems.is_empty() {
                return Ok(Verification { ok: false, problems });
            }
            return Ok(Verification {
                ok: true,
                problems: vec!["provenance incomplete: recorded before checksums existed; unchanged since it was \
                                marked, but its bytes at recording time were never attested"
                    .into()],
            });
        }
        Ok(Verification {
            ok: true,
            problems: vec!["unverified: no checksum file, not marked (recorded before P0)".into()],
        })
    }

    /// Why the run was aborted, as its manifest says, or None.
    fn aborted(&self, label: &str) -> Result<Option<String>, Error> {
        let m = self.path(label, ".meta.json")?;
        let Ok(text) = std::fs::read_to_string(&m) else { return Ok(None) };
        let Ok(v) = pyjson::loads(&text) else { return Ok(None) };
        Ok(match v.get("aborted") {
            Some(why) if why.truthy() => Some(why.py_str()),
            _ => None,
        })
    }

    /// `label`, verified, read, and corrected in memory where it was
    /// recorded by code with a defect since fixed. A recording that fails
    /// verification is refused.
    pub fn load(&self, label: &str) -> Result<Recording, Error> {
        self.load_with(label, true, true)
    }

    /// `label` exactly as it is on disk: verified, but with no correction.
    pub fn load_as_recorded(&self, label: &str) -> Result<Recording, Error> {
        self.load_with(label, true, false)
    }

    pub(crate) fn load_with(&self, label: &str, verify: bool, repair: bool) -> Result<Recording, Error> {
        if verify {
            let v = self.verify(label)?;
            if !v.ok {
                let list = Value::Array(v.problems.iter().map(|p| Value::Str(p.clone())).collect());
                return Err(error(format!("{label}: integrity check FAILED: {}", list.py_repr())));
            }
        }
        let npz_path = self.path(label, ".npz")?;
        let bytes = std::fs::read(&npz_path).map_err(|e| error(format!("{}: {e}", npz_path.display())))?;
        let mut arrays = read_npz(&bytes).map_err(|e| error(format!("{}: {e}", npz_path.display())))?;
        let meta_path = self.path(label, ".meta.json")?;
        let manifest_text =
            std::fs::read_to_string(&meta_path).map_err(|e| error(format!("{}: {e}", meta_path.display())))?;
        let meta = pyjson::loads(&manifest_text).map_err(|e| error(format!("{}: {}", meta_path.display(), e.0)))?;

        let mut take = |name: &str| arrays.iter().position(|(n, _)| n == name).map(|i| arrays.remove(i).1);
        let counts = take("counts").ok_or_else(|| error(format!("{label}: the recording has no 'counts'")))?;
        let index = take("index").ok_or_else(|| error(format!("{label}: the recording has no 'index'")))?;
        let host_time = take("host_time").ok_or_else(|| error(format!("{label}: the recording has no 'host_time'")))?;
        let device_ticks = take("device_ticks");
        let leadoff = take("leadoff");
        let gap_before = take("gap_before");
        let leadoff_active = take("leadoff_active");
        if counts.shape.len() != 2 {
            return Err(error(format!("{label}: the counts are not channels by samples")));
        }
        let mut r = Recording {
            label: label.to_string(),
            counts,
            index,
            host_time,
            device_ticks,
            leadoff,
            gap_before,
            leadoff_active,
            meta,
            manifest_text,
            repairs: Vec::new(),
        };
        if repair {
            r.dedupe_delivery()?;
            r.repin_clock();
        }
        Ok(r)
    }
}

/// The sha256 of a file's bytes, in lowercase hex.
pub(crate) fn sha256_file(p: &Path) -> Option<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut f = std::fs::File::open(p).ok()?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Some(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// The arrays of an `.npz`, by name, without the `.npy` suffix.
pub(crate) fn read_npz(bytes: &[u8]) -> Result<Vec<(String, Array)>, Error> {
    let members = zip::read(bytes).map_err(|e| error(e.0))?;
    let mut out = Vec::new();
    for (name, data) in members {
        let key = name.strip_suffix(".npy").unwrap_or(&name).to_string();
        let a = npy::read(&data).map_err(|e| error(format!("{name}: {}", e.0)))?;
        out.push((key, a));
    }
    Ok(out)
}

/// One recording in memory: the converter's counts for every channel, each
/// sample's index and clocks and marks, and the manifest that explains them.
#[derive(Debug, Clone)]
pub struct Recording {
    label: String,
    pub(crate) counts: Array,
    pub(crate) index: Array,
    pub(crate) host_time: Array,
    pub(crate) device_ticks: Option<Array>,
    pub(crate) leadoff: Option<Array>,
    pub(crate) gap_before: Option<Array>,
    pub(crate) leadoff_active: Option<Array>,
    pub(crate) meta: Value,
    manifest_text: String,
    repairs: Vec<String>,
}

impl Recording {
    /// The recording's name.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// How many channels it holds.
    pub fn channels(&self) -> usize {
        self.counts.shape[0]
    }

    /// How many samples it holds, counted on the counts, which are the
    /// record.
    pub fn samples(&self) -> usize {
        self.counts.shape[1]
    }

    /// Channel `c`'s counts, as the converter gave them.
    pub fn counts(&self, c: usize) -> Vec<i64> {
        self.counts.row_indices(c).into_iter().map(|i| self.counts.data.i64_at(i)).collect()
    }

    /// Each sample's index, as the device numbered it.
    pub fn index(&self) -> Vec<i64> {
        (0..self.index.data.len()).map(|i| self.index.data.i64_at(i)).collect()
    }

    /// Each sample's time on this computer's clock, in seconds.
    pub fn host_time(&self) -> Vec<f64> {
        (0..self.host_time.data.len()).map(|i| self.host_time.data.f64_at(i)).collect()
    }

    /// Each sample's time on the device's clock, in its ticks, when the
    /// recording kept it.
    pub fn device_ticks(&self) -> Option<Vec<i64>> {
        self.device_ticks.as_ref().map(|a| (0..a.data.len()).map(|i| a.data.i64_at(i)).collect())
    }

    /// The lead-off byte that arrived with each sample, one bit per
    /// electrode, when the recording kept it.
    pub fn leadoff(&self) -> Option<Vec<u8>> {
        self.leadoff.as_ref().map(|a| (0..a.data.len()).map(|i| a.data.i64_at(i) as u8).collect())
    }

    /// Whether each sample is the first after a break in the timeline.
    pub fn gap_before(&self) -> Option<Vec<bool>> {
        self.gap_before.as_ref().map(|a| (0..a.data.len()).map(|i| a.data.bool_at(i)).collect())
    }

    /// Whether lead-off detection was running at each sample. None for a
    /// recording made before this was kept, which cannot say whether
    /// contact was checked at all.
    pub fn leadoff_active(&self) -> Option<Vec<bool>> {
        self.leadoff_active.as_ref().map(|a| (0..a.data.len()).map(|i| a.data.bool_at(i)).collect())
    }

    /// The manifest, as its file holds it.
    pub fn manifest_json(&self) -> &str {
        &self.manifest_text
    }

    /// What reading had to correct, in sentences. Empty for every recording
    /// made by code with the fixes in it.
    pub fn repairs(&self) -> &[String] {
        &self.repairs
    }

    /// Whether the device generated this signal rather than measuring it.
    pub fn synthetic(&self) -> bool {
        self.meta.field("synthetic").truthy()
    }

    /// The effective sample rate. The manifest's `fs_effective`, unless it
    /// disagrees with the host clock by more than five percent, which a
    /// recording whose index reset before the fix did, and then the host
    /// clock's.
    pub fn fs(&self) -> f64 {
        let stored = self.meta.field("fs_effective");
        let fs = if stored.truthy() { stored.as_f64().unwrap_or(0.0) } else { 0.0 };
        let n = self.host_time.data.len();
        if n > 1 {
            let (first, last) = (self.host_time.data.f64_at(0), self.host_time.data.f64_at(n - 1));
            if last > first {
                let fs_ht = (n - 1) as f64 / (last - first);
                if fs <= 0.0 || (fs - fs_ht).abs() / fs_ht > 0.05 {
                    return fs_ht;
                }
            }
        }
        fs
    }

    /// Microvolts per count, from what the device said: its reference, its
    /// gain and its converter's width. A manifest that does not record the
    /// width, which version 1 does not, cannot be converted.
    pub fn lsb_uv(&self) -> Result<f64, Error> {
        let bits = self.meta.field("adc_bits");
        if !bits.truthy() {
            let who = match self.meta.get("label") {
                Some(v) => v.py_str(),
                None => "this capture".into(),
            };
            return Err(error(format!(
                "{who} does not record adc_bits, so the microvolts it holds cannot be computed. \
                 Manifests from version 2 onward record it."
            )));
        }
        let Value::Int(b) = bits else { return Err(error("adc_bits is not a whole number")) };
        let b: u32 = b.parse().map_err(|_| error("adc_bits is not a converter width"))?;
        let vref =
            self.meta.get("vref_uv").and_then(Value::as_f64).ok_or_else(|| error("the manifest has no vref_uv"))?;
        let gain = self.meta.get("gain").ok_or_else(|| error("the manifest has no gain"))?;
        let scale = 2f64.powi(b as i32);
        let denom = match gain {
            Value::Int(g) => {
                // An integer product, exact, then one conversion, as Python's
                // `gain * (1 << bits)` is.
                let g: i128 = g.parse().map_err(|_| error("gain is not a number"))?;
                (g * (1i128 << b)) as f64
            }
            Value::Float(g) => g * scale,
            Value::Bool(g) => (*g as u8 as f64) * scale,
            _ => return Err(error("gain is not a number")),
        };
        Ok((2.0 * vref) / denom)
    }

    /// Channel `c` in microvolts: its counts times [`Self::lsb_uv`].
    pub fn uv(&self, c: usize) -> Result<Vec<f64>, Error> {
        let lsb = self.lsb_uv()?;
        Ok(self.counts.row_indices(c).into_iter().map(|i| self.counts.data.f64_at(i) * lsb).collect())
    }

    /// Two subscriptions on one device deliver every packet twice, and an
    /// old recorder stored each copy. The signature is exact: a device time
    /// repeated with every channel's count the same. Copies are removed, the
    /// first arrival kept, and the marks go with their samples.
    fn dedupe_delivery(&mut self) -> Result<(), Error> {
        let Some(ticks) = &self.device_ticks else { return Ok(()) };
        let n = ticks.data.len();
        if n < 2 {
            return Ok(());
        }
        let t: Vec<i64> = (0..n).map(|i| ticks.data.i64_at(i)).collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| (t[i], i));
        // The first arrival of each device time, and which first each sample
        // repeats.
        let mut first_of = vec![0usize; n];
        let mut firsts = Vec::new();
        let mut k = 0;
        while k < n {
            let f = order[k];
            firsts.push(f);
            let mut j = k;
            while j < n && t[order[j]] == t[f] {
                first_of[order[j]] = f;
                j += 1;
            }
            k = j;
        }
        if firsts.len() == n {
            return Ok(());
        }
        let n_extra = n - firsts.len();
        let nch = self.counts.shape[0];
        let rows: Vec<Vec<usize>> = (0..nch).map(|c| self.counts.row_indices(c)).collect();
        let same = (0..n)
            .all(|i| rows.iter().all(|r| self.counts.data.i64_at(r[i]) == self.counts.data.i64_at(r[first_of[i]])));
        if !same {
            self.repairs.push(format!(
                "{} samples repeat a device time with DIFFERENT counts behind it. Not duplicate delivery, and not \
                 something this can correct: the sample stream is unusable as a timeline.",
                pyfmt::grouped(n_extra as i64)
            ));
            return Ok(());
        }
        let mut keep = firsts.clone();
        keep.sort_unstable();
        if !keep.windows(2).all(|w| t[w[1]] > t[w[0]]) {
            self.repairs.push(format!(
                "{} duplicated samples, and removing them still does not leave the device clock increasing.",
                pyfmt::grouped(n_extra as i64)
            ));
            return Ok(());
        }
        // `round(x, 3)`: the exact value rounded half to even at three places.
        let dupes: f64 = pyfmt::fixed(n as f64 / firsts.len() as f64, 3).parse().unwrap_or(f64::NAN);
        let check = |a: &Array, what: &str| -> Result<(), Error> {
            if a.data.len() < n {
                Err(error(format!(
                    "{}: {what} is shorter than the samples, so its copies cannot be removed",
                    self.label
                )))
            } else {
                Ok(())
            }
        };
        for (a, what) in [
            (Some(&self.index), "index"),
            (Some(&self.host_time), "host_time"),
            (self.leadoff.as_ref(), "leadoff"),
            (self.gap_before.as_ref(), "gap_before"),
            (self.leadoff_active.as_ref(), "leadoff_active"),
        ] {
            if let Some(a) = a {
                check(a, what)?;
            }
        }
        // The counts, channels by samples, kept in the order they were stored.
        let mut kept = Vec::with_capacity(nch * keep.len());
        let cols = keep.len();
        let data = if self.counts.fortran {
            for &i in &keep {
                for r in &rows {
                    kept.push(r[i]);
                }
            }
            self.counts.data.take(&kept)
        } else {
            for r in &rows {
                for &i in &keep {
                    kept.push(r[i]);
                }
            }
            self.counts.data.take(&kept)
        };
        self.counts = Array { shape: vec![nch, cols], fortran: self.counts.fortran && nch > 1, data };
        let take = |a: &Array| Array::vector(a.data.take(&keep));
        self.index = take(&self.index);
        self.host_time = take(&self.host_time);
        self.device_ticks = Some(take(ticks));
        self.leadoff = self.leadoff.as_ref().map(take);
        self.gap_before = self.gap_before.as_ref().map(take);
        self.leadoff_active = self.leadoff_active.as_ref().map(take);
        self.repairs.push(format!(
            "every packet was delivered {} times over, so {} of {} samples were copies. Removed: {} samples remain. \
             (Two BLE subscriptions were open on the instrument at once.)",
            pyfmt::general(dupes, 6),
            pyfmt::grouped(n_extra as i64),
            pyfmt::grouped(n as i64),
            pyfmt::grouped(cols as i64)
        ));
        Ok(())
    }

    /// A clock fitted at a skew no pair of oscillators can have stretched
    /// every sample time. Where the recording kept the sync exchanges, the
    /// times are restamped from the device clock with the skew pinned at
    /// one, exactly as the Python library restamps them.
    fn repin_clock(&mut self) {
        let empty = Value::Object(Vec::new());
        let c = match self.meta.get("clock") {
            Some(v) if v.truthy() => v.clone(),
            _ => empty,
        };
        let skew = c.field("skew");
        if !skew.truthy() {
            return;
        }
        let Some(skew) = skew.as_f64() else { return };
        if (skew - 1.0).abs() * 1e6 <= 5000.0 {
            return;
        }
        let ppm = (skew - 1.0).abs() * 1e6;
        let off = c.field("offset");
        let anchors = match c.field("anchors") {
            Value::Array(a) => a.clone(),
            _ => Vec::new(),
        };
        let tick_hz = self.meta.field("tick_hz");
        let tick_ok = tick_hz.truthy();
        if off.is_null() || anchors.len() < 2 || !tick_ok || self.device_ticks.is_none() {
            self.repairs.push(format!(
                "the clock was fitted at skew {} ({} ppm), which no pair of oscillators can do, and this capture does \
                 not carry the exchanges needed to refit it. Every sample time is stretched by that factor. Counts are \
                 unaffected.",
                pyfmt::fixed(skew, 4),
                pyfmt::grouped_fixed(ppm, 0)
            ));
            return;
        }
        let (Some(off), Some(tick_hz)) = (off.as_f64(), tick_hz.as_f64()) else { return };
        let pair = |a: &Value, i: usize| match a {
            Value::Array(p) => p.get(i).and_then(Value::as_f64).unwrap_or(f64::NAN),
            _ => f64::NAN,
        };
        let real: Vec<f64> = anchors.iter().map(|a| pair(a, 0)).collect();
        let mono: Vec<f64> = anchors.iter().map(|a| pair(a, 1)).collect();
        let dev: Vec<f64> = mono.iter().map(|m| (m - off) / skew).collect();
        // numpy's argmin takes the first of equal minima, and a NaN wins.
        let mut at = 0;
        for (i, &m) in mono.iter().enumerate() {
            if m.is_nan() {
                at = i;
                break;
            }
            if m < mono[at] {
                at = i;
            }
        }
        let mono_min = mono[at];
        let frame = real[at] - mono_min;
        let shift = pyfmt::numpy_mean(&mono) - pyfmt::numpy_mean(&dev);
        let ticks = self.device_ticks.as_ref().expect("checked above");
        let ht: Vec<f64> = (0..ticks.data.len()).map(|i| ticks.data.f64_at(i) / tick_hz + shift + frame).collect();
        let span = if ht.is_empty() { 0.0 } else { ht[ht.len() - 1] - ht[0] };
        let span_s = c.field("span_s");
        let baseline = if span_s.truthy() { span_s.as_f64().unwrap_or(0.0) } else { 0.0 };
        self.host_time = Array::vector(Data::F64(ht));
        self.repairs.push(format!(
            "the clock was fitted at skew {} ({} ppm) over a {} s baseline, stretching every sample time by that \
             factor. Restamped from the device clock with the skew pinned at 1.0; the samples now span {} s.",
            pyfmt::fixed(skew, 4),
            pyfmt::grouped_fixed(ppm, 0),
            pyfmt::fixed(baseline, 2),
            pyfmt::grouped_fixed(span, 0)
        ));
    }
}
