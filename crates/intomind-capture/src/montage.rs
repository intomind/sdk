//! Where the electrodes are, as data: the part of the Python library's
//! `montage` module a manifest is built with.
//!
//! A montage is a list of channels, each a difference between two named
//! sites, with the bias and the shared reference beside them. It is read
//! from what an application declares, which arrives as JSON, so every rule
//! here is Python's rule on a Python dict, refusal words included.

use crate::pyjson::{repr_str, Value};

pub(crate) const BIPOLAR: &str = "bipolar";
pub(crate) const REFERENTIAL: &str = "referential";

/// The 10-20 and 10-10 names a site's `ten_twenty` may take.
const TEN_TWENTY: &[&str] = &[
    "Nz", "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F9", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6",
    "F8", "F10", "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "FT10", "T9", "T7", "C5", "C3",
    "C1", "Cz", "C2", "C4", "C6", "T8", "T10", "TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "TP10", "P9", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "P10", "PO7", "PO3", "POz", "PO4", "PO8", "O1",
    "Oz", "O2", "Iz", "A1", "A2", "M1", "M2",
];

/// Whether a value is one of the 10-20 and 10-10 names.
pub(crate) fn ten_twenty(v: &Value) -> bool {
    matches!(v, Value::Str(s) if TEN_TWENTY.contains(&s.as_str()))
}

/// Why a montage or a placement could not be recorded.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Fail {
    /// `MontageError`: the montage as written cannot be true. A manifest
    /// records the words.
    Montage(String),
    /// `PlacementError`: the placement cannot describe a real head. A
    /// manifest records the words.
    Placement(String),
    /// What Python raises past both, a missing key or a value of the wrong
    /// kind, and so what stops a recording from being saved there too.
    Other(String),
}

impl Fail {
    /// Python's `KeyError` on a missing key.
    pub(crate) fn key(key: &str) -> Fail {
        Fail::Other(format!("KeyError: {}", repr_str(key)))
    }
}

/// `d[key]`.
pub(crate) fn item<'a>(d: &'a Value, key: &str) -> Result<&'a Value, Fail> {
    match d {
        Value::Object(_) => d.get(key).ok_or_else(|| Fail::key(key)),
        other => Err(Fail::Other(format!("TypeError: {} is not a dict", other.py_repr()))),
    }
}

/// `d.get(key)`, which needs a dict.
pub(crate) fn get<'a>(d: &'a Value, key: &str) -> Result<&'a Value, Fail> {
    match d {
        Value::Object(_) => Ok(d.field(key)),
        other => Err(Fail::Other(format!("AttributeError: {} has no get", other.py_repr()))),
    }
}

/// `x or []`, for a value that is then iterated as a list.
pub(crate) fn list(v: &Value) -> Result<Vec<&Value>, Fail> {
    match v {
        Value::Array(a) => Ok(a.iter().collect()),
        v if !v.truthy() => Ok(Vec::new()),
        other => Err(Fail::Other(format!("TypeError: {} is not a list", other.py_repr()))),
    }
}

/// Whether `sorted(idx) == list(range(1, len(idx) + 1))`.
pub(crate) fn one_to_n(idx: &[&Value]) -> Result<bool, Fail> {
    if idx.iter().all(|v| v.is_number()) {
        let mut x: Vec<f64> = idx.iter().map(|v| v.as_f64().unwrap_or(f64::NAN)).collect();
        x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        return Ok(x.iter().enumerate().all(|(i, v)| *v == (i + 1) as f64));
    }
    if idx.len() <= 1 || idx.iter().all(|v| matches!(v, Value::Str(_))) {
        return Ok(false);
    }
    Err(Fail::Other("TypeError: '<' not supported between the channel indices".into()))
}

/// `int(value)`, for a channel index.
fn int_of(v: &Value) -> Result<i64, Fail> {
    v.as_i64().ok_or_else(|| Fail::Other(format!("ValueError: invalid literal for int(): {}", v.py_repr())))
}

/// One electrode position, refused when it has no name or claims a 10-20
/// name it is not.
pub(crate) fn site(name: &Value, ten_twenty_name: &Value, description: &Value) -> Result<Value, Fail> {
    if !name.truthy() {
        return Err(Fail::Montage("a site needs a name".into()));
    }
    if !ten_twenty_name.is_null() && !ten_twenty(ten_twenty_name) {
        return Err(Fail::Montage(format!("{} is not a 10-20/10-10 position", ten_twenty_name.py_repr())));
    }
    Ok(Value::obj([
        ("name", name.clone()),
        ("ten_twenty", ten_twenty_name.clone()),
        ("description", description.clone()),
    ]))
}

/// One recorded channel: a difference between two sites.
pub(crate) fn channel(
    index: &Value,
    derivation: &Value,
    positive: Value,
    negative: Value,
    label: &Value,
) -> Result<Value, Fail> {
    if !is_derivation(derivation) {
        return Err(Fail::Montage(format!(
            "derivation must be one of ('bipolar', 'referential'), got {}",
            derivation.py_repr()
        )));
    }
    let i = int_of(index)?;
    if i < 1 {
        return Err(Fail::Montage("channel index is 1-based".into()));
    }
    let label = if label.truthy() {
        label.clone()
    } else {
        Value::str(format!("{}-{}", item(&positive, "name")?.py_str(), item(&negative, "name")?.py_str()))
    };
    Ok(Value::obj([
        ("index", Value::int(i)),
        ("derivation", derivation.clone()),
        ("positive", positive),
        ("negative", negative),
        ("label", label),
    ]))
}

fn is_derivation(v: &Value) -> bool {
    matches!(v, Value::Str(s) if s == BIPOLAR || s == REFERENTIAL)
}

/// A montage, assembled and checked.
pub(crate) fn build(
    channels: Vec<Value>,
    electrode: Value,
    bias: Value,
    reference: Value,
    prose: Value,
    source: Value,
) -> Result<Value, Fail> {
    let m = Value::obj([
        ("electrode", electrode),
        ("channels", Value::Array(channels)),
        ("bias", bias),
        ("reference", reference),
        ("prose", prose),
        ("source", source),
    ]);
    validate(&m)?;
    Ok(m)
}

/// Refuse a montage that cannot describe a real recording.
pub(crate) fn validate(m: &Value) -> Result<(), Fail> {
    let chans = list(get(m, "channels")?)?;
    if chans.is_empty() {
        return Err(Fail::Montage("a montage with no channels describes nothing".into()));
    }
    let idx: Vec<&Value> = chans.iter().map(|c| item(c, "index")).collect::<Result<_, _>>()?;
    if !one_to_n(&idx)? {
        return Err(Fail::Montage(format!(
            "channel indices must be 1..n with no gaps, got {}",
            Value::Array(idx.iter().map(|v| (*v).clone()).collect()).py_repr()
        )));
    }
    let reference = get(m, "reference")?;
    for c in &chans {
        let at = item(c, "index")?.py_str();
        let derivation = item(c, "derivation")?;
        if !is_derivation(derivation) {
            return Err(Fail::Montage(format!("ch{at}: bad derivation {}", derivation.py_repr())));
        }
        for pole in ["positive", "negative"] {
            let s = get(c, pole)?;
            if !s.truthy() || !get(s, "name")?.truthy() {
                return Err(Fail::Montage(format!("ch{at}: {pole} site is missing")));
            }
            let tt = get(s, "ten_twenty")?;
            if !tt.is_null() && !ten_twenty(tt) {
                return Err(Fail::Montage(format!("ch{at}: {} is not a 10-20/10-10 position", tt.py_repr())));
            }
        }
        if matches!(derivation, Value::Str(d) if d == REFERENTIAL) {
            if !reference.truthy() {
                return Err(Fail::Montage(format!("ch{at} is referential but the montage names no reference")));
            }
            let theirs = item(item(c, "negative")?, "name")?;
            let ours = item(reference, "name")?;
            if !theirs.py_eq(ours) {
                return Err(Fail::Montage(format!(
                    "ch{at} is referential against {}, but the montage's reference is {}",
                    theirs.py_repr(),
                    ours.py_repr()
                )));
            }
        }
    }
    Ok(())
}

/// How many channels a montage has, 0 when it names none.
pub(crate) fn channel_count(m: &Value) -> usize {
    match m.get("channels") {
        Some(Value::Array(a)) => a.len(),
        _ => 0,
    }
}

/// The montage a site declares, checked, or None when it declares none.
pub(crate) fn from_site(site_cfg: &Value) -> Result<Option<Value>, Fail> {
    let m = if site_cfg.truthy() { get(site_cfg, "montage")? } else { &Value::Null };
    if !m.truthy() {
        return Ok(None);
    }
    validate(m)?;
    Ok(Some(m.clone()))
}
