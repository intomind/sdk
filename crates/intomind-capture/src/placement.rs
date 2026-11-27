//! Where the electrodes went, and what is under them: the part of the
//! Python library's `placement` module a manifest is built with.
//!
//! A layout is what a device offers: its slots and how they are wired into
//! channels. A placement is where a person put it, this time. A manifest
//! records both, the placement resolved against the scalp positions and the
//! regions under them, and the montage generated from the two. Both arrive
//! as JSON from an application, so every rule here is Python's rule on a
//! Python dict, refusal words included.
//!
//! What is under a 10-10 position was measured, as the Python library has
//! it: for 65 of the 77 positions, Table 3 of C. L. Scrivener and A. T.
//! Reader, "Variability of EEG electrode positions and their underlying brain
//! regions: visualizing gel artifacts from a simultaneous EEG-fMRI dataset",
//! Brain and Behavior 12 (2022), e2476, doi:10.1002/brb3.2476, published
//! under the Creative Commons Attribution 4.0 International license,
//! <https://creativecommons.org/licenses/by/4.0/>. Changed here: the paper's
//! abbreviations are written out as the paper defines them, the names are in
//! lower case, the one "Precuneous" is spelled "precuneus" as the paper
//! spells it elsewhere, and its column of Brodmann areas is left out.

use crate::montage::{self as m, get, item, list, one_to_n, Fail};
use crate::pyfmt;
use crate::pyjson::Value;

const LEFT: &str = "left";
const RIGHT: &str = "right";
const MIDLINE: &str = "midline";
const RECORDING: &str = "recording";
const BIAS: &str = "bias";
const REFERENCE: &str = "reference";

/// What can be under an electrode: key, words, and kind.
const REGIONS: [(&str, &str, &str); 44] = [
    ("frontal-pole", "frontal pole", "cortical"),
    ("dlpfc", "dorsolateral prefrontal", "cortical"),
    ("vlpfc", "ventrolateral prefrontal", "cortical"),
    ("orbitofrontal", "orbitofrontal", "cortical"),
    ("premotor", "premotor and supplementary motor", "cortical"),
    ("motor", "primary motor", "cortical"),
    ("somatosensory", "primary somatosensory", "cortical"),
    ("superior-parietal", "superior parietal", "cortical"),
    ("tpj", "inferior parietal and temporoparietal junction", "cortical"),
    ("temporal-pole", "temporal pole", "cortical"),
    ("superior-temporal", "superior temporal", "cortical"),
    ("inferior-temporal", "middle and inferior temporal", "cortical"),
    ("lateral-occipital", "lateral occipital", "cortical"),
    ("medial-occipital", "medial occipital", "cortical"),
    ("temporalis", "temporalis (jaw muscle)", "muscle"),
    ("frontalis", "frontalis (brow muscle)", "muscle"),
    ("ocular", "eye movement", "ocular"),
    ("mastoid-bone", "mastoid bone, little cortex beneath", "bone"),
    // The regions a study found under the 10-10 positions, named as it
    // names them with its abbreviations written out. Two of them, the frontal
    // pole and the temporal pole, are above already under the same names.
    ("superior-frontal-gyrus", "superior frontal gyrus", "cortical"),
    ("middle-frontal-gyrus", "middle frontal gyrus", "cortical"),
    ("inferior-frontal-gyrus-pars-triangularis", "inferior frontal gyrus (pars triangularis)", "cortical"),
    ("inferior-frontal-gyrus-pars-opercularis", "inferior frontal gyrus (pars opercularis)", "cortical"),
    ("juxtapositional-lobule-cortex", "juxtapositional lobule cortex (formerly supplementary motor cortex)", "cortical"),
    ("precentral-gyrus", "precentral gyrus", "cortical"),
    ("postcentral-gyrus", "postcentral gyrus", "cortical"),
    ("central-opercular-cortex", "central opercular cortex", "cortical"),
    ("anterior-supramarginal-gyrus", "anterior supramarginal gyrus", "cortical"),
    ("posterior-supramarginal-gyrus", "posterior supramarginal gyrus", "cortical"),
    ("angular-gyrus", "angular gyrus", "cortical"),
    ("superior-parietal-lobule", "superior parietal lobule", "cortical"),
    ("precuneus-cortex", "precuneus cortex", "cortical"),
    ("anterior-superior-temporal-gyrus", "anterior superior temporal gyrus", "cortical"),
    ("posterior-superior-temporal-gyrus", "posterior superior temporal gyrus", "cortical"),
    ("planum-temporale", "planum temporale", "cortical"),
    ("anterior-middle-temporal-gyrus", "anterior middle temporal gyrus", "cortical"),
    ("posterior-middle-temporal-gyrus", "posterior middle temporal gyrus", "cortical"),
    ("middle-temporal-gyrus-temporooccipital-part", "middle temporal gyrus (temporooccipital part)", "cortical"),
    ("anterior-inferior-temporal-gyrus", "anterior inferior temporal gyrus", "cortical"),
    ("posterior-inferior-temporal-gyrus", "posterior inferior temporal gyrus", "cortical"),
    ("inferior-temporal-gyrus-temporooccipital-part", "inferior temporal gyrus (temporooccipital part)", "cortical"),
    ("superior-lateral-occipital-cortex", "superior lateral occipital cortex", "cortical"),
    ("inferior-lateral-occipital-cortex", "inferior lateral occipital cortex", "cortical"),
    ("occipital-pole", "occipital pole", "cortical"),
    ("cuneal-cortex", "cuneal cortex", "cortical"),
];

/// Where a person can put an electrode: key, words, the sides it comes in
/// with each side's nearest 10-10 position, and what it is over, most
/// dominant first.
type Zone = (&'static str, &'static str, &'static [(&'static str, &'static str)], &'static [&'static str]);

const ZONES: [Zone; 18] = [
    (
        "forehead-outer",
        "outer forehead, above the brow's outer end",
        &[(LEFT, "AF7"), (RIGHT, "AF8")],
        &["frontal-pole", "dlpfc", "ocular", "frontalis"],
    ),
    (
        "forehead-inner",
        "inner forehead, above the brow",
        &[(LEFT, "Fp1"), (RIGHT, "Fp2")],
        &["frontal-pole", "orbitofrontal", "ocular", "frontalis"],
    ),
    ("forehead-center", "center of the forehead", &[(MIDLINE, "Fpz")], &["frontal-pole", "ocular", "frontalis"]),
    (
        "hairline",
        "forehead at the hairline",
        &[(LEFT, "F5"), (RIGHT, "F6"), (MIDLINE, "Fz")],
        &["dlpfc", "vlpfc", "frontalis"],
    ),
    ("temple-front", "temple, front", &[(LEFT, "F7"), (RIGHT, "F8")], &["vlpfc", "temporalis", "temporal-pole"]),
    ("temple-back", "temple, back", &[(LEFT, "FT7"), (RIGHT, "FT8")], &["temporalis", "superior-temporal", "vlpfc"]),
    ("in-front-of-ear", "in front of the ear", &[(LEFT, "FT9"), (RIGHT, "FT10")], &["temporalis", "temporal-pole"]),
    ("above-ear", "above the ear", &[(LEFT, "T7"), (RIGHT, "T8")], &["superior-temporal", "temporalis"]),
    (
        "behind-ear",
        "behind the ear, on the bone",
        &[(LEFT, "TP9"), (RIGHT, "TP10")],
        &["mastoid-bone", "inferior-temporal"],
    ),
    ("behind-above-ear", "behind and above the ear", &[(LEFT, "TP7"), (RIGHT, "TP8")], &["tpj", "superior-temporal"]),
    (
        "side-of-head",
        "side of the head, above and forward of the ear",
        &[(LEFT, "C5"), (RIGHT, "C6")],
        &["somatosensory", "motor"],
    ),
    ("crown-side", "crown, off to the side", &[(LEFT, "C3"), (RIGHT, "C4")], &["motor", "somatosensory", "premotor"]),
    ("top-of-head", "top of the head", &[(MIDLINE, "Cz")], &["motor", "premotor"]),
    (
        "back-of-head-side",
        "back of the head, off to the side",
        &[(LEFT, "P3"), (RIGHT, "P4")],
        &["superior-parietal", "tpj"],
    ),
    ("crown-back", "crown, toward the back", &[(MIDLINE, "Pz")], &["superior-parietal"]),
    (
        "back-of-head-low",
        "back of the head, low",
        &[(LEFT, "O1"), (RIGHT, "O2")],
        &["lateral-occipital", "medial-occipital"],
    ),
    ("back-of-head-center", "back of the head, center", &[(MIDLINE, "Oz")], &["medial-occipital"]),
    ("nape", "the bump at the base of the skull", &[(MIDLINE, "Iz")], &["medial-occipital"]),
];

/// Every scalp position, in the order the Python library builds them.
fn scalp_table() -> Vec<(String, Value)> {
    let mut out = Vec::new();
    for (zone, human, sides, over) in ZONES {
        for (side, tt) in sides {
            let (key, words) = if *side == MIDLINE {
                (zone.to_string(), human.to_string())
            } else {
                (format!("{side}-{zone}"), format!("{side} {human}"))
            };
            out.push((
                key.clone(),
                Value::obj([
                    ("key", Value::str(key)),
                    ("zone", Value::str(zone)),
                    ("side", Value::str(*side)),
                    ("human", Value::str(words)),
                    ("ten_ten", Value::str(*tt)),
                    ("over", Value::Array(over.iter().map(|r| Value::str(*r)).collect())),
                ]),
            ));
        }
    }
    out
}

/// The whole 10-10 system, front to back and left to right, then the
/// earlobes and the mastoids: the positions a placement may name by their
/// codes.
const TEN_TEN: [&str; 77] = [
    "Nz", "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F9", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6",
    "F8", "F10", "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "FT10", "T9", "T7", "C5", "C3",
    "C1", "Cz", "C2", "C4", "C6", "T8", "T10", "TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "TP10", "P9", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "P10", "PO7", "PO3", "POz", "PO4", "PO8", "O1",
    "Oz", "O2", "Iz", "A1", "A2", "M1", "M2",
];

/// The side a 10-10 code names: z the midline, an odd number the left, an
/// even one the right.
fn side_of(code: &str) -> &'static str {
    if code.ends_with('z') {
        return MIDLINE;
    }
    let number = code.trim_start_matches(|c: char| c.is_ascii_alphabetic());
    if number.parse::<u32>().is_ok_and(|n| n % 2 == 1) {
        LEFT
    } else {
        RIGHT
    }
}

/// What is under each 10-10 position the study covered, as measured: Table
/// 3 of the study this module credits, row by row in its order, every region
/// it found with its share, the percentage of the participants usable at
/// that position whose electrode lay over it, as the paper prints it. The
/// other positions are absent: nothing is said about them.
const UNDER: [(&str, &[(&str, f64)]); 65] = [
    ("AF3", &[("frontal-pole", 100.0)]),
    ("AF4", &[("frontal-pole", 100.0)]),
    ("AF7", &[("frontal-pole", 100.0)]),
    ("AF8", &[("frontal-pole", 100.0)]),
    ("AFz", &[("frontal-pole", 62.5), ("superior-frontal-gyrus", 37.5)]),
    ("C1", &[("precentral-gyrus", 71.0), ("postcentral-gyrus", 29.0)]),
    ("C2", &[("precentral-gyrus", 87.5), ("postcentral-gyrus", 12.5)]),
    ("C3", &[("postcentral-gyrus", 87.5), ("precentral-gyrus", 12.5)]),
    ("C4", &[("postcentral-gyrus", 85.0), ("precentral-gyrus", 15.0)]),
    ("C5", &[("postcentral-gyrus", 50.0), ("anterior-supramarginal-gyrus", 44.4), ("precentral-gyrus", 5.6)]),
    ("C6", &[("anterior-supramarginal-gyrus", 72.0), ("postcentral-gyrus", 28.0)]),
    ("CP1", &[("superior-parietal-lobule", 66.67), ("postcentral-gyrus", 27.78),
        ("superior-lateral-occipital-cortex", 5.56)]),
    ("CP2", &[("superior-parietal-lobule", 80.0), ("postcentral-gyrus", 13.0),
        ("superior-lateral-occipital-cortex", 7.0)]),
    ("CP3", &[("posterior-supramarginal-gyrus", 47.0), ("superior-parietal-lobule", 27.0), ("angular-gyrus", 13.0),
        ("postcentral-gyrus", 13.0)]),
    ("CP4", &[("angular-gyrus", 50.0), ("superior-parietal-lobule", 42.0), ("postcentral-gyrus", 8.0)]),
    ("CP5", &[("posterior-supramarginal-gyrus", 61.11), ("anterior-supramarginal-gyrus", 16.67),
        ("angular-gyrus", 11.11), ("posterior-superior-temporal-gyrus", 5.56),
        ("superior-lateral-occipital-cortex", 5.56)]),
    ("CP6", &[("angular-gyrus", 53.0), ("posterior-supramarginal-gyrus", 40.0),
        ("superior-lateral-occipital-cortex", 7.0)]),
    ("CPz", &[("postcentral-gyrus", 62.5), ("precuneus-cortex", 18.75), ("superior-parietal-lobule", 12.5),
        ("superior-lateral-occipital-cortex", 6.25)]),
    ("Cz", &[("precentral-gyrus", 87.0), ("postcentral-gyrus", 13.0)]),
    ("F1", &[("superior-frontal-gyrus", 71.0), ("frontal-pole", 29.0)]),
    ("F2", &[("superior-frontal-gyrus", 77.0), ("frontal-pole", 23.0)]),
    ("F3", &[("middle-frontal-gyrus", 64.0), ("frontal-pole", 36.0)]),
    ("F4", &[("frontal-pole", 53.0), ("middle-frontal-gyrus", 47.0)]),
    ("F5", &[("middle-frontal-gyrus", 65.0), ("frontal-pole", 35.0)]),
    ("F6", &[("middle-frontal-gyrus", 50.0), ("frontal-pole", 50.0)]),
    ("F7", &[("inferior-frontal-gyrus-pars-triangularis", 83.0), ("frontal-pole", 11.0),
        ("inferior-frontal-gyrus-pars-opercularis", 6.0)]),
    ("F8", &[("inferior-frontal-gyrus-pars-triangularis", 66.67), ("inferior-frontal-gyrus-pars-opercularis", 16.67),
        ("middle-frontal-gyrus", 5.56), ("precentral-gyrus", 5.56), ("frontal-pole", 5.56)]),
    ("FC1", &[("superior-frontal-gyrus", 93.75), ("middle-frontal-gyrus", 6.25)]),
    ("FC2", &[("superior-frontal-gyrus", 82.0), ("precentral-gyrus", 18.0)]),
    ("FC3", &[("middle-frontal-gyrus", 89.0), ("precentral-gyrus", 11.0)]),
    ("FC4", &[("middle-frontal-gyrus", 92.0), ("precentral-gyrus", 8.0)]),
    ("FC5", &[("precentral-gyrus", 88.0), ("middle-frontal-gyrus", 6.0),
        ("inferior-frontal-gyrus-pars-opercularis", 6.0)]),
    ("FC6", &[("precentral-gyrus", 81.0), ("postcentral-gyrus", 13.0), ("middle-frontal-gyrus", 6.0)]),
    ("FCz", &[("juxtapositional-lobule-cortex", 66.7), ("superior-frontal-gyrus", 33.3)]),
    ("Fp1", &[("frontal-pole", 100.0)]),
    ("Fp2", &[("frontal-pole", 100.0)]),
    ("Fpz", &[("frontal-pole", 100.0)]),
    ("FT10", &[("anterior-middle-temporal-gyrus", 66.7), ("posterior-inferior-temporal-gyrus", 11.1),
        ("temporal-pole", 11.1), ("posterior-middle-temporal-gyrus", 5.6), ("anterior-inferior-temporal-gyrus", 5.6)]),
    ("FT7", &[("precentral-gyrus", 64.29), ("temporal-pole", 14.29), ("anterior-superior-temporal-gyrus", 14.29),
        ("inferior-frontal-gyrus-pars-opercularis", 7.14)]),
    ("FT8", &[("precentral-gyrus", 62.5), ("anterior-superior-temporal-gyrus", 25.0),
        ("central-opercular-cortex", 6.25), ("posterior-middle-temporal-gyrus", 6.25)]),
    ("FT9", &[("anterior-middle-temporal-gyrus", 42.0), ("temporal-pole", 37.0),
        ("posterior-middle-temporal-gyrus", 16.0), ("anterior-inferior-temporal-gyrus", 5.0)]),
    ("Fz", &[("superior-frontal-gyrus", 100.0)]),
    ("O1", &[("occipital-pole", 100.0)]),
    ("O2", &[("occipital-pole", 100.0)]),
    ("Oz", &[("occipital-pole", 100.0)]),
    ("P1", &[("superior-lateral-occipital-cortex", 95.0), ("superior-parietal-lobule", 5.0)]),
    ("P2", &[("superior-lateral-occipital-cortex", 100.0)]),
    ("P3", &[("superior-lateral-occipital-cortex", 95.0), ("angular-gyrus", 5.0)]),
    ("P4", &[("superior-lateral-occipital-cortex", 100.0)]),
    ("P5", &[("superior-lateral-occipital-cortex", 89.47), ("inferior-lateral-occipital-cortex", 5.26),
        ("angular-gyrus", 5.26)]),
    ("P6", &[("superior-lateral-occipital-cortex", 100.0)]),
    ("P7", &[("inferior-lateral-occipital-cortex", 75.0), ("superior-lateral-occipital-cortex", 20.0),
        ("middle-temporal-gyrus-temporooccipital-part", 5.0)]),
    ("P8", &[("inferior-lateral-occipital-cortex", 85.0), ("superior-lateral-occipital-cortex", 10.0),
        ("angular-gyrus", 5.0)]),
    ("PO3", &[("superior-lateral-occipital-cortex", 65.0), ("occipital-pole", 35.0)]),
    ("PO4", &[("occipital-pole", 58.0), ("superior-lateral-occipital-cortex", 42.0)]),
    ("PO7", &[("inferior-lateral-occipital-cortex", 80.0), ("superior-lateral-occipital-cortex", 15.0),
        ("occipital-pole", 5.0)]),
    ("PO8", &[("occipital-pole", 33.33), ("superior-lateral-occipital-cortex", 33.33),
        ("inferior-lateral-occipital-cortex", 33.33)]),
    ("POz", &[("cuneal-cortex", 37.5), ("occipital-pole", 31.3), ("superior-lateral-occipital-cortex", 25.0),
        ("precuneus-cortex", 6.3)]),
    ("Pz", &[("precuneus-cortex", 60.0), ("superior-lateral-occipital-cortex", 33.33),
        ("superior-parietal-lobule", 6.66)]),
    ("T7", &[("posterior-superior-temporal-gyrus", 72.2), ("posterior-middle-temporal-gyrus", 16.6),
        ("anterior-superior-temporal-gyrus", 5.6), ("anterior-supramarginal-gyrus", 5.6)]),
    ("T8", &[("posterior-superior-temporal-gyrus", 52.9), ("posterior-middle-temporal-gyrus", 29.4),
        ("planum-temporale", 5.9), ("anterior-supramarginal-gyrus", 5.9), ("central-opercular-cortex", 5.9)]),
    ("TP7", &[("middle-temporal-gyrus-temporooccipital-part", 73.7), ("posterior-supramarginal-gyrus", 15.8),
        ("posterior-middle-temporal-gyrus", 10.5)]),
    ("TP8", &[("middle-temporal-gyrus-temporooccipital-part", 75.0), ("angular-gyrus", 20.0),
        ("posterior-supramarginal-gyrus", 5.0)]),
    ("TP9", &[("inferior-temporal-gyrus-temporooccipital-part", 100.0)]),
    ("TP10", &[("inferior-temporal-gyrus-temporooccipital-part", 89.0),
        ("middle-temporal-gyrus-temporooccipital-part", 11.0)]),
];

const UNDER_SOURCE: &str = "Scrivener and Reader 2022, doi:10.1002/brb3.2476";

/// The study's row for a 10-10 position, empty where it did not cover it.
fn under(code: &str) -> &'static [(&'static str, f64)] {
    UNDER.iter().find(|(c, _)| *c == code).map_or(&[], |(_, row)| *row)
}

/// One 10-10 position as the Python library offers it, with the regions the
/// study found under it, if it covered it.
fn ten_ten(code: &str) -> Value {
    Value::obj([
        ("key", Value::str(code)),
        ("zone", Value::Null),
        ("side", Value::str(side_of(code))),
        ("human", Value::str(code)),
        ("ten_ten", Value::str(code)),
        ("over", Value::Array(under(code).iter().map(|(r, _)| Value::str(*r)).collect())),
    ])
}

/// The position for anywhere the 10-10 system has no position, the place
/// written in the placement's note.
fn other() -> Value {
    Value::obj([
        ("key", Value::str("other")),
        ("zone", Value::Null),
        ("side", Value::Null),
        ("human", Value::str("a position off the 10-10 system")),
        ("ten_ten", Value::Null),
        ("over", Value::Array(Vec::new())),
    ])
}

/// One position by key, as the Python library's `placement.scalp` reads
/// it: a 10-10 code, "other", or the key of one of the 32 named positions
/// of `scalp_table`, such as `left-forehead-outer`. Any other text, a
/// number or a null is refused in the library's own words, the names in
/// them the library's, because the refusal never reaches a Rust program as
/// an error: a recording's manifest keeps it under `montage_error`, word for
/// word as the library writes it, in the file format every program reads.
pub(crate) fn scalp(key: &Value) -> Result<Value, Fail> {
    let table = scalp_table();
    if matches!(key, Value::Array(_) | Value::Object(_)) {
        return Err(Fail::Other(format!("TypeError: unhashable type: {}", key.rust_repr())));
    }
    if let Value::Str(k) = key {
        if let Some((_, v)) = table.iter().find(|(t, _)| t == k) {
            return Ok(v.clone());
        }
        if TEN_TEN.contains(&k.as_str()) {
            return Ok(ten_ten(k));
        }
        if k == "other" {
            return Ok(other());
        }
    }
    Err(Fail::Placement(format!(
        "{} is not a position key. There are {} on offer, the 10-10 codes and \"other\" for anywhere else, its \
         place written in the note; `placement.positions()` lists them. The {} names in `placement.SCALP` are \
         accepted as well.",
        key.py_repr(),
        TEN_TEN.len() + 1,
        table.len()
    )))
}

fn region(key: &Value) -> Option<(&'static str, &'static str, &'static str)> {
    match key {
        Value::Str(k) => REGIONS.iter().copied().find(|(r, _, _)| r == k),
        _ => None,
    }
}

/// What is under a scalp position, its side carried down, each entry
/// marked as inferred. Under a 10-10 position each region carries its share
/// and the study it comes from. A position the study did not cover, and
/// "other", carry no region.
pub(crate) fn regions_under(scalp_key: &Value) -> Result<Vec<Value>, Fail> {
    let s = scalp(scalp_key)?;
    let side = s.field("side").py_str();
    let code = s.field("key").py_str();
    let shares: &[(&str, f64)] = if TEN_TEN.contains(&code.as_str()) { under(&code) } else { &[] };
    let mut out = Vec::new();
    for key in list(s.field("over"))? {
        let (r, human, kind) = region(key).ok_or_else(|| Fail::key(&key.py_str()))?;
        let sided = if side != MIDLINE && kind != "ocular" { format!("{side} {human}") } else { human.to_string() };
        let mut entry = vec![
            ("region", Value::str(r)),
            ("human", Value::str(sided)),
            ("kind", Value::str(kind)),
            ("side", Value::str(side.clone())),
            ("derived", Value::Bool(true)),
        ];
        if let Some((_, share)) = shares.iter().find(|(k, _)| *k == r) {
            entry.push(("share", Value::Float(*share)));
            entry.push(("source", Value::str(UNDER_SOURCE)));
        }
        out.push(Value::Object(entry.into_iter().map(|(k, v)| (k.to_string(), v)).collect()));
    }
    Ok(out)
}

/// A key a Python dict can hold. A list or an object cannot be one.
fn hashable(v: &Value) -> Result<(), Fail> {
    match v {
        Value::Array(_) | Value::Object(_) => Err(Fail::Other(format!("TypeError: unhashable type: {}", v.rust_repr()))),
        _ => Ok(()),
    }
}

/// `{s["key"]: s for s in slots}`: insertion order, a repeated key keeping
/// its first place and its last slot.
fn slots_by_key(slots: &[&Value]) -> Result<Vec<(Value, Value)>, Fail> {
    let mut out: Vec<(Value, Value)> = Vec::new();
    for s in slots {
        let k = item(s, "key")?;
        hashable(k)?;
        match out.iter_mut().find(|(have, _)| have.py_eq(k)) {
            Some(slot) => slot.1 = (*s).clone(),
            None => out.push((k.clone(), (*s).clone())),
        }
    }
    Ok(out)
}

fn has(slots: &[(Value, Value)], k: &Value) -> Result<bool, Fail> {
    hashable(k)?;
    Ok(slots.iter().any(|(have, _)| have.py_eq(k)))
}

fn slot_of<'a>(slots: &'a [(Value, Value)], k: &Value) -> Option<&'a Value> {
    slots.iter().find(|(have, _)| have.py_eq(k)).map(|(_, s)| s)
}

/// The keys of a JSON object, as values: what `for k in d` iterates.
fn keys(d: &Value) -> Result<Vec<Value>, Fail> {
    match d {
        Value::Object(o) => Ok(o.iter().map(|(k, _)| Value::str(k.clone())).collect()),
        v if !v.truthy() => Ok(Vec::new()),
        other => Err(Fail::Other(format!("AttributeError: {} has no keys", other.rust_repr()))),
    }
}

/// `k in d` for a JSON object, whose keys are strings.
fn in_object(d: &Value, k: &Value) -> Result<bool, Fail> {
    hashable(k)?;
    Ok(match (d, k) {
        (Value::Object(o), Value::Str(k)) => o.iter().any(|(have, _)| have == k),
        _ => false,
    })
}

/// What `for r in value` yields: a list's items, a string's characters,
/// an object's keys.
pub(crate) fn iterate(v: &Value) -> Result<Vec<Value>, Fail> {
    match v {
        Value::Array(a) => Ok(a.clone()),
        Value::Str(s) => Ok(s.chars().map(|c| Value::str(c.to_string())).collect()),
        Value::Object(_) => keys(v),
        other => Err(Fail::Other(format!("TypeError: {} is not iterable", other.rust_repr()))),
    }
}

fn role_is(s: &Value, role: &str) -> Result<bool, Fail> {
    Ok(matches!(item(s, "role")?, Value::Str(r) if r == role))
}

/// Refuse a layout that could not be wired.
fn validate_layout(f: &Value) -> Result<(), Fail> {
    let slots = slots_by_key(&list(get(f, "slots")?)?)?;
    if slots.is_empty() {
        return Err(Fail::Placement("a layout with no slots describes nothing".into()));
    }
    let chans = list(get(f, "channels")?)?;
    if chans.is_empty() {
        return Err(Fail::Placement(format!("{} declares no channels", get(f, "key")?.py_repr())));
    }
    let idx: Vec<&Value> = chans.iter().map(|c| item(c, "index")).collect::<Result<_, _>>()?;
    if !one_to_n(&idx)? {
        return Err(Fail::Placement(format!(
            "channel indices must be 1..n with no gaps, got {}",
            Value::Array(idx.iter().map(|v| (*v).clone()).collect()).py_repr()
        )));
    }
    let mut refs = Vec::new();
    for (_, s) in &slots {
        if role_is(s, REFERENCE)? {
            refs.push(s);
        }
    }
    for c in &chans {
        let at = item(c, "index")?.py_str();
        let derivation = item(c, "derivation")?;
        if !matches!(derivation, Value::Str(d) if d == m::BIPOLAR || d == m::REFERENTIAL) {
            return Err(Fail::Placement(format!("ch{at}: bad derivation {}", derivation.py_repr())));
        }
        for pole in ["positive", "negative"] {
            let named = get(c, pole)?;
            if !has(&slots, named)? {
                return Err(Fail::Placement(format!(
                    "ch{at}: {pole} names slot {}, which this layout does not have",
                    named.py_repr()
                )));
            }
        }
        if matches!(derivation, Value::Str(d) if d == m::REFERENTIAL) {
            if refs.len() != 1 {
                return Err(Fail::Placement(format!(
                    "ch{at} is single-ended, so this layout needs exactly one reference slot; it has {}",
                    refs.len()
                )));
            }
            let negative = item(c, "negative")?;
            let reference = item(refs[0], "key")?;
            if !negative.py_eq(reference) {
                return Err(Fail::Placement(format!(
                    "ch{at} is single-ended against {}, but the reference slot is {}",
                    negative.py_repr(),
                    reference.py_repr()
                )));
            }
        }
    }
    for p in list(get(f, "presets")?)? {
        validate_preset(f, p, &slots)?;
    }
    Ok(())
}

fn validate_preset(f: &Value, p: &Value, slots: &[(Value, Value)]) -> Result<(), Fail> {
    // `f['key']` is read only where the words need it, as Python reads it.
    let lay = || item(f, "key").map(|v| v.py_str());
    if !get(p, "key")?.truthy() {
        return Err(Fail::Placement(format!("{}: a preset needs a key", lay()?)));
    }
    let sites = get(p, "sites")?;
    let mut missing = Vec::new();
    for (k, s) in slots {
        if !in_object(sites, k)? && !role_is(s, BIAS)? {
            missing.push(k);
        }
    }
    if !missing.is_empty() {
        let mut names = Vec::new();
        for k in missing {
            match k {
                Value::Str(s) => names.push(s.clone()),
                other => {
                    return Err(Fail::Other(format!("TypeError: sequence item: expected str, {}", other.rust_repr())))
                }
            }
        }
        names.sort();
        return Err(Fail::Placement(format!(
            "{}/{}: no site for slot(s) {}",
            lay()?,
            item(p, "key")?.py_str(),
            names.join(", ")
        )));
    }
    if let Value::Object(o) = sites {
        for (k, v) in o {
            if !has(slots, &Value::str(k.clone()))? {
                return Err(Fail::Placement(format!(
                    "{}/{}: {} is not a slot",
                    lay()?,
                    item(p, "key")?.py_str(),
                    Value::str(k.clone()).py_repr()
                )));
            }
            scalp(v)?;
        }
    } else if sites.truthy() {
        return Err(Fail::Other("AttributeError: a preset's sites is not a dict".into()));
    }
    if !get(p, "rotation_deg")?.is_null() && !get(f, "rotatable")?.truthy() {
        let lay = lay()?;
        return Err(Fail::Placement(format!(
            "{lay}/{} states a rotation, but {lay} is not rotatable",
            item(p, "key")?.py_str()
        )));
    }
    Ok(())
}

/// A layout as an application declares it, checked. None for nothing.
pub(crate) fn layout_from(d: &Value) -> Result<Option<&Value>, Fail> {
    if !d.truthy() {
        return Ok(None);
    }
    validate_layout(d)?;
    Ok(Some(d))
}

/// `p.get(key, default)`.
fn get_or<'a>(d: &'a Value, key: &str, default: &'a Value) -> Result<&'a Value, Fail> {
    match d {
        Value::Object(_) => Ok(d.get(key).unwrap_or(default)),
        other => Err(Fail::Other(format!("AttributeError: {} has no get", other.rust_repr()))),
    }
}

/// Refuse a placement that does not put every electrode somewhere.
fn validate_placement(lay: &Value, p: &Value) -> Result<(), Fail> {
    let theirs = get(p, "layout")?;
    let ours = get(lay, "key")?;
    if !theirs.py_eq(ours) {
        return Err(Fail::Placement(format!(
            "this placement is for layout {}, not {}",
            theirs.py_repr(),
            ours.py_repr()
        )));
    }
    let name = || item(lay, "key").map(|v| v.py_str());
    let slots = slots_by_key(&list(item(lay, "slots")?)?)?;
    let sites = get(p, "sites")?;
    for k in keys(sites)? {
        if !has(&slots, &k)? {
            return Err(Fail::Placement(format!("{} is not a slot on {}", k.py_repr(), name()?)));
        }
        scalp(sites.field(&k.py_str()))?;
    }
    let yes = Value::Bool(true);
    for (k, s) in &slots {
        if in_object(sites, k)? {
            continue;
        }
        if role_is(s, BIAS)? && !get_or(p, "bias_driven", &yes)?.truthy() {
            continue;
        }
        return Err(Fail::Placement(format!(
            "{}: no site for slot {} ({})",
            name()?,
            k.py_repr(),
            item(s, "human")?.py_str()
        )));
    }
    let over = get(p, "over")?;
    for k in keys(over)? {
        if !has(&slots, &k)? {
            return Err(Fail::Placement(format!("override names {}, which is not a slot", k.py_repr())));
        }
        for r in iterate(over.field(&k.py_str()))? {
            if region(&r).is_none() {
                return Err(Fail::Placement(format!("{} is not a region; there are {}", r.py_repr(), REGIONS.len())));
            }
        }
    }
    let rotation = get(p, "rotation_deg")?;
    if !rotation.is_null() && !get(lay, "rotatable")?.truthy() {
        return Err(Fail::Placement(format!(
            "{} is not rotatable, so a rotation of {} means nothing",
            name()?,
            rotation.py_str()
        )));
    }
    Ok(())
}

/// The placement with every slot's site and regions filled in.
pub(crate) fn resolve(lay: &Value, p: &Value) -> Result<Value, Fail> {
    validate_placement(lay, p)?;
    let slots = slots_by_key(&list(item(lay, "slots")?)?)?;
    let sites = get(p, "sites")?;
    let over = get(p, "over")?;
    let mut out = Vec::new();
    for (key, s) in &slots {
        let site_key = match (sites, key) {
            (Value::Object(_), Value::Str(k)) => sites.field(k),
            _ => &Value::Null,
        };
        if site_key.is_null() {
            continue;
        }
        let sc = scalp(site_key)?;
        let regions = if in_object(over, key)? {
            iterate(over.field(&key.py_str()))?
                .iter()
                .map(|r| {
                    let (k, human, kind) = region(r).ok_or_else(|| Fail::key(&r.py_str()))?;
                    Ok(Value::obj([
                        ("region", Value::str(k)),
                        ("human", Value::str(human)),
                        ("kind", Value::str(kind)),
                        ("side", sc.field("side").clone()),
                        ("derived", Value::Bool(false)),
                    ]))
                })
                .collect::<Result<Vec<_>, Fail>>()?
        } else {
            regions_under(site_key)?
        };
        out.push(Value::obj([
            ("slot", key.clone()),
            ("role", item(s, "role")?.clone()),
            ("name", item(s, "name")?.clone()),
            ("human", item(s, "human")?.clone()),
            ("scalp", sc),
            ("over", Value::Array(regions)),
        ]));
    }
    let empty = Value::str("");
    let yes = Value::Bool(true);
    Ok(Value::obj([
        ("layout", item(lay, "key")?.clone()),
        ("preset", get(p, "preset")?.clone()),
        ("rotation_deg", get(p, "rotation_deg")?.clone()),
        ("note", get_or(p, "note", &empty)?.clone()),
        ("bias_driven", get_or(p, "bias_driven", &yes)?.clone()),
        ("slots", Value::Array(out)),
    ]))
}

/// The resolved slots, by slot key.
fn by_slot(r: &Value) -> Vec<(Value, Value)> {
    match r.get("slots") {
        Some(Value::Array(a)) => a.iter().map(|s| (s.field("slot").clone(), s.clone())).collect(),
        _ => Vec::new(),
    }
}

fn first_with_role<'a>(slots: &[&'a Value], role: &str) -> Result<Option<&'a Value>, Fail> {
    for s in slots {
        if role_is(s, role)? {
            return Ok(Some(s));
        }
    }
    Ok(None)
}

/// `sorted(channels, key=lambda c: c["index"])`, stable.
fn by_index<'a>(chans: &[&'a Value]) -> Result<Vec<&'a Value>, Fail> {
    let mut keyed = Vec::new();
    for c in chans {
        let i = item(c, "index")?;
        let k = i
            .as_f64()
            .filter(|_| i.is_number())
            .ok_or_else(|| Fail::Other("TypeError: an index is not a number".into()))?;
        keyed.push((k, *c));
    }
    keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(keyed.into_iter().map(|(_, c)| c).collect())
}

/// The placement as a montage, so everything downstream works.
pub(crate) fn to_montage(lay: &Value, p: &Value) -> Result<Value, Fail> {
    let r = resolve(lay, p)?;
    let placed = by_slot(&r);
    let site = |key: &Value| -> Result<Value, Fail> {
        let s = slot_of(&placed, key).ok_or_else(|| Fail::key(&key.py_str()))?;
        let sc = item(s, "scalp")?;
        m::site(item(s, "name")?, item(sc, "ten_ten")?, item(sc, "human")?)
    };
    let slots = list(item(lay, "slots")?)?;
    let ref_slot = first_with_role(&slots, REFERENCE)?;
    let mut chans = Vec::new();
    for c in by_index(&list(item(lay, "channels")?)?)? {
        chans.push(m::channel(
            item(c, "index")?,
            item(c, "derivation")?,
            site(item(c, "positive")?)?,
            site(item(c, "negative")?)?,
            get(c, "label")?,
        )?);
    }
    let bias_slot = first_with_role(&slots, BIAS)?;
    let bias = match bias_slot {
        Some(b) if slot_of(&placed, item(b, "key")?).is_some() && r.field("bias_driven").truthy() => {
            site(item(b, "key")?)?
        }
        _ => Value::Null,
    };
    let electrode = match get(lay, "electrode")? {
        e if e.truthy() => e.clone(),
        _ => Value::str("unknown"),
    };
    let reference = match ref_slot {
        Some(s) => site(item(s, "key")?)?,
        None => Value::Null,
    };
    m::build(chans, electrode, bias, reference, Value::str(describe(lay, p, false)?), Value::Null)
}

/// The placement as a sentence, regenerated from the structure.
pub(crate) fn describe(lay: &Value, p: &Value, extras: bool) -> Result<String, Fail> {
    let r = resolve(lay, p)?;
    let placed = by_slot(&r);
    let resolved = list(r.field("slots"))?;
    let rec: Vec<&Value> =
        resolved.iter().copied().filter(|s| matches!(s.field("role"), Value::Str(x) if x == RECORDING)).collect();
    let reference = resolved.iter().copied().find(|s| matches!(s.field("role"), Value::Str(x) if x == REFERENCE));
    let mut spots: Vec<String> = rec.iter().map(|s| s.field("scalp").field("key").py_str()).collect();
    spots.sort();
    spots.dedup();
    let last_part = |s: &Value| -> String { s.field("human").py_str().rsplit(", ").next().unwrap_or("").to_string() };
    let mut parts = Vec::new();
    let chans = list(item(lay, "channels")?)?;
    if rec.len() > 1 && spots.len() == 1 {
        parts.push(format!(
            "{} channels in one cluster at {}: {}",
            chans.len(),
            rec[0].field("scalp").field("human").py_str(),
            rec.iter().map(|s| last_part(s)).collect::<Vec<_>>().join(", ")
        ));
        if let Some(r) = reference {
            parts.push(format!("referenced to the {}", last_part(r)));
        }
    } else {
        for c in by_index(&chans)? {
            let human = |pole: &str| -> Result<String, Fail> {
                let s = slot_of(&placed, item(c, pole)?)
                    .ok_or_else(|| Fail::key(&item(c, pole).map(|v| v.py_str()).unwrap_or_default()))?;
                Ok(s.field("scalp").field("human").py_str())
            };
            let (a, b) = (human("positive")?, human("negative")?);
            let at = item(c, "index")?.py_str();
            parts.push(if matches!(item(c, "derivation")?, Value::Str(d) if d == m::REFERENTIAL) {
                format!("ch{at} = {a}, referenced to {b}")
            } else {
                format!("ch{at} = {a} / {b}")
            });
        }
    }
    let slots = list(item(lay, "slots")?)?;
    if let Some(b) = first_with_role(&slots, BIAS)? {
        let key = item(b, "key")?;
        parts.push(format!(
            "bias = {}",
            match slot_of(&placed, key) {
                Some(s) if r.field("bias_driven").truthy() => s.field("scalp").field("human").py_str(),
                _ => "not driven".into(),
            }
        ));
    }
    let mut tail = String::new();
    if extras {
        let rotation = r.field("rotation_deg");
        if rotation.truthy() {
            tail.push_str(&format!(" Rotated {}°.", pyfmt::general(rotation.as_f64().unwrap_or(f64::NAN), 6)));
        }
        let note = r.field("note");
        if note.truthy() {
            tail.push_str(&format!(" {}", note.py_str()));
        }
    }
    let Some((first, rest)) = parts.split_first() else {
        return Err(Fail::Other("IndexError: list index out of range".into()));
    };
    let mut said = vec![first.clone()];
    for p in rest {
        let head: String = p.chars().take(14).collect();
        said.push(if head.contains(" = ") {
            p.clone()
        } else {
            let mut cs = p.chars();
            match cs.next() {
                Some(c) => c.to_uppercase().collect::<String>() + cs.as_str(),
                None => String::new(),
            }
        });
    }
    Ok(said.join(". ") + "." + &tail)
}
