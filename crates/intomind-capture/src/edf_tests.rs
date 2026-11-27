//! Every EDF+ and BDF+ export held to the rules EDFlib opens a file by.
//!
//! EDFlib, the library EDFbrowser and pyedflib read files with, refuses to
//! open an EDF+ or BDF+ file that breaks any rule of `edflib_refusal`: its
//! header check, edflib_check_edf_file in edflib.c, rule for rule and in its
//! order, then its check of the annotation signal in every data record,
//! edflib_get_annotations, as the Python library's suite writes them again,
//! so the suite here holds every export to them with nothing installed.

use std::path::{Path, PathBuf};

use serde_json::{json, Value as Json};

use crate::edf::{self, Input, Width};
use crate::export;
use crate::npy::{self, Array, Data};
use crate::recording::Collection;
use crate::zip;

const MONTHS: [&str; 12] = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];

fn printable(b: &[u8]) -> bool {
    b.iter().all(|&c| (32..=126).contains(&c))
}

fn digit(c: u8) -> bool {
    c.is_ascii_digit()
}

/// edflib_is_integer_number: a sign, then digits, then only spaces.
fn integer(text: &[u8]) -> bool {
    if text.is_empty() {
        return false;
    }
    let start = usize::from(matches!(text[0], b'+' | b'-'));
    let (mut digits, mut space) = (0, false);
    for &c in &text[start..] {
        if c == b' ' {
            if digits == 0 {
                return false;
            }
            space = true;
        } else if digit(c) && !space {
            digits += 1;
        } else {
            return false;
        }
    }
    digits > 0
}

/// edflib_is_number: a sign, digits with one dot, an exponent, then only
/// spaces.
fn number(text: &[u8]) -> bool {
    if text.is_empty() {
        return false;
    }
    let mut i = usize::from(matches!(text[0], b'+' | b'-'));
    let (mut digits, mut space, mut dot) = (0, false, false);
    while i < text.len() {
        let c = text[i];
        if c == b'e' || c == b'E' {
            if digits == 0 {
                return false;
            }
            i += 1;
            if i == text.len() {
                return false;
            }
            if matches!(text[i], b'+' | b'-') {
                i += 1;
            }
            digits = 0;
            for &d in &text[i..] {
                if d == b' ' {
                    if digits == 0 {
                        return false;
                    }
                    space = true;
                } else if digit(d) && !space {
                    digits += 1;
                } else {
                    return false;
                }
            }
            return digits > 0;
        }
        if c == b' ' {
            if digits == 0 {
                return false;
            }
            space = true;
        } else if (digit(c) || c == b'.') && !space {
            if c == b'.' {
                if dot {
                    return false;
                }
                dot = true;
            } else {
                digits += 1;
            }
        } else {
            return false;
        }
        i += 1;
    }
    digits > 0
}

/// edflib_atoi_nonlocalized: spaces, a sign, then the digits up to the
/// first other character.
fn atoi(text: &[u8]) -> i64 {
    let mut s = text;
    while s.first() == Some(&b' ') {
        s = &s[1..];
    }
    let sign = if s.first() == Some(&b'-') { -1 } else { 1 };
    if matches!(s.first(), Some(b'+' | b'-')) {
        s = &s[1..];
    }
    let mut v: i64 = 0;
    for &c in s {
        if !digit(c) {
            break;
        }
        v = v * 10 + i64::from(c - b'0');
    }
    sign * v
}

/// A time as EDFlib reads it, edflib_get_long_time: in units of 100 ns.
fn ticks(text: &[u8]) -> i128 {
    let neg = text.first() == Some(&b'-');
    let text = if matches!(text.first(), Some(b'+' | b'-')) { &text[1..] } else { text };
    let (whole, frac) = match text.iter().position(|&c| c == b'.') {
        Some(at) => (&text[..at], &text[at + 1..]),
        None => (text, &[][..]),
    };
    let mut value: i128 = 0;
    for (k, &c) in whole.iter().rev().enumerate() {
        value += (i128::from(c) - 48) * 10i128.pow(7 + k as u32);
    }
    for (k, &c) in frac.iter().take(7).enumerate() {
        value += (i128::from(c) - 48) * 10i128.pow(6 - k as u32);
    }
    if neg {
        -value
    } else {
        value
    }
}

/// edflib_is_onset_number: a sign, then digits with one dot, the dot
/// neither first nor last.
fn onset_number(text: &[u8]) -> bool {
    if text.len() < 2 || !matches!(text[0], b'+' | b'-') || text[1] == b'.' || text[text.len() - 1] == b'.' {
        return false;
    }
    let body = &text[1..];
    body.iter().filter(|&&c| c == b'.').count() <= 1 && body.iter().all(|&c| digit(c) || c == b'.')
}

/// edflib_is_duration_number: digits with one dot, the dot neither first
/// nor last.
fn duration_number(text: &[u8]) -> bool {
    if text.is_empty() || text[0] == b'.' || text[text.len() - 1] == b'.' {
        return false;
    }
    text.iter().filter(|&&c| c == b'.').count() <= 1 && text.iter().all(|&c| digit(c) || c == b'.')
}

fn float(text: &[u8]) -> f64 {
    std::str::from_utf8(text).unwrap().trim().parse().unwrap()
}

fn two(b: &[u8]) -> i64 {
    atoi(b)
}

/// The first of EDFlib's rules the file `b` breaks, named, or None for a
/// file EDFlib opens. Its header check first, then, unless `header_only`,
/// its check of the annotation signal in every data record, which it runs
/// as it opens an EDF+ or BDF+ file.
pub(crate) fn edflib_refusal(b: &[u8], header_only: bool) -> Option<String> {
    let no = |s: &str| Some(s.to_string());
    if b.len() < 256 {
        return no("a header of fewer than 256 bytes");
    }
    let head = &b[..256];
    // The version.
    let bdf = head[0] == 0xff;
    if bdf {
        if !printable(&head[1..8]) || &head[1..8] != b"BIOSEMI" {
            return no("the version field");
        }
    } else if !printable(&head[..8]) || &head[..8] != b"0       " {
        return no("the version field");
    }
    // The patient and recording fields, each printable.
    if !printable(&head[8..88]) {
        return no("the patient field holds a character that is not printable");
    }
    if !printable(&head[88..168]) {
        return no("the recording field holds a character that is not printable");
    }
    // The start date, dd.mm.yy.
    let d = &head[168..176];
    if !printable(d) || d[2] != b'.' || d[5] != b'.' || ![0, 1, 3, 4, 6, 7].iter().all(|&k| digit(d[k])) {
        return no("the start date field is not dd.mm.yy");
    }
    let (day, month, yy) = (two(&d[0..2]), two(&d[3..5]), two(&d[6..8]));
    if !(1..=31).contains(&day) || !(1..=12).contains(&month) {
        return no("the start date field holds no day or month");
    }
    let year = yy + if yy > 84 { 1900 } else { 2000 };
    // The start time, hh.mm.ss.
    let t = &head[176..184];
    if !printable(t) || t[2] != b'.' || t[5] != b'.' || ![0, 1, 3, 4, 6, 7].iter().all(|&k| digit(t[k])) {
        return no("the start time field is not hh.mm.ss");
    }
    if two(&t[0..2]) > 23 || two(&t[3..5]) > 59 || two(&t[6..8]) > 59 {
        return no("the start time field holds no time of day");
    }
    // The number of signals, and of header bytes.
    if !printable(&head[252..256]) || !integer(&head[252..256]) {
        return no("the number of signals is not a whole number");
    }
    let ns = atoi(&head[252..256]);
    if !(1..=4096).contains(&ns) {
        return no("the number of signals is not 1 to 4096");
    }
    let ns = ns as usize;
    if !printable(&head[184..192]) || !integer(&head[184..192]) || atoi(&head[184..192]) != 256 * (ns as i64 + 1) {
        return no("the number of header bytes is not 256 for each signal and 256 more");
    }
    // The reserved field.
    let reserved = &head[192..236];
    if !printable(reserved) {
        return no("the reserved field holds a character that is not printable");
    }
    let family = if bdf { "BDF" } else { "EDF" };
    let padded = |s: String| format!("{s:<44}").into_bytes();
    let plus = reserved == padded(format!("{family}+C")) || reserved == padded(format!("{family}+D"));
    if !bdf && !plus && reserved.iter().any(|&c| c != b' ') {
        return no("the reserved field is neither EDF+C, EDF+D nor empty");
    }
    let discontinuous = plus && reserved[4] == b'D';
    // The number of data records and their duration.
    if !printable(&head[236..244]) || !integer(&head[236..244]) || atoi(&head[236..244]) < 1 {
        return no("the number of data records is not a whole number of at least 1");
    }
    let n_records = atoi(&head[236..244]) as usize;
    if !printable(&head[244..252]) || !number(&head[244..252]) || float(&head[244..252]) < -0.000001 {
        return no("the data record duration is not a number of at least 0");
    }
    let duration = ticks(head[244..252].split(|&c| c == b' ').next().unwrap());
    if b.len() < 256 * (ns + 1) {
        return no("a header cut short");
    }
    let sig = &b[256..256 * (ns + 1)];
    let col = |o: usize, w: usize| -> Vec<&[u8]> { (0..ns).map(|i| &sig[o * ns + i * w..o * ns + (i + 1) * w]).collect() };
    let blank = |x: &[u8]| x.iter().all(|&c| c == b' ');
    // The labels, and the annotation signal.
    let labels = col(0, 16);
    if !labels.iter().all(|x| printable(x)) {
        return no("a label holds a character that is not printable");
    }
    let annot: Vec<bool> = labels.iter().map(|x| plus && *x == format!("{family} Annotations ").as_bytes()).collect();
    if plus && !annot.iter().any(|&a| a) {
        return Some(format!("an {family}+ file with no {family} Annotations signal"));
    }
    if (!annot.iter().all(|&a| a) || !plus) && duration < 1 {
        return no("a data record duration of 0 in a file of signals");
    }
    // The transducers, dimensions, physical and digital ranges, prefilters,
    // samples per record and reserved fields of every signal.
    for (i, x) in col(16, 80).iter().enumerate() {
        if !printable(x) || (annot[i] && !blank(x)) {
            return Some(format!("signal {}'s transducer field", i + 1));
        }
    }
    if !col(96, 8).iter().all(|x| printable(x)) {
        return no("a physical dimension holds a character that is not printable");
    }
    let pmin = col(104, 8);
    for (i, x) in pmin.iter().enumerate() {
        if !printable(x) || !number(x) {
            return Some(format!("signal {}'s physical minimum is not a number", i + 1));
        }
    }
    for (i, x) in col(112, 8).iter().enumerate() {
        if !printable(x) || !number(x) {
            return Some(format!("signal {}'s physical maximum is not a number", i + 1));
        }
        if float(x) == float(pmin[i]) {
            return Some(format!("signal {}'s physical maximum is its minimum", i + 1));
        }
    }
    let (low, high) = if bdf { (-0x80_0000, 0x7f_ffff) } else { (-0x8000, 0x7fff) };
    let dmin = col(120, 8);
    for (i, x) in dmin.iter().enumerate() {
        if !printable(x) || !integer(x) {
            return Some(format!("signal {}'s digital minimum is not a whole number", i + 1));
        }
        let v = atoi(x);
        if (annot[i] && v != low) || !(low..=high).contains(&v) {
            return Some(format!("signal {}'s digital minimum", i + 1));
        }
    }
    for (i, x) in col(128, 8).iter().enumerate() {
        if !printable(x) || !integer(x) {
            return Some(format!("signal {}'s digital maximum is not a whole number", i + 1));
        }
        let v = atoi(x);
        if (annot[i] && v != high) || !(low..=high).contains(&v) {
            return Some(format!("signal {}'s digital maximum", i + 1));
        }
        if v < atoi(dmin[i]) + 1 {
            return Some(format!("signal {}'s digital maximum is not above its minimum", i + 1));
        }
    }
    for (i, x) in col(136, 80).iter().enumerate() {
        if !printable(x) || (annot[i] && !blank(x)) {
            return Some(format!("signal {}'s prefilter field", i + 1));
        }
    }
    let mut nsr = Vec::new();
    for (i, x) in col(216, 8).iter().enumerate() {
        if !printable(x) || !integer(x) || atoi(x) < 1 {
            return Some(format!("signal {}'s samples per data record", i + 1));
        }
        nsr.push(atoi(x) as usize);
    }
    let sample_bytes = if bdf { 3 } else { 2 };
    let record_bytes = nsr.iter().sum::<usize>() * sample_bytes;
    if record_bytes > if bdf { 15 } else { 10 } * 1024 * 1024 {
        return no("a data record of more bytes than EDFlib reads");
    }
    for (i, x) in col(224, 32).iter().enumerate() {
        if (!bdf || plus) && !blank(x) {
            return Some(format!("signal {}'s reserved field is not empty", i + 1));
        }
        if bdf && !plus && !printable(x) {
            return Some(format!("signal {}'s reserved field holds a character that is not printable", i + 1));
        }
    }
    if plus {
        // The patient field: code, sex, birthdate and name, each X where
        // not known.
        let mut p = head[8..88].to_vec();
        p.extend([0u8; 48]);
        let mut dot = match p[..80].iter().position(|&c| c == b' ') {
            Some(at) => at + 1,
            None => 1,
        };
        if !(2..=73).contains(&dot) {
            return no("the patient field");
        }
        if p[dot + 2] != b'X' && dot > 65 {
            return no("the patient field");
        }
        if !b"MFX".contains(&p[dot]) {
            return no("the patient field's sex");
        }
        dot += 1;
        if p[dot] != b' ' {
            return no("the patient field");
        }
        if p[dot + 1] == b'X' {
            if p[dot + 2] != b' ' || p[dot + 3] == b' ' {
                return no("the patient field");
            }
        } else {
            if p[dot + 12] != b' ' || p[dot + 13] == b' ' {
                return no("the patient field");
            }
            let born = &p[dot + 1..dot + 12];
            if born[2] != b'-'
                || born[6] != b'-'
                || ![0, 1, 7, 8, 9, 10].iter().all(|&k| digit(born[k]))
                || !(1..=31).contains(&two(&born[..2]))
                || !MONTHS.iter().any(|m| m.as_bytes() == &born[3..6])
            {
                return no("the patient field's birthdate");
            }
        }
        // The recording field: Startdate, the start date as DD-MMM-YYYY or
        // X, then the administration, technician and equipment codes, each
        // one or more characters, separated by single spaces.
        let mut r = head[88..168].to_vec();
        r.extend([0u8; 48]);
        if &r[..10] != b"Startdate " {
            return no("the recording field does not begin with Startdate");
        }
        let k;
        let mut date: &[u8] = &[];
        if r[10] == b'X' {
            if r[11] != b' ' || r[12] == b' ' {
                return no("the recording field's start date");
            }
            k = 12;
        } else {
            if r[21] != b' ' || r[22] == b' ' {
                return no("the recording field's start date");
            }
            k = 22;
            date = &r[10..21];
            if date[2] != b'-'
                || date[6] != b'-'
                || ![0, 1, 7, 8, 9, 10].iter().all(|&j| digit(date[j]))
                || !(1..=31).contains(&two(&date[..2]))
                || !MONTHS.iter().any(|m| m.as_bytes() == &date[3..6])
            {
                return no("the recording field's start date is not DD-MMM-YYYY");
            }
        }
        let mut spaces = 0;
        for i in k..80 {
            if i > 78 {
                return no("the recording field holds fewer than five subfields");
            }
            if r[i] == b' ' {
                spaces += 1;
                if r[i + 1] == b' ' {
                    return no("the recording field holds two spaces where one separates its subfields");
                }
            }
            if spaces > 1 {
                break;
            }
        }
        if r[10] != b'X' {
            let m = MONTHS.iter().position(|m| m.as_bytes() == &date[3..6]).unwrap() as i64 + 1;
            if (day, month, yy) != (two(&date[..2]), m, two(&date[9..11])) {
                return no("the recording field's start date is not the header's");
            }
            if year != atoi(&date[7..11]) {
                return no("the recording field's year is not the header's");
            }
        }
    }
    // The file's size.
    if b.len() != 256 * (ns + 1) + record_bytes * n_records {
        return no("the file's size is not its header and its data records");
    }
    if header_only || !plus {
        return None;
    }
    // Every data record's annotation signal: zeros at its end, its first
    // list the record's onset, exactly one duration after the last record's
    // in a continuous file, then lists of onset, duration and texts.
    let mut elapsed: i128 = 0;
    let annots: Vec<usize> = (0..ns).filter(|&i| annot[i]).collect();
    for rec in 0..n_records {
        let base = 256 * (ns + 1) + rec * record_bytes;
        for (which, &a) in annots.iter().enumerate() {
            let start = base + nsr[..a].iter().sum::<usize>() * sample_bytes;
            let blob = &b[start..start + nsr[a] * sample_bytes];
            if blob[blob.len() - 1] != 0 {
                return Some(format!("data record {}'s annotations do not end in a zero byte", rec + 1));
            }
            if which == 0 {
                let cut = blob[..blob.len() - 2].iter().position(|&c| c == 0x14);
                let Some(cut) = cut.filter(|&c| blob[c + 1] == 0x14) else {
                    return Some(format!("data record {} begins with no list of its onset", rec + 1));
                };
                let onset = &blob[..cut];
                if !onset_number(onset) {
                    return Some(format!("data record {}'s onset is not a number", rec + 1));
                }
                let t0 = ticks(onset);
                if rec == 0 && !(0..10_000_000).contains(&t0) {
                    return no("the first data record's onset is not 0 to 1 s");
                }
                if rec > 0 && !discontinuous && t0 - elapsed != duration {
                    return Some(format!(
                        "data record {}'s onset is not the data record duration after data record {rec}'s",
                        rec + 1
                    ));
                }
                if rec > 0 && discontinuous && t0 - elapsed < duration {
                    return Some(format!("data record {} begins before data record {rec} ends", rec + 1));
                }
                elapsed = t0;
            }
            let mut text: Vec<u8> = Vec::new();
            let (mut zero, mut onset_read, mut dur_read, mut dur_open, mut in_tal, mut in_record) =
                (0, false, false, false, 0, 0);
            for (k, &c) in blob.iter().enumerate() {
                if c == 0 {
                    if zero == 0 {
                        if k > 0 && blob[k - 1] != 0x14 {
                            return Some(format!("data record {}: a list of annotations not closed by 0x14", rec + 1));
                        }
                        text.clear();
                        (onset_read, dur_read, dur_open, in_tal) = (false, false, false, 0);
                    }
                    zero += 1;
                    continue;
                }
                if zero > 1 {
                    return Some(format!("data record {}: a list of annotations after its padding", rec + 1));
                }
                zero = 0;
                if c == 0x14 || c == 0x15 {
                    if c == 0x15 {
                        if dur_read || dur_open || onset_read || in_tal > 0 {
                            return Some(format!("data record {}: a duration not right after an onset", rec + 1));
                        }
                        dur_open = true;
                    }
                    if c == 0x14 && onset_read && !dur_open {
                        if (which > 0 || in_record > 0)
                            && text.iter().take(512).any(|&x| (x < 32 || x == 127) && ![9, 10, 13].contains(&x))
                        {
                            return Some(format!(
                                "data record {}: an annotation holds a character that is not printable",
                                rec + 1
                            ));
                        }
                        in_tal += 1;
                        in_record += 1;
                        text.clear();
                        continue;
                    }
                    if !onset_read {
                        if !onset_number(&text) {
                            return Some(format!("data record {}: an onset that is not a number", rec + 1));
                        }
                        onset_read = true;
                        text.clear();
                        continue;
                    }
                    if dur_open {
                        if !duration_number(&text) {
                            return Some(format!("data record {}: a duration that is not a number", rec + 1));
                        }
                        dur_read = true;
                        dur_open = false;
                        text.clear();
                        continue;
                    }
                }
                text.push(c);
            }
        }
    }
    None
}

// ---------------------------------------------------------------- the files

/// A folder of its own for one test's recordings and exports.
fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("intomind-edf-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The files of a recording on two channels at `fs` a second, numbered by
/// `index`, its values a DC-coupled channel drifting under a small rhythm,
/// as the library's export tests make one, with the manifest `extra` gives
/// over a device named "fake instrument", a revision of "unknown" and the
/// electrode "fixture electrode", and `events` as its markers.
fn recording(dir: &Path, label: &str, extra: Json, index: &[i64], fs: f64, events: Option<Json>) {
    let n = index.len();
    let lsb = (2.0 * 2_400_000.0) / (12.0 * f64::from(1u32 << 24));
    let mut counts = Vec::with_capacity(2 * n);
    for c in 0..2 {
        for k in 0..n {
            let ramp = if n > 1 { 8000.0 * k as f64 / (n - 1) as f64 } else { 0.0 };
            let uv = 1500.0 * (c + 1) as f64 + ramp + 20.0 * (2.0 * std::f64::consts::PI * 10.0 * k as f64 / fs).sin();
            counts.push((uv / lsb).round() as i64);
        }
    }
    let members: Vec<(&str, Array)> = vec![
        ("counts", Array { shape: vec![2, n], fortran: false, data: Data::I64(counts) }),
        ("index", Array::vector(Data::I64(index.to_vec()))),
        ("device_ticks", Array::vector(Data::I64(index.to_vec()))),
        ("host_time", Array::vector(Data::F64(index.iter().map(|&k| k as f64 / fs).collect()))),
    ];
    let members: Vec<(String, Vec<u8>)> = members.iter().map(|(k, a)| (format!("{k}.npy"), npy::write(a))).collect();
    std::fs::write(dir.join(format!("{label}.npz")), zip::write(&members, 1_767_225_600)).unwrap();
    let mut meta = json!({
        "label": label, "n": n, "fs_effective": fs, "vref_uv": 2_400_000, "gain": 12, "adc_bits": 24,
        "device_name": "fake instrument", "board_revision": "unknown", "montage": {"electrode": "fixture electrode"},
    });
    for (k, v) in extra.as_object().unwrap() {
        if v.as_str() == Some(ABSENT) {
            meta.as_object_mut().unwrap().remove(k);
        } else {
            meta[k] = v.clone();
        }
    }
    std::fs::write(dir.join(format!("{label}.meta.json")), meta.to_string()).unwrap();
    if let Some(e) = events {
        std::fs::write(dir.join(format!("{label}.events.json")), e.to_string()).unwrap();
    }
}

/// A manifest value that says the key is left out.
const ABSENT: &str = "\u{0}absent";

fn samples(n: i64) -> Vec<i64> {
    (0..n).collect()
}

/// The bytes of `label` exported as `fmt`.
fn exported(dir: &Path, label: &str, fmt: &str) -> Vec<u8> {
    let path = dir.join(format!("{label}.out.{fmt}"));
    export::export(&Collection::open(dir), label, fmt, Some(&path)).unwrap_or_else(|e| panic!("{label} {fmt}: {e}"));
    std::fs::read(path).unwrap()
}

/// The fixed-width header, at the offsets the standard puts it, each field
/// without its padding.
struct Header {
    recording: String,
    startdate: String,
    starttime: String,
    n_records: usize,
    duration: String,
    labels: Vec<String>,
    transducer: Vec<String>,
    prefilter: Vec<String>,
}

fn header(b: &[u8]) -> Header {
    let f = |o: usize, n: usize| b[o..o + n].iter().map(|&c| c as char).collect::<String>().trim().to_string();
    let ns: usize = f(252, 4).parse().unwrap();
    let each = |o: usize, w: usize| (0..ns).map(|i| f(256 + o * ns + i * w, w)).collect::<Vec<_>>();
    Header {
        recording: f(88, 80),
        startdate: f(168, 8),
        starttime: f(176, 8),
        n_records: f(236, 8).parse().unwrap(),
        duration: f(244, 8),
        labels: each(0, 16),
        transducer: each(16, 80),
        prefilter: each(136, 80),
    }
}

fn known() -> Json {
    json!({"started_utc": "2026-11-25T10:20:30Z"})
}

fn with(a: &Json, b: &Json) -> Json {
    let mut out = a.clone();
    for (k, v) in b.as_object().unwrap() {
        out[k] = v.clone();
    }
    out
}

fn generated() -> Json {
    json!({"synthetic": true, "signal_source": "synthetic"})
}

/// A ramp on one channel named `name`, 1200 samples.
fn ramp(name: &str) -> Vec<(String, Vec<f64>)> {
    vec![(name.to_string(), (0..1200).map(|k| -50.0 + 100.0 * k as f64 / 1199.0).collect())]
}

/// What the writer is given for one file of `signals` at 500 a second, from
/// 2026-11-25 10:20:30, with no recording field.
fn base<'a>(signals: &'a [(String, Vec<f64>)], index: &'a [i64]) -> Input<'a> {
    Input {
        signals,
        fs: 500.0,
        start: (2026, 11, 25, 10, 20, 30),
        index,
        events: &[],
        patient: "X X X X",
        recording: "",
        prefilter: "",
        transducer: "",
        stretches: &[],
    }
}

fn written(input: &Input, w: &Width) -> Vec<u8> {
    edf::write(input, w).unwrap().0
}

fn holds(b: &[u8], text: &str) -> bool {
    b.windows(text.len()).any(|w| w == text.as_bytes())
}

// ---------------------------------------------------------------- the tests

#[test]
fn every_header_holds_the_rules_edflib_opens_a_file_by_and_so_does_the_writers_own_field() {
    // EDFlib refuses an EDF+ or BDF+ file whose recording field does not
    // begin with Startdate, the start date the header holds, and the
    // administration, technician and equipment codes. Every export's header
    // holds every rule of EDFlib's header check, for a start that is known,
    // one that is not, a generated signal and a device not named, and so
    // does the header the writer writes when it is given no recording field.
    let dir = scratch("rules");
    let cases = [
        ("known", known()),
        ("unknown", json!({})),
        ("generated", with(&known(), &generated())),
        ("generated_unknown", generated()),
        ("no_device", with(&known(), &json!({"device_name": null, "device_id": null}))),
    ];
    for (name, extra) in &cases {
        recording(&dir, name, extra.clone(), &samples(1500), 500.0, None);
        for fmt in ["edf", "bdf"] {
            assert_eq!(edflib_refusal(&exported(&dir, name, fmt), true), None, "{name} {fmt}");
        }
    }
    let none = header(&exported(&dir, "no_device", "edf")).recording;
    assert!(none.starts_with("Startdate 25-NOV-2026 X X X 16-bit lossy, unknown;"), "{none}");
    let (signals, index) = (ramp("ch1"), samples(1200));
    for w in [&edf::EDF, &edf::BDF] {
        let b = written(&base(&signals, &index), w);
        assert_eq!(header(&b).recording, "Startdate 25-NOV-2026 X X X", "{}", w.family);
        assert_eq!(edflib_refusal(&b, true), None, "{}", w.family);
    }
}

#[test]
fn a_start_date_names_its_month_in_english_capitals_whatever_language_the_host_is_set_to() {
    let (signals, index) = (ramp("ch1"), samples(1200));
    for (m, month) in MONTHS.iter().enumerate() {
        let b = written(&Input { start: (987, m as u32 + 1, 7, 1, 2, 3), ..base(&signals, &index) }, &edf::EDF);
        assert_eq!(header(&b).recording, format!("Startdate 07-{month}-0987 X X X"));
    }
}

#[test]
fn a_generated_signal_and_a_lossy_file_say_so_inside_the_80_characters_whatever_the_device() {
    // However long the device's name, it is cut to leave room for the words
    // about a generated signal and the width, so only the revision and the
    // raw record's name can be cut away.
    let dir = scratch("eighty");
    let name = "a device whose name runs on for far longer than any field could hold";
    for started in [Some("2026-11-25T10:20:30Z"), None] {
        for synthetic in [true, false] {
            let label = format!("eighty_{}_{synthetic}", started.is_some());
            let extra =
                json!({"started_utc": started, "synthetic": synthetic, "device_name": name, "board_revision": "fixture board"});
            recording(&dir, &label, extra, &samples(1500), 500.0, None);
            let b = exported(&dir, &label, "edf");
            let field = header(&b).recording;
            assert!(field.len() == 80 && field.contains("16-bit lossy"), "{field}");
            assert_eq!(field.contains("SYNTHETIC, not measured: "), synthetic, "{field}");
            let lead =
                if started.is_some() { "Startdate 25-NOV-2026 X X a_device_whose_n" } else { "Startdate X X X a_device_whose_name_runs_o" };
            assert!(field.starts_with(lead), "{field}");
            assert_eq!(edflib_refusal(&b, true), None, "{label}");
        }
    }
}

#[test]
fn a_start_that_is_not_known_is_startdate_x_x_x_and_the_device_as_the_equipment() {
    // A start that is not known, none given, null, or a time not written as
    // the library writes one, is the recording field's start date, and the
    // two subfields after it, each an X, then the device as the equipment,
    // its name with an underscore for each space, and a start date and time
    // of 01.01.00 at 00.00.00, which the X says is not the start. A start
    // that is known keeps its date and time, and the field gives it as EDF+
    // does.
    let dir = scratch("start");
    let unknown = [json!({}), json!({"started_utc": null}), json!({"started_utc": "not a time"}), json!({"started_utc": "2026-11-25T10:20:30"})];
    for (k, extra) in unknown.iter().enumerate() {
        recording(&dir, &format!("start_{k}"), extra.clone(), &samples(1500), 500.0, None);
    }
    recording(&dir, "start_known", known(), &samples(1500), 500.0, None);
    for (fmt, bits) in [("edf", "16-bit lossy"), ("bdf", "24-bit")] {
        for k in 0..unknown.len() {
            let h = header(&exported(&dir, &format!("start_{k}"), fmt));
            let want = format!("Startdate X X X fake_instrument {bits}, unknown; raw record is start_{k}.npz");
            assert_eq!(h.recording, want.chars().take(80).collect::<String>(), "{fmt} {k}");
            assert_eq!((h.startdate.as_str(), h.starttime.as_str()), ("01.01.00", "00.00.00"), "{fmt} {k}");
        }
        let h = header(&exported(&dir, "start_known", fmt));
        let want = format!("Startdate 25-NOV-2026 X X fake_instrument {bits}, unknown; raw record is start_known.npz");
        assert_eq!(h.recording, want.chars().take(80).collect::<String>(), "{fmt}");
        assert_eq!((h.startdate.as_str(), h.starttime.as_str()), ("25.11.26", "10.20.30"), "{fmt}");
    }
}

#[test]
fn an_export_whose_start_is_not_known_lays_out_the_library_words_as_one_whose_start_is_known() {
    let dir = scratch("equipment");
    for (g, extra) in [("measured", json!({})), ("generated", generated())] {
        recording(&dir, &format!("eq_{g}_known"), with(&extra, &known()), &samples(1500), 500.0, None);
        recording(&dir, &format!("eq_{g}_unknown"), with(&extra, &json!({"started_utc": null})), &samples(1500), 500.0, None);
        for fmt in ["edf", "bdf"] {
            let fields: Vec<String> = ["known", "unknown"]
                .iter()
                .map(|s| {
                    let b = exported(&dir, &format!("eq_{g}_{s}"), fmt);
                    let field = header(&b).recording;
                    let parts: Vec<&str> = field.split(' ').collect();
                    assert!(parts[0] == "Startdate" && parts[2..5] == ["X", "X", "fake_instrument"], "{s} {fmt} {field}");
                    assert_eq!(edflib_refusal(&b, true), None, "{s} {fmt}");
                    field
                })
                .collect();
            let (known, unknown) = (&fields[0], &fields[1]);
            assert!(known.starts_with("Startdate 25-NOV-2026 ") && unknown.starts_with("Startdate X "), "{known} | {unknown}");
            // The known start's date is ten characters longer than X, so its
            // field is cut ten characters sooner.
            let after = &known["Startdate 25-NOV-2026 ".len()..];
            assert!(unknown["Startdate X ".len()..].starts_with(after), "{known} | {unknown}");
            assert_eq!(after.contains("SYNTHETIC, not measured: "), g == "generated", "{after}");
        }
    }
}

#[test]
fn the_equipment_code_joins_each_run_of_white_space_as_pythons_split_finds_it() {
    // Python's split takes the separators of files, groups, records and
    // units, and next line, as white space, and not a byte order mark or a
    // zero width space, which are written as question marks, as a character
    // ASCII cannot hold is.
    let dir = scratch("spaces");
    let extra = with(&known(), &json!({"device_name": " a\u{1c}b\u{85}c\u{feff}d\u{3000}\u{3000}e f\u{200b}g\t"}));
    recording(&dir, "spaces", extra, &samples(1500), 500.0, None);
    for fmt in ["edf", "bdf"] {
        let b = exported(&dir, "spaces", fmt);
        assert_eq!(header(&b).recording.split(' ').nth(4), Some("a_b_c?d_e_f?g"), "{fmt}");
        assert_eq!(edflib_refusal(&b, true), None, "{fmt}");
    }
}

#[test]
fn a_board_revision_of_null_or_of_no_text_is_written_as_for_a_recording_that_states_none() {
    let dir = scratch("revision");
    let revisions = [("absent", json!(ABSENT)), ("null", json!(null)), ("empty", json!("")), ("spaces", json!("  ")), ("python_spaces", json!("\t\u{1c}\u{3000}"))];
    for started in [Some("2026-11-25T10:20:30Z"), None] {
        for fmt in ["edf", "bdf"] {
            let fields: Vec<String> = revisions
                .iter()
                .map(|(name, rev)| {
                    let label = format!("rev_{name}_{}", started.is_some());
                    recording(&dir, &label, json!({"started_utc": started, "board_revision": rev}), &samples(1500), 500.0, None);
                    header(&exported(&dir, &label, fmt)).recording.split("raw record is ").next().unwrap().to_string()
                })
                .collect();
            assert!(fields[0].ends_with(", unknown; "), "{}", fields[0]);
            assert!(fields.iter().all(|f| f == &fields[0]), "{fields:?}");
        }
    }
}

/// Control characters: NUL, the bell, tab, line feed, carriage return, the
/// two separators of an annotation list, escape, delete, and next line.
const CONTROLS: &str = "\u{0}\u{7}\t\n\r\u{14}\u{15}\u{1b}\u{7f}\u{85}";

#[test]
fn a_control_character_in_any_text_an_export_writes_is_a_question_mark_and_the_file_opens() {
    // EDFlib refuses a file whose header holds a character that is not
    // printable or whose annotation holds a control character, and a
    // control character passed into both from the text a recording carries:
    // its device's name or id, its board revision, its electrode, its
    // processing description or signal word, and an event's text. Each is
    // written as a question mark, and every file passes the whole of
    // EDFlib's check. In the device's name, as in the equipment code it
    // becomes, each run of white space is one underscore, and a control
    // character that is white space, as a tab or a line feed is, is part of
    // a run.
    let dir = scratch("controls");
    let q = "?".repeat(CONTROLS.chars().count());
    let events = json!([{"host_time": 0.5, "cond": format!("cue{CONTROLS}")}, {"host_time": 1.0, "kind": format!("marker{CONTROLS}")}]);
    let cases = [
        (
            "named",
            json!({"device_name": format!("dev{CONTROLS}ice"), "board_revision": format!("rev{CONTROLS}"),
                   "montage": {"electrode": format!("electrode{CONTROLS}")}, "processing": {"description": format!("chain{CONTROLS}")}}),
            format!("chain{q}"),
        ),
        ("by_id", json!({"device_name": null, "device_id": format!("dev{CONTROLS}ice"), "signal": format!("notch{CONTROLS}")}), format!("notch{q}")),
    ];
    for (name, extra, prefilter) in &cases {
        for started in [Some("2026-11-25T10:20:30Z"), None] {
            let label = format!("controls_{name}_{}", started.is_some());
            recording(&dir, &label, with(extra, &json!({"started_utc": started})), &samples(1500), 500.0, Some(events.clone()));
            for fmt in ["edf", "bdf"] {
                let b = exported(&dir, &label, fmt);
                assert_eq!(edflib_refusal(&b, false), None, "{label} {fmt}");
                let h = header(&b);
                assert_eq!(h.recording.split(' ').nth(4), Some("dev??_????_ice"), "{label} {fmt} {}", h.recording);
                if *name == "named" {
                    assert!(h.recording.contains(&format!(", rev{q}; raw record")), "{}", h.recording);
                    assert_eq!(h.transducer[..2], [format!("electrode{q}"), format!("electrode{q}")], "{label} {fmt}");
                }
                assert_eq!(h.prefilter[..2], [prefilter.clone(), prefilter.clone()], "{label} {fmt}");
                assert!(holds(&b, &format!("\u{14}cue{q}\u{14}")) && holds(&b, &format!("\u{14}marker{q}\u{14}")), "{label} {fmt}");
            }
        }
    }
    // The writer itself, given a control character in each text it writes.
    let (signals, index) = (ramp(&format!("ch{CONTROLS}")), samples(1200));
    let events = [edf::Event { t: 0.5, label: format!("cue{CONTROLS}") }];
    let stretches = [(format!("BAD{CONTROLS}"), (0..1200).map(|k| k < 10).collect::<Vec<bool>>())];
    let recording_field = format!("Startdate 25-NOV-2026 X X dev{CONTROLS}ice");
    let (prefilter, transducer) = (format!("chain{CONTROLS}"), format!("electrode{CONTROLS}"));
    for w in [&edf::EDF, &edf::BDF] {
        let input = Input {
            recording: &recording_field,
            prefilter: &prefilter,
            transducer: &transducer,
            events: &events,
            stretches: &stretches,
            ..base(&signals, &index)
        };
        let b = written(&input, w);
        assert_eq!(edflib_refusal(&b, false), None, "{}", w.family);
        let h = header(&b);
        assert!(h.labels[0] == format!("ch{q}") && h.transducer[0] == format!("electrode{q}"), "{:?}", h.labels);
        assert_eq!(h.recording, format!("Startdate 25-NOV-2026 X X dev{q}ice"));
        assert!(holds(&b, &format!("\u{14}cue{q}\u{14}")) && holds(&b, &format!("\u{14}BAD{q}\u{14}")), "{}", w.family);
    }
}

#[test]
fn every_data_record_begins_the_record_duration_the_header_states_after_the_one_before_it() {
    // EDF+C requires each data record to begin where the one before it
    // ends, and EDFlib refuses a file whose record onsets are not exactly
    // the header's record duration apart. At a rate that is not a whole
    // number the header holds the duration rounded to its eight characters,
    // and the onsets followed the rate instead, so at 498.8996 samples a
    // second a record began a microsecond late by the header's count within
    // a few records. Every export of such a recording passes the whole of
    // EDFlib's check, for a known start, an unknown start and a generated
    // signal, with a hole in it too.
    let dir = scratch("onsets");
    let holed: Vec<i64> = (0..2500).chain(2600..5100).collect();
    for (name, extra) in [("known", known()), ("unknown", json!({})), ("generated", with(&known(), &generated()))] {
        let extra = with(&extra, &json!({"fs_effective": 498.8996}));
        recording(&dir, &format!("onsets_{name}"), extra.clone(), &samples(5000), 498.8996, None);
        recording(&dir, &format!("onsets_{name}_holed"), extra, &holed, 498.8996, None);
        for label in [format!("onsets_{name}"), format!("onsets_{name}_holed")] {
            for fmt in ["edf", "bdf"] {
                let b = exported(&dir, &label, fmt);
                let h = header(&b);
                assert!(h.duration == "1.000201" && h.n_records > 3, "{label} {fmt} {} {}", h.duration, h.n_records);
                assert_eq!(edflib_refusal(&b, false), None, "{label} {fmt}");
            }
        }
    }
}
