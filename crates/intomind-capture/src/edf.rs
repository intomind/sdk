//! EDF+ and BDF+, written byte for byte as the Python library writes them.
//!
//! Every sample is written at the time it was recorded, counted from its
//! index. A hole holds one placeholder for each lost sample, repeating the
//! last recorded value, under a `BAD_ACQ_SKIP` annotation over exactly the
//! hole. The file is whole records, and the last one is completed after the
//! last sample the same way. A stretch where an electrode was off is a
//! `BAD_LEADOFF` annotation, and the part of a recording where detection
//! was off is `LEADOFF_NOT_CHECKED`, written only when it ran for the rest.
//! The record duration carries the measured rate, and each signal's range
//! is rounded outward so no sample is clipped. docs.intomind.com/files.html
//! states these rules, and this module follows the Python library's
//! `edf.py` line by line.

use crate::pyfmt::{self, Toward};
use crate::recording::{error, Error};

/// The annotation over each hole where samples were lost.
pub(crate) const SKIP_LABEL: &str = "BAD_ACQ_SKIP";
/// The annotation over each stretch an electrode was off, followed by the
/// channel's label.
pub(crate) const LEADOFF_LABEL: &str = "BAD_LEADOFF";
/// The annotation over the part of a recording where detection was off,
/// written only when it ran for the rest.
pub(crate) const NOT_CHECKED_LABEL: &str = "LEADOFF_NOT_CHECKED";

/// What differs between the two widths of the one container.
pub(crate) struct Width {
    pub(crate) dig_min: i64,
    pub(crate) dig_max: i64,
    pub(crate) sample_bytes: usize,
    version: &'static [u8; 8],
    annot: &'static str,
    pub(crate) family: &'static str,
}

pub(crate) const EDF: Width = Width {
    dig_min: -32768,
    dig_max: 32767,
    sample_bytes: 2,
    version: b"0       ",
    annot: "EDF Annotations",
    family: "EDF+",
};
pub(crate) const BDF: Width = Width {
    dig_min: -8_388_608,
    dig_max: 8_388_607,
    sample_bytes: 3,
    version: b"\xffBIOSEMI",
    annot: "BDF Annotations",
    family: "BDF+",
};

/// ASCII, left-justified, exactly `width` bytes: the first `width`
/// characters, each outside ASCII written as `?`, padded with spaces.
pub(crate) fn field(s: &str, width: usize) -> Vec<u8> {
    let mut b: Vec<u8> = s.chars().take(width).map(|c| if c.is_ascii() { c as u8 } else { b'?' }).collect();
    b.resize(width, b' ');
    b
}

/// A number in eight characters, losing no more than it must: `.8g`, and
/// failing that the most fixed places that fit, and failing that cut.
pub(crate) fn num(x: f64) -> String {
    let mut s = pyfmt::general(x, 8);
    if s.len() > 8 {
        for prec in (1..=6).rev() {
            s = pyfmt::fixed(x, prec);
            if s.len() <= 8 {
                break;
            }
        }
    }
    if s.len() > 8 {
        s.truncate(8);
    }
    s
}

/// `float(text)` for what `num` writes.
fn parse(s: &str) -> Option<f64> {
    s.trim().parse::<f64>().ok()
}

/// A bound as the header field will hold it.
fn written(x: f64) -> f64 {
    parse(&num(x)).unwrap_or(x)
}

/// A bound as the field will hold it, moved away from the samples where
/// the field cannot hold it exactly. None where Python's decimal context
/// refuses the bound and the export fails.
fn bound(x: f64, below: bool) -> Option<f64> {
    if !x.is_finite() {
        return Some(written(x));
    }
    let toward = if below { Toward::Floor } else { Toward::Ceiling };
    for places in (0..=7).rev() {
        let s = pyfmt::quantize(x, places, toward)?;
        let v = parse(&s)?;
        if s.len() <= 8 && written(v) == v {
            return Some(v);
        }
    }
    Some(written(x))
}

/// The physical range of one signal: its own extent with a one percent
/// margin, and never degenerate.
fn scale(x: &[f64]) -> (f64, f64) {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut nan = false;
    for &v in x {
        if v.is_nan() {
            nan = true;
        }
        lo = lo.min(v);
        hi = hi.max(v);
    }
    if nan {
        lo = f64::NAN;
        hi = f64::NAN;
    }
    if !lo.is_finite() || !hi.is_finite() || hi - lo < 1e-9 {
        lo -= 1.0;
        hi += 1.0;
    }
    let pad = 0.01 * (hi - lo);
    (lo - pad, hi + pad)
}

/// numpy's `round(...).clip(...).astype(int32)` for one value.
fn digitize(x: f64, pmin: f64, pmax: f64, dmin: i64, dmax: i64) -> i32 {
    let g = (dmax - dmin) as f64 / (pmax - pmin);
    let d = ((x - pmin) * g + dmin as f64).round_ties_even();
    // numpy's clip keeps a NaN, and its conversion to int32 then gives the
    // smallest int32.
    if d.is_nan() {
        return i32::MIN;
    }
    d.max(dmin as f64).min(dmax as f64) as i32
}

/// One time-stamped annotation list.
fn tal(onset: f64, duration: Option<f64>, text: &str) -> Vec<u8> {
    let mut out = format!("{}{}", if onset >= 0.0 { "+" } else { "-" }, pyfmt::fixed(onset.abs(), 6)).into_bytes();
    if let Some(d) = duration {
        out.push(0x15);
        out.extend(pyfmt::fixed(d, 6).as_bytes());
    }
    out.push(0x14);
    out.extend(text.as_bytes());
    out.extend([0x14, 0x00]);
    out
}

/// One event: its onset from the start of the file, in seconds, and its
/// text.
pub(crate) struct Event {
    pub(crate) t: f64,
    pub(crate) label: String,
}

/// The start of a recording, as Python's `strptime(started, "%Y-%m-%dT%H:%M:%SZ")`
/// reads it, or 2000-01-01 where it cannot.
pub(crate) fn start_of(started: Option<&str>) -> (i64, u32, u32, u32, u32, u32) {
    const DEFAULT: (i64, u32, u32, u32, u32, u32) = (2000, 1, 1, 0, 0, 0);
    let Some(s) = started else { return DEFAULT };
    parse_start(s).unwrap_or(DEFAULT)
}

fn parse_start(s: &str) -> Option<(i64, u32, u32, u32, u32, u32)> {
    let b = s.as_bytes();
    let mut i = 0;
    // One field: the digits strptime's own pattern allows, tried as it tries them.
    let num = |i: &mut usize, max_len: usize, ok: &dyn Fn(&str) -> bool| -> Option<u32> {
        let start = *i;
        let mut end = start;
        while end < b.len() && end - start < max_len && b[end].is_ascii_digit() {
            end += 1;
        }
        for e in (start + 1..=end).rev() {
            let t = &s[start..e];
            if ok(t) {
                *i = e;
                return t.parse().ok();
            }
        }
        None
    };
    let lit = |i: &mut usize, c: u8| -> Option<()> {
        if *i < b.len() && b[*i].eq_ignore_ascii_case(&c) {
            *i += 1;
            Some(())
        } else {
            None
        }
    };
    let two = |t: &str, lo: u32, hi: u32, one_ok: bool| -> bool {
        let v: u32 = match t.parse() {
            Ok(v) => v,
            Err(_) => return false,
        };
        (t.len() == 2 || (one_ok && t.len() == 1)) && v >= lo && v <= hi
    };
    let y = num(&mut i, 4, &|t| t.len() == 4)? as i64;
    lit(&mut i, b'-')?;
    let m = num(&mut i, 2, &|t| two(t, 1, 12, t != "0"))?;
    lit(&mut i, b'-')?;
    // %d also allows a space before a single digit.
    let d = if i < b.len() && b[i] == b' ' {
        i += 1;
        num(&mut i, 1, &|t| two(t, 1, 9, true))?
    } else {
        num(&mut i, 2, &|t| two(t, 1, 31, true))?
    };
    lit(&mut i, b'T')?;
    let hh = num(&mut i, 2, &|t| two(t, 0, 23, true))?;
    lit(&mut i, b':')?;
    let mm = num(&mut i, 2, &|t| two(t, 0, 59, true))?;
    lit(&mut i, b':')?;
    let ss = num(&mut i, 2, &|t| two(t, 0, 61, true))?;
    lit(&mut i, b'Z')?;
    if i != b.len() || y < 1 || ss > 59 {
        return None;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let days = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][(m - 1) as usize];
    if d > days {
        return None;
    }
    Some((y, m, d, hh, mm, ss))
}

/// What a file holds and what it cost, as `write_edf` reports it.
pub(crate) struct Written {
    pub(crate) n_records: usize,
    pub(crate) samples_per_record: usize,
    pub(crate) record_duration_s: f64,
    pub(crate) continuity: String,
    pub(crate) fs: f64,
    pub(crate) samples_written: usize,
    pub(crate) samples_filled: usize,
    pub(crate) gaps: usize,
    pub(crate) tail_filled: usize,
    pub(crate) n_annotations: usize,
    /// Each stretch text and how many annotations it got, in the order given.
    pub(crate) stretch_counts: Vec<(String, usize)>,
    pub(crate) quantum_uv: f64,
    pub(crate) max_error_uv: f64,
}

/// What one file is made of.
pub(crate) struct Input<'a> {
    pub(crate) signals: &'a [(String, Vec<f64>)],
    pub(crate) fs: f64,
    pub(crate) start: (i64, u32, u32, u32, u32, u32),
    pub(crate) index: &'a [i64],
    pub(crate) events: &'a [Event],
    pub(crate) patient: &'a str,
    pub(crate) recording: &'a str,
    pub(crate) prefilter: &'a str,
    pub(crate) transducer: &'a str,
    pub(crate) stretches: &'a [(String, Vec<bool>)],
}

/// One EDF+ or BDF+ file, and what it holds.
pub(crate) fn write(input: &Input, w: &Width) -> Result<(Vec<u8>, Written), Error> {
    let (dmin, dmax, sample_bytes) = (w.dig_min, w.dig_max, w.sample_bytes);
    let signals = input.signals;
    if signals.is_empty() {
        return Err(error("no signals"));
    }
    let n = signals[0].1.len();
    for (lbl, x) in signals {
        if x.len() != n {
            return Err(error(format!("signal {lbl} has {} samples, expected {n}", x.len())));
        }
    }
    if n == 0 {
        return Err(error("no samples"));
    }
    let idx = input.index;
    if idx.len() != n {
        return Err(error(format!("index has {} entries for {n} samples", idx.len())));
    }
    if let Some(at) = idx.windows(2).position(|p| p[1] - p[0] < 1) {
        return Err(error(format!(
            "refusing to write {}: the sample index steps back after sample {at}, a break whose length is not known, \
             so the samples after it have no place on this file's timeline",
            w.family
        )));
    }
    // Where each sample belongs, counted in samples from the first.
    let pos: Vec<usize> = idx.iter().map(|&i| (i - idx[0]) as usize).collect();
    let span = pos[n - 1] + 1;
    let fs = input.fs;
    if !fs.is_finite() {
        return Err(error("the sample rate is not a number"));
    }
    let nsr = (fs.round_ties_even() as i64).max(1) as usize;
    let dur = nsr as f64 / fs;
    let n_rec = span.div_ceil(nsr);
    let used = n_rec * nsr;
    let tail = used - span;
    let holes: Vec<(usize, usize)> = (0..n - 1)
        .filter(|&i| idx[i + 1] - idx[i] > 1)
        .map(|i| (pos[i] + 1, (idx[i + 1] - idx[i] - 1) as usize))
        .collect();

    // Runs of flagged samples on the file's timeline.
    let mut runs: Vec<(usize, usize, String)> = Vec::new();
    let mut stretch_counts = Vec::new();
    for (text, flags) in input.stretches {
        if flags.len() != n {
            return Err(error(format!("stretch '{text}' has {} flags for {n} samples", flags.len())));
        }
        let mut on = vec![false; used + 1];
        for (i, &f) in flags.iter().enumerate() {
            if f {
                on[pos[i]] = true;
            }
        }
        let mut count = 0;
        let mut start: Option<usize> = None;
        for (p, &v) in on.iter().enumerate() {
            match (start, v) {
                (None, true) => start = Some(p),
                (Some(a), false) => {
                    runs.push((a, p - a, text.clone()));
                    count += 1;
                    start = None;
                }
                _ => {}
            }
        }
        stretch_counts.push((text.clone(), count));
    }

    // Annotations bucketed into the record whose span holds them: events,
    // then holes and the tail, then stretches.
    let mut ann: Vec<Vec<(f64, Option<f64>, String)>> = vec![Vec::new(); n_rec];
    let mut n_events = 0;
    for e in input.events {
        let q = e.t / dur;
        if !q.is_finite() {
            return Err(error("an event's time is not a number"));
        }
        let r = (q.trunc() as i64).max(0).min(n_rec as i64 - 1) as usize;
        ann[r].push((e.t, None, e.label.clone()));
        n_events += 1;
    }
    let mut filled_spans = holes.clone();
    if tail > 0 {
        filled_spans.push((span, tail));
    }
    for &(first, length) in &filled_spans {
        ann[first / nsr].push((first as f64 / fs, Some(length as f64 / fs), SKIP_LABEL.into()));
    }
    for (first, length, text) in &runs {
        ann[first / nsr].push((*first as f64 / fs, Some(*length as f64 / fs), text.clone()));
    }
    let annot_bytes: Vec<Vec<u8>> = (0..n_rec)
        .map(|r| {
            let mut blob = tal((r * nsr) as f64 / fs, None, "");
            for (t, d, label) in &ann[r] {
                blob.extend(tal(*t, *d, label));
            }
            blob
        })
        .collect();
    let widest = annot_bytes.iter().map(Vec::len).max().unwrap_or(0);
    let ann_nsr = (widest.div_ceil(sample_bytes) + 1).max(1);

    let ns = signals.len() + 1;
    let hdr_bytes = 256 * (ns + 1);
    let reserved = format!("{}+C", &w.family[..3]);

    // Each range fitted to the recorded samples, its bounds as the header
    // will hold them.
    let mut ranges = Vec::with_capacity(signals.len());
    for (_, x) in signals {
        let (lo, hi) = scale(x);
        let lo_w = bound(lo, true).ok_or_else(|| error("a signal's range is too wide for the header"))?;
        let hi_w = bound(hi, false).ok_or_else(|| error("a signal's range is too wide for the header"))?;
        ranges.push(if hi_w > lo_w { (lo_w, hi_w) } else { (lo, hi) });
    }

    let (y, mo, d, hh, mi, ss) = input.start;
    let mut out = Vec::new();
    out.extend(w.version);
    out.extend(field(input.patient, 80));
    out.extend(field(input.recording, 80));
    out.extend(field(&format!("{d:02}.{mo:02}.{:02}", y.rem_euclid(100)), 8));
    out.extend(field(&format!("{hh:02}.{mi:02}.{ss:02}"), 8));
    out.extend(field(&hdr_bytes.to_string(), 8));
    out.extend(field(&reserved, 44));
    out.extend(field(&n_rec.to_string(), 8));
    out.extend(field(&num(dur), 8));
    out.extend(field(&ns.to_string(), 4));
    debug_assert_eq!(out.len(), 256);
    let each = |out: &mut Vec<u8>, f: &dyn Fn(Option<usize>) -> Vec<u8>| {
        for i in 0..signals.len() {
            out.extend(f(Some(i)));
        }
        out.extend(f(None));
    };
    each(&mut out, &|s| field(s.map_or(w.annot, |i| signals[i].0.as_str()), 16));
    each(&mut out, &|s| field(if s.is_some() { input.transducer } else { "" }, 80));
    each(&mut out, &|s| field(if s.is_some() { "uV" } else { "" }, 8));
    each(&mut out, &|s| field(&s.map_or("-1".to_string(), |i| num(ranges[i].0)), 8));
    each(&mut out, &|s| field(&s.map_or("1".to_string(), |i| num(ranges[i].1)), 8));
    each(&mut out, &|_| field(&dmin.to_string(), 8));
    each(&mut out, &|_| field(&dmax.to_string(), 8));
    each(&mut out, &|s| field(if s.is_some() { input.prefilter } else { "" }, 80));
    each(&mut out, &|s| field(&s.map_or(ann_nsr, |_| nsr).to_string(), 8));
    each(&mut out, &|_| field("", 32));
    debug_assert_eq!(out.len(), hdr_bytes);

    // Every recorded sample where it belongs, every other place the last
    // recorded value before it.
    let mut last = vec![0usize; used];
    {
        let mut k = 0;
        for (p, slot) in last.iter_mut().enumerate() {
            while k + 1 < n && pos[k + 1] <= p {
                k += 1;
            }
            *slot = k;
        }
    }
    let mut real = Vec::with_capacity(signals.len());
    let mut quantum: f64 = 0.0;
    let mut max_err: f64 = 0.0;
    for (si, (_, x)) in signals.iter().enumerate() {
        let (pmin, pmax) = ranges[si];
        let dg: Vec<i32> = x.iter().map(|&v| digitize(v, pmin, pmax, dmin, dmax)).collect();
        let q = (pmax - pmin) / (dmax - dmin) as f64;
        quantum = if q > quantum { q } else { quantum };
        // What a reader gets back, by the standard's formula, against what
        // was handed in. numpy's max propagates a NaN, and Python's max
        // keeps the first of a NaN and a number.
        let mut worst = f64::NEG_INFINITY;
        let mut nan = false;
        for (i, &dv) in dg.iter().enumerate() {
            let back = ((dv as f64 - dmin as f64) * (pmax - pmin)) / (dmax - dmin) as f64 + pmin;
            let e = (back - x[i]).abs();
            if e.is_nan() {
                nan = true;
            } else if e > worst {
                worst = e;
            }
        }
        let worst = if nan { f64::NAN } else { worst };
        if worst > max_err {
            max_err = worst;
        }
        real.push(dg);
    }
    for (r, record_annotations) in annot_bytes.iter().enumerate() {
        let (a, b) = (r * nsr, (r + 1) * nsr);
        for dg in &real {
            for &k in &last[a..b] {
                let v = dg[k];
                if sample_bytes == 2 {
                    out.extend((v as i16).to_le_bytes());
                } else {
                    out.extend(&v.to_le_bytes()[..3]);
                }
            }
        }
        let mut blob = record_annotations.clone();
        blob.resize(ann_nsr * sample_bytes, 0);
        out.extend(blob);
    }
    Ok((
        out,
        Written {
            n_records: n_rec,
            samples_per_record: nsr,
            record_duration_s: dur,
            continuity: reserved,
            fs,
            samples_written: n,
            samples_filled: holes.iter().map(|h| h.1).sum(),
            gaps: holes.len(),
            tail_filled: tail,
            n_annotations: n_events,
            stretch_counts,
            quantum_uv: quantum,
            max_error_uv: max_err,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_numbers_are_pythons() {
        assert_eq!(num(-4231.12345678), "-4231.12");
        assert_eq!(num(498.80345678), "498.8035");
        assert_eq!(num(1e-9), "1e-09");
        assert_eq!(num(-0.0), "-0");
        assert_eq!(bound(-4231.123456789, true), Some(-4231.13));
        assert_eq!(bound(4231.123456789, false), Some(4231.124));
    }

    #[test]
    fn a_start_is_read_as_strptime_reads_it() {
        assert_eq!(start_of(Some("2026-10-26T03:43:37Z")), (2026, 10, 26, 3, 43, 37));
        assert_eq!(start_of(Some("2026-1-5T3:4:5z")), (2026, 1, 5, 3, 4, 5));
        assert_eq!(start_of(Some("2026-02-30T00:00:00Z")), (2000, 1, 1, 0, 0, 0));
        assert_eq!(start_of(Some("2026-10-26T03:43:60Z")), (2000, 1, 1, 0, 0, 0));
        assert_eq!(start_of(None), (2000, 1, 1, 0, 0, 0));
    }

    #[test]
    fn a_hole_repeats_the_last_value_and_is_labeled() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let idx: Vec<i64> = (0..10).chain(15..25).collect();
        let signals = vec![("ch1".to_string(), x)];
        let input = Input {
            signals: &signals,
            fs: 10.0,
            start: (2026, 1, 1, 0, 0, 0),
            index: &idx,
            events: &[],
            patient: "X X X X",
            recording: "r",
            prefilter: "",
            transducer: "",
            stretches: &[],
        };
        let (bytes, w) = write(&input, &EDF).unwrap();
        assert_eq!((w.gaps, w.samples_filled, w.tail_filled, w.n_records), (1, 5, 5, 3));
        // Two signals in the header, ch1 and the annotations. ch1's samples
        // of the second record are the hole's five places, then five samples.
        let ann_nsr: usize =
            std::str::from_utf8(&bytes[256 + 2 * 216..256 + 2 * 216 + 16]).unwrap()[8..].trim().parse().unwrap();
        let rec = 10 * 2 + ann_nsr * 2;
        assert_eq!(bytes.len(), 256 * 3 + 3 * rec);
        let second: Vec<i16> =
            bytes[256 * 3 + rec..256 * 3 + rec + 20].chunks(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
        let ninth = i16::from_le_bytes([bytes[256 * 3 + 18], bytes[256 * 3 + 19]]);
        assert!(second[..5].iter().all(|&v| v == ninth), "the hole repeats the last value before it");
    }
}
