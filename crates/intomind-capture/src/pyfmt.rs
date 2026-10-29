//! Numbers written exactly as the Python library writes them.
//!
//! Every export is held to the bytes the Python library produces, and those
//! bytes come from Python's own formatting: `repr` for a float in JSON,
//! `f"{x:.6f}"` in text and annotations, `f"{x:.8g}"` in an EDF header, and
//! `decimal.Decimal` when a header bound is moved outward. Each is
//! reproduced here from its definition, and the fixtures in
//! `contract/recordings` hold them to what Python printed.

/// The shortest decimal digits that read back as `x`, and the exponent of
/// the first of them: `x = d.ddd × 10^exp`. `x` is finite and not zero.
///
/// Python's `repr` takes the fewest digits that round trip, and of those
/// the closest to `x`, and where two are equally close, the even one. Rust's
/// `{:e}` finds the same number of digits, and breaks that tie upward:
/// 15.5887603759765625 is 15.588760375976563 there and 15.588760375976562
/// in Python. So the count comes from `{:e}`, and the digits from rounding
/// `x` to that many places half to even, which is the closest, whenever
/// that reads back as `x`.
fn shortest(x: f64) -> (String, i32) {
    let split = |s: &str| -> (String, i32) {
        let (mant, exp) = s.split_once('e').expect("{:e} always has an exponent");
        (mant.replace('.', ""), exp.parse().expect("an integer exponent"))
    };
    let a = x.abs();
    let (digits, exp) = split(&format!("{a:e}"));
    let even = format!("{:.*e}", digits.len() - 1, a);
    if even.parse::<f64>() == Ok(a) {
        return split(&even);
    }
    (digits, exp)
}

/// `d.ddd` and the exponent, as Python writes an exponent: a sign and at
/// least two digits.
fn scientific(neg: bool, digits: &str, exp: i32) -> String {
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    out.push_str(&digits[..1]);
    if digits.len() > 1 {
        out.push('.');
        out.push_str(&digits[1..]);
    }
    out.push('e');
    out.push(if exp < 0 { '-' } else { '+' });
    out.push_str(&format!("{:02}", exp.unsigned_abs()));
    out
}

/// Python's `repr` of a float, which is also what `json.dumps` writes for
/// a finite one.
pub(crate) fn repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    let (digits, exp) = shortest(x);
    let neg = x < 0.0;
    // Where the decimal point falls, counted from the left of the digits.
    let point = exp + 1;
    if point <= -4 || point > 16 {
        return scientific(neg, &digits, exp);
    }
    let mut out = String::from(if neg { "-" } else { "" });
    let n = digits.len() as i32;
    if point <= 0 {
        out.push_str("0.");
        out.push_str(&"0".repeat((-point) as usize));
        out.push_str(&digits);
    } else if point < n {
        out.push_str(&digits[..point as usize]);
        out.push('.');
        out.push_str(&digits[point as usize..]);
    } else {
        out.push_str(&digits);
        out.push_str(&"0".repeat((point - n) as usize));
        out.push_str(".0");
    }
    out
}

/// `f"{x:.{places}f}"`. Rust's fixed formatting rounds the exact binary
/// value half to even, as Python's does, and keeps the sign of a negative
/// zero, as Python's does. Only the spelling of the two special values
/// differs.
pub(crate) fn fixed(x: f64, places: usize) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    format!("{x:.places$}")
}

/// `f"{x:.{precision}g}"`: `precision` significant digits, trailing zeros
/// and a bare decimal point removed, and an exponent when the rounded
/// value's exponent is below -4 or at least `precision`.
pub(crate) fn general(x: f64, precision: usize) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    let p = precision.max(1);
    // Rounded to p significant digits, half to even on the exact value.
    let s = format!("{:.*e}", p - 1, x.abs());
    let (mant, exp) = s.split_once('e').expect("{:e} always has an exponent");
    let exp: i32 = exp.parse().expect("an integer exponent");
    let digits = mant.replace('.', "");
    let neg = x.is_sign_negative();
    if exp < -4 || exp >= p as i32 {
        let kept = digits.trim_end_matches('0');
        let kept = if kept.is_empty() { "0" } else { kept };
        return scientific(neg, kept, exp);
    }
    // Fixed, with p - 1 - exp places, then trimmed.
    let places = (p as i32 - 1 - exp).max(0) as usize;
    let mut out = format!("{:.places$}", x.abs());
    if out.contains('.') {
        out = out.trim_end_matches('0').trim_end_matches('.').to_string();
    }
    if neg {
        out.insert(0, '-');
    }
    out
}

/// Which way `quantize` rounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Toward {
    /// `decimal.ROUND_FLOOR`.
    Floor,
    /// `decimal.ROUND_CEILING`.
    Ceiling,
}

/// `f"{Decimal(repr(x)).quantize(Decimal(1).scaleb(-places), rounding=…):f}"`
/// for a finite `x`, or None where Python's default context (28 digits)
/// refuses the quantize and raises.
pub(crate) fn quantize(x: f64, places: u32, toward: Toward) -> Option<String> {
    let r = repr(x);
    let neg = r.starts_with('-');
    let body = r.trim_start_matches('-');
    // The decimal's coefficient and exponent, from the repr's own digits.
    let (mant, e10) = match body.split_once('e') {
        Some((m, e)) => (m.to_string(), e.parse::<i64>().ok()?),
        None => (body.to_string(), 0),
    };
    let (int_part, frac_part) = mant.split_once('.').unwrap_or((&mant, ""));
    let mut coeff: Vec<u8> = format!("{int_part}{frac_part}").into_bytes();
    let exp = e10 - frac_part.len() as i64;
    let target = -(places as i64);
    if exp >= target {
        coeff.resize(coeff.len() + (exp - target) as usize, b'0');
    } else {
        let drop = (target - exp) as usize;
        let (keep, rest) = if drop >= coeff.len() {
            (Vec::new(), coeff.clone())
        } else {
            let k = coeff.len() - drop;
            (coeff[..k].to_vec(), coeff[k..].to_vec())
        };
        let inexact = rest.iter().any(|&d| d != b'0');
        coeff = if keep.is_empty() { vec![b'0'] } else { keep };
        // Away from zero when the rounding direction points away from it.
        let away = inexact && matches!((toward, neg), (Toward::Ceiling, false) | (Toward::Floor, true));
        if away {
            let mut i = coeff.len();
            loop {
                if i == 0 {
                    coeff.insert(0, b'1');
                    break;
                }
                i -= 1;
                if coeff[i] == b'9' {
                    coeff[i] = b'0';
                } else {
                    coeff[i] += 1;
                    break;
                }
            }
        }
    }
    // The coefficient without leading zeros, as Decimal holds it.
    let digits: String = {
        let s = String::from_utf8(coeff).expect("ascii digits");
        let t = s.trim_start_matches('0');
        if t.is_empty() {
            "0".to_string()
        } else {
            t.to_string()
        }
    };
    if digits.len() > 28 {
        return None;
    }
    // Written in fixed notation with exactly `places` places, the sign of a
    // zero kept, as Decimal formats it.
    let places = places as usize;
    let padded =
        if digits.len() <= places { format!("{}{}", "0".repeat(places + 1 - digits.len()), digits) } else { digits };
    let split = padded.len() - places;
    let mut out = String::from(if neg { "-" } else { "" });
    out.push_str(&padded[..split]);
    if places > 0 {
        out.push('.');
        out.push_str(&padded[split..]);
    }
    Some(out)
}

/// The sum numpy computes for a contiguous float64 array: pairwise within
/// blocks of 8192, the blocks added left to right.
pub(crate) fn numpy_sum(a: &[f64]) -> f64 {
    fn pairwise(a: &[f64]) -> f64 {
        let n = a.len();
        if n < 8 {
            let mut r = 0.0;
            for &v in a {
                r += v;
            }
            r
        } else if n <= 128 {
            let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
            let mut i = 8;
            while i < n - (n % 8) {
                for j in 0..8 {
                    r[j] += a[i + j];
                }
                i += 8;
            }
            let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            while i < n {
                res += a[i];
                i += 1;
            }
            res
        } else {
            let mut n2 = n / 2;
            n2 -= n2 % 8;
            pairwise(&a[..n2]) + pairwise(&a[n2..])
        }
    }
    let mut chunks = a.chunks(8192);
    let mut s = match chunks.next() {
        Some(c) => pairwise(c),
        None => return 0.0,
    };
    for c in chunks {
        s += pairwise(c);
    }
    s
}

/// `numpy.mean` of a contiguous float64 array.
pub(crate) fn numpy_mean(a: &[f64]) -> f64 {
    numpy_sum(a) / a.len() as f64
}

/// Digits with a comma between every three, counted from the right.
fn commas(digits: &str) -> String {
    let mut rev = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            rev.push(',');
        }
        rev.push(c);
    }
    rev.chars().rev().collect()
}

/// An integer with thousands separators, as `f"{n:,}"` writes it.
pub(crate) fn grouped(n: i64) -> String {
    let out = commas(&n.unsigned_abs().to_string());
    if n < 0 {
        format!("-{out}")
    } else {
        out
    }
}

/// `f"{x:,.{places}f}"`.
pub(crate) fn grouped_fixed(x: f64, places: usize) -> String {
    let s = fixed(x, places);
    if !x.is_finite() {
        return s;
    }
    let (neg, body) = match s.strip_prefix('-') {
        Some(b) => (true, b.to_string()),
        None => (false, s.clone()),
    };
    let (int_part, frac) = match body.split_once('.') {
        Some((i, f)) => (i.to_string(), Some(f.to_string())),
        None => (body.clone(), None),
    };
    let mut out = commas(&int_part);
    if let Some(f) = frac {
        out.push('.');
        out.push_str(&f);
    }
    if neg {
        out.insert(0, '-');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repr_is_pythons() {
        for (x, want) in [
            (1e-5, "1e-05"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (0.0001, "0.0001"),
            (498.80345678912346, "498.80345678912346"),
            (2.0, "2.0"),
            (-0.0, "-0.0"),
            (5e-324, "5e-324"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (0.1, "0.1"),
            (-1.5e-7, "-1.5e-07"),
            (123456789012345680.0, "1.2345678901234568e+17"),
            (f64::NAN, "nan"),
            (f64::NEG_INFINITY, "-inf"),
            // Exactly 15.5887603759765625, halfway between two 17-digit spellings.
            (f64::from_bits(0x402f_2d72_0000_0000), "15.588760375976562"),
        ] {
            assert_eq!(repr(x), want, "{x:?}");
        }
    }

    #[test]
    fn general_is_pythons() {
        for (x, p, want) in [
            (0.0001, 8, "0.0001"),
            (1e-5, 8, "1e-05"),
            (123456789.0, 8, "1.2345679e+08"),
            (-0.0, 8, "-0"),
            (0.0, 8, "0"),
            (498.8034, 8, "498.8034"),
            (12345678.0, 8, "12345678"),
            (99999999.5, 8, "1e+08"),
            (1.0, 8, "1"),
            (-4231.123456789, 8, "-4231.1235"),
            (0.5, 1, "0.5"),
            (2.5, 1, "2"),
        ] {
            assert_eq!(general(x, p), want, "{x:?} {p}");
        }
    }

    #[test]
    fn quantize_is_decimals() {
        use Toward::*;
        for (x, places, t, want) in [
            (-4231.1234567891, 7, Floor, "-4231.1234568"),
            (-4231.1234567891, 2, Ceiling, "-4231.12"),
            (-1e-9, 7, Ceiling, "-0.0000000"),
            (1e-9, 7, Floor, "0.0000000"),
            (1e-9, 7, Ceiling, "0.0000001"),
            (9.99999999, 2, Ceiling, "10.00"),
            (-9.99999999, 2, Floor, "-10.00"),
            (123.0, 0, Floor, "123"),
            (0.5, 0, Ceiling, "1"),
            (-0.5, 0, Ceiling, "-0"),
        ] {
            assert_eq!(quantize(x, places, t).as_deref(), Some(want), "{x:?} {places} {t:?}");
        }
        assert_eq!(quantize(1e30, 7, Floor), None, "past 28 digits Python's quantize raises");
    }

    #[test]
    fn sums_add_up_as_numpys_do() {
        let a: Vec<f64> = (0..20).map(|i| (i as f64).sin() * 10f64.powi(i % 7)).collect();
        assert_eq!(numpy_sum(&a[..5]), a[..5].iter().fold(0.0, |s, v| s + v));
        assert_eq!(grouped(1234567), "1,234,567");
        assert_eq!(grouped_fixed(1945.4, 0), "1,945");
    }
}
