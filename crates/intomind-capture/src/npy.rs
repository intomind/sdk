//! numpy's `.npy` arrays, read and written as numpy writes them.
//!
//! Only the element types a recording or an export holds are here:
//! booleans, integers, floats, and the fixed-width unicode text an export
//! keeps its manifest in. A two-dimensional array may be stored in either
//! order, and the recorder stores its counts sample by sample, which numpy
//! calls Fortran order for a channels by samples array.

use std::fmt;

/// Why an array could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NpyError(pub(crate) String);

impl fmt::Display for NpyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

fn bad(what: impl Into<String>) -> NpyError {
    NpyError(what.into())
}

/// The values of an array, in the order they are stored.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Data {
    Bool(Vec<bool>),
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    /// Fixed-width text of `width` code points per element.
    Unicode {
        width: usize,
        values: Vec<String>,
    },
}

impl Data {
    pub(crate) fn len(&self) -> usize {
        match self {
            Data::Bool(v) => v.len(),
            Data::U8(v) => v.len(),
            Data::I8(v) => v.len(),
            Data::U16(v) => v.len(),
            Data::I16(v) => v.len(),
            Data::U32(v) => v.len(),
            Data::I32(v) => v.len(),
            Data::U64(v) => v.len(),
            Data::I64(v) => v.len(),
            Data::F32(v) => v.len(),
            Data::F64(v) => v.len(),
            Data::Unicode { values, .. } => values.len(),
        }
    }

    /// numpy's name for the element type, as `dtype.str` spells it.
    pub(crate) fn descr(&self) -> String {
        match self {
            Data::Bool(_) => "|b1".into(),
            Data::U8(_) => "|u1".into(),
            Data::I8(_) => "|i1".into(),
            Data::U16(_) => "<u2".into(),
            Data::I16(_) => "<i2".into(),
            Data::U32(_) => "<u4".into(),
            Data::I32(_) => "<i4".into(),
            Data::U64(_) => "<u8".into(),
            Data::I64(_) => "<i8".into(),
            Data::F32(_) => "<f4".into(),
            Data::F64(_) => "<f8".into(),
            Data::Unicode { width, .. } => format!("<U{width}"),
        }
    }

    /// Element `i` as a float, as numpy's `astype(float64)` converts it.
    pub(crate) fn f64_at(&self, i: usize) -> f64 {
        match self {
            Data::Bool(v) => v[i] as u8 as f64,
            Data::U8(v) => v[i] as f64,
            Data::I8(v) => v[i] as f64,
            Data::U16(v) => v[i] as f64,
            Data::I16(v) => v[i] as f64,
            Data::U32(v) => v[i] as f64,
            Data::I32(v) => v[i] as f64,
            Data::U64(v) => v[i] as f64,
            Data::I64(v) => v[i] as f64,
            Data::F32(v) => v[i] as f64,
            Data::F64(v) => v[i],
            Data::Unicode { .. } => f64::NAN,
        }
    }

    /// Element `i` as an integer, as numpy's `astype(int64)` converts it.
    /// A float is truncated toward zero.
    pub(crate) fn i64_at(&self, i: usize) -> i64 {
        match self {
            Data::Bool(v) => v[i] as i64,
            Data::U8(v) => v[i] as i64,
            Data::I8(v) => v[i] as i64,
            Data::U16(v) => v[i] as i64,
            Data::I16(v) => v[i] as i64,
            Data::U32(v) => v[i] as i64,
            Data::I32(v) => v[i] as i64,
            Data::U64(v) => v[i] as i64,
            Data::I64(v) => v[i],
            Data::F32(v) => v[i] as i64,
            Data::F64(v) => v[i] as i64,
            Data::Unicode { .. } => 0,
        }
    }

    /// Element `i` as numpy's `astype(bool)` converts it: nonzero is true.
    pub(crate) fn bool_at(&self, i: usize) -> bool {
        match self {
            Data::Bool(v) => v[i],
            Data::F32(v) => v[i] != 0.0,
            Data::F64(v) => v[i] != 0.0,
            Data::Unicode { values, .. } => !values[i].is_empty(),
            other => other.i64_at(i) != 0,
        }
    }

    /// The elements at `keep`, in that order, as numpy's fancy indexing
    /// takes them.
    pub(crate) fn take(&self, keep: &[usize]) -> Data {
        fn t<T: Clone>(v: &[T], keep: &[usize]) -> Vec<T> {
            keep.iter().map(|&i| v[i].clone()).collect()
        }
        match self {
            Data::Bool(v) => Data::Bool(t(v, keep)),
            Data::U8(v) => Data::U8(t(v, keep)),
            Data::I8(v) => Data::I8(t(v, keep)),
            Data::U16(v) => Data::U16(t(v, keep)),
            Data::I16(v) => Data::I16(t(v, keep)),
            Data::U32(v) => Data::U32(t(v, keep)),
            Data::I32(v) => Data::I32(t(v, keep)),
            Data::U64(v) => Data::U64(t(v, keep)),
            Data::I64(v) => Data::I64(t(v, keep)),
            Data::F32(v) => Data::F32(t(v, keep)),
            Data::F64(v) => Data::F64(t(v, keep)),
            Data::Unicode { width, values } => Data::Unicode { width: *width, values: t(values, keep) },
        }
    }

    /// The elements from `start` on.
    pub(crate) fn skip(&self, start: usize) -> Data {
        let keep: Vec<usize> = (start.min(self.len())..self.len()).collect();
        self.take(&keep)
    }

    /// The bytes numpy stores for these elements, little endian.
    fn bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            Data::Bool(v) => out.extend(v.iter().map(|&b| b as u8)),
            Data::U8(v) => out.extend(v),
            Data::I8(v) => out.extend(v.iter().map(|&x| x as u8)),
            Data::U16(v) => v.iter().for_each(|x| out.extend(x.to_le_bytes())),
            Data::I16(v) => v.iter().for_each(|x| out.extend(x.to_le_bytes())),
            Data::U32(v) => v.iter().for_each(|x| out.extend(x.to_le_bytes())),
            Data::I32(v) => v.iter().for_each(|x| out.extend(x.to_le_bytes())),
            Data::U64(v) => v.iter().for_each(|x| out.extend(x.to_le_bytes())),
            Data::I64(v) => v.iter().for_each(|x| out.extend(x.to_le_bytes())),
            Data::F32(v) => v.iter().for_each(|x| out.extend(x.to_le_bytes())),
            Data::F64(v) => v.iter().for_each(|x| out.extend(x.to_le_bytes())),
            Data::Unicode { width, values } => {
                for s in values {
                    let mut n = 0;
                    for c in s.chars() {
                        out.extend((c as u32).to_le_bytes());
                        n += 1;
                    }
                    out.resize(out.len() + 4 * (width - n), 0);
                }
            }
        }
        out
    }
}

/// One array: its shape, the order its elements are stored in, and the
/// elements.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Array {
    pub(crate) shape: Vec<usize>,
    /// Stored first dimension fastest, numpy's `fortran_order`.
    pub(crate) fortran: bool,
    pub(crate) data: Data,
}

impl Array {
    /// A one-dimensional array.
    pub(crate) fn vector(data: Data) -> Array {
        Array { shape: vec![data.len()], fortran: false, data }
    }

    /// A zero-dimensional text array, as `np.array(text)` makes one. numpy
    /// gives an empty text a width of one.
    pub(crate) fn text(s: &str) -> Array {
        let width = s.chars().count().max(1);
        Array { shape: vec![], fortran: false, data: Data::Unicode { width, values: vec![s.to_string()] } }
    }

    /// Row `r` of a two-dimensional array, in order, whichever order it is
    /// stored in. A one-dimensional array has one row.
    pub(crate) fn row_indices(&self, r: usize) -> Vec<usize> {
        match self.shape.as_slice() {
            [rows, cols] => {
                if self.fortran {
                    (0..*cols).map(|c| c * rows + r).collect()
                } else {
                    (0..*cols).map(|c| r * cols + c).collect()
                }
            }
            _ => (0..self.data.len()).collect(),
        }
    }
}

/// The header numpy writes for an array: the dictionary with its keys in
/// order, the spare room numpy leaves for the growing axis, and padding so
/// that the data starts on a 64-byte boundary.
fn header(a: &Array) -> Vec<u8> {
    let shape = match a.shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", a.shape[0]),
        _ => format!("({})", a.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut h = format!(
        "{{'descr': '{}', 'fortran_order': {}, 'shape': {}, }}",
        a.data.descr(),
        if a.fortran { "True" } else { "False" },
        shape
    );
    if !a.shape.is_empty() {
        let growth = if a.fortran { a.shape[a.shape.len() - 1] } else { a.shape[0] };
        h.push_str(&" ".repeat(21usize.saturating_sub(growth.to_string().len())));
    }
    // Version 1.0 holds a header length in two bytes, and numpy moves to
    // 2.0, four bytes, only when it does not fit.
    let hlen = h.len() + 1;
    let pad1 = 64 - ((10 + hlen) % 64);
    let (version, prefix_len, pad) =
        if hlen + pad1 <= 65_535 { (1u8, 10usize, pad1) } else { (2u8, 12usize, 64 - ((12 + hlen) % 64)) };
    let total = hlen + pad;
    let mut out = Vec::with_capacity(prefix_len + total);
    out.extend(b"\x93NUMPY");
    out.push(version);
    out.push(0);
    if version == 1 {
        out.extend((total as u16).to_le_bytes());
    } else {
        out.extend((total as u32).to_le_bytes());
    }
    out.extend(h.as_bytes());
    out.resize(out.len() + pad, b' ');
    out.push(b'\n');
    out
}

/// The `.npy` file numpy's `save` writes for `a`.
pub(crate) fn write(a: &Array) -> Vec<u8> {
    let mut out = header(a);
    out.extend(a.data.bytes());
    out
}

/// One value of the header dictionary, as text.
fn header_field<'a>(h: &'a str, key: &str) -> Option<&'a str> {
    let at = h.find(&format!("'{key}'"))?;
    let rest = h[at + key.len() + 2..].trim_start().strip_prefix(':')?.trim_start();
    let end = if rest.starts_with('(') {
        rest.find(')')? + 1
    } else if let Some(q) = rest.strip_prefix('\'') {
        q.find('\'')? + 2
    } else {
        rest.find([',', '}'])?
    };
    Some(rest[..end].trim())
}

/// An `.npy` file's array.
pub(crate) fn read(b: &[u8]) -> Result<Array, NpyError> {
    if !b.starts_with(b"\x93NUMPY") || b.len() < 10 {
        return Err(bad("not a numpy array file"));
    }
    let (hlen, start) = match b[6] {
        1 => (u16::from_le_bytes([b[8], b[9]]) as usize, 10),
        2 | 3 => {
            let s = b.get(8..12).ok_or_else(|| bad("the array file ends early"))?;
            (u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize, 12)
        }
        v => return Err(bad(format!("numpy array format version {v} is not one numpy writes"))),
    };
    let h = b.get(start..start + hlen).ok_or_else(|| bad("the array file ends early"))?;
    let h = std::str::from_utf8(h).map_err(|_| bad("the array header is not text"))?;
    let descr = header_field(h, "descr").ok_or_else(|| bad("the array header has no descr"))?.trim_matches('\'');
    let fortran = match header_field(h, "fortran_order") {
        Some("True") => true,
        Some("False") => false,
        _ => return Err(bad("the array header has no fortran_order")),
    };
    let shape_s = header_field(h, "shape").ok_or_else(|| bad("the array header has no shape"))?;
    let shape: Vec<usize> = shape_s
        .trim_matches(|c| c == '(' || c == ')')
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.trim_end_matches('L').parse::<usize>().map_err(|_| bad("the array shape is not a list of sizes")))
        .collect::<Result<_, _>>()?;
    let count: usize = shape.iter().product();
    let body = &b[start + hlen..];
    let (order, kind) = descr.split_at(1);
    let big = match order {
        "<" | "|" | "=" => false,
        ">" => true,
        _ => return Err(bad(format!("element type {descr} is not one a recording holds"))),
    };
    macro_rules! nums {
        ($t:ty, $n:expr, $variant:ident) => {{
            let need = count * $n;
            let raw = body.get(..need).ok_or_else(|| bad("the array file ends early"))?;
            Data::$variant(
                raw.chunks_exact($n)
                    .map(|c| {
                        let a: [u8; $n] = c.try_into().expect("a chunk of the element size");
                        if big {
                            <$t>::from_be_bytes(a)
                        } else {
                            <$t>::from_le_bytes(a)
                        }
                    })
                    .collect(),
            )
        }};
    }
    let data = match kind {
        "b1" => Data::Bool(
            body.get(..count).ok_or_else(|| bad("the array file ends early"))?.iter().map(|&x| x != 0).collect(),
        ),
        "u1" => Data::U8(body.get(..count).ok_or_else(|| bad("the array file ends early"))?.to_vec()),
        "i1" => Data::I8(
            body.get(..count).ok_or_else(|| bad("the array file ends early"))?.iter().map(|&x| x as i8).collect(),
        ),
        "u2" => nums!(u16, 2, U16),
        "i2" => nums!(i16, 2, I16),
        "u4" => nums!(u32, 4, U32),
        "i4" => nums!(i32, 4, I32),
        "u8" => nums!(u64, 8, U64),
        "i8" => nums!(i64, 8, I64),
        "f4" => nums!(f32, 4, F32),
        "f8" => nums!(f64, 8, F64),
        k if k.starts_with('U') => {
            let width: usize =
                k[1..].parse().map_err(|_| bad(format!("element type {descr} is not one a recording holds")))?;
            let raw = body.get(..count * width * 4).ok_or_else(|| bad("the array file ends early"))?;
            let mut values = Vec::with_capacity(count);
            for item in raw.chunks_exact((width * 4).max(1)).take(count) {
                let mut s = String::new();
                for u in item.chunks_exact(4) {
                    let a: [u8; 4] = u.try_into().expect("four bytes");
                    let cp = if big { u32::from_be_bytes(a) } else { u32::from_le_bytes(a) };
                    if cp == 0 {
                        break;
                    }
                    s.push(char::from_u32(cp).ok_or_else(|| bad("text that is not unicode"))?);
                }
                values.push(s);
            }
            Data::Unicode { width, values }
        }
        _ => return Err(bad(format!("element type {descr} is not one a recording holds"))),
    };
    Ok(Array { shape, fortran, data })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headers_are_numpys() {
        // numpy 2.2: np.save of np.zeros(1500, np.float32)
        let a = Array::vector(Data::F32(vec![0.0; 1500]));
        let h = header(&a);
        assert_eq!(h.len() % 64, 0);
        assert!(std::str::from_utf8(&h[10..])
            .unwrap()
            .starts_with("{'descr': '<f4', 'fortran_order': False, 'shape': (1500,), }"));
        let t = Array::text("{}");
        assert!(std::str::from_utf8(&header(&t)[10..])
            .unwrap()
            .starts_with("{'descr': '<U2', 'fortran_order': False, 'shape': (), }"));
    }

    #[test]
    fn arrays_read_back_as_written() {
        let counts = Array { shape: vec![2, 3], fortran: true, data: Data::I64(vec![1, 10, 2, 20, 3, 30]) };
        let back = read(&write(&counts)).unwrap();
        assert_eq!(back, counts);
        assert_eq!(back.row_indices(1).iter().map(|&i| back.data.i64_at(i)).collect::<Vec<_>>(), vec![10, 20, 30]);
        let t = Array::text("é{\"a\": 1}");
        assert_eq!(read(&write(&t)).unwrap(), t);
    }
}
