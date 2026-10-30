//! JSON read and written as the Python library reads and writes it.
//!
//! A manifest is read with `json.loads` and an export carries it again as
//! `json.dumps(manifest, sort_keys=True)`, so an export written here holds
//! the same text only if this module keeps what Python keeps: an integer
//! apart from a float, every float exactly, the order keys arrived in, and
//! `NaN` and `Infinity`, which Python reads and writes although JSON does
//! not define them.

use crate::pyfmt;

/// One JSON value as Python holds it after `json.loads`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Value {
    Null,
    Bool(bool),
    /// An integer, as its canonical decimal digits. Python's integers have
    /// no limit, and digits keep every one exactly.
    Int(String),
    Float(f64),
    Str(String),
    Array(Vec<Value>),
    /// Keys in the order they first arrived. A key repeated keeps its first
    /// place and takes its last value, as a Python dict does.
    Object(Vec<(String, Value)>),
}

impl Value {
    /// The value under `key`, when this is an object that has it.
    pub(crate) fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(items) => items.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// `meta.get(key)` in Python: the value, or Null where it is absent.
    pub(crate) fn field(&self, key: &str) -> &Value {
        const NULL: Value = Value::Null;
        self.get(key).unwrap_or(&NULL)
    }

    pub(crate) fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Python's truth value.
    pub(crate) fn truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(b) => *b,
            Value::Int(d) => d != "0",
            Value::Float(f) => *f != 0.0,
            Value::Str(s) => !s.is_empty(),
            Value::Array(a) => !a.is_empty(),
            Value::Object(o) => !o.is_empty(),
        }
    }

    /// `float(value)` in Python, for a number, a boolean or a numeric
    /// string. None where Python would raise.
    pub(crate) fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            Value::Int(d) => d.parse::<f64>().ok(),
            Value::Float(f) => Some(*f),
            Value::Str(s) => python_float(s.trim()),
            _ => None,
        }
    }

    /// `int(value)` in Python for an integer, a whole float, a boolean, or
    /// a string of digits. None where Python would raise or the result does
    /// not fit 64 bits.
    pub(crate) fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Bool(b) => Some(*b as i64),
            Value::Int(d) => d.parse().ok(),
            Value::Float(f) if f.is_finite() => {
                let t = f.trunc();
                if t.abs() < 9.2e18 {
                    Some(t as i64)
                } else {
                    None
                }
            }
            Value::Str(s) => s.trim().replace('_', "").parse().ok(),
            _ => None,
        }
    }

    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    /// `str(value)` in Python.
    pub(crate) fn py_str(&self) -> String {
        match self {
            Value::Str(s) => s.clone(),
            other => other.py_repr(),
        }
    }

    /// An integer.
    pub(crate) fn int(i: i64) -> Value {
        Value::Int(i.to_string())
    }

    /// A string.
    pub(crate) fn str(s: impl Into<String>) -> Value {
        Value::Str(s.into())
    }

    /// An object from its items, in order.
    pub(crate) fn obj<const N: usize>(items: [(&str, Value); N]) -> Value {
        Value::Object(items.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    /// `d[key] = value` on a Python dict: a key already there keeps its
    /// place and takes the value, a new one goes last.
    pub(crate) fn set(&mut self, key: &str, value: Value) {
        if let Value::Object(items) = self {
            match items.iter_mut().find(|(k, _)| k == key) {
                Some(slot) => slot.1 = value,
                None => items.push((key.to_string(), value)),
            }
        }
    }

    /// `d.pop(key, None)` on a Python dict.
    pub(crate) fn pop(&mut self, key: &str) -> Value {
        match self {
            Value::Object(items) => match items.iter().position(|(k, _)| k == key) {
                Some(i) => items.remove(i).1,
                None => Value::Null,
            },
            _ => Value::Null,
        }
    }

    /// Python's `==` between two values JSON can hold: numbers by value,
    /// whatever their type, and `True` as 1.
    pub(crate) fn py_eq(&self, other: &Value) -> bool {
        use Value::*;
        match (self, other) {
            (Null, Null) => true,
            (Str(a), Str(b)) => a == b,
            (Array(a), Array(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.py_eq(y)),
            (Object(a), Object(b)) => {
                a.len() == b.len() && a.iter().all(|(k, v)| b.iter().any(|(k2, v2)| k == k2 && v.py_eq(v2)))
            }
            (Int(a), Int(b)) => a == b,
            (a, b) if a.is_number() && b.is_number() => match (a.exact_int(), b.exact_int()) {
                (Some(x), Some(y)) => x == y,
                // At least one is a float with a fraction, or not finite,
                // which no integer equals.
                _ => matches!((a, b), (Float(x), Float(y)) if x == y),
            },
            _ => false,
        }
    }

    /// A boolean, an integer or a float.
    pub(crate) fn is_number(&self) -> bool {
        matches!(self, Value::Bool(_) | Value::Int(_) | Value::Float(_))
    }

    /// The value as an integer's canonical digits, whatever its size, for a
    /// boolean, an integer, or a finite float with no fraction written out
    /// in full: Python compares an integer and a float by their exact values.
    fn exact_int(&self) -> Option<String> {
        match self {
            Value::Bool(b) => Some(if *b { "1" } else { "0" }.into()),
            Value::Int(d) => Some(d.clone()),
            Value::Float(f) if f.is_finite() && f.fract() == 0.0 => {
                Some(if *f == 0.0 { "0".into() } else { format!("{f:.0}") })
            }
            _ => None,
        }
    }

    /// The text `json.dumps` writes for this value as a dict's key, and None
    /// for a list or an object, which no dict holds as a key.
    pub(crate) fn json_key(&self) -> Option<String> {
        match self {
            Value::Str(s) => Some(s.clone()),
            Value::Int(d) => Some(d.clone()),
            Value::Float(f) => {
                let mut s = String::new();
                write_float(&mut s, *f);
                Some(s)
            }
            Value::Bool(b) => Some(if *b { "true" } else { "false" }.into()),
            Value::Null => Some("null".into()),
            Value::Array(_) | Value::Object(_) => None,
        }
    }

    /// The value from `serde_json`'s, with its integers and floats kept
    /// apart and its keys in the order it holds them.
    pub(crate) fn from_serde(v: &serde_json::Value) -> Value {
        match v {
            serde_json::Value::Null => Value::Null,
            serde_json::Value::Bool(b) => Value::Bool(*b),
            serde_json::Value::Number(n) => match (n.as_i64(), n.as_u64()) {
                (Some(i), _) => Value::Int(i.to_string()),
                (_, Some(u)) => Value::Int(u.to_string()),
                _ => Value::Float(n.as_f64().unwrap_or(f64::NAN)),
            },
            serde_json::Value::String(s) => Value::Str(s.clone()),
            serde_json::Value::Array(a) => Value::Array(a.iter().map(Value::from_serde).collect()),
            serde_json::Value::Object(o) => {
                Value::Object(o.iter().map(|(k, v)| (k.clone(), Value::from_serde(v))).collect())
            }
        }
    }

    /// The value as `serde_json`'s, for an application. A number too large
    /// for 64 bits becomes a float, and a float that is not finite becomes
    /// null, as `serde_json` holds no other.
    pub(crate) fn to_serde(&self) -> serde_json::Value {
        match self {
            Value::Null => serde_json::Value::Null,
            Value::Bool(b) => serde_json::Value::Bool(*b),
            Value::Int(d) => match (d.parse::<i64>(), d.parse::<u64>()) {
                (Ok(i), _) => serde_json::Value::from(i),
                (_, Ok(u)) => serde_json::Value::from(u),
                _ => serde_json::Number::from_f64(d.parse::<f64>().unwrap_or(f64::NAN))
                    .map_or(serde_json::Value::Null, serde_json::Value::Number),
            },
            Value::Float(f) => serde_json::Number::from_f64(*f).map_or(serde_json::Value::Null, serde_json::Value::Number),
            Value::Str(s) => serde_json::Value::String(s.clone()),
            Value::Array(a) => serde_json::Value::Array(a.iter().map(Value::to_serde).collect()),
            Value::Object(o) => serde_json::Value::Object(o.iter().map(|(k, v)| (k.clone(), v.to_serde())).collect()),
        }
    }

    /// `repr(value)` in Python, for the values JSON can hold.
    pub(crate) fn py_repr(&self) -> String {
        match self {
            Value::Null => "None".into(),
            Value::Bool(true) => "True".into(),
            Value::Bool(false) => "False".into(),
            Value::Int(d) => d.clone(),
            Value::Float(f) => pyfmt::repr(*f),
            Value::Str(s) => repr_str(s),
            Value::Array(a) => format!("[{}]", a.iter().map(|v| v.py_repr()).collect::<Vec<_>>().join(", ")),
            Value::Object(o) => format!(
                "{{{}}}",
                o.iter().map(|(k, v)| format!("{}: {}", repr_str(k), v.py_repr())).collect::<Vec<_>>().join(", ")
            ),
        }
    }
}

/// `float(text)` for the spellings Python accepts.
fn python_float(s: &str) -> Option<f64> {
    let t = s.replace('_', "");
    match t.to_ascii_lowercase().as_str() {
        "nan" | "+nan" | "-nan" => return Some(f64::NAN),
        "inf" | "+inf" | "infinity" | "+infinity" => return Some(f64::INFINITY),
        "-inf" | "-infinity" => return Some(f64::NEG_INFINITY),
        _ => {}
    }
    t.parse::<f64>().ok()
}

/// Python's `repr` of a string: single quotes unless the text holds a single
/// quote and no double quote, and every character Python does not print as
/// itself written as its escape, from Python's own table.
pub(crate) fn repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if unprintable(c) => match c as u32 {
                x if x < 0x100 => out.push_str(&format!("\\x{x:02x}")),
                x if x < 0x10000 => out.push_str(&format!("\\u{x:04x}")),
                x => out.push_str(&format!("\\U{x:08x}")),
            },
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Whether Python's `repr` writes this character as an escape.
fn unprintable(c: char) -> bool {
    let x = c as u32;
    crate::unprintable::UNPRINTABLE
        .binary_search_by(|&(lo, hi)| {
            if hi < x {
                std::cmp::Ordering::Less
            } else if lo > x {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// Why a document is not JSON as Python reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParseError(pub(crate) String);

/// `json.loads(text)`.
pub(crate) fn loads(text: &str) -> Result<Value, ParseError> {
    let mut p = Parser { s: text.as_bytes(), i: 0, text };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.s.len() {
        return Err(p.err("extra data"));
    }
    Ok(v)
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
    text: &'a str,
}

impl<'a> Parser<'a> {
    fn err(&self, what: &str) -> ParseError {
        ParseError(format!("{what} at character {}", self.text[..self.i.min(self.text.len())].chars().count()))
    }

    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn eat(&mut self, lit: &str) -> bool {
        if self.s[self.i..].starts_with(lit.as_bytes()) {
            self.i += lit.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Result<Value, ParseError> {
        let Some(&c) = self.s.get(self.i) else { return Err(self.err("expecting value")) };
        match c {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => Ok(Value::Str(self.string()?)),
            b'n' if self.eat("null") => Ok(Value::Null),
            b't' if self.eat("true") => Ok(Value::Bool(true)),
            b'f' if self.eat("false") => Ok(Value::Bool(false)),
            b'N' if self.eat("NaN") => Ok(Value::Float(f64::NAN)),
            b'I' if self.eat("Infinity") => Ok(Value::Float(f64::INFINITY)),
            b'-' if self.s[self.i..].starts_with(b"-Infinity") => {
                self.i += "-Infinity".len();
                Ok(Value::Float(f64::NEG_INFINITY))
            }
            b'-' | b'0'..=b'9' => self.number(),
            _ => Err(self.err("expecting value")),
        }
    }

    /// `(-?(?:0|[1-9]\d*))(\.\d+)?([eE][-+]?\d+)?`, as Python's scanner
    /// matches a number.
    fn number(&mut self) -> Result<Value, ParseError> {
        let start = self.i;
        let digits = |p: &mut Parser| {
            let s = p.i;
            while p.i < p.s.len() && p.s[p.i].is_ascii_digit() {
                p.i += 1;
            }
            p.i - s
        };
        if self.s[self.i] == b'-' {
            self.i += 1;
        }
        match self.s.get(self.i) {
            Some(b'0') => self.i += 1,
            Some(b'1'..=b'9') => {
                digits(self);
            }
            _ => {
                self.i = start;
                return Err(self.err("expecting value"));
            }
        }
        let int_end = self.i;
        let mut float = false;
        if self.s.get(self.i) == Some(&b'.') && self.s.get(self.i + 1).is_some_and(|c| c.is_ascii_digit()) {
            self.i += 1;
            digits(self);
            float = true;
        }
        if matches!(self.s.get(self.i), Some(b'e' | b'E')) {
            let save = self.i;
            self.i += 1;
            if matches!(self.s.get(self.i), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if digits(self) == 0 {
                self.i = save;
            } else {
                float = true;
            }
        }
        let text = &self.text[start..self.i];
        if float {
            text.parse::<f64>().map(Value::Float).map_err(|_| self.err("a number"))
        } else {
            let t = &self.text[start..int_end];
            let neg = t.starts_with('-');
            let body = t.trim_start_matches('-');
            Ok(Value::Int(if body == "0" || !neg { body.to_string() } else { format!("-{body}") }))
        }
    }

    fn hex4(&mut self) -> Result<u32, ParseError> {
        let h = self.s.get(self.i..self.i + 4).ok_or_else(|| self.err("invalid \\uXXXX escape"))?;
        let h = std::str::from_utf8(h).map_err(|_| self.err("invalid \\uXXXX escape"))?;
        if !h.bytes().all(|c| c.is_ascii_hexdigit()) {
            return Err(self.err("invalid \\uXXXX escape"));
        }
        self.i += 4;
        Ok(u32::from_str_radix(h, 16).expect("four hex digits"))
    }

    fn string(&mut self) -> Result<String, ParseError> {
        self.i += 1; // the opening quote
        let mut out = String::new();
        loop {
            let start = self.i;
            while self.i < self.s.len() && self.s[self.i] != b'"' && self.s[self.i] != b'\\' && self.s[self.i] >= 0x20 {
                self.i += 1;
            }
            out.push_str(&self.text[start..self.i]);
            let Some(&c) = self.s.get(self.i) else { return Err(self.err("unterminated string")) };
            match c {
                b'"' => {
                    self.i += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.i += 1;
                    let Some(&e) = self.s.get(self.i) else { return Err(self.err("unterminated string")) };
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let u = self.hex4()?;
                            let code = if (0xd800..0xdc00).contains(&u) && self.s[self.i..].starts_with(b"\\u") {
                                let save = self.i;
                                self.i += 2;
                                let lo = self.hex4()?;
                                if (0xdc00..0xe000).contains(&lo) {
                                    0x10000 + ((u - 0xd800) << 10) + (lo - 0xdc00)
                                } else {
                                    self.i = save;
                                    u
                                }
                            } else {
                                u
                            };
                            // Python keeps a lone surrogate. A Rust string cannot
                            // hold one, so such a manifest is refused rather than
                            // altered.
                            out.push(char::from_u32(code).ok_or_else(|| self.err("a lone surrogate"))?);
                        }
                        _ => return Err(self.err("invalid escape")),
                    }
                }
                _ => return Err(self.err("invalid control character")),
            }
        }
    }

    fn array(&mut self) -> Result<Value, ParseError> {
        self.i += 1;
        let mut items = Vec::new();
        self.ws();
        if self.s.get(self.i) == Some(&b']') {
            self.i += 1;
            return Ok(Value::Array(items));
        }
        loop {
            self.ws();
            items.push(self.value()?);
            self.ws();
            match self.s.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Value::Array(items));
                }
                _ => return Err(self.err("expecting ',' delimiter")),
            }
        }
    }

    fn object(&mut self) -> Result<Value, ParseError> {
        self.i += 1;
        let mut items: Vec<(String, Value)> = Vec::new();
        self.ws();
        if self.s.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Ok(Value::Object(items));
        }
        loop {
            self.ws();
            if self.s.get(self.i) != Some(&b'"') {
                return Err(self.err("expecting property name enclosed in double quotes"));
            }
            let k = self.string()?;
            self.ws();
            if self.s.get(self.i) != Some(&b':') {
                return Err(self.err("expecting ':' delimiter"));
            }
            self.i += 1;
            self.ws();
            let v = self.value()?;
            match items.iter_mut().find(|(have, _)| *have == k) {
                Some(slot) => slot.1 = v,
                None => items.push((k, v)),
            }
            self.ws();
            match self.s.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Value::Object(items));
                }
                _ => return Err(self.err("expecting ',' delimiter")),
            }
        }
    }
}

/// A string as `json.dumps` writes it with `ensure_ascii`: everything
/// outside the printable ASCII range escaped, in lowercase hex.
fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            c => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{:04x}", unit));
                }
            }
        }
    }
    out.push('"');
}

fn write_float(out: &mut String, f: f64) {
    if f.is_nan() {
        out.push_str("NaN");
    } else if f.is_infinite() {
        out.push_str(if f > 0.0 { "Infinity" } else { "-Infinity" });
    } else {
        out.push_str(&pyfmt::repr(f));
    }
}

/// `json.dumps(value, sort_keys=True)`: one line, `", "` and `": "`.
pub(crate) fn dumps_sorted(v: &Value) -> String {
    one_line(v, true)
}

/// `json.dumps(value)`: one line, keys in their order.
pub(crate) fn dumps(v: &Value) -> String {
    one_line(v, false)
}

fn one_line(v: &Value, sort: bool) -> String {
    fn go(out: &mut String, v: &Value, sort: bool) {
        match v {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Value::Int(d) => out.push_str(d),
            Value::Float(f) => write_float(out, *f),
            Value::Str(s) => write_str(out, s),
            Value::Array(a) => {
                out.push('[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    go(out, x, sort);
                }
                out.push(']');
            }
            Value::Object(o) => {
                let mut items: Vec<&(String, Value)> = o.iter().collect();
                if sort {
                    items.sort_by(|a, b| a.0.cmp(&b.0));
                }
                out.push('{');
                for (i, (k, x)) in items.into_iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    write_str(out, k);
                    out.push_str(": ");
                    go(out, x, sort);
                }
                out.push('}');
            }
        }
    }
    let mut out = String::new();
    go(&mut out, v, sort);
    out
}

/// `json.dumps(value, indent=indent)`: keys in their order, `","` at line
/// ends, and `": "` after a key.
pub(crate) fn dumps_indented(v: &Value, indent: usize) -> String {
    fn go(out: &mut String, v: &Value, indent: usize, level: usize) {
        let pad = |out: &mut String, n: usize| out.push_str(&" ".repeat(indent * n));
        match v {
            Value::Array(a) if !a.is_empty() => {
                out.push('[');
                for (i, x) in a.iter().enumerate() {
                    out.push_str(if i > 0 { ",\n" } else { "\n" });
                    pad(out, level + 1);
                    go(out, x, indent, level + 1);
                }
                out.push('\n');
                pad(out, level);
                out.push(']');
            }
            Value::Object(o) if !o.is_empty() => {
                out.push('{');
                for (i, (k, x)) in o.iter().enumerate() {
                    out.push_str(if i > 0 { ",\n" } else { "\n" });
                    pad(out, level + 1);
                    write_str(out, k);
                    out.push_str(": ");
                    go(out, x, indent, level + 1);
                }
                out.push('\n');
                pad(out, level);
                out.push('}');
            }
            other => out.push_str(&dumps_sorted(other)),
        }
    }
    let mut out = String::new();
    go(&mut out, v, indent, 0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_manifest_reads_back_to_the_same_text() {
        let text = r#"{"b": 1, "a": [1.5, -0, 2e-05, "é😀", null, true], "n": NaN, "i": -Infinity, "b": 2}"#;
        let v = loads(text).unwrap();
        assert_eq!(
            dumps_sorted(&v),
            r#"{"a": [1.5, 0, 2e-05, "\u00e9\ud83d\ude00", null, true], "b": 2, "i": -Infinity, "n": NaN}"#
        );
        assert_eq!(v.get("b"), Some(&Value::Int("2".into())), "a repeated key takes its last value");
        assert!(matches!(&v, Value::Object(o) if o[0].0 == "b"), "and keeps its first place");
    }

    #[test]
    fn indented_is_pythons() {
        let v = loads(r#"{"a": [], "b": {}, "c": [1, {"d": "x"}]}"#).unwrap();
        assert_eq!(
            dumps_indented(&v, 2),
            "{\n  \"a\": [],\n  \"b\": {},\n  \"c\": [\n    1,\n    {\n      \"d\": \"x\"\n    }\n  ]\n}"
        );
    }

    #[test]
    fn what_python_refuses_is_refused() {
        for bad in ["{\"a\": 1,}", "[1 2]", "01", "1.", "\"\u{1}\"", "{'a': 1}", "[1] x"] {
            assert!(loads(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn str_is_pythons() {
        let v = loads(r#"[null, true, 1.0, "it's", {"k": "v"}]"#).unwrap();
        assert_eq!(v.py_str(), r#"[None, True, 1.0, "it's", {'k': 'v'}]"#);
    }
}
