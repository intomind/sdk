//! Device time on the host's clocks, as the Python library maps it, and
//! the clock record a recording carries.
//!
//! Every time exchange is read on two host clocks. The device's time is
//! fitted against the monotonic one, which cannot step, and the wall clock
//! read beside it is the anchor that carries the fit back into the time a
//! recording is written in. While a stream runs the fit that dates its
//! samples is held still, so a recording is on one ruler from its first
//! sample to its last. Exchanges keep arriving and what they say is
//! reported beside it. A step of the wall clock is found between anchors
//! and written down, never smoothed into the samples.
//!
//! The arithmetic is the library's, operation for operation, so a fit made
//! here is the same bits as the one the library makes from the same
//! exchanges.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use crate::pyfmt;
use crate::pyjson::Value;

/// Beyond this, a slope is not two crystals disagreeing.
pub const MAX_SKEW_PPM: f64 = 1000.0;
/// And a slope has to be measured this well to beat assuming none.
pub const MAX_SKEW_STDERR_PPM: f64 = 100.0;
/// How far realtime and monotonic may drift apart between two anchors
/// without the wall clock having stepped: a floor for the two reads, and
/// the rate at which NTP slews.
const STEP_FLOOR_S: f64 = 0.050;
const SLEW_PPM: f64 = 500e-6;

/// The host's two clocks, read one after the other: monotonic first, then
/// the wall clock, always in that order, so the gap between the two reads
/// is a fixed bias rather than a coin flip.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HostClocks {
    /// Seconds on a clock that never steps, from a point of its own.
    pub monotonic: f64,
    /// Seconds since 1970 on the wall clock.
    pub realtime: f64,
}

impl HostClocks {
    /// Both clocks, now.
    pub fn now() -> HostClocks {
        static START: OnceLock<std::time::Instant> = OnceLock::new();
        let start = *START.get_or_init(std::time::Instant::now);
        let monotonic = start.elapsed().as_secs_f64();
        let realtime =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
        HostClocks { monotonic, realtime }
    }
}

/// One time exchange, as the fit holds it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sync {
    /// The wall clock at the middle of the exchange.
    pub host: f64,
    /// The device's time, in seconds.
    pub device: f64,
    /// The exchange's round trip on the wall clock.
    pub rtt: f64,
    /// The monotonic clock at the middle of the exchange.
    pub mono: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct Learned {
    skew: f64,
    offset: f64,
    pinned: Option<String>,
}

/// The map from device time to host time: the exchanges, the fit, and
/// whether the fit is being held still under a running stream.
#[derive(Debug)]
pub struct Clock {
    tick_hz: f64,
    syncs: Vec<Sync>,
    skew: f64,
    offset: f64,
    pinned: Option<String>,
    frame_offset: f64,
    frame_mono: Option<f64>,
    learned: Option<Learned>,
    frozen: AtomicBool,
}

impl Clone for Clock {
    fn clone(&self) -> Clock {
        Clock {
            tick_hz: self.tick_hz,
            syncs: self.syncs.clone(),
            skew: self.skew,
            offset: self.offset,
            pinned: self.pinned.clone(),
            frame_offset: self.frame_offset,
            frame_mono: self.frame_mono,
            learned: self.learned.clone(),
            frozen: AtomicBool::new(self.frozen.load(Ordering::Relaxed)),
        }
    }
}

impl Clock {
    /// A clock for a device whose time runs at `tick_hz`, with no exchange
    /// taken: the device's own seconds until one is.
    pub fn new(tick_hz: u32) -> Clock {
        Clock {
            tick_hz: tick_hz as f64,
            syncs: Vec::new(),
            skew: 1.0,
            offset: 0.0,
            pinned: None,
            frame_offset: 0.0,
            frame_mono: None,
            learned: None,
            frozen: AtomicBool::new(false),
        }
    }

    /// One exchange: both host clocks before the request and after the
    /// answer, and the device's time in its ticks. The fit is redone over
    /// every exchange so far.
    pub fn observe(&mut self, device_ticks: u64, before: HostClocks, after: HostClocks) {
        let device = device_ticks as f64 / self.tick_hz;
        self.syncs.push(Sync {
            host: (before.realtime + after.realtime) / 2.0,
            device,
            rtt: after.realtime - before.realtime,
            mono: (before.monotonic + after.monotonic) / 2.0,
        });
        self.fit();
    }

    /// Hold the map still: a stream has started, and its samples are dated
    /// by the fit in force now.
    pub fn freeze(&self) {
        self.frozen.store(true, Ordering::Relaxed);
    }

    /// Let the map move again: the stream has ended.
    pub fn thaw(&self) {
        self.frozen.store(false, Ordering::Relaxed);
    }

    /// Whether the map is being held still.
    pub fn frozen(&self) -> bool {
        self.frozen.load(Ordering::Relaxed)
    }

    /// The exchanges so far.
    pub fn syncs(&self) -> &[Sync] {
        &self.syncs
    }

    /// Device ticks on the monotonic clock.
    pub fn mono_time(&self, device_ticks: i64) -> f64 {
        (device_ticks as f64 / self.tick_hz) * self.skew + self.offset
    }

    /// Device ticks on the wall clock of the first exchange: one continuous
    /// ruler, whatever the wall clock did afterwards.
    pub fn host_time(&self, device_ticks: i64) -> f64 {
        self.mono_time(device_ticks) + self.frame_offset
    }

    /// (wall clock, monotonic) for every exchange.
    pub fn anchors(&self) -> Vec<(f64, f64)> {
        self.syncs.iter().map(|s| (s.host, s.mono)).collect()
    }

    fn fit(&mut self) {
        let n = self.syncs.len();
        if n == 0 {
            return;
        }
        let mut first = self.syncs[0];
        for s in &self.syncs[1..] {
            if s.mono < first.mono {
                first = *s;
            }
        }
        let (mut skew, mut offset, pinned);
        if n == 1 {
            skew = 1.0;
            offset = self.syncs[0].mono - self.syncs[0].device;
            pinned = Some("one exchange: a point has no slope".to_string());
        } else {
            let ds: Vec<f64> = self.syncs.iter().map(|s| s.device).collect();
            let hs: Vec<f64> = self.syncs.iter().map(|s| s.mono).collect();
            let (dbar, hbar) = (fsum(&ds) / n as f64, fsum(&hs) / n as f64);
            let sdd = fsum(&ds.iter().map(|d| (d - dbar) * (d - dbar)).collect::<Vec<_>>());
            let sdh = fsum(&ds.iter().zip(&hs).map(|(d, h)| (d - dbar) * (h - hbar)).collect::<Vec<_>>());
            skew = if sdd > 0.0 { sdh / sdd } else { 1.0 };
            offset = hbar - skew * dbar;
            pinned = self.reject(skew, offset, sdd, n);
            if pinned.is_some() {
                skew = 1.0;
                offset = hbar - dbar;
            }
        }
        // What the exchanges say, always. What the samples are dated with,
        // only while no stream is running.
        self.learned = Some(Learned { skew, offset, pinned: pinned.clone() });
        if self.frozen() {
            return;
        }
        self.frame_mono = Some(first.mono);
        self.frame_offset = first.host - first.mono;
        self.skew = skew;
        self.offset = offset;
        self.pinned = pinned;
    }

    /// Why a slope cannot be believed: past what two crystals can differ
    /// by, or not measured well enough to beat assuming none.
    fn reject(&self, skew: f64, offset: f64, sdd: f64, n: usize) -> Option<String> {
        if sdd <= 0.0 {
            return Some("every exchange landed on the same device time".into());
        }
        let err_ppm = (skew - 1.0).abs() * 1e6;
        if err_ppm > MAX_SKEW_PPM {
            return Some(format!(
                "fitted {} ppm from unity, past the {} ppm two oscillators can differ by",
                grouped(err_ppm),
                grouped(MAX_SKEW_PPM)
            ));
        }
        if n <= 2 {
            return Some("two exchanges: a slope with no residual to check it".into());
        }
        let resid: Vec<f64> = self.syncs.iter().map(|s| s.mono - (skew * s.device + offset)).collect();
        let sse = fsum(&resid.iter().map(|r| r * r).collect::<Vec<_>>());
        let se_ppm = ((sse / (n - 2) as f64) / sdd).sqrt() * 1e6;
        if se_ppm > MAX_SKEW_STDERR_PPM {
            let hi = self.syncs.iter().map(|s| s.device).fold(f64::NEG_INFINITY, f64::max);
            let lo = self.syncs.iter().map(|s| s.device).fold(f64::INFINITY, f64::min);
            return Some(format!(
                "slope {} +- {} ppm over a {} s baseline: not measured well enough to beat assuming 1.0",
                grouped(err_ppm),
                grouped(se_ppm),
                pyfmt::fixed(hi - lo, 2)
            ));
        }
        None
    }

    /// The clock record a recording carries: how far `host_time` can be
    /// trusted, the anchors that align events afterwards, any step of the
    /// wall clock, and what the exchanges said while the map was held.
    pub fn quality(&self) -> Value {
        self.record()
    }

    pub(crate) fn record(&self) -> Value {
        let n = self.syncs.len();
        let anchors = self.anchors();
        let steps = find_steps(&anchors);
        let opt_str = |s: &Option<String>| s.clone().map_or(Value::Null, Value::Str);
        let stepped = !matches!(&steps, Value::Array(a) if a.is_empty());
        let mut out = Value::obj([
            ("n_syncs", Value::int(n as i64)),
            ("skew", Value::Float(self.skew)),
            ("offset", Value::Float(self.offset)),
            ("skew_pinned", Value::Bool(self.pinned.is_some())),
            ("skew_pinned_why", opt_str(&self.pinned)),
            ("residual_ms", Value::Null),
            ("max_half_rtt_ms", Value::Null),
            ("span_s", Value::Null),
            ("fitted_against", Value::str("monotonic")),
            (
                "anchors",
                Value::Array(
                    anchors.iter().map(|(t, m)| Value::Array(vec![Value::Float(*t), Value::Float(*m)])).collect(),
                ),
            ),
            ("frame_mono", self.frame_mono.map_or(Value::Null, Value::Float)),
            ("wall_clock_steps", steps),
            ("wall_clock_stepped", Value::Bool(stepped)),
        ]);
        let mut span = None;
        if n > 0 {
            let rtt = self.syncs.iter().map(|s| s.rtt).fold(f64::NEG_INFINITY, f64::max);
            out.set("max_half_rtt_ms", Value::Float(rtt / 2.0 * 1e3));
        }
        if n >= 2 {
            let s = self.syncs[n - 1].device - self.syncs[0].device;
            span = Some(s);
            out.set("span_s", Value::Float(s));
            let errs: Vec<f64> = self.syncs.iter().map(|s| s.mono - (self.skew * s.device + self.offset)).collect();
            out.set(
                "residual_ms",
                Value::Float((fsum(&errs.iter().map(|e| e * e).collect::<Vec<_>>()) / n as f64).sqrt() * 1e3),
            );
        }
        let learned = self.learned.as_ref();
        out.set("skew_measured", learned.map_or(Value::Null, |l| Value::Float(l.skew)));
        out.set("skew_measured_pinned_why", learned.map_or(Value::Null, |l| opt_str(&l.pinned)));
        out.set("map_frozen_at_stream_start", Value::Bool(self.frozen()));
        if let (Some(l), Some(s)) = (learned, span) {
            if l.skew != 0.0 && s != 0.0 {
                out.set("uncorrected_drift_s", Value::Float((l.skew - self.skew) * s));
            }
        }
        out
    }
}

/// Steps of the wall clock between consecutive anchors, in monotonic order:
/// where each is bracketed and how far it jumped.
pub(crate) fn find_steps(anchors: &[(f64, f64)]) -> Value {
    let mut a = anchors.to_vec();
    a.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = Vec::new();
    for w in a.windows(2) {
        let ((t0, m0), (t1, m1)) = (w[0], w[1]);
        let drift = (t1 - t0) - (m1 - m0);
        let tol = STEP_FLOOR_S + SLEW_PPM * (m1 - m0).abs();
        if drift.abs() > tol {
            out.push(Value::obj([
                ("after_mono", Value::Float(m0)),
                ("before_mono", Value::Float(m1)),
                ("jump_s", Value::Float(drift)),
                ("tolerance_s", Value::Float(tol)),
                ("realtime_before", Value::Float(t0)),
                ("realtime_after", Value::Float(t1)),
            ]));
        }
    }
    Value::Array(out)
}

/// `f"{x:,.0f}"`: rounded to a whole number as Python rounds, with a comma
/// between each three digits.
fn grouped(x: f64) -> String {
    let s = pyfmt::fixed(x, 0);
    let (sign, digits) = match s.strip_prefix('-') {
        Some(d) => ("-", d),
        None => ("", s.as_str()),
    };
    if !digits.bytes().all(|b| b.is_ascii_digit()) {
        return s;
    }
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    format!("{sign}{out}")
}

/// `math.fsum`: the correctly rounded sum, by Python's own partials, so a
/// sum here is the same bits as the library's.
pub(crate) fn fsum(xs: &[f64]) -> f64 {
    let mut p: Vec<f64> = Vec::new();
    let mut special = 0.0f64;
    let mut inf = 0.0f64;
    for &x0 in xs {
        let mut x = x0;
        let mut i = 0;
        for j in 0..p.len() {
            let mut y = p[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                p[i] = lo;
                i += 1;
            }
            x = hi;
        }
        p.truncate(i);
        if x != 0.0 {
            if !x.is_finite() {
                if x0.is_infinite() {
                    inf += x0;
                }
                special += x0;
                p.clear();
            } else {
                p.push(x);
            }
        }
    }
    if special != 0.0 {
        return if inf.is_nan() { f64::NAN } else { special };
    }
    let mut hi = 0.0;
    let mut n = p.len();
    if n > 0 {
        n -= 1;
        hi = p[n];
        let mut lo = 0.0;
        while n > 0 {
            let x = hi;
            n -= 1;
            let y = p[n];
            hi = x + y;
            let yr = hi - x;
            lo = y - yr;
            if lo != 0.0 {
                break;
            }
        }
        if n > 0 && ((lo < 0.0 && p[n - 1] < 0.0) || (lo > 0.0 && p[n - 1] > 0.0)) {
            let y = lo * 2.0;
            let x = hi + y;
            let yr = x - hi;
            if y == yr {
                hi = x;
            }
        }
    }
    hi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fsum_rounds_as_python_does() {
        assert_eq!(fsum(&[1e-16, 1.0, 1e16]), 1.0000000000000002e16);
        assert_eq!(fsum(&[0.1; 10]), 1.0);
        assert_eq!(fsum(&[1e100, 1.0, -1e100, 1e-100, 1e50, -1.0, -1e50]), 1e-100);
        assert_eq!(fsum(&[]), 0.0);
    }

    #[test]
    fn grouped_puts_a_comma_between_each_three_digits() {
        assert_eq!(grouped(2990.6), "2,991");
        assert_eq!(grouped(1000.0), "1,000");
        assert_eq!(grouped(999.5), "1,000");
        assert_eq!(grouped(192.4), "192");
        assert_eq!(grouped(1234567.0), "1,234,567");
    }
}
