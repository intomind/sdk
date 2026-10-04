//! The processing chain a device runs on its signal.
//!
//! Filters are one class of the preprocessing the firmware hosts. A chain
//! is stages as data: the catalog names the kinds this firmware
//! implements, a chain is an ordered list of instances with parameters,
//! and every stage belongs to a class that says what it does to the
//! signal. Map stages keep the sample rate and the kind of signal;
//! representation changes and detectors are classes the contract
//! defines and this firmware does not yet implement, so the catalog does
//! not list them and a chain that names one is refused.
//!
//! The rules are pure functions over a chain and a rate, so the device
//! and a host can both say in advance whether a chain runs. The design is
//! the same arithmetic as the proven earlier firmware: cookbook biquads at
//! Butterworth Q, a notch given by its band's two edges, float carried
//! across the whole cascade and quantized once at the end.

#![cfg_attr(not(test), no_std)]
#![deny(unsafe_code)]
#![deny(missing_docs)]

pub use intomind_protocol::pipeline::{class, input_class, kind, CatalogEntry, Chain, Stage, MAX_STAGES};

/// Channels a runtime carries state for.
pub const CHANNELS: usize = 4;
/// Notch bands a chain may hold at once.
pub const MAX_NOTCHES: usize = 8;
/// The automatic low-pass corner, as a share of the sample rate.
pub const LOW_PASS_AUTO_SHARE: (u32, u32) = (2, 5);
/// A corner at or above this share of the Nyquist rate is refused.
pub const NYQUIST_SHARE_MAX: f32 = 0.9;
/// The factory default high-pass corner, in tenths of a hertz.
pub const DEFAULT_HIGH_PASS_DHZ: u16 = 5;
/// The factory default mains bands: fundamental, second and third harmonic
/// of both mains frequencies, four hertz wide, until a region is set.
pub const DEFAULT_MAINS_BANDS_DHZ: [(u16, u16); 6] =
    [(480, 520), (980, 1020), (1480, 1520), (580, 620), (1180, 1220), (1780, 1820)];

/// The kinds this firmware implements.
pub const CATALOG: [CatalogEntry; 3] = [
    CatalogEntry::new(kind::HIGH_PASS, class::MAP, 1, 1, b"highpass"),
    CatalogEntry::new(kind::LOW_PASS, class::MAP, 1, 1, b"lowpass\0"),
    CatalogEntry::new(kind::NOTCH, class::MAP, 2, MAX_NOTCHES as u8, b"notch\0\0\0"),
];

/// Why a chain cannot run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// A kind the catalog does not list.
    UnknownKind(u8),
    /// A stage carries the wrong number of parameters for its kind.
    ParamCount(u8),
    /// More instances of a kind than the catalog allows.
    TooMany(u8),
    /// A corner or band edge at or above nine tenths of Nyquist. Carries
    /// the kind and the offending value in tenths of a hertz.
    AboveNyquist(u8, u16),
    /// A corner of zero where the kind needs one.
    Zero(u8),
    /// A notch whose high edge is not above its low edge.
    NotABand,
    /// The high-pass corner at or above the low-pass corner.
    HighPassMeetsLowPass,
}

/// The automatic low-pass corner for a rate, in tenths of a hertz.
pub fn low_pass_auto_dhz(rate_sps: u32) -> u16 {
    (rate_sps * 10 * LOW_PASS_AUTO_SHARE.0 / LOW_PASS_AUTO_SHARE.1).min(u16::MAX as u32) as u16
}

/// The highest corner a rate can represent, in tenths of a hertz,
/// exclusive: a value at or above this is refused.
pub fn corner_limit_dhz(rate_sps: u32) -> u32 {
    // 0.9 × (rate / 2) × 10 = rate × 4.5
    rate_sps * 45 / 10
}

/// The effective low-pass corner of a stage, in tenths of a hertz.
pub fn low_pass_corner_dhz(corner: u16, rate_sps: u32) -> u16 {
    if corner == 0 { low_pass_auto_dhz(rate_sps) } else { corner }
}

/// Whether a chain can run at a rate, and why not if it cannot. The
/// device asks before installing a chain and before changing rate; a
/// host may ask before sending, to explain a refusal in words.
pub fn check(chain: &Chain, rate_sps: u32) -> Result<(), Refusal> {
    let limit = corner_limit_dhz(rate_sps);
    let mut counts = [0u8; 256];
    let mut hp: Option<u16> = None;
    let mut lp: Option<u16> = None;
    for s in chain.stages() {
        let Some(entry) = CATALOG.iter().find(|e| e.kind == s.kind) else {
            return Err(Refusal::UnknownKind(s.kind));
        };
        if s.n_params != entry.n_params {
            return Err(Refusal::ParamCount(s.kind));
        }
        counts[s.kind as usize] += 1;
        if counts[s.kind as usize] > entry.max_instances {
            return Err(Refusal::TooMany(s.kind));
        }
        let p = s.params();
        match s.kind {
            kind::HIGH_PASS => {
                if p[0] == 0 {
                    return Err(Refusal::Zero(s.kind));
                }
                if u32::from(p[0]) >= limit {
                    return Err(Refusal::AboveNyquist(s.kind, p[0]));
                }
                hp = Some(p[0]);
            }
            kind::LOW_PASS => {
                let c = low_pass_corner_dhz(p[0], rate_sps);
                if u32::from(c) >= limit {
                    return Err(Refusal::AboveNyquist(s.kind, c));
                }
                lp = Some(c);
            }
            kind::NOTCH => {
                if p[0] == 0 {
                    return Err(Refusal::Zero(s.kind));
                }
                if p[1] <= p[0] {
                    return Err(Refusal::NotABand);
                }
                if u32::from(p[1]) >= limit {
                    return Err(Refusal::AboveNyquist(s.kind, p[1]));
                }
            }
            _ => return Err(Refusal::UnknownKind(s.kind)),
        }
    }
    if let (Some(h), Some(l)) = (hp, lp) {
        if h >= l {
            return Err(Refusal::HighPassMeetsLowPass);
        }
    }
    Ok(())
}

/// The class of signal a chain produces, as an `input_class` bit. Every
/// kind this firmware implements keeps the time domain.
pub fn output_class(chain: &Chain) -> u8 {
    let representation = chain.stages().iter().any(|s| {
        CATALOG.iter().find(|e| e.kind == s.kind).map(|e| e.class == class::REPRESENTATION).unwrap_or(false)
    });
    if representation { input_class::REPRESENTATION } else { input_class::TIME_DOMAIN }
}

/// The device's default chain for a rate: the minimal high-pass, the
/// mains bands the rate can represent, and the automatic low-pass. It is
/// recomposed at every rate change while it is the chain in force.
pub fn default_for(rate_sps: u32) -> Chain {
    let limit = corner_limit_dhz(rate_sps);
    let mut c = Chain::NATURAL;
    let _ = c.push(Stage::one(kind::HIGH_PASS, DEFAULT_HIGH_PASS_DHZ));
    for (lo, hi) in DEFAULT_MAINS_BANDS_DHZ {
        if u32::from(hi) < limit {
            let _ = c.push(Stage::two(kind::NOTCH, lo, hi));
        }
    }
    let _ = c.push(Stage::one(kind::LOW_PASS, 0));
    c
}

/// The mains bands for a region, fundamental, second and third harmonic,
/// four hertz wide, in tenths of a hertz.
pub fn mains_bands_dhz(mains_hz: u16) -> [(u16, u16); 3] {
    let f = mains_hz * 10;
    [(f - 20, f + 20), (2 * f - 20, 2 * f + 20), (3 * f - 20, 3 * f + 20)]
}

/// One second-order section in double precision, for the high-pass.
///
/// A corner far below the rate puts the section's poles a hair inside the
/// unit circle, and there single precision leaks: the state holds
/// numbers the size of the electrode offset, a million counts and more,
/// each step rounds them at the eighth digit, and the near-unity feedback
/// multiplies that rounding by hundreds. Measured on the host: a 30 mV
/// offset left 70 counts of residual after ten seconds in single
/// precision and nothing in double. The part has no double-precision
/// hardware, so this costs software arithmetic, about five percent of
/// the core at a thousand samples per second for the one such stage a
/// chain can hold. The other stages see the offset already removed, or
/// sit far from the unit circle, and run in single precision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Biquad64 {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: [f64; CHANNELS],
    z2: [f64; CHANNELS],
}

/// The cookbook's terms for one corner, in single precision but without
/// the cancellation that spoils a corner far below the rate: the halves
/// `1 - cos w = 2 sin²(w/2)` and `1 + cos w = 2 cos²(w/2)` come from a
/// sine or cosine of a small angle, which single precision holds to its
/// full relative accuracy. Double-precision transcendentals would do the
/// same and cost several kilobytes of flash the part does not have.
struct Terms {
    alpha: f32,
    cos_w: f32,
    one_minus_cos: f32,
    one_plus_cos: f32,
}

/// Sine and cosine of a half angle, which is never past 0.45 pi here
/// because every corner is refused at nine tenths of Nyquist. Plain
/// Taylor series to the thirteenth and fourteenth powers, evaluated by
/// Horner's rule: past the last term the remainder is below 1e-10, under
/// single precision's own resolution. The library's sine and cosine cost
/// three kilobytes of flash for an argument reduction this never needs.
fn sin_cos_half(x: f32) -> (f32, f32) {
    let x2 = x * x;
    let sin = x * (1.0 - x2 / 6.0 * (1.0 - x2 / 20.0 * (1.0 - x2 / 42.0 * (1.0 - x2 / 72.0 * (1.0 - x2 / 110.0 * (1.0 - x2 / 156.0))))));
    let cos = 1.0 - x2 / 2.0 * (1.0 - x2 / 12.0 * (1.0 - x2 / 30.0 * (1.0 - x2 / 56.0 * (1.0 - x2 / 90.0 * (1.0 - x2 / 132.0 * (1.0 - x2 / 182.0))))));
    (sin, cos)
}

fn terms(f_hz: f32, fs_hz: f32, q: f32) -> Terms {
    let half = core::f32::consts::PI * f_hz / fs_hz;
    let (s, c) = sin_cos_half(half);
    let sin_w = 2.0 * s * c;
    Terms { alpha: sin_w / (2.0 * q), cos_w: c * c - s * s, one_minus_cos: 2.0 * s * s, one_plus_cos: 2.0 * c * c }
}

impl Biquad64 {
    /// Cookbook high-pass at Butterworth Q. The middle coefficient is
    /// held at exactly minus twice the outer ones, so the numerator sums
    /// to exactly zero at zero frequency and an offset is rejected rather
    /// than leaked.
    pub fn high_pass(fc_hz: f32, fs_hz: f32) -> Biquad64 {
        let t = terms(fc_hz, fs_hz, core::f32::consts::FRAC_1_SQRT_2);
        let a0 = 1.0 + t.alpha;
        let b0 = f64::from((t.one_plus_cos / 2.0) / a0);
        Biquad64 {
            b0,
            b1: -2.0 * b0,
            b2: b0,
            a1: f64::from((-2.0 * t.cos_w) / a0),
            a2: f64::from((1.0 - t.alpha) / a0),
            z1: [0.0; CHANNELS],
            z2: [0.0; CHANNELS],
        }
    }

    /// One sample of one channel through this section, in double precision.
    #[inline]
    pub fn step(&mut self, ch: usize, x: f32) -> f32 {
        let x = f64::from(x);
        let y = self.b0 * x + self.z1[ch];
        self.z1[ch] = self.b1 * x - self.a1 * y + self.z2[ch];
        self.z2[ch] = self.b2 * x - self.a2 * y;
        y as f32
    }

    /// Forget this section's state.
    pub fn reset(&mut self) {
        self.z1 = [0.0; CHANNELS];
        self.z2 = [0.0; CHANNELS];
    }
}

/// One second-order section, direct form II transposed, with state per
/// channel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: [f32; CHANNELS],
    z2: [f32; CHANNELS],
}

impl Biquad {
    const IDENTITY: Biquad = Biquad { b0: 1.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0, z1: [0.0; CHANNELS], z2: [0.0; CHANNELS] };

    fn from_rbj(b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32) -> Biquad {
        Biquad { b0: b0 / a0, b1: b1 / a0, b2: b2 / a0, a1: a1 / a0, a2: a2 / a0, z1: [0.0; CHANNELS], z2: [0.0; CHANNELS] }
    }

    /// Cookbook low-pass at Butterworth Q.
    pub fn low_pass(fc_hz: f32, fs_hz: f32) -> Biquad {
        let t = terms(fc_hz, fs_hz, core::f32::consts::FRAC_1_SQRT_2);
        Biquad::from_rbj(t.one_minus_cos / 2.0, t.one_minus_cos, t.one_minus_cos / 2.0, 1.0 + t.alpha, -2.0 * t.cos_w, 1.0 - t.alpha)
    }

    /// Cookbook notch for the band between two edges: the null sits at the
    /// middle of the band and Q follows from the width. The middle, not the
    /// geometric mean: a mains band is written around its mains frequency,
    /// and the geometric mean of 58 and 62 Hz put the null at 59.97 Hz,
    /// which left 60 Hz mains at 60.03 Hz down only 32 dB.
    pub fn notch(lo_hz: f32, hi_hz: f32, fs_hz: f32) -> Biquad {
        let f0 = (lo_hz + hi_hz) / 2.0;
        let q = f0 / (hi_hz - lo_hz);
        let t = terms(f0, fs_hz, q);
        Biquad::from_rbj(1.0, -2.0 * t.cos_w, 1.0, 1.0 + t.alpha, -2.0 * t.cos_w, 1.0 - t.alpha)
    }

    /// One sample of one channel through this section.
    #[inline]
    pub fn step(&mut self, ch: usize, x: f32) -> f32 {
        let y = self.b0 * x + self.z1[ch];
        self.z1[ch] = self.b1 * x - self.a1 * y + self.z2[ch];
        self.z2[ch] = self.b2 * x - self.a2 * y;
        y
    }

    /// Forget this section's state.
    pub fn reset(&mut self) {
        self.z1 = [0.0; CHANNELS];
        self.z2 = [0.0; CHANNELS];
    }
}

/// One designed stage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Section {
    /// One biquad section.
    Single(Biquad),
    /// One double-precision section, for the high-pass.
    Double(Biquad64),
    /// Two identical sections in series, for a notch. One narrow section
    /// loses depth as fast as the mains frequency drifts off its null; two
    /// double it in decibels: 60.03 Hz mains in a 58 to 62 Hz band goes
    /// from 36 dB down to 71 dB down at 500 samples a second.
    Pair(Biquad, Biquad),
}

impl Section {
    #[inline]
    fn step(&mut self, ch: usize, x: f32) -> f32 {
        match self {
            Section::Single(b) => b.step(ch, x),
            Section::Double(b) => b.step(ch, x),
            Section::Pair(a, b) => b.step(ch, a.step(ch, x)),
        }
    }

    fn reset(&mut self) {
        match self {
            Section::Single(b) => b.reset(),
            Section::Double(b) => b.reset(),
            Section::Pair(a, b) => {
                a.reset();
                b.reset();
            }
        }
    }
}

/// A chain designed for one rate, ready to run. Stages run in the order
/// the chain lists them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Runtime {
    stages: [Section; MAX_STAGES],
    len: usize,
}

impl Runtime {
    /// The natural signal: nothing runs.
    pub const NATURAL: Runtime = Runtime { stages: [Section::Single(Biquad::IDENTITY); MAX_STAGES], len: 0 };

    /// Design every stage for the rate. A chain that fails [`check`] is
    /// refused here with the same reason, so a runtime always matches a
    /// chain that was checked.
    pub fn design(chain: &Chain, rate_sps: u32) -> Result<Runtime, Refusal> {
        check(chain, rate_sps)?;
        let fs = rate_sps as f32;
        let mut rt = Runtime::NATURAL;
        for s in chain.stages() {
            let p = s.params();
            rt.stages[rt.len] = match s.kind {
                kind::HIGH_PASS => Section::Double(Biquad64::high_pass(p[0] as f32 / 10.0, fs)),
                kind::LOW_PASS => Section::Single(Biquad::low_pass(low_pass_corner_dhz(p[0], rate_sps) as f32 / 10.0, fs)),
                kind::NOTCH => {
                    let n = Biquad::notch(p[0] as f32 / 10.0, p[1] as f32 / 10.0, fs);
                    Section::Pair(n, n)
                }
                other => return Err(Refusal::UnknownKind(other)),
            };
            rt.len += 1;
        }
        Ok(rt)
    }

    /// Whether this runtime is the natural signal: nothing runs.
    pub fn is_natural(&self) -> bool {
        self.len == 0
    }

    /// Stages in this runtime.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this runtime holds no stages.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// One sample of one channel through the whole cascade.
    #[inline]
    pub fn step(&mut self, ch: usize, x: f32) -> f32 {
        let mut y = x;
        for b in self.stages[..self.len].iter_mut() {
            y = b.step(ch, y);
        }
        y
    }

    /// Forget the past. Called at every stream start, so a stream begins
    /// from a known state and a window never spans two of them.
    pub fn reset(&mut self) {
        for b in self.stages[..self.len].iter_mut() {
            b.reset();
        }
    }
}

/// The largest magnitude a converter count can hold: 24 bits, signed.
pub const COUNT_MAX: f32 = 8_388_607.0;

/// A processed value back into the converter's own integer domain, rounded
/// and held inside 24 bits, so the stream's scaling rule is unchanged.
#[inline]
pub fn to_count(y: f32) -> i32 {
    let c = if y > COUNT_MAX {
        COUNT_MAX
    } else if y < -COUNT_MAX - 1.0 {
        -COUNT_MAX - 1.0
    } else {
        y
    };
    libm::roundf(c) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(rt: &mut Runtime, ch: usize, input: impl Iterator<Item = f32>) -> Vec<f32> {
        input.map(|x| rt.step(ch, x)).collect()
    }

    fn sine(f: f32, fs: f32, n: usize) -> impl Iterator<Item = f32> {
        (0..n).map(move |i| libm::sinf(2.0 * core::f32::consts::PI * f * i as f32 / fs))
    }

    fn rms(v: &[f32]) -> f32 {
        libm::sqrtf(v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32)
    }

    #[test]
    fn the_default_chain_fits_every_rate_and_recomposes_with_it() {
        for rate in [250u32, 500, 1000] {
            let c = default_for(rate);
            assert_eq!(check(&c, rate), Ok(()), "{rate}");
            assert_eq!(c.stages()[0], Stage::one(kind::HIGH_PASS, 5));
            assert_eq!(c.stages()[c.len as usize - 1], Stage::one(kind::LOW_PASS, 0));
        }
        // At 250 samples per second the 120, 150 and 180 Hz bands are above
        // the limit; at 500 and above every band fits.
        assert_eq!(default_for(250).len, 1 + 3 + 1, "the 120, 150 and 180 Hz bands are above what 250 can represent");
        assert_eq!(default_for(500).len, 1 + 6 + 1);
        assert_eq!(default_for(1000).len, 1 + 6 + 1);
        assert_eq!(low_pass_auto_dhz(250), 1000);
        assert_eq!(low_pass_auto_dhz(500), 2000);
        assert_eq!(low_pass_auto_dhz(1000), 4000);
        assert_eq!(mains_bands_dhz(50), [(480, 520), (980, 1020), (1480, 1520)]);
        assert_eq!(mains_bands_dhz(60), [(580, 620), (1180, 1220), (1780, 1820)]);
    }

    #[test]
    fn the_rules_refuse_exactly_what_the_contract_says() {
        let mut c = Chain::NATURAL;
        c.push(Stage::one(9, 5)).unwrap();
        assert_eq!(check(&c, 500), Err(Refusal::UnknownKind(9)));

        let mut c = Chain::NATURAL;
        c.push(Stage::two(kind::HIGH_PASS, 5, 0)).unwrap();
        assert_eq!(check(&c, 500), Err(Refusal::ParamCount(kind::HIGH_PASS)));

        let mut c = Chain::NATURAL;
        c.push(Stage::one(kind::HIGH_PASS, 5)).unwrap();
        c.push(Stage::one(kind::HIGH_PASS, 10)).unwrap();
        assert_eq!(check(&c, 500), Err(Refusal::TooMany(kind::HIGH_PASS)));

        let mut c = Chain::NATURAL;
        for _ in 0..=MAX_NOTCHES {
            c.push(Stage::two(kind::NOTCH, 480, 520)).unwrap();
        }
        assert_eq!(check(&c, 500), Err(Refusal::TooMany(kind::NOTCH)));

        // 0.9 × Nyquist at 500 is 225 Hz: 2250 tenths is refused, 2249 runs.
        let mut c = Chain::NATURAL;
        c.push(Stage::one(kind::LOW_PASS, 2250)).unwrap();
        assert_eq!(check(&c, 500), Err(Refusal::AboveNyquist(kind::LOW_PASS, 2250)));
        let mut c = Chain::NATURAL;
        c.push(Stage::one(kind::LOW_PASS, 2249)).unwrap();
        assert_eq!(check(&c, 500), Ok(()));
        // The same corner is fine at 1000 and refused at 250.
        assert_eq!(check(&c, 1000), Ok(()));
        assert_eq!(check(&c, 250), Err(Refusal::AboveNyquist(kind::LOW_PASS, 2249)));

        let mut c = Chain::NATURAL;
        c.push(Stage::one(kind::HIGH_PASS, 0)).unwrap();
        assert_eq!(check(&c, 500), Err(Refusal::Zero(kind::HIGH_PASS)));

        let mut c = Chain::NATURAL;
        c.push(Stage::two(kind::NOTCH, 520, 480)).unwrap();
        assert_eq!(check(&c, 500), Err(Refusal::NotABand));
        let mut c = Chain::NATURAL;
        c.push(Stage::two(kind::NOTCH, 0, 480)).unwrap();
        assert_eq!(check(&c, 500), Err(Refusal::Zero(kind::NOTCH)));
        let mut c = Chain::NATURAL;
        c.push(Stage::two(kind::NOTCH, 1180, 1220)).unwrap();
        assert_eq!(check(&c, 250), Err(Refusal::AboveNyquist(kind::NOTCH, 1220)));

        let mut c = Chain::NATURAL;
        c.push(Stage::one(kind::HIGH_PASS, 300)).unwrap();
        c.push(Stage::one(kind::LOW_PASS, 300)).unwrap();
        assert_eq!(check(&c, 500), Err(Refusal::HighPassMeetsLowPass));
        // The automatic low-pass counts as its effective corner.
        let mut c = Chain::NATURAL;
        c.push(Stage::one(kind::HIGH_PASS, 2000)).unwrap();
        c.push(Stage::one(kind::LOW_PASS, 0)).unwrap();
        assert_eq!(check(&c, 500), Err(Refusal::HighPassMeetsLowPass));
        assert_eq!(check(&c, 1000), Ok(()));

        assert_eq!(check(&Chain::NATURAL, 500), Ok(()));
        assert_eq!(output_class(&Chain::NATURAL), input_class::TIME_DOMAIN);
        assert_eq!(output_class(&default_for(500)), input_class::TIME_DOMAIN);
    }

    #[test]
    fn the_design_matches_the_check_and_the_natural_runtime_is_identity() {
        let mut c = Chain::NATURAL;
        c.push(Stage::one(kind::LOW_PASS, 2250)).unwrap();
        assert_eq!(Runtime::design(&c, 500).unwrap_err(), Refusal::AboveNyquist(kind::LOW_PASS, 2250));
        let mut rt = Runtime::NATURAL;
        assert!(rt.is_natural());
        for x in [0.0f32, 1.0, -123.5, 8e6] {
            assert_eq!(rt.step(0, x), x);
        }
    }

    #[test]
    fn a_high_pass_removes_an_offset_and_a_low_pass_keeps_slow_signal() {
        let fs = 500.0;
        let mut c = Chain::NATURAL;
        c.push(Stage::one(kind::HIGH_PASS, 5)).unwrap();
        let mut rt = Runtime::design(&c, 500).unwrap();
        // A 30 mV electrode offset in counts at gain 24 is about 1.3
        // million. After ten seconds the output has settled to nothing.
        let y = run(&mut rt, 0, core::iter::repeat(1_300_000.0f32).take(5000));
        assert!(y[0].abs() > 1_000_000.0, "the step passes at first: {}", y[0]);
        assert!(y[4999].abs() < 0.5, "and is gone after ten seconds: {}", y[4999]);
        // Channels are independent: channel 3 never saw anything.
        assert_eq!(rt.step(3, 0.0), 0.0);

        let mut c = Chain::NATURAL;
        c.push(Stage::one(kind::LOW_PASS, 0)).unwrap();
        let mut rt = Runtime::design(&c, 500).unwrap();
        // 10 Hz passes a 200 Hz corner untouched; 240 Hz is well down.
        let slow = run(&mut rt, 1, sine(10.0, fs, 5000));
        let a_slow = rms(&slow[2500..]);
        rt.reset();
        let fast = run(&mut rt, 1, sine(240.0, fs, 5000));
        let a_fast = rms(&fast[2500..]);
        let a_in = rms(&sine(10.0, fs, 5000).collect::<Vec<_>>()[2500..]);
        assert!((a_slow / a_in - 1.0).abs() < 0.02, "10 Hz through a 200 Hz corner: {}", a_slow / a_in);
        assert!(a_fast / a_in < 0.5, "240 Hz through a 200 Hz corner: {}", a_fast / a_in);
    }

    #[test]
    fn a_notch_takes_out_its_band_and_leaves_its_neighbors() {
        let fs = 500.0;
        let mut c = Chain::NATURAL;
        c.push(Stage::two(kind::NOTCH, 580, 620)).unwrap();
        let mut rt = Runtime::design(&c, 500).unwrap();
        let a_in = rms(&sine(60.0, fs, 5000).collect::<Vec<_>>()[2500..]);
        let at_60 = rms(&run(&mut rt, 2, sine(60.0, fs, 5000))[2500..]);
        rt.reset();
        let at_45 = rms(&run(&mut rt, 2, sine(45.0, fs, 5000))[2500..]);
        rt.reset();
        let at_80 = rms(&run(&mut rt, 2, sine(80.0, fs, 5000))[2500..]);
        // The null sits at the middle of the band, so 60.00 Hz is gone.
        assert!(at_60 / a_in < 1e-4, "60 Hz in a 58 to 62 notch: {}", at_60 / a_in);
        // Mains drifts off its nominal frequency, to 60.03 Hz for example.
        // Two sections keep it more than 60 dB down,
        // where one section at the geometric mean left it 32 dB down.
        rt.reset();
        let at_drift = rms(&run(&mut rt, 2, sine(60.03, fs, 20000))[10000..]);
        assert!(at_drift / a_in < 1e-3, "60.03 Hz mains: {}", at_drift / a_in);
        rt.reset();
        let at_56 = rms(&run(&mut rt, 2, sine(56.0, fs, 5000))[2500..]);
        assert!(at_56 / a_in > 0.8, "56 Hz beside it: {}", at_56 / a_in);
        assert!(at_45 / a_in > 0.9, "45 Hz beside it: {}", at_45 / a_in);
        assert!(at_80 / a_in > 0.9, "80 Hz beside it: {}", at_80 / a_in);
    }

    #[test]
    fn the_default_chain_at_500_leaves_a_10_hz_signal_and_removes_mains_and_offset() {
        let fs = 500.0;
        let mut rt = Runtime::design(&default_for(500), 500).unwrap();
        let x: Vec<f32> = sine(10.0, fs, 6000)
            .zip(sine(50.0, fs, 6000))
            .zip(sine(60.0, fs, 6000))
            .map(|((a, b), c)| 200_000.0 + 1000.0 * a + 1000.0 * b + 1000.0 * c)
            .collect();
        let y = run(&mut rt, 0, x.iter().copied());
        let tail = &y[3000..];
        let mean = tail.iter().sum::<f32>() / tail.len() as f32;
        assert!(mean.abs() < 50.0, "the offset is gone: {mean}");
        // What is left is the 10 Hz component, near its 1000 amplitude.
        let a = rms(tail) * core::f32::consts::SQRT_2;
        assert!((a / 1000.0 - 1.0).abs() < 0.05, "10 Hz survives at {a}");
    }

    #[test]
    fn the_series_matches_the_library_over_every_angle_a_corner_can_take() {
        for i in 0..=1000 {
            let x = 0.45 * core::f32::consts::PI * i as f32 / 1000.0;
            let (s, c) = sin_cos_half(x);
            assert!((s - libm::sinf(x)).abs() <= 2e-7 * libm::sinf(x).abs().max(1e-3), "sin {x}: {s} vs {}", libm::sinf(x));
            assert!((c - libm::cosf(x)).abs() <= 2e-7, "cos {x}: {c} vs {}", libm::cosf(x));
        }
    }

    #[test]
    fn processed_values_go_back_into_24_bits() {
        assert_eq!(to_count(0.4), 0);
        assert_eq!(to_count(0.6), 1);
        assert_eq!(to_count(-2.5), -3);
        assert_eq!(to_count(1e9), 8_388_607);
        assert_eq!(to_count(-1e9), -8_388_608);
    }
}
