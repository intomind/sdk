//! Processing, new in 1.1: the chain a device runs on its signal, as the
//! wire carries it, and the catalog of stage kinds a device offers.
//!
//! A chain is an ordered list of stages. Each stage names a kind from the
//! device's catalog and carries up to [`MAX_PARAMS`] sixteen-bit parameters
//! whose units the kind defines. The empty chain is the natural signal.
//! Nothing here designs or runs a filter; that is the device's pipeline.
//! This is the contract's shape of a chain, and it is what a recording
//! stores to say what produced its samples.

use crate::Error;

/// Stage classes. A map stage keeps the rate and the kind of signal. A
/// representation change replaces the signal with something else, at its
/// own rate. A detector adds annotations and passes the signal on.
pub mod class {
    /// Keeps the rate and the kind of signal.
    pub const MAP: u8 = 0;
    /// Replaces the signal with something else, at its own rate.
    pub const REPRESENTATION: u8 = 1;
    /// Adds annotations and passes the signal on.
    pub const DETECTOR: u8 = 2;
}

/// Stage kinds this contract defines. A device's catalog says which of
/// them it implements, and a kind outside the catalog is refused.
pub mod kind {
    /// One parameter: the corner in tenths of a hertz, at least 1.
    pub const HIGH_PASS: u8 = 1;
    /// One parameter: the corner in tenths of a hertz, or 0 for the
    /// automatic corner at four tenths of the sample rate.
    pub const LOW_PASS: u8 = 2;
    /// Two parameters: the band's low and high edges in tenths of a hertz.
    pub const NOTCH: u8 = 3;
}

/// Where the model's input comes from.
pub mod input_source {
    /// The stream as it is: the chain's output, or the natural signal when
    /// the chain is empty.
    pub const STREAM: u8 = 0;
    /// The natural signal, whatever the stream carries.
    pub const NATURAL: u8 = 1;
    /// A chain of the model's own, run beside the stream's.
    pub const OWN_CHAIN: u8 = 2;
    /// 1.3: the synthetic signal, while the device generates it. Reported,
    /// never requested: a device in synthetic mode marks every embedding
    /// and prediction with it, whatever source was asked for.
    pub const SYNTHETIC: u8 = 3;
}

/// Signal classes a model declares it can take, as bits.
pub mod input_class {
    /// The signal as samples over time.
    pub const TIME_DOMAIN: u8 = 1 << 0;
    /// The signal after a representation change.
    pub const REPRESENTATION: u8 = 1 << 1;
}

/// GET_PIPELINE `origin` values.
pub mod origin {
    /// The device's own default for the current rate.
    pub const DEFAULT: u8 = 0;
    /// Set by a host.
    pub const HOST: u8 = 1;
}

/// SET_BIAS modes.
pub mod bias_mode {
    /// The bias amplifier off.
    pub const OFF: u8 = 0;
    /// The bias amplifier on, driving its electrode.
    pub const ON: u8 = 1;
    /// The amplifier on with no electrode routed into it, a diagnostic.
    pub const LOOP_OPEN: u8 = 2;
}

/// Stages a chain may hold.
pub const MAX_STAGES: usize = 12;
/// Parameters a stage may hold.
pub const MAX_PARAMS: usize = 4;
/// The longest encoding of a chain.
pub const MAX_ENCODED_LEN: usize = 1 + MAX_STAGES * (2 + 2 * MAX_PARAMS);

/// One stage in a chain: a kind and its parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Stage {
    /// The stage's kind, from the device's catalog.
    pub kind: u8,
    /// Parameters this stage carries.
    pub n_params: u8,
    /// The stage's parameters, in the units its kind defines. Only the
    /// first `n_params` are meaningful.
    pub params: [u16; MAX_PARAMS],
}

impl Stage {
    /// A stage with one parameter.
    pub const fn one(kind: u8, p0: u16) -> Stage {
        Stage { kind, n_params: 1, params: [p0, 0, 0, 0] }
    }

    /// A stage with two parameters.
    pub const fn two(kind: u8, p0: u16, p1: u16) -> Stage {
        Stage { kind, n_params: 2, params: [p0, p1, 0, 0] }
    }

    /// This stage's parameters, `n_params` of them.
    pub fn params(&self) -> &[u16] {
        &self.params[..(self.n_params as usize).min(MAX_PARAMS)]
    }
}

/// An ordered chain of stages. Empty is the natural signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chain {
    /// The chain's stages. Only the first `len` are meaningful.
    pub stages: [Stage; MAX_STAGES],
    /// Stages in the chain. Zero is the natural signal.
    pub len: u8,
}

impl Default for Chain {
    fn default() -> Self {
        Chain::NATURAL
    }
}

impl Chain {
    /// The empty chain: the natural signal.
    pub const NATURAL: Chain = Chain { stages: [Stage { kind: 0, n_params: 0, params: [0; MAX_PARAMS] }; MAX_STAGES], len: 0 };

    /// Whether this chain is empty, the natural signal.
    pub fn is_natural(&self) -> bool {
        self.len == 0
    }

    /// This chain's stages, in order.
    pub fn stages(&self) -> &[Stage] {
        &self.stages[..(self.len as usize).min(MAX_STAGES)]
    }

    /// Append a stage. Refused once the chain is full.
    pub fn push(&mut self, s: Stage) -> Result<(), Error> {
        if s.kind == 0 || s.n_params as usize > MAX_PARAMS {
            return Err(Error::Invalid);
        }
        let i = self.len as usize;
        if i >= MAX_STAGES {
            return Err(Error::NoRoom);
        }
        self.stages[i] = s;
        self.len += 1;
        Ok(())
    }

    /// This chain's encoded length, in bytes.
    pub fn encoded_len(&self) -> usize {
        1 + self.stages().iter().map(|s| 2 + 2 * s.n_params as usize).sum::<usize>()
    }

    /// `u8 n_stages`, then per stage `u8 kind, u8 n_params, u16[n_params]`.
    pub fn encode(&self, out: &mut [u8]) -> Result<usize, Error> {
        let n = self.encoded_len();
        if out.len() < n {
            return Err(Error::NoRoom);
        }
        out[0] = self.len;
        let mut at = 1;
        for s in self.stages() {
            out[at] = s.kind;
            out[at + 1] = s.n_params;
            at += 2;
            for p in s.params() {
                out[at..at + 2].copy_from_slice(&p.to_le_bytes());
                at += 2;
            }
        }
        Ok(n)
    }

    /// Parse a descriptor that is the whole slice.
    pub fn parse(b: &[u8]) -> Result<Chain, Error> {
        let (c, rest) = Self::parse_prefix(b)?;
        if !rest.is_empty() {
            return Err(Error::Invalid);
        }
        Ok(c)
    }

    /// Parse a descriptor from the start of a slice and return what follows.
    pub fn parse_prefix(b: &[u8]) -> Result<(Chain, &[u8]), Error> {
        let (&n, mut rest) = b.split_first().ok_or(Error::Truncated)?;
        if n as usize > MAX_STAGES {
            return Err(Error::Invalid);
        }
        let mut c = Chain::NATURAL;
        for _ in 0..n {
            if rest.len() < 2 {
                return Err(Error::Truncated);
            }
            let (kind, n_params) = (rest[0], rest[1]);
            if kind == 0 || n_params as usize > MAX_PARAMS {
                return Err(Error::Invalid);
            }
            rest = &rest[2..];
            let mut s = Stage { kind, n_params, params: [0; MAX_PARAMS] };
            for i in 0..n_params as usize {
                if rest.len() < 2 {
                    return Err(Error::Truncated);
                }
                s.params[i] = u16::from_le_bytes([rest[0], rest[1]]);
                rest = &rest[2..];
            }
            c.push(s)?;
        }
        Ok((c, rest))
    }
}

/// GET_PIPELINE payload: `u8 origin`, then the chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineState {
    /// Whether the chain in force is the device's default or a host's, an
    /// `origin` value.
    pub origin: u8,
    /// The chain in force.
    pub chain: Chain,
}

impl PipelineState {
    /// Encode into `out`.
    pub fn encode(&self, out: &mut [u8]) -> Result<usize, Error> {
        if out.is_empty() {
            return Err(Error::NoRoom);
        }
        out[0] = self.origin;
        Ok(1 + self.chain.encode(&mut out[1..])?)
    }

    /// Decode a GET_PIPELINE answer.
    pub fn parse(b: &[u8]) -> Result<Self, Error> {
        let (&origin, rest) = b.split_first().ok_or(Error::Truncated)?;
        if origin > origin::HOST {
            return Err(Error::Invalid);
        }
        Ok(PipelineState { origin, chain: Chain::parse(rest)? })
    }
}

/// SET_PREDICTION_INPUT request payload and GET_PREDICTION_INPUT response:
/// `u8 source`, then a chain. In a request the chain is empty unless the
/// source is the model's own chain. In a response it is the chain in
/// effect: the stream's, none, or the model's own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PredictionInput {
    /// Where the model's input comes from, an `input_source` value.
    pub source: u8,
    /// The chain that goes with `source`.
    pub chain: Chain,
}

impl PredictionInput {
    /// The model following the stream, the power-on default.
    pub const STREAM: PredictionInput = PredictionInput { source: input_source::STREAM, chain: Chain::NATURAL };

    /// Encode into `out`.
    pub fn encode(&self, out: &mut [u8]) -> Result<usize, Error> {
        if out.is_empty() {
            return Err(Error::NoRoom);
        }
        out[0] = self.source;
        Ok(1 + self.chain.encode(&mut out[1..])?)
    }

    /// Parse either direction: a source and the chain that goes with it.
    pub fn parse(b: &[u8]) -> Result<Self, Error> {
        let (&source, rest) = b.split_first().ok_or(Error::Truncated)?;
        if source > input_source::OWN_CHAIN {
            return Err(Error::Invalid);
        }
        Ok(PredictionInput { source, chain: Chain::parse(rest)? })
    }

    /// Parse a request, where only the model's own source carries a chain.
    pub fn parse_request(b: &[u8]) -> Result<Self, Error> {
        let p = Self::parse(b)?;
        if p.source != input_source::OWN_CHAIN && !p.chain.is_natural() {
            return Err(Error::Invalid);
        }
        Ok(p)
    }
}

/// One GET_PIPELINE_CATALOG record, 12 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CatalogEntry {
    /// The stage kind.
    pub kind: u8,
    /// The stage's class, a `class` value.
    pub class: u8,
    /// Parameters this kind takes.
    pub n_params: u8,
    /// Instances of this kind a chain may hold at once.
    pub max_instances: u8,
    /// ASCII, zero padded.
    pub name: [u8; 8],
}

impl CatalogEntry {
    /// Bytes in one record.
    pub const LEN: usize = 12;

    /// Build an entry for `kind`.
    pub const fn new(kind: u8, class: u8, n_params: u8, max_instances: u8, name: &[u8; 8]) -> CatalogEntry {
        CatalogEntry { kind, class, n_params, max_instances, name: *name }
    }

    /// Decode one record.
    pub fn parse(b: &[u8]) -> Result<Self, Error> {
        if b.len() < Self::LEN {
            return Err(Error::Truncated);
        }
        let mut name = [0u8; 8];
        name.copy_from_slice(&b[4..12]);
        Ok(CatalogEntry { kind: b[0], class: b[1], n_params: b[2], max_instances: b[3], name })
    }

    /// Encode one record into `out`.
    pub fn encode(&self, out: &mut [u8]) -> Result<usize, Error> {
        if out.len() < Self::LEN {
            return Err(Error::NoRoom);
        }
        out[0] = self.kind;
        out[1] = self.class;
        out[2] = self.n_params;
        out[3] = self.max_instances;
        out[4..12].copy_from_slice(&self.name);
        Ok(Self::LEN)
    }

    /// The name without its padding.
    pub fn name_str(&self) -> &str {
        let end = self.name.iter().position(|&b| b == 0).unwrap_or(8);
        core::str::from_utf8(&self.name[..end]).unwrap_or("")
    }
}

/// GET_PIPELINE_CATALOG payload: `u8 n_kinds`, then the records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Catalog<'a> {
    records: &'a [u8],
}

impl<'a> Catalog<'a> {
    /// The length must be exactly `1 + 12 * n_kinds`.
    pub fn parse(b: &'a [u8]) -> Result<Self, Error> {
        let (&n, rest) = b.split_first().ok_or(Error::Truncated)?;
        if rest.len() != n as usize * CatalogEntry::LEN {
            return Err(Error::Invalid);
        }
        Ok(Catalog { records: rest })
    }

    /// Records in the catalog.
    pub fn len(&self) -> usize {
        self.records.len() / CatalogEntry::LEN
    }

    /// Whether the catalog holds no records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// The record at `i`, if the catalog holds it.
    pub fn get(&self, i: usize) -> Option<CatalogEntry> {
        let start = i.checked_mul(CatalogEntry::LEN)?;
        CatalogEntry::parse(self.records.get(start..)?).ok()
    }

    /// The records in order.
    pub fn iter(&self) -> impl Iterator<Item = CatalogEntry> + 'a {
        self.records.chunks_exact(CatalogEntry::LEN).map(|c| {
            CatalogEntry::parse(c).unwrap_or(CatalogEntry { kind: 0, class: 0, n_params: 0, max_instances: 0, name: [0; 8] })
        })
    }

    /// The entry for a kind, if the catalog has it.
    pub fn kind(&self, kind: u8) -> Option<CatalogEntry> {
        self.iter().find(|e| e.kind == kind)
    }
}

/// Encode a GET_PIPELINE_CATALOG payload from `entries`.
pub fn encode_catalog(entries: &[CatalogEntry], out: &mut [u8]) -> Result<usize, Error> {
    if entries.len() > u8::MAX as usize {
        return Err(Error::Invalid);
    }
    let n = 1 + entries.len() * CatalogEntry::LEN;
    if out.len() < n {
        return Err(Error::NoRoom);
    }
    out[0] = entries.len() as u8;
    for (i, e) in entries.iter().enumerate() {
        e.encode(&mut out[1 + i * CatalogEntry::LEN..])?;
    }
    Ok(n)
}

/// GET_BIAS_DIAGNOSTIC payload: the bias output's mean, spread, minimum
/// and maximum in millivolts over the device's own window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BiasDiagnostic {
    /// Mean of the bias output, in millivolts.
    pub mean_mv: i16,
    /// Spread of the bias output, in millivolts.
    pub sd_mv: i16,
    /// Minimum of the bias output, in millivolts.
    pub min_mv: i16,
    /// Maximum of the bias output, in millivolts.
    pub max_mv: i16,
}

impl BiasDiagnostic {
    /// Bytes in the payload.
    pub const LEN: usize = 8;

    /// Decode a GET_BIAS_DIAGNOSTIC answer.
    pub fn parse(b: &[u8]) -> Result<Self, Error> {
        if b.len() < Self::LEN {
            return Err(Error::Truncated);
        }
        let at = |o: usize| i16::from_le_bytes([b[o], b[o + 1]]);
        Ok(BiasDiagnostic { mean_mv: at(0), sd_mv: at(2), min_mv: at(4), max_mv: at(6) })
    }

    /// Encode into `out`.
    pub fn encode(&self, out: &mut [u8]) -> Result<usize, Error> {
        if out.len() < Self::LEN {
            return Err(Error::NoRoom);
        }
        out[0..2].copy_from_slice(&self.mean_mv.to_le_bytes());
        out[2..4].copy_from_slice(&self.sd_mv.to_le_bytes());
        out[4..6].copy_from_slice(&self.min_mv.to_le_bytes());
        out[6..8].copy_from_slice(&self.max_mv.to_le_bytes());
        Ok(Self::LEN)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_chain() -> Chain {
        let mut c = Chain::NATURAL;
        c.push(Stage::one(kind::HIGH_PASS, 5)).unwrap();
        c.push(Stage::two(kind::NOTCH, 480, 520)).unwrap();
        c.push(Stage::two(kind::NOTCH, 980, 1020)).unwrap();
        c.push(Stage::one(kind::LOW_PASS, 0)).unwrap();
        c
    }

    #[test]
    fn a_chain_round_trips_and_its_length_is_what_it_says() {
        let c = sample_chain();
        let mut buf = [0u8; MAX_ENCODED_LEN];
        let n = c.encode(&mut buf).unwrap();
        // One byte of count, then each stage's two bytes plus two per parameter.
        assert_eq!(n, 1 + 4 + 6 + 6 + 4);
        assert_eq!(n, c.encoded_len());
        assert_eq!(Chain::parse(&buf[..n]).unwrap(), c);
        // The natural signal is one byte.
        let mut one = [0u8; 1];
        assert_eq!(Chain::NATURAL.encode(&mut one).unwrap(), 1);
        assert_eq!(Chain::parse(&one).unwrap(), Chain::NATURAL);
        assert!(Chain::parse(&one).unwrap().is_natural());
    }

    #[test]
    fn a_descriptor_is_refused_when_it_lies_about_itself() {
        // Says two stages, carries one.
        assert_eq!(Chain::parse(&[2, 1, 1, 5, 0]), Err(Error::Truncated));
        // A stage with a parameter it does not carry.
        assert_eq!(Chain::parse(&[1, 1, 1, 5]), Err(Error::Truncated));
        // Kind zero is not a kind.
        assert_eq!(Chain::parse(&[1, 0, 0]), Err(Error::Invalid));
        // Too many parameters for any kind.
        assert_eq!(Chain::parse(&[1, 1, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), Err(Error::Invalid));
        // Too many stages.
        assert_eq!(Chain::parse(&[13]), Err(Error::Invalid));
        // Bytes after the last stage.
        assert_eq!(Chain::parse(&[0, 7]), Err(Error::Invalid));
        // Nothing at all.
        assert_eq!(Chain::parse(&[]), Err(Error::Truncated));
        // A full chain takes no more.
        let mut c = Chain::NATURAL;
        for _ in 0..MAX_STAGES {
            c.push(Stage::two(kind::NOTCH, 480, 520)).unwrap();
        }
        assert_eq!(c.push(Stage::one(kind::LOW_PASS, 0)), Err(Error::NoRoom));
    }

    #[test]
    fn pipeline_state_and_prediction_input_round_trip_with_their_rules() {
        let st = PipelineState { origin: origin::HOST, chain: sample_chain() };
        let mut buf = [0u8; 1 + MAX_ENCODED_LEN];
        let n = st.encode(&mut buf).unwrap();
        assert_eq!(PipelineState::parse(&buf[..n]).unwrap(), st);
        assert_eq!(PipelineState::parse(&[2, 0]), Err(Error::Invalid));

        let own = PredictionInput { source: input_source::OWN_CHAIN, chain: sample_chain() };
        let n = own.encode(&mut buf).unwrap();
        assert_eq!(PredictionInput::parse(&buf[..n]).unwrap(), own);
        let n = PredictionInput::STREAM.encode(&mut buf).unwrap();
        assert_eq!(n, 2);
        assert_eq!(PredictionInput::parse(&buf[..n]).unwrap(), PredictionInput::STREAM);
        // A chain with a source that does not take one is fine in an
        // answer, where the device reports the stream's chain, and invalid
        // in a request.
        let stream = PredictionInput { source: input_source::STREAM, chain: sample_chain() };
        let n = stream.encode(&mut buf).unwrap();
        assert_eq!(PredictionInput::parse(&buf[..n]).unwrap(), stream);
        assert_eq!(PredictionInput::parse_request(&buf[..n]), Err(Error::Invalid));
        assert_eq!(PredictionInput::parse_request(&[2, 0]).unwrap().source, input_source::OWN_CHAIN);
        assert_eq!(PredictionInput::parse(&[3, 0]), Err(Error::Invalid));
        assert_eq!(PredictionInput::parse(&[]), Err(Error::Truncated));
    }

    #[test]
    fn a_catalog_round_trips_and_names_its_kinds() {
        let entries = [
            CatalogEntry::new(kind::HIGH_PASS, class::MAP, 1, 1, b"highpass"),
            CatalogEntry::new(kind::LOW_PASS, class::MAP, 1, 1, b"lowpass\0"),
            CatalogEntry::new(kind::NOTCH, class::MAP, 2, 8, b"notch\0\0\0"),
        ];
        let mut buf = [0u8; 64];
        let n = encode_catalog(&entries, &mut buf).unwrap();
        assert_eq!(n, 1 + 3 * 12);
        let cat = Catalog::parse(&buf[..n]).unwrap();
        assert_eq!(cat.len(), 3);
        assert_eq!(cat.get(2).unwrap(), entries[2]);
        assert_eq!(cat.kind(kind::LOW_PASS).unwrap().name_str(), "lowpass");
        assert_eq!(cat.kind(kind::HIGH_PASS).unwrap().name_str(), "highpass");
        assert!(cat.kind(9).is_none());
        assert_eq!(Catalog::parse(&buf[..n - 1]), Err(Error::Invalid));
        assert_eq!(Catalog::parse(&[]), Err(Error::Truncated));
    }

    #[test]
    fn the_bias_diagnostic_round_trips() {
        let d = BiasDiagnostic { mean_mv: -12, sd_mv: 3, min_mv: -40, max_mv: 25 };
        let mut buf = [0u8; BiasDiagnostic::LEN];
        assert_eq!(d.encode(&mut buf).unwrap(), 8);
        assert_eq!(BiasDiagnostic::parse(&buf).unwrap(), d);
        assert_eq!(BiasDiagnostic::parse(&buf[..7]), Err(Error::Truncated));
    }
}
