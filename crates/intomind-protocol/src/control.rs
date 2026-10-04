//! Control Point (write) and Control Response (indicate).
//!
//! Request: `[opcode, arg?]`, one or two bytes. Response:
//! `[opcode, status, payload?]`. Status 0 is OK, else an error code.
//! Config changes are refused with Busy while streaming so every sample in
//! an epoch sits under one configuration.

use crate::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Opcode {
    StartStream = 0x01,
    StopStream = 0x02,
    SetRate = 0x10,
    SetGain = 0x11,
    SetMode = 0x20,
    SetLeadoff = 0x30,
    TimeSync = 0x40,
    SetSamplesPerPacket = 0x41,
    GetBattery = 0x42,
    GetBootInfo = 0x43,
    ResetEpoch = 0x50,
    ClearBonds = 0x53,
    SetPredictions = 0x80,
    SelectHead = 0x81,
    ListHeads = 0x82,
    RemoveHead = 0x83,
    GetModelInfo = 0x84,
    /// 1.1: where the model's input comes from, with a chain of its own.
    SetPredictionInput = 0x85,
    GetPredictionInput = 0x86,
    /// 1.1: the bias drive, on devices that claim it.
    SetBias = 0x32,
    GetBiasDiagnostic = 0x33,
    /// 1.2: the converter's registers, read only, on devices that claim it.
    GetConverterRegisters = 0x34,
    /// 1.2: the status lamp's level, and identify.
    GetIndicator = 0x44,
    SetIndicator = 0x45,
    Identify = 0x46,
    /// 1.2: which form of the encoder's output the device sends.
    SetEmbeddings = 0x87,
    /// 1.3: the interval between the windows the model describes.
    SetModelInterval = 0x88,
    GetModelInterval = 0x89,
    /// 1.4: the encoder id each head names, one record for each LIST_HEADS
    /// record.
    ListHeadEncoders = 0x8A,
    /// 1.3: the name and the adjective the device composes its name from.
    GetName = 0x47,
    SetName = 0x48,
    /// 1.1: the processing chain.
    GetPipelineCatalog = 0x90,
    GetPipeline = 0x91,
    SetPipeline = 0x92,
    ClearPipeline = 0x93,
    RestorePipelineDefault = 0x94,
    SoftReset = 0xF0,
}

impl Opcode {
    pub fn from_u8(b: u8) -> Option<Self> {
        Some(match b {
            0x01 => Self::StartStream,
            0x02 => Self::StopStream,
            0x10 => Self::SetRate,
            0x11 => Self::SetGain,
            0x20 => Self::SetMode,
            0x30 => Self::SetLeadoff,
            0x40 => Self::TimeSync,
            0x41 => Self::SetSamplesPerPacket,
            0x42 => Self::GetBattery,
            0x43 => Self::GetBootInfo,
            0x50 => Self::ResetEpoch,
            0x53 => Self::ClearBonds,
            0x80 => Self::SetPredictions,
            0x81 => Self::SelectHead,
            0x82 => Self::ListHeads,
            0x83 => Self::RemoveHead,
            0x84 => Self::GetModelInfo,
            0x85 => Self::SetPredictionInput,
            0x86 => Self::GetPredictionInput,
            0x32 => Self::SetBias,
            0x33 => Self::GetBiasDiagnostic,
            0x34 => Self::GetConverterRegisters,
            0x44 => Self::GetIndicator,
            0x45 => Self::SetIndicator,
            0x46 => Self::Identify,
            0x87 => Self::SetEmbeddings,
            0x88 => Self::SetModelInterval,
            0x89 => Self::GetModelInterval,
            0x8A => Self::ListHeadEncoders,
            0x47 => Self::GetName,
            0x48 => Self::SetName,
            0x90 => Self::GetPipelineCatalog,
            0x91 => Self::GetPipeline,
            0x92 => Self::SetPipeline,
            0x93 => Self::ClearPipeline,
            0x94 => Self::RestorePipelineDefault,
            0xF0 => Self::SoftReset,
            _ => return None,
        })
    }

    /// Whether the contract defines an argument byte for this opcode.
    pub fn takes_arg(self) -> bool {
        matches!(
            self,
            Self::SetRate
                | Self::SetGain
                | Self::SetMode
                | Self::SetLeadoff
                | Self::SetSamplesPerPacket
                | Self::SetPredictions
                | Self::SelectHead
                | Self::RemoveHead
                | Self::SetBias
                | Self::SetIndicator
                | Self::Identify
                | Self::SetEmbeddings
        )
    }

    /// Whether the opcode carries a variable-length payload after the
    /// opcode byte, defined in the `pipeline` module. New in 1.1: these
    /// are the only requests longer than two bytes.
    pub fn takes_payload(self) -> bool {
        matches!(self, Self::SetPipeline | Self::SetPredictionInput | Self::SetModelInterval | Self::SetName)
    }

    /// Whether the opcode is new in 1.1. A 1.0 device answers these with
    /// status 1, so a host sends them only to a device whose Device Info
    /// claims the capability behind them.
    pub fn new_in_1_1(self) -> bool {
        matches!(
            self,
            Self::SetPredictionInput
                | Self::GetPredictionInput
                | Self::SetBias
                | Self::GetBiasDiagnostic
                | Self::GetPipelineCatalog
                | Self::GetPipeline
                | Self::SetPipeline
                | Self::ClearPipeline
                | Self::RestorePipelineDefault
        )
    }

    /// Whether the opcode is new in 1.2. A 1.1 device answers these with
    /// status 1, so a host sends them only to a device whose Device Info
    /// claims the capability behind them.
    pub fn new_in_1_2(self) -> bool {
        matches!(self, Self::GetConverterRegisters | Self::GetIndicator | Self::SetIndicator | Self::Identify | Self::SetEmbeddings)
    }

    /// Whether the opcode is new in 1.3. A 1.2 device answers these with
    /// status 1, so a host sends them only to a device whose Device Info
    /// claims the capability behind them.
    pub fn new_in_1_3(self) -> bool {
        matches!(self, Self::SetModelInterval | Self::GetModelInterval | Self::GetName | Self::SetName)
    }

    /// Whether the opcode is new in 1.4. A device before 1.4 answers it with
    /// status 1, so a host sends it only to a device that reports 1.4 or
    /// later.
    pub fn new_in_1_4(self) -> bool {
        matches!(self, Self::ListHeadEncoders)
    }

    /// Every opcode the contract defines.
    pub const ALL: [Opcode; 37] = [
        Self::StartStream,
        Self::StopStream,
        Self::SetRate,
        Self::SetGain,
        Self::SetMode,
        Self::SetLeadoff,
        Self::TimeSync,
        Self::SetSamplesPerPacket,
        Self::GetBattery,
        Self::GetBootInfo,
        Self::ResetEpoch,
        Self::ClearBonds,
        Self::SetPredictions,
        Self::SelectHead,
        Self::ListHeads,
        Self::RemoveHead,
        Self::GetModelInfo,
        Self::SetPredictionInput,
        Self::GetPredictionInput,
        Self::SetBias,
        Self::GetBiasDiagnostic,
        Self::GetConverterRegisters,
        Self::GetIndicator,
        Self::SetIndicator,
        Self::Identify,
        Self::SetEmbeddings,
        Self::SetModelInterval,
        Self::GetModelInterval,
        Self::ListHeadEncoders,
        Self::GetName,
        Self::SetName,
        Self::GetPipelineCatalog,
        Self::GetPipeline,
        Self::SetPipeline,
        Self::ClearPipeline,
        Self::RestorePipelineDefault,
        Self::SoftReset,
    ];

    /// The longest payload this opcode's answer carries on a device with
    /// `head_slots` user head slots and `catalog_kinds` stage kinds in its
    /// processing catalog. Every opcode has one, so a new opcode cannot be
    /// added without saying how long its answer can be, and a device checks
    /// at build time that every answer fits `RESPONSE_MAX`.
    pub const fn answer_max(self, head_slots: usize, catalog_kinds: usize) -> usize {
        match self {
            Self::StartStream
            | Self::StopStream
            | Self::SetRate
            | Self::SetGain
            | Self::SetMode
            | Self::SetLeadoff
            | Self::ResetEpoch
            | Self::ClearBonds
            | Self::SetPredictions
            | Self::SelectHead
            | Self::RemoveHead
            | Self::SetPredictionInput
            | Self::SetBias
            | Self::SetIndicator
            | Self::Identify
            | Self::SetEmbeddings
            | Self::SetModelInterval
            | Self::SetName
            | Self::SetPipeline
            | Self::ClearPipeline
            | Self::RestorePipelineDefault
            | Self::SoftReset => 0,
            // `u64 device_time`.
            Self::TimeSync => 8,
            // `u8 applied`.
            Self::SetSamplesPerPacket => 1,
            Self::GetBattery => BatteryInfo::LEN,
            Self::GetBootInfo => BootInfo::LEN,
            Self::GetModelInfo => ModelInfo::LEN,
            Self::GetModelInterval => interval::ModelInterval::LEN,
            Self::GetBiasDiagnostic => crate::pipeline::BiasDiagnostic::LEN,
            // `u8 level`.
            Self::GetIndicator => 1,
            // `family, first, count`, then the values.
            Self::GetConverterRegisters => 3 + ConverterRegisters::MAX_VALUES,
            // Two lengths and two parts, each at most a whole name.
            Self::GetName => 2 + 2 * name::MAX_COMPOSED,
            // `u8 origin` or `u8 source`, then a chain.
            Self::GetPipeline | Self::GetPredictionInput => 1 + crate::pipeline::MAX_ENCODED_LEN,
            Self::GetPipelineCatalog => 1 + catalog_kinds * crate::pipeline::CatalogEntry::LEN,
            // The built-in slot and the user slots.
            Self::ListHeads => 2 + (1 + head_slots) * HeadEntry::LEN,
            Self::ListHeadEncoders => 1 + (1 + head_slots) * HeadEncoder::LEN,
        }
    }
}

/// 1.3: the interval between the windows the model describes.
pub mod interval {
    use crate::Error;

    /// SET_MODEL_INTERVAL's argument: seconds, little-endian. Zero is every
    /// window the device can, each the newest complete one; `T` is one
    /// window every `T` seconds, the windows starting at sample
    /// `n × T × rate` from the epoch.
    pub fn parse_seconds(payload: &[u8]) -> Result<u16, Error> {
        match payload {
            [lo, hi] => Ok(u16::from_le_bytes([*lo, *hi])),
            _ => Err(Error::Invalid),
        }
    }

    /// GET_MODEL_INTERVAL's payload: the interval in force, then the
    /// smallest interval the device can keep.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct ModelInterval {
        pub interval_s: u16,
        pub minimum_s: u16,
    }

    impl ModelInterval {
        pub const LEN: usize = 4;

        pub fn parse(b: &[u8]) -> Result<Self, Error> {
            if b.len() < Self::LEN {
                return Err(Error::Truncated);
            }
            Ok(ModelInterval { interval_s: u16::from_le_bytes([b[0], b[1]]), minimum_s: u16::from_le_bytes([b[2], b[3]]) })
        }

        pub fn encode(&self, out: &mut [u8]) -> Result<usize, Error> {
            if out.len() < Self::LEN {
                return Err(Error::NoRoom);
            }
            out[0..2].copy_from_slice(&self.interval_s.to_le_bytes());
            out[2..4].copy_from_slice(&self.minimum_s.to_le_bytes());
            Ok(Self::LEN)
        }
    }
}

/// 1.3: the device's name, composed from a name and an adjective a host
/// sets. The composed name is the device's Bluetooth name and must fit
/// what a scan response carries, so every scan list shows the whole of it.
pub mod name {
    use crate::Error;

    /// The most bytes a composed name may have: a scan response carries a
    /// 31 byte structure, two of which are its length and type.
    pub const MAX_COMPOSED: usize = 29;

    /// The bytes after a name: `'s`.
    const POSSESSIVE: &[u8] = b"'s";

    /// Whether one part is acceptable: UTF-8, no control character, no
    /// leading or trailing space. An empty part is acceptable and means
    /// none.
    pub fn valid_part(part: &[u8]) -> bool {
        if core::str::from_utf8(part).is_err() {
            return false;
        }
        if part.iter().any(|&b| b < 0x20 || b == 0x7F) {
            return false;
        }
        !(part.first() == Some(&b' ') || part.last() == Some(&b' '))
    }

    /// How long the composed name would be.
    pub fn composed_len(name_len: usize, adjective_len: usize, product_len: usize) -> usize {
        let mut n = product_len;
        if name_len > 0 {
            n += name_len + POSSESSIVE.len() + 1;
        }
        if adjective_len > 0 {
            n += adjective_len + 1;
        }
        n
    }

    /// Compose the name: the name and `'s` when a name is set, the
    /// adjective when set, the product name, separated by single spaces.
    /// `NoRoom` when `out` is too small.
    pub fn compose(name: &[u8], adjective: &[u8], product: &[u8], out: &mut [u8]) -> Result<usize, Error> {
        let n = composed_len(name.len(), adjective.len(), product.len());
        if out.len() < n {
            return Err(Error::NoRoom);
        }
        let mut at = 0;
        let mut put = |bytes: &[u8], at: &mut usize| {
            out[*at..*at + bytes.len()].copy_from_slice(bytes);
            *at += bytes.len();
        };
        if !name.is_empty() {
            put(name, &mut at);
            put(POSSESSIVE, &mut at);
            put(b" ", &mut at);
        }
        if !adjective.is_empty() {
            put(adjective, &mut at);
            put(b" ", &mut at);
        }
        put(product, &mut at);
        Ok(n)
    }

    /// Whether a name and an adjective may be set on a device with this
    /// product name: both parts valid and the composition within the limit.
    pub fn acceptable(name: &[u8], adjective: &[u8], product: &[u8]) -> bool {
        valid_part(name) && valid_part(adjective) && composed_len(name.len(), adjective.len(), product.len()) <= MAX_COMPOSED
    }

    /// The two parts as SET_NAME carries them and GET_NAME answers:
    /// `u8 name_len, name, u8 adjective_len, adjective`.
    pub fn parse_parts(b: &[u8]) -> Result<(&[u8], &[u8]), Error> {
        let (&nl, rest) = b.split_first().ok_or(Error::Truncated)?;
        let nl = nl as usize;
        if rest.len() < nl + 1 {
            return Err(Error::Invalid);
        }
        let (name, rest) = rest.split_at(nl);
        let (&al, adjective) = rest.split_first().ok_or(Error::Invalid)?;
        if adjective.len() != al as usize {
            return Err(Error::Invalid);
        }
        Ok((name, adjective))
    }

    pub fn encode_parts(name: &[u8], adjective: &[u8], out: &mut [u8]) -> Result<usize, Error> {
        if name.len() > MAX_COMPOSED || adjective.len() > MAX_COMPOSED {
            return Err(Error::Invalid);
        }
        let n = 2 + name.len() + adjective.len();
        if out.len() < n {
            return Err(Error::NoRoom);
        }
        out[0] = name.len() as u8;
        out[1..1 + name.len()].copy_from_slice(name);
        out[1 + name.len()] = adjective.len() as u8;
        out[2 + name.len()..n].copy_from_slice(adjective);
        Ok(n)
    }
}

/// The indicator's levels of verbosity, SET_INDICATOR's argument and
/// GET_INDICATOR's answer. One language; the level decides what is shown.
pub mod indicator {
    /// Nothing, ever.
    pub const SILENT: u8 = 0;
    /// Low battery and a converter fault at power on. The default.
    pub const RESERVED: u8 = 1;
    /// Reserved plus waiting for a host, connected, streaming, and update.
    pub const VERBOSE: u8 = 2;
    /// The longest identify a host may ask for, in seconds.
    pub const IDENTIFY_MAX_SECONDS: u8 = 30;
}

/// GET_CONVERTER_REGISTERS payload: which family of chip the bytes belong
/// to, and the bytes as the chip returned them.
pub mod converter_family {
    pub const ADS129X: u8 = 1;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConverterRegisters<'a> {
    pub family: u8,
    /// Address of the first register carried.
    pub first: u8,
    pub values: &'a [u8],
}

impl<'a> ConverterRegisters<'a> {
    pub const MAX_VALUES: usize = 64;

    /// `family, first, count, values[count]`.
    pub fn parse(b: &'a [u8]) -> Result<Self, Error> {
        if b.len() < 3 {
            return Err(Error::Truncated);
        }
        let count = b[2] as usize;
        if b.len() != 3 + count || count == 0 {
            return Err(Error::Invalid);
        }
        Ok(ConverterRegisters { family: b[0], first: b[1], values: &b[3..] })
    }

    pub fn encode(&self, out: &mut [u8]) -> Result<usize, Error> {
        if self.values.is_empty() || self.values.len() > Self::MAX_VALUES {
            return Err(Error::Invalid);
        }
        let n = 3 + self.values.len();
        if out.len() < n {
            return Err(Error::NoRoom);
        }
        out[0] = self.family;
        out[1] = self.first;
        out[2] = self.values.len() as u8;
        out[3..n].copy_from_slice(self.values);
        Ok(n)
    }
}

/// Opcode numbers held by earlier firmware or by factory builds. A
/// production device answers `Status::Unsupported` to them and never
/// treats them as malformed.
pub fn reserved(op: u8) -> bool {
    matches!(op, 0x12 | 0x13 | 0x31 | 0x70 | 0x71) || (0x60..=0x6F).contains(&op)
}

/// Response status codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Status {
    Ok = 0,
    InvalidArg = 1,
    Unsupported = 2,
    Busy = 3,
    NotStreaming = 4,
    Hardware = 5,
    /// 1.4: refused while USB power is present. A device that does not
    /// run on the wearer while plugged in refuses to start a stream of the
    /// natural signal, from the electrodes or from the converter's own test
    /// signal, and answers this. The generated signal is not refused. 6 is
    /// left unassigned here because the Update service uses 6.
    UsbPower = 7,
}

impl Status {
    pub fn from_u8(b: u8) -> Option<Self> {
        Some(match b {
            0 => Self::Ok,
            1 => Self::InvalidArg,
            2 => Self::Unsupported,
            3 => Self::Busy,
            4 => Self::NotStreaming,
            5 => Self::Hardware,
            7 => Self::UsbPower,
            _ => return None,
        })
    }
}

/// A parsed inbound request, device side. For an opcode that takes a
/// payload, `arg` is `None` and the payload is the bytes after the opcode
/// in the write, which the device parses with the `pipeline` module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Request {
    pub opcode: Opcode,
    pub arg: Option<u8>,
}

impl Request {
    /// Parse a write. Precedence for the device's handler is fixed by the
    /// contract: invalid argument outranks unsupported outranks busy
    /// outranks hardware, so validation happens on the parse result before
    /// any state is consulted. `Error::Reserved` maps to Unsupported, every
    /// other error to InvalidArg.
    pub fn parse(buf: &[u8]) -> Result<Self, Error> {
        let (&op, rest) = buf.split_first().ok_or(Error::Truncated)?;
        if reserved(op) {
            return Err(Error::Reserved);
        }
        let opcode = Opcode::from_u8(op).ok_or(Error::Invalid)?;
        if opcode.takes_payload() {
            // The payload's own parser judges it, so a malformed chain is
            // reported as invalid by the device rather than here.
            return Ok(Request { opcode, arg: None });
        }
        let arg = match (opcode.takes_arg(), rest) {
            (true, [a]) => Some(*a),
            (false, []) => None,
            _ => return Err(Error::Invalid),
        };
        Ok(Request { opcode, arg })
    }

    /// The payload of a write, for an opcode that takes one.
    pub fn payload(buf: &[u8]) -> &[u8] {
        buf.get(1..).unwrap_or(&[])
    }

    pub fn encode(&self, out: &mut [u8]) -> Result<usize, Error> {
        let n = 1 + self.arg.is_some() as usize;
        if out.len() < n {
            return Err(Error::NoRoom);
        }
        out[0] = self.opcode as u8;
        if let Some(a) = self.arg {
            out[1] = a;
        }
        Ok(n)
    }
}

/// Encode a request that carries a payload: the opcode, then the bytes.
pub fn encode_request_with_payload(opcode: Opcode, payload: &[u8], out: &mut [u8]) -> Result<usize, Error> {
    if !opcode.takes_payload() {
        return Err(Error::Invalid);
    }
    let n = 1 + payload.len();
    if out.len() < n {
        return Err(Error::NoRoom);
    }
    out[0] = opcode as u8;
    out[1..n].copy_from_slice(payload);
    Ok(n)
}

/// The longest answer on Control Response, opcode and status included: an
/// indication at the smallest MTU the contract allows for anything but
/// EEG Data, 159, less its three byte header (section 3 of 1.0).
pub const RESPONSE_MAX: usize = 156;

/// The longest payload an answer may carry.
pub const PAYLOAD_MAX: usize = RESPONSE_MAX - 2;

/// The status and length of an answer whose payload was just composed: OK
/// and the payload's length, or status 5 and nothing when it could not be
/// composed. A device never answers 2, or a short OK, for an answer it
/// failed to compose (section 26.3 of 1.4).
pub fn composed(r: Result<usize, Error>) -> (Status, usize) {
    match r {
        Ok(n) => (Status::Ok, n),
        Err(_) => (Status::Hardware, 0),
    }
}

/// Compose an answer into a buffer of the contract's size and return its
/// length, never zero. A payload longer than an answer may carry is
/// answered status 5 with nothing after it.
pub fn compose_response(opcode: u8, status: Status, payload: &[u8], out: &mut [u8; RESPONSE_MAX]) -> usize {
    match encode_response_raw(opcode, status, payload, out) {
        Ok(n) => n,
        Err(_) => {
            out[0] = opcode;
            out[1] = Status::Hardware as u8;
            2
        }
    }
}

/// An outbound response, device side. `payload` is appended verbatim.
/// For a reserved or unknown opcode the device echoes the raw byte, so this
/// takes the number rather than the enum.
pub fn encode_response_raw(
    opcode: u8,
    status: Status,
    payload: &[u8],
    out: &mut [u8],
) -> Result<usize, Error> {
    let n = 2 + payload.len();
    if out.len() < n {
        return Err(Error::NoRoom);
    }
    out[0] = opcode;
    out[1] = status as u8;
    out[2..n].copy_from_slice(payload);
    Ok(n)
}

pub fn encode_response(
    opcode: Opcode,
    status: Status,
    payload: &[u8],
    out: &mut [u8],
) -> Result<usize, Error> {
    encode_response_raw(opcode as u8, status, payload, out)
}

/// A parsed response, host side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Response<'a> {
    pub opcode: Opcode,
    pub status: Status,
    pub payload: &'a [u8],
}

impl<'a> Response<'a> {
    pub fn parse(buf: &'a [u8]) -> Result<Self, Error> {
        if buf.len() < 2 {
            return Err(Error::Truncated);
        }
        Ok(Response {
            opcode: Opcode::from_u8(buf[0]).ok_or(Error::Invalid)?,
            status: Status::from_u8(buf[1]).ok_or(Error::Invalid)?,
            payload: &buf[2..],
        })
    }
}

// ---- payloads ------------------------------------------------------------

/// GET_BATTERY payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatteryInfo {
    pub battery_mv: u16,
    pub battery_percent: u8,
    pub charger_state: u8,
}

impl BatteryInfo {
    pub const LEN: usize = 4;

    pub fn parse(b: &[u8]) -> Result<Self, Error> {
        if b.len() < Self::LEN {
            return Err(Error::Truncated);
        }
        Ok(BatteryInfo {
            battery_mv: u16::from_le_bytes([b[0], b[1]]),
            battery_percent: b[2],
            charger_state: b[3],
        })
    }

    pub fn encode(&self, out: &mut [u8]) -> Result<usize, Error> {
        if out.len() < Self::LEN {
            return Err(Error::NoRoom);
        }
        out[0..2].copy_from_slice(&self.battery_mv.to_le_bytes());
        out[2] = self.battery_percent;
        out[3] = self.charger_state;
        Ok(Self::LEN)
    }
}

/// GET_BOOT_INFO `boot_reason` values.
pub mod boot_reason {
    pub const POWER_ON: u8 = 0;
    pub const RESET_PIN: u8 = 1;
    pub const SOFTWARE: u8 = 2;
    pub const WATCHDOG: u8 = 3;
    pub const LOCKUP: u8 = 4;
    pub const UPDATE: u8 = 5;
    pub const ROLLBACK: u8 = 6;
    pub const UNKNOWN: u8 = 0xFF;
}

/// GET_BOOT_INFO `slot_state` values.
pub mod slot_state {
    pub const TRIAL: u8 = 0;
    pub const CONFIRMED: u8 = 1;
}

/// GET_BOOT_INFO payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BootInfo {
    pub active_slot: u8,
    pub boot_reason: u8,
    pub slot_state: u8,
    pub boot_count: u32,
}

impl BootInfo {
    pub const LEN: usize = 8;

    pub fn parse(b: &[u8]) -> Result<Self, Error> {
        if b.len() < Self::LEN {
            return Err(Error::Truncated);
        }
        Ok(BootInfo {
            active_slot: b[0],
            boot_reason: b[1],
            slot_state: b[2],
            boot_count: u32::from_le_bytes([b[4], b[5], b[6], b[7]]),
        })
    }

    pub fn encode(&self, out: &mut [u8]) -> Result<usize, Error> {
        if out.len() < Self::LEN {
            return Err(Error::NoRoom);
        }
        out[0] = self.active_slot;
        out[1] = self.boot_reason;
        out[2] = self.slot_state;
        out[3] = 0;
        out[4..8].copy_from_slice(&self.boot_count.to_le_bytes());
        Ok(Self::LEN)
    }
}

/// LIST_HEADS entry `state` values.
pub mod head_state {
    pub const EMPTY: u8 = 0;
    pub const VALID: u8 = 1;
    pub const INVALID: u8 = 2;
    /// 1.3: a sound head whose input width is not the loaded encoder's. It
    /// is kept and never run; it runs again if weights of its width return.
    pub const WIDTH_MISMATCH: u8 = 3;
}

/// One LIST_HEADS record, 30 bytes, and (1.3) the encoder the head says it
/// was trained beside, all zero when the head does not say. A device sends
/// the encoder in LIST_HEAD_ENCODERS (1.4), never in LIST_HEADS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadEntry {
    pub slot: u8,
    pub state: u8,
    pub out_dim: u16,
    pub head_id: [u8; 8],
    pub name: [u8; 16],
    pub encoder_id: [u8; 8],
}

impl HeadEntry {
    pub const LEN: usize = 30;

    pub fn parse(b: &[u8]) -> Result<Self, Error> {
        if b.len() < Self::LEN {
            return Err(Error::Truncated);
        }
        let mut head_id = [0u8; 8];
        head_id.copy_from_slice(&b[4..12]);
        let mut name = [0u8; 16];
        name.copy_from_slice(&b[12..28]);
        Ok(HeadEntry {
            slot: b[0],
            state: b[1],
            out_dim: u16::from_le_bytes([b[2], b[3]]),
            head_id,
            name,
            encoder_id: [0; 8],
        })
    }

    pub fn encode(&self, out: &mut [u8]) -> Result<usize, Error> {
        if out.len() < Self::LEN {
            return Err(Error::NoRoom);
        }
        out[0] = self.slot;
        out[1] = self.state;
        out[2..4].copy_from_slice(&self.out_dim.to_le_bytes());
        out[4..12].copy_from_slice(&self.head_id);
        out[12..28].copy_from_slice(&self.name);
        out[28] = 0;
        out[29] = 0;
        Ok(Self::LEN)
    }
}

/// LIST_HEADS payload: `u8 active_slot, u8 n_entries`, then the records.
/// 1.3 defined a trailer of one eight byte encoder id per record, which no
/// device sent and 1.4 withdraws. A host still reads one if it comes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ListHeads<'a> {
    pub active_slot: u8,
    records: &'a [u8],
    /// Empty on a 1.2 device.
    encoders: &'a [u8],
}

impl<'a> ListHeads<'a> {
    /// Parse a payload. The length is `2 + 30 × n_entries`, or with the
    /// 1.3 trailer `2 + 38 × n_entries`.
    pub fn parse(b: &'a [u8]) -> Result<Self, Error> {
        if b.len() < 2 {
            return Err(Error::Truncated);
        }
        let n = b[1] as usize;
        let records_end = 2 + n * HeadEntry::LEN;
        if b.len() == records_end {
            return Ok(ListHeads { active_slot: b[0], records: &b[2..], encoders: &[] });
        }
        if b.len() != records_end + n * 8 {
            return Err(Error::Invalid);
        }
        Ok(ListHeads { active_slot: b[0], records: &b[2..records_end], encoders: &b[records_end..] })
    }

    pub fn len(&self) -> usize {
        self.records.len() / HeadEntry::LEN
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn get(&self, i: usize) -> Option<HeadEntry> {
        let start = i.checked_mul(HeadEntry::LEN)?;
        let mut e = HeadEntry::parse(self.records.get(start..)?).ok()?;
        if let Some(id) = self.encoders.get(i * 8..i * 8 + 8) {
            e.encoder_id.copy_from_slice(id);
        }
        Some(e)
    }

    pub fn iter(&self) -> impl Iterator<Item = HeadEntry> + 'a {
        let this = *self;
        (0..self.len()).map(move |i| {
            this.get(i).unwrap_or(HeadEntry { slot: 0, state: 0, out_dim: 0, head_id: [0; 8], name: [0; 16], encoder_id: [0; 8] })
        })
    }
}

/// Encode a LIST_HEADS payload: the records and nothing after them.
pub fn encode_list_heads(active_slot: u8, entries: &[HeadEntry], out: &mut [u8]) -> Result<usize, Error> {
    if entries.len() > u8::MAX as usize {
        return Err(Error::Invalid);
    }
    let n = 2 + entries.len() * HeadEntry::LEN;
    if out.len() < n {
        return Err(Error::NoRoom);
    }
    out[0] = active_slot;
    out[1] = entries.len() as u8;
    for (i, e) in entries.iter().enumerate() {
        e.encode(&mut out[2 + i * HeadEntry::LEN..])?;
    }
    Ok(n)
}

/// One LIST_HEAD_ENCODERS record (1.4): a slot and the encoder id its head
/// names, all zero for an empty slot or a head that does not say.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadEncoder {
    pub slot: u8,
    pub encoder_id: [u8; 8],
}

impl HeadEncoder {
    pub const LEN: usize = 9;
}

/// LIST_HEAD_ENCODERS payload (1.4): `u8 n_entries`, then one record for
/// each LIST_HEADS record, in the same order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadEncoders<'a> {
    records: &'a [u8],
}

impl<'a> HeadEncoders<'a> {
    pub fn parse(b: &'a [u8]) -> Result<Self, Error> {
        let (&n, records) = b.split_first().ok_or(Error::Truncated)?;
        if records.len() != n as usize * HeadEncoder::LEN {
            return Err(Error::Invalid);
        }
        Ok(HeadEncoders { records })
    }

    pub fn len(&self) -> usize {
        self.records.len() / HeadEncoder::LEN
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn get(&self, i: usize) -> Option<HeadEncoder> {
        let r = self.records.get(i.checked_mul(HeadEncoder::LEN)?..)?.get(..HeadEncoder::LEN)?;
        let mut encoder_id = [0u8; 8];
        encoder_id.copy_from_slice(&r[1..9]);
        Some(HeadEncoder { slot: r[0], encoder_id })
    }

    pub fn iter(&self) -> impl Iterator<Item = HeadEncoder> + 'a {
        let this = *self;
        (0..self.len()).filter_map(move |i| this.get(i))
    }
}

/// Encode a LIST_HEAD_ENCODERS payload from the same entries LIST_HEADS
/// lists, in the same order.
pub fn encode_head_encoders(entries: &[HeadEntry], out: &mut [u8]) -> Result<usize, Error> {
    if entries.len() > u8::MAX as usize {
        return Err(Error::Invalid);
    }
    let n = 1 + entries.len() * HeadEncoder::LEN;
    if out.len() < n {
        return Err(Error::NoRoom);
    }
    out[0] = entries.len() as u8;
    for (i, e) in entries.iter().enumerate() {
        let r = &mut out[1 + i * HeadEncoder::LEN..1 + (i + 1) * HeadEncoder::LEN];
        r[0] = e.slot;
        r[1..9].copy_from_slice(&e.encoder_id);
    }
    Ok(n)
}

/// GET_MODEL_INFO `model_state` values.
/// SET_MODE values, and the mode field of Status and of the stream header.
pub mod mode {
    pub const NORMAL: u8 = 0;
    pub const TEST: u8 = 1;
    pub const SHORT: u8 = 2;
    /// 1.3: the converter is not driven; the device generates the signal.
    /// Refused with Unsupported on a device without the synthetic
    /// capability, with Busy while streaming, and with InvalidArg at a rate
    /// other than the generator's 500 samples a second. Never persisted.
    pub const SYNTHETIC: u8 = 3;
}

pub mod model_state {
    pub const NONE: u8 = 0;
    pub const NO_WEIGHTS: u8 = 1;
    pub const READY: u8 = 2;
    /// 1.3: the weights are being replaced; the model is stopped.
    pub const UPDATING: u8 = 3;
}

/// GET_MODEL_INFO payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelInfo {
    pub model_state: u8,
    /// Slot, or 0xFF when none is selected.
    pub active_head: u8,
    pub predictions_on: u8,
    pub encoder_id: [u8; 8],
    pub weights_version: (u8, u8, u8),
    /// 1.1: the signal classes the loaded model declares it can take, as
    /// `pipeline::input_class` bits. A 1.0 device sends 0, which a host
    /// reads as the time domain.
    pub input_classes: u8,
    /// 1.2: tokens the loaded model produces per channel per window, the
    /// shape of the token form of embeddings. A 1.1 device's message ends
    /// before this byte, and a host reads 0: it sends no tokens.
    pub tokens_per_channel: u8,
    /// 1.3: wall time of the last pass in milliseconds, 0 before the first.
    pub pass_ms: u16,
    /// 1.3: the interval in force, as SET_MODEL_INTERVAL set it.
    pub interval_s: u16,
    /// 1.3: bit 0, the device holds a generator and offers synthetic signal.
    pub generator: u8,
}

pub const NO_HEAD: u8 = 0xFF;

impl ModelInfo {
    /// The 1.1 length. A 1.2 device appends one byte, a 1.3 device five more.
    pub const LEN_1_1: usize = 16;
    pub const LEN_1_2: usize = 17;
    pub const LEN: usize = 22;

    pub fn parse(b: &[u8]) -> Result<Self, Error> {
        if b.len() < Self::LEN_1_1 {
            return Err(Error::Truncated);
        }
        let mut encoder_id = [0u8; 8];
        encoder_id.copy_from_slice(&b[4..12]);
        Ok(ModelInfo {
            model_state: b[0],
            active_head: b[1],
            predictions_on: b[2],
            encoder_id,
            weights_version: (b[12], b[13], b[14]),
            input_classes: b[15],
            tokens_per_channel: b.get(16).copied().unwrap_or(0),
            pass_ms: b.get(17..19).map(|x| u16::from_le_bytes([x[0], x[1]])).unwrap_or(0),
            interval_s: b.get(19..21).map(|x| u16::from_le_bytes([x[0], x[1]])).unwrap_or(0),
            generator: b.get(21).copied().unwrap_or(0),
        })
    }

    pub fn encode(&self, out: &mut [u8]) -> Result<usize, Error> {
        if out.len() < Self::LEN {
            return Err(Error::NoRoom);
        }
        out[0] = self.model_state;
        out[1] = self.active_head;
        out[2] = self.predictions_on;
        out[3] = 0;
        out[4..12].copy_from_slice(&self.encoder_id);
        out[12] = self.weights_version.0;
        out[13] = self.weights_version.1;
        out[14] = self.weights_version.2;
        out[15] = self.input_classes;
        out[16] = self.tokens_per_channel;
        out[17..19].copy_from_slice(&self.pass_ms.to_le_bytes());
        out[19..21].copy_from_slice(&self.interval_s.to_le_bytes());
        out[21] = self.generator;
        Ok(Self::LEN)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_round_trip_with_and_without_arg() {
        for req in [
            Request { opcode: Opcode::StartStream, arg: None },
            Request { opcode: Opcode::SetRate, arg: Some(5) },
            Request { opcode: Opcode::SelectHead, arg: Some(0) },
            Request { opcode: Opcode::GetModelInfo, arg: None },
        ] {
            let mut buf = [0u8; 2];
            let n = req.encode(&mut buf).unwrap();
            assert_eq!(Request::parse(&buf[..n]).unwrap(), req);
        }
    }

    #[test]
    fn arg_arity_is_enforced_both_ways() {
        // SET_RATE without its argument
        assert_eq!(Request::parse(&[0x10]), Err(Error::Invalid));
        // START_STREAM with a stray argument
        assert_eq!(Request::parse(&[0x01, 0x00]), Err(Error::Invalid));
        // SET_PREDICTIONS without its argument
        assert_eq!(Request::parse(&[0x80]), Err(Error::Invalid));
        // unknown opcode
        assert_eq!(Request::parse(&[0x99]), Err(Error::Invalid));
        // empty write
        assert_eq!(Request::parse(&[]), Err(Error::Truncated));
    }

    #[test]
    fn reserved_numbers_are_reserved_not_malformed() {
        for op in [0x12u8, 0x13, 0x31, 0x70, 0x71, 0x60, 0x6F] {
            assert_eq!(Request::parse(&[op]), Err(Error::Reserved), "{op:#x}");
            assert!(Opcode::from_u8(op).is_none(), "{op:#x}");
        }
        assert!(!reserved(0x5F));
        assert!(!reserved(0x72));
        assert!(!reserved(0x80));
    }

    #[test]
    fn every_opcode_survives_the_byte_round_trip() {
        for op in Opcode::ALL {
            assert_eq!(Opcode::from_u8(op as u8), Some(op));
            assert!(!reserved(op as u8));
        }
        // The list is every opcode, once.
        let defined: Vec<u8> = (0..=255u8).filter(|&b| Opcode::from_u8(b).is_some()).collect();
        let mut listed: Vec<u8> = Opcode::ALL.iter().map(|&o| o as u8).collect();
        listed.sort_unstable();
        assert_eq!(listed, defined);
    }

    #[test]
    fn the_1_4_opcode_is_new_and_takes_nothing() {
        assert!(Opcode::ListHeadEncoders.new_in_1_4());
        assert!(!Opcode::ListHeadEncoders.new_in_1_3());
        assert!(!Opcode::ListHeads.new_in_1_4());
        assert!(!reserved(0x8A));
        assert_eq!(Request::parse(&[0x8A]).unwrap(), Request { opcode: Opcode::ListHeadEncoders, arg: None });
        assert_eq!(Request::parse(&[0x8A, 0]), Err(Error::Invalid));
    }

    #[test]
    fn the_1_2_numbers_are_new_and_take_their_arguments() {
        for op in [0x34u8, 0x44, 0x45, 0x46, 0x87] {
            assert!(!reserved(op));
            let o = Opcode::from_u8(op).unwrap();
            assert!(o.new_in_1_2(), "{op:#x}");
            assert!(!o.new_in_1_1(), "{op:#x}");
        }
        assert!(!Opcode::SetBias.new_in_1_2());
        assert_eq!(Request::parse(&[0x45, 2]).unwrap(), Request { opcode: Opcode::SetIndicator, arg: Some(2) });
        assert_eq!(Request::parse(&[0x45]), Err(Error::Invalid));
        assert_eq!(Request::parse(&[0x46, 10]).unwrap(), Request { opcode: Opcode::Identify, arg: Some(10) });
        assert_eq!(Request::parse(&[0x87, 3]).unwrap(), Request { opcode: Opcode::SetEmbeddings, arg: Some(3) });
        assert_eq!(Request::parse(&[0x44]).unwrap(), Request { opcode: Opcode::GetIndicator, arg: None });
        assert_eq!(Request::parse(&[0x44, 0]), Err(Error::Invalid));
        assert_eq!(Request::parse(&[0x34]).unwrap(), Request { opcode: Opcode::GetConverterRegisters, arg: None });
    }

    #[test]
    fn converter_registers_round_trip_and_length_rule() {
        let values: [u8; 24] = core::array::from_fn(|i| i as u8 * 3);
        let r = ConverterRegisters { family: converter_family::ADS129X, first: 0, values: &values };
        let mut buf = [0u8; 40];
        let n = r.encode(&mut buf).unwrap();
        assert_eq!(n, 27);
        let back = ConverterRegisters::parse(&buf[..n]).unwrap();
        assert_eq!(back, r);
        assert_eq!(ConverterRegisters::parse(&buf[..n - 1]), Err(Error::Invalid));
        assert_eq!(ConverterRegisters::parse(&buf[..2]), Err(Error::Truncated));
        assert_eq!(ConverterRegisters { family: 1, first: 0, values: &[] }.encode(&mut buf), Err(Error::Invalid));
    }

    #[test]
    fn payload_opcodes_parse_at_any_length_and_encode_with_their_bytes() {
        // The chain's own parser judges the payload; the request parser
        // only names the opcode.
        assert_eq!(Request::parse(&[0x92]).unwrap(), Request { opcode: Opcode::SetPipeline, arg: None });
        assert_eq!(Request::parse(&[0x92, 0]).unwrap(), Request { opcode: Opcode::SetPipeline, arg: None });
        assert_eq!(Request::parse(&[0x85, 2, 1, 1, 1, 5, 0]).unwrap().opcode, Opcode::SetPredictionInput);
        assert_eq!(Request::payload(&[0x92, 1, 2, 3]), &[1, 2, 3]);
        let mut buf = [0u8; 8];
        let n = encode_request_with_payload(Opcode::SetPipeline, &[0], &mut buf).unwrap();
        assert_eq!(&buf[..n], &[0x92, 0]);
        assert_eq!(encode_request_with_payload(Opcode::SetRate, &[5], &mut buf), Err(Error::Invalid));
        // The fixed-length opcodes keep their arity rule.
        assert_eq!(Request::parse(&[0x32]), Err(Error::Invalid));
        assert_eq!(Request::parse(&[0x32, 1]).unwrap(), Request { opcode: Opcode::SetBias, arg: Some(1) });
        assert_eq!(Request::parse(&[0x91, 0]), Err(Error::Invalid));
        // The 1.1 numbers stay clear of everything reserved.
        for op in [0x85u8, 0x86, 0x32, 0x33, 0x90, 0x91, 0x92, 0x93, 0x94] {
            assert!(!reserved(op));
            assert!(Opcode::from_u8(op).unwrap().new_in_1_1());
        }
        assert!(!Opcode::SetRate.new_in_1_1());
    }

    #[test]
    fn time_sync_response_round_trip() {
        let t: u64 = 0x0102_0304_0506_0708;
        let mut buf = [0u8; 10];
        let n = encode_response(Opcode::TimeSync, Status::Ok, &t.to_le_bytes(), &mut buf).unwrap();
        let r = Response::parse(&buf[..n]).unwrap();
        assert_eq!(r.opcode, Opcode::TimeSync);
        assert_eq!(r.status, Status::Ok);
        assert_eq!(u64::from_le_bytes(r.payload.try_into().unwrap()), t);
    }

    #[test]
    fn payload_round_trips_and_sizes() {
        let b = BatteryInfo { battery_mv: 3712, battery_percent: 61, charger_state: 1 };
        let mut buf = [0u8; BatteryInfo::LEN];
        assert_eq!(b.encode(&mut buf).unwrap(), 4);
        assert_eq!(BatteryInfo::parse(&buf).unwrap(), b);

        let bi = BootInfo { active_slot: 1, boot_reason: boot_reason::ROLLBACK, slot_state: slot_state::CONFIRMED, boot_count: 70_000 };
        let mut buf = [0u8; BootInfo::LEN];
        assert_eq!(bi.encode(&mut buf).unwrap(), 8);
        assert_eq!(BootInfo::parse(&buf).unwrap(), bi);

        let mi = ModelInfo {
            model_state: model_state::READY,
            active_head: 0,
            predictions_on: 1,
            encoder_id: [1, 2, 3, 4, 5, 6, 7, 8],
            pass_ms: 3150,
            interval_s: 0,
            generator: 0,
            weights_version: (1, 0, 3),
            input_classes: crate::pipeline::input_class::TIME_DOMAIN,
            tokens_per_channel: 20,
        };
        let mut buf = [0u8; ModelInfo::LEN];
        assert_eq!(mi.encode(&mut buf).unwrap(), 22);
        assert_eq!(ModelInfo::parse(&buf).unwrap(), mi);
        // A 1.2 device's seventeen bytes still parse, and say the device
        // paces itself: no pass time, no interval, no generator.
        let mid = ModelInfo::parse(&buf[..17]).unwrap();
        assert_eq!((mid.pass_ms, mid.interval_s, mid.generator), (0, 0, 0));
        assert_eq!(mid.tokens_per_channel, mi.tokens_per_channel);
        // A 1.1 device's sixteen bytes still parse, and say no tokens.
        let old = ModelInfo::parse(&buf[..16]).unwrap();
        assert_eq!(old.tokens_per_channel, 0);
        assert_eq!(old.input_classes, mi.input_classes);
        assert_eq!(ModelInfo::parse(&buf[..15]), Err(Error::Truncated));
    }

    #[test]
    fn list_heads_round_trip_and_length_rule() {
        let mut name = [0u8; 16];
        name[..5].copy_from_slice(b"focus");
        let entries = [
            HeadEntry { slot: 0, state: head_state::VALID, out_dim: 2, head_id: [9; 8], name: [0; 16], encoder_id: [0xE2; 8] },
            HeadEntry { slot: 1, state: head_state::EMPTY, out_dim: 0, head_id: [0; 8], name: [0; 16], encoder_id: [0; 8] },
            HeadEntry { slot: 2, state: head_state::VALID, out_dim: 3, head_id: [7; 8], name, encoder_id: [0xA7; 8] },
        ];
        let mut buf = [0u8; PAYLOAD_MAX];
        let n = encode_list_heads(0, &entries, &mut buf).unwrap();
        assert_eq!(n, 2 + 3 * HeadEntry::LEN, "the records and nothing after them");
        let l = ListHeads::parse(&buf[..n]).unwrap();
        assert_eq!(l.active_slot, 0);
        assert_eq!(l.len(), 3);
        // The encoder ids travel in LIST_HEAD_ENCODERS, so the list reads none.
        assert_eq!(l.get(2).unwrap(), HeadEntry { encoder_id: [0; 8], ..entries[2] });
        assert!(l.get(3).is_none());
        assert_eq!(l.iter().count(), 3);
        // A payload whose length disagrees with n_entries is malformed.
        assert_eq!(ListHeads::parse(&buf[..n - 1]), Err(Error::Invalid));
        assert_eq!(ListHeads::parse(&buf[..n + 1]), Err(Error::Invalid));
        // A list too long for its buffer is refused, never cut short.
        assert_eq!(encode_list_heads(0, &entries, &mut buf[..n - 1]), Err(Error::NoRoom));
        // A 1.3 payload with the withdrawn trailer is still read, ids and all.
        let mut old = [0u8; 2 + 3 * (HeadEntry::LEN + 8)];
        old[..n].copy_from_slice(&buf[..n]);
        old[n..n + 8].copy_from_slice(&[0xE2; 8]);
        let l = ListHeads::parse(&old).unwrap();
        assert_eq!(l.get(0).unwrap().encoder_id, [0xE2; 8]);
        assert_eq!(l.get(1).unwrap().encoder_id, [0; 8]);
    }

    #[test]
    fn head_encoders_round_trip_and_length_rule() {
        let entries = [
            HeadEntry { slot: 0, state: head_state::EMPTY, out_dim: 0, head_id: [0; 8], name: [0; 16], encoder_id: [0; 8] },
            HeadEntry { slot: 1, state: head_state::VALID, out_dim: 1, head_id: [3; 8], name: [0; 16], encoder_id: [0xE2, 0xF9, 0xB6, 0x0F, 0x41, 0x1F, 0x0E, 0x73] },
            HeadEntry { slot: 2, state: head_state::VALID, out_dim: 1, head_id: [4; 8], name: [0; 16], encoder_id: [0; 8] },
        ];
        let mut buf = [0u8; PAYLOAD_MAX];
        let n = encode_head_encoders(&entries, &mut buf).unwrap();
        assert_eq!(n, 1 + 3 * HeadEncoder::LEN);
        let h = HeadEncoders::parse(&buf[..n]).unwrap();
        assert_eq!(h.len(), 3);
        assert_eq!(h.get(1).unwrap(), HeadEncoder { slot: 1, encoder_id: entries[1].encoder_id });
        assert_eq!(h.iter().map(|e| e.slot).collect::<Vec<_>>(), [0, 1, 2]);
        assert!(h.get(3).is_none());
        assert_eq!(HeadEncoders::parse(&buf[..n - 1]), Err(Error::Invalid));
        assert_eq!(HeadEncoders::parse(&buf[..n + 1]), Err(Error::Invalid));
        assert_eq!(HeadEncoders::parse(&[]), Err(Error::Truncated));
        assert_eq!(encode_head_encoders(&entries, &mut buf[..n - 1]), Err(Error::NoRoom));
    }

    /// Every answer, built at its largest by its own encoder, is as long as
    /// `answer_max` says, and on a device with the IntoMind One's four head
    /// slots and a catalog of up to twelve kinds every one fits.
    #[test]
    fn every_answer_at_its_largest_fits_the_limit() {
        use crate::pipeline::{self, CatalogEntry, Chain, PipelineState, PredictionInput, Stage};
        const SLOTS: usize = 4;
        const KINDS: usize = 12;
        let mut buf = [0u8; 512];
        let mut full = Chain::NATURAL;
        while full.push(Stage { kind: 1, n_params: pipeline::MAX_PARAMS as u8, params: [1; pipeline::MAX_PARAMS] }).is_ok() {}
        let regs = [0u8; ConverterRegisters::MAX_VALUES];
        let long = [b'a'; name::MAX_COMPOSED];
        let kinds = [CatalogEntry::new(1, 0, 1, 1, b"highpass"); KINDS];
        let heads = [HeadEntry { slot: 1, state: head_state::VALID, out_dim: 32, head_id: [1; 8], name: [b'n'; 16], encoder_id: [2; 8] }; 1 + SLOTS];
        let mi = ModelInfo {
            model_state: model_state::READY,
            active_head: 1,
            predictions_on: 1,
            encoder_id: [1; 8],
            weights_version: (1, 0, 0),
            input_classes: 1,
            tokens_per_channel: 20,
            pass_ms: 3525,
            interval_s: 0,
            generator: 1,
        };
        for op in Opcode::ALL {
            let built = match op {
                Opcode::TimeSync => 8,
                Opcode::SetSamplesPerPacket | Opcode::GetIndicator => 1,
                Opcode::GetBattery => BatteryInfo { battery_mv: 4200, battery_percent: 100, charger_state: 0 }.encode(&mut buf).unwrap(),
                Opcode::GetBootInfo => BootInfo { active_slot: 1, boot_reason: 0, slot_state: 1, boot_count: 9 }.encode(&mut buf).unwrap(),
                Opcode::GetModelInfo => mi.encode(&mut buf).unwrap(),
                Opcode::GetModelInterval => interval::ModelInterval { interval_s: 60, minimum_s: 4 }.encode(&mut buf).unwrap(),
                Opcode::GetBiasDiagnostic => pipeline::BiasDiagnostic { mean_mv: 1, sd_mv: 1, min_mv: 1, max_mv: 1 }.encode(&mut buf).unwrap(),
                Opcode::GetConverterRegisters => ConverterRegisters { family: 1, first: 0, values: &regs }.encode(&mut buf).unwrap(),
                Opcode::GetName => name::encode_parts(&long, &long, &mut buf).unwrap(),
                Opcode::GetPipeline => PipelineState { origin: 1, chain: full }.encode(&mut buf).unwrap(),
                Opcode::GetPredictionInput => PredictionInput { source: pipeline::input_source::OWN_CHAIN, chain: full }.encode(&mut buf).unwrap(),
                Opcode::GetPipelineCatalog => pipeline::encode_catalog(&kinds, &mut buf).unwrap(),
                Opcode::ListHeads => encode_list_heads(1, &heads, &mut buf).unwrap(),
                Opcode::ListHeadEncoders => encode_head_encoders(&heads, &mut buf).unwrap(),
                _ => 0,
            };
            assert_eq!(built, op.answer_max(SLOTS, KINDS), "{op:?}");
            assert!(2 + op.answer_max(SLOTS, KINDS) <= RESPONSE_MAX, "{op:?}: {} bytes", 2 + built);
        }
        // The list with the withdrawn trailer was 194 bytes, and a fifth user
        // slot would not fit even without it.
        assert_eq!(2 + 2 + (1 + SLOTS) * (HeadEntry::LEN + 8), 194);
        assert!(2 + Opcode::ListHeads.answer_max(SLOTS + 1, KINDS) > RESPONSE_MAX);
        assert!(2 + Opcode::GetPipelineCatalog.answer_max(SLOTS, KINDS + 1) > RESPONSE_MAX);
    }

    #[test]
    fn an_answer_that_cannot_be_composed_says_so() {
        let mut out = [0u8; RESPONSE_MAX];
        assert_eq!(composed(Ok(4)), (Status::Ok, 4));
        assert_eq!(composed(Err(Error::NoRoom)), (Status::Hardware, 0));
        let n = compose_response(0x82, Status::Ok, &[7; PAYLOAD_MAX], &mut out);
        assert_eq!(n, RESPONSE_MAX);
        assert_eq!(&out[..2], &[0x82, 0]);
        // One byte too many is answered status 5, never cut short or 2.
        let n = compose_response(0x82, Status::Ok, &[7; PAYLOAD_MAX + 1], &mut out);
        assert_eq!(&out[..n], &[0x82, Status::Hardware as u8]);
    }

    #[test]
    fn the_name_is_composed_within_what_the_air_carries() {
        use super::name::*;
        let mut out = [0u8; 64];
        let n = compose(b"Ada", b"Blue", b"IntoMind One", &mut out).unwrap();
        assert_eq!(&out[..n], b"Ada's Blue IntoMind One");
        assert_eq!(n, composed_len(3, 4, 12));
        let n = compose(b"", b"Blue", b"IntoMind One", &mut out).unwrap();
        assert_eq!(&out[..n], b"Blue IntoMind One");
        let n = compose(b"Ada", b"", b"IntoMind One", &mut out).unwrap();
        assert_eq!(&out[..n], b"Ada's IntoMind One");
        let n = compose(b"", b"", b"IntoMind One", &mut out).unwrap();
        assert_eq!(&out[..n], b"IntoMind One");
        // The limit is on the result: a long name leaves less room for an adjective.
        assert!(acceptable(b"Beatrice", b"Green", b"IntoMind One"), "8 + 5 makes 29 exactly");
        assert!(!acceptable(b"Beatrice", b"Purple", b"IntoMind One"), "8 + 6 makes 30");
        assert!(acceptable(b"Christopher Ro", b"", b"IntoMind One"), "a name alone may use 14");
        assert!(!acceptable(b"Christopher Rob", b"", b"IntoMind One"));
        assert!(acceptable("José".as_bytes(), b"", b"IntoMind One"), "UTF-8 is counted in bytes and allowed");
        // Parts are checked as parts.
        assert!(!valid_part(b" Ada"));
        assert!(!valid_part(b"Ada "));
        assert!(!valid_part(b"Kr\x01s"));
        assert!(!valid_part(b"Kr\x7Fs"));
        assert!(!valid_part(&[0xC3, 0x28]), "not UTF-8");
        assert!(valid_part(b""));
        assert!(valid_part(b"O'Neil"));
        // The wire form of the two parts.
        let mut buf = [0u8; 64];
        let n = encode_parts(b"Ada", b"Blue", &mut buf).unwrap();
        assert_eq!(&buf[..n], b"\x03Ada\x04Blue");
        assert_eq!(parse_parts(&buf[..n]).unwrap(), (&b"Ada"[..], &b"Blue"[..]));
        assert_eq!(parse_parts(b"\x00\x00").unwrap(), (&b""[..], &b""[..]));
        assert_eq!(parse_parts(b"\x04Ada\x04Blue"), Err(Error::Invalid));
        assert_eq!(parse_parts(b"\x03Ada\x03Blue"), Err(Error::Invalid));
        assert_eq!(parse_parts(b"\x03Ada"), Err(Error::Invalid));
        assert_eq!(parse_parts(b""), Err(Error::Truncated));
        // Composition refuses a buffer that is too small rather than cutting.
        let mut small = [0u8; 10];
        assert_eq!(compose(b"Ada", b"Blue", b"IntoMind One", &mut small), Err(Error::NoRoom));
    }

    #[test]
    fn the_model_interval_and_the_name_are_payload_opcodes() {
        use super::interval::*;
        assert_eq!(parse_seconds(&[60, 0]).unwrap(), 60);
        assert_eq!(parse_seconds(&[0, 1]).unwrap(), 256);
        assert_eq!(parse_seconds(&[1]), Err(Error::Invalid));
        assert_eq!(parse_seconds(&[]), Err(Error::Invalid));
        let mi = ModelInterval { interval_s: 60, minimum_s: 4 };
        let mut buf = [0u8; ModelInterval::LEN];
        assert_eq!(mi.encode(&mut buf).unwrap(), 4);
        assert_eq!(ModelInterval::parse(&buf).unwrap(), mi);
        assert_eq!(ModelInterval::parse(&buf[..3]), Err(Error::Truncated));
        for (op, code) in [(Opcode::SetModelInterval, 0x88), (Opcode::GetModelInterval, 0x89), (Opcode::GetName, 0x47), (Opcode::SetName, 0x48)] {
            assert_eq!(Opcode::from_u8(code), Some(op));
            assert!(op.new_in_1_3());
            assert!(!op.new_in_1_2() && !op.new_in_1_1());
        }
        assert!(Opcode::SetModelInterval.takes_payload() && Opcode::SetName.takes_payload());
        assert!(!Opcode::GetModelInterval.takes_payload() && !Opcode::GetName.takes_payload());
        // A payload opcode's request parses whatever follows the opcode;
        // the payload's own parser judges it.
        assert_eq!(Request::parse(&[0x88, 60, 0]).unwrap(), Request { opcode: Opcode::SetModelInterval, arg: None });
        assert_eq!(Request::parse(&[0x89]).unwrap(), Request { opcode: Opcode::GetModelInterval, arg: None });
        assert_eq!(Request::parse(&[0x89, 1]), Err(Error::Invalid));
    }
}
