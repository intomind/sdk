//! Update service: Update Control (write, indicate) and Update Data (write
//! without response). Carries application images, weights images, and user
//! heads. Images travel in an envelope whose first 32 bytes are plain and
//! whose payload is opaque to the host.

use crate::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Op {
    Start = 0x01,
    Query = 0x02,
    Finish = 0x03,
    Activate = 0x04,
    Abort = 0x05,
}

impl Op {
    pub fn from_u8(b: u8) -> Option<Self> {
        Some(match b {
            0x01 => Self::Start,
            0x02 => Self::Query,
            0x03 => Self::Finish,
            0x04 => Self::Activate,
            0x05 => Self::Abort,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Status {
    Ok = 0,
    InvalidArg = 1,
    Unsupported = 2,
    Busy = 3,
    NoTransfer = 4,
    Flash = 5,
    Verify = 6,
}

impl Status {
    pub fn from_u8(b: u8) -> Option<Self> {
        Some(match b {
            0 => Self::Ok,
            1 => Self::InvalidArg,
            2 => Self::Unsupported,
            3 => Self::Busy,
            4 => Self::NoTransfer,
            5 => Self::Flash,
            6 => Self::Verify,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Target {
    App = 1,
    Weights = 2,
    Head = 3,
}

impl Target {
    pub fn from_u8(b: u8) -> Option<Self> {
        Some(match b {
            1 => Self::App,
            2 => Self::Weights,
            3 => Self::Head,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum State {
    Idle = 0,
    Receiving = 1,
    Complete = 2,
    Failed = 3,
}

impl State {
    pub fn from_u8(b: u8) -> Option<Self> {
        Some(match b {
            0 => Self::Idle,
            1 => Self::Receiving,
            2 => Self::Complete,
            3 => Self::Failed,
            _ => return None,
        })
    }
}

/// FINISH's `verify_result`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum VerifyResult {
    Verified = 0,
    Length = 1,
    Malformed = 2,
    Target = 3,
    Slot = 4,
    Hash = 5,
    Signature = 6,
    SecurityCounter = 7,
    HeadShape = 8,
    Flash = 9,
    KeyId = 10,
}

impl VerifyResult {
    pub fn from_u8(b: u8) -> Option<Self> {
        Some(match b {
            0 => Self::Verified,
            1 => Self::Length,
            2 => Self::Malformed,
            3 => Self::Target,
            4 => Self::Slot,
            5 => Self::Hash,
            6 => Self::Signature,
            7 => Self::SecurityCounter,
            8 => Self::HeadShape,
            9 => Self::Flash,
            10 => Self::KeyId,
            _ => return None,
        })
    }
}

/// A parsed Update Control write, device side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Request {
    Start(Start),
    Query,
    Finish,
    Activate,
    Abort,
}

/// START request body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Start {
    pub target: Target,
    /// Head slot 1..head_slots for a head, 0 otherwise.
    pub slot: u8,
    pub total_len: u32,
    pub transfer_id: [u8; 8],
}

impl Start {
    /// Body length after the op byte.
    pub const BODY_LEN: usize = 14;
    /// Whole request length.
    pub const LEN: usize = 1 + Self::BODY_LEN;

    fn parse_body(b: &[u8]) -> Result<Self, Error> {
        if b.len() != Self::BODY_LEN {
            return Err(Error::Invalid);
        }
        let target = Target::from_u8(b[0]).ok_or(Error::Invalid)?;
        let mut transfer_id = [0u8; 8];
        transfer_id.copy_from_slice(&b[6..14]);
        Ok(Start {
            target,
            slot: b[1],
            total_len: u32::from_le_bytes([b[2], b[3], b[4], b[5]]),
            transfer_id,
        })
    }

    pub fn encode(&self, out: &mut [u8]) -> Result<usize, Error> {
        if out.len() < Self::LEN {
            return Err(Error::NoRoom);
        }
        out[0] = Op::Start as u8;
        out[1] = self.target as u8;
        out[2] = self.slot;
        out[3..7].copy_from_slice(&self.total_len.to_le_bytes());
        out[7..15].copy_from_slice(&self.transfer_id);
        Ok(Self::LEN)
    }
}

impl Request {
    /// Parse a write. Ops without a body must arrive as exactly one byte.
    pub fn parse(buf: &[u8]) -> Result<Self, Error> {
        let (&op, body) = buf.split_first().ok_or(Error::Truncated)?;
        let op = Op::from_u8(op).ok_or(Error::Invalid)?;
        match op {
            Op::Start => Start::parse_body(body).map(Request::Start),
            _ if !body.is_empty() => Err(Error::Invalid),
            Op::Query => Ok(Request::Query),
            Op::Finish => Ok(Request::Finish),
            Op::Activate => Ok(Request::Activate),
            Op::Abort => Ok(Request::Abort),
        }
    }

    pub fn op(&self) -> Op {
        match self {
            Request::Start(_) => Op::Start,
            Request::Query => Op::Query,
            Request::Finish => Op::Finish,
            Request::Activate => Op::Activate,
            Request::Abort => Op::Abort,
        }
    }

    pub fn encode(&self, out: &mut [u8]) -> Result<usize, Error> {
        match self {
            Request::Start(s) => s.encode(out),
            other => {
                if out.is_empty() {
                    return Err(Error::NoRoom);
                }
                out[0] = other.op() as u8;
                Ok(1)
            }
        }
    }
}

/// START response payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StartResponse {
    /// For an application: 0 = slot A, 1 = slot B will receive the image.
    pub target_slot: u8,
    pub chunk_max: u16,
    pub resume_offset: u32,
}

impl StartResponse {
    pub const LEN: usize = 7;

    pub fn parse(b: &[u8]) -> Result<Self, Error> {
        if b.len() < Self::LEN {
            return Err(Error::Truncated);
        }
        Ok(StartResponse {
            target_slot: b[0],
            chunk_max: u16::from_le_bytes([b[1], b[2]]),
            resume_offset: u32::from_le_bytes([b[3], b[4], b[5], b[6]]),
        })
    }

    pub fn encode(&self, out: &mut [u8]) -> Result<usize, Error> {
        if out.len() < Self::LEN {
            return Err(Error::NoRoom);
        }
        out[0] = self.target_slot;
        out[1..3].copy_from_slice(&self.chunk_max.to_le_bytes());
        out[3..7].copy_from_slice(&self.resume_offset.to_le_bytes());
        Ok(Self::LEN)
    }
}

/// QUERY response payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryResponse {
    pub state: State,
    pub offset: u32,
    pub crc32: u32,
}

impl QueryResponse {
    pub const LEN: usize = 9;

    pub fn parse(b: &[u8]) -> Result<Self, Error> {
        if b.len() < Self::LEN {
            return Err(Error::Truncated);
        }
        Ok(QueryResponse {
            state: State::from_u8(b[0]).ok_or(Error::Invalid)?,
            offset: u32::from_le_bytes([b[1], b[2], b[3], b[4]]),
            crc32: u32::from_le_bytes([b[5], b[6], b[7], b[8]]),
        })
    }

    pub fn encode(&self, out: &mut [u8]) -> Result<usize, Error> {
        if out.len() < Self::LEN {
            return Err(Error::NoRoom);
        }
        out[0] = self.state as u8;
        out[1..5].copy_from_slice(&self.offset.to_le_bytes());
        out[5..9].copy_from_slice(&self.crc32.to_le_bytes());
        Ok(Self::LEN)
    }
}

/// Encode a response `[op, status, payload...]`.
pub fn encode_response(op: Op, status: Status, payload: &[u8], out: &mut [u8]) -> Result<usize, Error> {
    let n = 2 + payload.len();
    if out.len() < n {
        return Err(Error::NoRoom);
    }
    out[0] = op as u8;
    out[1] = status as u8;
    out[2..n].copy_from_slice(payload);
    Ok(n)
}

/// A parsed response, host side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Response<'a> {
    pub op: Op,
    pub status: Status,
    pub payload: &'a [u8],
}

impl<'a> Response<'a> {
    pub fn parse(buf: &'a [u8]) -> Result<Self, Error> {
        if buf.len() < 2 {
            return Err(Error::Truncated);
        }
        Ok(Response {
            op: Op::from_u8(buf[0]).ok_or(Error::Invalid)?,
            status: Status::from_u8(buf[1]).ok_or(Error::Invalid)?,
            payload: &buf[2..],
        })
    }
}

/// The plain 32-byte head of an image envelope. Everything after it is the
/// encrypted payload, opaque to the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Envelope {
    pub target: Target,
    /// Application: 0 = built for slot A, 1 = built for slot B. Weights:
    /// `SLOT_LINK_NONE`.
    pub slot_link: u8,
    pub key_id: u8,
    pub nonce: [u8; 16],
    pub plain_len: u32,
}

pub const ENVELOPE_MAGIC: [u8; 4] = *b"IMUP";
pub const ENVELOPE_VERSION: u8 = 1;
pub const ENVELOPE_LEN: usize = 32;
pub const SLOT_LINK_NONE: u8 = 0xFF;

impl Envelope {
    pub fn parse(b: &[u8]) -> Result<Self, Error> {
        if b.len() < ENVELOPE_LEN {
            return Err(Error::Truncated);
        }
        if b[0..4] != ENVELOPE_MAGIC || b[4] != ENVELOPE_VERSION {
            return Err(Error::Invalid);
        }
        let target = match Target::from_u8(b[5]) {
            Some(Target::App) => Target::App,
            Some(Target::Weights) => Target::Weights,
            _ => return Err(Error::Invalid),
        };
        let slot_link = b[6];
        match target {
            Target::App if slot_link > 1 => return Err(Error::Invalid),
            Target::Weights if slot_link != SLOT_LINK_NONE => return Err(Error::Invalid),
            _ => {}
        }
        let mut nonce = [0u8; 16];
        nonce.copy_from_slice(&b[8..24]);
        Ok(Envelope {
            target,
            slot_link,
            key_id: b[7],
            nonce,
            plain_len: u32::from_le_bytes([b[24], b[25], b[26], b[27]]),
        })
    }

    pub fn encode(&self, out: &mut [u8]) -> Result<usize, Error> {
        if out.len() < ENVELOPE_LEN {
            return Err(Error::NoRoom);
        }
        out[0..4].copy_from_slice(&ENVELOPE_MAGIC);
        out[4] = ENVELOPE_VERSION;
        out[5] = self.target as u8;
        out[6] = self.slot_link;
        out[7] = self.key_id;
        out[8..24].copy_from_slice(&self.nonce);
        out[24..28].copy_from_slice(&self.plain_len.to_le_bytes());
        out[28..32].fill(0);
        Ok(ENVELOPE_LEN)
    }

    /// Total transfer length for this envelope.
    pub fn total_len(&self) -> u32 {
        ENVELOPE_LEN as u32 + self.plain_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_round_trip_is_fifteen_bytes() {
        let s = Start { target: Target::Head, slot: 2, total_len: 3392, transfer_id: [0xAB; 8] };
        let mut buf = [0u8; Start::LEN];
        assert_eq!(s.encode(&mut buf).unwrap(), 15);
        assert_eq!(Request::parse(&buf).unwrap(), Request::Start(s));
    }

    #[test]
    fn arity_is_enforced() {
        assert_eq!(Request::parse(&[0x02]), Ok(Request::Query));
        assert_eq!(Request::parse(&[0x02, 0x00]), Err(Error::Invalid));
        assert_eq!(Request::parse(&[0x01, 1, 0, 0, 0]), Err(Error::Invalid));
        assert_eq!(Request::parse(&[0x09]), Err(Error::Invalid));
        assert_eq!(Request::parse(&[]), Err(Error::Truncated));
        // Unknown target inside an otherwise well formed START.
        let mut buf = [0u8; Start::LEN];
        Start { target: Target::App, slot: 0, total_len: 1, transfer_id: [0; 8] }.encode(&mut buf).unwrap();
        buf[1] = 7;
        assert_eq!(Request::parse(&buf), Err(Error::Invalid));
    }

    #[test]
    fn responses_round_trip() {
        let sr = StartResponse { target_slot: 1, chunk_max: 244, resume_offset: 4096 };
        let mut p = [0u8; StartResponse::LEN];
        sr.encode(&mut p).unwrap();
        let mut buf = [0u8; 2 + StartResponse::LEN];
        let n = encode_response(Op::Start, Status::Ok, &p, &mut buf).unwrap();
        let r = Response::parse(&buf[..n]).unwrap();
        assert_eq!((r.op, r.status), (Op::Start, Status::Ok));
        assert_eq!(StartResponse::parse(r.payload).unwrap(), sr);

        let q = QueryResponse { state: State::Receiving, offset: 123_456, crc32: 0xCBF4_3926 };
        let mut p = [0u8; QueryResponse::LEN];
        q.encode(&mut p).unwrap();
        assert_eq!(QueryResponse::parse(&p).unwrap(), q);
        p[0] = 9;
        assert_eq!(QueryResponse::parse(&p), Err(Error::Invalid));

        let mut buf = [0u8; 3];
        let n = encode_response(Op::Finish, Status::Verify, &[VerifyResult::Signature as u8], &mut buf).unwrap();
        let r = Response::parse(&buf[..n]).unwrap();
        assert_eq!(r.status, Status::Verify);
        assert_eq!(VerifyResult::from_u8(r.payload[0]), Some(VerifyResult::Signature));
        assert_eq!(VerifyResult::from_u8(11), None);
    }

    #[test]
    fn envelope_round_trip_and_rules() {
        let e = Envelope { target: Target::App, slot_link: 1, key_id: 1, nonce: [5; 16], plain_len: 200_000 };
        let mut buf = [0u8; ENVELOPE_LEN];
        assert_eq!(e.encode(&mut buf).unwrap(), 32);
        assert_eq!(&buf[0..4], b"IMUP");
        assert_eq!(Envelope::parse(&buf).unwrap(), e);
        assert_eq!(e.total_len(), 200_032);

        // Wrong magic, wrong version, a head is not an envelope target, and
        // slot links out of range are all malformed.
        let mut bad = buf;
        bad[0] = b'X';
        assert_eq!(Envelope::parse(&bad), Err(Error::Invalid));
        let mut bad = buf;
        bad[4] = 2;
        assert_eq!(Envelope::parse(&bad), Err(Error::Invalid));
        let mut bad = buf;
        bad[5] = Target::Head as u8;
        assert_eq!(Envelope::parse(&bad), Err(Error::Invalid));
        let mut bad = buf;
        bad[6] = 2;
        assert_eq!(Envelope::parse(&bad), Err(Error::Invalid));
        let w = Envelope { target: Target::Weights, slot_link: SLOT_LINK_NONE, key_id: 1, nonce: [0; 16], plain_len: 1 };
        w.encode(&mut buf).unwrap();
        assert_eq!(Envelope::parse(&buf).unwrap(), w);
        buf[6] = 0;
        assert_eq!(Envelope::parse(&buf), Err(Error::Invalid));
        assert_eq!(Envelope::parse(&buf[..31]), Err(Error::Truncated));
    }
}
