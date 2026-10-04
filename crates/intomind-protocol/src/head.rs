//! User heads: a weights blob mapping the model's embedding to a few
//! outputs. This module knows the blob layout and the reference output
//! arithmetic. Hashing is the caller's job (the codec has no dependencies),
//! and `HeadLayout::hashed` is the exact byte range to hash.
//!
//! A head consumes the embedding as signed sixteen bit integers, in units
//! of `1 / EMBED_SCALE`. The device quantizes its own embedding that way
//! before running a head, and the published runtime does the same, so a
//! head trained on a host produces the same numbers on the device. Fixing
//! the scale here is what makes a head a portable blob rather than
//! something that has to be recalibrated per device.

use crate::Error;

/// Counts per unit of embedding. The embedding comes out of a layer
/// normalization, so it sits within a few units of zero, and this scale
/// covers plus or minus eight with a resolution of a quarter of a
/// thousandth. Changing it would change every head ever trained, so it is
/// part of the contract.
pub const EMBED_SCALE: f32 = 4096.0;

/// Quantize an embedding the way a head expects to receive it. Values
/// beyond the representable range saturate rather than wrapping.
pub fn quantize_embedding(embedding: &[f32], out: &mut [i16]) -> Result<(), Error> {
    if out.len() < embedding.len() {
        return Err(Error::NoRoom);
    }
    for (o, v) in out.iter_mut().zip(embedding) {
        let scaled = v * EMBED_SCALE;
        *o = if scaled >= i16::MAX as f32 {
            i16::MAX
        } else if scaled <= i16::MIN as f32 {
            i16::MIN
        } else {
            // Round half away from zero, which is what every framework's
            // rounding does for the values a head will ever see.
            let r = if scaled >= 0.0 { scaled + 0.5 } else { scaled - 0.5 };
            r as i16
        };
    }
    Ok(())
}

pub const MAGIC: [u8; 4] = *b"IMHD";
/// The 1.0 format: a 32 byte header.
pub const FORMAT_VERSION: u8 = 1;
/// The 1.3 format: the same header followed by the eight byte id of the
/// encoder whose embeddings the head was trained on, all zero when the
/// trainer did not say. The device stores and reports it and never judges
/// it; a host may warn on a mismatch, and nothing blocks.
pub const FORMAT_VERSION_2: u8 = 2;
pub const KIND_LINEAR: u8 = 1;
pub const HEADER_LEN: usize = 32;
pub const HEADER_LEN_2: usize = 40;
pub const HASH_LEN: usize = 32;
pub const NAME_LEN: usize = 16;

/// The header length of a format version, or 0 for a version this codec
/// does not know.
pub const fn header_len(version: u8) -> usize {
    match version {
        FORMAT_VERSION => HEADER_LEN,
        FORMAT_VERSION_2 => HEADER_LEN_2,
        _ => 0,
    }
}

/// Total blob length for a head of the given format and shape.
pub const fn blob_len(version: u8, in_dim: u16, out_dim: u16) -> usize {
    let o = out_dim as usize;
    header_len(version) + o * in_dim as usize + 4 * o + 4 * o + HASH_LEN
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadHeader {
    pub version: u8,
    pub kind: u8,
    pub in_dim: u16,
    pub out_dim: u16,
    pub name: [u8; NAME_LEN],
    /// Format 2 only; all zero in format 1 and when the trainer did not say.
    pub encoder_id: [u8; 8],
}

impl HeadHeader {
    pub fn encode(&self, out: &mut [u8]) -> Result<usize, Error> {
        let n = header_len(self.version);
        if n == 0 {
            return Err(Error::Invalid);
        }
        if out.len() < n {
            return Err(Error::NoRoom);
        }
        out[0..4].copy_from_slice(&MAGIC);
        out[4] = self.version;
        out[5] = self.kind;
        out[6..8].copy_from_slice(&self.in_dim.to_le_bytes());
        out[8..10].copy_from_slice(&self.out_dim.to_le_bytes());
        out[10..26].copy_from_slice(&self.name);
        out[26..32].fill(0);
        if self.version == FORMAT_VERSION_2 {
            out[32..40].copy_from_slice(&self.encoder_id);
        }
        Ok(n)
    }

    fn parse(b: &[u8]) -> Result<Self, HeadError> {
        if b.len() < HEADER_LEN {
            return Err(HeadError::Length);
        }
        if b[0..4] != MAGIC || header_len(b[4]) == 0 || b[5] != KIND_LINEAR {
            return Err(HeadError::Malformed);
        }
        let version = b[4];
        if b.len() < header_len(version) {
            return Err(HeadError::Length);
        }
        let mut name = [0u8; NAME_LEN];
        name.copy_from_slice(&b[10..26]);
        let mut encoder_id = [0u8; 8];
        if version == FORMAT_VERSION_2 {
            encoder_id.copy_from_slice(&b[32..40]);
        }
        Ok(HeadHeader {
            version,
            kind: b[5],
            in_dim: u16::from_le_bytes([b[6], b[7]]),
            out_dim: u16::from_le_bytes([b[8], b[9]]),
            name,
            encoder_id,
        })
    }
}

/// Why a blob was refused. Maps onto the update service's verify results:
/// `Malformed` and `Length` to Malformed (2), `Shape` and `Width` to
/// HeadShape (8). `Width` is the one refusal the contract allows on a
/// sound head: its input is not the loaded encoder's width, so it cannot
/// be computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadError {
    Malformed,
    Shape,
    Width,
    Length,
}

/// A validated view of a head blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadLayout<'a> {
    pub header: HeadHeader,
    weights: &'a [u8],
    bias: &'a [u8],
    scale: &'a [u8],
    /// Every byte before the hash, which is what the hash covers.
    pub hashed: &'a [u8],
    pub hash: &'a [u8],
}

/// Check a blob against this device's embedding width and output limit and
/// return its layout. The hash is not checked here.
pub fn layout(blob: &[u8], embed_dim: u16, max_outputs: u8) -> Result<HeadLayout<'_>, HeadError> {
    let header = HeadHeader::parse(blob)?;
    if header.out_dim == 0 || header.out_dim > max_outputs as u16 {
        return Err(HeadError::Shape);
    }
    if header.in_dim != embed_dim {
        return Err(HeadError::Width);
    }
    layout_of(blob, header)
}

/// The layout of a head of any input width: what a device needs to list a
/// stored head whose width is not the loaded encoder's. Such a head is
/// described and never run.
pub fn layout_any(blob: &[u8], max_outputs: u8) -> Result<HeadLayout<'_>, HeadError> {
    let header = HeadHeader::parse(blob)?;
    if header.out_dim == 0 || header.out_dim > max_outputs as u16 {
        return Err(HeadError::Shape);
    }
    layout_of(blob, header)
}

fn layout_of(blob: &[u8], header: HeadHeader) -> Result<HeadLayout<'_>, HeadError> {
    let want = blob_len(header.version, header.in_dim, header.out_dim);
    if blob.len() != want {
        return Err(HeadError::Length);
    }
    let o = header.out_dim as usize;
    let h_len = header_len(header.version);
    let w_end = h_len + o * header.in_dim as usize;
    let b_end = w_end + 4 * o;
    let s_end = b_end + 4 * o;
    Ok(HeadLayout {
        header,
        weights: &blob[h_len..w_end],
        bias: &blob[w_end..b_end],
        scale: &blob[b_end..s_end],
        hashed: &blob[..s_end],
        hash: &blob[s_end..],
    })
}

impl HeadLayout<'_> {
    pub fn head_id(&self) -> [u8; 8] {
        let mut id = [0u8; 8];
        id.copy_from_slice(&self.hash[..8]);
        id
    }

    pub fn weight(&self, output: usize, input: usize) -> i8 {
        self.weights[output * self.header.in_dim as usize + input] as i8
    }

    pub fn bias(&self, output: usize) -> i32 {
        let b = &self.bias[output * 4..output * 4 + 4];
        i32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }

    pub fn scale(&self, output: usize) -> f32 {
        let b = &self.scale[output * 4..output * 4 + 4];
        f32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }

    /// The reference output arithmetic:
    /// `out[o] = scale[o] × (bias[o] + Σ_i W[o][i] × e[i])`, accumulated
    /// exactly in 64 bits. `embedding.len()` must be `in_dim` and
    /// `out.len()` at least `out_dim`.
    pub fn evaluate(&self, embedding: &[i16], out: &mut [f32]) -> Result<(), Error> {
        let (i_dim, o_dim) = (self.header.in_dim as usize, self.header.out_dim as usize);
        if embedding.len() != i_dim {
            return Err(Error::Invalid);
        }
        if out.len() < o_dim {
            return Err(Error::NoRoom);
        }
        for o in 0..o_dim {
            let mut acc: i64 = self.bias(o) as i64;
            let row = &self.weights[o * i_dim..(o + 1) * i_dim];
            for (w, e) in row.iter().zip(embedding) {
                acc += (*w as i8) as i64 * *e as i64;
            }
            out[o] = self.scale(o) * acc as f32;
        }
        Ok(())
    }
}

/// Write a complete blob except its hash. Returns the length written, which
/// is `blob_len(in_dim, out_dim)`. The caller hashes `out[..len - 32]` and
/// stores the digest in `out[len - 32..]`.
pub fn encode_unhashed(
    header: &HeadHeader,
    weights: &[i8],
    bias: &[i32],
    scale: &[f32],
    out: &mut [u8],
) -> Result<usize, Error> {
    let (i_dim, o_dim) = (header.in_dim as usize, header.out_dim as usize);
    if weights.len() != i_dim * o_dim || bias.len() != o_dim || scale.len() != o_dim {
        return Err(Error::Invalid);
    }
    let n = blob_len(header.version, header.in_dim, header.out_dim);
    if out.len() < n {
        return Err(Error::NoRoom);
    }
    header.encode(out)?;
    let mut p = header_len(header.version);
    for w in weights {
        out[p] = *w as u8;
        p += 1;
    }
    for b in bias {
        out[p..p + 4].copy_from_slice(&b.to_le_bytes());
        p += 4;
    }
    for s in scale {
        out[p..p + 4].copy_from_slice(&s.to_le_bytes());
        p += 4;
    }
    out[p..n].fill(0);
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedding_quantizes_to_the_published_scale() {
        let mut out = [0i16; 6];
        quantize_embedding(&[0.0, 1.0, -1.0, 0.000_244_140_625, 100.0, -100.0], &mut out).unwrap();
        assert_eq!(out[0], 0);
        assert_eq!(out[1], 4096);
        assert_eq!(out[2], -4096);
        assert_eq!(out[3], 1, "one count is a quarter of a thousandth");
        assert_eq!(out[4], i16::MAX, "beyond the range it saturates, never wraps");
        assert_eq!(out[5], i16::MIN);
        // Rounding is to nearest, away from zero at a half.
        let mut out = [0i16; 4];
        quantize_embedding(&[0.000_1, -0.000_1, 0.5 / EMBED_SCALE, -0.5 / EMBED_SCALE], &mut out).unwrap();
        assert_eq!(out, [0, 0, 1, -1]);
        assert_eq!(quantize_embedding(&[0.0; 4], &mut [0i16; 3]), Err(Error::NoRoom));
    }

    #[test]
    fn sizes_follow_the_spec() {
        assert_eq!(blob_len(1, 2, 1), 32 + 2 + 4 + 4 + 32);
        assert_eq!(blob_len(1, 96, 2), 32 + 192 + 8 + 8 + 32);
        assert_eq!(blob_len(2, 96, 2), 40 + 192 + 8 + 8 + 32);
        // The IntoMind One's largest head fits its 4096-byte slot in either format.
        assert_eq!(blob_len(1, 96, 32), 3392);
        assert_eq!(blob_len(2, 96, 32), 3400);
        assert!(blob_len(2, 96, 32) <= 4096);
        assert_eq!(header_len(3), 0, "an unknown format has no header length");
    }

    #[test]
    fn a_format_two_head_carries_the_encoder_it_was_trained_beside() {
        let header = HeadHeader { version: FORMAT_VERSION_2, kind: KIND_LINEAR, in_dim: 2, out_dim: 1, name: [0; NAME_LEN], encoder_id: [0xA5; 8] };
        let mut blob = [0u8; blob_len(2, 2, 1)];
        let n = encode_unhashed(&header, &[1, 1], &[0], &[1.0], &mut blob).unwrap();
        assert_eq!(n, 40 + 2 + 4 + 4 + 32);
        let l = layout(&blob, 2, 32).unwrap();
        assert_eq!(l.header, header);
        assert_eq!(l.header.encoder_id, [0xA5; 8]);
        assert_eq!(l.hashed.len(), n - 32);
        assert_eq!((l.weight(0, 0), l.weight(0, 1), l.bias(0)), (1, 1, 0));
        // A format 1 blob of the same head parses with no encoder id.
        let header1 = HeadHeader { version: FORMAT_VERSION, encoder_id: [0; 8], ..header };
        let mut blob1 = [0u8; blob_len(1, 2, 1)];
        encode_unhashed(&header1, &[1, 1], &[0], &[1.0], &mut blob1).unwrap();
        assert_eq!(layout(&blob1, 2, 32).unwrap().header, header1);
    }

    #[test]
    fn layout_and_reference_arithmetic() {
        let mut name = [0u8; NAME_LEN];
        name[..4].copy_from_slice(b"test");
        let header = HeadHeader { version: FORMAT_VERSION, kind: KIND_LINEAR, in_dim: 2, out_dim: 2, name, encoder_id: [0; 8] };
        let mut blob = [0u8; blob_len(1, 2, 2)];
        let n = encode_unhashed(&header, &[1, -2, 3, 4], &[5, -6], &[0.5, 2.0], &mut blob).unwrap();
        assert_eq!(n, blob.len());
        blob[n - 32..].copy_from_slice(&[0xEE; 32]);
        let l = layout(&blob, 2, 32).unwrap();
        assert_eq!(l.header, header);
        assert_eq!(l.hashed.len(), n - 32);
        assert_eq!(l.head_id(), [0xEE; 8]);
        assert_eq!((l.weight(0, 0), l.weight(0, 1), l.weight(1, 0), l.weight(1, 1)), (1, -2, 3, 4));
        assert_eq!((l.bias(0), l.bias(1)), (5, -6));
        let mut out = [0f32; 2];
        l.evaluate(&[3, 4], &mut out).unwrap();
        // out0 = 0.5 × (5 + 1×3 + (−2)×4) = 0.5 × 0 = 0
        // out1 = 2.0 × (−6 + 3×3 + 4×4) = 2.0 × 19 = 38
        assert_eq!(out, [0.0, 38.0]);
        assert_eq!(l.evaluate(&[3], &mut out), Err(Error::Invalid));
    }

    #[test]
    fn refusals() {
        let header = HeadHeader { version: FORMAT_VERSION, kind: KIND_LINEAR, in_dim: 2, out_dim: 1, name: [0; NAME_LEN], encoder_id: [0; 8] };
        let mut blob = [0u8; blob_len(1, 2, 1)];
        encode_unhashed(&header, &[1, 1], &[0], &[1.0], &mut blob).unwrap();
        assert!(layout(&blob, 2, 32).is_ok());
        // An embedding width that is not the encoder's is its own refusal;
        // out_dim over the limit is a shape error.
        assert_eq!(layout(&blob, 96, 32).unwrap_err(), HeadError::Width);
        assert_eq!(layout(&blob, 2, 0).unwrap_err(), HeadError::Shape);
        // Length must equal the shape's length exactly.
        assert_eq!(layout(&blob[..blob.len() - 1], 2, 32).unwrap_err(), HeadError::Length);
        // Wrong magic or an unknown kind is malformed.
        let mut bad = blob;
        bad[0] = b'x';
        assert_eq!(layout(&bad, 2, 32).unwrap_err(), HeadError::Malformed);
        let mut bad = blob;
        bad[5] = 2;
        assert_eq!(layout(&bad, 2, 32).unwrap_err(), HeadError::Malformed);
        // Accumulation is exact at the largest allowed magnitudes.
        let header = HeadHeader { version: FORMAT_VERSION, kind: KIND_LINEAR, in_dim: 96, out_dim: 1, name: [0; NAME_LEN], encoder_id: [0; 8] };
        let mut blob = [0u8; blob_len(1, 96, 1)];
        encode_unhashed(&header, &[-128; 96], &[i32::MIN], &[1.0], &mut blob).unwrap();
        let l = layout(&blob, 96, 32).unwrap();
        let mut out = [0f32; 1];
        l.evaluate(&[i16::MIN; 96], &mut out).unwrap();
        let expect = (i32::MIN as i64 + 96 * 128 * 32768) as f32;
        assert_eq!(out[0], expect);
    }
}
