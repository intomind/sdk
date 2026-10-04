//! The weights blob, and the encoder that runs from it.
//!
//! The blob is read in place from flash. Nothing is copied except one
//! dequantized weight row at a time, and every row is reused across the
//! whole sequence, so the conversion is amortized.

use crate::math::{dot, f32_at, gelu, layer_norm, softmax};
use crate::prep::normalize_window;

/// Where a window comes from. The encoder reads one channel at a time and
/// normalizes it in its own scratch, so nothing has to hold the whole
/// window contiguously and the feeder's buffer is the only copy.
pub trait Window {
    fn channels(&self) -> usize;
    /// Write this channel's samples, oldest first, in microvolts.
    fn read_channel(&self, channel: usize, out: &mut [f32]);
}

/// A window that is already laid out channel major. For hosts and tests.
pub struct Slice<'a> {
    pub data: &'a [f32],
    pub channels: usize,
    pub window_samples: usize,
}

impl Window for Slice<'_> {
    fn channels(&self) -> usize {
        self.channels
    }
    fn read_channel(&self, channel: usize, out: &mut [f32]) {
        let from = channel * self.window_samples;
        out[..self.window_samples].copy_from_slice(&self.data[from..from + self.window_samples]);
    }
}

pub const MAGIC: [u8; 4] = *b"IMW1";
/// The protocol's `pipeline::input_class::TIME_DOMAIN`, restated here so
/// this crate stays free of the protocol crate.
pub const INPUT_CLASS_TIME_DOMAIN: u8 = 1 << 0;
pub const FORMAT_VERSION: u8 = 1;
pub const HEADER_LEN: usize = 32;
/// Tokens whose feed forward hidden values are held at once. Larger is
/// faster and needs more memory: each weight row is dequantized once per
/// tile rather than once per token.
pub const FF_TILE: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Not a weights blob, or a format this firmware does not know.
    Malformed,
    /// The blob ends before its own tensors do.
    Truncated,
    /// The caller's window, workspace, or output is the wrong size.
    BadShape,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub n_layers: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub n_heads: usize,
    pub patch: usize,
    pub n_time: usize,
    pub native_sps: u32,
    /// Names the trained weights, and travels with every prediction.
    pub encoder_id: [u8; 8],
    /// Offset of a built-in head inside the blob, or zero for none.
    pub head_offset: usize,
    /// The signal classes this model takes, as the protocol's
    /// `pipeline::input_class` bits. Byte 18 of the header; a blob that
    /// leaves it zero, as every blob before this field existed does,
    /// declares the time domain.
    pub input_classes: u8,
}

impl Header {
    pub fn window_samples(&self) -> usize {
        self.patch * self.n_time
    }

    pub fn d_head(&self) -> usize {
        self.d_model / self.n_heads
    }
}

fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

/// One int8 matrix with a scale per output row, and its float bias.
#[derive(Debug, Clone, Copy)]
struct Linear<'a> {
    q: &'a [u8],
    scale: &'a [u8],
    bias: &'a [u8],
    #[allow(dead_code)]
    out_dim: usize,
    in_dim: usize,
}

impl<'a> Linear<'a> {
    const fn len(out_dim: usize, in_dim: usize) -> usize {
        out_dim * in_dim + 8 * out_dim
    }

    fn take(blob: &'a [u8], at: &mut usize, out_dim: usize, in_dim: usize) -> Result<Linear<'a>, Error> {
        let q = take(blob, at, out_dim * in_dim)?;
        let scale = take(blob, at, out_dim * 4)?;
        let bias = take(blob, at, out_dim * 4)?;
        Ok(Linear { q, scale, bias, out_dim, in_dim })
    }

    /// Dequantize one output row into `dst`, which must hold `in_dim`.
    fn row(&self, o: usize, dst: &mut [f32]) {
        self.row_part(o, 0, self.in_dim, dst);
    }

    /// Dequantize part of one output row: inputs `from` to `from + n`.
    fn row_part(&self, o: usize, from: usize, n: usize, dst: &mut [f32]) {
        let s = f32_at(self.scale, o);
        let base = o * self.in_dim + from;
        for (d, &q) in dst[..n].iter_mut().zip(&self.q[base..base + n]) {
            *d = (q as i8) as f32 * s;
        }
    }

    fn bias(&self, o: usize) -> f32 {
        f32_at(self.bias, o)
    }
}

fn take<'a>(blob: &'a [u8], at: &mut usize, n: usize) -> Result<&'a [u8], Error> {
    let end = at.checked_add(n).ok_or(Error::Truncated)?;
    let s = blob.get(*at..end).ok_or(Error::Truncated)?;
    *at = end;
    Ok(s)
}

struct Layer<'a> {
    norm1_w: &'a [u8],
    norm1_b: &'a [u8],
    qkv: Linear<'a>,
    out: Linear<'a>,
    norm2_w: &'a [u8],
    norm2_b: &'a [u8],
    ff1: Linear<'a>,
    ff2: Linear<'a>,
}

/// The encoder, read in place from flash.
pub struct Model<'a> {
    pub header: Header,
    blob: &'a [u8],
    descriptor: &'a [u8],
    time_emb: &'a [u8],
    proj: Linear<'a>,
    layers_at: usize,
    layer_len: usize,
    final_norm_w: &'a [u8],
    final_norm_b: &'a [u8],
}

impl<'a> Model<'a> {
    pub fn parse(blob: &'a [u8]) -> Result<Model<'a>, Error> {
        if blob.len() < HEADER_LEN || blob[0..4] != MAGIC || blob[4] != FORMAT_VERSION {
            return Err(Error::Malformed);
        }
        let header = Header {
            n_layers: blob[5] as usize,
            d_model: u16_at(blob, 6) as usize,
            d_ff: u16_at(blob, 8) as usize,
            n_heads: blob[10] as usize,
            patch: u16_at(blob, 12) as usize,
            n_time: u16_at(blob, 14) as usize,
            native_sps: u16_at(blob, 16) as u32,
            encoder_id: blob[20..28].try_into().map_err(|_| Error::Malformed)?,
            head_offset: u32::from_le_bytes([blob[28], blob[29], blob[30], blob[31]]) as usize,
            input_classes: if blob[18] == 0 { INPUT_CLASS_TIME_DOMAIN } else { blob[18] },
        };
        let (d, dff) = (header.d_model, header.d_ff);
        if d == 0 || header.n_heads == 0 || d % header.n_heads != 0 || header.patch == 0 || header.n_time == 0 || header.n_layers == 0 {
            return Err(Error::Malformed);
        }
        let mut at = HEADER_LEN;
        let descriptor = take(blob, &mut at, d * 4)?;
        let time_emb = take(blob, &mut at, header.n_time * d * 4)?;
        let proj = Linear::take(blob, &mut at, d, header.patch)?;
        let layers_at = at;
        let layer_len = 4 * d * 4
            + Linear::len(3 * d, d)
            + Linear::len(d, d)
            + Linear::len(dff, d)
            + Linear::len(d, dff);
        at = layers_at.checked_add(layer_len * header.n_layers).ok_or(Error::Truncated)?;
        let final_norm_w = take(blob, &mut at, d * 4)?;
        let final_norm_b = take(blob, &mut at, d * 4)?;
        if header.head_offset != 0 && header.head_offset >= blob.len() {
            return Err(Error::Malformed);
        }
        Ok(Model { header, blob, descriptor, time_emb, proj, layers_at, layer_len, final_norm_w, final_norm_b })
    }

    /// The built-in head that shipped with these weights, if any.
    pub fn built_in_head(&self) -> Option<&'a [u8]> {
        match self.header.head_offset {
            0 => None,
            o => self.blob.get(o..),
        }
    }

    fn layer(&self, i: usize) -> Result<Layer<'a>, Error> {
        let (d, dff) = (self.header.d_model, self.header.d_ff);
        let mut at = self.layers_at + i * self.layer_len;
        Ok(Layer {
            norm1_w: take(self.blob, &mut at, d * 4)?,
            norm1_b: take(self.blob, &mut at, d * 4)?,
            qkv: Linear::take(self.blob, &mut at, 3 * d, d)?,
            out: Linear::take(self.blob, &mut at, d, d)?,
            norm2_w: take(self.blob, &mut at, d * 4)?,
            norm2_b: take(self.blob, &mut at, d * 4)?,
            ff1: Linear::take(self.blob, &mut at, dff, d)?,
            ff2: Linear::take(self.blob, &mut at, d, dff)?,
        })
    }

    /// Floats of workspace this model needs for a given channel count.
    pub fn workspace_len(&self, channels: usize) -> usize {
        let h = &self.header;
        let seq = channels * h.n_time;
        let dh = h.d_head();
        let row = if h.d_ff > h.d_model { h.d_ff } else { h.d_model };
        2 * seq * h.d_model + 3 * seq * dh + seq + row + FF_TILE * h.d_ff + h.window_samples()
    }

    /// Turn a window of microvolts into an embedding.
    ///
    /// `window` is channel major, `channels * window_samples` microvolts on
    /// the model's native grid, and is normalized in place. `live` says
    /// which channels are attached: a channel that is not attached is not
    /// attended to and is not pooled. `out` receives `d_model` values.
    ///
    /// This runs the whole pass at once. A caller that has other things to
    /// do meanwhile drives a [`Forward`] one step at a time instead; the
    /// arithmetic is the same code either way.
    pub fn embed(
        &self,
        window: &impl Window,
        live: Option<&[bool]>,
        ws: &mut [f32],
        out: &mut [f32],
    ) -> Result<(), Error> {
        let mut pass = Forward::new(self, window.channels(), live, ws.len(), out.len())?;
        loop {
            if let Progress::Done = pass.step(Some(window), ws, out)? {
                return Ok(());
            }
        }
    }

    /// The tokens of the last `embed` on this workspace: every channel's
    /// `n_time` vectors of `d_model`, channel-major, after the final
    /// normalization. Valid until the workspace is used again.
    pub fn tokens<'w>(&self, ws: &'w [f32], channels: usize) -> &'w [f32] {
        let (d, nt) = (self.header.d_model, self.header.n_time);
        let seq = channels * nt;
        &ws[seq * d..2 * seq * d]
    }
}

/// Rows of the attention projections one `Kvq` step computes, keys, values
/// and queries together. Tokens one `Attend` step attends. Output rows one
/// `OutProj` step adds. Each step is a few tens of milliseconds on the
/// part, which is how long the caller's other work waits.
const KVQ_ROWS: usize = 4;
const ATTEND_TOKENS: usize = 16;
const OUT_ROWS: usize = 24;

/// What a step of a [`Forward`] reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Progress {
    /// The next step reads the window; call again with it.
    NeedsWindow,
    /// A step ran; there is more.
    Running,
    /// `out` holds the embedding and the workspace holds the tokens.
    Done,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Read and normalize one channel of the window into the workspace.
    ReadChannel { c: usize },
    /// The copied channel is normalized in the workspace. No window is
    /// needed, so the caller holds no lock while the medians are found.
    Normalize { c: usize },
    /// Project that channel's patches, `KVQ_ROWS * 8` output rows at a time.
    Project { c: usize, o0: usize },
    Norm1 { li: usize },
    Kvq { li: usize, head: usize, j0: usize },
    Attend { li: usize, head: usize, t0: usize },
    OutProj { li: usize, head: usize, o0: usize },
    Norm2 { li: usize },
    FfUp { li: usize, base: usize },
    FfDown { li: usize, base: usize },
    Pool,
    Done,
}

/// The workspace, named.
struct Parts<'w> {
    x: &'w mut [f32],
    normed: &'w mut [f32],
    k: &'w mut [f32],
    v: &'w mut [f32],
    a: &'w mut [f32],
    scores: &'w mut [f32],
    row: &'w mut [f32],
    hidden: &'w mut [f32],
    channel: &'w mut [f32],
}

/// One forward pass, run a step at a time.
///
/// The pass on the part takes seconds, and a caller that also serves a
/// radio cannot be away that long. Each step is a bounded piece of the
/// arithmetic; between steps the caller yields, and the pass resumes where
/// it was. The window is read only in the first steps, one channel each,
/// so a caller that must hold a lock to hand the window over holds it for
/// a copy, not for the pass.
pub struct Forward<'m, 'a> {
    model: &'m Model<'a>,
    channels: usize,
    /// Bit per channel: attended and pooled.
    live: u32,
    phase: Phase,
    inv_sqrt_dh: f32,
}

impl<'m, 'a> Forward<'m, 'a> {
    /// Check the shapes and stand at the first step. `ws_len` and `out_len`
    /// are the lengths the caller will pass to every step.
    pub fn new(model: &'m Model<'a>, channels: usize, live: Option<&[bool]>, ws_len: usize, out_len: usize) -> Result<Self, Error> {
        let h = &model.header;
        if channels == 0 || channels > 32 || out_len < h.d_model || ws_len < model.workspace_len(channels) {
            return Err(Error::BadShape);
        }
        if live.is_some_and(|l| l.len() != channels) {
            return Err(Error::BadShape);
        }
        // Every channel dead is the same as no information about which are,
        // which is what the training code does rather than attending to
        // nothing at all.
        let any_live = live.is_none_or(|l| l.iter().any(|&v| v));
        let mut mask = 0u32;
        for c in 0..channels {
            if !any_live || live.is_none_or(|l| l[c]) {
                mask |= 1 << c;
            }
        }
        Ok(Forward {
            model,
            channels,
            live: mask,
            phase: Phase::ReadChannel { c: 0 },
            inv_sqrt_dh: 1.0 / libm::sqrtf(h.d_head() as f32),
        })
    }

    /// Whether every channel of the window has been read: from here on the
    /// window is not needed and may change under the caller.
    pub fn window_read(&self) -> bool {
        !matches!(self.phase, Phase::ReadChannel { .. } | Phase::Normalize { .. } | Phase::Project { .. })
    }

    fn is_live(&self, c: usize) -> bool {
        self.live & (1 << c) != 0
    }

    fn parts<'w>(&self, ws: &'w mut [f32]) -> Parts<'w> {
        let h = &self.model.header;
        let (d, dff, dh, nt) = (h.d_model, h.d_ff, h.d_head(), h.n_time);
        let seq = self.channels * nt;
        let row_len = if dff > d { dff } else { d };
        let (x, rest) = ws.split_at_mut(seq * d);
        let (normed, rest) = rest.split_at_mut(seq * d);
        let (k, rest) = rest.split_at_mut(seq * dh);
        let (v, rest) = rest.split_at_mut(seq * dh);
        let (a, rest) = rest.split_at_mut(seq * dh);
        let (scores, rest) = rest.split_at_mut(seq);
        let (row, rest) = rest.split_at_mut(row_len);
        let (hidden, rest) = rest.split_at_mut(FF_TILE * dff);
        let channel = &mut rest[..h.window_samples()];
        Parts { x, normed, k, v, a, scores, row, hidden, channel }
    }


    /// One step. `window` is needed only while the channels are being
    /// read; a step that needs it and is not given it reports so and does
    /// nothing. `ws` and `out` must be the same buffers every time.
    pub fn step<W: Window>(&mut self, window: Option<&W>, ws: &mut [f32], out: &mut [f32]) -> Result<Progress, Error> {
        let m = self.model;
        let h = &m.header;
        let (d, dff, dh, nt, patch) = (h.d_model, h.d_ff, h.d_head(), h.n_time, h.patch);
        let seq = self.channels * nt;
        if ws.len() < m.workspace_len(self.channels) || out.len() < d {
            return Err(Error::BadShape);
        }
        let p = self.parts(ws);
        self.phase = match self.phase {
            Phase::ReadChannel { c } => {
                let Some(window) = window else { return Ok(Progress::NeedsWindow) };
                if window.channels() != self.channels {
                    return Err(Error::BadShape);
                }
                // Copy it out and nothing more. The caller may hold the
                // window under a lock that stops interrupts, and finding
                // two medians over a channel took milliseconds there: on
                // the device one conversion per channel was missed at every
                // window, measured on the bench. The copy is microseconds.
                window.read_channel(c, p.channel);
                Phase::Normalize { c }
            }
            Phase::Normalize { c } => {
                // Normalize against the channel's own median and spread,
                // in a scratch the later phases will overwrite anyway.
                let n = p.channel.len();
                if p.normed.len() < n {
                    return Err(Error::BadShape);
                }
                normalize_window(p.channel, &mut p.normed[..n]);
                Phase::Project { c, o0: 0 }
            }
            Phase::Project { c, o0 } => {
                let o1 = (o0 + KVQ_ROWS * 8).min(d);
                for o in o0..o1 {
                    m.proj.row(o, p.row);
                    let bias = m.proj.bias(o) + f32_at(m.descriptor, o);
                    for t in 0..nt {
                        let at = t * patch;
                        p.x[(c * nt + t) * d + o] = bias + f32_at(m.time_emb, t * d + o) + dot(&p.row[..patch], &p.channel[at..at + patch]);
                    }
                }
                if o1 < d {
                    Phase::Project { c, o0: o1 }
                } else if c + 1 < self.channels {
                    Phase::ReadChannel { c: c + 1 }
                } else {
                    Phase::Norm1 { li: 0 }
                }
            }
            Phase::Norm1 { li } => {
                let layer = m.layer(li)?;
                for t in 0..seq {
                    layer_norm(&p.x[t * d..(t + 1) * d], layer.norm1_w, layer.norm1_b, &mut p.normed[t * d..(t + 1) * d]);
                }
                Phase::Kvq { li, head: 0, j0: 0 }
            }
            Phase::Kvq { li, head, j0 } => {
                let layer = m.layer(li)?;
                let j1 = (j0 + KVQ_ROWS).min(dh);
                // Keys and values for this head, every token, one
                // dequantized row at a time. Queries land in the attention
                // buffer and are consumed token by token in the next phase.
                for j in j0..j1 {
                    layer.qkv.row(d + head * dh + j, p.row);
                    let kb = layer.qkv.bias(d + head * dh + j);
                    for t in 0..seq {
                        p.k[t * dh + j] = kb + dot(&p.row[..d], &p.normed[t * d..(t + 1) * d]);
                    }
                    layer.qkv.row(2 * d + head * dh + j, p.row);
                    let vb = layer.qkv.bias(2 * d + head * dh + j);
                    for t in 0..seq {
                        p.v[t * dh + j] = vb + dot(&p.row[..d], &p.normed[t * d..(t + 1) * d]);
                    }
                    layer.qkv.row(head * dh + j, p.row);
                    let qb = layer.qkv.bias(head * dh + j);
                    for t in 0..seq {
                        p.a[t * dh + j] = qb + dot(&p.row[..d], &p.normed[t * d..(t + 1) * d]);
                    }
                }
                if j1 < dh { Phase::Kvq { li, head, j0: j1 } } else { Phase::Attend { li, head, t0: 0 } }
            }
            Phase::Attend { li, head, t0 } => {
                let t1 = (t0 + ATTEND_TOKENS).min(seq);
                for t in t0..t1 {
                    let q = &mut p.row[..dh];
                    q.copy_from_slice(&p.a[t * dh..(t + 1) * dh]);
                    for (s, score) in p.scores.iter_mut().enumerate() {
                        *score = if self.is_live(s / nt) { dot(q, &p.k[s * dh..(s + 1) * dh]) * self.inv_sqrt_dh } else { f32::NEG_INFINITY };
                    }
                    softmax(p.scores);
                    let dst = &mut p.a[t * dh..(t + 1) * dh];
                    dst.iter_mut().for_each(|o| *o = 0.0);
                    for (value, &pr) in p.v.chunks_exact(dh).zip(p.scores.iter()) {
                        if pr == 0.0 {
                            continue;
                        }
                        for (o, v) in dst.iter_mut().zip(value) {
                            *o += pr * v;
                        }
                    }
                }
                if t1 < seq { Phase::Attend { li, head, t0: t1 } } else { Phase::OutProj { li, head, o0: 0 } }
            }
            Phase::OutProj { li, head, o0 } => {
                let layer = m.layer(li)?;
                let o1 = (o0 + OUT_ROWS).min(d);
                // This head's share of the output projection, added
                // straight into the residual stream.
                for o in o0..o1 {
                    layer.out.row_part(o, head * dh, dh, p.row);
                    let bias = if head == 0 { layer.out.bias(o) } else { 0.0 };
                    for t in 0..seq {
                        p.x[t * d + o] += bias + dot(&p.row[..dh], &p.a[t * dh..(t + 1) * dh]);
                    }
                }
                if o1 < d {
                    Phase::OutProj { li, head, o0: o1 }
                } else if head + 1 < h.n_heads {
                    Phase::Kvq { li, head: head + 1, j0: 0 }
                } else {
                    Phase::Norm2 { li }
                }
            }
            Phase::Norm2 { li } => {
                let layer = m.layer(li)?;
                for t in 0..seq {
                    layer_norm(&p.x[t * d..(t + 1) * d], layer.norm2_w, layer.norm2_b, &mut p.normed[t * d..(t + 1) * d]);
                }
                Phase::FfUp { li, base: 0 }
            }
            Phase::FfUp { li, base } => {
                // Feed forward, a tile of tokens at a time so each weight row
                // is dequantized once per tile rather than once per token.
                let layer = m.layer(li)?;
                let n = FF_TILE.min(seq - base);
                for o in 0..dff {
                    layer.ff1.row(o, p.row);
                    let bias = layer.ff1.bias(o);
                    for i in 0..n {
                        let t = base + i;
                        p.hidden[i * dff + o] = gelu(bias + dot(&p.row[..d], &p.normed[t * d..(t + 1) * d]));
                    }
                }
                Phase::FfDown { li, base }
            }
            Phase::FfDown { li, base } => {
                let layer = m.layer(li)?;
                let n = FF_TILE.min(seq - base);
                for o in 0..d {
                    layer.ff2.row(o, p.row);
                    let bias = layer.ff2.bias(o);
                    for i in 0..n {
                        p.x[(base + i) * d + o] += bias + dot(&p.row[..dff], &p.hidden[i * dff..(i + 1) * dff]);
                    }
                }
                let next = base + n;
                if next < seq {
                    Phase::FfUp { li, base: next }
                } else if li + 1 < h.n_layers {
                    Phase::Norm1 { li: li + 1 }
                } else {
                    Phase::Pool
                }
            }
            Phase::Pool => {
                // The final normalization of every token, kept in the
                // workspace so the tokens can be read out afterwards
                // (`Model::tokens`), then the mean over live channels and
                // over time, which is the readout the model was measured
                // under.
                out[..d].iter_mut().for_each(|o| *o = 0.0);
                let mut counted = 0usize;
                for c in 0..self.channels {
                    for t in 0..nt {
                        let token = c * nt + t;
                        layer_norm(&p.x[token * d..(token + 1) * d], m.final_norm_w, m.final_norm_b, &mut p.normed[token * d..(token + 1) * d]);
                    }
                    if !self.is_live(c) {
                        continue;
                    }
                    counted += 1;
                    for t in 0..nt {
                        let token = c * nt + t;
                        for o in 0..d {
                            out[o] += p.normed[token * d + o];
                        }
                    }
                }
                let scale = 1.0 / (counted.max(1) * nt) as f32;
                out[..d].iter_mut().for_each(|o| *o *= scale);
                Phase::Done
            }
            Phase::Done => Phase::Done,
        };
        Ok(if self.phase == Phase::Done { Progress::Done } else { Progress::Running })
    }
}
