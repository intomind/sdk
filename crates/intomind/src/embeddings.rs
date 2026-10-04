//! The encoder's output, put back together from its notifications.
//!
//! A device sends its encoder's output for each window in two forms: the
//! window embedding, one vector a head consumes, and the tokens, one vector
//! per channel per slice of the window, which reconstruction turns back
//! into signal. Each vector is one notification, or several when it does
//! not fit one. This module collects them into whole windows.

use std::collections::BTreeMap;

use intomind_protocol::embeddings::{self, EmbeddingHeader, WINDOW_TOKEN};
use intomind_protocol::head::EMBED_SCALE;

/// One notification, decoded: a whole vector or a part of one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Embedding {
    /// Raw stream index of the window's first sample.
    pub index: u32,
    /// Device time of that sample.
    pub device_time: u64,
    /// Window length in raw samples at the current rate.
    pub window_samples: u16,
    /// The encoder that produced it, as the model info reports it.
    pub encoder_id: [u8; 8],
    /// A `protocol::pipeline::input_source` value.
    pub input_source: u8,
    /// Values in the whole vector.
    pub embed_dim: u8,
    /// Index of the first value carried here.
    pub first: u8,
    /// None for the window embedding; otherwise the token's index,
    /// channel-major over all channels.
    pub token: Option<u8>,
    /// Another notification carries the rest of this vector.
    pub more_parts: bool,
    /// The window spans a loss in the stream.
    pub gap_in_window: bool,
    /// The device skipped windows to stay within its compute budget.
    pub duty_reduced: bool,
    /// An electrode was off during the window.
    pub leadoff_in_window: bool,
    /// Quantized values: the encoder's output times 4096.
    pub values: Vec<i16>,
}

impl Embedding {
    /// From one notification's bytes.
    pub fn parse(bytes: &[u8]) -> Result<Embedding, intomind_protocol::Error> {
        let (h, v) = EmbeddingHeader::parse(bytes)?;
        Ok(Embedding::from_parts(&h, v.iter().collect()))
    }

    fn from_parts(h: &EmbeddingHeader, values: Vec<i16>) -> Embedding {
        Embedding {
            index: h.sample_index,
            device_time: h.device_time,
            window_samples: h.window_samples,
            encoder_id: h.encoder_id,
            input_source: h.input_source,
            embed_dim: h.embed_dim,
            first: h.first,
            token: if h.token == WINDOW_TOKEN { None } else { Some(h.token) },
            more_parts: h.flags & embeddings::flags::MORE_PARTS != 0,
            gap_in_window: h.flags & embeddings::flags::GAP_IN_WINDOW != 0,
            duty_reduced: h.flags & embeddings::flags::DUTY_REDUCED != 0,
            leadoff_in_window: h.flags & embeddings::flags::LEADOFF_IN_WINDOW != 0,
            values,
        }
    }
}

/// Which form a host asks for, `SET_EMBEDDINGS`'s argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Form {
    /// The one vector a head consumes.
    Window,
    /// The per-slice vectors reconstruction consumes.
    Tokens,
    /// Both, for the same windows.
    Both,
}

impl Form {
    /// The contract's byte.
    pub fn code(self) -> u8 {
        match self {
            Form::Window => embeddings::form::WINDOW,
            Form::Tokens => embeddings::form::TOKENS,
            Form::Both => embeddings::form::BOTH,
        }
    }
}

/// One window's embeddings, whole.
#[derive(Debug, Clone, PartialEq)]
pub struct Window {
    /// Raw stream index of the window's first sample.
    pub index: u32,
    /// Device time of that sample.
    pub device_time: u64,
    /// Window length in raw samples at the current rate.
    pub window_samples: u16,
    /// The encoder that produced it.
    pub encoder_id: [u8; 8],
    /// A `protocol::pipeline::input_source` value.
    pub input_source: u8,
    /// Values per vector.
    pub embed_dim: usize,
    /// The window spans a loss in the stream.
    pub gap_in_window: bool,
    /// The device skipped windows to stay within its compute budget.
    pub duty_reduced: bool,
    /// An electrode was off during the window.
    pub leadoff_in_window: bool,
    /// The window embedding, quantized, when asked for.
    pub embedding: Option<Vec<i16>>,
    /// The tokens, quantized, `channels × tokens_per_channel` vectors of
    /// `embed_dim`, channel-major, when asked for.
    pub tokens: Option<Vec<Vec<i16>>>,
}

impl Window {
    /// The window embedding as the encoder produced it, before quantization.
    pub fn embedding_f32(&self) -> Option<Vec<f32>> {
        self.embedding.as_ref().map(|e| e.iter().map(|&v| v as f32 / EMBED_SCALE).collect())
    }

    /// The tokens as the encoder produced them, `(channels × tokens_per_channel)` rows.
    pub fn tokens_f32(&self) -> Option<Vec<Vec<f32>>> {
        self.tokens.as_ref().map(|t| t.iter().map(|row| row.iter().map(|&v| v as f32 / EMBED_SCALE).collect()).collect())
    }
}

struct InFlight {
    meta: Embedding,
    embedding: Vec<Option<i16>>,
    tokens: Vec<Vec<Option<i16>>>,
}

/// Puts notifications back together into whole windows.
///
/// Built for a device's channel count, the model's tokens per channel, and
/// the form asked for. `feed` takes each notification and returns the
/// window when its last piece arrives. Windows older than the two most
/// recent that never completed are dropped and counted in `incomplete`.
#[derive(Debug)]
pub struct Assembler {
    channels: usize,
    tokens_per_channel: usize,
    form: Form,
    /// Windows dropped before they were whole.
    pub incomplete: u32,
    windows: BTreeMap<u32, InFlight>,
}

impl std::fmt::Debug for InFlight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "window {}", self.meta.index)
    }
}

impl Assembler {
    /// For a device with this many channels, a model with this many tokens
    /// per channel, and the form asked for.
    pub fn new(channels: usize, tokens_per_channel: usize, form: Form) -> Assembler {
        Assembler { channels, tokens_per_channel, form, incomplete: 0, windows: BTreeMap::new() }
    }

    fn n_tokens(&self) -> usize {
        self.channels * self.tokens_per_channel
    }

    /// One notification in; a whole window out when this completed it.
    pub fn feed(&mut self, e: Embedding) -> Option<Window> {
        let n_tokens = self.n_tokens();
        let dim = e.embed_dim as usize;
        if !self.windows.contains_key(&e.index) {
            self.windows.insert(
                e.index,
                InFlight { meta: e.clone(), embedding: vec![None; dim], tokens: vec![vec![None; dim]; n_tokens] },
            );
            while self.windows.len() > 3 {
                let oldest = *self.windows.keys().next().unwrap();
                self.windows.remove(&oldest);
                self.incomplete += 1;
            }
        }
        let want_embedding = matches!(self.form, Form::Window | Form::Both);
        let want_tokens = matches!(self.form, Form::Tokens | Form::Both);
        let w = self.windows.get_mut(&e.index)?;
        let target = match e.token {
            None => &mut w.embedding,
            Some(t) if (t as usize) < n_tokens => &mut w.tokens[t as usize],
            Some(_) => return None,
        };
        let first = e.first as usize;
        if first + e.values.len() > target.len() {
            return None;
        }
        for (slot, v) in target[first..first + e.values.len()].iter_mut().zip(&e.values) {
            *slot = Some(*v);
        }
        let embedding_done = !want_embedding || w.embedding.iter().all(Option::is_some);
        let tokens_done = !want_tokens || w.tokens.iter().all(|t| t.iter().all(Option::is_some));
        if !(embedding_done && tokens_done) {
            return None;
        }
        let w = self.windows.remove(&e.index)?;
        let m = w.meta;
        Some(Window {
            index: m.index,
            device_time: m.device_time,
            window_samples: m.window_samples,
            encoder_id: m.encoder_id,
            input_source: m.input_source,
            embed_dim: dim,
            gap_in_window: m.gap_in_window,
            duty_reduced: m.duty_reduced,
            leadoff_in_window: m.leadoff_in_window,
            embedding: want_embedding.then(|| w.embedding.iter().map(|v| v.unwrap()).collect()),
            tokens: want_tokens.then(|| w.tokens.iter().map(|t| t.iter().map(|v| v.unwrap()).collect()).collect()),
        })
    }
}
