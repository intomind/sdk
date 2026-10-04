//! The on-device encoder.
//!
//! A window of microvolts goes in, an embedding as wide as the weights say
//! comes out (76 numbers for the encoder the IntoMind One ships). A head
//! turns that embedding into whatever a user asked for, and heads are
//! weights, never code.
//!
//! The weights live in flash as a blob whose format is defined here and in
//! the exporter that writes it. Neither side trusts the other: the tests
//! run a small model exported by that same script and assert this
//! arithmetic reproduces the embeddings the training framework produced, so
//! a drift between the two shows up as a numeric failure rather than a
//! silent misread.
//!
//! Weights are int8 with one scale per output row. Activations are f32,
//! because the part has a floating point unit, so quantizing the
//! activations would save memory we do not need and cost accuracy we have
//! no reason to spend.
//! Each weight row is dequantized once and reused across every token in
//! the sequence, so the conversion is amortized and the inner loop is a
//! plain dot product.
//!
//! Nothing here allocates. The caller hands in the workspace, which the
//! application owns statically and never shares with the acquisition path.

#![cfg_attr(not(test), no_std)]
#![deny(unsafe_code)]
#![deny(missing_docs)]

pub mod blob;
mod math;
mod prep;
pub mod stream;

pub use blob::{Error, Forward, Header, Model, Progress, Slice, Window};
pub use prep::{normalize_window, resample_into, MAD_FLOOR_UV};
pub use stream::{Feeder, Ratio, WindowAt};

/// Scratch the encoder needs, in floats. Use `Model::workspace_len`, which
/// knows the shapes. This is the same arithmetic for a build that has to
/// size a static buffer before the weights are parsed.
pub const fn workspace_len(seq: usize, d_model: usize, d_ff: usize, d_head: usize, window_samples: usize) -> usize {
    // The residual stream and its normalized copy dominate. Then one
    // head's keys, values and attention output, the scores for one query,
    // one dequantized weight row, a tile of feed forward hidden values,
    // and the channel being normalized.
    let row = if d_ff > d_model { d_ff } else { d_model };
    2 * seq * d_model + 3 * seq * d_head + seq + row + blob::FF_TILE * d_ff + window_samples
}
