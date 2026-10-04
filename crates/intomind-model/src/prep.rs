//! Getting a window ready: the rate the model was trained on, and the
//! normalization it was trained under.

use crate::math::median_into;

/// The floor under the deviation, in microvolts, so a flat channel cannot
/// explode when it is divided by its own spread. The training code's
/// value, and it is absolute, which is why the window has to be in real
/// microvolts rather than converter counts.
pub const MAD_FLOOR_UV: f32 = 0.1;

/// The constant that turns a median absolute deviation into the standard
/// deviation of a normal distribution.
const MAD_TO_SIGMA: f32 = 1.4826;

/// Normalize one channel in place: subtract its median, divide by its
/// spread. Per window and per channel, which makes the model blind to
/// amplifier gain and to a resting offset, and is what it was trained
/// under. `buf` must hold at least `channel.len()` floats.
pub fn normalize_window(channel: &mut [f32], buf: &mut [f32]) {
    let median = median_into(channel.iter().copied(), buf);
    let mad = median_into(channel.iter().map(|v| (v - median).abs()), buf) * MAD_TO_SIGMA;
    let scale = 1.0 / if mad > MAD_FLOOR_UV { mad } else { MAD_FLOOR_UV };
    for v in channel.iter_mut() {
        *v = (*v - median) * scale;
    }
}

/// Resample a channel onto the model's native grid, linearly.
///
/// The model has one native rate. The instrument has three, and the raw
/// stream is never resampled, so the model gets its own copy. Linear
/// interpolation is enough here because the converter has already band
/// limited the signal well below every rate the device offers: its
/// decimation filter puts the usable band at about a quarter of the rate,
/// so nothing near the new Nyquist frequency survives to alias.
///
/// Returns how many samples were written.
pub fn resample_into(source: &[f32], source_sps: u32, out: &mut [f32], native_sps: u32) -> usize {
    if source.is_empty() || source_sps == 0 || native_sps == 0 {
        return 0;
    }
    if source_sps == native_sps {
        let n = source.len().min(out.len());
        out[..n].copy_from_slice(&source[..n]);
        return n;
    }
    let step = source_sps as f64 / native_sps as f64;
    let last = source.len() - 1;
    let mut written = 0;
    for (i, o) in out.iter_mut().enumerate() {
        let at = i as f64 * step;
        if at > last as f64 {
            break;
        }
        let lo = at as usize;
        let frac = (at - lo as f64) as f32;
        let hi = (lo + 1).min(last);
        *o = source[lo] + (source[hi] - source[lo]) * frac;
        written += 1;
    }
    written
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizing_removes_offset_and_gain() {
        let mut buf = [0.0f32; 16];
        let base: Vec<f32> = (0..8).map(|i| i as f32 - 3.5).collect();
        let mut a: Vec<f32> = base.iter().map(|v| v * 10.0 + 500.0).collect();
        let mut b: Vec<f32> = base.iter().map(|v| v * 1000.0 - 20.0).collect();
        normalize_window(&mut a, &mut buf);
        normalize_window(&mut b, &mut buf);
        // A hundredfold difference in gain and a large offset produce the
        // same normalized window.
        for i in 0..8 {
            assert!((a[i] - b[i]).abs() < 1e-3, "{i}: {} against {}", a[i], b[i]);
        }
    }

    #[test]
    fn a_flat_channel_cannot_explode() {
        let mut buf = [0.0f32; 8];
        let mut flat = [7.0f32; 8];
        normalize_window(&mut flat, &mut buf);
        assert_eq!(flat, [0.0f32; 8], "no spread means no signal, not infinity");
        // A channel whose spread is below the floor is scaled by the floor,
        // not by its own spread.
        let mut tiny = [0.0f32, 0.01, 0.0, 0.01, 0.0, 0.01, 0.0, 0.01];
        normalize_window(&mut tiny, &mut buf);
        assert!(tiny.iter().all(|v| v.abs() <= 0.11), "{tiny:?}");
    }

    #[test]
    fn a_channel_read_from_a_window_normalizes_like_a_copy_of_it() {
        use crate::blob::{Slice, Window};
        let raw: Vec<f32> = (0..24).map(|i| ((i * 37 % 19) as f32 - 9.0) * 3.0 + 250.0).collect();
        let w = Slice { data: &raw, channels: 3, window_samples: 8 };
        let mut read = [0.0f32; 8];
        for c in 0..3 {
            let mut copy = raw[c * 8..(c + 1) * 8].to_vec();
            let mut buf = [0.0f32; 8];
            normalize_window(&mut copy, &mut buf);
            w.read_channel(c, &mut read);
            normalize_window(&mut read, &mut buf);
            for i in 0..8 {
                assert!((copy[i] - read[i]).abs() < 1e-6, "channel {c} sample {i}");
            }
        }
    }

    #[test]
    fn resampling_hits_the_native_grid() {
        // Same rate: a copy, exactly.
        let src: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let mut out = [0.0f32; 10];
        assert_eq!(resample_into(&src, 500, &mut out, 500), 10);
        assert_eq!(&out[..], &src[..]);

        // Half rate in, so every other output sample is interpolated.
        let mut out = [0.0f32; 19];
        let n = resample_into(&src, 250, &mut out, 500);
        assert_eq!(n, 19);
        assert_eq!(out[0], 0.0);
        assert!((out[1] - 0.5).abs() < 1e-6);
        assert_eq!(out[2], 1.0);
        assert!((out[17] - 8.5).abs() < 1e-6);

        // Double rate in, so every other input sample is dropped.
        let mut out = [0.0f32; 5];
        assert_eq!(resample_into(&src, 1000, &mut out, 500), 5);
        assert_eq!(&out[..], &[0.0, 2.0, 4.0, 6.0, 8.0]);

        // A line stays a line at any rate, which is the property that
        // matters: interpolation adds no shape of its own.
        let line: Vec<f32> = (0..100).map(|i| 3.0 * i as f32 - 7.0).collect();
        let mut out = [0.0f32; 120];
        let n = resample_into(&line, 400, &mut out, 500);
        for (i, v) in out[..n].iter().enumerate() {
            let want = 3.0 * (i as f32 * 0.8) - 7.0;
            assert!((v - want).abs() < 1e-3, "{i}: {v} against {want}");
        }
    }

    #[test]
    fn nothing_is_written_past_what_the_source_can_fill() {
        let src = [1.0f32, 2.0];
        let mut out = [-1.0f32; 10];
        let n = resample_into(&src, 500, &mut out, 500);
        assert_eq!(n, 2);
        assert_eq!(out[2], -1.0);
        assert_eq!(resample_into(&[], 500, &mut out, 500), 0);
        assert_eq!(resample_into(&src, 0, &mut out, 500), 0);
    }
}
