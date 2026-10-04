//! The arithmetic the encoder is made of. Every function here is the one
//! the training framework used, in the same form, because a different but
//! equally valid form would move the answer.

/// Gaussian error linear unit, the exact form: `0.5 x (1 + erf(x / sqrt 2))`.
/// The tanh approximation is a different function and would show up in the
/// reference check.
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * core::f32::consts::FRAC_1_SQRT_2))
}

/// Layer normalization over `x`, into `out`. Biased variance, as every
/// framework computes it. The weight and bias are read straight out of the
/// blob, so nothing is copied for a step that runs once per token.
pub fn layer_norm(x: &[f32], weight: &[u8], bias: &[u8], out: &mut [f32]) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / libm::sqrtf(var + 1e-5);
    for (i, o) in out.iter_mut().enumerate().take(x.len()) {
        *o = (x[i] - mean) * inv * f32_at(weight, i) + f32_at(bias, i);
    }
}

/// One little-endian float out of a byte slice.
pub fn f32_at(b: &[u8], i: usize) -> f32 {
    let o = i * 4;
    f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// Softmax in place, shifted by the maximum so it cannot overflow.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        // Every entry was masked out. Leave zeros rather than producing
        // not-a-number, which would poison the embedding.
        x.iter_mut().for_each(|v| *v = 0.0);
        return;
    }
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = libm::expf(*v - max);
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        x.iter_mut().for_each(|v| *v *= inv);
    }
}

/// Inner product, four lanes at a time. The part's floating point unit
/// issues one multiply or add a cycle; what costs is the loop around
/// them, so four independent sums take the loop overhead once per four
/// products. The sum is reassociated against a plain left fold, which
/// moves the result by rounding noise and nothing more: the reference
/// test holds it to the training framework at one part in ten thousand.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut ca = a.chunks_exact(4);
    let mut cb = b.chunks_exact(4);
    let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for (x, y) in (&mut ca).zip(&mut cb) {
        s0 += x[0] * y[0];
        s1 += x[1] * y[1];
        s2 += x[2] * y[2];
        s3 += x[3] * y[3];
    }
    let mut tail = 0.0f32;
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        tail += x * y;
    }
    (s0 + s1) + (s2 + s3) + tail
}

/// The lower median of a slice, which is what the training framework's
/// median returns for an even count. Found by selection rather than a
/// full sort: the same value, in linear time, and without the sorting
/// machinery that a full sort links into a firmware image.
pub fn lower_median(buf: &mut [f32]) -> f32 {
    if buf.is_empty() {
        return 0.0;
    }
    let k = (buf.len() - 1) / 2;
    *buf.select_nth_unstable_by(k, f32::total_cmp).1
}

/// The lower median of a sequence. `buf` is used as scratch and its
/// contents are lost.
pub fn median_into(values: impl Iterator<Item = f32>, buf: &mut [f32]) -> f32 {
    let mut n = 0;
    for v in values {
        buf[n] = v;
        n += 1;
    }
    lower_median(&mut buf[..n])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_is_the_exact_form() {
        // Values from the exact definition, not the tanh approximation.
        assert!((gelu(0.0) - 0.0).abs() < 1e-7);
        assert!((gelu(1.0) - 0.841_345).abs() < 1e-5, "{}", gelu(1.0));
        assert!((gelu(-1.0) + 0.158_655).abs() < 1e-5, "{}", gelu(-1.0));
        assert!((gelu(3.0) - 2.995_950).abs() < 1e-4, "{}", gelu(3.0));
    }

    #[test]
    fn layer_norm_centers_and_scales() {
        fn bytes(v: [f32; 4]) -> [u8; 16] {
            let mut o = [0u8; 16];
            for (i, f) in v.iter().enumerate() {
                o[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes());
            }
            o
        }
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let w = bytes([1.0; 4]);
        let b = bytes([0.0; 4]);
        let mut out = [0.0f32; 4];
        layer_norm(&x, &w, &b, &mut out);
        let mean = out.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5);
        let var = out.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 1e-3, "{var}");
        // Weight and bias are applied after normalizing.
        let w = bytes([2.0; 4]);
        let b = bytes([0.5; 4]);
        let mut scaled = [0.0f32; 4];
        layer_norm(&x, &w, &b, &mut scaled);
        for i in 0..4 {
            assert!((scaled[i] - (out[i] * 2.0 + 0.5)).abs() < 1e-5);
        }
    }

    #[test]
    fn softmax_sums_to_one_and_survives_extremes() {
        let mut x = [1.0f32, 2.0, 3.0];
        softmax(&mut x);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(x[2] > x[1] && x[1] > x[0]);
        // Large values do not overflow.
        let mut x = [1000.0f32, 1001.0];
        softmax(&mut x);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        // Everything masked out gives zeros, never not-a-number.
        let mut x = [f32::NEG_INFINITY; 3];
        softmax(&mut x);
        assert_eq!(x, [0.0; 3]);
    }

    #[test]
    fn the_median_is_the_lower_one_for_an_even_count() {
        let mut buf = [0.0f32; 8];
        assert_eq!(median_into([3.0f32, 1.0, 2.0, 4.0].into_iter(), &mut buf), 2.0);
        assert_eq!(median_into([3.0f32, 1.0, 2.0].into_iter(), &mut buf), 2.0);
        assert_eq!(median_into([5.0f32].into_iter(), &mut buf), 5.0);
        assert_eq!(median_into(core::iter::empty(), &mut buf), 0.0);
    }
}
