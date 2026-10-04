//! The check that the firmware's arithmetic is the training framework's.
//!
//! The fixture is a small model exported by `export_device.py`, with the
//! embeddings that framework produced for four windows of microvolts. If
//! this runtime and that exporter ever disagree, about the blob's layout
//! or about any step of the computation, these numbers move and the test
//! fails. That is the whole point: neither side is trusted, they are
//! compared.
//!
//! The full sized model is checked the same way where its weights live,
//! against `dev_distil_sometimes_vectors.bin`, and the result is recorded.

use intomind_model::{Model, Slice};

const BLOB: &[u8] = include_bytes!("data/tiny.imw");
const VECTORS: &[u8] = include_bytes!("data/tiny_vectors.bin");

struct Vectors {
    windows: Vec<Vec<f32>>,
    /// What the trained weights produce.
    exact: Vec<Vec<f32>>,
    /// What those weights produce after a round trip through int8, which is
    /// what the device holds and therefore what it must reproduce.
    quantized: Vec<Vec<f32>>,
    channels: usize,
    window_samples: usize,
    d_model: usize,
}

fn read_vectors(b: &[u8]) -> Vectors {
    let u32_at = |i: usize| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()) as usize;
    let (n, channels, window_samples, d_model) = (u32_at(0), u32_at(1), u32_at(2), u32_at(3));
    let mut at = 16;
    let mut take = |count: usize| {
        let v: Vec<f32> = (0..count)
            .map(|i| f32::from_le_bytes(b[at + i * 4..at + i * 4 + 4].try_into().unwrap()))
            .collect();
        at += count * 4;
        v
    };
    let windows = (0..n).map(|_| take(channels * window_samples)).collect();
    let exact = (0..n).map(|_| take(d_model)).collect();
    let quantized = (0..n).map(|_| take(d_model)).collect();
    Vectors { windows, exact, quantized, channels, window_samples, d_model }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

#[test]
fn the_runtime_reproduces_the_framework_exactly() {
    // Against the weights the device actually holds. Any difference here
    // is a difference in arithmetic, not in precision, so the bar is the
    // bar for float arithmetic that should agree step for step.
    let model = Model::parse(BLOB).expect("the fixture is a weights blob");
    let v = read_vectors(VECTORS);
    assert_eq!(model.header.d_model, v.d_model);
    assert_eq!(model.header.window_samples(), v.window_samples);

    let mut ws = vec![0.0f32; model.workspace_len(v.channels)];
    let mut out = vec![0.0f32; v.d_model];
    for (i, window) in v.windows.iter().enumerate() {
        let w = Slice { data: window, channels: v.channels, window_samples: v.window_samples };
        model.embed(&w, None, &mut ws, &mut out).expect("embed");
        let want = &v.quantized[i];
        let worst = out.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let magnitude = want.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let c = cosine(&out, want);
        assert!(
            c > 0.999_999_9 && worst < 1e-4 * magnitude.max(1.0),
            "window {i}: cosine {c:.9}, worst difference {worst:.7} against a magnitude of {magnitude:.3}\n got  {:?}\n want {:?}",
            &out[..4.min(out.len())],
            &want[..4.min(want.len())]
        );
    }
}

#[test]
fn the_tokens_are_what_the_embedding_is_the_mean_of() {
    // The token form of embeddings is the encoder's output before pooling,
    // so the mean over every token of every channel, all live, must be the
    // pooled embedding the head receives: the same numbers, not a second
    // computation of them.
    let model = Model::parse(BLOB).expect("the fixture is a weights blob");
    let v = read_vectors(VECTORS);
    let (d, nt) = (model.header.d_model, model.header.n_time);
    let mut ws = vec![0.0f32; model.workspace_len(v.channels)];
    let mut out = vec![0.0f32; d];
    let w = Slice { data: &v.windows[0], channels: v.channels, window_samples: v.window_samples };
    model.embed(&w, None, &mut ws, &mut out).expect("embed");
    let tokens = model.tokens(&ws, v.channels);
    assert_eq!(tokens.len(), v.channels * nt * d);
    let mut mean = vec![0.0f32; d];
    for t in 0..v.channels * nt {
        for o in 0..d {
            mean[o] += tokens[t * d + o];
        }
    }
    let scale = 1.0 / (v.channels * nt) as f32;
    for (m, o) in mean.iter_mut().zip(&out) {
        *m *= scale;
        assert!((*m - o).abs() < 1e-4, "token mean {m} against embedding {o}");
    }
    // Tokens differ from one another: they describe different slices.
    assert!(tokens[..d] != tokens[d..2 * d]);
}

#[test]
fn the_pass_runs_in_steps_and_lands_on_the_same_numbers() {
    // The device serves a radio while it thinks, so the pass is driven a
    // step at a time with yields between. The steps must add up to the
    // whole pass exactly, read the window only at the start, and be many.
    use intomind_model::{Forward, Progress};
    let model = Model::parse(BLOB).expect("the fixture is a weights blob");
    let v = read_vectors(VECTORS);
    let mut ws = vec![0.0f32; model.workspace_len(v.channels)];
    let mut whole = vec![0.0f32; v.d_model];
    let w = Slice { data: &v.windows[0], channels: v.channels, window_samples: v.window_samples };
    model.embed(&w, None, &mut ws, &mut whole).expect("embed");
    let tokens_whole = model.tokens(&ws, v.channels).to_vec();

    let mut ws2 = vec![0.0f32; model.workspace_len(v.channels)];
    let mut stepped = vec![0.0f32; v.d_model];
    let mut pass = Forward::new(&model, v.channels, None, ws2.len(), stepped.len()).unwrap();
    let (mut steps, mut window_reads) = (0usize, 0usize);
    loop {
        match pass.step(None::<&Slice>, &mut ws2, &mut stepped).unwrap() {
            Progress::NeedsWindow => {
                window_reads += 1;
                assert!(!pass.window_read());
                assert_eq!(pass.step(Some(&w), &mut ws2, &mut stepped).unwrap(), Progress::Running);
            }
            Progress::Running => steps += 1,
            Progress::Done => break,
        }
    }
    assert_eq!(window_reads, v.channels, "the window is read once per channel and never again");
    // At least a step per normalization, per attention head phase, and per
    // feed forward tile: the fixture is tiny, the shipping model has far
    // more, and either way a radio breathes between them.
    let h = &model.header;
    let least = v.channels + h.n_layers * (2 + 3 * h.n_heads + 2) + 1;
    assert!(steps >= least, "{steps} steps, fewer than the {least} the shape calls for");
    assert!(pass.window_read());
    assert_eq!(stepped, whole, "the stepped pass is the whole pass, bit for bit");
    assert_eq!(model.tokens(&ws2, v.channels), &tokens_whole[..]);
    // Done stays done.
    assert_eq!(pass.step(None::<&Slice>, &mut ws2, &mut stepped).unwrap(), Progress::Done);
}

#[test]
fn quantization_costs_what_it_was_measured_to_cost() {
    // Against the trained weights. This difference is the price of int8,
    // and it is the number the model work measured and accepted.
    let model = Model::parse(BLOB).unwrap();
    let v = read_vectors(VECTORS);
    let mut ws = vec![0.0f32; model.workspace_len(v.channels)];
    let mut out = vec![0.0f32; v.d_model];
    let mut worst = 1.0f64;
    for (i, window) in v.windows.iter().enumerate() {
        let w = Slice { data: window, channels: v.channels, window_samples: v.window_samples };
        model.embed(&w, None, &mut ws, &mut out).unwrap();
        worst = worst.min(cosine(&out, &v.exact[i]));
    }
    assert!(worst > 0.9975, "int8 cost the embedding a cosine of {worst:.6}, worse than the model work measured");
}

#[test]
fn the_header_describes_the_fixture() {
    let m = Model::parse(BLOB).unwrap();
    assert_eq!(m.header.n_layers, 2);
    assert_eq!(m.header.n_heads, 2);
    assert_eq!(m.header.d_model, 8);
    assert_eq!(m.header.d_ff, 16);
    assert_eq!(m.header.patch, 10);
    assert_eq!(m.header.n_time, 4);
    assert_eq!(m.header.native_sps, 500);
    assert_eq!(m.header.window_samples(), 40);
    assert_eq!(m.header.d_head(), 4);
    assert!(m.built_in_head().is_none());
}

#[test]
fn a_damaged_blob_is_refused_and_never_read_past() {
    use intomind_model::Error;
    assert_eq!(Model::parse(&[]).err(), Some(Error::Malformed));
    assert_eq!(Model::parse(&BLOB[..16]).err(), Some(Error::Malformed));
    let mut bad = BLOB.to_vec();
    bad[0] = b'X';
    assert_eq!(Model::parse(&bad).err(), Some(Error::Malformed));
    let mut bad = BLOB.to_vec();
    bad[4] = 2;
    assert_eq!(Model::parse(&bad).err(), Some(Error::Malformed));
    // A blob whose header promises more than it holds is truncated, not a
    // read past the end.
    assert_eq!(Model::parse(&BLOB[..BLOB.len() - 4]).err(), Some(Error::Truncated));
    // Shapes that cannot be run at all.
    let mut bad = BLOB.to_vec();
    bad[10] = 3; // three heads do not divide eight
    assert_eq!(Model::parse(&bad).err(), Some(Error::Malformed));
}

#[test]
fn the_caller_is_told_when_its_buffers_are_wrong() {
    use intomind_model::Error;
    let m = Model::parse(BLOB).unwrap();
    let mut ws = vec![0.0f32; m.workspace_len(3)];
    let mut out = vec![0.0f32; 8];
    let data = vec![0.0f32; 3 * 40];
    let w = Slice { data: &data, channels: 3, window_samples: 40 };
    let none = Slice { data: &data, channels: 0, window_samples: 40 };
    assert_eq!(m.embed(&none, None, &mut ws, &mut out), Err(Error::BadShape));
    assert_eq!(m.embed(&w, None, &mut ws[..10], &mut out), Err(Error::BadShape));
    assert_eq!(m.embed(&w, None, &mut ws, &mut out[..4]), Err(Error::BadShape));
    assert_eq!(m.embed(&w, Some(&[true, true]), &mut ws, &mut out), Err(Error::BadShape));
}

#[test]
fn a_channel_that_fell_off_is_left_out_rather_than_believed() {
    let m = Model::parse(BLOB).unwrap();
    let v = read_vectors(VECTORS);
    let mut ws = vec![0.0f32; m.workspace_len(v.channels)];
    let mut all = vec![0.0f32; v.d_model];
    let mut two = vec![0.0f32; v.d_model];
    let mut nonsense = v.windows[0].clone();
    // The third channel goes to a large flat offset, the way a detached
    // electrode reads.
    for s in nonsense[2 * v.window_samples..].iter_mut() {
        *s = 9_000.0;
    }
    let w = Slice { data: &nonsense, channels: v.channels, window_samples: v.window_samples };
    m.embed(&w, None, &mut ws, &mut all).unwrap();
    m.embed(&w, Some(&[true, true, false]), &mut ws, &mut two).unwrap();
    assert!(cosine(&all, &two) < 0.9999, "excluding a dead channel has to change the answer");

    // And excluding it gives the same answer as never having had it, up to
    // the sequence being shorter, which is what permutation invariance
    // under a withheld position buys.
    let mut every_dead = vec![0.0f32; v.d_model];
    m.embed(&w, Some(&[false, false, false]), &mut ws, &mut every_dead).unwrap();
    assert!(cosine(&all, &every_dead) > 0.999_99, "no channel live is treated as no information, not as nothing to attend to");
}
