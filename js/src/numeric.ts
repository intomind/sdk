/**
 * The signal processing the analyses need, written here so the package
 * carries no dependency: a Fourier transform of any length, Welch's
 * spectrum, Butterworth and notch filter design, forward and backward
 * filtering, and the few numpy functions the analyses lean on. Each does
 * what the Python library gets from scipy 1.15 and numpy 2.2, written from
 * the documented method and checked against that library's own results by
 * `contract/analysis`: identical where the arithmetic is exact, and within a
 * stated tolerance where a different order of floating point operations
 * leaves the last digits apart. Internal: the analyses are the surface.
 */

import { numpyMean, numpySum, repr } from "./capture/pyfmt.ts";

/** A refusal, in the words scipy refuses with. */
export class DesignError extends Error {}

// --- numpy's small functions -------------------------------------------------------

/** `numpy.mean`, NaN for nothing. */
export function mean(a: ArrayLike<number>): number {
  return a.length === 0 ? NaN : numpyMean(a);
}

/** `numpy.std` with `ddof` degrees of freedom: the mean, then the mean square of what is left over. */
export function std(a: ArrayLike<number>, ddof = 0): number {
  const n = a.length;
  if (n === 0) return NaN;
  const m = numpyMean(a);
  const sq = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const d = a[i]! - m;
    sq[i] = d * d;
  }
  return Math.sqrt(numpySum(sq) / (n - ddof));
}

/** `numpy.median`: the middle value, or the mean of the two middle ones. NaN for nothing. */
export function median(a: ArrayLike<number>): number {
  const n = a.length;
  if (n === 0) return NaN;
  const s = Float64Array.from(a).sort();
  if (Number.isNaN(s[n - 1]!)) return NaN;
  const h = n >> 1;
  return n % 2 === 1 ? s[h]! : (s[h - 1]! + s[h]!) / 2;
}

/** `numpy.trapezoid(y, x)`: each interval's width times its two ends' mean, summed as numpy sums. */
export function trapezoid(y: ArrayLike<number>, x: ArrayLike<number>): number {
  const n = y.length;
  if (n < 2) return 0;
  const terms = new Float64Array(n - 1);
  for (let i = 0; i < n - 1; i++) terms[i] = ((x[i + 1]! - x[i]!) * (y[i + 1]! + y[i]!)) / 2.0;
  return numpySum(terms);
}

/** `numpy.polyfit(x, y, 1)`: the least squares line, as slope then intercept. */
export function polyfit1(x: ArrayLike<number>, y: ArrayLike<number>): [number, number] {
  const n = x.length;
  const xm = numpyMean(x);
  const ym = numpyMean(y);
  const xx = new Float64Array(n);
  const xy = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const dx = x[i]! - xm;
    xx[i] = dx * dx;
    xy[i] = dx * (y[i]! - ym);
  }
  const slope = numpySum(xy) / numpySum(xx);
  return [slope, ym - slope * xm];
}

/** `numpy.corrcoef(a, b)[0, 1]`: the covariance divided by each standard deviation in turn, held to between -1 and 1. */
export function corrcoef(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = a.length;
  const am = numpyMean(a);
  const bm = numpyMean(b);
  const aa = new Float64Array(n);
  const bb = new Float64Array(n);
  const ab = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const da = a[i]! - am;
    const db = b[i]! - bm;
    aa[i] = da * da;
    bb[i] = db * db;
    ab[i] = da * db;
  }
  const f = 1 / (n - 1);
  const c01 = numpySum(ab) * f;
  const r = c01 / Math.sqrt(numpySum(aa) * f) / Math.sqrt(numpySum(bb) * f);
  return Number.isNaN(r) ? r : Math.min(1, Math.max(-1, r));
}

/** `numpy.searchsorted(a, v)`: where `v` would go in the sorted `a`, before any value equal to it. */
export function searchsorted(a: ArrayLike<number>, v: number): number {
  let lo = 0;
  let hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (a[mid]! < v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** `numpy.argmax`: the first place the largest value is. */
export function argmax(a: ArrayLike<number>): number {
  let k = 0;
  for (let i = 1; i < a.length; i++) {
    if (a[i]! > a[k]! || (Number.isNaN(a[i]!) && !Number.isNaN(a[k]!))) {
      k = i;
      if (Number.isNaN(a[i]!)) break;
    }
  }
  return k;
}

/** `numpy.ptp`: the largest value less the smallest. */
export function ptp(a: ArrayLike<number>): number {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < a.length; i++) {
    const v = a[i]!;
    if (Number.isNaN(v)) return NaN;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return hi - lo;
}

// --- the Fourier transform ---------------------------------------------------------

/**
 * The discrete Fourier transform of `re + i im`, in place, for a length that
 * is a power of two: iterative radix 2, each twiddle factor taken from the
 * cosine and sine of its own angle.
 */
function fftPow2(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]!;
      re[i] = re[j]!;
      re[j] = t;
      t = im[i]!;
      im[i] = im[j]!;
      im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const step = n / len;
    for (let k = 0; k < half; k++) {
      const angle = (-2 * Math.PI * (k * step)) / n;
      const wr = Math.cos(angle);
      const wi = Math.sin(angle);
      for (let s = 0; s < n; s += len) {
        const a = s + k;
        const b = a + half;
        const xr = re[b]! * wr - im[b]! * wi;
        const xi = re[b]! * wi + im[b]! * wr;
        re[b] = re[a]! - xr;
        im[b] = im[a]! - xi;
        re[a] = re[a]! + xr;
        im[a] = im[a]! + xi;
      }
    }
  }
}

/**
 * The discrete Fourier transform of any length, in place: radix 2 where the
 * length is a power of two, and otherwise Bluestein's method, which writes
 * the transform as a convolution with a chirp and does that convolution with
 * power-of-two transforms. The chirp's angles are taken from `k * k` reduced
 * modulo twice the length, so they stay exact for long transforms.
 */
export function fft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  if (n <= 1) return;
  if ((n & (n - 1)) === 0) {
    fftPow2(re, im);
    return;
  }
  let m = 1;
  while (m < 2 * n - 1) m <<= 1;
  const cr = new Float64Array(n);
  const ci = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const angle = (Math.PI * ((k * k) % (2 * n))) / n;
    cr[k] = Math.cos(angle);
    ci[k] = -Math.sin(angle);
  }
  const ar = new Float64Array(m);
  const ai = new Float64Array(m);
  for (let k = 0; k < n; k++) {
    ar[k] = re[k]! * cr[k]! - im[k]! * ci[k]!;
    ai[k] = re[k]! * ci[k]! + im[k]! * cr[k]!;
  }
  const br = new Float64Array(m);
  const bi = new Float64Array(m);
  br[0] = cr[0]!;
  bi[0] = -ci[0]!;
  for (let k = 1; k < n; k++) {
    br[k] = br[m - k] = cr[k]!;
    bi[k] = bi[m - k] = -ci[k]!;
  }
  fftPow2(ar, ai);
  fftPow2(br, bi);
  for (let k = 0; k < m; k++) {
    const r = ar[k]! * br[k]! - ai[k]! * bi[k]!;
    const i = ar[k]! * bi[k]! + ai[k]! * br[k]!;
    ar[k] = r;
    ai[k] = -i;
  }
  // The inverse transform, as the forward transform of the conjugate.
  fftPow2(ar, ai);
  for (let k = 0; k < n; k++) {
    const r = ar[k]! / m;
    const i = -ai[k]! / m;
    re[k] = r * cr[k]! - i * ci[k]!;
    im[k] = r * ci[k]! + i * cr[k]!;
  }
}

// --- Welch's spectrum --------------------------------------------------------------

/**
 * The periodic Hann window of `n` points, as scipy's `get_window("hann", n)`
 * makes it: half plus half the cosine of `n + 1` points spaced evenly from
 * minus pi to pi, the last dropped.
 */
export function hann(n: number): Float64Array {
  const w = new Float64Array(n);
  if (n === 1) {
    w[0] = 1;
    return w;
  }
  const step = (Math.PI - -Math.PI) / n;
  for (let i = 0; i < n; i++) w[i] = 0.5 + 0.5 * Math.cos(i * step + -Math.PI);
  return w;
}

/** A spectrum: each frequency, and the power density at it. */
export interface Spectrum {
  /** The frequencies, in hertz. */
  readonly f: Float64Array;
  /** The power density at each, in the signal's units squared per hertz. */
  readonly p: Float64Array;
}

/**
 * The periodogram of each segment, one row per segment: the segment less
 * its mean, windowed, transformed, squared and scaled to a one-sided
 * density, with the power at every frequency between zero and Nyquist
 * counted twice.
 */
function periodograms(x: ArrayLike<number>, fs: number, nperseg: number): { f: Float64Array; rows: Float64Array[] } {
  const win = hann(nperseg);
  const sq = new Float64Array(nperseg);
  for (let i = 0; i < nperseg; i++) sq[i] = win[i]! * win[i]!;
  const scale = 1.0 / (fs * numpySum(sq));
  const step = nperseg - (nperseg >> 1);
  const nseg = Math.floor((x.length - nperseg) / step) + 1;
  const nf = (nperseg >> 1) + 1;
  const val = 1.0 / (nperseg * (1 / fs));
  const f = new Float64Array(nf);
  for (let k = 0; k < nf; k++) f[k] = k * val;
  const last = nperseg % 2 ? nf : nf - 1;
  const rows: Float64Array[] = [];
  const seg = new Float64Array(nperseg);
  for (let s = 0; s < nseg; s++) {
    for (let i = 0; i < nperseg; i++) seg[i] = x[s * step + i]!;
    const m = numpyMean(seg);
    const re = new Float64Array(nperseg);
    const im = new Float64Array(nperseg);
    for (let i = 0; i < nperseg; i++) re[i] = win[i]! * (seg[i]! - m);
    fft(re, im);
    const p = new Float64Array(nf);
    for (let k = 0; k < nf; k++) {
      p[k] = (re[k]! * re[k]! + im[k]! * im[k]!) * scale;
      if (k >= 1 && k < last) p[k] = p[k]! * 2;
    }
    rows.push(p);
  }
  return { f, rows };
}

/** Rows averaged frequency by frequency, added in order, as numpy averages along an axis that is not the innermost. */
function averageRows(rows: readonly Float64Array[], nf: number): Float64Array {
  const out = new Float64Array(nf);
  for (const r of rows) for (let k = 0; k < nf; k++) out[k] = out[k]! + r[k]!;
  for (let k = 0; k < nf; k++) out[k] = out[k]! / rows.length;
  return out;
}

/** `scipy.signal.welch(x, fs, nperseg)` with its defaults: a periodic Hann window, half overlap, each segment's mean removed, a one-sided density, the mean over segments. */
export function welch(x: ArrayLike<number>, fs: number, nperseg: number): Spectrum {
  let n = Math.trunc(nperseg);
  if (n < 1) throw new DesignError("nperseg must be a positive integer");
  if (n > x.length) n = x.length;
  const { f, rows } = periodograms(x, fs, n);
  return { f, p: rows.length === 1 ? rows[0]! : averageRows(rows, f.length) };
}

/** `scipy.signal.welch` of each row of `rows`, every row one segment, then `p.mean(axis=0)`: the epochs' spectra averaged. */
export function welchEpochs(rows: readonly Float64Array[], fs: number): Spectrum {
  const n = rows[0]!.length;
  const spectra = rows.map((r) => periodograms(r, fs, n));
  const f = spectra[0]!.f;
  return { f, p: averageRows(spectra.map((s) => s.rows[0]!), f.length) };
}

// --- filter design -----------------------------------------------------------------

/** A complex number. */
type C = readonly [number, number];

const cmul = (a: C, b: C): C => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];

/** Complex division by Smith's method, as numpy divides. */
function cdiv(a: C, b: C): C {
  const [ar, ai] = a;
  const [br, bi] = b;
  if (Math.abs(br) >= Math.abs(bi)) {
    if (br === 0 && bi === 0) return [ar / Math.abs(br), ai / Math.abs(bi)];
    const rat = bi / br;
    const scl = 1.0 / (br + bi * rat);
    return [(ar + ai * rat) * scl, (ai - ar * rat) * scl];
  }
  const rat = br / bi;
  const scl = 1.0 / (bi + br * rat);
  return [(ar * rat + ai) * scl, (ai * rat - ar) * scl];
}

/** The product of complex numbers, multiplied left to right. */
function cprod(xs: readonly C[]): C {
  let acc: C = [1, 0];
  for (const x of xs) acc = cmul(acc, x);
  return acc;
}

/** One second-order section: three numerator and three denominator coefficients, the first denominator coefficient one. */
export type Section = readonly [number, number, number, number, number, number];

/**
 * `scipy.signal.butter(4, hz, btype, fs=fs, output="sos")` for a low-pass or
 * a high-pass: the analog prototype's poles on the unit circle, moved to the
 * pre-warped cutoff, carried through the bilinear transform, and paired
 * into sections as scipy's `zpk2sos` pairs them, the pole nearest the unit
 * circle last and the gain in the first section.
 */
export function butter4(hz: number, btype: "lowpass" | "highpass", fs: number): Section[] {
  const N = 4;
  const wn = hz / (fs / 2);
  if (wn <= 0) throw new DesignError("filter critical frequencies must be greater than 0");
  if (wn >= 1) {
    throw new DesignError(`Digital filter critical frequencies must be 0 < Wn < fs/2 (fs=${repr(fs)} -> fs/2=${repr(fs / 2)})`);
  }
  // The prototype: poles -exp(i pi m / 2N) for m = -N+1, -N+3, ..., N-1.
  const proto: C[] = [];
  for (let m = -N + 1; m < N; m += 2) {
    const angle = (Math.PI * m) * 0.125;
    proto.push([-Math.cos(angle), -Math.sin(angle)]);
  }
  const warped = 2 * 2.0 * Math.tan((Math.PI * wn) / 2.0);
  let z: C[];
  let p: C[];
  let k: number;
  if (btype === "lowpass") {
    z = [];
    p = proto.map((q) => [warped * q[0], warped * q[1]] as C);
    k = warped ** N;
  } else {
    z = [[0, 0], [0, 0], [0, 0], [0, 0]];
    p = proto.map((q) => cdiv([warped, 0], q));
    k = cdiv([1, 0], cprod(proto.map((q) => [-q[0], -q[1]] as C)))[0];
  }
  // The bilinear transform at fs = 2.
  const fs2 = 4.0;
  const zz: C[] = z.map((q) => cdiv([fs2 + q[0], q[1]], [fs2 - q[0], -q[1]]));
  const pz: C[] = p.map((q) => cdiv([fs2 + q[0], q[1]], [fs2 - q[0], -q[1]]));
  while (zz.length < pz.length) zz.push([-1, 0]);
  const num = cprod(z.map((q) => [fs2 - q[0], -q[1]] as C));
  const den = cprod(p.map((q) => [fs2 - q[0], -q[1]] as C));
  const kz = k * cdiv(num, den)[0];
  // `_cplxreal`: each conjugate pair as the mean of its two members, ordered by real part.
  const sorted = pz.map((q, i) => ({ q, i })).sort((a, b) => a.q[0] - b.q[0] || Math.abs(a.q[1]) - Math.abs(b.q[1]) || a.i - b.i);
  const upper = sorted.filter((e) => e.q[1] > 0).map((e) => e.q);
  const lower = sorted.filter((e) => e.q[1] < 0).map((e) => e.q);
  const pairs: C[] = upper.map((u, i) => [(u[0] + lower[i]![0]) / 2, (u[1] + -lower[i]![1]) / 2]);
  const zero = zz[0]![0];
  const sections: Section[] = new Array(2);
  for (let si = 1; si >= 0; si--) {
    let worst = 0;
    for (let i = 1; i < pairs.length; i++) {
      if (Math.abs(1 - Math.hypot(pairs[i]![0], pairs[i]![1])) < Math.abs(1 - Math.hypot(pairs[worst]![0], pairs[worst]![1]))) worst = i;
    }
    const [pr, pi] = pairs.splice(worst, 1)[0]!;
    sections[si] = [1, -(zero + zero), zero * zero, 1, -pr + -pr, pr * pr - pi * -pi];
  }
  const s0 = sections[0]!;
  sections[0] = [s0[0] * kz, s0[1] * kz, s0[2] * kz, s0[3], s0[4], s0[5]];
  return sections;
}

/** `scipy.signal.iirnotch(hz, q, fs)`: a second-order notch at `hz` with quality `q`. */
export function iirnotch(hz: number, q: number, fs: number): [number[], number[]] {
  let w0 = (2 * hz) / fs;
  if (w0 > 1.0 || w0 < 0.0) throw new DesignError("w0 should be such that 0 < w0 < 1");
  let bw = w0 / q;
  bw = bw * Math.PI;
  w0 = w0 * Math.PI;
  const beta = Math.tan(bw / 2.0);
  const gain = 1.0 / (1.0 + beta);
  const c = Math.cos(w0);
  return [
    [gain * 1.0, gain * (-2.0 * c), gain * 1.0],
    [1.0, -2.0 * gain * c, 2.0 * gain - 1.0],
  ];
}

// --- filtering ---------------------------------------------------------------------

/**
 * `scipy.signal.lfilter_zi(b, a)`: the state that holds a unit step steady,
 * solved by Gaussian elimination with partial pivoting as LAPACK solves it.
 * A system whose pivot is exactly zero is refused, as numpy refuses it.
 */
export function lfilterZi(bIn: readonly number[], aIn: readonly number[]): number[] {
  let b = [...bIn];
  let a = [...aIn];
  while (a.length > 1 && a[0] === 0.0) a = a.slice(1);
  if (a[0] !== 1.0) {
    const a0 = a[0]!;
    b = b.map((x) => x / a0);
    a = a.map((x) => x / a0);
  }
  const n = Math.max(a.length, b.length);
  while (a.length < n) a.push(0);
  while (b.length < n) b.push(0);
  const m = n - 1;
  // I - companion(a).T, whose first column is -a[1:] and whose superdiagonal is ones.
  const M: number[][] = [];
  for (let i = 0; i < m; i++) {
    const row: number[] = [];
    for (let j = 0; j < m; j++) {
      const companionT = j === 0 ? -a[i + 1]! : i === j - 1 ? 1 : 0;
      row.push((i === j ? 1 : 0) - companionT);
    }
    M.push(row);
  }
  const B: number[] = [];
  for (let i = 1; i < n; i++) B.push(b[i]! - a[i]! * b[0]!);
  for (let col = 0; col < m; col++) {
    let piv = col;
    for (let r = col + 1; r < m; r++) if (Math.abs(M[r]![col]!) > Math.abs(M[piv]![col]!)) piv = r;
    if (piv !== col) {
      [M[piv], M[col]] = [M[col]!, M[piv]!];
      [B[piv], B[col]] = [B[col]!, B[piv]!];
    }
    if (M[col]![col] === 0) throw new DesignError("Singular matrix");
    const recip = 1 / M[col]![col]!;
    for (let r = col + 1; r < m; r++) {
      const l = M[r]![col]! * recip;
      for (let c = col + 1; c < m; c++) M[r]![c] = M[r]![c]! - l * M[col]![c]!;
      B[r] = B[r]! - l * B[col]!;
    }
  }
  const zi = new Array<number>(m);
  for (let r = m - 1; r >= 0; r--) {
    let s = B[r]!;
    for (let c = r + 1; c < m; c++) s -= M[r]![c]! * zi[c]!;
    zi[r] = s / M[r]![r]!;
  }
  return zi;
}

/** `scipy.signal.sosfilt_zi(sos)`: each section's steady state, scaled by the gain of the sections before it. */
export function sosfiltZi(sos: readonly Section[]): number[][] {
  let scale = 1.0;
  return sos.map((s) => {
    const b = [s[0], s[1], s[2]];
    const a = [s[3], s[4], s[5]];
    const zi = lfilterZi(b, a).map((v) => scale * v);
    scale *= (b[0]! + b[1]! + b[2]!) / (a[0]! + a[1]! + a[2]!);
    return zi;
  });
}

/** `scipy.signal.lfilter(b, a, x, zi=zi)` for a second-order filter, transposed direct form II. */
function lfilter2(b: readonly number[], a: readonly number[], x: Float64Array, zi: readonly number[]): Float64Array {
  const a0 = a[0]!;
  const bn = b.map((v) => v / a0);
  const an = a.map((v) => v / a0);
  let z0 = zi[0]!;
  let z1 = zi[1]!;
  const y = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const xi = x[i]!;
    const yi = z0 + bn[0]! * xi;
    z0 = z1 + xi * bn[1]! - yi * an[1]!;
    z1 = xi * bn[2]! - yi * an[2]!;
    y[i] = yi;
  }
  return y;
}

/** `scipy.signal.sosfilt(sos, x, zi=zi)`: the sections in cascade, each in transposed direct form II. */
function sosfilt(sos: readonly Section[], x: Float64Array, zi: readonly (readonly number[])[]): Float64Array {
  const z = zi.map((s) => [s[0]!, s[1]!]);
  const y = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    let cur = x[i]!;
    for (let s = 0; s < sos.length; s++) {
      const c = sos[s]!;
      const zs = z[s]!;
      const next = c[0] * cur + zs[0]!;
      zs[0] = c[1] * cur - c[4] * next + zs[1]!;
      zs[1] = c[2] * cur - c[5] * next;
      cur = next;
    }
    y[i] = cur;
  }
  return y;
}

/** `scipy.signal`'s odd extension: `n` points reflected through each end. */
function oddExt(x: Float64Array, n: number): Float64Array {
  const len = x.length;
  const out = new Float64Array(len + 2 * n);
  for (let i = 0; i < n; i++) out[i] = 2 * x[0]! - x[n - i]!;
  out.set(x, n);
  for (let i = 0; i < n; i++) out[n + len + i] = 2 * x[len - 1]! - x[len - 2 - i]!;
  return out;
}

function padOrRefuse(x: Float64Array, edge: number): Float64Array {
  if (x.length <= edge) {
    throw new DesignError(`The length of the input vector x must be greater than padlen, which is ${edge}.`);
  }
  return oddExt(x, edge);
}

function reversed(x: Float64Array): Float64Array {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[x.length - 1 - i]!;
  return out;
}

/** `scipy.signal.filtfilt(b, a, x)` with its defaults: odd padding three times the filter's length, forward then backward from the steady state. */
export function filtfilt(b: readonly number[], a: readonly number[], x: Float64Array): Float64Array {
  const edge = 3 * Math.max(a.length, b.length);
  const ext = padOrRefuse(x, edge);
  const zi = lfilterZi(b, a);
  let y = lfilter2(b, a, ext, zi.map((v) => v * ext[0]!));
  y = lfilter2(b, a, reversed(y), zi.map((v) => v * y[y.length - 1]!));
  y = reversed(y);
  return y.slice(edge, y.length - edge);
}

/** `scipy.signal.sosfiltfilt(sos, x)` with its defaults. */
export function sosfiltfilt(sos: readonly Section[], x: Float64Array): Float64Array {
  let ntaps = 2 * sos.length + 1;
  const zb = sos.filter((s) => s[2] === 0).length;
  const za = sos.filter((s) => s[5] === 0).length;
  ntaps -= Math.min(zb, za);
  const edge = ntaps * 3;
  const ext = padOrRefuse(x, edge);
  const zi = sosfiltZi(sos);
  let y = sosfilt(sos, ext, zi.map((s) => s.map((v) => v * ext[0]!)));
  const last = y[y.length - 1]!;
  y = sosfilt(sos, reversed(y), zi.map((s) => s.map((v) => v * last)));
  y = reversed(y);
  return y.slice(edge, y.length - edge);
}
