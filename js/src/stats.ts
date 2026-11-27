/**
 * The statistical tests the analyses report, written here so the package
 * carries no dependency: Wilcoxon's signed-rank test, the paired t test and
 * the Mann-Whitney U test, each choosing its method as scipy 1.15 chooses
 * it, with the normal and Student tail probabilities they need. Written from
 * the tests' documented definitions and held to the Python library's results
 * by `contract/analysis`. Internal: the analyses are the surface.
 */

import { numpySum } from "./capture/pyfmt.ts";

/** One test's statistic and p value. */
export interface Result {
  /** The test's statistic. */
  readonly statistic: number;
  /** The probability of a statistic at least this extreme if nothing differed. */
  readonly pvalue: number;
}

// --- tails ------------------------------------------------------------------------

const SQRT_HALF = 0.7071067811865476;
const TWO_OVER_SQRT_PI = 1.1283791670955126;
const ONE_OVER_SQRT_PI = 0.5641895835477563;

/** The error function by its series of positive terms, `2x exp(-x^2) / sqrt(pi)` times the sum of `(2x^2)^n / (1 * 3 * ... * (2n + 1))`. */
function erfSeries(x: number): number {
  const x2 = x * x;
  let term = 1;
  let sum = 1;
  for (let n = 1; n < 500; n++) {
    term *= (2 * x2) / (2 * n + 1);
    sum += term;
    if (term < sum * 1e-17) break;
  }
  return TWO_OVER_SQRT_PI * x * Math.exp(-x2) * sum;
}

/** The complementary error function for `x >= 2`, by its continued fraction evaluated with Lentz's method. */
function erfcFraction(x: number): number {
  // erfc(x) = exp(-x^2) / sqrt(pi) * 1 / (x + (1/2) / (x + 1 / (x + (3/2) / (x + 2 / (x + ...)))))
  const tiny = 1e-300;
  let f = x;
  let c = x;
  let d = 0;
  for (let k = 1; k < 1000; k++) {
    const a = k / 2;
    d = x + a * d;
    d = d === 0 ? tiny : d;
    c = x + a / c;
    c = c === 0 ? tiny : c;
    d = 1 / d;
    const delta = c * d;
    f *= delta;
    if (Math.abs(delta - 1) < 1e-16) break;
  }
  return (ONE_OVER_SQRT_PI * Math.exp(-x * x)) / f;
}

/** The error function. */
export function erf(x: number): number {
  if (Number.isNaN(x)) return NaN;
  const a = Math.abs(x);
  const v = a < 2 ? erfSeries(a) : 1 - erfcFraction(a);
  return x < 0 ? -v : v;
}

/** The largest power of e a double holds, below which cephes' complementary error function underflows to zero. */
const MAXLOG = 709.782712893384;

/** The complementary error function, zero where `exp(-x^2)` would underflow, as cephes stops there. */
export function erfc(x: number): number {
  if (Number.isNaN(x)) return NaN;
  const a = Math.abs(x);
  if (a >= 1 && a * a > MAXLOG) return x < 0 ? 2 : 0;
  const v = a < 2 ? 1 - erfSeries(a) : erfcFraction(a);
  return x < 0 ? 2 - v : v;
}

/** `scipy.special.ndtr`: the standard normal distribution below `a`, from the error function near zero and its complement beyond, as cephes computes it. */
export function ndtr(a: number): number {
  if (Number.isNaN(a)) return NaN;
  const x = a * SQRT_HALF;
  const z = Math.abs(x);
  if (z < SQRT_HALF) return 0.5 + 0.5 * erf(x);
  const y = 0.5 * erfc(z);
  return x > 0 ? 1.0 - y : y;
}

const LANCZOS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
  12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];

/** The logarithm of the gamma function for positive arguments, by Lanczos' approximation with g = 7. */
function lgamma(x: number): number {
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  const z = x - 1;
  let a = LANCZOS[0]!;
  const t = z + 7.5;
  for (let i = 1; i < 9; i++) a += LANCZOS[i]! / (z + i);
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
}

/** The continued fraction of the incomplete beta function, by Lentz's method. */
function betaFraction(a: number, b: number, x: number): number {
  const tiny = 1e-300;
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  d = Math.abs(d) < tiny ? tiny : d;
  d = 1 / d;
  let h = d;
  for (let m = 1; m < 10000; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    d = Math.abs(d) < tiny ? tiny : d;
    c = 1 + aa / c;
    c = Math.abs(c) < tiny ? tiny : c;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    d = Math.abs(d) < tiny ? tiny : d;
    c = 1 + aa / c;
    c = Math.abs(c) < tiny ? tiny : c;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-16) break;
  }
  return h;
}

/** The regularized incomplete beta function `I_x(a, b)`. */
function incbeta(a: number, b: number, x: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const front = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log1p(-x));
  if (x < (a + 1) / (a + b + 2)) return (front * betaFraction(a, b, x)) / a;
  return 1 - (front * betaFraction(b, a, 1 - x)) / b;
}

/** `scipy.special.stdtr`: Student's t distribution with `df` degrees of freedom below `t`. */
export function stdtr(df: number, t: number): number {
  if (Number.isNaN(t) || Number.isNaN(df)) return NaN;
  if (t === 0) return 0.5;
  if (t === -Infinity) return 0;
  if (t === Infinity) return 1;
  const tail = 0.5 * incbeta(df / 2, 0.5, df / (df + t * t));
  return t < 0 ? tail : 1 - tail;
}

// --- ranks ------------------------------------------------------------------------

/**
 * scipy's average ranks: each value's position counting from one, ties
 * given the mean of their positions, with NaN after every number and each
 * NaN its own group. `ties` holds each group's size at its first member
 * and zero elsewhere, as scipy's `_rankdata` returns it.
 */
export function rankAverage(x: ArrayLike<number>): { ranks: Float64Array; ties: Float64Array } {
  const n = x.length;
  const order = Array.from({ length: n }, (_, i) => i);
  order.sort((i, j) => {
    const a = x[i]!;
    const b = x[j]!;
    if (Number.isNaN(a)) return Number.isNaN(b) ? i - j : 1;
    if (Number.isNaN(b)) return -1;
    return a < b ? -1 : a > b ? 1 : i - j;
  });
  const ranks = new Float64Array(n);
  const ties = new Float64Array(n);
  let start = 0;
  while (start < n) {
    let end = start + 1;
    const v = x[order[start]!]!;
    while (end < n && x[order[end]!]! === v) end++;
    const count = end - start;
    const r = start + 1 + (count - 1) / 2;
    for (let k = start; k < end; k++) ranks[order[k]!] = r;
    ties[order[start]!] = count;
    start = end;
  }
  return { ranks, ties };
}

// --- Wilcoxon's signed-rank test ---------------------------------------------------

/** The exact null distribution of the signed-rank sum of `n` observations, as probabilities, halving and shifting one observation at a time. */
function wilcoxonDistribution(n: number): Float64Array {
  let c = new Float64Array([1]);
  for (let k = 1; k <= n; k++) {
    const prev = c;
    c = new Float64Array((k * (k + 1)) / 2 + 1);
    const m = prev.length;
    for (let i = 0; i < m; i++) c[i] = prev[i]! * 0.5;
    for (let i = 0; i < m; i++) c[c.length - m + i] = c[c.length - m + i]! + prev[i]! * 0.5;
  }
  return c;
}

/** The signed-rank sum of the positive differences, zeros left out as Wilcoxon left them out. */
function signedRanks(d: Float64Array): { rPlus: number; count: number; tieCorrect: number; hasTies: boolean; ranks: Float64Array } {
  const abs = new Float64Array(d.length);
  let count = 0;
  for (let i = 0; i < d.length; i++) {
    abs[i] = d[i] === 0 || Number.isNaN(d[i]!) ? NaN : Math.abs(d[i]!);
    if (!Number.isNaN(abs[i]!)) count++;
  }
  const { ranks, ties } = rankAverage(abs);
  const plus = new Float64Array(d.length);
  for (let i = 0; i < d.length; i++) plus[i] = (d[i]! > 0 ? 1 : 0) * ranks[i]!;
  const cube = new Float64Array(d.length);
  let hasTies = false;
  for (let i = 0; i < d.length; i++) {
    const t = ties[i]!;
    cube[i] = t ** 3 - t;
    if (t === 0) hasTies = true;
  }
  return { rPlus: numpySum(plus), count, tieCorrect: numpySum(cube), hasTies, ranks };
}

/**
 * `scipy.stats.wilcoxon(x, y, alternative="greater")` with its defaults in
 * scipy 1.15: zeros dropped, no continuity correction, and the method by its
 * "auto" rule: the normal approximation past fifty pairs, and otherwise the
 * exact distribution when no difference is zero and none tie, every sign
 * flip counted when there are and at most thirteen pairs, and the normal
 * approximation for more.
 */
export function wilcoxonGreater(x: ArrayLike<number>, y: ArrayLike<number>): Result {
  const n = x.length;
  const d = new Float64Array(n);
  let zeros = 0;
  for (let i = 0; i < n; i++) {
    d[i] = x[i]! - y[i]!;
    if (d[i] === 0) zeros++;
  }
  if (n === 0) return { statistic: NaN, pvalue: NaN };
  const { rPlus, count, tieCorrect, hasTies, ranks } = signedRanks(d);
  const mn = count * (count + 1.0) * 0.25;
  let se = count * (count + 1.0) * (2.0 * count + 1.0);
  se -= tieCorrect / 2;
  se = Math.sqrt(se / 24);
  const z = (rPlus - mn) / se;
  let method: "asymptotic" | "exact" | "permutation";
  if (n > 50) method = "asymptotic";
  else if (!(hasTies || zeros > 0)) method = "exact";
  else if (n <= 13) method = "permutation";
  else method = "asymptotic";
  let p: number;
  if (method === "asymptotic") {
    p = ndtr(-z);
  } else if (method === "exact") {
    const pmf = wilcoxonDistribution(count);
    const k = Math.floor(rPlus);
    const mean = (count * (count + 1)) / 4;
    p = k <= mean ? numpySum(pmf.subarray(k)) : 1 - numpySum(pmf.subarray(0, k));
  } else {
    // Every pattern of signs, the observed one among them: the share whose
    // signed-rank sum reaches the observed one, within scipy's tolerance.
    const gamma = Math.abs(2.220446049250313e-16 * 100 * rPlus);
    const total = 2 ** n;
    let reach = 0;
    const plus = new Float64Array(n);
    for (let mask = 0; mask < total; mask++) {
      for (let i = 0; i < n; i++) {
        const s = (mask >> i) & 1 ? -d[i]! : d[i]!;
        plus[i] = (s > 0 ? 1 : 0) * ranks[i]!;
      }
      if (numpySum(plus) >= rPlus - gamma) reach++;
    }
    p = reach / total;
  }
  return { statistic: rPlus, pvalue: p };
}

// --- the paired t test --------------------------------------------------------------

/** `scipy.stats.ttest_rel(a, b)`: the t statistic of the differences and its two-sided p value. */
export function ttestRel(a: ArrayLike<number>, b: ArrayLike<number>): Result {
  const n = a.length;
  const d = new Float64Array(n);
  for (let i = 0; i < n; i++) d[i] = a[i]! - b[i]!;
  if (n === 0) return { statistic: NaN, pvalue: NaN };
  const mean = numpySum(d) / n;
  const sq = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const dev = d[i]! - mean;
    sq[i] = dev * dev;
  }
  let v = numpySum(sq) / n;
  v *= n / (n - 1);
  const denom = Math.sqrt(v / n);
  const t = mean / denom;
  return { statistic: t, pvalue: 2 * stdtr(n - 1, -Math.abs(t)) };
}

// --- the Mann-Whitney U test --------------------------------------------------------

/** The binomial coefficient, as a double. */
function binom(n: number, k: number): number {
  let r = 1;
  for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i;
  return Math.round(r);
}

/** The exact distribution of U for samples of `n1` and `n2`, counts of each value by the recurrence scipy uses, as probabilities. */
function mwuDistribution(n1: number, n2: number, maxu: number): Float64Array {
  const a = Math.min(n1, n2);
  const b = Math.max(n1, n2);
  const sigma = new Float64Array(maxu + 1);
  for (let d = 1; d <= a; d++) for (let i = d; i <= maxu; i += d) sigma[i] = sigma[i]! + d;
  for (let d = b + 1; d <= b + a; d++) for (let i = d; i <= maxu; i += d) sigma[i] = sigma[i]! - d;
  const counts = new Float64Array(maxu + 1);
  counts[0] = 1;
  for (let u = 1; u <= maxu; u++) {
    let dot = 0;
    for (let i = 0; i < u; i++) dot += counts[i]! * sigma[u - i]!;
    counts[u] = Math.trunc(dot / u);
  }
  const total = binom(a + b, a);
  const out = new Float64Array(maxu + 1);
  for (let u = 0; u <= maxu; u++) out[u] = counts[u]! / total;
  return out;
}

/** The share of U at or above `k`. */
function mwuSf(n1: number, n2: number, k: number): number {
  const kc = n1 * n2 - k;
  const cdf = (q: number): number => {
    const pmf = mwuDistribution(n1, n2, q);
    let acc = 0;
    for (let u = 0; u <= q; u++) acc += pmf[u]!;
    return acc;
  };
  if (k < kc) return 1 - cdf(k) + mwuDistribution(n1, n2, k)[k]!;
  return cdf(kc);
}

/**
 * `scipy.stats.mannwhitneyu(x, y, alternative="greater")` with its
 * defaults in scipy 1.15: the exact distribution when one sample has at
 * most eight values and none tie, and otherwise the normal approximation
 * with the tie correction and the continuity correction.
 */
export function mannwhitneyuGreater(x: ArrayLike<number>, y: ArrayLike<number>): Result {
  const n1 = x.length;
  const n2 = y.length;
  const xy = new Float64Array(n1 + n2);
  for (let i = 0; i < n1; i++) xy[i] = x[i]!;
  for (let i = 0; i < n2; i++) xy[n1 + i] = y[i]!;
  const { ranks, ties } = rankAverage(xy);
  const r1 = numpySum(ranks.subarray(0, n1));
  const u1 = r1 - (n1 * (n1 + 1)) / 2;
  let tied = false;
  for (const t of ties) if (t > 1) tied = true;
  let p: number;
  if ((n1 > 8 && n2 > 8) || tied) {
    const mu = (n1 * n2) / 2;
    const n = n1 + n2;
    const cube = new Float64Array(ties.length);
    for (let i = 0; i < ties.length; i++) cube[i] = ties[i]! ** 3 - ties[i]!;
    const tieTerm = numpySum(cube);
    const s = Math.sqrt(((n1 * n2) / 12) * (n + 1 - tieTerm / (n * (n - 1))));
    const z = (u1 - mu - 0.5) / s;
    p = ndtr(-z);
  } else {
    p = mwuSf(n1, n2, Math.trunc(u1));
  }
  return { statistic: u1, pvalue: Math.min(1, Math.max(0, p)) };
}
