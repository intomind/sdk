/**
 * Head training as the Python library's `model` module does it, and the
 * arithmetic under it: what the library takes as rows of numbers and as
 * targets, and its words where they are not, rounding half to even, the
 * eigenvalues of a symmetric matrix, and the solution of a linear system
 * with partial pivoting.
 *
 * Internal. `model.ts` publishes `gapRidge`, `trainHead` and
 * `blockedFolds`. The tests reach `train` here, which also hands back what
 * every weight, bias and scale held before it was rounded, so that where
 * this arithmetic and numpy's differ in the last bits the tests can show by
 * how much, and that a byte rounded the other way sat on a rounding
 * boundary.
 *
 * What the library refuses with Python's `ValueError` is refused with the
 * package's `Invalid`, its words kept and `ValueError` at their head, and
 * what it refuses with `TypeError` with a `TypeError` in its words. What
 * the library refuses with its own `Invalid` is refused with this package's
 * `Invalid`, in the same words. Where the library's words show None, True,
 * nan or inf, the package's show null, true, NaN or Infinity.
 */

import { general, repr } from "./capture/pyfmt.ts";
import { holdsTooLong, quoted, tooLong } from "./capture/pyjson.ts";
import { pyFloat } from "./py.ts";
import { Invalid, buildHead, quantizeEmbedding } from "./protocol.ts";

// --- what a refusal shows -------------------------------------------------------

function isSequence(v: unknown): v is ArrayLike<unknown> {
  return Array.isArray(v) || (ArrayBuffer.isView(v) && !(v instanceof DataView));
}

/** A refusal of a value, as the library's `ValueError` is refused here. */
function valueError(words: string): Invalid {
  return new Invalid(`ValueError: ${words}`);
}

/** A number as a refusal shows it: a whole number as one, NaN and the infinities as JavaScript writes them, and any other number as Python writes a float, such as 1e+39. */
function shownNumber(x: number): string {
  if (!Number.isFinite(x)) return String(x);
  if (Number.isSafeInteger(x) && !Object.is(x, -0)) return String(x);
  return repr(x);
}

/** A number as Python's `{x:g}` writes it, NaN and the infinities as JavaScript writes them. */
function shownGeneral(x: number): string {
  return Number.isFinite(x) ? general(x, 6) : String(x);
}

/**
 * A value a caller passed, as a refusal names it, as the library's `_given`
 * does: a number, text, a boolean, null or a row as JavaScript writes it,
 * text in double quotes, and anything else by its type. A whole number of
 * more digits than Python turns into text, 4,300, and text written as one,
 * are named by how many digits they have, never printed, and a row holding
 * such a whole number by its type, as Python, which will not print it,
 * names it.
 */
function given(v: unknown): string {
  if (v === null) return "null";
  if (v === undefined) return "undefined";
  if (typeof v === "number") return shownNumber(v);
  const long = tooLong(v);
  if (long !== null) return long;
  if (typeof v === "bigint" || typeof v === "boolean") return String(v);
  if (typeof v === "string") return quoted(v);
  if (isSequence(v) && holdsTooLong(v)) return "a list of more digits than Python prints";
  if (isSequence(v)) return `[${Array.from(v, given).join(", ")}]`;
  if (typeof v === "function") return "a function";
  const type = (v as { constructor?: { name?: string } }).constructor?.name;
  return type && type !== "Object" ? `a ${type}` : "an object";
}

/** `n` values, as the library counts them. */
function values(n: number): string {
  return n === 1 ? "1 value" : `${n} values`;
}

// --- rows of numbers, as the library reads them ----------------------------------

/**
 * What numpy makes of one value in an array of 64-bit floats: the number,
 * NaN for null, as numpy makes None, or undefined where numpy refuses the
 * value, such as a row, text that is not a number, or a whole number past
 * every float.
 */
function asFloat(v: unknown): number | undefined {
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (typeof v === "bigint") {
    const x = Number(v);
    return Number.isFinite(x) ? x : undefined;
  }
  if (v === null || v === undefined) return Number.NaN;
  if (typeof v === "string") return pyFloat(v) ?? undefined;
  return undefined;
}

/** `value` as rows of 64-bit floats, where numpy makes rows of it: a row a window, every row as long as the first. Null otherwise. */
function asRows(value: unknown): Float64Array[] | null {
  if (!isSequence(value) || value.length === 0) return null;
  const first = value[0];
  if (!isSequence(first)) return null;
  const d = first.length;
  const rows: Float64Array[] = [];
  for (let i = 0; i < value.length; i++) {
    const r = value[i];
    if (!isSequence(r) || r.length !== d) return null;
    const f = new Float64Array(d);
    for (let j = 0; j < d; j++) {
      const x = asFloat(r[j]);
      if (x === undefined) return null;
      f[j] = x;
    }
    rows.push(f);
  }
  return rows;
}

/** The refusal, in `words`, of a value `v` that is not a number: a TypeError for null, which is no number at all, as the library's for None, and a ValueError for anything else. */
function notANumber(v: unknown, words: string): Error {
  return v === null || v === undefined ? new TypeError(words) : valueError(words);
}

/** Refuse `v`, found at `where`, unless the library's `float` takes it. */
function refuseUnlessNumber(v: unknown, where: string): void {
  if (typeof v === "bigint" && !Number.isFinite(Number(v))) throw valueError(`${where} is ${given(v)}, which a 64-bit float cannot hold`);
  if (v !== null && v !== undefined && asFloat(v) !== undefined) return;
  if (typeof v === "string") throw valueError(`${where} is ${given(v)}, which is not a number`);
  throw new TypeError(`${where} is ${given(v)}, which is not a number`);
}

/** Refuse the first row of `rows` that is not as long as the first, or the first value in a row that is not a number, naming it. */
function refuseParts(rows: ArrayLike<ArrayLike<unknown>>, name: string): void {
  const width = rows[0]!.length;
  for (let i = 0; i < rows.length; i++) {
    const n = rows[i]!.length;
    if (n !== width) throw valueError(`${name}[${i}] holds ${values(n)} where ${name}[0] holds ${width}, and every row is as long as the first`);
  }
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i]!;
    for (let j = 0; j < row.length; j++) {
      const v = row[j];
      if (isSequence(v)) throw valueError(`${name}[${i}][${j}] is ${given(v)}, and each value in a row is a number`);
      refuseUnlessNumber(v, `${name}[${i}][${j}]`);
    }
  }
}

/** Refuse the first value of `rows`, read from `value`, for which `ok` is false, naming it as the caller passed it. */
function refuseUnless(rows: readonly Float64Array[], ok: (x: number) => boolean, value: ArrayLike<ArrayLike<unknown>>, name: string, what: string): void {
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i]!;
    for (let j = 0; j < row.length; j++) {
      if (ok(row[j]!)) continue;
      const v = value[i]![j];
      throw notANumber(v, `${name}[${i}][${j}] is ${given(v)}, ${what}`);
    }
  }
}

/**
 * `value`, one row of numbers a window, every row as long as the first, as
 * 64-bit floats, as the library's `_rows` reads it; anything else refused
 * in words naming the first part of it that is wrong. `rule` says what
 * `name` is. A value that is not a number is refused, unless `nan` allows
 * it.
 */
function rowsOf(value: unknown, rule: string, name: string, nan = false): Float64Array[] {
  const rows = asRows(value);
  if (rows !== null) {
    if (!nan) refuseUnless(rows, (x) => !Number.isNaN(x), value as ArrayLike<ArrayLike<unknown>>, name, "which is not a number");
    return rows;
  }
  if (!isSequence(value)) throw new TypeError(`${rule}, not ${given(value)}`);
  if (value.length === 0) throw valueError(`${rule}, and no rows were given`);
  for (let i = 0; i < value.length; i++) {
    if (!isSequence(value[i])) throw valueError(`${rule}, and ${name}[${i}] is ${given(value[i])}`);
  }
  refuseParts(value as ArrayLike<ArrayLike<unknown>>, name);
  throw valueError(rule);
}

/**
 * `rows` checked as the library's `blocked_folds` checks them before it
 * trains a head: one row of numbers a window, every row as long as the
 * first, and of at least two values, refused in the library's words
 * otherwise.
 */
export function checkRows(rows: unknown): void {
  const d = rowsOf(rows, "rows are one embedding per window", "rows")[0]!.length;
  if (d < 2) throw valueError(`every row of rows holds ${values(d)}, and a ridge is chosen from rows of at least two values`);
}

/** What `trainHead`'s targets are. */
const TARGETS = "targets are one number or one row of numbers per window";

/**
 * The words every refusal of embeddings that do not determine one head
 * holds, so that a caller can tell that refusal from the others.
 */
export const UNDETERMINED = "do not determine one head";

/**
 * The refusal of embeddings that with the ridge chosen from them do not
 * determine one head, as the library's `_DEAD` is.
 */
export const DEAD =
  `with the ridge chosen from these embeddings, they ${UNDETERMINED}, because there are two or more ` +
  "directions along which they do not vary";

/**
 * The advice `trainHead` adds to `DEAD`, which only a caller that can give a
 * ridge is given, as the library's `_GIVE_A_RIDGE` is.
 */
export const GIVE_A_RIDGE = "; give a ridge above 0";

/**
 * `targets` as one row a window and one column an output, in 64-bit
 * floats, as the library's `_target_rows` reads them, with how many outputs
 * they have; anything else refused in words naming the first part of it
 * that is wrong, a value that is not a finite number among them.
 */
function targetRows(targets: unknown): { rows: Float64Array[]; outputs: number } {
  if (isSequence(targets)) {
    // One number a window.
    const one: number[] = [];
    for (let i = 0; i < targets.length; i++) {
      const v = targets[i];
      const x = isSequence(v) ? undefined : asFloat(v);
      if (x === undefined) break;
      one.push(x);
    }
    if (one.length === targets.length) {
      const i = one.findIndex((x) => !Number.isFinite(x));
      if (i >= 0) throw notANumber(targets[i], `targets[${i}] is ${given(targets[i])}, which is not a finite number`);
      return { rows: one.map((x) => Float64Array.of(x)), outputs: 1 };
    }
    // One row a window.
    const rows = asRows(targets);
    if (rows !== null) {
      refuseUnless(rows, Number.isFinite, targets as ArrayLike<ArrayLike<unknown>>, "targets", "which is not a finite number");
      return { rows, outputs: rows[0]!.length };
    }
  }
  if (!isSequence(targets)) throw new TypeError(`${TARGETS}, not ${given(targets)}`);
  const rowsAt: number[] = [];
  for (let i = 0; i < targets.length; i++) if (isSequence(targets[i])) rowsAt.push(i);
  if (rowsAt.length === 0) {
    for (let i = 0; i < targets.length; i++) refuseUnlessNumber(targets[i], `targets[${i}]`);
    throw valueError(TARGETS);
  }
  if (rowsAt.length < targets.length) {
    let i = 0;
    while (isSequence(targets[i])) i++;
    throw valueError(`${TARGETS}, and targets[${rowsAt[0]}] is a row where targets[${i}] is ${given(targets[i])}`);
  }
  refuseParts(targets as ArrayLike<ArrayLike<unknown>>, "targets");
  throw valueError(TARGETS);
}

const INTEGER_ARRAYS = [
  Int8Array, Uint8Array, Uint8ClampedArray, Int16Array, Uint16Array, Int32Array, Uint32Array, BigInt64Array, BigUint64Array,
];

/**
 * Whether embeddings are the device's integers rather than the encoder's
 * floats, decided over the whole input as `np.asarray` decides an array's
 * type in Python. A row in an integer typed array is integers and a row in
 * any other typed array is floats. A plain row is integers when every value
 * in it is a safe integer, as the package's embedding windows carry them and
 * as integers loaded from JSON arrive, and floats otherwise. One float row
 * makes every row float, as one float in a Python list makes numpy's array
 * float.
 */
function integerRows(rows: ArrayLike<unknown>): boolean {
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    if (ArrayBuffer.isView(r)) {
      if (!INTEGER_ARRAYS.some((T) => r instanceof T)) return false;
      continue;
    }
    const row = r as ArrayLike<unknown>;
    for (let j = 0; j < row.length; j++) if (!Number.isSafeInteger(row[j])) return false;
  }
  return true;
}

// --- numbers ------------------------------------------------------------------

/** Round half to even, as numpy's `rint` and Python's `round` do. */
export function rint(x: number): number {
  const f = Math.floor(x);
  const d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

/**
 * The eigenvalues of a symmetric matrix, ascending, by cyclic Jacobi
 * rotations with Rutishauser's thresholds, to the matrix's last bits.
 * `a` is `n` by `n`, row-major, and is left as it was. A matrix holding a
 * value that is not finite has eigenvalues that are not numbers, as numpy
 * hands back. Null where they have not settled in 100 sweeps.
 */
export function symmetricEigenvalues(a: Float64Array, n: number): Float64Array | null {
  for (const v of a) if (!Number.isFinite(v)) return new Float64Array(n).fill(Number.NaN);
  const m = Float64Array.from(a);
  const d = new Float64Array(n);
  const b = new Float64Array(n);
  const z = new Float64Array(n);
  for (let p = 0; p < n; p++) d[p] = b[p] = m[p * n + p]!;
  const rotate = (i: number, j: number, k: number, l: number, s: number, tau: number): void => {
    const g = m[i * n + j]!;
    const h = m[k * n + l]!;
    m[i * n + j] = g - s * (h + g * tau);
    m[k * n + l] = h + s * (g - h * tau);
  };
  for (let sweep = 1; sweep <= 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) off += Math.abs(m[p * n + q]!);
    if (off === 0) return Float64Array.from(d).sort();
    const threshold = sweep < 4 ? (0.2 * off) / (n * n) : 0;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = m[p * n + q]!;
        const g = 100 * Math.abs(apq);
        if (sweep > 4 && Math.abs(d[p]!) + g === Math.abs(d[p]!) && Math.abs(d[q]!) + g === Math.abs(d[q]!)) {
          m[p * n + q] = 0;
        } else if (Math.abs(apq) > threshold) {
          const diff = d[q]! - d[p]!;
          let t: number;
          if (Math.abs(diff) + g === Math.abs(diff)) {
            t = apq / diff;
          } else {
            const theta = (0.5 * diff) / apq;
            t = 1 / (Math.abs(theta) + Math.sqrt(1 + theta * theta));
            if (theta < 0) t = -t;
          }
          const c = 1 / Math.sqrt(1 + t * t);
          const s = t * c;
          const tau = s / (1 + c);
          const h = t * apq;
          z[p] = z[p]! - h;
          z[q] = z[q]! + h;
          d[p] = d[p]! - h;
          d[q] = d[q]! + h;
          m[p * n + q] = 0;
          for (let j = 0; j < p; j++) rotate(j, p, j, q, s, tau);
          for (let j = p + 1; j < q; j++) rotate(p, j, j, q, s, tau);
          for (let j = q + 1; j < n; j++) rotate(p, j, q, j, s, tau);
        }
      }
    }
    for (let p = 0; p < n; p++) {
      b[p] = b[p]! + z[p]!;
      d[p] = b[p]!;
      z[p] = 0;
    }
  }
  return null;
}

/**
 * `a x = b` for `x`, `a` square of size `m` and `b` `m` by `k`, both
 * row-major, by elimination with partial pivoting as LAPACK's `gesv` does
 * it: the first row with the largest magnitude in the column is the pivot.
 * A pivot of exactly zero is refused in numpy's words.
 */
export function solve(a: Float64Array, b: Float64Array, m: number, k: number): Float64Array {
  const A = Float64Array.from(a);
  const B = Float64Array.from(b);
  for (let col = 0; col < m; col++) {
    let pivot = col;
    let best = Math.abs(A[col * m + col]!);
    for (let r = col + 1; r < m; r++) {
      const v = Math.abs(A[r * m + col]!);
      if (v > best) {
        best = v;
        pivot = r;
      }
    }
    if (best === 0) throw new Invalid("LinAlgError: Singular matrix");
    if (pivot !== col) {
      for (let j = 0; j < m; j++) {
        const t = A[col * m + j]!;
        A[col * m + j] = A[pivot * m + j]!;
        A[pivot * m + j] = t;
      }
      for (let j = 0; j < k; j++) {
        const t = B[col * k + j]!;
        B[col * k + j] = B[pivot * k + j]!;
        B[pivot * k + j] = t;
      }
    }
    const p = A[col * m + col]!;
    for (let r = col + 1; r < m; r++) {
      const f = A[r * m + col]! / p;
      if (f === 0) continue;
      for (let j = col; j < m; j++) A[r * m + j] = A[r * m + j]! - f * A[col * m + j]!;
      for (let j = 0; j < k; j++) B[r * k + j] = B[r * k + j]! - f * B[col * k + j]!;
    }
  }
  const X = new Float64Array(m * k);
  for (let r = m - 1; r >= 0; r--) {
    for (let j = 0; j < k; j++) {
      let s = B[r * k + j]!;
      for (let c = r + 1; c < m; c++) s -= A[r * m + c]! * X[c * k + j]!;
      X[r * k + j] = s / A[r * m + r]!;
    }
  }
  return X;
}

// --- the ridge ----------------------------------------------------------------

/** The Gram matrix of rows `n` by `d`, each row centered on the rows' mean, `d` by `d`, row-major. */
function centeredGram(rows: readonly Float64Array[], d: number): Float64Array {
  const n = rows.length;
  const mean = new Float64Array(d);
  for (const r of rows) for (let j = 0; j < d; j++) mean[j] = mean[j]! + r[j]!;
  for (let j = 0; j < d; j++) mean[j] = mean[j]! / n;
  const gram = new Float64Array(d * d);
  const xc = new Float64Array(d);
  for (const r of rows) {
    for (let j = 0; j < d; j++) xc[j] = r[j]! - mean[j]!;
    for (let i = 0; i < d; i++) {
      const xi = xc[i]!;
      for (let j = i; j < d; j++) gram[i * d + j] = gram[i * d + j]! + xi * xc[j]!;
    }
  }
  for (let i = 0; i < d; i++) for (let j = 0; j < i; j++) gram[i * d + j] = gram[j * d + i]!;
  return gram;
}

/** The gap ridge of a centered Gram matrix `d` by `d`, `d` at least two, or null where its eigenvalues have not settled. */
function ridgeOf(gram: Float64Array, d: number): number | null {
  const ev = symmetricEigenvalues(gram, d);
  if (ev === null) return null;
  return Math.sqrt(Math.max(ev[0]!, 1) * ev[1]!);
}

/**
 * The ridge `trainHead` uses unless told otherwise, from the embeddings
 * alone, as the library's `model.gap_ridge` chooses it: the geometric mean
 * of the two smallest eigenvalues of the centered embeddings' Gram matrix,
 * the smallest floored at one. No label enters the choice. `quantized` that
 * is not one row of numbers a window, every row as long as the first and of
 * at least two values, is refused in the library's words, naming what is
 * wrong in it, and so are values the eigenvalues cannot be found for.
 */
export function gapRidge(quantized: unknown): number {
  const x = rowsOf(quantized, "quantized is one row per window", "quantized", true);
  const d = x[0]!.length;
  if (d < 2) throw valueError(`every row of quantized holds ${values(d)}, and a ridge is chosen from rows of at least two values`);
  const ridge = ridgeOf(centeredGram(x, d), d);
  if (ridge === null) throw valueError("the eigenvalues a ridge is chosen from did not converge for quantized");
  return ridge;
}

// --- the states a head tells apart ---------------------------------------------

/**
 * What a head for `states` states is fitted to, as the library's `_targets`
 * gives it: for two, the labels themselves, so its one output says the
 * state labeled 1, and for more, one row a window and one column a state,
 * 1 for the window's own state and 0 for the others.
 */
export function targets(labels: readonly number[], states: number): readonly number[] | number[][] {
  if (states === 2) return labels;
  return labels.map((y) => Array.from({ length: states }, (_, s) => (y === s ? 1.0 : 0.0)));
}

/**
 * The state a head with one output for each state says, as the library's
 * `_picked` reads it: the one whose output is the largest, as the device
 * picks. When two are equal, the first.
 */
export function picked(outputs: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < outputs.length; i++) if (outputs[i]! > outputs[best]!) best = i;
  return best;
}

// --- training -----------------------------------------------------------------

/** How a head is trained: the options of the library's `train_head`. */
export interface TrainOptions {
  /** The head's name, at most sixteen bytes of it kept. */
  readonly name?: string;
  /** The ridge. Chosen from the embeddings by `gapRidge` when absent. */
  readonly ridge?: number | null;
  /** The encoder the embeddings came from, sixteen hex digits. None named when absent. */
  readonly encoderId?: string | null;
}

/** A trained head, and what every number in it held before it was rounded. */
export interface Trained {
  /** The head blob, ready to upload. */
  readonly head: Uint8Array;
  /** The ridge the fit used. */
  readonly ridge: number;
  /** Each output's scale, as a float64, before the head stores it as a float32. */
  readonly scale: number[];
  /** Each output's weights divided by its scale, before rounding. */
  readonly weightsOverScale: number[][];
  /** Each output's intercept divided by its scale, before rounding. */
  readonly biasOverScale: number[];
}

/** A ridge given to `train_head`, as the library takes it: a finite number, refused in its words otherwise. */
function givenRidge(ridge: unknown): number {
  let x: number;
  if (typeof ridge === "number") x = ridge;
  else if (typeof ridge === "boolean") x = ridge ? 1 : 0;
  else if (typeof ridge === "bigint") x = Number(ridge);
  else throw new TypeError(`ridge is ${given(ridge)}, and a ridge is a number`);
  if (!Number.isFinite(x)) throw valueError(`ridge is ${given(ridge)}, and a ridge is a finite number`);
  return x;
}

/** `train_head`'s steps, in its order, refusing what it refuses where it refuses it. */
export function train(embeddings: unknown, targets: unknown, options: TrainOptions = {}): Trained {
  let e = rowsOf(embeddings, "embeddings are one row per window", "embeddings");
  const n = e.length;
  const d = e[0]!.length;
  if (!integerRows(embeddings as ArrayLike<unknown>)) e = e.map((row) => Float64Array.from(quantizeEmbedding(row)));
  const { rows: y, outputs: k } = targetRows(targets);
  if (n !== y.length) throw valueError(`${n} embeddings and ${y.length} targets`);
  if (n <= d) {
    throw valueError(
      `fitting ${d} weights needs more than ${n} windows, ` +
        "or the head will describe the noise in this recording rather than the signal",
    );
  }
  if (d === 0) throw valueError("every row of embeddings holds 0 values, and a head is fitted to rows of at least one value");
  const chosen = options.ridge === null || options.ridge === undefined;
  let ridge = chosen ? Number.NaN : givenRidge(options.ridge);

  // Embeddings with two or more directions along which they do not vary
  // give a ridge of 0, or one that is not a number where rounding puts the
  // second smallest eigenvalue below 0.
  const dead = DEAD + GIVE_A_RIDGE;
  if (chosen) {
    if (d < 2) {
      throw valueError(
        "every row of embeddings holds 1 value, and a ridge is chosen from rows of at least two values; " +
          "give a ridge to fit a head to rows of one value",
      );
    }
    const gram = centeredGram(e, d);
    ridge = ridgeOf(gram, d) ?? Number.NaN;
    if (Number.isNaN(ridge)) {
      if (!gram.every(Number.isFinite)) {
        throw valueError(
          "the values of embeddings are too large: choosing a ridge from them overflows the 64-bit floats it is computed in",
        );
      }
      throw valueError(dead);
    }
  }

  // The normal equations with an intercept, the intercept never penalized.
  const m = d + 1;
  const A = new Float64Array(m * m);
  const x = new Float64Array(m);
  for (const row of e) {
    x.set(row);
    x[d] = 1;
    for (let i = 0; i < m; i++) {
      const xi = x[i]!;
      for (let j = i; j < m; j++) A[i * m + j] = A[i * m + j]! + xi * x[j]!;
    }
  }
  for (let i = 0; i < m; i++) for (let j = 0; j < i; j++) A[i * m + j] = A[j * m + i]!;
  for (let i = 0; i < d; i++) A[i * m + i] = A[i * m + i]! + ridge;

  const B = new Float64Array(m * k);
  for (let r = 0; r < n; r++) {
    x.set(e[r]!);
    x[d] = 1;
    for (let i = 0; i < m; i++) {
      const xi = x[i]!;
      for (let o = 0; o < k; o++) B[i * k + o] = B[i * k + o]! + xi * y[r]![o]!;
    }
  }
  let W: Float64Array;
  try {
    W = solve(A, B, m, k);
  } catch (err) {
    if (!(err instanceof Invalid)) throw err;
    if (chosen && ridge === 0) throw valueError(dead);
    throw valueError(`with a ridge of ${shownGeneral(ridge)}, these embeddings ${UNDETERMINED}; give a larger ridge`);
  }
  for (let o = 0; o < k; o++) {
    for (let j = 0; j <= d; j++) {
      if (!Number.isFinite(W[j * k + o]!)) {
        throw valueError(`fitting a head to output ${o}'s targets overflows the 64-bit floats the fit is computed in`);
      }
    }
  }

  const rows: number[][] = [];
  const bias: number[] = [];
  const scale: number[] = [];
  const weightsOverScale: number[][] = [];
  const biasOverScale: number[] = [];
  for (let o = 0; o < k; o++) {
    let peak = 0;
    for (let j = 0; j < d; j++) peak = Math.max(peak, Math.abs(W[j * k + o]!));
    const s = peak > 0 ? peak / 127 : 1;
    const over: number[] = [];
    for (let j = 0; j < d; j++) over.push(W[j * k + o]! / s);
    const b = W[d * k + o]! / s;
    rows.push(over.map((v) => Math.max(-127, Math.min(127, rint(v)))));
    // An infinite bias is kept as it is, for the check below to name it.
    if (Number.isNaN(b)) throw valueError("cannot convert float NaN to integer");
    bias.push(Number.isFinite(b) ? rint(b) : b);
    scale.push(s);
    weightsOverScale.push(over);
    biasOverScale.push(b);
  }
  // A head holds each bias in four bytes and each scale in a four-byte
  // float: targets that need more are refused in words that name them.
  bias.forEach((b, o) => {
    if (!Number.isFinite(b) || b < -(2 ** 31) || b >= 2 ** 31) {
      throw valueError(
        `a head fitted to output ${o}'s targets needs a bias of ${shownGeneral(b)}, and a head's bias is a whole number of four bytes, -2147483648 to 2147483647`,
      );
    }
  });
  scale.forEach((v, o) => {
    if (Number.isFinite(v) && !Number.isFinite(Math.fround(v))) {
      throw valueError(`a head fitted to output ${o}'s targets needs a scale of ${shownGeneral(v)}, which a head's four-byte float cannot hold`);
    }
  });

  // numpy turns a weight that is not a number into the smallest int64, which
  // the library's build_head then refuses as out of range. The package's
  // buildHead does the rest of what build_head does, in its order.
  for (const row of rows) {
    for (const w of row) {
      if (!Number.isInteger(w)) throw new Invalid("head weights are signed bytes, so they are trained or scaled to fit");
    }
  }
  const head = buildHead(rows, bias, scale, {
    ...(options.name === undefined ? {} : { name: options.name }),
    ...(options.encoderId === undefined || options.encoderId === null ? {} : { encoderId: options.encoderId }),
  });
  return { head, ridge, scale, weightsOverScale, biasOverScale };
}
