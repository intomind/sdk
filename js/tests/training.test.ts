/**
 * Head training held to the Python library's: every head in
 * `contract/training` trained again here from the same embeddings and
 * targets, every refusal refused in the library's words, each value it
 * shows as this package shows it (`asShownHere`), and the ridge chosen as
 * `model.gap_ridge` chooses it. Embeddings are integers or floats
 * as numpy decides over the whole input, so the integers are trained as
 * they arrive from JSON and as typed arrays, and the floats as float typed
 * arrays. The demos' case is trained again from this package's own embedding
 * windows. Also `buildHead` held to the library's `build_head` on values a
 * head cannot hold as given, and `fromHex` to `bytes.fromhex`. And the
 * held-out score, `blockedFolds`, held to `model.blocked_folds`: the same
 * counts from the same windows, the encoder's floats among them, and the
 * same refusals.
 *
 * The arithmetic here is the package's own and numpy's is LAPACK's, so the
 * floating point can differ in its last bits. Each number is held within a
 * tolerance measured on this fixture, written below with the largest
 * difference found. A head is held byte for byte, and a byte that differs is
 * allowed only where the library's own value before rounding sat within that
 * tolerance of a rounding boundary, and the two came out one step apart.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { sha256 } from "../src/digest.ts";
import * as model from "../src/model.ts";
import { gapRidge, trainHead } from "../src/model.ts";
import {
  EmbeddingAssembler,
  HEAD_HASH_LEN,
  HEAD_HEADER_LEN_2,
  Invalid,
  buildHead,
  decodeEmbedding,
  encodeEmbedding,
  fromHex,
  toHex,
  type EmbeddingWindow,
} from "../src/protocol.ts";
import { rint, solve, symmetricEigenvalues, train } from "../src/training.ts";
import { inJavaScript } from "./words.ts";

// Measured on this fixture, numpy 2.2.4 on MKL, the largest difference first.
/** The ridge, relative. Largest found: 9.5e-8, the demos' case, whose smallest eigenvalue is rounding noise numpy and this both carry. */
const RIDGE = 1e-7;
/** A weight divided by its scale, before rounding, in steps of the signed byte. Largest found: 4.4e-7. */
const WEIGHT = 5e-7;
/** A scale, relative. Largest found: 1.9e-9. */
const SCALE = 2e-9;
/** An intercept divided by its scale, before rounding, relative to the larger of one and itself. Largest found: 1.9e-9. */
const BIAS = 2e-9;

const fixture = JSON.parse(readFileSync(new URL("../../contract/training/training.json", import.meta.url), "utf8"));

/** NaN and the infinities, which JSON cannot carry, written as words in the fixture. */
function special(v: unknown): unknown {
  if (Array.isArray(v)) return v.map(special);
  if (v === "nan") return Number.NaN;
  if (v === "inf") return Number.POSITIVE_INFINITY;
  if (v === "-inf") return Number.NEGATIVE_INFINITY;
  return v;
}

/** A case's input: the name of one of the fixture's shared inputs, or the input itself. */
function input(v: unknown): unknown {
  return typeof v === "string" ? fixture.inputs[v] : special(v);
}

/** Every row of an input in one kind of typed array. */
function typed(v: unknown, T: Int32ArrayConstructor | Float64ArrayConstructor): unknown[] {
  return (input(v) as number[][]).map((row) => T.from(row));
}

/** Whether every value of every row is a safe integer, which makes plain rows integers. */
function wholeNumbers(v: unknown): boolean {
  return (input(v) as number[][]).every((row) => row.every((x) => Number.isSafeInteger(x)));
}

function options(c: { options: { name: string | null; ridge: number | null; encoder_id: string | null } }) {
  return {
    ...(c.options.name === null ? {} : { name: c.options.name }),
    ridge: c.options.ridge,
    encoderId: c.options.encoder_id,
  };
}

const PYTHON = /^(ValueError|IndexError|LinAlgError|OverflowError|AxisError|error): /;

/** A refusal as the fixture writes the library's, `Exception: message`. */
function said(e: unknown): string {
  if (e instanceof Invalid) return PYTHON.test(e.message) ? e.message : `Invalid: ${e.message}`;
  if (e instanceof TypeError) return `TypeError: ${e.message}`;
  throw e;
}

/**
 * The library's refusal of a value as this package shows the same value: a
 * whole number a Python program passes as a float, such as 2147483648.0, is
 * one JavaScript number, which the package shows as a whole number where it
 * is a safe integer, and Python's None, nan and inf are read through the
 * name table as null, NaN and Infinity.
 */
function asShownHere(refusal: string): string {
  const whole = refusal.replace(/(?<![\w.])-?\d+\.0(?![\w.])/g, (m) => {
    const v = Number(m);
    return Number.isSafeInteger(v) && !Object.is(v, -0) ? String(v) : m;
  });
  return inJavaScript(whole);
}

function refusal(f: () => unknown): string {
  try {
    f();
  } catch (e) {
    return said(e);
  }
  return "nothing refused";
}

/** How far a value sits from the nearest half step. */
function fromHalf(v: number): number {
  return Math.abs(Math.abs(v - Math.floor(v)) - 0.5);
}

/** The float32 next to `f` on the side of `toward`, and the midpoint between them. */
function float32Midpoint(f: number, toward: number): number {
  const view = new DataView(new ArrayBuffer(4));
  view.setFloat32(0, f, true);
  const bits = view.getUint32(0, true);
  const up = (toward > f) === (f >= 0);
  view.setUint32(0, up ? bits + 1 : bits - 1, true);
  return (f + view.getFloat32(0, true)) / 2;
}

type Case = Record<string, any>;

/**
 * Where a head made here differs from the library's head for case `c`: a
 * byte of a weight or a bias rounded one step the other way is allowed only
 * where the library's own value before rounding sat within the tolerance of
 * a rounding boundary, and a scale only where it sat within the tolerance of
 * the midpoint between two float32 values. Anything else is a problem.
 */
function differences(where: string, ours: Uint8Array, c: Case, scales: readonly number[]): string[] {
  const problems: string[] = [];
  const theirs = Uint8Array.from(Buffer.from(c.head, "hex"));
  const outputs = c.scale.length as number;
  const width = c.weights_over_scale[0].length as number;
  const weightsAt = HEAD_HEADER_LEN_2;
  const biasAt = weightsAt + outputs * width;
  const scaleAt = biasAt + 4 * outputs;
  const hashAt = scaleAt + 4 * outputs;
  assert.equal(ours.length, theirs.length, where);
  assert.equal(hashAt + HEAD_HASH_LEN, theirs.length, where);
  const view = (b: Uint8Array) => new DataView(b.buffer, b.byteOffset, b.byteLength);
  let differs = false;
  for (let i = 0; i < hashAt; i++) {
    if (ours[i] === theirs[i]) continue;
    differs = true;
    if (i < weightsAt) {
      problems.push(`${where}: header byte ${i}`);
    } else if (i < biasAt) {
      const o = Math.floor((i - weightsAt) / width);
      const j = (i - weightsAt) % width;
      const w = c.weights_over_scale[o][j] as number;
      const step = Math.abs(((ours[i]! << 24) >> 24) - ((theirs[i]! << 24) >> 24));
      if (step !== 1 || fromHalf(w) > WEIGHT) problems.push(`${where}: weight ${o}.${j} rounds apart off a boundary (${w})`);
    } else if (i < scaleAt) {
      const o = Math.floor((i - biasAt) / 4);
      const b = c.bias_over_scale[o] as number;
      const step = Math.abs(view(ours).getInt32(biasAt + 4 * o, true) - view(theirs).getInt32(biasAt + 4 * o, true));
      if (step !== 1 || fromHalf(b) > BIAS * Math.max(1, Math.abs(b))) problems.push(`${where}: bias ${o} rounds apart off a boundary (${b})`);
    } else {
      const o = Math.floor((i - scaleAt) / 4);
      const s = c.scale[o] as number;
      const mid = float32Midpoint(view(theirs).getFloat32(scaleAt + 4 * o, true), scales[o]!);
      if (Math.abs(s - mid) > SCALE * s) problems.push(`${where}: scale ${o} rounds apart off a boundary (${s})`);
    }
  }
  if (!differs) assert.deepEqual(ours, theirs, `${where}: the same head with a different hash`);
  return problems;
}

/** The ridge and every number before rounding, against the library's, within the measured tolerances. */
function numbers(where: string, t: ReturnType<typeof train>, c: Case): string[] {
  const problems: string[] = [];
  if (c.options.ridge !== null) {
    if (t.ridge !== c.ridge) problems.push(`${where}: ridge ${t.ridge}, the library's ${c.ridge}`);
  } else if (Math.abs(t.ridge - c.ridge) > RIDGE * Math.abs(c.ridge)) {
    problems.push(`${where}: ridge ${t.ridge}, the library's ${c.ridge}`);
  }
  for (let o = 0; o < c.scale.length; o++) {
    if (Math.abs(t.scale[o]! - c.scale[o]) > SCALE * c.scale[o]) problems.push(`${where}: scale ${o}`);
    const b = c.bias_over_scale[o] as number;
    if (Math.abs(t.biasOverScale[o]! - b) > BIAS * Math.max(1, Math.abs(b))) problems.push(`${where}: bias ${o}`);
    c.weights_over_scale[o].forEach((w: number, j: number) => {
      if (Math.abs(t.weightsOverScale[o]![j]! - w) > WEIGHT) problems.push(`${where}: weight ${o}.${j}`);
    });
  }
  return problems;
}

test("every head is the library's, byte for byte or one step apart on a rounding boundary", () => {
  const heads = fixture.heads as Case[];
  assert.ok(heads.length >= 19, String(heads.length));
  const problems: string[] = [];
  for (const c of heads) {
    const where = c.case as string;
    // Integers as they arrive from JSON, plain rows of whole numbers. Floats
    // in float typed arrays, which say they are floats whatever their values.
    const given = c.integer ? input(c.embeddings) : typed(c.embeddings, Float64Array);
    const t = train(given, input(c.targets), options(c as never));
    problems.push(...numbers(where, t, c), ...differences(where, t.head, c, t.scale));
    const head = (rows: unknown) => trainHead(rows as never, input(c.targets) as never, options(c as never));
    if (c.integer) {
      assert.deepEqual(head(typed(c.embeddings, Int32Array)), t.head, `${where}, in Int32Array rows`);
    } else if (!wholeNumbers(c.embeddings)) {
      assert.deepEqual(head(input(c.embeddings)), t.head, `${where}, in plain rows`);
    }
  }
  assert.deepEqual(problems, []);
});

test("integers or floats is decided over the whole input, as numpy decides it", () => {
  const find = (name: string): Case => (fixture.heads as Case[]).find((c) => c.case === name)!;
  const asIntegers = find("small whole numbers, as integers");
  const asFloats = find("small whole numbers, as floats");
  const mixed = find("integers with one row of floats");
  const targets = input(asIntegers.targets) as never;
  const theirs = (c: Case) => Uint8Array.from(Buffer.from(c.head, "hex"));
  assert.notDeepEqual(theirs(asIntegers), theirs(asFloats), "the library trains whole numbers given as floats differently");
  // The same whole numbers: plain rows are integers, a Float64Array of whole numbers is floats.
  assert.deepEqual(trainHead(input(asFloats.embeddings) as never, targets), theirs(asIntegers));
  assert.deepEqual(trainHead(typed(asFloats.embeddings, Float64Array) as never, targets), theirs(asFloats));
  // One row of floats makes every row float, however the rest are held.
  const rows = typed(mixed.embeddings, Int32Array);
  rows[rows.length - 1] = Float64Array.from((input(mixed.embeddings) as number[][]).at(-1)!);
  assert.deepEqual(trainHead(rows as never, input(mixed.targets) as never), theirs(mixed));
  assert.deepEqual(trainHead(input(mixed.embeddings) as never, input(mixed.targets) as never), theirs(mixed));
});

test("the package's own embedding windows train to the library's head from its own windows", () => {
  const c = fixture.windows as Case;
  const rows = input(c.embeddings) as number[][];
  const sent = c.sent as { encoder_id: string; split: number; index_step: number; time_step: number; window_samples: number };
  const notes: Uint8Array[] = [];
  rows.forEach((row, w) => {
    for (let first = 0; first < row.length; first += sent.split) {
      notes.push(
        encodeEmbedding({
          index: sent.index_step * w,
          deviceTime: BigInt(sent.time_step * w),
          windowSamples: sent.window_samples,
          encoderId: sent.encoder_id,
          embedDim: row.length,
          first,
          token: null,
          moreParts: first + sent.split < row.length,
          values: row.slice(first, first + sent.split),
        }),
      );
    }
  });
  assert.equal(notes.length, c.notifications);
  const all = new Uint8Array(notes.reduce((n, b) => n + b.length, 0));
  notes.reduce((at, b) => (all.set(b, at), at + b.length), 0);
  assert.equal(toHex(sha256(all)), c.notifications_sha256, "the notifications are the bytes the library decoded");

  const assembler = new EmbeddingAssembler(c.channels, c.tokens_per_channel, "window");
  const windows: EmbeddingWindow[] = [];
  for (const b of notes) {
    const w = assembler.feed(decodeEmbedding(b));
    if (w !== null) windows.push(w);
  }
  assert.equal(windows.length, rows.length);
  const embeddings = windows.map((w) => w.embedding!);
  assert.ok(embeddings.every((e) => Array.isArray(e)), "the windows hand out plain arrays");
  const t = train(embeddings, input(c.targets), { name: c.options.name, encoderId: windows[0]!.encoderId });
  assert.deepEqual([...numbers(c.case, t, c), ...differences(c.case, t.head, c, t.scale)], []);
  assert.deepEqual(trainHead(embeddings, input(c.targets) as never, { name: c.options.name, encoderId: windows[0]!.encoderId }), t.head);
});

test("every refusal is the library's, where it refuses and in its words", () => {
  const refusals = fixture.refusals as Array<Record<string, any>>;
  assert.ok(refusals.length >= 27, String(refusals.length));
  for (const c of refusals) {
    const got = refusal(() => trainHead(input(c.embeddings) as never, input(c.targets) as never, options(c as never)));
    assert.equal(got, asShownHere(c.refusal), c.case);
  }
});

test("the ridge is the one the library chooses, and refused where the library refuses it", () => {
  const cases = fixture.gap_ridge as Array<Record<string, any>>;
  assert.ok(cases.length >= 13, String(cases.length));
  for (const c of cases) {
    const quantized = input(c.quantized) as never;
    if (c.refusal !== undefined) {
      assert.equal(refusal(() => gapRidge(quantized)), asShownHere(c.refusal), c.case);
    } else if (c.ridge === "nan") {
      assert.ok(Number.isNaN(gapRidge(quantized)), c.case);
    } else {
      const r = gapRidge(quantized);
      assert.ok(Math.abs(r - c.ridge) <= RIDGE * Math.abs(c.ridge), `${c.case}: ${r}, the library's ${c.ridge}`);
    }
  }
});

test("rounding is half to even, as numpy's rint and Python's round", () => {
  const cases: Array<[number, number]> = [[0.5, 0], [1.5, 2], [2.5, 2], [-0.5, 0], [-1.5, -2], [-2.5, -2], [2.4, 2], [2.6, 3], [-2.6, -3], [126.5, 126], [127.5, 128]];
  for (const [x, want] of cases) assert.ok(rint(x) === want, `${x} rounds to ${rint(x)}`);
  assert.ok(Number.isNaN(rint(Number.NaN)));
});

test("eigenvalues and solutions of small systems, and a pivot of zero refused", () => {
  assert.deepEqual(Array.from(symmetricEigenvalues(Float64Array.of(2, 1, 1, 2), 2)!), [1, 3]);
  const ev = symmetricEigenvalues(Float64Array.of(4, 1, 0, 1, 3, 1, 0, 1, 2), 3)!;
  const known = [3 - Math.sqrt(3), 3, 3 + Math.sqrt(3)];
  ev.forEach((v, i) => assert.ok(Math.abs(v - known[i]!) < 1e-14, `${v} and ${known[i]}`));
  assert.ok(Number.isNaN(symmetricEigenvalues(Float64Array.of(1, Number.NaN, Number.NaN, 1), 2)![0]!));
  assert.deepEqual(Array.from(solve(Float64Array.of(2, 1, 1, 3), Float64Array.of(3, 5), 2, 1)), [0.8, 1.4]);
  assert.equal(refusal(() => solve(new Float64Array(4), Float64Array.of(1, 1), 2, 1)), "LinAlgError: Singular matrix");
});

test("a head is written as the library writes it from the same values, or refused in its words", () => {
  const cases = fixture.build_head as Case[];
  assert.ok(cases.length >= 51, String(cases.length));
  for (const c of cases) {
    const o = c.options as { name: string | null; in_dim: number | null; encoder_id: string | null; version: number | null };
    const options = {
      ...(o.name === null ? {} : { name: o.name }),
      ...(o.in_dim === null ? {} : { inDim: o.in_dim }),
      ...(o.encoder_id === null ? {} : { encoderId: o.encoder_id }),
      ...(o.version === null ? {} : { version: o.version }),
    };
    const got = (() => {
      try {
        return toHex(buildHead(special(c.weights) as number[][], special(c.bias) as number[], special(c.scale) as number[], options));
      } catch (e) {
        return said(e);
      }
    })();
    assert.equal(got, c.head ?? asShownHere(c.refusal), c.case);
  }
});

test("values the fixture does not hold are refused in the library's words, with null, NaN and Infinity where it says None, nan and inf", () => {
  // Each is the library's refusal of the same value passed from Python,
  // with None, nan and inf as null, NaN and Infinity, and text in double
  // quotes.
  const rows = [[1, 2], [3, 4], [5, 7], [2, 9]];
  assert.equal(refusal(() => trainHead([[1, null], ...rows.slice(1)] as never, [1, 2, 3, 4])), "TypeError: embeddings[0][1] is null, which is not a number");
  assert.equal(refusal(() => trainHead(rows, [1, null, 3, 4] as never)), "TypeError: targets[1] is null, which is not a finite number");
  assert.equal(refusal(() => trainHead(rows, [[1, Infinity], [1, 2], [1, 2], [1, 2]])), "ValueError: targets[0][1] is Infinity, which is not a finite number");
  assert.equal(refusal(() => trainHead(rows, [1, 2, 3, 4], { ridge: Number.NaN })), "ValueError: ridge is NaN, and a ridge is a finite number");
  assert.equal(refusal(() => trainHead(rows, [1, 2, 3, 4], { ridge: "x" as never })), 'TypeError: ridge is "x", and a ridge is a number');
  assert.equal(refusal(() => trainHead(null as never, [1])), "TypeError: embeddings are one row per window, not null");
  assert.equal(refusal(() => gapRidge(null as never)), "TypeError: quantized is one row per window, not null");
  assert.equal(refusal(() => model.blockedFolds([...Array(8).fill([1, 2]), [1, null]], [0, 1, 0, 1, 0, 1, 0, 1, 0])), "TypeError: rows[8][1] is null, which is not a number");
  assert.equal(refusal(() => buildHead([[1, 2, 3]], [0], [1], { name: null as never })), "Invalid: a head's name is text, not null");
  assert.equal(refusal(() => buildHead([[1, 2, 3]], [0], [1], { encoderId: 5 as never })), "Invalid: an encoder id is text, sixteen hex digits, not 5");
  assert.equal(refusal(() => buildHead([[1, 2, 3]], [0], [null as never])), "Invalid: scale[0] is null, and a head's scale is a finite number");
  assert.equal(refusal(() => buildHead([[1, 2, 3]], [0], [1e39])), "Invalid: scale[0] is 1e+39, which a head's four-byte float cannot hold");
});

test("a whole number of more digits than Python prints, and text written as one, is refused by how many digits it has, in a head's values and a trained head's embeddings, targets and ridge, as the library refuses it", () => {
  // Python turns no whole number of more than 4,300 digits into text, so
  // the library names each by its digits, never printing it, and so the
  // package names a bigint and text of as many; a row holding such a whole
  // number is named by its type, as Python, which will not print it, names
  // it. One of 4,300 digits still prints.
  const rows = [[1, 2], [3, 4], [5, 7], [2, 9]];
  const big = 10n ** 5000n;
  assert.equal(refusal(() => trainHead([[big, 1], ...rows.slice(1)] as never, [1, 2, 3, 4])), "ValueError: embeddings[0][0] is a whole number of 5001 digits, which a 64-bit float cannot hold");
  assert.equal(refusal(() => trainHead(rows, [1, -(10n ** 4300n), 3, 4] as never)), "ValueError: targets[1] is a whole number of 4301 digits, which a 64-bit float cannot hold");
  assert.equal(refusal(() => trainHead(rows, [1, 2, 3, 4], { ridge: big as never })), "ValueError: ridge is a whole number of 5001 digits, and a ridge is a finite number");
  assert.equal(refusal(() => trainHead(rows, [1, 2, 3, 4], { ridge: "1".repeat(5000) as never })), "TypeError: ridge is text of 5000 digits, and a ridge is a number");
  assert.equal(refusal(() => trainHead(rows, [1, 2, 3, 4], { ridge: `-${"1_0".repeat(2200)}` as never })), "TypeError: ridge is text of 4400 digits, and a ridge is a number");
  assert.equal(refusal(() => trainHead(rows, [[1], [2], [big], [4]] as never)), "ValueError: targets[2][0] is a whole number of 5001 digits, which a 64-bit float cannot hold");
  assert.equal(refusal(() => trainHead([[[big], 1], ...rows.slice(1)] as never, [1, 2, 3, 4])), "ValueError: embeddings[0][0] is a list of more digits than Python prints, and each value in a row is a number");
  assert.equal(refusal(() => trainHead(rows, [[1, 2], [3, 4], [5, [big]], [7, 8]] as never)), "ValueError: targets[2][1] is a list of more digits than Python prints, and each value in a row is a number");
  assert.equal(refusal(() => buildHead([[1, 2, 3]], [0], [big as never])), "Invalid: scale[0] is a whole number of 5001 digits, which a head's four-byte float cannot hold");
  assert.equal(refusal(() => buildHead([[1, 2, 3]], [-(10n ** 4300n) as never], [1.0])), "Invalid: bias[0] is a whole number of 4301 digits, and a head's bias is a whole number of four bytes, -2147483648 to 2147483647");
  assert.equal(refusal(() => buildHead([[1, 2, 3]], [0], ["1".repeat(5000) as never])), "Invalid: scale[0] is text of 5000 digits, and a head's scale is a finite number");
  assert.ok(refusal(() => buildHead([[1, 2, 3]], [(10n ** 4300n - 1n) as never], [1.0])).startsWith(`Invalid: bias[0] is ${10n ** 4300n - 1n}, and `));
  assert.ok(refusal(() => buildHead([[1, 2, 3]], [0], ["9".repeat(4300) as never])).startsWith(`Invalid: scale[0] is "${"9".repeat(4300)}", and `));
});

test("held-out blocks refuse rows that determine no head in trainHead's words, without its advice to give a ridge, as the library's do", () => {
  // blockedFolds takes no ridge, so where the ridge chosen from a block's
  // training windows determines no head, it refuses as trainHead does,
  // without trainHead's advice to give a ridge.
  assert.equal(
    refusal(() => model.blockedFolds(Array.from({ length: 10 }, () => [0, 0]), [0, 1, 0, 1, 0, 1, 0, 1, 0, 1])),
    "ValueError: with the ridge chosen from these embeddings, they do not determine one head, because there are two or more " +
      "directions along which they do not vary",
  );
});

test("hex is read as the library reads it, and refused where it refuses it, in its words", () => {
  const cases = fixture.fromhex as Case[];
  assert.ok(cases.length >= 15, String(cases.length));
  for (const c of cases) {
    const got = (() => {
      try {
        return toHex(fromHex(c.text));
      } catch (e) {
        return said(e);
      }
    })();
    assert.equal(got, c.bytes ?? c.refusal, JSON.stringify(c.text));
  }
});

test("the model module publishes the library's three names and nothing else", () => {
  assert.deepEqual(Object.keys(model).sort(), ["blockedFolds", "gapRidge", "trainHead"]);
});

test("every held-out score is the library's: the same counts, the same fraction, the same refusals", () => {
  const cases = fixture.blocked_folds as {
    case: string;
    rows: unknown;
    labels: unknown;
    options: { encoder_id?: string; folds?: number };
    score?: { accuracy: number; correct: number; total: number; chance: number } | null;
    refusal?: string;
  }[];
  assert.ok(cases.length >= 10, String(cases.length));
  for (const c of cases) {
    const rows = input(c.rows) as number[][];
    const labels = input(c.labels) as number[];
    const opts = {
      ...(c.options.encoder_id === undefined ? {} : { encoderId: c.options.encoder_id }),
      ...(c.options.folds === undefined ? {} : { folds: c.options.folds }),
    };
    if (c.refusal !== undefined) {
      assert.equal(refusal(() => model.blockedFolds(rows, labels, opts)), asShownHere(c.refusal), c.case);
      continue;
    }
    const got = model.blockedFolds(rows, labels, opts);
    assert.deepEqual(got, c.score ?? null, c.case);
    // Rows as typed arrays are the same windows: integers in an integer
    // array, and the floats in a float array.
    const asTyped = rows.map((r) => (r.every((x) => Number.isSafeInteger(x)) ? Int32Array.from(r) : Float64Array.from(r)));
    if (c.score) assert.deepEqual(model.blockedFolds(asTyped, labels, opts), c.score, c.case);
  }
});
