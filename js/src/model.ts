/**
 * Training a head, as the Python library's `model` module trains one: a
 * ridge fit whose ridge comes from the embeddings alone, quantized the way
 * the device holds it, and written as the head blob the device runs.
 *
 * ```ts
 * import { trainHead } from "@intomind/sdk/model";
 *
 * const blob = trainHead(windows.map((w) => w.embedding ?? []), targets, { name: "focus", encoderId: windows[0]?.encoderId ?? null });
 * ```
 *
 * `blockedFolds` says how well such a head tells two or more states apart
 * on windows it never saw, as the library's `model.blocked_folds` does.
 *
 * The head comes out as the library's does, byte for byte, held to the
 * library's own heads by the fixture in `contract/training`. The arithmetic
 * is this package's own and numpy's is LAPACK's, so their floating point
 * can differ in the last bits. Where it does, a weight, bias or scale that
 * sat on a rounding boundary can come out one step apart, and the fixture's
 * tests show that every such difference is one of those.
 */

import { Invalid, decodeHead, evaluateHead } from "./protocol.ts";
import { DEAD, GIVE_A_RIDGE, checkRows, gapRidge as ridge, picked, targets, train, type TrainOptions } from "./training.ts";

/**
 * The ridge `trainHead` uses unless told otherwise, from the embeddings
 * alone, as the library's `model.gap_ridge` chooses it: the geometric mean of
 * the two smallest eigenvalues of the centered embeddings' Gram matrix, the
 * smallest floored at one. No label enters the choice.
 *
 * `quantized` that is not one row of numbers a window, every row as long as
 * the first and of at least two values, is refused in words naming what is
 * wrong in it, and so are values the eigenvalues cannot be found for.
 */
export function gapRidge(quantized: ReadonlyArray<ArrayLike<number>>): number {
  return ridge(quantized);
}

/**
 * Fit a head to embeddings, and return the blob to upload, as the library's
 * `model.train_head` does.
 *
 * `embeddings` is one row per window, either the floats the encoder
 * produced, quantized first, or the integers a head receives, as the
 * package's embedding windows carry them. Which they are is decided over the
 * whole input as numpy decides it in Python: integer typed arrays and plain
 * rows of whole numbers are integers, and a row in a float typed array, or
 * holding any value that is not a whole number, makes every row float.
 * `targets` is one number per window for a head of one output, or one row
 * per window with one column per output.
 *
 * The ridge comes from the embeddings unless given (`gapRidge`), and
 * `encoderId` names the encoder the embeddings came from, so a host can warn
 * when the head is uploaded to a device running another. A head is refused
 * where the library refuses it, in its words. Embeddings that are not rows
 * of numbers, every row as long as the first, targets that are not finite
 * numbers in one of the two shapes above, and a ridge that is not a finite
 * number are refused in words naming what is wrong in them. So are
 * embeddings of one value a row when no ridge is given, embeddings that with
 * the ridge given or chosen do not determine one head, and a fit that
 * overflows the 64-bit floats it is computed in. Targets that need a bias
 * past the four bytes a head holds it in, or a scale past what its four-byte
 * float holds, are refused in words naming the output.
 */
export function trainHead(
  embeddings: ReadonlyArray<ArrayLike<number>>,
  targets: ArrayLike<number> | ReadonlyArray<ArrayLike<number>>,
  options: TrainOptions = {},
): Uint8Array {
  return train(embeddings, targets, options).head;
}

/** A held-out score: the share of windows right, the counts behind it, and what guessing the commonest state scores. */
export interface HeldOut {
  /** The share of held-out windows the heads got right. */
  readonly accuracy: number;
  /** How many they got right. */
  readonly correct: number;
  /** How many were held out and scored. */
  readonly total: number;
  /** What always guessing the commonest state would score. */
  readonly chance: number;
}

/** Whether `v` has a length, as Python's `len` takes one: a row or text. */
function hasLength(v: unknown): boolean {
  return Array.isArray(v) || (ArrayBuffer.isView(v) && !(v instanceof DataView)) || typeof v === "string";
}

/** A label as the number of a state, as the library's `_state_number` reads it, or null when it is not a whole number from 0. A boolean counts as Python counts one. */
function stateNumber(y: unknown): number | null {
  const v = typeof y === "boolean" ? Number(y) : y;
  return typeof v === "number" && Number.isInteger(v) && v >= 0 ? v : null;
}

/**
 * How well a head trained on these windows tells their states apart on
 * windows it never saw, as the library's `model.blocked_folds` scores it,
 * or null when there are too few to hold any out or every block is
 * skipped.
 *
 * `rows` are the windows' embeddings, as `trainHead` takes them: one row of
 * numbers per window, every row as long as the first. When a head is to be
 * trained on them, `rows` that are not, or whose rows hold fewer than two
 * values, are refused first, in words naming what is wrong in them.
 *
 * `labels` give each window's state, one per window, as a number from 0.
 * There are as many states as there are different labels, and at least
 * two, so the labels are 0 and 1 for two states and 0, 1 and 2 for three.
 *
 * The windows are cut into contiguous blocks in the order they were
 * collected, so neighboring windows, which share most of their signal,
 * never sit on both sides of a split and flatter the score. Of n windows,
 * each block holds ceil(n / k), the last what is left, so there are k
 * blocks, or fewer where the windows run out first. k is `folds`, or more,
 * up to twenty, where a block's training windows would not otherwise
 * outnumber an embedding's values. Each block is scored by a head
 * `trainHead` fits to the other blocks, its outputs computed by
 * `evaluateHead` as the device computes them. For two states the head has
 * one output, and the head's output above one half means the state labeled
 * 1. For more, it has one output for each state, fitted to one-hot targets,
 * and a window counts as right when the output for its own state is the
 * largest, as the device picks. A block is skipped when its training windows lack one of the
 * states. A refusal of `trainHead` is thrown as it is, but for one: where
 * the ridge chosen from a block's training windows determines no head, the
 * refusal leaves out `trainHead`'s advice to give a ridge, since this takes
 * none.
 *
 * The answer is the share of held-out windows the heads got right
 * (`accuracy`), the counts behind it (`correct`, `total`), and what always
 * guessing the commonest state would score over all the windows, those of
 * skipped blocks included (`chance`).
 */
export function blockedFolds(
  rows: ReadonlyArray<ArrayLike<number>>,
  labels: ArrayLike<number>,
  options: { readonly encoderId?: string | null; readonly folds?: number } = {},
): HeldOut | null {
  const r = Array.from(rows);
  // A boolean label is the number Python counts it as, wherever it is read.
  const y = Array.from(labels, (v) => (typeof v === "boolean" ? Number(v) : v));
  if (r.length === 0) throw new Invalid("ValueError: held-out folds need at least one window");
  if (y.length !== r.length) throw new Invalid(`ValueError: ${r.length} windows and ${y.length} labels`);
  const numbers = y.map(stateNumber);
  const states = Math.max(2, new Set(numbers.filter((i) => i !== null)).size);
  if (numbers.some((i) => i === null || i >= states)) {
    throw new Invalid(
      states === 2
        ? "ValueError: the labels are 0 and 1, one for each window"
        : `ValueError: the labels are 0 to ${states - 1} for ${states} states, one for each window`,
    );
  }
  const folds = options.folds ?? 5;
  if (folds < 1) throw new Invalid(`ValueError: folds is a count of at least one, not ${folds}`);
  const n = r.length;
  // A first window with no length is refused here, in the words that say
  // what is wrong in the rows.
  if (!hasLength(r[0])) checkRows(r);
  const d = r[0]!.length;
  let k = folds;
  while (n - Math.ceil(n / k) <= d && k < 20) k++;
  if (n - Math.ceil(n / k) <= d) return null;
  const size = Math.ceil(n / k);
  // Two states are scored on the labels as given.
  const given = states === 2 ? y : (numbers as number[]);
  let correct = 0;
  let total = 0;
  let checked = false;
  for (let f = 0; f < k; f++) {
    const lo = f * size;
    const hi = Math.min(n, (f + 1) * size);
    if (lo >= hi) continue;
    const trainRows = [...r.slice(0, lo), ...r.slice(hi)];
    const trainY = [...given.slice(0, lo), ...given.slice(hi)];
    let lacking = false;
    for (let st = 0; st < states && !lacking; st++) lacking = !trainY.includes(st);
    if (lacking) continue;
    // Every window is trained on or scored from here, so every row is
    // checked before the first head is trained.
    if (!checked) {
      checkRows(r);
      checked = true;
    }
    let blob: Uint8Array;
    try {
      blob = trainHead(trainRows, targets(trainY, states), { name: "fold", encoderId: options.encoderId ?? null });
    } catch (e) {
      // This takes no ridge, so its caller is not told to give one.
      if (e instanceof Invalid && e.message === `ValueError: ${DEAD}${GIVE_A_RIDGE}`) throw new Invalid(`ValueError: ${DEAD}`);
      throw e;
    }
    const head = decodeHead(blob);
    for (let i = lo; i < hi; i++) {
      const out = evaluateHead(head, r[i]!);
      if (states === 2) {
        if ((out[0]! > 0.5 ? 1 : 0) === y[i]) correct++;
      } else if (picked(out) === numbers[i]) correct++;
    }
    total += hi - lo;
  }
  if (total === 0) return null;
  if (states > 2) {
    let most = 0;
    for (let st = 0; st < states; st++) most = Math.max(most, numbers.filter((i) => i === st).length);
    return { accuracy: correct / total, correct, total, chance: most / n };
  }
  let ones = 0;
  for (const v of y) ones += v;
  return { accuracy: correct / total, correct, total, chance: Math.max(ones, n - ones) / n };
}
