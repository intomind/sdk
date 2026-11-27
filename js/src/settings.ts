/**
 * A setting of an analysis as the Python library reads one and says it in a
 * refusal: a number as Python's arithmetic takes one, a setting as a
 * sentence shows it, and a select held to its choices. Written apart from
 * the analyses so the rule for a select is held to the library's own, by
 * `contract/analysis`, for choices of any kind, where every select of the
 * package's own analyses offers 0 and 1. Internal: the analyses are the
 * surface.
 */

import { general } from "./capture/pyfmt.ts";
import * as J from "./capture/pyjson.ts";
import type { Value } from "./capture/pyjson.ts";

/** A value as a number, as Python's arithmetic uses it. Anything else is refused. */
export function num(v: Value): number {
  const f = J.asF64(v);
  if (f === null || v.kind === "str") throw new TypeError(`${J.jsRepr(v)} is not a number`);
  return f;
}

/** A number as Python's `{x:g}` writes it, NaN and the infinities as JavaScript writes them. */
export const g = (x: number): string => (Number.isFinite(x) ? general(x, 6) : String(x));

/** Whether a value is a number as Python's `numbers.Real` takes one: an integer, a float or a boolean. */
export const isReal = (v: Value): boolean => v.kind === "int" || v.kind === "float" || v.kind === "bool";

/**
 * A setting as a sentence shows it: "60 Hz" or "0.5 s" with its `unit`, NaN
 * or Infinity, or a value that is not a number as JavaScript writes it.
 */
export function amount(v: Value, unit = ""): string {
  if (!isReal(v)) return J.jsRepr(v);
  const x = num(v);
  return Number.isFinite(x) ? `${g(x)} ${unit}`.trimEnd() : g(x);
}

/**
 * Why `v` is not a value the select `name` takes, in the library's words,
 * naming the recording `label` and the setting with what it is, `means`,
 * and listing its `choices`, or null where it is one of them. A value is
 * one of the choices by Python's equality: a number where a choice is that
 * number, true where one is 1, and text only where a choice is the same
 * text.
 */
export function selectRefusal(label: string, name: string, means: string, choices: readonly unknown[], v: Value): string | null {
  const offered = choices.map((c) => J.fromPlain(c));
  if ((isReal(v) || v.kind === "str") && offered.some((c) => J.pyEq(c, v))) return null;
  return `${label} cannot be analyzed with ${name} set to ${amount(v)}: ${name}, ${means}, is ${offered.map((c) => amount(c)).join(" or ")}.`;
}
