/**
 * Internal.
 */

import * as J from "./capture/pyjson.ts";
import { Invalid } from "./protocol.ts";

/** The four separators of files, groups, records and units, which `str.isspace` counts as white space beside Unicode's own. */
const SEPARATORS = new Set([0x1c, 0x1d, 0x1e, 0x1f]);

/** Whether `c` is white space as `str.isspace` says, the characters `str.strip` and `str.split` take away. */
export function isSpace(c: number): boolean {
  return J.isWhitespace(c) || SEPARATORS.has(c);
}

/** `s.strip()`. */
export function strip(s: string): string {
  let a = 0;
  let b = s.length;
  while (a < b && isSpace(s.charCodeAt(a))) a++;
  while (b > a && isSpace(s.charCodeAt(b - 1))) b--;
  return s.slice(a, b);
}

/** A Python type's name for `x`, as a refusal names it. */
export function pyType(x: unknown): string {
  if (x === null || x === undefined) return "NoneType";
  if (Array.isArray(x)) return "list";
  if (typeof x === "string") return "str";
  if (typeof x === "number") return Number.isInteger(x) ? "int" : "float";
  if (typeof x === "boolean") return "bool";
  if (typeof x === "object") {
    const proto = Object.getPrototypeOf(x);
    return proto === Object.prototype || proto === null ? "dict" : (x as object).constructor.name;
  }
  return typeof x;
}

/**
 * A type's name for `x`, as a refusal thrown to a program names it: Python's
 * name for each type, as `pyType` gives it, but null and undefined in
 * JavaScript's own words.
 */
export function jsType(x: unknown): string {
  if (x === null) return "null";
  if (x === undefined) return "undefined";
  return pyType(x);
}

/**
 * `int(x)` for a value a program hands over, as Python takes one: a number
 * cut toward zero, a bigint as it is, true and false as one and zero, and
 * text read as Python reads a whole number written in ASCII digits, refused
 * in Python's words where it is not one: an `Invalid` naming the ValueError
 * or OverflowError, or a TypeError. Text in the digits of another script,
 * such as ３, which Python's `int` takes, is refused here.
 */
export function pyIntOf(x: unknown): bigint {
  if (typeof x === "bigint") return x;
  if (typeof x === "boolean") return x ? 1n : 0n;
  if (typeof x === "number") {
    if (Number.isNaN(x)) throw new Invalid("ValueError: cannot convert float NaN to integer");
    if (!Number.isFinite(x)) throw new Invalid("OverflowError: cannot convert float infinity to integer");
    return BigInt(Math.trunc(x));
  }
  if (typeof x === "string") {
    const v = pyInt(x);
    if (v === null) throw new Invalid(`ValueError: invalid literal for int() with base 10: ${J.reprStr(x)}`);
    return v;
  }
  throw new TypeError(`int() argument must be a string, a bytes-like object or a real number, not '${jsType(x)}'`);
}

/** Digits with one underscore at most between two, as Python reads a number written with them. */
function digitpart(s: string): boolean {
  return /^[0-9]+(_[0-9]+)*$/.test(s);
}

/** `int(text)` for text written in ASCII digits, or null where Python refuses it. Text in the digits of another script, such as ３, which Python takes, gives null here. */
export function pyInt(text: string): bigint | null {
  const s = strip(text);
  const m = /^([+-]?)(.*)$/s.exec(s)!;
  if (!digitpart(m[2]!)) return null;
  const v = BigInt(m[2]!.replace(/_/g, ""));
  return m[1] === "-" ? -v : v;
}

/** `float(text)` for text written in ASCII digits, or null where Python refuses it. Text in the digits of another script, such as ３, which Python takes, gives null here. */
export function pyFloat(text: string): number | null {
  const s = strip(text).toLowerCase();
  const m = /^([+-]?)(.*)$/s.exec(s)!;
  const sign = m[1] === "-" ? -1 : 1;
  const body = m[2]!;
  if (body === "inf" || body === "infinity") return sign * Infinity;
  if (body === "nan") return NaN;
  const parts = /^([0-9_]*)(?:\.([0-9_]*))?(?:e([+-]?)([0-9_]+))?$/.exec(body);
  if (parts === null) return null;
  const [, whole = "", frac, esign = "", exp] = parts;
  if (whole === "" && (frac === undefined || frac === "")) return null;
  if (whole !== "" && !digitpart(whole)) return null;
  if (frac !== undefined && frac !== "" && !digitpart(frac)) return null;
  if (exp !== undefined && !digitpart(exp)) return null;
  const literal = `${whole.replace(/_/g, "") || "0"}.${(frac ?? "").replace(/_/g, "") || "0"}e${esign}${(exp ?? "0").replace(/_/g, "")}`;
  return sign * Number(literal);
}
