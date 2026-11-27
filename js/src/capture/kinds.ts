/**
 * Which numbers the Python library holds as floats.
 *
 * Python keeps `250` apart from `250.0`, and a manifest says which it held.
 * JavaScript has one kind of number, so a whole number made by this package
 * is written as an integer unless the package knows the library would hold
 * it as a float: a rate a file states as a division, a reference computed
 * from a grid, a number a stream carried with a fraction. Those are marked
 * here, on the object that holds them, under the name each is held by.
 *
 * Internal.
 */

import * as J from "./pyjson.ts";
import type { Value } from "./pyjson.ts";

const FLOATS = new WeakMap<object, Set<string>>();

/** Say that `o` holds each of `keys` as a float, and give back `o`. */
export function markFloats<T extends object>(o: T, keys: Iterable<string>): T {
  let s = FLOATS.get(o);
  if (s === undefined) {
    s = new Set();
    FLOATS.set(o, s);
  }
  for (const k of keys) s.add(k);
  return o;
}

/** Whether `o` holds `key` as a float. */
export function isFloat(o: object | null | undefined, key: string): boolean {
  return o !== null && o !== undefined && (FLOATS.get(o)?.has(key) ?? false);
}

/** A number as Python writes it: an integer when it is a whole number held as one, a float otherwise. */
export function plain(v: number): Value {
  return Number.isSafeInteger(v) && !Object.is(v, -0) ? J.int(v) : J.float(v);
}

/** `v`, which `o` holds under `key`, as Python writes it: a float where `o` holds it as one, otherwise as `plain` writes it. */
export function held(o: object | null | undefined, key: string, v: number): Value {
  return isFloat(o, key) ? J.float(v) : plain(v);
}

/** `v`, or null, as `held` writes it. */
export function heldOrNull(o: object | null | undefined, key: string, v: number | null): Value {
  return v === null ? J.NULL : held(o, key, v);
}
