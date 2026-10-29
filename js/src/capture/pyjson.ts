/**
 * JSON read and written as the Python library reads and writes it.
 *
 * A manifest is read with `json.loads` and an export carries it again as
 * `json.dumps(manifest, sort_keys=True)`, so an export written here holds
 * the same text only if this module keeps what Python keeps: an integer
 * apart from a float, every float exactly, every integer exactly, the order
 * keys arrived in, and `NaN` and `Infinity`, which Python reads and writes
 * although JSON does not define them. `JSON.parse` keeps none of these, so
 * nothing here uses it.
 */

import { repr } from "./pyfmt.ts";

/** One JSON value as Python holds it after `json.loads`. */
export type Value =
  | { readonly kind: "null" }
  | { readonly kind: "bool"; readonly value: boolean }
  /** An integer, as its canonical decimal digits. Python's integers have no limit, and digits keep every one exactly. */
  | { readonly kind: "int"; readonly value: string }
  | { readonly kind: "float"; readonly value: number }
  | { readonly kind: "str"; readonly value: string }
  | { readonly kind: "array"; readonly items: Value[] }
  /** Keys in the order they first arrived. A key repeated keeps its first place and takes its last value, as a Python dict does. */
  | { readonly kind: "object"; readonly entries: [string, Value][] };

/** The null value. */
export const NULL: Value = { kind: "null" };

/** A string value. */
export function str(value: string): Value {
  return { kind: "str", value };
}

/** An integer value, from a number that is a whole number or from a bigint. */
export function int(value: number | bigint): Value {
  return { kind: "int", value: BigInt(value).toString() };
}

/** A float value. */
export function float(value: number): Value {
  return { kind: "float", value };
}

/** A boolean value. */
export function bool(value: boolean): Value {
  return { kind: "bool", value };
}

/** An array value. */
export function array(items: Value[]): Value {
  return { kind: "array", items };
}

/** An object value, its keys in the order given. */
export function object(entries: [string, Value][]): Value {
  return { kind: "object", entries };
}

/** The value under `key`, when `v` is an object that has it. */
export function get(v: Value, key: string): Value | undefined {
  if (v.kind !== "object") return undefined;
  for (const [k, x] of v.entries) if (k === key) return x;
  return undefined;
}

/** `meta.get(key)` in Python: the value, or null where it is absent. */
export function field(v: Value, key: string): Value {
  return get(v, key) ?? NULL;
}

/** Python's truth value. */
export function truthy(v: Value): boolean {
  switch (v.kind) {
    case "null":
      return false;
    case "bool":
      return v.value;
    case "int":
      return v.value !== "0";
    case "float":
      return v.value !== 0;
    case "str":
      return v.value !== "";
    case "array":
      return v.items.length > 0;
    case "object":
      return v.entries.length > 0;
  }
}

// The characters Rust's `char::is_whitespace` and `str::trim` take as
// whitespace, which are Unicode's White_Space property. JavaScript's own
// `trim` adds U+FEFF and leaves out U+0085.
const WHITE_SPACE = new Set([
  0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x20, 0x85, 0xa0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005,
  0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000,
]);

/** Whether one code unit is whitespace. Every whitespace character is a single code unit. */
export function isWhitespace(code: number): boolean {
  return WHITE_SPACE.has(code);
}

/** `s` without whitespace at either end. */
export function trim(s: string): string {
  let a = 0;
  let b = s.length;
  while (a < b && WHITE_SPACE.has(s.charCodeAt(a))) a++;
  while (b > a && WHITE_SPACE.has(s.charCodeAt(b - 1))) b--;
  return s.slice(a, b);
}

/** `s` without whitespace at its start. */
export function trimStart(s: string): string {
  let a = 0;
  while (a < s.length && WHITE_SPACE.has(s.charCodeAt(a))) a++;
  return s.slice(a);
}

const FLOAT_TEXT = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;

/** `float(text)` for the spellings Python accepts, or null where Python would raise. */
function pythonFloat(s: string): number | null {
  const t = s.replaceAll("_", "");
  switch (t.toLowerCase()) {
    case "nan":
    case "+nan":
    case "-nan":
      return NaN;
    case "inf":
    case "+inf":
    case "infinity":
    case "+infinity":
      return Infinity;
    case "-inf":
    case "-infinity":
      return -Infinity;
  }
  return FLOAT_TEXT.test(t) ? Number(t) : null;
}

/** `float(value)` in Python, for a number, a boolean or a numeric string. Null where Python would raise. */
export function asF64(v: Value): number | null {
  switch (v.kind) {
    case "bool":
      return v.value ? 1 : 0;
    case "int":
      return Number(v.value);
    case "float":
      return v.value;
    case "str":
      return pythonFloat(trim(v.value));
    default:
      return null;
  }
}

const I64_MIN = -(2n ** 63n);
const I64_MAX = 2n ** 63n - 1n;

/**
 * `int(value)` in Python for an integer, a whole float, a boolean, or a
 * string of digits. Null where Python would raise or the result does not fit
 * 64 bits.
 */
export function asI64(v: Value): bigint | null {
  switch (v.kind) {
    case "bool":
      return v.value ? 1n : 0n;
    case "int": {
      const b = BigInt(v.value);
      return b >= I64_MIN && b <= I64_MAX ? b : null;
    }
    case "float": {
      if (!Number.isFinite(v.value)) return null;
      const t = Math.trunc(v.value);
      return Math.abs(t) < 9.2e18 ? BigInt(t) : null;
    }
    case "str": {
      const t = trim(v.value).replaceAll("_", "");
      if (!/^[+-]?\d+$/.test(t)) return null;
      const b = BigInt(t);
      return b >= I64_MIN && b <= I64_MAX ? b : null;
    }
    default:
      return null;
  }
}

/** The text of a string value, or null for anything else. */
export function asStr(v: Value): string | null {
  return v.kind === "str" ? v.value : null;
}

/** `str(value)` in Python. */
export function pyStr(v: Value): string {
  return v.kind === "str" ? v.value : pyRepr(v);
}

/** `repr(value)` in Python, for the values JSON can hold. */
export function pyRepr(v: Value): string {
  switch (v.kind) {
    case "null":
      return "None";
    case "bool":
      return v.value ? "True" : "False";
    case "int":
      return v.value;
    case "float":
      return repr(v.value);
    case "str":
      return reprStr(v.value);
    case "array":
      return `[${v.items.map(pyRepr).join(", ")}]`;
    case "object":
      return `{${v.entries.map(([k, x]) => `${reprStr(k)}: ${pyRepr(x)}`).join(", ")}}`;
  }
}

/**
 * Python's `repr` of a string: single quotes unless the text holds a single
 * quote and no double quote. Escapes are Python's for ASCII. A character
 * outside ASCII is written as itself.
 */
export function reprStr(s: string): string {
  const quote = s.includes("'") && !s.includes('"') ? '"' : "'";
  let out = quote;
  for (const c of s) {
    const code = c.codePointAt(0)!;
    if (c === "\\") out += "\\\\";
    else if (c === "\n") out += "\\n";
    else if (c === "\r") out += "\\r";
    else if (c === "\t") out += "\\t";
    else if (c === quote) out += "\\" + c;
    else if (code < 0x20 || code === 0x7f) out += "\\x" + code.toString(16).padStart(2, "0");
    else out += c;
  }
  return out + quote;
}

/** Why a document is not JSON as Python reads it. */
export class ParseError extends Error {
  /** The reason, and the character it was found at. */
  constructor(message: string) {
    super(message);
    this.name = "ParseError";
  }
}

/** `json.loads(text)`. Throws a `ParseError` where Python would raise. */
export function loads(text: string): Value {
  const p = new Parser(text);
  p.ws();
  const v = p.value();
  p.ws();
  if (p.i !== text.length) throw p.err("extra data");
  return v;
}

class Parser {
  readonly s: string;
  i = 0;

  constructor(s: string) {
    this.s = s;
  }

  err(what: string): ParseError {
    // Counted in characters, a pair of surrogates being one.
    let n = 0;
    const end = Math.min(this.i, this.s.length);
    for (let k = 0; k < end; k++) {
      const c = this.s.charCodeAt(k);
      if (c >= 0xd800 && c < 0xdc00 && k + 1 < end) {
        const d = this.s.charCodeAt(k + 1);
        if (d >= 0xdc00 && d < 0xe000) k++;
      }
      n++;
    }
    return new ParseError(`${what} at character ${n}`);
  }

  ws(): void {
    const s = this.s;
    while (this.i < s.length) {
      const c = s.charCodeAt(this.i);
      if (c === 0x20 || c === 0x09 || c === 0x0a || c === 0x0d) this.i++;
      else break;
    }
  }

  eat(lit: string): boolean {
    if (this.s.startsWith(lit, this.i)) {
      this.i += lit.length;
      return true;
    }
    return false;
  }

  value(): Value {
    if (this.i >= this.s.length) throw this.err("expecting value");
    const c = this.s[this.i]!;
    if (c === "{") return this.object();
    if (c === "[") return this.array();
    if (c === '"') return { kind: "str", value: this.string() };
    if (c === "n" && this.eat("null")) return NULL;
    if (c === "t" && this.eat("true")) return { kind: "bool", value: true };
    if (c === "f" && this.eat("false")) return { kind: "bool", value: false };
    if (c === "N" && this.eat("NaN")) return { kind: "float", value: NaN };
    if (c === "I" && this.eat("Infinity")) return { kind: "float", value: Infinity };
    if (c === "-" && this.s.startsWith("-Infinity", this.i)) {
      this.i += "-Infinity".length;
      return { kind: "float", value: -Infinity };
    }
    if (c === "-" || (c >= "0" && c <= "9")) return this.number();
    throw this.err("expecting value");
  }

  /** `(-?(?:0|[1-9]\d*))(\.\d+)?([eE][-+]?\d+)?`, as Python's scanner matches a number. */
  number(): Value {
    const s = this.s;
    const start = this.i;
    const isDigit = (k: number): boolean => k < s.length && s.charCodeAt(k) >= 48 && s.charCodeAt(k) <= 57;
    const digits = (): number => {
      const from = this.i;
      while (isDigit(this.i)) this.i++;
      return this.i - from;
    };
    if (s[this.i] === "-") this.i++;
    if (s[this.i] === "0") {
      this.i++;
    } else if (isDigit(this.i)) {
      digits();
    } else {
      this.i = start;
      throw this.err("expecting value");
    }
    const intEnd = this.i;
    let isFloat = false;
    if (s[this.i] === "." && isDigit(this.i + 1)) {
      this.i++;
      digits();
      isFloat = true;
    }
    if (s[this.i] === "e" || s[this.i] === "E") {
      const save = this.i;
      this.i++;
      if (s[this.i] === "+" || s[this.i] === "-") this.i++;
      if (digits() === 0) this.i = save;
      else isFloat = true;
    }
    if (isFloat) return { kind: "float", value: Number(s.slice(start, this.i)) };
    const t = s.slice(start, intEnd);
    const neg = t.startsWith("-");
    const body = neg ? t.slice(1) : t;
    return { kind: "int", value: body === "0" || !neg ? body : "-" + body };
  }

  hex4(): number {
    const h = this.s.slice(this.i, this.i + 4);
    if (h.length < 4 || !/^[0-9a-fA-F]{4}$/.test(h)) throw this.err("invalid \\uXXXX escape");
    this.i += 4;
    return parseInt(h, 16);
  }

  string(): string {
    const s = this.s;
    this.i++; // the opening quote
    let out = "";
    for (;;) {
      const start = this.i;
      while (this.i < s.length) {
        const c = s.charCodeAt(this.i);
        if (c === 0x22 || c === 0x5c || c < 0x20) break;
        this.i++;
      }
      out += s.slice(start, this.i);
      if (this.i >= s.length) throw this.err("unterminated string");
      const c = s[this.i]!;
      if (c === '"') {
        this.i++;
        return out;
      }
      if (c !== "\\") throw this.err("invalid control character");
      this.i++;
      if (this.i >= s.length) throw this.err("unterminated string");
      const e = s[this.i]!;
      this.i++;
      switch (e) {
        case '"':
          out += '"';
          break;
        case "\\":
          out += "\\";
          break;
        case "/":
          out += "/";
          break;
        case "b":
          out += "\b";
          break;
        case "f":
          out += "\f";
          break;
        case "n":
          out += "\n";
          break;
        case "r":
          out += "\r";
          break;
        case "t":
          out += "\t";
          break;
        case "u": {
          const u = this.hex4();
          let code = u;
          if (u >= 0xd800 && u < 0xdc00 && s.startsWith("\\u", this.i)) {
            const save = this.i;
            this.i += 2;
            const lo = this.hex4();
            if (lo >= 0xdc00 && lo < 0xe000) code = 0x10000 + ((u - 0xd800) << 10) + (lo - 0xdc00);
            else this.i = save;
          }
          // Python keeps a lone surrogate. The Rust crate cannot hold one,
          // so such a manifest is refused there, and here alike.
          if (code >= 0xd800 && code < 0xe000) throw this.err("a lone surrogate");
          out += String.fromCodePoint(code);
          break;
        }
        default:
          throw this.err("invalid escape");
      }
    }
  }

  array(): Value {
    this.i++;
    const items: Value[] = [];
    this.ws();
    if (this.s[this.i] === "]") {
      this.i++;
      return { kind: "array", items };
    }
    for (;;) {
      this.ws();
      items.push(this.value());
      this.ws();
      const c = this.s[this.i];
      if (c === ",") this.i++;
      else if (c === "]") {
        this.i++;
        return { kind: "array", items };
      } else throw this.err("expecting ',' delimiter");
    }
  }

  object(): Value {
    this.i++;
    const entries: [string, Value][] = [];
    const at = new Map<string, number>();
    this.ws();
    if (this.s[this.i] === "}") {
      this.i++;
      return { kind: "object", entries };
    }
    for (;;) {
      this.ws();
      if (this.s[this.i] !== '"') throw this.err("expecting property name enclosed in double quotes");
      const k = this.string();
      this.ws();
      if (this.s[this.i] !== ":") throw this.err("expecting ':' delimiter");
      this.i++;
      this.ws();
      const v = this.value();
      const slot = at.get(k);
      if (slot === undefined) {
        at.set(k, entries.length);
        entries.push([k, v]);
      } else {
        entries[slot]![1] = v;
      }
      this.ws();
      const c = this.s[this.i];
      if (c === ",") this.i++;
      else if (c === "}") {
        this.i++;
        return { kind: "object", entries };
      } else throw this.err("expecting ',' delimiter");
    }
  }
}

/** A string as `json.dumps` writes it with `ensure_ascii`: everything outside the printable ASCII range escaped, in lowercase hex. */
function writeStr(s: string): string {
  let out = '"';
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (c === 0x5c) out += "\\\\";
    else if (c === 0x22) out += '\\"';
    else if (c === 0x0a) out += "\\n";
    else if (c === 0x0d) out += "\\r";
    else if (c === 0x09) out += "\\t";
    else if (c === 0x08) out += "\\b";
    else if (c === 0x0c) out += "\\f";
    else if (c >= 0x20 && c <= 0x7e) out += s[i];
    else out += "\\u" + c.toString(16).padStart(4, "0");
  }
  return out + '"';
}

function writeFloat(f: number): string {
  if (Number.isNaN(f)) return "NaN";
  if (f === Infinity) return "Infinity";
  if (f === -Infinity) return "-Infinity";
  return repr(f);
}

/** Two keys in Python's order, which is code point order, not UTF-16's. */
function compareKeys(a: string, b: string): number {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const x = a.codePointAt(i)!;
    const y = b.codePointAt(i)!;
    if (x !== y) return x < y ? -1 : 1;
    if (x > 0xffff) i++;
  }
  return a.length - b.length;
}

/** `json.dumps(value, sort_keys=True)`: one line, `", "` and `": "`. */
export function dumpsSorted(v: Value): string {
  const parts: string[] = [];
  const go = (v: Value): void => {
    switch (v.kind) {
      case "null":
        parts.push("null");
        return;
      case "bool":
        parts.push(v.value ? "true" : "false");
        return;
      case "int":
        parts.push(v.value);
        return;
      case "float":
        parts.push(writeFloat(v.value));
        return;
      case "str":
        parts.push(writeStr(v.value));
        return;
      case "array":
        parts.push("[");
        v.items.forEach((x, i) => {
          if (i > 0) parts.push(", ");
          go(x);
        });
        parts.push("]");
        return;
      case "object": {
        const items = [...v.entries].sort((a, b) => compareKeys(a[0], b[0]));
        parts.push("{");
        items.forEach(([k, x], i) => {
          if (i > 0) parts.push(", ");
          parts.push(writeStr(k), ": ");
          go(x);
        });
        parts.push("}");
        return;
      }
    }
  };
  go(v);
  return parts.join("");
}

/** `json.dumps(value, indent=indent)`: keys in their order, `","` at line ends, and `": "` after a key. */
export function dumpsIndented(v: Value, indent: number): string {
  const parts: string[] = [];
  const go = (v: Value, level: number): void => {
    const pad = (n: number): void => {
      parts.push(" ".repeat(indent * n));
    };
    if (v.kind === "array" && v.items.length > 0) {
      parts.push("[");
      v.items.forEach((x, i) => {
        parts.push(i > 0 ? ",\n" : "\n");
        pad(level + 1);
        go(x, level + 1);
      });
      parts.push("\n");
      pad(level);
      parts.push("]");
    } else if (v.kind === "object" && v.entries.length > 0) {
      parts.push("{");
      v.entries.forEach(([k, x], i) => {
        parts.push(i > 0 ? ",\n" : "\n");
        pad(level + 1);
        parts.push(writeStr(k), ": ");
        go(x, level + 1);
      });
      parts.push("\n");
      pad(level);
      parts.push("}");
    } else {
      parts.push(dumpsSorted(v));
    }
  };
  go(v, 0);
  return parts.join("");
}

/**
 * A value as plain JavaScript: an integer as a number, or as a bigint where
 * a number cannot hold it exactly, and an object as a plain object in its
 * key order.
 */
export function toPlain(v: Value): unknown {
  switch (v.kind) {
    case "null":
      return null;
    case "bool":
    case "float":
    case "str":
      return v.value;
    case "int": {
      const b = BigInt(v.value);
      return b >= -(2n ** 53n) && b <= 2n ** 53n ? Number(b) : b;
    }
    case "array":
      return v.items.map(toPlain);
    case "object": {
      const out: Record<string, unknown> = {};
      for (const [k, x] of v.entries) {
        // Defined rather than assigned, so a key named `__proto__` is a key.
        Object.defineProperty(out, k, { value: toPlain(x), enumerable: true, writable: true, configurable: true });
      }
      return out;
    }
  }
}
