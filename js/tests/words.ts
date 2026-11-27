/**
 * The one table by which a refusal in the Python library's words is read in
 * this package's: each name a Python program passes, as a JavaScript program
 * passes it, the library's `Session` of several devices as the package's
 * `DaisyChain`, Python's None, True and False as null, true and false, a
 * float Python writes as nan or inf as NaN or Infinity, as JavaScript writes
 * it, "this library" and "to the library", as the library speaks of itself, as
 * "this package" and "to the package", the sentence that points to the
 * library's `placement.SCALP`, which this package does not have, as the
 * package words it. Text Python's `repr` quotes, such as a value a
 * program passed, is read in double quotes as JSON quotes it, so a list
 * and a dict are read as JSON writes them: ['a', None] as ["a", null] and
 * {'a': 1} as {"a": 1}. Python's own words, which the package says word for
 * word, keep Python's quotes: see `PYTHONS_OWN`. Both packages refuse the
 * same input for the same reason, each naming what its own developer
 * passes, and with this table a fixture's refusal is compared with the
 * package's word for word. `valuesInJavaScript` reads only the values, for
 * a refusal whose names are the library's own in both packages, such as an
 * analysis's settings and a recording's manifest keys.
 */

/** Python's words for a value, each with the package's, the longest first. */
const VALUES_TABLE: readonly (readonly [string, string])[] = [
  ["NoneType", "null"],
  ["None", "null"],
  ["True", "true"],
  ["False", "false"],
  ["nan", "NaN"],
  ["inf", "Infinity"],
];

/** Python's words, each with the package's, the longest first: the names, then the values. */
const TABLE: readonly (readonly [string, string])[] = [
  ["The 32 names in `placement.SCALP` are accepted as well.", 'The 32 named positions, such as "top-of-head", are accepted as well.'],
  ["this session's devices", "this chain's devices"],
  ["leadoff_active", "leadoffActive"],
  ["'NoneType'", "'null'"],
  ["this library", "this package"],
  ["to the library", "to the package"],
  ["record_to", "recordTo"],
  ...VALUES_TABLE,
];

/** A quoted string as Python's `repr` writes one, opening and closing where no letter or digit touches the quote, so an apostrophe in a word is not one. */
const QUOTED = /(?<![A-Za-z0-9])'(?:[^'\\]|\\.)*'(?![A-Za-z0-9])|(?<![A-Za-z0-9])"(?:[^"\\]|\\.)*"(?![A-Za-z0-9])/g;

/** Every word of `table` in `text`, a quoted string aside, as the package says it. */
function plain(text: string, table: readonly (readonly [string, string])[] = TABLE): string {
  let out = text;
  for (const [python, js] of table) {
    if (/^\w+$/.test(python)) out = out.replace(new RegExp(`\\b${python}\\b`, "g"), js);
    else out = out.split(python).join(js);
  }
  return out;
}

/** `python`, a refusal in the Python library's words, as this package says it. */
export function inJavaScript(python: string): string {
  const text = python.split("'NoneType'").join("'null'");
  return outsideQuotes(text, (t) => plain(t));
}

/**
 * `python`, a refusal in the Python library's words, with only its values
 * as this package writes them: for a refusal whose names are the library's
 * own in both packages, such as an analysis's settings and a recording's
 * manifest keys.
 */
export function valuesInJavaScript(python: string): string {
  return outsideQuotes(python.split("'NoneType'").join("'null'"), (t) => plain(t, VALUES_TABLE));
}

/**
 * Where a quoted string keeps Python's quotes: in the words of Python's own
 * errors, which the package says word for word, a type's, an attribute's or
 * a codec's name, a keyword argument's name, a KeyError's key, the text
 * `int()` and `float()` could not read, and a whole refusal that is one quoted
 * string, as a KeyError's words are. Each holds the text before the quoted
 * string and the text after it.
 */
const PYTHONS_OWN: readonly (readonly [RegExp, RegExp])[] = [
  [/$/, /^ object\b/],
  [/$/, /^ codec\b/],
  [/(?:has no attribute|unhashable type:|real number, not|keyword argument|KeyError:|with base 10:|string to float:) $/, /^/],
  [/^$/, /^$/],
];

/** The text a quoted string Python's `repr` wrote holds. */
function unquoted(q: string): string {
  const escapes: Readonly<Record<string, string>> = { n: "\n", r: "\r", t: "\t", "\\": "\\", "'": "'", '"': '"' };
  return q.slice(1, -1).replace(/\\(x[0-9a-f]{2}|u[0-9a-f]{4}|U[0-9a-f]{8}|.)/g, (_, e: string) =>
    e.length > 1 ? String.fromCodePoint(parseInt(e.slice(1), 16)) : escapes[e] ?? `\\${e}`);
}

/** `text` with `read` applied to each stretch outside a quoted string, and each quoted string, but one in Python's own words, in double quotes as the package quotes text. */
function outsideQuotes(text: string, read: (t: string) => string): string {
  let out = "";
  let at = 0;
  for (const m of text.matchAll(QUOTED)) {
    const before = text.slice(0, m.index);
    const after = text.slice(m.index + m[0].length);
    const own = PYTHONS_OWN.some(([b, a]) => b.test(before) && a.test(after));
    out += read(text.slice(at, m.index)) + (own ? m[0] : JSON.stringify(unquoted(m[0])));
    at = m.index + m[0].length;
  }
  return out + read(text.slice(at));
}

/** `value`, plain data a program gave back, with each string under a key `refused` or `raised` as this package says it. */
export function refusalsInJavaScript(value: unknown, key: string | null = null): unknown {
  if (typeof value === "string") return key === "refused" || key === "raised" ? inJavaScript(value) : value;
  if (Array.isArray(value)) return value.map((v) => refusalsInJavaScript(v, key));
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(Object.entries(value as Record<string, unknown>).map(([k, v]) => [k, refusalsInJavaScript(v, k)]));
  }
  return value;
}
