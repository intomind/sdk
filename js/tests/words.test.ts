/**
 * The table by which the fixtures' refusals, in the Python library's words,
 * are read in this package's: see `words.ts`.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { inJavaScript, refusalsInJavaScript, valuesInJavaScript } from "./words.ts";

test("a refusal in the library's words is read with each name a program passes, None, True, False, nan and inf, and text, a list and a dict as Python's repr writes them, as this package says them, and Python's own words as they are", () => {
  assert.equal(
    inJavaScript("TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'"),
    "TypeError: int() argument must be a string, a bytes-like object or a real number, not 'null'",
  );
  assert.equal(
    inJavaScript("ValueError: IntoMind One (00000000000000a3) is not one of this session's devices"),
    "ValueError: IntoMind One (00000000000000a3) is not one of this chain's devices",
  );
  assert.equal(
    inJavaScript("RuntimeError: Stand-in (stand-in-1) says it sends its encoder's output, and offers no way to hand its windows to the library, so no windows can be gathered from it"),
    "RuntimeError: Stand-in (stand-in-1) says it sends its encoder's output, and offers no way to hand its windows to the package, so no windows can be gathered from it",
  );
  assert.equal(
    inJavaScript("PlacementError: 'cpz' is not a position key. There are 78 on offer, the 10-10 codes and \"other\" for anywhere else, its place written in the note; `placement.positions()` lists them. The 32 names in `placement.SCALP` are accepted as well."),
    "PlacementError: \"cpz\" is not a position key. There are 78 on offer, the 10-10 codes and \"other\" for anywhere else, its place written in the note; `placement.positions()` lists them. The 32 named positions, such as \"top-of-head\", are accepted as well.",
  );
  assert.equal(inJavaScript("ValueError: targets[1] is nan, which is not a finite number"), "ValueError: targets[1] is NaN, which is not a finite number");
  assert.equal(inJavaScript("Invalid: bias[0] is -inf, and a head's bias is a whole number of four bytes"), "Invalid: bias[0] is -Infinity, and a head's bias is a whole number of four bytes");
  assert.equal(inJavaScript("ValueError: an inflow of 'nan' is not a number, nor is 'inf'"), 'ValueError: an inflow of "nan" is not a number, nor is "inf"');
  // A list and a dict as JSON writes them, text in either as JSON quotes it.
  assert.equal(
    inJavaScript("ValueError: the form is one of ['off', 'window', 'tokens', 'both'], not 'all'"),
    'ValueError: the form is one of ["off", "window", "tokens", "both"], not "all"',
  );
  assert.equal(
    valuesInJavaScript("rest-4ch cannot be analyzed with a min_pp_uv of {'a': 1, 'b': [None, True, nan], \"it's\": '\\x00'}: x"),
    'rest-4ch cannot be analyzed with a min_pp_uv of {"a": 1, "b": [null, true, NaN], "it\'s": "\\u0000"}: x',
  );
  // An analysis's settings and a recording's manifest keys are the library's own names in both packages.
  assert.equal(valuesInJavaScript("no-adc-bits does not record adc_bits, so the microvolts it holds cannot be computed"), "no-adc-bits does not record adc_bits, so the microvolts it holds cannot be computed");
  // Python's own words, which the package says word for word, keep Python's quotes.
  for (const own of [
    "AttributeError: 'bool' object has no attribute 'get'",
    "TypeError: unhashable type: 'list'",
    "TypeError: summary() got an unexpected keyword argument 'nonsense'",
    "KeyError: 'nonsense'",
    "ValueError: invalid literal for int() with base 10: '202²'",
    "ValueError: could not convert string to float: 'x'",
    "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc3 in position 2: unexpected end of data",
    "'leadoff is not a file in the archive'",
  ]) assert.equal(inJavaScript(own), own);
  assert.deepEqual(refusalsInJavaScript({ refused: "None is not a position key", note: "None" }), { refused: "null is not a position key", note: "None" });
  assert.deepEqual(refusalsInJavaScript({ refused: "'x' is not a position key", note: "'x'" }), { refused: '"x" is not a position key', note: "'x'" });
});
