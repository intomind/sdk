/**
 * Where the electrodes went, and what is under them: the part of the Python
 * library's `placement` module a manifest is built with.
 *
 * A layout is what a device offers: its slots and how they are wired into
 * channels. A placement is where a person put it, this time. A manifest
 * records both, the placement resolved against the scalp positions and the
 * regions under them, and the montage generated from the two. Both arrive as
 * JSON from an application, so every rule here is Python's rule on a Python
 * dict, refusal words included.
 *
 * What is under a 10-10 position was measured, as the Python library has it:
 * for 65 of the 77 positions, Table 3 of C. L. Scrivener and A. T. Reader,
 * "Variability of EEG electrode positions and their underlying brain regions:
 * visualizing gel artifacts from a simultaneous EEG-fMRI dataset", Brain and
 * Behavior 12 (2022), e2476, doi:10.1002/brb3.2476, published under the
 * Creative Commons Attribution 4.0 International license,
 * https://creativecommons.org/licenses/by/4.0/. Changed here: the paper's
 * abbreviations are written out as the paper defines them, the names are in
 * lower case, the one "Precuneous" is spelled "precuneus" as the paper spells
 * it elsewhere, and its column of Brodmann areas is left out.
 */

import * as M from "./montage.ts";
import { Fail, get, item, keyError, list, oneToN } from "./montage.ts";

/**
 * A placement refusal, in the library's words a manifest records, each
 * value `words` shows written by Python's `repr`, and in the words thrown to
 * a program, each written as JavaScript writes it.
 */
function placementFail(words: (r: (v: Value) => string) => string): Fail {
  return new Fail("placement", words(J.pyRepr), words(J.jsRepr));
}
import { general } from "./pyfmt.ts";
import * as J from "./pyjson.ts";
import type { Value } from "./pyjson.ts";

const LEFT = "left";
const RIGHT = "right";
const MIDLINE = "midline";
const RECORDING = "recording";
const BIAS = "bias";
const REFERENCE = "reference";

/** What can be under an electrode: key, words, and kind. */
export const REGIONS: readonly (readonly [string, string, string])[] = [
  ["frontal-pole", "frontal pole", "cortical"],
  ["dlpfc", "dorsolateral prefrontal", "cortical"],
  ["vlpfc", "ventrolateral prefrontal", "cortical"],
  ["orbitofrontal", "orbitofrontal", "cortical"],
  ["premotor", "premotor and supplementary motor", "cortical"],
  ["motor", "primary motor", "cortical"],
  ["somatosensory", "primary somatosensory", "cortical"],
  ["superior-parietal", "superior parietal", "cortical"],
  ["tpj", "inferior parietal and temporoparietal junction", "cortical"],
  ["temporal-pole", "temporal pole", "cortical"],
  ["superior-temporal", "superior temporal", "cortical"],
  ["inferior-temporal", "middle and inferior temporal", "cortical"],
  ["lateral-occipital", "lateral occipital", "cortical"],
  ["medial-occipital", "medial occipital", "cortical"],
  ["temporalis", "temporalis (jaw muscle)", "muscle"],
  ["frontalis", "frontalis (brow muscle)", "muscle"],
  ["ocular", "eye movement", "ocular"],
  ["mastoid-bone", "mastoid bone, little cortex beneath", "bone"],
  // The regions a study found under the 10-10 positions, named as it names them with its abbreviations written out. Two of them, the frontal pole and the temporal pole, are above already under the same names.
  ["superior-frontal-gyrus", "superior frontal gyrus", "cortical"],
  ["middle-frontal-gyrus", "middle frontal gyrus", "cortical"],
  ["inferior-frontal-gyrus-pars-triangularis", "inferior frontal gyrus (pars triangularis)", "cortical"],
  ["inferior-frontal-gyrus-pars-opercularis", "inferior frontal gyrus (pars opercularis)", "cortical"],
  ["juxtapositional-lobule-cortex", "juxtapositional lobule cortex (formerly supplementary motor cortex)", "cortical"],
  ["precentral-gyrus", "precentral gyrus", "cortical"],
  ["postcentral-gyrus", "postcentral gyrus", "cortical"],
  ["central-opercular-cortex", "central opercular cortex", "cortical"],
  ["anterior-supramarginal-gyrus", "anterior supramarginal gyrus", "cortical"],
  ["posterior-supramarginal-gyrus", "posterior supramarginal gyrus", "cortical"],
  ["angular-gyrus", "angular gyrus", "cortical"],
  ["superior-parietal-lobule", "superior parietal lobule", "cortical"],
  ["precuneus-cortex", "precuneus cortex", "cortical"],
  ["anterior-superior-temporal-gyrus", "anterior superior temporal gyrus", "cortical"],
  ["posterior-superior-temporal-gyrus", "posterior superior temporal gyrus", "cortical"],
  ["planum-temporale", "planum temporale", "cortical"],
  ["anterior-middle-temporal-gyrus", "anterior middle temporal gyrus", "cortical"],
  ["posterior-middle-temporal-gyrus", "posterior middle temporal gyrus", "cortical"],
  ["middle-temporal-gyrus-temporooccipital-part", "middle temporal gyrus (temporooccipital part)", "cortical"],
  ["anterior-inferior-temporal-gyrus", "anterior inferior temporal gyrus", "cortical"],
  ["posterior-inferior-temporal-gyrus", "posterior inferior temporal gyrus", "cortical"],
  ["inferior-temporal-gyrus-temporooccipital-part", "inferior temporal gyrus (temporooccipital part)", "cortical"],
  ["superior-lateral-occipital-cortex", "superior lateral occipital cortex", "cortical"],
  ["inferior-lateral-occipital-cortex", "inferior lateral occipital cortex", "cortical"],
  ["occipital-pole", "occipital pole", "cortical"],
  ["cuneal-cortex", "cuneal cortex", "cortical"],
];

/** Where a person can put an electrode: key, words, the sides it comes in with each side's nearest 10-10 position, and what it is over, most dominant first. */
type Zone = readonly [string, string, readonly (readonly [string, string])[], readonly string[]];

const ZONES: readonly Zone[] = [
  ["forehead-outer", "outer forehead, above the brow's outer end", [[LEFT, "AF7"], [RIGHT, "AF8"]], ["frontal-pole", "dlpfc", "ocular", "frontalis"]],
  ["forehead-inner", "inner forehead, above the brow", [[LEFT, "Fp1"], [RIGHT, "Fp2"]], ["frontal-pole", "orbitofrontal", "ocular", "frontalis"]],
  ["forehead-center", "center of the forehead", [[MIDLINE, "Fpz"]], ["frontal-pole", "ocular", "frontalis"]],
  ["hairline", "forehead at the hairline", [[LEFT, "F5"], [RIGHT, "F6"], [MIDLINE, "Fz"]], ["dlpfc", "vlpfc", "frontalis"]],
  ["temple-front", "temple, front", [[LEFT, "F7"], [RIGHT, "F8"]], ["vlpfc", "temporalis", "temporal-pole"]],
  ["temple-back", "temple, back", [[LEFT, "FT7"], [RIGHT, "FT8"]], ["temporalis", "superior-temporal", "vlpfc"]],
  ["in-front-of-ear", "in front of the ear", [[LEFT, "FT9"], [RIGHT, "FT10"]], ["temporalis", "temporal-pole"]],
  ["above-ear", "above the ear", [[LEFT, "T7"], [RIGHT, "T8"]], ["superior-temporal", "temporalis"]],
  ["behind-ear", "behind the ear, on the bone", [[LEFT, "TP9"], [RIGHT, "TP10"]], ["mastoid-bone", "inferior-temporal"]],
  ["behind-above-ear", "behind and above the ear", [[LEFT, "TP7"], [RIGHT, "TP8"]], ["tpj", "superior-temporal"]],
  ["side-of-head", "side of the head, above and forward of the ear", [[LEFT, "C5"], [RIGHT, "C6"]], ["somatosensory", "motor"]],
  ["crown-side", "crown, off to the side", [[LEFT, "C3"], [RIGHT, "C4"]], ["motor", "somatosensory", "premotor"]],
  ["top-of-head", "top of the head", [[MIDLINE, "Cz"]], ["motor", "premotor"]],
  ["back-of-head-side", "back of the head, off to the side", [[LEFT, "P3"], [RIGHT, "P4"]], ["superior-parietal", "tpj"]],
  ["crown-back", "crown, toward the back", [[MIDLINE, "Pz"]], ["superior-parietal"]],
  ["back-of-head-low", "back of the head, low", [[LEFT, "O1"], [RIGHT, "O2"]], ["lateral-occipital", "medial-occipital"]],
  ["back-of-head-center", "back of the head, center", [[MIDLINE, "Oz"]], ["medial-occipital"]],
  ["nape", "the bump at the base of the skull", [[MIDLINE, "Iz"]], ["medial-occipital"]],
];

/** Every scalp position, in the order the Python library builds them. */
function scalpTable(): [string, Value][] {
  const out: [string, Value][] = [];
  for (const [zone, human, sides, over] of ZONES) {
    for (const [side, tt] of sides) {
      const key = side === MIDLINE ? zone : `${side}-${zone}`;
      const words = side === MIDLINE ? human : `${side} ${human}`;
      out.push([
        key,
        J.object([
          ["key", J.str(key)],
          ["zone", J.str(zone)],
          ["side", J.str(side)],
          ["human", J.str(words)],
          ["ten_ten", J.str(tt)],
          ["over", J.array(over.map(J.str))],
        ]),
      ]);
    }
  }
  return out;
}

/** The whole 10-10 system, front to back and left to right, then the earlobes and the mastoids: the positions a placement may name by their codes. */
const TEN_TEN = [
  "Nz", "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F9", "F7",
  "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "F10", "FT9", "FT7",
  "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "FT10", "T9", "T7",
  "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "T10", "TP9", "TP7",
  "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP10", "P9", "P7",
  "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "P10", "PO7", "PO3",
  "POz", "PO4", "PO8", "O1", "Oz", "O2", "Iz", "A1", "A2", "M1", "M2",
];

/** The side a 10-10 code names: z the midline, an odd number the left, an even one the right. */
function sideOf(code: string): string {
  if (code.endsWith("z")) return MIDLINE;
  return Number(code.replace(/^[A-Za-z]+/, "")) % 2 === 1 ? LEFT : RIGHT;
}

/**
 * What is under each 10-10 position the study covered, as measured: Table 3
 * of the study this module credits, row by row in its order, every region it
 * found with its share, the percentage of the participants usable at that
 * position whose electrode lay over it, as the paper prints it. The other
 * positions are absent: nothing is said about them.
 */
const UNDER: readonly (readonly [string, readonly (readonly [string, number])[]])[] = [
  ["AF3", [["frontal-pole", 100.0]]],
  ["AF4", [["frontal-pole", 100.0]]],
  ["AF7", [["frontal-pole", 100.0]]],
  ["AF8", [["frontal-pole", 100.0]]],
  ["AFz", [["frontal-pole", 62.5], ["superior-frontal-gyrus", 37.5]]],
  ["C1", [["precentral-gyrus", 71.0], ["postcentral-gyrus", 29.0]]],
  ["C2", [["precentral-gyrus", 87.5], ["postcentral-gyrus", 12.5]]],
  ["C3", [["postcentral-gyrus", 87.5], ["precentral-gyrus", 12.5]]],
  ["C4", [["postcentral-gyrus", 85.0], ["precentral-gyrus", 15.0]]],
  ["C5", [["postcentral-gyrus", 50.0], ["anterior-supramarginal-gyrus", 44.4], ["precentral-gyrus", 5.6]]],
  ["C6", [["anterior-supramarginal-gyrus", 72.0], ["postcentral-gyrus", 28.0]]],
  ["CP1", [["superior-parietal-lobule", 66.67], ["postcentral-gyrus", 27.78], ["superior-lateral-occipital-cortex", 5.56]]],
  ["CP2", [["superior-parietal-lobule", 80.0], ["postcentral-gyrus", 13.0], ["superior-lateral-occipital-cortex", 7.0]]],
  ["CP3", [["posterior-supramarginal-gyrus", 47.0], ["superior-parietal-lobule", 27.0], ["angular-gyrus", 13.0], ["postcentral-gyrus", 13.0]]],
  ["CP4", [["angular-gyrus", 50.0], ["superior-parietal-lobule", 42.0], ["postcentral-gyrus", 8.0]]],
  ["CP5", [["posterior-supramarginal-gyrus", 61.11], ["anterior-supramarginal-gyrus", 16.67], ["angular-gyrus", 11.11], ["posterior-superior-temporal-gyrus", 5.56], ["superior-lateral-occipital-cortex", 5.56]]],
  ["CP6", [["angular-gyrus", 53.0], ["posterior-supramarginal-gyrus", 40.0], ["superior-lateral-occipital-cortex", 7.0]]],
  ["CPz", [["postcentral-gyrus", 62.5], ["precuneus-cortex", 18.75], ["superior-parietal-lobule", 12.5], ["superior-lateral-occipital-cortex", 6.25]]],
  ["Cz", [["precentral-gyrus", 87.0], ["postcentral-gyrus", 13.0]]],
  ["F1", [["superior-frontal-gyrus", 71.0], ["frontal-pole", 29.0]]],
  ["F2", [["superior-frontal-gyrus", 77.0], ["frontal-pole", 23.0]]],
  ["F3", [["middle-frontal-gyrus", 64.0], ["frontal-pole", 36.0]]],
  ["F4", [["frontal-pole", 53.0], ["middle-frontal-gyrus", 47.0]]],
  ["F5", [["middle-frontal-gyrus", 65.0], ["frontal-pole", 35.0]]],
  ["F6", [["middle-frontal-gyrus", 50.0], ["frontal-pole", 50.0]]],
  ["F7", [["inferior-frontal-gyrus-pars-triangularis", 83.0], ["frontal-pole", 11.0], ["inferior-frontal-gyrus-pars-opercularis", 6.0]]],
  ["F8", [["inferior-frontal-gyrus-pars-triangularis", 66.67], ["inferior-frontal-gyrus-pars-opercularis", 16.67], ["middle-frontal-gyrus", 5.56], ["precentral-gyrus", 5.56], ["frontal-pole", 5.56]]],
  ["FC1", [["superior-frontal-gyrus", 93.75], ["middle-frontal-gyrus", 6.25]]],
  ["FC2", [["superior-frontal-gyrus", 82.0], ["precentral-gyrus", 18.0]]],
  ["FC3", [["middle-frontal-gyrus", 89.0], ["precentral-gyrus", 11.0]]],
  ["FC4", [["middle-frontal-gyrus", 92.0], ["precentral-gyrus", 8.0]]],
  ["FC5", [["precentral-gyrus", 88.0], ["middle-frontal-gyrus", 6.0], ["inferior-frontal-gyrus-pars-opercularis", 6.0]]],
  ["FC6", [["precentral-gyrus", 81.0], ["postcentral-gyrus", 13.0], ["middle-frontal-gyrus", 6.0]]],
  ["FCz", [["juxtapositional-lobule-cortex", 66.7], ["superior-frontal-gyrus", 33.3]]],
  ["Fp1", [["frontal-pole", 100.0]]],
  ["Fp2", [["frontal-pole", 100.0]]],
  ["Fpz", [["frontal-pole", 100.0]]],
  ["FT10", [["anterior-middle-temporal-gyrus", 66.7], ["posterior-inferior-temporal-gyrus", 11.1], ["temporal-pole", 11.1], ["posterior-middle-temporal-gyrus", 5.6], ["anterior-inferior-temporal-gyrus", 5.6]]],
  ["FT7", [["precentral-gyrus", 64.29], ["temporal-pole", 14.29], ["anterior-superior-temporal-gyrus", 14.29], ["inferior-frontal-gyrus-pars-opercularis", 7.14]]],
  ["FT8", [["precentral-gyrus", 62.5], ["anterior-superior-temporal-gyrus", 25.0], ["central-opercular-cortex", 6.25], ["posterior-middle-temporal-gyrus", 6.25]]],
  ["FT9", [["anterior-middle-temporal-gyrus", 42.0], ["temporal-pole", 37.0], ["posterior-middle-temporal-gyrus", 16.0], ["anterior-inferior-temporal-gyrus", 5.0]]],
  ["Fz", [["superior-frontal-gyrus", 100.0]]],
  ["O1", [["occipital-pole", 100.0]]],
  ["O2", [["occipital-pole", 100.0]]],
  ["Oz", [["occipital-pole", 100.0]]],
  ["P1", [["superior-lateral-occipital-cortex", 95.0], ["superior-parietal-lobule", 5.0]]],
  ["P2", [["superior-lateral-occipital-cortex", 100.0]]],
  ["P3", [["superior-lateral-occipital-cortex", 95.0], ["angular-gyrus", 5.0]]],
  ["P4", [["superior-lateral-occipital-cortex", 100.0]]],
  ["P5", [["superior-lateral-occipital-cortex", 89.47], ["inferior-lateral-occipital-cortex", 5.26], ["angular-gyrus", 5.26]]],
  ["P6", [["superior-lateral-occipital-cortex", 100.0]]],
  ["P7", [["inferior-lateral-occipital-cortex", 75.0], ["superior-lateral-occipital-cortex", 20.0], ["middle-temporal-gyrus-temporooccipital-part", 5.0]]],
  ["P8", [["inferior-lateral-occipital-cortex", 85.0], ["superior-lateral-occipital-cortex", 10.0], ["angular-gyrus", 5.0]]],
  ["PO3", [["superior-lateral-occipital-cortex", 65.0], ["occipital-pole", 35.0]]],
  ["PO4", [["occipital-pole", 58.0], ["superior-lateral-occipital-cortex", 42.0]]],
  ["PO7", [["inferior-lateral-occipital-cortex", 80.0], ["superior-lateral-occipital-cortex", 15.0], ["occipital-pole", 5.0]]],
  ["PO8", [["occipital-pole", 33.33], ["superior-lateral-occipital-cortex", 33.33], ["inferior-lateral-occipital-cortex", 33.33]]],
  ["POz", [["cuneal-cortex", 37.5], ["occipital-pole", 31.3], ["superior-lateral-occipital-cortex", 25.0], ["precuneus-cortex", 6.3]]],
  ["Pz", [["precuneus-cortex", 60.0], ["superior-lateral-occipital-cortex", 33.33], ["superior-parietal-lobule", 6.66]]],
  ["T7", [["posterior-superior-temporal-gyrus", 72.2], ["posterior-middle-temporal-gyrus", 16.6], ["anterior-superior-temporal-gyrus", 5.6], ["anterior-supramarginal-gyrus", 5.6]]],
  ["T8", [["posterior-superior-temporal-gyrus", 52.9], ["posterior-middle-temporal-gyrus", 29.4], ["planum-temporale", 5.9], ["anterior-supramarginal-gyrus", 5.9], ["central-opercular-cortex", 5.9]]],
  ["TP7", [["middle-temporal-gyrus-temporooccipital-part", 73.7], ["posterior-supramarginal-gyrus", 15.8], ["posterior-middle-temporal-gyrus", 10.5]]],
  ["TP8", [["middle-temporal-gyrus-temporooccipital-part", 75.0], ["angular-gyrus", 20.0], ["posterior-supramarginal-gyrus", 5.0]]],
  ["TP9", [["inferior-temporal-gyrus-temporooccipital-part", 100.0]]],
  ["TP10", [["inferior-temporal-gyrus-temporooccipital-part", 89.0], ["middle-temporal-gyrus-temporooccipital-part", 11.0]]],
];

const UNDER_BY_CODE = new Map(UNDER);
const UNDER_SOURCE = "Scrivener and Reader 2022, doi:10.1002/brb3.2476";

/** One 10-10 position as the Python library offers it, with the regions the study found under it, if it covered it. */
function tenTen(code: string): Value {
  return J.object([
    ["key", J.str(code)],
    ["zone", J.NULL],
    ["side", J.str(sideOf(code))],
    ["human", J.str(code)],
    ["ten_ten", J.str(code)],
    ["over", J.array((UNDER_BY_CODE.get(code) ?? []).map(([r]) => J.str(r)))],
  ]);
}

/** The position for anywhere the 10-10 system has no position, the place written in the placement's note. */
function other(): Value {
  return J.object([
    ["key", J.str("other")],
    ["zone", J.NULL],
    ["side", J.NULL],
    ["human", J.str("a position off the 10-10 system")],
    ["ten_ten", J.NULL],
    ["over", J.array([])],
  ]);
}

/** One position by key: a 10-10 code, "other", or one of the 32 named positions the library's `SCALP` holds, or the refusal naming how many there are on offer and saying the named positions are accepted as well. */
export function scalp(key: Value): Value {
  const table = scalpTable();
  if (key.kind === "array" || key.kind === "object") {
    throw new Fail("other", `TypeError: unhashable type: ${J.jsRepr(key)}`);
  }
  if (key.kind === "str") {
    const found = table.find(([t]) => t === key.value);
    if (found !== undefined) return found[1];
    if (TEN_TEN.includes(key.value)) return tenTen(key.value);
    if (key.value === "other") return other();
  }
  const named = table.length;
  const refusal = (shown: string, also: string): string =>
    `${shown} is not a position key. There are ${TEN_TEN.length + 1} on offer, the 10-10 codes and "other" for anywhere else, its place written in the note; \`placement.positions()\` lists them. ${also}`;
  // A recording keeps the library's words, which name its `SCALP`; a
  // program is told in words that name nothing this package does not have.
  throw new Fail(
    "placement",
    refusal(J.pyRepr(key), `The ${named} names in \`placement.SCALP\` are accepted as well.`),
    refusal(J.jsRepr(key), `The ${named} named positions, such as "top-of-head", are accepted as well.`),
  );
}

function region(key: Value): readonly [string, string, string] | null {
  if (key.kind !== "str") return null;
  return REGIONS.find(([r]) => r === key.value) ?? null;
}

/**
 * What is under a scalp position, its side carried down, each entry marked as
 * inferred. Under a 10-10 position each region carries its share and the
 * study it comes from. A position the study did not cover, and "other",
 * carry no region.
 */
export function regionsUnder(scalpKey: Value): Value[] {
  const s = scalp(scalpKey);
  const side = J.pyStr(J.field(s, "side"));
  const code = J.pyStr(J.field(s, "key"));
  const shares = new Map(TEN_TEN.includes(code) ? (UNDER_BY_CODE.get(code) ?? []) : []);
  const out: Value[] = [];
  for (const key of list(J.field(s, "over"))) {
    const r = region(key);
    if (r === null) throw keyError(J.pyStr(key));
    const [k, human, kind] = r;
    const sided = side !== MIDLINE && kind !== "ocular" ? `${side} ${human}` : human;
    const entry: [string, Value][] = [
      ["region", J.str(k)],
      ["human", J.str(sided)],
      ["kind", J.str(kind)],
      ["side", J.str(side)],
      ["derived", J.bool(true)],
    ];
    const share = shares.get(k);
    if (share !== undefined) entry.push(["share", J.float(share)], ["source", J.str(UNDER_SOURCE)]);
    out.push(J.object(entry));
  }
  return out;
}

/** A key a Python dict can hold. A list or an object cannot be one. */
function hashable(v: Value): void {
  if (v.kind === "array" || v.kind === "object") throw new Fail("other", `TypeError: unhashable type: ${J.jsRepr(v)}`);
}

/** `{s["key"]: s for s in slots}`: insertion order, a repeated key keeping its first place and its last slot. */
function slotsByKey(slots: Value[]): [Value, Value][] {
  const out: [Value, Value][] = [];
  for (const s of slots) {
    const k = item(s, "key");
    hashable(k);
    const have = out.find(([h]) => J.pyEq(h, k));
    if (have !== undefined) have[1] = s;
    else out.push([k, s]);
  }
  return out;
}

function has(slots: [Value, Value][], k: Value): boolean {
  hashable(k);
  return slots.some(([h]) => J.pyEq(h, k));
}

function slotOf(slots: [Value, Value][], k: Value): Value | null {
  return slots.find(([h]) => J.pyEq(h, k))?.[1] ?? null;
}

/** The keys of a JSON object, as values: what `for k in d` iterates. */
function keys(d: Value): Value[] {
  if (d.kind === "object") return d.entries.map(([k]) => J.str(k));
  if (!J.truthy(d)) return [];
  throw new Fail("other", `AttributeError: ${J.jsRepr(d)} has no keys`);
}

/** `k in d` for a JSON object, whose keys are strings. */
function inObject(d: Value, k: Value): boolean {
  hashable(k);
  return d.kind === "object" && k.kind === "str" && d.entries.some(([h]) => h === k.value);
}

/** What `for r in value` yields: a list's items, a string's characters, an object's keys. */
function iterate(v: Value): Value[] {
  if (v.kind === "array") return v.items;
  if (v.kind === "str") return Array.from(v.value, (c) => J.str(c));
  if (v.kind === "object") return keys(v);
  throw new Fail("other", `TypeError: ${J.jsRepr(v)} is not iterable`);
}

function roleIs(s: Value, role: string): boolean {
  const r = item(s, "role");
  return r.kind === "str" && r.value === role;
}

/** Refuse a layout that could not be wired. */
function validateLayout(f: Value): void {
  const slots = slotsByKey(list(get(f, "slots")));
  if (slots.length === 0) throw new Fail("placement", "a layout with no slots describes nothing");
  const chans = list(get(f, "channels"));
  if (chans.length === 0) throw placementFail((r) => `${r(get(f, "key"))} declares no channels`);
  const idx = chans.map((c) => item(c, "index"));
  if (!oneToN(idx)) {
    throw placementFail((r) => `channel indices must be 1..n with no gaps, got ${r(J.array(idx))}`);
  }
  const refs: Value[] = [];
  for (const [, s] of slots) if (roleIs(s, REFERENCE)) refs.push(s);
  for (const c of chans) {
    const at = J.pyStr(item(c, "index"));
    const derivation = item(c, "derivation");
    if (!(derivation.kind === "str" && (derivation.value === M.BIPOLAR || derivation.value === M.REFERENTIAL))) {
      throw placementFail((r) => `ch${at}: bad derivation ${r(derivation)}`);
    }
    for (const pole of ["positive", "negative"]) {
      const named = get(c, pole);
      if (!has(slots, named)) {
        throw placementFail((r) => `ch${at}: ${pole} names slot ${r(named)}, which this layout does not have`);
      }
    }
    if (derivation.value === M.REFERENTIAL) {
      if (refs.length !== 1) {
        throw new Fail(
          "placement",
          `ch${at} is single-ended, so this layout needs exactly one reference slot; it has ${refs.length}`,
        );
      }
      const negative = item(c, "negative");
      const reference = item(refs[0]!, "key");
      if (!J.pyEq(negative, reference)) {
        throw placementFail((r) => `ch${at} is single-ended against ${r(negative)}, but the reference slot is ${r(reference)}`);
      }
    }
  }
  for (const p of list(get(f, "presets"))) validatePreset(f, p, slots);
}

function validatePreset(f: Value, p: Value, slots: [Value, Value][]): void {
  // `f['key']` is read only where the words need it, as Python reads it.
  const lay = (): string => J.pyStr(item(f, "key"));
  if (!J.truthy(get(p, "key"))) throw new Fail("placement", `${lay()}: a preset needs a key`);
  const sites = get(p, "sites");
  const missing: Value[] = [];
  for (const [k, s] of slots) if (!inObject(sites, k) && !roleIs(s, BIAS)) missing.push(k);
  if (missing.length > 0) {
    const names: string[] = [];
    for (const k of missing) {
      if (k.kind !== "str") throw new Fail("other", `TypeError: sequence item: expected str, ${J.jsRepr(k)}`);
      names.push(k.value);
    }
    names.sort(codePointOrder);
    throw new Fail("placement", `${lay()}/${J.pyStr(item(p, "key"))}: no site for slot(s) ${names.join(", ")}`);
  }
  if (sites.kind === "object") {
    for (const [k, v] of sites.entries) {
      if (!has(slots, J.str(k))) {
        throw placementFail((r) => `${lay()}/${J.pyStr(item(p, "key"))}: ${r(J.str(k))} is not a slot`);
      }
      scalp(v);
    }
  } else if (J.truthy(sites)) {
    throw new Fail("other", "AttributeError: a preset's sites is not a dict");
  }
  if (get(p, "rotation_deg").kind !== "null" && !J.truthy(get(f, "rotatable"))) {
    const l = lay();
    throw new Fail("placement", `${l}/${J.pyStr(item(p, "key"))} states a rotation, but ${l} is not rotatable`);
  }
}

/** Two strings in Python's order, which is code point order. */
function codePointOrder(a: string, b: string): number {
  const x = Array.from(a, (c) => c.codePointAt(0)!);
  const y = Array.from(b, (c) => c.codePointAt(0)!);
  for (let i = 0; i < Math.min(x.length, y.length); i++) if (x[i] !== y[i]) return x[i]! - y[i]!;
  return x.length - y.length;
}

/** Every position on offer, in head order, front to back, then "other": the picker. */
export function positions(): Value[] {
  return [...TEN_TEN.map(tenTen), other()];
}

/** A dict's key as JSON writes it: a string as itself, anything else as its JSON. */
function jsonKey(k: Value): string {
  return k.kind === "str" ? k.value : J.dumps(k);
}

/** `dict(x)` for a mapping: a copy with the same entries in the same order. */
function dictOf(v: Value): Value {
  if (v.kind === "object") return J.object(v.entries.map(([k, x]) => [k, x]));
  throw new Fail("other", `TypeError: ${J.jsRepr(v)} is not a dict`);
}

/** What `placement` takes after the layout and the sites, by Python's keyword names. */
export interface PlacementOptions {
  /** The name of a preset the layout declares, which supplies the sites. */
  readonly preset?: Value | undefined;
  /** How far the layout was turned, in degrees, as Python's `rotation_deg`. */
  readonly rotationDeg?: Value | undefined;
  /** Regions named by hand for slots, in place of what is under their positions. */
  readonly over?: Value | undefined;
  /** Free text, never parsed: what fits no vocabulary, such as where an electrode at "other" sat. */
  readonly note?: Value | undefined;
  /** Whether the bias was driven, as Python's `bias_driven`. True when not given. */
  readonly biasDriven?: Value | undefined;
  /** One position for every electrode, a 10-10 code or "other". */
  readonly position?: Value | undefined;
}

/**
 * Where a layout was put, on this head, this time, as the Python library's
 * `placement` makes it. `sites` maps slot key to scalp key and may be left
 * null when `preset` names one the layout declares. `position` puts every
 * electrode at one position, leaving out a bias that is not driven, and
 * takes neither sites nor a preset.
 */
export function placement(lay: Value, sites: Value = J.NULL, options: PlacementOptions = {}): Value {
  const preset = options.preset ?? J.NULL;
  let rotation = options.rotationDeg ?? J.NULL;
  const over = options.over ?? J.NULL;
  const note = options.note ?? J.str("");
  const biasDriven = options.biasDriven ?? J.bool(true);
  const position = options.position ?? J.NULL;
  if (position.kind !== "null") {
    if (sites.kind !== "null" || preset.kind !== "null") {
      throw new Fail(
        "placement",
        `${J.pyStr(item(lay, "key"))}: \`position\` puts every electrode at one position, so \`sites\` and \`preset\` are not taken with it`,
      );
    }
    const every: [string, Value][] = [];
    for (const s of list(item(lay, "slots"))) {
      if (roleIs(s, BIAS) && !J.truthy(biasDriven)) continue;
      const k = item(s, "key");
      hashable(k);
      const key = jsonKey(k);
      const have = every.find(([h]) => h === key);
      if (have !== undefined) have[1] = position;
      else every.push([key, position]);
    }
    sites = J.object(every);
  }
  let chosen: Value | null = null;
  if (preset.kind !== "null") {
    const declared = list(getOr(lay, "presets", J.NULL));
    chosen = declared.find((p) => J.pyEq(item(p, "key"), preset)) ?? null;
    if (chosen === null) {
      const have = declared.map((p) => J.pyStr(item(p, "key"))).join(", ");
      throw placementFail((r) => `${J.pyStr(item(lay, "key"))} has no preset ${r(preset)}` + (have !== "" ? `; it has ${have}` : ""));
    }
  }
  let given = sites;
  if (!J.truthy(given)) given = chosen === null ? J.NULL : getOr(chosen, "sites", J.NULL);
  const placedSites = J.truthy(given) ? dictOf(given) : J.object([]);
  if (rotation.kind === "null" && chosen !== null) rotation = getOr(chosen, "rotation_deg", J.NULL);
  const p = J.object([
    ["layout", item(lay, "key")],
    ["preset", preset],
    ["sites", placedSites],
    ["rotation_deg", rotation],
    ["over", J.truthy(over) ? dictOf(over) : J.object([])],
    ["note", J.truthy(note) ? note : J.str("")],
    ["bias_driven", J.bool(J.truthy(biasDriven))],
  ]);
  validatePlacement(lay, p);
  return p;
}

/** A layout as an application declares it, checked. Null for nothing. */
export function layoutFrom(d: Value): Value | null {
  if (!J.truthy(d)) return null;
  validateLayout(d);
  return d;
}

/** `p.get(key, default)`. */
function getOr(d: Value, key: string, fallback: Value): Value {
  if (d.kind !== "object") throw new Fail("other", `AttributeError: ${J.jsRepr(d)} has no get`);
  return J.get(d, key) ?? fallback;
}

/** Refuse a placement that does not put every electrode somewhere. */
function validatePlacement(lay: Value, p: Value): void {
  const theirs = get(p, "layout");
  const ours = get(lay, "key");
  if (!J.pyEq(theirs, ours)) {
    throw placementFail((r) => `this placement is for layout ${r(theirs)}, not ${r(ours)}`);
  }
  const name = (): string => J.pyStr(item(lay, "key"));
  const slots = slotsByKey(list(item(lay, "slots")));
  const sites = get(p, "sites");
  for (const k of keys(sites)) {
    if (!has(slots, k)) throw placementFail((r) => `${r(k)} is not a slot on ${name()}`);
    scalp(J.field(sites, J.pyStr(k)));
  }
  const yes = J.bool(true);
  for (const [k, s] of slots) {
    if (inObject(sites, k)) continue;
    if (roleIs(s, BIAS) && !J.truthy(getOr(p, "bias_driven", yes))) continue;
    throw placementFail((r) => `${name()}: no site for slot ${r(k)} (${J.pyStr(item(s, "human"))})`);
  }
  const over = get(p, "over");
  for (const k of keys(over)) {
    if (!has(slots, k)) throw placementFail((r) => `override names ${r(k)}, which is not a slot`);
    for (const r of iterate(J.field(over, J.pyStr(k)))) {
      if (region(r) === null) throw placementFail((w) => `${w(r)} is not a region; there are ${REGIONS.length}`);
    }
  }
  const rotation = get(p, "rotation_deg");
  if (rotation.kind !== "null" && !J.truthy(get(lay, "rotatable"))) {
    throw new Fail("placement", `${name()} is not rotatable, so a rotation of ${J.pyStr(rotation)} means nothing`);
  }
}

/** The placement with every slot's site and regions filled in. */
export function resolve(lay: Value, p: Value): Value {
  validatePlacement(lay, p);
  const slots = slotsByKey(list(item(lay, "slots")));
  const sites = get(p, "sites");
  const over = get(p, "over");
  const out: Value[] = [];
  for (const [key, s] of slots) {
    const siteKey = sites.kind === "object" && key.kind === "str" ? J.field(sites, key.value) : J.NULL;
    if (siteKey.kind === "null") continue;
    const sc = scalp(siteKey);
    let regions: Value[];
    if (inObject(over, key)) {
      regions = iterate(J.field(over, J.pyStr(key))).map((r) => {
        const found = region(r);
        if (found === null) throw keyError(J.pyStr(r));
        const [k, human, kind] = found;
        return J.object([
          ["region", J.str(k)],
          ["human", J.str(human)],
          ["kind", J.str(kind)],
          ["side", J.field(sc, "side")],
          ["derived", J.bool(false)],
        ]);
      });
    } else {
      regions = regionsUnder(siteKey);
    }
    out.push(
      J.object([
        ["slot", key],
        ["role", item(s, "role")],
        ["name", item(s, "name")],
        ["human", item(s, "human")],
        ["scalp", sc],
        ["over", J.array(regions)],
      ]),
    );
  }
  return J.object([
    ["layout", item(lay, "key")],
    ["preset", get(p, "preset")],
    ["rotation_deg", get(p, "rotation_deg")],
    ["note", getOr(p, "note", J.str(""))],
    ["bias_driven", getOr(p, "bias_driven", J.bool(true))],
    ["slots", J.array(out)],
  ]);
}

/** The resolved slots, by slot key. */
function bySlot(r: Value): [Value, Value][] {
  const s = J.get(r, "slots");
  return s !== undefined && s.kind === "array" ? s.items.map((x) => [J.field(x, "slot"), x]) : [];
}

function firstWithRole(slots: Value[], role: string): Value | null {
  for (const s of slots) if (roleIs(s, role)) return s;
  return null;
}

/** `sorted(channels, key=lambda c: c["index"])`, stable. */
function byIndex(chans: Value[]): Value[] {
  const keyed = chans.map((c): [number, Value] => {
    const i = item(c, "index");
    const k = J.isNumber(i) ? J.asF64(i) : null;
    if (k === null) throw new Fail("other", "TypeError: an index is not a number");
    return [k, c];
  });
  keyed.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  return keyed.map(([, c]) => c);
}

/** The placement as a montage, so everything downstream works. */
export function toMontage(lay: Value, p: Value): Value {
  const r = resolve(lay, p);
  const placed = bySlot(r);
  const site = (key: Value): Value => {
    const s = slotOf(placed, key);
    if (s === null) throw keyError(J.pyStr(key));
    const sc = item(s, "scalp");
    return M.site(item(s, "name"), item(sc, "ten_ten"), item(sc, "human"));
  };
  const slots = list(item(lay, "slots"));
  const refSlot = firstWithRole(slots, REFERENCE);
  const chans: Value[] = [];
  for (const c of byIndex(list(item(lay, "channels")))) {
    chans.push(
      M.channel(item(c, "index"), item(c, "derivation"), site(item(c, "positive")), site(item(c, "negative")), get(c, "label")),
    );
  }
  const biasSlot = firstWithRole(slots, BIAS);
  const bias =
    biasSlot !== null && slotOf(placed, item(biasSlot, "key")) !== null && J.truthy(J.field(r, "bias_driven"))
      ? site(item(biasSlot, "key"))
      : J.NULL;
  const e = get(lay, "electrode");
  const electrode = J.truthy(e) ? e : J.str("unknown");
  const reference = refSlot !== null ? site(item(refSlot, "key")) : J.NULL;
  return M.build(chans, electrode, bias, reference, J.str(describe(lay, p, false)), J.NULL);
}

/** The placement as a sentence, regenerated from the structure. */
export function describe(lay: Value, p: Value, extras: boolean): string {
  const r = resolve(lay, p);
  const placed = bySlot(r);
  const resolved = list(J.field(r, "slots"));
  const roleOf = (s: Value, role: string): boolean => {
    const x = J.field(s, "role");
    return x.kind === "str" && x.value === role;
  };
  const rec = resolved.filter((s) => roleOf(s, RECORDING));
  const reference = resolved.find((s) => roleOf(s, REFERENCE));
  const spots = new Set(rec.map((s) => J.pyStr(J.field(J.field(s, "scalp"), "key"))));
  const lastPart = (s: Value): string => {
    const h = J.pyStr(J.field(s, "human")).split(", ");
    return h[h.length - 1]!;
  };
  const parts: string[] = [];
  const chans = list(item(lay, "channels"));
  if (rec.length > 1 && spots.size === 1) {
    parts.push(
      `${chans.length} channels in one cluster at ${J.pyStr(J.field(J.field(rec[0]!, "scalp"), "human"))}: ` +
        rec.map(lastPart).join(", "),
    );
    if (reference !== undefined) parts.push(`referenced to the ${lastPart(reference)}`);
  } else {
    for (const c of byIndex(chans)) {
      const human = (pole: string): string => {
        const s = slotOf(placed, item(c, pole));
        if (s === null) throw keyError(J.pyStr(item(c, pole)));
        return J.pyStr(J.field(J.field(s, "scalp"), "human"));
      };
      const a = human("positive");
      const b = human("negative");
      const at = J.pyStr(item(c, "index"));
      const d = item(c, "derivation");
      parts.push(d.kind === "str" && d.value === M.REFERENTIAL ? `ch${at} = ${a}, referenced to ${b}` : `ch${at} = ${a} / ${b}`);
    }
  }
  const slots = list(item(lay, "slots"));
  const b = firstWithRole(slots, BIAS);
  if (b !== null) {
    const s = slotOf(placed, item(b, "key"));
    const where = s !== null && J.truthy(J.field(r, "bias_driven")) ? J.pyStr(J.field(J.field(s, "scalp"), "human")) : "not driven";
    parts.push(`bias = ${where}`);
  }
  let tail = "";
  if (extras) {
    const rotation = J.field(r, "rotation_deg");
    if (J.truthy(rotation)) tail += ` Rotated ${general(J.asF64(rotation) ?? NaN, 6)}°.`;
    const note = J.field(r, "note");
    if (J.truthy(note)) tail += ` ${J.pyStr(note)}`;
  }
  if (parts.length === 0) throw new Fail("other", "IndexError: list index out of range");
  const said = [parts[0]!];
  for (const part of parts.slice(1)) {
    const head = Array.from(part).slice(0, 14).join("");
    if (head.includes(" = ")) {
      said.push(part);
    } else {
      const chars = Array.from(part);
      said.push(chars.length === 0 ? "" : chars[0]!.toUpperCase() + chars.slice(1).join(""));
    }
  }
  return said.join(". ") + "." + tail;
}
