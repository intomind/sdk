/**
 * The analysis fixture: every analysis this package runs, held to what the
 * Python library reported for the same synthetic recordings.
 *
 * `contract/analysis/make.py` emits the fixture from the Python library on
 * scipy and numpy, whose versions it records. Every key must be there in the
 * library's order, every integer, boolean, string and null the same, every
 * verdict and criterion the same, and every refusal in the same words, a
 * value it shows written as this package writes it (`words.ts`). A
 * float the library computed through scipy's filters, transforms and tests
 * must be within the tolerance stated here of the library's: the arithmetic
 * is the same, the order of its floating point operations is not. The
 * largest error each output reached is checked against that tolerance, and
 * printed when `ANALYSIS_ERRORS` is set.
 */

import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import test from "node:test";

import { ANALYSES, compare, events, inputs, listCaptures, type Param, type Params } from "../src/analysis.ts";
import * as N from "../src/numeric.ts";
import { selectRefusal } from "../src/settings.ts";
import * as S from "../src/stats.ts";
import { CaptureError } from "../src/capture/errors.ts";
import { Collection } from "../src/capture/recording.ts";
import * as npy from "../src/capture/npy.ts";
import * as J from "../src/capture/pyjson.ts";
import * as zip from "../src/capture/zip.ts";
import { Random } from "../src/pyrandom.ts";
import type { Value } from "../src/capture/pyjson.ts";
import { valuesInJavaScript } from "./words.ts";

const ROOT = new URL("../../contract/analysis/", import.meta.url);
const CAPTURES = new URL("captures/", ROOT);
const fixture = J.loads(readFileSync(new URL("analysis.json", ROOT), "utf8"));

function recordings(): Collection {
  const files = new Map<string, Uint8Array>();
  for (const name of readdirSync(CAPTURES)) files.set(name, new Uint8Array(readFileSync(new URL(name, CAPTURES))));
  return new Collection(files, { dir: "captures" });
}

function items(v: Value | undefined): Value[] {
  assert.ok(v !== undefined && v.kind === "array", "a list in the fixture");
  return v.items;
}

function text(v: Value): string {
  assert.equal(v.kind, "str");
  return (v as { value: string }).value;
}

function params(v: Value): Record<string, Value> {
  assert.equal(v.kind, "object");
  return Object.fromEntries((v as { entries: [string, Value][] }).entries);
}

/**
 * A float's error against the library's, relative: to its own size, or, in
 * a list of numbers such as a spectrum, to the largest in the list, since a
 * spectrum's smallest values sit at the level of rounding and their own
 * digits mean nothing. Exactly zero where the library has exactly zero.
 */
const relative = (a: number, b: number, scale: number): number => Math.abs(a - b) / Math.max(scale, Number.MIN_VALUE);

/**
 * The largest error allowed. The largest measured on this fixture is 4e-12,
 * on the median power of a flat channel's filtered spectrum, a value near
 * 1e-36 made of rounding alone, and the next 3e-13, on Cohen's d, a ratio of
 * a mean difference to a small standard deviation that magnifies rounding.
 * Most numbers are within 1e-15, the filtered signals, the frequencies and
 * many statistics are identical, and the normal and Student tails are
 * within 2e-14 of scipy's across their range.
 */
const TOLERANCE = 1e-11;

const worst = new Map<string, { err: number; at: string }>();

function note(key: string, err: number, at: string): void {
  const w = worst.get(key);
  if (w === undefined || err > w.err) worst.set(key, { err, at });
}

function same(mine: Value, theirs: Value, at: string, key: string, scale: number | null): void {
  assert.equal(mine.kind, theirs.kind, `${at}: ${J.pyRepr(mine)} against ${J.pyRepr(theirs)}`);
  switch (theirs.kind) {
    case "object": {
      const m = (mine as { entries: [string, Value][] }).entries;
      const t = theirs.entries;
      assert.deepEqual(m.map(([k]) => k), t.map(([k]) => k), `${at}: the keys`);
      m.forEach(([k, v], i) => same(v, t[i]![1], `${at}/${k}`, k, null));
      return;
    }
    case "array": {
      const m = (mine as { items: Value[] }).items;
      assert.equal(m.length, theirs.items.length, `${at}: the length`);
      let big = 0;
      for (const v of theirs.items) if (v.kind === "float" && Number.isFinite(v.value)) big = Math.max(big, Math.abs(v.value));
      m.forEach((v, i) => same(v, theirs.items[i]!, `${at}/${i}`, key, big > 0 ? big : null));
      return;
    }
    case "float": {
      const a = (mine as { value: number }).value;
      const b = theirs.value;
      if (!Number.isFinite(b)) {
        assert.ok(Object.is(a, b) || (Number.isNaN(a) && Number.isNaN(b)), `${at}: ${a} against ${b}`);
        return;
      }
      assert.ok(Number.isFinite(a), `${at}: ${a} against ${b}`);
      const err = relative(a, b, scale ?? Math.abs(b));
      note(key, err, at);
      assert.ok(err <= TOLERANCE, `${at}: ${a} against ${b}, off by ${err.toExponential(2)}`);
      return;
    }
    default:
      assert.equal(J.dumps(mine), J.dumps(theirs), at);
  }
}

test("the registry is the library's: names, words, whether cues are needed, and the parameters a form offers", () => {
  const theirs = items(J.get(fixture, "registry"));
  assert.deepEqual([...ANALYSES.keys()], theirs.map((s) => text(J.field(s, "name"))));
  for (const s of theirs) {
    const name = text(J.field(s, "name"));
    const mine = ANALYSES.get(name)!;
    assert.equal(mine.doc, text(J.field(s, "doc")), name);
    assert.equal(mine.needsEvents, J.truthy(J.field(s, "needs_events")), name);
    assert.equal(JSON.stringify(mine.params), JSON.stringify(J.toPlain(J.field(s, "params"))), name);
  }
});

test("every analysis reports what the library reported, key for key, or refuses in its words", () => {
  const rec = recordings();
  const cases = items(J.get(fixture, "cases"));
  assert.ok(cases.length >= 95, String(cases.length));
  let refused = 0;
  for (const c of cases) {
    const name = text(J.field(c, "analysis"));
    const labels = items(J.get(c, "labels")).map(text);
    const p = params(J.field(c, "params"));
    const at = `${name} ${labels.join(",")} ${J.dumps(J.field(c, "params"))}`;
    const run = (): Value => (name === "psd" && labels.length > 1 ? compare(rec, labels, p) : ANALYSES.get(name)!.fn(rec, labels[0]!, p));
    const refusal = J.get(c, "refused");
    if (refusal !== undefined) {
      refused++;
      assert.throws(run, (e: unknown) => {
        assert.ok(e instanceof CaptureError, `${at}: ${String(e)}`);
        assert.equal(e.message, valuesInJavaScript(text(J.field(refusal, "message"))), at);
        return true;
      });
      continue;
    }
    same(run(), J.field(c, "result"), at, name, null);
  }
  assert.ok(refused >= 40, String(refused));
  if (process.env.ANALYSIS_ERRORS) {
    const rows = [...worst].sort((x, y) => y[1].err - x[1].err);
    for (const [key, w] of rows) console.log(`${key.padEnd(24)} ${w.err.toExponential(2)}  ${w.at}`);
  }
});

test("a setting no filter can be made from is refused in the library's words, naming null, NaN and the infinities as JavaScript writes them", () => {
  // The library's words for the same settings, each naming None, nan or
  // inf as Python writes them, and a list as Python's repr writes it.
  const rec = recordings();
  const rule = (kind: string, most: string): string =>
    `a ${kind} frequency is above 0 Hz and ${most} 124.975 Hz, half the recording's rate of 249.95 Hz, or 0 for no ${kind}.`;
  const sharp = "notch_q, the notch frequency over the notch's width at half power, is a finite number above 0.";
  const cases: [Params, string][] = [
    [{ notch_q: null }, `rest-4ch cannot be filtered with a notch of 60 Hz whose notch_q is None: ${sharp}`],
    [{ notch_q: NaN }, `rest-4ch cannot be filtered with a notch of 60 Hz whose notch_q is nan: ${sharp}`],
    [{ hp: NaN }, `rest-4ch cannot be filtered with a high-pass of nan: ${rule("high-pass", "below")}`],
    [{ lp: Infinity }, `rest-4ch cannot be filtered with a low-pass of inf: ${rule("low-pass", "below")}`],
    [{ notch: -Infinity }, `rest-4ch cannot be filtered with a notch of -inf: ${rule("notch", "at most")}`],
    [{ hp: [1] }, `rest-4ch cannot be filtered with a high-pass of [1]: ${rule("high-pass", "below")}`],
  ];
  for (const [given, python] of cases) {
    const words = valuesInJavaScript(python);
    assert.throws(() => ANALYSES.get("psd")!.fn(rec, "rest-4ch", { filtered: 1, ...given }), (e: unknown) => {
      assert.ok(e instanceof CaptureError, String(e));
      assert.equal(e.message, words);
      return true;
    });
  }
  assert.equal(
    valuesInJavaScript(`rest-4ch cannot be filtered with a notch of -inf: ${rule("notch", "at most")}`),
    `rest-4ch cannot be filtered with a notch of -Infinity: ${rule("notch", "at most")}`,
  );
});

test("a setting outside the range the registry gives it is refused in the library's words, naming null, NaN, the infinities, a list and an object as JavaScript writes them", () => {
  // The library's words for the same settings, each naming None, nan, inf,
  // a list or a dict as Python writes them.
  const rec = recordings();
  const cases: [string, string, Params, string][] = [
    ["summary", "rest-4ch", { line_hz: null },
      "rest-4ch cannot be analyzed with a line_hz of None: line_hz, the mains frequency whose line is reported, is a finite number of at least 0 Hz."],
    ["summary", "rest-4ch", { slope_hi: NaN },
      "rest-4ch cannot be analyzed with a slope_hi of nan: slope_hi, the highest frequency of the slope fit, is a finite number above 0 Hz."],
    ["summary", "rest-4ch", { nperseg: Infinity },
      "rest-4ch cannot be analyzed with an nperseg of inf: nperseg, the samples in each segment of the spectrum, is a finite number of at least 2."],
    ["psd", "rest-4ch", { fmax: Infinity },
      "rest-4ch cannot be analyzed with an fmax of inf: fmax, the highest frequency of the curves, is a finite number above 0 Hz."],
    ["psd", "rest-4ch", { filtered: null },
      "rest-4ch cannot be analyzed with filtered set to None: filtered, whether the spectrum is of the filtered signal, is 0 or 1."],
    ["saccade", "saccade-pass", { pre_s: NaN },
      "saccade-pass cannot be analyzed with a pre_s of nan: pre_s, the time each trial starts before its cue, is a finite number of seconds that holds at least 1 sample, 0.00401 s at the recording's rate of 249.95 Hz."],
    ["jaw", "jaw-pass", { alpha: -Infinity },
      "jaw-pass cannot be analyzed with an alpha of -inf: alpha, the p-value below which a channel counts as significant, is a finite number from 0 to 1."],
    ["jaw", "jaw-pass", { window_s: Infinity },
      "jaw-pass cannot be analyzed with a window_s of inf: window_s, the length of each rest window and each clench window, is a finite number of seconds that holds at least 32 samples, 0.129 s at the recording's rate of 249.95 Hz, the fewest a window's power is taken from."],
    ["berger", "berger-blocks", { settle: NaN },
      "berger-blocks cannot be analyzed with a settle of nan: settle, the time left out at the start of each block, is at least 0 s and below 10 s, the recording's block length."],
    // A list and an object as JSON writes them, where the library writes Python's repr.
    ["blink", "blink-pass", { min_pp_uv: [1] },
      "blink-pass cannot be analyzed with a min_pp_uv of [1]: min_pp_uv, the smallest peak to peak that counts as a blink, is a finite number of at least 0 µV."],
    ["blink", "blink-pass", { min_pp_uv: { a: 1 } },
      "blink-pass cannot be analyzed with a min_pp_uv of {'a': 1}: min_pp_uv, the smallest peak to peak that counts as a blink, is a finite number of at least 0 µV."],
  ];
  for (const [name, label, given, python] of cases) {
    const words = valuesInJavaScript(python);
    assert.throws(() => ANALYSES.get(name)!.fn(rec, label, given), (e: unknown) => {
      assert.ok(e instanceof CaptureError, String(e));
      assert.equal(e.message, words);
      return true;
    });
  }
  // A truth value is a number to the library, 1 or 0.
  assert.doesNotThrow(() => ANALYSES.get("psd")!.fn(rec, "rest-4ch", { filtered: true }));
  assert.throws(() => ANALYSES.get("psd")!.fn(rec, "rest-4ch", { fmax: false }), {
    message: "rest-4ch cannot be analyzed with an fmax of 0 Hz: fmax, the highest frequency of the curves, is a finite number above 0 Hz.",
  });
  assert.throws(() => ANALYSES.get("blink")!.fn(rec, "blink-pass", { min_pp_uv: { a: 1 } }), {
    message: 'blink-pass cannot be analyzed with a min_pp_uv of {"a": 1}: min_pp_uv, the smallest peak to peak that counts as a blink, is a finite number of at least 0 µV.',
  });
});

test("the collection lists as the library lists it: manifests, cues, and whether each recording is intact", () => {
  const listed = J.field(fixture, "list_captures");
  const order = items(J.get(listed, "labels")).map(text);
  same(listCaptures(recordings(), order), J.field(listed, "result"), "list_captures", "list", null);
});

test("an analysis registered with several is handed every selected recording at once, and every one of the package's takes one label", () => {
  const rec = recordings();
  const theirs = J.field(fixture, "several");
  const registered = J.field(theirs, "registered");
  const handed: string[][] = [];
  const lengths = (recordings: Collection, labels: string | readonly string[], p: Params = {}): Value => {
    const scale = p.scale === undefined ? 1 : J.asF64(J.fromPlain(p.scale))!;
    const list = [...(labels as readonly string[])];
    handed.push(list);
    return J.object([
      ["analysis", J.str("lengths")],
      ["labels", J.array(list.map((l) => J.str(l)))],
      ["samples", J.array(list.map((l) => J.int(Math.trunc(scale * recordings.load(l).samples))))],
    ]);
  };
  assert.ok(J.truthy(J.field(registered, "several")));
  ANALYSES.set("lengths", {
    fn: lengths,
    needsEvents: false,
    several: true,
    doc: text(J.field(registered, "doc")),
    params: J.toPlain(J.field(registered, "params")) as Param[],
  });
  try {
    for (const c of items(J.get(theirs, "cases"))) {
      const name = text(J.field(c, "analysis"));
      const labels = items(J.get(c, "labels")).map(text);
      const p = params(J.field(c, "params"));
      const spec = ANALYSES.get(name)!;
      // As an application runs it, as the registry says: once, handed every
      // label in a list, or on each label separately.
      const results = spec.several ? [spec.fn(rec, labels, p)] : labels.map((l) => spec.fn(rec, l, p));
      same(J.array(results), J.field(c, "results"), `${name} ${labels.join(",")}`, name, null);
    }
  } finally {
    ANALYSES.delete("lengths");
  }
  assert.deepEqual(handed, J.toPlain(J.field(theirs, "handed")));
  assert.deepEqual([...ANALYSES.keys()], J.toPlain(J.field(theirs, "takes_one")));
  for (const [name, spec] of ANALYSES) assert.ok(!("several" in spec), `${name} takes one label`);
});

test("a select is one of its choices as the library holds one, whether they are numbers or text, and any other value is refused in its words", () => {
  // The library's own check, asked for an entry registered with it whose
  // choices are text, and text and numbers, and for its own selects of 0 and
  // 1. Each setting given is held in the order its entry lists them.
  const theirs = J.field(fixture, "selects");
  const label = text(J.field(theirs, "label"));
  const registered = J.field(theirs, "registered");
  const means = J.toPlain(J.field(theirs, "means")) as Record<string, string>;
  const entries = new Map<string, readonly Param[]>([[text(J.field(registered, "name")), J.toPlain(J.field(registered, "params")) as Param[]]]);
  for (const [name, spec] of ANALYSES) entries.set(name, spec.params);
  let refused = 0;
  for (const c of items(J.get(theirs, "cases"))) {
    const given = params(J.field(c, "given"));
    let mine: string | null = null;
    for (const p of entries.get(text(J.field(c, "analysis")))!) {
      if (mine !== null || !Object.hasOwn(given, p.name)) continue;
      assert.equal(p.type, "select");
      mine = selectRefusal(label, p.name, means[p.name]!, p.choices!, given[p.name]!);
    }
    const want = J.field(c, "refused");
    if (want.kind !== "null") refused++;
    assert.equal(mine, want.kind === "null" ? null : valuesInJavaScript(text(J.field(want, "message"))), J.dumps(J.field(c, "given")));
  }
  assert.ok(refused >= 30 && items(J.get(theirs, "cases")).length - refused >= 20);
});

/**
 * The files of a recording of 40 samples whose every electrode reads off
 * where lead-off detection was on, as on firmware that cannot check contact,
 * with the manifest `extra` gives and no `leadoff_judged` unless `judged` is
 * given, as the library's own test makes one.
 */
function contactRecording(files: Map<string, Uint8Array>, label: string, extra: Record<string, unknown>, active: boolean[], judged: boolean[] | null = null, withActive = true): void {
  const n = 40;
  const vector = (kind: "i8" | "f8" | "u1" | "b1", values: number[] | bigint[]): npy.NpyArray =>
    npy.vector(
      kind === "i8"
        ? { kind, values: BigInt64Array.from(values as bigint[]) }
        : kind === "f8"
          ? { kind, values: Float64Array.from(values as number[]) }
          : { kind, values: Uint8Array.from(values as number[]) },
    );
  const ticks = Array.from({ length: n }, (_, k) => BigInt(k * 2000));
  const members: [string, npy.NpyArray][] = [
    ["counts", { shape: [2, n], fortran: false, data: { kind: "i8", values: new BigInt64Array(2 * n).fill(1000n) } }],
    ["index", vector("i8", Array.from({ length: n }, (_, k) => BigInt(k)))],
    ["device_ticks", vector("i8", ticks)],
    ["host_time", vector("f8", ticks.map((t) => Number(t) / 1e6))],
    ["leadoff", vector("u1", active.map((a) => (a ? 0x03 : 0x00)))],
    ["gap_before", vector("b1", new Array<number>(n).fill(0))],
  ];
  if (withActive) members.push(["leadoff_active", vector("b1", active.map(Number))]);
  if (judged !== null) members.push(["leadoff_judged", vector("b1", judged.map(Number))]);
  files.set(`${label}.npz`, zip.write(members.map(([k, a]) => [`${k}.npy`, npy.write(a)]), 1_767_225_600));
  const meta = { label, n, fs_effective: 500.0, vref_uv: 4_500_000.0, gain: 24, adc_bits: 24, tick_hz: 1_000_000, manifest_version: 7, ...extra };
  files.set(`${label}.meta.json`, new TextEncoder().encode(JSON.stringify(meta)));
}

test("a recording of firmware that cannot check contact has no lead-off verdict, read as the library reads it", () => {
  // An IntoMind device whose firmware is older than 1.4.6 cannot check
  // electrode contact: its lead-off detection does not work, and every
  // electrode reads off whether it touches skin or not. A recording of one
  // that holds no leadoff_judged loads with every sample taken with
  // detection on not judged and the rest judged, says so in its repairs, and
  // has no lead-off figures in the contact analysis. Loaded as recorded, it
  // is as it is on disk. The firmware is read from firmware_version, and
  // from fw in an older manifest.
  const files = new Map<string, Uint8Array>();
  const every = new Array<boolean>(40).fill(true);
  const half = Array.from({ length: 40 }, (_, k) => k >= 20);
  const old: [string, Record<string, unknown>, boolean[], string][] = [
    ["t_fw_old", { firmware_version: "1.4.5" }, every, "1.4.5"],
    ["t_fw_old_legacy", { fw: "1.2.0" }, every, "1.2.0"],
    ["t_fw_old_half", { firmware_version: "1.4.5" }, half, "1.4.5"],
    ["t_fw_old_spaced", { firmware_version: " 01.04.05 " }, every, "1.4.5"],
  ];
  for (const [label, extra, active] of old) contactRecording(files, label, extra, active);
  const kept: [string, Record<string, unknown>, boolean[] | null][] = [
    ["t_fw_new", { firmware_version: "1.4.6" }, null],
    ["t_fw_unknown", { firmware_version: "test" }, null],
    ["t_fw_two_parts", { firmware_version: "1.4" }, null],
    ["t_fw_old_marked", { firmware_version: "1.4.5" }, Array.from({ length: 40 }, (_, k) => k < 5)],
  ];
  for (const [label, extra, judged] of kept) contactRecording(files, label, extra, every, judged);
  contactRecording(files, "t_fw_old_no_flag", { firmware_version: "1.4.5" }, every, null, false);
  const rec = new Collection(files);
  const contact = ANALYSES.get("contact")!.fn;
  for (const [label, , active, firmware] of old) {
    const cap = rec.load(label);
    assert.deepEqual(cap.leadoffJudged, active.map((a) => !a), label);
    const on = active.filter((a) => a).length;
    assert.deepEqual(cap.repairs, [
      `the device ran firmware ${firmware}, which cannot check electrode contact: its lead-off detection does not ` +
        "work, and every electrode reads off whether it touches skin or not. The lead-off bits of the " +
        `${on} samples taken with detection on are read as no verdict on contact.`,
    ], label);
    assert.equal(rec.loadAsRecorded(label).leadoffJudged, null, label);
    const got = J.toPlain(contact(rec, label)) as Record<string, unknown>;
    // No sample's contact was checked: detection was off for half of
    // t_fw_old_half, and the rest are no verdict.
    assert.deepEqual([got.leadoff_bytes, got.leadoff_nonzero_frac], [[], null], label);
    assert.deepEqual((got.channels as Record<string, unknown>[]).map((c) => c.contact), ["not checked", "not checked"], label);
  }
  for (const [label, , judged] of kept) {
    const cap = rec.load(label);
    assert.deepEqual(cap.repairs, [], label);
    if (judged === null) {
      assert.equal(cap.leadoffJudged, null, label);
      const got = J.toPlain(contact(rec, label)) as Record<string, unknown>;
      assert.deepEqual([got.leadoff_bytes, got.leadoff_nonzero_frac], [[3], 1], label);
    } else {
      assert.deepEqual(cap.leadoffJudged, judged, label);
    }
  }
  const unsaid = rec.load("t_fw_old_no_flag");
  assert.deepEqual([unsaid.leadoffActive, unsaid.leadoffJudged, unsaid.repairs], [null, null, []]);
});

/**
 * The files of a recording of two channels holding `x` in counts, at 500
 * samples a second, its sample index `index` or none missing, with the cues
 * `events` beside it and the manifest `extra` gives, as the library's own
 * analysis tests write one.
 */
function cueRecording(files: Map<string, Uint8Array>, label: string, x: readonly number[], events: readonly Record<string, unknown>[], index: readonly number[] | null = null, extra: Record<string, unknown> = {}): void {
  const n = x.length;
  const idx = index ?? Array.from({ length: n }, (_, k) => k);
  const vector = (data: npy.NpyArray["data"]): npy.NpyArray => npy.vector(data);
  const counts = new BigInt64Array(2 * n);
  for (let k = 0; k < n; k++) counts[k] = counts[n + k] = BigInt(Math.round(x[k]!));
  const members: [string, npy.NpyArray][] = [
    ["counts", { shape: [2, n], fortran: false, data: { kind: "i8", values: counts } }],
    ["index", vector({ kind: "i8", values: BigInt64Array.from(idx, BigInt) })],
    ["device_ticks", vector({ kind: "i8", values: BigInt64Array.from(idx, BigInt) })],
    ["host_time", vector({ kind: "f8", values: Float64Array.from(idx, (i) => i / 500) })],
    ["leadoff", vector({ kind: "u1", values: new Uint8Array(n) })],
    ["gap_before", vector({ kind: "b1", values: new Uint8Array(n) })],
  ];
  files.set(`${label}.npz`, zip.write(members.map(([k, a]) => [`${k}.npy`, npy.write(a)]), 1_767_225_600));
  const meta = { label, n, fs_effective: 500.0, vref_uv: 4_500_000.0, gain: 24, adc_bits: 24, ...extra };
  files.set(`${label}.meta.json`, new TextEncoder().encode(JSON.stringify(meta)));
  files.set(`${label}.events.json`, new TextEncoder().encode(JSON.stringify(events)));
}

/** A signal of `n` samples, in counts, that is not flat. */
const wavy = (n: number): number[] => Array.from({ length: n }, (_, k) => Math.round(4000 * Math.sin(k * 0.37) + 1500 * Math.sin(k * 1.3)));

test("saccade and blink filter the whole recording in one pass, the samples on each side of a break taken as next to each other, as the library does", () => {
  // In a recording that lost 50 samples, the signal carrying on across them
  // as it does across lost packets, each condition's numbers are those of its
  // trials taken from all the recording's samples filtered at once. They are
  // not those of the same trials taken from each stretch without a break
  // filtered on its own, which the filters' response to the ends of a stretch
  // changes. Every trial lies more than the high-pass's settle time from the
  // break, so none is left out.
  const n = 30050;
  const cut = 15000;
  const lost = 50;
  const pre = 250;
  const post = 750;
  // Each trial's first sample, once the lost samples are gone: three trials
  // before the break and three after it, the nearest 5 s away.
  const first = [2000, 6000, 10000, 17500, 21500, 25500];
  const cues = first.map((f) => f + pre);
  const keep = Array.from({ length: n - lost }, (_, k) => (k < cut ? k : k + lost));
  // The filters as the analyses run them by default, at the recording's rate.
  const [nb, na] = N.iirnotch(60.0, 30.0, 500);
  const filtered = (y: Float64Array): Float64Array =>
    N.sosfiltfilt(N.butter4(10.0, "lowpass", 500), N.filtfilt(nb, na, N.sosfiltfilt(N.butter4(0.1, "highpass", 500), y)));
  const random = new Random(63);
  for (const analysis of ["saccade", "blink"] as const) {
    const cond = analysis === "saccade" ? "left" : "blink";
    // Noise of about 8 µV that wanders slowly, in counts: a microvolt is
    // about 45 of them at this gain and reference.
    let drift = 0;
    const x = Array.from({ length: n }, () => (drift = 0.995 * drift + 125 * (random.getrandbits(32) / 2 ** 32 - 0.5)));
    for (const c of cues) {
      for (let k = 0; k < post; k++) {
        const r = analysis === "saccade" ? 120 * (1 - Math.exp(-k / (0.05 * 500))) : 150 * Math.sin((Math.PI * Math.min(Math.max(k - 0.2 * 500, 0), 0.25 * 500)) / (0.25 * 500));
        x[keep[c]! + k] = x[keep[c]! + k]! + Math.round(45 * r);
      }
    }
    const label = `t_${analysis}_one_pass`;
    const files = new Map<string, Uint8Array>();
    cueRecording(files, label, keep.map((p) => x[p]!), cues.map((c) => ({ cond, kind: "cue", host_time: keep[c]! / 500 })), keep);
    const rec = new Collection(files);
    const result = ANALYSES.get(analysis)!.fn(rec, label, {});
    const got = J.toPlain(J.field(result, "conditions")) as Record<string, { n_trials: number; channels: Record<string, number>[] }>;
    assert.equal(got[cond]!.n_trials, 6, `${analysis}: ${JSON.stringify(got)}`);
    const cap = rec.load(label);
    const whole = [0, 1].map((c) => filtered(cap.uv(c)));
    const apart = [0, 1].map((c) => {
      const y = cap.uv(c);
      const a = filtered(y.slice(0, cut));
      const b = filtered(y.slice(cut));
      const out = new Float64Array(a.length + b.length);
      out.set(a);
      out.set(b, a.length);
      return out;
    });
    const number = (y: Float64Array): number => {
      const E = cues.map((c) => {
        const seg = y.slice(c - pre, c + post);
        const m = N.mean(seg.subarray(0, pre));
        return seg.map((v) => v - m);
      });
      if (analysis === "blink") return N.mean(Float64Array.from(E, (e) => N.ptp(e.subarray(pre, pre + 400))));
      const mean = new Float64Array(pre + post);
      for (const e of E) for (let k = 0; k < mean.length; k++) mean[k] = mean[k]! + e[k]!;
      const w = mean.map((v) => v / E.length).slice(pre, pre + 500);
      return w[N.argmax(w.map(Math.abs))]!;
    };
    for (const c of [0, 1]) {
      const said = got[cond]!.channels[c]![analysis === "blink" ? "mean_pp_uv" : "deflection_uv"]!;
      assert.equal(said, number(whole[c]!), `${analysis} ${c}`);
      assert.notEqual(said, number(apart[c]!), `${analysis} ${c}`);
    }
    assert.ok(J.asI64(J.field(result, "analysis_version"))! >= 15n);
  }
});

test("a trial in a short stretch between two breaks is not refused, as the library has it", () => {
  // The filters run over the whole recording, so a stretch without a break
  // too short to be filtered on its own is no reason to refuse. With the
  // high-pass off, when no trial is left out for a settle time, a trial of
  // two samples inside a stretch of 10 between two breaks is analyzed. With
  // the default high-pass it lies within the settle time of a break and is
  // left out: its condition is not among the conditions, the result counts
  // the one left out, and there is no criterion and no verdict.
  const long1 = 500;
  const short = 10;
  const long2 = 500;
  const index = [
    ...Array.from({ length: long1 }, (_, k) => k),
    ...Array.from({ length: short }, (_, k) => long1 + 5 + k),
    ...Array.from({ length: long2 }, (_, k) => long1 + short + 10 + k),
  ];
  const x = wavy(index.length);
  const files = new Map<string, Uint8Array>();
  for (const [analysis, cond] of [["saccade", "left"], ["blink", "blink"]] as const) {
    // The trial's samples are the fifth and sixth of the short stretch.
    cueRecording(files, `t_${analysis}_short`, x, [{ cond, kind: "cue", host_time: index[long1 + 5]! / 500 }], index);
  }
  const rec = new Collection(files);
  for (const [analysis, cond] of [["saccade", "left"], ["blink", "blink"]] as const) {
    const fn = ANALYSES.get(analysis)!.fn;
    const got = J.toPlain(fn(rec, `t_${analysis}_short`, { pre_s: 1 / 500, post_s: 1 / 500, hp: 0 })) as { conditions: Record<string, { n_trials: number }> };
    assert.equal(got.conditions[cond]!.n_trials, 1, analysis);
    const left = fn(rec, `t_${analysis}_short`, { pre_s: 1 / 500, post_s: 1 / 500 });
    assert.ok(!Object.hasOwn(J.toPlain(J.field(left, "conditions")!) as object, cond), analysis);
    assert.deepEqual(J.toPlain(J.field(left, "left_out")!), { [cond]: leftOut({ gap_nearby: 1 }) }, analysis);
    assert.equal(J.dumps(J.field(left, "criteria")), "{}", analysis);
    assert.equal(J.dumps(J.field(left, "verdict")), "null", analysis);
  }
});

/** What `left_out` reads with the counts given and every other 0. */
function leftOut(counts: { gap_inside?: number; gap_nearby?: number; outside_recording?: number } = {}): { gap_inside: number; gap_nearby: number; outside_recording: number } {
  return { gap_inside: counts.gap_inside ?? 0, gap_nearby: counts.gap_nearby ?? 0, outside_recording: counts.outside_recording ?? 0 };
}

/** Where the six trials of `settleRecording` lie: three stretches of 5,000, 7,500 and 5,000 samples, each trial 0.5 s before its cue and 1.5 s after. */
const SETTLE_STRETCHES = [5000, 7500, 5000] as const;
const SETTLE_PRE = 250;
const SETTLE_POST = 750;

/**
 * A recording of three stretches with a break of 50 samples between each
 * two, or of the same samples without a break, and six trials of
 * `analysis`'s first condition, given by their first samples: one at the
 * recording's first sample, one ending `inside` samples before the first
 * break, one beginning `outside` samples after it, one ending `outside`
 * samples before the second break, one beginning `inside` samples after it,
 * and one ending at the last sample a trial can. Each trial's response, a
 * step for saccade and a bump for blink, rises after its cue. `kept`, where
 * given, are the trials cued, by number, and `more` is further cues with no
 * response, each a condition and the position of its cue. Gives the cue
 * positions of all six.
 */
function settleRecording(files: Map<string, Uint8Array>, label: string, analysis: "saccade" | "blink", inside: number, outside: number, broken = true, kept: readonly number[] | null = null, more: readonly (readonly [string, number])[] = []): number[] {
  const [l1, l2, l3] = SETTLE_STRETCHES;
  const n = l1 + l2 + l3;
  const w = SETTLE_PRE + SETTLE_POST;
  const first = [0, l1 - inside - w, l1 + outside, l1 + l2 - outside - w, l1 + l2 + inside, n - w];
  const cues = first.map((p) => p + SETTLE_PRE);
  const cond = analysis === "saccade" ? "left" : "blink";
  // In counts: a microvolt is about 45 of them at this gain and reference.
  const x = wavy(n);
  for (const p of cues) {
    for (let k = 0; k < SETTLE_POST; k++) {
      const r = analysis === "saccade" ? 120 * (1 - Math.exp(-k / (0.05 * 500))) : 150 * Math.sin((Math.PI * Math.min(Math.max(k - 0.2 * 500, 0), 0.25 * 500)) / (0.25 * 500));
      x[p + k] = x[p + k]! + Math.round(45 * r);
    }
  }
  const index = Array.from({ length: n }, (_, k) => k + (broken ? (k >= l1 ? 50 : 0) + (k >= l1 + l2 ? 50 : 0) : 0));
  const cued = kept ?? [0, 1, 2, 3, 4, 5];
  cueRecording(files, label, x, [
    ...cued.map((k) => ({ cond, kind: "cue", host_time: index[cues[k]!]! / 500 })),
    ...more.map(([c, at]) => ({ cond: c, kind: "cue", host_time: index[at]! / 500 })),
  ], index);
  return cues;
}

test("saccade and blink leave out a trial within the high-pass's settle time of a break, after it or before it, as they leave out a trial that holds a break, and keep one just outside it and the trials at the recording's own first and last samples, as the library does", () => {
  // The settle time is three time constants of the high-pass, 3 / (2π·hp)
  // seconds: 4.8 s, 2,387.3 samples at 500 Hz, at the default hp of 0.1 Hz,
  // and half that at 0.2 Hz. Every trial kept is analyzed as it is where the
  // trials left out were never cued.
  for (const analysis of ["saccade", "blink"] as const) {
    const cond = analysis === "saccade" ? "left" : "blink";
    for (const hp of [null, 0.2]) {
      const room = (3 / (2 * Math.PI * (hp ?? 0.1))) * 500;
      const inside = Math.floor(room);
      assert.ok(inside < room && room < inside + 1, String(room));
      const files = new Map<string, Uint8Array>();
      settleRecording(files, "settle", analysis, inside, inside + 1);
      settleRecording(files, "settle_without", analysis, inside, inside + 1, true, [0, 2, 3, 5]);
      const rec = new Collection(files);
      const p = hp === null ? {} : { hp };
      const got = ANALYSES.get(analysis)!.fn(rec, "settle", p);
      const want = ANALYSES.get(analysis)!.fn(rec, "settle_without", p);
      const conditions = J.toPlain(J.field(got, "conditions")) as Record<string, { n_trials: number }>;
      assert.equal(conditions[cond]!.n_trials, 4, `${analysis} ${hp}`);
      assert.deepEqual(J.toPlain(J.field(got, "left_out")!), { [cond]: leftOut({ gap_nearby: 2 }) }, `${analysis} ${hp}`);
      assert.deepEqual(J.toPlain(J.field(want, "left_out")!), { [cond]: leftOut() }, `${analysis} ${hp}`);
      for (const key of ["conditions", "criteria", "verdict"]) assert.equal(J.dumps(J.field(got, key)), J.dumps(J.field(want, key)), `${analysis} ${hp} ${key}`);
      assert.ok(J.asI64(J.field(got, "analysis_version"))! >= 12n);
    }
  }
});

test("saccade and blink count the trials they leave out by reason in the result's own left_out, a condition whose trials were all left out staying out of the conditions and the verdict, and count none where none is left out, as the library does", () => {
  // The result's `left_out` gives, for each condition the recording holds a
  // cue of, how many of its trials were left out: `outside_recording` for
  // one that begins before the recording's first sample or ends after its
  // last, `gap_inside` for one whose samples hold a break, and `gap_nearby`
  // for one within the high-pass's settle time of a break, each trial
  // counted once. The sham trials, cued in the recording's last second, are
  // all left out: sham is not among the conditions, as before, its count is
  // in `left_out`, and the conditions, the criteria and the verdict are
  // those of the same recording without the sham cues.
  const l1 = SETTLE_STRETCHES[0];
  const n = SETTLE_STRETCHES[0] + SETTLE_STRETCHES[1] + SETTLE_STRETCHES[2];
  for (const analysis of ["saccade", "blink"] as const) {
    const first = analysis === "saccade" ? "left" : "blink";
    const more: [string, number][] = [
      [first, 100], // begins before the first sample
      [first, n - 100], // ends after the last
      [first, l1 - 50], // holds the first break
      ["sham", n - 300], // ends after the last
      ["sham", n - 200],
    ];
    const files = new Map<string, Uint8Array>();
    settleRecording(files, "reasons", analysis, 2387, 2388, true, null, more);
    settleRecording(files, "reasons_no_sham", analysis, 2387, 2388, true, null, more.slice(0, 3));
    settleRecording(files, "reasons_none", analysis, 2387, 2388, false);
    const rec = new Collection(files);
    const fn = ANALYSES.get(analysis)!.fn;
    const got = fn(rec, "reasons", {});
    const conditions = J.toPlain(J.field(got, "conditions")!) as Record<string, { n_trials: number }>;
    assert.deepEqual(Object.keys(conditions), [first], analysis);
    assert.equal(conditions[first]!.n_trials, 4, analysis);
    assert.deepEqual(J.toPlain(J.field(got, "left_out")!), { [first]: { gap_inside: 1, gap_nearby: 2, outside_recording: 2 }, sham: leftOut({ outside_recording: 2 }) }, analysis);
    assert.ok(!Object.hasOwn(J.toPlain(J.field(got, "criteria")!) as object, "sham_is_null"), analysis);
    const want = fn(rec, "reasons_no_sham", {});
    assert.deepEqual(J.toPlain(J.field(want, "left_out")!), { [first]: { gap_inside: 1, gap_nearby: 2, outside_recording: 2 } }, analysis);
    for (const key of ["conditions", "criteria", "verdict"]) assert.equal(J.dumps(J.field(got, key)), J.dumps(J.field(want, key)), `${analysis} ${key}`);
    assert.deepEqual(J.toPlain(J.field(fn(rec, "reasons_none", {}), "left_out")!), { [first]: leftOut() }, analysis);
    assert.ok(J.asI64(J.field(got, "analysis_version"))! >= 16n);
  }
});

test("with no high-pass, its hp 0 or null, no trial of saccade or blink is left out for the settle time, with the other filters on or off, as the library has it", () => {
  const files = new Map<string, Uint8Array>();
  for (const analysis of ["saccade", "blink"] as const) settleRecording(files, `no_hp_${analysis}`, analysis, 2387, 2388);
  const rec = new Collection(files);
  for (const analysis of ["saccade", "blink"] as const) {
    const cond = analysis === "saccade" ? "left" : "blink";
    for (const off of [{ hp: 0 }, { hp: null }, { hp: 0.0, notch: 0 }, { hp: 0, lp: 0, notch: 0 }]) {
      const got = J.toPlain(ANALYSES.get(analysis)!.fn(rec, `no_hp_${analysis}`, off)) as { conditions: Record<string, { n_trials: number }> };
      assert.equal(got.conditions[cond]!.n_trials, 6, `${analysis} ${JSON.stringify(off)}`);
    }
  }
});

test("a recording without a break has no trial of saccade or blink left out for the settle time, its first and last samples not being breaks, as the library has it", () => {
  const files = new Map<string, Uint8Array>();
  for (const analysis of ["saccade", "blink"] as const) settleRecording(files, `unbroken_${analysis}`, analysis, 2387, 2388, false);
  const rec = new Collection(files);
  for (const analysis of ["saccade", "blink"] as const) {
    const cond = analysis === "saccade" ? "left" : "blink";
    const got = J.toPlain(ANALYSES.get(analysis)!.fn(rec, `unbroken_${analysis}`, {})) as { conditions: Record<string, { n_trials: number }> };
    assert.equal(got.conditions[cond]!.n_trials, 6, analysis);
  }
});

test("on the shared recordings with breaks, saccade and blink leave out each trial that holds a break and each within the default high-pass's 4.8 s settle time of one, and pass, as the library gives them", () => {
  const rec = recordings();
  const sac = J.toPlain(ANALYSES.get("saccade")!.fn(rec, "saccade-broken", {})) as { conditions: Record<string, { n_trials: number }>; verdict: string; failed_criteria: string[] };
  assert.deepEqual(Object.fromEntries(Object.entries(sac.conditions).map(([c, d]) => [c, d.n_trials])), { left: 3, right: 3, sham: 4 });
  assert.deepEqual([sac.verdict, sac.failed_criteria], ["PASS", []]);
  const bl = J.toPlain(ANALYSES.get("blink")!.fn(rec, "blink-broken", {})) as { conditions: Record<string, { n_trials: number }>; verdict: string; failed_criteria: string[] };
  assert.deepEqual(Object.fromEntries(Object.entries(bl.conditions).map(([c, d]) => [c, d.n_trials])), { blink: 2, sham: 1 });
  assert.deepEqual([bl.verdict, bl.failed_criteria], ["PASS", []]);
});

test("saccade gives each condition's ch1_ch2_corr as null where its response window holds one sample, as a post_s of one sample makes it, as the library gives it", () => {
  // The recording's rate is 249.95 Hz, so a post_s of 0.006 s is one sample.
  const got = ANALYSES.get("saccade")!.fn(recordings(), "saccade-pass", { post_s: 0.006 });
  const conditions = J.field(got, "conditions");
  assert.equal(conditions.kind, "object");
  const corr = (conditions as { entries: [string, Value][] }).entries.map(([cond, c]) => [cond, J.dumps(J.field(c, "ch1_ch2_corr"))]);
  assert.deepEqual(corr, [["left", "null"], ["right", "null"], ["sham", "null"]]);
  assert.equal(J.dumps(J.field(got, "analysis_version")), "16");
});

test("saccade and blink keep a trial whose samples, from pre_s before its cue to post_s after it, end at the recording's last sample, and leave out one that would end a sample after it, as the library does, which made the analysis version 13", () => {
  // A trial whose samples end at the recording's last sample runs past
  // neither end of the recording. With the filters on or off it is counted,
  // and with them off it is analyzed as it is in the same recording with one
  // more sample after it.
  const off = { hp: 0, lp: 0, notch: 0 };
  for (const analysis of ["saccade", "blink"] as const) {
    const cond = analysis === "saccade" ? "left" : "blink";
    const n = 4000;
    const cues = [1000, n - SETTLE_POST];
    // In counts: a microvolt is about 45 of them at this gain and reference.
    const x = wavy(n + 1);
    for (const p of cues) {
      for (let k = 0; k < SETTLE_POST; k++) {
        const r = analysis === "saccade" ? 120 * (1 - Math.exp(-k / (0.05 * 500))) : 150 * Math.sin((Math.PI * Math.min(Math.max(k - 0.2 * 500, 0), 0.25 * 500)) / (0.25 * 500));
        x[p + k] = x[p + k]! + Math.round(45 * r);
      }
    }
    const events = cues.map((p) => ({ cond, kind: "cue", host_time: p / 500 }));
    const files = new Map<string, Uint8Array>();
    cueRecording(files, "ends_at_last", x.slice(0, n), events);
    cueRecording(files, "one_more", x, events);
    cueRecording(files, "one_past", x.slice(0, n - 1), events);
    const rec = new Collection(files);
    const fn = ANALYSES.get(analysis)!.fn;
    const trials = (label: string, p: Record<string, number>): number =>
      (J.toPlain(J.field(fn(rec, label, p), "conditions")) as Record<string, { n_trials: number }>)[cond]!.n_trials;
    for (const p of [{}, off]) {
      const said = `${analysis} ${JSON.stringify(p)}`;
      assert.equal(trials("ends_at_last", p), 2, said);
      assert.ok(J.asI64(J.field(fn(rec, "ends_at_last", p), "analysis_version"))! >= 13n, said);
      assert.equal(trials("one_past", p), 1, said);
    }
    assert.equal(J.dumps(J.field(fn(rec, "ends_at_last", off), "conditions")), J.dumps(J.field(fn(rec, "one_more", off), "conditions")), analysis);
  }
});

test("jaw keeps a trial whose clench window ends at the recording's last sample, and leaves out one whose clench window would end a sample after it, as the library does, which made the analysis version 14", () => {
  // A clench window that ends at the recording's last sample runs past
  // neither end of the recording, and its trial is analyzed as it is in the
  // same recording with one more sample after it.
  const w = 500;
  const n = 12 * 500;
  const cues = [1000, 2500, 4000, n - w];
  const random = new Random(70);
  // Noise of `uv` microvolts' spread, in counts: a microvolt is about 45 of
  // them at this gain and reference.
  const noise = (uv: number): number => uv * 45 * Math.sqrt(12) * (random.getrandbits(32) / 2 ** 32 - 0.5);
  const y = Array.from({ length: n + 1 }, () => noise(5));
  for (const p of cues) for (let k = 0; k < w; k++) y[p + k] = y[p + k]! + noise(40);
  const events = cues.map((p) => ({ cond: "clench", kind: "cue", host_time: p / 500 }));
  const files = new Map<string, Uint8Array>();
  cueRecording(files, "jaw_ends_at_last", y.slice(0, n), events);
  cueRecording(files, "jaw_one_more", y, events);
  cueRecording(files, "jaw_one_past", y.slice(0, n - 1), events);
  const rec = new Collection(files);
  assert.equal(rec.load("jaw_ends_at_last").fs, 500);
  const jaw = ANALYSES.get("jaw")!.fn;
  const trials = (label: string): number[] => (J.toPlain(J.field(jaw(rec, label, {}), "channels")) as { n_trials: number }[]).map((c) => c.n_trials);
  assert.deepEqual(trials("jaw_ends_at_last"), [4, 4]);
  assert.equal(J.dumps(J.field(jaw(rec, "jaw_ends_at_last", {}), "analysis_version")), "16");
  assert.deepEqual(J.toPlain(J.field(jaw(rec, "jaw_ends_at_last", {}), "left_out")!), { clench: leftOut() });
  assert.equal(J.dumps(J.field(jaw(rec, "jaw_ends_at_last", {}), "channels")), J.dumps(J.field(jaw(rec, "jaw_one_more", {}), "channels")));
  assert.deepEqual(trials("jaw_one_past"), [3, 3]);
  assert.deepEqual(J.toPlain(J.field(jaw(rec, "jaw_one_past", {}), "left_out")!), { clench: leftOut({ outside_recording: 1 }) });
});

test("counts of three dimensions or more are refused in words as the counts are not channels by samples, by the reader and by every analysis, as the library refuses them, which made the analysis version 11", () => {
  const n = 600;
  const files = new Map<string, Uint8Array>();
  const shapes: [string, number[]][] = [["t_counts_three_dims", [2, 3, n]], ["t_counts_thin_dim", [2, 1, n]], ["t_counts_four_dims", [1, 2, 3, n]]];
  for (const [label, shape] of shapes) {
    const size = shape.reduce((a, b) => a * b, 1);
    const counts = BigInt64Array.from({ length: size }, (_, k) => BigInt(50_000 + ((k * 37) % 2000) - 1000));
    const index = BigInt64Array.from({ length: n }, (_, k) => BigInt(k));
    const members: [string, npy.NpyArray][] = [
      ["counts", { shape, fortran: false, data: { kind: "i8", values: counts } }],
      ["index", npy.vector({ kind: "i8", values: index })],
      ["device_ticks", npy.vector({ kind: "i8", values: index.map((k) => k * 2000n) })],
      ["host_time", npy.vector({ kind: "f8", values: Float64Array.from({ length: n }, (_, k) => k / 500) })],
      ["leadoff", npy.vector({ kind: "u1", values: new Uint8Array(n) })],
      ["gap_before", npy.vector({ kind: "b1", values: new Uint8Array(n) })],
      ["leadoff_active", npy.vector({ kind: "b1", values: new Uint8Array(n).fill(1) })],
    ];
    files.set(`${label}.npz`, zip.write(members.map(([k, a]) => [`${k}.npy`, npy.write(a)]), 1_767_225_600));
    const meta = { label, n, fs_effective: 500.0, vref_uv: 2_400_000, gain: 12, adc_bits: 24, tick_hz: 1_000_000 };
    files.set(`${label}.meta.json`, new TextEncoder().encode(JSON.stringify(meta)));
  }
  const rec = new Collection(files);
  for (const [label] of shapes) {
    const words = `${label}: the counts are not channels by samples`;
    assert.throws(() => rec.load(label), (e: unknown) => e instanceof CaptureError && e.message === words, label);
    for (const analysis of ["summary", "contact"]) {
      assert.throws(() => ANALYSES.get(analysis)!.fn(rec, label, {}), (e: unknown) => (e as Error).message === words, `${analysis} ${label}`);
    }
  }
});

test("saccade and blink refuse a recording whose values cannot be computed before any trial is taken, whether or not it holds a trial of theirs, as the library does", () => {
  // A recording that does not state its converter's width holds no values
  // that can be computed.
  const files = new Map<string, Uint8Array>();
  const x = wavy(5000);
  for (const [analysis, cond] of [["saccade", "left"], ["blink", "blink"]] as const) {
    cueRecording(files, `t_${analysis}_unscaled_marker`, x, [{ cond: "marker", kind: "marker", host_time: 5.0 }], null, { adc_bits: null });
    cueRecording(files, `t_${analysis}_unscaled_cue`, x, [{ cond, kind: "cue", host_time: 5.0 }], null, { adc_bits: null });
  }
  const rec = new Collection(files);
  for (const analysis of ["saccade", "blink"]) {
    for (const label of [`t_${analysis}_unscaled_marker`, `t_${analysis}_unscaled_cue`]) {
      assert.throws(
        () => ANALYSES.get(analysis)!.fn(rec, label, {}),
        (e: unknown) => {
          assert.ok(e instanceof CaptureError, String(e));
          assert.equal(e.message, `${label} does not record adc_bits, so the microvolts it holds cannot be computed. Manifests from version 2 onward record it.`);
          return true;
        },
        label,
      );
    }
  }
});

test("a recording's cues and a game pad's inputs land on the same timeline as its samples, as the library puts them there", () => {
  const rec = recordings();
  const ev = J.field(fixture, "events");
  assert.equal(ev.kind, "object");
  for (const [label, want] of (ev as { entries: [string, Value][] }).entries) {
    assert.equal(J.dumps(J.array(events(rec, label))), J.dumps(want), `events ${label}`);
  }
  assert.ok(events(rec, "saccade-pass").length >= 10 && events(rec, "no-events").length === 0);
  const cases = items(J.get(fixture, "inputs"));
  let most = 0;
  for (const c of cases) {
    const label = text(J.field(c, "label"));
    const kinds = J.field(c, "kinds");
    const got = kinds.kind === "null" ? inputs(rec, label) : inputs(rec, label, items(kinds).map(text));
    assert.equal(J.dumps(J.array(got)), J.dumps(J.field(c, "result")), `inputs ${label} ${J.dumps(kinds)}`);
    most = Math.max(most, got.length);
  }
  assert.ok(most >= 10, String(most));
  assert.deepEqual(inputs(rec, "no-events"), []);
});

test("an analysis takes the library's parameters by name, and refuses a name the library's function does not take", () => {
  assert.throws(() => ANALYSES.get("summary")!.fn(recordings(), "rest-4ch", { nonsense: 1 }), {
    name: "TypeError",
    message: "summary() got an unexpected keyword argument 'nonsense'",
  });
  assert.throws(() => compare(recordings(), ["rest-4ch"], { line_hz: 50 }), {
    name: "TypeError",
    message: "psd_curve() got an unexpected keyword argument 'line_hz'",
  });
});

function floatsOf(v: Value): number[] {
  return items(v).map((x) => {
    assert.ok(x.kind === "float" || x.kind === "int");
    return Number((x as { value: number | string }).value);
  });
}

function num(v: Value): number {
  return Number((v as { value: number | string }).value);
}

/** A list of numbers held to the library's, each against the largest of them. */
function sameList(mine: ArrayLike<number>, theirs: readonly number[], at: string, key: string): void {
  assert.equal(mine.length, theirs.length, `${at}: the length`);
  const big = Math.max(...theirs.map(Math.abs));
  for (let i = 0; i < theirs.length; i++) {
    const err = relative(mine[i]!, theirs[i]!, big);
    note(key, err, `${at}/${i}`);
    assert.ok(err <= TOLERANCE, `${at}/${i}: ${mine[i]} against ${theirs[i]}, off by ${err.toExponential(2)}`);
  }
}

/** One number held to the library's, against its own size. */
function sameNumber(mine: number, theirs: number, at: string, key: string): void {
  if (!Number.isFinite(theirs)) {
    assert.ok(Object.is(mine, theirs) || (Number.isNaN(mine) && Number.isNaN(theirs)), `${at}: ${mine} against ${theirs}`);
    return;
  }
  const err = relative(mine, theirs, Math.abs(theirs));
  note(key, err, at);
  assert.ok(err <= TOLERANCE, `${at}: ${mine} against ${theirs}, off by ${err.toExponential(2)}`);
}

const numerics = J.field(fixture, "numerics");

test("filter design is scipy's: Butterworth sections and notches at other rates and cutoffs", () => {
  for (const d of items(J.get(numerics, "butter"))) {
    const hz = num(J.field(d, "hz"));
    const fs = num(J.field(d, "fs"));
    const btype = text(J.field(d, "btype")) === "hp" ? "highpass" : "lowpass";
    const mine = N.butter4(hz, btype, fs).flat();
    sameList(mine, items(J.get(d, "sos")).flatMap(floatsOf), `butter ${hz} ${btype} ${fs}`, "sos");
  }
  for (const d of items(J.get(numerics, "iirnotch"))) {
    const [b, a] = N.iirnotch(num(J.field(d, "hz")), num(J.field(d, "q")), num(J.field(d, "fs")));
    const [tb, ta] = items(J.get(d, "ba")).map(floatsOf);
    sameList(b, tb!, "iirnotch b", "notch");
    sameList(a, ta!, "iirnotch a", "notch");
  }
});

test("forward and backward filtering is scipy's, with and without sections", () => {
  const f = J.field(numerics, "filtering");
  const x = Float64Array.from(floatsOf(J.field(f, "x")));
  for (const d of items(J.get(f, "sosfiltfilt"))) {
    const btype = text(J.field(d, "btype")) === "hp" ? "highpass" : "lowpass";
    const sos = N.butter4(num(J.field(d, "hz")), btype, num(J.field(d, "fs")));
    sameList(N.sosfiltfilt(sos, x), floatsOf(J.field(d, "y")), `sosfiltfilt ${J.dumps(J.field(d, "hz"))} ${btype}`, "filtered");
  }
  for (const d of items(J.get(f, "filtfilt"))) {
    const [b, a] = N.iirnotch(num(J.field(d, "hz")), num(J.field(d, "q")), num(J.field(d, "fs")));
    sameList(N.filtfilt(b, a, x), floatsOf(J.field(d, "y")), `filtfilt ${J.dumps(J.field(d, "hz"))}`, "filtered");
  }
});

test("Welch's spectrum is scipy's at transform lengths of every kind, a power of two, a prime and the rest", () => {
  for (const d of items(J.get(numerics, "welch"))) {
    const x = floatsOf(J.field(d, "x"));
    const s = N.welch(x, num(J.field(d, "fs")), num(J.field(d, "nperseg")));
    const at = `welch ${x.length} ${J.dumps(J.field(d, "nperseg"))}`;
    sameList(s.f, floatsOf(J.field(d, "f")), at, "frequency");
    sameList(s.p, floatsOf(J.field(d, "p")), at, "spectrum");
  }
});

test("the three tests are scipy's on every one of their methods", () => {
  for (const c of items(J.get(numerics, "tests"))) {
    const name = text(J.field(c, "test"));
    const x = floatsOf(J.field(c, "x"));
    const y = floatsOf(J.field(c, "y"));
    const r = name === "wilcoxon" ? S.wilcoxonGreater(x, y) : name === "ttest_rel" ? S.ttestRel(x, y) : S.mannwhitneyuGreater(x, y);
    const at = `${name}, ${text(J.field(c, "why"))}`;
    sameNumber(r.statistic, num(J.field(c, "statistic")), at, `${name} statistic`);
    sameNumber(r.pvalue, num(J.field(c, "p")), at, `${name} p`);
  }
});

test("the normal and Student tails are scipy's across their range", () => {
  for (const row of items(J.get(numerics, "ndtr"))) {
    const [z, p] = floatsOf(row);
    sameNumber(S.ndtr(z!), p!, `ndtr ${z}`, "ndtr");
  }
  for (const row of items(J.get(numerics, "stdtr"))) {
    const [df, t, p] = floatsOf(row);
    sameNumber(S.stdtr(df!, t!), p!, `stdtr ${df} ${t}`, "stdtr");
  }
  if (process.env.ANALYSIS_ERRORS) {
    for (const key of ["sos", "notch", "filtered", "frequency", "spectrum", "wilcoxon statistic", "wilcoxon p", "ttest_rel statistic",
      "ttest_rel p", "mannwhitneyu statistic", "mannwhitneyu p", "ndtr", "stdtr"]) {
      const w = worst.get(key);
      if (w !== undefined) console.log(`${key.padEnd(24)} ${w.err.toExponential(2)}  ${w.at}`);
    }
  }
});
