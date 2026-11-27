/**
 * The analyses, as the Python library's `analysis` module runs them on a
 * recording: the same seven, with the same parameters, the same filters,
 * spectra and tests, and the same report, key for key, held to the
 * library's own results by the fixture in `contract/analysis`. On that
 * fixture every verdict, count and pass or fail is the library's, and every
 * number the library gets from scipy's filters, transforms and tests agrees
 * with it here to within one part in a hundred billion, the bound the
 * package's tests hold (a value in a list such as a spectrum measured
 * against the largest in the list). The arithmetic is the same, but its
 * last bits may differ, as they may between two computers running Python,
 * so a number that lands within that difference of a threshold can give a
 * different verdict.
 *
 * An analysis reads a recording from a collection, as `capture` reads one,
 * so it runs wherever the recording's files are: in a browser, from the
 * recorder or a file picker, or in Node, from `capture/node`. Parameters are
 * the ones the library takes, by its names, as a form sends them, and the
 * report is a JSON value whose numbers keep Python's integers and floats
 * apart. A recording or a setting the library refuses is refused with the
 * package's `CaptureError`, in the library's words, a value passed written
 * as JavaScript writes it: null, true, NaN, Infinity, "text", [1] and
 * {"a": 1}. A parameter the library's function does not take is refused
 * with a `TypeError`, and so is a setting of `contact` that is not a
 * number.
 *
 * `ANALYSES` registers every analysis by name, as the library's registry
 * does. An analysis takes one recording's label first, and an application
 * runs it on each selected recording separately. One whose entry says
 * `several: true` takes a list of recording labels first instead, and an
 * application runs it once, on all the selected recordings together. An
 * entry without that key takes one label, as every entry of the package's
 * does. `events` and `inputs` put a recording's cues and its controller
 * events on the same timeline as its samples, as the library's do.
 *
 * Each entry's `params` lists the settings a form shows for the analysis,
 * each with its `name`, its `type` and its `default`, and with the range an
 * analysis can use it in, as data an application reads, as the library's
 * registry gives them:
 *
 * - A "select" is one of its `choices`.
 * - A "number" is a finite number, at least its `min`, above its `above`,
 *   at most its `max`, and below its `below`, where it has them. A `below`
 *   is a number, or the name of another setting of the same analysis: a
 *   band's low edge is below its high edge. A filter stage's setting has an
 *   `off` value, 0, which turns the stage off, as null does, and is
 *   accepted outside the range.
 * - `rate` and `samples` bound a number by the recording's sample rate, so
 *   they are checked once the rate is known. A `rate` of "below half" is
 *   below half the rate, and "at most half" at most half the rate; one of
 *   "lowered" is taken as 1 Hz below half the rate where it is at or above
 *   half the rate. A `samples` bounds a time in seconds: that time times the
 *   rate, its fraction dropped, is at least that many samples, and that
 *   time times the rate is at most the largest number a 64-bit float holds.
 * - A notch's width at half power, its frequency over `notch_q`, is also
 *   below half the rate, and a filter whose starting state cannot be
 *   computed at the rate, as one whose frequency is too close to 0 Hz for
 *   the rate, is refused as well.
 *
 * Every analysis refuses with a `CaptureError`, naming the recording and
 * the setting, a setting outside its range, those whose range needs no
 * recording before it reads the recording. contact's settings carry no
 * range, and berger's `settle`, which no form shows, is at least 0 and
 * below the recording's block length.
 *
 * ```ts
 * import { analysis, capture } from "@intomind/sdk";
 *
 * const recordings = new capture.Collection(files);
 * const spec = analysis.ANALYSES.get("berger")!;
 * const report = spec.fn(recordings, "berger_20260101-102030", { epoch_s: 2 });
 * ```
 */

import { CaptureError } from "./capture/errors.ts";
import { fixed, grouped, groupedFixed, numpySum } from "./capture/pyfmt.ts";
import * as J from "./capture/pyjson.ts";
import type { Value } from "./capture/pyjson.ts";
import * as npy from "./capture/npy.ts";
import { Collection, checkLabel, readNpz } from "./capture/recording.ts";
import type { Recording } from "./capture/recording.ts";
import { sha256 } from "./digest.ts";
import { contactFromLeadoff } from "./instrument.ts";
import * as D from "./py.ts";
import * as N from "./numeric.ts";
import { amount, g, isReal, num, selectRefusal } from "./settings.ts";
import * as S from "./stats.ts";

/**
 * Bumped on any change that can alter a reported number, as the library's
 * is. 3: a recording's rate is measured from the host's clock over the
 * stretches where its sample index climbs by one, so a hole is not read as
 * slow sampling. 4: the contact analysis's lead-off figures count only the
 * samples whose lead-off bits are a verdict on contact, so a recording of an
 * IntoMind device whose firmware is older than 1.4.6, which cannot check
 * electrode contact, taken with detection on, has none. 5: a filtered
 * spectrum of a recording with no stretch that holds one segment once
 * filtered is the filtered signal's, in one segment as long as its longest
 * stretch, where it was the unfiltered signal's. Berger's blocks start only
 * at events of the condition closed or open, so its count of reps counts no
 * other event. The table psd and compare return is of the spectrum the
 * curves show, filtered where they are and in their segments, where it was
 * the unfiltered summary's in segments of 2,048. No spectrum is taken across
 * a break: a recording with no stretch long enough for a segment and a
 * second more has the spectrum of its longest stretch without a break, where
 * it had that of the whole recording, and jaw leaves out a trial whose rest
 * or clench window holds a break. A number with no value is null, where it
 * was NaN or infinity: a median PSD over no frequency, a slope with too few
 * frequencies to fit, and saccade's correlation, berger's and jaw's
 * p-values, berger's effect size and reference band ratio and jaw's ratio
 * where they have none. A jaw channel with power in neither its rest nor
 * its clench windows does not meet power_ratio, where 0 over 0 met it. 6:
 * the contact analysis gives each channel `contact`, the device's own
 * judgment, and its lead-off figures, leadoff_bytes and
 * leadoff_nonzero_frac, count only the samples where lead-off detection was
 * running and, where the recording says, the bits are a verdict, where they
 * counted every sample whose bits were a verdict, detection running or not:
 * a recording that does not say whether detection was running counts none.
 * After the version became 6, with no bump of its own, summary's
 * saturation_pct came to be measured against the full scale of a converter
 * as wide as the recording states, its adc_bits, where it was measured
 * against a 24-bit converter's for every recording, so a version 6 result
 * for a recording of another width may hold either figure. 7:
 * summary's n_runs and kept_frac count the stretches without a break its
 * spectrum was computed from, as `welchPsd` chooses them for the nperseg
 * given, where they counted every stretch of at least 2,048 samples whatever
 * nperseg was. With the default nperseg, a stretch of 2,048 samples or more
 * that is shorter than 2,048 and a second more is no longer counted, and a
 * recording with no stretch that long counts its longest stretch, as 1. 8:
 * saccade and blink leave out a trial whose samples, from pre_s before its
 * cue to post_s after it, hold a break, as jaw leaves out a trial whose
 * windows hold one. 9: saccade and blink filter each stretch without a break
 * on its own, by the code that filters each stretch of `welchPsd`'s filtered
 * spectrum, where they filtered the whole recording in one pass, so a trial
 * in a recording with a break carries nothing of the filters' response to
 * the break. A recording without a break gives what it gave. A trial in a
 * stretch too short for the filters is refused, as `welchPsd` refuses such a
 * stretch, in its words. 10: the contact analysis of a recording that holds
 * no samples gives each channel's dc_uv and ac_rms_uv as null, where they
 * were NaN. 11: the Python library's `load` refuses a recording whose counts
 * are of three dimensions or more, saying they are not channels by samples,
 * as this package's reader already refused it, where the library read them,
 * taking the first dimension as the channels. Every analysis of such a
 * recording is refused. A recording whose counts are channels by samples
 * gives what it gave. 12: saccade and blink leave out a trial whose samples,
 * from pre_s before its cue to post_s after it, begin less than the
 * high-pass's settle time after a break or end less than that time before
 * one, as they leave out a trial that holds a break, where they measured it
 * on the high-pass's swing after the start of a stretch and before its end.
 * The settle time is three time constants of the high-pass, 3 / (2π·hp)
 * seconds, 4.8 s at the default hp of 0.1 Hz, and none where no high-pass
 * runs. The recording's own first and last samples are not breaks, so a
 * recording without a break gives what it gave. 13: saccade and blink keep a
 * trial whose samples, from pre_s before its cue to post_s after it, end at
 * the recording's last sample, where they left it out as one that ran past
 * the end of the recording. A recording with no such trial gives what it
 * gave. 14: jaw keeps a trial whose clench window ends at the recording's
 * last sample, where it left it out as one whose windows ran past the end of
 * the recording. A recording with no such trial gives what it gave. 15:
 * saccade and blink filter the whole recording in one pass again, the
 * samples on each side of a break taken as next to each other, as they did
 * before version 9, where they filtered each stretch without a break on its
 * own. On real recordings a trial near a break of the size lost packets
 * leave, 0.05 to 0.2 s, was changed more by the filters' response to the
 * ends of its stretch than by their response to the break. A trial that
 * holds a break, or lies within the high-pass's settle time of one, is left
 * out as before, and a trial in a stretch too short for the filters is no
 * longer refused. A recording without a break gives what it gave. 16:
 * saccade, blink and jaw give `left_out`: for each condition the recording
 * holds a cue of, how many of its trials were left out, by reason,
 * `gap_inside`, `gap_nearby` and `outside_recording`. Nothing else in a
 * result changes: a condition whose trials were all left out is not among
 * the `conditions`, as before.
 */
const ANALYSIS_VERSION = 16;

/** The bands a summary splits its power into. */
const EEG_BANDS: readonly (readonly [string, number, number])[] = [
  ["delta", 1, 4],
  ["theta", 4, 8],
  ["alpha", 8, 13],
  ["beta", 13, 30],
  ["gamma", 30, 45],
];

// --- what an analysis is -----------------------------------------------------------

/** The parameters an analysis was given, by the library's names, as a form sends them. A number keeps Python's integer or float as `capture.fromPlain` reads it. */
export type Params = Readonly<Record<string, unknown>>;

/**
 * A parameter an analysis declares, as a form draws it, with the range the
 * analysis can use it in. A number with none of `min`, `above`, `max`,
 * `below`, `rate` and `samples` carries no range, as contact's do.
 */
export interface Param {
  /** The parameter's name. */
  readonly name: string;
  /** How a form takes it: number or select. */
  readonly type: string;
  /** What it is unless a person changes it. */
  readonly default: unknown;
  /** For a select, what may be chosen. */
  readonly choices?: readonly unknown[];
  /** For a filter stage's setting, the value that turns the stage off, 0, accepted outside the range, as null is. */
  readonly off?: number;
  /** The least the number may be. */
  readonly min?: number;
  /** A number the number must be above. */
  readonly above?: number;
  /** The most the number may be. */
  readonly max?: number;
  /** A number the number must be below, or the name of another setting of the same analysis it must be below, as a band's low edge is below its high edge. */
  readonly below?: number | string;
  /**
   * A bound by the recording's sample rate, checked once the rate is known:
   * "below half" is below half the rate, "at most half" at most half the
   * rate, and "lowered" is taken as 1 Hz below half the rate where it is at
   * or above half the rate.
   */
  readonly rate?: string;
  /** For a time in seconds, the fewest samples it holds: that time times the recording's rate, its fraction dropped, is at least this many. */
  readonly samples?: number;
}

/** What an analysis does: read one recording from a collection, by its label, and report on it. */
export type AnalysisFn = (recordings: Collection, label: string, params?: Params) => Value;

/** One analysis, as the library's registry holds it. */
export interface Spec {
  /** What it does, handed one recording's label, or, registered with `several`, a list of the selected recordings' labels. */
  fn(recordings: Collection, label: string | readonly string[], params?: Params): Value;
  /** Whether it needs the recording's cues. */
  readonly needsEvents: boolean;
  /** What it is, in a line. */
  readonly doc: string;
  /** The parameters a form offers. */
  readonly params: readonly Param[];
  /**
   * Whether `fn` takes a list of recording labels first, so an application
   * runs it once, on all the selected recordings together. Absent, as in
   * every entry of the package's, `fn` takes one label, and an application
   * runs it on each selected recording separately.
   */
  readonly several?: boolean;
}

// --- reading values as Python does ---------------------------------------------------

/** A parameter's value, as given or the library's default, refusing a name the library's function does not take as Python refuses it. */
function args(fn: string, p: Params, defaults: Readonly<Record<string, Value>>): Record<string, Value> {
  for (const k of Object.keys(p)) {
    if (p[k] !== undefined && !Object.hasOwn(defaults, k)) {
      throw new TypeError(`${fn}() got an unexpected keyword argument '${k}'`);
    }
  }
  const out: Record<string, Value> = {};
  for (const [k, d] of Object.entries(defaults)) out[k] = p[k] === undefined ? d : J.fromPlain(p[k]);
  return out;
}

const F = J.float;
const I = (n: number): Value => J.int(n);
const B = J.bool;
const T = J.str;
const obj = (...entries: [string, Value][]): Value => J.object(entries);
const floats = (a: ArrayLike<number>): Value => J.array(Array.from(a, (x) => F(x)));

/** Run a step that may be refused by the numerics, refusing in their words. */
function design<X>(step: () => X): X {
  try {
    return step();
  } catch (e) {
    if (e instanceof N.DesignError) throw new CaptureError(e.message);
    throw e;
  }
}

function readText(recordings: Collection, name: string): string | null {
  const b = recordings.file(name);
  if (b === null) return null;
  return new TextDecoder("utf-8", { fatal: true }).decode(b);
}

/** The recording's sample rate as the library's `Capture.fs` reports it: the manifest's own number, an integer kept an integer, unless the host clock overrules it. */
function fsValue(rec: Recording): Value {
  const stored = J.field(rec.manifest, "fs_effective");
  const fs = rec.fs;
  if (J.truthy(stored) && J.isNumber(stored) && J.asF64(stored) === fs) return stored;
  return F(fs);
}

function npzSha256(recordings: Collection, label: string): Value {
  const bytes = recordings.file(`${checkLabel(label)}.npz`)!;
  let s = "";
  for (const x of sha256(bytes)) s += x.toString(16).padStart(2, "0");
  return T(s);
}

/** `x[a:b]`. */
const slice = (x: Float64Array, a: number, b: number): Float64Array => x.slice(Math.max(0, a), Math.max(0, b));

/** `x - x.mean()`. */
function demean(x: Float64Array): Float64Array {
  const m = N.mean(x);
  return x.map((v) => v - m);
}

/** The samples in microvolts, channel by channel. */
const channels = (rec: Recording): Float64Array[] => Array.from({ length: rec.channels }, (_, c) => rec.uv(c));

// --- events ----------------------------------------------------------------------

function offsetAt(anchors: readonly [Value, Value][], mono: number): [Value, Value] {
  const a = [...anchors].sort((x, y) => num(x[1]) - num(y[1]));
  let best = a[0]!;
  for (const pair of a) {
    if (num(pair[1]) <= mono) best = pair;
    else break;
  }
  return best;
}

/** `timebase.to_frame`: an instant on the monotonic clock, on the recording's one realtime timeline. */
function toFrame(anchors: readonly [Value, Value][], mono: Value, frame: Value): Value {
  const frameMono = frame.kind === "null" ? Math.min(...anchors.map((p) => num(p[1]))) : num(frame);
  const [t, m] = offsetAt(anchors, frameMono);
  if (mono.kind === "int" && t.kind === "int" && m.kind === "int") {
    return J.int(BigInt(mono.value) + (BigInt(t.value) - BigInt(m.value)));
  }
  return F(num(mono) + (num(t) - num(m)));
}

/** The recording's manifest, or nothing where there is none, as `events` and `inputs` read it for the clock. */
function manifestOf(recordings: Collection, label: string): Value {
  const text = readText(recordings, `${label}.meta.json`);
  return text === null ? J.object([]) : J.loads(text);
}

/**
 * Every event of `raw` on the recording's own realtime frame, as the
 * library's `timebase.align_events` puts it there and `events` hands it on:
 * `host_time` the aligned instant, the time it was stamped with kept as
 * `host_time_recorded`, and `aligned` saying whether a correction was
 * possible. An event with no `mono` cannot be corrected and keeps its
 * `host_time`, marked `aligned` false rather than passed off as corrected.
 */
function aligned(meta: Value, raw: readonly Value[], file: string): Value[] {
  const clock = J.truthy(J.field(meta, "clock")) ? J.field(meta, "clock") : J.object([]);
  const anchorList = J.truthy(J.field(clock, "anchors")) ? J.field(clock, "anchors") : J.array([]);
  const anchors: [Value, Value][] = anchorList.kind === "array"
    ? anchorList.items.map((a) => {
        if (a.kind !== "array" || a.items.length !== 2) throw new CaptureError("an anchor is a realtime and a monotonic reading");
        return [a.items[0]!, a.items[1]!];
      })
    : [];
  const frame = J.field(clock, "frame_mono");
  return raw.map((item) => {
    if (item.kind !== "object") throw new CaptureError(`${file} holds an event that is not an object`);
    const e = J.copyObject(item);
    const m = J.field(e, "mono");
    if (m.kind === "null" || anchors.length === 0) {
      J.set(e, "host_time_aligned", J.field(e, "host_time"));
      J.set(e, "aligned", B(false));
    } else {
      J.set(e, "host_time_aligned", toFrame(anchors, m, frame));
      J.set(e, "aligned", B(true));
    }
    J.set(e, "host_time_recorded", J.field(e, "host_time"));
    J.set(e, "host_time", J.pop(e, "host_time_aligned"));
    return e;
  });
}

/**
 * The recording's cues on the same timeline as its samples' host times, as the
 * library's `analysis.events` returns them: each event's `host_time` the
 * aligned instant, which is what indexes correctly into the samples' host
 * times, the clock it was actually stamped with kept as
 * `host_time_recorded`, and `aligned` saying whether a correction was
 * possible at all. The file is never rewritten: it stays the record of what
 * was stamped. Nothing for a recording with no events file.
 */
export function events(recordings: Collection, label: string): Value[] {
  const text = readText(recordings, `${checkLabel(label)}.events.json`);
  if (text === null) return [];
  const raw = J.loads(text);
  if (raw.kind !== "array") throw new CaptureError(`${label}.events.json is not a list of events`);
  return aligned(manifestOf(recordings, label), raw.items, `${label}.events.json`);
}

// The kernel's event types and the names the library's controller module
// gives the controls it names. The raw numbers are kept beside them, so an
// unrecognized control is still data.
const EV_SYN = 0x00;
const EV_KEY = 0x01;
const EV_ABS = 0x03;
/** The code of Linux's report, of type EV_SYN, that it threw away events from a device before they were read. */
const SYN_DROPPED = 0x03;
const BTN_NAMES: Readonly<Record<number, string>> = {
  0x130: "BTN_SOUTH", 0x131: "BTN_EAST", 0x132: "BTN_C", 0x133: "BTN_NORTH", 0x134: "BTN_WEST", 0x135: "BTN_Z",
  0x136: "BTN_TL", 0x137: "BTN_TR", 0x138: "BTN_TL2", 0x139: "BTN_TR2", 0x13a: "BTN_SELECT", 0x13b: "BTN_START",
  0x13c: "BTN_MODE", 0x13d: "BTN_THUMBL", 0x13e: "BTN_THUMBR",
};
const ABS_NAMES: Readonly<Record<number, string>> = {
  0x00: "ABS_X", 0x01: "ABS_Y", 0x02: "ABS_Z", 0x03: "ABS_RX", 0x04: "ABS_RY", 0x05: "ABS_RZ", 0x10: "ABS_HAT0X", 0x11: "ABS_HAT0Y",
};
const EV_NAMES: Readonly<Record<number, string>> = { 0x00: "SYN", 0x01: "KEY", 0x02: "REL", 0x03: "ABS", 0x04: "MSC" };

/** A number as Python's `#x` format writes it. */
function hexOf(code: number): string {
  return code < 0 ? `-0x${(-code).toString(16)}` : `0x${code.toString(16)}`;
}

/** A readable name for a control, as the library's `controller.code_name` gives one, falling back to the number. Never invents a meaning. */
function codeName(type: number, code: number): string {
  if (type === EV_KEY) return BTN_NAMES[code] ?? `KEY_${hexOf(code)}`;
  if (type === EV_ABS) return ABS_NAMES[code] ?? `ABS_${hexOf(code)}`;
  // Linux's loss marker, of type EV_SYN and code SYN_DROPPED, is named SYN_DROPPED.
  if (type === EV_SYN && code === SYN_DROPPED) return "SYN_DROPPED";
  return `${EV_NAMES[type] ?? type}_${hexOf(code)}`;
}

/**
 * Controller events, on the same timeline as the recording's samples, as the
 * library's `analysis.inputs` returns them. They are stored as arrays in
 * `<label>.inputs.npz` rather than as JSON because of their rate, and
 * handed back here in the same event shape `events` returns, so an analysis
 * does not have to care which container a label came out of: `host_time`
 * is the aligned time, for the same reason and with the same guarantee as
 * in `events`, with each event's `type`, `code` and `value` as the kernel
 * gave them and `cond` the control's name. `kinds` filters by what the
 * control is: `button` for a key, the sparse, discrete labels most analyses
 * want, and `axis` for an absolute axis, continuous, and `other` for
 * anything else, left out unless `kinds` names it: among them each row
 * where Linux reported that it threw away events before they were read, of
 * type EV_SYN and code SYN_DROPPED, which a recording keeps where it came
 * and which is named SYN_DROPPED. Nothing for a recording with no inputs
 * file.
 */
export function inputs(recordings: Collection, label: string, kinds: readonly string[] = ["button", "axis"]): Value[] {
  const bytes = recordings.file(`${checkLabel(label)}.inputs.npz`);
  if (bytes === null) return [];
  const meta = manifestOf(recordings, label);
  const arrays = new Map(readNpz(bytes));
  const column = (name: string): npy.Data => {
    const a = arrays.get(name);
    if (a === undefined) throw new CaptureError(`'${name} is not a file in the archive'`);
    return a.data;
  };
  const mono = column("input_mono");
  const host = column("input_host_time");
  const type = column("input_type");
  const code = column("input_code");
  const value = column("input_value");
  const want = new Set(kinds);
  const n = Math.min(npy.len(mono), npy.len(host), npy.len(type), npy.len(code), npy.len(value));
  const raw: Value[] = [];
  for (let i = 0; i < n; i++) {
    const t = npy.intAt(type, i);
    const c = npy.intAt(code, i);
    const kind = t === EV_KEY ? "button" : t === EV_ABS ? "axis" : "other";
    if (!want.has(kind)) continue;
    raw.push(obj(
      ["kind", T(kind)],
      ["cond", T(codeName(t, c))],
      ["host_time", F(npy.f64At(host, i))],
      ["mono", F(npy.f64At(mono, i))],
      ["type", I(t)],
      ["code", I(c)],
      ["value", I(npy.intAt(value, i))],
    ));
  }
  return aligned(meta, raw, `${label}.inputs.npz`);
}

// --- the pieces every analysis is made of --------------------------------------------

/** `contiguous_runs`: the bounds of each run of the index that climbs by one, at least `minlen` long. */
function contiguousRuns(index: BigInt64Array, minlen: number): [number, number][] {
  const edges = [0];
  for (let i = 0; i + 1 < index.length; i++) if (index[i + 1]! - index[i]! !== 1n) edges.push(i + 1);
  edges.push(index.length);
  const out: [number, number][] = [];
  for (let k = 0; k + 1 < edges.length; k++) if (edges[k + 1]! - edges[k]! >= minlen) out.push([edges[k]!, edges[k + 1]!]);
  return out;
}

/**
 * `_unbroken`: a test of whether the samples from position a to position b,
 * b excluded, hold no break in `index`, where the sample index does not
 * climb by one: true or false for each a and b.
 */
function unbrokenIn(index: BigInt64Array): (a: number, b: number) => boolean {
  // breaks[j] counts the breaks among the first j + 1 samples, so the
  // samples from a to b, b excluded, hold breaks[b - 1] - breaks[a].
  const breaks = new Int32Array(Math.max(1, index.length));
  for (let j = 1; j < index.length; j++) breaks[j] = breaks[j - 1]! + (index[j]! - index[j - 1]! !== 1n ? 1 : 0);
  return (a, b) => b - a < 2 || breaks[b - 1] === breaks[a];
}

/**
 * `_settle_s`: the time, in seconds, a high-pass of `hp` Hz run through a
 * break is given to settle on each side of it: three of its time constants,
 * 3 / (2π·hp), which is 4.8 s at 0.1 Hz, and 0 where no high-pass runs, `hp`
 * being 0 or null.
 */
function settleS(hp: Value): number {
  return J.truthy(hp) ? 3 / (2 * Math.PI * num(hp)) : 0.0;
}

/**
 * `_settled`: a test of whether the samples from position a to position b,
 * b excluded, which hold no break in `index`, begin at least `settle`
 * seconds after a break and end at least that long before one, at the rate
 * `fs`: true or false for each a and b. The time after a break runs from the
 * first sample after it to sample a, and the time before one from sample
 * b - 1 to the last sample before it, each that many samples over `fs`. The
 * first and last samples of the recording are not breaks, and with a
 * `settle` of 0 every a and b is true.
 */
function settledIn(index: BigInt64Array, fs: number, settle: number): (a: number, b: number) => boolean {
  if (!settle) return () => true;
  const runs = contiguousRuns(index, 1);
  const n = index.length;
  return (a, b) => {
    // The last stretch that starts at or before a.
    let k = 0;
    for (let lower = 0, upper = runs.length; lower < upper;) {
      const mid = (lower + upper) >>> 1;
      if (runs[mid]![0] <= a) {
        k = mid;
        lower = mid + 1;
      } else upper = mid;
    }
    const [first, end] = runs[k]!;
    return !((first > 0 && (a - first) / fs < settle) || (end < n && (end - b) / fs < settle));
  };
}

/** A spectrum `welchPsd` computed, and the position bounds of the stretches it was computed from, in order. */
interface Welch extends N.Spectrum {
  readonly stretches: readonly (readonly [number, number])[];
}

interface Filters {
  readonly hp: Value;
  readonly lp: Value;
  readonly notch: Value;
  readonly notch_q: Value;
  readonly trim: Value;
}

/** `preprocess`: high-pass, notch and low-pass, each skipped where its setting is 0, then half a second trimmed from each end. */
function preprocess(xIn: Float64Array, fs: number, f: Filters): Float64Array {
  let x = xIn;
  const sosHp = J.truthy(f.hp) ? design(() => N.butter4(num(f.hp), "highpass", fs)) : null;
  const sosLp = J.truthy(f.lp) ? design(() => N.butter4(num(f.lp), "lowpass", fs)) : null;
  const bn = J.truthy(f.notch) ? design(() => N.iirnotch(num(f.notch), num(f.notch_q), fs)) : null;
  if (sosHp !== null) x = design(() => N.sosfiltfilt(sosHp, x));
  if (bn !== null) x = design(() => N.filtfilt(bn[0], bn[1], x));
  if (sosLp !== null) x = design(() => N.sosfiltfilt(sosLp, x));
  const k = Math.trunc(num(f.trim) * fs);
  return k !== 0 && x.length > 2 * k + 8 ? x.slice(k, x.length - k) : x;
}

/** `n` samples, in words: "1 sample", "2,048 samples". */
const samplesSaid = (n: number): string => (n === 1 ? `${grouped(n)} sample` : `${grouped(n)} samples`);

/**
 * `n` samples at `fs` a second, as seconds to three significant digits,
 * rounded up so the time said is never short of `n` samples.
 */
function secondsUp(n: number, fs: number): string {
  const s = n / fs;
  const places = Math.max(0, 2 - Math.floor(Math.log10(s)));
  const scale = 10 ** places;
  const up = Math.ceil(Number(fixed(s * scale, 6))) / scale;
  return places ? groupedFixed(up, places).replace(/0+$/, "").replace(/\.$/, "") : groupedFixed(up, 0);
}

/** A frequency setting as a sentence shows it: "60 Hz", NaN, or a value that is not a number as JavaScript writes it. */
const hz = (v: Value): string => amount(v, "Hz");

/**
 * Refuse a time setting `v`, in seconds, that is not a finite number,
 * holds fewer than `least` samples at the rate `fs`, its fraction dropped,
 * or times `fs` is past the largest number a 64-bit float holds, so that
 * its samples cannot be counted, naming the recording and the setting.
 * `name` is the setting's name with its article, as "a pre_s"; `means`
 * says what it is; `why`, where given, closes the sentence about the fewest
 * samples. The time said is rounded up, so that time at `fs` holds `least`
 * samples. `least` is the setting's `samples` in the registry.
 */
function refuseShortTime(label: string, name: string, means: string, v: Value, fs: number, least = 1, why = ""): void {
  const bare = name.slice(name.indexOf(" ") + 1);
  // `least` is a whole number, so the product reaches it exactly where its
  // whole part does, and an infinite product is refused in words of its own.
  if (isReal(v) && Number.isFinite(num(v)) && num(v) * fs >= least) {
    if (Number.isFinite(num(v) * fs)) return;
    throw new CaptureError(
      `${label} cannot be analyzed with ${name} of ${amount(v, "s")}: ${bare}, ${means}, times the recording's ` +
        `rate of ${g(fs)} Hz is past the largest number a 64-bit float holds, so its samples cannot be counted.`,
    );
  }
  throw new CaptureError(
    `${label} cannot be analyzed with ${name} of ${amount(v, "s")}: ${bare}, ${means}, is a finite number of ` +
      `seconds that holds at least ${samplesSaid(least)}, ${secondsUp(least, fs)} s at the recording's rate of ` +
      `${g(fs)} Hz${why}.`,
  );
}

/** The fewest samples the time setting `name` of `analysis` holds at a recording's rate, its `samples` in the registry. */
const fewestSamples = (analysis: string, name: string): number => PARAMS[analysis]!.find((p) => p.name === name)!.samples!;

/** The unit each setting of the analyses is given in, as a refusal shows it. */
const UNITS: Readonly<Record<string, string>> = {
  line_hz: "Hz", slope_lo: "Hz", slope_hi: "Hz", fmax: "Hz", lo_hz: "Hz", hi_hz: "Hz", alpha_lo: "Hz", alpha_hi: "Hz",
  ref_lo: "Hz", ref_hi: "Hz", min_deflection_uv: "µV", min_pp_uv: "µV", ppmax: "µV", max_latency_sd_s: "s",
};

/** What each setting of the analyses is, as a refusal says it. */
const MEANS: Readonly<Record<string, string>> = {
  line_hz: "the mains frequency whose line is reported",
  slope_lo: "the lowest frequency of the slope fit",
  slope_hi: "the highest frequency of the slope fit",
  filtered: "whether the spectrum is of the filtered signal",
  fmax: "the highest frequency of the curves",
  min_deflection_uv: "the smallest deflection that counts",
  max_corr: "the highest correlation of ch1 and ch2 that counts as inverted",
  min_pp_uv: "the smallest peak to peak that counts as a blink",
  min_detect_frac: "the least fraction of blink trials that must show one",
  max_latency_sd_s: "the largest spread of the blinks' latencies that counts as consistent",
  sham_max_pp_ratio: "the response over baseline below which sham trials count as null",
  lo_hz: "the band's low edge",
  hi_hz: "the band's high edge",
  min_ratio: "the smallest clench power over rest power that counts",
  alpha: "the p-value below which a channel counts as significant",
  ppmax: "the largest peak to peak an epoch may have and be kept",
  alpha_lo: "the alpha band's low edge",
  alpha_hi: "the alpha band's high edge",
  ref_lo: "the reference band's low edge",
  ref_hi: "the reference band's high edge",
  allow_pseudoreplication: "whether the epochs are also tested pooled",
};

/**
 * Settings whose range is checked where they are used, in words of their
 * own: the filters, by `refuseImpossibleFilters`; the times that hold
 * samples, by `refuseShortTime`; and `nperseg`, by `welchPsd`.
 */
const CHECKED_ON_USE = new Set(["hp", "lp", "notch", "notch_q", "pre_s", "post_s", "window_s", "epoch_s", "nperseg"]);

/** `name` with its article: "a line_hz", "an alpha", "an fmax". */
const article = (name: string): string => ("aeiou".includes(name[0]!) || name === "fmax" || name === "nperseg" ? `an ${name}` : `a ${name}`);

/** The bounds a registry parameter `p` gives, in words. */
function bounds(p: Param, unit: string): string {
  const u = unit ? ` ${unit}` : "";
  const lower = p.min !== undefined ? `at least ${g(p.min)}` : p.above !== undefined ? `above ${g(p.above)}` : null;
  const upper = p.max !== undefined ? `at most ${g(p.max)}` : typeof p.below === "number" ? `below ${g(p.below)}` : null;
  if (p.min !== undefined && p.max !== undefined) return `from ${g(p.min)} to ${g(p.max)}${u}`;
  if (lower !== null && upper !== null) return `${lower}${u} and ${upper}${u}`;
  return lower !== null ? (lower.startsWith("at") ? `of ${lower}${u}` : `${lower}${u}`) : `${upper}${u}`;
}

/** The parameters of `analysis` in its registry entry that `given` holds and that are not checked where they are used. */
const ranged = (analysis: string, given: Readonly<Record<string, Value>>): Param[] =>
  PARAMS[analysis]!.filter((p) => Object.hasOwn(given, p.name) && !CHECKED_ON_USE.has(p.name));

/**
 * Refuse, naming the recording and the setting, a setting `given` to
 * `analysis` that the registry's range for it does not hold, where that
 * range needs no recording: a select that is not one of its choices, and a
 * number that is not finite or not within its `min`, `above`, `max` and
 * `below`, a low edge first held to its own bounds and then to its high
 * edge. Settings in `CHECKED_ON_USE` are left to their own checks.
 */
function checkSettings(label: string, analysis: string, given: Readonly<Record<string, Value>>): void {
  const params = ranged(analysis, given);
  for (const p of params) {
    const name = p.name;
    const v = given[name]!;
    const unit = UNITS[name] ?? "";
    if (p.type === "select") {
      const refused = selectRefusal(label, name, MEANS[name]!, p.choices!, v);
      if (refused !== null) throw new CaptureError(refused);
      continue;
    }
    const x = isReal(v) ? num(v) : NaN;
    const ok = Number.isFinite(x)
      && (p.min === undefined || x >= p.min) && (p.above === undefined || x > p.above)
      && (p.max === undefined || x <= p.max) && (typeof p.below !== "number" || x < p.below);
    if (!ok) {
      throw new CaptureError(
        `${label} cannot be analyzed with ${article(name)} of ${amount(v, unit)}: ${name}, ${MEANS[name]}, is a finite number ${bounds(p, unit)}.`,
      );
    }
  }
  for (const p of params) {
    const name = p.name;
    const high = p.below;
    if (typeof high === "string" && Object.hasOwn(given, high) && !(num(given[name]!) < num(given[high]!))) {
      const unit = UNITS[name] ?? "";
      throw new CaptureError(
        `${label} cannot be analyzed with ${article(name)} of ${amount(given[name]!, unit)} and ` +
          `${article(high)} of ${amount(given[high]!, unit)}: ${name}, ${MEANS[name]}, is below ` +
          `${high}, ${MEANS[high]}.`,
      );
    }
  }
}

/**
 * Refuse a setting `given` to `analysis` whose range in the registry
 * depends on the recording's rate `fs`, above 0, once that is known: a
 * `rate` of "below half" holds a value below half the rate, and a low edge
 * below a high edge whose `rate` is "lowered" is below that edge as taken,
 * 1 Hz below half the rate where it is at or above half the rate. Returns
 * the settings as taken. Settings in `CHECKED_ON_USE` are left to their own
 * checks.
 */
function checkRateSettings(label: string, analysis: string, given: Readonly<Record<string, Value>>, fs: number): Record<string, Value> {
  const params = ranged(analysis, given);
  const taken: Record<string, Value> = { ...given };
  for (const p of params) {
    const name = p.name;
    const v = given[name]!;
    if (p.rate === "below half" && !(num(v) < fs / 2)) {
      throw new CaptureError(
        `${label} cannot be analyzed with ${article(name)} of ${amount(v, "Hz")}: ${name}, ` +
          `${MEANS[name]}, is below ${g(fs / 2)} Hz, half the recording's rate of ${g(fs)} Hz.`,
      );
    }
    if (p.rate === "lowered" && num(v) >= fs / 2) taken[name] = F(fs / 2 - 1.0);
  }
  for (const p of params) {
    const name = p.name;
    const high = p.below;
    if (typeof high === "string" && taken[high] !== given[high] && !(num(given[name]!) < num(taken[high]!))) {
      throw new CaptureError(
        `${label} cannot be analyzed with ${article(name)} of ${amount(given[name]!, "Hz")}: ${name}, ` +
          `${MEANS[name]}, is below ${high}, ${MEANS[high]}, and ${article(high)} of ${amount(given[high]!, "Hz")}, ` +
          `at or above ${g(fs / 2)} Hz, half the recording's rate of ${g(fs)} Hz, is taken as ${g(num(taken[high]!))} Hz, ` +
          "1 Hz below half the rate.",
      );
    }
  }
  return taken;
}

/**
 * Refuse a recording whose sample rate `fs` is not above 0, before any
 * filter setting is checked against it. A recording's rate is 0 where it
 * states no rate above 0 Hz and the host times of its samples measure none,
 * as those of fewer than 2 samples measure none. `needs` closes the
 * sentence: what needs a rate.
 */
function refuseUnmeasuredRate(label: string, fs: number, needs: string): void {
  if (!(fs > 0)) {
    throw new CaptureError(
      `${label} has no sample rate that can be measured, and ${needs}: the recording states no rate above ` +
        "0 Hz, and the host times of its samples measure none over its stretches without a break.",
    );
  }
}

/** `x` as a float, or null where it is NaN or infinite, which JSON cannot carry. */
const reported = (x: number): Value => (Number.isFinite(x) ? F(x) : J.NULL);

/**
 * Refuse filter settings from which no filter can be made at the rate
 * `fs`, in the library's words rather than in the numerics', naming what
 * was passed as JavaScript writes it. A high-pass or low-pass frequency is
 * above 0 and below half the rate, and a notch frequency above 0 and at most
 * half the rate, each 0 or null for no such filter. A notch's `notch_q` is a
 * finite number above 0, and its width at half power, its frequency over
 * `notch_q`, below half the rate: no notch is made from a `notch_q` of 0,
 * and an unstable one, which turns the signal into NaN, from one below 0 or
 * one too small for the rate. A filter that meets these but whose starting
 * state cannot be solved, which a frequency too close to 0 Hz for the rate
 * makes, is refused as well. `label` names the recording.
 */
function refuseImpossibleFilters(label: string, fs: number, f: Filters): void {
  const shown = hz;
  const solvable = (step: () => unknown): boolean => {
    try {
      step();
      return true;
    } catch (e) {
      if (e instanceof N.DesignError && e.message === "Singular matrix") return false;
      throw e;
    }
  };
  for (const [kind, v, btype] of [["high-pass", f.hp, "highpass"], ["low-pass", f.lp, "lowpass"]] as const) {
    if (!J.truthy(v)) continue;
    if (!(isReal(v) && fs > 0 && 0 < (2 * num(v)) / fs && (2 * num(v)) / fs < 1)) {
      throw new CaptureError(
        `${label} cannot be filtered with a ${kind} of ${shown(v)}: a ${kind} frequency is above 0 Hz ` +
          `and below ${g(fs / 2)} Hz, half the recording's rate of ${g(fs)} Hz, or 0 for no ${kind}.`,
      );
    }
    if (!solvable(() => N.sosfiltZi(design(() => N.butter4(num(v), btype, fs))))) {
      throw new CaptureError(
        `${label} cannot be filtered with a ${kind} of ${shown(v)}: a ${kind} frequency that close to ` +
          `0 Hz cannot be computed at the recording's rate of ${g(fs)} Hz; a higher one can, or 0 for ` +
          `no ${kind}.`,
      );
    }
  }
  const notch = f.notch;
  if (!J.truthy(notch)) return;
  if (!(isReal(notch) && fs > 0 && 0 <= (2 * num(notch)) / fs && (2 * num(notch)) / fs <= 1)) {
    throw new CaptureError(
      `${label} cannot be filtered with a notch of ${shown(notch)}: a notch frequency is above 0 Hz ` +
        `and at most ${g(fs / 2)} Hz, half the recording's rate of ${g(fs)} Hz, or 0 for no notch.`,
    );
  }
  const qv = f.notch_q;
  const q = isReal(qv) ? g(num(qv)) : J.jsRepr(qv);
  if (!(isReal(qv) && Number.isFinite(num(qv)) && num(qv) > 0)) {
    throw new CaptureError(
      `${label} cannot be filtered with a notch of ${shown(notch)} whose notch_q is ${q}: notch_q, ` +
        "the notch frequency over the notch's width at half power, is a finite number above 0.",
    );
  }
  const width = num(notch) / num(qv);
  if (!(width < fs / 2)) {
    throw new CaptureError(
      `${label} cannot be filtered with a notch of ${shown(notch)} whose notch_q is ${q}: that notch is ` +
        `${g(width)} Hz wide at half power, its frequency over notch_q, and a notch's width is ` +
        `below ${g(fs / 2)} Hz, half the recording's rate of ${g(fs)} Hz.`,
    );
  }
  if (!solvable(() => N.lfilterZi(...design(() => N.iirnotch(num(notch), num(qv), fs))))) {
    throw new CaptureError(
      `${label} cannot be filtered with a notch of ${shown(notch)} whose notch_q is ${q}: its filter ` +
        `cannot be computed at the recording's rate of ${g(fs)} Hz. No notch can be whose frequency is too ` +
        `close to 0 Hz, or whose width at half power, its frequency over notch_q, is too close to ` +
        `${g(fs / 2)} Hz, half the rate.`,
    );
  }
}

/**
 * The fewest samples `preprocess` can filter with these stages, 0 where
 * every stage is off: each stage is filtered forward and backward after
 * each end of the trace is padded with three samples for each of the
 * filter's taps, and a trace must hold more samples than that padding. A
 * high-pass or low-pass, an order 4 Butterworth filter in two second-order
 * sections, counts five taps, since the last denominator coefficient of
 * each section, the squared magnitude of a pair of its poles, is above 0,
 * and a notch counts three, whatever their frequencies and the rate, so
 * this needs no rate.
 */
function filterNeeds(f: Filters): number {
  const pads: number[] = [];
  for (const v of [f.hp, f.lp]) if (J.truthy(v)) pads.push(3 * 5);
  if (J.truthy(f.notch)) pads.push(3 * 3);
  return pads.length > 0 ? Math.max(...pads) + 1 : 0;
}

/**
 * Refuse `n` samples too few for `preprocess` to filter with these
 * settings, before the filtering refuses them in its own words. `said`
 * opens the sentence: which recording, or which stretch of one, holds them.
 * The time the filters need is said where the rate `fs` is above 0.
 */
function refuseUnfilterable(said: string, n: number, fs: number, f: Filters): void {
  const needs = filterNeeds(f);
  if (n >= needs) return;
  const stages: string[] = [];
  for (const [kind, v] of [["high-pass", f.hp], ["notch", f.notch], ["low-pass", f.lp]] as const) {
    if (J.truthy(v)) stages.push(`a ${hz(v)} ${kind}`);
  }
  const listed = stages.length === 1 ? stages[0]! : `${stages.slice(0, -1).join(", ")} and ${stages[stages.length - 1]}`;
  throw new CaptureError(
    `${said}, too few to filter with ${listed}: ` +
      `${stages.length === 1 ? "that filter needs" : "those filters need"} at least ${grouped(needs)} samples` +
      (fs > 0 ? `, ${secondsUp(needs, fs)} s at the recording's rate of ${g(fs)} Hz.` : "."),
  );
}

/**
 * `_stretch_said`: what opens a refusal of the samples from position `a` to
 * `b`, `b` excluded, of the recording `label` of `n` samples, a stretch
 * without a break: the recording, where the stretch is all of it; its
 * longest stretch without a break, where `longest`; or one of its stretches.
 */
function stretchSaid(label: string, n: number, a: number, b: number, longest = false): string {
  const m = b - a;
  if (a === 0 && b === n) return `${label} holds ${samplesSaid(m)}`;
  if (longest) return `${label} holds no stretch without a break longer than ${samplesSaid(m)}`;
  return `${label} has a stretch of ${samplesSaid(m)} without a break`;
}

/**
 * `_filtered`: the samples `x` of one stretch without a break, filtered and
 * trimmed on their own by `preprocess` at the rate `fs` with the settings
 * `f`, and refused by `refuseUnfilterable` where too few to filter. `said`
 * opens the refusal.
 */
function filtered(x: Float64Array, fs: number, said: string, f: Filters): Float64Array {
  refuseUnfilterable(said, x.length, fs, f);
  return preprocess(x, fs, f);
}

/**
 * `welch_psd`: Welch's spectrum in segments of `nperseg` samples, averaged
 * over each stretch of at least `nperseg` samples and a second more in which
 * the sample index climbs by one without a break. A recording with no such
 * stretch has the spectrum of its longest stretch without a break, the
 * first of several equally long; a recording with no break is itself that
 * stretch. No spectrum is taken across a break. With `filters`, each
 * stretch is filtered and trimmed by `preprocess`; otherwise its mean is
 * subtracted. Where no stretch then holds `nperseg` samples, the spectrum is
 * that of the longest, in one segment as long as it is.
 *
 * Beside the spectrum it gives the position bounds of the stretches the
 * spectrum was computed from, in order, as the library's `_welch` does. This
 * is the one place those stretches are chosen, so what counts them counts
 * what the spectrum holds.
 *
 * Refused, in this order: a recording, or a longest stretch without a
 * break, of fewer than 2 samples; a recording whose rate cannot be
 * measured, its rate not above 0; an `nperseg` that is not a finite number
 * of at least 2, taken as a whole number of samples by dropping its
 * fraction; and, with `filters`, settings from which no filter can be made
 * at the recording's rate and a stretch too short for the filters.
 */
function welchPsd(rec: Recording, c: number, npersegV: Value, filters: Filters | null): Welch {
  const fs = rec.fs;
  const label = rec.label;
  const n = rec.samples;
  if (n < 2) throw new CaptureError(`${label} holds ${samplesSaid(n)}, too few for a spectrum: a spectrum needs at least 2 samples.`);
  // The longest stretch without a break, the first of several as long, is
  // counted before anything the rate decides.
  let stretch: [number, number] | null = null;
  for (const r of contiguousRuns(rec.index, 1)) if (stretch === null || r[1] - r[0] > stretch[1] - stretch[0]) stretch = r;
  if (stretch![1] - stretch![0] < 2) {
    throw new CaptureError(
      `${label} holds no stretch without a break longer than ${samplesSaid(stretch![1] - stretch![0])}, too few for a spectrum: ` +
        "a spectrum needs at least 2 samples.",
    );
  }
  refuseUnmeasuredRate(label, fs, "a spectrum needs one");
  // A segment of 1 sample holds only 0 Hz.
  if (!(isReal(npersegV) && Number.isFinite(num(npersegV)) && num(npersegV) >= 2)) {
    throw new CaptureError(
      `${label} cannot be analyzed with an nperseg of ${amount(npersegV)}: nperseg, the samples in each segment of the ` +
        "spectrum, is a finite number of at least 2.",
    );
  }
  const nperseg = num(npersegV);
  if (filters !== null) refuseImpossibleFilters(label, fs, filters);
  let runs = contiguousRuns(rec.index, nperseg + Math.trunc(fs));
  const longest = runs.length === 0;
  if (longest) {
    // A spectrum across a break would splatter its power, so the longest
    // stretch without one stands for the recording.
    runs = [stretch!];
  }
  const said = (a: number, b: number): string => stretchSaid(label, n, a, b, longest);
  if (runs[0]![1] - runs[0]![0] < 2) {
    throw new CaptureError(`${said(runs[0]![0], runs[0]![1])}, too few for a spectrum: a spectrum needs at least 2 samples.`);
  }
  const acc: Float64Array[] = [];
  const short: { readonly x: Float64Array; readonly stretch: [number, number] }[] = [];
  const used: [number, number][] = [];
  let f: Float64Array | null = null;
  for (const [a, b] of runs) {
    const piece = rec.uv(c).slice(a, b);
    const x: Float64Array = filters !== null ? filtered(piece, fs, said(a, b), filters) : demean(piece);
    if (x.length < nperseg) {
      short.push({ x, stretch: [a, b] });
      continue;
    }
    const s = design(() => N.welch(x, fs, nperseg));
    f = s.f;
    acc.push(s.p);
    used.push([a, b]);
  }
  if (acc.length === 0) {
    // No stretch holds one segment, so the longest, as filtered or not as
    // asked, is taken as one segment of its own length.
    let longestShort = short[0]!;
    for (const y of short) if (y.x.length > longestShort.x.length) longestShort = y;
    const s = design(() => N.welch(longestShort.x, fs, longestShort.x.length));
    return { f: s.f, p: s.p, stretches: [longestShort.stretch] };
  }
  const p = new Float64Array(acc[0]!.length);
  for (const row of acc) for (let k = 0; k < p.length; k++) p[k] = p[k]! + row[k]!;
  for (let k = 0; k < p.length; k++) p[k] = p[k]! / acc.length;
  return { f: f!, p, stretches: used };
}

/** The values of `a` where `keep` holds. */
function where(a: ArrayLike<number>, keep: (i: number) => boolean): Float64Array {
  const out: number[] = [];
  for (let i = 0; i < a.length; i++) if (keep(i)) out.push(a[i]!);
  return Float64Array.from(out);
}

/** `_band`: the power between `lo` and below `hi`, by the trapezoid rule. */
function band(s: N.Spectrum, lo: number, hi: number): number {
  const keep = (i: number): boolean => s.f[i]! >= lo && s.f[i]! < hi;
  const p = where(s.p, keep);
  return p.length > 0 ? N.trapezoid(p, where(s.f, keep)) : 0.0;
}

/** `_median_psd`: the median of the spectrum from `lo` to `hi` Hz, both included, or null where it holds none of those frequencies. */
function medianPsd(s: N.Spectrum, lo: number, hi: number): Value {
  const p = where(s.p, (i) => s.f[i]! >= lo && s.f[i]! <= hi);
  return p.length > 0 ? F(N.median(p)) : J.NULL;
}

/** `spectral_slope`: the slope of the log-log spectrum between `lo` and `hi`, outside `exclude`. NaN with fewer than eight points. */
function spectralSlope(s: N.Spectrum, lo = 2.0, hi = 40.0, exclude: readonly [number, number] = [7.0, 14.0]): number {
  const keep = (i: number): boolean => {
    const f = s.f[i]!;
    return f >= lo && f <= hi && !(f >= exclude[0] && f <= exclude[1]) && s.p[i]! > 0;
  };
  const f = where(s.f, keep);
  if (f.length < 8) return NaN;
  const p = where(s.p, keep);
  return N.polyfit1(f.map(Math.log10), p.map(Math.log10))[0];
}

/**
 * `clean_epochs`: the filtered epochs of `epochS` seconds whose peak to peak
 * is within `ppmax`, and how many there were in all. Refused, in this order:
 * a recording whose rate cannot be measured; an `epochS` that is not a
 * finite number of seconds holding at least 1 sample at the recording's
 * rate, or that times the rate is past the largest number a 64-bit float
 * holds; settings from which no filter can be made at the recording's rate;
 * and a stretch kept for an epoch but too short for the filters.
 */
function cleanEpochs(rec: Recording, c: number, epochV: Value, ppmax: number, filters: Filters, sel: number[] | null): [Float64Array[], number] {
  const fs = rec.fs;
  refuseUnmeasuredRate(rec.label, fs, "epochs need one");
  refuseShortTime(rec.label, "an epoch_s", "the length of each epoch", epochV, fs);
  refuseImpossibleFilters(rec.label, fs, filters);
  const L = Math.trunc(num(epochV) * fs);
  const keep: Float64Array[] = [];
  let total = 0;
  const all = rec.uv(c);
  const index = rec.index;
  const idx = sel === null ? index : BigInt64Array.from(sel, (i) => index[i]!);
  const dat = sel === null ? all : Float64Array.from(sel, (i) => all[i]!);
  for (const [a, b] of contiguousRuns(idx, L + Math.trunc(fs))) {
    refuseUnfilterable(`${rec.label} has a stretch of ${samplesSaid(b - a)} without a break`, b - a, fs, filters);
    const x = preprocess(dat.slice(a, b), fs, filters);
    for (let k = 0; k < Math.floor(x.length / L); k++) {
      const e = x.slice(k * L, (k + 1) * L);
      total += 1;
      if (N.ptp(e) <= ppmax) keep.push(e);
    }
  }
  return [keep, total];
}

/** The count of the trials an analysis left out, by reason. */
interface LeftOut { gap_inside: number; gap_nearby: number; outside_recording: number }

/**
 * `_none_left_out`: the count, all zero, of the trials an analysis left out,
 * by reason: `gap_inside`, a trial whose samples hold a break in the
 * recording; `gap_nearby`, one within the high-pass's settle time of a
 * break; and `outside_recording`, one that runs past either end of the
 * recording.
 */
function noneLeftOut(): LeftOut {
  return { gap_inside: 0, gap_nearby: 0, outside_recording: 0 };
}

/** A count of trials left out as a result gives it, its reasons in the library's order. */
function leftOutValue(left: LeftOut): Value {
  return obj(["gap_inside", I(left.gap_inside)], ["gap_nearby", I(left.gap_nearby)], ["outside_recording", I(left.outside_recording)]);
}

/** Whether a count of trials left out counts any. */
function anyLeftOut(left: LeftOut): boolean {
  return left.gap_inside + left.gap_nearby + left.outside_recording > 0;
}

/**
 * `_cue_epochs`: for each cue of `cond` with room around it, each channel's
 * samples of the signals `sig`, each channel's filtered samples, from `npre`
 * before to `npost` after, less their mean before the cue. A trial that runs
 * past either end of the recording is left out, and so is one whose samples
 * hold a break in the recording, as jaw leaves out a trial whose windows
 * hold one, and one whose samples begin less than `settle` seconds, the
 * high-pass's settle time `settleS` gives, after a break or end less than
 * that before one, as `settledIn` measures them at the recording's rate.
 * Returns the trials and how many of the condition's trials were left out
 * for each of those reasons, as `noneLeftOut` names them, each trial counted
 * under the first that applies.
 */
function cueEpochs(rec: Recording, evs: readonly Value[], cond: string, npre: number, npost: number, sig: readonly Float64Array[], ht: Float64Array, settle = 0.0): [Float64Array[][], LeftOut] {
  const unbroken = unbrokenIn(rec.index);
  const settled = settledIn(rec.index, rec.fs, settle);
  const out: Float64Array[][] = Array.from({ length: rec.channels }, () => []);
  const left = noneLeftOut();
  for (const e of evs) {
    if (J.asStr(J.field(e, "cond")) !== cond) continue;
    const i = N.searchsorted(ht, num(J.field(e, "host_time")));
    if (i - npre < 0 || i + npost > ht.length) { left.outside_recording++; continue; }
    if (!unbroken(i - npre, i + npost)) { left.gap_inside++; continue; }
    if (!settled(i - npre, i + npost)) { left.gap_nearby++; continue; }
    for (let c = 0; c < rec.channels; c++) {
      const seg = sig[c]!.slice(i - npre, i + npost);
      const m = N.mean(seg.subarray(0, npre));
      out[c]!.push(seg.map((v) => v - m));
    }
  }
  return [out, left];
}

/** `E.mean(axis=0)`: the trials averaged sample by sample, added in order. */
function meanOverTrials(trials: readonly Float64Array[]): Float64Array {
  const out = new Float64Array(trials[0]!.length);
  for (const t of trials) for (let k = 0; k < out.length; k++) out[k] = out[k]! + t[k]!;
  return out.map((v) => v / trials.length);
}

/** `E[:, a:b].std()`: every sample in those columns of every trial taken together, its mean from the rows' sums added in order, as numpy reduces a strided block. */
function stdOfBlock(trials: readonly Float64Array[], a: number, b: number): number {
  const count = trials.length * (b - a);
  if (count === 0) return NaN;
  let total = 0;
  for (const t of trials) total += numpySum(t.subarray(a, b));
  const m = total / count;
  const sq = new Float64Array(count);
  let k = 0;
  for (const t of trials) {
    for (let i = a; i < b; i++) {
      const d = t[i]! - m;
      sq[k++] = d * d;
    }
  }
  return Math.sqrt(numpySum(sq) / count);
}

/** The largest and smallest values, NaN if any is. */
function extremes(a: ArrayLike<number>): [number, number] {
  let hi = -Infinity;
  let lo = Infinity;
  for (let i = 0; i < a.length; i++) {
    const v = a[i]!;
    if (Number.isNaN(v)) return [NaN, NaN];
    if (v > hi) hi = v;
    if (v < lo) lo = v;
  }
  return [hi, lo];
}

// --- the analyses ------------------------------------------------------------------

const FILTER_DEFAULTS = { hp: F(1.0), lp: F(45.0), notch: F(60.0), notch_q: F(30.0) };

function summaryOf(recordings: Collection, label: string, p: Params = {}): Value {
  const a = args("summary", p, { line_hz: F(60.0), nperseg: I(2048), slope_lo: F(2.0), slope_hi: F(40.0) });
  checkSettings(label, "summary", { line_hz: a.line_hz!, slope_lo: a.slope_lo!, slope_hi: a.slope_hi! });
  const rec = recordings.load(label);
  const n = rec.samples;
  const fs = fsValue(rec);
  const meta = rec.manifest;
  // A converter's counts saturate at its own full scale, from its width.
  const stated = J.field(meta, "adc_bits");
  const bits = J.truthy(stated) ? Number(D.pyIntOf(J.toPlain(stated))) : 24;
  if (bits - 1 < 0) throw new CaptureError("negative shift count");
  const fullscale = 2 ** (bits - 1) - 1;
  const lsb = rec.lsbUv;
  // Each channel's spectrum, with the stretches it was computed from, which
  // are the same for every channel: n_runs and kept_frac count those, so
  // they describe the spectrum beside them.
  const spectra: Welch[] = [];
  for (let c = 0; c < rec.channels; c++) spectra.push(welchPsd(rec, c, a.nperseg!, null));
  const runs = spectra.length > 0 ? spectra[0]!.stretches : [];
  let kept = 0;
  for (const [x, y] of runs) kept += y - x;
  const lineHz = num(a.line_hz!);
  const chans: Value[] = [];
  for (let c = 0; c < rec.channels; c++) {
    const x = rec.uv(c);
    const raw = Float64Array.from(rec.counts(c), (v) => Number(v));
    const s = spectra[c]!;
    const ac = demean(x);
    const med = (lo: number, hi: number): Value => medianPsd(s, lo, hi);
    const inBand = (i: number): boolean => s.f[i]! >= 1 && s.f[i]! <= 45;
    const tot = band(s, 1, 45);
    const acStd = N.std(ac);
    const [acMax, acMin] = extremes(ac);
    let railed = 0;
    for (const v of raw) if (Math.abs(v) > 0.99 * fullscale) railed++;
    if (s.p.length < 2) throw new CaptureError("attempt to get argmax of an empty sequence");
    const peak = N.argmax(s.p.subarray(1));
    chans.push(obj(
      ["dc_uv", F(N.mean(x))],
      ["ac_rms_uv", F(acStd)],
      ["ac_pp_uv", F(acMax - acMin)],
      ["pp_over_rms", F(acStd !== 0 ? (acMax - acMin) / acStd : 0.0)],
      ["saturation_pct", F((raw.length === 0 ? NaN : railed / raw.length) * 100)],
      ["rms_1_45_uv", F(Math.sqrt(N.trapezoid(where(s.p, inBand), where(s.f, inBand))))],
      ["psd_5_45", med(5, 45)],
      ["psd_60_90", med(60, 90)],
      ["psd_100_200", med(100, 200)],
      ["slope_2_40", reported(spectralSlope(s, num(a.slope_lo!), num(a.slope_hi!)))],
      ["peak_hz", F(s.f[1 + peak]!)],
      ["bands", J.object(EEG_BANDS.map(([name, lo, hi]) => [name, F(tot !== 0 ? (band(s, lo, hi) / tot) * 100 : 0.0)]))],
      ["psd_60", med(lineHz - 0.6, lineHz + 0.6)],
      ["line_hz", a.line_hz!],
    ));
  }
  return obj(
    ["label", T(label)],
    ["meta", meta],
    ["fs", fs],
    ["lsb_uv", F(lsb)],
    ["n", I(n)],
    ["n_runs", I(runs.length)],
    ["kept_frac", F(kept / Math.max(1, n))],
    ["channels", J.array(chans)],
  );
}

const PSD_DEFAULTS = { filtered: B(false), fmax: F(100.0), nperseg: I(2048), ...FILTER_DEFAULTS };

/**
 * `psd_curve`: each channel's spectrum up to `fmax`, filtered where
 * `filtered` is true, as `welchPsd` takes it, and a table of one row, the
 * row `compare` also returns for the recording: each channel's median power
 * from 5 to 45 Hz, of the same spectrum as its curve, or null where that
 * spectrum holds no frequency from 5 to 45 Hz, as the coarse spectrum of a
 * short recording can. Refused: first a setting outside the range the
 * registry gives it, and then as `welchPsd` refuses.
 */
function psdCurve(recordings: Collection, label: string, p: Params = {}): Value {
  const a = args("psd_curve", p, PSD_DEFAULTS);
  checkSettings(label, "psd", { filtered: a.filtered!, fmax: a.fmax! });
  const rec = recordings.load(label);
  const filtered = J.truthy(a.filtered!);
  const filters: Filters | null = filtered ? { hp: a.hp!, lp: a.lp!, notch: a.notch!, notch_q: a.notch_q!, trim: F(0.5) } : null;
  const fmax = num(a.fmax!);
  const curves: Value[] = [];
  const row: [string, Value][] = [["label", T(label)]];
  for (let c = 0; c < rec.channels; c++) {
    const s = welchPsd(rec, c, a.nperseg!, filters);
    const keep = (i: number): boolean => s.f[i]! > 0 && s.f[i]! <= fmax;
    curves.push(obj(["f", floats(where(s.f, keep))], ["p", floats(where(s.p, keep))]));
    row.push([`ch${c + 1}_psd_5_45`, medianPsd(s, 5, 45)]);
  }
  return obj(
    ["label", T(label)],
    ["fs", fsValue(rec)],
    ["gain", J.field(rec.manifest, "gain")],
    ["rld_sens", J.field(rec.manifest, "rld_sens")],
    ["curves", J.array(curves)],
    ["table", J.array([J.object(row)])],
  );
}

/**
 * Several recordings' spectra overlaid, as the library's `compare` lays
 * them out: each recording's spectrum as `psd` reports it, then one row
 * per recording of each channel's median power from 5 to 45 Hz, of the
 * same spectrum as its curve, or null where that spectrum holds no
 * frequency from 5 to 45 Hz, the row `psd` reports for it. The parameters
 * are `psd`'s.
 */
export function compare(recordings: Collection, labels: readonly string[], params: Params = {}): Value {
  const fmax = params.fmax === undefined ? F(100.0) : J.fromPlain(params.fmax);
  const rest: Record<string, unknown> = { ...params, fmax };
  const series = labels.map((l) => psdCurve(recordings, l, rest));
  const table = series.map((s) => {
    const t = J.field(s, "table");
    return t.kind === "array" ? t.items[0]! : J.NULL;
  });
  return obj(["kind", T("compare")], ["series", J.array(series)], ["table", J.array(table)]);
}

function contactOf(recordings: Collection, label: string, p: Params = {}): Value {
  const a = args("contact", p, { dc_min_uv: F(1000.0), ac_min_uv: F(2.0) });
  const rec = recordings.load(label);
  // Every word at its full width, a byte for the device, as the recording
  // was read.
  const words = rec.leadoffWords;
  if (words === null) throw new CaptureError("'leadoff is not a file in the archive'");
  // Contact was checked where detection was running and, where the
  // recording says, the bits are a verdict. A recording that does not say
  // whether detection was running says nothing of contact.
  const active = rec.leadoffActive;
  const judged = rec.leadoffJudged;
  const checked = (k: number): boolean =>
    active !== null && active.length === words.length && active[k] === true
    && (judged === null || judged.length !== words.length || judged[k] === true);
  const lo = words.filter((_, k) => checked(k));
  // The bits cover the eight channels of the IntoMind device's byte, as the
  // exports read them.
  const covered = Math.min(rec.channels, 8);
  const dcMin = num(a.dc_min_uv!);
  const acMin = num(a.ac_min_uv!);
  let nonzero = 0;
  for (const v of lo) if (v !== 0n) nonzero++;
  const bytes = [...new Set(lo)].sort((x, y) => (x < y ? -1 : x > y ? 1 : 0));
  // A recording with no samples has no offset or level to judge, so its
  // offset and level are null, the number with no value, and its verdict
  // says it was not checked, as its contact does.
  const empty = rec.samples === 0;
  const chans: Value[] = [];
  for (let c = 0; c < rec.channels; c++) {
    let dc: number | null = null;
    let ac: number | null = null;
    let hasDc = false;
    let hasAc = false;
    if (!empty) {
      const x = rec.uv(c);
      dc = N.mean(x);
      ac = N.std(demean(x));
      hasDc = Math.abs(dc) >= dcMin;
      hasAc = ac >= acMin;
    }
    chans.push(obj(
      ["dc_uv", dc === null ? J.NULL : F(dc)],
      ["ac_rms_uv", ac === null ? J.NULL : F(ac)],
      ["dc_ok", B(hasDc)],
      ["ac_ok", B(hasAc)],
      ["verdict", T(empty ? "NOT CHECKED (no samples)"
        : hasDc && hasAc ? "contact" : !hasDc ? "NO CONTACT (no half-cell offset)" : "suspect (offset but near-floor AC)")],
      ["contact", T(lo.length === 0 || c >= covered ? "not checked"
        : contactFromLeadoff(Array.from(lo, (w) => Number((w >> BigInt(c)) & 1n)), 1)[0] ? "contact" : "no contact")],
    ));
  }
  return obj(
    ["label", T(label)],
    ["analysis", T("contact")],
    ["analysis_version", I(ANALYSIS_VERSION)],
    ["dc_min_uv", a.dc_min_uv!],
    ["ac_min_uv", a.ac_min_uv!],
    ["leadoff_bytes", J.array(bytes.map((b) => J.int(b)))],
    ["leadoff_nonzero_frac", lo.length === 0 ? J.NULL : F(nonzero / lo.length)],
    ["channels", J.array(chans)],
  );
}

const CUE_FILTERS = { hp: F(0.1), lp: F(10.0), notch: F(60.0), notch_q: F(30.0) };

/** The filter settings an event analysis reports, with no trim. */
function cueFilters(a: Record<string, Value>): Filters {
  return { hp: a.hp!, lp: a.lp!, notch: a.notch!, notch_q: a.notch_q!, trim: F(0.0) };
}

const filtersValue = (f: Filters): Value => obj(["hp", f.hp], ["lp", f.lp], ["notch", f.notch], ["notch_q", f.notch_q], ["trim", f.trim]);

function saccadeOf(recordings: Collection, label: string, p: Params = {}): Value {
  const a = args("saccade", p, { pre_s: F(0.5), post_s: F(1.5), ...CUE_FILTERS, min_deflection_uv: F(50.0), max_corr: F(-0.5) });
  checkSettings(label, "saccade", { min_deflection_uv: a.min_deflection_uv!, max_corr: a.max_corr! });
  const rec = recordings.load(label);
  const ev = events(recordings, label);
  if (ev.length === 0) throw new CaptureError(`${label} has no events.json`);
  const fs = rec.fs;
  const filt = cueFilters(a);
  const n = rec.samples;
  // Counted before anything the rate decides: fewer than 2 samples measure
  // no rate.
  if (n < 2) refuseUnfilterable(`${label} holds ${samplesSaid(n)}`, n, fs, filt);
  refuseUnmeasuredRate(label, fs, "saccade needs one");
  // The baseline is taken from the samples before the cue, and the response
  // from those after it, so each holds at least one.
  refuseShortTime(label, "a pre_s", "the time each trial starts before its cue", a.pre_s!, fs, fewestSamples("saccade", "pre_s"));
  refuseShortTime(label, "a post_s", "the time each trial runs after its cue", a.post_s!, fs, fewestSamples("saccade", "post_s"));
  const npre = Math.trunc(num(a.pre_s!) * fs);
  const npost = Math.trunc(num(a.post_s!) * fs);
  refuseImpossibleFilters(label, fs, filt);
  refuseUnfilterable(`${label} holds ${samplesSaid(n)}`, n, fs, filt);
  const sig = channels(rec).map((x) => preprocess(x, fs, filt));
  const ht = rec.hostTime;
  const nch = rec.channels;
  const minDef = num(a.min_deflection_uv!);
  const maxCorr = num(a.max_corr!);
  const resp: [number, number] = [npre, npre + Math.trunc(1.0 * fs)];
  const conditions: [string, Value][] = [];
  const means = new Map<string, Float64Array[]>();
  const found = new Map<string, { channels: { deflection: number; sign: number; snr: number }[]; corr?: number | null }>();
  const leftOut: [string, Value][] = [];
  for (const cond of ["left", "right", "sham"]) {
    const [E, left] = cueEpochs(rec, ev, cond, npre, npost, sig, ht, settleS(a.hp!));
    if (E[0]!.length > 0 || anyLeftOut(left)) leftOut.push([cond, leftOutValue(left)]);
    if (E[0]!.length === 0) continue;
    const m = E.map(meanOverTrials);
    means.set(cond, m);
    const baseSd = N.mean(Float64Array.from(E, (e) => stdOfBlock(e, 0, npre)));
    const chs: { deflection: number; sign: number; snr: number }[] = [];
    for (let c = 0; c < nch; c++) {
      const w = m[c]!.slice(resp[0], resp[1]);
      const peak = w[N.argmax(w.map(Math.abs))]!;
      if (Number.isNaN(peak)) throw new CaptureError("cannot convert float NaN to integer");
      chs.push({ deflection: peak, sign: Math.sign(peak) + 0, snr: baseSd !== 0 ? Math.abs(peak) / baseSd : 0.0 });
    }
    const entry: [string, Value][] = [
      ["n_trials", I(E[0]!.length)],
      ["baseline_sd_uv", F(baseSd)],
      ["channels", J.array(chs.map((ch) => obj(["deflection_uv", F(ch.deflection)], ["sign", I(ch.sign)], ["snr", F(ch.snr)])))],
    ];
    const record: { channels: typeof chs; corr?: number | null } = { channels: chs };
    if (nch >= 2) {
      // No correlation where either mean response is constant over the
      // window, as a flat channel's is, or where the window holds one
      // sample, as a post_s of one sample makes it.
      const w0 = m[0]!.slice(resp[0], resp[1]);
      const w1 = m[1]!.slice(resp[0], resp[1]);
      const corr = w0.length < 2 ? J.NULL : reported(N.corrcoef(w0, w1));
      entry.push(["ch1_ch2_corr", corr]);
      record.corr = corr.kind === "null" ? null : num(corr);
    }
    found.set(cond, record);
    conditions.push([cond, J.object(entry)]);
  }
  const crit: [string, boolean][] = [];
  const haveLr = means.has("left") && means.has("right");
  const two = Math.min(2, nch);
  if (haveLr) {
    const L = found.get("left")!;
    const R = found.get("right")!;
    // A condition with no correlation, as where a channel is flat, has
    // not inverted.
    const inverted = (d: { corr?: number | null }): boolean => d.corr !== null && (d.corr ?? 0) <= maxCorr;
    crit.push(["polarity_inverted_left", inverted(L)]);
    crit.push(["polarity_inverted_right", inverted(R)]);
    let reverses = true;
    for (let c = 0; c < two; c++) reverses &&= L.channels[c]!.sign === -R.channels[c]!.sign && -R.channels[c]!.sign !== 0;
    crit.push(["sign_reverses_with_direction", reverses]);
    let big = true;
    for (const d of [L, R]) for (let c = 0; c < two; c++) big &&= Math.abs(d.channels[c]!.deflection) >= minDef;
    crit.push(["deflection_big_enough", big]);
  }
  if (found.has("sham")) crit.push(["sham_is_null", found.get("sham")!.channels.every((ch) => ch.snr < 3.0)]);
  const out: [string, Value][] = [
    ["label", T(label)],
    ["analysis", T("saccade")],
    ["analysis_version", I(ANALYSIS_VERSION)],
    ["capture_sha256", npzSha256(recordings, label)],
    ["params", obj(["pre_s", a.pre_s!], ["post_s", a.post_s!], ["filters", filtersValue(filt)], ["min_deflection_uv", a.min_deflection_uv!], ["max_corr", a.max_corr!])],
    ["left_out", J.object(leftOut)],
    ["conditions", J.object(conditions)],
    ["criteria", J.object(crit.map(([k, v]) => [k, B(v)]))],
    ["verdict", J.NULL],
  ];
  if (haveLr) {
    const required = ["polarity_inverted_left", "polarity_inverted_right", "sign_reverses_with_direction", "deflection_big_enough"];
    if (crit.some(([k]) => k === "sham_is_null")) required.push("sham_is_null");
    const holds = (k: string): boolean => crit.find(([n]) => n === k)![1];
    out[8] = ["verdict", T(required.every(holds) ? "PASS" : "FAIL")];
    out.push(["failed_criteria", J.array(required.filter((k) => !holds(k)).map(T))]);
  }
  return J.object(out);
}

function blinkOf(recordings: Collection, label: string, p: Params = {}): Value {
  const a = args("blink", p, {
    pre_s: F(0.5), post_s: F(1.5), ...CUE_FILTERS, min_pp_uv: F(100.0), min_detect_frac: F(0.9),
    max_latency_sd_s: F(0.25), sham_max_pp_ratio: F(1.5),
  });
  checkSettings(label, "blink", {
    min_pp_uv: a.min_pp_uv!, min_detect_frac: a.min_detect_frac!, max_latency_sd_s: a.max_latency_sd_s!,
    sham_max_pp_ratio: a.sham_max_pp_ratio!,
  });
  const rec = recordings.load(label);
  const ev = events(recordings, label);
  if (ev.length === 0) throw new CaptureError(`${label} has no events.json`);
  const fs = rec.fs;
  const filt = cueFilters(a);
  const n = rec.samples;
  // Counted before anything the rate decides: fewer than 2 samples measure
  // no rate.
  if (n < 2) refuseUnfilterable(`${label} holds ${samplesSaid(n)}`, n, fs, filt);
  refuseUnmeasuredRate(label, fs, "blink needs one");
  // The baseline is taken from the samples before the cue, and the response
  // from those after it, so each holds at least one.
  refuseShortTime(label, "a pre_s", "the time each trial starts before its cue", a.pre_s!, fs, fewestSamples("blink", "pre_s"));
  refuseShortTime(label, "a post_s", "the time each trial runs after its cue", a.post_s!, fs, fewestSamples("blink", "post_s"));
  const npre = Math.trunc(num(a.pre_s!) * fs);
  const npost = Math.trunc(num(a.post_s!) * fs);
  refuseImpossibleFilters(label, fs, filt);
  refuseUnfilterable(`${label} holds ${samplesSaid(n)}`, n, fs, filt);
  const sig = channels(rec).map((x) => preprocess(x, fs, filt));
  const ht = rec.hostTime;
  const resp: [number, number] = [npre, npre + Math.trunc(0.8 * fs)];
  const minPp = num(a.min_pp_uv!);
  const minDetect = num(a.min_detect_frac!);
  const maxLatSd = num(a.max_latency_sd_s!);
  const shamRatio = num(a.sham_max_pp_ratio!);
  interface Ch { meanPp: number; basePp: number; ratio: number; detected: number; lat: number; latSd: number; baseSd: number }
  const conditions: [string, Value][] = [];
  const found = new Map<string, Ch[]>();
  const leftOut: [string, Value][] = [];
  for (const cond of ["blink", "sham"]) {
    const [E, left] = cueEpochs(rec, ev, cond, npre, npost, sig, ht, settleS(a.hp!));
    if (E[0]!.length > 0 || anyLeftOut(left)) leftOut.push([cond, leftOutValue(left)]);
    if (E[0]!.length === 0) continue;
    const chs: Ch[] = E.map((trialsOfC) => {
      const pps = Float64Array.from(trialsOfC, (t) => N.ptp(t.subarray(resp[0], resp[1])));
      const basePp = Float64Array.from(trialsOfC, (t) => N.ptp(t.subarray(0, npre)));
      const baseSd = stdOfBlock(trialsOfC, 0, npre);
      const lat = Float64Array.from(trialsOfC, (t) => N.argmax(t.slice(resp[0], resp[1]).map(Math.abs)) / fs);
      let detected = 0;
      for (const v of pps) if (v >= minPp) detected++;
      const meanPp = N.mean(pps);
      const meanBase = N.mean(basePp);
      return {
        meanPp, basePp: meanBase, ratio: meanBase !== 0 ? meanPp / meanBase : 0.0,
        detected: detected / pps.length, lat: N.mean(lat), latSd: N.std(lat), baseSd,
      };
    });
    found.set(cond, chs);
    conditions.push([cond, obj(
      ["n_trials", I(E[0]!.length)],
      ["channels", J.array(chs.map((ch) => obj(
        ["mean_pp_uv", F(ch.meanPp)],
        ["baseline_pp_uv", F(ch.basePp)],
        ["pp_ratio", F(ch.ratio)],
        ["detected_frac", F(ch.detected)],
        ["latency_s", F(ch.lat)],
        ["latency_sd_s", F(ch.latSd)],
        ["baseline_sd_uv", F(ch.baseSd)],
      )))],
    )]);
  }
  const crit: [string, boolean][] = [];
  const b = found.get("blink");
  if (b !== undefined) {
    crit.push(["amplitude", b.some((ch) => ch.meanPp >= minPp)]);
    crit.push(["reliable_across_trials", b.some((ch) => ch.detected >= minDetect)]);
    crit.push(["consistent_latency", b.some((ch) => ch.latSd <= maxLatSd)]);
  }
  const sham = found.get("sham");
  if (sham !== undefined) crit.push(["sham_is_null", sham.every((ch) => ch.ratio < shamRatio && ch.meanPp < minPp)]);
  const out: [string, Value][] = [
    ["label", T(label)],
    ["analysis", T("blink")],
    ["analysis_version", I(ANALYSIS_VERSION)],
    ["capture_sha256", npzSha256(recordings, label)],
    ["params", obj(
      ["pre_s", a.pre_s!], ["post_s", a.post_s!], ["filters", filtersValue(filt)], ["min_pp_uv", a.min_pp_uv!],
      ["min_detect_frac", a.min_detect_frac!], ["max_latency_sd_s", a.max_latency_sd_s!], ["sham_max_pp_ratio", a.sham_max_pp_ratio!],
    )],
    ["left_out", J.object(leftOut)],
    ["conditions", J.object(conditions)],
    ["criteria", J.object(crit.map(([k, v]) => [k, B(v)]))],
    ["verdict", J.NULL],
  ];
  if (crit.length > 0) {
    const required = ["amplitude", "reliable_across_trials", "consistent_latency", "sham_is_null"].filter((k) => crit.some(([n]) => n === k));
    const holds = (k: string): boolean => crit.find(([n]) => n === k)![1];
    out[8] = ["verdict", T(required.every(holds) ? "PASS" : "FAIL")];
    out.push(["failed_criteria", J.array(required.filter((k) => !holds(k)).map(T))]);
  }
  return J.object(out);
}

function jawOf(recordings: Collection, label: string, p: Params = {}): Value {
  const a = args("jaw", p, { window_s: F(1.0), lo_hz: F(20.0), hi_hz: F(200.0), min_ratio: F(10.0), alpha: F(0.001) });
  checkSettings(label, "jaw", { lo_hz: a.lo_hz!, hi_hz: a.hi_hz!, min_ratio: a.min_ratio!, alpha: a.alpha! });
  const rec = recordings.load(label);
  const ev = events(recordings, label);
  if (ev.length === 0) throw new CaptureError(`${label} has no events.json`);
  const fs = rec.fs;
  // Counted before anything the rate decides: fewer than 2 samples measure
  // no rate.
  if (rec.samples < 2) {
    throw new CaptureError(`${label} holds ${samplesSaid(rec.samples)}, too few for a spectrum: a spectrum needs at least 2 samples.`);
  }
  refuseUnmeasuredRate(label, fs, "jaw needs one");
  refuseShortTime(label, "a window_s", "the length of each rest window and each clench window", a.window_s!, fs,
    fewestSamples("jaw", "window_s"), ", the fewest a window's power is taken from");
  // A band reaching half the rate is cut 1 Hz below it.
  const hiHz = checkRateSettings(label, "jaw", { lo_hz: a.lo_hz!, hi_hz: a.hi_hz! }, fs).hi_hz!;
  const n = Math.trunc(num(a.window_s!) * fs);
  const ht = rec.hostTime;
  const raw = channels(rec);
  const lo = num(a.lo_hz!);
  const hi = num(hiHz);
  const bandPower = (x: Float64Array): number => {
    if (x.length < 32) return NaN;
    const s = design(() => N.welch(demean(x), fs, Math.min(256, x.length)));
    return band(s, lo, hi);
  };
  // A window that holds a break is left out: a spectrum taken across it
  // would splatter power into the band.
  const unbroken = unbrokenIn(rec.index);
  const pairs: [number, number][][] = raw.map(() => []);
  const left = noneLeftOut();
  let cued = 0;
  for (const e of ev) {
    if (J.asStr(J.field(e, "cond")) !== "clench") continue;
    cued++;
    const i = N.searchsorted(ht, num(J.field(e, "host_time")));
    if (i - n < 0 || i + n > ht.length) { left.outside_recording++; continue; }
    if (!(unbroken(i - n, i) && unbroken(i, i + n))) { left.gap_inside++; continue; }
    raw.forEach((x, c) => pairs[c]!.push([bandPower(slice(x, i - n, i)), bandPower(slice(x, i, i + n))]));
  }
  const minRatio = num(a.min_ratio!);
  const alpha = num(a.alpha!);
  const chans: Value[] = [];
  const good: { ratio: number | null; clench: number; p: number | null }[] = [];
  for (let c = 0; c < rec.channels; c++) {
    const pr = pairs[c]!.filter(([r, k]) => Number.isFinite(r) && Number.isFinite(k));
    if (pr.length < 3) {
      chans.push(obj(["n_trials", I(pr.length)], ["note", T("too few trials")]));
      continue;
    }
    const rest = Float64Array.from(pr, (q) => q[0]);
    const clench = Float64Array.from(pr, (q) => q[1]);
    const pval = S.wilcoxonGreater(clench, rest).pvalue;
    const restMean = N.mean(rest);
    const clenchMean = N.mean(clench);
    let wins = 0;
    for (let i = 0; i < pr.length; i++) if (clench[i]! > rest[i]!) wins++;
    const ratio = restMean !== 0 ? reported(clenchMean / restMean) : J.NULL;
    const pv = reported(pval);
    good.push({ ratio: ratio.kind === "null" ? null : num(ratio), clench: clenchMean, p: pv.kind === "null" ? null : num(pv) });
    chans.push(obj(
      ["n_trials", I(pr.length)],
      ["rest", F(restMean)],
      ["clench", F(clenchMean)],
      ["ratio", ratio],
      ["wins", I(wins)],
      ["p", pv],
      ["p_floor", F(2.0 ** -pr.length)],
    ));
  }
  const crit: [string, boolean][] = [];
  if (good.length > 0) {
    // A ratio of null is infinite where there is clench power, and 0 over
    // 0, no evidence of a clench, where there is none.
    crit.push(["power_ratio", good.some((ch) => (ch.ratio === null ? ch.clench > 0 : ch.ratio >= minRatio))]);
    crit.push(["significant", good.some((ch) => ch.p !== null && ch.p < alpha)]);
  }
  const out: [string, Value][] = [
    ["label", T(label)],
    ["analysis", T("jaw")],
    ["analysis_version", I(ANALYSIS_VERSION)],
    ["capture_sha256", npzSha256(recordings, label)],
    ["params", obj(["window_s", a.window_s!], ["band", J.array([a.lo_hz!, hiHz])], ["min_ratio", a.min_ratio!], ["alpha", a.alpha!], ["test", T("wilcoxon_paired_trials")])],
    ["left_out", cued > 0 ? obj(["clench", leftOutValue(left)]) : J.object([])],
    ["channels", J.array(chans)],
    ["criteria", J.object(crit.map(([k, v]) => [k, B(v)]))],
    ["verdict", J.NULL],
  ];
  if (good.length > 0) {
    out[8] = ["verdict", T(crit.every(([, v]) => v) ? "PASS" : "FAIL")];
    out.push(["failed_criteria", J.array(crit.filter(([, v]) => !v).map(([k]) => T(k)))]);
  }
  return J.object(out);
}

function bergerOf(recordings: Collection, label: string, p: Params = {}): Value {
  const a = args("berger", p, {
    epoch_s: F(2.0), ppmax: F(150.0), settle: J.NULL, alpha_lo: F(8.0), alpha_hi: F(13.0), ref_lo: F(4.0), ref_hi: F(30.0),
    ...FILTER_DEFAULTS, allow_pseudoreplication: B(false),
  });
  checkSettings(label, "berger", {
    ppmax: a.ppmax!, alpha_lo: a.alpha_lo!, alpha_hi: a.alpha_hi!, ref_lo: a.ref_lo!, ref_hi: a.ref_hi!,
    allow_pseudoreplication: a.allow_pseudoreplication!,
  });
  const rec = recordings.load(label);
  const all = events(recordings, label);
  if (all.length === 0) throw new CaptureError(`${label} has no events.json -- not a cued experiment`);
  // Each block starts at an event of the condition closed or open. An event
  // of any other condition, such as a marker a person stamped, does not
  // start one, as saccade, blink and jaw read only the events of their own
  // conditions.
  const ev = all.filter((e) => {
    const cond = J.asStr(J.field(e, "cond"));
    return cond === "closed" || cond === "open";
  });
  if (ev.length === 0) {
    const conds = [...new Set(all.map((e) => J.asStr(J.field(e, "cond"))).filter((c): c is string => c !== null))].sort(J.compareKeys);
    const listed = conds.length > 1 ? `${conds.slice(0, -1).join(", ")} and ${conds[conds.length - 1]}` : conds.join("");
    throw new CaptureError(
      `${label} has no event of the condition closed or open, the blocks berger compares: ` +
        (conds.length > 0 ? `its events are of the condition${conds.length > 1 ? "s" : ""} ${listed}.` : "none of its events names a condition."),
    );
  }
  const meta = rec.manifest;
  const settleV = a.settle!.kind === "null" ? J.get(meta, "settle") ?? F(3.0) : a.settle!;
  const blockV = J.get(meta, "block") ?? F(20.0);
  const block = num(blockV);
  const ht = rec.hostTime;
  const fs = rec.fs;
  const filt: Filters = { hp: a.hp!, lp: a.lp!, notch: a.notch!, notch_q: a.notch_q!, trim: F(0.5) };
  // Counted before anything the rate decides: fewer than 2 samples measure
  // no rate.
  if (rec.samples < 2) {
    throw new CaptureError(`${label} holds ${samplesSaid(rec.samples)}, too few for a spectrum: a spectrum needs at least 2 samples.`);
  }
  refuseUnmeasuredRate(label, fs, "berger needs one");
  refuseShortTime(label, "an epoch_s", "the length of each epoch", a.epoch_s!, fs, fewestSamples("berger", "epoch_s"));
  if (!(isReal(settleV) && 0 <= num(settleV) && num(settleV) < block)) {
    throw new CaptureError(
      `${label} cannot be analyzed with a settle of ${amount(settleV, "s")}: settle, the time left out at the ` +
        `start of each block, is at least 0 s and below ${g(block)} s, the recording's block length.`,
    );
  }
  checkRateSettings(label, "berger", { alpha_lo: a.alpha_lo!, alpha_hi: a.alpha_hi!, ref_lo: a.ref_lo!, ref_hi: a.ref_hi! }, fs);
  refuseImpossibleFilters(label, fs, filt);
  const settle = num(settleV);
  const epochS = num(a.epoch_s!);
  const ppmax = num(a.ppmax!);
  const alphaLo = num(a.alpha_lo!);
  const alphaHi = num(a.alpha_hi!);
  const refLo = num(a.ref_lo!);
  const refHi = num(a.ref_hi!);
  const params = obj(
    ["epoch_s", a.epoch_s!], ["ppmax", a.ppmax!], ["settle", settleV], ["alpha", J.array([a.alpha_lo!, a.alpha_hi!])],
    ["reference", J.array([a.ref_lo!, a.ref_hi!])], ["filters", obj(["hp", filt.hp], ["lp", filt.lp], ["notch", filt.notch], ["notch_q", filt.notch_q])],
    ["allow_pseudoreplication", a.allow_pseudoreplication!],
  );
  let pseudoreplicated = false;
  const chans: Value[] = [];
  for (let c = 0; c < rec.channels; c++) {
    interface Block { rep: Value; cond: Value; abs: number; rel: number; ref: number; clean: number; total: number }
    const perBlock: Block[] = [];
    const pooled: Record<string, number[]> = { closed: [], open: [] };
    for (const e of ev) {
      const t = num(J.field(e, "host_time"));
      const sel: number[] = [];
      for (let i = 0; i < ht.length; i++) if (ht[i]! >= t + settle && ht[i]! < t + block) sel.push(i);
      if (sel.length < Math.trunc(epochS * fs) + Math.trunc(fs)) continue;
      const [eps, tot] = cleanEpochs(rec, c, a.epoch_s!, ppmax, filt, sel);
      const rel: number[] = [];
      const absol: number[] = [];
      const refs: number[] = [];
      for (const seg of eps) {
        const s = design(() => N.welch(seg, fs, seg.length));
        const numer = band(s, alphaLo, alphaHi);
        const den = band(s, refLo, refHi);
        absol.push(numer);
        refs.push(den);
        rel.push(den !== 0 ? (numer / den) * 100 : 0.0);
      }
      if (eps.length === 0) continue;
      const cond = J.field(e, "cond");
      pooled[J.asStr(cond)!]!.push(...rel);
      // A block whose event carries no rep is analyzed with rep null.
      const rep = J.field(e, "rep");
      perBlock.push({
        rep, cond, abs: N.mean(Float64Array.from(absol)), rel: N.mean(Float64Array.from(rel)),
        ref: N.mean(Float64Array.from(refs)), clean: eps.length, total: tot,
      });
    }
    const res: [string, Value][] = [["blocks", J.array(perBlock.map((b) => obj(
      ["rep", b.rep], ["cond", b.cond], ["alpha_abs", F(b.abs)], ["alpha_rel", F(b.rel)],
      ["reference_abs", F(b.ref)], ["clean", I(b.clean)], ["total", I(b.total)],
    )))]];
    const tests = new Map<string, { p: number | null; ratio: number }>();
    for (const [key, tag] of [["abs", "absolute"], ["rel", "relative"]] as const) {
      const cl0 = perBlock.filter((b) => J.asStr(b.cond) === "closed").map((b) => b[key]);
      const op0 = perBlock.filter((b) => J.asStr(b.cond) === "open").map((b) => b[key]);
      const n = Math.min(cl0.length, op0.length);
      if (n < 2) {
        res.push([tag, obj(["n_blocks", I(n)], ["note", T("too few blocks to test")])]);
        continue;
      }
      const cl = Float64Array.from(cl0.slice(0, n));
      const op = Float64Array.from(op0.slice(0, n));
      const pw = S.wilcoxonGreater(cl, op).pvalue;
      const tt = S.ttestRel(cl, op);
      const pt = tt.statistic > 0 ? tt.pvalue / 2 : 1 - tt.pvalue / 2;
      const diff = cl.map((v, i) => v - op[i]!);
      // No effect size where every difference is the same: its spread is 0.
      const d = reported(N.mean(diff) / N.std(diff, 1));
      const clMean = N.mean(cl);
      const opMean = N.mean(op);
      let wins = 0;
      for (let i = 0; i < n; i++) if (cl[i]! > op[i]!) wins++;
      const ratio = opMean !== 0 ? clMean / opMean : 0.0;
      const p = reported(pw);
      tests.set(tag, { p: p.kind === "null" ? null : num(p), ratio });
      res.push([tag, obj(
        ["n_blocks", I(n)], ["closed", F(clMean)], ["open", F(opMean)], ["ratio", F(ratio)], ["wins", I(wins)],
        ["test", T("wilcoxon_signed_rank_paired_blocks")], ["p", p], ["p_paired_t", reported(pt)], ["cohens_dz", d],
        ["p_floor", F(2.0 ** -n)],
      )]);
    }
    if (J.truthy(a.allow_pseudoreplication!) && pooled.closed!.length > 0 && pooled.open!.length > 0) {
      const pp = S.mannwhitneyuGreater(pooled.closed!, pooled.open!).pvalue;
      res.push(["pooled_epochs", obj(
        ["n_closed", I(pooled.closed!.length)], ["n_open", I(pooled.open!.length)], ["p", F(pp)],
        ["test", T("mannwhitneyu_pooled_epochs")],
        ["warning", T("PSEUDOREPLICATED: epochs within a block are not independent; this p-value is inflated")],
      )]);
      pseudoreplicated = true;
    }
    const ab = tests.get("absolute");
    const rl = tests.get("relative");
    if (ab !== undefined && rl !== undefined) {
      const refc = perBlock.filter((b) => J.asStr(b.cond) === "closed").map((b) => b.ref);
      const refo = perBlock.filter((b) => J.asStr(b.cond) === "open").map((b) => b.ref);
      const nref = Math.min(refc.length, refo.length);
      const opRef = nref > 0 ? N.mean(Float64Array.from(refo.slice(0, nref))) : 0;
      const refRatio = nref > 0 && opRef !== 0 ? reported(N.mean(Float64Array.from(refc.slice(0, nref))) / opRef) : J.NULL;
      res.push(["reference_band_ratio", refRatio]);
      if (rl.p !== null && ab.p !== null && rl.p < 0.05 && ab.p > 0.05) {
        const refBand = `${J.pyStr(a.ref_lo!)}-${J.pyStr(a.ref_hi!)} Hz reference band`;
        const changed = refRatio.kind !== "null"
          ? `The ${refBand} changed by ${fixed(num(refRatio), 3)}x.`
          : `The ratio of the ${refBand}'s mean power, closed over open, could not be computed.`;
        res.push(["denominator_artifact", B(true)]);
        res.push(["interpretation", T(
          "Relative alpha moved but ABSOLUTE alpha did not " +
            `(ratio ${fixed(ab.ratio, 3)}, p=${fixed(ab.p, 3)}). ${changed} This is a denominator effect, not a ` +
            "Berger effect.",
        )]);
      } else {
        res.push(["denominator_artifact", B(false)]);
      }
    }
    const allc = cleanEpochs(rec, c, a.epoch_s!, ppmax, filt, null)[0];
    if (allc.length > 0) {
      const s = N.welchEpochs(allc, fs);
      const pkLo = alphaLo - 1.0;
      const pkHi = alphaHi + 1.0;
      res.push(["slope_2_40", reported(spectralSlope(s, 2.0, 40.0, [pkLo, pkHi]))]);
      const kk = (i: number): boolean => {
        const f = s.f[i]!;
        return f >= 2 && f <= 40 && !(f >= pkLo && f <= pkHi) && s.p[i]! > 0;
      };
      const fk = where(s.f, kk);
      if (fk.length >= 8) {
        const [slope, intercept] = N.polyfit1(fk.map(Math.log10), where(s.p, kk).map(Math.log10));
        const ka = (i: number): boolean => s.f[i]! >= pkLo && s.f[i]! <= pkHi;
        const fa = where(s.f, ka);
        const pa = where(s.p, ka);
        if (fa.length === 0) throw new CaptureError("zero-size array to reduction operation maximum which has no identity");
        const resid = pa.map((v, i) => v / 10 ** ((0 * Math.log10(fa[i]!) + slope) * Math.log10(fa[i]!) + intercept));
        const top = N.argmax(resid);
        const most = extremes(resid)[0];
        res.push(["alpha_peak_ratio", F(most)]);
        res.push(["alpha_peak_hz", F(fa[top]!)]);
        res.push(["has_alpha_peak", B(most >= 1.5)]);
      }
      res.push(["n_epochs", I(allc.length)]);
    }
    chans.push(J.object(res));
  }
  const nTests = 2 * rec.channels;
  return obj(
    ["label", T(label)],
    ["analysis", T("berger")],
    ["analysis_version", I(ANALYSIS_VERSION)],
    ["params", params],
    ["capture_sha256", npzSha256(recordings, label)],
    ["block", blockV],
    ["settle", settleV],
    ["pseudoreplicated", B(pseudoreplicated)],
    ["n_reps", I(Math.floor(ev.length / 2))],
    ["channels", J.array(chans)],
    ["multiple_comparisons", obj(["n_tests", I(nTests)], ["bonferroni_alpha", F(0.05 / nTests)])],
  );
}

// --- the registry ------------------------------------------------------------------

/** A number parameter, with the range `range` gives it, its keys in the library's order. */
const n = (name: string, value: number, range: Omit<Param, "name" | "type" | "default" | "choices"> = {}): Param =>
  ({ name, type: "number", default: value, ...range });
const select = (name: string): Param => ({ name, type: "select", default: 0, choices: [0, 1] });

/** A filter's four settings, each stage turned off by 0 and its frequency bounded by the rate. */
const filterParams = (hp: number, lp: number, notch: number, notchQ: number): Param[] => [
  n("hp", hp, { off: 0, above: 0, rate: "below half" }),
  n("lp", lp, { off: 0, above: 0, rate: "below half" }),
  n("notch", notch, { off: 0, above: 0, rate: "at most half" }),
  n("notch_q", notchQ, { above: 0 }),
];

/** The time a cued trial starts before its cue and runs after it, each holding at least 1 sample. */
const aroundCue = (): Param[] => [n("pre_s", 0.5, { above: 0, samples: 1 }), n("post_s", 1.5, { above: 0, samples: 1 })];

/** Each analysis's parameters, as its registry entry gives them and as its own settings are checked against them. */
const PARAMS: Readonly<Record<string, readonly Param[]>> = {
  summary: [n("line_hz", 60.0, { min: 0 }), n("nperseg", 2048, { min: 2 }),
    n("slope_lo", 2.0, { above: 0, below: "slope_hi" }), n("slope_hi", 40.0, { above: 0 })],
  psd: [select("filtered"), n("fmax", 100.0, { above: 0 }), n("nperseg", 2048, { min: 2 }), ...filterParams(1.0, 45.0, 60.0, 30.0)],
  contact: [n("dc_min_uv", 1000.0), n("ac_min_uv", 2.0)],
  saccade: [...aroundCue(), ...filterParams(0.1, 10.0, 60.0, 30.0),
    n("min_deflection_uv", 50.0, { min: 0 }), n("max_corr", -0.5, { min: -1, max: 1 })],
  blink: [...aroundCue(), ...filterParams(0.1, 10.0, 60.0, 30.0),
    n("min_pp_uv", 100.0, { min: 0 }), n("min_detect_frac", 0.9, { min: 0, max: 1 }),
    n("max_latency_sd_s", 0.25, { min: 0 }), n("sham_max_pp_ratio", 1.5, { min: 0 })],
  jaw: [n("window_s", 1.0, { above: 0, samples: 32 }),
    n("lo_hz", 20.0, { min: 0, below: "hi_hz" }), n("hi_hz", 200.0, { above: 0, rate: "lowered" }),
    n("min_ratio", 10.0, { min: 0 }), n("alpha", 0.001, { min: 0, max: 1 })],
  berger: [n("epoch_s", 2.0, { above: 0, samples: 1 }), n("ppmax", 150.0, { min: 0 }),
    n("alpha_lo", 8.0, { min: 0, below: "alpha_hi", rate: "below half" }), n("alpha_hi", 13.0, { above: 0 }),
    n("ref_lo", 4.0, { min: 0, below: "ref_hi", rate: "below half" }), n("ref_hi", 30.0, { above: 0 }),
    ...filterParams(1.0, 45.0, 60.0, 30.0), select("allow_pseudoreplication")],
};

/**
 * Every analysis, by name, in the library's order: what it does, whether it
 * needs cues, a line about it, and the parameters a form offers. An
 * application adds its own here, as it adds one to the library's registry,
 * with `several: true` when it takes a list of the selected recordings'
 * labels and is run once on all of them. Every entry of the package's takes
 * one label.
 */
export const ANALYSES = new Map<string, Spec>([
  ["summary", {
    fn: summaryOf, needsEvents: false,
    doc: "DC/AC split, band powers, spectral slope, noise floor (unfiltered)",
    params: PARAMS.summary!,
  }],
  ["psd", {
    fn: psdCurve, needsEvents: false,
    doc: "Power spectral density (multi-select overlays)",
    params: PARAMS.psd!,
  }],
  ["contact", {
    fn: contactOf, needsEvents: false,
    doc: "Electrode contact: the device's lead-off judgment for each channel, with its DC offset and AC level",
    params: PARAMS.contact!,
  }],
  ["saccade", {
    fn: saccadeOf, needsEvents: true,
    doc: "Ocular dipole in cued saccades, ch1 and ch2 on electrodes at the left and right temples: tests whether the two channels invert and their signs reverse with direction",
    params: PARAMS.saccade!,
  }],
  ["blink", {
    fn: blinkOf, needsEvents: true,
    doc: "Cued blinks: amplitude, per-trial reliability, latency consistency, sham null",
    params: PARAMS.blink!,
  }],
  ["jaw", {
    fn: jawOf, needsEvents: true,
    doc: "Cued jaw clench: 20-200 Hz EMG, paired over trials",
    params: PARAMS.jaw!,
  }],
  ["berger", {
    fn: bergerOf, needsEvents: true,
    doc: "Alpha closed-vs-open: paired block-level test, absolute + relative, 1/f peak",
    params: PARAMS.berger!,
  }],
]);

// --- the collection ----------------------------------------------------------------

/**
 * The recordings named by `labels`, each as the library's `list_captures`
 * lists it: its manifest with its `label`, whether it has cues, and whether
 * its bytes are the ones it was finalized with, as `ok`, `incomplete`,
 * `unverified`, `ABORTED` or `FAILED`, with the first thing a reader should
 * know. A `Collection` keeps no modification times, so the recordings come
 * back in the order of `labels`: the library lists them newest first, and a
 * caller that wants that order passes the labels newest first. A name that
 * is not a recording's label, and a manifest that is not an object, are
 * left out, as the library leaves them out.
 */
export function listCaptures(recordings: Collection, labels: readonly string[]): Value {
  const out: Value[] = [];
  for (const label of labels) {
    try {
      checkLabel(label);
    } catch {
      continue;
    }
    let d: Value;
    try {
      const text = readText(recordings, `${label}.meta.json`);
      if (text === null) continue;
      d = J.loads(text);
      if (d.kind !== "object") continue;
    } catch {
      continue;
    }
    d = J.copyObject(d);
    J.set(d, "label", T(label));
    J.set(d, "has_events", B(recordings.file(`${label}.events.json`) !== null));
    const { ok, problems } = recordings.verify(label);
    const first = problems[0] ?? "";
    const integrity = ok && problems.length === 0 ? "ok"
      : ok && first.startsWith("provenance incomplete") ? "incomplete"
      : ok ? "unverified"
      : J.truthy(J.field(d, "aborted")) ? "ABORTED" : "FAILED";
    J.set(d, "integrity", T(integrity));
    J.set(d, "integrity_detail", T(first));
    out.push(d);
  }
  return J.array(out);
}
