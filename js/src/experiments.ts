/**
 * The experiments, and the recording every one of them makes, as the
 * Python library's `experiments` module registers and runs them: the same
 * eight protocols with the same parameters, the same cues in the same order
 * at the same times, into the same recordings, held to the library's own
 * runs by the fixtures in `contract/experiments` and `contract/together`.
 *
 * An experiment is an async function registered with `experiment`, under a
 * name and with the parameters a form shows for it. It is handed
 * `(dev, emit, params, ctx)`: the connection or connections it runs on,
 * `emit`, the function its messages are sent through, its parameters, and a
 * `RunCtx`, which stamps its events and whose `stop`, once set, asks it to
 * stop. An experiment never sets an instrument
 * setting. It records at whatever the device is set to and writes that into
 * the manifest. An experiment may declare an `enforced` precondition, and
 * no run checks one: an application that wants a run refused when its
 * parameters break the precondition calls `instrument.checkEnforced` before
 * starting the run. The package never starts a run on its own. `REGISTRY`
 * holds the package's experiments and any an application registers, and
 * `DaisyChain.run` runs any of them.
 *
 * ONE DEVICE OR SEVERAL. Every experiment here, and every step it is built
 * from, takes one connection or an array of them: the run's own, or any
 * connected since the run started. Each device handed to a step is recorded
 * as its own recording. Handed one connection outside a run, a step records
 * it into that device's own session recorder. In a run, a step records into
 * the recorder the run keeps, which keeps one recording for each device a
 * step hands it.
 *
 * A RUN. `DaisyChain.run` runs a registered experiment on the devices it
 * names, or on every connected device. Its function is handed the
 * connection itself on a run of one, and a new array of the connections, in
 * the order given, on a run of several. Several runs may go at once on
 * different devices, each with its own `RunCtx`, so its own `stop`, markers
 * and events, and its own recordings. An experiment's own code hands groups
 * of its devices to the package's experiments, in turn or at the same time,
 * passing on the `emit` and `ctx` it was handed, `params` here carrying the
 * parameters of both:
 *
 * ```ts
 * import { experiments } from "@intomind/sdk";
 *
 * const berger = experiments.REGISTRY.get("berger")!.fn;
 * const baseline = experiments.REGISTRY.get("baseline")!.fn;
 *
 * experiments.experiment("in_turn", "Two groups in turn.", [], async (dev, emit, p, ctx) => {
 *   const all = Array.isArray(dev) ? dev : [dev];                 // a run of one device hands the device itself
 *   const first = await berger(all.slice(0, 2), emit, p, ctx);    // the others rest
 *   const second = await berger(all.slice(2), emit, p, ctx);
 *   return [...first, ...second];
 * });
 *
 * experiments.experiment("at_once", "Two groups at once.", [], async (dev, emit, p, ctx) => {
 *   const all = Array.isArray(dev) ? dev : [dev];
 *   const [a, b] = await Promise.all([berger(all.slice(0, 2), emit, p, ctx), baseline(all.slice(2), emit, p, ctx)]);
 *   return [...a, ...b];
 * });
 * ```
 *
 * A device that is not handed to a step is never reset, stopped, armed or
 * saved by it, and no consumer of its samples is added, so a device that is
 * streaming between steps goes on streaming, unrecorded. Setting the run's
 * `ctx.stop` ends every step of the run. An experiment waits for every step
 * it starts. JavaScript cannot end a promise from outside, so a step that
 * fails does not end the others beside it: an experiment that wants that
 * sets the run's `ctx.stop`, with `ctx.stop.set()`, and each ends at once
 * and saves what it recorded. A step still running
 * when its experiment returns keeps its devices until it ends, and is not
 * counted in what the run gives back.
 *
 * WHAT A STEP DOES TO A DEVICE. A step starts the streams of the devices it
 * records that were not running, and stops only those it started. A device
 * another step or another run is recording is refused, in words that name
 * it, before anything is sent to any device of the step, since a recording
 * restarts the count of a device that can restart it. A setup that gives an IntoMind
 * device a gain other than the gain in force, which its Status states, is
 * refused in words naming both gains before anything records. A device
 * whose link is down when a step starts is not recorded, and the step says
 * so and goes on with the others, unless every device's link is down: then
 * the step fails with the error raised by the first device it writes to.
 * One whose link drops mid-step has its recording saved with `aborted` set
 * to "link_lost" while the others carry on. One that ends its stream itself
 * because USB power appeared is saved with every sample up to that moment,
 * with `ended_by_usb` in its manifest. A failure while arming leaves no
 * device armed, held or started by it. A parameter of the package's
 * experiments that is not given, that cannot be read as the number or the
 * whole number it needs, or whose text the experiment cannot read, such as
 * a line of `steps` with no "|", is refused, in words that name it, before
 * anything is sent to any device. Infinity and NaN are refused as any other
 * value that is not a number is, a whole number given with a fraction is
 * cut to its whole part, and no number is checked against a range. A step
 * whose recording cannot be saved still stops the streams it started and
 * lets its devices go.
 *
 * WHAT EACH RECORDING SAYS. Every recording a run makes carries `run`, the
 * experiment's registered name and when the run started, the same in every
 * one, and `together`, the names of the other recordings its step saved at
 * the same moment. Each device's setup in the run's `cfgs`, such as who
 * wears it, lands in its own manifest. A cue a step stamps belongs to the
 * devices that step handles, and is written only into their recordings.
 * Every event stamped anywhere else, a person's marker, an outside source's
 * event, or a cue an experiment's own code stamps between steps, is written
 * into every recording the run saves after it is stamped. A run on one
 * device saves its recording under the name its step gives it. A run of
 * several adds each device's id to that name, and where the name gives a
 * gain, puts in it the gain the device's own setup gives, or `na` where its
 * setup gives none. A name a run of the same chain already saved has a number added. A
 * run gives back the name of every recording it saved, and every message it
 * sends carries `devices`, the id of each device it concerns.
 *
 * A run says what is happening as it happens through `emit`: a note, each
 * cue's words, progress, the time elapsed, the marker buttons, and each
 * recording saved, the messages the library sends. Where the library
 * writes a recording's files into its folder, the message that says it was
 * saved carries them, for an application to keep however it keeps files, and
 * `capture/node`'s `writeRecording` puts them in a folder. What the library
 * refuses with one of Python's own exceptions is refused here with the
 * package's, its words kept: `Invalid` for a value, `TypeError` for a type,
 * and `SessionError` for a device that did not answer ok or a refusal of
 * the library's own.
 *
 * ```ts
 * import { experiments } from "@intomind/sdk";
 *
 * const spec = experiments.REGISTRY.get("berger")!;
 * const show = (m) => {
 *   if (m.type === "saved") keep(m.recording);
 * };
 * const labels = await spec.fn(link, show, params, new experiments.RunCtx(show, { gain: 24 }));
 * ```
 */

import { CHOICES } from "./audio.ts";
import { utc } from "./capture/record.ts";
import { fixed } from "./capture/pyfmt.ts";
import * as J from "./capture/pyjson.ts";
import type { Json, Value } from "./capture/pyjson.ts";
import { blockedFolds, trainHead } from "./model.ts";
import { UNDETERMINED, targets } from "./training.ts";
import { BIAS_MODES, Invalid, toHex, type ModelInfo } from "./protocol.ts";
import { Random } from "./pyrandom.ts";
import {
  arm,
  cue,
  finish,
  holdFor,
  jsonable,
  labels,
  ready,
  recordFor,
  recorded,
  runtime,
  several,
  step,
  tell,
  utcMs,
  windowsOf,
  gainNamed,
  idOfUnit,
  named,
  listed,
  type Armed,
  type Done,
  type Unit,
} from "./run.ts";
import { SessionError } from "./session.ts";
import type { Connection } from "./web-bluetooth.ts";

// --- what a run is ---------------------------------------------------------------

/**
 * One message a run sends while it runs: its `type`, and its fields, as the
 * library's. A `saved` message carries the recording's `label` and its
 * manifest as `meta`, as the library's does, and `recording`, the files the
 * library writes into its folder itself, as `Session.saveRecording` hands
 * them back.
 */
export interface Message {
  /** What kind of message: note, cue, progress, elapsed, markers or saved. In a run, every message also carries `devices`, the id of each device it concerns. */
  readonly type: string;
  /** The message's fields. */
  readonly [field: string]: unknown;
}

/** Where a run's messages go. */
export type Emit = (message: Message) => void | Promise<void>;

/** The parameters a run was given, by name, as a form sends them. */
export type Params = Readonly<Record<string, unknown>>;

/** A parameter an experiment declares, as a form draws it. */
export interface Param {
  /** The parameter's name, the key a run reads. */
  readonly name: string;
  /** How a form takes it: number, text, textarea or select. */
  readonly type: string;
  /** What it is unless a person changes it. */
  readonly default: unknown;
  /** For a select, what may be chosen. */
  readonly choices?: readonly unknown[];
}

/**
 * What an experiment does: drive the device over this connection, or each
 * device over these connections, with these parameters, and hand back the
 * names of the recordings it made. In a run of one it is handed the
 * connection itself, and in a run of several a new array of them. A device
 * here is a browser connection to an IntoMind device.
 */
export type ExperimentFn = (
  dev: Connection | Connection[],
  emit: Emit,
  params: Params,
  ctx: RunCtx,
) => Promise<string[]>;

/** One registered experiment, as the library's registry holds it. */
export interface Spec {
  /** Its name, the key it is registered under. */
  readonly name: string;
  /** What it is, in a sentence or two. */
  readonly doc: string;
  /** The parameters it takes. */
  readonly params: readonly Param[];
  /** What it does. */
  readonly fn: ExperimentFn;
  /** A genuinely unsafe precondition that refuses the run, by parameter. None declares one. */
  readonly enforced: Value;
  /** The analysis it was registered with, fixed before any data, stamped into the manifest by the application. */
  readonly prereg: Value;
}

/** Every registered experiment, by name, in the order registered: the library's eight, then any an application adds. */
export const REGISTRY = new Map<string, Spec>();

/**
 * Register an experiment, as the library's `experiment` decorator does: its
 * parameters, an optional `enforced` precondition, and its preregistered
 * analysis. It declares no instrument setting, because a recording captures
 * whatever the instrument is set to.
 */
export function experiment(
  name: string,
  doc: string,
  params: readonly Param[],
  fn: ExperimentFn,
  options: { readonly enforced?: Json; readonly prereg?: Json } = {},
): Spec {
  const asObject = (x: Json): Value => {
    const v = J.fromPlain(x ?? null);
    if (!J.truthy(v)) return J.object([]);
    if (v.kind !== "object") throw new TypeError(`an experiment's ${J.jsRepr(v)} is not an object`);
    return J.copyObject(v);
  };
  const spec: Spec = { name, doc, params, fn, enforced: asObject(options.enforced), prereg: asObject(options.prereg) };
  REGISTRY.set(name, spec);
  return spec;
}

/** One of the package's own experiments, registered as the library registers its own. */
function register(
  name: string,
  doc: string,
  params: readonly Param[],
  fn: ExperimentFn,
  options: { readonly enforced?: Json; readonly prereg?: Json },
): void {
  experiment(name, doc, params, fn, options);
}

/** A flag a run checks: the library's `asyncio.Event`, as far as a run uses one. */
class Flag {
  #set = false;

  set(): void {
    this.#set = true;
  }

  clear(): void {
    this.#set = false;
  }

  isSet(): boolean {
    return this.#set;
  }
}

/**
 * A live handle on a running experiment: stop it, or mark an event.
 *
 * `cfg` is the setup the run records with, given when the context is made,
 * as `DaisyChain.run` makes one from the first of its `cfgs` where no
 * context is passed: what the instrument was set to, and who wears it. It
 * goes into the manifest as it is. An experiment never configures the
 * device it records from. Every event
 * carries both host clocks: `host_time`, the record's timebase, and `mono`,
 * the witness to whether the wall clock moved underneath it. Events are
 * held as the library holds them, so the events file is the same bytes;
 * `capture.toPlain` gives plain data.
 *
 * A gain in `cfg` never replaces the `gain` a recording states. An
 * IntoMind device is never set: a gain in `cfg`
 * other than the gain in force on it, which its Status states, is refused
 * in words naming both gains before anything is recorded. From firmware
 * 1.4.6 the device samples at gain 1 while lead-off detection is on,
 * whatever gain was set.
 *
 * A cue one of the package's experiments stamps belongs to the devices that
 * experiment handles, and is written only into their recordings. An event
 * stamped anywhere else, such as a person's marker, an outside source's
 * event, or a cue an experiment's own code stamps between steps, is written
 * into every recording the run saves after it is stamped. What ties a cue to
 * its devices is kept beside the events, never in them.
 */
export class RunCtx {
  /** Where the run's messages go. */
  readonly emit: Emit;
  /** Raised with `set()` to end the run early. `isSet()` says whether it has been. */
  readonly stop: { set(): void; clear(): void; isSet(): boolean } = new Flag();
  /** The events stamped so far: markers, cues, and anything else. */
  readonly events: Value[] = [];
  /** What the instrument was, for the manifest. */
  readonly cfg: Value;

  /** A context for one run: where its messages go, and what the instrument was. */
  constructor(emit: Emit, cfg: Json = null) {
    this.emit = emit;
    const c = J.fromPlain(cfg ?? null);
    if (J.truthy(c) && c.kind !== "object") throw new TypeError(`cfg is an object, not ${J.jsRepr(c)}`);
    this.cfg = J.truthy(c) ? J.copyObject(c) : J.object([]);
  }

  #stamp(cond: string, kind: string, kw: Json): Value {
    // Monotonic is read first and the wall clock second, always in that
    // order, so the gap between the two reads is a fixed bias.
    const mono = runtime.time.monotonic();
    const hostTime = runtime.time.time();
    const e = J.object([
      ["cond", J.str(cond)],
      ["kind", J.str(kind)],
      ["host_time", J.float(hostTime)],
      ["mono", J.float(mono)],
    ]);
    const extra = J.fromPlain(kw ?? {});
    if (extra.kind !== "object") throw new TypeError(`an event's own fields are an object, not ${J.jsRepr(extra)}`);
    for (const [k, v] of extra.entries) {
      if (J.get(e, k) !== undefined) throw new TypeError(`dict() got multiple values for keyword argument '${k}'`);
      J.set(e, k, v);
    }
    this.events.push(e);
    return e;
  }

  /** Stamp a marker named `name`, a person's own, and return it. */
  marker(name: string, kw: Json = {}): Value {
    return this.#stamp(name, "marker", kw);
  }

  /** Stamp a cue for the condition `cond`, the protocol's, and return it. */
  cue(cond: string, kw: Json = {}): Value {
    return this.#stamp(cond, "cue", kw);
  }

  /**
   * Stamp an event of any kind, the path an outside source takes, so every
   * source lands in one stream on one timebase rather than each inventing
   * its own.
   */
  event(kind: string, cond: string, kw: Json = {}): Value {
    return this.#stamp(cond, kind, kw);
  }
}

/** One cue point: `none` is silent, `speech` speaks `text`, any other name plays that tone. */
export function cueSound(tone: unknown, text = ""): void {
  runtime.play(tone, text);
}

/** Speak `text` aloud through the host's audio output. */
export function say(text: string): void {
  runtime.play("speech", text);
}

/** The three cue points every subject-facing protocol takes, each speech unless changed. */
export const TONE_PARAMS: readonly Param[] = [
  { name: "tone_start", type: "select", default: "speech", choices: CHOICES },
  { name: "tone_change", type: "select", default: "speech", choices: CHOICES },
  { name: "tone_end", type: "select", default: "speech", choices: CHOICES },
];

// --- Python's own ways with values ------------------------------------------------

/** `p[key]`, refused in the words Python refuses a key that is not there with. */
function need(p: Params, key: string): unknown {
  if (!Object.hasOwn(p, key) || p[key] === undefined) throw new Invalid(`KeyError: ${J.reprStr(key)}`);
  return p[key];
}

/** `p.get(key, fallback)`. */
function get(p: Params, key: string, fallback: unknown): unknown {
  return Object.hasOwn(p, key) && p[key] !== undefined ? p[key] : fallback;
}

/** The name Python gives the type of a value JSON can hold, as a refusal thrown to a program names it: null in JavaScript's own word. */
function pyType(v: Value): string {
  switch (v.kind) {
    case "null":
      return "null";
    case "array":
      return "list";
    case "object":
      return "dict";
    default:
      return v.kind;
  }
}

/** `float(x)`, refused in Python's words. */
function toFloat(x: unknown): number {
  const v = J.fromPlain(x);
  const f = J.asF64(v);
  if (f !== null) return f;
  if (v.kind === "str") throw new Invalid(`ValueError: could not convert string to float: ${J.reprStr(v.value)}`);
  throw new TypeError(`float() argument must be a string or a real number, not '${pyType(v)}'`);
}

/**
 * `int(x)`, as the whole number it is, refused in Python's words: a whole
 * number as it is, true and false as one and zero, a float cut toward zero,
 * and text read where it is written as a whole number, of no more digits
 * than Python reads, 4,300.
 */
function toWhole(x: unknown): bigint {
  const v = J.fromPlain(x);
  switch (v.kind) {
    case "bool":
      return v.value ? 1n : 0n;
    case "int":
      return BigInt(v.value);
    case "float":
      if (Number.isNaN(v.value)) throw new Invalid("ValueError: cannot convert float NaN to integer");
      if (!Number.isFinite(v.value)) throw new Invalid("OverflowError: cannot convert float infinity to integer");
      return BigInt(Math.trunc(v.value));
    case "str": {
      const t = J.trim(v.value).replaceAll("_", "");
      if (J.tooLong(v) !== null) throw new Invalid("ValueError: Exceeds the limit (4300 digits) for integer string conversion");
      if (/^[+-]?\d+$/.test(t)) return BigInt(t);
      throw new Invalid(`ValueError: invalid literal for int() with base 10: ${J.reprStr(v.value)}`);
    }
    default:
      throw new TypeError(`int() argument must be a string, a bytes-like object or a real number, not '${pyType(v)}'`);
  }
}

/** `str(x)`, as an f-string writes a value. */
function pyText(x: unknown): string {
  return J.pyStr(J.fromPlain(x));
}

/** A value the library's refusal shows by its `repr`, as a refusal in the package shows a value: as JavaScript writes it, readably. */
const said = (x: unknown): string => J.shown(x);

/**
 * A parameter's value as a refusal shows it, as the library's
 * `_value_said` shows it: as `said` shows it, but a whole number of more
 * digits than Python turns into text, 4,300, and text written as one, by
 * how many digits it has, never printed, and a list or an object holding
 * such a whole number by its type, as Python, which will not print it,
 * names it.
 */
function valueSaid(x: unknown): string {
  return J.tooLong(x) ?? (J.holdsTooLong(x) ? `a ${pyType(J.fromPlain(x))} of more digits than Python prints` : said(x));
}

/**
 * The parameter `name` of an experiment, given as `value`, as the number
 * it must be, as the library's `_number` reads it. One that is not a
 * finite number, or is not given, is refused in words that name the
 * parameter, before anything is sent to any device, showing the value as
 * `valueSaid` shows it.
 */
function number(name: string, value: unknown): number {
  if (value === null || value === undefined) throw new Invalid(`ValueError: ${name} is not given, and needs a number`);
  let n: number;
  try {
    n = toFloat(value);
  } catch (e) {
    if (e instanceof Invalid || e instanceof TypeError) throw new Invalid(`ValueError: ${name} needs a number, not ${valueSaid(value)}`);
    throw e;
  }
  if (!Number.isFinite(n)) throw new Invalid(`ValueError: ${name} needs a number, not ${valueSaid(value)}`);
  return n;
}

/**
 * The parameter `name` of an experiment, given as `value`, as the whole
 * number it must be, refused as `number` refuses one, as the library's
 * `_count` reads it, and so is a number that is not finite. A number with a
 * fraction is cut to its whole part, and text is read only when it is
 * written as a whole number, of no more digits than Python reads, 4,300:
 * text of more is refused, named by how many digits it holds.
 */
function whole(name: string, value: unknown): bigint {
  if (value === null || value === undefined) throw new Invalid(`ValueError: ${name} is not given, and needs a whole number`);
  try {
    return toWhole(value);
  } catch (e) {
    if (e instanceof Invalid || e instanceof TypeError) throw new Invalid(`ValueError: ${name} needs a whole number, not ${valueSaid(value)}`);
    throw e;
  }
}

/** The parameter `name` of an experiment, given as `value`, as the whole number `whole` reads, as a number. */
function count(name: string, value: unknown): number {
  return Number(whole(name, value));
}

/** A whole number a parameter was read as, as a refusal shows it: by how many digits it has where it has more than Python turns into text, 4,300. */
function wholeSaid(n: bigint): string {
  return J.tooLong(n) ?? n.toString();
}

/** Words as a sentence lists them, as the library's `_series` writes them: "A", "A and B", or "A, B and C", with `last` in place of "and". */
function series(words: readonly string[], last: string): string {
  return words.length === 1 ? words[0]! : `${words.slice(0, -1).join(", ")} ${last} ${words[words.length - 1]}`;
}

/** `s.casefold()`, as far as a name a person writes goes: lowercase, and a sharp s folded to two. */
function casefold(s: string): string {
  return s.toLowerCase().replaceAll("\u00df", "ss");
}

/** What Python's `str.isspace` calls whitespace. */
const SPACE = new Set([
  0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003,
  0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000,
]);

/** `s.strip()`. */
function strip(s: string): string {
  let a = 0;
  let b = s.length;
  while (a < b && SPACE.has(s.charCodeAt(a))) a++;
  while (b > a && SPACE.has(s.charCodeAt(b - 1))) b--;
  return s.slice(a, b);
}

/** What Python's `str.splitlines` breaks a line at. A carriage return and a line feed together are one break. */
const BREAKS = new Set([0x0a, 0x0b, 0x0c, 0x0d, 0x1c, 0x1d, 0x1e, 0x85, 0x2028, 0x2029]);

/** `s.splitlines()`. */
function splitlines(s: string): string[] {
  const out: string[] = [];
  let start = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (!BREAKS.has(c)) continue;
    out.push(s.slice(start, i));
    if (c === 0x0d && s.charCodeAt(i + 1) === 0x0a) i++;
    start = i + 1;
  }
  if (start < s.length) out.push(s.slice(start));
  return out;
}

/** `[item] * n`. */
function repeat<T>(items: readonly T[], n: number): T[] {
  const out: T[] = [];
  for (let k = 0; k < n; k++) out.push(...items);
  return out;
}

/**
 * `dict(k=v, ..., **a, **b)`: the entries in order, then each object's, and
 * a key given twice refused as Python refuses it.
 */
function dict(entries: readonly (readonly [string, Value])[], ...more: Value[]): Value {
  const out = J.object([]);
  const put = (k: string, v: Value): void => {
    if (J.get(out, k) !== undefined) throw new TypeError(`dict() got multiple values for keyword argument '${k}'`);
    J.set(out, k, v);
  };
  for (const [k, v] of entries) put(k, v);
  for (const m of more) {
    if (m.kind !== "object") throw new TypeError(`dict() argument after ** must be a mapping, not ${pyType(m)}`);
    for (const [k, v] of m.entries) put(k, v);
  }
  return out;
}

/** Local time as `YYYYmmdd-HHMMSS`, as `time.strftime` writes the time now. */
function localStamp(unix: number): string {
  const d = new Date(Math.floor(unix) * 1000);
  const two = (v: number): string => String(v).padStart(2, "0");
  return `${d.getFullYear()}${two(d.getMonth() + 1)}${two(d.getDate())}-${two(d.getHours())}${two(d.getMinutes())}${two(d.getSeconds())}`;
}

/**
 * A recording's name, as the library's `_lbl` makes it: `prefix`, then the
 * bias mode the run's `ctx.cfg` gives under `bias`, one of the protocol's
 * `BIAS_MODES` in its own word, and `na` where it gives no mode, as the setup
 * of a device that drives no bias electrode, the IntoMind One among them,
 * never does; then `g` and the gain it gives, `na` where it gives none or
 * gives null; and the local date and time, as `YYYYMMDD-HHMMSS`, the four
 * joined by underscores. A gain of
 * more digits than Python turns into text, 4,300, or text written as one,
 * is refused in words naming it by how many digits it has, too long for a
 * recording's name, before anything is recorded. The package's own
 * experiments name their recordings with it, and an experiment of one's own
 * hands it to `cuedTrials` as its `label`.
 */
export function label(ctx: RunCtx, prefix: string): string {
  const given = J.get(ctx.cfg, "bias");
  const bias = given !== undefined && given.kind === "str" && Object.hasOwn(BIAS_MODES, given.value) ? given.value : "na";
  return `${prefix}_${bias}_g${gainNamed(J.get(ctx.cfg, "gain"))}_${localStamp(runtime.time.time())}`;
}

const f = J.float;
const s = J.str;

// --- the experiments --------------------------------------------------------------

register(
  "baseline",
  "Fixed-duration resting recording.",
  [{ name: "seconds", type: "number", default: 30 }, { name: "note", type: "text", default: "" }, ...TONE_PARAMS],
  async (dev, emit, p, ctx) => {
    const seconds = number("seconds", get(p, "seconds", null));
    const here = step(dev, emit, ctx);
    const lbl = label(ctx, "baseline");
    await tell(here, { type: "note", text: `recording ${pyText(get(p, "seconds", null))}s → ${lbl}` });
    runtime.play(get(p, "tone_start", "speech"), "Recording.");
    const extra = dict(
      [
        ["experiment", s("baseline")],
        ["note", J.fromPlain(get(p, "note", ""))],
      ],
      ctx.cfg,
    );
    const done = await recordFor(dev, emit, ctx, lbl, seconds, extra);
    runtime.play(get(p, "tone_end", "speech"), "Done.");
    return labels(done);
  },
  { enforced: {} },
);

register(
  "berger",
  "Cued eyes-closed / eyes-open alpha test.",
  [
    { name: "block", type: "number", default: 20 },
    { name: "reps", type: "number", default: 8 },
    { name: "settle", type: "number", default: 3 },
    ...TONE_PARAMS,
  ],
  async (dev, emit, p, ctx) => {
    const block = number("block", get(p, "block", null));
    const reps = count("reps", get(p, "reps", null));
    const settle = number("settle", get(p, "settle", null));
    const here = step(dev, emit, ctx);
    const cfg = J.copyObject(ctx.cfg);
    const lbl = label(ctx, "berger");
    const started = utc(runtime.time.time());
    const armed = await arm(here, dev);
    let done: Done[];
    try {
      runtime.play(get(p, "tone_start", "speech"), "Starting. Sit still, breathe normally.");
      await tell(here, { type: "cue", text: "Sit still — starting in 6 s" });
      await runtime.time.sleep(6);
      const total = reps * 2 * block;
      let sofar = 0.0;
      for (let r = 0; r < reps; r++) {
        for (const cond of ["closed", "open"]) {
          const txt = cond === "closed" ? "Close your eyes" : "Open your eyes";
          runtime.play(get(p, "tone_change", "speech"), txt);
          cue(here, cond, { rep: r });
          await tell(here, { type: "cue", text: `${txt}  (rep ${r + 1}/${reps})` });
          const t0 = runtime.time.time();
          while (runtime.time.time() - t0 < block && !ctx.stop.isSet()) {
            await runtime.time.sleep(0.25);
            await tell(here, { type: "progress", frac: (sofar + runtime.time.time() - t0) / total });
          }
          sofar += block;
        }
      }
      runtime.play(get(p, "tone_end", "speech"), "Done. You can relax.");
      await tell(here, { type: "cue", text: "Done — relax" });
    } finally {
      // The fields the manifest adds are built once the streams the step
      // started are stopped, inside the cleanup that lets its devices go.
      done = await finish(here, dev, armed, lbl, () =>
        dict(
          [
            ["experiment", s("berger")],
            ["block", f(block)],
            ["reps", J.int(reps)],
            ["started_utc", s(started)],
            ["settle", f(settle)],
          ],
          cfg,
        ),
      );
    }
    return labels(done);
  },
  {
    enforced: {},
    prereg: J.object([
      ["name", s("berger")],
      ["alpha_band", J.array([f(8.0), f(13.0)])],
      ["reference_band", J.array([f(4.0), f(30.0)])],
      ["epoch_s", f(2.0)],
      ["reject_pp_uv", f(150.0)],
      ["test", s("wilcoxon_paired_blocks")],
      ["allow_pseudoreplication", J.bool(false)],
    ]),
  },
);

register(
  "record",
  "Open-ended recording. Runs until you press Stop or the time set to stop after has passed, ten minutes by default. Use the marker buttons to timestamp events as they happen.",
  [
    { name: "markers", type: "text", default: "event 1,event 2,event 3,event 4,event 5" },
    { name: "max_seconds", type: "number", default: 600 },
    { name: "note", type: "text", default: "" },
    ...TONE_PARAMS,
  ],
  async (dev, emit, p, ctx) => {
    const limit = number("max_seconds", get(p, "max_seconds", null));
    const here = step(dev, emit, ctx);
    const cfg = J.copyObject(ctx.cfg);
    const lbl = label(ctx, "record");
    const started = utc(runtime.time.time());
    await tell(here, { type: "note", text: `recording → ${lbl} (Stop to end)` });
    const names = pyText(need(p, "markers")).split(",").map(strip).filter((m) => m !== "");
    await tell(here, { type: "markers", names });
    const armed = await arm(here, dev);
    runtime.play(get(p, "tone_start", "speech"), "Recording.");
    const t0 = runtime.time.time();
    let done: Done[];
    try {
      while (!ctx.stop.isSet() && runtime.time.time() - t0 < limit) {
        await runtime.time.sleep(0.25);
        await tell(here, { type: "elapsed", seconds: runtime.time.time() - t0, markers: ctx.events.length });
      }
    } finally {
      done = await finish(
        here,
        dev,
        armed,
        lbl,
        () =>
          dict(
            [
              ["experiment", s("record")],
              ["note", J.fromPlain(get(p, "note", ""))],
              ["started_utc", s(started)],
              ["duration_s", f(runtime.time.time() - t0)],
            ],
            cfg,
          ),
        { then: () => runtime.play(get(p, "tone_end", "speech"), "Done.") },
      );
    }
    return labels(done);
  },
  { enforced: {} },
);

register(
  "protocol",
  "Custom cued protocol. One step per line: 'cue text | seconds'. The steps run as many times as the repetitions set. Each cue is timestamped, and spoken when the cue at each change is speech.",
  [
    { name: "steps", type: "textarea", default: "Close your eyes | 20\nOpen your eyes | 20" },
    { name: "reps", type: "number", default: 3 },
    { name: "settle", type: "number", default: 3 },
    { name: "lead_in", type: "number", default: 6 },
    { name: "note", type: "text", default: "" },
    ...TONE_PARAMS,
  ],
  async (dev, emit, p, ctx) => {
    const steps: [string, number][] = [];
    const written = get(p, "steps", null);
    splitlines(written === null ? "" : pyText(written)).forEach((line, i) => {
      if (strip(line) === "") return;
      if (!line.includes("|")) throw new Invalid(`ValueError: step needs "cue | seconds": ${J.quoted(line)}`);
      const bar = line.lastIndexOf("|");
      let secs: number | null = null;
      try {
        secs = toFloat(line.slice(bar + 1));
      } catch (e) {
        if (!(e instanceof Invalid)) throw e;
      }
      if (secs === null || !Number.isFinite(secs)) {
        throw new Invalid(`ValueError: line ${i + 1} of steps needs a number of seconds after its "|": ${J.quoted(strip(line))}`);
      }
      steps.push([strip(line.slice(0, bar)), secs]);
    });
    if (steps.length === 0) throw new Invalid("ValueError: no steps");
    const reps = count("reps", get(p, "reps", null));
    const settle = number("settle", get(p, "settle", null));
    const leadIn = number("lead_in", get(p, "lead_in", null));

    const here = step(dev, emit, ctx);
    const cfg = J.copyObject(ctx.cfg);
    const lbl = label(ctx, "protocol");
    const started = utc(runtime.time.time());
    const armed = await arm(here, dev);
    let done: Done[];
    try {
      runtime.play(get(p, "tone_start", "speech"), "Starting.");
      await tell(here, { type: "cue", text: `Starting in ${pyText(get(p, "lead_in", null))} s` });
      await runtime.time.sleep(leadIn);
      let sum = 0;
      for (const [, secs] of steps) sum += secs;
      const total = reps * sum;
      let sofar = 0.0;
      for (let r = 0; r < reps; r++) {
        for (const [text, secs] of steps) {
          if (ctx.stop.isSet()) break;
          runtime.play(get(p, "tone_change", "speech"), text);
          cue(here, text, { rep: r, planned_s: f(secs) });
          await tell(here, { type: "cue", text: `${text}  (rep ${r + 1}/${reps})` });
          const t0 = runtime.time.time();
          while (runtime.time.time() - t0 < secs && !ctx.stop.isSet()) {
            await runtime.time.sleep(0.25);
            await tell(here, { type: "progress", frac: (sofar + runtime.time.time() - t0) / total });
          }
          sofar += secs;
        }
      }
      runtime.play(get(p, "tone_end", "speech"), "Done.");
      await tell(here, { type: "cue", text: "Done" });
    } finally {
      // The longest step, the first of equals kept, as Python's `max` keeps it.
      let block = steps[0]![1];
      for (const [, secs] of steps) if (secs > block) block = secs;
      done = await finish(here, dev, armed, lbl, () =>
        dict(
          [
            ["experiment", s("protocol")],
            ["steps", J.array(steps.map(([text, secs]) => J.array([s(text), f(secs)])))],
            ["reps", J.int(reps)],
            ["started_utc", s(started)],
            ["settle", f(settle)],
            ["block", f(block)],
            ["note", J.fromPlain(get(p, "note", ""))],
          ],
          cfg,
        ),
      );
    }
    return labels(done);
  },
  { enforced: {} },
);

/** What one event-locked protocol is: its trials, what each cue says, and its timing. */
interface Trials {
  /** The recording's name, as `label` makes it. */
  readonly label: string;
  /** Each trial's condition, in the order the trials run. */
  readonly conds: readonly string[];
  /** The words that cue each condition, by condition. */
  readonly texts: Readonly<Record<string, string>>;
  /** The seconds before the first trial, 6 unless given. */
  readonly leadIn?: number;
  /** The seconds of the rest before each cue, 2 unless given. */
  readonly restS?: number;
  /** The seconds each cue is held, 2 unless given. */
  readonly holdS?: number;
  /** The words said as the run starts, none unless given. */
  readonly intro?: string;
  /** The words that start each rest, "Rest" unless given. */
  readonly restText?: string;
  /** What the manifest adds, such as the experiment's name, as an object `capture.fromPlain` makes. */
  readonly extra?: Value | null;
}

/**
 * One cue a trial, stamped at the cue instant, with a rest between, shared
 * by the blink and jaw clench protocols, as the library's `_cued_trials`
 * runs them. The cues come in the order of
 * `conds`, which each of them shuffles with a fixed seed of its own before
 * it calls this, and that order is written into the manifest as `order`,
 * because the order is part of the record: a subject who can predict the
 * next cue is a different subject.
 * Both edges are spoken: the rest is voiced at the start of every rest,
 * because a cue that starts a state needs a cue that ends it.
 *
 * An experiment of one's own that is event-locked registers with
 * `experiment` and hands this its `dev`, `emit`, parameters and `ctx`, with
 * its trials in `t`: a recording's name from `label`, its conditions in the
 * order it shuffled them, the words for each, and its timing. The manifest
 * keeps `order`, `rest_s`, `hold_s` and the run's `note` beside what `extra`
 * adds and the run's `ctx.cfg`, and the names of the recordings saved are
 * given back.
 */
export async function cuedTrials(dev: Unit | Unit[], emit: Emit, p: Params, ctx: RunCtx, t: Trials): Promise<string[]> {
  const here = step(dev, emit, ctx);
  const leadIn = t.leadIn ?? 6.0;
  const restS = t.restS ?? 2.0;
  const holdS = t.holdS ?? 2.0;
  const intro = t.intro ?? "";
  const restText = t.restText ?? "Rest";
  const n = t.conds.length;
  const started = utc(runtime.time.time());
  const armed = await arm(here, dev);
  let done: Done[];
  try {
    runtime.play(get(p, "tone_start", "speech"), intro);
    await tell(here, { type: "cue", text: `${intro}  starting in ${fixed(leadIn, 0)} s` });
    await runtime.time.sleep(leadIn);
    for (let i = 0; i < n; i++) {
      if (ctx.stop.isSet()) break;
      const cond = t.conds[i]!;
      runtime.play(get(p, "tone_change", "speech"), restText);
      await tell(here, { type: "cue", text: `${restText}  (${i + 1}/${n})` });
      let t0 = runtime.time.time();
      while (runtime.time.time() - t0 < restS && !ctx.stop.isSet()) await runtime.time.sleep(0.05);

      const text = t.texts[cond]!;
      runtime.play(get(p, "tone_change", "speech"), text);
      cue(here, cond, { trial: i });
      await tell(here, { type: "cue", text: `${text}  (${i + 1}/${n})` });
      t0 = runtime.time.time();
      while (runtime.time.time() - t0 < holdS && !ctx.stop.isSet()) {
        await runtime.time.sleep(0.05);
        await tell(here, { type: "progress", frac: (i + 1) / n });
      }
    }
    runtime.play(get(p, "tone_end", "speech"), "Done. You can relax.");
    await tell(here, { type: "cue", text: "Done" });
  } finally {
    done = await finish(here, dev, armed, t.label, () =>
      dict(
        [
          ["started_utc", s(started)],
          ["order", J.array(t.conds.map(s))],
          ["rest_s", f(restS)],
          ["hold_s", f(holdS)],
          ["note", J.fromPlain(get(p, "note", ""))],
        ],
        t.extra ?? J.object([]),
        J.copyObject(ctx.cfg),
      ),
    );
  }
  return labels(done);
}

/** The trials and, when `sham` is not 0, the shams that control for the cue itself, shuffled with the protocol's own seed. */
function withShams(conds: string[], sham: number, seed: number, trials: number): string[] {
  if (sham !== 0) conds.push(...repeat(["sham"], Math.max(2, Math.floor(trials / 4))));
  new Random(seed).shuffle(conds);
  return conds;
}

const SHAM_PARAM: Param = { name: "include_sham", type: "select", default: 1, choices: [0, 1] };

register(
  "blink",
  "Cued blinks. Electrodes at the forehead typically see a blink of 100 to 300 uV. Sham trials ('stay still') control for the cue itself.",
  [
    { name: "trials", type: "number", default: 30 },
    { name: "hold_s", type: "number", default: 1.5 },
    { name: "rest_s", type: "number", default: 2.0 },
    SHAM_PARAM,
    { name: "note", type: "text", default: "" },
    ...TONE_PARAMS,
  ],
  async (dev, emit, p, ctx) => {
    const trials = count("trials", get(p, "trials", null));
    const sham = count("include_sham", get(p, "include_sham", null));
    const restS = number("rest_s", get(p, "rest_s", null));
    const holdS = number("hold_s", get(p, "hold_s", null));
    const conds = withShams(repeat(["blink"], trials), sham, 24601, trials);
    return cuedTrials(dev, emit, p, ctx, {
      label: label(ctx, "blink"),
      conds,
      texts: { blink: "Blink", sham: "Stay still" },
      restS,
      holdS,
      restText: "Rest",
      intro: "Blink test. Blink once, hard, when told, then rest.",
      extra: J.object([
        ["experiment", s("blink")],
        ["trials", J.int(trials)],
      ]),
    });
  },
  {
    enforced: {},
    prereg: J.object([
      ["name", s("blink")],
      ["pre_s", f(0.5)],
      ["post_s", f(1.5)],
      ["hp", f(0.1)],
      ["lp", f(10.0)],
      ["min_pp_uv", f(100.0)],
      ["min_detect_frac", f(0.9)],
      ["max_latency_sd_s", f(0.25)],
      ["sham_must_be_null", J.bool(true)],
    ]),
  },
);

register(
  "jaw_clench",
  "Cued jaw clench. Electrodes at the temples sit over the temporalis muscle, where a clench is typically broadband EMG an order of magnitude above baseline in 20-200 Hz.",
  [
    { name: "trials", type: "number", default: 20 },
    { name: "hold_s", type: "number", default: 2.0 },
    { name: "rest_s", type: "number", default: 3.0 },
    SHAM_PARAM,
    { name: "note", type: "text", default: "" },
    ...TONE_PARAMS,
  ],
  async (dev, emit, p, ctx) => {
    const trials = count("trials", get(p, "trials", null));
    const sham = count("include_sham", get(p, "include_sham", null));
    const restS = number("rest_s", get(p, "rest_s", null));
    const holdS = number("hold_s", get(p, "hold_s", null));
    const conds = withShams(repeat(["clench"], trials), sham, 31337, trials);
    return cuedTrials(dev, emit, p, ctx, {
      label: label(ctx, "jaw"),
      conds,
      texts: { clench: "Clench your jaw", sham: "Stay still" },
      restS,
      holdS,
      restText: "Relax",
      intro: "Jaw clench test. Clench when told, and relax when told.",
      extra: J.object([
        ["experiment", s("jaw_clench")],
        ["trials", J.int(trials)],
      ]),
    });
  },
  {
    enforced: {},
    prereg: J.object([
      ["name", s("jaw")],
      ["window_s", f(1.0)],
      ["lo_hz", f(20.0)],
      ["hi_hz", f(200.0)],
      ["min_ratio", f(10.0)],
      ["alpha", f(0.001)],
      ["test", s("wilcoxon_paired_trials")],
    ]),
  },
);

register(
  "saccade",
  "Cued left/right eye movements. The eye is a dipole, so electrodes at the left and right temples deflect in opposite directions, and the signs reverse with the direction of the movement.",
  [
    { name: "trials", type: "number", default: 20 },
    { name: "hold_s", type: "number", default: 2.0 },
    { name: "center_s", type: "number", default: 2.0 },
    SHAM_PARAM,
    { name: "note", type: "text", default: "" },
    ...TONE_PARAMS,
  ],
  async (dev, emit, p, ctx) => {
    // The sham cue, stay still, is the control for the cue itself: the
    // speaker, the audio output, and any cue-locked electrical artifact.
    const trials = count("trials", get(p, "trials", null));
    const hold = number("hold_s", get(p, "hold_s", null));
    // `centre_s` is the older spelling of this parameter. Recordings may
    // carry it and a saved form may still send it, so it is accepted on the
    // way in and never written on the way out.
    const center = number("center_s", get(p, "center_s", get(p, "centre_s", 2.0)));
    const sham = count("include_sham", get(p, "include_sham", null)) !== 0;
    const here = step(dev, emit, ctx);
    const cfg = J.copyObject(ctx.cfg);
    const lbl = label(ctx, "saccade");
    const conds = trials > 0 ? repeat(["left", "right"], Math.floor((trials + 1) / 2)).slice(0, trials) : [];
    if (sham) conds.push(...repeat(["sham"], Math.max(2, Math.floor(trials / 4))));
    new Random(12345).shuffle(conds);

    const started = utc(runtime.time.time());
    const armed = await arm(here, dev);
    let done: Done[];
    try {
      runtime.play(get(p, "tone_start", "speech"), "Saccade test. Keep your head still. Move only your eyes.");
      await tell(here, { type: "cue", text: "Head still — starting in 6 s" });
      await runtime.time.sleep(6);
      const n = conds.length;
      for (let i = 0; i < n; i++) {
        if (ctx.stop.isSet()) break;
        const cond = conds[i]!;
        runtime.play(get(p, "tone_change", "speech"), "Look at the center");
        await tell(here, { type: "cue", text: `Center  (${i + 1}/${n})` });
        let t0 = runtime.time.time();
        while (runtime.time.time() - t0 < center && !ctx.stop.isSet()) await runtime.time.sleep(0.1);

        const text = ({ left: "Look left", right: "Look right", sham: "Stay still" } as Record<string, string>)[cond]!;
        runtime.play(get(p, "tone_change", "speech"), text);
        cue(here, cond, { trial: i });
        await tell(here, { type: "cue", text: `${text}  (${i + 1}/${n})` });
        t0 = runtime.time.time();
        while (runtime.time.time() - t0 < hold && !ctx.stop.isSet()) {
          await runtime.time.sleep(0.1);
          await tell(here, { type: "progress", frac: (i + 1) / n });
        }
      }
      runtime.play(get(p, "tone_end", "speech"), "Done. You can relax.");
      await tell(here, { type: "cue", text: "Done" });
    } finally {
      done = await finish(here, dev, armed, lbl, () =>
        dict(
          [
            ["experiment", s("saccade")],
            ["trials", J.int(trials)],
            ["hold_s", f(hold)],
            ["center_s", f(center)],
            ["include_sham", J.bool(sham)],
            ["order", J.array(conds.map(s))],
            ["started_utc", s(started)],
            ["note", J.fromPlain(get(p, "note", ""))],
          ],
          cfg,
        ),
      );
    }
    return labels(done);
  },
  {
    enforced: {},
    prereg: J.object([
      ["name", s("saccade")],
      ["window_s", J.array([f(0.1), f(0.9)])],
      ["baseline_s", J.array([f(-0.5), f(0.0)])],
      ["min_deflection_uv", f(50.0)],
      ["require_opposite_polarity", J.bool(true)],
      ["require_sign_reversal", J.bool(true)],
      ["sham_must_be_null", J.bool(true)],
      ["test", s("wilcoxon_paired_trials")],
    ]),
  },
);

// --- the head training ----------------------------------------------------------

/** A state the head training teaches: its name, and the words that cue it. */
type State = readonly [string, string];

/** The pairs of states the head training offers, by the name a run is given: each state's name and the words that cue it. A run whose pair is "own" teaches the states a person writes in its `states` instead. */
const PAIRS: Readonly<Record<string, readonly [State, State]>> = {
  mental: [
    ["Relaxed", "Relax. Breathe slowly and let your mind wander."],
    ["Mental math", "Mental arithmetic. Start at three hundred and keep subtracting seven."],
  ],
  eyes: [
    ["Eyes open", "Eyes open."],
    ["Eyes closed", "Close your eyes, and keep them closed until the next cue."],
  ],
  jaw: [
    ["Relaxed", "Relax your face."],
    ["Jaw clenched", "Clench your jaw gently, and hold it."],
  ],
};

/** What `states` holds until a person writes their own: the "mental" pair, one state to a line, so the field shows the form it takes. */
const STATES = PAIRS.mental!.map(([name, words]) => `${name} | ${words}`).join("\n");

/** Windows delivered after a cue that may have begun before it: the one that straddles it, and the one before that. */
const SETTLE = 2;

/**
 * The states a person wrote, one to a line as "name | cue words", as the
 * library's `_own_states` reads them: each state's name and words, in the
 * order given. A blank line is passed over. Refused before anything is
 * sent to any device, saying which line and why, unless there are two or
 * more, each with a name and cue words, and no two share a name, whatever
 * their capitals.
 */
function ownStates(text: unknown): State[] {
  const out: State[] = [];
  const seen = new Map<string, number>();
  splitlines(text === null || text === undefined ? "" : pyText(text)).forEach((line, i) => {
    const at = i + 1;
    if (strip(line) === "") return;
    if (!line.includes("|")) throw new Invalid(`ValueError: line ${at} of states needs "name | cue words": ${J.quoted(strip(line))}`);
    const bar = line.indexOf("|");
    const name = strip(line.slice(0, bar));
    const words = strip(line.slice(bar + 1));
    if (name === "") throw new Invalid(`ValueError: line ${at} of states has no name before its "|": ${J.quoted(strip(line))}`);
    if (words === "") throw new Invalid(`ValueError: line ${at} of states has no cue words after its "|": ${J.quoted(strip(line))}`);
    const earlier = seen.get(casefold(name));
    if (earlier !== undefined) {
      throw new Invalid(`ValueError: line ${at} of states names ${J.quoted(name)}, as line ${earlier} does, and each state needs a name of its own`);
    }
    seen.set(casefold(name), at);
    out.push([name, words]);
  });
  if (out.length < 2) {
    throw new Invalid(
      `ValueError: states names ${out.length === 0 ? "no state" : "one state"}, and a head tells two or more apart: one to a line, written "name | cue words"`,
    );
  }
  return out;
}

/** What a head trained on `states` is called, as the library's `_head_name` names it, cut to 15 characters: the second state's name for two, the state its one output says, and every state's name joined by "/" for more. */
function headName(states: readonly string[]): string {
  // Python slices a string by code points, as `Array.from` walks one.
  return Array.from(states.length === 2 ? states[1]! : states.join("/"))
    .slice(0, 15)
    .join("");
}

/** A head's training data as text, one window to a line, as the library writes `<label>.training.json`. */
function trainingText(head: Value, windows: readonly { index: number; round: number; label: number; embedding: number[] }[]): string {
  const top = J.dumpsIndented(J.object([...(head as { entries: [string, Value][] }).entries, ["windows", J.array([])]]), 2);
  const lines = windows
    .map((w) => `    {"index":${w.index},"round":${w.round},"label":${w.label},"embedding":[${w.embedding.join(",")}]}`)
    .join(",\n");
  return `${top.replace('"windows": []', `"windows": [\n${lines}\n  ]`)}\n`;
}

/** One device's windows kept by a head training: the rows and labels a head trains on, and what was dropped. */
/** A device head training asks for its model and its windows: a browser connection. */
type Thinks = Connection & { modelInfo(): Promise<ModelInfo> };

interface Taught {
  rows: number[][];
  labels: number[];
  windows: { index: number; round: number; label: number; embedding: number[] }[];
  settle: number;
  flagged: number;
}

/**
 * The head one device's kept windows train, its held-out score, or why
 * there is none, as the library says it. Every state needs windows, and
 * there must be more windows than an embedding has values. Windows that do
 * not vary enough to determine a head, all of them or those left once a
 * block is held out to score it, are said so in plain words, and any other
 * refusal of `trainHead` or `blockedFolds` in the library's.
 */
function taughtHead(
  tk: Taught,
  states: readonly string[],
  name: string,
  info: ModelInfo,
  linkOk: boolean,
  finished: boolean,
): { head: Uint8Array | null; score: ReturnType<typeof blockedFolds>; noHead: string | null } {
  const none = (noHead: string) => ({ head: null, score: null, noHead });
  const width = tk.windows.length > 0 ? tk.windows[0]!.embedding.length : 0;
  if (!linkOk) return none("the link to the device dropped, so no head was trained");
  if (!finished) return none("the run ended on an error, so no head was trained");
  if (tk.rows.length === 0) return none("no windows were kept, so no head was trained");
  const missing = states.filter((_name, y) => !tk.labels.includes(y));
  if (missing.length > 0) {
    return none(
      `no windows of ${series(missing, "or")} were kept, and a head that tells ` +
        (states.length === 2 ? "two states apart needs both" : `${states.length} states apart needs windows of every one`),
    );
  }
  if (tk.rows.length <= width) {
    return none(`${tk.rows.length} windows were kept, and a head for an embedding of ${width} values needs more than ${width}`);
  }
  // The library's ValueError, numpy's LinAlgError among them, in its words.
  const refused = (e: unknown): string => {
    if (!(e instanceof Invalid) || !/^(ValueError|LinAlgError): /.test(e.message)) throw e;
    return e.message.slice(e.message.indexOf(": ") + 2);
  };
  let head: Uint8Array;
  try {
    head = trainHead(tk.rows, targets(tk.labels, states.length), { name, encoderId: info.encoderId });
  } catch (e) {
    const words = refused(e);
    if (words.includes(UNDETERMINED)) return none("the windows kept do not vary enough to determine a head, so no head was trained");
    return none(`no head was trained: ${words}`);
  }
  let score: ReturnType<typeof blockedFolds>;
  try {
    score = blockedFolds(tk.rows, tk.labels, { encoderId: info.encoderId });
  } catch (e) {
    const words = refused(e);
    if (words.includes(UNDETERMINED)) {
      return none(
        "the windows kept determine a head, but with a block of them held out to score it, the rest do not vary " +
          "enough to determine one, so no head was trained",
      );
    }
    return none(`no head was trained: ${words}`);
  }
  return { head, score, noHead: null };
}

register(
  "head_training",
  "Train a head to tell two or more states apart: one of the pairs of states offered, or, with the pair set to own, " +
    "the person's own states, written one to a line as 'name | cue words'. Reps go through the states in turn, each " +
    "cued aloud and lasting the rep length plus four of the device's windows, of which up to the rep length are kept, " +
    "labeled by the state cued. A head is trained on every window kept, and scored, where enough can be held out, " +
    "by heads each trained without one block of the windows and tested on that block. The head is not put on the " +
    "device: that is done separately, into a slot the person chooses.",
  [
    { name: "pair", type: "select", default: "mental", choices: [...Object.keys(PAIRS).sort(), "own"] },
    { name: "states", type: "textarea", default: STATES },
    { name: "rep_length", type: "number", default: 15 },
    { name: "reps", type: "number", default: 8 },
    { name: "note", type: "text", default: "" },
  ],
  async (dev, emit, p, ctx) => {
    // Reps of two or more states, each cued aloud, while the device
    // describes every window it can. The states are one of the pairs offered,
    // named by `pair`, or with `pair` "own" the person's own, which
    // `states` holds one to a line, written "name | cue words", two lines
    // or more, refused before anything is sent to any device when not so
    // written. With any other pair `states` is not read. The states are
    // labeled 0, 1, 2 and so on in the order given, and the reps go
    // through them in that order, so two states alternate. A `rep_length`
    // below 1 would keep no window, so it is refused, saying it needs at
    // least 1, before anything is sent to any device. Fewer reps than
    // states would leave a state with no rep, so `reps` below the number of
    // states is refused, saying how many it needs, before anything is sent
    // to any device.
    const pair = get(p, "pair", null);
    if (typeof pair !== "string" || (pair !== "own" && !Object.hasOwn(PAIRS, pair))) {
      throw new Invalid(`ValueError: pair needs one of ${series([...Object.keys(PAIRS).sort(), "own"], "or")}, not ${said(pair)}`);
    }
    const cued: readonly State[] = pair === "own" ? ownStates(get(p, "states", null)) : PAIRS[pair]!;
    const repLengthWhole = whole("rep_length", get(p, "rep_length", null));
    const repsWhole = whole("reps", get(p, "reps", null));
    // A rep keeps up to rep_length windows, so with none no window is kept
    // and no head can be trained.
    if (repLengthWhole < 1n) throw new Invalid(`ValueError: rep_length needs at least 1, a window kept from each rep, not ${wholeSaid(repLengthWhole)}`);
    // The reps go through the states in order, so with fewer reps than
    // states a state is never cued and no head can be trained.
    if (repsWhole < BigInt(cued.length)) throw new Invalid(`ValueError: reps needs at least ${cued.length}, one for each state, not ${wholeSaid(repsWhole)}`);
    const repLength = Number(repLengthWhole);
    const reps = Number(repsWhole);
    const states = cued.map(([name]) => name);
    for (const m of listed(dev)) {
      if (!m.can("model")) {
        throw new SessionError(`RuntimeError: ${named(m)} runs no model, so it sends no embeddings and no head can be trained on it`);
      }
      if (typeof (m as { modelInfo?: unknown }).modelInfo !== "function") {
        throw new SessionError(`RuntimeError: ${named(m)} says it runs a model, and does not say what its model is doing, so no head can be trained on it`);
      }
    }
    // Across several devices each rep is cued once and every device's own
    // windows are labeled, kept and trained on, so each device's recording
    // comes with its own training data and its own head, for that device.
    // On one device, a link that drops ends the run at once, whether it
    // drops between reps or while a rep's windows are waited for: the
    // recording is saved with `aborted` set to "link_lost", the whole reps
    // kept beside it, and no head is trained.
    const here = step(dev, emit, ctx);
    const many = several(dev);
    let members = (await ready(here, dev)) as Thinks[];
    // A device whose link is down is neither asked nor recorded.
    const target: Thinks | Thinks[] = many ? members : (dev as Thinks);
    const who = (m: Thinks): string => (many ? `${idOfUnit(m)}: ` : "");
    // Held before anything is asked of them, so no other step starts
    // recording them while their models are read.
    const owner = {};
    const letGo = holdFor(here, members, owner);
    const infos = new Map<Thinks, ModelInfo>();
    let lbl: string;
    let started: string;
    let armed: Armed;
    const taught = new Map<Thinks, Taught>();
    try {
      for (const m of members) {
        const info = await m.modelInfo();
        if (info.modelStateName !== "ready") {
          throw new SessionError(`RuntimeError: ${who(m)}the device's model is ${info.modelStateName}, so it sends no embeddings`);
        }
        if (info.intervalS > 0) {
          throw new SessionError(
            `RuntimeError: ${who(m)}the device describes one window every ${info.intervalS} seconds, and a head ` +
              "is trained on every window: set the model's interval to 0 first",
          );
        }
        infos.set(m, info);
      }
      lbl = label(ctx, "head_training");
      started = utc(runtime.time.time());
      for (const m of members) taught.set(m, { rows: [], labels: [], windows: [], settle: 0, flagged: 0 });
      armed = await arm(here, target, 25, owner);
    } finally {
      // What arming did not take over is let go.
      letGo();
    }
    members = recorded(target, armed).map(([m]) => m as Thinks);
    let finished = false;
    let done: Done[];
    try {
      for (let rep = 0; rep < reps; rep++) {
        if (ctx.stop.isSet()) break;
        const y = rep % cued.length;
        const [state, words] = cued[y]!;
        runtime.play("speech", words);
        cue(here, state, { rep, label: y });
        await tell(here, { type: "cue", text: `${state}  (rep ${rep + 1}/${reps})` });
        const got = await windowsOf(many ? members : target, repLength + SETTLE + 2, ctx);
        for (let k = 0; k < members.length; k++) {
          const m = members[k]!;
          const tk = taught.get(m)!;
          let n = 0;
          got[k]!.forEach((w, i) => {
            if (i < SETTLE) tk.settle++;
            else if (w.gapInWindow || w.leadoffInWindow) tk.flagged++;
            else if (n < repLength) {
              const e = Array.from(w.embedding ?? []);
              tk.rows.push(e);
              tk.labels.push(y);
              tk.windows.push({ index: w.index, round: rep, label: y, embedding: e });
              n++;
            }
          });
          await tell(here, { type: "note", text: `${who(m)}kept ${n} window${n === 1 ? "" : "s"}` }, [m]);
        }
        await tell(here, { type: "progress", frac: (rep + 1) / reps });
      }
      runtime.play("speech", "Done.");
      for (const m of members) {
        const tk = taught.get(m)!;
        await tell(
          here,
          {
            type: "note",
            text:
              `${who(m)}${tk.rows.length} windows kept, ${tk.settle} dropped since they may have begun before a cue, ` +
              `${tk.flagged} dropped for a gap or an electrode off.`,
          },
          [m],
        );
      }
      finished = true;
    } finally {
      // The words a person heard are part of how the recording was made,
      // so it keeps each state's cue words beside its name, as the cued
      // protocol keeps its steps, while the training data names the states.
      done = await finish(
        here,
        target,
        armed,
        lbl,
        () =>
          dict(
            [
              ["experiment", s("head_training")],
              ["pair", s(pair)],
              ["rep_length", J.int(repLengthWhole)],
              ["reps", J.int(repsWhole)],
              ["states", J.array(cued.map(([name, words]) => J.array([s(name), s(words)])))],
              ["started_utc", s(started)],
              ["note", J.fromPlain(get(p, "note", ""))],
            ],
            J.copyObject(ctx.cfg),
          ),
        { say: false },
      );
      const name = headName(states);
      const labeled = series(states.map((state, y) => `${y} for ${state}`), "and");
      for (const { device, saved: one, ok } of done) {
        const m = device as Thinks;
        const tk = taught.get(m)!;
        const info = infos.get(m)!;
        const { head, score, noHead } = taughtHead(tk, states, name, info, ok, finished);
        let training: string | null = null;
        if (tk.rows.length > 0) {
          const width = tk.windows[0]!.embedding.length;
          training = trainingText(
            J.object([
              [
                "what",
                s(
                  `Training data for a head, from the head_training experiment: one ${width}-dimensional embedding for each ` +
                    `window the device described and the run kept, in the order they were collected, labeled ${labeled}.`,
                ),
              ],
              ["made", s(utcMs())],
              ["name", s(name)],
              ["encoderId", s(info.encoderId)],
              ["states", J.array(states.map(s))],
              ["dropped", J.int(tk.settle + tk.flagged)],
              [
                "score",
                score === null
                  ? J.NULL
                  : J.object([
                      ["accuracy", f(score.accuracy)],
                      ["correct", J.int(score.correct)],
                      ["total", J.int(score.total)],
                      ["chance", f(score.chance)],
                    ]),
              ],
            ]),
            tk.windows,
          );
        }
        await tell(
          here,
          {
            type: "saved",
            label: one.label,
            meta: J.toPlain(jsonable(one.manifest)),
            recording: one,
            head: head === null ? null : toHex(head),
            score,
            no_head: noHead,
            training,
          },
          [m],
        );
      }
    }
    return labels(done);
  },
  { enforced: {} },
);
