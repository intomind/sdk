/**
 * A recording written as the Python library's recorder writes one: the
 * samples in `name.npz`, the manifest in `name.meta.json`, the markers in
 * `name.events.json`, and `name.sha256` last, so that a run which dies on the
 * way leaves no checksums and is never taken for a whole recording.
 *
 * `Take` holds what a run collected: every sample and every packet that
 * carried them. `save` makes the recording's files from it, with the
 * manifest the library writes at its current version, built the way it
 * builds one, from what the device said, what the application declares in a
 * `Setup`, and what the run adds in a `Run`. Every recording made here reads
 * back through `Collection` and every IntoMind program, and is held to the
 * Python library's own recordings by the fixture in `contract/recorder`.
 *
 * Nothing here writes a disk: `save` hands back the files' bytes, in the
 * order the library writes them, for a browser to keep however it keeps
 * files. `capture/node` writes them into a folder.
 */

import { sha256 } from "../digest.ts";
import { GAIN_BY_CODE, INDICATOR_LEVELS, MODE_BY_CODE, RATE_BY_CODE, composeName, lowPassAutoDhz } from "../protocol.ts";
import type { ModelInterval, PipelineState, Stage, Status } from "../protocol.ts";
import { CaptureError } from "./errors.ts";
import * as montage from "./montage.ts";
import { Fail } from "./montage.ts";
import * as npy from "./npy.ts";
import type { NpyArray } from "./npy.ts";
import * as placement from "./placement.ts";
import { general } from "./pyfmt.ts";
import * as J from "./pyjson.ts";
import type { Json, Value } from "./pyjson.ts";
import { checkLabel } from "./recording.ts";
import { held, heldOrNull } from "./kinds.ts";
import { PACKAGE_VERSION } from "./version.ts";
import * as zip from "./zip.ts";
import type { Clock } from "./clock.ts";
import type { Batch } from "../session.ts";

/** The manifest version this writes. 7 is the version whose npz carries `leadoff_active`, whether lead-off detection ran at each sample. */
export const MANIFEST_VERSION = 7;

/** What one packet said about the samples it carried, as a manifest reads it. */
export interface Packet {
  /** The device's index for the packet's first sample. */
  readonly index: number;
  /** The device's time for the packet's first sample, in its ticks. */
  readonly deviceTime: bigint;
  /** The gain the samples were taken at. */
  readonly gain: number;
  /** The rate the samples were taken at. Null for a rate code the contract does not define. */
  readonly sampleRateHz: number | null;
  /** What the samples were: `normal`, `test`, `short` or `synthetic`. */
  readonly mode: string;
  /** Whether the device generated the samples rather than measuring them. */
  readonly synthetic: boolean;
}

const LITTLE = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

/** A typed array that grows as it is appended to. */
class Grow<T extends BigInt64Array | BigUint64Array | Float64Array | Int32Array | Uint8Array | Int8Array> {
  #buf: T;
  #n = 0;
  readonly #make: (n: number) => T;

  constructor(make: (n: number) => T) {
    this.#make = make;
    this.#buf = make(1024);
  }

  push(v: T[number]): void {
    if (this.#n === this.#buf.length) {
      const grown = this.#make(this.#buf.length * 2);
      (grown as unknown as { set(a: T): void }).set(this.#buf);
      this.#buf = grown;
    }
    (this.#buf as unknown as T[number][])[this.#n++] = v;
  }

  get length(): number {
    return this.#n;
  }

  /** The values appended, as their own array. */
  values(): T {
    return this.#buf.slice(0, this.#n) as T;
  }
}

const INT64_MIN = -(2n ** 63n);
const INT64_MAX = 2n ** 63n - 1n;
const TWO_63 = 2 ** 63;
const TWO_31 = 2 ** 31;
/** What numpy says of a whole number past 64 bits as it makes an array of them. */
const TOO_LARGE = "Python int too large to convert to C long";

/** What a run collected: every sample, and every packet that carried them. */
export class Take {
  readonly #index = new Grow((n) => new BigInt64Array(n));
  readonly #ticks = new Grow((n) => new BigInt64Array(n));
  readonly #host = new Grow((n) => new Float64Array(n));
  // Counts and lead-off words as the numbers they arrived as: the recording
  // keeps every count as numpy's 64 bits do, and refuses, as numpy does, a lead-off word that
  // does not fit in the type the recording keeps it in. Once a count arrives that a number does not hold
  // exactly, every count is kept as a bigint from then on.
  readonly #counts = new Grow((n) => new Float64Array(n));
  #exact: Grow<BigInt64Array> | null = null;
  readonly #leadoff = new Grow((n) => new Float64Array(n));
  // A lead-off word past what a number holds exactly, once one arrives:
  // every word from then on, as the bigint it is.
  #leadoffExact: bigint[] | null = null;
  readonly #gap = new Grow((n) => new Uint8Array(n));
  readonly #active = new Grow((n) => new Int8Array(n));
  readonly #judged = new Grow((n) => new Uint8Array(n));
  readonly #packets: Packet[] = [];
  #channels: number | null = null;
  #mixed = false;
  /** Whether a count falls outside 32 bits, so the counts are widened one by one. */
  #wide = false;
  /** What numpy would refuse first of each array as the recording is saved, in its words, or null. */
  #indexWrong: string | null = null;
  #ticksWrong: string | null = null;
  #countsWrong: string | null = null;

  /** One packet, as it arrived. */
  packet(p: Packet): void {
    this.#packets.push(p);
  }

  /**
   * One sample: its index, its time on the device, its time on the host, its counts across the channels,
   * the lead-off word of its packet, whether a break comes before it,
   * whether lead-off detection was running, null where the source did not
   * say, and whether its lead-off bits are a verdict on each electrode's
   * contact, false where they are none, as for a sample before the device's
   * first lead-off judgment of a stream. A count or a lead-off word past
   * what a number holds exactly is given as a bigint.
   */
  sample(
    index: bigint,
    deviceTicks: bigint,
    hostTime: number,
    counts: ArrayLike<number | bigint>,
    leadoff: number | bigint,
    gapBefore: boolean,
    leadoffActive: boolean | null,
    leadoffJudged = true,
  ): void {
    if (this.#channels === null) this.#channels = counts.length;
    else if (this.#channels !== counts.length) this.#mixed = true;
    if (this.#ticksWrong === null && (deviceTicks < INT64_MIN || deviceTicks > INT64_MAX)) this.#ticksWrong = TOO_LARGE;
    if (this.#indexWrong === null && (index < INT64_MIN || index > INT64_MAX)) this.#indexWrong = TOO_LARGE;
    this.#index.push(index);
    this.#ticks.push(deviceTicks);
    this.#host.push(hostTime);
    for (let c = 0; c < counts.length; c++) {
      const v = counts[c]!;
      if (typeof v === "bigint") {
        if (v >= 2n ** 31n || v < -(2n ** 31n)) {
          this.#wide = true;
          if (this.#countsWrong === null && (v > INT64_MAX || v < INT64_MIN)) this.#countsWrong = TOO_LARGE;
        }
        if (this.#exact === null && !Number.isSafeInteger(Number(v))) {
          // From here on every count is kept as the bigint it is.
          const kept = new Grow((n) => new BigInt64Array(n));
          for (const old of this.#counts.values()) kept.push(BigInt(old));
          this.#exact = kept;
        }
        if (this.#exact !== null) this.#exact.push(BigInt.asIntN(64, v));
        else this.#counts.push(Number(v));
        continue;
      }
      if (v >= TWO_31 || v < -TWO_31) {
        this.#wide = true;
        if (this.#countsWrong === null && (v >= TWO_63 || v < -TWO_63)) this.#countsWrong = TOO_LARGE;
      }
      if (this.#exact !== null) this.#exact.push(BigInt(v));
      else this.#counts.push(v);
    }
    if (this.#leadoffExact === null && typeof leadoff === "bigint" && !Number.isSafeInteger(Number(leadoff))) {
      this.#leadoffExact = Array.from(this.#leadoff.values(), (v) => BigInt(v));
    }
    if (this.#leadoffExact !== null) this.#leadoffExact.push(BigInt(leadoff));
    else this.#leadoff.push(Number(leadoff));
    this.#gap.push(gapBefore ? 1 : 0);
    this.#active.push(leadoffActive === null ? -1 : leadoffActive ? 1 : 0);
    this.#judged.push(leadoffJudged ? 1 : 0);
  }

  /** How many samples were collected. */
  get samples(): number {
    return this.#index.length;
  }

  /** The packets, in the order they arrived. */
  get packets(): readonly Packet[] {
    return this.#packets;
  }

  /** Each sample's index. */
  get index(): BigInt64Array {
    return this.#index.values();
  }

  /** Each sample's device time, in its ticks. */
  get deviceTicks(): BigInt64Array {
    return this.#ticks.values();
  }

  /**
   * The arrays as `np.savez_compressed` writes them for the recorder, in
   * its order, refused where numpy refuses to make one, in its words, in
   * the order the recorder makes them. The lead-off words are kept at the
   * width the library keeps them at: a byte for the IntoMind device.
   */
  arrays(): [string, NpyArray][] {
    const n = this.samples;
    if (this.#indexWrong !== null) throw new CaptureError(this.#indexWrong);
    if (this.#ticksWrong !== null) throw new CaptureError(this.#ticksWrong);
    if (this.#mixed) {
      throw new CaptureError(
        `setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (${n},) + inhomogeneous part.`,
      );
    }
    if (this.#countsWrong !== null) throw new CaptureError(this.#countsWrong);
    const leadoff = this.#leadoffArray();
    const channels = this.#channels ?? 0;
    // The counts channels by samples, stored sample after sample, which
    // numpy marks as Fortran order wherever both dimensions exceed one.
    const flat = this.#counts.values();
    let values: BigInt64Array;
    if (this.#exact !== null) {
      values = this.#exact.values();
    } else if (this.#wide) {
      values = BigInt64Array.from(flat, (v) => BigInt(v));
    } else {
      // Each count widened to 64 bits, its sign carried into the high word,
      // in the platform's own order, without making a bigint of every one.
      const words = new Int32Array(flat.length * 2);
      const [low, high] = LITTLE ? [0, 1] : [1, 0];
      for (let k = 0; k < flat.length; k++) {
        words[2 * k + low] = flat[k]!;
        words[2 * k + high] = flat[k]! < 0 ? -1 : 0;
      }
      values = new BigInt64Array(words.buffer);
    }
    const counts: NpyArray =
      n === 0
        ? npy.vector({ kind: "i8", values: new BigInt64Array(0) })
        : { shape: [channels, n], fortran: channels > 1 && n > 1, data: { kind: "i8", values } };
    const out: [string, NpyArray][] = [
      ["counts", counts],
      ["index", npy.vector({ kind: "i8", values: this.#index.values() })],
      ["device_ticks", npy.vector({ kind: "i8", values: this.#ticks.values() })],
      ["host_time", npy.vector({ kind: "f8", values: this.#host.values() })],
      ["leadoff", leadoff],
      ["gap_before", npy.vector({ kind: "b1", values: this.#gap.values() })],
    ];
    // Whether detection was running at each sample, from the packet's own
    // flag. A zero lead-off word means an attached electrode only where this
    // is true, so it is kept beside the word, and only when every sample
    // said.
    const active = this.#active.values();
    if (active.every((a) => a >= 0)) {
      out.push(["leadoff_active", npy.vector({ kind: "b1", values: Uint8Array.from(active) })]);
    }
    // Whether each sample's lead-off bits are a verdict on contact, kept
    // only where one is not: from firmware 1.4.6 the IntoMind device's bits
    // read 0 until its first judgment of a stream, which says neither
    // attached nor off, and on an older firmware, which cannot check
    // electrode contact, no bit sent with detection on is a verdict.
    const judged = this.#judged.values();
    if (!judged.every((j) => j === 1)) out.push(["leadoff_judged", npy.vector({ kind: "b1", values: Uint8Array.from(judged) })]);
    return out;
  }

  /** The lead-off words as the library's recorder keeps them, a byte wide or as wide as the words need, refused as numpy refuses a word its type does not hold. */
  #leadoffArray(): NpyArray {
    const words: bigint[] = this.#leadoffExact ?? Array.from(this.#leadoff.values(), (v) => BigInt(v));
    let bits = 8;
    let most: bigint | null = null;
    for (const w of words) if (most === null || w > most) most = w;
    if (most !== null) {
      const m = most < 0n ? -most : most;
      bits = Math.max(bits, m === 0n ? 0 : m.toString(2).length);
    }
    const width = [8, 16, 32, 64].find((b) => bits <= b) ?? 64;
    for (const w of words) {
      if (w < 0n || w >= 1n << BigInt(width)) throw new CaptureError(`Python integer ${w} out of bounds for uint${width}`);
    }
    switch (width) {
      case 8:
        return npy.vector({ kind: "u1", values: Uint8Array.from(words, Number) });
      case 16:
        return npy.vector({ kind: "u2", values: Uint16Array.from(words, Number) });
      case 32:
        return npy.vector({ kind: "u4", values: Uint32Array.from(words, Number) });
      default:
        return npy.vector({ kind: "u8", values: BigUint64Array.from(words) });
    }
  }
}

/** What each mode code's samples are, as a manifest names them. */
const MODE_WORDS = ["normal", "test", "short", "synthetic"] as const;

/**
 * Give `take` a batch and its samples, each dated as the Python library
 * dates it: the packet's device time stepped by the rate for each sample
 * after the first, and the clock's map now. What a session's own recorder
 * and a run's recorder both collect. Internal.
 */
export function collect(take: Take, b: Batch, tickHz: number, clock: Clock | null): void {
  const rate = b.sampleRateHz;
  take.packet({
    index: b.index,
    deviceTime: b.deviceTime,
    gain: b.gain,
    sampleRateHz: rate,
    mode: MODE_WORDS[b.mode & 3]!,
    synthetic: b.synthetic,
  });
  const tickPer = rate !== null && rate > 0 ? tickHz / rate : 0;
  const rows = b.rows;
  for (let i = 0; i < rows; i++) {
    const ticks = tickPer === 0 ? b.deviceTime : BigInt(Math.trunc(Number(b.deviceTime) + i * tickPer));
    const host = clock !== null ? clock.hostTime(ticks) : Number(ticks) / tickHz;
    take.sample(
      BigInt(b.index) + BigInt(i),
      ticks,
      host,
      b.counts.subarray(i * b.channels, (i + 1) * b.channels),
      b.leadoff,
      b.gapBefore && i === 0,
      b.leadoffActive,
      b.leadoffJudged(i),
    );
  }
}

/** What the device said about itself, read when the recording is saved. */
export interface DeviceFacts {
  /** The device's id, as lowercase hex. */
  readonly deviceId: string;
  /** Its firmware version, `major.minor.patch`. */
  readonly firmware: string;
  /** The protocol version it speaks, `major.minor`. */
  readonly protocol: string | null;
  /** Its clock's ticks per second. */
  readonly tickHz: number;
  /** Its converter's reference, in microvolts. */
  readonly vrefUv: number;
  /** Its converter's width, in bits. */
  readonly adcBits: number | null;
  /** Its channels. */
  readonly channels: number | null;
  /** The clock record at saving: how far `host_time` can be trusted. */
  readonly clock: Value | null;
  /** Breaks the device announced while this run was armed. */
  readonly gapsInRun: number;
}

/** The commit of the application's source tree, and whether it has changes, as `capture/node`'s `gitState` reads them. */
export interface GitState {
  /** The commit, or null where it could not be read. */
  readonly commit: string | null;
  /** Whether the tree has changes, or null where it could not be read. */
  readonly dirty: boolean | null;
}

/**
 * What the application declares about the setup a recording was made on.
 * Nothing here is guessed: what is not declared is written as unknown. A
 * JSON field takes plain data or a `Value`, as `fromPlain` reads them.
 */
export interface Setup {
  /**
   * Facts about the hardware no code can discover, as an object: the keys the
   * Python library's `use_site` takes, `board_revision`, `mains_hz`,
   * `psu_volts`, `firmware_sha256`, `firmware_artifact`,
   * `electrode_montage`, `subject_id`, `notes`, and a structured `montage`.
   */
  readonly site?: Json;
  /** The commit of the application doing the recording and whether it has changes. Unknown when not declared. */
  readonly git?: GitState | null;
  /** The adapter that carried the samples: `name`, `version`, `first_party` and `transport`. Unknown when not declared. */
  readonly adapter?: Json;
  /** The filter the device was set to, when one was: `on`, `hp_hz`, `lp_hz` and `bands`. */
  readonly filter?: Json;
  /** Where a filter ran, `device` unless the adapter filtered on the host. */
  readonly filtersIn?: string | null;
}

/** What the run adds to its recording. A JSON field takes plain data or a `Value`, as `fromPlain` reads them. */
export interface Run {
  /**
   * The run's own fields, as an object. `experiment` names what was run.
   * `started_utc`, `electrode_layout`, `placement`, `regs_before` and
   * `config_before` are read as the Python library reads them, and every
   * other field joins the manifest. A null never replaces a value the device
   * measured.
   */
  readonly extra?: Json;
  /** The markers, written as `name.events.json` when there are any to keep, an empty list included. */
  readonly events?: Json;
  /** The registers after the run, from an adapter that reads them. */
  readonly regsAfter?: Json;
  /** What the device said it was set to after the run. */
  readonly configAfter?: Json;
  /** Why the run ended early, `link_lost` for a dropped link. An aborted recording gets no checksums. */
  readonly aborted?: string | null;
  /** When the recording finished, in seconds since 1970. The host clock now, when not given. */
  readonly finishedUnix?: number;
}

/** A recording, as the files the library writes for it. */
export interface Saved {
  /** The recording's name. */
  readonly label: string;
  /** Its files, by name, in the order they are written: the npz, the manifest, the events when there are any, and the checksums last when the run finished. */
  readonly files: readonly (readonly [string, Uint8Array])[];
  /** The manifest, as Python holds it. */
  readonly manifest: Value;
  /** The manifest file's text. */
  readonly manifestJson: string;
  /** The line the recordings folder's session log, `SESSIONS.jsonl`, gains. */
  readonly sessionLine: string;
  /** Whether the run finished, so the recording carries checksums and is made read-only in a folder. */
  readonly finalized: boolean;
}

/** What a recording made by this package says made it. */
export function hostVersions(): Value {
  const p = (globalThis as { process?: { platform?: string; arch?: string } }).process;
  const platform = p !== undefined && typeof p.platform === "string" ? `${p.platform}-${p.arch}` : "browser";
  return J.object([
    ["intomind_js", J.str(PACKAGE_VERSION)],
    ["platform", J.str(platform)],
  ]);
}

function hex(b: Uint8Array): string {
  let s = "";
  for (const x of b) s += x.toString(16).padStart(2, "0");
  return s;
}

/** The checksum file the library writes over `files`: each file's sha256 and name, in name order. */
export function checksums(files: readonly (readonly [string, Uint8Array])[]): Uint8Array {
  const sums = files.map(([name, bytes]): [string, string] => [name, hex(sha256(bytes))]);
  sums.sort((a, b) => J.compareKeys(a[0], b[0]));
  return new TextEncoder().encode(sums.map(([k, v]) => `${v}  ${k}\n`).join(""));
}

/**
 * The recording `label` from `take`, as the files the Python library writes
 * for it, with its manifest. `versions` names the software that made it,
 * this package by default.
 */
export function save(
  label: string,
  take: Take,
  device: DeviceFacts,
  setup: Setup = {},
  run: Run = {},
  versions: Value = hostVersions(),
): Saved {
  return saveWith(label, take, device, setup, run, versions, { run: null, endedByUsb: false });
}

/** What a manifest says beyond what `save` is handed, as the library's recorder adds it. Internal. */
export interface Made {
  /** For a recording a run made, which run it was: the experiment's registered name and when the run started. */
  readonly run: Value | null;
  /** Whether the device ended its stream itself because USB power appeared, as the library's `Device.ended_by_usb` says. */
  readonly endedByUsb: boolean;
  /** For an IntoMind device, the gain it last stated, as the library's `Device._gain`, which its recording carries where it kept no packet. */
  readonly statedGain?: number | null;
}

/** The gain each IntoMind device last stated, by its session. */
const STATED_GAIN = new WeakMap<object, number>();

/**
 * The gain the IntoMind device of `session` last stated, as the library's
 * `Device._gain`: its last packet's, the gain last set on it, or its Status
 * as last read, and 6 before any of them has said. A recording carries it
 * where it kept no packet. Internal.
 */
export function statedGain(session: object): number {
  return STATED_GAIN.get(session) ?? 6;
}

/** Keep `gain` as the gain the IntoMind device of `session` last stated. Internal. */
export function stateGain(session: object, gain: number): void {
  STATED_GAIN.set(session, gain);
}

/**
 * `save`, and what the library's recorder adds after the names every
 * earlier reader reads: `run`, for a recording a run made, and
 * `ended_by_usb`, for a device that ended its stream because USB power
 * appeared. Internal.
 */
export function saveWith(
  label: string,
  take: Take,
  device: DeviceFacts,
  setup: Setup,
  run: Run,
  versions: Value,
  made: Made,
): Saved {
  checkLabel(label);
  const finished = run.finishedUnix ?? Date.now() / 1000;
  const npzBytes = zip.write(
    take.arrays().map(([k, a]) => [`${k}.npy`, npy.write(a)]),
    Math.trunc(Math.max(0, finished)),
  );

  const given = J.fromPlain(run.extra ?? {});
  if (given.kind !== "object") throw new CaptureError("a run's extra fields are an object");
  const extra = J.copyObject(given);
  const protocol = J.pop(extra, "experiment");
  const regsBefore = J.pop(extra, "regs_before");
  const params = J.copyObject(extra);
  let meta: Value;
  try {
    meta = buildManifest(
      label,
      device,
      setup,
      take.packets,
      regsBefore,
      J.fromPlain(run.regsAfter),
      protocol,
      params,
      extra,
      take.index,
      take.deviceTicks,
      J.fromPlain(run.configAfter),
      finished,
      versions,
      made.statedGain ?? null,
    );
  } catch (e) {
    if (e instanceof Fail) throw new CaptureError(e.shown);
    throw e;
  }
  J.set(meta, "n", J.int(take.samples));
  // The names every earlier reader reads.
  for (const [k, v] of [
    ["fw", J.str(device.firmware)],
    ["experiment", protocol],
    ["regs_before", regsBefore],
  ] as const) {
    if (J.get(meta, k) === undefined) J.set(meta, k, v);
  }
  if (made.run !== null) J.set(meta, "run", J.copyObject(made.run));
  if (made.endedByUsb) J.set(meta, "ended_by_usb", J.bool(true));
  const aborted = run.aborted ?? null;
  if (aborted !== null && aborted !== "") J.set(meta, "aborted", J.str(aborted));
  const manifestJson = J.dumpsIndented(meta, 2);
  const encode = (s: string): Uint8Array => new TextEncoder().encode(s);
  const files: [string, Uint8Array][] = [
    [`${label}.npz`, npzBytes],
    [`${label}.meta.json`, encode(manifestJson)],
  ];
  const events = run.events === undefined ? J.NULL : J.fromPlain(run.events);
  if (events.kind !== "null") files.push([`${label}.events.json`, encode(J.dumpsIndented(events, 2))]);
  // Checksums mark a recording finalized. An aborted run never gets them.
  const finalized = aborted === null || aborted === "";
  if (finalized) files.push([`${label}.sha256`, checksums(files)]);
  return { label, files, manifest: meta, manifestJson, sessionLine: sessionLine(meta), finalized };
}

/** One line for the recordings folder's session log, as the library appends it, with `run`, which run made the recording, when the manifest has it. Internal. */
export function sessionLine(meta: Value): string {
  const row = J.object([]);
  for (const k of [
    "label",
    "started_utc",
    "finished_utc",
    "protocol_id",
    "board_revision",
    "firmware_version",
    "gain",
    "rate_sps",
    "device_config_changed_by_run",
    "git_commit",
  ]) {
    J.set(row, k, J.field(meta, k));
  }
  J.set(row, "n", J.field(meta, "n"));
  J.set(row, "samples_lost", J.field(J.field(meta, "quality"), "samples_lost"));
  const run = J.get(meta, "run");
  if (run !== undefined) J.set(row, "run", run);
  return J.dumps(row) + "\n";
}

/** The library's `site()`: every fact it records, unknown until declared. */
function site(declared: Json): Value {
  const s = J.object([
    ["board_revision", J.str("unknown")],
    ["mains_hz", J.NULL],
    ["psu_volts", J.NULL],
    ["firmware_sha256", J.NULL],
    ["firmware_artifact", J.NULL],
    ["electrode_montage", J.NULL],
    ["subject_id", J.NULL],
    ["notes", J.str("")],
  ]);
  const d = J.fromPlain(declared);
  if (d.kind === "object") for (const [k, v] of d.entries) J.set(s, k, v);
  return s;
}

/**
 * The structured montage, from where the electrodes were put or from the
 * site, and whether it agrees with the device. A montage that cannot be true
 * is recorded with its error rather than refusing a recording that has
 * already been made.
 */
export function montageFields(siteCfg: Value, devChannels: number | null, layout: Value, place: Value): Value {
  const ok = (m: Value): Value => (devChannels === null ? J.NULL : J.bool(montage.channelCount(m) === devChannels));
  const fields = (m: Value, err: Value, matches: Value, source: Value, lay: Value, placed: Value): Value =>
    J.object([
      ["montage", m],
      ["montage_error", err],
      ["montage_matches_device", matches],
      ["montage_source", source],
      ["electrode_layout", lay],
      ["placement", placed],
    ]);
  const recorded = (e: unknown, kinds: readonly string[]): Fail | null =>
    e instanceof Fail && kinds.includes(e.kind) ? e : null;
  if (layout.kind !== "null" && place.kind !== "null") {
    let m: Value;
    let r: Value;
    try {
      const lay = placement.layoutFrom(layout) ?? J.NULL;
      m = placement.toMontage(lay, place);
      r = placement.resolve(lay, place);
    } catch (e) {
      const f = recorded(e, ["montage", "placement"]);
      if (f === null) throw e;
      return fields(J.NULL, J.str(f.message), J.bool(false), J.str("placement"), layout, J.NULL);
    }
    return fields(m, J.NULL, ok(m), J.str("placement"), layout, r);
  }
  let m: Value | null;
  try {
    m = montage.fromSite(siteCfg);
  } catch (e) {
    const f = recorded(e, ["montage"]);
    if (f === null) throw e;
    return fields(J.NULL, J.str(f.message), J.bool(false), J.str("site"), J.NULL, J.NULL);
  }
  if (m === null) return fields(J.NULL, J.NULL, J.NULL, J.NULL, J.NULL, J.NULL);
  return fields(m, J.NULL, ok(m), J.str("site"), J.NULL, J.NULL);
}

/** Where the breaks are, not only how many. A count that steps back is an epoch reset and lost nothing. */
export function gapLedger(index: BigInt64Array): Value {
  const gaps: Value[] = [];
  let resets = 0;
  let lost = 0n;
  for (let i = 0; i + 1 < index.length; i++) {
    const d = index[i + 1]! - index[i]!;
    if (d > 1n) {
      gaps.push(J.object([["after_sample", J.int(index[i]!)], ["lost", J.int(d - 1n)]]));
      lost += d - 1n;
    } else if (d < 0n) {
      resets += 1;
    }
  }
  const received = BigInt(index.length);
  return J.object([
    ["gaps", J.array(gaps)],
    ["n_gaps", J.int(gaps.length)],
    ["epoch_resets", J.int(resets)],
    ["samples_received", J.int(received)],
    ["samples_expected", J.int(received + lost)],
    ["samples_lost", J.int(lost)],
  ]);
}

/**
 * The rate from the device's clock: samples over the span of their ticks.
 * Given the sample `index`, the ticks are measured over the stretches where
 * the index climbs by exactly one, as `rateOverRuns` measures a clock, so a
 * hole, where the index skips the samples that were lost, is not read as
 * slow sampling. For a recording without holes or breaks, and for one with
 * no two samples in a row, where no stretch can be measured, the rate is
 * the samples less one over the span of all their ticks.
 */
export function fsFromTicks(ticks: BigInt64Array, tickHz: number, index: ArrayLike<bigint> | null = null): number {
  if (ticks.length < 2 || tickHz === 0) return 0;
  if (index === null || index.length !== ticks.length) {
    const span = (Number(ticks[ticks.length - 1]!) - Number(ticks[0]!)) / tickHz;
    return span > 0 ? (ticks.length - 1) / span : 0;
  }
  let count = 0;
  let span = 0;
  let a = 0;
  const n = index.length;
  for (let i = 1; i <= n; i++) {
    if (i === n || index[i]! - index[i - 1]! !== 1n) {
      if (i - a >= 2) {
        count += i - a - 1;
        span += Number(ticks[i - 1]!) - Number(ticks[a]!);
      }
      a = i;
    }
  }
  if (count === 0) return fsFromTicks(ticks, tickHz);
  const seconds = span / tickHz;
  return seconds > 0 ? count / seconds : 0;
}

/** The rate from the first and last packets' index and device time. */
function fsFromPackets(packets: readonly Packet[], tickHz: number): number {
  if (packets.length < 2) return 0;
  const p0 = packets[0]!;
  const p1 = packets[packets.length - 1]!;
  const span = Number(p1.deviceTime - p0.deviceTime) / tickHz;
  return span > 0 ? (p1.index - p0.index) / span : 0;
}

/** What `for x in value` yields: a list's items, a string's characters, an object's keys. */
function pyIter(v: Value): Value[] {
  if (v.kind === "array") return [...v.items];
  if (v.kind === "str") return Array.from(v.value, (c) => J.str(c));
  if (v.kind === "object") return v.entries.map(([k]) => J.str(k));
  throw new Fail("other", `TypeError: ${J.jsRepr(v)} is not iterable`);
}

/**
 * What `dict(v)` makes of a value a program declared: an object as itself,
 * and a list of key and value pairs as an object, keys Python holds equal
 * (`1`, `1.0` and `True`) kept once in their first form with the last
 * value, and written as `json.dumps` writes a key that is not text.
 */
function pyDict(v: Value): Value {
  if (v.kind === "object") return v;
  const out: [Value, Value][] = [];
  pyIter(v).forEach((item, i) => {
    if (item.kind !== "array" && item.kind !== "str" && item.kind !== "object") {
      throw new Fail("other", `TypeError: cannot convert dictionary update sequence element #${i} to a sequence`);
    }
    const pair = pyIter(item);
    if (pair.length !== 2) {
      throw new Fail("other", `ValueError: dictionary update sequence element #${i} has length ${pair.length}; 2 is required`);
    }
    const [key, value] = pair as [Value, Value];
    if (J.jsonKey(key) === null) throw new Fail("other", `TypeError: unhashable type: ${J.jsRepr(key)}`);
    const slot = out.find(([have]) => J.pyEq(have, key));
    if (slot) slot[1] = value;
    else out.push([key, value]);
  });
  return J.object(out.map(([k, x]) => [J.jsonKey(k)!, x]));
}

/** Which adapter produced the recording, and where it was filtered. */
export function signalChain(setup: Setup): Value {
  const declared = J.fromPlain(setup.adapter);
  const adapter = J.truthy(declared)
    ? pyDict(declared)
    : J.object([
        ["name", J.str("unknown")],
        ["version", J.NULL],
        ["first_party", J.bool(false)],
        ["transport", J.NULL],
      ]);
  // `getattr(dev, "filter", None) or {}`, read with `.get`, which only a dict has.
  const f = J.fromPlain(setup.filter);
  if (J.truthy(f) && f.kind !== "object") throw new Fail("other", `AttributeError: ${J.jsRepr(f)} has no attribute 'get'`);
  const on = J.truthy(J.field(f, "on"));
  let filters: Value = J.NULL;
  if (on) {
    // `[list(b) for b in (f.get("bands") or [])]`
    const bands = J.field(f, "bands");
    const each = J.truthy(bands) ? pyIter(bands) : [];
    filters = J.object([
      ["hp_hz", J.field(f, "hp_hz")],
      ["lp_hz", J.field(f, "lp_hz")],
      ["bands", J.array(each.map((b) => J.array(pyIter(b))))],
    ]);
  }
  return J.object([
    ["adapter", adapter],
    ["filtering", J.str(on ? (setup.filtersIn ?? "device") : "none")],
    ["filters", filters],
  ]);
}

/** What the samples were, off the packets that carried them. */
function signalSource(packets: readonly Packet[]): [Value, Value] {
  const modes = [...new Set(packets.map((p) => p.mode))].sort(J.compareKeys);
  const synthetic = J.bool(packets.some((p) => p.synthetic));
  if (modes.length === 0) return [J.NULL, J.NULL];
  if (modes.length === 1) return [J.str(modes[0]!), synthetic];
  return [J.array(modes.map(J.str)), synthetic];
}

/** UTC as `YYYY-MM-DDTHH:MM:SSZ`, from seconds since 1970, as `time.strftime` writes it from `time.gmtime`. */
export function utc(unix: number): string {
  const secs = Math.floor(unix);
  const days = Math.floor(secs / 86400);
  const rem = secs - days * 86400;
  const z = days + 719468;
  const era = Math.floor(z / 146097);
  const doe = z - era * 146097;
  const yoe = Math.floor((doe - Math.floor(doe / 1460) + Math.floor(doe / 36524) - Math.floor(doe / 146096)) / 365);
  const doy = doe - (365 * yoe + Math.floor(yoe / 4) - Math.floor(yoe / 100));
  const mp = Math.floor((5 * doy + 2) / 153);
  const d = doy - Math.floor((153 * mp + 2) / 5) + 1;
  const m = mp < 10 ? mp + 3 : mp - 9;
  const y = yoe + era * 400 + (m <= 2 ? 1 : 0);
  const two = (v: number): string => String(v).padStart(2, "0");
  return `${String(y).padStart(4, "0")}-${two(m)}-${two(d)}T${two(Math.floor(rem / 3600))}:${two(Math.floor(rem / 60) % 60)}:${two(rem % 60)}Z`;
}

/** What changed between two readings of the instrument, in key order. */
function diff(before: Value, after: Value): Value {
  if (before.kind !== "object" || after.kind !== "object") return J.object([]);
  const keys = [...new Set([...before.entries.map(([k]) => k), ...after.entries.map(([k]) => k)])].sort(J.compareKeys);
  const out: [string, Value][] = [];
  for (const k of keys) {
    const x = J.field(before, k);
    const y = J.field(after, k);
    if (!J.pyEq(x, y)) out.push([k, J.array([x, y])]);
  }
  return J.object(out);
}

/** The manifest, built as the library's `build_manifest` builds it. Internal. */
export function buildManifest(
  label: string,
  device: DeviceFacts,
  setup: Setup,
  packets: readonly Packet[],
  regsBefore: Value,
  regsAfter: Value,
  protocol: Value,
  params: Value,
  extra: Value,
  index: BigInt64Array,
  ticks: BigInt64Array,
  configAfter: Value,
  finished: number,
  versions: Value,
  statedGain: number | null = null,
): Value {
  const s = site(setup.site);
  const layout = J.pop(extra, "electrode_layout");
  const place = J.pop(extra, "placement");
  // The rate from the device's clock, which a reset cannot bend, read over
  // the stretches where the index climbs by one, so a recording missing
  // samples reads its true rate, not one slowed by its holes, and otherwise
  // the packets'.
  const fs = ticks.length >= 2 ? fsFromTicks(ticks, device.tickHz, index) : fsFromPackets(packets, device.tickHz);
  const configBefore = J.field(extra, "config_before");
  const changed =
    J.truthy(regsBefore) && J.truthy(regsAfter)
      ? diff(regsBefore, regsAfter)
      : J.truthy(configBefore) && J.truthy(configAfter)
        ? diff(configBefore, configAfter)
        : J.object([]);
  const started = J.pop(extra, "started_utc");
  const opt = (v: number | null): Value => (v === null ? J.NULL : J.int(v));
  const m = J.object([
    ["manifest_version", J.int(MANIFEST_VERSION)],
    ["label", J.str(label)],
    ["started_utc", started],
    ["finished_utc", J.str(utc(finished))],
    ["captured_unix", J.float(finished)],
    ["device_id", J.str(device.deviceId)],
    ["firmware_version", J.str(device.firmware)],
    ["firmware_sha256", J.field(s, "firmware_sha256")],
    ["firmware_artifact", J.field(s, "firmware_artifact")],
    ["tick_hz", held(device, "tickHz", device.tickHz)],
    ["vref_uv", held(device, "vrefUv", device.vrefUv)],
    ["adc_bits", opt(device.adcBits)],
    ["channels", opt(device.channels)],
    ["device_proto", device.protocol === null ? J.NULL : J.str(device.protocol)],
    ["board_revision", J.field(s, "board_revision")],
    ["electrode_montage", J.field(s, "electrode_montage")],
  ]);
  const placed = montageFields(s, device.channels, layout, place);
  if (placed.kind === "object") for (const [k, v] of placed.entries) J.set(m, k, v);
  const last = packets.length > 0 ? packets[packets.length - 1]! : null;
  const [source, synthetic] = signalSource(packets);
  const git = setup.git ?? null;
  const pairs: [string, Value][] = [
    ["mains_hz", J.field(s, "mains_hz")],
    ["psu_volts", J.field(s, "psu_volts")],
    ["subject_id", J.field(s, "subject_id")],
    ["regs_before", regsBefore],
    ["regs_after", regsAfter],
    ["regs_diff", changed],
    ["config_after", configAfter],
    ["device_config_changed_by_run", J.bool(J.truthy(changed))],
    ["gain", last === null ? J.NULL : held(last, "gain", last.gain)],
    ["rate_sps", last === null ? J.NULL : heldOrNull(last, "sampleRateHz", last.sampleRateHz)],
    ["fs_effective", J.float(fs)],
    ["protocol_id", protocol],
    ["protocol_params", params],
    ["signal_chain", signalChain(setup)],
    ["signal_source", source],
    ["synthetic", synthetic],
    ["quality", gapLedger(index)],
    ["gaps_announced", J.int(device.gapsInRun)],
    ["clock", device.clock ?? J.NULL],
    ["git_commit", git === null || git.commit === null ? J.NULL : J.str(git.commit)],
    ["git_dirty", git === null || git.dirty === null ? J.NULL : J.bool(git.dirty)],
    ["host_versions", versions],
  ];
  for (const [k, v] of pairs) J.set(m, k, v);
  if (J.field(m, "gain").kind === "null" && statedGain !== null) {
    // The IntoMind device's gain is the one its packets carried, or where
    // none was kept the one it last stated, and never a setup's: from
    // firmware 1.4.6 it samples at gain 1 while lead-off detection is on,
    // whatever gain was set.
    J.set(m, "gain", J.int(statedGain));
  }
  // A run's fields may add to the manifest. They may not erase a
  // measurement: a null never wins against a value the device gave, and a
  // recording's gain is the one its device states, never a setup's.
  if (extra.kind === "object") {
    for (const [k, v] of extra.entries) {
      if (v.kind === "null" && J.field(m, k).kind !== "null") continue;
      if (k === "gain") continue;
      J.set(m, k, v);
    }
  }
  return m;
}

/** A stage's kind in words, `kind N` for one the contract does not define. */
function kindName(kind: number): string {
  switch (kind) {
    case 1:
      return "highpass";
    case 2:
      return "lowpass";
    case 3:
      return "notch";
    default:
      return `kind ${kind}`;
  }
}

/**
 * A processing chain in words, as the Python library says it in a manifest
 * and in a BDF or EDF header: `natural signal` for no stage, corners in
 * hertz, and the automatic low-pass at the rate it follows.
 */
export function describeChain(stages: readonly Stage[], rateSps: number | null = null): string {
  if (stages.length === 0) return "natural signal";
  const hz = (dhz: number): string => general(dhz / 10, 6);
  const words: string[] = [];
  for (const s of stages) {
    const p = s.params;
    if (s.kind === 1 && p.length >= 1) {
      words.push(`high-pass ${hz(p[0]!)} Hz`);
    } else if (s.kind === 2 && p.length >= 1 && p[0] !== 0) {
      words.push(`low-pass ${hz(p[0]!)} Hz`);
    } else if (s.kind === 2 && p.length >= 1) {
      words.push(
        rateSps !== null && rateSps !== 0
          ? `low-pass ${general(lowPassAutoDhz(rateSps) / 10, 6)} Hz (automatic at ${rateSps} SPS)`
          : "low-pass automatic",
      );
    } else if (s.kind === 3 && p.length >= 2) {
      words.push(`notch ${hz(p[0]!)} to ${hz(p[1]!)} Hz`);
    } else {
      words.push(`${kindName(s.kind)} [${p.join(", ")}]`);
    }
  }
  return words.join(", ");
}

/** A chain in force as a manifest stores it: whose it is, its stages, and the chain in words at the rate given. */
function chainValue(state: PipelineState, rateSps: number | null): Value {
  const origin = state.origin === 0 ? "default" : state.origin === 1 ? "host" : `origin ${state.origin}`;
  return J.object([
    ["origin", J.str(origin)],
    [
      "stages",
      J.array(
        state.stages.map((s) =>
          J.object([
            ["kind", J.int(s.kind)],
            ["name", J.str(kindName(s.kind))],
            ["params", J.array(s.params.map((x) => J.int(x)))],
          ]),
        ),
      ),
    ],
    ["description", J.str(describeChain(state.stages, rateSps))],
  ]);
}

/**
 * What a recording stores as `processing`, the chain the device ran on the
 * signal, as the Python library reads it after a run: the chain in force at
 * the rate the device was set to, read from a device that claims a
 * processing chain, or the natural signal from one that does not. Null where
 * a device that claims one did not answer.
 */
export function processing(capable: boolean, state: PipelineState | null, rateSps: number | null): Value | null {
  if (!capable) {
    return J.object([
      ["origin", J.str("none")],
      ["stages", J.array([])],
      ["description", J.str("natural signal")],
    ]);
  }
  return state === null ? null : chainValue(state, rateSps);
}

const INDICATOR_NAMES: Readonly<Record<number, string>> = Object.fromEntries(
  Object.entries(INDICATOR_LEVELS).map(([name, level]) => [level, name]),
);

/**
 * What the device said it was set to, read after a run, as the Python
 * library reads it back: the status, which every device has, the lead-off
 * setting as last known, and then what the device claims: the chain in
 * force, the lamp's level, the model's cadence in seconds, and the name and
 * adjective it composes its name from. Pass null for what the device does
 * not claim.
 */
export function configAfter(
  status: Status,
  leadoffSetting: boolean | null,
  pipeline: PipelineState | null = null,
  indicator: number | null = null,
  modelInterval: ModelInterval | null = null,
  name: { readonly name: string; readonly adjective: string } | null = null,
): Value {
  const gain = GAIN_BY_CODE[status.gainCode];
  const rate = RATE_BY_CODE[status.rateCode] ?? null;
  const c = J.object([
    ["gain", gain === undefined ? J.NULL : J.int(gain)],
    ["rate_sps", rate === null ? J.NULL : J.int(rate)],
    ["mode", J.str(MODE_BY_CODE[status.mode] ?? `mode ${status.mode}`)],
    ["leadoff", leadoffSetting === null ? J.NULL : J.bool(leadoffSetting)],
    ["leadoff_status", J.int(status.loffStatp)],
  ]);
  if (pipeline !== null) J.set(c, "processing", chainValue(pipeline, rate));
  if (indicator !== null) J.set(c, "indicator", J.str(INDICATOR_NAMES[indicator] ?? `level ${indicator}`));
  if (modelInterval !== null) {
    J.set(c, "model_interval_s", J.int(modelInterval.intervalS));
    J.set(c, "model_interval_min_s", J.int(modelInterval.minimumS));
  }
  if (name !== null) {
    J.set(c, "name", J.str(name.name));
    J.set(c, "adjective", J.str(name.adjective));
    J.set(c, "composed_name", J.str(composeName(name.name, name.adjective)));
  }
  return c;
}
