/**
 * What a run uses that is not part of the package's surface: the clocks it
 * reads, the wait it takes and how a cue sounds, and the steps every
 * experiment is built from, on one device or several, with the recorder a
 * run keeps for each device and what says a device is being recorded.
 *
 * The steps are the Python library's (`_arm`, `_record`, `_safe_stop`,
 * `_config_after`, `_processing`, `_finish`, `_windows`): each takes one
 * device or a list of them, a device being a browser connection, and asks
 * each device what it can do before asking it to do it. What one step call
 * finds and does on each device
 * (which streams it started, each one's link, what each says it is set to
 * after) belongs to that call alone, so two steps on different devices at
 * the same moment never mix.
 *
 * The library tells what a step stamps and says which devices it belongs to
 * through context variables, which a browser does not have. Here every step
 * carries them explicitly in a `Step`: the devices it was handed, and the
 * recorder it records into, which is the run's when the step was handed the
 * run's own `emit`, or the context of a run going on, and the device's own
 * session recorder otherwise. A cue a step stamps is tied to the step's
 * devices, and a message it sends in a run names them.
 *
 * Internal. The package's own tests replace the clocks and the cues in
 * `runtime`, as the Python library's fixture replaces its experiments
 * module's clock, sleep and cues.
 */

import { isConnection } from "./boundary.ts";
import { general } from "./capture/pyfmt.ts";
import * as J from "./capture/pyjson.ts";
import type { Json, Value } from "./capture/pyjson.ts";
import { Take, collect, hostVersions, processing, saveWith, utc, type DeviceFacts, type Saved } from "./capture/record.ts";
import { GAIN_BY_CODE, Invalid, type EmbeddingWindow } from "./protocol.ts";
import { LINK_DOWN, refusal } from "./refusal.ts";
import type { Emit, Message, Params, RunCtx, Spec } from "./experiments.ts";
import { runtime } from "./runtime.ts";
import { SessionError } from "./session.ts";
import { embeddingsFor, own, tap } from "./taps.ts";
import type { Connection } from "./web-bluetooth.ts";

export { HOST_TIME, runtime, type RunTime } from "./runtime.ts";

/** A device as a step takes one: a browser connection to an IntoMind device. */
export type Unit = Connection;

// --- a device, as a step reads it -------------------------------------------------

/** Whether a step was handed a list of devices rather than one. */
export function several(dev: Unit | readonly Unit[]): dev is readonly Unit[] {
  return Array.isArray(dev);
}

/** The devices a step was handed, as a new list. */
export function listed(dev: Unit | readonly Unit[]): Unit[] {
  return several(dev) ? [...dev] : [dev as Unit];
}

/** Whether a device's link is up, by its own account. */
export function alive(link: Unit): boolean {
  return link.connected;
}

/** Whether a device ended its stream itself because USB power appeared, as the library's `Device.ended_by_usb` says. */
export function endedByUsb(link: Unit): boolean {
  return link.session.endedByUsb === true;
}

/** Whether a device can still send what a step waits for: its link is up, and its stream was not ended by USB power. */
export function flowing(link: Unit): boolean {
  return link.connected && !endedByUsb(link);
}

/** Whether a device is streaming, by its own account. */
export function streamingOf(link: Unit): boolean {
  return link.session.streaming;
}

/** A device as a message names it: its name, and its id when it has one, since two devices may share a name. */
export function named(dev: unknown): string {
  if (typeof dev === "string") return dev;
  const d = dev as { device?: { name?: string; id?: string }; info?: { deviceId?: string } } | null | undefined;
  const name = d?.device?.name || d?.device?.id || "a device";
  const id = d?.info?.deviceId;
  return id !== undefined ? `${name} (${id})` : name;
}

/** A device's id, as a message and a recording's name say it: an IntoMind device's own. */
export function idOfUnit(dev: Unit): string {
  return dev.session.requireInfo().deviceId;
}

/** Each device's id, as a message names it. */
export function ids(devs: readonly Unit[]): (string | null)[] {
  return devs.map((d) => d.info?.deviceId ?? null);
}

/** Who carries the samples, as every recording names it: the connection's own class's `ADAPTER`, as the library's `signal_chain` reads `dev.ADAPTER`. */
function adapterOf(link: Unit): Json {
  return (link.constructor as { readonly ADAPTER?: Json }).ADAPTER ?? null;
}

/** Now, in UTC, as ISO 8601 with milliseconds and a Z. */
export function utcMs(): string {
  return new Date(Math.floor(runtime.time.time() * 1000)).toISOString();
}

/** The manifest as a message carries it, a float that is not a number written as null. */
export function jsonable(meta: Value): Value {
  if (meta.kind !== "object") return meta;
  return J.object(meta.entries.map(([k, v]) => [k, v.kind === "float" && Number.isNaN(v.value) ? J.NULL : v]));
}

/** Start a fresh epoch of a stream's sample index, as `Session.resetEpoch` says, refused before anything is written on a link that is down, as the library's `Device.reset_epoch`. */
async function resetEpoch(link: Unit): Promise<void> {
  if (!link.connected) throw new SessionError(LINK_DOWN);
  const answer = await link.request(link.session.resetEpoch());
  if (!answer.ok) throw new SessionError(refusal(answer.opcode, answer.status));
}

/**
 * The gain a recording's name carries for a setup that gives `gain`, as the
 * library's `_gain_named` names it: `na` where it gives none or gives null,
 * and otherwise the gain as Python prints it. A whole number of more digits
 * than Python turns into text, 4,300, and text written as one, are refused
 * in words naming the gain by how many digits it has, and the device `dev`
 * the setup is for where one is given, since a recording's name carries its
 * gain.
 */
export function gainNamed(gain: Value | undefined, dev?: Unit): string {
  if (gain === undefined || gain.kind === "null") return "na";
  const long = J.tooLong(gain);
  if (long !== null) {
    const whose = dev === undefined ? "the setup" : `the setup for ${named(dev)}`;
    throw new Invalid(`ValueError: ${whose} gives a gain of ${long}, too long for a recording's name, which carries the gain`);
  }
  return J.pyStr(gain);
}

/**
 * One device's recording in a run of several: the run's name with that
 * device's own gain where the name says the gain, then the device's id, as
 * the library's `_tagged` names it, with any character a recording's name
 * cannot hold, written as an underscore. A gain no name can hold is refused,
 * as `gainNamed` refuses it.
 */
export function tagged(lbl: string, dev: Unit, cfg: Value): string {
  const m = /^(.*)_g[^_]*_(\d{8}-\d{6})$/.exec(lbl);
  const named = m ? `${m[1]}_g${gainNamed(J.get(cfg, "gain"), dev)}_${m[2]}` : lbl;
  const id = idOfUnit(dev).replace(/[^A-Za-z0-9._-]+/gu, "_").replace(/^[._-]+/, "").replace(/[._-]+$/, "");
  return `${named}_${Array.from(id).slice(0, 64).join("")}`;
}

// --- events, by the step that stamped them ----------------------------------------

/** The devices of the step each event was stamped in, kept beside the events, never in them, as the library's `RunCtx._ties`. */
const TIES = new WeakMap<object, readonly Unit[]>();

/** The events of `events` that belong in `dev`'s recording: every one stamped outside any step, and those a step that was handed `dev` stamped. */
export function eventsFor(events: readonly Value[], dev: Unit): Value[] {
  return events.filter((e) => {
    const t = TIES.get(e);
    return t === undefined || t.includes(dev);
  });
}

// --- what says a device is being recorded -----------------------------------------

/** A recorder a device can be held for: a run's recorder for one device, or the device's own session recorder. */
interface Holdable {
  /** Whether it is collecting samples now. */
  readonly armed: boolean;
}

/**
 * A device being recorded, as the library's `_Held`: the recorder its
 * samples go to, the step that holds it, whether that step is still arming
 * it, and whether its recording is still to be saved.
 */
class Held {
  readonly device: Unit;
  readonly recorder: Holdable;
  readonly owner: object | null;
  arming = true;
  unsaved = false;

  constructor(device: Unit, recorder: Holdable, owner: object | null) {
    this.device = device;
    this.recorder = recorder;
    this.owner = owner;
  }

  /** Whether the device is still being recorded. */
  live(): boolean {
    return this.arming || this.unsaved || this.recorder.armed;
  }
}

/**
 * The devices being recorded now. A device is being recorded from the
 * moment a step starts arming it until its recording is saved, or until the
 * step ends without saving it, since a second recording in between would
 * restart the count the first counts by.
 */
const RECORDING = new Map<Unit, Held>();

/** What holds `dev` now, or null. */
function holder(dev: Unit): Held | null {
  const held = RECORDING.get(dev);
  return held !== undefined && held.live() ? held : null;
}

/** Forget that `dev` was being recorded, once it no longer is. */
function letGo(dev: Unit): void {
  const held = RECORDING.get(dev);
  if (held !== undefined && !held.live()) RECORDING.delete(dev);
}

/** Mark each of `devs` as being recorded into `rec`, and give back the marks. */
function hold(devs: readonly Unit[], rec: StepRecorder, owner: object | null): Held[] {
  for (const [d, held] of [...RECORDING]) if (!held.live()) RECORDING.delete(d);
  return devs.map((d) => {
    const held = new Held(d, rec.holdable(d), owner);
    RECORDING.set(d, held);
    return held;
  });
}

/** `dev`'s recording into `recorder` is saved, or given up: once that recorder is disarmed, the device is no longer being recorded. */
function savedInto(dev: Unit, recorder: Holdable): void {
  const held = RECORDING.get(dev);
  if (held !== undefined && held.recorder === recorder && !held.arming) {
    held.unsaved = false;
    letGo(dev);
  }
}

/** A step is done with `devs`, saved or not. */
function release(devs: readonly Unit[], rec: StepRecorder): void {
  for (const d of devs) savedInto(d, rec.holdable(d));
}

/**
 * Refuse, in words that name it, a device another step or another run is
 * recording now, or one handed to a step twice, since each recording
 * restarts the device's count. A device `owner` holds is its own.
 */
function refuse(devs: readonly Unit[], owner: object | null): void {
  devs.forEach((d, i) => {
    if (devs.slice(0, i).includes(d)) {
      throw new SessionError(
        `RuntimeError: ${named(d)} is handed to this step twice, and a second recording of a device that can restart its sample count restarts that count in the middle of the first`,
      );
    }
    const held = holder(d);
    if (held !== null && (owner === null || held.owner !== owner)) {
      throw new SessionError(
        `RuntimeError: ${named(d)} is already being recorded, and a second recording of a device that can restart its sample count restarts that count in the middle of the first`,
      );
    }
  });
}

/** Whether anything still says `dev` is being recorded, a mark a step forgot to take off included. Internal, for the package's own tests. */
export function marked(dev: Unit): boolean {
  return RECORDING.has(dev);
}

// --- the recorders ----------------------------------------------------------------

/** Where a step records: a run's recorder, or a device's own session recorder. */
interface StepRecorder {
  /** Whether this is a run's recorder, so a step's messages name their devices. */
  readonly inRun: boolean;
  /** What a hold of `dev` names as its recorder. */
  holdable(dev: Unit): Holdable;
  /** Arm the recorder of each of `devs`. */
  arm(devs: readonly Unit[]): void;
  /** Disarm the recorder of each of `devs`, and nothing else. */
  disarm(devs: readonly Unit[]): void;
  /** Save each device's recording under `label`, each with its own fields, events, end and settings read after, and give back what each saved. */
  store(
    devs: readonly Unit[],
    label: string,
    extras: readonly Value[],
    events: readonly Value[][],
    aborted: readonly (string | null)[],
    configs: readonly (Value | null)[],
  ): Saved[];
}

/**
 * What a recorder keeps for one device: what it collected since it was
 * armed, whether it is armed, the device's count of breaks when it was, and
 * how to take it off the device.
 */
interface Each {
  take: Take;
  armed: boolean;
  gapsAtArm: number;
  unwire: (() => void) | null;
}

/** A recorder's place for one device, empty. */
function eachNew(): Each {
  return { take: new Take(), armed: false, gapsAtArm: 0, unwire: null };
}

/**
 * `e` armed afresh on `d`: a new take, and the device's count of breaks now,
 * as the library's `Recorder.arm`.
 */
function armEach(e: Each, d: Unit): void {
  e.take = new Take();
  e.gapsAtArm = d.session.gapsAnnounced;
}

/** Whether each device's own session recorder is armed, as far as the package's steps armed it. */
const SOLO = new WeakMap<Unit, Each>();

function soloOf(link: Unit): Each {
  let s = SOLO.get(link);
  if (s === undefined) {
    s = eachNew();
    SOLO.set(link, s);
  }
  return s;
}

/**
 * One device's own recorder, as a step outside a run records into it: the
 * counterpart of the Recorder a program hands the library's steps, fed by
 * that device. For a browser connection it is the session's own recorder.
 * It holds one device's samples.
 */
const OWN_RECORDER: StepRecorder = {
  inRun: false,
  holdable: soloOf,
  arm(devs) {
    const d = devs[0];
    if (d === undefined) return;
    const e = soloOf(d);
    d.session.armRecorder();
    e.armed = true;
  },
  disarm(devs) {
    for (const d of devs) {
      const e = soloOf(d);
      d.session.disarmRecorder();
      e.armed = false;
      letGo(d);
    }
  },
  store(devs, label, extras, events, aborted, configs) {
    const d = devs[0]!;
    const e = soloOf(d);
    const run = {
      extra: extras[0]!,
      events: J.array(events[0]!),
      configAfter: configs[0] ?? null,
      aborted: aborted[0] ?? null,
      finishedUnix: runtime.time.time(),
    };
    const saved = d.session.saveRecording(label, { adapter: adapterOf(d) }, run);
    savedInto(d, e);
    return [saved];
  },
};

/**
 * What a device said about itself when its recording is saved, as
 * `Session.saveRecording` reads it.
 */
function facts(link: Unit, gapsAtArm: number): DeviceFacts {
  const s = link.session;
  const info = s.requireInfo();
  return {
    deviceId: info.deviceId,
    firmware: info.firmware.join("."),
    protocol: `${info.protocol[0]}.${info.protocol[1]}`,
    tickHz: info.tickHz,
    vrefUv: info.vrefUv,
    adcBits: info.adcBits,
    channels: info.channels,
    clock: s.clock?.quality() ?? null,
    gapsInRun: Math.max(0, s.gapsAnnounced - gapsAtArm),
  };
}

/** One recording a run saved: the name its step asked for, the name it was saved under, the device, and what it saved. */
interface Save {
  readonly asked: string;
  readonly name: string;
  readonly device: Unit;
  readonly saved: Saved;
}

/**
 * The recorder a run keeps, as the library's `_RunRecorder`: one recording
 * for each device a step hands it, made when first needed, put on the
 * device by the package's own path when the step arms it and taken off when
 * the step is done with it, and it arms, disarms and saves only the
 * devices a step names. Each device's recording is the files a recording of
 * that device alone saves, under its own name, with that device's
 * settings, `run`, and `together`: the names of the other recordings its
 * step saved at the same moment.
 */
class RunRecorder implements StepRecorder {
  readonly inRun = true;
  /** The run's devices, in the order given. */
  readonly devices: readonly Unit[];
  /** Each device's setup, as the run was given it. */
  readonly cfgs: Map<Unit, Value>;
  /** Which run this is, the same in every recording it makes. */
  readonly run: Value;
  /** What the run's own context says the instrument was: a step writes it into every recording, and each device's own replaces it there. */
  readonly lead: Value;
  /** Every recording saved, in order. */
  readonly saves: Save[] = [];
  /** The names whose saved message has been said. */
  readonly said = new Set<string>();
  /** The names a run of the same chain has saved a recording under. */
  readonly #taken: Set<string>;
  readonly #each = new Map<Unit, Each>();

  constructor(devices: readonly Unit[], cfgs: readonly Value[], ctx: RunCtx, run: Value, taken: Set<string>) {
    this.devices = [...devices];
    this.cfgs = new Map(devices.map((d, i) => [d, J.copyObject(cfgs[i]!)]));
    this.run = run;
    this.lead = J.copyObject(ctx.cfg);
    this.#taken = taken;
  }

  #eachOf(dev: Unit): Each {
    let e = this.#each.get(dev);
    if (e === undefined) {
      e = eachNew();
      this.#each.set(dev, e);
    }
    return e;
  }

  holdable(dev: Unit): Holdable {
    return this.#eachOf(dev);
  }

  arm(devs: readonly Unit[]): void {
    for (const d of devs) {
      const e = this.#eachOf(d);
      if (e.unwire === null) {
        e.unwire = tap(d, "batch", (b) => {
          if (e.armed) collect(e.take, b, d.session.requireInfo().tickHz, d.session.clock);
        });
      }
      armEach(e, d);
      e.armed = true;
    }
  }

  disarm(devs: readonly Unit[]): void {
    for (const d of devs) {
      const e = this.#each.get(d);
      if (e === undefined) continue;
      e.armed = false;
      if (e.unwire !== null) {
        e.unwire();
        e.unwire = null;
      }
      letGo(d);
    }
  }

  store(
    devs: readonly Unit[],
    label: string,
    extras: readonly Value[],
    events: readonly Value[][],
    aborted: readonly (string | null)[],
    configs: readonly (Value | null)[],
  ): Saved[] {
    const names = this.#names(devs, label);
    return devs.map((d, i) => {
      const extra = this.#own(d, extras[i]!);
      J.set(extra, "together", J.array(names.filter((_, j) => j !== i).map((n) => J.str(n))));
      const e = this.#eachOf(d);
      const saved = saveWith(
        names[i]!,
        e.take,
        facts(d, e.gapsAtArm),
        { adapter: adapterOf(d) },
        {
          extra,
          events: J.array(events[i]!),
          configAfter: configs[i] ?? null,
          aborted: aborted[i] ?? null,
          finishedUnix: runtime.time.time(),
        },
        hostVersions(),
        { run: this.run, endedByUsb: endedByUsb(d), statedGain: own(d).gain },
      );
      this.#taken.add(names[i]!);
      this.saves.push({ asked: label, name: names[i]!, device: d, saved });
      savedInto(d, e);
      return saved;
    });
  }

  /** What a step wrote, with the device's own settings in place of what the run's context said. */
  #own(dev: Unit, given: Value): Value {
    const extra = J.copyObject(given);
    const cfg = this.cfgs.get(dev) ?? J.object([]);
    if (this.lead.kind === "object") for (const [k] of this.lead.entries) if (J.get(cfg, k) === undefined) J.pop(extra, k);
    if (cfg.kind === "object") for (const [k, v] of cfg.entries) J.set(extra, k, v);
    return extra;
  }

  /**
   * What each recording is called. A run on one device saves its recording
   * under the name its experiment asked for. Otherwise each name ends in
   * its device's id, and says its device's own gain where the name gives a
   * gain. A name already taken
   * has the first number from 2 that makes it new added.
   */
  #names(devs: readonly Unit[], label: string): string[] {
    const alone = this.devices.length === 1 && devs.length === 1 && devs[0] === this.devices[0];
    const out: string[] = [];
    for (const d of devs) {
      const name = alone ? label : tagged(label, d, this.cfgs.get(d) ?? J.object([]));
      let free = name;
      for (let k = 2; out.includes(free) || this.#taken.has(free); k++) free = `${name}_${k}`;
      out.push(free);
    }
    return out;
  }

  /** Whether a step is recording `dev` into this recorder now. */
  #stepping(dev: Unit): boolean {
    const held = holder(dev);
    return held !== null && held.recorder === this.#each.get(dev);
  }

  /**
   * The names of every recording saved: those the experiment gave back
   * first, in its order, a name it asked a step for standing for every
   * recording saved under it, then any it did not give back.
   */
  given(got: unknown): string[] {
    const asked: unknown[] =
      typeof got === "string"
        ? [got]
        : got === null || got === undefined
          ? []
          : typeof (got as Iterable<unknown>)[Symbol.iterator] === "function"
            ? [...(got as Iterable<unknown>)]
            : [];
    const out: string[] = [];
    for (const n of asked) {
      let found = this.saves.filter((s) => s.name === n).map((s) => s.name);
      if (found.length === 0) found = this.saves.filter((s) => s.asked === n).map((s) => s.name);
      if (found.length === 0) found = [n as string];
      for (const m of found) if (!out.includes(m)) out.push(m);
    }
    for (const s of this.saves) if (!out.includes(s.name)) out.push(s.name);
    return out;
  }

  /**
   * The run is over: every device is disarmed and taken off, and let go if
   * its recording was never saved, except a device a step is still
   * recording, which that step lets go when it ends.
   */
  close(): void {
    const devs = [...this.#each.keys()].filter((d) => !this.#stepping(d));
    this.disarm(devs);
    release(devs, this);
  }
}

/** The run each run's own `emit` belongs to, and the run each context is running now. */
const RUN_OF_EMIT = new WeakMap<Emit, RunRecorder>();
const RUN_OF_CTX = new WeakMap<RunCtx, RunRecorder>();

// --- a step -----------------------------------------------------------------------

/** One call of a step: the devices it was handed, where it records, where it says what it does, and the run's context. */
export interface Step {
  /** The devices the step was handed, a device whose link is down among them. */
  readonly devices: readonly Unit[];
  /** Where it records. */
  readonly rec: StepRecorder;
  /** Where its messages go. */
  readonly emit: Emit;
  /** The context it stamps its events in. */
  readonly ctx: RunCtx;
}

/**
 * A step of `dev`, one device or a list: in a run when it was handed the
 * run's own `emit`, or the context of a run going on, and on each device's
 * own session recorder otherwise.
 */
export function step(dev: Unit | readonly Unit[], emit: Emit, ctx: RunCtx): Step {
  return { devices: listed(dev), rec: RUN_OF_EMIT.get(emit) ?? RUN_OF_CTX.get(ctx) ?? OWN_RECORDER, emit, ctx };
}

/** Say `m` as the step says it: in a run, its `devices` names the step's devices, or `about` when it concerns fewer. Outside a run it is said as it is. */
export async function tell(s: Step, m: Message, about?: readonly Unit[]): Promise<void> {
  await s.emit(s.rec.inRun ? { ...m, devices: ids(about ?? s.devices) } : m);
}

/** Stamp a cue for `cond`, the step's own, so it belongs to the step's devices. */
export function cue(s: Step, cond: string, kw: Json = {}): Value {
  const e = s.ctx.cue(cond, kw);
  TIES.set(e, s.devices);
  return e;
}

/**
 * The devices of a step that it records, refused or passed over before
 * anything is sent to any of them, as the library's `_ready`.
 *
 * A device another step or another run is recording now, or one handed to
 * the step twice, is refused in words that name it. A device whose link is
 * down is passed over, and the step says it was not recorded and why,
 * unless every device's link is down: then all of them are given back, and
 * the step fails at the first thing it asks of a device, as a step on one
 * device whose link is down fails. A device's own session recorder holds
 * one device's samples, so several devices are recorded only in a run. A
 * device `owner` holds already is its own.
 */
export async function ready(s: Step, dev: Unit | readonly Unit[], owner: object | null = null): Promise<Unit[]> {
  const devs = listed(dev);
  if (devs.length > 1 && !s.rec.inRun) {
    throw new Invalid(
      "ValueError: a session's recorder holds one device's samples: several devices are recorded by DaisyChain.run, which keeps a recorder for each",
    );
  }
  refuse(devs, owner);
  const live = devs.filter(alive);
  if (live.length === 0) return devs;
  for (const d of devs) {
    if (!live.includes(d)) await tell(s, { type: "note", text: `${named(d)} was not recorded: its link is down` }, [d]);
  }
  return live;
}

/** The setup each of `devs` is recorded with, one a device: its own, in a run, and otherwise the setup of the step's context. */
function setupsOf(s: Step, devs: readonly Unit[]): Value[] {
  const rec = s.rec;
  if (rec instanceof RunRecorder) return devs.map((d) => rec.cfgs.get(d) ?? J.object([]));
  return devs.map(() => s.ctx.cfg);
}

/** A setup's gain as Python's `float` reads it, or null where it is not a number. */
function gainNumber(v: Value): number | null {
  if (v.kind === "int" || v.kind === "float") return J.asF64(v);
  if (v.kind === "bool") return v.value ? 1 : 0;
  if (v.kind === "str") {
    const t = J.trim(v.value).replace(/_/g, "");
    if (/^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/i.test(t)) return Number(t);
    if (/^[+-]?(inf|infinity)$/i.test(t)) return t.startsWith("-") ? -Infinity : Infinity;
    if (/^[+-]?nan$/i.test(t)) return NaN;
  }
  return null;
}

/**
 * The gain in force on each IntoMind device of `devs` whose setup, one of
 * `setups` a device, gives a gain, as the device's Status states it, or
 * null for a gain code the contract does not define, as the library's
 * `_gains_in_force`. Reading the Status changes nothing on the device. A
 * device whose link is down is not read and is left out of what comes back,
 * so no gain its setup gives is refused for it.
 */
export async function gainsInForce(devs: readonly Unit[], setups: readonly Value[]): Promise<Map<Unit, number | null>> {
  const out = new Map<Unit, number | null>();
  for (let i = 0; i < devs.length; i++) {
    const d = devs[i]!;
    const want = J.get(setups[i] ?? J.object([]), "gain");
    if (want === undefined || want.kind === "null" || !alive(d)) continue;
    const status = await d.readStatus();
    const gain = GAIN_BY_CODE[status.gainCode] ?? null;
    if (gain !== null) own(d).gain = gain;
    out.set(d, gain);
  }
  return out;
}

/** The first firmware of the IntoMind device that samples at gain 1 while lead-off detection is on, whatever gain was set. */
const UNITY_WHILE_DETECTING: readonly number[] = [1, 4, 6];

/** Whether firmware `a` is `b` or later, compared as Python compares tuples. */
function atLeast(a: readonly number[], b: readonly number[]): boolean {
  for (let i = 0; i < Math.min(a.length, b.length); i++) if (a[i] !== b[i]) return a[i]! > b[i]!;
  return a.length >= b.length;
}

/**
 * Refuse, in words naming both gains, a setup that gives the IntoMind
 * device `dev` the gain `want` while its gain in force is `now`, since its
 * recording carries the gain its packets carry, as the library's
 * `_held_to_gain`. Nothing is refused where the gain in force was not read.
 * A whole number too large for a float is one no gain in force equals, and
 * is named by how many digits it has, never printed, and so is text written
 * as a whole number of more digits than Python turns into text, 4,300.
 */
function heldToGain(dev: Connection, want: Value, now: number | null): void {
  if (now === null) return;
  const number = gainNumber(want);
  if (number === null) {
    throw new Invalid(`ValueError: the setup for ${named(dev)} gives its gain as ${J.jsRepr(want)}, and a gain is a number`);
  }
  if (number === now) return;
  const firmware = dev.info?.firmware;
  const why =
    now === 1 && firmware !== undefined && atLeast(firmware, UNITY_WHILE_DETECTING)
      ? " From firmware 1.4.6 the device samples at gain 1 while lead-off detection is on, whatever gain was set."
      : "";
  // A whole number past every float, which Python's `float` refuses as too
  // large, is named by its digits rather than printed.
  const shown = want.kind === "int" && !Number.isFinite(number) ? J.digitsSaid(want.value) : (J.tooLong(want) ?? J.pyStr(want));
  const inForce = general(now, 6);
  throw new Invalid(
    `ValueError: the setup for ${named(dev)} gives a gain of ${shown}, and the gain in force on it is ${inForce}, so the run is refused: its recording would carry a gain of ${inForce} beside a setup that gives ${shown}.${why}`,
  );
}

/**
 * An IntoMind device is never asked to change its gain: its recording
 * carries the gain its packets carry, and a setup that gives it a gain other
 * than the gain in force, one of `inForce` as `gainsInForce` reads it, is
 * refused in words naming both gains.
 */
export function gainsAsked(devs: readonly Unit[], setups: readonly Value[], inForce: ReadonlyMap<Unit, number | null> = new Map()): void {
  devs.forEach((d, i) => {
    const want = J.get(setups[i] ?? J.object([]), "gain");
    if (want === undefined || want.kind === "null") return;
    heldToGain(d, want, inForce.get(d) ?? null);
  });
}

/** What `arm` says of each device: whether it started the stream, or null for a device not recorded because its link was down. */
export type Armed = boolean | (boolean | null)[];

/**
 * Arm the recorder where the device's sample index starts again, and say
 * whether this started the stream, as the library's `_arm` does. Every
 * recording begins here, so that none can arm in any other order. A stream already
 * running has its count reset first and the recorder arms once packets
 * numbered before the reset have drained. One that is not is armed first and
 * then started, which resets the count, so its first samples are not missed.
 *
 * A list of devices takes the same steps, each device its own, and what
 * comes back is then a list, one entry a device in the order given. A
 * device another step or another run is recording is refused before
 * anything is sent to any device. A failure while arming leaves no device
 * armed, held or started by it. A device armed here counts as being
 * recorded until its recording is saved, or until its step ends without
 * saving it.
 */
export async function arm(s: Step, dev: Unit | readonly Unit[], spp = 25, owner: object | null = null): Promise<Armed> {
  const devs = await ready(s, dev, owner);
  const setups = setupsOf(s, devs);
  gainsAsked(devs, setups, await gainsInForce(devs, setups));
  // Asked again with no wait between the asking and the holding, since
  // saying which devices were passed over lets another step go first.
  refuse(devs, owner);
  const holds = hold(devs, s.rec, owner);
  const tried: Unit[] = [];
  let streaming: boolean[] = [];
  try {
    streaming = devs.map(streamingOf);
    // Each device is asked what it can do before it is asked to do it.
    for (let i = 0; i < devs.length; i++) if (streaming[i] && devs[i]!.can("reset_count")) await resetEpoch(devs[i]!);
    if (streaming.some((x) => x)) await runtime.time.sleep(0.05);
    s.rec.arm(s.rec.inRun ? devs : devs.slice(0, 1));
    for (let i = 0; i < devs.length; i++) {
      if (streaming[i]) continue;
      const m = devs[i]!;
      tried.push(m);
      if (m.can("packet_size")) await m.start({ samplesPerPacket: spp });
      else await m.start();
    }
  } catch (e) {
    // The streams this tried to start are stopped, whether or not the start
    // answered, and nothing it armed or put on a device is left.
    for (const held of holds) held.arming = false;
    s.rec.disarm(devs);
    for (const m of tried) {
      try {
        await m.stop();
      } catch {
        // The device that refused to start is the error being reported.
      }
    }
    throw e;
  }
  for (const held of holds) {
    held.arming = false;
    held.unsaved = true;
  }
  const started = new Map(devs.map((m, i) => [m, !streaming[i]]));
  if (!several(dev)) return started.get(dev as Unit) ?? false;
  return dev.map((d) => started.get(d) ?? null);
}

/** The devices a step recorded, each with whether the step started its stream, from what `arm` gave back. */
export function recorded(dev: Unit | readonly Unit[], armed: Armed): [Unit, boolean][] {
  if (several(dev)) {
    const each = armed as (boolean | null)[];
    const out: [Unit, boolean][] = [];
    dev.forEach((d, i) => {
      const s = each[i];
      if (s !== null && s !== undefined) out.push([d, s]);
    });
    return out;
  }
  return [[dev as Unit, armed as boolean]];
}

/**
 * Stop streaming, and say whether the link was still alive. A recording is
 * finalized only if it was. A device that ended its stream itself because
 * USB power appeared is no longer streaming, and answers the stop with ok,
 * as it answers any stop while it is not streaming, so its link counts as
 * alive.
 */
export async function safeStop(link: Unit): Promise<boolean> {
  try {
    await link.stop();
  } catch {
    return false;
  }
  return link.connected;
}

/** What the device says it is set to after a run, or nothing from a link that dropped. */
async function configAfter(link: Unit, linkOk: boolean): Promise<Value | null> {
  if (!linkOk) return null;
  try {
    return await link.refreshConfig();
  } catch {
    return null;
  }
}

/**
 * What the device did to the signal, in its own words: the chain in force
 * on a device that claims one, the natural signal on one that does not.
 * Only a link failure leaves it unknown.
 */
async function processingOf(link: Unit): Promise<Value | null> {
  if (!link.session.can("pipeline")) return processing(false, null, null);
  let state;
  try {
    state = await link.pipeline();
  } catch {
    return null;
  }
  const rate = link.config === null ? null : J.asF64(J.field(link.config, "rate_sps"));
  return processing(true, state, rate);
}

/** One device's end of a step: the device, what it saved, and whether its link was still up. */
export interface Done {
  /** The device. */
  readonly device: Unit;
  /** Its recording, as saved. */
  readonly saved: Saved;
  /** Whether its link was still up when the step stopped it. */
  readonly ok: boolean;
}

/**
 * The end of a step, as the library's `_finish`: for each device it
 * recorded, its stream stopped only where the step started it, a wait for
 * what is still in flight, its recording disarmed, what it says it is set
 * to read by its own link, and its recording saved and said, each with its
 * own link, settings, the events that belong to it and, with `processing`,
 * what the device did to its signal. `extra` is what each recording's
 * manifest adds, or a function that gives it once the settings are read.
 * `then` runs once every recording is disarmed.
 */
export async function finish(
  s: Step,
  dev: Unit | readonly Unit[],
  armed: Armed,
  label: string,
  extra: Value | (() => Value),
  options: { readonly processing?: boolean; readonly then?: () => void; readonly say?: boolean } = {},
): Promise<Done[]> {
  const pairs = recorded(dev, armed);
  const devs = pairs.map(([d]) => d);
  const links: boolean[] = [];
  let saves: Saved[];
  try {
    try {
      // Only stop what this started. Otherwise the trace goes dark the moment
      // a recording ends, and the device is still streaming.
      for (const [d, started] of pairs) links.push(started ? await safeStop(d) : alive(d));
      await runtime.time.sleep(0.3);
    } finally {
      s.rec.disarm(devs);
    }
    options.then?.();
    const configs: (Value | null)[] = [];
    for (let i = 0; i < devs.length; i++) configs.push(await configAfter(devs[i]!, links[i]!));
    let procs: (Value | null)[] | null = null;
    if (options.processing) {
      procs = [];
      for (const d of devs) procs.push(await processingOf(d));
    }
    const given = typeof extra === "function" ? extra() : extra;
    const extras = devs.map((_, i) => {
      const e = J.copyObject(given);
      // What signal this is: the chain the device ran on it, in the
      // device's own words, or the natural signal. Never a raw-or-filtered
      // flag: a flag says nothing about what a signal is.
      if (procs !== null) {
        const proc = procs[i]!;
        const known = proc !== null && J.truthy(proc);
        if (known) J.set(e, "processing", proc);
        J.set(e, "signal", known ? J.field(proc, "description") : J.str("unknown"));
      }
      return e;
    });
    saves = s.rec.store(
      devs,
      label,
      extras,
      devs.map((d) => eventsFor(s.ctx.events, d)),
      links.map((ok) => (ok ? null : "link_lost")),
      configs,
    );
  } finally {
    // Each device stays held until its recording is saved, so no other step
    // restarts its count in between, and is let go however this ends.
    release(devs, s.rec);
  }
  if (options.say ?? true) {
    for (let i = 0; i < devs.length; i++) {
      const one = saves[i]!;
      await tell(s, { type: "saved", label: one.label, meta: J.toPlain(jsonable(one.manifest)), recording: one }, [devs[i]!]);
    }
  }
  return devs.map((d, i) => ({ device: d, saved: saves[i]!, ok: links[i]! }));
}

/** The names a step saved its recordings under, in order. */
export function labels(done: readonly Done[]): string[] {
  return done.map((d) => d.saved.label);
}

/**
 * Record from the stream for `seconds`, starting it only if it was not
 * running, and stopping only what this started, as the library's `_record`.
 * A list of devices is recorded side by side, each as its own recording. A
 * recording that fails before its end is not saved: it is disarmed and its
 * devices let go, and the streams it started are stopped.
 */
export async function recordFor(
  dev: Unit | readonly Unit[],
  emit: Emit,
  ctx: RunCtx,
  label: string,
  seconds: number,
  given: Value,
): Promise<Done[]> {
  const s = step(dev, emit, ctx);
  const extra = J.copyObject(given);
  J.set(extra, "started_utc", J.str(utc(runtime.time.time())));
  const armed = await arm(s, dev);
  try {
    const t0 = runtime.time.time();
    while (runtime.time.time() - t0 < seconds) {
      if (ctx.stop.isSet()) break;
      await runtime.time.sleep(0.25);
      await tell(s, { type: "progress", frac: (runtime.time.time() - t0) / seconds });
    }
  } catch (e) {
    const pairs = recorded(dev, armed);
    const devs = pairs.map(([d]) => d);
    s.rec.disarm(devs);
    release(devs, s.rec);
    for (const [d, started] of pairs) if (started) await safeStop(d);
    throw e;
  }
  return finish(s, dev, armed, label, extra, { processing: true });
}

/**
 * Up to `n` whole windows of the encoder's output while each device
 * streams, gathered as `Connection.collectEmbeddings` gathers them, with
 * embeddings left as they were found and `onEmbedding` never touched, and
 * ended as soon as `ctx.stop` is set: checked every 50 ms, as the cued
 * trials check it. One list a device, as the library's `_windows` gives them.
 *
 * A list of devices each send their own, gathered side by side until every
 * device still connected has sent `n`. A device whose link has dropped, or
 * whose stream ended because USB power appeared, is not asked, and not
 * waited for. Each is asked within the cleanup that undoes it, so when one
 * fails, every device is let go: no consumer of its windows is left, and
 * its embeddings are put back as they were found. One device on its own whose
 * stream ended on USB power is not asked either, and not waited for, and
 * one alone whose link drops while its windows are waited for is not
 * waited for either: the wait ends at once with `ConnectionError`, as
 * asking the device would, checked every 50 ms as `ctx.stop` is, never by
 * waiting out the timeout.
 */
export async function windowsOf(dev: Unit | readonly Unit[], n: number, ctx: RunCtx): Promise<EmbeddingWindow[][]> {
  for (const m of listed(dev)) {
    if (!m.can("embeddings")) {
      throw new SessionError(`RuntimeError: ${named(m)} does not send its encoder's output, so no windows can be gathered from it`);
    }
    if (!handsWindows(m)) {
      throw new SessionError(
        `RuntimeError: ${named(m)} says it sends its encoder's output, and offers no way to hand its windows to the package, so no windows can be gathered from it`,
      );
    }
  }
  if (several(dev)) {
    const got: EmbeddingWindow[][] = dev.map(() => []);
    const undo: (() => Promise<void>)[] = [];
    try {
      for (let i = 0; i < dev.length; i++) {
        const m = dev[i]!;
        if (!flowing(m)) continue;
        undo.push(await windowsFrom(m, (w) => got[i]!.push(w)));
      }
      const t0 = runtime.time.time();
      const limit = 60 + 30 * n;
      while (!ctx.stop.isSet()) {
        const waiting = dev.map((m, i) => (got[i]!.length < n && flowing(m) ? i : -1)).filter((i) => i >= 0);
        if (waiting.length === 0) break;
        if (runtime.time.time() - t0 > limit) {
          const i = waiting[0]!;
          throw new SessionError(
            `TimeoutError: ${idOfUnit(dev[i]!)}: ${got[i]!.length} of ${n} windows arrived in ${limit.toFixed(0)} seconds`,
          );
        }
        await runtime.time.sleep(0.05);
      }
    } finally {
      for (const u of undo) await u();
    }
    return got.map((g) => g.slice(0, n));
  }
  const one = dev as Unit;
  const got: EmbeddingWindow[] = [];
  if (endedByUsb(one)) return [got];
  let undo: (() => Promise<void>) | null = null;
  try {
    undo = await windowsFrom(one, (w) => got.push(w));
    const t0 = runtime.time.time();
    const limit = 60 + 30 * n;
    while (got.length < n && !ctx.stop.isSet() && !endedByUsb(one)) {
      if (!alive(one)) throw new SessionError(`ConnectionError: the link dropped after ${got.length} of ${n} windows arrived`);
      if (runtime.time.time() - t0 > limit) {
        throw new SessionError(`TimeoutError: ${got.length} of ${n} windows arrived in ${limit.toFixed(0)} seconds`);
      }
      await runtime.time.sleep(0.05);
    }
  } finally {
    if (undo !== null) await undo();
  }
  return [got.slice(0, n)];
}

/** Whether `m` offers a way to hand its windows to the package: a browser connection's own embeddings. Anything else a program hands in that says it sends windows offers none. */
function handsWindows(m: unknown): boolean {
  return isConnection(m);
}

/** Hand every whole window `m` sends that carries the window embedding to `take`, by the package's own path, and give back how to stop. */
function windowsFrom(m: Unit, take: (w: EmbeddingWindow) => void): Promise<() => Promise<void>> {
  return embeddingsFor(m, "window", take);
}

/**
 * Hold `devs` for a step before anything is asked of them, as head training
 * holds its devices while it reads their models, so no other step starts
 * recording them meanwhile. The holds are `owner`'s, and `arm` with the same
 * owner takes them over. Give back how to let go whatever arming did not
 * take over.
 */
export function holdFor(s: Step, devs: readonly Unit[], owner: object): () => void {
  refuse(devs, null);
  const asking = hold(devs, s.rec, owner);
  return () => {
    for (const held of asking) {
      if (RECORDING.get(held.device) === held) {
        held.arming = false;
        letGo(held.device);
      }
    }
  };
}

// --- a run ------------------------------------------------------------------------

/**
 * `DaisyChain.run`, as the library's `_run_together`: one registered
 * experiment, whoever registered it, on the devices given, its function
 * handed the connection itself when there is one and a new array of them in
 * the order given when there are several. Every recording its steps make is
 * one recording a device in the run's recorder, marked as this run, and every
 * message it sends names the devices it concerns: those its step was handed,
 * one device's for what is said about its recording, and the run's for what
 * the run says. `taken` holds the names the chain's runs have saved, so a
 * name already taken has a number added.
 */
export async function runTogether(
  spec: Spec,
  devices: readonly Unit[],
  cfgs: readonly Value[],
  params: Params,
  emit: Emit,
  ctx: RunCtx,
  taken: Set<string>,
): Promise<string[]> {
  const rec = new RunRecorder(
    devices,
    cfgs,
    ctx,
    J.object([
      ["experiment", J.str(spec.name)],
      ["started_utc", J.str(utcMs())],
    ]),
    taken,
  );
  gainsAsked(devices, cfgs, await gainsInForce(devices, cfgs));
  const said: Emit = async (m) => {
    if (m === null || typeof m !== "object") {
      await emit(m);
      return;
    }
    let about: readonly Unit[] = rec.devices;
    if (m.type === "saved") {
      const label = m.label;
      const fresh = rec.saves.filter((s) => s.name === label && !rec.said.has(s.name));
      if (fresh.length > 0) {
        rec.said.add(fresh[0]!.name);
        about = [fresh[0]!.device];
      } else if (rec.saves.some((s) => s.asked === label || s.name === label)) {
        // An experiment that says it saved under the name it asked a step
        // for has each recording saved under that name and not said yet said,
        // under its own, so each recording is said once.
        for (const s of rec.saves) {
          if (s.asked !== label || rec.said.has(s.name)) continue;
          rec.said.add(s.name);
          await emit({ ...m, label: s.name, meta: J.toPlain(jsonable(s.saved.manifest)), recording: s.saved, devices: ids([s.device]) });
        }
        return;
      }
    }
    await emit("devices" in m ? m : { ...m, devices: ids(about) });
  };
  RUN_OF_EMIT.set(said, rec);
  RUN_OF_CTX.set(ctx, rec);
  const handed = devices.length === 1 ? devices[0]! : [...devices];
  let got: unknown;
  try {
    got = await spec.fn(handed, said, params, ctx);
  } finally {
    rec.close();
    if (RUN_OF_CTX.get(ctx) === rec) RUN_OF_CTX.delete(ctx);
  }
  return rec.given(got);
}
