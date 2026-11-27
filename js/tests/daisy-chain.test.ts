/**
 * The daisy chain held to the Python library's multi-device `Session`. The
 * files `record_to` writes, byte for byte, from the same packets. Every run
 * in `contract/together` replayed through browser connections to devices
 * made of nothing, on the same scripted clocks, compared step for step with
 * what the library did. And every program there, from the scan, run as the
 * library's program ran it, two runs at once and two groups at once
 * included, a device whose link drops in the middle of head training and
 * comes back, and one that refuses to turn embeddings off until it does
 * not. Compared are every clock exchange at a connection, every command
 * written to each device, every read, every wait and the packets each device
 * sent during it, every delivery of windows, every cue and every message, in
 * order, then each device's recording, its manifest and events byte for byte
 * and its arrays element for element, what each run gave back, the error a
 * run raised, in the library's words, and what each run left on each device.
 *
 * The differences allowed are the two the experiments are allowed: the
 * software a recording names, and the adapter, the library's own Bluetooth
 * device where the browser connection names itself.
 *
 * Where the library's program waits on its event loop, these wait on the
 * host's: a scripted wait ends with a turn of the event loop, and what the
 * library starts as a task beside another is started on the next turn, so
 * the steps that go at once take their turns in the order the library's do.
 */

import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import process from "node:process";
import test from "node:test";

import { sha256 } from "../src/digest.ts";
import { writeRecording } from "../src/capture/node.ts";
import * as npy from "../src/capture/npy.ts";
import * as J from "../src/capture/pyjson.ts";
import type { Json, Value } from "../src/capture/pyjson.ts";
import type { Saved } from "../src/capture/record.ts";
import { readNpz } from "../src/capture/recording.ts";
import { PACKAGE_VERSION } from "../src/capture/version.ts";
import { DaisyChain } from "../src/daisy-chain.ts";
import { REGISTRY, RunCtx, experiment, type Emit, type ExperimentFn, type Message, type Params } from "../src/experiments.ts";
import * as P from "../src/protocol.ts";
import { HOST_TIME, marked, runtime } from "../src/run.ts";
import { SessionError } from "../src/session.ts";
import { own } from "../src/taps.ts";
import { audio } from "../src/index.ts";
import { Connection, connect, type BluetoothDeviceLike } from "../src/web-bluetooth.ts";
import { Browser } from "./browser.ts";
import { inJavaScript, refusalsInJavaScript } from "./words.ts";

process.env.TZ = "UTC";

const fixture = J.loads(readFileSync(new URL("../../contract/together/together.json", import.meta.url), "utf8"));

function items(v: Value | undefined): Value[] {
  assert.ok(v !== undefined && v.kind === "array", "a list in the fixture");
  return v.items;
}

function text(v: Value): string {
  assert.equal(v.kind, "str");
  return (v as { value: string }).value;
}

function num(v: Value): number {
  const x = J.asF64(v);
  assert.ok(x !== null, "a number in the fixture");
  return x;
}

function hex(b: Uint8Array): string {
  let s = "";
  for (const x of b) s += x.toString(16).padStart(2, "0");
  return s;
}

function plain(v: unknown): string {
  return JSON.stringify(v, (_k, x) => (typeof x === "bigint" ? x.toString() : x));
}

function said(e: unknown): string {
  const err = e as Error;
  return err instanceof P.Invalid || err instanceof SessionError ? err.message : `${err.name}: ${err.message}`;
}

/** One turn of the host's event loop, as the library's program yields one of its own. */
function turn(): Promise<void> {
  return new Promise((resolve) => setImmediate(resolve));
}

/** Start `f` on the next turn of the event loop, as the library starts a task beside the one running. */
function later<T>(f: () => Promise<T>): Promise<T> {
  return new Promise((resolve, reject) => setImmediate(() => f().then(resolve, reject)));
}

class FakeCharacteristic extends EventTarget {
  readonly uuid: string;
  value?: DataView;
  onWrite?: (bytes: Uint8Array) => void;
  onRead?: () => void;

  constructor(uuid: string, value?: Uint8Array) {
    super();
    this.uuid = uuid;
    if (value !== undefined) this.value = new DataView(value.slice().buffer);
  }

  readValue(): Promise<DataView> {
    this.onRead?.();
    assert.ok(this.value !== undefined, `${this.uuid} has nothing to read`);
    return Promise.resolve(this.value);
  }

  writeValueWithResponse(value: BufferSource): Promise<void> {
    return this.#write(value);
  }

  writeValueWithoutResponse(value: BufferSource): Promise<void> {
    return this.#write(value);
  }

  notify(bytes: Uint8Array): void {
    this.value = new DataView(bytes.slice().buffer);
    this.dispatchEvent(new Event("characteristicvaluechanged"));
  }

  #write(value: BufferSource): Promise<void> {
    const bytes = ArrayBuffer.isView(value)
      ? new Uint8Array(value.buffer, value.byteOffset, value.byteLength).slice()
      : new Uint8Array(value).slice();
    this.onWrite?.(bytes);
    return Promise.resolve();
  }
}

/** A device made of nothing, as a fixture unit describes it, before anything connects to it. */
interface Fake {
  readonly device: BluetoothDeviceLike;
  readonly data: FakeCharacteristic;
  readonly embeddings: FakeCharacteristic;
  /** Takes its link down. */
  drop(): void;
  /** Brings its link back up, as a person's hand does between runs. */
  back(): void;
  /** The forms of embeddings it refuses to be set to, by their byte, until a person's hand clears them. */
  readonly refuseForms: Set<number>;
}

/** What a stand-in tells the script: each write as it is made, and whether it refused it, each write again once it is done, as the device handles it, before its answer, each write once its answer is given, and each read of its status. */
interface Hooks {
  readonly onWrite?: (bytes: Uint8Array, refused: boolean) => void;
  readonly onHandled?: (bytes: Uint8Array, refused: boolean) => void;
  readonly onAnswered?: (bytes: Uint8Array, refused: boolean) => void;
  readonly onRead?: () => void;
}

function fakeDevice(u: Value, hooks: Hooks = {}): Fake {
  const answers = new Map<number, Uint8Array>();
  const given = J.field(u, "answers");
  if (given.kind === "object") for (const [op, a] of given.entries) answers.set(Number(op), P.fromHex(text(a)));
  const refuse = new Set(items(J.get(u, "refuse") ?? J.array([])).map(num));
  const refuseForms = new Set(
    items(J.get(u, "refuse_forms") ?? J.array([])).map((f) => P.EMBEDDING_FORMS[text(f) as keyof typeof P.EMBEDDING_FORMS]),
  );
  const characteristics = new Map<string, FakeCharacteristic>([
    [P.DEVICE_INFO, new FakeCharacteristic(P.DEVICE_INFO, P.fromHex(text(J.field(u, "info"))))],
    [P.CONTROL, new FakeCharacteristic(P.CONTROL)],
    [P.CONTROL_RESPONSE, new FakeCharacteristic(P.CONTROL_RESPONSE)],
    [P.EEG_DATA, new FakeCharacteristic(P.EEG_DATA)],
    [P.STATUS, new FakeCharacteristic(P.STATUS, P.fromHex(text(J.field(u, "status"))))],
    [P.EMBEDDINGS, new FakeCharacteristic(P.EMBEDDINGS)],
  ]);
  const browser = new Browser(characteristics, { id: "fake", name: "IntoMind One" });
  const fake: Fake = {
    device: browser.device,
    data: characteristics.get(P.EEG_DATA)!,
    embeddings: characteristics.get(P.EMBEDDINGS)!,
    drop: () => browser.drop(),
    back: () => browser.back(),
    refuseForms,
  };
  characteristics.get(P.CONTROL)!.onWrite = (bytes) => {
    const op = bytes[0]!;
    const refused = refuse.has(op) || (op === P.OPCODES.set_embeddings && fake.refuseForms.has(bytes[1]!));
    hooks.onWrite?.(bytes, refused);
    // A stop is answered with ok whether or not the device is streaming, as
    // the firmware answers it.
    const answer = refused ? Uint8Array.of(op, 1) : (answers.get(op) ?? Uint8Array.of(op, 0));
    // The device acknowledges the write first, and handles what it carried
    // on a later turn of the event loop, as the library's fixture radio
    // does. Its answer reaches the connection on the turn after that, the
    // turn on which the library's exchange, waiting on a queue, takes it up.
    setImmediate(() => {
      hooks.onHandled?.(bytes, refused);
      setImmediate(() => {
        characteristics.get(P.CONTROL_RESPONSE)!.notify(answer);
        hooks.onAnswered?.(bytes, refused);
      });
    });
  };
  characteristics.get(P.STATUS)!.onRead = () => hooks.onRead?.();
  return fake;
}

/** A clock exchange as the library's connection took it, on the script's clocks. */
function exchange(link: Connection, before: readonly [number, number], after: readonly [number, number], ticks: string): void {
  const bytes = new Uint8Array(8);
  new DataView(bytes.buffer).setBigUint64(0, BigInt(ticks), true);
  link.session.onTimeSyncClocks(bytes, { monotonic: before[0], realtime: before[1] }, { monotonic: after[0], realtime: after[1] });
}

/** One device made of nothing, connected through the browser transport, with the clock exchange the unit says it took. */
async function fakeLink(u: Value, hooks: Hooks = {}): Promise<Fake & { link: Connection }> {
  const fake = fakeDevice(u, hooks);
  const link = await connect(fake.device);
  const sync = J.field(u, "sync");
  const pair = (k: string) => items(J.get(sync, k)).map(num) as [number, number];
  exchange(link, pair("before"), pair("after"), J.pyStr(J.field(sync, "ticks")));
  return Object.assign(fake, { link });
}

/** The library's manifest as a run through a connection writes it: its software and its adapter named as here, everything else as it is. */
function asMadeHere(python: string): string {
  const start = python.indexOf('  "host_versions": {');
  assert.ok(start >= 0, "a manifest names its software");
  const close = python.indexOf("\n  }", start);
  const ours =
    `  "host_versions": {\n    "intomind_js": "${PACKAGE_VERSION}",\n` +
    `    "platform": "${process.platform}-${process.arch}"\n  }`;
  const named = python.slice(0, start) + ours + python.slice(close + "\n  }".length);
  const theirs = '"adapter": {\n      "name": "intomind.ble",';
  const at = named.indexOf(theirs);
  assert.ok(at >= 0 && named.indexOf(theirs, at + 1) < 0, "the library's recording names its own adapter, once");
  return named.replace(theirs, `"adapter": {\n      "name": "${Connection.ADAPTER.name}",`);
}

function npzDigest(archive: Uint8Array): unknown[] {
  return readNpz(archive).map(([name, a]) => {
    let data = a.data;
    if (a.fortran && a.shape.length === 2) {
      const [rows, cols] = a.shape as [number, number];
      const keep = new Int32Array(rows * cols);
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) keep[r * cols + c] = c * rows + r;
      data = npy.take(a.data, keep);
    }
    return [name, npy.descr(a.data), a.shape, hex(sha256(npy.bytesOf(data))), a.fortran];
  });
}

/** What a run left on a device, as the library's fixture reads it of its own. */
function left(link: Connection): unknown {
  const o = own(link);
  return {
    streaming: link.session.streaming,
    held: marked(link),
    tapped: o.batch.length > 0 || o.window.length > 0,
    embeddings: o.form,
  };
}

/** A message as the fixture's trace keeps it: a saved message as its name, the devices it concerns and what a head training adds. */
function kept(m: Message): unknown {
  if (m.type !== "saved") return m;
  return {
    type: "saved",
    label: m.label,
    ...("head" in m ? { head: m.head, score: m.score, no_head: m.no_head } : {}),
    ...("devices" in m ? { devices: m.devices } : {}),
  };
}

/** Everything a run or a program left in its folder, compared with what the library's left in its own. */
function compareFolder(dir: string, want: Value, kept: readonly Saved[], problem: (what: string) => void): number {
  let recordings = 0;
  for (const rec of items(J.get(want, "recordings"))) {
    const label = text(J.field(rec, "label"));
    const read = (file: string): string | null => (existsSync(join(dir, file)) ? readFileSync(join(dir, file), "utf8") : null);
    const meta = read(`${label}.meta.json`);
    if (meta === null) {
      problem(`${label}: not saved here`);
      continue;
    }
    if (meta !== asMadeHere(text(J.field(rec, "meta_json")))) problem(`${label}: the manifest differs:\n${meta}\n${text(J.field(rec, "meta_json"))}`);
    const wantEvents = J.field(rec, "events_json");
    if (read(`${label}.events.json`) !== (wantEvents.kind === "null" ? null : text(wantEvents))) problem(`${label}: the events differ`);
    const wantTaught = J.field(rec, "training_json");
    if (read(`${label}.training.json`) !== (wantTaught.kind === "null" ? null : text(wantTaught))) problem(`${label}: the training data differs`);
    const sums = read(`${label}.sha256`);
    const names = sums === null ? null : sums.split("\n").filter((l) => l !== "").map((l) => l.split("  ")[1]);
    if (plain(names) !== plain(J.toPlain(J.field(rec, "sha256_names")))) problem(`${label}: checksums for ${plain(names)}`);
    const arrays = plain(npzDigest(new Uint8Array(readFileSync(join(dir, `${label}.npz`)))));
    if (arrays !== plain(J.toPlain(J.field(rec, "arrays")))) problem(`${label}: the arrays differ`);
    if (!kept.some((s) => s.label === label)) problem(`${label}: no saved message carried it`);
    recordings++;
  }
  const files = existsSync(dir)
    ? readdirSync(dir)
        .sort()
        .map((f) => [f, (statSync(join(dir, f)).mode & 0o222) === 0])
    : [];
  const wantFiles = J.field(want, "files");
  const theirFiles =
    wantFiles.kind === "object"
      ? wantFiles.entries.map(([f, v]) => [f, J.truthy(J.field(v, "read_only"))]).sort((a, b) => ((a[0] as string) < (b[0] as string) ? -1 : 1))
      : [];
  if (plain(files) !== plain(theirFiles)) problem(`files ${plain(files)}, the library's ${plain(theirFiles)}`);
  const log = existsSync(join(dir, "SESSIONS.jsonl")) ? readFileSync(join(dir, "SESSIONS.jsonl"), "utf8") : null;
  const wantLog = J.field(want, "sessions_jsonl");
  if (log !== (wantLog.kind === "null" ? null : text(wantLog))) problem("the session log differs");
  return recordings;
}

/** The trace here against the library's, step for step, and how many steps agreed. */
function compareTrace(trace: readonly unknown[], want: Value, problem: (what: string) => void): number {
  const theirs = items(J.get(want, "trace")).map((t) => J.toPlain(t) as unknown[]);
  const n = Math.max(trace.length, theirs.length);
  for (let i = 0; i < n; i++) {
    const mine = trace[i] === undefined ? "nothing" : plain(trace[i]);
    const them = theirs[i] === undefined ? "nothing" : plain(theirs[i]);
    if (mine !== them) {
      problem(`step ${i}: here ${mine}, the library ${them}`);
      return i;
    }
  }
  return n;
}

/**
 * The scripted world a run or a program runs in, replaying the library's
 * own: its waits, with the packets that arrived and what the person did, its
 * deliveries of windows, and its clock exchanges at a connection.
 */
class Script {
  readonly trace: unknown[][] = [];
  tracing = false;
  readonly clocks = { mono: 0, real: 0 };
  readonly ctxs = new Map<string, RunCtx>();
  ctx: RunCtx | null = null;
  readonly fakes: Fake[] = [];
  readonly #waits: Value[];
  readonly #deliveries: Value[];
  readonly #connects: Value[];
  #waited = 0;
  #connected = 0;
  readonly #delivered: number[];
  /**
   * The units whose link is lost after a rep's windows, which drops once the
   * device has answered the request that asked for them, as the library's
   * fixture radio drops it: nothing reaches the host on a link that is down.
   */
  readonly #dropAfterAnswer = new Set<number>();
  readonly #program: boolean;

  constructor(want: Value, units: number, program: boolean) {
    const theirs = items(J.get(want, "trace"));
    const kind = (t: Value) => text(items(t)[0]!);
    this.#waits = theirs.filter((t) => kind(t) === "sleep");
    this.#deliveries = theirs.filter((t) => kind(t) === "embeddings");
    this.#connects = theirs.filter((t) => kind(t) === "connect");
    this.#delivered = Array.from({ length: units }, () => 0);
    this.#program = program;
  }

  /** The hooks a unit's stand-in writes and reads through: each traced, and windows delivered when the window form is asked for. */
  hooks(i: number): Hooks {
    return {
      onWrite: (bytes) => {
        if (this.tracing) this.trace.push(["write", i, hex(bytes)]);
      },
      onHandled: (bytes, refused) => {
        if (this.tracing && !refused && bytes[0] === P.OPCODES.set_embeddings && bytes[1] === P.EMBEDDING_FORMS.window) {
          this.#deliver(i);
        }
      },
      onAnswered: () => {
        if (this.#dropAfterAnswer.delete(i)) this.fakes[i]!.drop();
      },
      onRead: () => {
        if (this.tracing) this.trace.push(["read", i, "status"]);
      },
    };
  }

  /** One thing the person did, as the library's script has it done. USB power ending a stream is in the packets the library's device sent, and needs nothing here. */
  #act(a: Value[]): void {
    const what = text(a[0]!);
    if (what === "marker") (a.length === 2 ? this.ctx! : this.ctxs.get(text(a[1]!))!).marker(text(a[a.length - 1]!));
    else if (what === "event") this.ctxs.get(text(a[1]!))!.event(text(a[2]!), text(a[3]!), J.toPlain(a[4]!) as Json);
    else if (what === "stop") (a.length === 1 ? this.ctx! : this.ctxs.get(text(a[1]!))!).stop.set();
    else if (what === "link_lost") this.fakes[num(a[1]!)]!.drop();
  }

  #deliver(i: number): void {
    const theirs = this.#deliveries.filter((t) => num(items(t)[1]!) === i)[this.#delivered[i]!++];
    const windows = theirs === undefined ? [] : items(items(theirs)[2]).map((w) => items(w));
    const acts = theirs === undefined ? [] : items(items(theirs)[3]).map((a) => items(a).map(text));
    const sent: unknown[] = [];
    for (const [packet, body] of windows) {
      this.clocks.mono += 4;
      this.clocks.real += 4;
      const h = packet!.kind === "null" ? null : text(packet!);
      if (h !== null) this.fakes[i]!.data.notify(P.fromHex(h));
      this.fakes[i]!.embeddings.notify(P.fromHex(text(body!)));
      sent.push([h, text(body!)]);
    }
    for (const a of acts) {
      if (a[0] === "stop") this.ctx!.stop.set();
      else if (a[0] === "link_lost") this.#dropAfterAnswer.add(i);
    }
    this.trace.push(["embeddings", i, sent, acts]);
  }

  /** The clocks and the wait every run reads, as the library's fixture sets them. */
  install(): void {
    runtime.time = {
      monotonic: () => this.clocks.mono,
      time: () => this.clocks.real,
      sleep: async (seconds: number) => {
        this.clocks.mono += seconds;
        this.clocks.real += seconds;
        const theirs = this.#waits[this.#waited++];
        const packets = theirs === undefined ? [] : items(items(theirs)[2]).map((p) => items(p));
        const acts = theirs === undefined ? [] : items(items(theirs)[3]).map((a) => items(a));
        const sent: unknown[] = [];
        for (const [i, h] of packets) {
          this.fakes[num(i!)]!.data.notify(P.fromHex(text(h!)));
          sent.push([num(i!), text(h!)]);
        }
        for (const a of acts) this.#act(a);
        this.trace.push(["sleep", seconds, sent, acts.map((a) => J.toPlain(J.array(a)))]);
        if (this.#program) await turn();
        else await Promise.resolve();
      },
    };
    runtime.play = (tone, words) => this.trace.push(["cue", tone ?? null, words]);
  }

  /** The clock exchange the library's connection of unit `i` took, taken here on the same clocks. */
  connected(i: number, link: Connection): void {
    const theirs = this.#connects[this.#connected++];
    assert.ok(theirs !== undefined, "the library connected as many devices");
    const t = items(theirs);
    assert.equal(num(t[1]!), i, "the library connected the same device here");
    const before = items(t[2]).map(num) as [number, number];
    const after = items(t[3]).map(num) as [number, number];
    exchange(link, before, after, J.pyStr(t[4]!));
    this.clocks.mono = after[0];
    this.clocks.real = after[1];
    this.trace.push(["connect", i, before, after, Number(J.pyStr(t[4]!))]);
  }

  /** An emit that writes what it hears into the trace, under `who` in a program, and keeps every recording said. */
  emit(dir: string, saves: Saved[], who: string | null, problem: (what: string) => void): Emit {
    return (m) => {
      if (m.type === "saved") {
        const saved = m.recording as Saved;
        if (saved.label !== m.label) problem(`a saved message's recording is ${saved.label}, its label ${String(m.label)}`);
        saves.push(saved);
        writeRecording(dir, saved);
        if (typeof m.training === "string") writeFileSync(join(dir, `${saved.label}.training.json`), m.training);
      }
      this.trace.push(who === null ? ["emit", kept(m)] : ["emit", who, kept(m)]);
    };
  }
}

function restore(): void {
  runtime.time = HOST_TIME;
  runtime.play = audio.play;
}

test("the file is the library's file, byte for byte, from the same packets on two devices", async () => {
  const f = J.field(fixture, "file");
  const units = items(J.get(f, "units"));
  const links = [];
  for (const u of units) links.push(await fakeLink(u));
  const chain = new DaisyChain();
  chain.devices = links.map((l) => l.link);
  const chunks: Uint8Array[] = [];
  chain.recordTo((c) => chunks.push(c));
  await chain.startAll();
  for (const p of items(J.get(f, "packets"))) {
    const [i, h] = items(p);
    links[num(i!)]!.data.notify(P.fromHex(text(h!)));
  }
  await chain.stopAll();
  const got = new TextDecoder().decode(Uint8Array.from(chunks.flatMap((c) => [...c])));
  assert.equal(got, text(J.field(f, "csv")));
});

test("every file a program writes is the library's: its own callback beside it, a file stopped, a device let go, a device too wide refused", async () => {
  const cases = items(J.get(fixture, "files"));
  assert.ok(cases.length >= 3, String(cases.length));
  for (const c of cases) {
    const name = text(J.field(c, "name"));
    const want = J.field(c, "expected");
    const script = new Script(want, 0, false);
    const fakes: (Fake & { link: Connection })[] = [];
    for (const [i, u] of items(J.get(c, "units")).entries()) fakes.push(await fakeLink(u, script.hooks(i)));
    const links = fakes.map((f) => f.link);
    script.tracing = true;
    const chain = new DaisyChain();
    chain.devices = [...links];
    const written = new Map<string, Uint8Array[]>();
    const stops = new Map<string, () => void>();
    const heard: number[][] = [];
    const packets = items(J.get(want, "trace")).filter((t) => text(items(t)[0]!) === "packet");
    let sent = 0;
    let raised: string | null = null;
    try {
      for (const step of items(J.get(c, "script"))) {
        const s = items(step);
        const what = text(s[0]!);
        if (what === "record_to") {
          const n = J.pyStr(s[1]!);
          const all = chain.devices;
          if (s.length > 2) chain.devices = items(s[2]).map((i) => links[num(i)]!);
          const chunks: Uint8Array[] = [];
          written.set(n, chunks);
          try {
            stops.set(n, chain.recordTo((b) => chunks.push(b)));
          } finally {
            chain.devices = all;
          }
        } else if (what === "callback") {
          const i = num(s[1]!);
          links[i]!.on((e) => {
            if (e.type === "samples") for (let r = 0; r < e.batch.rows; r++) heard.push([i, e.batch.index + r]);
          });
        } else if (what === "start_all") {
          await chain.startAll();
        } else if (what === "packet") {
          const t = items(packets[sent++]);
          const i = num(t[1]!);
          script.trace.push(["packet", i, text(t[2]!)]);
          fakes[i]!.data.notify(P.fromHex(text(t[2]!)));
        } else if (what === "stop") {
          stops.get(J.pyStr(s[1]!))!();
        } else if (what === "disconnect") {
          await chain.disconnect(links[num(s[1]!)]!);
        } else if (what === "close") {
          await chain.close();
        }
      }
    } catch (e) {
      raised = said(e);
    }
    const problems: string[] = [];
    compareTrace(script.trace, want, (w) => problems.push(w));
    const wantRaised = J.field(want, "raised");
    if (raised !== (wantRaised.kind === "null" ? null : inJavaScript(text(wantRaised)))) problems.push(`raised ${raised}, the library ${J.dumps(wantRaised)}`);
    if (plain(heard) !== plain(J.toPlain(J.field(want, "heard")))) problems.push(`the program's own callback heard ${plain(heard)}`);
    const csv = J.field(want, "csv");
    for (const [n, theirs] of csv.kind === "object" ? csv.entries : []) {
      const got = new TextDecoder().decode(Uint8Array.from((written.get(n) ?? []).flatMap((b) => [...b])));
      if (got !== text(theirs)) problems.push(`file ${n} differs:\n${got}\nthe library's:\n${text(theirs)}`);
    }
    const devices = links.map((l, i) => (chain.devices.includes(l) ? i : -1)).filter((i) => i >= 0);
    if (plain(devices) !== plain(J.toPlain(J.field(want, "devices")))) problems.push(`devices ${plain(devices)}`);
    assert.equal(problems.length, 0, `${name}:\n${problems.join("\n")}`);
  }
});

test("a file needs a connected device to name its channels", () => {
  assert.throws(() => new DaisyChain().recordTo(() => undefined), /RuntimeError: connect a device before recording/);
});

test("a run needs a device, a registered experiment and a configuration for each device", async () => {
  const chain = new DaisyChain();
  await assert.rejects(chain.run("baseline", {}, () => undefined), /ValueError: a run needs at least one connected device/);
  const u = items(J.get(items(J.get(fixture, "runs"))[0]!, "units"))[0]!;
  chain.devices = [(await fakeLink(u)).link];
  await assert.rejects(chain.run("nothing", {}, () => undefined), /KeyError: 'nothing'/);
  await assert.rejects(chain.run("baseline", {}, () => undefined, { cfgs: [{}, {}] }), /ValueError: 2 setups were given for 1 devices, and a run needs one setup for each device/);
});

test("every run on several devices does what the library's run does, step for step, and records what it records", async () => {
  const runs = items(J.get(fixture, "runs"));
  assert.ok(runs.length >= 33, String(runs.length));
  const problems: string[] = [];
  let steps = 0;
  let recordings = 0;
  for (const r of runs) {
    const name = text(J.field(r, "name"));
    const problem = (what: string): void => {
      problems.push(`${name}: ${what}`);
    };
    const want = J.field(r, "expected");
    const units = items(J.get(r, "units"));
    const script = new Script(want, units.length, false);
    const dir = mkdtempSync(join(tmpdir(), "intomind-together-"));
    try {
      const links: Connection[] = [];
      for (let i = 0; i < units.length; i++) {
        const u = units[i]!;
        const fake = await fakeLink(u, script.hooks(i));
        script.fakes.push(fake);
        links.push(fake.link);
        const [a0, a1] = items(J.get(J.field(u, "sync"), "after")).map(num) as [number, number];
        script.clocks.mono = a0;
        script.clocks.real = a1;
        if (J.truthy(J.field(u, "streaming"))) {
          await fake.link.start();
          for (const h of items(J.get(u, "setup_packets"))) fake.data.notify(P.fromHex(text(h)));
        }
        if (J.truthy(J.get(u, "down") ?? J.bool(false))) fake.drop();
      }
      script.install();
      const saves: Saved[] = [];
      const emit = script.emit(dir, saves, null, problem);
      const cfgs = items(J.get(r, "cfgs")).map((c) => J.toPlain(c) as Json);
      script.ctx = new RunCtx(emit, cfgs[0] ?? null);
      const chain = new DaisyChain();
      chain.devices = [...links];
      const params = J.toPlain(J.field(r, "params")) as Params;
      let labels: string[] | null = null;
      let raised: string | null = null;
      script.tracing = true;
      try {
        labels = await chain.run(text(J.field(r, "experiment")), params, emit, { cfgs, ctx: script.ctx });
      } catch (e) {
        raised = said(e);
      } finally {
        restore();
      }
      script.tracing = false;
      steps += compareTrace(script.trace, want, problem);
      const wantRaised = J.field(want, "raised");
      if (raised !== (wantRaised.kind === "null" ? null : inJavaScript(text(wantRaised)))) problem(`raised ${raised}, the library ${J.dumps(wantRaised)}`);
      if (plain(labels) !== plain(J.toPlain(J.field(want, "labels")))) problem(`labels ${plain(labels)}, the library ${J.dumps(J.field(want, "labels"))}`);
      recordings += compareFolder(dir, want, saves, problem);
      if (plain(links.map(left)) !== plain(J.toPlain(J.field(want, "left")))) problem(`left ${plain(links.map(left))}, the library ${J.dumps(J.field(want, "left"))}`);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }
  assert.equal(problems.length, 0, `${problems.length} problems:\n${problems.join("\n")}`);
  assert.ok(steps >= 1000, String(steps));
  assert.ok(recordings >= 35, String(recordings));
});

// --- the programs --------------------------------------------------------------------

/** What a program has: the chain, the stand-ins on the air, and the scripted world. */
interface Bench {
  readonly chain: DaisyChain;
  readonly script: Script;
  /** Every stand-in on the air, in the order a scan heard them. */
  readonly found: Fake[];
  /** Connect the stand-ins a person picked, by their place in the scan, each with the library's clock exchange. */
  connect(...picked: number[]): Promise<void>;
  /** A run's context, named so a person's acts can reach it. */
  ctx(name: string, cfg: Json): RunCtx;
  /** An emit for the run named `who`. */
  emit(who: string): Emit;
  /** An experiment's parameters as a page's form sends them: every default, then what is given. */
  given(name: string, over?: Params): Params;
  /** A step the programs' experiments hand their groups, by the fixture's name for it. */
  step(key: string): [ExperimentFn, Params];
}

/** The experiments the programs' application registers, as the library's fixture writes its own. */
const APPLICATION: Readonly<Record<string, (b: Bench) => ExperimentFn>> = {
  two_groups: (b) => async (dev, emit, _p, ctx) => {
    const all = dev as Connection[];
    const [berger, pa] = b.step("berger_long");
    const [baseline, pb] = b.step("baseline_long");
    const first = berger(all.slice(0, 2), emit, pa, ctx);
    const second = later(() => baseline(all.slice(2), emit, pb, ctx));
    const [x, y] = await Promise.all([first, second]);
    return [...x, ...y];
  },
  swap: (b) => async (dev, emit, _p, ctx) => {
    const all = dev as Connection[];
    const [berger, pa] = b.step("berger_short");
    await emit({ type: "note", text: "the first group works" });
    const first = await berger(all.slice(0, 2), emit, pa, ctx);
    const second = await berger(all.slice(2), emit, pa, ctx);
    return [...first, ...second];
  },
  in_turn: (b) => async (dev, emit, _p, ctx) => {
    const all = dev as Connection[];
    const [blink, pa] = b.step("blink_short");
    const [saccade, pb] = b.step("saccade_short");
    ctx.cue("begin");
    const first = await blink(all.slice(0, 1), emit, pa, ctx);
    ctx.cue("swap");
    const second = await saccade(all.slice(1), emit, pb, ctx);
    return [...first, ...second];
  },
  twice: (b) => async (dev, emit, _p, ctx) => {
    const all = dev as Connection[];
    const [baseline, pa] = b.step("baseline_one");
    return baseline([all[1]!, all[1]!], emit, pa, ctx);
  },
  again: (b) => async (dev, emit, _p, ctx) => {
    const [baseline, pa] = b.step("baseline_quarter");
    const first = await baseline(dev, emit, pa, ctx);
    const second = await baseline(dev, emit, pa, ctx);
    return [...first, ...second];
  },
  late: (b) => async (dev, emit, _p, ctx) => {
    const [baseline, pa] = b.step("baseline_one");
    await b.connect(1);
    return baseline([dev as Connection, b.chain.devices[b.chain.devices.length - 1]!], emit, pa, ctx);
  },
  outside: (b) => async (dev, emit, _p, ctx) => {
    const all = dev as Connection[];
    const [baseline, pa] = b.step("baseline_one");
    await emit({ type: "note", text: "before any step" });
    const got = await baseline(all.slice(1), emit, pa, ctx);
    await emit({ type: "note", text: "after it", devices: ["as the experiment says"] });
    return got;
  },
};

/** The programs, as the library's fixture runs its own, from the scan. */
const PROGRAMS: Readonly<Record<string, (b: Bench) => Promise<unknown>>> = {
  async everyone(b) {
    await b.connect(0);
    await b.connect(2);
    await b.connect(3);
    b.script.tracing = true;
    const wearers = ["S01", "S02", "S03"].map((w) => ({ gain: 24, subject_id: w }));
    return b.chain.run("record", b.given("record", { max_seconds: 600 }), b.emit("run"), { cfgs: wearers, ctx: b.ctx("run", { gain: 24 }) });
  },
  async two_runs(b) {
    await b.connect(0, 1, 2);
    b.script.tracing = true;
    const [one, two, three] = b.chain.devices as [Connection, Connection, Connection];
    const first = b.ctx("first", { gain: 24 });
    const second = b.ctx("second", { gain: 24 });
    const a = b.chain.run("record", b.given("record", { max_seconds: 60 }), b.emit("first"), {
      cfgs: [{ subject_id: "S01" }],
      ctx: first,
      devices: [one],
    });
    const c = later(() =>
      b.chain.run("record", b.given("record", { max_seconds: 60 }), b.emit("second"), {
        cfgs: [{ subject_id: "S02" }, { subject_id: "S03" }],
        ctx: second,
        devices: [two, three],
      }),
    );
    return Promise.all([a, c]);
  },
  async groups(b) {
    await b.connect(0, 1, 2, 3);
    b.script.tracing = true;
    const wearers = ["S01", "S02", "S03", "S04"].map((w) => ({ gain: 24, subject_id: w }));
    return b.chain.run("two_groups", {}, b.emit("run"), { cfgs: wearers, ctx: b.ctx("run", { gain: 24 }) });
  },
  async resting(b) {
    await b.connect(0, 1, 2, 3);
    b.script.tracing = true;
    await b.chain.startAll();
    const wearers = ["S01", "S02", "S03", "S04"].map((w) => ({ gain: 24, subject_id: w }));
    return b.chain.run("swap", {}, b.emit("run"), { cfgs: wearers, ctx: b.ctx("run", { gain: 24 }) });
  },
  async naming(b) {
    await b.connect(0, 1);
    b.script.tracing = true;
    const [first, second] = b.chain.devices as [Connection, Connection];
    const stranger = await connect(b.found[2]!.device);
    const names: string[][] = [];
    for (const given of [[second], [second], new Set([second]), second]) {
      names.push(await b.chain.run("baseline", b.given("baseline", { seconds: 1 }), b.emit("run"), { devices: given }));
    }
    const refused: string[] = [];
    for (const devices of [
      [first, stranger],
      [second, first, second],
    ]) {
      try {
        await b.chain.run("baseline", b.given("baseline", { seconds: 1 }), b.emit("refused"), { devices });
      } catch (e) {
        refused.push(said(e));
      }
    }
    return { names, refused };
  },
  async busy(b) {
    await b.connect(0, 1);
    b.script.tracing = true;
    const [one] = b.chain.devices as [Connection];
    const first = b.chain.run("record", b.given("record", { max_seconds: 60 }), b.emit("first"), { ctx: b.ctx("first", { gain: 24 }), devices: [one] });
    for (let k = 0; k < 3; k++) await turn();
    const refused: string[] = [];
    for (const [name, params] of [
      ["baseline", b.given("baseline", { seconds: 1 })],
      ["twice", {}],
    ] as const) {
      try {
        await b.chain.run(name, params, b.emit(name));
      } catch (e) {
        refused.push(said(e));
      }
    }
    const names = await first;
    const again = await b.chain.run("baseline", b.given("baseline", { seconds: 1 }), b.emit("again"), { devices: [one] });
    return { refused, names, again };
  },
  async turns(b) {
    await b.connect(0, 1);
    b.script.tracing = true;
    return b.chain.run("in_turn", {}, b.emit("run"), { ctx: b.ctx("run", { gain: 24 }) });
  },
  async taken(b) {
    await b.connect(0, 1);
    b.script.tracing = true;
    const several = await b.chain.run("again", {}, b.emit("several"), { cfgs: [{ gain: 24 }, { gain: 24 }] });
    const one = await b.chain.run("again", {}, b.emit("one"), { devices: [b.chain.devices[0]!] });
    return { several, one };
  },
  async latecomer(b) {
    await b.connect(0);
    b.script.tracing = true;
    return b.chain.run("late", {}, b.emit("run"));
  },
  async own_messages(b) {
    await b.connect(0, 1);
    b.script.tracing = true;
    return b.chain.run("outside", {}, b.emit("run"));
  },
  async comeback(b) {
    await b.connect(0);
    b.script.tracing = true;
    let raised: string | null = null;
    try {
      await b.chain.run("head_training", b.given("head_training", { rep_length: 40, reps: 2 }), b.emit("dropped"));
    } catch (e) {
      raised = said(e);
    }
    b.found[0]!.back();
    const names = await b.chain.run("head_training", b.given("head_training", { rep_length: 40, reps: 2 }), b.emit("back"));
    return { raised, names };
  },
  async refused_off(b) {
    await b.connect(0);
    b.script.tracing = true;
    const first = await b.chain.run("head_training", b.given("head_training", { rep_length: 2, reps: 2 }), b.emit("refusing"));
    b.found[0]!.refuseForms.clear();
    const second = await b.chain.run("head_training", b.given("head_training", { rep_length: 2, reps: 2 }), b.emit("taken"));
    return { first, second };
  },
};

test("every program does what the library's program does, from the scan: one run on all, two at once, two groups at once, one group resting, a link back, a turn off refused", async () => {
  const programs = items(J.get(fixture, "programs"));
  assert.ok(programs.length >= 12, String(programs.length));
  const problems: string[] = [];
  let steps = 0;
  let recordings = 0;
  for (const prog of programs) {
    const name = text(J.field(prog, "name"));
    const problem = (what: string): void => {
      problems.push(`${name}: ${what}`);
    };
    const want = J.field(prog, "expected");
    const units = items(J.get(prog, "units"));
    const script = new Script(want, units.length, true);
    const dir = mkdtempSync(join(tmpdir(), "intomind-program-"));
    const saves: Saved[] = [];
    const chain = new DaisyChain();
    for (let i = 0; i < units.length; i++) script.fakes.push(fakeDevice(units[i]!, script.hooks(i)));
    const stepsOf = J.field(prog, "steps");
    const bench: Bench = {
      chain,
      script,
      found: script.fakes,
      async connect(...picked) {
        for (const i of picked) {
          await chain.connect(script.fakes[i]!.device);
          script.connected(i, chain.devices[chain.devices.length - 1] as Connection);
        }
      },
      ctx(n, cfg) {
        const ctx = new RunCtx(() => undefined, cfg);
        script.ctxs.set(n, ctx);
        return ctx;
      },
      emit(who) {
        return script.emit(dir, saves, who, problem);
      },
      given(n, over = {}) {
        const spec = REGISTRY.get(n)!;
        return { ...Object.fromEntries(spec.params.map((p) => [p.name, p.default])), ...over };
      },
      step(key) {
        const s = J.field(stepsOf, key);
        return [REGISTRY.get(text(J.field(s, "experiment")))!.fn, J.toPlain(J.field(s, "params")) as Params];
      },
    };
    const added: string[] = [];
    for (const n of items(J.get(prog, "application")).map(text)) {
      experiment(n, n, [], APPLICATION[n]!(bench));
      added.push(n);
    }
    let gave: unknown = null;
    let raised: string | null = null;
    script.install();
    try {
      gave = await PROGRAMS[text(J.field(prog, "program"))]!(bench);
    } catch (e) {
      raised = said(e);
    } finally {
      restore();
      for (const n of added) REGISTRY.delete(n);
    }
    script.tracing = false;
    try {
      steps += compareTrace(script.trace, want, problem);
      const wantRaised = J.field(want, "raised");
      if (raised !== (wantRaised.kind === "null" ? null : inJavaScript(text(wantRaised)))) problem(`raised ${raised}, the library ${J.dumps(wantRaised)}`);
      if (plain(gave) !== plain(refusalsInJavaScript(J.toPlain(J.field(want, "gave"))))) problem(`gave back ${plain(gave)}, the library ${J.dumps(J.field(want, "gave"))}`);
      recordings += compareFolder(dir, want, saves, problem);
      const linkOf = (fake: Fake) => (chain.devices as Connection[]).find((l) => l.device === fake.device);
      const devices = script.fakes.map((f, i) => (linkOf(f) !== undefined ? i : -1)).filter((i) => i >= 0);
      if (plain(devices) !== plain(J.toPlain(J.field(want, "devices")))) problem(`devices ${plain(devices)}`);
      const theirLeft = items(J.get(want, "left"));
      script.fakes.forEach((f, i) => {
        const link = linkOf(f);
        if (link === undefined) return;
        if (plain(left(link)) !== plain(J.toPlain(theirLeft[i]!))) problem(`left on ${i}: ${plain(left(link))}, the library ${J.dumps(theirLeft[i]!)}`);
      });
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }
  assert.equal(problems.length, 0, `${problems.length} problems:\n${problems.join("\n")}`);
  assert.ok(steps >= 900, String(steps));
  assert.ok(recordings >= 25, String(recordings));
});

test("every device of a chain is placed on one reference point, the first anchor any took, and a device alone keeps its own", async () => {
  // Three stand-ins whose own clocks started counting at different moments
  // on the host's monotonic clock. Two are connected through one chain, the
  // wall clock stepped two and a half seconds between their exchanges, and
  // the third is connected on its own after the step.
  const u = items(J.get(items(J.get(fixture, "runs"))[0]!, "units"))[0]!;
  const chain = new DaisyChain();
  const [a, b, c] = [fakeDevice(u), fakeDevice(u), fakeDevice(u)];
  await chain.connect(a.device, b.device);
  const [first, second] = chain.devices as [Connection, Connection];
  const alone = await connect(c.device);
  let mono = 1000;
  let wall = 1_767_225_600;
  const boots = new Map<Connection, number>([
    [first, 10],
    [second, 400],
    [alone, 70],
  ]);
  const ticksNow = (link: Connection): bigint => BigInt(Math.round((mono - boots.get(link)!) * 1_000_000));
  const take = (link: Connection): void => exchange(link, [mono, mono + wall], [mono, mono + wall], ticksNow(link).toString());
  take(first);
  mono += 5;
  wall += 2.5; // the wall clock is stepped
  take(second);
  take(alone);
  mono += 30;
  const at = (link: Connection): number => link.session.clock!.hostTime(ticksNow(link));
  assert.ok(Math.abs(at(first) - at(second)) < 1e-5, `two timelines in one chain: ${at(first)} and ${at(second)}`);
  assert.ok(Math.abs(at(alone) - (at(first) + 2.5)) < 1e-5, "a device alone keeps its own reference point");
  const anchors = (link: Connection) => link.session.clock!.anchors();
  assert.deepEqual(anchors(alone), alone.session.clock!.syncs.map((s) => [s.host, s.mono]), "and lists only its own anchors");
  assert.deepEqual(anchors(second)[0], anchors(first)[0], "the reference point is among the anchors of the device placed on it");
  assert.equal(anchors(second).length, 2);
  const quality = second.session.clock!.quality();
  assert.ok(J.truthy(J.field(quality, "wall_clock_stepped")), "the step after the reference point is said");
  assert.equal(J.asF64(J.field(quality, "frame_mono")), anchors(first)[0]![1]);
  assert.ok(!J.truthy(J.field(first.session.clock!.quality(), "wall_clock_stepped")));
});
