/**
 * The experiments held to the Python library's: every run in
 * `contract/experiments` replayed through a browser connection to a device
 * made of nothing, on the same scripted clocks, and compared step for step
 * with what the library did. Every command written to the device, every
 * read, every wait and how long it was, every cue and every message, in
 * order; then the recording, its manifest and events byte for byte and its
 * arrays element for element; and the error a run raised, in the library's
 * words. Also the registry, Python's seeded shuffle, and the cue tones.
 *
 * The one difference allowed is the software a recording names: a recording
 * made here names this package where the library names Python. A head
 * training's windows are sent as the library's run was sent them, its head,
 * held-out score and the reason it has none compared in its saved message,
 * and its training data byte for byte, Stop pressed in the middle of a rep
 * included, a person's own states, two and three, and the link dropping
 * while a rep's windows are waited for. Every experiment given one
 * parameter it cannot use refuses it by name before anything is sent.
 */

import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import process from "node:process";
import test from "node:test";

import { CHOICES, TONES, available, play } from "../src/audio.ts";
import { sha256 } from "../src/digest.ts";
import { writeRecording } from "../src/capture/node.ts";
import * as npy from "../src/capture/npy.ts";
import * as J from "../src/capture/pyjson.ts";
import type { Value } from "../src/capture/pyjson.ts";
import type { Saved } from "../src/capture/record.ts";
import { readNpz } from "../src/capture/recording.ts";
import { PACKAGE_VERSION } from "../src/capture/version.ts";
import { REGISTRY, RunCtx, TONE_PARAMS, cueSound, say, type Message, type Params } from "../src/experiments.ts";
import * as experimentsEntry from "../src/experiments.ts";
import * as sdk from "../src/index.ts";
import * as P from "../src/protocol.ts";
import { Random } from "../src/pyrandom.ts";
import { HOST_TIME, gainsAsked, marked, runtime, tagged } from "../src/run.ts";
import { Session, SessionError } from "../src/session.ts";
import { wav } from "../src/tones.ts";
import { Connection, connect } from "../src/web-bluetooth.ts";
import { Browser } from "./browser.ts";
import { inJavaScript } from "./words.ts";

// A recording's name carries the local time, and the library's runs were
// made with the time zone at UTC.
process.env.TZ = "UTC";

const fixture = J.loads(readFileSync(new URL("../../contract/experiments/experiments.json", import.meta.url), "utf8"));

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

/** Plain data, written the one way, so two values are compared by what they hold. */
function plain(v: unknown): string {
  return JSON.stringify(v, (_k, x) => (typeof x === "bigint" ? x.toString() : x));
}

/**
 * A refusal as the fixture writes the library's, `Exception: message`. The
 * package's own errors already begin with the name of the exception the
 * library raises, and the ones that share Python's name begin with it here.
 */
function said(e: unknown): string {
  const err = e as Error;
  return err instanceof P.Invalid || err instanceof SessionError ? err.message : `${err.name}: ${err.message}`;
}

/** Every run reads the host's own clocks and plays its own cues, as it does outside these tests. */
function restore(): void {
  runtime.time = HOST_TIME;
  runtime.play = play;
}

test("the registry is the library's: names, words, parameters, preconditions and analyses", () => {
  const theirs = items(J.get(fixture, "registry"));
  assert.deepEqual([...REGISTRY.keys()], theirs.map((s) => text(J.field(s, "name"))));
  for (const s of theirs) {
    const mine = REGISTRY.get(text(J.field(s, "name")))!;
    assert.equal(mine.doc, text(J.field(s, "doc")));
    assert.equal(plain(mine.params), plain(J.toPlain(J.field(s, "params"))), mine.name);
    // What a manifest is stamped with, so the same bytes, floats and all.
    assert.equal(J.dumps(mine.enforced), J.dumps(J.field(s, "enforced")), mine.name);
    assert.equal(J.dumps(mine.prereg), J.dumps(J.field(s, "prereg")), mine.name);
  }
  assert.equal(plain(TONE_PARAMS), plain(J.toPlain(J.field(fixture, "tone_params"))));
});

test("Python's seeded draws and shuffles come out the same, for every seed", () => {
  const cases = items(J.get(fixture, "random"));
  assert.ok(cases.length >= 10);
  for (const c of cases) {
    const seed = BigInt(J.pyStr(J.field(c, "seed")));
    const words = new Random(seed);
    assert.deepEqual(Array.from({ length: 10 }, () => words.getrandbits(32)), items(J.get(c, "words")).map(num), `seed ${seed}`);
    const bits = new Random(seed);
    assert.deepEqual(Array.from({ length: 33 }, (_, k) => bits.getrandbits(k)), items(J.get(c, "bits")).map(num), `seed ${seed}`);
    const below = new Random(seed);
    assert.deepEqual(
      [1, 2, 3, 7, 100, 2 ** 31 - 1, 2 ** 32 - 1].map((n) => below.randbelow(n)),
      items(J.get(c, "below")).map(num),
      `seed ${seed}`,
    );
    const x = Array.from({ length: 40 }, (_, i) => i);
    new Random(seed).shuffle(x);
    assert.deepEqual(x, items(J.get(c, "shuffled")).map(num), `seed ${seed}`);
    const cues = [...Array(30).fill("blink"), ...Array(7).fill("sham")];
    new Random(seed).shuffle(cues);
    assert.deepEqual(cues, items(J.get(c, "cues")).map(text), `seed ${seed}`);
  }
});

test("every cue tone is the library's WAV file, byte for byte", () => {
  const tones = items(J.get(fixture, "tones"));
  assert.deepEqual(tones.map((t) => text(J.field(t, "name"))), [...CHOICES]);
  for (const t of tones) {
    const name = text(J.field(t, "name"));
    const b = wav(name);
    const want = J.field(t, "sha256");
    if (want.kind === "null") {
      assert.equal(b, null, name);
      continue;
    }
    assert.ok(b !== null, name);
    assert.equal(b.length, num(J.field(t, "length")), name);
    assert.equal(hex(sha256(b)), text(want), name);
  }
  assert.equal(wav("not a tone"), null);
  assert.equal(Object.keys(TONES).length, 9);
});

test("a cue never throws and never waits, and is silent where nothing can sound", () => {
  // Node has no speech and no audio, as a computer without either is.
  assert.deepEqual(available(), { tones: [...CHOICES], aplay: false, speech: false });
  for (const tone of [...CHOICES, "", null, undefined, 5, "not a tone"]) play(tone, "Recording.");

  const spoken: { text: string; rate: number }[] = [];
  const played: number[] = [];
  const g = globalThis as Record<string, unknown>;
  g.speechSynthesis = { speak: (u: { text: string; rate: number }) => spoken.push(u) };
  g.SpeechSynthesisUtterance = class {
    text: string;
    rate = 1;
    constructor(t: string) {
      this.text = t;
    }
  };
  g.AudioContext = class {
    state = "suspended";
    destination = {};
    resume() {
      this.state = "running";
      return Promise.resolve();
    }
    createBuffer(_c: number, n: number) {
      const data = new Float32Array(n);
      return { getChannelData: () => data };
    }
    createBufferSource() {
      return {
        buffer: null as unknown,
        connect() {},
        start() {
          played.push((this.buffer as { getChannelData(): Float32Array }).getChannelData().length);
        },
      };
    }
  };
  try {
    assert.deepEqual(available(), { tones: [...CHOICES], aplay: true, speech: true });
    play("speech", "Close your eyes");
    play("speech", "");
    play("beep_low");
    play("none");
    assert.deepEqual(spoken.map((u) => [u.text, u.rate]), [["Close your eyes", 0.8]]);
    assert.deepEqual(played, [(wav("beep_low")!.length - 44) / 2]);
    g.speechSynthesis = {
      speak() {
        throw new Error("no voice");
      },
    };
    play("speech", "still nothing thrown");
  } finally {
    delete g.speechSynthesis;
    delete g.SpeechSynthesisUtterance;
    delete g.AudioContext;
  }
});

// --- the runs --------------------------------------------------------------

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

/**
 * The library's manifest as a run through a connection writes it, with the
 * two substitutions allowed and nothing else: the software it names, put as
 * this package names itself, and the adapter, the library's own Bluetooth
 * device where the browser connection names itself. Everything else is
 * compared byte for byte.
 */
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

/**
 * A device made of nothing, as a fixture run describes it: what it is, what
 * it answers and refuses, and its status, connected through the browser
 * transport. `drop` takes its link down. It answers a stop with ok whether or
 * not it is streaming, as the firmware does. `hooks.onWrite` hears each
 * control write as it is made, `hooks.onHandled` hears it once the write
 * is done, as the device handles it, before its answer, and
 * `hooks.onAnswered` once its answer is given.
 */
async function fakeLink(
  r: Value,
  hooks: {
    readonly onWrite?: (bytes: Uint8Array, refused: boolean) => void;
    readonly onHandled?: (bytes: Uint8Array, refused: boolean) => void;
    readonly onAnswered?: (bytes: Uint8Array, refused: boolean) => void;
    readonly onRead?: () => void;
  } = {},
): Promise<{ link: Connection; characteristics: Map<string, FakeCharacteristic>; drop: () => void }> {
  const answers = new Map<number, Uint8Array>();
  const given = J.field(r, "answers");
  if (given.kind === "object") for (const [op, a] of given.entries) answers.set(Number(op), P.fromHex(text(a)));
  const refuse = new Set(items(J.get(r, "refuse") ?? J.array([])).map(num));
  const characteristics = new Map<string, FakeCharacteristic>([
    [P.DEVICE_INFO, new FakeCharacteristic(P.DEVICE_INFO, P.fromHex(text(J.field(r, "info"))))],
    [P.CONTROL, new FakeCharacteristic(P.CONTROL)],
    [P.CONTROL_RESPONSE, new FakeCharacteristic(P.CONTROL_RESPONSE)],
    [P.EEG_DATA, new FakeCharacteristic(P.EEG_DATA)],
    [P.STATUS, new FakeCharacteristic(P.STATUS, P.fromHex(text(J.field(r, "status"))))],
    [P.EMBEDDINGS, new FakeCharacteristic(P.EMBEDDINGS)],
  ]);
  characteristics.get(P.CONTROL)!.onWrite = (bytes) => {
    const op = bytes[0]!;
    const refused = refuse.has(op);
    hooks.onWrite?.(bytes, refused);
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
  const browser = new Browser(characteristics, { id: "fake", name: "IntoMind One" });
  const link = await connect(browser.device);
  return {
    link,
    characteristics,
    drop: () => browser.drop(),
  };
}

/** The arrays of an npz as the fixture lists them. */
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

/** One step of a trace, written the one way, so a step here and the library's compare as text. */
function step(t: unknown[]): string {
  return plain(t);
}

test("every run does what the library's run does, step for step, and records what it records", async () => {
  const runs = items(J.get(fixture, "runs"));
  assert.ok(runs.length >= 72, String(runs.length));
  const problems: string[] = [];
  let steps = 0;
  let recordings = 0;
  for (const r of runs) {
    const name = text(J.field(r, "name"));
    const problem = (what: string): void => {
      problems.push(`${name}: ${what}`);
    };
    const want = J.field(r, "expected");
    const dir = mkdtempSync(join(tmpdir(), "intomind-experiments-"));
    try {
      // The device, and every write and read made of it while the run runs.
      const trace: unknown[][] = [];
      let tracing = false;
      // What the device sent each time the run asked for the window form, in
      // the order the library's run asked, replayed when this run asks.
      const deliveries = items(J.get(want, "trace")).filter((t) => text(items(t as Value)[0]!) === "embeddings");
      let delivered = 0;
      let deliver: (() => void) | null = null;
      // A link lost after a rep's windows drops once the device has answered
      // the request that asked for them, as the library's fixture radio drops
      // it: nothing reaches the host on a link that is down.
      let dropAfterAnswer = false;
      const { link, characteristics, drop } = await fakeLink(r, {
        onWrite: (bytes) => {
          if (tracing) trace.push(["write", hex(bytes)]);
        },
        onHandled: (bytes, refused) => {
          if (tracing && !refused && bytes[0] === P.OPCODES.set_embeddings && bytes[1] === P.EMBEDDING_FORMS.window) deliver?.();
        },
        onAnswered: () => {
          if (dropAfterAnswer) {
            dropAfterAnswer = false;
            drop();
          }
        },
        onRead: () => {
          if (tracing) trace.push(["read", "status"]);
        },
      });

      // The exchange at connect, on the script's clocks.
      const sync = J.field(r, "sync");
      const [b0, b1] = items(J.get(sync, "before")).map(num) as [number, number];
      const [a0, a1] = items(J.get(sync, "after")).map(num) as [number, number];
      const ticks = new Uint8Array(8);
      new DataView(ticks.buffer).setBigUint64(0, BigInt(J.pyStr(J.field(sync, "ticks"))), true);
      link.session.onTimeSyncClocks(ticks, { monotonic: b0, realtime: b1 }, { monotonic: a0, realtime: a1 });
      const clocks = { mono: a0, real: a1 };
      if (J.truthy(J.field(r, "streaming"))) {
        await link.start();
        for (const h of items(J.get(r, "setup_packets"))) characteristics.get(P.EEG_DATA)!.notify(P.fromHex(text(h)));
      }
      if (J.truthy(J.get(r, "down") ?? J.bool(false))) drop();

      // What the library's run waited, what arrived meanwhile, and what the
      // person did after each wait, replayed in the order the library met them.
      // USB power ending a stream is in the packets the library's device sent,
      // and the device answers a stop with ok all the same.
      const waits = items(J.get(want, "trace")).filter((t) => text(items(t as Value)[0]!) === "sleep");
      let waited = 0;
      let ctx: RunCtx | null = null;
      const time = {
        monotonic: () => clocks.mono,
        time: () => clocks.real,
        sleep: async (seconds: number) => {
          clocks.mono += seconds;
          clocks.real += seconds;
          const theirs = waits[waited++];
          const packets = theirs === undefined ? [] : items(items(theirs)[2]).map(text);
          const acts = theirs === undefined ? [] : items(items(theirs)[3]).map((a) => items(a).map(text));
          for (const h of packets) characteristics.get(P.EEG_DATA)!.notify(P.fromHex(h));
          for (const a of acts) {
            if (a[0] === "marker") ctx!.marker(a[1]!);
            else if (a[0] === "stop") ctx!.stop.set();
            else if (a[0] === "link_lost") drop();
          }
          trace.push(["sleep", seconds, packets, acts]);
          await Promise.resolve();
        },
      };
      deliver = () => {
        const theirs = deliveries[delivered++];
        const windows = theirs === undefined ? [] : items(items(theirs)[1]).map((w) => items(w));
        const acts = theirs === undefined ? [] : items(items(theirs)[2]).map((a) => items(a).map(text));
        const sent: unknown[] = [];
        for (const [packet, body] of windows) {
          clocks.mono += 4;
          clocks.real += 4;
          const h = packet!.kind === "null" ? null : text(packet!);
          if (h !== null) characteristics.get(P.EEG_DATA)!.notify(P.fromHex(h));
          characteristics.get(P.EMBEDDINGS)!.notify(P.fromHex(text(body!)));
          sent.push([h, text(body!)]);
        }
        for (const a of acts) {
          if (a[0] === "stop") ctx!.stop.set();
          else if (a[0] === "link_lost") dropAfterAnswer = true;
        }
        trace.push(["embeddings", sent, acts]);
      };
      // The recording's files come with the message that says it was saved,
      // where the library writes them into its folder, and a head's training
      // data beside them.
      const kept: Saved[] = [];
      const emit = (m: Message): void => {
        if (m.type === "saved") {
          const saved = m.recording as Saved;
          if (saved.label !== m.label) problem(`a saved message's recording is ${saved.label}, its label ${String(m.label)}`);
          kept.push(saved);
          writeRecording(dir, saved);
          if (typeof m.training === "string") writeFileSync(join(dir, `${saved.label}.training.json`), m.training);
        }
        trace.push([
          "emit",
          m.type === "saved"
            ? { type: "saved", label: m.label, ...("head" in m ? { head: m.head, score: m.score, no_head: m.no_head } : {}) }
            : m,
        ]);
      };
      ctx = new RunCtx(emit, J.field(r, "cfg"));
      // A program's own callback for whole windows, which hears every window
      // the run takes and is still the program's after it.
      const heard: number[] = [];
      const mine = (_l: Connection, w: P.EmbeddingWindow): void => {
        heard.push(w.index);
      };
      link.onEmbedding = mine;
      // The clocks and the cues, as the library's fixture sets them for its run.
      runtime.time = time;
      runtime.play = (tone, said) => trace.push(["cue", tone ?? null, said]);
      const spec = REGISTRY.get(text(J.field(r, "experiment")))!;
      const params = J.toPlain(J.field(r, "params")) as Record<string, unknown>;
      let labels: string[] | null = null;
      let raised: string | null = null;
      tracing = true;
      try {
        labels = await spec.fn(link, emit, params, ctx);
      } catch (e) {
        raised = said(e);
      } finally {
        restore();
      }
      tracing = false;

      // Step for step.
      const theirTrace = items(J.get(want, "trace")).map((t) => J.toPlain(t) as unknown[]);
      const n = Math.max(trace.length, theirTrace.length);
      for (let i = 0; i < n; i++) {
        const mine = trace[i] === undefined ? "nothing" : step(trace[i]!);
        const theirs = theirTrace[i] === undefined ? "nothing" : step(theirTrace[i]!);
        if (mine !== theirs) {
          problem(`step ${i}: here ${mine}, the library ${theirs}`);
          break;
        }
        steps++;
      }
      const wantRaised = J.field(want, "raised");
      if (raised !== (wantRaised.kind === "null" ? null : inJavaScript(text(wantRaised)))) problem(`raised ${raised}, the library ${J.dumps(wantRaised)}`);
      const wantLabels = J.field(want, "labels");
      if (plain(labels) !== plain(J.toPlain(wantLabels))) problem(`labels ${plain(labels)}, the library ${J.dumps(wantLabels)}`);
      const windows = deliveries.flatMap((t) => items(items(t as Value)[1]));
      if (heard.length !== windows.length) problem(`the program's own callback heard ${heard.length} of ${windows.length} windows`);
      if (link.onEmbedding !== mine) problem("the program's own callback was not left as it was");

      // The recording.
      for (const rec of items(J.get(want, "recordings"))) {
        const label = text(J.field(rec, "label"));
        const read = (file: string): string | null => (existsSync(join(dir, file)) ? readFileSync(join(dir, file), "utf8") : null);
        const meta = read(`${label}.meta.json`);
        if (meta !== asMadeHere(text(J.field(rec, "meta_json")))) problem(`${label}: the manifest differs`);
        const events = read(`${label}.events.json`);
        const wantEvents = J.field(rec, "events_json");
        if (events !== (wantEvents.kind === "null" ? null : text(wantEvents))) problem(`${label}: the events differ:\n${events}\n${J.pyStr(wantEvents)}`);
        const taught = read(`${label}.training.json`);
        const wantTaught = J.field(rec, "training_json");
        if (wantTaught.kind !== "null" || taught !== null) {
          if (taught !== (wantTaught.kind === "null" ? null : text(wantTaught))) problem(`${label}: the training data differs`);
        }
        const sums = read(`${label}.sha256`);
        const names = sums === null ? null : sums.split("\n").filter((l) => l !== "").map((l) => l.split("  ")[1]);
        if (plain(names) !== plain(J.toPlain(J.field(rec, "sha256_names")))) problem(`${label}: checksums for ${plain(names)}`);
        const arrays = plain(npzDigest(new Uint8Array(readFileSync(join(dir, `${label}.npz`)))));
        if (arrays !== plain(J.toPlain(J.field(rec, "arrays")))) problem(`${label}: the arrays differ`);
        const carried = kept.find((s) => s.label === label);
        if (carried === undefined) problem(`${label}: no saved message carried it`);
        else if (J.dumps(J.field(J.field(carried.manifest, "signal_chain"), "adapter")) !== J.dumps(J.fromPlain(Connection.ADAPTER))) {
          problem(`${label}: the recording does not name the connection as its adapter`);
        }
        recordings++;
      }
      const files = readdirSync(dir)
        .sort()
        .map((f) => [f, (statSync(join(dir, f)).mode & 0o222) === 0]);
      const wantFiles = J.field(want, "files");
      const theirFiles =
        wantFiles.kind === "object"
          ? wantFiles.entries.map(([f, v]) => [f, J.truthy(J.field(v, "read_only"))]).sort((a, b) => ((a[0] as string) < (b[0] as string) ? -1 : 1))
          : [];
      if (plain(files) !== plain(theirFiles)) problem(`files ${plain(files)}, the library's ${plain(theirFiles)}`);
      const log = existsSync(join(dir, "SESSIONS.jsonl")) ? readFileSync(join(dir, "SESSIONS.jsonl"), "utf8") : null;
      const wantLog = J.field(want, "sessions_jsonl");
      if (log !== (wantLog.kind === "null" ? null : text(wantLog))) problem("the session log differs");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }
  assert.equal(problems.length, 0, `${problems.length} problems:\n${problems.join("\n")}`);
  assert.ok(steps >= 300, String(steps));
  assert.ok(recordings >= 13, String(recordings));
});

test("a recording's name carries the bias mode its setup gives under bias, in the mode's own word, and na where the setup gives none, as the library's does", () => {
  // The part of a name before its gain: no bias, a bias of null, as the
  // setup of a device that drives no bias electrode has, one that is not a
  // mode, and the key a setup of old gave the bias drive under, are each na.
  for (const mode of Object.keys(P.BIAS_MODES)) {
    const name = experimentsEntry.label(new RunCtx(() => {}, { gain: 24, bias: mode }), "x");
    assert.match(name, new RegExp(`^x_${mode}_g24_\\d{8}-\\d{6}$`), name);
  }
  for (const setup of [{ gain: 24 }, { gain: 24, bias: null }, { gain: 24, bias: "sideways" }, { gain: 24, bias: ["on"] }, { gain: 24, rld: "on" }]) {
    const name = experimentsEntry.label(new RunCtx(() => {}, setup), "x");
    assert.match(name, /^x_na_g24_\d{8}-\d{6}$/, `${JSON.stringify(setup)}: ${name}`);
  }
});

test("a recording's name carries na for its gain where its setup gives none or gives null, as the library's does, never the word null", () => {
  for (const setup of [{ bias: "on" }, { gain: null, bias: "on" }]) {
    const name = experimentsEntry.label(new RunCtx(() => {}, setup), "x");
    assert.match(name, /^x_on_gna_\d{8}-\d{6}$/, `${JSON.stringify(setup)}: ${name}`);
  }
  assert.match(experimentsEntry.label(new RunCtx(() => {}, { gain: null }), "x"), /^x_na_gna_\d{8}-\d{6}$/);
});

test("a run's events carry both clocks and refuse a field given twice", () => {
  let mono = 10;
  let real = 1_767_225_600;
  runtime.time = { monotonic: () => mono, time: () => real, sleep: () => Promise.resolve() };
  try {
    const ctx = new RunCtx(() => undefined, { gain: 24 });
    const e = ctx.marker("event 1");
    mono += 1.5;
    real += 1.5;
    ctx.cue("closed", { rep: 0 });
    ctx.event("button", "left", { pressed: true });
    assert.equal(J.dumps(e), '{"cond": "event 1", "kind": "marker", "host_time": 1767225600.0, "mono": 10.0}');
    assert.equal(J.dumps(J.array(ctx.events)).includes('"kind": "button"'), true);
    assert.throws(() => ctx.cue("open", { mono: 1 }), /multiple values for keyword argument 'mono'/);
    assert.throws(() => new RunCtx(() => undefined, [1]), TypeError);
    assert.equal(J.dumps(new RunCtx(() => undefined).cfg), "{}");
    assert.ok(!ctx.stop.isSet());
    ctx.stop.set();
    assert.ok(ctx.stop.isSet());
  } finally {
    restore();
  }
});

test("a cue point and spoken words go where every run's cues go", () => {
  const heard: unknown[][] = [];
  runtime.play = (tone, text) => heard.push([tone, text]);
  try {
    cueSound("beep_low");
    cueSound("speech", "Close your eyes");
    say("Done.");
  } finally {
    restore();
  }
  assert.deepEqual(heard, [
    ["beep_low", ""],
    ["speech", "Close your eyes"],
    ["speech", "Done."],
  ]);
});

test("a parameter an experiment cannot use is refused in the library's words, by name, before anything is sent", async () => {
  const spec = REGISTRY.get("baseline")!;
  const ctx = new RunCtx(() => undefined);
  const notALink = {} as Parameters<typeof spec.fn>[0];
  const refused = (name: string, params: Params, why: string): Promise<void> =>
    assert.rejects(REGISTRY.get(name)!.fn(notALink, () => undefined, params, ctx), (e: Error) => e instanceof P.Invalid && e.message === why);
  await refused("baseline", {}, "ValueError: seconds is not given, and needs a number");
  await refused("berger", { block: "long", reps: 1, settle: 1 }, 'ValueError: block needs a number, not "long"');
  await refused("berger", { block: 1, reps: "8.5", settle: 1 }, 'ValueError: reps needs a whole number, not "8.5"');
  await refused("berger", { block: 1, reps: 1, settle: null }, "ValueError: settle is not given, and needs a number");
  await refused("protocol", { steps: "Close | 1", reps: 1, settle: 1, lead_in: [6] }, "ValueError: lead_in needs a number, not [6]");
  await refused("protocol", { steps: "Close | 1\n\nOpen | twenty", reps: 1 }, 'ValueError: line 3 of steps needs a number of seconds after its "|": "Open | twenty"');
  await refused("protocol", { reps: 1 }, "ValueError: no steps");
  await refused("saccade", { trials: 2, hold_s: 0.1, center_s: "x", include_sham: 0 }, 'ValueError: center_s needs a number, not "x"');
  await refused("head_training", { pair: "colors" }, 'ValueError: pair needs one of eyes, jaw, mental or own, not "colors"');
  await refused("head_training", { pair: "own", states: "Focus | Look at the dot.\n\nRest" }, 'ValueError: line 3 of states needs "name | cue words": "Rest"');
  await refused("head_training", { pair: "own", states: "Rest | Rest now.\nFocus | Look at the dot.\nREST | Rest again." }, 'ValueError: line 3 of states names "REST", as line 1 does, and each state needs a name of its own');
  // A whole number that is not finite is refused by name, as the library
  // refuses one, as JavaScript writes it.
  await refused("blink", { trials: Infinity }, "ValueError: trials needs a whole number, not Infinity");
  await refused("blink", { trials: NaN }, "ValueError: trials needs a whole number, not NaN");
});

test("a rep_length below 1 is refused before anything is sent, saying it needs at least 1, a window kept from each rep, before fewer reps than states are, as the library refuses it", async () => {
  // A rep keeps up to rep_length windows, so a rep_length of 0 or less would
  // go through every rep and keep nothing. A fraction is cut to its whole
  // part first, and a rep_length of 1 is taken.
  const ctx = new RunCtx(() => undefined);
  const notALink = {} as Parameters<NonNullable<ReturnType<typeof REGISTRY.get>>["fn"]>[0];
  const refused = (params: Params, why: string): Promise<void> =>
    assert.rejects(REGISTRY.get("head_training")!.fn(notALink, () => undefined, params, ctx), (e: Error) => e instanceof P.Invalid && e.message === why);
  for (const [given, said] of [[0, 0], [-2, -2], [0.5, 0], [-(10n ** 5000n), "a whole number of 5001 digits"]] as const) {
    await refused({ pair: "eyes", rep_length: given, reps: 2 }, `ValueError: rep_length needs at least 1, a window kept from each rep, not ${said}`);
    await refused({ pair: "eyes", rep_length: given, reps: 1 }, `ValueError: rep_length needs at least 1, a window kept from each rep, not ${said}`);
  }
  await refused({ pair: "eyes", rep_length: 1, reps: 1 }, "ValueError: reps needs at least 2, one for each state, not 1");
  // A count of more digits than Python prints is named by its digits.
  await refused({ pair: "eyes", rep_length: 12, reps: -(10n ** 5000n) }, "ValueError: reps needs at least 2, one for each state, not a whole number of 5001 digits");
});

test("a parameter given as a whole number of more digits than Python prints, or as text written as one, is refused by name and how many digits it has, as the library refuses it, and text of 4,300 digits is still read as a whole number", async () => {
  // Python turns no whole number of more than 4,300 digits into text, and
  // reads none from text, so the library names each by its digits, never
  // printing it: a bigint here, and text of as many digits, its sign,
  // white space and underscores aside. Text of as many digits is refused as
  // a whole number however small it is, and taken as a number, which
  // Python's float reads at any length.
  const ctx = new RunCtx(() => undefined);
  const notALink = {} as Parameters<NonNullable<ReturnType<typeof REGISTRY.get>>["fn"]>[0];
  const refused = (name: string, params: Params, why: string): Promise<void> =>
    assert.rejects(REGISTRY.get(name)!.fn(notALink, () => undefined, params, ctx), (e: Error) => e instanceof P.Invalid && e.message === why);
  for (const [value, digits] of [["1".repeat(5000), 5000], [` -${"9".repeat(4301)}\n`, 4301]] as const) {
    await refused("baseline", { seconds: value }, `ValueError: seconds needs a number, not text of ${digits} digits`);
    await refused("berger", { block: 1, reps: value, settle: 1 }, `ValueError: reps needs a whole number, not text of ${digits} digits`);
  }
  await refused("berger", { block: 1, reps: `${"0".repeat(4999)}8`, settle: 1 }, "ValueError: reps needs a whole number, not text of 5000 digits");
  for (const [value, digits] of [[10n ** 5000n, 5001], [-(10n ** 5000n), 5001], [10n ** 4300n, 4301]] as const) {
    await refused("baseline", { seconds: value }, `ValueError: seconds needs a number, not a whole number of ${digits} digits`);
  }
  await refused("baseline", { seconds: [10n ** 5000n] }, "ValueError: seconds needs a number, not a list of more digits than Python prints");
  await refused("head_training", { pair: "eyes", rep_length: `-${"9".repeat(4300)}`, reps: 2 }, `ValueError: rep_length needs at least 1, a window kept from each rep, not -${"9".repeat(4300)}`);
});

/** An IntoMind device of firmware 1.4.6, never connected, as the library's tests make one, for what names it, says its id and says its firmware. */
const TEST_DEVICE = {
  device: { name: "IntoMind-TEST" },
  info: { deviceId: "00000000c0de0001", firmware: [1, 4, 6] },
  session: { requireInfo: () => ({ deviceId: "00000000c0de0001" }) },
} as unknown as Connection;

test("a setup gain too large for a float, or text written as a whole number of more digits than Python reads, is refused as a gain other than the gain in force, named by how many digits it has, as the library refuses it", () => {
  for (const [gain, shown] of [
    [10n ** 400n, "a whole number of 401 digits"],
    [10n ** 5000n, "a whole number of 5001 digits"],
    ["1".repeat(5000), "text of 5000 digits"],
  ] as const) {
    assert.throws(
      () => gainsAsked([TEST_DEVICE], [J.fromPlain({ gain })], new Map([[TEST_DEVICE, 24]])),
      (e: Error) =>
        e instanceof P.Invalid &&
        e.message ===
          `ValueError: the setup for IntoMind-TEST (00000000c0de0001) gives a gain of ${shown}, and the gain in force on it is 24, so the run is refused: its recording would carry a gain of 24 beside a setup that gives ${shown}.`,
      shown,
    );
  }
});

test("a setup gain no recording's name can hold is refused by how many digits it has as the recording is named, one device's name in a run naming that device, as the library refuses it", () => {
  for (const [gain, shown] of [[10n ** 5000n, "a whole number of 5001 digits"], ["1".repeat(5000), "text of 5000 digits"]] as const) {
    for (const [whose, name] of [
      ["the setup", () => experimentsEntry.label(new RunCtx(() => {}, { gain, bias: "on" }), "x")],
      ["the setup for IntoMind-TEST (00000000c0de0001)", () => tagged("x_on_g24_20260101-000000", TEST_DEVICE, J.fromPlain({ gain }))],
    ] as const) {
      assert.throws(name, (e: Error) => e instanceof P.Invalid && e.message === `ValueError: ${whose} gives a gain of ${shown}, too long for a recording's name, which carries the gain`, whose);
    }
  }
  assert.match(experimentsEntry.label(new RunCtx(() => {}, { gain: 24, bias: "on" }), "x"), /^x_on_g24_\d{8}-\d{6}$/);
  assert.equal(tagged("x_on_g24_20260101-000000", TEST_DEVICE, J.fromPlain({ gain: 12 })), "x_on_g12_20260101-000000_00000000c0de0001");
  assert.equal(tagged("x_on_g24_20260101-000000", TEST_DEVICE, J.object([])), "x_on_gna_20260101-000000_00000000c0de0001");
});

test("a number that is not finite is refused by name, for a parameter and for a step's seconds, in the words that refuse any other value that is not a number, and a finite number is taken", async () => {
  const spec = REGISTRY.get("baseline")!;
  const ctx = new RunCtx(() => undefined);
  const notALink = {} as Parameters<typeof spec.fn>[0];
  const refused = (name: string, params: Params, why: string): Promise<void> =>
    assert.rejects(REGISTRY.get(name)!.fn(notALink, () => undefined, params, ctx), (e: Error) => e instanceof P.Invalid && e.message === why);
  const notFinite: [unknown, string][] = [
    ["nan", '"nan"'],
    ["inf", '"inf"'],
    ["-inf", '"-inf"'],
    ["Infinity", '"Infinity"'],
    [" NaN ", '" NaN "'],
    [Infinity, "Infinity"],
    [-Infinity, "-Infinity"],
    [NaN, "NaN"],
    // A whole number too large for a float is one that is not finite here.
    [10n ** 400n, (10n ** 400n).toString()],
  ];
  // Nothing sounds and nothing waits, should a value get past its check.
  runtime.play = () => undefined;
  runtime.time = { ...HOST_TIME, sleep: () => Promise.resolve() };
  try {
    for (const [value, shown] of notFinite) {
      await refused("baseline", { seconds: value }, `ValueError: seconds needs a number, not ${shown}`);
    }
    // A finite number before the one refused is taken, as text or as a number.
    await refused("blink", { trials: 2, include_sham: 0, rest_s: "-1e300", hold_s: NaN }, "ValueError: hold_s needs a number, not NaN");
    await refused("protocol", { steps: "Close your eyes | 2.5", reps: 1, settle: "inf" }, 'ValueError: settle needs a number, not "inf"');
    await refused("protocol", { steps: "Close your eyes | -inf", reps: 1 }, 'ValueError: line 1 of steps needs a number of seconds after its "|": "Close your eyes | -inf"');
    await refused("protocol", { steps: "Close your eyes | 1\nOpen | nan", reps: 1 }, 'ValueError: line 2 of steps needs a number of seconds after its "|": "Open | nan"');
  } finally {
    restore();
  }
});

test("a failure while building the manifest fields still stops the streams and frees the devices", async () => {
  const { link } = await fakeLink(items(J.get(fixture, "runs"))[0]!);
  let clock = 1_767_225_600;
  runtime.time = {
    monotonic: () => clock,
    time: () => clock,
    sleep: async (seconds) => {
      clock += seconds;
    },
  };
  runtime.play = () => undefined;
  const saved: Saved[] = [];
  const emit = (m: Message): void => {
    if (m.type === "saved") saved.push(m.recording as Saved);
  };
  // Parameters whose note cannot be read, so that building the fields a
  // recording's manifest adds fails at the end of the step, after the
  // streams the step started are stopped, inside the cleanup that lets its
  // devices go. Berger reads no note, so it is not trapped this way.
  const trapped = (given: Record<string, unknown>): Params => {
    const target: Record<string, unknown> = { ...given, note: "" };
    return new Proxy(target, {
      get(t, k) {
        if (k === "note") throw new Error("the note could not be read");
        return t[k as string];
      },
    });
  };
  const steps: [string, Record<string, unknown>][] = [
    ["record", { markers: "a", max_seconds: 0.5 }],
    ["protocol", { steps: "One | 0.5", reps: 1, settle: 1, lead_in: 0 }],
    ["blink", { trials: 1, hold_s: 0.1, rest_s: 0.1, include_sham: 0 }],
    ["jaw_clench", { trials: 1, hold_s: 0.1, rest_s: 0.1, include_sham: 0 }],
    ["saccade", { trials: 1, hold_s: 0.1, center_s: 0.1, include_sham: 0 }],
  ];
  try {
    for (const [name, given] of steps) {
      await assert.rejects(REGISTRY.get(name)!.fn(link, emit, trapped(given), new RunCtx(emit, { gain: 24 })), { message: "the note could not be read" });
      assert.equal(link.session.streaming, false, `${name} left the stream running`);
      assert.equal(marked(link), false, `${name} left the device held`);
    }
    assert.equal(saved.length, 0, "nothing was saved");
    const names = await REGISTRY.get("baseline")!.fn(link, emit, { seconds: 0.5 }, new RunCtx(emit, { gain: 24 }));
    assert.equal(names.length, 1, "the next step records the same device");
    assert.equal(saved.length, 1);
    assert.ok(saved[0]!.finalized);
  } finally {
    restore();
  }
});

test("every recording a run saves through a connection names the connection as its adapter", async () => {
  assert.deepEqual({ ...Connection.ADAPTER }, { name: "intomind.web-bluetooth", version: "1", first_party: true, transport: "bluetooth-le" });
  assert.ok(Object.isFrozen(Connection.ADAPTER));
  const { link } = await fakeLink(items(J.get(fixture, "runs"))[0]!);
  let clock = 1_767_225_600;
  runtime.time = {
    monotonic: () => clock,
    time: () => clock,
    sleep: async (seconds) => {
      clock += seconds;
    },
  };
  runtime.play = () => undefined;
  const saved: Saved[] = [];
  try {
    const emit = (m: Message): void => {
      if (m.type === "saved") saved.push(m.recording as Saved);
    };
    await REGISTRY.get("baseline")!.fn(link, emit, { seconds: 0.5 }, new RunCtx(emit));
  } finally {
    restore();
  }
  assert.equal(saved.length, 1);
  assert.equal(
    J.dumps(J.field(J.field(saved[0]!.manifest, "signal_chain"), "adapter")),
    '{"name": "intomind.web-bluetooth", "version": "1", "first_party": true, "transport": "bluetooth-le"}',
  );
});

test("several devices handed to an experiment outside a run are refused before anything is sent", async () => {
  const r = items(J.get(fixture, "runs"))[0]!;
  const writes: Uint8Array[] = [];
  const [a, b] = [await fakeLink(r, { onWrite: (w) => writes.push(w) }), await fakeLink(r, { onWrite: (w) => writes.push(w) })];
  const heard: Message[] = [];
  runtime.time = { monotonic: () => 1, time: () => 1_767_225_600, sleep: () => Promise.resolve() };
  runtime.play = () => undefined;
  try {
    await assert.rejects(
      REGISTRY.get("berger")!.fn([a.link, b.link], (m) => {
        heard.push(m);
      }, { block: 1, reps: 1, settle: 1 }, new RunCtx(() => undefined)),
      (e: Error) =>
        e instanceof P.Invalid &&
        e.message ===
          "ValueError: a session's recorder holds one device's samples: several devices are recorded by DaisyChain.run, which keeps a recorder for each",
    );
  } finally {
    restore();
  }
  assert.deepEqual(writes, [], "nothing was sent to either device");
  assert.deepEqual(heard, []);
});

test("a recording saved with no adapter declared says the adapter is unknown", () => {
  const session = new Session();
  session.onDeviceInfo(P.fromHex(text(J.field(items(J.get(fixture, "runs"))[0]!, "info"))));
  session.armRecorder();
  const saved = session.saveRecording("no-adapter-declared");
  assert.equal(
    J.dumps(J.field(J.field(saved.manifest, "signal_chain"), "adapter")),
    '{"name": "unknown", "version": null, "first_party": false, "transport": null}',
  );
});

test("an experiment of one's own, built from the cued trials and a recording's name as a Python user builds one from the library's, is registered and run through the public entry alone", async () => {
  const { capture, experiments } = sdk;
  assert.equal(typeof experiments.cuedTrials, "function", "the cued trials are reached from the package");
  assert.equal(typeof experiments.label, "function", "a recording's name is reached from the package");
  assert.equal(experiments.cuedTrials, experimentsEntry.cuedTrials, "the package's experiments entry gives the same cued trials");
  assert.equal(experiments.label, experimentsEntry.label, "the package's experiments entry gives the same name");
  const { link } = await fakeLink(items(J.get(fixture, "runs"))[0]!);
  experiments.experiment(
    "my_protocol",
    "One line describing what it measures.",
    [
      { name: "trials", type: "number", default: 20 },
      { name: "hold_s", type: "number", default: 2.0 },
      { name: "note", type: "text", default: "" },
      ...experiments.TONE_PARAMS,
    ],
    async (dev, emit, p, ctx) => {
      const conds = Array.from({ length: Math.floor(Number(p.trials) / 2) }, () => ["go", "sham"]).flat();
      return experiments.cuedTrials(dev, emit, p, ctx, {
        label: experiments.label(ctx, "my"),
        conds,
        texts: { go: "Do the thing", sham: "Stay still" },
        restText: "Rest",
        holdS: Number(p.hold_s),
        restS: 0.05,
        leadIn: 0,
        intro: "My protocol. Do the thing when told, rest when told.",
        extra: capture.fromPlain({ experiment: "my_protocol" }),
      });
    },
    { prereg: { name: "my_protocol", window_s: 1.5, min_ratio: 5.5 } },
  );
  const cues: string[] = [];
  const saved: Saved[] = [];
  const emit = (m: Message): void => {
    if (m.type === "cue") cues.push(String(m.text));
    if (m.type === "saved") saved.push(m.recording as Saved);
  };
  try {
    const spec = experiments.REGISTRY.get("my_protocol")!;
    assert.equal(capture.dumps(spec.prereg), '{"name": "my_protocol", "window_s": 1.5, "min_ratio": 5.5}');
    const ctx = new experiments.RunCtx(emit);
    const names = await spec.fn(link, emit, { trials: 4, hold_s: 0.05, note: "desk" }, ctx);
    assert.deepEqual(cues, [
      "My protocol. Do the thing when told, rest when told.  starting in 0 s",
      "Rest  (1/4)",
      "Do the thing  (1/4)",
      "Rest  (2/4)",
      "Stay still  (2/4)",
      "Rest  (3/4)",
      "Do the thing  (3/4)",
      "Rest  (4/4)",
      "Stay still  (4/4)",
      "Done",
    ]);
    assert.equal(saved.length, 1);
    assert.deepEqual(names, [saved[0]!.label]);
    assert.match(saved[0]!.label, /^my_na_gna_\d{8}-\d{6}$/, "named as the library names it, with no bias drive or gain in the setup");
    const meta = saved[0]!.manifest;
    assert.equal(capture.dumps(J.field(meta, "experiment")), '"my_protocol"');
    assert.equal(capture.dumps(J.field(meta, "order")), '["go", "sham", "go", "sham"]');
    assert.equal(capture.dumps(J.field(meta, "hold_s")), "0.05");
    assert.equal(capture.dumps(J.field(meta, "rest_s")), "0.05");
    assert.equal(capture.dumps(J.field(meta, "note")), '"desk"');
    assert.deepEqual(
      ctx.events.map((e) => capture.toPlain(e) as { cond: string; kind: string; trial: number }).map((e) => [e.cond, e.kind, e.trial]),
      [
        ["go", "cue", 0],
        ["sham", "cue", 1],
        ["go", "cue", 2],
        ["sham", "cue", 3],
      ],
    );
    assert.equal(link.session.streaming, false, "the stream it started is stopped");
  } finally {
    experiments.REGISTRY.delete("my_protocol");
  }
});
