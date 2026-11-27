/**
 * The browser connection held to what the Python library's `Device` does
 * with its control link, its deliveries and its embeddings, on devices made
 * of nothing that answer when a test says they answer: exchanges that take
 * turns, an answer that comes after its exchange stopped waiting, a
 * listener that throws, USB power ending a stream, the same connection
 * connected again after its link dropped, from either side, behind a browser
 * that keeps subscriptions as Chrome keeps them (see `browser.ts`), and
 * embeddings left as they were found by everything the package turns them on
 * for.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test, { type TestContext } from "node:test";

import { hostClocks } from "../src/capture/clock.ts";
import { DaisyChain } from "../src/daisy-chain.ts";
import { crc32 } from "../src/digest.ts";
import * as P from "../src/protocol.ts";
import { RunCtx } from "../src/experiments.ts";
import { windowsOf } from "../src/run.ts";
import { SessionError, Unanswered, type Event } from "../src/session.ts";
import { own, tap } from "../src/taps.ts";
import { LONGEST_MS } from "../src/timers.ts";
import { ChecksumMismatch, OffsetMismatch, Refused, Rejected, Transfer } from "../src/transfer.ts";
import { connect, requestAndConnect, type Connection } from "../src/web-bluetooth.ts";
import { Browser } from "./browser.ts";

const VECTORS = JSON.parse(readFileSync(new URL("../../contract/conformance.json", import.meta.url), "utf8"));

function vector(name: string): Uint8Array {
  return P.fromHex(VECTORS.decode.find((v: { name: string }) => v.name.startsWith(name)).bytes);
}

const INFO = vector("device info, the IntoMind One");
const STATUS = P.fromHex(VECTORS.decode.find((v: { kind: string }) => v.kind === "status").bytes);
const READY = vector("a model that is ready, with a head selected");
const OP = P.OPCODES;

class FakeCharacteristic extends EventTarget {
  readonly uuid: string;
  value?: DataView;
  onWrite?: (bytes: Uint8Array) => void;

  constructor(uuid: string, value?: Uint8Array) {
    super();
    this.uuid = uuid;
    if (value !== undefined) this.value = new DataView(value.slice().buffer);
  }

  readValue(): Promise<DataView> {
    assert.ok(this.value !== undefined, `${this.uuid} has nothing to read`);
    return Promise.resolve(this.value);
  }

  writeValueWithResponse(value: BufferSource): Promise<void> {
    const bytes = new Uint8Array((value as Uint8Array).buffer, (value as Uint8Array).byteOffset, (value as Uint8Array).byteLength).slice();
    this.onWrite?.(bytes);
    return Promise.resolve();
  }

  writeValueWithoutResponse(value: BufferSource): Promise<void> {
    return this.writeValueWithResponse(value);
  }

  notify(bytes: Uint8Array): void {
    this.value = new DataView(bytes.slice().buffer);
    this.dispatchEvent(new Event("characteristicvaluechanged"));
  }
}

/**
 * A device whose every answer the test decides: by opcode, the next of
 * `answers` when there is a list, after the next of `delays` in
 * milliseconds, never when that is null, and with no delay once the write
 * has returned, since the device acknowledges a write first and answers it
 * afterwards. Every write and answer is logged in the order it happens.
 */
function busyDevice(answers: Map<number, Uint8Array[]> = new Map(), delays: Map<number, (number | null)[]> = new Map()) {
  const characteristics = new Map<string, FakeCharacteristic>([
    [P.DEVICE_INFO, new FakeCharacteristic(P.DEVICE_INFO, INFO)],
    [P.CONTROL, new FakeCharacteristic(P.CONTROL)],
    [P.CONTROL_RESPONSE, new FakeCharacteristic(P.CONTROL_RESPONSE)],
    [P.EEG_DATA, new FakeCharacteristic(P.EEG_DATA)],
    [P.STATUS, new FakeCharacteristic(P.STATUS)],
    [P.EMBEDDINGS, new FakeCharacteristic(P.EMBEDDINGS)],
  ]);
  const log: [string, number][] = [];
  const writes: Uint8Array[] = [];
  const respond = characteristics.get(P.CONTROL_RESPONSE)!;
  characteristics.get(P.CONTROL)!.onWrite = (bytes) => {
    const op = bytes[0]!;
    writes.push(bytes);
    log.push(["write", op]);
    const queued = answers.get(op);
    const answer = queued !== undefined && queued.length > 0 ? queued.shift()! : Uint8Array.of(op, 0);
    const waits = delays.get(op);
    const delay = waits !== undefined && waits.length > 0 ? waits.shift()! : 0;
    if (delay === null) return;
    const send = () => {
      log.push(["answer", answer[0]!]);
      respond.notify(answer);
    };
    if (delay === 0) setImmediate(send);
    else setTimeout(send, delay);
  };
  const browser = new Browser(characteristics, { id: "busy", name: "IntoMind One" });
  return {
    device: browser.device,
    browser,
    characteristics,
    log,
    writes,
    connects: () => browser.links,
    /** The device drops the link, as one that restarts, is switched off or goes out of range does. */
    drop() {
      browser.drop();
    },
  };
}

function battery(mv: number): Uint8Array {
  const b = new Uint8Array(6);
  b[0] = OP.get_battery;
  new DataView(b.buffer).setUint16(2, mv, true);
  b[4] = 61;
  b[5] = 1;
  return b;
}

function millivolts(r: P.Response): number {
  return new DataView(r.payload.buffer, r.payload.byteOffset, r.payload.byteLength).getUint16(0, true);
}

const BATTERY = { characteristic: P.FILL.CONTROL, bytes: Uint8Array.of(OP.get_battery), withResponse: true } as const;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

test("two exchanges at once on one connection each take their own answer, in turn", async () => {
  const d = busyDevice(new Map([[OP.get_battery, [battery(3712)]]]), new Map([[OP.get_battery, [30]], [OP.get_boot_info, [5]]]));
  const link = await connect(d.device);
  const boot = { characteristic: P.FILL.CONTROL, bytes: Uint8Array.of(OP.get_boot_info), withResponse: true };
  const [b, bi] = await Promise.all([link.request(BATTERY), link.request(boot)]);
  assert.equal(millivolts(b), 3712);
  assert.equal(bi.opcode, OP.get_boot_info);
  assert.deepEqual(d.log, [
    ["write", OP.get_battery],
    ["answer", OP.get_battery],
    ["write", OP.get_boot_info],
    ["answer", OP.get_boot_info],
  ]);

  const same = busyDevice(new Map([[OP.get_battery, [battery(3712), battery(3800)]]]), new Map([[OP.get_battery, [30, 5]]]));
  const again = await connect(same.device);
  const [first, second] = await Promise.all([again.request(BATTERY), again.request(BATTERY)]);
  assert.deepEqual([millivolts(first), millivolts(second)], [3712, 3800], "the same opcode, each its own answer");
});

test("an answer that arrives after its exchange stopped waiting is never taken for a later one", async () => {
  // Late while the next exchange of the same opcode waits for its own, which
  // is taken the moment it arrives, the late one dropped.
  let d = busyDevice(new Map([[OP.get_battery, [battery(3111), battery(3222)]]]), new Map([[OP.get_battery, [200, 100]]]));
  let link = await connect(d.device, { timeoutMs: 150 });
  await assert.rejects(link.request(BATTERY), /did not answer opcode 0x/);
  const t0 = performance.now();
  assert.equal(millivolts(await link.request(BATTERY)), 3222, "the late answer was taken for the next one's");
  assert.ok(performance.now() - t0 < 140, `the second answer took ${performance.now() - t0} ms to be taken`);

  // Two exchanges gave up, and both late answers come before the third
  // exchange's own: neither is taken for it.
  d = busyDevice(
    new Map([[OP.get_battery, [battery(3111), battery(3222), battery(3333)]]]),
    new Map([[OP.get_battery, [440, 260, 120]]]),
  );
  link = await connect(d.device, { timeoutMs: 200 });
  for (let k = 0; k < 2; k++) await assert.rejects(link.request(BATTERY));
  assert.equal(millivolts(await link.request(BATTERY)), 3333, "a late answer was taken for the third one's");

  // Late, and arriving while no exchange waits, before the next begins.
  d = busyDevice(new Map([[OP.get_battery, [battery(3111), battery(3222)]]]), new Map([[OP.get_battery, [30, 0]]]));
  link = await connect(d.device, { timeoutMs: 10 });
  await assert.rejects(link.request(BATTERY));
  await sleep(50);
  assert.equal(millivolts(await link.request(BATTERY)), 3222, "a stale answer was taken");
});

test("an exchange never answered costs the next of its opcode its timeout and no more, a refusal included", async () => {
  for (const second of [battery(3222), Uint8Array.of(OP.get_battery, 1)]) {
    const d = busyDevice(new Map([[OP.get_battery, [battery(3111), second, battery(3333)]]]), new Map([[OP.get_battery, [null, 0, 0]]]));
    const link = await connect(d.device, { timeoutMs: 50 });
    await assert.rejects(link.request(BATTERY));
    let t0 = performance.now();
    const got = await link.request(BATTERY);
    const waited = performance.now() - t0;
    if (second[1] === 0) assert.equal(millivolts(got), 3222);
    else assert.equal(got.ok, false, "the refusal held is the exchange's own");
    assert.ok(waited >= 45, `taken after ${waited} ms, before its timeout said nothing followed`);
    t0 = performance.now();
    const third = await link.request(BATTERY);
    assert.equal(millivolts(third), 3333);
    assert.ok(performance.now() - t0 < 40, "the next exchange was answered at once");
  }
});

test("a clock exchange never counts a wait for its turn, and is dated by its answer's arrival when it was held", async () => {
  const ticks = new Uint8Array(8);
  new DataView(ticks.buffer).setBigUint64(0, 5_000_000n, true);
  const sync = P.encodeResponse(OP.time_sync, 0, ticks);
  const d = busyDevice(new Map([[OP.time_sync, [sync, sync, sync]]]), new Map([[OP.get_battery, [60]], [OP.time_sync, [0, null, 0]]]));
  const link = await connect(d.device, { timeoutMs: 150 });
  await Promise.all([link.request(BATTERY), link.timeSync()]);
  const clock = link.session.clock!;
  assert.ok(clock.syncs.at(-1)!.rtt < 0.03, `a round trip of ${clock.syncs.at(-1)!.rtt} s counted the wait`);
  // Never answered, so the next is held until its deadline says it was its own.
  await assert.rejects(link.timeSync());
  const t0 = performance.now();
  const before = hostClocks();
  await link.timeSync();
  assert.ok(performance.now() - t0 >= 140, "the held answer was taken at the deadline");
  const s = clock.syncs.at(-1)!;
  assert.ok(s.rtt < 0.05, `dated by the wait, ${s.rtt} s, not by the answer's arrival`);
  assert.ok(s.mono - before.monotonic < 0.05);
  assert.equal(s.device, 5.0);
});

test("answers owed on a link are forgotten when it goes, and the same connection connects again with its listeners", async () => {
  const d = busyDevice(new Map([[OP.get_battery, [battery(3111), battery(3222)]]]), new Map([[OP.get_battery, [null, 0]]]));
  let drops = 0;
  const link = await connect(d.device, { timeoutMs: 40, onDisconnect: () => drops++ });
  const heard: Event["type"][] = [];
  link.on((e) => heard.push(e.type));
  await assert.rejects(link.request(BATTERY));
  d.drop();
  assert.equal(link.connected, false);
  await assert.rejects(link.request(BATTERY), (e: Error) => e instanceof SessionError && e.message === "ConnectionError: BLE link is down");
  assert.equal(await link.connect(), link, "the same connection, connected again");
  assert.equal(link.connected, true);
  assert.equal(d.connects(), 2);
  const t0 = performance.now();
  assert.equal(millivolts(await link.request(BATTERY)), 3222, "nothing owed on the link that went");
  assert.ok(performance.now() - t0 < 30);
  d.characteristics.get(P.STATUS)!.notify(STATUS);
  assert.ok(heard.includes("status"), "its listeners still hear the device");
  d.drop();
  await link.connect();
  d.drop();
  assert.equal(drops, 3, "every drop is reported, not only the first");
});

test("a program that connects to a device again after each drop hears each drop once, by its newest connection", async () => {
  const d = busyDevice();
  const heard = [0, 0, 0];
  await connect(d.device, { onDisconnect: () => heard[0]!++ });
  d.drop();
  await connect(d.device, { onDisconnect: () => heard[1]!++ });
  d.drop();
  await connect(d.device, { onDisconnect: () => heard[2]!++ });
  const before = [...heard];
  d.drop();
  assert.deepEqual(heard.map((n, i) => n - before[i]!), [0, 0, 1], "the last drop is heard by the newest connection, once, and by no earlier one");
  assert.deepEqual(heard, [1, 1, 1], "each connection heard the drop of its own link, and no later one");
});

test("a connection that connects itself again hears each drop of its links once, and the drop its own disconnect makes", async () => {
  const d = busyDevice();
  let drops = 0;
  const link = await connect(d.device, { onDisconnect: () => drops++ });
  d.drop();
  assert.equal(drops, 1);
  await link.connect();
  d.drop();
  assert.equal(drops, 2);
  await link.connect();
  d.drop();
  assert.equal(drops, 3);
  await link.connect();
  await link.disconnect();
  assert.equal(drops, 4, "its own disconnect is a drop it hears");
});

test("disconnect takes the stream as stopped, as the library's disconnect does, on a link that is up and on one the device already dropped, and writes nothing for it", async () => {
  const d = busyDevice();
  const link = await connect(d.device);
  await link.start();
  assert.equal(link.session.streaming, true);
  const written = d.writes.length;
  await link.disconnect();
  assert.equal(link.session.streaming, false, "the session takes the stream as stopped");
  assert.equal(d.writes.length, written, "nothing was written to stop it");
  await link.connect();
  await link.start();
  assert.equal(link.session.streaming, true);
  d.drop();
  await link.disconnect();
  assert.equal(link.session.streaming, false, "the session takes the stream as stopped once a dropped link is let go");
});

test("an answer that does not come in time throws Unanswered, to a control request, a clock exchange and a setting alike, its message as it was", async () => {
  const d = busyDevice(new Map(), new Map([[OP.get_battery, [null]], [OP.time_sync, [null]], [OP.set_gain, [null]]]));
  const link = await connect(d.device, { timeoutMs: 30 });
  const unanswered = (op: number) => (e: unknown) =>
    e instanceof Unanswered && !(e instanceof SessionError) && e.name === "Unanswered" && e.message === `the device did not answer opcode 0x${op.toString(16)} in time`;
  await assert.rejects(link.request(BATTERY), unanswered(OP.get_battery));
  await assert.rejects(link.timeSync(), unanswered(OP.time_sync));
  await assert.rejects(link.setGain(24), unanswered(OP.set_gain));
});

test("a control request given a wait longer than a timer keeps to, or an infinite one, waits for its answer and does not give up at once", async () => {
  for (const timeoutMs of [LONGEST_MS + 5_000, Infinity]) {
    const d = busyDevice(new Map([[OP.get_battery, [battery(3712)]]]), new Map([[OP.get_battery, [30]]]));
    const link = await connect(d.device, { timeoutMs });
    assert.equal(millivolts(await link.request(BATTERY)), 3712, `a wait of ${timeoutMs} ms`);
  }
});

test("a wait for control requests that is not a number is refused by connect before any link is opened, in words naming timeoutMs and what it was given", async () => {
  for (const [timeoutMs, shown] of [["soon", '"soon"'], ["5000", '"5000"'], [NaN, "NaN"], [{}, "{}"]] as const) {
    const d = busyDevice();
    await assert.rejects(connect(d.device, { timeoutMs: timeoutMs as unknown as number }), (e: unknown) => {
      assert.ok(e instanceof TypeError, `${shown}: ${String(e)}`);
      assert.equal(e.message, `timeoutMs is a number of milliseconds, and was given ${shown}`);
      return true;
    });
    assert.equal(d.connects(), 0, `${shown}: a link was opened`);
  }
  // Refused before the browser's chooser is put up.
  await assert.rejects(requestAndConnect({ timeoutMs: "soon" as unknown as number }), {
    name: "TypeError",
    message: 'timeoutMs is a number of milliseconds, and was given "soon"',
  });
  // Not given, or null, is the five seconds it was.
  for (const options of [{}, { timeoutMs: null as unknown as number }]) {
    const d = busyDevice(new Map([[OP.get_battery, [battery(3712)]]]));
    const link = await connect(d.device, options);
    assert.equal(millivolts(await link.request(BATTERY)), 3712);
  }
});

test("a clock calibration waits the whole gap it is given between its exchanges, a gap longer than a timer keeps to included, and an infinite gap waits", async (t) => {
  const ticks = new Uint8Array(8);
  new DataView(ticks.buffer).setBigUint64(0, 5_000_000n, true);
  const sync = P.encodeResponse(OP.time_sync, 0, ticks);
  const d = busyDevice(new Map([[OP.time_sync, [sync, sync, sync, sync]]]));
  const link = await connect(d.device);
  const exchanges = (): number => d.writes.filter((w) => w[0] === OP.time_sync).length;
  const turn = (): Promise<void> => new Promise((resolve) => setImmediate(resolve));
  t.mock.timers.enable({ apis: ["setTimeout"] });
  try {
    const done = link.calibrateClock(2, LONGEST_MS + 5_000);
    await turn();
    assert.equal(exchanges(), 1, "the first exchange goes at once");
    t.mock.timers.tick(LONGEST_MS);
    await turn();
    assert.equal(exchanges(), 1, "the second exchange went before the gap had passed");
    t.mock.timers.tick(5_000);
    await done;
    assert.equal(exchanges(), 2, "the second exchange goes once the gap has passed");

    // Never settles: its gap never passes.
    void link.calibrateClock(2, Infinity);
    await turn();
    assert.equal(exchanges(), 3);
    t.mock.timers.tick(LONGEST_MS);
    await turn();
    assert.equal(exchanges(), 3, "an infinite gap ended");
  } finally {
    t.mock.timers.reset();
  }
});

test("a clock calibration given a gap that is not a number takes no gap between its exchanges", async (t) => {
  const ticks = new Uint8Array(8);
  new DataView(ticks.buffer).setBigUint64(0, 5_000_000n, true);
  const sync = P.encodeResponse(OP.time_sync, 0, ticks);
  const d = busyDevice(new Map([[OP.time_sync, [sync, sync, sync, sync]]]));
  const link = await connect(d.device);
  const exchanges = (): number => d.writes.filter((w) => w[0] === OP.time_sync).length;
  const turn = (): Promise<void> => new Promise((resolve) => setImmediate(resolve));
  t.mock.timers.enable({ apis: ["setTimeout"] });
  try {
    for (const gapMs of [NaN, "soon" as unknown as number]) {
      const before = exchanges();
      const done = link.calibrateClock(2, gapMs);
      await turn();
      t.mock.timers.tick(0);
      await turn();
      assert.equal(exchanges() - before, 2, `a gap of ${String(gapMs)} was waited for`);
      await done;
    }
  } finally {
    t.mock.timers.reset();
  }
});

/**
 * Send what the device sends in answer to an update request once the write
 * that carried the request has returned. The device acknowledges the write
 * first, and only then handles the request and answers it.
 */
function afterTheWrite(answer: () => void): void {
  setTimeout(answer, 0);
}

/**
 * A device whose update service answers by `answer`, the body of its answer
 * to each operation or null for none at all, and turns away the write of
 * each operation `refuse` names, as the browser turns away a write that
 * fails. It answers once the write has returned, as a device does. Every
 * write to it is listed by its operation.
 */
function updatingDevice(answer: (op: number) => Uint8Array | null, refuse: (op: number) => boolean = () => false) {
  const d = busyDevice();
  const control = new FakeCharacteristic(P.UPDATE_CONTROL);
  d.characteristics.set(P.UPDATE_CONTROL, control);
  d.characteristics.set(P.UPDATE_DATA, new FakeCharacteristic(P.UPDATE_DATA));
  const asked: number[] = [];
  control.onWrite = (bytes) => {
    const op = bytes[0]!;
    asked.push(op);
    if (refuse(op)) throw new DOMException("GATT operation failed for unknown reason.", "NetworkError");
    const body = answer(op);
    if (body !== null) afterTheWrite(() => control.notify(P.encodeUpdateResponse(op, 0, body)));
  };
  return { ...d, asked };
}

/** The update service's answer to a start: chunks of 100 bytes. */
function started(op: number): Uint8Array | null {
  if (op !== P.UPDATE_OPS.start) return null;
  const body = new Uint8Array(7);
  new DataView(body.buffer).setUint16(1, 100, true);
  return body;
}

/** What `f` settles with, its value or what it threw, on mocked timers moved a tenth of a second at a time, up to `limitMs`. */
async function onMockedTime(t: TestContext, f: () => Promise<unknown>, limitMs: number): Promise<unknown> {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  try {
    let settled: { readonly value: unknown } | null = null;
    f().then(
      (value) => (settled = { value }),
      (value: unknown) => (settled = { value }),
    );
    for (let ms = 0; settled === null && ms <= limitMs; ms += 100) {
      await new Promise((resolve) => setImmediate(resolve));
      t.mock.timers.tick(100);
    }
    await new Promise((resolve) => setImmediate(resolve));
    assert.ok(settled !== null, `still waiting after ${limitMs} ms`);
    return (settled as { readonly value: unknown }).value;
  } finally {
    t.mock.timers.reset();
  }
}

test("an exchange with the update service that goes unanswered, and a progress check unanswered the last time it is asked, throw Unanswered", async (t) => {
  const img = Uint8Array.from({ length: 300 }, (_, i) => i & 0xff);
  const silent = updatingDevice(() => null);
  const quiet = await connect(silent.device);
  let got = await onMockedTime(t, () => quiet.transfer(Transfer.app(quiet.session, 0, img), img), 36_000);
  assert.ok(got instanceof Unanswered && !(got instanceof SessionError), `the start went unanswered and threw ${String(got)}`);
  assert.equal((got as Error).message, "the device did not answer the update service in time");

  const unchecked = updatingDevice(started);
  const link = await connect(unchecked.device);
  got = await onMockedTime(t, () => link.transfer(Transfer.app(link.session, 0, img), img), 26_000);
  assert.ok(got instanceof Unanswered, `the check went unanswered and threw ${String(got)}`);
  assert.equal((got as Error).message, "the device did not answer the update service in time");
  assert.equal(unchecked.asked.filter((op) => op === P.UPDATE_OPS.query).length, 3, "the check was asked three times");
});

test("a failure of another kind is not Unanswered: a write the browser turns away, a link that is down, and a progress check turned away the last time it is asked", async (t) => {
  const d = busyDevice();
  d.characteristics.get(P.CONTROL)!.onWrite = () => {
    throw new DOMException("GATT operation failed for unknown reason.", "NetworkError");
  };
  const link = await connect(d.device, { timeoutMs: 30 });
  await assert.rejects(link.request(BATTERY), (e: Error) => !(e instanceof Unanswered) && e.name === "NetworkError");
  d.drop();
  await assert.rejects(
    link.request(BATTERY),
    (e: Error) => !(e instanceof Unanswered) && e instanceof SessionError && e.message === "ConnectionError: BLE link is down",
  );

  const img = Uint8Array.from({ length: 300 }, (_, i) => i & 0xff);
  const refusing = updatingDevice(started, (op) => op === P.UPDATE_OPS.query);
  const updating = await connect(refusing.device);
  const got = await onMockedTime(t, () => updating.transfer(Transfer.app(updating.session, 0, img), img), 8_000);
  assert.ok(got instanceof Error && !(got instanceof Unanswered) && got.name === "NetworkError", String(got));
  assert.equal(refusing.asked.filter((op) => op === P.UPDATE_OPS.query).length, 3, "the check was asked three times");
});

/** A transfer's states, by the codes a progress check gives them. */
const IDLE = 0;
const RECEIVING = 1;
const COMPLETE = 2;
const FAILED = 3;

/**
 * A device that carries a transfer as a device's update engine does, behind
 * the browser's link. START begins a transfer, or carries on with the same
 * one, and is refused as busy while another is under way. The device holds
 * what arrives, counts it and checksums it. A progress check says the
 * transfer's state, the count and the checksum. FINISH checks the count,
 * and is refused with the device's verdict where there is no transfer under
 * way. ABORT drops the transfer, and is refused where there is none.
 * ACTIVATE puts a transfer that verified in force, and the device restarts
 * once it has answered, dropping the link; it is refused as no transfer
 * where there is none, as verification failed where the transfer did not
 * verify, and as unsupported for a head, as a device's update engine
 * refuses it, or with `activateStatus` where that is set. Each update
 * request is answered once the write that carried it has returned, as a
 * device answers. The byte at `corruptAt`, counted from the start of the
 * transfer, arrives changed. A device that is `streaming` refuses START as
 * busy. The answers to the operations in `silent` never come, though the
 * device still restarts after an ACTIVATE it accepted, the browser turns
 * away the write of each operation in `turnedAway`, and the device drops
 * the link when the operation `dropAt` names arrives, instead of answering
 * it. A write on a link that is down never reaches the device, and the
 * browser refuses it. Every update request that reaches the device, every
 * answer it sends and its restart are listed in order, each request and
 * answer by its operation's name.
 */
function carryingDevice() {
  const d = busyDevice();
  const control = new FakeCharacteristic(P.UPDATE_CONTROL);
  const data = new FakeCharacteristic(P.UPDATE_DATA);
  d.characteristics.set(P.UPDATE_CONTROL, control);
  d.characteristics.set(P.UPDATE_DATA, data);
  let state = IDLE;
  let transfer = "";
  let target: string | null = null;
  let total = 0;
  let verdict: number | null = null;
  const device = {
    ...d,
    control,
    held: [] as number[],
    log: [] as string[],
    corruptAt: null as number | null,
    streaming: false,
    silent: new Set<number>(),
    turnedAway: new Set<number>(),
    dropAt: null as number | null,
    activateStatus: null as number | null,
  };
  const fail = (why: number): void => {
    verdict = why;
    state = FAILED;
  };
  const reached = (): void => {
    if (!d.device.gatt!.connected) {
      throw new DOMException("GATT Server is disconnected. Cannot perform GATT operations. (Re)connect first with `device.gatt.connect`.", "NetworkError");
    }
  };
  data.onWrite = (bytes) => {
    reached();
    if (state !== RECEIVING) return;
    if (device.held.length + bytes.length > total) return fail(1); // length
    const out = Array.from(bytes);
    const i = device.corruptAt === null ? -1 : device.corruptAt - device.held.length;
    if (i >= 0 && i < out.length) {
      out[i] = out[i]! ^ 0x01;
      device.corruptAt = null;
    }
    device.held.push(...out);
  };
  control.onWrite = (bytes) => {
    reached();
    const request = P.decodeUpdateRequest(bytes);
    if (device.turnedAway.has(request.op)) throw new DOMException("GATT operation failed for unknown reason.", "NetworkError");
    device.log.push(`${request.opName} asked`);
    if (device.dropAt === request.op) {
      d.drop();
      return;
    }
    let status = 0;
    let body = new Uint8Array(0);
    let restarts = false;
    if (request.opName === "start") {
      const asked = P.toHex(bytes.subarray(1));
      if (device.streaming || (state === RECEIVING && asked !== transfer)) {
        status = 3; // busy
      } else {
        if (state !== RECEIVING) {
          device.held.length = 0;
          transfer = asked;
          target = request.targetName;
          total = request.totalLen!;
          verdict = null;
          state = RECEIVING;
        }
        body = new Uint8Array(7);
        body[0] = 1;
        new DataView(body.buffer).setUint16(1, 100, true);
        new DataView(body.buffer).setUint32(3, device.held.length, true);
      }
    } else if (request.opName === "query") {
      body = new Uint8Array(9);
      body[0] = state;
      new DataView(body.buffer).setUint32(1, device.held.length, true);
      new DataView(body.buffer).setUint32(5, crc32(Uint8Array.from(device.held)), true);
    } else if (request.opName === "finish") {
      if (state !== RECEIVING) status = 4; // no transfer
      else if (device.held.length !== total) {
        fail(1); // length
        status = 6;
      } else {
        verdict = 0; // verified
        state = COMPLETE;
      }
      if (verdict !== null) body = Uint8Array.of(verdict);
    } else if (request.opName === "abort") {
      if (state === IDLE) status = 4; // no transfer
      else {
        device.held.length = 0;
        transfer = "";
        verdict = null;
        state = IDLE;
      }
    } else if (request.opName === "activate") {
      if (device.activateStatus !== null) status = device.activateStatus;
      else if (state === IDLE) status = 4; // no transfer
      else if (state !== COMPLETE) status = 6; // verification failed
      else if (target === "head") status = 2; // unsupported
      else {
        transfer = "";
        state = IDLE;
        restarts = true;
      }
    }
    // A device that restarts drops the link once the write has returned and
    // its answer, where one comes, has gone.
    const restart = (): void => {
      device.log.push("restarted");
      d.drop();
    };
    if (device.silent.has(request.op)) {
      if (restarts) afterTheWrite(restart);
      return;
    }
    const answer = P.encodeUpdateResponse(request.op, status, body);
    afterTheWrite(() => {
      device.log.push(`${request.opName} answered`);
      control.notify(answer);
      if (restarts) restart();
    });
  };
  return device;
}

/** An image of 300 bytes, which goes as three writes of 100. */
const IMAGE = Uint8Array.from({ length: 300 }, (_, i) => i & 0xff);

/** An image of 5,000 bytes. */
const IMAGE_5000 = Uint8Array.from({ length: 5000 }, (_, i) => (i * 7 + 13) & 0xff);

/** A transfer of `IMAGE` that starts at the smallest size, where no smaller one remains to start the file again at, so the first fault ends it. */
function atTheSmallest(link: Connection): Transfer {
  const t = Transfer.app(link.session, 0, IMAGE);
  t.limitWrites(20);
  return t;
}

test("a transfer that fails with the link up, followed at once by a new transfer on the same link, has the new START take START's answer", async () => {
  const d = carryingDevice();
  const link = await connect(d.device);
  d.corruptAt = 50;
  await assert.rejects(link.transfer(atTheSmallest(link), IMAGE), ChecksumMismatch);
  assert.equal(await link.transfer(Transfer.app(link.session, 0, IMAGE), IMAGE), "verified");
  assert.deepEqual(Uint8Array.from(d.held), IMAGE, "the device holds the new transfer, every byte once");
});

test("an update answer to one operation never settles a request waiting for another", async () => {
  const d = carryingDevice();
  const link = await connect(d.device);
  const going = link.transfer(Transfer.app(link.session, 0, IMAGE), IMAGE);
  // START waits for its answer, and a late answer to a progress check comes first.
  d.control.notify(P.encodeUpdateResponse(P.UPDATE_OPS.query, 0, new Uint8Array(9)));
  assert.equal(await going, "verified");
  assert.deepEqual(Uint8Array.from(d.held), IMAGE);
});

test("an update answer no request waits for is dropped, and never taken by a request that comes after it", async () => {
  const d = carryingDevice();
  const link = await connect(d.device);
  // A START answer saying the device already holds 200 bytes, while nothing waits.
  const resumed = new Uint8Array(7);
  new DataView(resumed.buffer).setUint16(1, 100, true);
  new DataView(resumed.buffer).setUint32(3, 200, true);
  d.control.notify(P.encodeUpdateResponse(P.UPDATE_OPS.start, 0, resumed));
  const going = link.transfer(Transfer.app(link.session, 0, IMAGE), IMAGE);
  // A FINISH answer refusing an image, while only START waits.
  d.control.notify(P.encodeUpdateResponse(P.UPDATE_OPS.finish, 6, Uint8Array.of(6)));
  assert.equal(await going, "verified", "the transfer took its own FINISH answer");
  assert.deepEqual(Uint8Array.from(d.held), IMAGE, "the transfer took its own START answer, and skipped nothing");
});

test("a transfer that fails with the link up waits for the device's answer to ABORT, and then throws its own error", async () => {
  const d = carryingDevice();
  const link = await connect(d.device);
  d.corruptAt = 50;
  await assert.rejects(
    link.transfer(atTheSmallest(link), IMAGE).finally(() => d.log.push("thrown")),
    ChecksumMismatch,
  );
  assert.deepEqual(d.log.slice(-3), ["abort asked", "abort answered", "thrown"]);
  assert.deepEqual(d.held, [], "the device dropped what it held");
});

test("an ABORT the device refuses, or whose write the browser turns away, changes nothing, and the transfer throws its own error", async () => {
  // A device that streams refuses START as busy, and then has no transfer to abort.
  const streaming = carryingDevice();
  streaming.streaming = true;
  let link = await connect(streaming.device);
  await assert.rejects(
    link.transfer(Transfer.app(link.session, 0, IMAGE), IMAGE).finally(() => streaming.log.push("thrown")),
    (e: unknown) => e instanceof Refused && e.op === "start" && e.status === "busy",
  );
  assert.deepEqual(streaming.log, ["start asked", "start answered", "abort asked", "abort answered", "thrown"]);

  const d = carryingDevice();
  d.turnedAway.add(P.UPDATE_OPS.abort);
  d.corruptAt = 50;
  link = await connect(d.device);
  await assert.rejects(link.transfer(Transfer.app(link.session, 0, IMAGE), IMAGE), ChecksumMismatch);
});

test("an ABORT the device does not answer is waited for five seconds, and then the transfer throws its own error", async (t) => {
  const d = carryingDevice();
  d.silent.add(P.UPDATE_OPS.abort);
  d.corruptAt = 50;
  const link = await connect(d.device);
  t.mock.timers.enable({ apis: ["setTimeout"] });
  try {
    let settled: { readonly value: unknown } | null = null;
    link.transfer(atTheSmallest(link), IMAGE).then(
      (value) => (settled = { value }),
      (value: unknown) => (settled = { value }),
    );
    let asked: number | null = null;
    let ms = 0;
    for (; settled === null && ms <= 10_000; ms += 100) {
      await new Promise((resolve) => setImmediate(resolve));
      if (asked === null && d.log.includes("abort asked")) asked = ms;
      t.mock.timers.tick(100);
    }
    await new Promise((resolve) => setImmediate(resolve));
    assert.ok(settled !== null, "still waiting after ten seconds");
    const got = (settled as { readonly value: unknown }).value;
    assert.ok(got instanceof ChecksumMismatch, String(got));
    assert.ok(asked !== null, "ABORT was never asked");
    assert.ok(ms - asked >= 5_000 && ms - asked < 5_500, `the transfer threw ${ms - asked} ms after it asked ABORT`);
  } finally {
    t.mock.timers.reset();
  }
});

test("a transfer whose link went down sends nothing more to the device, ABORT included, and throws its own error", async (t) => {
  // The device drops the link when the progress check arrives.
  const d = carryingDevice();
  d.dropAt = P.UPDATE_OPS.query;
  const link = await connect(d.device);
  const got = await onMockedTime(t, () => link.transfer(Transfer.app(link.session, 0, IMAGE), IMAGE), 10_000);
  assert.ok(got instanceof Error && !(got instanceof Unanswered) && got.name === "NetworkError", String(got));
  assert.deepEqual(d.log, ["start asked", "start answered", "query asked"]);
});

test("after a progress check says the transfer failed, no progress is reported, as the library reports none there, and the transfer ends with the device's verdict", async () => {
  // The device took the envelope, the image's first 32 bytes, and refused
  // the image there, as it refuses one built for its other slot. A progress
  // check says the transfer failed and that the device holds 32 bytes, with
  // its checksum over the whole of the first write, of 100 bytes, as the
  // firmware gives it, and the answer to the finish that follows gives the
  // verdict.
  const failed = new Uint8Array(9);
  failed[0] = 3;
  new DataView(failed.buffer).setUint32(1, 32, true);
  new DataView(failed.buffer).setUint32(5, crc32(IMAGE.subarray(0, 100)), true);
  const d = updatingDevice((op) =>
    op === P.UPDATE_OPS.query ? failed : op === P.UPDATE_OPS.finish ? Uint8Array.of(4) : op === P.UPDATE_OPS.abort ? new Uint8Array(0) : started(op),
  );
  const link = await connect(d.device);
  const heard: [number | undefined, number, number][] = [];
  await assert.rejects(
    link.transfer(Transfer.app(link.session, 0, IMAGE), IMAGE, ({ taken, total }) => heard.push([d.asked.at(-1), taken, total])),
    (e: unknown) => e instanceof Rejected && e.result === "slot",
  );
  assert.deepEqual(d.asked, [P.UPDATE_OPS.start, P.UPDATE_OPS.query, P.UPDATE_OPS.finish, P.UPDATE_OPS.abort]);
  assert.deepEqual(heard, [], "nothing is reported, the answer to the first START among them");
});

test("activate puts an image the device verified in force, as the library's activate does: the device answers ok and restarts, and the session takes the stream as stopped", async () => {
  const d = carryingDevice();
  const link = await connect(d.device);
  assert.equal(await link.transfer(Transfer.app(link.session, 0, IMAGE), IMAGE), "verified");
  await link.start();
  assert.equal(link.session.streaming, true);
  assert.equal(await link.activate(), undefined);
  assert.deepEqual(d.log.slice(-3), ["activate asked", "activate answered", "restarted"]);
  assert.equal(link.connected, false, "the device restarted");
  assert.equal(link.session.streaming, false, "the session takes the stream as stopped");
});

test("an ACTIVATE the device refuses throws Refused with the device's status in words, for every status, and the link stays up", async () => {
  for (const status of [1, 2, 3, 4, 5, 6]) {
    const d = carryingDevice();
    d.activateStatus = status;
    const link = await connect(d.device);
    await assert.rejects(link.activate(), (e: unknown) => {
      assert.ok(e instanceof Refused, `status ${status}: ${String(e)}`);
      assert.equal(e.op, "activate");
      assert.equal(e.status, P.UPDATE_STATUS[status]);
      assert.equal(e.message, `the device refused to activate the image: ${P.UPDATE_STATUS[status]}`);
      return true;
    });
    assert.equal(link.connected, true, `status ${status}: the device did not restart`);
  }
  // As a device's update engine refuses: nothing to put in force, and a head, which is selected rather than put in force.
  const idle = carryingDevice();
  const link = await connect(idle.device);
  await assert.rejects(link.activate(), { name: "Refused", message: "the device refused to activate the image: no transfer" });
  const blob = P.buildHead(Array.from({ length: 2 }, () => Array.from({ length: 96 }, () => 1)), [0, 0], [1, 1], { name: "focus" });
  assert.equal(await link.transfer(Transfer.head(link.session, 1, blob), blob), "verified");
  await assert.rejects(link.activate(), { name: "Refused", message: "the device refused to activate the image: unsupported" });
  assert.equal(link.connected, true);
});

test("an ACTIVATE the device does not answer is waited for five seconds, as the library waits, and then throws Unanswered", async (t) => {
  // Nothing was transferred, so the device refuses and stays, and its answer is lost.
  const d = carryingDevice();
  d.silent.add(P.UPDATE_OPS.activate);
  const link = await connect(d.device);
  t.mock.timers.enable({ apis: ["setTimeout"] });
  try {
    let settled: { readonly value: unknown } | null = null;
    link.activate().then(
      (value) => (settled = { value }),
      (value: unknown) => (settled = { value }),
    );
    let ms = 0;
    for (; settled === null && ms <= 40_000; ms += 100) {
      await new Promise((resolve) => setImmediate(resolve));
      t.mock.timers.tick(100);
    }
    await new Promise((resolve) => setImmediate(resolve));
    assert.ok(settled !== null, "still waiting after forty seconds");
    const got = (settled as { readonly value: unknown }).value;
    assert.ok(got instanceof Unanswered && !(got instanceof SessionError), String(got));
    assert.equal((got as Error).message, "the device did not answer the update service in time");
    assert.ok(ms >= 5_000 && ms < 5_500, `ACTIVATE was given up ${ms} ms after it was asked`);
    assert.equal(link.connected, true);
  } finally {
    t.mock.timers.reset();
  }
});

test("a link that drops before ACTIVATE's answer comes throws Unanswered once five seconds have passed, as the library raises when no answer comes, since only the answer says the device accepted, and a write that fails throws its own error", async (t) => {
  // The device accepts, and restarts before its answer reaches the page.
  const d = carryingDevice();
  d.silent.add(P.UPDATE_OPS.activate);
  const link = await connect(d.device);
  assert.equal(await link.transfer(Transfer.app(link.session, 0, IMAGE), IMAGE), "verified");
  t.mock.timers.enable({ apis: ["setTimeout"] });
  try {
    let settled: { readonly value: unknown } | null = null;
    link.activate().then(
      (value) => (settled = { value }),
      (value: unknown) => (settled = { value }),
    );
    let dropped: number | null = null;
    let ms = 0;
    for (; settled === null && ms <= 10_000; ms += 100) {
      await new Promise((resolve) => setImmediate(resolve));
      if (dropped === null && d.log.includes("restarted")) dropped = ms;
      t.mock.timers.tick(100);
    }
    await new Promise((resolve) => setImmediate(resolve));
    assert.ok(settled !== null, "still waiting after ten seconds");
    const got = (settled as { readonly value: unknown }).value;
    assert.ok(got instanceof Unanswered && !(got instanceof SessionError), `the drop was taken for ${String(got)}`);
    assert.equal((got as Error).message, "the device did not answer the update service in time");
    assert.ok(dropped !== null && dropped <= 100, `the link dropped ${String(dropped)} ms after ACTIVATE was written`);
    assert.ok(ms >= 5_000 && ms < 5_500, `ACTIVATE was given up ${ms} ms after it was asked`);
  } finally {
    t.mock.timers.reset();
  }
  assert.deepEqual(d.log.slice(-2), ["activate asked", "restarted"]);
  assert.equal(link.connected, false);

  // A write the browser turns away, and a write on a link that is down, never reached the device.
  const turned = carryingDevice();
  turned.turnedAway.add(P.UPDATE_OPS.activate);
  const refused = await connect(turned.device);
  await assert.rejects(refused.activate(), (e: Error) => !(e instanceof Unanswered) && e.name === "NetworkError");
  turned.drop();
  await assert.rejects(refused.activate(), (e: Error) => !(e instanceof Unanswered) && e.name === "NetworkError");
  assert.deepEqual(turned.log, []);
});

/** A packet of four channels, three samples, numbered from `index`, with USB power present when `usb` says. */
function packet(index: number, usb = false): Uint8Array {
  return P.encodePacket({
    index,
    deviceTime: BigInt(1_000_000 + 2000 * index),
    gainCode: P.GAIN_BY_CODE.indexOf(24),
    rateCode: 5,
    counts: [
      [1, 2, 3, 4],
      [1, 2, 3, 4],
      [1, 2, 3, 4],
    ],
    discontinuity: index === 0,
    usbPresent: usb,
  });
}

test("a listener that throws stops neither the others, nor the package's own consumers, nor the assembly of windows", async () => {
  const d = busyDevice(new Map([[OP.get_model_info, [P.encodeResponse(OP.get_model_info, 0, READY)]]]));
  const link = await connect(d.device);
  const errors: unknown[] = [];
  const g = globalThis as { reportError?: (e: unknown) => void };
  g.reportError = (e) => errors.push(e);
  try {
    const mine: number[] = [];
    const windows: number[] = [];
    const theirs: number[] = [];
    await link.setEmbeddings("window");
    link.on(() => {
      throw new Error("the program's own mistake");
    });
    link.on((e) => {
      if (e.type === "samples") mine.push(e.batch.index);
    });
    const off = tap(link, "batch", (b) => theirs.push(b.index));
    link.onEmbedding = (_l, w) => {
      windows.push(w.index);
      throw new Error("and another");
    };
    d.characteristics.get(P.EEG_DATA)!.notify(packet(0));
    d.characteristics.get(P.EEG_DATA)!.notify(packet(3));
    d.characteristics.get(P.EMBEDDINGS)!.notify(vector("the same window embedding as firmware before 1.4.2"));
    assert.deepEqual(mine, [0, 3], "the other listener heard every batch");
    assert.deepEqual(theirs, [0, 3], "the package's own consumer heard every batch");
    assert.equal(windows.length, 1, "the window was put together and handed on");
    assert.equal(errors.length, 4, "each error reported, none breaking delivery");
    assert.equal(link.session.gapsAnnounced, 1, "no break where there was none");
    off();
  } finally {
    delete g.reportError;
  }
});

test("onUsbStop is called once, after the package's own consumers have the packet, when USB power ends the stream", async () => {
  const d = busyDevice();
  const link = await connect(d.device);
  const order: string[] = [];
  tap(link, "batch", () => order.push("own"));
  link.on((e) => {
    if (e.type === "samples") order.push("listener");
  });
  let stops = 0;
  link.onUsbStop = (l) => {
    assert.equal(l, link);
    order.push("usb");
    stops++;
  };
  await link.start();
  d.characteristics.get(P.EEG_DATA)!.notify(packet(0));
  d.characteristics.get(P.EEG_DATA)!.notify(packet(3, true));
  d.characteristics.get(P.EEG_DATA)!.notify(packet(6, true));
  assert.equal(stops, 1);
  assert.equal(link.session.endedByUsb, true);
  assert.deepEqual(order, ["own", "listener", "own", "usb", "listener", "own", "listener"]);
});

/** Whole windows of the form the device was last told, one each time `send` is called, nothing while it is off. */
function embeddingDevice() {
  const model = P.encodeResponse(OP.get_model_info, 0, READY);
  const d = busyDevice(new Map([[OP.get_model_info, Array.from({ length: 20 }, () => model)]]));
  let form = "off";
  let index = 0;
  const control = d.characteristics.get(P.CONTROL)!;
  const answer = control.onWrite!;
  control.onWrite = (bytes) => {
    if (bytes[0] === OP.set_embeddings) form = (Object.entries(P.EMBEDDING_FORMS).find(([, v]) => v === bytes[1]) ?? ["off"])[0];
    answer(bytes);
  };
  const tokensPerChannel = P.decodeModelInfo(READY).tokensPerChannel;
  const send = (): void => {
    const tokens: (number | null)[] =
      form === "window" ? [null] : form === "tokens" ? range(4 * tokensPerChannel) : form === "both" ? [null, ...range(4 * tokensPerChannel)] : [];
    for (const token of tokens) {
      d.characteristics.get(P.EMBEDDINGS)!.notify(
        P.encodeEmbedding({ index, deviceTime: BigInt(1000 * index), windowSamples: 2000, encoderId: "a7c61680d9a202db", embedDim: 3, token, values: [1, 2, 3] }),
      );
    }
    index++;
  };
  const forms = (): string[] =>
    d.writes.filter((w) => w[0] === OP.set_embeddings).map((w) => (Object.entries(P.EMBEDDING_FORMS).find(([, v]) => v === w[1]) ?? ["?"])[0]);
  return { ...d, send, forms };
}

function range(n: number): number[] {
  return Array.from({ length: n }, (_, i) => i);
}

/** `collectEmbeddings` while the device sends a window every millisecond, and `meanwhile`, a program's own request, after the first. */
async function collect(link: Connection, send: () => void, windows: number, form = "window", timeout = 2, meanwhile?: () => Promise<void>) {
  let sent = 0;
  let running = true;
  const feed = (async () => {
    while (running) {
      await sleep(1);
      send();
      sent++;
      if (sent === 1 && meanwhile !== undefined) await meanwhile();
    }
  })();
  try {
    return await link.collectEmbeddings(windows, form, timeout);
  } finally {
    running = false;
    await feed;
  }
}

/**
 * What `link` hears of one of each thing its device sends: the answer to a
 * request, the batch numbered `index`, a status, and, once embeddings are
 * turned on, a whole window, as its listeners hear them, the breaks between
 * batches aside, and the windows `onEmbedding` hears.
 */
async function heard(link: Connection, d: ReturnType<typeof embeddingDevice>, index: number): Promise<{ events: string[]; windows: number }> {
  const events: string[] = [];
  const off = link.on((e) => {
    if (e.type === "answer") events.push(`answer 0x${e.answer.opcode.toString(16)}`);
    else if (e.type === "samples") events.push(`samples ${e.batch.index}`);
    else if (e.type !== "gap") events.push(e.type);
  });
  let windows = 0;
  try {
    await link.request(BATTERY);
    d.characteristics.get(P.EEG_DATA)!.notify(packet(index));
    d.characteristics.get(P.STATUS)!.notify(STATUS);
    await link.setEmbeddings("window", () => {
      windows += 1;
    });
    d.send();
  } finally {
    off();
  }
  return { events, windows };
}

/** All of what `heard` asks for, each heard once. */
function once(index: number): { events: string[]; windows: number } {
  const answer = (op: number) => `answer 0x${op.toString(16)}`;
  return {
    events: [answer(OP.get_battery), `samples ${index}`, "status", answer(OP.get_model_info), answer(OP.set_embeddings), "embedding"],
    windows: 1,
  };
}

test("a first connection subscribes to everything its device notifies, and hears each answer, batch, status and window once", async () => {
  const d = embeddingDevice();
  const link = await connect(d.device, { timeoutMs: 200 });
  for (const uuid of [P.EEG_DATA, P.STATUS, P.CONTROL_RESPONSE]) assert.ok(d.browser.held(uuid), `not subscribed to ${uuid}`);
  assert.ok(!d.browser.held(P.EMBEDDINGS), "embeddings are subscribed to when a form is first asked for");
  assert.deepEqual(await heard(link, d, 0), once(0));
  assert.ok(d.browser.held(P.EMBEDDINGS));
  assert.equal(d.connects(), 1);
});

test("after the device drops the link, the connection connected again hears answers, batches, status and embeddings again, each once", async () => {
  const d = embeddingDevice();
  const link = await connect(d.device, { timeoutMs: 200 });
  assert.deepEqual(await heard(link, d, 0), once(0));
  // As a device drops it that restarts into new firmware, is switched off
  // and on, or goes out of range and comes back.
  d.drop();
  assert.equal(await link.connect(), link);
  assert.deepEqual(await heard(link, d, 3), once(3), "heard on the new link");
  d.drop();
  await link.connect();
  assert.deepEqual(await heard(link, d, 6), once(6), "and on the link after that");
  // A new connection made to the device after it dropped the link hears it,
  // and the connection whose link went hears nothing more.
  d.drop();
  const fresh = await connect(d.device, { timeoutMs: 200 });
  const stale: string[] = [];
  link.on((e) => stale.push(e.type));
  assert.deepEqual(await heard(fresh, d, 9), once(9));
  assert.deepEqual(stale, []);
});

test("after the link is dropped from this side, the connection connected again hears answers, batches, status and embeddings again, each once", async () => {
  const d = embeddingDevice();
  const link = await connect(d.device, { timeoutMs: 200 });
  assert.deepEqual(await heard(link, d, 0), once(0));
  // Its own disconnect, which ends every subscription before the link drops.
  await link.disconnect();
  for (const uuid of [P.EEG_DATA, P.STATUS, P.CONTROL_RESPONSE, P.EMBEDDINGS]) assert.ok(!d.browser.held(uuid), `still subscribed to ${uuid}`);
  await link.connect();
  assert.deepEqual(await heard(link, d, 3), once(3), "after its own disconnect");
  // The browser's own disconnect, around the connection, which ends none.
  d.device.gatt!.disconnect();
  assert.equal(link.connected, false);
  await link.connect();
  assert.deepEqual(await heard(link, d, 6), once(6), "after the browser's disconnect");
  // Connected again while it is connected.
  await link.connect();
  assert.deepEqual(await heard(link, d, 9), once(9), "after connecting again while connected");
  assert.equal(d.connects(), 4);
});

test("an embeddings subscription the browser refuses leaves nothing behind, and the one asked for next hears each window once", async () => {
  const d = embeddingDevice();
  const link = await connect(d.device, { timeoutMs: 200 });
  d.browser.refuseNextStart(P.EMBEDDINGS);
  await assert.rejects(link.setEmbeddings("window"), (e: Error) => e.name === "NetworkError" && e.message === "GATT operation failed for unknown reason.");
  assert.equal(d.browser.held(P.EMBEDDINGS), false);
  const events: number[] = [];
  link.on((e) => {
    if (e.type === "embedding") events.push(e.embedding.index);
  });
  const windows: number[] = [];
  await link.setEmbeddings("window", (_l, w) => windows.push(w.index));
  d.send();
  d.send();
  assert.deepEqual(events, [0, 1], "each notification heard once");
  assert.deepEqual(windows, [0, 1], "each window handed on once");
});

test("collecting embeddings leaves them as they were found, and never touches onEmbedding", async () => {
  let d = embeddingDevice();
  let link = await connect(d.device);
  let got = await collect(link, d.send, 3);
  assert.equal(got.length, 3);
  assert.deepEqual(d.forms(), ["window", "off"]);
  assert.equal(own(link).form, "off");
  assert.equal(link.onEmbedding, null);
  assert.equal(own(link).window.length, 0, "nothing of the package's left on the device");

  // A form the program had on, with its own callback, is sent nothing and
  // keeps both, and the callback hears every window.
  d = embeddingDevice();
  link = await connect(d.device);
  const heard: number[] = [];
  await link.setEmbeddings("window", (_l, w) => heard.push(w.index));
  const mine = link.onEmbedding;
  got = await collect(link, d.send, 3);
  assert.deepEqual(d.forms(), ["window"], "nothing sent but the program's own request");
  assert.equal(own(link).form, "window");
  assert.equal(link.onEmbedding, mine);
  assert.deepEqual(heard.slice(0, 3), got.map((w) => w.index));

  // A form that lacks what is collected is widened for it and put back.
  d = embeddingDevice();
  link = await connect(d.device);
  await link.setEmbeddings("tokens");
  got = await collect(link, d.send, 2);
  assert.equal(got.length, 2);
  assert.deepEqual(got[0]!.embedding, [1, 2, 3]);
  assert.ok(got[0]!.tokens !== null);
  assert.deepEqual(d.forms(), ["tokens", "both", "tokens"]);

  // A form the program sets during a collection is its own, and stays.
  d = embeddingDevice();
  link = await connect(d.device);
  got = await collect(link, d.send, 3, "window", 2, () => link.setEmbeddings("both"));
  assert.equal(got.length, 3);
  assert.deepEqual(d.forms(), ["window", "both"]);
  assert.equal(own(link).form, "both");

  // A collection that times out says how many windows arrived, and still
  // leaves them as found.
  d = embeddingDevice();
  link = await connect(d.device);
  await link.setEmbeddings("tokens");
  await assert.rejects(collect(link, () => undefined, 2, "window", 0.05), (e: unknown) => {
    assert.ok(e instanceof SessionError);
    assert.equal(e.message, "TimeoutError: 0 of 2 windows arrived in 0.05 seconds");
    return true;
  });
  assert.deepEqual(d.forms(), ["tokens", "both", "tokens"]);
  assert.equal(own(link).window.length, 0);
});

/**
 * `collectEmbeddings(2, ask)` on a device the program set to `have`, with its
 * own callback, that sends one whole window in `have` when it receives the
 * request for the model's information `setEmbeddings` makes first in
 * widening the form, before it answers that request, and a window every
 * millisecond once it has been told the wider form: the windows collected,
 * that window's index, the indexes the program's callback heard, and the
 * forms the device was told.
 */
async function widenedPastAnOldWindow(have: string, ask: string) {
  const d = embeddingDevice();
  const link = await connect(d.device);
  const heard: number[] = [];
  await link.setEmbeddings(have, (_l, w) => heard.push(w.index));
  let sent = 0;
  const send = (): void => {
    d.send();
    sent++;
  };
  let old = null as number | null;
  const control = d.characteristics.get(P.CONTROL)!;
  const answer = control.onWrite!;
  control.onWrite = (bytes) => {
    if (bytes[0] === OP.get_model_info && old === null) {
      old = sent;
      send();
    }
    answer(bytes);
  };
  const got = await collect(link, () => (d.forms().length >= 2 ? send() : undefined), 2, ask);
  return { got, old, heard, forms: d.forms() };
}

// A window the device sends in the form the program set, while a collection
// widens that form, still reaches onEmbedding and is not collected. Such a
// window was once collected without the form asked for.
test("a collection that widens tokens collects only windows with the window embedding", async () => {
  const { got, old, heard, forms } = await widenedPastAnOldWindow("tokens", "window");
  assert.deepEqual(got.filter((w) => w.embedding === null).map((w) => w.index), [], "collected without the window embedding");
  assert.equal(got.length, 2);
  for (const w of got) assert.deepEqual(w.embedding, [1, 2, 3]);
  assert.ok(old !== null && !got.some((w) => w.index === old));
  assert.ok(heard.includes(old), "onEmbedding did not hear the window in the form the program set");
  assert.deepEqual(forms, ["tokens", "both", "tokens"]);
});

test("a collection that widens the window embedding collects only windows with tokens", async () => {
  const { got, old, heard, forms } = await widenedPastAnOldWindow("window", "tokens");
  assert.deepEqual(got.filter((w) => w.tokens === null).map((w) => w.index), [], "collected without the tokens");
  assert.equal(got.length, 2);
  for (const w of got) assert.equal(w.tokens?.length, 4 * P.decodeModelInfo(READY).tokensPerChannel);
  assert.ok(old !== null && !got.some((w) => w.index === old));
  assert.ok(heard.includes(old), "onEmbedding did not hear the window in the form the program set");
  assert.deepEqual(forms, ["window", "both", "window"]);
});

test("when one device cannot turn embeddings on, none is left hooked or switched on", async () => {
  const first = embeddingDevice();
  const second = embeddingDevice();
  const refusing = second.characteristics.get(P.CONTROL)!;
  const answer = refusing.onWrite!;
  refusing.onWrite = (bytes) => {
    if (bytes[0] === OP.set_embeddings) {
      second.writes.push(bytes);
      setImmediate(() => second.characteristics.get(P.CONTROL_RESPONSE)!.notify(Uint8Array.of(OP.set_embeddings, 1)));
      return;
    }
    answer(bytes);
  };
  const links = [await connect(first.device), await connect(second.device)];
  await assert.rejects(windowsOf(links, 3, new RunCtx(() => undefined)), /Refused: op 0x/);
  assert.deepEqual(first.forms(), ["window", "off"], "the device turned on is turned off again");
  for (const l of links) {
    assert.equal(own(l).window.length, 0);
    assert.equal(own(l).form, "off");
    assert.equal(own(l).lent, null);
  }
});

test("a transfer through the connection asks a check the device turned away again after half a second, and carries on from the device's count past writes it threw away", async () => {
  // The Python library's busy device behind the browser's link: it falls
  // behind at every thirteenth write and throws away what arrives until it
  // is asked where it is, and it turns the first check's write away.
  const d = busyDevice();
  const control = new FakeCharacteristic(P.UPDATE_CONTROL);
  const data = new FakeCharacteristic(P.UPDATE_DATA);
  d.characteristics.set(P.UPDATE_CONTROL, control);
  d.characteristics.set(P.UPDATE_DATA, data);
  const held: number[] = [];
  let writes = 0;
  let checks = 0;
  let behind = false;
  data.onWrite = (bytes) => {
    writes += 1;
    if (writes % 13 === 0) behind = true;
    if (!behind) held.push(...bytes);
  };
  control.onWrite = (bytes) => {
    const op = bytes[0]!;
    let body = new Uint8Array(0);
    if (op === P.UPDATE_OPS.start) {
      body = new Uint8Array(7);
      new DataView(body.buffer).setUint16(1, 100, true);
    } else if (op === P.UPDATE_OPS.query) {
      checks += 1;
      behind = false;
      if (checks === 1) throw new DOMException("GATT operation failed for unknown reason.", "NetworkError");
      body = new Uint8Array(9);
      body[0] = 1; // receiving
      new DataView(body.buffer).setUint32(1, held.length, true);
      new DataView(body.buffer).setUint32(5, crc32(Uint8Array.from(held)), true);
    } else if (op === P.UPDATE_OPS.finish) {
      body = Uint8Array.of(0); // verified
    }
    const answer = P.encodeUpdateResponse(op, 0, body);
    setTimeout(() => control.notify(answer), 0);
  };
  const link = await connect(d.device);
  const img = Uint8Array.from({ length: 5120 }, (_, i) => i & 0xff);
  const t0 = performance.now();
  assert.equal(await link.transfer(Transfer.app(link.session, 0, img), img), "verified");
  assert.ok(performance.now() - t0 >= 450, "the check turned away was asked again after half a second");
  assert.deepEqual(Uint8Array.from(held), img, "every byte arrived once, in order");
  assert.ok(writes > Math.ceil(5120 / 100), "what the device threw away went again");
});

/** Bytes in a page of the device's flash, which it writes whole. */
const PAGE = 4096;
/** Incoming writes the device holds while it writes a page. */
const HOLDS = 8;
/** How long writing a page takes, in milliseconds. */
const WRITING_MS = 2;

/**
 * A device with a real one's limits behind the browser's link, its control
 * link and its update service alike: writing a page of its flash takes a
 * moment, and meanwhile it holds eight incoming writes, a request among
 * them, and throws away what arrives beyond them. A request held is
 * answered once the page is written, and one thrown away is never
 * answered. Its update service answers a request once the write that
 * carried it has returned, as a device does. Every write that reaches it is
 * listed, in order, as where it went and its first byte. `verdict` is what
 * it says of the image at the end.
 */
function limitedDevice(verdict = 0) {
  const d = busyDevice();
  const control = d.characteristics.get(P.CONTROL)!;
  const respond = d.characteristics.get(P.CONTROL_RESPONSE)!;
  const update = new FakeCharacteristic(P.UPDATE_CONTROL);
  const data = new FakeCharacteristic(P.UPDATE_DATA);
  d.characteristics.set(P.UPDATE_CONTROL, update);
  d.characteristics.set(P.UPDATE_DATA, data);
  const held: number[] = [];
  const arrived: [string, number][] = [];
  let busy = false;
  let queue: [string, Uint8Array][] = [];
  let thrown = 0;
  let ticks = 0n;
  const answer = (where: string, bytes: Uint8Array): void => {
    const op = bytes[0]!;
    if (where === "control") {
      let body = new Uint8Array(0);
      if (op === OP.time_sync) {
        ticks += 1000n;
        body = new Uint8Array(8);
        new DataView(body.buffer).setBigUint64(0, ticks, true);
      }
      const response = P.encodeResponse(op, 0, body);
      setImmediate(() => respond.notify(response));
      return;
    }
    let status = 0;
    let body = new Uint8Array(0);
    if (op === P.UPDATE_OPS.start) {
      body = new Uint8Array(7);
      new DataView(body.buffer).setUint16(1, 244, true);
    } else if (op === P.UPDATE_OPS.query) {
      body = new Uint8Array(9);
      body[0] = 1; // receiving
      new DataView(body.buffer).setUint32(1, held.length, true);
      new DataView(body.buffer).setUint32(5, crc32(Uint8Array.from(held)), true);
    } else if (op === P.UPDATE_OPS.finish) {
      status = verdict ? 6 : 0;
      body = Uint8Array.of(verdict);
    }
    const response = P.encodeUpdateResponse(op, status, body);
    afterTheWrite(() => update.notify(response));
  };
  const take = (where: string, bytes: Uint8Array): void => {
    if (where !== "data") {
      answer(where, bytes);
      return;
    }
    const before = held.length;
    held.push(...bytes);
    if (!busy && Math.floor(before / PAGE) !== Math.floor(held.length / PAGE)) {
      busy = true;
      setTimeout(written, WRITING_MS);
    }
  };
  // The page is written, and then what arrived meanwhile, in order.
  const written = (): void => {
    const waiting = queue;
    queue = [];
    for (const [where, bytes] of waiting) take(where, bytes);
    busy = false;
  };
  const arrive = (where: string) => (bytes: Uint8Array): void => {
    arrived.push([where, bytes[0]!]);
    if (busy) {
      if (queue.length >= HOLDS) thrown += 1;
      else queue.push([where, bytes]);
      return;
    }
    take(where, bytes);
  };
  control.onWrite = arrive("control");
  update.onWrite = arrive("update");
  data.onWrite = arrive("data");
  return { device: d.device, held, arrived, thrown: () => thrown };
}

/**
 * A transfer through a daisy chain's connection to a device with a real
 * one's limits, the chain's resync running beside it with its time come at
 * every turn: what the transfer said, what reached the device while it
 * ran, and whether the resync asked the device for its clock before it and
 * after it.
 */
async function besideTheResync(verdict = 0) {
  const d = limitedDevice(verdict);
  const chain = new DaisyChain();
  await chain.connect(d.device);
  const link = chain.devices[0] as Connection;
  const stop = new AbortController();
  const loop = chain.resyncLoop(0, stop.signal);
  const synced = (from: number): boolean => d.arrived.slice(from).some(([where, op]) => where === "control" && op === OP.time_sync);
  for (let i = 0; i < 200 && !synced(0); i++) await sleep(1);
  const before = synced(0);
  const img = Uint8Array.from({ length: 10240 }, (_, i) => i & 0xff);
  let said: string | null;
  try {
    said = await link.transfer(Transfer.app(link.session, 0, img), img);
  } catch (e) {
    said = (e as Error).message;
  }
  // What reached the device from the transfer's first request to its end.
  const end = d.arrived.length;
  const during = d.arrived.slice(d.arrived.findIndex(([where, op]) => where === "update" && op === P.UPDATE_OPS.start), end);
  for (let i = 0; i < 200 && !synced(end); i++) await sleep(1);
  const after = synced(end);
  stop.abort();
  await loop;
  return { said, during, before, after, d, img };
}

/**
 * The device's verdict on an image's envelope, its first 32 bytes, in the
 * order the firmware checks them, for a transfer of `total` bytes to
 * `target`, on a device that holds the transport key 1 and takes an
 * application into slot 0: null when it takes it.
 */
function judged(envelope: Uint8Array, target: string, total: number): number | null {
  let e: P.Envelope;
  try {
    e = P.decodeEnvelope(envelope);
  } catch {
    return 2; // malformed
  }
  if (e.targetName !== target) return 3; // target
  if (target === "app" && e.slotLink !== 0) return 4; // slot
  if (e.keyId !== 1) return 10; // key id
  if (e.totalLen !== total) return 1; // length
  return null;
}

/**
 * An application image of `total` bytes whose envelope names slot 0, the
 * transport key 1 and the image's own length, the bytes after it counting
 * through every byte value, as the Python library's tests make one.
 */
function enveloped(total: number): Uint8Array {
  const out = new Uint8Array(total);
  out.set(P.encodeEnvelope({ target: "app", slotLink: 0, keyId: 1, nonce: Uint8Array.from({ length: 16 }, (_, i) => i), plainLen: total - P.ENVELOPE_LEN }));
  for (let i = P.ENVELOPE_LEN; i < total; i++) out[i] = (i - P.ENVELOPE_LEN) & 0xff;
  return out;
}

/** What a stand-in link does with a data write longer than it carries. */
type Way = "refuses" | "drops" | "cuts";

const WAYS: readonly Way[] = ["refuses", "drops", "cuts"];

/**
 * A device behind a link of `size`, as a system reports its size, which
 * carries a write of at most its size less 3 and treats a longer data write
 * its `way`: the browser refuses it, as Chrome on Linux refuses one; drops
 * it with no sign, the write resolving, as a Mac's browser resolves a write
 * whatever happens; or cuts it to its size less 3 with no sign, as Android
 * documents. The device has a real one's limits: writing a page of its flash
 * takes a moment, and meanwhile it holds eight incoming writes, a request
 * among them, and throws away what arrives beyond them, and a request held
 * is answered once the page is written. It answers an update request once
 * the write that carried it has returned. START begins a transfer, or
 * carries on with the same one, its answer saying the device takes writes of
 * 244 bytes; ABORT drops the transfer, and is refused where there is none;
 * FINISH verifies a transfer that holds every byte. The answers to the
 * operations in `silent` never come, the browser turns away the write of
 * each operation in `turnedAway`, so it never reaches the device, the device
 * drops the link when the operation `dropAt` names arrives, instead of
 * answering it, and the browser refuses the data write `refuseOnce` counts,
 * once, whatever its size, as a system may refuse a write while the link is
 * up. A write on a link that is down never reaches the device, and the
 * browser refuses it. Every update request, by its operation, and every data
 * write, by the bytes sent, are listed in order in `trace`, and so is each
 * progress `heard` reports, as "taken N".
 *
 * With `envelopes` set, the device judges an image's first 32 bytes as its
 * envelope once they are in, from whichever writes carried them, as the
 * firmware does, holding the transport key 1 and taking an application into
 * slot 0. An envelope it refuses fails the transfer: a progress check says
 * so, its count stays at the 32 bytes and its checksum covers every byte of
 * every write it took, it throws away every write that follows, and it
 * answers FINISH "no transfer" with its verdict.
 */
function sizedLink(size: number, way: Way) {
  const d = busyDevice();
  const update = new FakeCharacteristic(P.UPDATE_CONTROL);
  const data = new FakeCharacteristic(P.UPDATE_DATA);
  d.characteristics.set(P.UPDATE_CONTROL, update);
  d.characteristics.set(P.UPDATE_DATA, data);
  const held: number[] = [];
  const trace: (string | number)[] = [];
  const silent = new Set<number>();
  const turnedAway = new Set<number>();
  const options = { dropAt: null as number | null, refuseOnce: null as number | null, envelopes: false };
  let writes = 0;
  let receiving = false;
  let transfer = "";
  let target = "";
  let total = 0;
  let busy = false;
  let queue: [string, Uint8Array][] = [];
  let thrown = 0;
  // The verdict of a transfer the device failed, and every byte of every
  // write it took until then, which its checksum covers.
  let failed: number | null = null;
  let summed: number[] = [];
  const take = (where: string, bytes: Uint8Array): void => {
    if (where === "data") {
      if (!receiving || held.length + bytes.length > total) return;
      const before = held.length;
      if (options.envelopes && target !== "head" && before < P.ENVELOPE_LEN && before + bytes.length >= P.ENVELOPE_LEN) {
        const verdict = judged(Uint8Array.from([...held, ...bytes].slice(0, P.ENVELOPE_LEN)), target, total);
        if (verdict !== null) {
          summed = [...held, ...bytes];
          held.push(...bytes.subarray(0, P.ENVELOPE_LEN - before));
          failed = verdict;
          receiving = false;
          return;
        }
      }
      held.push(...bytes);
      if (!busy && Math.floor(before / PAGE) !== Math.floor(held.length / PAGE)) {
        busy = true;
        setTimeout(written, WRITING_MS);
      }
      return;
    }
    const request = P.decodeUpdateRequest(bytes);
    let status = 0;
    let body = new Uint8Array(0);
    if (request.opName === "start") {
      const asked = P.toHex(bytes.subarray(1));
      if (!(receiving && asked === transfer)) {
        held.length = 0;
        transfer = asked;
        target = request.targetName!;
        total = request.totalLen!;
        receiving = true;
        failed = null;
      }
      body = new Uint8Array(7);
      new DataView(body.buffer).setUint16(1, 244, true);
      new DataView(body.buffer).setUint32(3, held.length, true);
    } else if (request.opName === "query") {
      body = new Uint8Array(9);
      body[0] = failed !== null ? FAILED : receiving ? 1 : 0;
      new DataView(body.buffer).setUint32(1, held.length, true);
      new DataView(body.buffer).setUint32(5, crc32(Uint8Array.from(failed !== null ? summed : held)), true);
    } else if (request.opName === "finish" && failed !== null) {
      status = 4; // no transfer
      body = Uint8Array.of(failed);
    } else if (request.opName === "finish") {
      const whole = receiving && held.length === total;
      status = whole ? 0 : 6;
      body = Uint8Array.of(whole ? 0 : 1);
      receiving = false;
    } else if (request.opName === "abort") {
      if (!receiving && failed === null) status = 4;
      else {
        held.length = 0;
        transfer = "";
        receiving = false;
        failed = null;
      }
    }
    if (silent.has(request.op)) return;
    const answer = P.encodeUpdateResponse(request.op, status, body);
    afterTheWrite(() => update.notify(answer));
  };
  // The page is written, and then what arrived meanwhile, in order.
  const written = (): void => {
    const waiting = queue;
    queue = [];
    for (const [where, bytes] of waiting) take(where, bytes);
    busy = false;
  };
  const arrive = (where: string, bytes: Uint8Array): void => {
    if (busy) {
      if (queue.length >= HOLDS) thrown += 1;
      else queue.push([where, bytes]);
      return;
    }
    take(where, bytes);
  };
  const reached = (): void => {
    if (!d.device.gatt!.connected) {
      throw new DOMException("GATT Server is disconnected. Cannot perform GATT operations. (Re)connect first with `device.gatt.connect`.", "NetworkError");
    }
  };
  update.onWrite = (bytes) => {
    reached();
    const op = bytes[0]!;
    if (turnedAway.has(op)) throw new DOMException("GATT operation failed for unknown reason.", "NetworkError");
    trace.push(P.decodeUpdateRequest(bytes).opName!);
    if (options.dropAt === op) {
      d.drop();
      return;
    }
    arrive("update", bytes);
  };
  data.onWrite = (bytes) => {
    reached();
    trace.push(bytes.length);
    writes += 1;
    if (options.refuseOnce === writes) {
      options.refuseOnce = null;
      throw new DOMException("GATT operation not permitted.", "NotSupportedError");
    }
    if (bytes.length <= size - 3) arrive("data", bytes);
    else if (way === "refuses") throw new DOMException("GATT operation not permitted.", "NotSupportedError");
    else if (way === "cuts") arrive("data", bytes.subarray(0, size - 3));
  };
  const heard = ({ taken }: { taken: number }): void => {
    trace.push(`taken ${taken}`);
  };
  return { ...d, held, trace, silent, turnedAway, options, heard, thrown: () => thrown };
}

/**
 * The requests and writes a transfer of `total` bytes made before a link's
 * size was asked about, to a device that threw nothing away, with each
 * progress the transfer reports, as the Python library reports it: START,
 * writes of 244 bytes with a progress check after every eighth and after the
 * last, the count after each check, and FINISH.
 */
function today(total: number): (string | number)[] {
  const out: (string | number)[] = ["start"];
  let sent = 0;
  let since = 0;
  while (sent < total) {
    const n = Math.min(244, total - sent);
    out.push(n);
    sent += n;
    since += 1;
    if (since === 8 || sent === total) {
      out.push("query", `taken ${sent}`);
      since = 0;
    }
  }
  out.push("finish");
  return out;
}

/** The data writes of `trace` from `from` on, by the bytes sent. */
const writesOf = (trace: readonly (string | number)[], from = 0): number[] => trace.slice(from).filter((x): x is number => typeof x === "number");

/** An image of 30,000 bytes, which crosses several of a device's pages. */
const LARGE = Uint8Array.from({ length: 30_000 }, (_, i) => (i * 7 + 13) & 0xff);

test("a link that carries 244 bytes sees through the browser exactly the requests, writes and progress a transfer made before a link's size was asked about, whatever it does with a longer write", async () => {
  for (const way of WAYS) {
    const d = sizedLink(247, way);
    const link = await connect(d.device);
    assert.equal(await link.transfer(Transfer.app(link.session, 0, LARGE), LARGE, d.heard), "verified", way);
    assert.deepEqual(d.trace, today(LARGE.length), way);
    assert.equal(d.thrown(), 0, way);
    assert.deepEqual(Uint8Array.from(d.held), LARGE, way);
  }
});

test("through the browser, on a link of 104 that refuses, drops or cuts a longer write, a transfer starts the file again at 101 bytes after ABORT's answer, its progress the count the device's answer to START gives, and is verified", async () => {
  for (const way of WAYS) {
    const d = sizedLink(104, way);
    const link = await connect(d.device);
    assert.equal(await link.transfer(Transfer.app(link.session, 0, LARGE), LARGE, d.heard), "verified", way);
    assert.deepEqual(Uint8Array.from(d.held), LARGE, `${way}: the device holds exactly what was sent`);
    assert.equal(d.trace[1], 244, `${way}: the first write is 244 bytes`);
    const again = d.trace.indexOf("abort");
    assert.ok(again > 0 && d.trace.lastIndexOf("abort") === again, `${way}: the file started again once`);
    assert.deepEqual(d.trace.slice(again, again + 3), ["abort", "start", "taken 0"], `${way}: ABORT, START, and progress from the device's answer to it`);
    const after = writesOf(d.trace, again);
    assert.ok(after[0] === 101 && after.every((n) => n <= 101), `${way}: then 101 bytes at a time`);
    assert.equal(after.reduce((x, y) => x + y, 0), LARGE.length, `${way}: from the first byte, each byte once`);
    assert.deepEqual(d.trace.slice(-2), [`taken ${LARGE.length}`, "finish"], way);
  }
});

test("through the browser, on a link of 23 that refuses, drops or cuts a longer write, a transfer starts the file again at 101 bytes and then at 20, and is verified", async () => {
  for (const way of WAYS) {
    const d = sizedLink(23, way);
    const link = await connect(d.device);
    assert.equal(await link.transfer(Transfer.app(link.session, 0, IMAGE_5000), IMAGE_5000, d.heard), "verified", way);
    assert.deepEqual(Uint8Array.from(d.held), IMAGE_5000, way);
    assert.equal(d.trace.filter((x) => x === "abort").length, 2, `${way}: the file started again twice`);
    const after = writesOf(d.trace, d.trace.lastIndexOf("abort"));
    assert.ok(after[0] === 20 && after.every((n) => n <= 20), way);
    assert.equal(after.reduce((x, y) => x + y, 0), IMAGE_5000.length, way);
  }
});

test("on a link that cuts writes, progress is reported only for checks that find the device holding what was sent, and for the START answer that starts the file again", async () => {
  // Nothing is reported for a size whose writes did not arrive whole, and
  // when the file starts again the progress goes back to 0, as the Python
  // library reports it.
  const img = enveloped(24 * 256);
  for (const [size, last, restarts] of [[104, 101, 1], [23, 20, 2]] as const) {
    const d = sizedLink(size, "cuts");
    d.options.envelopes = true;
    const link = await connect(d.device);
    const heard: number[] = [];
    assert.equal(await link.transfer(Transfer.app(link.session, 0, img), img, ({ taken, total }) => {
      assert.equal(total, img.length);
      heard.push(taken);
    }), "verified", String(size));
    assert.deepEqual(Uint8Array.from(d.held), img, String(size));
    const checks: number[] = [];
    for (let k = 8 * last; k < img.length; k += 8 * last) checks.push(k);
    checks.push(img.length);
    assert.deepEqual(heard, [...Array.from({ length: restarts }, () => 0), ...checks], String(size));
  }
});

test("through the browser, where 20 bytes fail too, a transfer ends in the error and the words it ends in on any link, after ABORT", async () => {
  // A link that carries 17 bytes, less than every size a transfer starts
  // again at.
  const refusing = sizedLink(20, "refuses");
  let link = await connect(refusing.device);
  await assert.rejects(link.transfer(Transfer.app(link.session, 0, IMAGE_5000), IMAGE_5000), (e: Error) => e.name === "NotSupportedError");
  assert.equal(refusing.trace.at(-1), "abort", "the device is told to drop what it holds");
  const dropping = sizedLink(20, "drops");
  link = await connect(dropping.device);
  await assert.rejects(link.transfer(Transfer.app(link.session, 0, IMAGE_5000), IMAGE_5000), (e: unknown) => {
    assert.ok(e instanceof OffsetMismatch, String(e));
    assert.equal(e.message, `the device accepted ${e.device} bytes where ${e.sent} were sent, and the transfer is not advancing`);
    return true;
  });
  const cutting = sizedLink(20, "cuts");
  link = await connect(cutting.device);
  await assert.rejects(link.transfer(Transfer.app(link.session, 0, IMAGE_5000), IMAGE_5000), (e: unknown) => {
    assert.ok(e instanceof ChecksumMismatch, String(e));
    assert.match(e.message, /^the device's checksum over \d+ bytes is 0x[0-9a-f]{8} and ours is 0x[0-9a-f]{8}$/);
    return true;
  });
  for (const d of [refusing, dropping, cutting]) assert.equal(d.trace.filter((x) => x === "abort").length, 3, "two starts again, and the abort of the end");
});

test("a later transfer on the same connection starts at the size the last one finished at, and one on the connection connected again starts at 244", async () => {
  const d = sizedLink(104, "refuses");
  const link = await connect(d.device);
  assert.equal(await link.transfer(Transfer.app(link.session, 0, IMAGE_5000), IMAGE_5000), "verified");
  assert.equal(d.trace.filter((x) => x === "abort").length, 1);
  let from = d.trace.length;
  assert.equal(await link.transfer(Transfer.app(link.session, 0, IMAGE_5000), IMAGE_5000), "verified");
  assert.ok(!d.trace.slice(from).includes("abort"), "nothing started again");
  assert.equal(writesOf(d.trace, from)[0], 101, "101 bytes from the first write");
  await link.connect();
  from = d.trace.length;
  assert.equal(await link.transfer(Transfer.app(link.session, 0, IMAGE_5000), IMAGE_5000), "verified");
  assert.equal(writesOf(d.trace, from)[0], 244, "a new link starts at 244 again");
  assert.equal(d.trace.slice(from).filter((x) => x === "abort").length, 1);
});

test("a transfer that starts the file again carries the data from the count the device's answer to START gives, where its ABORT never reached the device", async () => {
  // The tenth write is refused while the link is up, and the browser turns
  // away the ABORT that follows, so the device still holds the nine writes
  // before it and says so when asked to start again.
  const d = sizedLink(247, "refuses");
  d.options.refuseOnce = 10;
  d.turnedAway.add(P.UPDATE_OPS.abort);
  const link = await connect(d.device);
  assert.equal(await link.transfer(Transfer.app(link.session, 0, LARGE), LARGE, d.heard), "verified");
  const start = d.trace.lastIndexOf("start");
  assert.deepEqual(d.trace.slice(start - 1, start + 2), [244, "start", `taken ${9 * 244}`], "START again, and progress from the device's count");
  const after = writesOf(d.trace, start);
  assert.ok(after[0] === 101 && after.every((n) => n <= 101), "then 101 bytes at a time");
  assert.equal(after.reduce((x, y) => x + y, 0), LARGE.length - 9 * 244, "from the device's count, and nothing past the file's length");
  assert.deepEqual(Uint8Array.from(d.held), LARGE, "each byte once");
});

test("a link that goes down during the ABORT of a transfer starting the file again at a smaller size ends the transfer in the error that stopped the larger size, as the library's transfer ends", async (t) => {
  const refusing = sizedLink(104, "refuses");
  refusing.options.dropAt = P.UPDATE_OPS.abort;
  let link = await connect(refusing.device);
  let got = await onMockedTime(t, () => link.transfer(Transfer.app(link.session, 0, IMAGE_5000), IMAGE_5000), 10_000);
  assert.ok(got instanceof Error && got.name === "NotSupportedError", `the refused write's own error, not ${String(got)}`);
  assert.equal(refusing.trace.at(-1), "abort", "nothing more is sent");
  const dropping = sizedLink(104, "drops");
  dropping.options.dropAt = P.UPDATE_OPS.abort;
  link = await connect(dropping.device);
  got = await onMockedTime(t, () => link.transfer(Transfer.app(link.session, 0, IMAGE_5000), IMAGE_5000), 10_000);
  assert.ok(got instanceof OffsetMismatch, String(got));
  assert.equal(got.message, `the device accepted 0 bytes where ${got.sent} were sent, and the transfer is not advancing`);
  assert.equal(dropping.trace.at(-1), "abort", "nothing more is sent");
  // A link of 23 that cuts writes to 20 bytes brings the device an envelope
  // made of two writes, which it refuses: its checksum is over the 20 bytes
  // of each, and ours over the first write, of 244.
  const cutting = sizedLink(23, "cuts");
  cutting.options.envelopes = true;
  cutting.options.dropAt = P.UPDATE_OPS.abort;
  link = await connect(cutting.device);
  const img = enveloped(24 * 256);
  got = await onMockedTime(t, () => link.transfer(Transfer.app(link.session, 0, img), img), 10_000);
  const hex = (n: number): string => `0x${(n >>> 0).toString(16).padStart(8, "0")}`;
  const theirs = crc32(Uint8Array.from([...img.subarray(0, 20), ...img.subarray(244, 264)]));
  assert.ok(got instanceof ChecksumMismatch, String(got));
  assert.equal(got.message, `the device's checksum over 244 bytes is ${hex(theirs)} and ours is ${hex(crc32(img.subarray(0, 244)))}`);
  assert.equal(cutting.trace.at(-1), "abort", "nothing more is sent");
});

test("an ABORT the device does not answer while a transfer starts the file again is waited for five seconds and passed over, and the transfer goes on to be verified", async (t) => {
  const d = sizedLink(104, "refuses");
  d.silent.add(P.UPDATE_OPS.abort);
  const link = await connect(d.device);
  t.mock.timers.enable({ apis: ["setTimeout"] });
  try {
    let settled: { readonly value: unknown } | null = null;
    link.transfer(Transfer.app(link.session, 0, IMAGE_5000), IMAGE_5000).then(
      (value) => (settled = { value }),
      (value: unknown) => (settled = { value }),
    );
    let asked: number | null = null;
    let started: number | null = null;
    for (let ms = 0; settled === null && ms <= 10_000; ms += 100) {
      await new Promise((resolve) => setImmediate(resolve));
      if (asked === null && d.trace.includes("abort")) asked = ms;
      if (asked !== null && started === null && d.trace.lastIndexOf("start") > d.trace.indexOf("abort")) started = ms;
      t.mock.timers.tick(100);
    }
    await new Promise((resolve) => setImmediate(resolve));
    assert.ok(settled !== null, "still waiting after ten seconds");
    assert.equal((settled as { readonly value: unknown }).value, "verified");
    assert.ok(asked !== null && started !== null, "ABORT and then START were asked");
    assert.ok(started - asked >= 5_000 && started - asked < 5_500, `START was asked ${started - asked} ms after ABORT`);
  } finally {
    t.mock.timers.reset();
  }
});

test("a daisy chain's resync asks nothing of a device while a transfer to it is under way, and asks again after", async () => {
  const { said, during, before, after, d, img } = await besideTheResync();
  assert.equal(said, "verified");
  assert.ok(before, "the resync never asked for the device's clock before the transfer");
  const others = during.filter(([where]) => where !== "data" && where !== "update");
  assert.equal(others.length, 0, `${others.length} request(s) besides the transfer's reached the device during it`);
  assert.deepEqual(Uint8Array.from(d.held), img, "every byte arrived once, in order");
  assert.equal(d.thrown(), 0, "nothing was thrown away");
  assert.ok(after, "the resync did not ask for the device's clock again after the transfer");
});

test("a transfer that fails lets a daisy chain's resync ask again", async () => {
  const { said, during, after } = await besideTheResync(7);
  assert.equal(said, "the device refused the image: security counter");
  const others = during.filter(([where]) => where !== "data" && where !== "update");
  assert.equal(others.length, 0, `${others.length} request(s) besides the transfer's reached the device during it`);
  assert.ok(after, "the resync did not ask for the device's clock again after the transfer failed");
});
