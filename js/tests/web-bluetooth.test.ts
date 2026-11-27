/**
 * The browser transport, against a device made of nothing.
 *
 * `connect` takes the device object the chooser handed back, so the whole
 * transport can be driven by a stand-in that answers the way a device
 * answers, behind a browser that keeps subscriptions as Chrome keeps them
 * (see `browser.ts`). What is left needing a real browser is the chooser
 * itself.
 */

import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import test from "node:test";

import { hostClocks } from "../src/capture/clock.ts";
import * as J from "../src/capture/pyjson.ts";
import * as P from "../src/protocol.ts";
import { NoBluetooth, connect, requestDevice } from "../src/web-bluetooth.ts";
import { SessionError, type Event } from "../src/session.ts";
import { safeStop } from "../src/run.ts";
import { Transfer } from "../src/transfer.ts";
import { Browser } from "./browser.ts";

const VECTORS = JSON.parse(readFileSync(new URL("../../contract/conformance.json", import.meta.url), "utf8"));

function vectorBytes(kind: string): Uint8Array {
  return P.fromHex(VECTORS.decode.find((v: { kind: string }) => v.kind === kind).bytes);
}

class FakeCharacteristic extends EventTarget {
  readonly uuid: string;
  value?: DataView;
  readonly writes: Uint8Array[] = [];
  onWrite?: (bytes: Uint8Array) => void;

  constructor(uuid: string, value?: Uint8Array) {
    super();
    this.uuid = uuid;
    if (value !== undefined) this.value = new DataView(value.buffer.slice(0) as ArrayBuffer);
  }

  readValue(): Promise<DataView> {
    assert.ok(this.value !== undefined, `${this.uuid} has nothing to read`);
    return Promise.resolve(this.value);
  }

  writeValueWithResponse(value: BufferSource): Promise<void> {
    return this.#write(value);
  }

  writeValueWithoutResponse(value: BufferSource): Promise<void> {
    return this.#write(value);
  }

  /** What the device would send on this characteristic. */
  notify(bytes: Uint8Array): void {
    this.value = new DataView(bytes.buffer.slice(0) as ArrayBuffer);
    this.dispatchEvent(new Event("characteristicvaluechanged"));
  }

  /** The device takes the write and acknowledges it, and only then handles it: what it sends in answer comes once the write is done. */
  #write(value: BufferSource): Promise<void> {
    const bytes = ArrayBuffer.isView(value)
      ? new Uint8Array(value.buffer, value.byteOffset, value.byteLength).slice()
      : new Uint8Array(value).slice();
    this.writes.push(bytes);
    setImmediate(() => this.onWrite?.(bytes));
    return Promise.resolve();
  }
}

function fakeDevice() {
  const characteristics = new Map<string, FakeCharacteristic>([
    [P.DEVICE_INFO, new FakeCharacteristic(P.DEVICE_INFO, vectorBytes("device_info"))],
    [P.CONTROL, new FakeCharacteristic(P.CONTROL)],
    [P.CONTROL_RESPONSE, new FakeCharacteristic(P.CONTROL_RESPONSE)],
    [P.EEG_DATA, new FakeCharacteristic(P.EEG_DATA)],
    [P.STATUS, new FakeCharacteristic(P.STATUS)],
  ]);
  // This device carries no predictions characteristic, which is what a
  // device without a model runtime looks like.
  const browser = new Browser(characteristics, { id: "fake", name: "IntoMind-0001", service: P.SERVICE });
  return { device: browser.device, server: browser.device.gatt!, characteristics, browser };
}

test("a page with no Web Bluetooth says so rather than failing later", () => {
  assert.throws(() => requestDevice(), NoBluetooth);
});

test("connecting reads what the device is and subscribes to what it notifies", async () => {
  const { device, characteristics, browser } = fakeDevice();
  const events: Event[] = [];
  const link = await connect(device, { onEvent: (e) => events.push(e) });

  assert.equal(link.info.channels, 4);
  assert.ok(link.connected);
  assert.ok(browser.held(P.EEG_DATA));
  assert.ok(browser.held(P.STATUS));

  // A command goes to the characteristic the session named.
  await link.send(link.session.startStream());
  assert.deepEqual(characteristics.get(P.CONTROL)!.writes, [new Uint8Array([0x01])]);

  // And a notification becomes an event.
  characteristics.get(P.EEG_DATA)!.notify(vectorBytes("eeg_data"));
  characteristics.get(P.STATUS)!.notify(vectorBytes("status"));
  assert.deepEqual(events.map((e) => e.type), ["samples", "status"]);

  await link.disconnect();
  assert.ok(!link.connected);
  // Every subscription ends before the link drops: on some platforms the
  // browser keeps a subscribed link open, and a later program on the same
  // computer would then hear every packet twice.
  for (const uuid of characteristics.keys()) {
    assert.ok(!browser.held(uuid), `still notifying: ${uuid}`);
  }
});

test("a time exchange is bracketed by the host clock", async () => {
  const { device, characteristics } = fakeDevice();
  const link = await connect(device);
  const control = characteristics.get(P.CONTROL)!;
  const answers = characteristics.get(P.CONTROL_RESPONSE)!;

  // The device answers a time request with its own clock, in its own ticks.
  control.onWrite = (bytes) => {
    if (bytes[0] !== P.OPCODES.time_sync) return;
    const ticks = new Uint8Array(8);
    new DataView(ticks.buffer).setBigUint64(0, 1_234_567_890n, true);
    answers.notify(P.encodeResponse(P.OPCODES.time_sync, 0, ticks));
  };

  const fit = await link.timeSync();
  assert.ok(fit !== null);
  assert.equal(fit.exchanges, 1);
  assert.ok(!fit.skewUsed, "one exchange claims no rate");
  // The device's instant lands inside the window the exchange bracketed.
  const mapped = link.session.hostTime(1_234_567_890n)!;
  const exchange = link.session.timebase!.exchanges[0]!;
  assert.ok(mapped >= exchange.before && mapped <= exchange.after);
});

test("a time exchange in the browser reaches the clock a recording is dated with, on both host clocks", async () => {
  const { device, characteristics } = fakeDevice();
  const link = await connect(device);
  const control = characteristics.get(P.CONTROL)!;
  const answers = characteristics.get(P.CONTROL_RESPONSE)!;
  control.onWrite = (bytes) => {
    if (bytes[0] !== P.OPCODES.time_sync) return;
    const ticks = new Uint8Array(8);
    new DataView(ticks.buffer).setBigUint64(0, 1_234_567_890n, true);
    answers.notify(P.encodeResponse(P.OPCODES.time_sync, 0, ticks));
  };

  const clock = link.session.clock!;
  assert.equal(clock.syncs.length, 0);
  const before = hostClocks();
  await link.timeSync();
  await link.timeSync();
  const after = hostClocks();
  assert.equal(clock.syncs.length, 2, "every exchange reaches the recording's clock");
  for (const s of clock.syncs) {
    assert.equal(s.device, 1_234_567_890 / link.info.tickHz);
    assert.ok(s.mono >= before.monotonic && s.mono <= after.monotonic, "the monotonic clock brackets it");
    assert.ok(s.host >= before.realtime && s.host <= after.realtime, "the wall clock brackets it");
  }
  assert.equal(link.session.timebase!.exchanges.length, 2, "the published fit still takes each exchange once");
});

test("a recording made through the browser connection is dated by the device's clock and comes back as its files", async () => {
  const { device, characteristics } = fakeDevice();
  const link = await connect(device);
  const control = characteristics.get(P.CONTROL)!;
  const answers = characteristics.get(P.CONTROL_RESPONSE)!;
  const data = characteristics.get(P.EEG_DATA)!;
  const tickHz = link.info.tickHz;
  // The device's clock reads 100 s when the test begins and runs with the host's.
  const t0 = hostClocks().monotonic;
  const ticksNow = () => BigInt(Math.round((hostClocks().monotonic - t0 + 100) * tickHz));
  control.onWrite = (bytes) => {
    const op = bytes[0]!;
    if (op === P.OPCODES.time_sync) {
      const ticks = new Uint8Array(8);
      new DataView(ticks.buffer).setBigUint64(0, ticksNow(), true);
      answers.notify(P.encodeResponse(op, 0, ticks));
    } else if (op === P.OPCODES.start_stream || op === P.OPCODES.stop_stream) {
      answers.notify(P.encodeResponse(op, 0));
    }
  };
  for (let i = 0; i < 3; i++) await link.timeSync();
  link.session.armRecorder();
  await link.request(link.session.startStream());
  assert.ok(link.session.streaming);
  const template = P.decodePacket(vectorBytes("eeg_data"), link.info.channels);
  const per = template.counts.length;
  const start = ticksNow();
  for (let k = 0; k < 5; k++) {
    const deviceTime = start + BigInt(Math.round((k * per * tickHz) / P.RATE_BY_CODE[template.rateCode]!));
    data.notify(P.encodePacket({ ...template, index: k * per, deviceTime, discontinuity: false, usbPresent: false, samplesLostBefore: 0 }));
  }
  // Saved while the stream runs, as the Python library's record-while-streaming session saves.
  const saved = link.session.saveRecording("browser-take");
  await link.request(link.session.stopStream());
  assert.ok(!link.session.streaming);

  const names = saved.files.map(([name]) => name);
  assert.ok(names.includes("browser-take.npz") && names.includes("browser-take.meta.json"), names.join(", "));
  const m = saved.manifest;
  assert.equal(J.asI64(J.field(m, "n")), BigInt(5 * per));
  const clock = J.field(m, "clock");
  assert.equal(J.asI64(J.field(clock, "n_syncs")), 3n, "the three exchanges dated the recording");
  assert.equal(J.asStr(J.field(clock, "fitted_against")), "monotonic");
  assert.ok(J.truthy(J.field(clock, "map_frozen_at_stream_start")), "the map held still while the stream ran");
});

test("a characteristic the device does not carry is not written to", async () => {
  const { device } = fakeDevice();
  const link = await connect(device);
  await assert.rejects(
    () => link.send({ characteristic: P.FILL.PREDICTIONS, bytes: new Uint8Array([1]), withResponse: true }),
    /has no characteristic/,
  );
});

test("a command written or a status read on a link that is down is refused as the browser refuses it, and never reaches the device", async () => {
  const { device, server, characteristics } = fakeDevice();
  const link = await connect(device);
  server.disconnect();
  const refused = (e: Error): boolean => e.name === "NetworkError" && e.message.startsWith("GATT Server is disconnected.");
  await assert.rejects(link.send(link.session.startStream()), refused);
  await assert.rejects(link.readStatus(), refused);
  assert.deepEqual(characteristics.get(P.CONTROL)!.writes, [], "nothing reached the device");
});

test("nothing the device notifies reaches a page while the link is down, and what it notifies once the link is up again does", async () => {
  // The browser stand-in keeps to it as Chrome does with a real device: no
  // notification crosses a link that is down.
  const { server, characteristics, browser } = fakeDevice();
  await server.connect();
  const handed = await (await server.getPrimaryService(P.SERVICE)).getCharacteristic(P.EEG_DATA);
  await handed.startNotifications();
  let heard = 0;
  handed.addEventListener("characteristicvaluechanged", () => heard++);
  const data = characteristics.get(P.EEG_DATA)!;
  data.notify(new Uint8Array([1]));
  browser.drop();
  data.notify(new Uint8Array([2]));
  assert.equal(heard, 1, "nothing while the link is down");
  browser.back();
  data.notify(new Uint8Array([3]));
  assert.equal(heard, 2, "heard again once it is up");
});

test("starting and stopping a stream do what the Python library's Device.start and Device.stop do", async () => {
  const { device, server, characteristics } = fakeDevice();
  const link = await connect(device);
  const control = characteristics.get(P.CONTROL)!;
  const answers = characteristics.get(P.CONTROL_RESPONSE)!;
  let refuse: number | null = null;
  control.onWrite = (bytes) => answers.notify(Uint8Array.of(bytes[0]!, bytes[0] === refuse ? 7 : 0));
  const written = (): number[][] => control.writes.splice(0).map((b) => [...b]);

  // As many samples a packet as fit one notification at four channels, the
  // count started again, the chain in force read, and the stream started.
  await link.start({ samplesPerPacket: 100 });
  assert.deepEqual(written(), [[0x41, 18], [0x50], [0x91], [0x01]]);
  assert.ok(link.session.streaming);
  await link.stop();
  assert.deepEqual(written(), [[0x02]]);
  assert.ok(!link.session.streaming);
  await link.start();
  assert.deepEqual(written()[0], [0x41, 18], "as many as fit when not asked");
  await link.start({ samplesPerPacket: 5, newEpoch: false });
  assert.deepEqual(written(), [[0x41, 5], [0x91], [0x01]], "the count runs on when asked to");
  await link.start({ samplesPerPacket: 0 });
  assert.deepEqual(written()[0], [0x41, 1], "at least one");

  // Refused in the library's words, and a stream that ended on USB power is
  // no longer said to have once a start is asked for.
  link.session.endedByUsb = true;
  refuse = 0x41;
  await assert.rejects(
    link.start(),
    (e: Error) =>
      e instanceof SessionError &&
      e.message === "Refused: op 0x41 -> status 7 (usb power): the device does not stream the natural signal while plugged in. Unplug it to record",
  );
  assert.ok(!link.session.endedByUsb);
  refuse = null;
  written();

  // A link that is down is refused before anything is written, and the
  // stream is taken as stopped all the same.
  await link.start();
  written();
  server.disconnect();
  await assert.rejects(link.stop(), (e: Error) => e instanceof SessionError && e.message === "ConnectionError: BLE link is down");
  assert.deepEqual(written(), []);
  assert.ok(!link.session.streaming);
});

test("a head's slot that is not a whole number from 0 to 255 is refused in the library's words before anything is sent, by uploadHead, selectHead, removeHead, a transfer and every request that carries it", async () => {
  // The slot is one byte, and 256 went to the device as slot 0 with no word.
  const info = P.encodeDeviceInfo({
    protocol: [1, 0], firmware: [1, 0, 0], hardware: [1, 1, 5], channels: 4, adcBits: 24, tickHz: 1_000_000, vrefUv: 4_500_000,
    capabilities: 0xffff, supportedRates: 0b111, deviceId: "0102030405060708", extensionPresent: true, firmwareBuildId: "0000000000000000",
    modelEmbedDim: 96, modelNativeSps: 500, modelWindowSamples: 2000, headSlots: 4, headMaxOutputs: 32, headSlotBytes: 4096,
    updateChunkMax: 244, appSlotBytes: 237_568, weightsImageBytes: 507_904,
  } as never);
  const characteristics = new Map<string, FakeCharacteristic>([
    [P.DEVICE_INFO, new FakeCharacteristic(P.DEVICE_INFO, info)],
    [P.CONTROL, new FakeCharacteristic(P.CONTROL)],
    [P.CONTROL_RESPONSE, new FakeCharacteristic(P.CONTROL_RESPONSE)],
    [P.EEG_DATA, new FakeCharacteristic(P.EEG_DATA)],
    [P.STATUS, new FakeCharacteristic(P.STATUS)],
    [P.UPDATE_CONTROL, new FakeCharacteristic(P.UPDATE_CONTROL)],
    [P.UPDATE_DATA, new FakeCharacteristic(P.UPDATE_DATA)],
  ]);
  const link = await connect(new Browser(characteristics, { id: "fake", name: "IntoMind-0001", service: P.SERVICE }).device);
  const control = characteristics.get(P.CONTROL)!;
  control.onWrite = (bytes) => characteristics.get(P.CONTROL_RESPONSE)!.notify(Uint8Array.of(bytes[0]!, 0));
  const blob = P.buildHead(Array.from({ length: 2 }, () => Array.from({ length: 96 }, () => 1)), [0, 0], [1, 1], { name: "focus" });
  const slots: [unknown, string][] = [
    [256, "256"], [-1, "-1"], [1.5, "1.5"], ["1", '"1"'], [null, "null"], [true, "true"], [undefined, "undefined"], [10n ** 5000n, "a whole number of 5001 digits"],
  ];
  for (const [slot, said] of slots) {
    const words = `a slot is a whole number from 0 to 255, not ${said}`;
    const refused = (e: unknown): boolean => e instanceof P.Invalid && e.message === words;
    const s = slot as number;
    await assert.rejects(link.uploadHead(blob, s), refused, `uploadHead ${said}`);
    await assert.rejects(link.selectHead(s), refused, `selectHead ${said}`);
    await assert.rejects(link.removeHead(s), refused, `removeHead ${said}`);
    assert.throws(() => Transfer.head(link.session, s, blob), refused, `Transfer.head ${said}`);
    assert.throws(() => link.session.selectHead(s), refused, `Session.selectHead ${said}`);
    assert.throws(() => link.session.removeHead(s), refused, `Session.removeHead ${said}`);
    assert.throws(() => P.encodeUpdateStart("head", s, blob.length, new Uint8Array(8)), refused, `encodeUpdateStart ${said}`);
    // A request left without its argument is a request given none.
    if (slot !== undefined) assert.throws(() => P.encodeRequest("select_head", s), refused, `encodeRequest ${said}`);
  }
  for (const uuid of [P.CONTROL, P.UPDATE_CONTROL, P.UPDATE_DATA]) assert.deepEqual(characteristics.get(uuid)!.writes, [], `nothing was sent on ${uuid}`);
  // A slot that fits goes as its byte, and whether the device has it is for
  // the device to say.
  await link.selectHead(255);
  await link.removeHead(0);
  assert.deepEqual(control.writes.map((b) => [...b]), [[P.OPCODES.select_head, 255], [P.OPCODES.remove_head, 0]]);
  assert.equal(P.encodeUpdateStart("head", 255, blob.length, new Uint8Array(8))[2], 255);
});

test("a device whose stream USB power ended answers the stop with ok, as the firmware answers any stop while it is not streaming, and safeStop counts its link alive, as the library's _safe_stop does, and a refused stop as no answer", async () => {
  // The firmware never refuses that stop. A stop refused all the same is not
  // taken as an answer that the device ended its stream itself.
  const { device, characteristics } = fakeDevice();
  const link = await connect(device);
  let status = 0;
  characteristics.get(P.CONTROL)!.onWrite = (bytes) => characteristics.get(P.CONTROL_RESPONSE)!.notify(Uint8Array.of(bytes[0]!, status));
  link.session.endedByUsb = true;
  assert.equal(await safeStop(link), true, "answered ok");
  link.session.endedByUsb = true;
  status = 4;
  assert.equal(await safeStop(link), false, "refused");
  assert.deepEqual(characteristics.get(P.CONTROL)!.writes.map((b) => [...b]), [[P.OPCODES.stop_stream], [P.OPCODES.stop_stream]]);
});
