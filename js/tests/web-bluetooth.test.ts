/**
 * The browser transport, against a device made of nothing.
 *
 * `connect` takes the device object the chooser handed back, so the whole
 * transport can be driven by a stand-in that answers the way a device
 * answers. What is left needing a real browser is the chooser itself.
 */

import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import test from "node:test";

import * as P from "../src/protocol.ts";
import { NoBluetooth, connect, requestDevice } from "../src/web-bluetooth.ts";
import type { Event } from "../src/session.ts";

const VECTORS = JSON.parse(readFileSync(new URL("../../contract/conformance.json", import.meta.url), "utf8"));

function vectorBytes(kind: string): Uint8Array {
  return P.fromHex(VECTORS.decode.find((v: { kind: string }) => v.kind === kind).bytes);
}

class FakeCharacteristic extends EventTarget {
  readonly uuid: string;
  value?: DataView;
  notifying = false;
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

  startNotifications(): Promise<this> {
    this.notifying = true;
    return Promise.resolve(this);
  }

  stopNotifications(): Promise<this> {
    this.notifying = false;
    return Promise.resolve(this);
  }

  /** What the device would send on this characteristic. */
  notify(bytes: Uint8Array): void {
    this.value = new DataView(bytes.buffer.slice(0) as ArrayBuffer);
    this.dispatchEvent(new Event("characteristicvaluechanged"));
  }

  #write(value: BufferSource): Promise<void> {
    const bytes = ArrayBuffer.isView(value)
      ? new Uint8Array(value.buffer, value.byteOffset, value.byteLength).slice()
      : new Uint8Array(value).slice();
    this.writes.push(bytes);
    this.onWrite?.(bytes);
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
  let connected = false;
  const server = {
    get connected() {
      return connected;
    },
    connect() {
      connected = true;
      return Promise.resolve(server);
    },
    disconnect() {
      connected = false;
    },
    getPrimaryService(uuid: string) {
      assert.equal(uuid, P.SERVICE);
      return Promise.resolve({
        getCharacteristic(want: string) {
          const c = characteristics.get(want);
          return c === undefined
            ? Promise.reject(new Error(`no characteristic ${want}`))
            : Promise.resolve(c);
        },
      });
    },
  };
  const device = Object.assign(new EventTarget(), { id: "fake", name: "IntoMind-0001", gatt: server });
  return { device, server, characteristics };
}

test("a page with no Web Bluetooth says so rather than failing later", () => {
  assert.throws(() => requestDevice(), NoBluetooth);
});

test("connecting reads what the device is and subscribes to what it notifies", async () => {
  const { device, characteristics } = fakeDevice();
  const events: Event[] = [];
  const link = await connect(device, { onEvent: (e) => events.push(e) });

  assert.equal(link.info.channels, 4);
  assert.ok(link.connected);
  assert.ok(characteristics.get(P.EEG_DATA)!.notifying);
  assert.ok(characteristics.get(P.STATUS)!.notifying);

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
  for (const [fill, c] of characteristics) {
    assert.ok(!c.notifying, `still notifying: ${fill}`);
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

test("a characteristic the device does not carry is not written to", async () => {
  const { device } = fakeDevice();
  const link = await connect(device);
  await assert.rejects(
    () => link.send({ characteristic: P.FILL.PREDICTIONS, bytes: new Uint8Array([1]), withResponse: true }),
    /has no characteristic/,
  );
});
