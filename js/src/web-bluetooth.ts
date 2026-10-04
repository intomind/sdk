/**
 * The browser transport.
 *
 * This is the only file that touches `navigator.bluetooth`. Everything
 * else in this package is bytes in and events out, so the protocol, the
 * session, and the timebase are all testable without a browser and without
 * a device.
 *
 * It stays thin on purpose: find a device, connect, resolve the
 * characteristics, subscribe, and hand every notification to the session.
 * What the notifications mean is the session's business.
 */

import {
  DEVICE_INFO,
  FILL,
  OPCODES,
  SERVICE,
  uuid,
  type DeviceInfo,
  type Response,
} from "./protocol.ts";
import { Session, type Command, type Event as SessionEvent } from "./session.ts";
import type { Fit } from "./timebase.ts";
import type { Transfer } from "./transfer.ts";

// The Web Bluetooth surface this file uses, and no more. Declaring it here
// rather than depending on a types package keeps the dependency count at
// zero and states exactly what the transport touches.

/** The part of a GATT characteristic this transport uses. */
export interface BluetoothRemoteGATTCharacteristicLike extends EventTarget {
  /** The characteristic's full identifier. */
  readonly uuid: string;
  /** The last value read or notified. */
  readonly value?: DataView;
  /** Read the current value. */
  readValue(): Promise<DataView>;
  /** Write a value and wait for the device's acknowledgement. */
  writeValueWithResponse(value: BufferSource): Promise<void>;
  /** Write a value without waiting for an acknowledgement. */
  writeValueWithoutResponse(value: BufferSource): Promise<void>;
  /** Subscribe to value-changed notifications. */
  startNotifications(): Promise<BluetoothRemoteGATTCharacteristicLike>;
  /** Unsubscribe from value-changed notifications. */
  stopNotifications(): Promise<BluetoothRemoteGATTCharacteristicLike>;
}

/** The part of a GATT service this transport uses. */
export interface BluetoothRemoteGATTServiceLike {
  /** Resolve one of the service's characteristics by its full identifier. */
  getCharacteristic(uuid: string): Promise<BluetoothRemoteGATTCharacteristicLike>;
}

/** The part of a GATT server this transport uses. */
export interface BluetoothRemoteGATTServerLike {
  /** Whether the link is currently connected. */
  readonly connected: boolean;
  /** Open the link. */
  connect(): Promise<BluetoothRemoteGATTServerLike>;
  /** Drop the link. */
  disconnect(): void;
  /** Resolve the device's primary service by its full identifier. */
  getPrimaryService(uuid: string): Promise<BluetoothRemoteGATTServiceLike>;
}

/** The part of a chosen Bluetooth device this transport uses. */
export interface BluetoothDeviceLike extends EventTarget {
  /** The browser's identifier for this device. */
  readonly id: string;
  /** The device's advertised name, when known. */
  readonly name?: string;
  /** The device's GATT server, when it has one. */
  readonly gatt?: BluetoothRemoteGATTServerLike;
}

/** The part of `navigator.bluetooth` this transport uses. */
export interface BluetoothLike {
  /** Put the browser's device chooser up. */
  requestDevice(options: {
    filters?: Array<{ services?: string[]; namePrefix?: string }>;
    optionalServices?: string[];
    acceptAllDevices?: boolean;
  }): Promise<BluetoothDeviceLike>;
  /** Whether this browser has Bluetooth available at all, when the browser reports it. */
  getAvailability?(): Promise<boolean>;
}

/** Thrown when the page cannot reach Web Bluetooth at all. */
export class NoBluetooth extends Error {
  constructor() {
    super("this page has no Web Bluetooth: it needs a supporting browser, a secure context, and a user gesture");
    this.name = "NoBluetooth";
  }
}

function bluetooth(): BluetoothLike {
  const nav = (globalThis as { navigator?: { bluetooth?: BluetoothLike } }).navigator;
  if (nav?.bluetooth === undefined) throw new NoBluetooth();
  return nav.bluetooth;
}

/**
 * The clock the exchanges are read on, in seconds.
 *
 * Monotonic where the runtime has one, because the fit between the two
 * clocks is a line and a wall clock step would bend it. A caller that puts
 * its own events on the same ruler reads this same function.
 */
export function hostClock(): number {
  const p = (globalThis as { performance?: { now(): number; timeOrigin?: number } }).performance;
  if (p !== undefined) return (p.timeOrigin ?? 0) / 1000 + p.now() / 1000;
  return Date.now() / 1000;
}

/** Options for `requestDevice`. */
export interface RequestOptions {
  /** Narrow the chooser further, for instance to one unit on a bench. */
  namePrefix?: string;
}

/**
 * Put the browser's device chooser up, filtered to devices that advertise
 * the IntoMind service. This has to be called from a user gesture.
 */
export function requestDevice(options: RequestOptions = {}): Promise<BluetoothDeviceLike> {
  const filter: { services: string[]; namePrefix?: string } = { services: [SERVICE] };
  if (options.namePrefix !== undefined) filter.namePrefix = options.namePrefix;
  return bluetooth().requestDevice({ filters: [filter], optionalServices: [SERVICE] });
}

/** Options for `connect`. */
export interface ConnectOptions {
  /** Drive an existing session rather than a new one. */
  session?: Session;
  /** Called with every event, in the order the session produced them. */
  onEvent?: (event: SessionEvent) => void;
  /** Called when the link drops, for whatever reason. */
  onDisconnect?: () => void;
  /** How long a control request waits for its answer. */
  timeoutMs?: number;
}

/** A live link to one device. */
export class Connection {
  /** The device this link is to. */
  readonly device: BluetoothDeviceLike;
  /** The session reading this device's notifications. */
  readonly session: Session;
  /** What the device said it is, read before anything else was asked of it. */
  readonly info: DeviceInfo;

  readonly #server: BluetoothRemoteGATTServerLike;
  readonly #characteristics: Map<number, BluetoothRemoteGATTCharacteristicLike>;
  readonly #listeners = new Set<(event: SessionEvent) => void>();
  readonly #waiting: Array<{ opcode: number; settle: (answer: Response, at: number) => void }> = [];
  /** Whoever is waiting for the next update answer, in order. */
  readonly #updateWaiting: Array<(bytes: Uint8Array) => void> = [];
  readonly #timeoutMs: number;
  /**
   * Verifying a finished image takes the device longer than answering a
   * command, because it hashes and checks a signature over the whole of
   * it before it says anything.
   */
  readonly #updateTimeoutMs = 30_000;

  /** Build one from an open server and the characteristics `connect` resolved. */
  constructor(f: {
    device: BluetoothDeviceLike;
    server: BluetoothRemoteGATTServerLike;
    session: Session;
    info: DeviceInfo;
    characteristics: Map<number, BluetoothRemoteGATTCharacteristicLike>;
    timeoutMs: number;
  }) {
    this.device = f.device;
    this.session = f.session;
    this.info = f.info;
    this.#server = f.server;
    this.#characteristics = f.characteristics;
    this.#timeoutMs = f.timeoutMs;
  }

  /** Whether the link is currently connected. */
  get connected(): boolean {
    return this.#server.connected;
  }

  /** Take every event. The returned function stops taking them. */
  on(listener: (event: SessionEvent) => void): () => void {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  }

  /** Write a command. The session says which characteristic it goes to. */
  async send(command: Command): Promise<void> {
    const c = this.#characteristics.get(command.characteristic);
    if (c === undefined) {
      throw new Error(`this device has no characteristic ${uuid(command.characteristic)}`);
    }
    const bytes = command.bytes.slice();
    if (command.withResponse) await c.writeValueWithResponse(bytes);
    else await c.writeValueWithoutResponse(bytes);
  }

  /** Write a command and wait for the device's answer to it. */
  async request(command: Command): Promise<Response> {
    const answer = this.#await(command.bytes[0]!);
    await this.send(command);
    return (await answer).answer;
  }

  /**
   * One time exchange, with the host clock read either side of it. The
   * device's answer is known to lie between those two readings, which is
   * the whole uncertainty in this one exchange.
   */
  async timeSync(): Promise<Fit | null> {
    const waiting = this.#await(OPCODES.time_sync);
    const before = hostClock();
    await this.send(this.session.timeSync());
    const { answer, at } = await waiting;
    return this.session.onTimeSync(answer.payload, before, at);
  }

  /**
   * Carry an image or a head to the device and have it verified there.
   *
   * Nothing is activated by this. Use `activate` afterwards, and expect
   * the link to drop when you do.
   *
   * ```javascript
   * const t = Transfer.head(link.session, 1, blob);
   * const result = await link.transfer(t, blob, ({ taken, total }) => {
   *   console.log(`${taken} of ${total}`);
   * });
   * ```
   */
  async transfer(
    transfer: Transfer,
    image: Uint8Array,
    onProgress?: (progress: { taken: number; total: number }) => void,
  ): Promise<string | null> {
    try {
      for (;;) {
        const step = transfer.step(image);
        if (step.type === "verified") return step.result;
        if (step.type === "send") {
          const answer = this.#awaitUpdate();
          await this.send(step.command);
          transfer.onAnswer(await answer, image);
          onProgress?.(transfer.progress);
          continue;
        }
        await this.send({
          characteristic: step.characteristic,
          bytes: image.subarray(step.from, step.to),
          withResponse: false,
        });
        transfer.sent(step.to - step.from);
      }
    } catch (e) {
      // A transfer that stopped leaves the device holding a part of
      // something. Tell it to drop that, then report what went wrong.
      try {
        await this.send(transfer.abort());
        this.#updateWaiting.shift();
      } catch {
        // The link is what failed. Nothing further to say to it.
      }
      throw e;
    }
  }

  /** Drop the link. */
  disconnect(): void {
    if (this.#server.connected) this.#server.disconnect();
  }

  #awaitUpdate(): Promise<Uint8Array> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        const i = this.#updateWaiting.indexOf(settle);
        if (i >= 0) this.#updateWaiting.splice(i, 1);
        reject(new Error("the device did not answer the update service in time"));
      }, this.#updateTimeoutMs);
      const settle = (bytes: Uint8Array) => {
        clearTimeout(timer);
        resolve(bytes);
      };
      this.#updateWaiting.push(settle);
    });
  }

  #await(opcode: number): Promise<{ answer: Response; at: number }> {
    return new Promise((resolve, reject) => {
      const waiter = {
        opcode,
        settle: (answer: Response, at: number) => {
          clearTimeout(timer);
          resolve({ answer, at });
        },
      };
      const timer = setTimeout(() => {
        const i = this.#waiting.indexOf(waiter);
        if (i >= 0) this.#waiting.splice(i, 1);
        reject(new Error(`the device did not answer opcode 0x${opcode.toString(16)} in time`));
      }, this.#timeoutMs);
      this.#waiting.push(waiter);
    });
  }

  /** @internal Called by `connect` for every notification that arrives. */
  deliver(characteristic: number, bytes: Uint8Array): void {
    const at = hostClock();
    if (characteristic === FILL.UPDATE_CONTROL) {
      // An update answer is not a session event. It goes to whichever
      // transfer asked for it.
      const settle = this.#updateWaiting.shift();
      if (settle !== undefined) settle(bytes);
      return;
    }
    for (const event of this.session.onNotification(characteristic, bytes)) {
      if (event.type === "answer") {
        const i = this.#waiting.findIndex((w) => w.opcode === event.answer.opcode);
        if (i >= 0) this.#waiting.splice(i, 1)[0]!.settle(event.answer, at);
      }
      for (const listener of this.#listeners) listener(event);
    }
  }
}

/** The characteristics this transport subscribes to, in the order it tries them. */
const NOTIFYING = [
  FILL.EEG_DATA,
  FILL.STATUS,
  FILL.CONTROL_RESPONSE,
  FILL.PREDICTIONS,
  // The update service answers on its own characteristic, by indication.
  // Without this subscription a transfer writes into silence.
  FILL.UPDATE_CONTROL,
];
const WRITABLE = [FILL.CONTROL, FILL.UPDATE_CONTROL, FILL.UPDATE_DATA];

/**
 * Connect to a device the user chose, read what it is, and subscribe to
 * everything it notifies.
 */
export async function connect(device: BluetoothDeviceLike, options: ConnectOptions = {}): Promise<Connection> {
  if (device.gatt === undefined) throw new NoBluetooth();
  const server = await device.gatt.connect();
  const service = await server.getPrimaryService(SERVICE);

  const session = options.session ?? new Session();
  const infoCharacteristic = await service.getCharacteristic(DEVICE_INFO);
  const info = session.onDeviceInfo(asBytes(await infoCharacteristic.readValue()));

  const characteristics = new Map<number, BluetoothRemoteGATTCharacteristicLike>();
  characteristics.set(FILL.DEVICE_INFO, infoCharacteristic);
  // A device carries the characteristics its firmware offers, so one that
  // is absent is not an error. What is absent shows up as a capability the
  // device never claimed.
  for (const fill of [...NOTIFYING, ...WRITABLE]) {
    const c = await optional(service, fill);
    if (c !== null) characteristics.set(fill, c);
  }

  const connection = new Connection({ device, server, session, info, characteristics, timeoutMs: options.timeoutMs ?? 5000 });
  if (options.onEvent !== undefined) connection.on(options.onEvent);

  for (const fill of NOTIFYING) {
    const c = characteristics.get(fill);
    if (c === undefined) continue;
    c.addEventListener("characteristicvaluechanged", (event: Event) => {
      const value = (event.target as BluetoothRemoteGATTCharacteristicLike).value;
      if (value !== undefined) connection.deliver(fill, asBytes(value));
    });
    await c.startNotifications();
  }

  if (options.onDisconnect !== undefined) {
    device.addEventListener("gattserverdisconnected", options.onDisconnect, { once: true });
  }
  return connection;
}

/** Ask for a device and connect to it, which is what most callers want. */
export async function requestAndConnect(options: RequestOptions & ConnectOptions = {}): Promise<Connection> {
  return connect(await requestDevice(options), options);
}

async function optional(
  service: BluetoothRemoteGATTServiceLike,
  fill: number,
): Promise<BluetoothRemoteGATTCharacteristicLike | null> {
  try {
    return await service.getCharacteristic(uuid(fill));
  } catch {
    return null;
  }
}

function asBytes(value: DataView): Uint8Array {
  return new Uint8Array(value.buffer, value.byteOffset, value.byteLength).slice();
}
