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

import { hostClocks, type HostClocks } from "./capture/clock.ts";
import * as J from "./capture/pyjson.ts";
import { configAfter } from "./capture/record.ts";
import { cannotCheckContact } from "./firmware.ts";
import {
  DEVICE_INFO,
  EMBEDDING_FORMS,
  EmbeddingAssembler,
  FILL,
  GAIN_BY_CODE,
  IDENTIFY_MAX_SECONDS,
  INDICATOR_LEVELS,
  INDICATOR_LEVEL_NAMES,
  Invalid,
  MODES,
  MODE_BY_CODE,
  NAME_MAX_COMPOSED,
  RATE_BY_CODE,
  SERVICE,
  UPDATE_OPS,
  checkChain,
  composeName,
  decodeCatalog,
  NO_ENCODER_ID,
  decodeConverterRegisters,
  decodeHead,
  decodeHeadEncoders,
  decodeHeads,
  decodeModelInfo,
  decodeModelInterval,
  decodeNameParts,
  decodePipelineState,
  decodeStatus,
  decodeTimeSync,
  decodeUpdateQuery,
  decodeUpdateResponse,
  encodeChain,
  encodeModelInterval,
  encodeNameParts,
  encodeRequest,
  nameFits,
  uuid,
  withEncoders,
  type CatalogEntry,
  type ConverterRegisters,
  type DeviceInfo,
  type EmbeddingWindow,
  type HeadEntry,
  type ModeName,
  type ModelInfo,
  type ModelInterval,
  type OpcodeName,
  type PipelineState,
  type Response,
  type Stage,
  type Status,
} from "./protocol.ts";
import { LINK_DOWN, refusal } from "./refusal.ts";
import { slotRefusal } from "./slot.ts";
import { Session, SessionError, Unanswered, type Command, type Event as SessionEvent } from "./session.ts";
import { embeddingsFor, hand, own } from "./taps.ts";
import { after, within } from "./timers.ts";
import type { Fit } from "./timebase.ts";
import { Refused, Transfer, activate as activation } from "./transfer.ts";

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
    filters?: Array<{ services?: string[]; name?: string; namePrefix?: string }>;
    optionalServices?: string[];
    acceptAllDevices?: boolean;
  }): Promise<BluetoothDeviceLike>;
  /** Whether this browser has Bluetooth available at all, when the browser reports it. */
  getAvailability?(): Promise<boolean>;
  /** The devices this page may reach without asking again, where the browser lists them. */
  getDevices?(): Promise<BluetoothDeviceLike[]>;
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
 * its own events on the same timeline reads this same function.
 */
export function hostClock(): number {
  const p = (globalThis as { performance?: { now(): number; timeOrigin?: number } }).performance;
  if (p !== undefined) return (p.timeOrigin ?? 0) / 1000 + p.now() / 1000;
  return Date.now() / 1000;
}

/** Options for `requestDevice`. */
export interface RequestOptions {
  /** Narrow the chooser to the devices whose advertised name starts with this. */
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
  /** Called each time the link drops, for whatever reason: the link this `connect` opened, and every link the connection's own `connect()` opens again to the same device, until a later `connect` to the device takes it off, so each drop is reported once, by the newest connection. */
  onDisconnect?: () => void;
  /** How long a control request waits for its answer, in milliseconds, five seconds when not given. An infinite wait waits until the answer comes, and a value that is not a number is refused by `connect`. */
  timeoutMs?: number;
}

/** A control answer as it arrived: the answer, and the host's clocks read as it did. */
interface Answered {
  readonly answer: Response;
  readonly at: number;
  readonly clocks: HostClocks;
}

/** A clock that never steps, in milliseconds, for an exchange's deadline. */
function nowMs(): number {
  const p = (globalThis as { performance?: { now(): number } }).performance;
  return p !== undefined ? p.now() : Date.now();
}

/**
 * Report an error a program's own callback threw, without breaking
 * delivery: to the host's `reportError` where it has one, as a browser
 * reports an uncaught error, and to the console where it has not.
 */
function report(e: unknown): void {
  const r = (globalThis as { reportError?: (e: unknown) => void }).reportError;
  if (typeof r === "function") r(e);
  else console.error(e);
}

/** Subscribe a connection to everything its device notifies, by the connection's own hand: shared by `connect` and `Connection.connect`. */
let listen: (connection: Connection) => Promise<void>;

/** A live link to one device. */
export class Connection {
  /**
   * Who carried the samples, as every recording a run saves through a
   * connection names it, the way the Python library's `Device.ADAPTER` names
   * its own: this package's browser transport.
   */
  static readonly ADAPTER = Object.freeze({
    name: "intomind.web-bluetooth",
    version: "1",
    first_party: true,
    transport: "bluetooth-le",
  });

  /** The device this link is to. */
  readonly device: BluetoothDeviceLike;
  /** The session reading this device's notifications. */
  readonly session: Session;
  #info: DeviceInfo;
  /** The Device Info this connection last read, which a `connect` again compares with what the device says then. */
  #infoBytes: Uint8Array;

  #server: BluetoothRemoteGATTServerLike;
  readonly #characteristics: Map<number, BluetoothRemoteGATTCharacteristicLike>;
  readonly #listeners = new Set<(event: SessionEvent) => void>();
  /** The listeners this link put on the characteristics it subscribed to, so a `connect` again leaves none behind. */
  #subscriptions: Array<[BluetoothRemoteGATTCharacteristicLike, (event: Event) => void]> = [];
  /**
   * Control answers that arrived and were not taken yet, the library's
   * `_rsp`. CONTROL EXCHANGES TAKE TURNS, one at a time on a device,
   * because the answers carry no number and come back on one queue: two at
   * once took each other's answers.
   */
  readonly #answers: Answered[] = [];
  /** Wakes the exchange waiting for the next answer. */
  #wake: (() => void) | null = null;
  /** By opcode, the answers still to come for exchanges that stopped waiting, the library's `_owed`, so a late one is told from a later exchange's own. */
  readonly #owed = new Map<number, number>();
  /** Settles when the last exchange to take a turn is done. */
  #turn: Promise<void> = Promise.resolve();
  /** How many exchanges hold or wait for a turn. */
  #pending = 0;
  /** The requests waiting for an update answer, in the order they were written, each with the operation it asked, its first byte. */
  readonly #updateWaiting: Array<{ readonly op: number; readonly settle: (bytes: Uint8Array) => void }> = [];
  readonly #timeoutMs: number;
  /**
   * Verifying a finished image takes the device longer than answering a
   * command, because it hashes and checks a signature over the whole of
   * it before it says anything.
   */
  readonly #updateTimeoutMs = 30_000;
  /**
   * A device answers a progress check as soon as it has written what came
   * before it, so a check unanswered for five seconds is asked again, as
   * the Python library waits and asks.
   */
  readonly #checkTimeoutMs = 5_000;
  /** ABORT's answer is waited for five seconds, as the Python library's `abort_transfer` waits for it. */
  readonly #abortTimeoutMs = 5_000;
  /** ACTIVATE's answer is waited for five seconds, as the Python library's `activate` waits for it. */
  readonly #activateTimeoutMs = 5_000;
  /** The size of write the last transfer on this link finished at, which the next transfer on it starts at, or null before any. */
  #largestWrite: number | null = null;
  /** What `refreshConfig` last read. */
  #config: J.Value | null = null;
  /** The device's stage catalog, read once, as the library keeps it. */
  #catalog: CatalogEntry[] | null = null;
  /** Puts embedding notifications back into whole windows while a form is on. */
  #assembler: EmbeddingAssembler | null = null;
  /** The embeddings subscription, made the first time a form is asked for, as the library makes it. */
  #embeddingsListener: ((event: Event) => void) | null = null;

  /**
   * Called with this connection and every whole window of the encoder's
   * output while a form is on, as the Python library calls
   * `Device.on_embedding(device, window)`. `setEmbeddings` sets it when
   * given one.
   */
  onEmbedding: ((connection: Connection, window: EmbeddingWindow) => void) | null = null;

  /**
   * Called once, with this connection, when the device ends its stream
   * because USB power appeared, as the Python library calls
   * `Device.on_usb_stop(device)`: after the package's own consumers have
   * every sample of the packet, and before any listener hears it.
   */
  onUsbStop: ((connection: Connection) => void) | null = null;

  /** Build one from an open server and the characteristics `connect` resolved. */
  constructor(f: {
    device: BluetoothDeviceLike;
    server: BluetoothRemoteGATTServerLike;
    session: Session;
    info: DeviceInfo;
    infoBytes?: Uint8Array;
    characteristics: Map<number, BluetoothRemoteGATTCharacteristicLike>;
    timeoutMs: number;
  }) {
    this.device = f.device;
    this.session = f.session;
    this.#info = f.info;
    this.#infoBytes = f.infoBytes ?? new Uint8Array(0);
    this.#server = f.server;
    this.#characteristics = f.characteristics;
    this.#timeoutMs = f.timeoutMs;
  }

  /** What the device said it is, read before anything else was asked of it, and again each time it is connected again. */
  get info(): DeviceInfo {
    return this.#info;
  }

  /** Whether the link is currently connected. */
  get connected(): boolean {
    return this.#server.connected;
  }

  /**
   * Whether the device claims a capability, or has one that every device
   * speaking this protocol has, which its Device Info does not list because
   * it never varies: it counts its own samples ("counter"), restarts that
   * count ("reset_count"), starts with a packet size ("packet_size"),
   * exchanges clocks ("clock"), and changes its gain ("gain"), as the
   * library's `Device.can` answers.
   */
  can(capability: string): boolean {
    return EVERY_DEVICE.has(capability) || this.session.can(capability);
  }

  /**
   * Connect again to the same device, as the Python library's
   * `Device.connect` reconnects the same object, and give back this
   * connection. Any link it holds is dropped first. Then the link is
   * opened, the device's characteristics are found again, and everything it
   * notifies is subscribed to again. Its listeners, `onEmbedding`,
   * `onUsbStop` and the `onDisconnect` it was connected with are kept, and
   * so is its session. Answers owed on the link that went are forgotten.
   */
  async connect(): Promise<Connection> {
    await this.disconnect();
    // A new link starts its transfers again at the largest size.
    this.#largestWrite = null;
    const gatt = this.device.gatt;
    if (gatt === undefined) throw new NoBluetooth();
    const server = await gatt.connect();
    const service = await server.getPrimaryService(SERVICE);
    const infoCharacteristic = await service.getCharacteristic(DEVICE_INFO);
    const bytes = asBytes(await infoCharacteristic.readValue());
    // A device that says something new, as one whose firmware was updated
    // does, is read again from the start: its own clock started again when
    // it booted. One that says what it said keeps its session as it is.
    if (!sameBytes(bytes, this.#infoBytes)) {
      this.#info = this.session.onDeviceInfo(bytes);
      this.#infoBytes = bytes;
    }
    const found = await characteristicsOf(service, infoCharacteristic);
    this.#server = server;
    this.#characteristics.clear();
    for (const [fill, c] of found) this.#characteristics.set(fill, c);
    await this.#listen();
    return this;
  }

  /** Lets the `connect` function beside this class subscribe a connection it made, by the connection's own hand. */
  static {
    listen = (connection) => connection.#listen();
  }

  /** Subscribe to everything the device notifies, each notification handed to `deliver`, taking off whatever this link put on before. */
  async #listen(): Promise<void> {
    for (const [c, l] of this.#subscriptions) c.removeEventListener("characteristicvaluechanged", l);
    this.#subscriptions = [];
    for (const fill of NOTIFYING) {
      const c = this.#characteristics.get(fill);
      if (c === undefined) continue;
      const l = (event: Event) => {
        const value = (event.target as BluetoothRemoteGATTCharacteristicLike).value;
        if (value !== undefined) this.deliver(fill, asBytes(value));
      };
      c.addEventListener("characteristicvaluechanged", l);
      this.#subscriptions.push([c, l]);
      // A browser holds one subscription to each characteristic, bound to the
      // object that made it, and keeps it through a link that dropped; asked
      // again on the new link's object it does nothing, and the device is
      // heard on the old one. So whatever it still holds is ended first.
      try {
        await c.stopNotifications();
      } catch {
        // Nothing was held, or nothing could be ended.
      }
      await c.startNotifications();
    }
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

  /**
   * Read the status characteristic: state, mode, gain, rate, contact and the
   * rest. Every device offers it, so it is the one reading of the device's
   * configuration that needs no capability.
   */
  async readStatus(): Promise<Status> {
    const c = this.#characteristics.get(FILL.STATUS);
    if (c === undefined) throw new Error(`this device has no characteristic ${uuid(FILL.STATUS)}`);
    return decodeStatus(asBytes(await c.readValue()));
  }

  /**
   * Write a command and wait for the device's answer to it, as the Python
   * library's `Device.command` does. Exchanges on one connection take
   * turns, so each takes its own answer. When an earlier exchange of the
   * same opcode stopped waiting, the first answer may be its late one, so
   * that answer is held: if another of the opcode follows within the
   * timeout, the one held was late and is dropped, and if none does, it
   * was this exchange's own. So an exchange whose answer never comes costs
   * the next exchange of the same opcode its full timeout, and when both
   * answers come, neither is taken for the other's. An answer that does not
   * come within the connection's timeout throws `Unanswered`.
   */
  async request(command: Command): Promise<Response> {
    return (await this.#taking(() => this.#exchange(command))).answer;
  }

  /**
   * Run `f` once every exchange before it on this connection is done, and
   * hold the turn until `f` is: at once when nobody holds it, as an
   * uncontended lock is taken, and in the order asked otherwise.
   */
  #taking<T>(f: () => Promise<T>): Promise<T> {
    const prior = this.#turn;
    const wait = this.#pending > 0;
    this.#pending += 1;
    let release!: () => void;
    this.#turn = new Promise<void>((resolve) => {
      release = resolve;
    });
    const go = async (): Promise<T> => {
      try {
        return await f();
      } finally {
        this.#pending -= 1;
        release();
      }
    };
    return wait ? prior.then(go) : go();
  }

  /**
   * One control exchange, on the turn the caller already holds, its answer
   * taken by the rule `request` states, as the library's `_exchange` takes
   * it, with the host's clocks read as that answer arrived.
   */
  async #exchange(command: Command): Promise<Answered> {
    if (!this.connected) throw new SessionError(LINK_DOWN);
    const op = command.bytes[0]!;
    // Whatever is waiting answered an earlier exchange: one that stopped
    // waiting, or a copy the system delivered twice.
    while (this.#answers.length > 0) this.#settle(this.#answers.shift()!.answer);
    await this.send(command);
    const deadline = nowMs() + this.#timeoutMs;
    let held: Answered | null = null;
    for (;;) {
      const got = this.#answers.shift() ?? (await this.#arrival(deadline));
      if (got === null) {
        if (held === null) {
          // The answer may still come, and whenever it does it is this
          // exchange's and nobody else's.
          this.#owed.set(op, (this.#owed.get(op) ?? 0) + 1);
          throw new Unanswered(`the device did not answer opcode 0x${op.toString(16)} in time`);
        }
        // Nothing of its opcode followed the answer held, so it was this
        // exchange's own, and the exchange it seemed late for was never
        // answered. Nothing is owed for the opcode now.
        this.#owed.delete(op);
        return held;
      }
      if (got.answer.opcode !== op) {
        this.#settle(got.answer);
        continue;
      }
      if (held !== null) {
        // Another of the opcode followed the one held, so that one was the
        // late answer.
        this.#owed.set(op, (this.#owed.get(op) ?? 0) - 1);
        held = null;
      }
      if ((this.#owed.get(op) ?? 0) > 0) {
        held = got;
        continue;
      }
      return got;
    }
  }

  /**
   * The next answer to arrive, or null once `deadline` passes first. The
   * wait is armed in steps a timer keeps to, so a deadline further off than
   * that is not cut short, and an infinite one waits.
   */
  #arrival(deadline: number): Promise<Answered | null> {
    return new Promise((resolve) => {
      const cancel = after(Math.max(0, deadline - nowMs()) / 1000, () => {
        this.#wake = null;
        resolve(null);
      });
      this.#wake = () => {
        cancel();
        this.#wake = null;
        resolve(this.#answers.shift() ?? null);
      };
    });
  }

  /** Whether `answer` is owed to an exchange that stopped waiting for it, and so is dropped and no longer owed. */
  #settle(answer: Response): boolean {
    const n = this.#owed.get(answer.opcode) ?? 0;
    if (n > 0) {
      this.#owed.set(answer.opcode, n - 1);
      return true;
    }
    return false;
  }

  /**
   * Start streaming, as the Python library's `Device.start` does.
   * `samplesPerPacket` is a preference, held to what fits one notification
   * on a 247-byte link, and as many as fit when not given. The device starts
   * its sample index again at every start, so a stream's index does not
   * carry on from the stream before it, whether `newEpoch` is true or false:
   * a stream's first sample is index 1, or 0 in the synthetic signal. Unless
   * `newEpoch` is false, the index is also reset first, as `resetEpoch`
   * resets it. A link that is down refuses before anything is written, and
   * an answer other than ok throws a `SessionError` in the library's words.
   */
  async start(options: { readonly samplesPerPacket?: number | null; readonly newEpoch?: boolean } = {}): Promise<void> {
    // A data packet must fit one notification: 247 bytes, less 3 of the
    // link's own and 20 of the packet's header, at 3 bytes a channel a
    // sample. Fewer a packet means more packets a second, and a link that
    // cannot carry them loses samples, so the default is as many as fit.
    const fit = Math.floor((247 - 3 - 20) / (3 * this.info.channels));
    const asked = options.samplesPerPacket ?? null;
    const n = asked === null ? fit : Math.max(1, Math.min(asked, fit));
    this.session.endedByUsb = false;
    await this.#ask({ characteristic: FILL.CONTROL, bytes: encodeRequest("set_samples_per_packet", n), withResponse: true });
    if (options.newEpoch ?? true) await this.#ask(this.session.resetEpoch());
    // The chain in force, read before every stream as the library reads it,
    // so a device is asked the same things by either.
    if (this.session.can("pipeline")) await this.#ask(this.session.pipeline());
    await this.#ask(this.session.startStream());
  }

  /** Stop streaming, as the library's `Device.stop` does: the session takes the stream as stopped even when the device does not answer. */
  async stop(): Promise<void> {
    await this.#ask(this.session.stopStream());
  }

  // --- the device's settings, as the Python library's Device sets and reads them ---
  //
  // Each does what the library's method of the same name does: the same
  // requests in the same order, refused in the library's words before
  // anything is written where the library refuses, and thrown in its words
  // when the device does not answer ok.

  /** What `refreshConfig` last read, or null before it has read anything: the library's `Device.config`. */
  get config(): J.Value | null {
    return this.#config;
  }

  /**
   * What the device is set to, read from the device, as the library's
   * `Device.refresh_config` reads it: the status, then the chain, the
   * lamp's level, the model's cadence and the name on a device that claims
   * each. The lead-off setting is the one this host knows, since the
   * protocol has no read for it, and null until it is known.
   */
  async refreshConfig(): Promise<J.Value> {
    const s = this.session;
    const status = await this.readStatus();
    own(this).gain = GAIN_BY_CODE[status.gainCode] || own(this).gain;
    const chain = s.can("pipeline") ? await this.pipeline() : null;
    const indicator = s.can("indicator") ? ((await this.#ask(s.indicator()))[0] ?? null) : null;
    const interval = s.can("model_cadence") ? await this.modelInterval() : null;
    const named = s.can("device_name") ? await this.getName() : null;
    this.#config = configAfter(status, s.leadoffSetting, chain, indicator, interval,
      named === null ? null : { name: named[0], adjective: named[1] });
    return this.#config;
  }

  /**
   * Several time exchanges up front, `gapMs` apart, so the map that dates
   * samples starts with an error bar, as the library's
   * `Device.calibrate_clock` does. Answers with the clock's record. A
   * `gapMs` that is not a number counts as no gap. An infinite `gapMs` never
   * passes, so a calibration of more than one exchange takes its first and
   * then waits forever.
   */
  async calibrateClock(n = 5, gapMs = 150): Promise<J.Value> {
    const gap = Number(gapMs) / 1000;
    for (let i = 0; i < Math.max(1, n); i++) {
      if (i) await new Promise<void>((resolve) => after(Number.isNaN(gap) ? 0 : gap, resolve));
      await this.timeSync();
    }
    return this.session.clock?.quality() ?? J.NULL;
  }

  /**
   * Set the analog gain to one of the contract's gains, as `Device.set_gain`
   * does. From firmware 1.4.6 the device samples at gain 1 while lead-off
   * detection is on, whatever gain was set.
   */
  async setGain(gain: number): Promise<void> {
    if (!GAIN_BY_CODE.includes(gain)) {
      throw new Invalid(`ValueError: this device has no gain ${J.written(gain)}× (it has ${intList(GAIN_BY_CODE)})`);
    }
    await this.#ask(this.session.setGain(gain));
    own(this).gain = gain;
  }

  /** Set the sample rate to one the device declares, as `Device.set_rate` does. */
  async setRate(sps: number): Promise<void> {
    const rates = this.info.rates.length ? this.info.rates : Object.values(RATE_BY_CODE).sort((a, b) => a - b);
    if (!rates.includes(sps)) {
      throw new Invalid(`ValueError: this device has no rate ${J.written(sps)} samples a second (it has ${intList(rates)})`);
    }
    await this.#ask(this.session.setRate(sps));
  }

  /**
   * The signal source, a name or a code, as `Device.set_mode` sets it. Only
   * the synthetic source is a capability this host checks; the device
   * refuses any other it does not offer.
   */
  async setMode(mode: number | ModeName | string): Promise<void> {
    let code: number;
    if (typeof mode === "string") {
      if (!(mode in MODES)) {
        throw new Invalid(`ValueError: the mode is one of ${strList(Object.keys(MODES))}, not ${J.shown(mode)}`);
      }
      code = MODES[mode as ModeName];
    } else {
      code = mode;
    }
    if (code === MODES.synthetic) this.#require("synthetic", "generate a synthetic signal");
    await this.#ask(this.#raw("set_mode", code));
  }

  /**
   * Turn lead-off detection on or off, as `Device.set_leadoff` does. The
   * device accepts this only while it is not streaming. While detection is
   * on, an IntoMind device passes a current through the electrodes, which
   * may distort the signal, and every packet carries the device's verdict,
   * one bit per electrode, set for an electrode that is off. From firmware
   * 1.4.6 the device samples at gain 1 while detection is on, whatever gain
   * was set, and judges each electrode over 0.512 s of samples, and again
   * every 0.512 s. The reference electrode is not judged.
   *
   * An IntoMind device whose firmware is older than 1.4.6 cannot check
   * electrode contact: its lead-off detection does not work, and every
   * electrode reads off whether it touches skin or not. Turning detection
   * on there is refused with a `SessionError` before anything is sent to
   * the device, in these words, with the device's own version: "This
   * device's firmware (1.4.5) cannot check electrode contact. Update it to
   * 1.4.6 or later." Turning it off is not refused.
   */
  async setLeadoff(on: boolean): Promise<void> {
    this.#require("leadoff", "detect lead-off");
    if (on) {
      const refused = cannotCheckContact(this.session.info?.firmware);
      if (refused !== null) throw new SessionError(`RuntimeError: ${refused}`);
    }
    await this.#ask(this.session.setLeadoff(on));
    if (this.#config !== null) J.set(this.#config, "leadoff", J.bool(on));
  }

  /** The stage kinds this device runs, from the device. */
  async pipelineCatalog(): Promise<CatalogEntry[]> {
    this.#require("pipeline", "run a processing chain");
    return decodeCatalog(await this.#ask(this.session.pipelineCatalog()));
  }

  /** The chain in force, and whether it is the device's own default or one a host set. */
  async pipeline(): Promise<PipelineState> {
    this.#require("pipeline", "run a processing chain");
    return decodePipelineState(await this.#ask(this.session.pipeline()));
  }

  /**
   * Put a chain on the device's signal path, checked against the device's
   * rate and catalog first, as `Device.set_pipeline` checks it.
   */
  async setPipeline(stages: readonly Stage[]): Promise<void> {
    this.#require("pipeline", "run a processing chain");
    checkChain(stages, await this.#rateNow(), await this.#catalogOnce());
    await this.#ask(this.#raw("set_pipeline", encodeChain(stages)));
  }

  /** The natural signal, as this host's choice. */
  async clearPipeline(): Promise<void> {
    this.#require("pipeline", "run a processing chain");
    await this.#ask(this.session.clearPipeline());
  }

  /** The device's own default chain for its current rate. */
  async restorePipelineDefault(): Promise<void> {
    this.#require("pipeline", "run a processing chain");
    await this.#ask(this.session.restorePipelineDefault());
  }

  /** The status lamp's level: silent, reserved, or verbose. */
  async indicator(): Promise<string> {
    this.#require("indicator", "have a status lamp a host can set");
    const level = (await this.#ask(this.session.indicator()))[0]!;
    return INDICATOR_LEVEL_NAMES[level] ?? `level ${level}`;
  }

  /** Put the lamp at a level. The device keeps it across power cycles. */
  async setIndicator(level: string): Promise<void> {
    this.#require("indicator", "have a status lamp a host can set");
    if (!(level in INDICATOR_LEVELS)) {
      throw new Invalid(`ValueError: the lamp's level is one of ${strList(Object.keys(INDICATOR_LEVELS))}, not ${J.shown(level)}`);
    }
    await this.#ask(this.#raw("set_indicator", INDICATOR_LEVELS[level as keyof typeof INDICATOR_LEVELS]));
  }

  /** Blink the lamp so this device can be told from others, 1 to 30 seconds, or 0 to stop one in progress. */
  async identify(seconds = 5): Promise<void> {
    this.#require("indicator", "have a status lamp a host can set");
    if (!Number.isInteger(seconds) || seconds < 0 || seconds > IDENTIFY_MAX_SECONDS) {
      throw new Invalid(`ValueError: identify runs 1 to ${IDENTIFY_MAX_SECONDS} seconds, or 0 to stop`);
    }
    await this.#ask(this.#raw("identify", seconds));
  }

  /** How often the model describes a window, and the least interval the device keeps, in seconds. */
  async modelInterval(): Promise<ModelInterval> {
    this.#require("model_cadence", "let a host set the model's cadence");
    return decodeModelInterval(await this.#ask(this.session.modelInterval()));
  }

  /** One described window every `seconds`; 0 is every window the device can. */
  async setModelInterval(seconds: number): Promise<void> {
    this.#require("model_cadence", "let a host set the model's cadence");
    await this.#ask(this.#raw("set_model_interval", encodeModelInterval(seconds)));
  }

  /** The name and the adjective a host set, each possibly empty. */
  async getName(): Promise<[string, string]> {
    this.#require("device_name", "take a name");
    const parts = decodeNameParts(await this.#ask(this.session.getName()));
    return [parts.name, parts.adjective];
  }

  /** Name the device. The composition must fit the 29 bytes a scan response carries. */
  async setName(name: string, adjective: string): Promise<void> {
    this.#require("device_name", "take a name");
    if (!nameFits(name, adjective)) {
      throw new Invalid(`ValueError: ${J.quoted(composeName(name, adjective))} does not fit the ${NAME_MAX_COMPOSED} bytes a scan response carries, or a part has a leading or trailing space or a control character`);
    }
    await this.#ask(this.#raw("set_name", encodeNameParts(name, adjective)));
  }

  /** The analog converter's configuration registers, raw, for debugging. Only while not streaming. */
  async converterRegisters(): Promise<ConverterRegisters> {
    this.#require("converter_registers", "read its converter's registers");
    return decodeConverterRegisters(await this.#ask(this.session.converterRegisters()));
  }

  /** What the device's model can do and is doing. */
  async modelInfo(): Promise<ModelInfo> {
    this.#require("model", "run a model");
    return decodeModelInfo(await this.#ask(this.session.getModelInfo()));
  }

  /**
   * The head slots, which one is selected, and the encoder each head names,
   * asked of a device on protocol 1.4 or later, as `Device.heads` asks. A
   * device that refuses that question has not said, and each head is shown
   * as one that does not say.
   */
  async heads(): Promise<[number | null, readonly HeadEntry[]]> {
    this.#require("heads", "hold heads");
    let list = decodeHeads(await this.#ask(this.session.listHeads()));
    const [major, minor] = this.info.protocol;
    if (major > 1 || (major === 1 && minor >= 4)) {
      let encoders: ReadonlyMap<number, string> = new Map();
      const answer = await this.#request(this.session.listHeadEncoders());
      if (answer.ok) encoders = decodeHeadEncoders(answer.payload);
      list = withEncoders(list, encoders);
    }
    return [list.activeSlot, list.heads];
  }

  /** Select the head in `slot` as the one that produces predictions. A slot that is not a whole number from 0 to 255 is refused with `Invalid`, in words, before anything is sent. */
  async selectHead(slot: number): Promise<void> {
    this.#require("heads", "hold heads");
    await this.#ask(this.session.selectHead(slot));
  }

  /** Turn the model's output on or off. A device refuses this until a head is selected. */
  async setPredictions(on: boolean): Promise<void> {
    this.#require("model", "run a model");
    await this.#ask(this.session.setPredictions(on));
  }

  /**
   * Ask for the encoder's output for each window while the device streams,
   * as `Device.set_embeddings` asks: "window", "tokens", "both", or "off".
   * Whole windows reach `onEmbedding(connection, window)`, set here when
   * `onWindow` is given.
   */
  async setEmbeddings(form: string, onWindow?: (connection: Connection, window: EmbeddingWindow) => void): Promise<void> {
    this.#require("embeddings", "send its encoder's output");
    if (!(form in EMBEDDING_FORMS)) {
      throw new Invalid(`ValueError: the form is one of ${strList(Object.keys(EMBEDDING_FORMS))}, not ${J.shown(form)}`);
    }
    const o = own(this);
    try {
      if (form !== "off") {
        const info = await this.modelInfo();
        if ((form === "tokens" || form === "both") && info.tokensPerChannel === 0) {
          throw new SessionError("RuntimeError: this device's model declares no tokens, so it cannot send the token form");
        }
        this.#assembler = new EmbeddingAssembler(this.info.channels, info.tokensPerChannel, form as "window" | "tokens" | "both");
        if (onWindow !== undefined) this.onEmbedding = onWindow;
        if (this.#embeddingsListener === null) await this.#subscribeEmbeddings();
      }
      await this.#ask(this.#raw("set_embeddings", EMBEDDING_FORMS[form as keyof typeof EMBEDDING_FORMS]));
    } catch (e) {
      // The form this side believes is what was last set with success, and
      // after a request that failed that may not be the device's.
      o.inDoubt = true;
      throw e;
    }
    o.form = form;
    o.changes += 1;
    o.inDoubt = false;
    if (form === "off") this.#assembler = null;
  }

  /**
   * Gather whole windows of the encoder's output while the device streams,
   * as `Device.collect_embeddings` does. The stream must already be running.
   *
   * Embeddings are turned on for this, or widened to carry `form` as well,
   * only when they do not carry it already, and the form found is put back
   * once none of the package's own consumers needs them any more, unless the
   * program has set the form since. A form the program has on stays on, and
   * so does its `onEmbedding`, which this never touches. A window carries at
   * least `form`, and more when the program has more on.
   *
   * Each window carries its own length in samples (`windowSamples`), and
   * the device may describe fewer than every window (`modelInterval`), so
   * unless `timeout`, in seconds, says otherwise the wait allows thirty
   * seconds a window and a minute more. A wait that runs out throws a
   * `SessionError` whose message begins `TimeoutError` and says how many of
   * the windows arrived.
   */
  async collectEmbeddings(windows: number, form = "window", timeout: number | null = null): Promise<EmbeddingWindow[]> {
    const limit = timeout ?? 60 + 30 * windows;
    const got: EmbeddingWindow[] = [];
    let finish: () => void = () => undefined;
    const done = new Promise<void>((resolve) => {
      finish = resolve;
    });
    const undo = await embeddingsFor(this, form, (w) => {
      got.push(w);
      if (got.length >= windows) finish();
    });
    try {
      // Never armed past what a timer keeps to, and never at all for an
      // infinite wait, which waits.
      await within(done, limit, () => new SessionError(`TimeoutError: ${got.length} of ${windows} windows arrived in ${limit} seconds`));
    } finally {
      await undo();
    }
    return got.slice(0, windows);
  }

  /**
   * Put a head in a slot, and select it if asked, as `Device.upload_head`
   * does, and answer the head's identity. The blob carries its own hash,
   * which the device checks, so a head that arrived wrong is refused rather
   * than run. A head trained beside an encoder other than the one the device
   * runs is stored all the same, with a warning written through
   * `console.warn`, as the library warns: the head is the person's. A slot
   * that is not a whole number from 0 to 255 is refused with `Invalid`, in
   * words, before anything is sent.
   */
  async uploadHead(blob: Uint8Array, slot: number, options: { readonly select?: boolean } = {}): Promise<string> {
    this.#require("heads", "hold heads");
    const refused = slotRefusal(slot);
    if (refused !== null) throw new Invalid(refused);
    const head = decodeHead(blob);
    if (this.info.modelEmbedDim && head.inDim !== this.info.modelEmbedDim) {
      throw new Invalid(
        `ValueError: this head takes an embedding of ${head.inDim} and the device's model produces ${this.info.modelEmbedDim}`,
      );
    }
    if (head.encoderId !== NO_ENCODER_ID && this.session.can("model")) {
      let running: string | null = null;
      try {
        running = (await this.modelInfo()).encoderId;
      } catch {
        running = null;
      }
      if (running && running !== head.encoderId) {
        console.warn(
          `this head was trained beside encoder ${head.encoderId} and the device runs ${running}; ` +
            "it is stored and will run, and its outputs may mean nothing",
        );
      }
    }
    this.#require("update", "accept updates");
    await this.transfer(Transfer.head(this.session, slot, blob), blob);
    if (options.select) await this.selectHead(slot);
    return head.headId;
  }

  /** Erase the head in `slot`, as `Device.remove_head` does. A slot that is not a whole number from 0 to 255 is refused with `Invalid`, in words, before anything is sent. */
  async removeHead(slot: number): Promise<void> {
    this.#require("heads", "hold heads");
    await this.#ask(this.session.removeHead(slot));
  }

  /** The library's `_require`: refused, in its words, before anything is written. */
  #require(capability: string, what: string): void {
    if (!this.session.can(capability)) {
      throw new SessionError(`RuntimeError: this device does not ${what}: it does not claim the ${capability} capability`);
    }
  }

  /** A request built from its opcode, as the library's `Device.command` builds one, for the few the session checks differently. */
  #raw(opcode: OpcodeName, arg: number | Uint8Array): Command {
    return { characteristic: FILL.CONTROL, bytes: encodeRequest(opcode, arg), withResponse: true };
  }

  /** A request whose refusal the caller handles, still refused before writing on a link that is down. */
  async #request(command: Command): Promise<Response> {
    if (!this.connected) throw new SessionError(LINK_DOWN);
    return this.request(command);
  }

  async #rateNow(): Promise<number> {
    let rate = this.#config === null ? null : J.asF64(J.field(this.#config, "rate_sps"));
    if (!rate) rate = J.asF64(J.field(await this.refreshConfig(), "rate_sps"));
    return rate || 500;
  }

  async #catalogOnce(): Promise<CatalogEntry[]> {
    this.#catalog ??= await this.pipelineCatalog();
    return this.#catalog;
  }

  async #subscribeEmbeddings(): Promise<void> {
    const c = this.#characteristics.get(FILL.EMBEDDINGS);
    if (c === undefined) {
      throw new SessionError(
        "RuntimeError: this device claims embeddings, and the browser shows no characteristic for them: " +
          "it may be holding an older description of the device, from before its firmware was updated. " +
          "Forget the device in the system's Bluetooth settings and connect again.",
      );
    }
    const listener = (event: Event) => {
      const value = (event.target as BluetoothRemoteGATTCharacteristicLike).value;
      if (value !== undefined) this.deliver(FILL.EMBEDDINGS, asBytes(value));
    };
    c.addEventListener("characteristicvaluechanged", listener);
    // The browser may still hold a subscription to embeddings from a link
    // that dropped, bound to that link's object, and then starting one on
    // this object does nothing. So whatever it holds is ended first.
    try {
      await c.stopNotifications();
    } catch {
      // Nothing was held, or nothing could be ended.
    }
    try {
      await c.startNotifications();
    } catch (e) {
      // A subscription that failed leaves no listener behind, so the one
      // asked for next is the only one, and each window is heard once.
      c.removeEventListener("characteristicvaluechanged", listener);
      throw e;
    }
    this.#embeddingsListener = listener;
  }

  /** One control exchange that has to go through: refused before writing on a link that is down, and thrown when the device does not answer ok. */
  async #ask(command: Command): Promise<Uint8Array> {
    if (!this.connected) throw new SessionError(LINK_DOWN);
    const answer = await this.request(command);
    if (!answer.ok) throw new SessionError(refusal(answer.opcode, answer.status));
    return answer.payload;
  }

  /**
   * One time exchange, with the host clock read either side of it. The
   * device's answer is known to lie between those two readings, which is
   * the whole uncertainty in this one exchange. The session's `clock`, which
   * dates a recording, takes the same exchange on both host clocks, as the
   * Python library's device does. The host's clocks are read when this
   * exchange's turn begins, so time spent waiting for another exchange to
   * finish is never counted in the round trip, and the clock readings taken
   * as its answer arrived are used even when that answer is then held a
   * while to tell it from a late answer to an earlier exchange.
   */
  async timeSync(): Promise<Fit | null> {
    return this.#taking(async () => {
      const before = hostClock();
      const beforeClocks = hostClocks();
      const { answer, at, clocks } = await this.#exchange(this.session.timeSync());
      this.session.clock?.observe(decodeTimeSync(answer.payload), beforeClocks, clocks);
      return this.session.onTimeSync(answer.payload, before, at);
    });
  }

  /**
   * Carry an image or a head to the device and have it verified there.
   *
   * Nothing is activated by this. Use `activate` afterwards, and expect
   * the link to drop when you do. A device that does not answer the update
   * service in time, or leaves a progress check unanswered the last time it
   * is asked, stops the transfer with `Unanswered`. A transfer that stops
   * while the link is up tells the device to drop what it holds, and waits
   * up to five seconds for the device's answer before it throws.
   *
   * A browser does not say how large a write the link carries, so a
   * transfer starts at 244 bytes, as `Transfer` does where the system gives
   * no figure. A data write the browser refuses while the link is up is told
   * to the transfer with `writeRefused`, and where the transfer starts the
   * file again at a smaller size, ABORT's answer is waited for five seconds,
   * as when a transfer stops, and whatever comes of it is passed over, but
   * for a link that goes down meanwhile, which ends the transfer in the error
   * that stopped the larger size, the write's own or the transfer's
   * `smallerBecause`, as the Python library's transfer ends. A later transfer
   * on this connection starts at the size the last one finished at, and a
   * `connect` again starts from 244.
   *
   * `onProgress` hears the transfer's progress as the Python library
   * reports it: after each progress check that finds the device holding what
   * was sent, and, where the transfer starts the file again at a smaller
   * size, for the device's answer to START, the count the data is then
   * carried from. Nothing is reported for any other answer: the first START,
   * a progress check whose count or checksum does not match what was sent or
   * that says the transfer failed, ABORT, and FINISH.
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
    own(this).transferring = true;
    if (this.#largestWrite !== null) transfer.limitWrites(this.#largestWrite);
    try {
      // The write the browser refused, where that made the transfer start
      // the file again at a smaller size, until its ABORT is done.
      let refused: { readonly error: unknown } | null = null;
      // Whether the device has answered a START: one answered after that
      // starts the file again.
      let started = false;
      for (;;) {
        const step = transfer.step(image);
        if (step.type === "verified") return step.result;
        if (step.type === "send") {
          // The ABORT of a transfer starting the file again at a smaller
          // size, with what stopped the larger size: a link that goes down
          // meanwhile ends the transfer in it, as the library's ends.
          const again = step.command.bytes[0] === UPDATE_OPS.abort;
          const stopped = again ? (refused ?? (transfer.smallerBecause === null ? null : { error: transfer.smallerBecause })) : null;
          refused = null;
          let answer: Uint8Array;
          try {
            answer = await this.#updateExchange(step.command);
          } catch (e) {
            if (stopped !== null && !this.connected) throw stopped.error;
            // A progress check the device turned away, or did not answer,
            // is asked again after a pause, as the library asks it.
            const pause = transfer.unanswered();
            if (pause === null) throw e;
            await new Promise((resolve) => setTimeout(resolve, pause));
            continue;
          }
          if (stopped !== null && !this.connected) throw stopped.error;
          const sent = transfer.progress.taken;
          transfer.onAnswer(answer, image);
          // Reported as the library reports it: a START that starts the file
          // again, and a progress check after which the transfer goes on
          // from what was sent, which is a check that found the device
          // holding it. One that carries on from the device's own count, or
          // starts the file again, found the device holding something else,
          // and one that says the transfer failed says nothing of what it
          // took that the transfer's own count would say.
          const op = step.command.bytes[0];
          if (op === UPDATE_OPS.start) {
            if (started) onProgress?.(transfer.progress);
            started = true;
          } else if (op === UPDATE_OPS.query && !saysFailed(answer) && transfer.smallerBecause === null && transfer.progress.taken === sent) {
            onProgress?.(transfer.progress);
          }
          continue;
        }
        try {
          await this.send({
            characteristic: step.characteristic,
            bytes: image.subarray(step.from, step.to),
            withResponse: false,
          });
        } catch (e) {
          // A write the system refuses while the link is up, as some systems
          // refuse one longer than the link carries, starts the file again
          // at a smaller size, where one remains.
          if (this.connected && this.#characteristics.has(step.characteristic) && transfer.writeRefused()) {
            refused = { error: e };
            continue;
          }
          throw e;
        }
        transfer.sent(step.to - step.from);
      }
    } catch (e) {
      // A transfer that stopped leaves the device holding a part of
      // something. On a link that is up, the device is told to drop that,
      // and its answer is waited for: the device acknowledges a write before
      // it handles what the write carried, so only its answer says that it
      // handled ABORT. A link that is down is told nothing. Then what went
      // wrong is reported.
      if (this.connected) {
        try {
          await this.#updateExchange(transfer.abort());
        } catch {
          // ABORT went unanswered, or its write failed. Whatever the device
          // answered, or did not, the transfer's own error is what is said.
        }
      }
      throw e;
    } finally {
      this.#largestWrite = transfer.largestWrite;
      own(this).transferring = false;
    }
  }

  /**
   * Put an image a transfer verified in force, as the Python library's
   * `Device.activate` does: ACTIVATE goes through the update exchange, and
   * its answer is waited for five seconds, as the library waits. Only an
   * answer of ok resolves. The device then restarts, so the link drops and
   * everything this connection knows about the device is stale, and the
   * session takes any stream as stopped. A device that refuses throws
   * `Refused`, with its status in words. An answer that does not come in
   * time throws `Unanswered`, a link that drops before the answer comes
   * among them, since only the answer says the device accepted. A write that
   * fails throws its own error.
   *
   * ```javascript
   * await link.transfer(Transfer.app(link.session, 0, image), image);
   * await link.activate();
   * ```
   */
  async activate(): Promise<void> {
    const answer = decodeUpdateResponse(await this.#updateExchange(activation()));
    if (!answer.ok) throw new Refused("activate", answer.statusName);
    // The restart ends any stream. Nothing is written for it: the device is restarting.
    if (this.session.streaming) this.session.stopStream();
  }

  /**
   * End every subscription this link started, then drop the link. The
   * order matters: on some platforms the browser keeps a subscribed link
   * open after a disconnect, and the next program on the same computer
   * then hears every packet twice. Safe on a link that is already down.
   * An answer owed on a link that is gone never comes, so none is owed
   * after. The session takes the stream as stopped, as the Python library's
   * `Device.disconnect` does, and nothing is written for it. The device
   * turns embeddings off at disconnect, so this side takes them as off.
   */
  async disconnect(): Promise<void> {
    const subscribed = this.#embeddingsListener === null ? NOTIFYING : [...NOTIFYING, FILL.EMBEDDINGS];
    for (const fill of subscribed) {
      const c = this.#characteristics.get(fill);
      if (c === undefined) continue;
      try {
        await c.stopNotifications();
      } catch {
        // A link that is already gone has nothing left to stop.
      }
    }
    if (this.#server.connected) this.#server.disconnect();
    this.#owed.clear();
    if (this.session.streaming) this.session.stopStream();
    const embeddings = this.#characteristics.get(FILL.EMBEDDINGS);
    if (embeddings !== undefined && this.#embeddingsListener !== null) {
      embeddings.removeEventListener("characteristicvaluechanged", this.#embeddingsListener);
    }
    this.#embeddingsListener = null;
    const o = own(this);
    o.form = "off";
    o.changes += 1;
    o.inDoubt = false;
    this.#assembler = null;
  }

  /**
   * Write to the update service and wait for its answer to the same
   * operation: five seconds for a progress check, an ABORT or an ACTIVATE,
   * and longer for the rest. A write that fails leaves nothing waiting.
   */
  async #updateExchange(command: Command): Promise<Uint8Array> {
    return (await this.#updateWrite(command)).answer;
  }

  /**
   * Write to the update service and start waiting for its answer to the
   * same operation, as `#updateExchange` waits for it. Gives back that
   * answer, still to come, and how to stop waiting for it, once the write
   * is done. A write that fails leaves nothing waiting.
   */
  async #updateWrite(command: Command): Promise<{ readonly answer: Promise<Uint8Array>; readonly cancel: () => void }> {
    const op = command.bytes[0]!;
    const waitMs =
      op === UPDATE_OPS.query
        ? this.#checkTimeoutMs
        : op === UPDATE_OPS.abort
          ? this.#abortTimeoutMs
          : op === UPDATE_OPS.activate
            ? this.#activateTimeoutMs
            : this.#updateTimeoutMs;
    let cancel = (): void => {};
    const answer = new Promise<Uint8Array>((resolve, reject) => {
      const drop = (): void => {
        const i = this.#updateWaiting.indexOf(waiter);
        if (i >= 0) this.#updateWaiting.splice(i, 1);
      };
      const waiter = {
        op,
        settle: (bytes: Uint8Array): void => {
          stop();
          resolve(bytes);
        },
      };
      const stop = after(waitMs / 1000, () => {
        drop();
        reject(new Unanswered("the device did not answer the update service in time"));
      });
      cancel = () => {
        stop();
        drop();
      };
      this.#updateWaiting.push(waiter);
    });
    // Late while the write is still under way is still late, and is said
    // when the write is done.
    answer.catch(() => {});
    try {
      await this.send(command);
    } catch (e) {
      cancel();
      throw e;
    }
    return { answer, cancel };
  }

  /**
   * @internal Called for every notification that arrives. The package's own
   * consumers hear each batch of samples and each whole window first, with
   * the session's counts already kept. Then `onUsbStop` hears that this
   * packet ended the stream on USB power, when it did, then every listener
   * hears each event, then `onEmbedding` each whole window. A listener or
   * callback that throws is reported, and stops neither the others, nor the
   * assembly of windows, nor the package's own consumers.
   */
  deliver(characteristic: number, bytes: Uint8Array): void {
    const at = hostClock();
    const clocks = hostClocks();
    if (characteristic === FILL.UPDATE_CONTROL) {
      // An update answer is not a session event. It goes to the first
      // request still waiting for an answer to the operation the answer
      // names in its first byte, and an answer no request waits for is
      // dropped, so a late answer, or the answer to an ABORT nobody waited
      // for, is never taken as another request's answer.
      const i = this.#updateWaiting.findIndex((w) => w.op === bytes[0]);
      if (i >= 0) this.#updateWaiting.splice(i, 1)[0]!.settle(bytes);
      return;
    }
    const usbBefore = this.session.endedByUsb;
    const events = this.session.onNotification(characteristic, bytes);
    // The package's own path first: the answer to the exchange waiting for
    // it, and every batch and whole window to the package's own consumers,
    // with the session's counts already kept.
    const windows: (EmbeddingWindow | null)[] = [];
    for (const event of events) {
      let window: EmbeddingWindow | null = null;
      if (event.type === "answer") {
        // An answer no exchange is waiting for answered an earlier one that
        // stopped waiting, or is a copy, and is settled now.
        if (this.#pending > 0) {
          this.#answers.push({ answer: event.answer, at, clocks });
          this.#wake?.();
        } else {
          this.#settle(event.answer);
        }
      } else if (event.type === "samples") {
        // What the packet said, from the wire rather than from memory.
        const wire = own(this);
        wire.gain = event.batch.gain;
        wire.rateWire = event.batch.sampleRateHz ?? 0;
        wire.modeWire = MODE_BY_CODE[event.batch.mode] ?? "unknown";
        hand(this, "batch", event.batch);
      } else if (event.type === "embedding" && this.#assembler !== null) {
        window = this.#assembler.feed(event.embedding);
        if (window !== null) hand(this, "window", window);
      }
      windows.push(window);
    }
    // Then the program's: each callback on its own, so one that throws is
    // reported and stops nothing else.
    const thrown: unknown[] = [];
    const call = (f: () => void): void => {
      try {
        f();
      } catch (e) {
        thrown.push(e);
      }
    };
    const usbStop = this.onUsbStop;
    if (!usbBefore && this.session.endedByUsb && usbStop !== null) call(() => usbStop(this));
    events.forEach((event, i) => {
      for (const listener of [...this.#listeners]) call(() => listener(event));
      const window = windows[i]!;
      const onEmbedding = this.onEmbedding;
      if (window !== null && onEmbedding !== null) call(() => onEmbedding(this, window));
    });
    for (const e of thrown) report(e);
  }
}

/** What every device that speaks this protocol can do, which no capability bit says because none of it varies. */
const EVERY_DEVICE: ReadonlySet<string> = new Set(["counter", "reset_count", "packet_size", "clock", "gain"]);

/** A list of whole numbers as a refusal shows it, in brackets. */
function intList(values: readonly number[]): string {
  return `[${values.map((v) => String(v)).join(", ")}]`;
}

/** A list of words as a refusal shows it: each in double quotes, in brackets. */
function strList(values: readonly string[]): string {
  return `[${values.map((v) => J.quoted(v)).join(", ")}]`;
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
 * Every characteristic a device carries that this transport uses, found on
 * its service. A device carries the characteristics its firmware offers, so
 * one that is absent is not an error: what is absent shows up as a
 * capability the device never claimed. The embeddings characteristic is
 * found here and subscribed only when a form is asked for, as the library
 * subscribes to it.
 */
async function characteristicsOf(
  service: BluetoothRemoteGATTServiceLike,
  infoCharacteristic: BluetoothRemoteGATTCharacteristicLike,
): Promise<Map<number, BluetoothRemoteGATTCharacteristicLike>> {
  const characteristics = new Map<number, BluetoothRemoteGATTCharacteristicLike>();
  characteristics.set(FILL.DEVICE_INFO, infoCharacteristic);
  for (const fill of [...NOTIFYING, ...WRITABLE, FILL.EMBEDDINGS]) {
    const c = await optional(service, fill);
    if (c !== null) characteristics.set(fill, c);
  }
  return characteristics;
}

/**
 * The `onDisconnect` each device was last connected with by `connect`, so a
 * later `connect` to the same device takes it off.
 */
const DROP_LISTENERS = new WeakMap<BluetoothDeviceLike, () => void>();

/**
 * Connect to a device the user chose, read what it is, and subscribe to
 * everything it notifies. `Connection.connect` connects the same
 * connection again after its link drops.
 */
export async function connect(device: BluetoothDeviceLike, options: ConnectOptions = {}): Promise<Connection> {
  const timeoutMs = timeoutOf(options);
  if (device.gatt === undefined) throw new NoBluetooth();
  const server = await device.gatt.connect();
  const service = await server.getPrimaryService(SERVICE);

  const session = options.session ?? new Session();
  const infoCharacteristic = await service.getCharacteristic(DEVICE_INFO);
  const infoBytes = asBytes(await infoCharacteristic.readValue());
  const info = session.onDeviceInfo(infoBytes);
  const characteristics = await characteristicsOf(service, infoCharacteristic);

  const connection = new Connection({
    device,
    server,
    session,
    info,
    infoBytes,
    characteristics,
    timeoutMs,
  });
  if (options.onEvent !== undefined) connection.on(options.onEvent);
  await listen(connection);

  // Every drop, not only the first: the listener stays on the device, which
  // is the same object however many times the connection connects again.
  // A later `connect` to the device is a new connection, and takes off the
  // listener an earlier one left, or a program that connects again after
  // each drop would hear each drop once for every connection it ever made.
  const earlier = DROP_LISTENERS.get(device);
  if (earlier !== undefined) device.removeEventListener("gattserverdisconnected", earlier);
  DROP_LISTENERS.delete(device);
  if (options.onDisconnect !== undefined) {
    device.addEventListener("gattserverdisconnected", options.onDisconnect);
    DROP_LISTENERS.set(device, options.onDisconnect);
  }
  return connection;
}

/**
 * The wait a control request is given, in milliseconds: five seconds where
 * `options` give none, and refused before anything is asked of the browser
 * where what they give is not a number, a wait no timer would ever end.
 */
function timeoutOf(options: ConnectOptions): number {
  const ms: unknown = options.timeoutMs;
  if (ms === undefined || ms === null) return 5000;
  if (typeof ms !== "number" || Number.isNaN(ms)) {
    throw new TypeError(`timeoutMs is a number of milliseconds, and was given ${J.shown(ms)}`);
  }
  return ms;
}

/** Ask for a device and connect to it, which is what most callers want. */
export async function requestAndConnect(options: RequestOptions & ConnectOptions = {}): Promise<Connection> {
  timeoutOf(options);
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

/** Whether two byte strings are the same bytes. */
function sameBytes(a: Uint8Array, b: Uint8Array): boolean {
  return a.length === b.length && a.every((x, i) => x === b[i]);
}

function asBytes(value: DataView): Uint8Array {
  return new Uint8Array(value.buffer, value.byteOffset, value.byteLength).slice();
}

/** Whether an update answer is a progress check's that says the transfer failed. */
function saysFailed(bytes: Uint8Array): boolean {
  const answer = decodeUpdateResponse(bytes);
  return answer.opName === "query" && answer.ok && decodeUpdateQuery(answer.payload).stateName === "failed";
}
