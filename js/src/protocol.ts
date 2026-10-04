/**
 * The IntoMind BLE protocol, version 1.0.
 *
 * This module is the wire and nothing else. It has no device in it, no
 * transport, no state, and no opinion about what a caller does with a
 * message. Every function takes bytes and returns what the contract says
 * those bytes mean, or throws a `ProtocolError` saying why they are not a
 * message.
 *
 * It is held to `contract/conformance.json`, which the firmware emits from
 * the codec it runs. Every message here decodes those vectors to the stated
 * fields and refuses the malformed ones for the stated reason, so this
 * implementation and the device cannot drift apart without a test failing.
 *
 * Three things are worth knowing before reading further.
 *
 * **Capabilities are the device's answer.** What a device can do is in the
 * bits it reports, never in a table of hardware here. A device that does not
 * claim a capability is not asked for it.
 *
 * **The timeline is never silently repaired.** A forward step in the sample
 * index is a loss with an exact count. A step that is not forward is a break
 * whose extent is not a number, and this module says so rather than
 * returning a count that would be a fiction.
 *
 * **Nothing is assumed about width.** The number of channels, the converter
 * resolution, the reference voltage, and the tick rate all come from the
 * device, and the scaling helper takes them as arguments.
 */

import { sha256 } from "./digest.ts";

export { Crc32, crc32, sha256 } from "./digest.ts";

/** The protocol version this module implements, major and minor. */
export const VERSION: readonly [number, number] = [1, 4];

/** The full identifier for one characteristic. */
export function uuid(fill: number): string {
  return `f3a1${fill.toString(16).padStart(4, "0")}-2c4b-4d1e-9a6f-1b2c3d4e5f60`;
}

/**
 * The sixteen bit fill each characteristic sits at. A notification is
 * routed by this rather than by its full identifier, so nothing downstream
 * has to format or compare strings.
 */
export const FILL = {
  SERVICE: 0x0001,
  DEVICE_INFO: 0x0002,
  CONTROL: 0x0003,
  CONTROL_RESPONSE: 0x0004,
  EEG_DATA: 0x0005,
  STATUS: 0x0006,
  /** Held by an earlier development firmware. Never reused. */
  RESERVED_0007: 0x0007,
  UPDATE_CONTROL: 0x0008,
  UPDATE_DATA: 0x0009,
  PREDICTIONS: 0x000a,
  /** 1.2: the encoder's output for each window. */
  EMBEDDINGS: 0x000b,
} as const;

/** The service's full identifier, advertised by every device. */
export const SERVICE = uuid(FILL.SERVICE);
/** Device Info's full identifier. */
export const DEVICE_INFO = uuid(FILL.DEVICE_INFO);
/** Control Point's full identifier. */
export const CONTROL = uuid(FILL.CONTROL);
/** Control Response's full identifier. */
export const CONTROL_RESPONSE = uuid(FILL.CONTROL_RESPONSE);
/** EEG Data's full identifier. */
export const EEG_DATA = uuid(FILL.EEG_DATA);
/** Status's full identifier. */
export const STATUS = uuid(FILL.STATUS);
/** Update Control's full identifier. */
export const UPDATE_CONTROL = uuid(FILL.UPDATE_CONTROL);
/** Update Data's full identifier. */
export const UPDATE_DATA = uuid(FILL.UPDATE_DATA);
/** Predictions' full identifier. */
export const PREDICTIONS = uuid(FILL.PREDICTIONS);
/** Embeddings' full identifier. */
export const EMBEDDINGS = uuid(FILL.EMBEDDINGS);

/** The name a device advertises under, before its own identifier. */
export const NAME_PREFIX = "IntoMind-";
/**
 * 1.3: the product name a composed device name ends with. A 1.3 device
 * advertises `[name's] [adjective] IntoMind One`; a 1.2 device advertised
 * `IntoMind-` and four hex digits.
 */
export const PRODUCT_NAME = "IntoMind One";

// --- refusals ---------------------------------------------------------------

/** Why bytes are not a message. The contract distinguishes these three. */
export type Refusal = "truncated" | "invalid" | "reserved";

/** Bytes that are not a message, with the reason they are not. */
export class ProtocolError extends Error {
  /** Which of the three reasons this is. */
  readonly reason: Refusal;

  /** Build one with its reason and message. A decoder throws one of the three subclasses. */
  constructor(reason: Refusal, message: string) {
    super(message);
    this.name = "ProtocolError";
    this.reason = reason;
  }
}

/** The message ends before the fields it promises. */
export class Truncated extends ProtocolError {
  /** Build one with the message saying what is missing. */
  constructor(message: string) {
    super("truncated", message);
    this.name = "Truncated";
  }
}

/** A field holds a value the contract does not define. */
export class Invalid extends ProtocolError {
  /** Build one with the message saying what is wrong. */
  constructor(message: string) {
    super("invalid", message);
    this.name = "Invalid";
  }
}

/**
 * A number the contract reserves. A device answers these unsupported rather
 * than treating them as malformed, and so does a host.
 */
export class Reserved extends ProtocolError {
  /** Build one with the message naming the reserved number. */
  constructor(message: string) {
    super("reserved", message);
    this.name = "Reserved";
  }
}

// --- bytes ------------------------------------------------------------------

const EMPTY = new Uint8Array(0);

function view(b: Uint8Array): DataView {
  return new DataView(b.buffer, b.byteOffset, b.byteLength);
}

function hex(b: Uint8Array): string {
  let s = "";
  for (let i = 0; i < b.length; i++) s += b[i]!.toString(16).padStart(2, "0");
  return s;
}

/** Bytes from a hex string, for tests and for anything that logs one. */
export function fromHex(s: string): Uint8Array {
  if (s.length % 2 !== 0) throw new Invalid("a hex string has an even number of characters");
  const out = new Uint8Array(s.length / 2);
  for (let i = 0; i < out.length; i++) {
    const byte = Number.parseInt(s.slice(i * 2, i * 2 + 2), 16);
    if (Number.isNaN(byte)) throw new Invalid(`${s.slice(i * 2, i * 2 + 2)} is not a hex byte`);
    out[i] = byte;
  }
  return out;
}

export { hex as toHex };

/** A fixed width name field, without the padding the wire carries. */
function nameOf(b: Uint8Array): string {
  let end = b.length;
  while (end > 0 && b[end - 1] === 0) end--;
  return new TextDecoder("utf-8").decode(b.subarray(0, end));
}

function paddedName(name: string, width: number): Uint8Array {
  const out = new Uint8Array(width);
  out.set(new TextEncoder().encode(name).subarray(0, width));
  return out;
}

// --- the control point ------------------------------------------------------

/** Every control point opcode this contract defines, by name. */
export const OPCODES = {
  start_stream: 0x01,
  stop_stream: 0x02,
  set_rate: 0x10,
  set_gain: 0x11,
  set_mode: 0x20,
  set_leadoff: 0x30,
  time_sync: 0x40,
  set_samples_per_packet: 0x41,
  get_battery: 0x42,
  get_boot_info: 0x43,
  reset_epoch: 0x50,
  clear_bonds: 0x53,
  set_predictions: 0x80,
  select_head: 0x81,
  list_heads: 0x82,
  remove_head: 0x83,
  get_model_info: 0x84,
  // New in 1.1.
  set_prediction_input: 0x85,
  get_prediction_input: 0x86,
  set_bias: 0x32,
  get_bias_diagnostic: 0x33,
  get_pipeline_catalog: 0x90,
  get_pipeline: 0x91,
  set_pipeline: 0x92,
  clear_pipeline: 0x93,
  restore_pipeline_default: 0x94,
  // New in 1.2.
  get_converter_registers: 0x34,
  get_indicator: 0x44,
  set_indicator: 0x45,
  identify: 0x46,
  set_embeddings: 0x87,
  // New in 1.3.
  set_model_interval: 0x88,
  get_model_interval: 0x89,
  get_name: 0x47,
  set_name: 0x48,
  // New in 1.4.
  list_head_encoders: 0x8a,
  soft_reset: 0xf0,
} as const;

/** The name of any opcode `OPCODES` defines. */
export type OpcodeName = keyof typeof OPCODES;

/** Opcode names by their number, the reverse of `OPCODES`. */
export const NAME_BY_OPCODE: ReadonlyMap<number, OpcodeName> = new Map(
  (Object.entries(OPCODES) as [OpcodeName, number][]).map(([name, code]) => [code, name]),
);

/** Opcodes that take one argument byte, and require it. */
export const TAKES_ARG: ReadonlySet<OpcodeName> = new Set<OpcodeName>([
  "set_rate",
  "set_gain",
  "set_mode",
  "set_leadoff",
  "set_samples_per_packet",
  "set_predictions",
  "select_head",
  "remove_head",
  "set_bias",
  "set_indicator",
  "identify",
  "set_embeddings",
]);

/**
 * Opcodes new in 1.1 that carry a payload after the opcode byte, laid out
 * as the processing section below says. The only requests longer than two
 * bytes.
 */
export const TAKES_PAYLOAD: ReadonlySet<OpcodeName> = new Set<OpcodeName>([
  "set_pipeline",
  "set_prediction_input",
  "set_model_interval",
  "set_name",
]);

/**
 * Opcodes a 1.0 device does not know. A host sends one only to a device
 * whose Device Info claims the capability behind it.
 */
export const NEW_IN_1_1: ReadonlySet<OpcodeName> = new Set<OpcodeName>([
  "set_prediction_input",
  "get_prediction_input",
  "set_bias",
  "get_bias_diagnostic",
  "get_pipeline_catalog",
  "get_pipeline",
  "set_pipeline",
  "clear_pipeline",
  "restore_pipeline_default",
]);

/** Opcodes a 1.1 device does not know, likewise gated on a capability. */
export const NEW_IN_1_2: ReadonlySet<OpcodeName> = new Set<OpcodeName>([
  "get_converter_registers",
  "get_indicator",
  "set_indicator",
  "identify",
  "set_embeddings",
]);
/** Opcodes a 1.2 device does not know: the model's cadence and the device's name. */
export const NEW_IN_1_3: ReadonlySet<OpcodeName> = new Set<OpcodeName>([
  "set_model_interval",
  "get_model_interval",
  "get_name",
  "set_name",
]);

/**
 * Numbers an earlier development firmware used, and numbers the contract
 * keeps for factory and bench builds. Never reused for anything else.
 */
export const RESERVED_OPCODES: ReadonlySet<number> = new Set([
  0x12,
  0x13,
  0x31,
  0x70,
  0x71,
  ...Array.from({ length: 16 }, (_, i) => 0x60 + i),
]);

/** A control response's status byte, in words. */
export const STATUS_CODES: Readonly<Record<number, string>> = {
  0: "ok",
  1: "invalid argument",
  2: "unsupported",
  3: "busy",
  4: "not streaming",
  5: "hardware",
  // 1.4: a stream of the natural signal refused while USB power is present.
  // 6 stays unassigned because the Update service uses it.
  7: "usb power",
};

/** The gain a `gain_code` of 0 to 6 selects. */
export const GAIN_BY_CODE: readonly number[] = [1, 2, 4, 6, 8, 12, 24];
/** The code for a gain, the reverse of `GAIN_BY_CODE`. */
export const CODE_BY_GAIN: ReadonlyMap<number, number> = new Map(GAIN_BY_CODE.map((g, i) => [g, i]));
/** The rate in samples per second a `rate_code` selects. */
export const RATE_BY_CODE: Readonly<Record<number, number>> = { 4: 1000, 5: 500, 6: 250 };
/** The code for a rate, the reverse of `RATE_BY_CODE`. */
export const CODE_BY_RATE: ReadonlyMap<number, number> = new Map([
  [1000, 4],
  [500, 5],
  [250, 6],
]);
/** The bit in Device Info's rate mask that stands for each rate. */
export const RATE_BY_INFO_BIT: Readonly<Record<number, number>> = { 0: 250, 1: 500, 2: 1000 };

/** 1.3 adds synthetic: the converter is not driven and the device generates the signal, on a device that claims it. */
export const MODES = { normal: 0, test: 1, short: 2, synthetic: 3 } as const;
/** The name of any mode `MODES` defines. */
export type ModeName = keyof typeof MODES;
/** Mode names by their code, the reverse of `MODES`. */
export const MODE_BY_CODE: Readonly<Record<number, ModeName>> = { 0: "normal", 1: "test", 2: "short", 3: "synthetic" };

/** What a charger state code means. */
export const CHARGER_STATES: Readonly<Record<number, string>> = {
  0: "no input",
  1: "charging",
  2: "complete",
  3: "fault",
  4: "standby",
};

/** Why the device last booted, by `boot_reason` code. */
export const BOOT_REASONS: Readonly<Record<number, string>> = {
  0: "power on",
  1: "reset pin",
  2: "software",
  3: "watchdog",
  4: "fault",
  5: "update",
  6: "rollback",
  0xff: "unknown",
};

/**
 * One control point write.
 *
 * `opcode` is a name or a number. An opcode that takes an argument requires
 * one, and one that does not takes none, because the device refuses either
 * mistake and a host should not make it.
 */
export function encodeRequest(
  opcode: OpcodeName | number,
  arg: number | Uint8Array | null = null,
): Uint8Array {
  const name = typeof opcode === "string" ? opcode : NAME_BY_OPCODE.get(opcode) ?? null;
  const code = name === null ? undefined : OPCODES[name];
  if (name === null || code === undefined) {
    const shown = typeof opcode === "string" ? opcode : toByte(opcode);
    throw new Invalid(`${shown} is not an opcode this contract defines`);
  }
  if (TAKES_PAYLOAD.has(name)) {
    if (!(arg instanceof Uint8Array)) throw new Invalid(`${name} takes a payload of bytes`);
    const out = new Uint8Array(1 + arg.length);
    out[0] = code;
    out.set(arg, 1);
    return out;
  }
  if (arg instanceof Uint8Array) throw new Invalid(`${name} takes no payload`);
  if (TAKES_ARG.has(name) !== (arg !== null)) {
    throw new Invalid(`${name} takes ${TAKES_ARG.has(name) ? "an argument" : "no argument"}`);
  }
  return arg === null ? new Uint8Array([code]) : new Uint8Array([code, arg & 0xff]);
}

/** One control point write, as `decodeRequest` reads it. */
export interface Request {
  /** The raw opcode byte, even one this contract does not define. */
  readonly opcode: number;
  /** The opcode's name. */
  readonly name: OpcodeName;
  /** The one argument byte, for an opcode that takes one. */
  readonly arg: number | null;
  /** The bytes after the opcode, for an opcode that takes a payload. */
  readonly payload: Uint8Array | null;
}

/**
 * A control point write, as the device reads it. For tests and for anything
 * that has to speak both sides of the contract.
 */
export function decodeRequest(data: Uint8Array): Request {
  if (data.length === 0) throw new Truncated("an empty write is not a request");
  const code = data[0]!;
  if (RESERVED_OPCODES.has(code)) throw new Reserved(`${toByte(code)} is reserved`);
  const name = NAME_BY_OPCODE.get(code);
  if (name === undefined) throw new Invalid(`${toByte(code)} is not an opcode this contract defines`);
  const rest = data.subarray(1);
  if (TAKES_PAYLOAD.has(name)) {
    // The payload's own decoder judges a chain; the 1.3 payloads are judged here.
    if (name === "set_model_interval" && rest.length !== 2) {
      throw new Invalid("set_model_interval takes two bytes, the seconds little-endian");
    }
    if (name === "set_name") decodeNameParts(rest);
    return { opcode: code, name, arg: null, payload: rest.slice() };
  }
  if (TAKES_ARG.has(name)) {
    if (rest.length !== 1) throw new Invalid(`${name} takes exactly one argument byte`);
    return { opcode: code, name, arg: rest[0]!, payload: null };
  }
  if (rest.length !== 0) throw new Invalid(`${name} takes no argument`);
  return { opcode: code, name, arg: null, payload: null };
}

/** One control response, as `decodeResponse` reads it. */
export interface Response {
  /** The opcode this answers, echoed back even when it is not one this contract defines. */
  readonly opcode: number;
  /** Null for a number this contract does not define, which is echoed back. */
  readonly name: OpcodeName | null;
  /** The status byte, 0 for success. */
  readonly status: number;
  /** `status` in words, from `STATUS_CODES`. */
  readonly statusName: string;
  /** Whether `status` is 0. */
  readonly ok: boolean;
  /** The bytes after the opcode and status, specific to what was asked. */
  readonly payload: Uint8Array;
}

/** One control response, as the device sends it. */
export function decodeResponse(data: Uint8Array): Response {
  if (data.length < 2) throw new Truncated("a response is at least an opcode and a status");
  const status = data[1]!;
  return {
    opcode: data[0]!,
    name: NAME_BY_OPCODE.get(data[0]!) ?? null,
    status,
    statusName: STATUS_CODES[status] ?? `undefined status ${status}`,
    ok: status === 0,
    payload: data.slice(2),
  };
}

/**
 * One control response. The opcode is a number rather than a name because a
 * device echoes back whatever it was sent, including a reserved number.
 */
export function encodeResponse(opcode: number, status: number, payload: Uint8Array = EMPTY): Uint8Array {
  const out = new Uint8Array(2 + payload.length);
  out[0] = opcode;
  out[1] = status;
  out.set(payload, 2);
  return out;
}

/** The device's own time, from the answer to a time request. */
export function decodeTimeSync(payload: Uint8Array): bigint {
  if (payload.length < 8) throw new Truncated("a device time is eight bytes");
  return view(payload).getBigUint64(0, true);
}

function toByte(n: number): string {
  return `0x${n.toString(16).padStart(2, "0")}`;
}

// --- device info ------------------------------------------------------------

/** What a device can do, as the bits it reports. */
export const CAPABILITIES = {
  battery_voltage: 1 << 0,
  battery_low_flag: 1 << 1,
  leadoff: 1 << 2,
  test_signal: 1 << 3,
  dcdc_mode: 1 << 4,
  input_short: 1 << 5,
  update: 1 << 6,
  model: 1 << 7,
  model_ready: 1 << 8,
  heads: 1 << 9,
  // New in 1.1.
  pipeline: 1 << 10,
  bias_drive: 1 << 11,
  // New in 1.2.
  embeddings: 1 << 12,
  indicator: 1 << 13,
  converter_registers: 1 << 14,
  // New in 1.3.
  model_cadence: 1 << 15,
} as const;

/** The name of any capability bit `CAPABILITIES` defines. */
export type CapabilityName = keyof typeof CAPABILITIES;

/** 1.3: capability bits 16 and up, the second word of Device Info at offset 76. Bit 16 of the contract is bit 0 here. */
export const CAPABILITIES_HIGH = {
  synthetic: 1 << 0,
  device_name: 1 << 1,
} as const;

/** The name of any capability bit `CAPABILITIES_HIGH` defines. */
export type CapabilityHighName = keyof typeof CAPABILITIES_HIGH;

/** Length of the v0.1 prefix, which 1.0 left where it was. */
export const INFO_V01_LEN = 27;
/** Length of the 1.0 layout. */
export const INFO_LEN_1_0 = 76;
/** 1.3: the 1.0 layout with a second capability word appended. */
export const INFO_LEN = 80;
/** What a device with no battery telemetry reports instead of a percent. */
export const BATTERY_UNKNOWN = 0xff;

/** What `DeviceInfo`'s constructor takes. The optional fields default to zero or null. */
export interface DeviceInfoFields {
  /** Protocol major and minor. */
  protocol: readonly [number, number];
  /** Firmware major, minor, and patch. */
  firmware: readonly [number, number, number];
  /** Channel count. */
  channels: number;
  /** Converter resolution, in bits. */
  adcBits: number;
  /** Device time ticks per second. */
  tickHz: number;
  /** Converter reference voltage, in microvolts. */
  vrefUv: number;
  /** The sixteen bit capability mask. */
  capabilities: number;
  /** Which rates the device supports, as a bit mask. */
  supportedRates: number;
  /** The device's stable per-unit identifier, as hex. */
  deviceId: string;
  /** Hardware major, minor, and patch, when the device reports the 1.0 extension. */
  hardware?: readonly [number, number, number] | null;
  /** The exact firmware build, when reported. */
  buildId?: string | null;
  /** The model's embedding width, zero for no model runtime. */
  modelEmbedDim?: number;
  /** The model's native sample rate. */
  modelNativeSps?: number;
  /** Samples per prediction window at the model's native rate. */
  modelWindowSamples?: number;
  /** User head slots. */
  headSlots?: number;
  /** The largest `out_dim` a head may have. */
  headMaxOutputs?: number;
  /** Capacity of one head slot, in bytes. */
  headSlotBytes?: number;
  /** The largest Update Data write the device accepts. */
  updateChunkMax?: number;
  /** Capacity of one application slot, including its header. */
  appSlotBytes?: number;
  /** Capacity of the weights image, including its header. */
  weightsImageBytes?: number;
  /** 1.3: capability bits 16 and up. Zero on a device whose info ends at 76 bytes. */
  capabilitiesHigh?: number;
}

/** Everything a device says about itself, and nothing a host assumed. */
export class DeviceInfo {
  /** Protocol major and minor. */
  readonly protocol: readonly [number, number];
  /** Firmware major, minor, and patch. */
  readonly firmware: readonly [number, number, number];
  /** Channel count. */
  readonly channels: number;
  /** Converter resolution, in bits. */
  readonly adcBits: number;
  /** Device time ticks per second. */
  readonly tickHz: number;
  /** Converter reference voltage, in microvolts. */
  readonly vrefUv: number;
  /** The sixteen bit capability mask, bits 0 to 15. `can` is how to read it. */
  readonly capabilities: number;
  /** Which rates the device supports, as a bit mask. `rates` is how to read it. */
  readonly supportedRates: number;
  /** The device's stable per-unit identifier, as hex. */
  readonly deviceId: string;
  /** Present on a device that reports the 1.0 layout. */
  readonly hardware: readonly [number, number, number] | null;
  /** The exact firmware build, or null when the device reports no extension. */
  readonly buildId: string | null;
  /** The model's embedding width, zero for no model runtime. */
  readonly modelEmbedDim: number;
  /** The model's native sample rate. */
  readonly modelNativeSps: number;
  /** Samples per prediction window at the model's native rate. */
  readonly modelWindowSamples: number;
  /** User head slots. */
  readonly headSlots: number;
  /** The largest `out_dim` a head may have. */
  readonly headMaxOutputs: number;
  /** Capacity of one head slot, in bytes. */
  readonly headSlotBytes: number;
  /** The largest Update Data write the device accepts. */
  readonly updateChunkMax: number;
  /** Capacity of one application slot, including its header. */
  readonly appSlotBytes: number;
  /** Capacity of the weights image, including its header. */
  readonly weightsImageBytes: number;
  /** The second capability word, bits 16 and up (1.3). Zero on a device whose info ends at 76 bytes. */
  readonly capabilitiesHigh: number;

  /** Build from the fields `decodeDeviceInfo` read, or assembled by hand. */
  constructor(f: DeviceInfoFields) {
    this.protocol = [f.protocol[0], f.protocol[1]];
    this.firmware = [f.firmware[0], f.firmware[1], f.firmware[2]];
    this.channels = f.channels;
    this.adcBits = f.adcBits;
    this.tickHz = f.tickHz;
    this.vrefUv = f.vrefUv;
    this.capabilities = f.capabilities;
    this.supportedRates = f.supportedRates;
    this.deviceId = f.deviceId;
    this.hardware = f.hardware ?? null;
    this.buildId = f.buildId ?? null;
    this.modelEmbedDim = f.modelEmbedDim ?? 0;
    this.modelNativeSps = f.modelNativeSps ?? 0;
    this.modelWindowSamples = f.modelWindowSamples ?? 0;
    this.headSlots = f.headSlots ?? 0;
    this.headMaxOutputs = f.headMaxOutputs ?? 0;
    this.headSlotBytes = f.headSlotBytes ?? 0;
    this.updateChunkMax = f.updateChunkMax ?? 0;
    this.appSlotBytes = f.appSlotBytes ?? 0;
    this.weightsImageBytes = f.weightsImageBytes ?? 0;
    this.capabilitiesHigh = f.capabilitiesHigh ?? 0;
  }

  /**
   * Whether the device claims a capability. Unknown names are false rather
   * than an error, so a host written against a later version of this
   * contract still runs.
   */
  can(capability: CapabilityName | CapabilityHighName | string): boolean {
    const high = (CAPABILITIES_HIGH as Record<string, number>)[capability];
    if (high !== undefined) return (this.capabilitiesHigh & high) !== 0;
    const bit = (CAPABILITIES as Record<string, number>)[capability] ?? 0;
    return (this.capabilities & bit) !== 0;
  }

  /** Whether the device reported the 1.0 extension. */
  get extensionPresent(): boolean {
    return this.hardware !== null;
  }

  /** The rates the device declares, from its own mask. */
  get rates(): number[] {
    return Object.entries(RATE_BY_INFO_BIT)
      .map(([bit, sps]) => [Number(bit), sps] as const)
      .filter(([bit]) => (this.supportedRates & (1 << bit)) !== 0)
      .map(([, sps]) => sps)
      .sort((a, b) => a - b);
  }

  /** `protocol` as `major.minor`. */
  get protocolString(): string {
    return `${this.protocol[0]}.${this.protocol[1]}`;
  }

  /** `firmware` as `major.minor.patch`. */
  get firmwareString(): string {
    return `${this.firmware[0]}.${this.firmware[1]}.${this.firmware[2]}`;
  }

  /** `hardware` as `major.minor.patch`, or null when the device reports no extension. */
  get hardwareString(): string | null {
    return this.hardware === null ? null : `${this.hardware[0]}.${this.hardware[1]}.${this.hardware[2]}`;
  }

  /** One count, in microvolts, at a gain. From the device's own reference and width. */
  microvoltsPerCount(gain: number): number {
    return (2.0 * this.vrefUv) / (gain * 2 ** this.adcBits);
  }
}

/** Device Info, as the device sends it. */
export function decodeDeviceInfo(data: Uint8Array): DeviceInfo {
  if (data.length < INFO_V01_LEN) throw new Truncated(`device info is at least ${INFO_V01_LEN} bytes`);
  const v = view(data);
  const head: DeviceInfoFields = {
    protocol: [data[0]!, data[1]!],
    firmware: [data[2]!, data[3]!, data[4]!],
    channels: data[5]!,
    adcBits: data[6]!,
    tickHz: v.getUint32(7, true),
    vrefUv: v.getUint32(11, true),
    capabilities: v.getUint16(15, true),
    supportedRates: data[17]!,
    deviceId: hex(data.subarray(19, 27)),
  };
  if (data.length === INFO_V01_LEN) return new DeviceInfo(head);
  const infoLen = data[27]!;
  if (infoLen < INFO_LEN_1_0) {
    throw new Invalid(
      `a device speaking ${head.protocol[0]}.${head.protocol[1]} states its info is ${infoLen} bytes, ` +
        `which is shorter than the ${INFO_LEN_1_0} the 1.0 layout defines`,
    );
  }
  if (data.length < INFO_LEN_1_0) {
    throw new Truncated(`device info promises ${infoLen} bytes and carries ${data.length}`);
  }
  // 1.3: the second capability word, when the device says its info reaches it.
  const capabilitiesHigh = infoLen >= INFO_LEN && data.length >= INFO_LEN ? v.getUint32(76, true) : 0;
  return new DeviceInfo({
    ...head,
    hardware: [data[28]!, data[29]!, data[30]!],
    buildId: hex(data.subarray(32, 40)),
    modelEmbedDim: v.getUint16(40, true),
    modelNativeSps: v.getUint16(42, true),
    modelWindowSamples: v.getUint16(44, true),
    headSlots: data[46]!,
    headMaxOutputs: data[47]!,
    headSlotBytes: v.getUint16(48, true),
    updateChunkMax: v.getUint16(50, true),
    appSlotBytes: v.getUint32(52, true),
    weightsImageBytes: v.getUint32(56, true),
    capabilitiesHigh,
  });
}

/**
 * Device info as a device reports it. A device that has no extension to
 * report encodes the v0.1 prefix, which is what an older one sends.
 */
export function encodeDeviceInfo(info: DeviceInfo): Uint8Array {
  // A device with nothing in its second word still writes the 1.3 length: the
  // contract's current layout is 80 bytes and a 1.2 reader stops at 76.
  const out = new Uint8Array(info.extensionPresent ? INFO_LEN : INFO_V01_LEN);
  const v = view(out);
  out[0] = info.protocol[0];
  out[1] = info.protocol[1];
  out[2] = info.firmware[0];
  out[3] = info.firmware[1];
  out[4] = info.firmware[2];
  out[5] = info.channels;
  out[6] = info.adcBits;
  v.setUint32(7, info.tickHz, true);
  v.setUint32(11, info.vrefUv, true);
  v.setUint16(15, info.capabilities, true);
  out[17] = info.supportedRates;
  out.set(fromHex(info.deviceId), 19);
  if (!info.extensionPresent) return out;
  out[27] = INFO_LEN;
  out[28] = info.hardware![0];
  out[29] = info.hardware![1];
  out[30] = info.hardware![2];
  out.set(fromHex(info.buildId ?? "0000000000000000"), 32);
  v.setUint16(40, info.modelEmbedDim, true);
  v.setUint16(42, info.modelNativeSps, true);
  v.setUint16(44, info.modelWindowSamples, true);
  out[46] = info.headSlots;
  out[47] = info.headMaxOutputs;
  v.setUint16(48, info.headSlotBytes, true);
  v.setUint16(50, info.updateChunkMax, true);
  v.setUint32(52, info.appSlotBytes, true);
  v.setUint32(56, info.weightsImageBytes, true);
  v.setUint32(76, info.capabilitiesHigh, true);
  return out;
}

// --- status -----------------------------------------------------------------

/** The length of a Status message, in bytes. */
export const STATUS_LEN = 12;
/** The bits of Status's flags byte. */
export const STATUS_FLAGS = { usb_present: 1 << 0, buffer_high_watermark: 1 << 1 } as const;

/** Status, decoded, read or notified. */
export interface Status {
  /** 0 idle, 1 streaming. */
  readonly state: number;
  /** Whether `state` is streaming. */
  readonly streaming: boolean;
  /** The mode code in force. See `MODES`. */
  readonly mode: number;
  /** `mode` in words. */
  readonly modeName: string;
  /** The gain code in force. */
  readonly gainCode: number;
  /** `gainCode` as a gain, or null for a code this contract does not define. */
  readonly gain: number | null;
  /** The rate code in force. */
  readonly rateCode: number;
  /** `rateCode` as samples per second, or null for a code this contract does not define. */
  readonly sps: number | null;
  /** The charger state code. */
  readonly chargerState: number;
  /** `chargerState` in words. */
  readonly chargerName: string;
  /** Null when the device has no battery telemetry, never 255 percent. */
  readonly batteryPercent: number | null;
  /** The raw byte: 0xFF for unknown, rather than null. */
  readonly batteryPercentRaw: number;
  /** Lead-off status latched with the most recently acquired sample. Bit n is channel n+1. */
  readonly loffStatp: number;
  /** Whether USB power is present (1.4). */
  readonly usbPresent: boolean;
  /** Whether the device's buffer is at or above three quarters full. */
  readonly bufferHighWatermark: boolean;
  /** Telemetry. The account of what a host missed is the sample index. */
  readonly droppedTotal: number;
  /** Samples currently buffered on the device. */
  readonly bufferFill: number;
}

/** Status, as the device sends it. */
export function decodeStatus(data: Uint8Array): Status {
  if (data.length < STATUS_LEN) throw new Truncated(`status is ${STATUS_LEN} bytes`);
  const v = view(data);
  const gainCode = data[2]!;
  const rateCode = data[3]!;
  const charger = data[4]!;
  const battery = data[5]!;
  const flags = data[7]!;
  return {
    state: data[0]!,
    streaming: data[0] === 1,
    mode: data[1]!,
    modeName: MODE_BY_CODE[data[1]!] ?? `mode ${data[1]}`,
    gainCode,
    gain: GAIN_BY_CODE[gainCode] ?? null,
    rateCode,
    sps: RATE_BY_CODE[rateCode] ?? null,
    chargerState: charger,
    chargerName: CHARGER_STATES[charger] ?? `charger state ${charger}`,
    batteryPercent: battery === BATTERY_UNKNOWN ? null : battery,
    batteryPercentRaw: battery,
    loffStatp: data[6]!,
    usbPresent: (flags & STATUS_FLAGS.usb_present) !== 0,
    bufferHighWatermark: (flags & STATUS_FLAGS.buffer_high_watermark) !== 0,
    droppedTotal: v.getUint16(8, true),
    bufferFill: v.getUint16(10, true),
  };
}

/** What `encodeStatus` takes. */
export interface StatusFields {
  /** 0 idle, 1 streaming. */
  state: number;
  /** The mode code in force. */
  mode: number;
  /** The gain code in force. */
  gainCode: number;
  /** The rate code in force. */
  rateCode: number;
  /** The charger state code. */
  chargerState: number;
  /** Null for a device with no battery telemetry. */
  batteryPercent: number | null;
  /** Lead-off status latched with the most recently acquired sample. */
  loffStatp: number;
  /** Whether USB power is present. */
  usbPresent?: boolean;
  /** Whether the device's buffer is at or above three quarters full. */
  bufferHighWatermark?: boolean;
  /** Samples acquired but never delivered since power on. */
  droppedTotal: number;
  /** Samples currently buffered on the device. */
  bufferFill: number;
}

/** Status, as the device sends it. */
export function encodeStatus(s: StatusFields): Uint8Array {
  const out = new Uint8Array(STATUS_LEN);
  const v = view(out);
  out[0] = s.state;
  out[1] = s.mode;
  out[2] = s.gainCode;
  out[3] = s.rateCode;
  out[4] = s.chargerState;
  out[5] = s.batteryPercent ?? BATTERY_UNKNOWN;
  out[6] = s.loffStatp;
  out[7] =
    (s.usbPresent ? STATUS_FLAGS.usb_present : 0) |
    (s.bufferHighWatermark ? STATUS_FLAGS.buffer_high_watermark : 0);
  v.setUint16(8, s.droppedTotal, true);
  v.setUint16(10, s.bufferFill, true);
  return out;
}

// --- the stream -------------------------------------------------------------

/** The EEG data packet type. */
export const PACKET_EEG = 0x01;
/** The prediction packet type. */
export const PACKET_PREDICTION = 0x02;
/** 1.3: samples the device generated with its converter off, in exactly the layout of `PACKET_EEG`. */
export const PACKET_SYNTHETIC = 0x04;
/** The length of an EEG data packet's header, before the samples. */
export const DATA_HEADER_LEN = 20;
/** The single bit flags of an EEG data packet's flags byte. Mode occupies the other two, via `MODE_SHIFT` and `MODE_MASK`. */
export const DATA_FLAGS = { discontinuity: 1 << 0, leadoff_active: 1 << 1, usb_present: 1 << 4 } as const;
/** Where the mode bits sit in an EEG data packet's flags byte. */
export const MODE_SHIFT = 2;
/** The mode bits of an EEG data packet's flags byte, before shifting. */
export const MODE_MASK = 0b11 << MODE_SHIFT;
/** `samples_lost_before` saturates here. The sample index is authoritative. */
export const LOST_SATURATED = 0xffff;

/**
 * One notification of samples, decoded but not scaled.
 *
 * `counts` is one row per sample across the channels, in converter counts.
 * Scaling is the caller's, with the device's own reference and width,
 * because this module assumes nothing about either.
 */
export interface Packet {
  /** 0x01 for a measured sample, 0x04 for one the device generated (1.3). */
  readonly packetType: number;
  /** Whether a break precedes this packet. */
  readonly discontinuity: boolean;
  /** Whether lead-off was active when this packet was built. */
  readonly leadoffActive: boolean;
  /** The mode code the samples were taken in. */
  readonly mode: number;
  /** `mode` in words. */
  readonly modeName: string;
  /** Whether USB power was present (1.4). */
  readonly usbPresent: boolean;
  /** A convenience count of samples lost before this packet, saturating at `LOST_SATURATED`. `index` continuity is authoritative. */
  readonly samplesLostBefore: number;
  /** The first sample's index, monotonic and wrapping at 2^32. */
  readonly index: number;
  /** The first sample's device time, in the device's ticks. */
  readonly deviceTime: bigint;
  /** Samples carried in this packet. */
  readonly nSamples: number;
  /** Lead-off status latched with the packet's last sample. */
  readonly loffStatp: number;
  /** The gain code the samples were taken at. */
  readonly gainCode: number;
  /** `gainCode` as a gain. */
  readonly gain: number;
  /** The rate code the samples were taken at. */
  readonly rateCode: number;
  /** `rateCode` as samples per second, or null for a code this contract does not define. */
  readonly sps: number | null;
  /** One row per sample, each row one converter count per channel. */
  readonly counts: number[][];
  /** 1.3: the samples were generated by the device, not measured (packet type 0x04). Never store one as a measurement. */
  readonly synthetic: boolean;
}

/** One EEG notification. `channels` comes from the device. */
export function decodePacket(data: Uint8Array, channels: number): Packet {
  if (data.length < DATA_HEADER_LEN) throw new Truncated("a data packet is at least its header");
  const v = view(data);
  const packetType = data[0]!;
  if (packetType !== PACKET_EEG && packetType !== PACKET_SYNTHETIC) {
    throw new Invalid(`packet type ${toByte(packetType)} is not sample data`);
  }
  const flags = data[1]!;
  const n = data[16]!;
  const gainCode = data[18]!;
  const rateCode = data[19]!;
  if (gainCode >= GAIN_BY_CODE.length) {
    throw new Invalid(`gain code ${gainCode} is not one this contract defines`);
  }
  const body = data.subarray(DATA_HEADER_LEN);
  const want = n * channels * 3;
  if (body.length !== want) {
    throw new Truncated(
      `${n} samples of ${channels} channels is ${want} bytes, and this packet carries ${body.length}`,
    );
  }
  const counts: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let c = 0; c < channels; c++) row.push(int24(body, (i * channels + c) * 3));
    counts.push(row);
  }
  const mode = (flags & MODE_MASK) >> MODE_SHIFT;
  return {
    packetType,
    discontinuity: (flags & DATA_FLAGS.discontinuity) !== 0,
    leadoffActive: (flags & DATA_FLAGS.leadoff_active) !== 0,
    mode,
    modeName: MODE_BY_CODE[mode] ?? `mode ${mode}`,
    usbPresent: (flags & DATA_FLAGS.usb_present) !== 0,
    samplesLostBefore: v.getUint16(2, true),
    index: v.getUint32(4, true),
    deviceTime: v.getBigUint64(8, true),
    nSamples: n,
    loffStatp: data[17]!,
    gainCode,
    gain: GAIN_BY_CODE[gainCode]!,
    rateCode,
    sps: RATE_BY_CODE[rateCode] ?? null,
    counts,
    synthetic: packetType === PACKET_SYNTHETIC,
  };
}

/** What `encodePacket` takes. */
export interface PacketFields {
  /** The first sample's index. */
  index: number;
  /** The first sample's device time, in the device's ticks. */
  deviceTime: bigint;
  /** The gain code the samples were taken at. */
  gainCode: number;
  /** The rate code the samples were taken at. */
  rateCode: number;
  /** One row per sample, each row one value per channel, in counts. */
  counts: ReadonlyArray<ArrayLike<number>>;
  /** Whether a break precedes this packet. */
  discontinuity?: boolean;
  /** Whether lead-off was active. */
  leadoffActive?: boolean;
  /** The mode code the samples were taken in. */
  mode?: number;
  /** Whether USB power was present. */
  usbPresent?: boolean;
  /** A convenience count of samples lost before this packet. */
  samplesLostBefore?: number;
  /** Lead-off status latched with the packet's last sample. */
  loffStatp?: number;
  /** 1.3: write the packet as synthetic samples, type 0x04. */
  synthetic?: boolean;
}

/**
 * One EEG notification, as the device sends it. The payload is converter
 * order, big endian, which is why the device copies its bytes out without
 * swapping any of them.
 */
export function encodePacket(p: PacketFields): Uint8Array {
  const n = p.counts.length;
  const channels = n === 0 ? 0 : p.counts[0]!.length;
  for (const row of p.counts) {
    if (row.length !== channels) throw new Invalid("every sample covers the same channels");
  }
  if (n > 0xff) throw new Invalid("a packet carries at most 255 samples");
  const out = new Uint8Array(DATA_HEADER_LEN + n * channels * 3);
  const v = view(out);
  out[0] = p.synthetic ? PACKET_SYNTHETIC : PACKET_EEG;
  out[1] =
    (p.discontinuity ? DATA_FLAGS.discontinuity : 0) |
    (p.leadoffActive ? DATA_FLAGS.leadoff_active : 0) |
    (((p.mode ?? 0) << MODE_SHIFT) & MODE_MASK) |
    (p.usbPresent ? DATA_FLAGS.usb_present : 0);
  v.setUint16(2, p.samplesLostBefore ?? 0, true);
  v.setUint32(4, p.index >>> 0, true);
  v.setBigUint64(8, p.deviceTime, true);
  out[16] = n;
  out[17] = p.loffStatp ?? 0;
  out[18] = p.gainCode;
  out[19] = p.rateCode;
  let at = DATA_HEADER_LEN;
  for (const row of p.counts) {
    for (let c = 0; c < channels; c++) {
      const value = row[c]! | 0;
      out[at++] = (value >> 16) & 0xff;
      out[at++] = (value >> 8) & 0xff;
      out[at++] = value & 0xff;
    }
  }
  return out;
}

/** One big endian, two's complement int24, sign extended. */
export function int24(b: Uint8Array, at = 0): number {
  return ((b[at]! << 24) | (b[at + 1]! << 16) | (b[at + 2]! << 8)) >> 8;
}

/**
 * One count in microvolts. Every term comes from the device: its reference,
 * the gain the packet was taken at, and the converter's width.
 */
export function microvolts(raw: number, vrefUv: number, gain: number, adcBits: number): number {
  return (raw * (2.0 * vrefUv)) / (gain * 2 ** adcBits);
}

/** What one packet's indexing says happened since the one before it. */
export interface Continuity {
  /** The contract's own words for the three cases. */
  readonly verdict: "continuous" | "gap" | "break";
  /** Samples missing. Null for a break, whose extent is not a number. */
  readonly lost: number | null;
  /** Whether this is a gap or a break rather than continuous. */
  readonly broken: boolean;
}

/**
 * Judge two packets by the contract's own rule.
 *
 * The subtraction is signed, so the index's own wrap at two to the thirty
 * second is exact. A step that is not forward is a break in the timeline
 * rather than a loss, and reporting the unsigned difference there claims a
 * loss of four billion samples that never happened.
 */
export function continuity(
  previousIndex: number,
  previousN: number,
  index: number,
  discontinuity: boolean,
): Continuity {
  const expected = (previousIndex + previousN) >>> 0;
  let step = (index - expected) >>> 0;
  if (step >= 0x80000000) step -= 0x100000000;
  if (step === 0 && !discontinuity) return { verdict: "continuous", lost: 0, broken: false };
  if (step > 0) return { verdict: "gap", lost: step, broken: true };
  return { verdict: "break", lost: null, broken: true };
}

// --- predictions ------------------------------------------------------------

/** The length of a prediction packet's header, before the outputs. */
export const PREDICTION_HEADER_LEN = 28;
/** The bits of a prediction packet's flags byte. */
export const PREDICTION_FLAGS = {
  gap_in_window: 1 << 0,
  duty_reduced: 1 << 1,
  leadoff_in_window: 1 << 2,
} as const;

/** One prediction notification, decoded. */
export interface Prediction {
  /** 0x02 for a prediction. */
  readonly packetType: number;
  /** The slot of the head that produced this. */
  readonly headSlot: number;
  /** The identity of the head that produced this. */
  readonly headId: string;
  /** The window's first sample's index. */
  readonly index: number;
  /** The window's first sample's device time. */
  readonly deviceTime: bigint;
  /** The window's length, in raw samples at the current rate. */
  readonly windowSamples: number;
  /** Whether a gap fell inside the window. */
  readonly gapInWindow: boolean;
  /** Whether the device skipped windows to stay within its compute budget. */
  readonly dutyReduced: boolean;
  /** Whether an electrode was off at some point in the window. */
  readonly leadoffInWindow: boolean;
  /** The head's outputs for this window. */
  readonly outputs: number[];
  /**
   * 1.1: what the window was taken from, an `INPUT_SOURCES` value: the
   * stream, the natural signal, or the model's own chain. A 1.0 device
   * says the stream.
   */
  readonly inputSource: number;
}

/** One prediction notification, as the device sends it. */
export function decodePrediction(data: Uint8Array): Prediction {
  if (data.length < PREDICTION_HEADER_LEN) throw new Truncated("a prediction is at least its header");
  const v = view(data);
  const packetType = data[0]!;
  if (packetType !== PACKET_PREDICTION) throw new Invalid(`packet type ${toByte(packetType)} is not a prediction`);
  const flags = data[1]!;
  const n = data[3]!;
  const body = data.subarray(PREDICTION_HEADER_LEN);
  if (body.length !== n * 4) {
    throw new Truncated(`${n} outputs is ${n * 4} bytes, and this packet carries ${body.length}`);
  }
  const outputs: number[] = [];
  for (let i = 0; i < n; i++) outputs.push(v.getFloat32(PREDICTION_HEADER_LEN + i * 4, true));
  return {
    packetType,
    headSlot: data[2]!,
    headId: hex(data.subarray(18, 26)),
    index: v.getUint32(4, true),
    deviceTime: v.getBigUint64(8, true),
    windowSamples: v.getUint16(16, true),
    gapInWindow: (flags & PREDICTION_FLAGS.gap_in_window) !== 0,
    dutyReduced: (flags & PREDICTION_FLAGS.duty_reduced) !== 0,
    leadoffInWindow: (flags & PREDICTION_FLAGS.leadoff_in_window) !== 0,
    outputs,
    inputSource: data[26]!,
  };
}

/** What `encodePrediction` takes. */
export interface PredictionFields {
  /** The slot of the head that produced this. */
  headSlot: number;
  /** The identity of the head that produced this. */
  headId: string;
  /** The window's first sample's index. */
  index: number;
  /** The window's first sample's device time. */
  deviceTime: bigint;
  /** The window's length, in raw samples. */
  windowSamples: number;
  /** The head's outputs for this window. */
  outputs: ArrayLike<number>;
  /** Whether a gap fell inside the window. */
  gapInWindow?: boolean;
  /** Whether the device skipped windows to stay within its compute budget. */
  dutyReduced?: boolean;
  /** Whether an electrode was off at some point in the window. */
  leadoffInWindow?: boolean;
  /** What the window was taken from, an `INPUT_SOURCES` value. */
  inputSource?: number;
}

/** One prediction notification, as the device sends it. */
export function encodePrediction(p: PredictionFields): Uint8Array {
  const n = p.outputs.length;
  if (n > 0xff) throw new Invalid("a head has at most 255 outputs");
  const out = new Uint8Array(PREDICTION_HEADER_LEN + n * 4);
  const v = view(out);
  out[0] = PACKET_PREDICTION;
  out[1] =
    (p.gapInWindow ? PREDICTION_FLAGS.gap_in_window : 0) |
    (p.dutyReduced ? PREDICTION_FLAGS.duty_reduced : 0) |
    (p.leadoffInWindow ? PREDICTION_FLAGS.leadoff_in_window : 0);
  out[2] = p.headSlot;
  out[3] = n;
  v.setUint32(4, p.index >>> 0, true);
  v.setBigUint64(8, p.deviceTime, true);
  v.setUint16(16, p.windowSamples, true);
  out.set(fromHex(p.headId), 18);
  out[26] = p.inputSource ?? 0;
  for (let i = 0; i < n; i++) v.setFloat32(PREDICTION_HEADER_LEN + i * 4, p.outputs[i]!, true);
  return out;
}

// --- what the device reports about itself beyond status ---------------------

/** GET_BATTERY's answer. */
export interface Battery {
  /** The measured pack voltage, in millivolts. */
  readonly millivolts: number;
  /** A state of charge estimate, or null when the device has no battery telemetry. */
  readonly percent: number | null;
  /** The charger state code. */
  readonly chargerState: number;
  /** `chargerState` in words. */
  readonly chargerName: string;
}

/** GET_BATTERY's answer, as the device sends it. */
export function decodeBattery(payload: Uint8Array): Battery {
  if (payload.length < 4) throw new Truncated("a battery reading is four bytes");
  const v = view(payload);
  const charger = payload[3]!;
  return {
    millivolts: v.getUint16(0, true),
    percent: payload[2] === BATTERY_UNKNOWN ? null : payload[2]!,
    chargerState: charger,
    chargerName: CHARGER_STATES[charger] ?? `charger state ${charger}`,
  };
}

/** GET_BATTERY's answer, as the device sends it. */
export function encodeBattery(b: { millivolts: number; percent: number | null; chargerState: number }): Uint8Array {
  const out = new Uint8Array(4);
  view(out).setUint16(0, b.millivolts, true);
  out[2] = b.percent ?? BATTERY_UNKNOWN;
  out[3] = b.chargerState;
  return out;
}

/** GET_BOOT_INFO's answer. */
export interface BootInfo {
  /** Which application slot is running: 0 A, 1 B. */
  readonly activeSlot: number;
  /** The boot reason code. */
  readonly bootReason: number;
  /** `bootReason` in words. */
  readonly bootReasonName: string;
  /** 0 trial, the running image has not yet confirmed itself. 1 confirmed. */
  readonly slotState: number;
  /** A slot on trial has not yet been confirmed by the firmware in it. */
  readonly confirmed: boolean;
  /** Boots of this unit since manufacture. */
  readonly bootCount: number;
}

/** GET_BOOT_INFO's answer, as the device sends it. */
export function decodeBootInfo(payload: Uint8Array): BootInfo {
  if (payload.length < 8) throw new Truncated("boot information is eight bytes");
  const reason = payload[1]!;
  return {
    activeSlot: payload[0]!,
    bootReason: reason,
    bootReasonName: BOOT_REASONS[reason] ?? `reason ${reason}`,
    slotState: payload[2]!,
    confirmed: payload[2] === 1,
    bootCount: view(payload).getUint32(4, true),
  };
}

/** GET_BOOT_INFO's answer, as the device sends it. */
export function encodeBootInfo(b: {
  activeSlot: number;
  bootReason: number;
  slotState: number;
  bootCount: number;
}): Uint8Array {
  const out = new Uint8Array(8);
  out[0] = b.activeSlot;
  out[1] = b.bootReason;
  out[2] = b.slotState;
  view(out).setUint32(4, b.bootCount, true);
  return out;
}

/** A head slot's state, in words, by its `state` code. */
export const HEAD_STATES: Readonly<Record<number, string>> = { 0: "empty", 1: "valid", 2: "invalid", 3: "width mismatch" };
/** The model runtime's state, in words, by its `model_state` code. */
export const MODEL_STATES: Readonly<Record<number, string>> = { 0: "no runtime", 1: "no weights", 2: "ready", 3: "updating" };
/** An encoder id a head that does not name its encoder carries. */
export const NO_ENCODER_ID = "0000000000000000";
/** The slot byte meaning no head is selected. */
export const NO_HEAD = 0xff;
/** The length of one LIST_HEADS record, in bytes. */
export const HEAD_ENTRY_LEN = 30;

/** One LIST_HEADS record: one head slot. */
export interface HeadEntry {
  /** 0 for the built-in head, 1 and up for user slots. */
  readonly slot: number;
  /** The slot's state code. See `HEAD_STATES`. */
  readonly state: number;
  /** `state` in words. */
  readonly stateName: string;
  /** The head's output count. */
  readonly outDim: number;
  /** The head's identity, the first eight bytes of its SHA-256. Zero when empty. */
  readonly headId: string;
  /** The head's name. */
  readonly name: string;
  /** Whether `state` is valid. */
  readonly usable: boolean;
  /**
   * 1.3: the encoder the head was trained beside, as its file recorded it; `NO_ENCODER_ID` when it did not say.
   * From 1.4 the list carries none: `decodeHeadEncoders` reads them, and `withEncoders` puts them here.
   */
  readonly encoderId: string;
}

/** LIST_HEADS's answer: the head slots, and which one is selected. */
export interface HeadList {
  /** Null means no head is selected. */
  readonly activeSlot: number | null;
  /** The slots, in the order the device reports them. */
  readonly heads: HeadEntry[];
}

/** The head slots, and which one is selected. */
export function decodeHeads(payload: Uint8Array): HeadList {
  if (payload.length < 2) throw new Truncated("a head list is at least its count");
  const active = payload[0]!;
  const n = payload[1]!;
  const recordsEnd = 2 + n * HEAD_ENTRY_LEN;
  // 1.3 defined one eight byte encoder id per record after the records. No
  // device sent it, because the IntoMind One's list was too long with it,
  // and 1.4 withdraws it. One that comes is still read.
  const withIds = payload.length === recordsEnd + n * 8;
  if (payload.length !== recordsEnd && !withIds) {
    throw new Invalid(
      `a list of ${n} heads is ${recordsEnd} bytes, or ${recordsEnd + n * 8} with its encoder ids, and this one is ${payload.length}`,
    );
  }
  const heads: HeadEntry[] = [];
  for (let i = 0; i < n; i++) {
    const r = payload.subarray(2 + i * HEAD_ENTRY_LEN, 2 + (i + 1) * HEAD_ENTRY_LEN);
    const state = r[1]!;
    heads.push({
      slot: r[0]!,
      state,
      stateName: HEAD_STATES[state] ?? `state ${state}`,
      outDim: view(r).getUint16(2, true),
      headId: hex(r.subarray(4, 12)),
      name: nameOf(r.subarray(12, 28)),
      usable: state === 1,
      encoderId: withIds ? hex(payload.subarray(recordsEnd + i * 8, recordsEnd + i * 8 + 8)) : NO_ENCODER_ID,
    });
  }
  return { activeSlot: active === NO_HEAD ? null : active, heads };
}

/** A LIST_HEADS payload as a device sends it: the records and nothing after them. */
export function encodeHeads(
  activeSlot: number | null,
  heads: ReadonlyArray<{ slot: number; state: number; outDim: number; headId: string; name: string }>,
): Uint8Array {
  const out = new Uint8Array(2 + heads.length * HEAD_ENTRY_LEN);
  out[0] = activeSlot ?? NO_HEAD;
  out[1] = heads.length;
  heads.forEach((h, i) => {
    const at = 2 + i * HEAD_ENTRY_LEN;
    out[at] = h.slot;
    out[at + 1] = h.state;
    view(out).setUint16(at + 2, h.outDim, true);
    out.set(fromHex(h.headId), at + 4);
    out.set(paddedName(h.name, 16), at + 12);
  });
  return out;
}

/** One LIST_HEAD_ENCODERS record (1.4): a slot and the encoder id its head names. */
export const HEAD_ENCODER_LEN = 9;

/**
 * 1.4: the encoder each head names, by slot, from LIST_HEAD_ENCODERS: `NO_ENCODER_ID` for an empty slot or a head
 * that does not say.
 */
export function decodeHeadEncoders(payload: Uint8Array): ReadonlyMap<number, string> {
  if (payload.length < 1) throw new Truncated("a list of head encoders is at least its count");
  const n = payload[0]!;
  if (payload.length !== 1 + n * HEAD_ENCODER_LEN) {
    throw new Invalid(`a list of ${n} head encoders is ${1 + n * HEAD_ENCODER_LEN} bytes, and this one is ${payload.length}`);
  }
  const out = new Map<number, string>();
  for (let i = 0; i < n; i++) {
    const at = 1 + i * HEAD_ENCODER_LEN;
    out.set(payload[at]!, hex(payload.subarray(at + 1, at + HEAD_ENCODER_LEN)));
  }
  return out;
}

/** A LIST_HEAD_ENCODERS payload: one record for each head, in the order given. */
export function encodeHeadEncoders(heads: ReadonlyArray<{ slot: number; encoderId: string }>): Uint8Array {
  const out = new Uint8Array(1 + heads.length * HEAD_ENCODER_LEN);
  out[0] = heads.length;
  heads.forEach((h, i) => {
    const at = 1 + i * HEAD_ENCODER_LEN;
    out[at] = h.slot;
    out.set(fromHex(h.encoderId), at + 1);
  });
  return out;
}

/** The heads, each with the encoder `encoders` says it names. A slot it does not mention keeps what it had. */
export function withEncoders(list: HeadList, encoders: ReadonlyMap<number, string>): HeadList {
  return {
    activeSlot: list.activeSlot,
    heads: list.heads.map((h) => ({ ...h, encoderId: encoders.get(h.slot) ?? h.encoderId })),
  };
}

/** GET_MODEL_INFO's answer. */
export interface ModelInfo {
  /** The model runtime's state code. See `MODEL_STATES`. */
  readonly modelState: number;
  /** `modelState` in words. */
  readonly modelStateName: string;
  /** Whether `modelState` is ready, so predictions can be enabled. */
  readonly ready: boolean;
  /** Null means no head is selected. */
  readonly activeHead: number | null;
  /** Whether predictions are enabled. */
  readonly predictionsOn: boolean;
  /** The loaded weights' identity. Zero unless ready. */
  readonly encoderId: string;
  /** The loaded weights' major, minor, and patch. */
  readonly weightsVersion: readonly [number, number, number];
  /**
   * 1.1: the signal classes the loaded model declares it takes, as
   * `INPUT_CLASSES` bits. A 1.0 device sends zero, which reads as the time
   * domain.
   */
  readonly inputClasses: number;
  /**
   * 1.2: tokens the loaded model produces per channel per window, the shape
   * of the token form of embeddings. A 1.1 device's message ends before this
   * byte and reads as zero: it sends no tokens.
   */
  readonly tokensPerChannel: number;
  /** 1.3: the time one pass of the encoder takes on the device, milliseconds; zero before the first pass. */
  readonly passMs: number;
  /** 1.3: the interval between described windows in force, seconds; zero is every window the device can. */
  readonly intervalS: number;
  /** 1.3: whether the device holds a generator, so it can offer the synthetic signal. */
  readonly generator: boolean;
}

/** GET_MODEL_INFO's answer, as the device sends it. */
export function decodeModelInfo(payload: Uint8Array): ModelInfo {
  if (payload.length < 16) throw new Truncated("model information is at least sixteen bytes");
  const state = payload[0]!;
  const active = payload[1]!;
  return {
    modelState: state,
    modelStateName: MODEL_STATES[state] ?? `state ${state}`,
    ready: state === 2,
    activeHead: active === NO_HEAD ? null : active,
    predictionsOn: payload[2] !== 0,
    encoderId: hex(payload.subarray(4, 12)),
    weightsVersion: [payload[12]!, payload[13]!, payload[14]!],
    inputClasses: payload[15]! || INPUT_CLASSES.time_domain,
    tokensPerChannel: payload.length >= 17 ? payload[16]! : 0,
    passMs: payload.length >= 22 ? view(payload).getUint16(17, true) : 0,
    intervalS: payload.length >= 22 ? view(payload).getUint16(19, true) : 0,
    generator: payload.length >= 22 && payload[21] !== 0,
  };
}

/** GET_MODEL_INFO's answer, as the device sends it. */
export function encodeModelInfo(m: {
  modelState: number;
  activeHead: number | null;
  predictionsOn: boolean;
  encoderId: string;
  weightsVersion: readonly [number, number, number];
  inputClasses?: number;
  /** 1.2: when given, the message is seventeen bytes. */
  tokensPerChannel?: number | undefined;
  /** 1.3: when given, the message is twenty two bytes. */
  passMs?: number | undefined;
  intervalS?: number | undefined;
  generator?: boolean | undefined;
}): Uint8Array {
  const out = new Uint8Array(m.passMs === undefined ? (m.tokensPerChannel === undefined ? 16 : 17) : 22);
  out[0] = m.modelState;
  out[1] = m.activeHead ?? NO_HEAD;
  out[2] = m.predictionsOn ? 1 : 0;
  out.set(fromHex(m.encoderId), 4);
  out[12] = m.weightsVersion[0];
  out[13] = m.weightsVersion[1];
  out[14] = m.weightsVersion[2];
  out[15] = m.inputClasses ?? 0;
  if (m.tokensPerChannel !== undefined) out[16] = m.tokensPerChannel;
  if (m.passMs !== undefined) {
    const v = view(out);
    v.setUint16(17, m.passMs, true);
    v.setUint16(19, m.intervalS ?? 0, true);
    out[21] = m.generator ? 1 : 0;
  }
  return out;
}

// --- the model's cadence and the device's name (1.3) --------------------------

/** GET_MODEL_INTERVAL's answer: the interval in force and the least the device keeps, seconds. */
export interface ModelInterval {
  /** The interval in force, seconds. 0 is every window the device can. */
  readonly intervalS: number;
  /** The least interval the device keeps, seconds. */
  readonly minimumS: number;
}

/** GET_MODEL_INTERVAL's answer, as the device sends it. */
export function decodeModelInterval(payload: Uint8Array): ModelInterval {
  if (payload.length < 4) throw new Truncated("a model interval is four bytes");
  const v = view(payload);
  return { intervalS: v.getUint16(0, true), minimumS: v.getUint16(2, true) };
}

/** SET_MODEL_INTERVAL's payload: the seconds, little endian. Zero is every window the device can. */
export function encodeModelInterval(seconds: number): Uint8Array {
  if (!Number.isInteger(seconds) || seconds < 0 || seconds > 0xffff) throw new Invalid("an interval is 0 to 65535 seconds");
  const out = new Uint8Array(2);
  view(out).setUint16(0, seconds, true);
  return out;
}

/** The most a composed name may be, in bytes: what a scan response carries. The same limit bounds each part. */
export const NAME_MAX_COMPOSED = 29;

const utf8 = new TextEncoder();
const utf8Decoder = new TextDecoder("utf-8");

/** A name or an adjective: UTF-8 of at most 29 bytes, no control characters, no leading or trailing space. Empty is allowed. */
export function namePartOk(part: string): boolean {
  const b = utf8.encode(part);
  if (b.length > NAME_MAX_COMPOSED) return false;
  for (const c of b) if (c < 0x20 || c === 0x7f) return false;
  return part === part.replace(/^ +| +$/g, "");
}

/** The device's name as it composes it: `name's adjective product`, the possessive only with a name, the adjective only when given. */
export function composeName(name: string, adjective: string, product = PRODUCT_NAME): string {
  const parts: string[] = [];
  if (name) parts.push(`${name}'s`);
  if (adjective) parts.push(adjective);
  parts.push(product);
  return parts.join(" ");
}

/** Whether the composition fits the air, so the device will accept it. */
export function nameFits(name: string, adjective: string, product = PRODUCT_NAME): boolean {
  return namePartOk(name) && namePartOk(adjective) && utf8.encode(composeName(name, adjective, product)).length <= NAME_MAX_COMPOSED;
}

/** SET_NAME's payload and GET_NAME's answer: each part as a length and its UTF-8 bytes. */
export function encodeNameParts(name: string, adjective: string): Uint8Array {
  const n = utf8.encode(name);
  const a = utf8.encode(adjective);
  if (n.length > NAME_MAX_COMPOSED || a.length > NAME_MAX_COMPOSED) {
    throw new Invalid(`a name and an adjective are each at most ${NAME_MAX_COMPOSED} bytes`);
  }
  const out = new Uint8Array(2 + n.length + a.length);
  out[0] = n.length;
  out.set(n, 1);
  out[1 + n.length] = a.length;
  out.set(a, 2 + n.length);
  return out;
}

/** GET_NAME's answer or SET_NAME's request, as the device reads or sends it: the name and the adjective. */
export function decodeNameParts(payload: Uint8Array): { name: string; adjective: string } {
  if (payload.length === 0) throw new Truncated("a name carries at least its lengths");
  const nl = payload[0]!;
  if (payload.length < nl + 2) throw new Invalid("a name that ends inside its own parts");
  const al = payload[1 + nl]!;
  const adjective = payload.subarray(2 + nl);
  if (adjective.length !== al) throw new Invalid("a name whose adjective is not the length it declares");
  return { name: utf8Decoder.decode(payload.subarray(1, 1 + nl)), adjective: utf8Decoder.decode(adjective) };
}

/**
 * Labels for a list a person picks devices from, in the list's order: each
 * device's own name, and when two in the list share one, a number after the
 * later ones (`Ada's Blue IntoMind One 2`). The number is the host's and
 * lasts as long as the list. The device knows nothing of it, since two
 * devices cannot agree on which is second without a host that sees them
 * both. A device heard before its name arrived is labeled with the product
 * name.
 */
export function displayNames(names: ReadonlyArray<string | null | undefined>): string[] {
  const seen = new Map<string, number>();
  return names.map((n) => {
    const name = n || PRODUCT_NAME;
    const count = (seen.get(name) ?? 0) + 1;
    seen.set(name, count);
    return count === 1 ? name : `${name} ${count}`;
  });
}

// --- processing (1.1) --------------------------------------------------------
//
// The device preprocesses its own signal. A chain is an ordered list of
// stages, each a kind from the device's catalog with up to four sixteen-bit
// parameters whose units the kind defines. The empty chain is the natural
// signal. When a chain is in force only its output streams, and the chain
// is what a recording stores to say what produced its samples: there is no
// raw-or-filtered flag anywhere, because a flag says nothing about what a
// signal is.

/** The processing stage kinds this version defines, by name. */
export const STAGE_KINDS = { highpass: 1, lowpass: 2, notch: 3 } as const;
/** The name of any stage kind `STAGE_KINDS` defines. */
export type StageKindName = keyof typeof STAGE_KINDS;
/** Stage kind names by their number, the reverse of `STAGE_KINDS`. */
export const KIND_NAMES: Readonly<Record<number, StageKindName>> = { 1: "highpass", 2: "lowpass", 3: "notch" };
/** The classes a stage kind belongs to: map, representation, or detector. */
export const STAGE_CLASSES = { map: 0, representation: 1, detector: 2 } as const;
/** Stage class names by their number, the reverse of `STAGE_CLASSES`. */
export const CLASS_NAMES: Readonly<Record<number, string>> = { 0: "map", 1: "representation", 2: "detector" };
/** 1.3 adds synthetic: reported by a device generating its signal, never requested. */
export const INPUT_SOURCES = { stream: 0, natural: 1, own_chain: 2, synthetic: 3 } as const;
/** The name of any source `INPUT_SOURCES` defines. */
export type InputSourceName = keyof typeof INPUT_SOURCES;
/** Input source names by their number, the reverse of `INPUT_SOURCES`. */
export const INPUT_SOURCE_NAMES: Readonly<Record<number, InputSourceName>> = { 0: "stream", 1: "natural", 2: "own_chain", 3: "synthetic" };
/** The signal classes a model declares it takes, as bits of `inputClasses`. */
export const INPUT_CLASSES = { time_domain: 1 << 0, representation: 1 << 1 } as const;
/** Whether the chain in force is the device's default or one a host set. */
export const PIPELINE_ORIGINS = { default: 0, host: 1 } as const;
/** The name of any origin `PIPELINE_ORIGINS` defines. */
export type OriginName = keyof typeof PIPELINE_ORIGINS;
/** Origin names by their number, the reverse of `PIPELINE_ORIGINS`. */
export const ORIGIN_NAMES: Readonly<Record<number, OriginName>> = { 0: "default", 1: "host" };
/** The bias drive's modes: off, on, or loop open. */
export const BIAS_MODES = { off: 0, on: 1, loop_open: 2 } as const;
/** The name of any mode `BIAS_MODES` defines. */
export type BiasModeName = keyof typeof BIAS_MODES;
/** The most stages one chain may hold. */
export const MAX_STAGES = 12;
/** The most parameters one stage may hold. */
export const MAX_PARAMS = 4;
/** The most notch stages one chain may hold. */
export const MAX_NOTCHES = 8;
/** The device's default high-pass corner, in tenths of a hertz. */
export const DEFAULT_HIGH_PASS_DHZ = 5;
/** Both mains fundamentals, second and third harmonics, four hertz wide: the device's default until a region is chosen. */
export const DEFAULT_MAINS_BANDS_DHZ: ReadonlyArray<readonly [number, number]> = [
  [480, 520],
  [980, 1020],
  [1480, 1520],
  [580, 620],
  [1180, 1220],
  [1780, 1820],
];

/** One stage of a chain: a kind and its parameters, in tenths of a hertz for every kind this version defines. */
export interface Stage {
  /** A `STAGE_KINDS` value. */
  readonly kind: number;
  /** The stage's parameters, in tenths of a hertz for every kind this version defines. */
  readonly params: readonly number[];
}

/** `kind` in words, from `KIND_NAMES`. */
export function stageName(s: Stage): string {
  return KIND_NAMES[s.kind] ?? `kind ${s.kind}`;
}

function dhz(hz: number): number {
  const v = Math.round(hz * 10);
  if (!(v >= 0 && v <= 0xffff)) throw new Invalid(`${hz} Hz is outside what a stage parameter can hold`);
  return v;
}

/** A high-pass at a corner, at least 0.1 Hz. */
export function highpass(cornerHz: number): Stage {
  return { kind: STAGE_KINDS.highpass, params: [dhz(cornerHz)] };
}

/** A low-pass at a corner, or the automatic corner at four tenths of the sample rate when none is given. */
export function lowpass(cornerHz?: number): Stage {
  return { kind: STAGE_KINDS.lowpass, params: [cornerHz === undefined ? 0 : dhz(cornerHz)] };
}

/** A notch over the band between two edges. Any band. */
export function notch(lowHz: number, highHz: number): Stage {
  return { kind: STAGE_KINDS.notch, params: [dhz(lowHz), dhz(highHz)] };
}

/** `u8 n_stages`, then per stage `u8 kind, u8 n_params, u16[n_params]`. */
export function encodeChain(stages: readonly Stage[]): Uint8Array {
  if (stages.length > MAX_STAGES) throw new Invalid(`a chain holds at most ${MAX_STAGES} stages`);
  const out: number[] = [stages.length];
  for (const s of stages) {
    if (!(s.kind >= 1 && s.kind <= 255)) throw new Invalid("a stage kind is one byte and never zero");
    if (s.params.length > MAX_PARAMS) throw new Invalid(`a stage takes at most ${MAX_PARAMS} parameters`);
    out.push(s.kind, s.params.length);
    for (const p of s.params) out.push(p & 0xff, (p >>> 8) & 0xff);
  }
  return new Uint8Array(out);
}

function decodeChainPrefix(data: Uint8Array): [Stage[], Uint8Array] {
  if (data.length === 0) throw new Truncated("a chain is at least its stage count");
  const n = data[0]!;
  if (n > MAX_STAGES) throw new Invalid(`${n} stages is more than a chain holds`);
  let at = 1;
  const stages: Stage[] = [];
  for (let i = 0; i < n; i++) {
    if (data.length < at + 2) throw new Truncated("a chain ends inside a stage");
    const kind = data[at]!;
    const nParams = data[at + 1]!;
    if (kind === 0) throw new Invalid("stage kind zero is not a kind");
    if (nParams > MAX_PARAMS) throw new Invalid(`a stage takes at most ${MAX_PARAMS} parameters`);
    at += 2;
    if (data.length < at + 2 * nParams) throw new Truncated("a stage promises parameters it does not carry");
    const params: number[] = [];
    for (let k = 0; k < nParams; k++) params.push(data[at + 2 * k]! | (data[at + 2 * k + 1]! << 8));
    at += 2 * nParams;
    stages.push({ kind, params });
  }
  return [stages, data.subarray(at)];
}

/** A chain descriptor that is the whole of `data`. */
export function decodeChain(data: Uint8Array): Stage[] {
  const [stages, rest] = decodeChainPrefix(data);
  if (rest.length !== 0) throw new Invalid("bytes after the last stage");
  return stages;
}

/** One kind a device runs: its number, class, parameter count, how many instances a chain may hold, and its name. */
export interface CatalogEntry {
  /** The stage kind's number. */
  readonly kind: number;
  /** The stage class's number. See `STAGE_CLASSES`. */
  readonly cls: number;
  /** `cls` in words. */
  readonly className: string;
  /** Parameters this kind takes. */
  readonly nParams: number;
  /** The most instances of this kind a chain may hold. */
  readonly maxInstances: number;
  /** The kind's name, as the device's catalog states it. */
  readonly name: string;
}

/**
 * The kinds this version of the contract defines, with their limits. A
 * device's own catalog is the authority; this is what a host checks against
 * when it has not read one.
 */
export const CONTRACT_CATALOG: readonly CatalogEntry[] = [
  { kind: 1, cls: 0, className: "map", nParams: 1, maxInstances: 1, name: "highpass" },
  { kind: 2, cls: 0, className: "map", nParams: 1, maxInstances: 1, name: "lowpass" },
  { kind: 3, cls: 0, className: "map", nParams: 2, maxInstances: MAX_NOTCHES, name: "notch" },
];

/** The length of one catalog record, in bytes. */
export const CATALOG_ENTRY_LEN = 12;

/** GET_PIPELINE_CATALOG: `u8 n_kinds`, then records of twelve bytes. */
export function decodeCatalog(payload: Uint8Array): CatalogEntry[] {
  if (payload.length === 0) throw new Truncated("a catalog is at least its count");
  const n = payload[0]!;
  if (payload.length !== 1 + CATALOG_ENTRY_LEN * n) throw new Invalid("a catalog is one byte and twelve per kind, exactly");
  const entries: CatalogEntry[] = [];
  for (let i = 0; i < n; i++) {
    const b = payload.subarray(1 + CATALOG_ENTRY_LEN * i, 13 + CATALOG_ENTRY_LEN * i);
    const raw = b.subarray(4, 12);
    let end = raw.indexOf(0);
    if (end < 0) end = raw.length;
    entries.push({
      kind: b[0]!,
      cls: b[1]!,
      className: CLASS_NAMES[b[1]!] ?? `class ${b[1]}`,
      nParams: b[2]!,
      maxInstances: b[3]!,
      name: new TextDecoder().decode(raw.subarray(0, end)),
    });
  }
  return entries;
}

/** GET_PIPELINE_CATALOG's answer, as the device sends it. */
export function encodeCatalog(entries: readonly CatalogEntry[]): Uint8Array {
  const out = new Uint8Array(1 + CATALOG_ENTRY_LEN * entries.length);
  out[0] = entries.length;
  entries.forEach((e, i) => {
    const at = 1 + CATALOG_ENTRY_LEN * i;
    out[at] = e.kind;
    out[at + 1] = e.cls;
    out[at + 2] = e.nParams;
    out[at + 3] = e.maxInstances;
    out.set(paddedName(e.name, 8), at + 4);
  });
  return out;
}

/** The chain in force, and whether it is the device's own default for its current rate or one a host set. */
export interface PipelineState {
  /** A `PIPELINE_ORIGINS` value. */
  readonly origin: number;
  /** `origin` in words. */
  readonly originName: OriginName;
  /** The chain in force. */
  readonly stages: Stage[];
}

/** GET_PIPELINE's answer, as the device sends it. */
export function decodePipelineState(payload: Uint8Array): PipelineState {
  if (payload.length === 0) throw new Truncated("a pipeline state is at least its origin");
  const origin = payload[0]!;
  const originName = ORIGIN_NAMES[origin];
  if (originName === undefined) throw new Invalid(`origin ${origin} is not one this contract defines`);
  return { origin, originName, stages: decodeChain(payload.subarray(1)) };
}

/** GET_PIPELINE's answer, as the device sends it. */
export function encodePipelineState(origin: number, stages: readonly Stage[]): Uint8Array {
  const chain = encodeChain(stages);
  const out = new Uint8Array(1 + chain.length);
  out[0] = origin;
  out.set(chain, 1);
  return out;
}

/**
 * Where the model's input comes from: the stream as it is, the natural
 * signal, or a chain of the model's own. In an answer the stages are the
 * chain in effect for the model.
 */
export interface PredictionInput {
  /** An `INPUT_SOURCES` value. */
  readonly source: number;
  /** `source` in words. */
  readonly sourceName: InputSourceName;
  /** The chain in effect for the model, when `source` is the model's own. */
  readonly stages: Stage[];
}

/** SET_PREDICTION_INPUT's request, as the device reads it. */
export function encodePredictionInput(source: number | InputSourceName, stages: readonly Stage[] = []): Uint8Array {
  const code = typeof source === "string" ? INPUT_SOURCES[source] : source;
  // Synthetic is reported by a device generating its signal, never requested.
  if (code === undefined || INPUT_SOURCE_NAMES[code] === undefined || code === INPUT_SOURCES.synthetic) {
    throw new Invalid(`${source} is not a source a host may ask for`);
  }
  const chain = encodeChain(stages);
  const out = new Uint8Array(1 + chain.length);
  out[0] = code;
  out.set(chain, 1);
  return out;
}

/** GET_PREDICTION_INPUT's answer, as the device sends it. */
export function decodePredictionInput(payload: Uint8Array): PredictionInput {
  if (payload.length === 0) throw new Truncated("a prediction input is at least its source");
  const source = payload[0]!;
  const sourceName = INPUT_SOURCE_NAMES[source];
  if (sourceName === undefined || source === INPUT_SOURCES.synthetic) {
    throw new Invalid(`source ${source} is not one a host may ask for`);
  }
  return { source, sourceName, stages: decodeChain(payload.subarray(1)) };
}

/** The bias output over the device's own window, in millivolts. */
export interface BiasDiagnostic {
  /** Mean bias output, in millivolts. */
  readonly meanMv: number;
  /** Standard deviation of the bias output, in millivolts. */
  readonly sdMv: number;
  /** Minimum bias output, in millivolts. */
  readonly minMv: number;
  /** Maximum bias output, in millivolts. */
  readonly maxMv: number;
}

/** GET_BIAS_DIAGNOSTIC's answer, as the device sends it. */
export function decodeBiasDiagnostic(payload: Uint8Array): BiasDiagnostic {
  if (payload.length < 8) throw new Truncated("a bias diagnostic is eight bytes");
  const v = view(payload);
  return { meanMv: v.getInt16(0, true), sdMv: v.getInt16(2, true), minMv: v.getInt16(4, true), maxMv: v.getInt16(6, true) };
}

/** GET_BIAS_DIAGNOSTIC's answer, as the device sends it. */
export function encodeBiasDiagnostic(d: BiasDiagnostic): Uint8Array {
  const out = new Uint8Array(8);
  const v = view(out);
  v.setInt16(0, d.meanMv, true);
  v.setInt16(2, d.sdMv, true);
  v.setInt16(4, d.minMv, true);
  v.setInt16(6, d.maxMv, true);
  return out;
}

// --- the status lamp (1.2) ---------------------------------------------------
//
// One language at three levels of verbosity. A blink means the same thing
// at every level that shows it; the level only decides what is shown. The
// blink shapes are the device's and documented on its page; a host explains
// the lamp in words from these tables and never decodes a light.

/** The status lamp's verbosity levels: silent, reserved, or verbose. */
export const INDICATOR_LEVELS = { silent: 0, reserved: 1, verbose: 2 } as const;
/** The name of any level `INDICATOR_LEVELS` defines. */
export type IndicatorLevelName = keyof typeof INDICATOR_LEVELS;
/** Indicator level names by their number, the reverse of `INDICATOR_LEVELS`. */
export const INDICATOR_LEVEL_NAMES: Readonly<Record<number, IndicatorLevelName>> = { 0: "silent", 1: "reserved", 2: "verbose" };
/** What each level shows, most important first. */
export const INDICATOR_SHOWS: Readonly<Record<IndicatorLevelName, readonly string[]>> = {
  silent: [],
  reserved: ["converter fault", "low battery"],
  verbose: ["converter fault", "low battery", "update in progress", "streaming", "connected", "waiting for a host"],
};
/** Low battery begins below this percent of the device's own estimate and ends above the clear percent, or on external power. */
export const LOW_BATTERY_PERCENT = 20;
/** Low battery clears once the device's own estimate rises back above this percent. */
export const LOW_BATTERY_CLEAR_PERCENT = 25;
/** The most seconds one identify request may run. */
export const IDENTIFY_MAX_SECONDS = 30;

// --- the converter's registers, read only (1.2) ------------------------------
//
// The raw configuration bytes of the analog converter, for debugging. The
// contract carries the bytes and a family number. Family one is the Texas
// Instruments ADS1299 family; the IntoMind One carries the ADS1299-4, as its
// datasheet says. There is no write.

/** Which chip family a converter register file belongs to, by its `family` code. */
export const CONVERTER_FAMILIES: Readonly<Record<number, string>> = { 1: "ads1299" };

/** GET_CONVERTER_REGISTERS's answer. */
export interface ConverterRegisters {
  /** Which chip family the registers belong to. See `CONVERTER_FAMILIES`. */
  readonly family: number;
  /** `family` in words. */
  readonly familyName: string;
  /** Address of the first register carried. */
  readonly first: number;
  /** The registers themselves, consecutive addresses starting at `first`. */
  readonly values: Uint8Array;
}

/** GET_CONVERTER_REGISTERS's answer, as the device sends it. */
export function decodeConverterRegisters(payload: Uint8Array): ConverterRegisters {
  if (payload.length < 3) throw new Truncated("registers are at least a family, a first address, and a count");
  const family = payload[0]!;
  const count = payload[2]!;
  if (count === 0 || payload.length !== 3 + count) {
    throw new Invalid(`${count} registers declared and ${payload.length - 3} carried`);
  }
  return { family, familyName: CONVERTER_FAMILIES[family] ?? `family ${family}`, first: payload[1]!, values: payload.slice(3) };
}

/** GET_CONVERTER_REGISTERS's answer, as the device sends it. */
export function encodeConverterRegisters(r: { family: number; first: number; values: Uint8Array }): Uint8Array {
  if (r.values.length === 0 || r.values.length > 64) throw new Invalid("registers carry one to sixty four values");
  const out = new Uint8Array(3 + r.values.length);
  out[0] = r.family;
  out[1] = r.first;
  out[2] = r.values.length;
  out.set(r.values, 3);
  return out;
}

const FAMILY_1_RATES: Readonly<Record<number, number>> = { 0: 16000, 1: 8000, 2: 4000, 3: 2000, 4: 1000, 5: 500, 6: 250 };
const FAMILY_1_GAINS: Readonly<Record<number, number>> = { 0: 1, 1: 2, 2: 4, 3: 6, 4: 8, 5: 12, 6: 24 };
const FAMILY_1_INPUTS = [
  "electrode",
  "shorted",
  "bias measurement",
  "supply",
  "temperature",
  "test signal",
  "bias drive, positive",
  "bias drive, negative",
] as const;
const FAMILY_1_LOFF_THRESHOLD = [95, 92.5, 90, 87.5, 85, 80, 75, 70] as const;
const FAMILY_1_LOFF_CURRENT = ["6 nA", "24 nA", "6 uA", "24 uA"] as const;
const FAMILY_1_LOFF_FREQUENCY = ["dc", "7.8 Hz", "31.2 Hz", "data rate / 4"] as const;

function channelsOf(mask: number): number[] {
  const out: number[] = [];
  for (let c = 0; c < 8; c++) if (mask & (1 << c)) out.push(c + 1);
  return out;
}

/**
 * An ADS1299 family register file in words, from address 0: the identity,
 * the data rate, the test signal, the reference and bias settings, lead-off
 * detection, each channel's gain and input, and the bias and lead-off sense
 * masks. Every field is the datasheet's (Texas Instruments SBAS499C). A
 * file shorter than 24 bytes is described as far as it goes.
 */
export function describeAds1299Registers(values: Uint8Array): Record<string, unknown> {
  const at = (i: number): number | undefined => (i < values.length ? values[i] : undefined);
  const out: Record<string, unknown> = { family: CONVERTER_FAMILIES[1] };
  const ident = at(0);
  if (ident !== undefined) {
    out.identity = {
      device: (ident & 0x1c) === 0x1c ? "ADS1299" : `not an ADS1299 (0x${ident.toString(16)})`,
      familyMember: (ident & 0x1c) === 0x1c,
      revision: (ident >> 5) & 0x7,
      channels: ({ 0: 4, 1: 6, 2: 8 } as Record<number, number>)[ident & 0x3] ?? `code ${ident & 0x3}`,
    };
  }
  const c1 = at(1);
  if (c1 !== undefined) {
    out.dataRateSps = FAMILY_1_RATES[c1 & 0x7] ?? null;
    out.multipleReadback = (c1 & 0x40) !== 0;
    out.clockOutput = (c1 & 0x20) !== 0;
  }
  const c2 = at(2);
  if (c2 !== undefined) {
    out.testSignal = {
      internal: (c2 & 0x10) !== 0,
      amplitude: c2 & 0x04 ? 2 : 1,
      frequency: ({ 0: "fclk / 2^21", 1: "fclk / 2^20", 3: "dc" } as Record<number, string>)[c2 & 0x3] ?? "do not use",
    };
  }
  const c3 = at(3);
  if (c3 !== undefined) {
    out.referenceBufferOn = (c3 & 0x80) !== 0;
    out.bias = {
      amplifierOn: (c3 & 0x04) !== 0,
      referenceInternal: (c3 & 0x08) !== 0,
      measurement: (c3 & 0x10) !== 0,
      sense: (c3 & 0x02) !== 0,
      connected: (c3 & 0x01) === 0,
    };
  }
  const lo = at(4);
  if (lo !== undefined) {
    out.leadOff = {
      thresholdPercent: FAMILY_1_LOFF_THRESHOLD[(lo >> 5) & 0x7],
      current: FAMILY_1_LOFF_CURRENT[(lo >> 2) & 0x3],
      frequency: FAMILY_1_LOFF_FREQUENCY[lo & 0x3],
    };
  }
  const channels: Array<Record<string, unknown>> = [];
  for (let i = 0; i < 8; i++) {
    const ch = at(5 + i);
    if (ch === undefined) break;
    channels.push({
      channel: i + 1,
      poweredDown: (ch & 0x80) !== 0,
      gain: FAMILY_1_GAINS[(ch >> 4) & 0x7] ?? "do not use",
      referenceSwitch: (ch & 0x08) !== 0,
      input: FAMILY_1_INPUTS[ch & 0x7],
    });
  }
  if (channels.length) out.channels = channels;
  const masks: Array<[number, string]> = [
    [13, "biasSensePositive"],
    [14, "biasSenseNegative"],
    [15, "leadOffSensePositive"],
    [16, "leadOffSenseNegative"],
    [17, "leadOffFlip"],
    [18, "leadOffStatusPositive"],
    [19, "leadOffStatusNegative"],
  ];
  for (const [i, name] of masks) {
    const v = at(i);
    if (v !== undefined) out[name] = channelsOf(v);
  }
  const m1 = at(21);
  if (m1 !== undefined) out.commonReferenceSwitch = (m1 & 0x20) !== 0;
  const c4 = at(23);
  if (c4 !== undefined) {
    out.singleShot = (c4 & 0x08) !== 0;
    out.leadOffComparatorsOn = (c4 & 0x02) !== 0;
  }
  return out;
}

// --- embeddings (1.2) --------------------------------------------------------
//
// The encoder's output for one window, in two forms. The window embedding is
// the mean over live channels and over time of the tokens, and is what a
// head consumes. The tokens are the encoder's output before that pooling,
// one vector per channel per slice of the window, and are what reconstruction
// turns back into signal. Both are quantized the way a head receives an
// embedding (times 4096). The weights never leave the device.

/** The embedding packet type. */
export const PACKET_EMBEDDING = 0x03;
/** The length of an embedding notification's header, before its values. */
export const EMBEDDING_HEADER_LEN = 28;
/** The `token` byte meaning this is the window embedding, not one token. */
export const WINDOW_TOKEN = 0xff;
/** Which embeddings a device streams: off, window, tokens, or both. */
export const EMBEDDING_FORMS = { off: 0, window: 1, tokens: 2, both: 3 } as const;
/** The name of any form `EMBEDDING_FORMS` defines. */
export type EmbeddingFormName = keyof typeof EMBEDDING_FORMS;
/** The bits of an embedding notification's flags byte. */
export const EMBEDDING_FLAGS = {
  gap_in_window: 1 << 0,
  duty_reduced: 1 << 1,
  leadoff_in_window: 1 << 2,
  more_parts: 1 << 3,
} as const;

/** One embedding notification, decoded: the whole vector, or the part carried here. */
export interface Embedding {
  /** 0x03 for an embedding. */
  readonly packetType: number;
  /** The window's first sample's index. */
  readonly index: number;
  /** The window's first sample's device time. */
  readonly deviceTime: bigint;
  /** The window's length, in raw samples at the current rate. */
  readonly windowSamples: number;
  /** The encoder that produced this, as `GET_MODEL_INFO` reports it. */
  readonly encoderId: string;
  /** An `INPUT_SOURCES` value. */
  readonly inputSource: number;
  /** Values in the whole vector. */
  readonly embedDim: number;
  /** Index of the first value carried here. */
  readonly first: number;
  /** Null for the window embedding; otherwise the token's index, channel-major over all channels. */
  readonly token: number | null;
  /** Whether another notification carries the rest of this vector. */
  readonly moreParts: boolean;
  /** Whether a gap fell inside the window. */
  readonly gapInWindow: boolean;
  /** Whether the device skipped windows to stay within its compute budget. */
  readonly dutyReduced: boolean;
  /** Whether an electrode was off at some point in the window. */
  readonly leadoffInWindow: boolean;
  /** Quantized values: the encoder's output times 4096. */
  readonly values: number[];
}

/** One embedding notification, as the device sends it. */
export function decodeEmbedding(data: Uint8Array): Embedding {
  if (data.length < EMBEDDING_HEADER_LEN) throw new Truncated("an embedding is at least its header");
  const v = view(data);
  const packetType = data[0]!;
  if (packetType !== PACKET_EMBEDDING) throw new Invalid(`packet type ${toByte(packetType)} is not an embedding`);
  const flags = data[1]!;
  const embedDim = data[2]!;
  const first = data[3]!;
  const body = data.subarray(EMBEDDING_HEADER_LEN);
  if (body.length === 0 || body.length % 2 !== 0) throw new Truncated("an embedding carries whole sixteen-bit values, at least one");
  const n = body.length / 2;
  if (first + n > embedDim) throw new Invalid(`values ${first} to ${first + n} of a vector of ${embedDim}`);
  const values: number[] = [];
  for (let i = 0; i < n; i++) values.push(v.getInt16(EMBEDDING_HEADER_LEN + i * 2, true));
  const token = data[27]!;
  return {
    packetType,
    index: v.getUint32(4, true),
    deviceTime: v.getBigUint64(8, true),
    windowSamples: v.getUint16(16, true),
    encoderId: hex(data.subarray(18, 26)),
    inputSource: data[26]!,
    embedDim,
    first,
    token: token === WINDOW_TOKEN ? null : token,
    moreParts: (flags & EMBEDDING_FLAGS.more_parts) !== 0,
    gapInWindow: (flags & EMBEDDING_FLAGS.gap_in_window) !== 0,
    dutyReduced: (flags & EMBEDDING_FLAGS.duty_reduced) !== 0,
    leadoffInWindow: (flags & EMBEDDING_FLAGS.leadoff_in_window) !== 0,
    values,
  };
}

/** What `encodeEmbedding` takes. */
export interface EmbeddingFields {
  /** The window's first sample's index. */
  index: number;
  /** The window's first sample's device time. */
  deviceTime: bigint;
  /** The window's length, in raw samples. */
  windowSamples: number;
  /** The encoder that produced this. */
  encoderId: string;
  /** An `INPUT_SOURCES` value. */
  inputSource?: number;
  /** Values in the whole embedding. */
  embedDim: number;
  /** Index of the first value carried here. */
  first?: number;
  /** Null for the window embedding, otherwise the token's index. */
  token?: number | null;
  /** Whether another notification carries the rest of this vector. */
  moreParts?: boolean;
  /** Whether a gap fell inside the window. */
  gapInWindow?: boolean;
  /** Whether the device skipped windows to stay within its compute budget. */
  dutyReduced?: boolean;
  /** Whether an electrode was off at some point in the window. */
  leadoffInWindow?: boolean;
  /** The values carried in this notification. */
  values: ArrayLike<number>;
}

/** One embedding notification, as the device sends it. */
export function encodeEmbedding(e: EmbeddingFields): Uint8Array {
  const n = e.values.length;
  const first = e.first ?? 0;
  if (n === 0 || first + n > e.embedDim) throw new Invalid("values must lie inside the vector they belong to");
  const out = new Uint8Array(EMBEDDING_HEADER_LEN + n * 2);
  const v = view(out);
  out[0] = PACKET_EMBEDDING;
  out[1] =
    (e.gapInWindow ? EMBEDDING_FLAGS.gap_in_window : 0) |
    (e.dutyReduced ? EMBEDDING_FLAGS.duty_reduced : 0) |
    (e.leadoffInWindow ? EMBEDDING_FLAGS.leadoff_in_window : 0) |
    (e.moreParts ? EMBEDDING_FLAGS.more_parts : 0);
  out[2] = e.embedDim;
  out[3] = first;
  v.setUint32(4, e.index >>> 0, true);
  v.setBigUint64(8, e.deviceTime, true);
  v.setUint16(16, e.windowSamples, true);
  out.set(fromHex(e.encoderId), 18);
  out[26] = e.inputSource ?? 0;
  out[27] = e.token === null || e.token === undefined ? WINDOW_TOKEN : e.token;
  for (let i = 0; i < n; i++) v.setInt16(EMBEDDING_HEADER_LEN + i * 2, e.values[i]!, true);
  return out;
}

/** One window's embeddings, whole. */
export interface EmbeddingWindow {
  /** The window's first sample's index. */
  readonly index: number;
  /** The window's first sample's device time. */
  readonly deviceTime: bigint;
  /** The window's length, in raw samples. */
  readonly windowSamples: number;
  /** The encoder that produced this. */
  readonly encoderId: string;
  /** An `INPUT_SOURCES` value. */
  readonly inputSource: number;
  /** Values in the whole embedding. */
  readonly embedDim: number;
  /** Whether a gap fell inside the window. */
  readonly gapInWindow: boolean;
  /** Whether the device skipped windows to stay within its compute budget. */
  readonly dutyReduced: boolean;
  /** Whether an electrode was off at some point in the window. */
  readonly leadoffInWindow: boolean;
  /** The window embedding, quantized, when asked for. */
  readonly embedding: number[] | null;
  /** The tokens, quantized, `channels × tokensPerChannel` vectors of `embedDim`, channel-major, when asked for. */
  readonly tokens: number[][] | null;
}

/**
 * Puts notifications back together into whole windows. Built for a device's
 * channel count, the model's tokens per channel, and the form asked for;
 * `feed` takes each decoded notification and returns the window when its
 * last piece arrives. Windows older than the two most recent that never
 * completed are dropped and counted in `incomplete`.
 */
export class EmbeddingAssembler {
  /** The device's channel count, as built. */
  readonly channels: number;
  /** Tokens per channel per window, as built. */
  readonly tokensPerChannel: number;
  /** Which form this was built for. */
  readonly form: EmbeddingFormName;
  /** Windows dropped before their last piece arrived. */
  incomplete = 0;
  readonly #windows = new Map<number, { meta: Embedding; embedding: Array<number | null>; tokens: Array<Array<number | null>> }>();

  /** Build one for a device's channel count, the model's tokens per channel, and the form asked for. */
  constructor(channels: number, tokensPerChannel: number, form: EmbeddingFormName = "both") {
    if (form === "off") throw new Invalid("the form is window, tokens, or both");
    this.channels = channels;
    this.tokensPerChannel = tokensPerChannel;
    this.form = form;
  }

  /** Tokens expected per window: `channels` times `tokensPerChannel`. */
  get nTokens(): number {
    return this.channels * this.tokensPerChannel;
  }

  /** Feed one decoded notification in. Returns the window once its last piece has arrived, else null. */
  feed(e: Embedding): EmbeddingWindow | null {
    const nTokens = this.nTokens;
    let w = this.#windows.get(e.index);
    if (w === undefined) {
      w = {
        meta: e,
        embedding: new Array<number | null>(e.embedDim).fill(null),
        tokens: Array.from({ length: nTokens }, () => new Array<number | null>(e.embedDim).fill(null)),
      };
      this.#windows.set(e.index, w);
      while (this.#windows.size > 3) {
        const oldest = Math.min(...this.#windows.keys());
        this.#windows.delete(oldest);
        this.incomplete += 1;
      }
    }
    const target = e.token === null ? w.embedding : e.token < nTokens ? w.tokens[e.token]! : null;
    if (target === null || e.first + e.values.length > target.length) return null;
    for (let i = 0; i < e.values.length; i++) target[e.first + i] = e.values[i]!;
    const wantEmbedding = this.form === "window" || this.form === "both";
    const wantTokens = this.form === "tokens" || this.form === "both";
    const embeddingDone = !wantEmbedding || w.embedding.every((x) => x !== null);
    const tokensDone = !wantTokens || w.tokens.every((t) => t.every((x) => x !== null));
    if (!(embeddingDone && tokensDone)) return null;
    this.#windows.delete(e.index);
    const m = w.meta;
    return {
      index: m.index,
      deviceTime: m.deviceTime,
      windowSamples: m.windowSamples,
      encoderId: m.encoderId,
      inputSource: m.inputSource,
      embedDim: m.embedDim,
      gapInWindow: m.gapInWindow,
      dutyReduced: m.dutyReduced,
      leadoffInWindow: m.leadoffInWindow,
      embedding: wantEmbedding ? (w.embedding as number[]) : null,
      tokens: wantTokens ? (w.tokens as number[][]) : null,
    };
  }
}

// The rules, the same ones the device applies, so a host can say before
// sending whether a chain runs at a rate and explain a refusal in words.

/** The automatic low-pass corner for a rate, in tenths of a hertz. */
export function lowPassAutoDhz(rateSps: number): number {
  return Math.trunc(rateSps) * 4;
}

/** The first corner a rate refuses, in tenths of a hertz: nine tenths of Nyquist. */
export function cornerLimitDhz(rateSps: number): number {
  return Math.floor((Math.trunc(rateSps) * 45) / 10);
}

/** Whether a chain can run at a rate. Throws `Invalid` saying why not. */
export function checkChain(stages: readonly Stage[], rateSps: number, catalog: readonly CatalogEntry[] = CONTRACT_CATALOG): void {
  const byKind = new Map(catalog.map((e) => [e.kind, e]));
  const limit = cornerLimitDhz(rateSps);
  const counts = new Map<number, number>();
  let hp: number | null = null;
  let lp: number | null = null;
  for (const s of stages) {
    const e = byKind.get(s.kind);
    if (e === undefined) throw new Invalid(`kind ${s.kind} is not in the device's catalog`);
    if (s.params.length !== e.nParams) throw new Invalid(`${e.name} takes ${e.nParams} parameter(s), not ${s.params.length}`);
    const count = (counts.get(s.kind) ?? 0) + 1;
    counts.set(s.kind, count);
    if (count > e.maxInstances) throw new Invalid(`a chain holds at most ${e.maxInstances} ${e.name} stage(s)`);
    const p = s.params;
    if (s.kind === STAGE_KINDS.highpass) {
      if (p[0] === 0) throw new Invalid("a high-pass corner is at least 0.1 Hz");
      if (p[0]! >= limit) throw new Invalid(`a high-pass at ${p[0]! / 10} Hz is at or above nine tenths of Nyquist at ${rateSps} SPS`);
      hp = p[0]!;
    } else if (s.kind === STAGE_KINDS.lowpass) {
      const c = p[0] || lowPassAutoDhz(rateSps);
      if (c >= limit) throw new Invalid(`a low-pass at ${c / 10} Hz is at or above nine tenths of Nyquist at ${rateSps} SPS`);
      lp = c;
    } else if (s.kind === STAGE_KINDS.notch) {
      if (p[0] === 0) throw new Invalid("a band starts above zero");
      if (p[1]! <= p[0]!) throw new Invalid("a band's high edge is above its low edge");
      if (p[1]! >= limit) throw new Invalid(`a band up to ${p[1]! / 10} Hz is at or above nine tenths of Nyquist at ${rateSps} SPS`);
    }
  }
  if (hp !== null && lp !== null && hp >= lp) throw new Invalid("the high-pass meets the low-pass: the band passes nothing");
}

/** The device's own default for a rate: a 0.5 Hz high-pass, the mains bands the rate can represent, the automatic low-pass. */
export function defaultChain(rateSps: number): Stage[] {
  const limit = cornerLimitDhz(rateSps);
  const stages: Stage[] = [{ kind: STAGE_KINDS.highpass, params: [DEFAULT_HIGH_PASS_DHZ] }];
  for (const [lo, hi] of DEFAULT_MAINS_BANDS_DHZ) if (hi < limit) stages.push({ kind: STAGE_KINDS.notch, params: [lo, hi] });
  stages.push({ kind: STAGE_KINDS.lowpass, params: [0] });
  return stages;
}

/**
 * The notch stages for a region's mains: the fundamental, its second and its third harmonic, four hertz wide.
 * Given a rate, only the bands that rate can represent: at 250 samples a second the device refuses the higher harmonics.
 */
export function mainsBands(mainsHz: number, rateSps?: number): Stage[] {
  const f = Math.trunc(mainsHz) * 10;
  const limit = rateSps === undefined ? Infinity : cornerLimitDhz(rateSps);
  return [1, 2, 3]
    .filter((k) => k * f + 20 < limit)
    .map((k) => ({ kind: STAGE_KINDS.notch, params: [k * f - 20, k * f + 20] }));
}

/** The class of signal a chain produces: the time domain unless a stage is a representation change. */
export function outputClass(stages: readonly Stage[], catalog: readonly CatalogEntry[] = CONTRACT_CATALOG): "time_domain" | "representation" {
  const byKind = new Map(catalog.map((e) => [e.kind, e]));
  return stages.some((s) => byKind.get(s.kind)?.cls === STAGE_CLASSES.representation) ? "representation" : "time_domain";
}

/** A chain in words, for a manifest, a header, or a person. */
export function describeChain(stages: readonly Stage[], rateSps?: number): string {
  if (stages.length === 0) return "natural signal";
  const words: string[] = [];
  for (const s of stages) {
    const p = s.params;
    if (s.kind === STAGE_KINDS.highpass && p.length > 0) words.push(`high-pass ${p[0]! / 10} Hz`);
    else if (s.kind === STAGE_KINDS.lowpass && p.length > 0) {
      if (p[0]) words.push(`low-pass ${p[0] / 10} Hz`);
      else if (rateSps) words.push(`low-pass ${lowPassAutoDhz(rateSps) / 10} Hz (automatic at ${rateSps} SPS)`);
      else words.push("low-pass automatic");
    } else if (s.kind === STAGE_KINDS.notch && p.length >= 2) words.push(`notch ${p[0]! / 10} to ${p[1]! / 10} Hz`);
    else words.push(`${stageName(s)} [${p.join(", ")}]`);
  }
  return words.join(", ");
}

// --- the update service -----------------------------------------------------

/** The update service's operations, by name. */
export const UPDATE_OPS = { start: 0x01, query: 0x02, finish: 0x03, activate: 0x04, abort: 0x05 } as const;
/** The name of any operation `UPDATE_OPS` defines. */
export type UpdateOpName = keyof typeof UPDATE_OPS;
/** Operation names by their number, the reverse of `UPDATE_OPS`. */
export const UPDATE_OP_BY_CODE: Readonly<Record<number, UpdateOpName>> = {
  1: "start",
  2: "query",
  3: "finish",
  4: "activate",
  5: "abort",
};

/** What a transfer carries: an application, weights, or a head. */
export const UPDATE_TARGETS = { app: 1, weights: 2, head: 3 } as const;
/** The name of any target `UPDATE_TARGETS` defines. */
export type UpdateTargetName = keyof typeof UPDATE_TARGETS;
/** Target names by their number, the reverse of `UPDATE_TARGETS`. */
export const UPDATE_TARGET_BY_CODE: Readonly<Record<number, UpdateTargetName>> = {
  1: "app",
  2: "weights",
  3: "head",
};

/** What an update control response's status byte means. */
export const UPDATE_STATUS: Readonly<Record<number, string>> = {
  0: "ok",
  1: "invalid argument",
  2: "unsupported",
  3: "busy",
  4: "no transfer",
  5: "flash",
  6: "verification failed",
};

/** A transfer's state, in words, by its QUERY `state` code. */
export const UPDATE_STATES: Readonly<Record<number, string>> = {
  0: "idle",
  1: "receiving",
  2: "complete",
  3: "failed",
};

/** Why FINISH accepted or refused an image, in words, by its `verify_result` code. */
export const VERIFY_RESULTS: Readonly<Record<number, string>> = {
  0: "verified",
  1: "length",
  2: "malformed",
  3: "target",
  4: "slot",
  5: "hash",
  6: "signature",
  7: "security counter",
  8: "head shape",
  9: "flash",
  10: "key id",
};

/** The length of a START request, in bytes. */
export const UPDATE_START_LEN = 15;
/** The length of an image envelope's plain header, in bytes. */
export const ENVELOPE_LEN = 32;
/** An image envelope's magic bytes, as hex. */
export const ENVELOPE_MAGIC = "494d5550";
/** The only envelope version this contract defines. */
export const ENVELOPE_VERSION = 1;
/** The `slot_link` byte for an image that is not linked to an application slot. */
export const SLOT_LINK_NONE = 0xff;

/** A START request, as the device reads it. */
export function encodeUpdateStart(
  target: UpdateTargetName | number,
  slot: number,
  totalLen: number,
  transferId: Uint8Array | string,
): Uint8Array {
  const code = typeof target === "string" ? UPDATE_TARGETS[target] : target;
  if (code === undefined || UPDATE_TARGET_BY_CODE[code] === undefined) {
    throw new Invalid(`${target} is not something this contract transfers`);
  }
  const id = typeof transferId === "string" ? fromHex(transferId) : transferId;
  if (id.length !== 8) throw new Invalid("a transfer identifier is eight bytes");
  const out = new Uint8Array(UPDATE_START_LEN);
  out[0] = UPDATE_OPS.start;
  out[1] = code;
  out[2] = slot;
  view(out).setUint32(3, totalLen, true);
  out.set(id, 7);
  return out;
}

/** A request with no body: QUERY, FINISH, ACTIVATE, or ABORT, as the device reads it. */
export function encodeUpdateOp(op: UpdateOpName): Uint8Array {
  return new Uint8Array([UPDATE_OPS[op]]);
}

/** An update control request, as `decodeUpdateRequest` reads it. */
export interface UpdateRequest {
  /** The raw operation byte. */
  readonly op: number;
  /** `op` in words. */
  readonly opName: UpdateOpName;
  /** What is being transferred, for a START request. Null otherwise. */
  readonly target: number | null;
  /** `target` in words. */
  readonly targetName: UpdateTargetName | null;
  /** The slot, for a START request. Null otherwise. */
  readonly slot: number | null;
  /** The exact byte count the host will send, for a START request. Null otherwise. */
  readonly totalLen: number | null;
  /** The transfer's identifier, for a START request. Null otherwise. */
  readonly transferId: string | null;
}

/**
 * An update control write, as the device reads it. An operation with no
 * body arrives as exactly one byte.
 */
export function decodeUpdateRequest(data: Uint8Array): UpdateRequest {
  if (data.length === 0) throw new Truncated("an empty write is not a request");
  const op = data[0]!;
  const opName = UPDATE_OP_BY_CODE[op];
  if (opName === undefined) throw new Invalid(`${toByte(op)} is not an update operation`);
  const body = data.subarray(1);
  if (opName !== "start") {
    if (body.length !== 0) throw new Invalid(`${opName} takes no body`);
    return { op, opName, target: null, targetName: null, slot: null, totalLen: null, transferId: null };
  }
  if (data.length !== UPDATE_START_LEN) {
    throw new Invalid(`a start is ${UPDATE_START_LEN} bytes, and this one is ${data.length}`);
  }
  const target = body[0]!;
  const targetName = UPDATE_TARGET_BY_CODE[target];
  if (targetName === undefined) throw new Invalid(`target ${target} is not something this contract transfers`);
  return {
    op,
    opName,
    target,
    targetName,
    slot: body[1]!,
    totalLen: view(body).getUint32(2, true),
    transferId: hex(body.subarray(6, 14)),
  };
}

/** An update control response, as `decodeUpdateResponse` reads it. */
export interface UpdateResponse {
  /** The operation byte this answers. */
  readonly op: number;
  /** `op` in words, or null for a number this contract does not define. */
  readonly opName: UpdateOpName | null;
  /** The status byte, 0 for success. */
  readonly status: number;
  /** `status` in words. */
  readonly statusName: string;
  /** Whether `status` is 0. */
  readonly ok: boolean;
  /** FINISH says why it refused an image. No other operation carries one. */
  readonly verifyResult: number | null;
  /** `verifyResult` in words, or null when there is none. */
  readonly verifyResultName: string | null;
  /** The bytes after the operation and status. */
  readonly payload: Uint8Array;
}

/** An update control response, as the device sends it. */
export function decodeUpdateResponse(data: Uint8Array): UpdateResponse {
  if (data.length < 2) throw new Truncated("an update answer is at least an operation and a status");
  const op = data[0]!;
  const status = data[1]!;
  // An operation or a status this contract does not define is refused
  // rather than guessed at. A status is what says whether an image was
  // accepted, and an unknown one must never read as success.
  const opName = UPDATE_OP_BY_CODE[op];
  if (opName === undefined) throw new Invalid(`${toByte(op)} is not an update operation this contract defines`);
  const statusName = UPDATE_STATUS[status];
  if (statusName === undefined) throw new Invalid(`status ${status} is not one this contract defines`);
  const payload = data.slice(2);
  // Only FINISH carries a verify result. Reading the first payload byte of
  // any answer would label a start answer's slot number as a verdict.
  const verify = op === UPDATE_OPS.finish && payload.length > 0 ? payload[0]! : null;
  return {
    op,
    opName,
    status,
    statusName,
    ok: status === 0,
    verifyResult: verify,
    verifyResultName: verify === null ? null : VERIFY_RESULTS[verify] ?? `result ${verify}`,
    payload,
  };
}

/** An update control response, as the device sends it. */
export function encodeUpdateResponse(op: number, status: number, payload: Uint8Array = EMPTY): Uint8Array {
  const out = new Uint8Array(2 + payload.length);
  out[0] = op;
  out[1] = status;
  out.set(payload, 2);
  return out;
}

/** START's answer: the slot, the chunk limit, and how much the device already has. */
export interface UpdateStart {
  /** Which slot the device wants the image in. */
  readonly targetSlot: number;
  /** The largest write it takes. */
  readonly chunkMax: number;
  /** How much of this transfer it already has. */
  readonly resumeOffset: number;
}

/** START's answer, as the device sends it. */
export function decodeUpdateStart(payload: Uint8Array): UpdateStart {
  if (payload.length < 7) throw new Truncated("a start answer is seven bytes");
  const v = view(payload);
  return { targetSlot: payload[0]!, chunkMax: v.getUint16(1, true), resumeOffset: v.getUint32(3, true) };
}

/** QUERY's answer: the transfer's state, how much the device holds, and its checksum over that. */
export interface UpdateQuery {
  /** The transfer's state code. See `UPDATE_STATES`. */
  readonly state: number;
  /** `state` in words. */
  readonly stateName: string;
  /** Bytes the device has accepted so far. */
  readonly offset: number;
  /** The device's own checksum over every byte it has accepted. */
  readonly crc32: number;
}

/** QUERY's answer, as the device sends it. */
export function decodeUpdateQuery(payload: Uint8Array): UpdateQuery {
  if (payload.length < 9) throw new Truncated("a query answer is nine bytes");
  const v = view(payload);
  const state = payload[0]!;
  return {
    state,
    stateName: UPDATE_STATES[state] ?? `state ${state}`,
    offset: v.getUint32(1, true),
    crc32: v.getUint32(5, true),
  };
}

/**
 * The plain head of a wrapped image. Everything after it is encrypted and
 * means nothing to a host, which is the point: a host carries an update
 * without being able to read or forge one.
 */
export interface Envelope {
  /** What is being transferred: application or weights. */
  readonly target: number;
  /** `target` in words. */
  readonly targetName: UpdateTargetName;
  /** Which application slot this image is built for. `SLOT_LINK_NONE` for weights. */
  readonly slotLink: number;
  /** The key identifier the envelope names. */
  readonly keyId: number;
  /** The encryption nonce. */
  readonly nonce: Uint8Array;
  /** Length of the encrypted payload, in bytes. */
  readonly plainLen: number;
  /** The envelope's length plus the payload's. */
  readonly totalLen: number;
}

/** An image envelope's plain header, as the device reads it. */
export function decodeEnvelope(data: Uint8Array): Envelope {
  if (data.length < ENVELOPE_LEN) throw new Truncated("an envelope is thirty two bytes");
  if (hex(data.subarray(0, 4)) !== ENVELOPE_MAGIC || data[4] !== ENVELOPE_VERSION) {
    throw new Invalid("not an image envelope this contract knows");
  }
  const target = data[5]!;
  const targetName = target === 1 ? "app" : target === 2 ? "weights" : null;
  if (targetName === null) throw new Invalid(`target ${target} is not something an envelope carries`);
  const slotLink = data[6]!;
  if (targetName === "app" && slotLink > 1) {
    throw new Invalid("an application image is built for slot zero or one");
  }
  if (targetName === "weights" && slotLink !== SLOT_LINK_NONE) {
    throw new Invalid("weights are not built for a slot");
  }
  const plainLen = view(data).getUint32(24, true);
  return {
    target,
    targetName,
    slotLink,
    keyId: data[7]!,
    nonce: data.slice(8, 24),
    plainLen,
    totalLen: ENVELOPE_LEN + plainLen,
  };
}

/** An image envelope's plain header, as the device reads it. */
export function encodeEnvelope(e: {
  target: UpdateTargetName | number;
  slotLink: number;
  keyId: number;
  nonce: Uint8Array;
  plainLen: number;
}): Uint8Array {
  const target = typeof e.target === "string" ? UPDATE_TARGETS[e.target] : e.target;
  if (e.nonce.length !== 16) throw new Invalid("a nonce is sixteen bytes");
  const out = new Uint8Array(ENVELOPE_LEN);
  out.set(fromHex(ENVELOPE_MAGIC), 0);
  out[4] = ENVELOPE_VERSION;
  out[5] = target;
  out[6] = e.slotLink;
  out[7] = e.keyId;
  out.set(e.nonce, 8);
  view(out).setUint32(24, e.plainLen, true);
  return out;
}

// --- heads ------------------------------------------------------------------

/** A head blob's magic bytes, as hex. */
export const HEAD_MAGIC = "494d4844";
/** The length of a format 1 head header, in bytes. */
export const HEAD_HEADER_LEN = 32;
/** 1.3: format 2 appends the eight byte id of the encoder the head was trained beside. */
export const HEAD_HEADER_LEN_2 = 40;
/** The length of a head blob's trailing hash, in bytes. */
export const HEAD_HASH_LEN = 32;
/** The only head kind this contract defines. */
export const HEAD_KIND_LINEAR = 1;
/** The length of a head's name field, in bytes. */
export const HEAD_NAME_LEN = 16;

/**
 * Counts per unit of embedding. Part of the contract: changing it would
 * change every head ever trained.
 */
export const EMBED_SCALE = 4096.0;

/** The exact length of a head blob for a shape and format version. */
export function headBlobLen(inDim: number, outDim: number, version = 2): number {
  return (version === 2 ? HEAD_HEADER_LEN_2 : HEAD_HEADER_LEN) + outDim * inDim + 8 * outDim + HEAD_HASH_LEN;
}

/**
 * The integers a head receives for a float embedding. Saturates rather than
 * wrapping, so an extreme value is clipped and never reversed.
 */
export function quantizeEmbedding(embedding: ArrayLike<number>): number[] {
  const out: number[] = [];
  for (let i = 0; i < embedding.length; i++) {
    const scaled = embedding[i]! * EMBED_SCALE;
    if (scaled >= 32767) out.push(32767);
    else if (scaled <= -32768) out.push(-32768);
    else out.push(Math.trunc(scaled >= 0 ? scaled + 0.5 : scaled - 0.5));
  }
  return out;
}

/** A head, decoded: the weights, and everything needed to run them. */
export interface HeadBlob {
  /** `HEAD_KIND_LINEAR`, the only kind this contract defines. */
  readonly kind: number;
  /** The embedding width this head takes. Must equal the loaded encoder's. */
  readonly inDim: number;
  /** Outputs this head produces. */
  readonly outDim: number;
  /** The head's name. */
  readonly name: string;
  /** The head's identity, the first eight bytes of its SHA-256. */
  readonly headId: string;
  /** One row per output, each row `inDim` signed byte values. */
  readonly weights: number[][];
  /** One integer per output. */
  readonly bias: number[];
  /** One float per output. */
  readonly scale: number[];
  /** The format the file was written in: 1, or 2 with an encoder id. */
  readonly version: number;
  /** 1.3: the encoder the head was trained beside; `NO_ENCODER_ID` when the file does not say. The device never judges it. */
  readonly encoderId: string;
}

/**
 * A head blob, ready to upload.
 *
 * `weights` is one row per output, each row `inDim` values in the range a
 * signed byte holds. `bias` is one integer per output and `scale` one float
 * per output, which together carry whatever range the outputs need. The
 * hash is computed here, because a head the device cannot check is a head
 * the device refuses.
 *
 * `encoderId` names the encoder the head was trained beside (1.3, format 2),
 * so a host can warn when it is uploaded to a device running another;
 * `version` 1 writes the 1.0 header for a device that predates format 2.
 */
export function buildHead(
  weights: ReadonlyArray<ArrayLike<number>>,
  bias: ArrayLike<number>,
  scale: ArrayLike<number>,
  options: { name?: string; inDim?: number; encoderId?: string; version?: number } = {},
): Uint8Array {
  const version = options.version ?? 2;
  if (version !== 1 && version !== 2) throw new Invalid("a head is written in format 1 or 2");
  const encoder = fromHex(options.encoderId ?? NO_ENCODER_ID);
  if (encoder.length !== 8) throw new Invalid("an encoder id is eight bytes, sixteen hex digits");
  if (version === 1 && options.encoderId !== undefined && options.encoderId !== NO_ENCODER_ID) {
    throw new Invalid("format 1 has no room for an encoder id");
  }
  const outDim = weights.length;
  if (outDim === 0) throw new Invalid("a head has at least one output");
  const inDim = options.inDim ?? weights[0]!.length;
  for (const row of weights) {
    if (row.length !== inDim) throw new Invalid("every row of a head is the same width");
  }
  if (bias.length !== outDim || scale.length !== outDim) {
    throw new Invalid("one bias and one scale per output");
  }
  const blob = new Uint8Array(headBlobLen(inDim, outDim, version));
  const v = view(blob);
  blob.set(fromHex(HEAD_MAGIC), 0);
  blob[4] = version;
  blob[5] = HEAD_KIND_LINEAR;
  v.setUint16(6, inDim, true);
  v.setUint16(8, outDim, true);
  blob.set(paddedName(options.name ?? "", HEAD_NAME_LEN), 10);
  if (version === 2) blob.set(encoder, 32);
  let at = version === 2 ? HEAD_HEADER_LEN_2 : HEAD_HEADER_LEN;
  for (const row of weights) {
    for (let i = 0; i < inDim; i++) {
      const w = row[i]!;
      if (!Number.isInteger(w) || w < -128 || w > 127) {
        throw new Invalid("head weights are signed bytes, so they are trained or scaled to fit");
      }
      blob[at++] = w & 0xff;
    }
  }
  for (let o = 0; o < outDim; o++, at += 4) v.setInt32(at, bias[o]!, true);
  for (let o = 0; o < outDim; o++, at += 4) v.setFloat32(at, scale[o]!, true);
  blob.set(sha256(blob.subarray(0, at)), at);
  return blob;
}

/** Read a head back, and check it the way the device does. */
export function decodeHead(blob: Uint8Array): HeadBlob {
  if (blob.length < HEAD_HEADER_LEN + HEAD_HASH_LEN) {
    throw new Truncated("a head is at least a header and a hash");
  }
  if (hex(blob.subarray(0, 4)) !== HEAD_MAGIC || (blob[4] !== 1 && blob[4] !== 2)) {
    throw new Invalid("not a head this contract knows");
  }
  const version = blob[4]!;
  if (blob[5] !== HEAD_KIND_LINEAR) {
    throw new Invalid(`head kind ${blob[5]} is not one this contract defines`);
  }
  const v = view(blob);
  const inDim = v.getUint16(6, true);
  const outDim = v.getUint16(8, true);
  if (blob.length !== headBlobLen(inDim, outDim, version)) {
    throw new Invalid(
      `a head of ${inDim} by ${outDim} is ${headBlobLen(inDim, outDim, version)} bytes, and this one is ${blob.length}`,
    );
  }
  const body = blob.subarray(0, blob.length - HEAD_HASH_LEN);
  const digest = blob.subarray(blob.length - HEAD_HASH_LEN);
  if (hex(sha256(body)) !== hex(digest)) {
    throw new Invalid("a head whose hash does not match its weights is refused, as the device refuses it");
  }
  let at = version === 2 ? HEAD_HEADER_LEN_2 : HEAD_HEADER_LEN;
  const weights: number[][] = [];
  for (let o = 0; o < outDim; o++) {
    const row: number[] = [];
    for (let i = 0; i < inDim; i++) row.push(v.getInt8(at + o * inDim + i));
    weights.push(row);
  }
  at += outDim * inDim;
  const bias: number[] = [];
  for (let o = 0; o < outDim; o++) bias.push(v.getInt32(at + o * 4, true));
  at += outDim * 4;
  const scale: number[] = [];
  for (let o = 0; o < outDim; o++) scale.push(v.getFloat32(at + o * 4, true));
  return {
    kind: blob[5]!,
    inDim,
    outDim,
    name: nameOf(blob.subarray(10, 26)),
    headId: hex(digest.subarray(0, 8)),
    weights,
    bias,
    scale,
    version,
    encoderId: version === 2 ? hex(blob.subarray(32, 40)) : NO_ENCODER_ID,
  };
}

/**
 * What a head produces, computed the way the device computes it: the
 * accumulation is exact in integers and only the scale is a float.
 *
 * `embedding` is the integers a head receives, which is what
 * `quantizeEmbedding` returns. Floats are not taken here, because a head
 * that was handed floats would silently produce something the device never
 * would.
 */
export function evaluateHead(head: HeadBlob, embedding: ArrayLike<number>): number[] {
  if (embedding.length !== head.inDim) {
    throw new Invalid(`this head takes an embedding of ${head.inDim}, and it was given ${embedding.length}`);
  }
  const out: number[] = [];
  for (let o = 0; o < head.outDim; o++) {
    // Every term is an integer under two to the fifty third, so the
    // accumulation is exact: a byte weight times a sixteen bit count is at
    // most two to the twenty second, and a head would need two billion
    // inputs before the sum could round.
    let acc = head.bias[o]!;
    for (let i = 0; i < head.inDim; i++) {
      const e = embedding[i]!;
      if (!Number.isInteger(e)) {
        throw new Invalid("a head takes the quantized embedding, from quantizeEmbedding");
      }
      acc += head.weights[o]![i]! * e;
    }
    out.push(head.scale[o]! * acc);
  }
  return out;
}
