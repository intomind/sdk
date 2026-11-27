/**
 * The instrument layer, as the Python library has it: what a device can do,
 * composed from what it declares; electrode contact, decoded from the
 * lead-off bits the stream carries; the tiers that gate what a person may
 * change; and the one precondition that refuses a run.
 *
 * Each does exactly what its Python counterpart does, held to the library's
 * own output by the fixture in `contract/profile`: `profile` is the
 * library's `Device.profile`, and the rest is its `instrument` module. What
 * the library refuses with one of Python's own exceptions is refused here
 * with the package's, its words kept: `Invalid` for a value, `TypeError` for
 * a type.
 */

import {
  BIAS_MODES,
  EMBEDDING_FORMS,
  GAIN_BY_CODE,
  INDICATOR_LEVELS,
  Invalid,
  NAME_MAX_COMPOSED,
  PRODUCT_NAME,
  type DeviceInfo,
} from "./protocol.ts";
import * as J from "./capture/pyjson.ts";
import { cannotCheckContact } from "./firmware.ts";
import type { Json, Value } from "./capture/pyjson.ts";

// --- what a device can do ----------------------------------------------------

/**
 * One control, as a profile describes it, with whatever else drawing it
 * needs. A control the device claims but cannot use on its firmware is
 * `supported: false` and carries `why`.
 */
export interface Control {
  /** Whether the device can use the control on its firmware: false for `leadoff` on an IntoMind device whose firmware is older than 1.4.6. */
  readonly supported: boolean;
  /** For a control that is not supported, a sentence an application shows in its place. */
  readonly why?: string;
  /** The control's other fields: its options, its tier, and the rest. */
  readonly [field: string]: unknown;
}

/** What a device can do, for an application that draws its controls from it. */
export interface Profile {
  /** Who made the device: "IntoMind", for every device this module describes. */
  readonly vendor: string;
  /** The hardware version the device reports, `major.minor.patch`, or null where it reports none. */
  readonly hardware: string | null;
  /** The controls, by name, in the order the library lists them. */
  readonly controls: Readonly<Record<string, Control>>;
  /**
   * How the electrodes are wired, present only for a hardware version with a
   * wiring preset: a layout as the library's `placement` builds one, and
   * whether every electrode sits at one locus, so an application asks a
   * person only where the device sat.
   */
  readonly wiring?: { readonly layout: { readonly [field: string]: unknown }; readonly one_locus: true };
}

/**
 * How a device's electrodes are wired, by the hardware version it reports,
 * because the protocol gives a device no way to report its wiring. The key
 * is the hardware version and never the name, which a person can change. Only a released device is here, and every other
 * hardware version gets no preset rather than a guess.
 */
const ONE_LOCUS: readonly (readonly [number, number, number])[] = [[1, 1, 7]];

/**
 * The wiring preset for the hardware a device reports, or undefined. One
 * locus: every electrode at one spot, each channel single-ended against a
 * reference at its center, and a bias electrode there too when the device
 * can drive one.
 */
function wiring(info: DeviceInfo): Profile["wiring"] {
  const hw = info.hardware;
  if (hw === null || !ONE_LOCUS.some((h) => h[0] === hw[0] && h[1] === hw[1] && h[2] === hw[2])) return undefined;
  const n = info.channels;
  const bias = info.can("bias_drive");
  const slot = (key: string, human: string, role: string, name: string) => ({ key, human, role, name, angle_deg: null });
  const slots = [];
  for (let k = 1; k <= n; k++) slots.push(slot(String(k), `channel ${k}`, "recording", `ch${k}`));
  // The part after the last comma is what a cluster's description calls it.
  slots.push(slot("ref", "reference electrode, center", "reference", "ref"));
  if (bias) slots.push(slot("bias", "bias electrode", "bias", "bias"));
  const channels = [];
  for (let k = 1; k <= n; k++) {
    channels.push({ index: k, derivation: "referential", positive: String(k), negative: "ref", label: `Channel ${k}` });
  }
  const s = n !== 1 ? "s" : "";
  const layout = {
    key: `one-locus-${n}` + (bias ? "-bias" : ""),
    human: `${n} channel${s} at one locus around a reference at its center` + (bias ? ", and a bias electrode" : ""),
    slots,
    channels,
    rotatable: false,
    presets: [],
    electrode: null,
    notes: null,
  };
  return { layout, one_locus: true };
}

/**
 * What the device can do, composed from its own capability bits and its own
 * rate mask, as the Python library's `Device.profile` composes it: the same
 * controls, the same fields, the same order. Given no device info, it is a
 * profile with the vendor, no hardware version and no controls.
 *
 * A table of any particular hardware here would be a defect: a fixed gain
 * list and a fixed control set would describe one device and describe every
 * other one wrongly. The one exception is `wiring`, because the protocol
 * gives a device no way to report how its electrodes are wired.
 *
 * A control the device claims but cannot use on its firmware is
 * `supported: false` and carries `why`, a sentence an application shows in
 * its place: `leadoff` on an IntoMind device whose firmware is older than
 * 1.4.6, which cannot check electrode contact, whose `why` is the sentence
 * `setLeadoff(true)` refuses in.
 */
export function profile(info: DeviceInfo | null): Profile {
  if (info === null) return { vendor: "IntoMind", hardware: null, controls: {} };
  const controls: Record<string, Control> = {
    gain: { supported: true, options: [...GAIN_BY_CODE] },
    sample_rate: { supported: true, options: info.rates },
  };
  if (info.can("test_signal") || info.can("input_short") || info.can("synthetic")) {
    // What the amplifier is listening to: a diagnostic rather than an
    // everyday knob. 1.3 adds the signal the device generates itself.
    const options = ["normal"];
    if (info.can("test_signal")) options.push("test");
    if (info.can("input_short")) options.push("short");
    if (info.can("synthetic")) options.push("synthetic");
    controls.signal_source = { supported: true, tier: "advanced", options };
  }
  if (info.can("leadoff")) {
    // The device claims lead-off detection, but on a firmware older than
    // 1.4.6 it does not work: an application shows `why` in place of a
    // contact check, and `setLeadoff(true)` refuses in the same words.
    const cannot = cannotCheckContact(info.firmware);
    controls.leadoff = cannot === null ? { supported: true, options: [false, true] } : { supported: false, options: [false], why: cannot };
  }
  if (info.can("model")) {
    controls.predictions = {
      supported: true,
      ready: info.can("model_ready"),
      head_slots: info.headSlots,
      embedding: info.modelEmbedDim,
    };
  }
  if (info.can("pipeline")) {
    // The chain the device runs on its signal: kinds from the device's
    // catalog, its own default in basic mode, the whole chain in advanced.
    controls.processing = { supported: true, tier: "basic", catalog: "pipeline_catalog()" };
  }
  if (info.can("bias_drive")) controls.bias = { supported: true, tier: "advanced", options: Object.keys(BIAS_MODES) };
  if (info.can("indicator")) {
    // The status lamp's level, which the device keeps across power cycles,
    // and a blink on request so one device can be told from another.
    controls.indicator = { supported: true, options: Object.keys(INDICATOR_LEVELS), identify: true };
  }
  if (info.can("converter_registers")) {
    // A diagnostic read, in words, for the converter family the device
    // declares. Nothing to set.
    controls.converter_registers = { supported: true, tier: "advanced", read_only: true };
  }
  if (info.can("embeddings")) {
    controls.embeddings = { supported: true, forms: Object.keys(EMBEDDING_FORMS).filter((f) => f !== "off") };
  }
  if (info.can("model_cadence")) {
    // 1.3: how often the model describes a window, in seconds, 0 for every
    // window it can. The device states its own minimum.
    controls.model_cadence = { supported: true, unit: "seconds", minimum: "model_interval()" };
  }
  if (info.can("device_name")) {
    // 1.3: a name and an adjective the device keeps and advertises,
    // composed within the bytes a scan response carries.
    controls.name = { supported: true, parts: ["name", "adjective"], product: PRODUCT_NAME, max_bytes: NAME_MAX_COMPOSED };
  }
  const wired = wiring(info);
  return wired === undefined
    ? { vendor: "IntoMind", hardware: info.hardwareString, controls }
    : { vendor: "IntoMind", hardware: info.hardwareString, controls, wiring: wired };
}

// --- contact -----------------------------------------------------------------
//
// Contact comes from the device's own electrode-off detection, carried in
// the stream, never from a heuristic on the signal. The contract's lead-off
// byte has one bit per channel, bit n for channel n+1, positive side only,
// and a set bit means that electrode is off. A device whose own packing
// differs is decoded by its adapter, never here, because the byte's values
// cannot tell one packing from another.

/** A channel has contact when its electrode reads on in at least this share of the samples `contactFromLeadoff` is given, so contact must hold for most of them. */
export const CONTACT_MIN_FRACTION = 0.75;

/** How many channels to decode contact for: the device's own count, or `fallback` where it has not said. */
export function contactChannels(device: { readonly info: DeviceInfo | null } | null, fallback = 2): number {
  return device?.info?.channels || fallback;
}

/** A number that does not fit a byte, refused in the words numpy refuses it with. */
function overflow(n: number): Invalid {
  return new Invalid(`OverflowError: Python integer ${n} out of bounds for uint8`);
}

/** One lead-off byte as numpy holds it in a byte: a boolean as 0 or 1, a fraction cut toward zero. */
function asByte(v: number | boolean): number {
  const n = typeof v === "boolean" ? (v ? 1 : 0) : Math.trunc(v);
  if (!Number.isInteger(n) || n < 0 || n > 255) throw overflow(n);
  return n;
}

/**
 * Contact for each channel, from the lead-off bytes the stream carried: one
 * byte, or one per sample. A channel has contact when the electrode the
 * contract reports for it is on in at least `CONTACT_MIN_FRACTION` of them.
 * No byte at all is no contact anywhere. A byte holds eight channels, so a
 * ninth is refused as the library refuses it.
 *
 * Only the bytes of samples whose `leadoffJudged` is true say anything of
 * contact. An IntoMind device whose firmware is older than 1.4.6 cannot
 * check electrode contact: its lead-off detection does not work, and its
 * bytes read every electrode off whether it touches skin or not.
 */
export function contactFromLeadoff(loff: number | boolean | ArrayLike<number | boolean>, nChannels = 2): boolean[] {
  const bytes = typeof loff === "number" || typeof loff === "boolean" ? [asByte(loff)] : Array.from(loff, asByte);
  if (bytes.length === 0) return Array.from({ length: nChannels }, () => false);
  const out: boolean[] = [];
  for (let ch = 0; ch < nChannels; ch++) {
    const mask = 2 ** ch;
    if (mask > 255) throw overflow(mask);
    let on = 0;
    for (const b of bytes) if ((b & mask) === 0) on++;
    out.push(on / bytes.length >= CONTACT_MIN_FRACTION);
  }
  return out;
}

// --- tiers and the one precondition --------------------------------------------
//
// The tiers gate what a person may change in an application. They never let
// an experiment impose a setting: the settings a person configured, within
// their tier, are exactly what a recording uses. The only precondition left
// is `enforced`, which refuses an unsafe run and never reconfigures anything.

/** What a person may change, from least to most. */
export const TIERS = ["basic", "intermediate", "advanced"] as const;

/** A genuinely unsafe precondition refuses the run at every tier. It refuses, and never reconfigures. */
export class EnforcedViolation extends Error {
  /** Build one with the reason the run was refused. */
  constructor(message: string) {
    super(message);
    this.name = "EnforcedViolation";
  }
}

/**
 * Refuse a run whose parameters break what the experiment declares
 * `enforced`, at every tier, Advanced included: a key whose requirement is
 * the string "required" must be given and truthy, and a key with any other
 * requirement must be given a value equal to it, as Python compares them.
 * None of the package's experiments declares one. Throws
 * `EnforcedViolation` with the library's message, and returns nothing when
 * the run may go. No run calls `checkEnforced`, so an application that
 * wants a run's parameters checked calls it before starting the run.
 */
export function checkEnforced(spec: { readonly name: string; readonly enforced?: Json }, params: Json): void {
  const declared = J.fromPlain(spec.enforced ?? null);
  const enforced = J.truthy(declared) ? declared : J.object([]);
  if (enforced.kind !== "object") throw new TypeError(`AttributeError: '${pyType(enforced)}' object has no attribute 'items'`);
  const given = J.fromPlain(params);
  const required = J.str("required");
  for (const [key, requirement] of enforced.entries) {
    if (given.kind !== "object") throw new TypeError(`AttributeError: '${pyType(given)}' object has no attribute 'get'`);
    const got = J.get(given, key) ?? J.NULL;
    if (J.pyEq(requirement, required) && !J.truthy(got)) {
      throw new EnforcedViolation(`${J.quoted(key)} is required for ${J.quoted(spec.name)} and cannot be overridden at any tier`);
    }
    if (!J.pyEq(requirement, required) && !J.pyEq(got, requirement)) {
      throw new EnforcedViolation(`${J.quoted(key)} must be ${J.jsRepr(requirement)} for ${J.quoted(spec.name)}`);
    }
  }
}

/**
 * Every setting's effective value, for the manifest: the instrument's actual
 * configuration as read from the device, what was applied on top of it, and
 * the experiment's parameters as `param.<name>`. Never a default the
 * experiment wished for, so a run can be reconstructed from its manifest
 * alone. A key already present keeps its place and takes the later value,
 * as a Python dict does.
 */
export function effectiveSettings(spec: unknown, actual: Json, applied: Json, params: Json): Value {
  void spec;
  const eff = J.object([]);
  for (const [what, source, prefix] of [
    ["the actual configuration", actual, ""],
    ["what was applied", applied, ""],
    ["the parameters", params, "param."],
  ] as const) {
    const v = J.fromPlain(source);
    if (v.kind !== "object") throw new TypeError(`${what} is an object of settings, not ${J.jsRepr(v)}`);
    for (const [k, x] of v.entries) J.set(eff, `${prefix}${k}`, x);
  }
  return eff;
}

/** The name Python gives the type of a value JSON can hold. */
function pyType(v: Value): string {
  switch (v.kind) {
    case "null":
      return "null";
    case "array":
      return "list";
    case "object":
      return "dict";
    default:
      return v.kind;
  }
}
