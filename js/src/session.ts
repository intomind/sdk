/**
 * One device, as a state machine.
 *
 * Bytes go in, events come out, and commands go back. Nothing here performs
 * input or output, so the same code runs behind a browser's Bluetooth
 * stack, a desktop one, or a test with no Bluetooth at all.
 *
 * A recording is made here too, as the Python library's recorder makes one:
 * give each time exchange with both host clocks to `onTimeSyncClocks`, which
 * the browser connection's `timeSync` does itself, arm the recorder with
 * `armRecorder`, and take what it collected with
 * `saveRecording`, whose files `capture.Collection` reads back and
 * `capture/node` writes into a folder.
 */

import {
  CODE_BY_GAIN,
  CODE_BY_RATE,
  DeviceInfo,
  FILL,
  Invalid,
  type CapabilityName,
  type OpcodeName,
  type Embedding,
  type Prediction,
  type Response,
  type Status,
  continuity,
  decodeDeviceInfo,
  decodeEmbedding,
  decodePacket,
  decodePrediction,
  decodeResponse,
  decodeStatus,
  decodeTimeSync,
  encodeRequest,
  microvolts,
  BIAS_MODES,
  EMBEDDING_FORMS,
  IDENTIFY_MAX_SECONDS,
  INDICATOR_LEVELS,
  checkChain,
  encodeChain,
  encodePredictionInput,
  MODES,
  OPCODES,
  NAME_MAX_COMPOSED,
  PRODUCT_NAME,
  composeName,
  encodeModelInterval,
  encodeNameParts,
  nameFits,
} from "./protocol.ts";
import type {
  BiasModeName,
  CapabilityHighName,
  CatalogEntry,
  EmbeddingFormName,
  IndicatorLevelName,
  InputSourceName,
  ModeName,
  Stage,
} from "./protocol.ts";
import { Exchange, Timebase, type Fit } from "./timebase.ts";
import { profile, type Profile } from "./instrument.ts";
import { Clock, frameOf, joinFrame, type HostClocks } from "./capture/clock.ts";
import { CaptureError } from "./capture/errors.ts";
import { written } from "./capture/pyjson.ts";
import { roundHalfEven } from "./capture/edf.ts";
import { slotRefusal } from "./slot.ts";
import { JUDGES_FROM, cannotCheckContact, firmwareAtLeast } from "./firmware.ts";
import { Take, collect, hostVersions, saveWith, stateGain, statedGain, type Run, type Saved, type Setup } from "./capture/record.ts";

/** Something a caller asked of a device that the device has not offered. */
export class SessionError extends Error {
  /** Build one with the message explaining what was asked for. */
  constructor(message: string) {
    super(message);
    this.name = "SessionError";
  }
}

/** The device has not said what it is yet, so nothing can be decoded. */
export class NoDeviceInfo extends SessionError {
  constructor() {
    super("read device info first, so nothing has to be assumed");
    this.name = "NoDeviceInfo";
  }
}

/** The device does not claim the capability this needs. */
export class NotCapable extends SessionError {
  /** Which capability was missing. */
  readonly capability: string;

  /** Build one naming the missing capability. */
  constructor(capability: string) {
    super(`this device does not claim the ${capability} capability`);
    this.name = "NotCapable";
    this.capability = capability;
  }
}

/**
 * The device did not answer in time: nothing came back for something asked
 * of it, a control request, an exchange with its update service, or a
 * progress check asked for the last time. It is not a `SessionError`, so it
 * is told apart from a refusal, a link that is down, or a write the system
 * turned away, each of which keeps its own class. The message says what
 * went unanswered.
 */
export class Unanswered extends Error {
  /** Build one with the message saying what went unanswered. */
  constructor(message: string) {
    super(message);
    this.name = "Unanswered";
  }
}

/** Bytes to write, and where to write them. */
export interface Command {
  /**
   * The sixteen bit fill of the characteristic. `protocol.uuid` turns it
   * into the full identifier your stack wants.
   */
  readonly characteristic: number;
  /** The bytes to write. */
  readonly bytes: Uint8Array;
  /** Whether the write must be acknowledged. */
  readonly withResponse: boolean;
}

/**
 * How long the device judges each electrode's contact over, in seconds,
 * from firmware 1.4.6, before its first verdict of a stream and between one
 * verdict and the next. Until the first, every lead-off bit reads 0.
 */
const JUDGMENT_S = 0.512;

/** A run of samples that arrived together. */
export class Batch {
  /** The device's own index for the first sample. */
  readonly index: number;
  /** The device time of the first sample, in the device's ticks. */
  readonly deviceTime: bigint;
  /** Channels per row. */
  readonly channels: number;
  /** Converter counts, one row per sample, laid out row after row. */
  readonly counts: Int32Array;
  /** Lead-off status latched with the most recent sample. */
  readonly leadoff: number;
  /** Whether lead-off detection was running when these samples were taken, from the packet's flag. `leadoff` is zero both when every electrode was attached and when nothing was checking, and this tells the two apart. */
  readonly leadoffActive: boolean;
  /** The gain the samples were taken at. */
  readonly gain: number;
  /** The rate the samples were taken at, or null for a code this contract does not define. */
  readonly sampleRateHz: number | null;
  /** The mode the samples were taken in: 0 normal, 1 test signal, 2 inputs shorted, 3 synthetic (1.3). */
  readonly mode: number;
  /** 1.3: the samples were generated by the device, not measured (packet type 0x04). Never store one as a measurement. */
  readonly synthetic: boolean;
  /** Whether the first sample follows a break, as the Python library's `Sample.gap_before` says of it: the packet says so, as the first of every stream does, or its index is not the next. */
  readonly gapBefore: boolean;
  /**
   * The first row whose lead-off bits are a verdict on each electrode's
   * contact, as the Python library's `Sample.leadoff_judged` says of each
   * sample: 0 where every row's are, and the batch's length where none are.
   * Every row carries the bits of the packet, so they are a verdict for
   * every row of a batch or for none. An IntoMind device whose firmware is
   * older than 1.4.6 cannot check electrode contact: its lead-off detection
   * does not work, and every electrode reads off whether it touches skin or
   * not, so with detection on no row of its batches carries a verdict. From
   * firmware 1.4.6 the device judges each electrode over a block of
   * round(0.512 * rate) consecutive samples, 0.512 s, starts the block over
   * at a sample it loses, and until its first judgment of a stream every bit
   * reads 0, which says neither that an electrode is attached nor that one
   * is off. Each packet carries the bits of its last sample, so in a stream
   * this session started with lead-off detection on no row carries a verdict
   * until a batch whose last sample is the round(0.512 * rate)th or later,
   * counted from the stream's first sample, or from the first sample after a
   * break in the sample index before then; that batch and every one after it
   * carry one. Detection was running for such samples all the same. See
   * `leadoffJudged`.
   */
  readonly judgedFrom: number;

  /** Build from decoded fields. `Session` is what calls this. */
  constructor(f: {
    index: number;
    deviceTime: bigint;
    channels: number;
    counts: Int32Array;
    leadoff: number;
    leadoffActive?: boolean;
    gain: number;
    sampleRateHz: number | null;
    mode?: number;
    synthetic?: boolean;
    gapBefore?: boolean;
    judgedFrom?: number;
  }) {
    this.index = f.index;
    this.deviceTime = f.deviceTime;
    this.channels = f.channels;
    this.counts = f.counts;
    this.leadoff = f.leadoff;
    this.leadoffActive = f.leadoffActive ?? false;
    this.gain = f.gain;
    this.sampleRateHz = f.sampleRateHz;
    this.mode = f.mode ?? 0;
    this.synthetic = f.synthetic ?? false;
    this.gapBefore = f.gapBefore ?? false;
    this.judgedFrom = f.judgedFrom ?? 0;
  }

  /** Whether row `row`'s lead-off bits are a verdict on each electrode's contact: false for every row with detection on from a device whose firmware is older than 1.4.6, and before the device's first lead-off judgment of a stream, see `judgedFrom`. */
  leadoffJudged(row: number): boolean {
    return row >= this.judgedFrom;
  }

  /** Samples in this batch. */
  get rows(): number {
    return this.channels === 0 ? 0 : this.counts.length / this.channels;
  }

  /** One converter count. */
  count(row: number, channel: number): number {
    return this.counts[row * this.channels + channel]!;
  }

  /**
   * One sample, in microvolts, with the device's own reference and width.
   * Nothing is assumed about either.
   */
  microvolts(row: number, channel: number, info: DeviceInfo): number {
    return microvolts(this.count(row, channel), info.vrefUv, this.gain, info.adcBits);
  }
}

/** A break in the timeline, written down rather than smoothed over. */
export interface Gap {
  /** The last index before the break. */
  readonly lastIndexBefore: number;
  /** The first index after the break. */
  readonly firstIndexAfter: number;
  /** Samples missing. Null for a re-base, whose extent is not known. */
  readonly samplesLost: number | null;
  /** Device time of the last sample before the break. */
  readonly deviceTimeBefore: bigint;
  /** Device time of the first sample after the break. */
  readonly deviceTimeAfter: bigint;
  /** Whether this is a re-base rather than a loss. See `samplesLost`. */
  readonly isRebase: boolean;
}

/** What arrived. */
export type Event =
  | { readonly type: "samples"; readonly batch: Batch }
  /** Announced before the batch that follows it. */
  | { readonly type: "gap"; readonly gap: Gap }
  | { readonly type: "status"; readonly status: Status }
  /** A control answer. A caller that issued a command matches on this. */
  | { readonly type: "answer"; readonly answer: Response }
  | { readonly type: "prediction"; readonly prediction: Prediction }
  /** 1.2: one notification of the encoder's output, a whole vector or a part of one. `protocol.EmbeddingAssembler` puts them into whole windows. */
  | { readonly type: "embedding"; readonly embedding: Embedding }
  /** A notification that was not a message. Counted, never guessed at. */
  | { readonly type: "undecodable"; readonly characteristic: number }
  /**
   * A data packet this session already passed on, delivered again by the
   * host's Bluetooth stack: dropped, never passed on twice, and counted. It
   * means another program on the same computer is listening to the device,
   * since the system sends each notification once per subscriber and every
   * subscriber hears all of the copies. `index` is its first sample.
   */
  | { readonly type: "duplicate"; readonly index: number };

/** What a recording is collecting, while the recorder is armed. */
interface Recorder {
  readonly take: Take;
  armed: boolean;
  readonly gapsAtArm: number;
}

/** `slot` as the byte a request carries, or refused with `Invalid`, in words, where it is not a whole number from 0 to 255. */
function slotByte(slot: unknown): number {
  const refused = slotRefusal(slot);
  if (refused !== null) throw new Invalid(refused);
  return Number(slot);
}

/** One device. */
export class Session {
  #info: DeviceInfo | null = null;
  #timebase: Timebase | null = null;
  #previous: { index: number; nSamples: number; deviceTime: bigint } | null = null;
  /** The first index and device time of the last few packets passed on, what a copy is recognized by. */
  #recent: string[] = [];
  /** The map a recording's samples are dated with, as the Python library keeps it. */
  #clock: Clock | null = null;
  #streaming = false;
  #leadoffSetting: boolean | null = null;
  /** What `setLeadoff` last asked for, until the device accepts it: 0 nothing, 1 off, 2 on. */
  #leadoffAsked = 0;
  #recorder: Recorder | null = null;
  // Whether a stream this session started has sent no packet yet, and until
  // the device's first lead-off verdict of it, the index of the sample this
  // session counts the device's block from: the stream's first sample, or
  // the first after the latest break. Null once the verdict has come, and
  // for a stream this session did not start. What tells the samples sent
  // before the device's first lead-off verdict of a stream.
  #streamPending = false;
  #blockFrom: number | null = null;

  /** Samples the device's own indexing says never arrived, over this session. */
  samplesLost = 0;
  /** Breaks counted as a re-base rather than a loss, over this session. */
  rebases = 0;
  /** Notifications that were not a message, over this session: see the "undecodable" event. */
  undecodable = 0;
  /** Data packets delivered twice by the host and dropped: see the "duplicate" event. */
  duplicates = 0;
  /**
   * Breaks the device announced over this session, counted as the Python
   * library counts them: a packet flagged as a discontinuity, which the
   * first packet of every stream is, or one whose index is not the next
   * expected.
   */
  gapsAnnounced = 0;
  /**
   * 1.4: whether USB power ended the last stream. A device that does not
   * stream the natural signal while plugged in ends that stream when USB
   * power appears, and its packets say so.
   */
  endedByUsb = false;

  /** What was read from the device info characteristic. */
  onDeviceInfo(bytes: Uint8Array): DeviceInfo {
    const info = decodeDeviceInfo(bytes);
    // A device of a daisy chain stays on the chain's one reference point.
    const frame = this.#clock === null ? undefined : frameOf(this.#clock);
    this.#timebase = new Timebase(info.tickHz);
    this.#clock = new Clock(info.tickHz);
    if (frame !== undefined) joinFrame(this.#clock, frame);
    this.#info = info;
    return info;
  }

  /** What the device said it is, or null before it has been read. */
  get info(): DeviceInfo | null {
    return this.#info;
  }

  /** The same, for a caller that cannot proceed without it. */
  requireInfo(): DeviceInfo {
    if (this.#info === null) throw new NoDeviceInfo();
    return this.#info;
  }

  /**
   * Whether the device claims a capability. A device that has not said what
   * it is claims nothing.
   */
  can(capability: CapabilityName | string): boolean {
    return this.#info?.can(capability) ?? false;
  }

  /**
   * What the device can do, for an application that draws its controls
   * from it, as the Python library's `Device.profile` composes it from the
   * device's capability bits and rate mask. Before device info, a profile
   * with the vendor, no hardware version and no controls.
   */
  profile(): Profile {
    return profile(this.#info);
  }

  /** The timebase fit from time exchanges, or null before `onDeviceInfo`. */
  get timebase(): Timebase | null {
    return this.#timebase;
  }

  /** Host time for a device time, once enough exchanges have been taken. */
  hostTime(deviceTicks: bigint): number | null {
    return this.#timebase?.hostTime(deviceTicks) ?? null;
  }

  // --- commands -------------------------------------------------------------

  #command(opcode: OpcodeName, arg: number | Uint8Array | null = null): Command {
    return { characteristic: FILL.CONTROL, bytes: encodeRequest(opcode, arg), withResponse: true };
  }

  #require(capability: CapabilityName): void {
    if (!this.can(capability)) throw new NotCapable(capability);
  }

  /** Whether the device claims a capability from the second word of Device Info (1.3): synthetic, device_name. */
  canHigh(capability: CapabilityHighName): boolean {
    return this.#info !== null && this.#info.can(capability);
  }

  #requireHigh(capability: CapabilityHighName): void {
    if (!this.canHigh(capability)) throw new NotCapable(capability);
  }

  /**
   * Ask the device to stream. The stream starts here, and the device's first
   * block of it is counted from the first sample of its first packet, which
   * is what tells its first samples' lead-off bits from verdicts: marked
   * before the command is sent, since a packet can arrive before its answer
   * is read.
   */
  startStream(): Command {
    const command = this.#command("start_stream");
    this.#streamPending = true;
    this.#blockFrom = null;
    return command;
  }

  /**
   * Ask the device to stop streaming. From here the host treats the stream
   * as stopped, answered or not, and the clock's map may move again.
   */
  stopStream(): Command {
    this.#streaming = false;
    this.#clock?.thaw();
    return this.#command("stop_stream");
  }

  /**
   * Set the rate, in samples per second. The device refuses a change while
   * it is streaming, so every sample of a recording sits under one
   * configuration.
   */
  setRate(sps: number): Command {
    const code = CODE_BY_RATE.get(sps);
    if (code === undefined) throw new Invalid(`${written(sps)} samples per second is not a rate this contract defines`);
    return this.#command("set_rate", code);
  }

  /** Set the gain. Refused while streaming. */
  setGain(gain: number): Command {
    const code = CODE_BY_GAIN.get(gain);
    if (code === undefined) throw new Invalid(`a gain of ${written(gain)} is not one this contract defines`);
    return this.#command("set_gain", code);
  }

  /**
   * What the amplifier listens to: 0 the electrodes, 1 the internal test
   * signal, 2 shorted inputs. The last two are capabilities a device
   * declares, so a device that does not offer one is not asked for it.
   */
  setMode(mode: number | ModeName): Command {
    const code = typeof mode === "string" ? MODES[mode] : mode;
    if (code === 1) this.#require("test_signal");
    else if (code === 2) this.#require("input_short");
    // 1.3: the device generates the signal, converter off. Only a device
    // that claims it; the device itself refuses it while streaming and at a
    // rate other than 500 samples a second.
    else if (code === 3) this.#requireHigh("synthetic");
    else if (code !== 0) throw new Invalid(`mode ${written(mode)} is not one this contract defines`);
    return this.#command("set_mode", code);
  }

  /**
   * Turn lead-off detection on or off.
   *
   * An IntoMind device whose firmware is older than 1.4.6 cannot check
   * electrode contact: its lead-off detection does not work, and every
   * electrode reads off whether it touches skin or not. Turning detection
   * on there is refused with a `SessionError` before a command is built, in
   * these words, with the device's own version: "This device's firmware
   * (1.4.5) cannot check electrode contact. Update it to 1.4.6 or later."
   * Turning it off is not refused.
   */
  setLeadoff(on: boolean): Command {
    this.#require("leadoff");
    if (on) {
      const refused = cannotCheckContact(this.#info?.firmware);
      if (refused !== null) throw new SessionError(refused);
    }
    this.#leadoffAsked = on ? 2 : 1;
    return this.#command("set_leadoff", on ? 1 : 0);
  }

  /**
   * Ask the device what time it is. Note the host clock either side and
   * give both to `onTimeSync`.
   */
  timeSync(): Command {
    return this.#command("time_sync");
  }

  /** The answer to a time request, with the host clock either side of it. */
  onTimeSync(payload: Uint8Array, before: number, after: number): Fit | null {
    const ticks = decodeTimeSync(payload);
    if (this.#timebase === null) throw new NoDeviceInfo();
    return this.#timebase.observe(new Exchange(before, after, ticks));
  }

  /**
   * The answer to a time request, with both host clocks read either side of
   * it, which is what a recording's clock needs: `capture.hostClocks()` reads
   * them, monotonic first. The single-clock fit `onTimeSync` keeps is fed
   * from the wall clock too, so `hostTime` still answers.
   */
  onTimeSyncClocks(payload: Uint8Array, before: HostClocks, after: HostClocks): void {
    const ticks = decodeTimeSync(payload);
    if (this.#timebase === null) throw new NoDeviceInfo();
    this.#timebase.observe(new Exchange(before.realtime, after.realtime, ticks));
    this.#clock?.observe(ticks, before, after);
  }

  /**
   * The device's clock as the Python library maps it: fitted against the
   * host's monotonic clock and anchored to its wall clock, and held still
   * while a stream runs. A recording's samples are dated with it and carry
   * its record. Null before device info.
   */
  get clock(): Clock | null {
    return this.#clock;
  }

  /** Whether the device is streaming, as far as this host knows: from a start the device accepted until a stop was asked for, or USB power ended the stream. */
  get streaming(): boolean {
    return this.#streaming;
  }

  /**
   * Whether lead-off detection is running: as the device's packets last
   * declared it, or as this host last set it. Null until either has said,
   * because the protocol has no read for the setting.
   */
  get leadoffSetting(): boolean | null {
    return this.#leadoffSetting;
  }

  /**
   * Arm the recorder: from here every sample and packet is collected until
   * `disarmRecorder`, with what was collected before dropped.
   *
   * Arm where the device's sample index starts again, as the Python library
   * does. On a stream that is running, send `resetEpoch`, let the packets
   * numbered before it drain, about fifty milliseconds, then arm. On one that
   * is not, arm, then send `resetEpoch` and `startStream`.
   */
  armRecorder(): void {
    this.#recorder = { take: new Take(), armed: true, gapsAtArm: this.gapsAnnounced };
  }

  /** Stop collecting. What was collected stays, for `saveRecording`. */
  disarmRecorder(): void {
    if (this.#recorder !== null) this.#recorder.armed = false;
  }

  /** What the recorder has collected since it was armed, or null before it was. */
  get recorded(): Take | null {
    return this.#recorder?.take ?? null;
  }

  /**
   * What the recorder collected as the recording `label`, with the manifest
   * the Python library writes: the device's own facts, its clock's record
   * now, and the breaks announced while armed, with what `setup` declares
   * and `run` adds. A device that ended its stream itself because USB power
   * appeared, as `endedByUsb` says, stopped recording there, and its
   * manifest says so in `ended_by_usb`, as the library's does. The
   * recording comes back as its files' bytes, and `capture/node`'s
   * `writeRecording` puts them in a folder.
   */
  saveRecording(label: string, setup: Setup = {}, run: Run = {}): Saved {
    const info = this.#info;
    if (info === null) throw new CaptureError("read device info first: a recording states what the device said");
    const r = this.#recorder;
    return saveWith(
      label,
      r?.take ?? new Take(),
      {
        deviceId: info.deviceId,
        firmware: info.firmware.join("."),
        protocol: `${info.protocol[0]}.${info.protocol[1]}`,
        tickHz: info.tickHz,
        vrefUv: info.vrefUv,
        adcBits: info.adcBits,
        channels: info.channels,
        clock: this.#clock?.quality() ?? null,
        gapsInRun: Math.max(0, this.gapsAnnounced - (r?.gapsAtArm ?? 0)),
      },
      setup,
      run,
      hostVersions(),
      { run: null, endedByUsb: this.endedByUsb === true, statedGain: statedGain(this) },
    );
  }

  /**
   * Start a fresh epoch of the sample index: the device gives index 1 to the
   * oldest sample it has not yet sent, or with none waiting to the next it
   * takes, and numbers every later sample on from there. In the synthetic
   * signal that sample is index 0.
   */
  resetEpoch(): Command {
    this.#previous = null;
    return this.#command("reset_epoch");
  }

  /** Ask for the measured battery voltage and charge. */
  getBattery(): Command {
    this.#require("battery_voltage");
    return this.#command("get_battery");
  }

  /** Ask why and how many times the device has booted. */
  getBootInfo(): Command {
    return this.#command("get_boot_info");
  }

  /** Turn predictions on or off. */
  setPredictions(on: boolean): Command {
    this.#require("model");
    return this.#command("set_predictions", on ? 1 : 0);
  }

  /** Ask for the model runtime's state. */
  getModelInfo(): Command {
    this.#require("model");
    return this.#command("get_model_info");
  }

  /** Select a head slot. 0 is the built-in head. A slot that is not a whole number from 0 to 255 is refused with `Invalid`, in words. */
  selectHead(slot: number): Command {
    this.#require("heads");
    return this.#command("select_head", slotByte(slot));
  }

  /** Ask for the head slots and which one is selected. */
  listHeads(): Command {
    this.#require("heads");
    return this.#command("list_heads");
  }

  /**
   * Ask which encoder each head names (1.4): one record for each LIST_HEADS record, read with `decodeHeadEncoders`.
   * A device before 1.4 is not asked, and its heads say nothing of their encoder.
   */
  listHeadEncoders(): Command {
    this.#require("heads");
    const [major, minor] = this.#info?.protocol ?? [0, 0];
    if (major < 1 || (major === 1 && minor < 4)) throw new NotCapable("head_encoders");
    return this.#command("list_head_encoders");
  }

  /** Remove a user head slot's contents. A slot that is not a whole number from 0 to 255 is refused with `Invalid`, in words. */
  removeHead(slot: number): Command {
    this.#require("heads");
    return this.#command("remove_head", slotByte(slot));
  }

  // --- processing, on devices that run a chain ------------------------------
  //
  // The device preprocesses its own signal. When a chain is in force only
  // its output streams, and the chain is readable at any time, so a
  // recording can say exactly what produced its samples. The rules a chain
  // must meet are the device's, restated in `protocol.checkChain`, so a
  // chain that cannot run is refused here in words before anything is
  // sent. The device refuses a change while streaming: stop, set, start.

  /** Ask for the stage kinds the device runs. `protocol.decodeCatalog` reads the answer. */
  pipelineCatalog(): Command {
    this.#require("pipeline");
    return this.#command("get_pipeline_catalog");
  }

  /** Ask for the chain in force. `protocol.decodePipelineState` reads the answer. */
  pipeline(): Command {
    this.#require("pipeline");
    return this.#command("get_pipeline");
  }

  /** Put a chain on the device's signal path, checked here first against the rate the device runs at. */
  setPipeline(stages: readonly Stage[], rateSps: number, catalog?: readonly CatalogEntry[]): Command {
    this.#require("pipeline");
    checkChain(stages, rateSps, catalog);
    return this.#command("set_pipeline", encodeChain(stages));
  }

  /** The natural signal, as this host's choice. */
  clearPipeline(): Command {
    this.#require("pipeline");
    return this.#command("clear_pipeline");
  }

  /** The device's own default chain for its current rate. */
  restorePipelineDefault(): Command {
    this.#require("pipeline");
    return this.#command("restore_pipeline_default");
  }

  /** Ask where the model's input comes from. `protocol.decodePredictionInput` reads the answer. */
  predictionInput(): Command {
    this.#require("model");
    return this.#command("get_prediction_input");
  }

  /**
   * Point the model at the stream as it is, at the natural signal, or at a
   * chain of its own, without touching the stream. Only the model's own
   * source carries a chain, checked here against the rate.
   */
  setPredictionInput(source: InputSourceName, stages: readonly Stage[] = [], rateSps = 500, catalog?: readonly CatalogEntry[]): Command {
    this.#require("model");
    if (source === "own_chain") checkChain(stages, rateSps, catalog);
    else if (stages.length !== 0) throw new Invalid("only the model's own chain carries stages");
    return this.#command("set_prediction_input", encodePredictionInput(source, stages));
  }

  /** The bias drive, on a device that claims one: off, on, or loop open. */
  setBias(mode: BiasModeName): Command {
    this.#require("bias_drive");
    const code = BIAS_MODES[mode];
    if (code === undefined) throw new Invalid(`${written(mode)} is not a bias mode this contract defines`);
    return this.#command("set_bias", code);
  }

  /** Ask for the bias output measured. `protocol.decodeBiasDiagnostic` reads the answer. */
  biasDiagnostic(): Command {
    this.#require("bias_drive");
    return this.#command("get_bias_diagnostic");
  }

  // --- the lamp, the converter's registers, embeddings (1.2) ----------------

  /** Ask for the status lamp's level. The answer's one byte is an `INDICATOR_LEVELS` value. */
  indicator(): Command {
    this.#require("indicator");
    return this.#command("get_indicator");
  }

  /**
   * Put the lamp at a level: silent, reserved (the default: low battery and a
   * converter fault at power on), or verbose. The device keeps it across
   * power cycles.
   */
  setIndicator(level: IndicatorLevelName): Command {
    this.#require("indicator");
    const code = INDICATOR_LEVELS[level];
    if (code === undefined) throw new Invalid(`${written(level)} is not a lamp level this contract defines`);
    return this.#command("set_indicator", code);
  }

  /** Blink the lamp so this device can be told from others, for 1 to 30 seconds at every level; 0 stops one in progress. */
  identify(seconds = 5): Command {
    this.#require("indicator");
    if (!Number.isInteger(seconds) || seconds < 0 || seconds > IDENTIFY_MAX_SECONDS) {
      throw new Invalid(`identify runs 1 to ${IDENTIFY_MAX_SECONDS} seconds, or 0 to stop`);
    }
    return this.#command("identify", seconds);
  }

  /** Ask for the converter's registers, raw, for debugging; read only, and only while not streaming. `protocol.decodeConverterRegisters` reads the answer. */
  converterRegisters(): Command {
    this.#require("converter_registers");
    return this.#command("get_converter_registers");
  }

  /**
   * Ask the device for its encoder's output for each window while it
   * streams: "window" for the one vector a head consumes, "tokens" for the
   * per-slice vectors reconstruction consumes, "both", or "off". Whole
   * windows come from `protocol.EmbeddingAssembler` fed with embedding events.
   */
  setEmbeddings(form: EmbeddingFormName): Command {
    this.#require("embeddings");
    const code = EMBEDDING_FORMS[form];
    if (code === undefined) throw new Invalid(`${form} is not an embedding form this contract defines`);
    return this.#command("set_embeddings", code);
  }

  // --- 1.3: the model's cadence and the device's name -----------------------

  /** Ask how often the model describes a window and the least interval the device keeps; `protocol.decodeModelInterval` reads the answer. */
  modelInterval(): Command {
    this.#require("model_cadence");
    return this.#command("get_model_interval");
  }

  /**
   * One described window every `seconds`, on a grid from the stream's start;
   * 0 is every window the device can. The device refuses a value below its
   * minimum and keeps the setting across power cycles.
   */
  setModelInterval(seconds: number): Command {
    this.#require("model_cadence");
    return this.#command("set_model_interval", encodeModelInterval(seconds));
  }

  /** Ask for the name and the adjective a host set; `protocol.decodeNameParts` reads the answer. */
  getName(): Command {
    this.#requireHigh("device_name");
    return this.#command("get_name");
  }

  /**
   * Name the device. It advertises `name's adjective IntoMind One` from the
   * next advertisement and keeps both parts across power cycles. The
   * composition must fit the 29 bytes a scan response carries, so every scan
   * list shows the whole of it; refused here when it does not, or when a
   * part has a leading or trailing space or a control character.
   */
  setName(name: string, adjective: string, product: string = PRODUCT_NAME): Command {
    this.#requireHigh("device_name");
    if (!nameFits(name, adjective, product)) {
      throw new Invalid(`"${composeName(name, adjective, product)}" does not fit the ${NAME_MAX_COMPOSED} bytes a scan response carries, or a part has a leading or trailing space or a control character`);
    }
    return this.#command("set_name", encodeNameParts(name, adjective));
  }

  // --- notifications --------------------------------------------------------

  /**
   * Hand in one notification, and take whatever it meant.
   *
   * A batch that follows a break is preceded by the gap, so a caller that
   * writes samples down writes the gap down first and can never splice.
   */
  onNotification(characteristic: number, bytes: Uint8Array): Event[] {
    switch (characteristic) {
      case FILL.EEG_DATA:
        return this.#onSamples(bytes);
      case FILL.STATUS:
        return this.#decoded(characteristic, () => ({ type: "status", status: decodeStatus(bytes) }));
      case FILL.CONTROL_RESPONSE:
        return this.#decoded(characteristic, () => {
          const answer = decodeResponse(bytes);
          this.#onAnswer(answer);
          return { type: "answer", answer };
        });
      case FILL.PREDICTIONS:
        return this.#decoded(characteristic, () => ({
          type: "prediction",
          prediction: decodePrediction(bytes),
        }));
      case FILL.EMBEDDINGS:
        return this.#decoded(characteristic, () => ({
          type: "embedding",
          embedding: decodeEmbedding(bytes),
        }));
      default:
        return this.#undecodable(characteristic);
    }
  }

  #decoded(characteristic: number, decode: () => Event): Event[] {
    try {
      return [decode()];
    } catch {
      return this.#undecodable(characteristic);
    }
  }

  /**
   * What an answer changes here: a start the device accepted holds the
   * clock's map still under the stream, and a lead-off setting it accepted
   * is the setting.
   */
  #onAnswer(answer: Response): void {
    if (!answer.ok) return;
    if (answer.opcode === OPCODES.start_stream) {
      this.#streaming = true;
      this.endedByUsb = false;
      this.#clock?.freeze();
    } else if (answer.opcode === OPCODES.set_leadoff) {
      const asked = this.#leadoffAsked;
      this.#leadoffAsked = 0;
      if (asked === 1) this.#leadoffSetting = false;
      else if (asked === 2) this.#leadoffSetting = true;
    }
  }

  #undecodable(characteristic: number): Event[] {
    this.undecodable += 1;
    return [{ type: "undecodable", characteristic }];
  }

  #onSamples(bytes: Uint8Array): Event[] {
    const info = this.#info;
    // A device that has not said how wide it is cannot have its samples
    // read, because the payload length only means something against a
    // channel count.
    if (info === null) return this.#undecodable(FILL.EEG_DATA);
    let packet;
    try {
      packet = decodePacket(bytes, info.channels);
    } catch {
      return this.#undecodable(FILL.EEG_DATA);
    }

    const events: Event[] = [];
    const previous = this.#previous;
    // A packet with the first index and the device time of one just passed
    // on is that packet again, delivered twice by the host. Dropped and
    // counted, so no sample is passed on twice and no break is announced
    // that did not happen. A device that restarts its count carries a new
    // device time, so a restart is never taken for a copy, and anything else
    // is judged by the contract's continuity rule unchanged.
    const key = `${packet.index}:${packet.deviceTime}`;
    // The gain from the wire, not from memory.
    stateGain(this, packet.gain);
    if (this.#recent.includes(key)) {
      this.duplicates += 1;
      return [{ type: "duplicate", index: packet.index }];
    }
    this.#recent.push(key);
    if (this.#recent.length > 8) this.#recent.shift();
    // A break as the Python library counts one, for a recording: the packet
    // says so, which the first of every stream does, or its index is not the
    // next.
    const flagged =
      packet.discontinuity || (previous !== null && packet.index !== (previous.index + previous.nSamples) >>> 0);
    if (flagged) this.gapsAnnounced += 1;
    if (previous !== null) {
      const verdict = continuity(previous.index, previous.nSamples, packet.index, packet.discontinuity);
      if (verdict.broken) {
        if (verdict.lost === null) this.rebases += 1;
        else this.samplesLost += verdict.lost;
        events.push({
          type: "gap",
          gap: {
            lastIndexBefore: (previous.index + previous.nSamples - 1) >>> 0,
            firstIndexAfter: packet.index,
            samplesLost: verdict.lost,
            deviceTimeBefore: previous.deviceTime,
            deviceTimeAfter: packet.deviceTime,
            isRebase: verdict.lost === null,
          },
        });
      }
    }
    this.#previous = { index: packet.index, nSamples: packet.nSamples, deviceTime: packet.deviceTime };
    const judged = this.#judged(packet, flagged, info);
    // 1.4: a device that does not stream the natural signal while plugged
    // in ends that stream when USB power appears, and says so in every
    // packet from then on. The samples already converted are still
    // delivered.
    const [major, minor] = info.protocol;
    if (packet.usbPresent && !packet.synthetic && this.#streaming && (major > 1 || (major === 1 && minor >= 4))) {
      this.#streaming = false;
      this.#clock?.thaw();
      this.endedByUsb = true;
    }
    // The packet says whether detection was running: the one read of the
    // setting the protocol offers.
    this.#leadoffSetting = packet.leadoffActive;

    const counts = new Int32Array(packet.nSamples * info.channels);
    for (let row = 0; row < packet.nSamples; row++) {
      for (let c = 0; c < info.channels; c++) counts[row * info.channels + c] = packet.counts[row]![c]!;
    }
    const batch = new Batch({
      index: packet.index,
      deviceTime: packet.deviceTime,
      channels: info.channels,
      counts,
      leadoff: packet.loffStatp,
      leadoffActive: packet.leadoffActive,
      gain: packet.gain,
      sampleRateHz: packet.sps,
      mode: packet.mode,
      synthetic: packet.synthetic,
      gapBefore: flagged,
      judgedFrom: judged ? 0 : packet.nSamples,
    });
    this.#collect(batch, info);
    events.push({ type: "samples", batch });
    return events;
  }

  /**
   * Whether the lead-off bits of `packet`, which each of its samples
   * carries, are a verdict on each electrode's contact, as the Python
   * library's `Device._judged`. `gap` is whether a break comes before the
   * packet: the device flagged one, or its first sample is not the one after
   * the last packet's.
   *
   * An IntoMind device whose firmware is older than 1.4.6 cannot check
   * electrode contact: its lead-off detection does not work, and every
   * electrode reads off whether it touches skin or not. With detection on,
   * no packet of such a device carries a verdict, whoever started the
   * stream.
   *
   * From firmware 1.4.6 the IntoMind device judges each electrode over a
   * block of round(0.512 * rate) consecutive samples, 0.512 s, and gives its
   * first verdict of a stream with the last sample of a whole block. Until
   * then every lead-off bit reads 0, which is no verdict. A sample it loses
   * starts its block over, and a packet carries the bits of its own last
   * sample. So with detection on, the packets of a stream this session
   * started carry no verdict until one whose last sample is the
   * round(0.512 * rate)th or later, counted from the stream's first sample,
   * or from the first after the latest break before then: a break the device
   * made started its block over, and one made on the way, which a host
   * cannot tell from it, is taken as one that did. From the first verdict
   * on, every packet carries one, since the device keeps its last verdict
   * through a break. A stream this session did not start is taken as judged
   * throughout, since nothing here says where it began.
   */
  #judged(packet: { index: number; nSamples: number; leadoffActive: boolean; sps: number | null }, gap: boolean, info: DeviceInfo): boolean {
    if (this.#streamPending) {
      this.#streamPending = false;
      this.#blockFrom = packet.index;
    } else if (gap && this.#blockFrom !== null) {
      this.#blockFrom = packet.index;
    }
    const start = this.#blockFrom;
    if (!packet.leadoffActive) return true;
    if (cannotCheckContact(info.firmware) !== null) return false;
    if (start === null || !packet.sps || !firmwareAtLeast(info.firmware, JUDGES_FROM)) return true;
    // Where the packet's last sample falls in the block, the block's first
    // sample being the 1st.
    const nth = (packet.index + packet.nSamples - start) >>> 0;
    if (nth < roundHalfEven(JUDGMENT_S * packet.sps)) return false;
    this.#blockFrom = null;
    return true;
  }

  /**
   * Give an armed recorder the packet and its samples, each dated as the
   * Python library dates it: the packet's device time stepped by the rate
   * for each sample after the first, and the clock's map now.
   */
  #collect(b: Batch, info: DeviceInfo): void {
    const r = this.#recorder;
    if (r === null || !r.armed) return;
    collect(r.take, b, info.tickHz, this.#clock);
  }
}
