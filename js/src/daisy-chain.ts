/**
 * Several devices on one host, streamed and recorded together: the Python
 * library's multi-device `Session`, its "daisy chain". This package's
 * `Session` has meant one device's protocol session since 1.4.0, so here it
 * is named for what it is.
 *
 * Each device keeps its own clock against the host's, and every device
 * connected here places its samples' host times on the chain's one
 * reference point, the first clock anchor any of them took, as the
 * library's `Session` does. Two IntoMind devices' samples taken at the same
 * moment carry nearly the same host time: apart by up to the round trips of
 * their clock exchanges, plus however far each device's clock has drifted
 * from the clock map in force when its stream started, the map that dates
 * every sample of that stream. Each device's
 * recording lists that reference point among its clock anchors, so its
 * events still line up with its samples.
 *
 * `recordTo` opens one comma separated file, the bytes the library's
 * `record_to` writes, and every sample of each device `startAll` started is
 * written into it while it is open. `run` runs any registered experiment on
 * any of the chain's devices, alone, in groups or all at once, each device
 * saved as its own recording, as the library's `Session.run` does, and
 * several runs may go at once on different devices.
 *
 * ```ts
 * import { DaisyChain, experiments } from "@intomind/sdk";
 *
 * const chain = new DaisyChain();
 * await chain.connect(first, second, third);    // the devices a person picked
 * const ctx = new experiments.RunCtx(show, { gain: 24 });
 * const wearers = [{ gain: 24, subject_id: "S01" }, { gain: 24, subject_id: "S02" }, { gain: 24, subject_id: "S03" }];
 * const names = await chain.run("record", params, show, { cfgs: wearers, ctx });
 * ```
 */

import { able } from "./boundary.ts";
import { joinFrame, newFrame } from "./capture/clock.ts";
import { fixed, repr } from "./capture/pyfmt.ts";
import * as J from "./capture/pyjson.ts";
import type { Json, Value } from "./capture/pyjson.ts";
import { REGISTRY, RunCtx, type Emit, type Params } from "./experiments.ts";
import { Invalid } from "./protocol.ts";
import { named, runTogether, runtime, type Unit } from "./run.ts";
import { SessionError, type Batch } from "./session.ts";
import { own, tap } from "./taps.ts";
import { Connection, connect, type BluetoothDeviceLike } from "./web-bluetooth.ts";

/** A field as Python's `csv` module writes it with its defaults: quoted only when it holds a comma, a quote or a line break. */
function csvField(s: string): string {
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/** Whether `x` can be walked as a collection of devices, a string excepted, as the library takes any collection or one device on its own. */
function isCollection(x: unknown): x is Iterable<Connection> {
  return typeof x !== "string" && x !== null && typeof x === "object" && typeof (x as Iterable<unknown>)[Symbol.iterator] === "function";
}

/**
 * The file `recordTo` writes: where its bytes go, and how many channels its
 * header names.
 */
interface File {
  readonly write: (chunk: Uint8Array) => void;
  readonly channels: number;
}

/** How many channels a device has, as the file's header counts them: what an IntoMind device said. */
function channelsOf(d: Unit): number {
  return d.session.info?.channels ?? 0;
}

/** Several devices connected to one host and driven together. */
export class DaisyChain {
  /** The connected devices, in the order they were connected: browser connections. */
  devices: Connection[] = [];
  #file: File | null = null;
  /** The file's sink on each device that writes to it, and how to take it off. */
  readonly #sinks = new Map<Connection, () => void>();
  readonly #encoder = new TextEncoder();
  /** The one reference point every device connected here is placed on. */
  readonly #frame = newFrame();
  /** The names this chain's runs have saved a recording under, so a name already taken has a number added. */
  readonly #taken = new Set<string>();

  /**
   * Connect the devices a person picked, in order. Nothing is connected
   * that was not chosen: with two devices in a room, the first to answer is
   * not necessarily the one anybody meant. Every device connected here
   * shares the chain's one reference point for host times, joined before
   * it takes its first clock anchor.
   */
  async connect(...devices: (BluetoothDeviceLike | Connection)[]): Promise<Connection[]> {
    for (const d of devices) await this.#connectDevice(d);
    return this.devices;
  }

  /** Connect one IntoMind device, or a connection already made again, on the chain's reference point. */
  async #connectDevice(src: BluetoothDeviceLike | Connection): Promise<void> {
    const link = src instanceof Connection ? await src.connect() : await connect(src);
    const clock = link.session.clock;
    if (clock !== null) joinFrame(clock, this.#frame);
    this.devices.push(link);
  }

  /**
   * Open a file, written as comma separated text handed to `write`, the lines
   * and bytes the library's `record_to` writes into its file, and give back a
   * function that stops writing. The header row is written at once, and every
   * sample of each device `startAll` started is written as it arrives while
   * the file is open. The header names as many channels as the connected
   * devices have, and every value sits under its own heading: a device with
   * fewer channels than the header names leaves the columns of the channels
   * it lacks empty. `synthetic` says whether the device generated a sample
   * rather than measured it, and the next column whether lead-off detection
   * was running. `leadoff` is left empty for a sample whose lead-off bits are
   * no verdict on contact, which from firmware 1.4.6 is each sample of an
   * IntoMind device's stream with detection on whose bits came before the
   * device's first judgment of it, 0.512 s in or later, and on firmware
   * older than 1.4.6, which cannot check electrode contact, every sample of
   * the device with detection on, see `Sample.leadoffJudged`. A
   * file already being written stops being written first.
   */
  recordTo(write: (chunk: Uint8Array) => void): () => void {
    const devices = [...this.devices, ...this.#sinks.keys()];
    const channels = Math.max(0, ...devices.map(channelsOf));
    if (!channels) {
      throw new SessionError(
        "RuntimeError: connect a device before recording, so the header can name the channels the file will actually carry",
      );
    }
    const file: File = { write, channels };
    this.#file = file;
    this.#row(file, [
      "host_time",
      "device",
      "sample_index",
      "device_time_ticks",
      ...Array.from({ length: channels }, (_, c) => `ch${c + 1}_uV`),
      "leadoff",
      "gap_before",
      "synthetic",
      "leadoff_active",
    ]);
    return () => {
      if (this.#file === file) this.#file = null;
    };
  }

  /**
   * Start streaming on every connected device, passing `options` to each
   * one's `start`. Every device's samples are written to the file
   * `recordTo` opened, by the package's own path, so a listener a program
   * added to a device is left as it is and still hears every sample. A
   * device with more channels than the file's header names is refused, in
   * words that name it and both counts, before any device is started: its
   * samples could not be written under their headings.
   */
  async startAll(options: { readonly samplesPerPacket?: number | null; readonly newEpoch?: boolean } = {}): Promise<void> {
    const file = this.#file;
    if (file !== null) {
      for (const d of this.devices) {
        const n = channelsOf(d);
        if (n > file.channels) {
          throw new SessionError(
            `RuntimeError: ${named(d)} has ${n} channels and the file recordTo opened names ${file.channels}, so its samples ` +
              "cannot be written under their headings: call recordTo again with every device connected",
          );
        }
      }
    }
    for (const d of this.devices) {
      if (!this.#sinks.has(d)) {
        this.#sinks.set(d, tap(d, "batch", (b) => this.#sink(d, b)));
      }
      await d.start(options);
    }
  }

  /** Stop streaming on every connected device. */
  async stopAll(): Promise<void> {
    for (const d of this.devices) await d.stop();
  }

  /**
   * Run one registered experiment, whoever registered it, on the devices
   * `devices` names, every connected device when it names none, and give
   * back the name of every recording it saved, as the library's
   * `Session.run` does.
   *
   * The experiment's function is handed the connection itself when the run
   * has one, as an experiment run on one device is, and a new array of the
   * connections, in the order given, when it has several. Its code may
   * hand any group of them, or a device connected since, to the package's
   * experiments, in turn or at the same time, passing on the `emit` and
   * `ctx` it was handed, and each device handed is saved as its own
   * recording. The `experiments` module shows how.
   *
   * Several runs may go at once on different devices of the chain, each with
   * its own `ctx`, so its own `stop`, markers and events, and its own
   * recordings. A run never resets, stops, arms or saves a device its code
   * did not hand to a step, and adds no consumer of that device's samples. A device that is
   * not one of the chain's, or that is named twice, is refused in words that
   * name it, before anything is sent to any device, and so is a device
   * another step or another run is recording, by the step it is handed to.
   *
   * A step starts the devices it records that were not streaming, and stops
   * only those it started, so a stream that was running before it is left
   * running after it. `ctx.stop` works as in a run on one device. A cue a
   * step stamps is written into the recordings of the devices that step
   * handles, and every other event, as a person's marker, into every
   * recording the run saves after it is stamped. A device whose link is down
   * when a step starts is not recorded, and the run says so, unless every
   * device of the step has its link down: then the step fails with the error
   * raised by the first device it writes to. One whose link drops mid-run has
   * its recording saved with `aborted` set to "link_lost" while the others
   * carry on, and one that ends its stream because USB
   * power appeared is saved complete, with `ended_by_usb` in its manifest.
   * Head training trains each device's own head from its own windows.
   *
   * A run on one device saves its recording under the name its step gives
   * it. A run of several adds each device's id to that name, and where the
   * name gives a gain, puts in it the gain the device's own setup gives, or
   * `na` where its setup gives none. A name a run of
   * this chain already saved has a number added. Each manifest says which run
   * made it under `run`, the experiment's registered name and when the run
   * started, in UTC, the same in every recording of the run, and names under
   * `together` the others its step saved at the same moment.
   *
   * `emit` hears what the run says, one `saved` for each recording, with
   * its files, and every message carries `devices`: the id of each device it
   * concerns, a step's devices for what a step says, one device's for what
   * is said about its recording, and the run's for what the run says.
   * `cfgs` is each device's setup, one a device of the run, in its order, as
   * a run on one device takes its `cfg`: what its instrument was, and who
   * wears it, such as `subject_id` and `notes`, which land in that device's
   * manifest. Pass `ctx` to stop the run or mark an event while it runs;
   * without `ctx`, the run makes its own context, whose `cfg` is the first
   * setup of `cfgs`.
   */
  async run(
    name: string,
    params: Params,
    emit: Emit,
    options: {
      readonly cfgs?: readonly Json[];
      readonly ctx?: RunCtx;
      readonly devices?: Connection | Iterable<Connection>;
    } = {},
  ): Promise<string[]> {
    let devices: Connection[];
    if (options.devices === undefined) {
      devices = [...this.devices];
    } else {
      devices = isCollection(options.devices) ? [...options.devices] : [options.devices];
      devices.forEach((d, i) => {
        if (!this.devices.includes(d)) throw new Invalid(`ValueError: ${named(d)} is not one of this chain's devices`);
        if (devices.slice(0, i).includes(d)) throw new Invalid(`ValueError: ${named(d)} is named twice`);
      });
    }
    const spec = REGISTRY.get(name);
    if (spec === undefined) throw new Invalid(`KeyError: ${J.reprStr(name)}`);
    if (devices.length === 0) throw new Invalid("ValueError: a run needs at least one connected device");
    const asObject = (c: Json): Value => {
      const v = J.fromPlain(c ?? null);
      return J.truthy(v) && v.kind === "object" ? J.copyObject(v) : J.object([]);
    };
    const ctx = options.ctx ?? new RunCtx(emit, options.cfgs?.[0] ?? null);
    const cfgs =
      options.cfgs !== undefined ? options.cfgs.map(asObject) : [J.copyObject(ctx.cfg), ...devices.slice(1).map(() => J.object([]))];
    if (cfgs.length !== devices.length) {
      throw new Invalid(`ValueError: ${cfgs.length} setups were given for ${devices.length} devices, and a run needs one setup for each device`);
    }
    return runTogether(spec, devices, cfgs, params, emit, ctx, this.#taken);
  }

  /**
   * Keep every device's clock fresh while it streams: wait `periodS`
   * seconds, take a time exchange with each device, and again, until
   * `signal` is aborted. A device that does not answer keeps its last fit.
   * A device taking a transfer is passed over until the transfer ends or
   * fails, since it takes nothing else meanwhile.
   */
  async resyncLoop(periodS = 120, signal?: AbortSignal): Promise<void> {
    while (signal === undefined || !signal.aborted) {
      await runtime.time.sleep(periodS);
      if (signal !== undefined && signal.aborted) return;
      for (const d of this.devices) {
        if (!able(d, "clock")) continue;
        if (d instanceof Connection && own(d).transferring) continue;
        try {
          await d.timeSync();
        } catch {
          // As the library: one device that does not answer stops no other.
        }
      }
    }
  }

  /**
   * Disconnect one device, take it out of `devices`, and stop writing its
   * samples to the file `recordTo` opened, as the library's
   * `Session.disconnect` does. The other
   * devices carry on.
   */
  async disconnect(device: Connection): Promise<void> {
    const off = this.#sinks.get(device);
    if (off !== undefined) {
      off();
      this.#sinks.delete(device);
    }
    const i = this.devices.indexOf(device);
    if (i >= 0) this.devices.splice(i, 1);
    await device.disconnect();
  }

  /** Disconnect every connected device, and stop writing the file `recordTo` opened. */
  async close(): Promise<void> {
    try {
      for (const d of this.devices) await d.disconnect();
    } finally {
      for (const off of this.#sinks.values()) off();
      this.#sinks.clear();
      this.#file = null;
    }
  }

  /** One line of the file, ended as Python's `csv` module ends one. */
  #row(file: File, fields: readonly string[]): void {
    file.write(this.#encoder.encode(`${fields.map(csvField).join(",")}\r\n`));
  }

  /** One sample's row. */
  #rowOf(file: File, dev: Connection, fields: string[], n: number): void {
    this.#row(file, fields);
  }

  /**
   * Every sample of a batch as the library's `_sink` writes it: the host's
   * time to the microsecond, the last four characters of the device's id,
   * the index, the device's time in ticks, each channel in microvolts as the
   * library's `Sample.uv` holds it and an empty field for each channel the
   * header names that the device lacks, the lead-off status, or nothing
   * where it is no verdict on contact, whether the sample follows a break,
   * whether it was generated, and whether detection ran. A device with more
   * channels than the header names is refused in the library's words, as
   * each sample of it is.
   */
  #sink(link: Connection, b: Batch): void {
    const file = this.#file;
    if (file === null) return;
    const empty = file.channels - b.channels;
    if (empty < 0) {
      throw new Invalid(`ValueError: ${named(link)} has ${b.channels} channels, and the file's header names ${file.channels}`);
    }
    const info = link.session.requireInfo();
    // `DeviceInfo.microvolts_per_count`, then each count times it rounded to
    // three places, in the library's order.
    const lsb = (2.0 * info.vrefUv) / (b.gain * 2 ** info.adcBits);
    const rate = b.sampleRateHz;
    const tickPer = rate !== null && rate > 0 ? info.tickHz / rate : 0;
    const clock = link.session.clock;
    for (let i = 0; i < b.rows; i++) {
      const ticks = tickPer === 0 ? b.deviceTime : BigInt(Math.trunc(Number(b.deviceTime) + i * tickPer));
      const host = clock !== null ? clock.hostTime(ticks) : Number(ticks) / info.tickHz;
      const uv: string[] = [];
      for (let c = 0; c < b.channels; c++) uv.push(repr(Number(fixed(b.count(i, c) * lsb, 3))));
      for (let c = 0; c < empty; c++) uv.push("");
      this.#rowOf(
        file,
        link,
        [
          fixed(host, 6),
          info.deviceId.slice(-4),
          String(b.index + i),
          ticks.toString(),
          ...uv,
          b.leadoffJudged(i) ? String(b.leadoff) : "",
          b.gapBefore && i === 0 ? "1" : "0",
          b.synthetic ? "1" : "0",
          b.leadoffActive ? "1" : "0",
        ],
        b.channels,
      );
    }
  }
}
