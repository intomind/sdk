/**
 * The session against the contract's own vectors.
 *
 * The wire is held down by `conformance.test.ts`. What these tests hold
 * down is everything built on top of it: that a batch carries the samples
 * the vector states, that a break in the timeline is announced before the
 * samples that follow it, that a loss is counted exactly while a re-base is
 * not counted at all, and that a device is never asked for something it did
 * not claim.
 */

import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import test from "node:test";

import * as P from "../src/protocol.ts";
import { Batch, NoDeviceInfo, NotCapable, Session, type Event } from "../src/session.ts";

const VECTORS = JSON.parse(readFileSync(new URL("../../contract/conformance.json", import.meta.url), "utf8"));

function vector(kind: string, at = 0): { name: string; bytes: string; fields: Record<string, unknown> } {
  return VECTORS.decode.filter((v: { kind: string }) => v.kind === kind)[at];
}

const INFO_BYTES = P.fromHex(vector("device_info").bytes);
const PACKET_BYTES = P.fromHex(vector("eeg_data").bytes);

function ready(): Session {
  const s = new Session();
  s.onDeviceInfo(INFO_BYTES);
  return s;
}

/** The contract's own packet, repointed at another index. */
function repoint(index: number, discontinuity: boolean, nSamples?: number): Uint8Array {
  const p = P.decodePacket(PACKET_BYTES, 4);
  const counts = nSamples === undefined ? p.counts : Array.from({ length: nSamples }, (_, i) => p.counts[i % p.counts.length]!);
  return P.encodePacket({
    index,
    deviceTime: p.deviceTime,
    gainCode: p.gainCode,
    rateCode: p.rateCode,
    counts,
    discontinuity,
    leadoffActive: p.leadoffActive,
    mode: p.mode,
    usbPresent: p.usbPresent,
    loffStatp: p.loffStatp,
  });
}

/** The event at a position, once it is the kind the test is about. */
function at<T extends Event["type"]>(events: Event[], index: number, type: T): Extract<Event, { type: T }> {
  const event = events[index];
  assert.ok(event !== undefined && event.type === type, `event ${index} is ${events[index]?.type}, not ${type}`);
  return event as Extract<Event, { type: T }>;
}

function samples(events: Event[]): Batch {
  const found = events.findIndex((e) => e.type === "samples");
  assert.ok(found >= 0, `no samples in ${JSON.stringify(events.map((e) => e.type))}`);
  return at(events, found, "samples").batch;
}

test("a device says what it is and nothing is assumed", () => {
  const s = ready();
  const info = s.requireInfo();
  assert.equal(info.channels, 4);
  assert.equal(info.adcBits, 24);
  assert.equal(info.tickHz, 1_000_000);
  assert.equal(info.vrefUv, 4_500_000);
  assert.equal(info.modelEmbedDim, 96);
  assert.equal(info.headSlots, 4);
  assert.ok(s.can("model") && s.can("heads") && s.can("update"));
  assert.ok(!s.can("battery_low_flag"));

  // A device that has not said what it is decodes nothing.
  const blank = new Session();
  assert.equal(blank.info, null);
  assert.throws(() => blank.requireInfo(), NoDeviceInfo);
  assert.deepEqual(
    blank.onNotification(P.FILL.EEG_DATA, PACKET_BYTES).map((e) => e.type),
    ["undecodable"],
  );
  assert.equal(blank.undecodable, 1);
});

test("a batch carries the samples the vector states", () => {
  const s = ready();
  const batch = samples(s.onNotification(P.FILL.EEG_DATA, PACKET_BYTES));
  const want = vector("eeg_data").fields.samples as number[][];
  assert.equal(batch.rows, want.length);
  assert.equal(batch.channels, 4);
  assert.equal(batch.gain, 24);
  assert.equal(batch.sampleRateHz, 500);
  assert.equal(batch.index, 3735928559);
  assert.equal(batch.deviceTime, 0x0123456789abcdefn);
  for (const [row, expected] of want.entries()) {
    for (const [channel, value] of expected.entries()) {
      assert.equal(batch.count(row, channel), value, `row ${row} channel ${channel}`);
    }
  }
  // And one count is worth what the device's own numbers say.
  assert.equal(batch.microvolts(0, 0, s.requireInfo()), 0.022351741790771484);
});

test("a break is announced before the samples that follow it", () => {
  const s = ready();

  const first = s.onNotification(P.FILL.EEG_DATA, repoint(100, false));
  assert.deepEqual(first.map((e) => e.type), ["samples"], "the first packet follows nothing");

  // Three samples never arrived: a loss, counted exactly.
  const lost = s.onNotification(P.FILL.EEG_DATA, repoint(105, false));
  assert.deepEqual(lost.map((e) => e.type), ["gap", "samples"], "a loss is announced first");
  const loss = at(lost, 0, "gap").gap;
  assert.equal(loss.samplesLost, 3);
  assert.equal(loss.lastIndexBefore, 101);
  assert.equal(loss.firstIndexAfter, 105);
  assert.ok(!loss.isRebase);
  assert.equal(s.samplesLost, 3);

  // The timeline re-based: a break whose extent is not a number.
  const rebased = s.onNotification(P.FILL.EEG_DATA, repoint(0, true));
  assert.deepEqual(rebased.map((e) => e.type), ["gap", "samples"]);
  const rebase = at(rebased, 0, "gap").gap;
  assert.ok(rebase.isRebase);
  assert.equal(rebase.samplesLost, null, "zero missing would be a different claim from unknown");
  assert.equal(s.samplesLost, 3, "a re-base adds nothing to the loss count");
  assert.equal(s.rebases, 1);
});

test("a packet delivered twice is passed on once", () => {
  // Measured on the bench: the system's Bluetooth service sends each
  // notification once per program subscribed, and every subscriber hears all
  // of the copies. A copy is the same packet, first index and device time.
  const s = ready();
  const packet = repoint(100, false);
  assert.deepEqual(s.onNotification(P.FILL.EEG_DATA, packet).map((e) => e.type), ["samples"]);
  const again = s.onNotification(P.FILL.EEG_DATA, packet);
  assert.deepEqual(again.map((e) => e.type), ["duplicate"], "a copy is not samples and not a break");
  assert.deepEqual([s.duplicates, s.rebases, s.samplesLost], [1, 0, 0]);
});

test("every continuity case in the contract comes out the same way", () => {
  for (const c of VECTORS.continuity) {
    const s = ready();
    s.onNotification(P.FILL.EEG_DATA, repoint(c.prev_index, false, c.prev_n_samples));
    const events = s.onNotification(P.FILL.EEG_DATA, repoint(c.new_index, c.discontinuity_flag));
    const where = JSON.stringify(c);
    if (c.verdict === "continuous") {
      assert.deepEqual(events.map((e) => e.type), ["samples"], where);
      continue;
    }
    const gap = at(events, 0, "gap").gap;
    if (c.verdict === "gap") {
      assert.equal(gap.samplesLost, c.lost, where);
      assert.equal(s.samplesLost, c.lost, where);
    } else {
      assert.ok(gap.isRebase, where);
      assert.equal(gap.samplesLost, null, where);
      assert.equal(s.samplesLost, 0, "a break is never counted as a loss");
    }
  }
});

test("the other notifications decode to what they say", () => {
  const s = ready();

  const status = at(s.onNotification(P.FILL.STATUS, P.fromHex(vector("status").bytes)), 0, "status").status;
  assert.equal(status.state, 1);
  assert.equal(status.bufferFill, 1900);

  const events = s.onNotification(P.FILL.PREDICTIONS, P.fromHex(vector("prediction").bytes));
  const prediction = at(events, 0, "prediction").prediction;
  assert.equal(prediction.headSlot, 2);
  assert.ok(prediction.dutyReduced && !prediction.gapInWindow);
  assert.equal(prediction.outputs.length, 3);

  const answered = s.onNotification(P.FILL.CONTROL_RESPONSE, P.fromHex(vector("control_response").bytes));
  const answer = at(answered, 0, "answer").answer;
  assert.equal(answer.opcode, P.OPCODES.time_sync);
  assert.ok(answer.ok);

  // Anything that is not a message is counted, never guessed at.
  const junk = s.onNotification(P.FILL.STATUS, new Uint8Array(3));
  assert.deepEqual(junk.map((e) => e.type), ["undecodable"]);
  assert.equal(s.undecodable, 1);
  // And so is a notification from a characteristic this session does not read.
  s.onNotification(P.FILL.UPDATE_DATA, new Uint8Array(4));
  assert.equal(s.undecodable, 2);
});

test("a command is the bytes the contract states", () => {
  const s = ready();
  type Fields = Record<string, unknown>;
  const built: Record<number, (f: Fields) => { bytes: Uint8Array; characteristic: number }> = {
    0x01: () => s.startStream(),
    0x10: () => s.setRate(500),
    0x11: () => s.setGain(24),
    0x20: (f) => s.setMode(f.arg as number),
    0x88: (f) => s.setModelInterval(f.interval_s as number),
    0x89: () => s.modelInterval(),
    0x47: () => s.getName(),
    0x48: (f) => s.setName(f.name as string, f.adjective as string),
    0x40: () => s.timeSync(),
    0x80: () => s.setPredictions(true),
    0x81: () => s.selectHead(2),
    0x44: () => s.indicator(),
    0x45: () => s.setIndicator("verbose"),
    0x46: () => s.identify(5),
    0x34: () => s.converterRegisters(),
    0x87: () => s.setEmbeddings("both"),
  };
  let checked = 0;
  for (const v of VECTORS.decode.filter((x: { kind: string }) => x.kind === "control_request")) {
    const make = built[v.fields.opcode as number];
    if (make === undefined) continue;
    checked += 1;
    const command = make(v.fields as Fields);
    assert.deepEqual(command.bytes, P.fromHex(v.bytes), v.name);
    assert.equal(command.characteristic, P.FILL.CONTROL);
  }
  assert.equal(checked, 18, "every control request vector was built from the session");
  // 1.3: a name that does not fit the air is refused here, and the mode the
  // device claims is sent as the contract's byte.
  assert.throws(() => s.setName("Beatrice", "Purple"), P.Invalid);
  assert.equal(s.setName("Beatrice", "Green").bytes[0], 0x48, "8 + 5 makes 29 exactly");
  // Two devices with one name are told apart by the host, never the device.
  assert.deepEqual(P.displayNames(["Ada's IntoMind One", "Blue IntoMind One", "Ada's IntoMind One", null, "Ada's IntoMind One"]),
    ["Ada's IntoMind One", "Blue IntoMind One", "Ada's IntoMind One 2", "IntoMind One", "Ada's IntoMind One 3"]);
  assert.deepEqual(s.setMode("synthetic").bytes, new Uint8Array([0x20, 3]));

  // A setting the contract does not define is refused here rather than on the wire.
  assert.throws(() => s.setRate(333), P.Invalid);
  assert.throws(() => s.setGain(3), P.Invalid);
  assert.throws(() => s.setMode(9), P.Invalid);
  assert.throws(() => s.identify(31), P.Invalid);
  assert.throws(() => s.setIndicator("loud" as P.IndicatorLevelName), P.Invalid);
  assert.deepEqual(s.setEmbeddings("off").bytes, new Uint8Array([0x87, 0]));
});

test("the registers and the embeddings come back whole", () => {
  const regs = P.decodeConverterRegisters(P.fromHex(vector("converter_registers").bytes));
  assert.equal(regs.family, 1);
  assert.equal(regs.values.length, 24);
  const words = P.describeAds1299Registers(regs.values);
  assert.deepEqual(words.identity, { device: "ADS1299", familyMember: true, revision: 1, channels: 4 });
  assert.equal(words.dataRateSps, 250);
  assert.deepEqual((words.channels as Array<Record<string, unknown>>)[4], {
    channel: 5,
    poweredDown: true,
    gain: 24,
    referenceSwitch: false,
    input: "shorted",
  });

  // The session hands an embedding notification on as an event, and the
  // assembler puts a window together from notifications in any order: one
  // channel, three tokens of four values, one token in two parts.
  const s = ready();
  const first = s.onNotification(P.FILL.EMBEDDINGS, P.fromHex(vector("embedding").bytes));
  assert.equal(first.length, 1);
  assert.equal(first[0]!.type, "embedding");
  const packet = (token: number | null, first: number, values: number[], moreParts = false): P.Embedding =>
    P.decodeEmbedding(
      P.encodeEmbedding({ index: 77, deviceTime: 1000n, windowSamples: 2000, encoderId: "0707070707070707", embedDim: 4, first, token, values, moreParts }),
    );
  const a = new P.EmbeddingAssembler(1, 3, "both");
  const fed = [
    a.feed(packet(0, 0, [1, 2, 3, 4])),
    a.feed(packet(2, 0, [9, 9], true)),
    a.feed(packet(null, 0, [5, 6, 7, 8])),
    a.feed(packet(1, 0, [-1, -2, -3, -4])),
    a.feed(packet(2, 2, [8, 8])),
  ];
  assert.deepEqual(fed.slice(0, 4), [null, null, null, null]);
  const w = fed[4]!;
  assert.deepEqual(w.embedding, [5, 6, 7, 8]);
  assert.deepEqual(w.tokens, [
    [1, 2, 3, 4],
    [-1, -2, -3, -4],
    [9, 9, 8, 8],
  ]);
  const b = new P.EmbeddingAssembler(1, 3, "window");
  assert.notEqual(b.feed(packet(null, 0, [1, 1, 1, 1])), null);
  const c = new P.EmbeddingAssembler(1, 3, "tokens");
  for (const index of [1, 2, 3, 4]) {
    c.feed({ ...packet(0, 0, [1, 2, 3, 4]), index });
  }
  assert.equal(c.incomplete, 1);
});

test("a capability the device does not claim is not asked for", () => {
  // A device that claims nothing, which is the same bytes with the
  // capability word cleared.
  const raw = INFO_BYTES.slice();
  raw[15] = 0;
  raw[16] = 0;
  const s = new Session();
  s.onDeviceInfo(raw);

  for (const [capability, ask] of [
    ["leadoff", () => s.setLeadoff(true)],
    ["model", () => s.setPredictions(true)],
    ["model", () => s.getModelInfo()],
    ["heads", () => s.selectHead(1)],
    ["heads", () => s.listHeads()],
    ["heads", () => s.listHeadEncoders()],
    ["heads", () => s.removeHead(1)],
    ["battery_voltage", () => s.getBattery()],
    ["test_signal", () => s.setMode(1)],
    ["input_short", () => s.setMode(2)],
  ] as Array<[string, () => unknown]>) {
    assert.throws(ask, (error: unknown) => {
      assert.ok(error instanceof NotCapable, `${capability}: refused as ${error}`);
      assert.equal(error.capability, capability);
      return true;
    }, `${capability} was asked for on a device that never claimed it`);
  }

  // And the things every device does are still there.
  assert.deepEqual(s.startStream().bytes, new Uint8Array([0x01]));
  assert.deepEqual(s.setMode(0).bytes, new Uint8Array([0x20, 0x00]));
  assert.deepEqual(s.setRate(250).bytes, new Uint8Array([0x10, 0x06]));
});

test("only a 1.4 device is asked which encoder its heads name", () => {
  for (const [minor, asked] of [[3, false], [4, true], [5, true]] as Array<[number, boolean]>) {
    const raw = INFO_BYTES.slice();
    raw[1] = minor;
    const s = new Session();
    s.onDeviceInfo(raw);
    if (asked) assert.deepEqual(s.listHeadEncoders().bytes, new Uint8Array([0x8a]), `1.${minor}`);
    else assert.throws(() => s.listHeadEncoders(), NotCapable, `1.${minor}`);
  }
});

test("a time exchange goes to the timebase the device's tick rate built", () => {
  const s = ready();
  assert.equal(s.hostTime(0n), null, "nothing is mapped before anything is measured");
  const answer = P.decodeResponse(P.fromHex(vector("control_response").bytes));
  const fit = s.onTimeSync(answer.payload, 1000, 1000.004);
  assert.ok(fit !== null);
  assert.equal(fit.exchanges, 1);
  assert.ok(!fit.skewUsed, "a rate cannot be measured from one point");
  // The device's ticks are its own, and the offset puts them on the host
  // clock inside the window this one exchange bracketed.
  const ticks = P.decodeTimeSync(answer.payload);
  const mapped = s.hostTime(ticks)!;
  assert.ok(mapped >= 1000 && mapped <= 1000.004, `${mapped} is outside the window the exchange measured`);

  // An epoch reset drops the previous packet, so the first packet after it
  // follows nothing rather than appearing to jump.
  const running = ready();
  running.onNotification(P.FILL.EEG_DATA, repoint(100, false));
  running.resetEpoch();
  assert.deepEqual(
    running.onNotification(P.FILL.EEG_DATA, repoint(0, false)).map((e) => e.type),
    ["samples"],
  );
  assert.equal(running.rebases, 0);
});
