/**
 * A transfer, driven end to end against a device that answers the way
 * the contract says a device answers.
 *
 * The device here is a few dozen lines rather than a mock: it holds
 * bytes, counts them, checksums them, and answers. That is enough to
 * test the state machine against, and the same suite exists in Rust so
 * the two are held to one behavior.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { crc32, sha256 } from "../src/digest.ts";
import * as P from "../src/protocol.ts";
import { Session } from "../src/session.ts";
import {
  ChecksumMismatch,
  OffsetMismatch,
  Refused,
  Rejected,
  Transfer,
  TransferError,
  type Step,
} from "../src/transfer.ts";

/** Bytes in a page of the device's flash, which it writes whole. */
const PAGE = 4096;
/** Incoming writes the device holds while it writes a page. */
const HOLDS = 8;
/** The transport key a device that judges envelopes holds. */
const KEY_ID = 1;
/** The slot such a device takes an application into, which its answer to START gives. */
const IDLE_SLOT = 0;

/**
 * The device's verdict on an image's envelope, its first 32 bytes, in the
 * order the firmware checks them, for a transfer of `total` bytes to
 * `target`, on a device that holds the transport key `KEY_ID` and takes an
 * application into `IDLE_SLOT`: null when it takes it.
 */
function judged(envelope: Uint8Array, target: string, total: number): number | null {
  let e: P.Envelope;
  try {
    e = P.decodeEnvelope(envelope);
  } catch {
    return 2; // malformed
  }
  if (e.targetName !== target) return 3; // target
  if (target === "app" && e.slotLink !== IDLE_SLOT) return 4; // slot
  if (e.keyId !== KEY_ID) return 10; // key id
  if (e.totalLen !== total) return 1; // length
  return null;
}

/**
 * An application image of `total` bytes whose envelope names `IDLE_SLOT`,
 * the transport key `keyId` and the image's own length, the bytes after it
 * counting through every byte value, as the Python library's tests make one.
 */
function enveloped(total: number, keyId = KEY_ID): Uint8Array {
  const out = new Uint8Array(total);
  out.set(P.encodeEnvelope({ target: "app", slotLink: IDLE_SLOT, keyId, nonce: Uint8Array.from({ length: 16 }, (_, i) => i), plainLen: total - P.ENVELOPE_LEN }));
  for (let i = P.ENVELOPE_LEN; i < total; i++) out[i] = (i - P.ENVELOPE_LEN) & 0xff;
  return out;
}

/**
 * A device that takes an image and says what it has, with a real one's
 * limits: writing a page of its flash takes time, and meanwhile it holds
 * eight incoming writes and throws away what arrives beyond them, a check
 * among them. Time passes only while this side waits for an answer or
 * pauses, so a burst of writes arrives back to back.
 *
 * With `judgesEnvelopes` set, it judges an image's first 32 bytes as its
 * envelope once they are in, from whichever writes carried them, as the
 * firmware does, by `judged`. An envelope it refuses fails the transfer, as
 * an `envelopeVerdict` does: its count stays at the 32 bytes, its checksum
 * covers every byte of every write it took, and it throws away every write
 * that follows.
 */
class Device {
  held: number[] = [];
  chunkMax = 244;
  /** Bytes this device already holds when the transfer starts. */
  already: Uint8Array = new Uint8Array(0);
  /** Answer one query with this instead of the truth. */
  lieOffset: number | null = null;
  corruptAt: number | null = null;
  verdict: { status: number; result: number } = { status: 0, result: 0 }; // verified
  /** Falls behind at every this many data writes, and throws away every write until it is next asked what it has, as the library's busy device does. */
  behindEvery = 0;
  /** How many checks this device turns away before it answers one. */
  turnAway = 0;
  /**
   * The verdict this device gives an image's envelope, its first 32 bytes,
   * when it refuses the image there, as a device does with an image built
   * for its other slot, or null when it takes the envelope.
   */
  envelopeVerdict: number | null = null;
  /** Whether it judges an image's envelope by what it holds, as the firmware does. */
  judgesEnvelopes = false;
  /** What the transfer's START named: its target, and its length. */
  #target = "";
  #total = 0;
  /** Whether it is writing a page, and the writes that arrived meanwhile. */
  busy = false;
  queue: Uint8Array[] = [];
  /** Data writes that arrived, checks asked, writes thrown away, and the most held at once. */
  writes = 0;
  checks = 0;
  thrown = 0;
  mostHeld = 0;
  /** Each pause this side took before asking again, in milliseconds. */
  waited: number[] = [];
  #behind = false;
  /** The verdict this device holds once it has stopped taking the transfer, or null while it takes it. */
  #failed: number | null = null;
  /** Every byte its checksum covers once it has stopped: the whole of each write it took, though it counted no further than the envelope. */
  #summed: number[] | null = null;
  /** Whether a finish has verified the transfer. The device holds that verdict, and takes nothing more of the transfer. */
  #verified = false;

  /** The answer to one request, or null when the device threw the request away. */
  control(bytes: Uint8Array): Uint8Array | null {
    if (this.busy) {
      // A request is one more incoming write. Held, it is answered once
      // what came before it is written, while this side waits.
      if (this.queue.length >= HOLDS) {
        this.thrown += 1;
        return null;
      }
      this.catchUp();
    }
    // The device's own decoder, so this refuses what a device refuses.
    const request = P.decodeUpdateRequest(bytes);
    switch (request.opName) {
      case "start": {
        assert.ok(request.totalLen! > 0);
        this.#target = request.targetName!;
        this.#total = request.totalLen!;
        this.held = Array.from(this.already);
        this.#failed = null;
        this.#summed = null;
        this.#verified = false;
        const body = new Uint8Array(7);
        const v = new DataView(body.buffer);
        body[0] = request.slot!;
        v.setUint16(1, this.chunkMax, true);
        v.setUint32(3, this.held.length, true);
        return P.encodeUpdateResponse(P.UPDATE_OPS.start, 0, body);
      }
      case "query": {
        this.checks += 1;
        this.#behind = false;
        if (this.turnAway > 0) {
          this.turnAway -= 1;
          return null;
        }
        const offset = this.lieOffset ?? this.held.length;
        this.lieOffset = null;
        const body = new Uint8Array(9);
        const v = new DataView(body.buffer);
        body[0] = this.#failed !== null ? 3 : this.#verified ? 2 : 1; // failed, complete, or receiving
        v.setUint32(1, offset, true);
        v.setUint32(5, crc32(Uint8Array.from(this.#summed ?? this.held)), true);
        return P.encodeUpdateResponse(P.UPDATE_OPS.query, 0, body);
      }
      case "finish": {
        // A transfer that failed, or one that verified, has nothing left to
        // finish: the device answers "no transfer", with the verdict it holds.
        const held = this.#failed ?? (this.#verified ? 0 : null);
        if (held !== null) return P.encodeUpdateResponse(P.UPDATE_OPS.finish, 4, Uint8Array.of(held));
        this.#verified = this.verdict.status === 0;
        return P.encodeUpdateResponse(
          P.UPDATE_OPS.finish,
          this.verdict.status,
          new Uint8Array([this.verdict.result]),
        );
      }
      case "abort":
        this.held = [];
        this.#failed = null;
        this.#summed = null;
        this.#verified = false;
        return P.encodeUpdateResponse(P.UPDATE_OPS.abort, 0);
      default:
        return P.encodeUpdateResponse(P.UPDATE_OPS[request.opName!], 0);
    }
  }

  /** One write without a response. */
  data(bytes: Uint8Array): void {
    this.writes += 1;
    if (this.behindEvery > 0 && this.writes % this.behindEvery === 0) this.#behind = true;
    if (this.#behind || (this.busy && this.queue.length >= HOLDS)) {
      this.thrown += 1;
      return;
    }
    if (this.busy) {
      this.queue.push(bytes.slice());
      this.mostHeld = Math.max(this.mostHeld, this.queue.length);
      return;
    }
    this.take(bytes);
  }

  /** Time passes: the page is written, and then what arrived meanwhile. */
  catchUp(): void {
    const held = this.queue;
    this.queue = [];
    for (const bytes of held) this.take(bytes);
    this.busy = false;
  }

  take(bytes: Uint8Array): void {
    // A device that stopped taking the transfer refuses every write after.
    if (this.#failed !== null || this.#verified) return;
    const out = Array.from(bytes);
    if (this.corruptAt !== null) {
      const i = this.corruptAt - this.held.length;
      if (i >= 0 && i < out.length) {
        out[i] = out[i]! ^ 0x01;
        this.corruptAt = null;
      }
    }
    if (
      (this.envelopeVerdict !== null || this.judgesEnvelopes) &&
      this.#target !== "head" &&
      this.held.length < P.ENVELOPE_LEN &&
      this.held.length + out.length >= P.ENVELOPE_LEN
    ) {
      // The envelope is the 32 bytes held, from whichever writes brought
      // them. One the device refuses stops it there: the write that
      // completes the envelope is checksummed whole, and its envelope bytes
      // are counted.
      const verdict = this.envelopeVerdict ?? judged(Uint8Array.from([...this.held, ...out].slice(0, P.ENVELOPE_LEN)), this.#target, this.#total);
      if (verdict !== null) {
        this.#summed = [...this.held, ...out];
        this.held.push(...out.slice(0, P.ENVELOPE_LEN - this.held.length));
        this.#failed = verdict;
        return;
      }
    }
    const before = this.held.length;
    this.held.push(...out);
    if (Math.floor(before / PAGE) !== Math.floor(this.held.length / PAGE)) this.busy = true;
  }
}

/** A session that has read a device claiming everything this needs. */
function session(capabilities = 0xffff): Session {
  const info = P.encodeDeviceInfo({
    protocol: [1, 0],
    firmware: [1, 0, 0],
    hardware: [1, 1, 5],
    channels: 4,
    adcBits: 24,
    tickHz: 1_000_000,
    vrefUv: 4_500_000,
    capabilities,
    supportedRates: 0b111,
    deviceId: "0102030405060708",
    extensionPresent: true,
    firmwareBuildId: "0000000000000000",
    modelEmbedDim: 96,
    modelNativeSps: 500,
    modelWindowSamples: 2000,
    headSlots: 4,
    headMaxOutputs: 32,
    headSlotBytes: 4096,
    updateChunkMax: 244,
    appSlotBytes: 237_568,
    weightsImageBytes: 507_904,
  } as never);
  const s = new Session();
  s.onDeviceInfo(info);
  return s;
}

/**
 * Drive a transfer to its end, returning the verdict and the write count.
 * A request the device threw away is never answered: the device catches up
 * while this side waits, and the transfer says whether to ask again, and
 * after how long.
 */
function run(t: Transfer, image: Uint8Array, dev: Device): { result: string | null; writes: number } {
  let writes = 0;
  for (let i = 0; i < 100_000; i += 1) {
    const step: Step = t.step(image);
    if (step.type === "verified") return { result: step.result, writes };
    if (step.type === "send") {
      assert.equal(step.command.withResponse, true, "a control write is answered");
      assert.equal(step.command.characteristic, P.FILL.UPDATE_CONTROL);
      const answer = dev.control(step.command.bytes);
      if (answer === null) {
        dev.catchUp();
        const pause = t.unanswered();
        if (pause === null) throw new Error("the device did not answer the update service in time");
        dev.waited.push(pause);
        continue;
      }
      t.onAnswer(answer, image);
      continue;
    }
    assert.equal(step.characteristic, P.FILL.UPDATE_DATA);
    assert.ok(step.to > step.from, "a write carries bytes");
    assert.ok(step.to - step.from <= 244, "no write is larger than the contract allows");
    dev.data(image.subarray(step.from, step.to));
    writes += 1;
    t.sent(step.to - step.from);
  }
  throw new Error("the transfer did not finish");
}

function image(n: number): Uint8Array {
  return Uint8Array.from({ length: n }, (_, i) => (i * 7 + 13) & 0xff);
}

/** What a stand-in link does with a write longer than it carries. */
type Way = "refuses" | "drops" | "cuts";

const WAYS: readonly Way[] = ["refuses", "drops", "cuts"];

/**
 * A link of `size`, as a system reports its size, which carries a write of
 * at most its size less 3 and treats a longer one its `way`: it refuses it,
 * as Linux does and Windows is reported to; drops it with no sign, the
 * write still succeeding, as iOS is reported to and as a Mac's browser
 * resolves a write whatever happens; or cuts it to its size less 3 with no
 * sign, as Android documents.
 */
interface StandIn {
  readonly size: number;
  readonly way: Way;
}

/** The system's refusal of a write, as a driver hears it. */
class WriteRefused extends Error {}

/**
 * Drive a transfer to its end through a stand-in link to `dev`, as
 * `Connection.transfer` drives one: a data write the link refuses is told
 * to the transfer, and is the end of it where the transfer says it does not
 * start again. Gives the verdict and every request and data write in order,
 * each request by its operation and each write by the bytes sent, which go
 * into `trace` as they are made, so a transfer that throws leaves them there.
 */
function over(t: Transfer, img: Uint8Array, dev: Device, link: StandIn, trace: (string | number)[] = []): { result: string | null; trace: (string | number)[] } {
  const carries = link.size - 3;
  for (let i = 0; i < 100_000; i += 1) {
    const step = t.step(img);
    if (step.type === "verified") return { result: step.result, trace };
    if (step.type === "send") {
      trace.push(P.decodeUpdateRequest(step.command.bytes).opName!);
      const answer = dev.control(step.command.bytes);
      if (answer === null) {
        dev.catchUp();
        const pause = t.unanswered();
        if (pause === null) throw new Error("the device did not answer the update service in time");
        dev.waited.push(pause);
        continue;
      }
      t.onAnswer(answer, img);
      continue;
    }
    const bytes = img.subarray(step.from, step.to);
    trace.push(bytes.length);
    if (bytes.length <= carries) dev.data(bytes);
    else if (link.way === "refuses") {
      if (t.writeRefused()) continue;
      throw new WriteRefused("the system refused the write");
    } else if (link.way === "cuts") dev.data(bytes.subarray(0, carries));
    t.sent(bytes.length);
  }
  throw new Error("the transfer did not finish");
}

/**
 * The requests and writes a transfer of `total` bytes made before a link's
 * size was asked about, to a device that threw nothing away: START, writes
 * of 244 bytes with a progress check after every eighth and after the last,
 * and FINISH.
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
      out.push("query");
      since = 0;
    }
  }
  out.push("finish");
  return out;
}

/** The writes of `trace` after its last ABORT, by their sizes. */
function writesAfterTheLastAbort(trace: readonly (string | number)[]): number[] {
  return trace.slice(trace.lastIndexOf("abort") + 1).filter((x): x is number => typeof x === "number");
}

/** The largest data write after each START in `trace`, in order. */
function largestAfterEachStart(trace: readonly (string | number)[]): number[] {
  const out: number[] = [];
  for (const x of trace) {
    if (x === "start") out.push(0);
    else if (typeof x === "number") out[out.length - 1] = Math.max(out.at(-1)!, x);
  }
  return out;
}

/** `n` data writes of `size` bytes, as a trace lists them. */
const writesOf = (n: number, size: number): number[] => Array.from({ length: n }, () => size);

test("a link that carries 244 bytes sees exactly the requests and writes a transfer made before a link's size was asked about, and no wait, whatever it does with a longer write", () => {
  // So for a device that judges an image's envelope, too.
  for (const [img, judges] of [[image(30_000), false], [enveloped(30_000), true]] as const) {
    for (const way of WAYS) {
      const dev = new Device();
      dev.judgesEnvelopes = judges;
      const t = Transfer.app(session(), 0, img);
      const { result, trace } = over(t, img, dev, { size: 247, way });
      assert.equal(result, "verified", way);
      assert.deepEqual(trace, today(img.length), way);
      assert.deepEqual(dev.waited, [], way);
      assert.equal(dev.thrown, 0, way);
      assert.deepEqual(Uint8Array.from(dev.held), img, way);
    }
  }
});

test("on a link of 104 that refuses, drops or cuts a longer write, a transfer starts the file again at 101 bytes, from the first byte, and is verified", () => {
  const img = image(30_000);
  for (const way of WAYS) {
    const dev = new Device();
    const t = Transfer.app(session(), 0, img);
    const { result, trace } = over(t, img, dev, { size: 104, way });
    assert.equal(result, "verified", way);
    assert.deepEqual(Uint8Array.from(dev.held), img, `${way}: the device holds exactly what was sent`);
    assert.equal(trace[0], "start", way);
    assert.equal(trace[1], 244, `${way}: the first write is 244 bytes`);
    assert.equal(trace.filter((x) => x === "abort").length, 1, `${way}: the file started again once`);
    assert.equal(trace[trace.lastIndexOf("abort") + 1], "start", way);
    const after = writesAfterTheLastAbort(trace);
    assert.ok(after.every((n) => n <= 101) && after[0] === 101, `${way}: then every write is 101 bytes or fewer`);
    assert.equal(after.reduce((x, y) => x + y, 0), img.length, `${way}: from the first byte, each byte once`);
    assert.equal(t.largestWrite, 101, way);
  }
});

test("on a link of 23 that refuses, drops or cuts a longer write, a transfer starts the file again at 101 bytes and then at 20, and is verified", () => {
  const img = image(5000);
  for (const way of WAYS) {
    const dev = new Device();
    const t = Transfer.app(session(), 0, img);
    const { result, trace } = over(t, img, dev, { size: 23, way });
    assert.equal(result, "verified", way);
    assert.deepEqual(Uint8Array.from(dev.held), img, way);
    assert.equal(trace.filter((x) => x === "abort").length, 2, `${way}: the file started again twice`);
    const between = trace.slice(trace.indexOf("abort") + 1, trace.lastIndexOf("abort")).filter((x): x is number => typeof x === "number");
    assert.ok(between[0] === 101 && between.every((n) => n <= 101), `${way}: at 101 bytes in between`);
    const after = writesAfterTheLastAbort(trace);
    assert.ok(after.every((n) => n <= 20), way);
    assert.equal(after.reduce((x, y) => x + y, 0), img.length, way);
    assert.equal(t.largestWrite, 20, way);
  }
});

test("where 20 bytes fail too, a transfer ends in the error and the words it ends in on any link", () => {
  // A link that carries 17 bytes, less than every size a transfer starts
  // again at.
  const img = image(5000);
  const refused = Transfer.app(session(), 0, img);
  assert.throws(() => over(refused, img, new Device(), { size: 20, way: "refuses" }), WriteRefused);
  assert.equal(refused.largestWrite, 20);
  assert.equal(refused.writeRefused(), false, "no smaller size remains");
  assert.throws(
    () => over(Transfer.app(session(), 0, img), img, new Device(), { size: 20, way: "drops" }),
    (e: unknown) => {
      assert.ok(e instanceof OffsetMismatch, String(e));
      assert.equal(e.message, `the device accepted ${e.device} bytes where ${e.sent} were sent, and the transfer is not advancing`);
      return true;
    },
  );
  assert.throws(
    () => over(Transfer.app(session(), 0, img), img, new Device(), { size: 20, way: "cuts" }),
    (e: unknown) => {
      assert.ok(e instanceof ChecksumMismatch, String(e));
      assert.match(e.message, /^the device's checksum over \d+ bytes is 0x[0-9a-f]{8} and ours is 0x[0-9a-f]{8}$/);
      return true;
    },
  );
});

test("a transfer held to the largest write a system reports writes no more than that, and starts again at the smaller of 101 and 20 below it", () => {
  const img = image(5000);
  // A link of 185 whose system says so carries every write.
  let t = Transfer.app(session(), 0, img);
  t.limitWrites(182);
  let got = over(t, img, new Device(), { size: 185, way: "drops" });
  assert.equal(got.result, "verified");
  assert.ok(!got.trace.includes("abort") && got.trace[1] === 182 && got.trace.every((x) => typeof x === "string" || x <= 182), "182 bytes at a time, and nothing started again");
  // A size held to that is still too large starts again at 101.
  t = Transfer.app(session(), 0, img);
  t.limitWrites(150);
  got = over(t, img, new Device(), { size: 104, way: "refuses" });
  assert.equal(got.result, "verified");
  assert.deepEqual([got.trace[1], got.trace.filter((x) => x === "abort").length, t.largestWrite], [150, 1, 101]);
  // The size the last transfer finished at, given to the next, starts it
  // there, with only 20 below it.
  t = Transfer.app(session(), 0, img);
  t.limitWrites(101);
  assert.equal(t.largestWrite, 101);
  got = over(t, img, new Device(), { size: 23, way: "cuts" });
  assert.equal(got.result, "verified");
  assert.deepEqual([got.trace[1], got.trace.filter((x) => x === "abort").length, t.largestWrite], [101, 1, 20]);
  // Never more than the device's own limit or this contract's.
  t = Transfer.app(session(), 0, img);
  t.limitWrites(512);
  assert.equal(t.largestWrite, 244);
});

test("a transfer that starts the file again carries the data from the count the device's answer to START gives, its progress that count, and passes over an ABORT turned away, unanswered or refused", () => {
  const img = image(5000);
  const dev = new Device();
  const t = Transfer.app(session(), 0, img);
  for (let step = t.step(img); step.type !== "data"; step = t.step(img)) {
    t.onAnswer(dev.control((step as { command: { bytes: Uint8Array } }).command.bytes)!, img);
  }
  dev.data(img.subarray(0, 244));
  t.sent(244);
  assert.deepEqual(t.progress, { taken: 244, total: 5000 });
  assert.equal(t.writeRefused(), true, "the second write is refused");
  assert.deepEqual(t.progress, { taken: 244, total: 5000 }, "nothing moves until the device answers START");
  assert.equal(t.smallerBecause, null, "a refused write is the driver's own error");
  const abort = t.step(img);
  assert.ok(abort.type === "send" && abort.command.bytes[0] === P.UPDATE_OPS.abort, "ABORT first");
  assert.equal(t.unanswered(), 0, "an ABORT turned away or unanswered is passed over at once");
  const start = t.step(img);
  assert.ok(start.type === "send" && start.command.bytes[0] === P.UPDATE_OPS.start, "then START");
  // The ABORT never reached the device, which still holds the first write
  // and says so when asked to start again.
  dev.already = Uint8Array.from(dev.held);
  t.onAnswer(dev.control(start.command.bytes)!, img);
  assert.deepEqual(t.progress, { taken: 244, total: 5000 }, "the count the device's answer to START gives");
  const next = t.step(img);
  assert.ok(next.type === "data" && next.from === 244 && next.to === 345, "the data goes on from there, 101 bytes at a time");
  assert.equal(run(t, img, dev).result, "verified");
  assert.deepEqual(Uint8Array.from(dev.held), img, "each byte once, and none past the file's length");

  // An ABORT the device refuses, as one with no transfer refuses it.
  const u = Transfer.app(session(), 0, img);
  for (let step = u.step(img); step.type !== "data"; step = u.step(img)) {
    u.onAnswer(new Device().control((step as { command: { bytes: Uint8Array } }).command.bytes)!, img);
  }
  assert.equal(u.writeRefused(), true);
  u.onAnswer(P.encodeUpdateResponse(P.UPDATE_OPS.abort, 4), img);
  const again = u.step(img);
  assert.ok(again.type === "send" && again.command.bytes[0] === P.UPDATE_OPS.start, "START after a refused ABORT");
  u.onAnswer(new Device().control(again.command.bytes)!, img);
  assert.deepEqual(u.progress, { taken: 0, total: 5000 }, "a device that handled ABORT starts from the first byte");
  assert.equal(u.largestWrite, 101);
});

test("while the ABORT that starts the file again at a smaller size is under way, the transfer gives what a progress check found, in the words it ends in where no smaller size remains", () => {
  // Every write of 244 bytes is lost on a link of 104, so the device's count
  // stops advancing at the second check.
  const img = image(5000);
  const dev = new Device();
  const t = Transfer.app(session(), 0, img);
  let why: TransferError | null = null;
  for (let i = 0; i < 1000 && why === null; i++) {
    const step = t.step(img);
    if (step.type === "send") {
      t.onAnswer(dev.control(step.command.bytes)!, img);
      why = t.smallerBecause;
    } else if (step.type === "data") t.sent(step.to - step.from);
  }
  assert.ok(why instanceof OffsetMismatch, String(why));
  assert.equal(why.message, `the device accepted 0 bytes where ${why.sent} were sent, and the transfer is not advancing`);
  assert.equal(t.largestWrite, 101);
  const abort = t.step(img);
  assert.ok(abort.type === "send" && abort.command.bytes[0] === P.UPDATE_OPS.abort);
  t.onAnswer(dev.control(abort.command.bytes)!, img);
  assert.equal(t.smallerBecause, null, "and nothing once ABORT is done");
});

test("a link of 23 that cuts writes ends in a verified transfer at 20 bytes, where the envelope the device judges at its first 32 bytes is made of two writes", () => {
  // The device gets the first 20 bytes of each write, so the envelope it
  // judges is made of two writes, and it fails the transfer there. Its
  // checksum is then not ours over the first write, so it did not get the
  // bytes that were sent: the file starts again at the next smaller size
  // with no FINISH, at 101 and then at 20, where the image is verified.
  const img = enveloped(24 * 256);
  const dev = new Device();
  dev.judgesEnvelopes = true;
  const t = Transfer.app(session(), 0, img);
  const { result, trace } = over(t, img, dev, { size: 23, way: "cuts" });
  assert.equal(result, "verified");
  assert.deepEqual(Uint8Array.from(dev.held), img);
  assert.deepEqual(largestAfterEachStart(trace), [244, 101, 20]);
  const failed = ["start", ...writesOf(8, 244), "query", "abort", "start", ...writesOf(8, 101), "query", "abort", "start"];
  assert.deepEqual(trace.slice(0, failed.length), failed);
  assert.equal(trace.indexOf("finish"), trace.length - 1, "FINISH only at the end");
});

test("a link of 104 that cuts writes starts the file again at 101 bytes at its first progress check, for an image and for a head", () => {
  // The device gets the first 101 bytes of each write. At the first check
  // its count is behind what was sent and its checksum is not ours over as
  // many of the file's first bytes, so it holds bytes that are not the
  // start of the file: the file starts again there, rather than carrying on
  // from the device's count. A head has no envelope.
  const blob = P.buildHead(Array.from({ length: 32 }, (_, o) => Array.from({ length: 96 }, (_, i) => ((o + i) % 7) - 3)), Array.from({ length: 32 }, () => 0), Array.from({ length: 32 }, () => 1), { name: "focus" });
  assert.ok(blob.length > 8 * 244, "the head is longer than eight writes");
  const img = enveloped(24 * 256);
  for (const [what, file, make] of [
    ["image", img, () => Transfer.app(session(), 0, img)],
    ["head", blob, () => Transfer.head(session(), 1, blob)],
  ] as const) {
    const dev = new Device();
    dev.judgesEnvelopes = true;
    const t = make();
    const { result, trace } = over(t, file, dev, { size: 104, way: "cuts" });
    assert.equal(result, "verified", what);
    assert.deepEqual(Uint8Array.from(dev.held), file, what);
    const first = ["start", ...writesOf(8, 244), "query", "abort", "start"];
    assert.deepEqual(trace.slice(0, first.length), first, what);
    assert.deepEqual(largestAfterEachStart(trace), [244, 101], what);
  }
});

test("an image the device refuses at its first 32 bytes on a link that carries 244 ends in its verdict, through FINISH, with no smaller size tried, and the next transfer starts at 244", () => {
  // Its checksum is ours over the first write: the device got the bytes
  // that were sent, and refused the image.
  const good = enveloped(24 * 256);
  const short = good.slice();
  new DataView(short.buffer).setUint32(24, 7, true);
  for (const [bad, words] of [
    [enveloped(24 * 256, 9), "the device refused the image: key id"],
    [short, "the device refused the image: length"],
  ] as const) {
    const dev = new Device();
    dev.judgesEnvelopes = true;
    const t = Transfer.app(session(), 0, bad);
    const trace: (string | number)[] = [];
    assert.throws(() => over(t, bad, dev, { size: 247, way: "refuses" }, trace), (e: unknown) => {
      assert.ok(e instanceof Rejected, String(e));
      assert.equal(e.message, words);
      return true;
    });
    assert.deepEqual(trace, ["start", ...writesOf(8, 244), "query", "finish"], words);
    assert.equal(t.largestWrite, 244, words);
    const next = Transfer.app(session(), 0, good);
    next.limitWrites(t.largestWrite);
    const then = over(next, good, dev, { size: 247, way: "refuses" });
    assert.equal(then.result, "verified", words);
    assert.deepEqual(largestAfterEachStart(then.trace), [244], words);
  }
});

test("at 20 bytes a device holding bytes that are not the start of the file ends the transfer at that check, in the words for a checksum that does not match, over the device's count", () => {
  // A link of 20 cuts each write to 17 bytes.
  const img = Uint8Array.from({ length: 12 * 256 }, (_, i) => i & 0xff);
  const trace: (string | number)[] = [];
  const held = Uint8Array.from(Array.from({ length: 8 }, (_, k) => Array.from(img.subarray(20 * k, 20 * k + 17))).flat());
  const theirs = crc32(held) >>> 0;
  const ours = crc32(img.subarray(0, held.length)) >>> 0;
  const hex = (n: number): string => `0x${n.toString(16).padStart(8, "0")}`;
  assert.throws(() => over(Transfer.app(session(), 0, img), img, new Device(), { size: 20, way: "cuts" }, trace), (e: unknown) => {
    assert.ok(e instanceof ChecksumMismatch, String(e));
    assert.equal(e.message, `the device's checksum over ${held.length} bytes is ${hex(theirs)} and ours is ${hex(ours)}`);
    return true;
  });
  assert.deepEqual(trace.slice(trace.lastIndexOf("start")), ["start", ...writesOf(8, 20), "query"]);
});

test("an image arrives byte for byte", () => {
  const img = image(5000);
  const dev = new Device();
  const t = Transfer.app(session(), 0, img);
  const { result, writes } = run(t, img, dev);
  assert.equal(result, "verified");
  assert.deepEqual(Uint8Array.from(dev.held), img, "the device holds exactly what was sent");
  assert.equal(writes, Math.ceil(5000 / 244));
  assert.deepEqual(t.progress, { taken: 5000, total: 5000 });
});

test("a device that already has part of it is not sent that part again", () => {
  const img = image(5000);
  const dev = new Device();
  dev.already = img.subarray(0, 2000);
  const t = Transfer.app(session(), 0, img);
  const { writes } = run(t, img, dev);
  assert.deepEqual(Uint8Array.from(dev.held), img);
  assert.equal(writes, Math.ceil(3000 / 244), "only what was missing went out");
});

test("a device claiming bytes it does not hold is caught at the first check", () => {
  const img = image(5000);
  const dev = new Device();
  dev.already = new Uint8Array(2000);
  const t = Transfer.app(session(), 0, img);
  assert.throws(() => run(t, img, dev), ChecksumMismatch);
});

test("a device that says it holds fewer bytes than it does is caught by the checksum at that check, which is over the count it gives", () => {
  const img = image(20_000);
  const dev = new Device();
  dev.lieOffset = 0;
  const t = Transfer.app(session(), 0, img);
  // At the smallest size, where no smaller one remains to start the file
  // again at.
  t.limitWrites(20);
  assert.throws(
    () => run(t, img, dev),
    (e: unknown) => {
      assert.ok(e instanceof ChecksumMismatch);
      // Our checksum over nothing is 0, and the device's is over the 160
      // bytes it holds.
      assert.deepEqual([e.sent, e.ours >>> 0, e.device >>> 0], [0, 0, crc32(img.subarray(0, 160)) >>> 0]);
      return true;
    },
  );
  assert.equal(dev.checks, 1, "it stopped at the check that said so, not later");
});

test("a device busy writing its flash takes a whole image or weights, and throws nothing away", () => {
  const img = image(30_000);
  for (const t of [Transfer.app(session(), 0, img), Transfer.weights(session(), img)]) {
    const dev = new Device();
    const { result, writes } = run(t, img, dev);
    assert.equal(result, "verified");
    assert.deepEqual(Uint8Array.from(dev.held), img, "the device holds exactly what was sent");
    assert.equal(writes, Math.ceil(30_000 / 244), "every write went once");
    assert.ok(dev.mostHeld > 0, "writes arrived while the device wrote a page");
    assert.equal(dev.thrown, 0, "the device threw nothing away");
    assert.deepEqual(dev.waited, [], "every check was answered");
  }
});

test("a transfer carries on from the device's own count past writes it threw away, and asks a check it turned away again", () => {
  // The Python library's busy device: it falls behind at every thirteenth
  // write and throws away what arrives until it is asked where it is, and
  // it turns the first check away.
  const img = Uint8Array.from({ length: 5120 }, (_, i) => i & 0xff);
  const dev = new Device();
  dev.chunkMax = 100;
  dev.behindEvery = 13;
  dev.turnAway = 1;
  const { result, writes } = run(Transfer.app(session(), 0, img), img, dev);
  assert.equal(result, "verified");
  assert.deepEqual(Uint8Array.from(dev.held), img, "every byte arrived once, in order");
  assert.ok(dev.thrown > 0, "the device threw writes away");
  assert.equal(writes, Math.ceil(5120 / 100) + dev.thrown, "what was thrown away went again, and nothing else");
  assert.ok(dev.checks > 5120 / (100 * 8), "progress is asked in short bursts");
});

test("a check the device does not answer is asked again after half a second and then a second, and a third unanswered stops the transfer", () => {
  const img = image(5000);
  const dev = new Device();
  dev.turnAway = 2;
  assert.equal(run(Transfer.app(session(), 0, img), img, dev).result, "verified");
  assert.deepEqual(Uint8Array.from(dev.held), img);
  assert.deepEqual(dev.waited, [500, 1000]);

  const deaf = new Device();
  deaf.turnAway = 3;
  assert.throws(() => run(Transfer.app(session(), 0, img), img, deaf), /did not answer the update service/);
  assert.deepEqual(deaf.waited, [500, 1000], "three times in all");

  const t = Transfer.app(session(), 0, img);
  assert.equal(t.unanswered(), null, "a start is not asked again");
});

test("a transfer whose count stops advancing is given up, in the library's words", () => {
  const dev = new Device();
  dev.behindEvery = 1;
  const img = new Uint8Array(5000);
  assert.throws(
    () => run(Transfer.app(session(), 0, img), img, dev),
    (e: unknown) => {
      assert.ok(e instanceof OffsetMismatch);
      assert.equal(e.device, 0);
      assert.ok(e.sent > 0 && e.sent < img.length, "it stopped where the count stopped, not at the end");
      assert.equal(e.message, `the device accepted 0 bytes where ${e.sent} were sent, and the transfer is not advancing`);
      return true;
    },
  );
});

test("bytes that arrived changed are caught by the checksum", () => {
  const img = image(20_000);
  const dev = new Device();
  dev.corruptAt = 100;
  const t = Transfer.app(session(), 0, img);
  // At the smallest size, where no smaller one remains to start the file
  // again at.
  t.limitWrites(20);
  assert.throws(() => run(t, img, dev), ChecksumMismatch);
});

test("an image the device refuses says why", () => {
  const img = image(500);
  const dev = new Device();
  dev.verdict = { status: 6, result: 6 }; // verification failed, bad signature
  const t = Transfer.app(session(), 0, img);
  assert.throws(
    () => run(t, img, dev),
    (e: unknown) => {
      assert.ok(e instanceof Rejected);
      assert.equal(e.result, "signature");
      return true;
    },
  );
});

test("an image the device refuses at its first bytes is finished at the first check, and the device's verdict says why", () => {
  // As a device does with an image built for its other slot: it stops at
  // the envelope's 32 bytes and holds its verdict, and a progress check
  // says the transfer failed.
  for (const n of [500, 20_000]) {
    const img = image(n);
    const dev = new Device();
    dev.envelopeVerdict = 4; // slot
    assert.throws(
      () => run(Transfer.app(session(), 0, img), img, dev),
      (e: unknown) => {
        assert.ok(e instanceof Rejected, `${n} bytes: ${String(e)}`);
        assert.equal(e.result, "slot");
        assert.equal(e.message, "the device refused the image: slot");
        return true;
      },
    );
    assert.equal(dev.checks, 1, `${n} bytes: the device said so at the first check`);
    assert.equal(dev.writes, Math.min(8, Math.ceil(n / 244)), `${n} bytes: nothing was written after it said so`);
  }
});

test("a finish the device answers ok after a progress check said the transfer failed ends in the refusal a refused finish gives, and never in a verified transfer", () => {
  // No firmware answers so. The device stops at the envelope, a progress
  // check says the transfer failed, and the answer to the finish that
  // follows says ok.
  const img = image(500);
  const failedAtFinish = (): Transfer => {
    const dev = new Device();
    dev.envelopeVerdict = 4; // The verdict is "slot".
    const t = Transfer.app(session(), 0, img);
    for (;;) {
      const step = t.step(img);
      if (step.type === "send" && step.command.bytes[0] === P.UPDATE_OPS.finish) return t;
      if (step.type === "send") t.onAnswer(dev.control(step.command.bytes)!, img);
      else if (step.type === "data") {
        dev.data(img.subarray(step.from, step.to));
        t.sent(step.to - step.from);
      } else assert.fail(`the transfer said verified before a finish was sent: ${step.result}`);
    }
  };
  const ok = (verdict: Uint8Array): Uint8Array => P.encodeUpdateResponse(P.UPDATE_OPS.finish, 0, verdict);
  assert.throws(
    () => failedAtFinish().onAnswer(ok(Uint8Array.of(4)), img),
    (e: unknown) => {
      assert.ok(e instanceof Rejected, String(e));
      assert.equal(e.result, "slot");
      assert.equal(e.message, "the device refused the image: slot");
      return true;
    },
    "a verdict that is not verified is the device's reason",
  );
  assert.throws(
    () => failedAtFinish().onAnswer(ok(Uint8Array.of(0)), img),
    (e: unknown) => {
      assert.ok(e instanceof Refused, String(e));
      assert.equal(e.op, "finish");
      assert.equal(e.status, "ok");
      assert.equal(e.message, "the device refused the image: ok");
      return true;
    },
    "a verdict that says verified is a refused finish with the device's status",
  );
  assert.throws(
    () => failedAtFinish().onAnswer(ok(new Uint8Array(0)), img),
    (e: unknown) => {
      assert.ok(e instanceof Refused, String(e));
      assert.equal(e.op, "finish");
      assert.equal(e.status, "ok");
      assert.equal(e.message, "the device refused the image: ok");
      return true;
    },
    "no verdict is a refused finish with the device's status, as a refused finish without one is",
  );
});

test("a finish refused with no verdict is a refused finish with the device's status, and one with a verdict the protocol does not define is rejected with that number, as the Python library says each", () => {
  // The transfer is carried to its FINISH, and the device's answer to it
  // is the one each case gives.
  const img = image(500);
  const atFinish = (): Transfer => {
    const dev = new Device();
    const t = Transfer.app(session(), 0, img);
    for (;;) {
      const step = t.step(img);
      if (step.type === "send" && step.command.bytes[0] === P.UPDATE_OPS.finish) return t;
      if (step.type === "send") t.onAnswer(dev.control(step.command.bytes)!, img);
      else if (step.type === "data") {
        dev.data(img.subarray(step.from, step.to));
        t.sent(step.to - step.from);
      } else assert.fail(`the transfer said verified before a finish was sent: ${step.result}`);
    }
  };
  for (const [status, name] of [[5, "flash"], [6, "verification failed"]] as const) {
    assert.throws(
      () => atFinish().onAnswer(P.encodeUpdateResponse(P.UPDATE_OPS.finish, status, new Uint8Array(0)), img),
      (e: unknown) => {
        assert.ok(e instanceof Refused, `status ${status}: ${String(e)}`);
        assert.equal(e.op, "finish");
        assert.equal(e.status, name);
        assert.equal(e.message, `the device refused the image: ${name}`);
        return true;
      },
      `status ${status} with no verdict is a refused finish with that status`,
    );
    assert.throws(
      () => atFinish().onAnswer(P.encodeUpdateResponse(P.UPDATE_OPS.finish, status, Uint8Array.of(0xee)), img),
      (e: unknown) => {
        assert.ok(e instanceof Rejected, `status ${status}: ${String(e)}`);
        assert.equal(e.result, "result 238");
        assert.equal(e.message, "the device refused the image: result 238");
        return true;
      },
      `status ${status} with a verdict the protocol does not define is rejected with its number, as the library says it`,
    );
  }
});

test("a head the device refuses at the end of its transfer is called the head, as the library calls it", () => {
  const blob = P.buildHead(Array.from({ length: 2 }, () => Array.from({ length: 96 }, () => 1)), [0, 0], [1, 1], { name: "focus" });
  const dev = new Device();
  dev.verdict = { status: 6, result: 6 };
  assert.throws(
    () => run(Transfer.head(session(), 1, blob), blob, dev),
    (e: unknown) => {
      assert.ok(e instanceof Rejected);
      assert.equal(e.message, "the device refused the head: signature");
      return true;
    },
  );
  const img = image(500);
  const again = new Device();
  again.verdict = { status: 6, result: 6 };
  assert.throws(() => run(Transfer.app(session(), 0, img), img, again), { message: "the device refused the image: signature" });
});

test("a finish asked again after the transfer verified is a refused finish, and never a transfer rejected as verified", () => {
  // The device verifies the image, and its answer does not arrive. Asked
  // to finish again, it refuses, as there is no transfer left to finish,
  // and still carries the verdict it holds: verified.
  const img = image(500);
  const dev = new Device();
  const t = Transfer.app(session(), 0, img);
  let finish: Uint8Array | null = null;
  while (finish === null) {
    const step = t.step(img);
    if (step.type === "send" && step.command.bytes[0] === P.UPDATE_OPS.finish) finish = step.command.bytes;
    else if (step.type === "send") t.onAnswer(dev.control(step.command.bytes)!, img);
    else if (step.type === "data") {
      dev.data(img.subarray(step.from, step.to));
      t.sent(step.to - step.from);
    } else assert.fail(`verified before a finish was sent: ${step.result}`);
  }
  const lost = dev.control(finish)!;
  assert.deepEqual(lost, P.encodeUpdateResponse(P.UPDATE_OPS.finish, 0, Uint8Array.of(0)), "the device verified it");
  const again = t.step(img);
  assert.ok(again.type === "send", again.type);
  assert.deepEqual(again.command.bytes, finish, "the transfer asks to finish again");
  const answer = dev.control(again.command.bytes)!;
  assert.deepEqual(answer, P.encodeUpdateResponse(P.UPDATE_OPS.finish, 4, Uint8Array.of(0)), "no transfer, and the verdict held: verified");
  assert.throws(
    () => t.onAnswer(answer, img),
    (e: unknown) => {
      assert.ok(e instanceof Refused, String(e));
      assert.equal(e.op, "finish");
      assert.equal(e.status, "no transfer");
      assert.equal(e.message, "the device refused the image: no transfer");
      return true;
    },
  );
});

test("a refused request is said in the Python library's words for what was asked, for each operation and each status, and a refused finish names what the transfer carried", () => {
  const asked = [
    ["start", "to start the transfer"],
    ["query", "a progress check"],
    ["activate", "to activate the image"],
    ["abort", "to abort the transfer"],
  ] as const;
  const statuses = Object.values(P.UPDATE_STATUS);
  assert.deepEqual(statuses, ["ok", "invalid argument", "unsupported", "busy", "no transfer", "flash", "verification failed"]);
  for (const status of statuses) {
    for (const [op, what] of asked) {
      const e = new Refused(op, status);
      assert.equal(e.message, `the device refused ${what}: ${status}`);
      assert.deepEqual([e.op, e.status], [op, status], "the operation and the status are kept as they were");
    }
    // A refused finish is said as a rejected transfer is: the head for a
    // head, and the image for anything else.
    for (const [target, what] of [[undefined, "the image"], ["app", "the image"], ["weights", "the image"], ["head", "the head"]] as const) {
      const e = new Refused("finish", status, target);
      assert.equal(e.message, `the device refused ${what}: ${status}`, String(target));
      assert.deepEqual([e.op, e.status], ["finish", status]);
    }
  }
  // The transfer gives each refusal the words of what it asked.
  const img = image(500);
  assert.throws(() => Transfer.app(session(), 0, img).onAnswer(P.encodeUpdateResponse(P.UPDATE_OPS.start, 3, new Uint8Array(7)), img), {
    name: "Refused",
    message: "the device refused to start the transfer: busy",
  });
  const blob = P.buildHead(Array.from({ length: 2 }, () => Array.from({ length: 96 }, () => 1)), [0, 0], [1, 1], { name: "focus" });
  const t = Transfer.head(session(), 1, blob);
  const dev = new Device();
  for (;;) {
    const step = t.step(blob);
    if (step.type === "send" && step.command.bytes[0] === P.UPDATE_OPS.finish) break;
    if (step.type === "send") t.onAnswer(dev.control(step.command.bytes)!, blob);
    else if (step.type === "data") {
      dev.data(blob.subarray(step.from, step.to));
      t.sent(step.to - step.from);
    } else assert.fail(`the transfer said verified before a finish was sent: ${step.result}`);
  }
  assert.throws(() => t.onAnswer(P.encodeUpdateResponse(P.UPDATE_OPS.finish, 5, new Uint8Array(0)), blob), {
    name: "Refused",
    message: "the device refused the head: flash",
  });
});

test("a head is checked against this device before any of it is sent", () => {
  const s = session();
  const weights = Array.from({ length: 2 }, () => Array.from({ length: 96 }, () => 1));

  // One for an encoder of the wrong width never reaches the wire.
  const wrong = P.buildHead([[1, 1, 1, 1], [1, 1, 1, 1]], [0, 0], [1, 1], { name: "focus" });
  assert.throws(() => Transfer.head(s, 1, wrong), TransferError);

  // One of the right width goes.
  const blob = P.buildHead(weights, [0, 0], [1, 1], { name: "focus" });
  const dev = new Device();
  const t = Transfer.head(s, 1, blob);
  const { result } = run(t, blob, dev);
  assert.equal(result, "verified");
  assert.deepEqual(Uint8Array.from(dev.held), blob, "the head arrived as it was built");
});

test("a device that does not claim the capability is not asked", () => {
  const bare = session(0);
  assert.throws(() => Transfer.app(bare, 0, image(100)));
  assert.throws(() => Transfer.head(bare, 1, image(100)));
  assert.throws(() => bare.listHeads());
  assert.throws(() => bare.removeHead(1));
  // And the capable one is.
  assert.ok(Transfer.app(session(), 0, image(100)));
});

test("the smallest and largest images both work", () => {
  for (const n of [1, 243, 244, 245, 488]) {
    const img = image(n);
    const dev = new Device();
    const t = Transfer.app(session(), 0, img);
    const { result } = run(t, img, dev);
    assert.equal(result, "verified", `${n} bytes`);
    assert.deepEqual(Uint8Array.from(dev.held), img, `${n} bytes`);
  }
  assert.throws(() => Transfer.app(session(), 0, new Uint8Array(0)), TransferError);
});

test("a device that wants smaller writes gets them", () => {
  const img = image(2000);
  const dev = new Device();
  dev.chunkMax = 20;
  const t = Transfer.app(session(), 0, img);
  const { writes } = run(t, img, dev);
  assert.deepEqual(Uint8Array.from(dev.held), img);
  assert.equal(writes, 100, "the device's limit, not ours");
});

test("the transfer identifier is the hash of what is being sent", () => {
  // Both implementations must name the same bytes the same way, or an
  // interrupted transfer cannot be picked up again.
  const img = image(1000);
  const t = Transfer.app(session(), 0, img);
  const step = t.step(img);
  assert.equal(step.type, "send");
  if (step.type !== "send") return;
  const request = P.decodeUpdateRequest(step.command.bytes);
  assert.equal(request.transferId, P.toHex(sha256(img).subarray(0, 8)));
});

/** The start of a transfer, as the package writes it. */
function start(t: Transfer, img: Uint8Array): string {
  const step = t.step(img);
  assert.equal(step.type, "send");
  return step.type === "send" ? P.toHex(step.command.bytes) : "";
}

test("a weights start is the bytes the Python library writes", () => {
  // The contract has the start name slot zero for everything but a head.
  // These are the bytes the Python library 1.4.4 gives for this image:
  //   img = bytes(((i * 7 + 13) & 0xFF) for i in range(1000))
  //   encode_update_start("weights", 0, len(img), hashlib.sha256(img).digest()[:8]).hex()
  const img = image(1000);
  assert.equal(start(Transfer.weights(session(), img), img), "010200e8030000c85e29b0cb8af116");
});

test("an application's start names slot zero, and another slot is refused before anything is sent", () => {
  // encode_update_start("app", 0, len(img), id) for the same image.
  const img = image(1000);
  assert.equal(start(Transfer.app(session(), 0, img), img), "010100e8030000c85e29b0cb8af116");
  assert.throws(() => Transfer.app(session(), 1, img), TransferError);
});
