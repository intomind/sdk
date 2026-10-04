/**
 * A transfer, driven end to end against a device that answers the way
 * the contract says a device answers.
 *
 * The device here is a few dozen lines rather than a mock: it holds
 * bytes, counts them, checksums them, and answers. That is enough to
 * prove the state machine, and the same suite exists in Rust so the two
 * are held to one behavior.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { crc32, sha256 } from "../src/digest.ts";
import * as P from "../src/protocol.ts";
import { Session } from "../src/session.ts";
import {
  ChecksumMismatch,
  OffsetMismatch,
  Rejected,
  Transfer,
  TransferError,
  type Step,
} from "../src/transfer.ts";

/** A device that takes an image and says what it has. */
class Device {
  held: number[] = [];
  chunkMax = 244;
  /** Bytes this device already holds when the transfer starts. */
  already: Uint8Array = new Uint8Array(0);
  /** Answer one query with this instead of the truth. */
  lieOffset: number | null = null;
  corruptAt: number | null = null;
  verdict: { status: number; result: number } = { status: 0, result: 0 }; // verified

  control(bytes: Uint8Array): Uint8Array {
    // The device's own decoder, so this refuses what a device refuses.
    const request = P.decodeUpdateRequest(bytes);
    switch (request.opName) {
      case "start": {
        assert.ok(request.totalLen! > 0);
        this.held = Array.from(this.already);
        const body = new Uint8Array(7);
        const v = new DataView(body.buffer);
        body[0] = request.slot!;
        v.setUint16(1, this.chunkMax, true);
        v.setUint32(3, this.held.length, true);
        return P.encodeUpdateResponse(P.UPDATE_OPS.start, 0, body);
      }
      case "query": {
        const offset = this.lieOffset ?? this.held.length;
        this.lieOffset = null;
        const body = new Uint8Array(9);
        const v = new DataView(body.buffer);
        body[0] = 1; // receiving
        v.setUint32(1, offset, true);
        v.setUint32(5, crc32(Uint8Array.from(this.held)), true);
        return P.encodeUpdateResponse(P.UPDATE_OPS.query, 0, body);
      }
      case "finish":
        return P.encodeUpdateResponse(
          P.UPDATE_OPS.finish,
          this.verdict.status,
          new Uint8Array([this.verdict.result]),
        );
      case "abort":
        this.held = [];
        return P.encodeUpdateResponse(P.UPDATE_OPS.abort, 0);
      default:
        return P.encodeUpdateResponse(P.UPDATE_OPS[request.opName!], 0);
    }
  }

  data(bytes: Uint8Array): void {
    const out = Array.from(bytes);
    if (this.corruptAt !== null) {
      const i = this.corruptAt - this.held.length;
      if (i >= 0 && i < out.length) {
        out[i] = out[i]! ^ 0x01;
        this.corruptAt = null;
      }
    }
    this.held.push(...out);
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

/** Drive a transfer to its end, returning the verdict and the write count. */
function run(t: Transfer, image: Uint8Array, dev: Device): { result: string | null; writes: number } {
  let writes = 0;
  for (let i = 0; i < 100_000; i += 1) {
    const step: Step = t.step(image);
    if (step.type === "verified") return { result: step.result, writes };
    if (step.type === "send") {
      assert.equal(step.command.withResponse, true, "a control write is answered");
      assert.equal(step.command.characteristic, P.FILL.UPDATE_CONTROL);
      t.onAnswer(dev.control(step.command.bytes), image);
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

test("an image arrives byte for byte", () => {
  const img = image(5000);
  const dev = new Device();
  const t = Transfer.app(session(), 1, img);
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
  const t = Transfer.app(session(), 1, img);
  const { writes } = run(t, img, dev);
  assert.deepEqual(Uint8Array.from(dev.held), img);
  assert.equal(writes, Math.ceil(3000 / 244), "only what was missing went out");
});

test("a device claiming bytes it does not hold is caught at the first check", () => {
  const img = image(5000);
  const dev = new Device();
  dev.already = new Uint8Array(2000);
  const t = Transfer.app(session(), 1, img);
  assert.throws(() => run(t, img, dev), ChecksumMismatch);
});

test("a write the device did not get stops the transfer where it happened", () => {
  const img = image(20_000);
  const dev = new Device();
  dev.lieOffset = 0;
  const t = Transfer.app(session(), 1, img);
  assert.throws(
    () => run(t, img, dev),
    (e: unknown) => {
      assert.ok(e instanceof OffsetMismatch);
      assert.equal(e.device, 0);
      assert.ok(e.sent > 0 && e.sent < img.length, "it stopped at the first check, not at the end");
      return true;
    },
  );
});

test("bytes that arrived changed are caught by the checksum", () => {
  const img = image(20_000);
  const dev = new Device();
  dev.corruptAt = 100;
  const t = Transfer.app(session(), 1, img);
  assert.throws(() => run(t, img, dev), ChecksumMismatch);
});

test("an image the device refuses says why", () => {
  const img = image(500);
  const dev = new Device();
  dev.verdict = { status: 6, result: 6 }; // verification failed, bad signature
  const t = Transfer.app(session(), 1, img);
  assert.throws(
    () => run(t, img, dev),
    (e: unknown) => {
      assert.ok(e instanceof Rejected);
      assert.equal(e.result, "signature");
      return true;
    },
  );
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
  assert.throws(() => Transfer.app(bare, 1, image(100)));
  assert.throws(() => Transfer.head(bare, 1, image(100)));
  assert.throws(() => bare.listHeads());
  assert.throws(() => bare.removeHead(1));
  // And the capable one is.
  assert.ok(Transfer.app(session(), 1, image(100)));
});

test("the smallest and largest images both work", () => {
  for (const n of [1, 243, 244, 245, 488]) {
    const img = image(n);
    const dev = new Device();
    const t = Transfer.app(session(), 1, img);
    const { result } = run(t, img, dev);
    assert.equal(result, "verified", `${n} bytes`);
    assert.deepEqual(Uint8Array.from(dev.held), img, `${n} bytes`);
  }
  assert.throws(() => Transfer.app(session(), 1, new Uint8Array(0)), TransferError);
});

test("a device that wants smaller writes gets them", () => {
  const img = image(2000);
  const dev = new Device();
  dev.chunkMax = 20;
  const t = Transfer.app(session(), 1, img);
  const { writes } = run(t, img, dev);
  assert.deepEqual(Uint8Array.from(dev.held), img);
  assert.equal(writes, 100, "the device's limit, not ours");
});

test("the transfer identifier is the hash of what is being sent", () => {
  // Both implementations must name the same bytes the same way, or an
  // interrupted transfer cannot be picked up again.
  const img = image(1000);
  const t = Transfer.app(session(), 1, img);
  const step = t.step(img);
  assert.equal(step.type, "send");
  if (step.type !== "send") return;
  const request = P.decodeUpdateRequest(step.command.bytes);
  assert.equal(request.transferId, P.toHex(sha256(img).subarray(0, 8)));
});
