/**
 * Carrying an image or a head to a device, without doing any input or
 * output.
 *
 * The same three things go over this path: an application image, the
 * model weights, and a head. The first two are ours, signed and
 * encrypted, and the device checks the signature against the key in its
 * own flash. A head is the user's, carries its own hash, and is checked
 * for that hash and for its shape. Nothing here can forge any of them,
 * which is the point: a host carries an update without being able to
 * read or write one.
 *
 * Nothing is activated by a transfer. The image lands in the idle slot
 * and is verified there, and putting it in force is a separate act.
 *
 * This is a state machine, like the session. Ask it what to do, do that,
 * and hand back what came of it. `Connection.transfer` drives it for you
 * in a browser.
 *
 * A write is never larger than the device's own limit or this contract's,
 * 244 bytes, and a link may carry less: a host writes at most three bytes
 * less than the link's size at a time. Whoever drives a transfer tells it
 * the largest write the system reports for the link, where the system
 * reports one above 20, with `limitWrites`; a figure of 20 or less is the
 * size every link starts at, and says nothing. Where the system gives no
 * figure, as a browser gives none, the transfer starts at 244. A data write
 * the system refuses while the link is up is told with `writeRefused`. Every
 * progress check compares the device's checksum over the bytes it holds with
 * ours over as many of the file's first bytes, whatever its count, and a
 * count behind what was sent whose checksum matches is the start of the file
 * with writes lost on the way, which the transfer carries on from. When a
 * write is refused, or a progress check shows a checksum that does not
 * match, a count not moving, or a device that failed the transfer at its
 * first 32 bytes with a checksum that is not ours over the file up to the end
 * of the write that carried its 32nd byte, the transfer starts the file
 * again at the next smaller of 101 and 20, each held to the
 * device's limit: it aborts what the device holds, starts again, and sends
 * from the count the device's answer to START gives, which after an ABORT
 * the device handled is the first byte, and its progress is that count. A
 * link that goes down during that ABORT ends the transfer in the error that
 * stopped the larger size, which `smallerBecause` gives where a progress
 * check found it. When 20 fails too, it ends in the error it ends in on any
 * link. A later transfer
 * on the same connection starts at the size the last one finished at,
 * `largestWrite`, given to it with `limitWrites`, and a new connection starts
 * again from the system's figure. On a link that carries 244 nothing of
 * this asks or writes anything more.
 */

import { crc32, sha256 } from "./digest.ts";
import {
  CAPABILITIES,
  ENVELOPE_LEN,
  FILL,
  decodeHead,
  decodeUpdateQuery,
  decodeUpdateResponse,
  decodeUpdateStart,
  encodeUpdateOp,
  encodeUpdateStart,
  type UpdateResponse,
  type UpdateTargetName,
} from "./protocol.ts";
import type { Command } from "./session.ts";
import { NotCapable, Session } from "./session.ts";

/**
 * How many writes go out before the device is asked what it has, as the
 * Python library asks. Writes without a response have no flow control: a
 * device writing a page of its flash holds eight incoming writes and throws
 * away what arrives beyond them, a progress check among them.
 */
const WRITES_PER_CHECK = 8;

/** How many times a transfer carries on from the device's own count before it gives up, as the library's. */
const RESUMES = 64;

/** How many times a progress check is asked in all, when the device turns it away or does not answer it, as the library asks it. */
const CHECK_ATTEMPTS = 3;

/** The largest write this contract allows, whatever a device claims. */
const CHUNK_CEILING = 244;

/**
 * The sizes a transfer starts again at, in turn, when its writes do not
 * arrive whole: 101 fits the link of 104 some computers report, and 20 fits
 * every link.
 */
const SMALLER: readonly number[] = [101, 20];

/** What a transfer needs done next. */
export type Step =
  /** Write this and hand the answer to `onAnswer`. */
  | { readonly type: "send"; readonly command: Command }
  /**
   * Write `image.subarray(from, to)` to this characteristic without a
   * response, then call `sent` with how many bytes went out.
   */
  | { readonly type: "data"; readonly characteristic: number; readonly from: number; readonly to: number }
  /** The device has the image and has checked it. Nothing was activated. */
  | { readonly type: "verified"; readonly result: string | null };

/** Why a transfer stopped. */
export class TransferError extends Error {
  /** Always "TransferError". */
  override readonly name: string = "TransferError";
}

/** What a refused request asked for, in the Python library's words for its refusal, by operation. */
const ASKED: Readonly<Record<string, string>> = {
  start: "to start the transfer",
  query: "a progress check",
  activate: "to activate the image",
  abort: "to abort the transfer",
};

/** The device refused a request. */
export class Refused extends TransferError {
  /** Always "Refused". */
  override readonly name = "Refused";
  /** Which operation the device refused. */
  readonly op: string;
  /** The device's status for that operation, in words. */
  readonly status: string;
  /**
   * Build one naming the refused operation and the device's status. The
   * message says what was asked in the Python library's words: "the device
   * refused to start the transfer: busy", "the device refused a progress
   * check: busy" and "the device refused to activate the image: no
   * transfer". A refused finish names what the transfer carried, as a
   * rejected one does: the head for a head, and the image for anything
   * else or no target given, as in "the device refused the image: flash".
   * The library never reports a refused abort, which is said as "to abort
   * the transfer", and any other operation is named as it is given.
   */
  constructor(op: string, status: string, target?: UpdateTargetName) {
    const asked = op === "finish" ? (target === "head" ? "the head" : "the image") : Object.hasOwn(ASKED, op) ? ASKED[op]! : op;
    super(`the device refused ${asked}: ${status}`);
    this.op = op;
    this.status = status;
  }
}

/**
 * The device holds a different number of bytes than were sent, and its
 * count is not advancing: carrying on from it did not move it, or did so
 * more times than a transfer allows.
 */
export class OffsetMismatch extends TransferError {
  /** Always "OffsetMismatch". */
  override readonly name = "OffsetMismatch";
  /** Bytes this side has sent. */
  readonly sent: number;
  /** Bytes the device says it holds. */
  readonly device: number;
  /** Build one with what was sent and what the device reports holding. */
  constructor(sent: number, device: number) {
    super(`the device accepted ${device} bytes where ${sent} were sent, and the transfer is not advancing`);
    this.sent = sent;
    this.device = device;
  }
}

/** The bytes arrived changed. */
export class ChecksumMismatch extends TransferError {
  /** Always "ChecksumMismatch". */
  override readonly name = "ChecksumMismatch";
  /** Bytes checked. */
  readonly sent: number;
  /** This side's checksum over those bytes. */
  readonly ours: number;
  /** The device's checksum over the same bytes. */
  readonly device: number;
  /** Build one with the bytes checked and the two checksums. */
  constructor(sent: number, ours: number, device: number) {
    super(
      `the device's checksum over ${sent} bytes is ${hex32(device)} and ours is ${hex32(ours)}`,
    );
    this.sent = sent;
    this.ours = ours;
    this.device = device;
  }
}

/**
 * The device refused the finished transfer, a head or an image, and said
 * why. A transfer that a progress check says has failed is finished at
 * once, so the device says why, and it ends here, or as `Refused` with the
 * device's status when the device's answer to that finish gives no verdict
 * or a verdict that says verified. One failed at its first 32 bytes with a
 * checksum that is not ours over the writes that carried them starts the
 * file again at the next smaller size instead, where one remains.
 */
export class Rejected extends TransferError {
  /** Always "Rejected". */
  override readonly name = "Rejected";
  /** Why the device refused the transfer, or null when it gave no reason. */
  readonly result: string | null;
  /**
   * Build one with the device's reason, or null when it gave none, and the
   * target the transfer was to: the message calls a head the head and
   * anything else, or no target given, the image.
   */
  constructor(result: string | null, target?: UpdateTargetName) {
    super(`the device refused ${target === "head" ? "the head" : "the image"}: ${result ?? "no reason given"}`);
    this.result = result;
  }
}

/** An answer arrived that this transfer did not ask for. */
export class Unexpected extends TransferError {
  /** Always "Unexpected". */
  override readonly name = "Unexpected";
  /** Build one, with a default message or one of your own. */
  constructor(message = "an answer arrived that this transfer did not ask for") {
    super(message);
  }
}

function hex32(n: number): string {
  return `0x${(n >>> 0).toString(16).padStart(8, "0")}`;
}

type Phase = "starting" | "sending" | "checking" | "finishing" | "aborting" | "done";

/** One image, on its way to one device. */
export class Transfer {
  readonly #target: UpdateTargetName;
  readonly #total: number;
  readonly #id: Uint8Array;
  /** The START request, made as the transfer is, so a slot it cannot carry is refused before anything is sent. */
  readonly #start: Uint8Array;
  #offset = 0;
  #chunk: number;
  #sinceCheck = 0;
  /** Times this transfer carried on from the device's count, and the count it last carried on from. */
  #resumes = 0;
  #resumedAt = -1;
  /** Times the progress check under way went unanswered. */
  #unanswered = 0;
  /** This is true once a progress check has said the transfer failed. The answer to the finish that follows is then a refusal, whatever its status. */
  #failed = false;
  /**
   * The end of the last write sent at this size that carried the file's
   * 32nd byte, or null before one: the device judges an image's envelope
   * when its 32 bytes are in, and its checksum then covers every byte of the
   * writes it took.
   */
  #enveloped: number | null = null;
  /** What a progress check found that made the transfer start the file again at a smaller size, while that ABORT is under way. */
  #smallerBecause: TransferError | null = null;
  #phase: Phase = "starting";
  #result: string | null = null;

  private constructor(target: UpdateTargetName, slot: number, image: Uint8Array, chunk: number) {
    this.#target = target;
    this.#total = image.length;
    // A transfer names itself by what is being sent, so an interrupted
    // one resumes only against the same bytes and never against
    // different ones that happen to be the same length.
    this.#id = sha256(image).subarray(0, 8);
    this.#start = encodeUpdateStart(target, slot, this.#total, this.#id);
    this.#chunk = chunk;
  }

  /**
   * An application image. The device takes it into the slot it does not
   * run from, and its answer to the transfer's START request says which, in
   * `UpdateStart.targetSlot`. The image must be built for that slot: the
   * device's check of what arrived refuses one built for the other. The
   * START request names slot zero, as the protocol has it for every target
   * but a head, so `slot` must be zero, and any other is refused before
   * anything is sent.
   */
  static app(session: Session, slot: number, image: Uint8Array): Transfer {
    if (slot !== 0) {
      throw new TransferError(
        `slot ${slot} is not for an application: an application's START request names slot zero, and the device picks the slot it does not run from`,
      );
    }
    return new Transfer("app", 0, image, chunkFor(session, image));
  }

  /**
   * The model weights. There is one copy of them, so the transfer's START
   * request names slot zero, as the protocol has it for every target but a
   * head.
   */
  static weights(session: Session, image: Uint8Array): Transfer {
    return new Transfer("weights", 0, image, chunkFor(session, image));
  }

  /**
   * A head, into one of the device's head slots.
   *
   * The blob is checked here against the width of the embedding this
   * device's model produces, so a head trained for a different encoder is
   * refused before any of it goes on the wire, and so is a slot that is not
   * a whole number from 0 to 255, with `Invalid`, in words.
   */
  static head(session: Session, slot: number, blob: Uint8Array): Transfer {
    const info = session.info;
    if (info === null) throw new NotCapable("heads");
    if ((info.capabilities & CAPABILITIES.heads) === 0) throw new NotCapable("heads");
    if (info.modelEmbedDim === null || info.headMaxOutputs === null) throw new NotCapable("heads");
    const head = decodeHead(blob);
    if (head.inDim !== info.modelEmbedDim) {
      throw new TransferError(
        `this head takes an embedding of ${head.inDim} and the device's model produces ${info.modelEmbedDim}`,
      );
    }
    if (head.outDim > info.headMaxOutputs) {
      throw new TransferError(
        `this head has ${head.outDim} outputs and the device takes at most ${info.headMaxOutputs}`,
      );
    }
    return new Transfer("head", slot, blob, chunkFor(session, blob));
  }

  /**
   * How much the device has taken, of how much there is. Where the transfer
   * starts the file again at a smaller size, it is what the device's answer
   * to START then gives, which after an ABORT the device handled is the
   * start.
   */
  get progress(): { taken: number; total: number } {
    return { taken: this.#offset, total: this.#total };
  }

  /**
   * Hold every write to at most `n` bytes: the largest write the system
   * reports for this link, where it reports one above 20, or the size the
   * last transfer on the same connection finished at, its `largestWrite`.
   * Given before the transfer starts. A write is never larger than the
   * device's own limit or this contract's either, and where a write does not
   * arrive whole the transfer starts again at the next smaller of 101 and
   * 20.
   *
   * ```javascript
   * const t = Transfer.app(session, 0, image);
   * t.limitWrites(largest);
   * ```
   */
  limitWrites(n: number): void {
    if (Number.isNaN(n)) return;
    this.#chunk = Math.min(this.#chunk, Math.max(1, Math.floor(n)));
  }

  /** The largest write this transfer makes now, which a later transfer on the same connection is held to with `limitWrites`. */
  get largestWrite(): number {
    return this.#chunk;
  }

  /**
   * Say that the system refused the data write `step` last asked for, while
   * the link is up, as some systems refuse a write longer than the link
   * carries. Where a smaller size remains, the transfer starts the file
   * again at it and this gives true: `step` asks for ABORT, then START, and
   * the data from the count the device's answer to START gives. Where none
   * remains it gives false and nothing changes, and the transfer ends in the
   * write's own error. A link that goes down during that ABORT ends the
   * transfer in the write's own error too.
   */
  writeRefused(): boolean {
    return this.#phase === "sending" && this.#smaller(null);
  }

  /**
   * What a progress check found that made the transfer start the file again
   * at a smaller size, while the ABORT that starts it again is under way, and
   * null otherwise: a count not advancing or a checksum over the device's
   * count not matching, in the words it ends in where no smaller size
   * remains, or, for a device that failed the transfer at its first 32 bytes,
   * a checksum not matching over the file up to the end of the write that
   * carried its 32nd byte. A link that goes down during that ABORT ends the
   * transfer in this error, as the Python library's transfer ends.
   */
  get smallerBecause(): TransferError | null {
    return this.#phase === "aborting" ? this.#smallerBecause : null;
  }

  /**
   * Go to the next smaller size, where one remains: the file starts again,
   * with ABORT, and this gives true, `why` kept as what stopped the larger
   * size. Where none remains this gives false and nothing changes.
   */
  #smaller(why: TransferError | null): boolean {
    const next = SMALLER.find((size) => size < this.#chunk);
    if (next === undefined) return false;
    this.#chunk = next;
    this.#smallerBecause = why;
    this.#sinceCheck = 0;
    this.#resumes = 0;
    this.#resumedAt = -1;
    this.#unanswered = 0;
    this.#failed = false;
    this.#enveloped = null;
    this.#phase = "aborting";
    return true;
  }

  /** What to do next. */
  step(image: Uint8Array): Step {
    switch (this.#phase) {
      case "starting":
        return send(this.#start);
      case "checking":
        return send(encodeUpdateOp("query"));
      case "finishing":
        return send(encodeUpdateOp("finish"));
      case "aborting":
        return send(encodeUpdateOp("abort"));
      case "done":
        return { type: "verified", result: this.#result };
      case "sending": {
        const to = Math.min(this.#offset + this.#chunk, image.length);
        return { type: "data", characteristic: FILL.UPDATE_DATA, from: this.#offset, to };
      }
    }
  }

  /** Say how many bytes went out, after writing what `step` asked for. */
  sent(n: number): void {
    if (this.#phase !== "sending") return;
    const end = Math.min(this.#offset + n, this.#total);
    if (this.#offset < ENVELOPE_LEN && ENVELOPE_LEN <= end) this.#enveloped = end;
    this.#offset = end;
    this.#sinceCheck += 1;
    if (this.#sinceCheck >= WRITES_PER_CHECK || this.#offset === this.#total) {
      this.#sinceCheck = 0;
      this.#unanswered = 0;
      this.#phase = "checking";
    }
  }

  /**
   * Say that the device turned away the request `step` last asked to be
   * sent, or did not answer it. A progress check is asked again, as the
   * Python library asks it: three times in all, half a second after the
   * first goes unanswered and a second after the second. Gives how many
   * milliseconds to wait before `step` asks for it again, or null when it
   * is not asked again and the transfer stops. Nothing else is asked
   * again. `Connection.transfer` waits five seconds for the answer to a
   * check, as the library waits. The ABORT of a transfer starting the file
   * again at a smaller size is passed over: this gives 0, and `step` asks
   * for START.
   */
  unanswered(): number | null {
    if (this.#phase === "aborting") {
      this.#smallerBecause = null;
      this.#phase = "starting";
      return 0;
    }
    if (this.#phase !== "checking") return null;
    this.#unanswered += 1;
    return this.#unanswered < CHECK_ATTEMPTS ? 500 * this.#unanswered : null;
  }

  /** Hand in the answer to whatever `step` last asked to be sent. */
  onAnswer(bytes: Uint8Array, image: Uint8Array): void {
    let answer: UpdateResponse;
    try {
      answer = decodeUpdateResponse(bytes);
    } catch (e) {
      throw new Unexpected(`that is not an update answer: ${(e as Error).message}`);
    }
    if (this.#phase === "starting" && answer.opName === "start") {
      this.#refuse(answer);
      const start = decodeUpdateStart(answer.payload);
      // The device may already hold part of this transfer, and says so.
      // Believe it, and test the claim at the next checkpoint like any
      // other.
      this.#offset = Math.min(start.resumeOffset, this.#total);
      if (start.chunkMax > 0) this.#chunk = Math.min(this.#chunk, start.chunkMax);
      this.#sinceCheck = 0;
      this.#unanswered = 0;
      this.#phase = this.#offset === this.#total ? "checking" : "sending";
      return;
    }
    if (this.#phase === "checking" && answer.opName === "query") {
      this.#refuse(answer);
      const q = decodeUpdateQuery(answer.payload);
      if (q.stateName === "failed") {
        // The device stopped taking this transfer, as it does at the first
        // bytes of an image built for its other slot, and holds its verdict.
        // Failed at its 32nd byte, where it judges an image's envelope, with
        // a checksum that is not ours over the writes that carried that
        // byte, it judged an envelope that is not the one sent: the writes
        // did not arrive whole, and the file starts again at a smaller size,
        // where one remains. Otherwise its answer to FINISH gives its
        // verdict, and the transfer stops there with the device's reason.
        if (q.offset === ENVELOPE_LEN && this.#enveloped !== null) {
          const ours = crc32(image.subarray(0, this.#enveloped));
          if ((q.crc32 >>> 0) !== (ours >>> 0) && this.#smaller(new ChecksumMismatch(this.#enveloped, ours, q.crc32))) return;
        }
        this.#failed = true;
        this.#phase = "finishing";
        return;
      }
      // The device's checksum is over the bytes it holds, so a count behind
      // what was sent is checked as well: one whose checksum does not match
      // holds bytes that are not the start of the file, as a link that cuts
      // writes leaves it, and the file starts again at a smaller size, where
      // one remains.
      const ours = crc32(image.subarray(0, q.offset));
      if ((q.crc32 >>> 0) !== (ours >>> 0)) {
        const changed = new ChecksumMismatch(q.offset, ours, q.crc32);
        if (this.#smaller(changed)) return;
        throw changed;
      }
      if (q.offset !== this.#offset) {
        // A device that threw writes away holds fewer bytes than were
        // sent, and they are the file's first bytes. The transfer carries on
        // from the device's own count, which is what a resume is for, as the
        // Python library does, and gives up when that count stops advancing.
        this.#resumes += 1;
        if (this.#resumes > RESUMES || q.offset <= this.#resumedAt) {
          // Writes that do not arrive whole on a link smaller than they are
          // look so: the file starts again at a smaller size, where one
          // remains.
          const stalled = new OffsetMismatch(this.#offset, q.offset);
          if (this.#smaller(stalled)) return;
          throw stalled;
        }
        this.#resumedAt = this.#offset = q.offset;
        this.#phase = this.#offset < this.#total ? "sending" : "finishing";
        return;
      }
      this.#phase = this.#offset === this.#total ? "finishing" : "sending";
      return;
    }
    if (this.#phase === "aborting" && answer.opName === "abort") {
      // Whatever the device answered, a refusal among them, the file starts
      // again, from the count the device's answer to START gives.
      this.#smallerBecause = null;
      this.#phase = "starting";
      return;
    }
    if (this.#phase === "finishing" && answer.opName === "finish") {
      // A transfer that a progress check said had failed is never verified:
      // the answer to its finish ends it as a refused finish ends it,
      // whatever the answer's status.
      if (!answer.ok || this.#failed) {
        // A refusal that gives a verdict other than verified is a rejected
        // transfer, the verdict in words, or "result N" for a number the
        // protocol does not define, as the Python library says it. One that
        // gives no verdict, or a verdict that says verified, is a refused
        // request, with the status the device gave. A device asked to finish
        // again after a transfer that verified refuses so: no transfer is
        // left to finish, and it still holds the verdict.
        const verdict = answer.verifyResultName;
        if (verdict === null || verdict === "verified") throw new Refused("finish", answer.statusName, this.#target);
        throw new Rejected(verdict, this.#target);
      }
      this.#result = answer.verifyResultName;
      this.#phase = "done";
      return;
    }
    throw new Unexpected();
  }

  /** Give up. The device drops what it has. */
  abort(): Command {
    return send(encodeUpdateOp("abort")).command;
  }

  #refuse(answer: UpdateResponse): void {
    if (!answer.ok) throw new Refused(answer.opName ?? "an update", answer.statusName);
  }
}

/**
 * Put a transferred image in force.
 *
 * The device resets, so the link drops and everything a caller knows
 * about it is stale. An application comes back on trial and rolls itself
 * back if it cannot confirm. Weights are verified at every boot, so
 * restarting is what puts new ones in force.
 */
export function activate(): Command {
  return send(encodeUpdateOp("activate")).command;
}

function send(bytes: Uint8Array): { type: "send"; command: Command } {
  return {
    type: "send",
    command: { characteristic: FILL.UPDATE_CONTROL, bytes, withResponse: true },
  };
}

function chunkFor(session: Session, image: Uint8Array): number {
  const info = session.info;
  if (info === null) throw new NotCapable("update");
  if ((info.capabilities & CAPABILITIES.update) === 0) throw new NotCapable("update");
  if (image.length === 0) throw new TransferError("nothing is not an image");
  const claimed = info.updateChunkMax ?? CHUNK_CEILING;
  return Math.min(claimed > 0 ? claimed : CHUNK_CEILING, CHUNK_CEILING);
}

