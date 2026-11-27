/**
 * The IntoMind instrument client, for a browser and for Node.
 *
 * The protocol, the session, and the timebase do no input and no output.
 * They are handed bytes that arrived and they hand back bytes to send,
 * which means they work over whatever Bluetooth stack you already have and
 * can be tested without one. `web-bluetooth.ts` is the browser transport,
 * and it is the only file in this package that touches a radio. `capture`
 * reads recordings and writes their exports as bytes in the same way, and
 * `capture/node` is the one part of it that touches a disk.
 *
 * ```ts
 * import { requestAndConnect } from "@intomind/sdk";
 *
 * const device = await requestAndConnect({
 *   onEvent(event) {
 *     if (event.type === "samples") write(event.batch);
 *     // Written down, never smoothed over.
 *     if (event.type === "gap") writeGap(event.gap);
 *   },
 * });
 * await device.send(device.session.startStream());
 * ```
 *
 * What a device can do is what the device says it can do. A capability is
 * a bit the device reports rather than a version number this package
 * compares against. The one table of hardware here is a wiring preset for
 * IntoMind hardware version 1.1.7, because the protocol gives a device no
 * way to report how its electrodes are wired.
 */

export * as protocol from "./protocol.ts";
export * as capture from "./capture/index.ts";
export * as instrument from "./instrument.ts";
export * as experiments from "./experiments.ts";
export * as audio from "./audio.ts";
export * as analysis from "./analysis.ts";
export * as model from "./model.ts";
export * as placement from "./placement.ts";

export {
  CAPABILITIES,
  CAPABILITIES_HIGH,
  DeviceInfo,
  EMBED_SCALE,
  FILL,
  GAIN_BY_CODE,
  Invalid,
  MODES,
  NAME_MAX_COMPOSED,
  NAME_PREFIX,
  NO_ENCODER_ID,
  OPCODES,
  PRODUCT_NAME,
  ProtocolError,
  RATE_BY_CODE,
  Reserved,
  SERVICE,
  Truncated,
  VERSION,
  buildHead,
  composeName,
  continuity,
  decodeHead,
  decodeModelInterval,
  decodeNameParts,
  decodePacket,
  displayNames,
  evaluateHead,
  microvolts,
  nameFits,
  quantizeEmbedding,
  uuid,
} from "./protocol.ts";

export type {
  CapabilityHighName,
  CapabilityName,
  Continuity,
  ModeName,
  ModelInterval,
  Envelope,
  HeadBlob,
  HeadEntry,
  OpcodeName,
  Packet,
  Prediction,
  Refusal,
  Request,
  Response,
  Status,
  UpdateResponse,
} from "./protocol.ts";

export { Batch, NoDeviceInfo, NotCapable, Session, SessionError, Unanswered } from "./session.ts";
export type { Command, Event, Gap } from "./session.ts";

export { Exchange, MAX_SKEW_PPM, MAX_SKEW_STDERR_PPM, Timebase } from "./timebase.ts";
export type { Fit } from "./timebase.ts";

export {
  ChecksumMismatch,
  OffsetMismatch,
  Refused,
  Rejected,
  Transfer,
  TransferError,
  Unexpected,
  activate,
} from "./transfer.ts";
export type { Step } from "./transfer.ts";

export { Connection, NoBluetooth, connect, hostClock, requestAndConnect, requestDevice } from "./web-bluetooth.ts";
export { DaisyChain } from "./daisy-chain.ts";
export type { ConnectOptions, RequestOptions } from "./web-bluetooth.ts";
