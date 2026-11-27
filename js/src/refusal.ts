/**
 * What a control request that did not go through is refused with, in the
 * Python library's words, carried by the package's own `SessionError` as
 * `capture` carries the library's words in its errors: the exception the
 * library raises, then its message. Internal: the browser connection and
 * the device an experiment drives both refuse in these words.
 */

import { STATUS_CODES } from "./protocol.ts";

/** The status a device answers with while USB power is present (1.4): it does not stream the natural signal while plugged in. */
const STATUS_USB_POWER = 7;

/** A write to a link that is down, refused before anything is written, as the library's `Device.command` refuses it. */
export const LINK_DOWN = "ConnectionError: BLE link is down";

/**
 * A request the device answered with a status other than ok, as the
 * library's `client.Refused` words it: the opcode, the status and its name,
 * and what a person can do about it where there is something.
 */
export function refusal(opcode: number, status: number): string {
  const name = STATUS_CODES[status] ?? `undefined status ${status}`;
  let text = `Refused: op 0x${opcode.toString(16).padStart(2, "0")} -> status ${status} (${name})`;
  if (status === STATUS_USB_POWER) text += ": the device does not stream the natural signal while plugged in. Unplug it to record";
  return text;
}
