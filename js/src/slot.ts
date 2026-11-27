/**
 * A head's slot as the one byte that carries it to the device, taken as the
 * Python library's `protocol._slot_byte` takes it. Internal: the protocol's
 * requests, a transfer and the browser connection refuse a slot in these
 * words, each with its own `Invalid`.
 */

import { shown, tooLong } from "./capture/pyjson.ts";

/**
 * The refusal of `slot`, in the library's own words, where it is not a whole
 * number from 0 to 255, which is all one byte holds, or null where it is.
 * True and false, text, null and undefined are not whole numbers here, and
 * neither is a number with a fraction. Whether the device has the slot is
 * for the device to say.
 */
export function slotRefusal(slot: unknown): string | null {
  if (typeof slot === "number" && Number.isInteger(slot) && slot >= 0 && slot <= 255) return null;
  if (typeof slot === "bigint" && slot >= 0n && slot <= 255n) return null;
  return `a slot is a whole number from 0 to 255, not ${tooLong(slot) ?? shown(slot)}`;
}
