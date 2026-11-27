/**
 * The firmware of the IntoMind device that checks electrode contact, and
 * the words a contact check is refused in on an earlier one. Internal: the
 * session, the browser connection and the profile all read it here.
 */

/**
 * The first firmware of the IntoMind device that checks electrode contact.
 * On an earlier one lead-off detection does not work, and every electrode
 * reads off whether it touches skin or not, so none of its lead-off bits is
 * a verdict and a contact check is refused. From this one the device's
 * lead-off bits are no verdict until its first judgment of a stream.
 */
export const JUDGES_FROM: readonly [number, number, number] = [1, 4, 6];

/** Whether firmware `a` is `b` or later, compared as Python compares tuples. */
export function firmwareAtLeast(a: readonly number[], b: readonly number[]): boolean {
  for (let i = 0; i < Math.min(a.length, b.length); i++) if (a[i] !== b[i]) return a[i]! > b[i]!;
  return a.length >= b.length;
}

/**
 * The words a contact check is refused in on an IntoMind device whose
 * firmware is older than 1.4.6, which cannot check electrode contact, its
 * own version filled in, or null for a device that can check it or has not
 * said which firmware it runs.
 */
export function cannotCheckContact(firmware: readonly number[] | null | undefined): string | null {
  if (firmware === null || firmware === undefined || firmware.length === 0 || firmwareAtLeast(firmware, JUDGES_FROM)) return null;
  return `This device's firmware (${firmware.join(".")}) cannot check electrode contact. Update it to ${JUDGES_FROM.join(".")} or later.`;
}
