/**
 * Timers that keep to the wait asked for. A runtime's `setTimeout` takes a
 * delay of at most 2**31 - 1 milliseconds, about 24.8 days, and fires at once
 * for any longer one, so a long wait is armed here in steps no longer than
 * that, and an infinite one is never armed at all: it waits.
 *
 * Internal: nothing here is part of the package's surface.
 */

/** The longest delay a runtime's `setTimeout` keeps to, in milliseconds. */
export const LONGEST_MS = 2 ** 31 - 1;

/**
 * Call `fn` once `seconds` have passed, and give back how to cancel it. A
 * wait longer than `LONGEST_MS` is armed again in steps until it has passed,
 * and an infinite one, or one that is not a number, is never called.
 */
export function after(seconds: number, fn: () => void): () => void {
  if (!(seconds < Infinity)) return () => {};
  let left = Math.max(0, seconds * 1000);
  let id: ReturnType<typeof setTimeout> | null = null;
  const arm = (): void => {
    const step = Math.min(left, LONGEST_MS);
    left -= step;
    id = setTimeout(() => {
      if (left > 0) {
        arm();
        return;
      }
      id = null;
      fn();
    }, step);
  };
  arm();
  return () => {
    if (id !== null) clearTimeout(id);
    id = null;
  };
}

/**
 * What `promise` settles with, or the error `late` makes once `seconds` have
 * passed first, timed as `after` times a wait. An infinite wait is the
 * promise alone.
 */
export async function within<T>(promise: Promise<T>, seconds: number, late: () => Error): Promise<T> {
  let cancel: () => void = () => {};
  const timeout = new Promise<never>((_resolve, reject) => {
    cancel = after(seconds, () => reject(late()));
  });
  try {
    return await Promise.race([promise, timeout]);
  } finally {
    cancel();
  }
}
