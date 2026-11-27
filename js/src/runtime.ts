/**
 * The clocks the package reads where the Python library reads its `time`
 * module, the wait a run takes, and how a cue sounds: one place, so a run
 * and the package's own tests all read the same
 * clocks, as the library's tests put one stand-in in place of every
 * module's clock.
 *
 * Internal. `run.ts` hands these on to the package's own tests.
 */

import { play } from "./audio.ts";
import { hostClocks } from "./capture/clock.ts";
import { after } from "./timers.ts";

/** The clocks a run reads and the wait it takes. */
export interface RunTime {
  /** Seconds on a clock that never steps. */
  monotonic(): number;
  /** Seconds since 1970 on the wall clock. */
  time(): number;
  /** Wait this many seconds. */
  sleep(seconds: number): Promise<void>;
}

/**
 * This reads the host's own clocks and waits on the package's own timer. A
 * wait longer than a timer holds is kept whole, an infinite one never ends,
 * and one that is not a number ends at once, as a timer ends it.
 */
export const HOST_TIME: RunTime = {
  monotonic: () => hostClocks().monotonic,
  time: () => Date.now() / 1000,
  sleep: (seconds) =>
    new Promise<void>((resolve) => {
      const s = Number(seconds);
      after(Number.isNaN(s) ? 0 : s, resolve);
    }),
};

/** What every run reads: the clocks and the wait, and how a cue sounds. */
export const runtime: {
  time: RunTime;
  play: (tone: unknown, text: string) => void;
} = { time: HOST_TIME, play };
