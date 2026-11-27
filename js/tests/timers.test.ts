/**
 * A wait longer than a runtime's timer keeps to, about 24.8 days, is not cut
 * short, and an infinite one waits.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { HOST_TIME } from "../src/runtime.ts";
import { LONGEST_MS, after } from "../src/timers.ts";

/** A wait of `ms` on the real clock. */
const pause = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

test("a wait past the longest a timer keeps to is armed again in steps until it has passed, and an infinite one is never called", async (t) => {
  let fired = 0;
  const cancelLong = after((LONGEST_MS + 5_000) / 1000, () => (fired += 1));
  const cancelEver = after(Infinity, () => (fired += 1));
  await pause(30);
  assert.equal(fired, 0, "neither fires at once, as an overflowing timer would");
  cancelLong();
  cancelEver();
  t.mock.timers.enable({ apis: ["setTimeout"] });
  let at = 0;
  after((2 * LONGEST_MS + 7) / 1000, () => (at += 1));
  t.mock.timers.tick(LONGEST_MS);
  t.mock.timers.tick(LONGEST_MS);
  assert.equal(at, 0, "not before the whole wait has passed");
  t.mock.timers.tick(7);
  assert.equal(at, 1, "once it has");
  t.mock.timers.reset();
});

test("a run's wait on the host's timer is kept whole when it is longer than a timer holds, never ends when it is infinite, and ends at once when it is not a number, as a timer ends it", async (t) => {
  const turn = (): Promise<void> => new Promise((resolve) => setImmediate(resolve));
  t.mock.timers.enable({ apis: ["setTimeout"] });
  try {
    const woke: string[] = [];
    void HOST_TIME.sleep((LONGEST_MS + 5_000) / 1000).then(() => woke.push("long"));
    void HOST_TIME.sleep(Infinity).then(() => woke.push("infinite"));
    t.mock.timers.tick(LONGEST_MS);
    await turn();
    assert.deepEqual(woke, [] as string[], "a wait ended before it had passed");
    t.mock.timers.tick(5_000);
    await turn();
    assert.deepEqual(woke, ["long"], "the long wait ends once it has passed, and the infinite one never does");
    void HOST_TIME.sleep(NaN).then(() => woke.push("not a number"));
    t.mock.timers.tick(0);
    await turn();
    assert.deepEqual(woke, ["long", "not a number"], "a wait that is not a number ends at once");
  } finally {
    t.mock.timers.reset();
  }
});
