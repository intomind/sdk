/**
 * The timebase, and what it refuses to believe.
 *
 * The fit between two clocks is a line. These tests hold down the three
 * refusals: one exchange never claims a rate, a skew two crystals cannot
 * have is dropped, and a slope drawn through noise is dropped. In each
 * case the offset is kept, because the offset is the part that was
 * measured.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { Exchange, MAX_SKEW_PPM, MAX_SKEW_STDERR_PPM, Timebase } from "../src/timebase.ts";

const TICK_HZ = 1_000_000;

/** Exchanges taken ten seconds apart, on a device running at `skewPpm`. */
function exchanges(skewPpm: number, offset: number, n: number, roundTrip = 0.004): Exchange[] {
  return Array.from({ length: n }, (_, i) => {
    const deviceS = i * 10;
    const host = deviceS * (1 + skewPpm * 1e-6) + offset;
    return new Exchange(host - roundTrip / 2, host + roundTrip / 2, BigInt(Math.trunc(deviceS * TICK_HZ)));
  });
}

function fitted(es: Exchange[]): Timebase {
  const tb = new Timebase(TICK_HZ);
  for (const e of es) tb.observe(e);
  return tb;
}

test("nothing is mapped before anything is measured", () => {
  const tb = new Timebase(TICK_HZ);
  assert.equal(tb.fit, null);
  assert.equal(tb.hostTime(0n), null);
  assert.equal(tb.exchanges.length, 0);
});

test("one exchange fixes an offset and claims no rate", () => {
  const tb = fitted(exchanges(20, 1000, 1));
  const fit = tb.fit!;
  assert.ok(!fit.skewUsed, "a rate cannot be measured from one point");
  assert.equal(fit.skewPpm, 0);
  assert.ok(Math.abs(fit.offsetS - 1000) < 1e-6);
  assert.equal(fit.exchanges, 1);
  // The uncertainty of that one exchange is half its round trip.
  assert.ok(Math.abs(fit.residualUs - 2000) < 1e-6);
});

test("several exchanges find the rate", () => {
  const tb = fitted(exchanges(20, 1000, 8));
  const fit = tb.fit!;
  assert.ok(fit.skewUsed);
  assert.ok(Math.abs(fit.skewPpm - 20) < 0.5, `${fit.skewPpm}`);
  assert.ok(Math.abs(fit.offsetS - 1000) < 1e-3);
  assert.equal(fit.exchanges, 8);
  // An hour of device time lands where the rate says it does.
  const atHour = tb.hostTime(BigInt(3600 * TICK_HZ))!;
  assert.ok(Math.abs(atHour - (1000 + 3600 * 1.00002)) < 0.01, `${atHour}`);
});

test("a skew that two crystals cannot have is refused", () => {
  const fit = fitted(exchanges(5000, 0, 8)).fit!;
  assert.ok(!fit.skewUsed, "five thousand parts per million is not two crystals");
  assert.equal(fit.skewPpm, 0, "the offset is kept and the rate is dropped");
  assert.equal(MAX_SKEW_PPM, 1000);
  // A skew inside the bound, measured cleanly, is used.
  assert.ok(fitted(exchanges(900, 0, 8)).fit!.skewUsed);
});

test("a rate drawn through noise is refused", () => {
  // One exchange whose answer came back late drags a line through nothing.
  // The fit says so rather than growing that error.
  const es = exchanges(20, 1000, 5);
  es[2] = new Exchange(es[2]!.before + 0.5, es[2]!.after + 0.5, es[2]!.deviceTicks);
  const fit = fitted(es).fit!;
  assert.ok(!fit.skewUsed, "a fit this noisy is not a measurement of rate");
  assert.equal(fit.skewPpm, 0);
  assert.ok(fit.residualUs > 1000);
  assert.equal(MAX_SKEW_STDERR_PPM, 100);
});

test("two exchanges never claim a rate on their own", () => {
  // With two points the line is exact and its standard error is not
  // defined, so the slope is not a measurement no matter how clean it looks.
  const fit = fitted(exchanges(20, 1000, 2)).fit!;
  assert.ok(!fit.skewUsed);
  assert.equal(fit.exchanges, 2);
});

test("an exchange states the window its answer lies in", () => {
  const e = new Exchange(1000, 1000.004, 5_000_000n);
  assert.ok(Math.abs(e.host - 1000.002) < 1e-9);
  assert.ok(Math.abs(e.uncertainty - 0.002) < 1e-9);
  // The device's own tick rate is what turns ticks into seconds.
  const tb = fitted([e]);
  assert.ok(Math.abs(tb.hostTime(5_000_000n)! - 1000.002) < 1e-9);
  assert.ok(Math.abs(tb.hostTime(6_000_000n)! - 1001.002) < 1e-9);
});
