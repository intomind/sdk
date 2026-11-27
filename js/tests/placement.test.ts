/**
 * Where a device was put, held to the Python library's placement: every
 * position it offers, each key and what is under it, the refusals word for
 * word, and the placements it made with one position, with sites or with a
 * preset, each resolved, turned into a montage and described here as the
 * library did in `contract/placement`.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { Fail } from "../src/capture/montage.ts";
import { describe, regionsUnder, resolve, scalp, toMontage } from "../src/capture/placement.ts";
import * as J from "../src/capture/pyjson.ts";
import type { Value } from "../src/capture/pyjson.ts";
import * as placementModule from "../src/placement.ts";
import { Invalid } from "../src/protocol.ts";
import { inJavaScript } from "./words.ts";

const fixture = J.loads(readFileSync(new URL("../../contract/placement/expected.json", import.meta.url), "utf8"));

function items(v: Value | undefined): Value[] {
  assert.ok(v !== undefined && v.kind === "array", "a list in the fixture");
  return v.items;
}

/** What the library returned or the words it raised, against what this did. */
function compare(name: string, want: Value, run: () => Value, problems: string[]): Value | null {
  let got: Value | null = null;
  let raised: string | null = null;
  try {
    got = run();
  } catch (e) {
    raised = e instanceof Fail && e.kind === "placement" ? `PlacementError: ${e.message}` : String(e);
  }
  const result = J.get(want, "result");
  const raises = J.get(want, "raises");
  if (result !== undefined) {
    if (raised !== null) problems.push(`${name}: raised ${raised} where the library returned`);
    else if (J.dumps(got!) !== J.dumps(result)) problems.push(`${name}: ${J.dumps(got!)} differs from ${J.dumps(result)}`);
  } else if (raised !== J.pyStr(raises!)) {
    problems.push(`${name}: ${raised ?? J.dumps(got!)} where the library raised ${J.pyStr(raises!)}`);
  }
  return got;
}

test("every position the library offers reads here as it offers it", () => {
  const offered = items(J.field(fixture, "positions"));
  assert.equal(offered.length, 78);
  for (const want of offered) assert.equal(J.dumps(scalp(J.field(want, "key"))), J.dumps(want));
});

test("the regions are the library's REGIONS, in its order, each with its words and kind", () => {
  const want = items(J.field(fixture, "regions"));
  const got = [...placementModule.REGIONS.values()];
  assert.equal(got.length, want.length);
  got.forEach((r, i) => assert.equal(J.dumps(J.fromPlain(r)), J.dumps(want[i]!), `region ${i}`));
  assert.deepEqual([...placementModule.REGIONS.keys()], got.map((r) => r.key));
});

test("every position, and every key that is not one, reads as the library reads it", () => {
  const problems: string[] = [];
  const cases = items(J.field(fixture, "scalp"));
  assert.ok(cases.length >= 110, String(cases.length));
  for (const c of cases) {
    const key = J.field(c, "key");
    compare(`scalp(${J.pyRepr(key)})`, c, () => scalp(key), problems);
  }
  for (const c of items(J.field(fixture, "regions_under"))) {
    const key = J.field(c, "key");
    compare(`regions_under(${J.pyRepr(key)})`, c, () => J.array(regionsUnder(key)), problems);
  }
  assert.equal(problems.length, 0, problems.join("\n"));
});

/** What the public module raised, written as the fixture writes the library's refusals. */
function publicWords(e: unknown): string {
  if (e instanceof Invalid || e instanceof TypeError) return e.message;
  return `unexpected ${String(e)}`;
}

/** The public module's answer against the library's: the same value, or the same refusal in the same words. */
function comparePublic(name: string, want: Value, run: () => Value, problems: string[]): void {
  let got: Value | null = null;
  let raised: string | null = null;
  try {
    got = run();
  } catch (e) {
    raised = publicWords(e);
  }
  const result = J.get(want, "result");
  if (result !== undefined) {
    if (raised !== null) problems.push(`${name}: raised ${raised} where the library returned`);
    else if (J.dumps(got!) !== J.dumps(result)) problems.push(`${name}: ${J.dumps(got!)} differs from ${J.dumps(result)}`);
  } else if (raised !== inJavaScript(J.pyStr(J.field(want, "raises")))) {
    problems.push(`${name}: ${raised ?? J.dumps(got!)} where the library raised ${J.pyStr(J.field(want, "raises"))}`);
  }
}

test("the published positions are the library's, in its order", () => {
  const offered = items(J.field(fixture, "positions"));
  const mine = placementModule.positions();
  assert.equal(mine.length, offered.length);
  for (let i = 0; i < mine.length; i++) assert.equal(J.dumps(mine[i]!), J.dumps(offered[i]!));
});

test("the published scalp and regionsUnder read every key as the library does, refusals in its words", () => {
  const problems: string[] = [];
  for (const c of items(J.field(fixture, "scalp"))) {
    const key = J.field(c, "key");
    comparePublic(`scalp(${J.pyRepr(key)})`, c, () => placementModule.scalp(key), problems);
  }
  for (const c of items(J.field(fixture, "regions_under"))) {
    const key = J.field(c, "key");
    comparePublic(`regionsUnder(${J.pyRepr(key)})`, c, () => J.array(placementModule.regionsUnder(key)), problems);
  }
  assert.equal(problems.length, 0, problems.join("\n"));
});

test("the published placement makes what the library's placement makes, from the same arguments, refusals in its words", () => {
  const problems: string[] = [];
  const cases = items(J.field(fixture, "placements"));
  assert.ok(cases.length >= 19, String(cases.length));
  for (const c of cases) {
    const name = J.pyStr(J.field(c, "name"));
    const args = J.field(c, "args");
    const arg = (k: string): Value | undefined => J.get(args, k);
    const bias = arg("bias_driven");
    comparePublic(
      name,
      c,
      () =>
        placementModule.placement(J.field(c, "layout"), arg("sites") ?? null, {
          ...(arg("preset") !== undefined ? { preset: arg("preset")! } : {}),
          ...(arg("rotation_deg") !== undefined ? { rotationDeg: arg("rotation_deg")! } : {}),
          ...(arg("over") !== undefined ? { over: arg("over")! } : {}),
          ...(arg("note") !== undefined ? { note: arg("note")! } : {}),
          ...(bias !== undefined ? { biasDriven: J.truthy(bias) } : {}),
          ...(arg("position") !== undefined ? { position: arg("position")! } : {}),
        }),
      problems,
    );
  }
  assert.equal(problems.length, 0, problems.join("\n"));
});

test("a placement the library made at one position, or with sites or a preset, is resolved and described as the library does it", () => {
  const problems: string[] = [];
  const cases = items(J.field(fixture, "placements")).filter((c) => J.get(c, "result") !== undefined);
  assert.ok(cases.length >= 11, String(cases.length));
  for (const c of cases) {
    const name = J.pyStr(J.field(c, "name"));
    const lay = J.field(c, "layout");
    const p = J.field(c, "result");
    const same = (what: string, run: () => Value, want: Value): void => {
      let got: string;
      try {
        got = J.dumps(run());
      } catch (e) {
        got = `raised ${String(e)}`;
      }
      if (got !== J.dumps(want)) problems.push(`${name}, ${what}: ${got} differs from ${J.dumps(want)}`);
    };
    same("resolved", () => resolve(lay, p), J.field(c, "resolved"));
    same("montage", () => toMontage(lay, p), J.field(c, "montage"));
    same("described", () => J.str(describe(lay, p, true)), J.field(c, "described"));
    same("described without extras", () => J.str(describe(lay, p, false)), J.field(c, "described_plain"));
  }
  assert.equal(problems.length, 0, problems.join("\n"));
});
