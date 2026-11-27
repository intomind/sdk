/**
 * The instrument layer held to the Python library's: what `Device.profile`
 * says each device can do, contact from lead-off bytes, the enforced
 * preconditions and their words, and the effective settings a manifest
 * records, all from `contract/profile`, which the library wrote.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import * as J from "../src/capture/pyjson.ts";
import type { Value } from "../src/capture/pyjson.ts";
import {
  CONTACT_MIN_FRACTION,
  EnforcedViolation,
  TIERS,
  checkEnforced,
  contactChannels,
  contactFromLeadoff,
  effectiveSettings,
  profile,
} from "../src/instrument.ts";
import { Invalid, decodeDeviceInfo, fromHex } from "../src/protocol.ts";
import { Session } from "../src/session.ts";
import { inJavaScript } from "./words.ts";

/**
 * A refusal as the fixture writes the library's, `Exception: message`.
 * `EnforcedViolation` shares the library's name. The package's `Invalid`
 * and `TypeError` carry the name of the exception the library raises at the
 * head of their message, and each is the class for that exception here.
 */
function said(e: unknown): string {
  const err = e as Error;
  if (err instanceof EnforcedViolation) return `${err.name}: ${err.message}`;
  const python = err.message.slice(0, err.message.indexOf(":"));
  const expected = { OverflowError: Invalid, AttributeError: TypeError }[python];
  assert.ok(expected !== undefined && err instanceof expected, `${err.name}: ${err.message}`);
  return err.message;
}

const fixture = J.loads(readFileSync(new URL("../../contract/profile/profile.json", import.meta.url), "utf8"));

function items(v: Value | undefined): Value[] {
  assert.ok(v !== undefined && v.kind === "array", "a list in the fixture");
  return v.items;
}

function text(v: Value): string {
  assert.equal(v.kind, "str");
  return (v as { value: string }).value;
}

test("every device's profile is what the library composes, field for field and in its order", () => {
  const cases = items(J.get(fixture, "profiles"));
  assert.ok(cases.length >= 40, String(cases.length));
  const problems: string[] = [];
  for (const c of cases) {
    const name = text(J.field(c, "name"));
    const hex = J.field(c, "info");
    const info = hex.kind === "null" ? null : decodeDeviceInfo(fromHex(text(hex)));
    const mine = JSON.stringify(profile(info));
    const theirs = JSON.stringify(J.toPlain(J.field(c, "profile")));
    if (mine !== theirs) problems.push(`${name}:\n  here    ${mine}\n  library ${theirs}`);
    if (info !== null) {
      const session = new Session();
      session.onDeviceInfo(fromHex(text(hex)));
      if (JSON.stringify(session.profile()) !== theirs) problems.push(`${name}: the session's profile differs`);
    }
  }
  assert.equal(problems.length, 0, problems.join("\n"));
  assert.deepEqual(new Session().profile(), { vendor: "IntoMind", hardware: null, controls: {} });
});

test("a device's wiring comes from the hardware version it reports and nothing else", () => {
  const byName = new Map(items(J.get(fixture, "profiles")).map((c) => [text(J.field(c, "name")), c]));
  const wiringOf = (name: string) => {
    const c = byName.get(name);
    assert.ok(c !== undefined, name);
    return profile(decodeDeviceInfo(fromHex(text(J.field(c, "info"))))).wiring;
  };
  const one = wiringOf("the IntoMind One's hardware, which has a wiring preset");
  assert.ok(one !== undefined && one.one_locus === true);
  assert.equal(one.layout.key, "one-locus-4");
  assert.equal(wiringOf("the IntoMind One's hardware, everything declared")?.layout.key, "one-locus-4-bias");
  assert.equal(wiringOf("the IntoMind One's hardware, eight channels")?.layout.key, "one-locus-8");
  assert.equal(wiringOf("other hardware, 1.0.0, which has no preset"), undefined);
  assert.equal(wiringOf("everything"), undefined);
});

test("contact is decoded from the lead-off bytes as the library decodes it, refusals included", () => {
  const cases = items(J.get(fixture, "contact"));
  assert.ok(cases.length >= 30, String(cases.length));
  const problems: string[] = [];
  for (const c of cases) {
    const loff = J.toPlain(J.field(c, "loff")) as number | boolean | (number | boolean)[];
    const n = J.asF64(J.field(c, "n_channels"))!;
    const want = J.get(c, "result");
    let got: string;
    try {
      got = JSON.stringify(contactFromLeadoff(loff, n));
    } catch (e) {
      got = said(e);
    }
    const theirs =
      want !== undefined ? JSON.stringify(J.toPlain(want)) : `${text(J.field(c, "raises"))}: ${text(J.field(c, "message"))}`;
    if (got !== theirs) problems.push(`${JSON.stringify(loff)} over ${n} channels: ${got}, the library ${theirs}`);
  }
  assert.equal(problems.length, 0, problems.join("\n"));
  assert.equal(CONTACT_MIN_FRACTION, J.asF64(J.field(fixture, "min_fraction")));
  assert.deepEqual([...TIERS], J.toPlain(J.field(fixture, "tiers")));
});

test("the channels contact is decoded for are the device's own, or the default", () => {
  for (const c of items(J.get(fixture, "contact_channels"))) {
    const channels = J.toPlain(J.field(c, "channels"));
    const fallback = J.asF64(J.field(c, "default"))!;
    const device =
      channels === "no device" ? null : { info: channels === null ? null : decodeDeviceInfoWithChannels(channels as number) };
    assert.equal(contactChannels(device, fallback), J.asF64(J.field(c, "result")), J.dumps(c));
  }
});

/** A device's info that says only how many channels it has. */
function decodeDeviceInfoWithChannels(channels: number) {
  const b = fromHex(text(J.field(items(J.get(fixture, "profiles"))[1]!, "info")));
  b[5] = channels;
  return decodeDeviceInfo(b);
}

test("an enforced precondition refuses as the library refuses, in the library's words", () => {
  const cases = items(J.get(fixture, "enforced"));
  assert.ok(cases.length >= 15, String(cases.length));
  const problems: string[] = [];
  for (const c of cases) {
    const name = text(J.field(c, "name"));
    const spec = J.field(c, "spec");
    const declared = J.get(spec, "enforced");
    let got = "ok";
    try {
      checkEnforced(declared === undefined ? { name: text(J.field(spec, "name")) } : { name: text(J.field(spec, "name")), enforced: declared }, J.field(c, "params"));
    } catch (e) {
      got = said(e);
    }
    const raised = J.get(c, "raises");
    const theirs = raised === undefined ? "ok" : `${text(raised)}: ${inJavaScript(text(J.field(c, "message")))}`;
    if (got !== theirs) problems.push(`${name}: ${got}, the library ${theirs}`);
  }
  assert.equal(problems.length, 0, problems.join("\n"));
});

test("the effective settings are the library's, in its order and with its types", () => {
  for (const c of items(J.get(fixture, "effective"))) {
    const got = effectiveSettings({}, J.field(c, "actual"), J.field(c, "applied"), J.field(c, "params"));
    assert.equal(J.dumps(got), J.dumps(J.field(c, "result")));
  }
  assert.throws(() => effectiveSettings({}, null, {}, {}), TypeError);
});
