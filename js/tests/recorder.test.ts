/**
 * The recorder held to the Python library's: every session in
 * `contract/recorder` replayed through a `Session`, and everything it writes
 * compared with what the library wrote from the same script. The clock
 * record after every exchange, the configuration and the processing read
 * back after a run, the manifest and the events byte for byte, the arrays
 * element for element and in the same layout, the checksums, the session
 * log, and which files are left read-only. Also where the electrodes went,
 * every way a manifest is told it, and processing chains in words.
 *
 * The one difference allowed is the software a recording names: a recording
 * made here names this package where the library names Python.
 */

import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import process from "node:process";
import test from "node:test";

import { sha256 } from "../src/digest.ts";
import { fsum, hostClocks } from "../src/capture/clock.ts";
import { openCollection, writeRecording } from "../src/capture/node.ts";
import { groupedFixed } from "../src/capture/pyfmt.ts";
import * as npy from "../src/capture/npy.ts";
import * as J from "../src/capture/pyjson.ts";
import type { Value } from "../src/capture/pyjson.ts";
import { configAfter, describeChain, hostVersions, montageFields, processing } from "../src/capture/record.ts";
import { readNpz } from "../src/capture/recording.ts";
import { PACKAGE_VERSION } from "../src/capture/version.ts";
import { FILL, decodeModelInterval, decodeNameParts, decodePipelineState, decodeStatus } from "../src/protocol.ts";
import type { PipelineState } from "../src/protocol.ts";
import { Session } from "../src/session.ts";

const FIXTURE = new URL("../../contract/recorder/sessions.json", import.meta.url);
const fixture = J.loads(readFileSync(FIXTURE, "utf8"));

function items(v: Value | undefined): Value[] {
  assert.ok(v !== undefined && v.kind === "array", "a list in the fixture");
  return v.items;
}

function text(v: Value): string {
  assert.equal(v.kind, "str");
  return (v as { value: string }).value;
}

function num(v: Value): number {
  const x = J.asF64(v);
  assert.ok(x !== null, "a number in the fixture");
  return x;
}

function bytes(v: Value): Uint8Array {
  const s = text(v);
  const out = new Uint8Array(s.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(s.slice(2 * i, 2 * i + 2), 16);
  return out;
}

/** An answer's payload: what follows its opcode and status. */
function payload(answer: Value): Uint8Array {
  return bytes(answer).subarray(2);
}

function hex(b: Uint8Array): string {
  let s = "";
  for (const x of b) s += x.toString(16).padStart(2, "0");
  return s;
}

function clocks(v: Value): { monotonic: number; realtime: number } {
  const [m, r] = items(v);
  return { monotonic: num(m!), realtime: num(r!) };
}

/** The library's manifest with the software it names put as this package names itself, so everything else is compared byte for byte. */
function asNamedHere(python: string): string {
  const start = python.indexOf('  "host_versions": {');
  assert.ok(start >= 0, "a manifest names its software");
  const close = python.indexOf("\n  }", start);
  assert.ok(close >= 0, "the block closes");
  const ours =
    `  "host_versions": {\n    "intomind_js": "${PACKAGE_VERSION}",\n` +
    `    "platform": "${process.platform}-${process.arch}"\n  }`;
  return python.slice(0, start) + ours + python.slice(close + "\n  }".length);
}

/** Where two texts first part, with a little of each. */
function firstDifference(a: string, b: string): string {
  let at = 0;
  while (at < a.length && at < b.length && a[at] === b[at]) at++;
  const from = Math.max(0, at - 120);
  return `written here: …${a.slice(from, from + 240)}…\nthe library's: …${b.slice(from, from + 240)}…`;
}

/** The arrays of an npz as the fixture lists them: name, numpy type, shape, the sha256 of the elements in C order, and whether the file stores them in Fortran order. */
function npzDigest(archive: Uint8Array): unknown[] {
  return readNpz(archive).map(([name, a]) => {
    let data = a.data;
    if (a.fortran && a.shape.length === 2) {
      const [rows, cols] = a.shape as [number, number];
      const keep = new Int32Array(rows * cols);
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) keep[r * cols + c] = c * rows + r;
      data = npy.take(a.data, keep);
    }
    return [name, npy.descr(a.data), a.shape, hex(sha256(npy.bytesOf(data))), a.fortran];
  });
}

test("the package names the version package.json states", () => {
  const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as { version: string };
  assert.equal(PACKAGE_VERSION, pkg.version);
});

test("every session records what the Python library records", () => {
  const sessions = items(J.get(fixture, "sessions"));
  assert.ok(sessions.length >= 19, String(sessions.length));
  const problems: string[] = [];
  let exchanges = 0;
  let saves = 0;
  for (const s of sessions) {
    const name = text(J.field(s, "name"));
    const want = J.field(s, "expected");
    const dir = mkdtempSync(join(tmpdir(), `intomind-recorder-${name}-`));
    const problem = (what: string): void => {
      problems.push(`${name}: ${what}`);
    };
    try {
      const session = new Session();
      // What the library's own Bluetooth adapter names itself, declared here
      // as an application declares its own.
      const adapter = J.fromPlain({ name: "intomind.ble", version: "1", first_party: true, transport: "bluetooth-le" });
      let site: Value | undefined;
      let syncs = 0;
      let linkOk = true;
      let config: Value | null = null;
      let proc: Value | null = null;
      for (const step of items(J.get(s, "script"))) {
        const op = text(J.field(step, "op"));
        switch (op) {
          case "info":
            session.onDeviceInfo(bytes(J.field(step, "bytes")));
            break;
          case "site":
            site = J.field(step, "site");
            break;
          case "sync": {
            const ticks = new Uint8Array(8);
            new DataView(ticks.buffer).setBigUint64(0, BigInt(J.pyStr(J.field(step, "ticks"))), true);
            session.onTimeSyncClocks(ticks, clocks(J.field(step, "before")), clocks(J.field(step, "after")));
            const mine = J.dumps(session.clock!.quality());
            const theirs = J.dumps(items(J.get(want, "clock"))[syncs]!);
            if (mine !== theirs) problem(`exchange ${syncs}: ${mine} differs from ${theirs}`);
            syncs++;
            exchanges++;
            break;
          }
          case "start":
            session.resetEpoch();
            session.startStream();
            session.onNotification(FILL.CONTROL_RESPONSE, Uint8Array.of(0x01, 0x00));
            break;
          case "reset_epoch":
            session.resetEpoch();
            break;
          case "arm":
            session.armRecorder();
            break;
          case "data":
            for (const p of items(J.get(step, "packets"))) session.onNotification(FILL.EEG_DATA, bytes(p));
            break;
          case "link_lost":
            linkOk = false;
            break;
          case "stop":
            session.stopStream();
            if (linkOk) session.onNotification(FILL.CONTROL_RESPONSE, Uint8Array.of(0x02, 0x00));
            break;
          case "disarm":
            session.disarmRecorder();
            break;
          case "config_after": {
            if (J.field(step, "link_ok").kind === "bool" && J.truthy(J.field(step, "link_ok"))) {
              const status = decodeStatus(bytes(J.field(step, "status")));
              const pipeline = session.can("pipeline") ? decodePipelineState(payload(J.field(step, "pipeline"))) : null;
              const indicator = session.can("indicator") ? payload(J.field(step, "indicator"))[0]! : null;
              const interval = session.can("model_cadence")
                ? decodeModelInterval(payload(J.field(step, "model_interval")))
                : null;
              const named = session.canHigh("device_name") ? decodeNameParts(payload(J.field(step, "name"))) : null;
              config = configAfter(status, session.leadoffSetting, pipeline, indicator, interval, named);
            } else {
              config = null;
            }
            const mine = config === null ? "null" : J.dumps(config);
            const theirs = J.dumps(J.field(want, "config_after"));
            if (mine !== theirs) problem(`config read back ${mine} differs from ${theirs}`);
            break;
          }
          case "processing": {
            const capable = session.can("pipeline");
            let state: PipelineState | null;
            try {
              state = decodePipelineState(payload(J.field(step, "pipeline")));
            } catch {
              state = null;
            }
            const rate = config === null ? null : J.asF64(J.field(config, "rate_sps"));
            proc = processing(capable, state, rate);
            const mine = proc === null ? "null" : J.dumps(proc);
            const theirs = J.dumps(J.field(want, "processing"));
            if (mine !== theirs) problem(`processing ${mine} differs from ${theirs}`);
            break;
          }
          case "save": {
            const label = text(J.field(step, "label"));
            const extra = J.copyObject(J.field(step, "extra"));
            if (J.truthy(J.field(step, "with_processing"))) {
              if (proc !== null) J.set(extra, "processing", proc);
              J.set(extra, "signal", proc !== null ? J.field(proc, "description") : J.str("unknown"));
            }
            const given = J.field(step, "config_after");
            const run = {
              extra,
              events: J.field(step, "events"),
              regsAfter: J.field(step, "regs_after"),
              configAfter: given.kind === "str" && given.value === "read" ? config : given,
              aborted: J.asStr(J.field(step, "aborted")),
              finishedUnix: num(items(J.get(step, "now"))[1]!),
            };
            const saved = session.saveRecording(label, site === undefined ? { adapter } : { adapter, site }, run);
            writeRecording(dir, saved);
            saves++;
            const read = (file: string): string | null =>
              existsSync(join(dir, file)) ? readFileSync(join(dir, file), "utf8") : null;
            const meta = read(`${label}.meta.json`);
            const theirs = asNamedHere(text(J.field(want, "meta_json")));
            if (meta === null) problem("no manifest");
            else if (meta !== theirs) problem(`the manifest differs:\n${firstDifference(meta, theirs)}`);
            if (saved.manifestJson !== meta) problem("the manifest handed back is not the one written");
            const events = read(`${label}.events.json`);
            const wantEvents = J.field(want, "events_json");
            if (events !== (wantEvents.kind === "null" ? null : text(wantEvents))) {
              problem(`events ${JSON.stringify(events)} differ from ${J.dumps(wantEvents)}`);
            }
            const log = read("SESSIONS.jsonl");
            if (log !== text(J.field(want, "sessions_jsonl"))) {
              problem(`session log ${JSON.stringify(log)} differs from ${J.dumps(J.field(want, "sessions_jsonl"))}`);
            }
            const sums = read(`${label}.sha256`);
            const names = sums === null ? null : sums.split("\n").filter((l) => l !== "").map((l) => l.split("  ")[1]);
            const wantNames = J.field(want, "sha256_names");
            const theirNames = wantNames.kind === "null" ? null : items(wantNames).map(text);
            if (JSON.stringify(names) !== JSON.stringify(theirNames)) {
              problem(`checksums for ${JSON.stringify(names)}, the library's for ${JSON.stringify(theirNames)}`);
            }
            for (const line of (sums ?? "").split("\n").filter((l) => l !== "")) {
              const [sum, file] = line.split("  ") as [string, string];
              if (hex(sha256(new Uint8Array(readFileSync(join(dir, file))))) !== sum) {
                problem(`${file}'s checksum is not its bytes'`);
              }
            }
            // The files handed back are the files written.
            for (const [file, b] of saved.files) {
              const on = new Uint8Array(readFileSync(join(dir, file)));
              if (hex(sha256(on)) !== hex(sha256(b))) problem(`${file} on disk is not the ${file} handed back`);
            }
            const files = readdirSync(dir)
              .sort()
              .map((f) => [f, (statSync(join(dir, f)).mode & 0o222) === 0]);
            const wantFiles = J.field(want, "files");
            assert.equal(wantFiles.kind, "object");
            const theirFiles =
              wantFiles.kind === "object"
                ? wantFiles.entries
                    .map(([f, v]) => [f, J.truthy(J.field(v, "read_only"))])
                    .sort((a, b) => ((a[0] as string) < (b[0] as string) ? -1 : 1))
                : [];
            if (JSON.stringify(files) !== JSON.stringify(theirFiles)) {
              problem(`files ${JSON.stringify(files)} differ from ${JSON.stringify(theirFiles)}`);
            }
            // And it reads back as every IntoMind program reads one.
            const v = openCollection(dir).verify(label);
            const aborted = J.asStr(J.field(step, "aborted"));
            if (aborted === null && !(v.ok && v.problems.length === 0)) problem(`does not verify: ${v.problems.join("; ")}`);
            if (aborted !== null && (v.ok || v.problems[0] !== `ABORTED (${aborted}): never finalized, do not analyze`)) {
              problem(`an aborted run verifies as ${JSON.stringify(v)}`);
            }
            // A recording of no samples stores its counts as one empty list,
            // which the Python library reads as no channels, and the reader
            // here, as the Rust crate's, refuses as not channels by samples.
            if (aborted === null && saved.finalized && J.asF64(J.field(saved.manifest, "n"))! > 0) {
              const back = openCollection(dir).loadAsRecorded(label);
              if (back.samples !== J.asF64(J.field(saved.manifest, "n"))) problem(`reads back ${back.samples} samples`);
            }
            const arrays = JSON.stringify(npzDigest(new Uint8Array(readFileSync(join(dir, `${label}.npz`)))));
            const theirArrays = JSON.stringify(J.toPlain(J.field(want, "arrays")));
            if (arrays !== theirArrays) problem(`arrays ${arrays} differ from ${theirArrays}`);
            break;
          }
          default:
            throw new Error(`no step ${op}`);
        }
      }
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }
  assert.equal(problems.length, 0, `${problems.length} problems:\n${problems.join("\n")}`);
  assert.equal(saves, sessions.length, "every session saves a recording");
  assert.equal(exchanges, sessions.reduce((n, s) => n + items(J.get(J.field(s, "expected"), "clock")).length, 0));
  assert.ok(exchanges >= 40, String(exchanges));
});

test("where the electrodes went is recorded as the library records it", () => {
  const cases = items(J.get(fixture, "montage"));
  assert.ok(cases.length >= 30, String(cases.length));
  const problems: string[] = [];
  for (const c of cases) {
    const name = text(J.field(c, "name"));
    const channels = J.field(c, "channels");
    let got: Value | null = null;
    let raised: string | null = null;
    try {
      got = montageFields(J.field(c, "site"), channels.kind === "null" ? null : num(channels), J.field(c, "layout"), J.field(c, "place"));
    } catch (e) {
      raised = (e as Error).message;
    }
    const want = J.get(c, "result");
    if (raised !== null) problems.push(`${name}: raised ${raised}, the library ${want !== undefined ? "did not" : "raised too"}`);
    else if (want === undefined) problems.push(`${name}: recorded where the library raised ${J.pyStr(J.field(c, "raises"))}`);
    else if (J.dumps(got!) !== J.dumps(want)) problems.push(`${name}: ${J.dumps(got!)} differs from ${J.dumps(want)}`);
  }
  assert.equal(problems.length, 0, problems.join("\n"));
});

test("a chain is described as the library describes it", () => {
  for (const c of items(J.get(fixture, "chains"))) {
    const stages = items(J.get(c, "stages")).map((s) => ({
      kind: num(J.field(s, "kind")),
      params: items(J.get(s, "params")).map(num),
    }));
    const rate = J.field(c, "rate");
    assert.equal(describeChain(stages, rate.kind === "null" ? null : num(rate)), text(J.field(c, "description")), J.dumps(c));
  }
});

test("a sum is math.fsum's, and a count is grouped as Python groups it", () => {
  assert.equal(fsum([1e-16, 1.0, 1e16]), 1.0000000000000002e16);
  assert.equal(fsum(Array(10).fill(0.1)), 1.0);
  assert.equal(fsum([1e100, 1.0, -1e100, 1e-100, 1e50, -1.0, -1e50]), 1e-100);
  assert.equal(fsum([]), 0.0);
  assert.throws(() => fsum([1.7e308, 1.7e308]), /intermediate overflow/);
  for (const [x, want] of [[2990.6, "2,991"], [1000.0, "1,000"], [999.5, "1,000"], [192.4, "192"], [1234567.0, "1,234,567"]] as const) {
    assert.equal(groupedFixed(x, 0), want);
  }
});

test("plain data is read as Python would hold it, and a value is kept as it is", () => {
  const v = J.fromPlain({ a: 1, b: 2.5, c: [true, null, "x"], d: 10n ** 30n, e: undefined, f: J.float(3) });
  assert.equal(J.dumps(v), '{"a": 1, "b": 2.5, "c": [true, null, "x"], "d": 1000000000000000000000000000000, "f": 3.0}');
  assert.equal(J.dumps(J.fromPlain(new Map([["z", 1], ["2", 2]]))), '{"z": 1, "2": 2}');
  assert.equal(J.dumps(J.fromPlain(2 ** 60)), "1.152921504606847e+18", "past what a double holds exactly, a float");
  const parsed = J.loads('{"settle": 3.0}');
  assert.equal(J.fromPlain(parsed), parsed);
  assert.equal(J.dumps(J.fromPlain({ kind: "str", value: "x" })), '{"kind": "str", "value": "x"}', "data shaped like a value is data");
  assert.throws(() => J.fromPlain(new Date(0)), TypeError);
});

test("both host clocks are read, and a recording names this package", () => {
  const before = hostClocks();
  const after = hostClocks();
  assert.ok(after.monotonic >= before.monotonic);
  assert.ok(Math.abs(before.realtime - Date.now() / 1000) < 60);
  assert.equal(J.dumps(hostVersions()), `{"intomind_js": "${PACKAGE_VERSION}", "platform": "${process.platform}-${process.arch}"}`);
});
