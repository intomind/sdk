/**
 * The recordings fixture: every export this package writes, held to what
 * the Python library wrote from the same synthetic recordings, and every
 * number and JSON document held to how Python formats it.
 *
 * `contract/recordings/make.py` emits the fixture from the Python library,
 * and the Rust crates are held to the same files. bdf, edf, csv and tsv must
 * match it byte for byte. npz must hold the same arrays, element for
 * element, and mat the same variables and attributes, read back here by
 * following the HDF5 structures from the superblock. Every summary must
 * carry the same keys and values, integers apart from floats, and every
 * refusal the same words.
 */

import assert from "node:assert/strict";
import { mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { sha256 } from "../src/digest.ts";
import { Collection, exportRecording } from "../src/capture/index.ts";
import { exportToFile, openCollection } from "../src/capture/node.ts";
import * as edf from "../src/capture/edf.ts";
import { roundHalfEven } from "../src/capture/edf.ts";
import { edflibRefusal } from "./edflib.ts";
import * as npy from "../src/capture/npy.ts";
import { bytesOf, descr } from "../src/capture/npy.ts";
import { fixed, general, quantize, repr } from "../src/capture/pyfmt.ts";
import { ParseError, dumpsIndented, dumpsSorted, get, loads, pyStr, reprStr } from "../src/capture/pyjson.ts";
import type { Value } from "../src/capture/pyjson.ts";
import { readNpz } from "../src/capture/recording.ts";
import * as zip from "../src/capture/zip.ts";

const ROOT = new URL("../../contract/recordings/", import.meta.url);
const CAPTURES = new URL("captures/", ROOT);

function fixture(name: string): Value {
  return loads(readFileSync(new URL(name, ROOT), "utf8"));
}

function items(v: Value | undefined): Value[] {
  assert.ok(v !== undefined && v.kind === "array", "a list in the fixture");
  return v.items;
}

function text(v: Value): string {
  assert.equal(v.kind, "str");
  return (v as { value: string }).value;
}

function intOf(v: Value): number {
  assert.equal(v.kind, "int");
  return Number((v as { value: string }).value);
}

function f64Of(hex: Value): number {
  const s = text(hex);
  const b = new Uint8Array(8);
  for (let i = 0; i < 8; i++) b[i] = parseInt(s.slice(2 * i, 2 * i + 2), 16);
  return new DataView(b.buffer).getFloat64(0, true);
}

function hex(b: Uint8Array): string {
  let s = "";
  for (const x of b) s += x.toString(16).padStart(2, "0");
  return s;
}

test("strings are quoted as Python quotes them", () => {
  const n = fixture("numbers.json");
  const rows = items(get(n, "strings"));
  assert.ok(rows.length > 2000, `${rows.length}`);
  for (const row of rows) {
    const [s, want] = items(row);
    assert.equal(reprStr(text(s!)), text(want!), JSON.stringify(text(s!)));
  }
});

test("floats are written as Python writes them", () => {
  const n = fixture("numbers.json");
  let checked = 0;
  for (const row of items(get(n, "repr"))) {
    const [x, want] = items(row);
    assert.equal(repr(f64Of(x!)), text(want!), `repr of ${f64Of(x!)}`);
    checked++;
  }
  for (const row of items(get(n, "fixed"))) {
    const [x, p, want] = items(row);
    assert.equal(fixed(f64Of(x!), intOf(p!)), text(want!), `${f64Of(x!)} to ${intOf(p!)} places`);
    checked++;
  }
  for (const row of items(get(n, "general"))) {
    const [x, p, want] = items(row);
    assert.equal(general(f64Of(x!), intOf(p!)), text(want!), `${f64Of(x!)} as .${intOf(p!)}g`);
    checked++;
  }
  for (const row of items(get(n, "quantize"))) {
    const [x, p, floor, ceiling] = items(row);
    const value = (v: Value): string | null => (v.kind === "null" ? null : text(v));
    assert.equal(quantize(f64Of(x!), intOf(p!), "floor"), value(floor!), `${f64Of(x!)} floor at ${intOf(p!)}`);
    assert.equal(quantize(f64Of(x!), intOf(p!), "ceiling"), value(ceiling!), `${f64Of(x!)} ceiling at ${intOf(p!)}`);
    checked++;
  }
  assert.ok(checked > 10_000, String(checked));
});

test("a value halfway between two integers is rounded to the even one, as numpy rounds it", () => {
  for (const [x, want] of [[0.5, 0], [1.5, 2], [2.5, 2], [-0.5, 0], [-1.5, -2], [-2.5, -2], [3.49, 3], [-3.51, -4]]) {
    assert.equal(roundHalfEven(x!) + 0, want, String(x));
  }
});

test("JSON reads and writes as Python's does", () => {
  const n = fixture("numbers.json");
  for (const row of items(get(n, "json"))) {
    const [input, sorted, indented] = items(row).map(text);
    const v = loads(input!);
    assert.equal(dumpsSorted(v), sorted);
    assert.equal(dumpsIndented(v, 2), indented);
  }
  for (const bad of ['{"a": 1,}', "[1 2]", "01", "1.", '"\u0001"', "{'a': 1}", "[1] x"]) {
    assert.throws(() => loads(bad), ParseError, bad);
  }
  assert.equal(pyStr(loads(`[null, true, 1.0, "it's", {"k": "v"}]`)), `[None, True, 1.0, "it's", {'k': 'v'}]`);
});

/** Every file of the fixture's recordings, by name, as a browser would hand them over. */
function fixtureFiles(): Map<string, Uint8Array> {
  const files = new Map<string, Uint8Array>();
  for (const name of readdirSync(CAPTURES)) files.set(name, new Uint8Array(readFileSync(new URL(name, CAPTURES))));
  return files;
}

/** Python's `repr` of a bytes object. */
function pyBytes(b: Uint8Array): string {
  const q = b.includes(0x27) && !b.includes(0x22) ? '"' : "'";
  let s = "b" + q;
  for (const c of b) {
    if (c === 0x5c) s += "\\\\";
    else if (c === 0x09) s += "\\t";
    else if (c === 0x0a) s += "\\n";
    else if (c === 0x0d) s += "\\r";
    else if (String.fromCharCode(c) === q) s += "\\" + q;
    else if (c >= 0x20 && c <= 0x7e) s += String.fromCharCode(c);
    else s += "\\x" + c.toString(16).padStart(2, "0");
  }
  return s + q;
}

/**
 * A MATLAB file as h5py reports it: the end of MATLAB's header, then each
 * variable's name, element type, shape, the sha256 of its elements and its
 * attributes, in the order h5py lists them.
 *
 * Read by following the HDF5 structures from the superblock, as the HDF5
 * specification lays them out, never from where this package's writer puts
 * them. It reads what a MATLAB export can hold: a version 0 superblock, a
 * symbol table group, version 1 object headers, and contiguous or compact
 * data. Anything else fails the test rather than being skipped.
 */
function matDigest(file: Uint8Array): [string, string[]] {
  const view = new DataView(file.buffer, file.byteOffset, file.byteLength);
  const u16 = (p: number): number => view.getUint16(p, true);
  const u32 = (p: number): number => view.getUint32(p, true);
  const u64 = (p: number): bigint => view.getBigUint64(p, true);
  const tail = hex(file.subarray(116, 128));
  // HDF5 looks for its superblock at 0, 512, 1024 and so on.
  const signature = [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];
  let sb = -1;
  for (let p = 0; p + 96 <= file.length; p += 512) {
    if (signature.every((c, k) => file[p + k] === c)) {
      sb = p;
      break;
    }
  }
  assert.ok(sb >= 0, "an HDF5 superblock");
  assert.equal(file[sb + 8], 0, "superblock version");
  assert.deepEqual([file[sb + 13], file[sb + 14]], [8, 8], "the size of addresses and lengths");
  const base = Number(u64(sb + 24));
  const at = (a: bigint): number => base + Number(a);
  // An object header's messages, as [type, body].
  const messages = (h: number): [number, Uint8Array][] => {
    assert.equal(file[h], 1, "object header version");
    const n = u16(h + 2);
    const size = u32(h + 8);
    const out: [number, Uint8Array][] = [];
    let p = h + 16;
    const end = h + 16 + size;
    while (out.length < n) {
      assert.ok(p + 8 <= end, "a message past the end of its header");
      const kind = u16(p);
      const len = u16(p + 2);
      assert.notEqual(kind, 0x0010, "a continuation, which a MATLAB export does not need");
      out.push([kind, file.subarray(p + 8, p + 8 + len)]);
      p += 8 + len;
    }
    return out;
  };
  // A datatype as numpy names it, and its size.
  const dtype = (t: Uint8Array): [string, number] => {
    const size = new DataView(t.buffer, t.byteOffset + 4, 4).getUint32(0, true);
    const order = size === 1 ? "|" : (t[1]! & 1) === 0 ? "<" : ">";
    switch (t[0]! & 0x0f) {
      case 0:
        return [`${order}${t[1]! & 0x08 ? "i" : "u"}${size}`, size];
      case 1:
        return [`${order}f${size}`, size];
      case 3:
        return [`|S${size}`, size];
      default:
        throw new Error(`datatype class ${t[0]! & 0x0f}`);
    }
  };
  const le64 = (b: Uint8Array, p: number): bigint => new DataView(b.buffer, b.byteOffset + p, 8).getBigUint64(0, true);
  // The root group's B-tree and local heap.
  const root = messages(at(u64(sb + 56 + 8)));
  const st = root.find(([k]) => k === 0x0011);
  assert.ok(st, "the root group's symbol table");
  const btree = le64(st[1], 0);
  const heap = le64(st[1], 8);
  const h = at(heap);
  assert.equal(String.fromCharCode(...file.subarray(h, h + 4)), "HEAP");
  const names = at(u64(h + 24));
  const name = (off: bigint): string => {
    const s = names + Number(off);
    let e = s;
    while (file[e] !== 0) e++;
    return new TextDecoder().decode(file.subarray(s, e));
  };
  // Every symbol node under the B-tree, in key order.
  const leaves = (node: bigint, out: bigint[]): void => {
    const p = at(node);
    assert.equal(String.fromCharCode(...file.subarray(p, p + 4)), "TREE");
    assert.equal(file[p + 4], 0, "a group B-tree");
    const level = file[p + 5]!;
    const used = u16(p + 6);
    for (let i = 0; i < used; i++) {
      const child = u64(p + 24 + 8 + 16 * i);
      if (level === 0) out.push(child);
      else leaves(child, out);
    }
  };
  const snods: bigint[] = [];
  leaves(btree, snods);
  const vars: string[] = [];
  for (const s of snods) {
    const p = at(s);
    assert.equal(String.fromCharCode(...file.subarray(p, p + 4)), "SNOD");
    for (let e = 0; e < u16(p + 6); e++) {
      const entry = p + 8 + 40 * e;
      const varName = name(u64(entry));
      const msgs = messages(at(u64(entry + 8)));
      const body = (kind: number): Uint8Array => {
        const m = msgs.find(([k]) => k === kind);
        assert.ok(m, "a dataset message");
        return m[1];
      };
      const space = body(0x0001);
      assert.equal(space[0], 1, "dataspace version");
      const shape: number[] = [];
      for (let i = 0; i < space[1]!; i++) shape.push(Number(le64(space, 8 + 8 * i)));
      const [dt] = dtype(body(0x0003));
      const layout = body(0x0008);
      assert.equal(layout[0], 3, "layout version");
      let data: Uint8Array;
      if (layout[1] === 0) {
        data = layout.subarray(4, 4 + (layout[2]! | (layout[3]! << 8)));
      } else if (layout[1] === 1) {
        const a = le64(layout, 2);
        const n = Number(le64(layout, 10));
        data = a === 0xffffffffffffffffn ? new Uint8Array(0) : file.subarray(at(a), at(a) + n);
      } else {
        throw new Error(`layout class ${layout[1]}`);
      }
      const attrs = msgs
        .filter(([k]) => k === 0x000c)
        .map(([, a]) => {
          assert.equal(a[0], 1, "attribute version");
          const nlen = a[2]! | (a[3]! << 8);
          const tlen = a[4]! | (a[5]! << 8);
          const slen = a[6]! | (a[7]! << 8);
          const pad = (n: number): number => Math.ceil(n / 8) * 8;
          const aname = new TextDecoder().decode(a.subarray(8, 8 + nlen - 1));
          const t = a.subarray(8 + pad(nlen), 8 + pad(nlen) + tlen);
          const sp = a.subarray(8 + pad(nlen) + pad(tlen));
          assert.equal(sp[1], 0, "a scalar attribute");
          const [adt, size] = dtype(t);
          const v = a.subarray(8 + pad(nlen) + pad(tlen) + pad(slen)).subarray(0, size);
          let value: string;
          if ((t[0]! & 0x0f) === 3) {
            // numpy drops a fixed string's trailing nulls.
            let end = v.length;
            while (end > 0 && v[end - 1] === 0) end--;
            value = pyBytes(v.subarray(0, end));
          } else if ((t[0]! & 0x0f) === 0) {
            const b = new Uint8Array(8);
            b.set(v);
            const x = new DataView(b.buffer).getBigUint64(0, true);
            value = (t[1]! & 0x08 ? BigInt.asIntN(size * 8, x) : x).toString();
          } else {
            throw new Error(`an attribute of datatype class ${t[0]! & 0x0f}`);
          }
          return `${aname} ${adt} ${value}`;
        })
        .sort();
      vars.push(`${varName} ${dt} ${JSON.stringify(shape)} ${hex(sha256(data))} ${JSON.stringify(attrs)}`);
    }
  }
  return [tail, vars];
}

test("every export is the Python library's", () => {
  const expected = fixture("expected.json");
  const recordings = get(expected, "recordings");
  assert.ok(recordings !== undefined && recordings.kind === "object", "no recordings");
  const collection = new Collection(fixtureFiles(), { dir: "captures" });
  const problems: string[] = [];
  let checked = 0;
  for (const [label, byFormat] of recordings.entries) {
    assert.equal(byFormat.kind, "object");
    if (byFormat.kind !== "object") continue;
    for (const [fmt, want] of byFormat.entries) {
      checked++;
      const refused = get(want, "refused");
      let got: ReturnType<typeof exportRecording> | null = null;
      let error: string | null = null;
      try {
        got = exportRecording(collection, label, fmt);
      } catch (e) {
        error = (e as Error).message;
      }
      if (refused !== undefined) {
        if (error === null) problems.push(`${label} ${fmt}: exported where Python refused: ${pyStr(refused)}`);
        else if (error !== pyStr(refused)) problems.push(`${label} ${fmt}: refused as ${error}, Python said ${pyStr(refused)}`);
        continue;
      }
      if (got === null) {
        problems.push(`${label} ${fmt}: refused where Python exported: ${error}`);
        continue;
      }
      const theirs = dumpsSorted(get(want, "summary")!);
      if (got.summaryJson !== theirs) problems.push(`${label} ${fmt}: summary ${got.summaryJson} differs from ${theirs}`);
      const sha = get(want, "sha256");
      if (sha !== undefined && hex(sha256(got.bytes)) !== text(sha)) {
        problems.push(`${label} ${fmt}: the file is not the bytes Python wrote`);
      }
      const arrays = get(want, "arrays");
      if (arrays !== undefined) {
        const mine = readNpz(got.bytes).map(([k, a]) => {
          const raw = bytesOf(a.data);
          return `${k} ${descr(a.data)} ${JSON.stringify(a.shape)} ${hex(sha256(raw))}`;
        });
        const want2 = items(arrays).map((r) => {
          const [k, d, shape, s] = items(r);
          return `${text(k!)} ${text(d!)} ${JSON.stringify(items(shape).map(intOf))} ${text(s!)}`;
        });
        if (JSON.stringify(mine) !== JSON.stringify(want2)) {
          problems.push(`${label} ${fmt}: arrays ${JSON.stringify(mine)} differ from ${JSON.stringify(want2)}`);
        }
      }
      const mat = get(want, "mat");
      if (mat !== undefined) {
        const mine = matDigest(got.bytes);
        const theirs2: [string, string[]] = [
          text(get(mat, "header_tail")!),
          items(get(mat, "variables")).map((v) => {
            const [k, d, shape, s, attrs] = items(v);
            const a = items(attrs).map((x) => items(x).map(text).join(" "));
            return `${text(k!)} ${text(d!)} ${JSON.stringify(items(shape).map(intOf))} ${text(s!)} ${JSON.stringify(a)}`;
          }),
        ];
        if (JSON.stringify(mine) !== JSON.stringify(theirs2)) {
          problems.push(`${label} ${fmt}: ${JSON.stringify(mine)} differs from ${JSON.stringify(theirs2)}`);
        }
      }
    }
  }
  assert.ok(checked >= 200, String(checked));
  assert.equal(problems.length, 0, `${problems.length} of ${checked} differ:\n${problems.join("\n")}`);
});

test("a recording's lead-off is its bytes where its words are bytes, never a word cut down to one, and every word at its full width", () => {
  const collection = new Collection(fixtureFiles(), { dir: "captures" });
  // A recording of the device: bytes, the first electrode off from sample 120
  // to 179 and the third from 300 to 329, and the same at full width.
  const bytes = collection.load("clean-v7");
  const want = new Uint8Array(600);
  for (let k = 120; k < 180; k++) want[k]! |= 0x01;
  for (let k = 300; k < 330; k++) want[k]! |= 0x04;
  assert.deepEqual(bytes.leadoff, want);
  assert.deepEqual(bytes.leadoffWords, BigUint64Array.from(want, (b) => BigInt(b)));
});

/**
 * The files of a recording of 1500 samples on four channels at 500 a second
 * of a device that cannot check contact, holding no `leadoff_judged`:
 * lead-off detection on throughout, or where `active` says, the first two
 * electrodes reading off while it was, with the manifest `extra` gives.
 */
function oldRecording(files: Map<string, Uint8Array>, label: string, extra: Record<string, unknown>, active: boolean[]): void {
  const n = 1500;
  const counts = new BigInt64Array(4 * n);
  for (let c = 0; c < 4; c++) for (let k = 0; k < n; k++) counts[c * n + k] = BigInt(((k * (c + 3)) % 97) - 48) * 1000n;
  const ticks = BigInt64Array.from({ length: n }, (_, k) => BigInt(k) * 2000n);
  const members: [string, npy.NpyArray][] = [
    ["counts", { shape: [4, n], fortran: false, data: { kind: "i8", values: counts } }],
    ["index", npy.vector({ kind: "i8", values: BigInt64Array.from({ length: n }, (_, k) => BigInt(k)) })],
    ["device_ticks", npy.vector({ kind: "i8", values: ticks })],
    ["host_time", npy.vector({ kind: "f8", values: Float64Array.from(ticks, (t) => 1_767_225_600 + Number(t) / 1e6) })],
    ["leadoff", npy.vector({ kind: "u1", values: Uint8Array.from(active, (a) => (a ? 0x03 : 0x00)) })],
    ["gap_before", npy.vector({ kind: "b1", values: new Uint8Array(n) })],
    ["leadoff_active", npy.vector({ kind: "b1", values: Uint8Array.from(active, Number) })],
  ];
  files.set(`${label}.npz`, zip.write(members.map(([k, a]) => [`${k}.npy`, npy.write(a)]), 1_767_225_600));
  const meta = {
    label, manifest_version: 7, started_utc: "2026-01-01T00:00:00Z", fs_effective: 500.0, rate_sps: 500, vref_uv: 4_500_000.0,
    gain: 24, adc_bits: 24, tick_hz: 1_000_000, channels: 4, run: { name: "together" }, ...extra,
  };
  files.set(`${label}.meta.json`, new TextEncoder().encode(JSON.stringify(meta)));
}

test("an old recording of firmware that cannot check contact leaves contact empty in every format", () => {
  // An IntoMind device whose firmware is older than 1.4.6 cannot check
  // electrode contact: its lead-off detection does not work, and every
  // electrode reads off whether it touches skin or not. A recording of one
  // that holds no leadoff_judged has its contact left empty by every format:
  // every summary says "not checked", the text formats have no lead-off
  // column, EDF+ and BDF+ hold no lead-off label, and the archives carry
  // leadoff_judged, false for every sample taken with detection on, beside
  // the lead-off bits as they were recorded. Recordings made together are
  // each read by their own manifest: the device on firmware 1.4.6 keeps every
  // electrode off as recorded.
  const files = new Map<string, Uint8Array>();
  const every = new Array<boolean>(1500).fill(true);
  const half = Array.from({ length: 1500 }, (_, k) => k >= 750);
  const empty: [string, boolean[]][] = [["t_fw_old_device", every], ["t_fw_old_half", half]];
  oldRecording(files, "t_fw_old_device", { firmware_version: "1.4.5" }, every);
  oldRecording(files, "t_fw_old_half", { firmware_version: "1.4.5" }, half);
  oldRecording(files, "t_fw_new_device", { firmware_version: "1.4.6" }, every);
  const rec = new Collection(files);
  const latin1 = new TextDecoder("latin1");
  const column = (bytes: Uint8Array, d: string): string[] => new TextDecoder().decode(bytes).split("\n")[0]!.split(d);
  for (const [label, active] of empty) {
    for (const fmt of ["npz", "edf", "bdf", "csv", "tsv", "mat"]) {
      const got = exportRecording(rec, label, fmt);
      assert.equal(got.summary["contact"], "not checked", `${label} ${fmt}`);
      if (fmt === "csv" || fmt === "tsv") {
        assert.deepEqual(column(got.bytes, fmt === "csv" ? "," : "\t").filter((h) => h.includes("leadoff")), [], `${label} ${fmt}`);
      }
      if (fmt === "edf" || fmt === "bdf") assert.ok(!latin1.decode(got.bytes).includes("LEADOFF"), `${label} ${fmt}`);
      if (fmt === "npz") {
        const arrays = new Map(readNpz(got.bytes));
        const judged = arrays.get("leadoff_judged")!;
        assert.equal(descr(judged.data), "|b1", label);
        assert.deepEqual(Array.from(judged.data.values as Uint8Array), active.map((a) => (a ? 0 : 1)), label);
        assert.deepEqual(Array.from(arrays.get("leadoff")!.data.values as Uint8Array), active.map((a) => (a ? 3 : 0)), label);
      }
      if (fmt === "mat") {
        const judged = matDigest(got.bytes)[1].find((v) => v.startsWith("leadoff_judged "));
        assert.ok(judged !== undefined && judged.startsWith("leadoff_judged |u1 [1,1500] ") && judged.includes("b'logical'"), label);
      }
    }
  }
  for (const label of ["t_fw_new_device"]) {
    for (const fmt of ["npz", "edf", "bdf", "csv", "tsv", "mat"]) {
      assert.equal(exportRecording(rec, label, fmt).summary["contact"], "checked", `${label} ${fmt}`);
    }
    const lines = new TextDecoder().decode(exportRecording(rec, label, "csv").bytes).trimEnd().split("\n").map((l) => l.split(","));
    const [c1, c2] = [lines[0]!.indexOf("ch1_leadoff"), lines[0]!.indexOf("ch2_leadoff")];
    assert.deepEqual([...new Set(lines.slice(1).flatMap((r) => [r[c1]!, r[c2]!]))], ["1"], label);
    for (const fmt of ["edf", "bdf"]) {
      const written = latin1.decode(exportRecording(rec, label, fmt).bytes);
      const labels = [...new Set(written.match(/BAD_LEADOFF ch\d+/g) ?? [])].sort();
      assert.deepEqual(labels, ["BAD_LEADOFF ch1", "BAD_LEADOFF ch2"], `${label} ${fmt}`);
    }
    assert.ok(!readNpz(exportRecording(rec, label, "npz").bytes).some(([k]) => k === "leadoff_judged"), label);
  }
});

/**
 * The files of a recording of `n` samples on two channels at `fs` a second,
 * numbered by `index`, its values a DC-coupled channel drifting under a small
 * rhythm, as the library's export tests make one, with the manifest `extra`
 * gives over a device named "fake instrument", a revision of "unknown" and
 * the electrode "fixture electrode", and `events` as its markers.
 */
function plainRecording(
  files: Map<string, Uint8Array>,
  label: string,
  extra: Record<string, unknown> = {},
  opts: { readonly n?: number; readonly fs?: number; readonly index?: number[]; readonly events?: unknown[] } = {},
): void {
  const fs = opts.fs ?? 500;
  const index = opts.index ?? Array.from({ length: opts.n ?? 1500 }, (_, k) => k);
  const n = index.length;
  const lsb = (2 * 2_400_000) / (12 * (1 << 24));
  const counts = new BigInt64Array(2 * n);
  for (let c = 0; c < 2; c++) {
    for (let k = 0; k < n; k++) {
      const uv = 1500 * (c + 1) + (n > 1 ? (8000 * k) / (n - 1) : 0) + 20 * Math.sin((2 * Math.PI * 10 * k) / fs);
      counts[c * n + k] = BigInt(Math.round(uv / lsb));
    }
  }
  const idx = BigInt64Array.from(index, (k) => BigInt(k));
  const members: [string, npy.NpyArray][] = [
    ["counts", { shape: [2, n], fortran: false, data: { kind: "i8", values: counts } }],
    ["index", npy.vector({ kind: "i8", values: idx })],
    ["device_ticks", npy.vector({ kind: "i8", values: idx.slice() })],
    ["host_time", npy.vector({ kind: "f8", values: Float64Array.from(index, (k) => k / fs) })],
  ];
  files.set(`${label}.npz`, zip.write(members.map(([k, a]) => [`${k}.npy`, npy.write(a)]), 1_767_225_600));
  const meta = {
    label, n, fs_effective: fs, vref_uv: 2_400_000, gain: 12, adc_bits: 24, device_name: "fake instrument",
    board_revision: "unknown", montage: { electrode: "fixture electrode" }, ...extra,
  };
  files.set(`${label}.meta.json`, new TextEncoder().encode(JSON.stringify(meta)));
  if (opts.events !== undefined) files.set(`${label}.events.json`, new TextEncoder().encode(JSON.stringify(opts.events)));
}

/** The fixed-width header of an EDF+ or BDF+ file, at the offsets the standard puts it, each field without its padding. */
function edfHeader(b: Uint8Array): { recording: string; startdate: string; starttime: string; nRecords: number; duration: string; labels: string[]; transducer: string[]; prefilter: string[] } {
  const f = (o: number, n: number): string => new TextDecoder("latin1").decode(b.subarray(o, o + n)).trim();
  const ns = Number(f(252, 4));
  const each = (o: number, w: number): string[] => Array.from({ length: ns }, (_, i) => f(256 + o * ns + i * w, w));
  return {
    recording: f(88, 80),
    startdate: f(168, 8),
    starttime: f(176, 8),
    nRecords: Number(f(236, 8)),
    duration: f(244, 8),
    labels: each(0, 16),
    transducer: each(16, 80),
    prefilter: each(136, 80),
  };
}

const KNOWN_START = { started_utc: "2026-11-25T10:20:30Z" };
const GENERATED = { synthetic: true, signal_source: "synthetic" };

/** One file of the library's writer at both widths: a ramp on one channel, 1200 samples at 500 a second, from 2026-11-25 10:20:30. */
function written(w: edf.Width, over: Partial<edf.Input> = {}): Uint8Array {
  const input: edf.Input = {
    signals: [["ch1", Float64Array.from({ length: 1200 }, (_, k) => -50 + (100 * k) / 1199)]],
    fs: 500,
    start: [2026, 11, 25, 10, 20, 30],
    index: BigInt64Array.from({ length: 1200 }, (_, k) => BigInt(k)),
    events: [],
    patient: "X X X X",
    recording: "",
    prefilter: "",
    transducer: "",
    stretches: [],
    ...over,
  };
  return edf.write(input, w)[0];
}

test("every EDF+ and BDF+ header holds the rules EDFlib opens a file by, whose recording field begins Startdate, the start date the header holds and the administration, technician and equipment codes, and so does the writer's own field", () => {
  // EDFlib, the library EDFbrowser and pyedflib read files with, refuses an
  // EDF+ or BDF+ file whose recording field does not begin with Startdate,
  // the start date the header holds, and the three codes. Every export's
  // header holds every rule of EDFlib's header check, for a start that is
  // known, one that is not, a generated signal and a device not named, and
  // so does the header the writer writes when it is given no recording field.
  const files = new Map<string, Uint8Array>();
  const cases: [string, Record<string, unknown>][] = [
    ["known", KNOWN_START],
    ["unknown", {}],
    ["generated", { ...KNOWN_START, ...GENERATED }],
    ["generated_unknown", GENERATED],
    ["no_device", { ...KNOWN_START, device_name: null, device_id: null }],
  ];
  for (const [name, extra] of cases) plainRecording(files, `t_${name}`, extra);
  const rec = new Collection(files);
  for (const [name] of cases) {
    for (const fmt of ["edf", "bdf"]) {
      const b = exportRecording(rec, `t_${name}`, fmt).bytes;
      assert.equal(edflibRefusal(b, true), null, `${name} ${fmt}`);
    }
  }
  const none = edfHeader(exportRecording(rec, "t_no_device", "edf").bytes).recording;
  assert.ok(none.startsWith("Startdate 25-NOV-2026 X X X 16-bit lossy, unknown;"), none);
  for (const w of [edf.EDF, edf.BDF]) {
    const b = written(w);
    assert.equal(edfHeader(b).recording, "Startdate 25-NOV-2026 X X X", w.family);
    assert.equal(edflibRefusal(b, true), null, w.family);
  }
});

test("a start date in a recording field names its month in English capitals, DD-MMM-YYYY, whatever language the host is set to", () => {
  const months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];
  months.forEach((month, m) => {
    const b = written(edf.EDF, { start: [987, m + 1, 7, 1, 2, 3] });
    assert.equal(edfHeader(b).recording, `Startdate 07-${month}-0987 X X X`);
  });
});

test("a generated signal and a lossy file say so inside the 80 characters of the recording field, whatever the device", () => {
  // What a reader must not lose comes before anything a cut can take: that a
  // generated signal is generated, and the width with its lossy note.
  // However long the device's name, it is cut to leave them room, so only the
  // revision and the raw record's name can be cut away.
  const files = new Map<string, Uint8Array>();
  const name = "a device whose name runs on for far longer than any field could hold";
  const cases: [string, string | null, boolean][] = [];
  for (const started of ["2026-11-25T10:20:30Z", null]) {
    for (const synthetic of [true, false]) {
      const label = `t_eighty_${started === null ? "unknown" : "known"}_${synthetic}`;
      plainRecording(files, label, { started_utc: started, synthetic, device_name: name, board_revision: "fixture board" });
      cases.push([label, started, synthetic]);
    }
  }
  const rec = new Collection(files);
  for (const [label, started, synthetic] of cases) {
    const b = exportRecording(rec, label, "edf").bytes;
    const field = edfHeader(b).recording;
    assert.ok(field.length === 80 && field.includes("16-bit lossy"), field);
    assert.equal(field.includes("SYNTHETIC, not measured: "), synthetic, field);
    assert.ok(field.startsWith(started === null ? "Startdate X X X a_device_whose_name_runs_o" : "Startdate 25-NOV-2026 X X a_device_whose_n"), field);
    assert.equal(edflibRefusal(b, true), null, label);
  }
});

test("an EDF+ or BDF+ export of a recording whose start cannot be read begins its recording field Startdate X X X and the device as the equipment, and holds 01.01.00 and 00.00.00 as its start date and time, as the library writes it", () => {
  // A start that is not known, none given, null, or a time not written as the
  // library writes one, is the recording field's start date, and the two
  // subfields after it, each an X, as EDF+ writes a subfield that is not
  // known, then the device as the equipment, its name with an underscore for
  // each space. The start date and time fields must hold a date, and hold one
  // the X says is not the start. A start that is known keeps its date and
  // time, and the recording field gives it as EDF+ does.
  const files = new Map<string, Uint8Array>();
  const unknown: Record<string, unknown>[] = [{}, { started_utc: null }, { started_utc: "not a time" }, { started_utc: "2026-11-25T10:20:30" }];
  unknown.forEach((extra, k) => plainRecording(files, `t_start_${k}`, extra));
  plainRecording(files, "t_start_known", KNOWN_START);
  const rec = new Collection(files);
  for (const [fmt, bits] of [["edf", "16-bit lossy"], ["bdf", "24-bit"]] as const) {
    unknown.forEach((extra, k) => {
      const h = edfHeader(exportRecording(rec, `t_start_${k}`, fmt).bytes);
      assert.equal(h.recording, `Startdate X X X fake_instrument ${bits}, unknown; raw record is t_start_${k}.npz`.slice(0, 80), `${fmt} ${JSON.stringify(extra)}`);
      assert.deepEqual([h.startdate, h.starttime], ["01.01.00", "00.00.00"], `${fmt} ${JSON.stringify(extra)}`);
    });
    const h = edfHeader(exportRecording(rec, "t_start_known", fmt).bytes);
    assert.equal(h.recording, `Startdate 25-NOV-2026 X X fake_instrument ${bits}, unknown; raw record is t_start_known.npz`.slice(0, 80), fmt);
    assert.deepEqual([h.startdate, h.starttime], ["25.11.26", "10.20.30"], fmt);
  }
});

test("an export whose start is not known names its device as the equipment, as one whose start is known, and lays the library's words out alike", () => {
  // Both give the device as the equipment code, and the library's words
  // after it laid out alike, for a measured and a generated signal, in EDF+
  // and BDF+, and only the start date differs.
  const files = new Map<string, Uint8Array>();
  for (const [g, generated] of [["measured", {}], ["generated", GENERATED]] as const) {
    plainRecording(files, `t_eq_${g}_known`, { ...generated, ...KNOWN_START });
    plainRecording(files, `t_eq_${g}_unknown`, { ...generated, started_utc: null });
  }
  const rec = new Collection(files);
  for (const g of ["measured", "generated"]) {
    for (const fmt of ["edf", "bdf"]) {
      const fields = ["known", "unknown"].map((s) => {
        const b = exportRecording(rec, `t_eq_${g}_${s}`, fmt).bytes;
        const field = edfHeader(b).recording;
        const parts = field.split(" ");
        assert.ok(parts[0] === "Startdate" && parts.slice(2, 5).join(" ") === "X X fake_instrument", `${s} ${fmt} ${field}`);
        assert.equal(edflibRefusal(b, true), null, `${s} ${fmt}`);
        return field;
      });
      const [known, unknown] = fields as [string, string];
      assert.ok(known.startsWith("Startdate 25-NOV-2026 ") && unknown.startsWith("Startdate X "), `${known} | ${unknown}`);
      // The known start's date is ten characters longer than X, so its field
      // is cut ten characters sooner.
      const after = known.slice("Startdate 25-NOV-2026 ".length);
      assert.ok(unknown.slice("Startdate X ".length).startsWith(after), `${known} | ${unknown}`);
      assert.equal(after.includes("SYNTHETIC, not measured: "), g === "generated", after);
    }
  }
});

test("the equipment code writes each run of white space in the device's name as one underscore, white space as Python's str.split finds it", () => {
  // Python's split takes the separators of files, groups, records and units,
  // and next line, as white space, and not a byte order mark or a zero width
  // space, which are written as question marks, as a character ASCII cannot
  // hold is.
  const files = new Map<string, Uint8Array>();
  plainRecording(files, "t_spaces", { ...KNOWN_START, device_name: " a\x1cb\x85c﻿d　　e f​g\t" });
  const rec = new Collection(files);
  for (const fmt of ["edf", "bdf"]) {
    const b = exportRecording(rec, "t_spaces", fmt).bytes;
    assert.equal(edfHeader(b).recording.split(" ")[4], "a_b_c?d_e_f?g", fmt);
    assert.equal(edflibRefusal(b, true), null, fmt);
  }
});

test("a board revision of null, or text with nothing in it, is written as for a recording that states none", () => {
  // A manifest whose board revision is null, or text with nothing in it,
  // white space as Python's strip finds it included, states no revision, and
  // the recording field says unknown, as for a recording that states none,
  // for a known start and an unknown one, in EDF+ and BDF+.
  const files = new Map<string, Uint8Array>();
  const revisions: [string, unknown][] = [["absent", undefined], ["null", null], ["empty", ""], ["spaces", "  "], ["python_spaces", "\t\x1c　"]];
  for (const started of ["2026-11-25T10:20:30Z", null]) {
    for (const [name, rev] of revisions) plainRecording(files, `t_rev_${name}_${started === null}`, { started_utc: started, board_revision: rev });
  }
  const rec = new Collection(files);
  for (const started of [false, true]) {
    for (const fmt of ["edf", "bdf"]) {
      const fields = revisions.map(([name]) => edfHeader(exportRecording(rec, `t_rev_${name}_${started}`, fmt).bytes).recording.split("raw record is ")[0]!);
      assert.ok(fields[0]!.endsWith(", unknown; "), fields[0]);
      assert.deepEqual(new Set(fields).size, 1, fields.join(" | "));
    }
  }
});

/** Control characters: NUL, the bell, tab, line feed, carriage return, the two separators of an annotation list, escape, delete, and next line. */
const CONTROLS = "\x00\x07\t\n\r\x14\x15\x1b\x7f\x85";

test("a control character in any text an export writes is a question mark, and the file opens", () => {
  // EDFlib refuses a file whose header holds a character that is not
  // printable or whose annotation holds a control character, and a control
  // character passed into both from the text a recording carries: its
  // device's name or id, its board revision, its electrode, its processing
  // description or signal word, and an event's text. Each is written as a
  // question mark, as a character the format cannot hold already is, in EDF+
  // and BDF+, for a known start and an unknown one, and every file passes the
  // whole of EDFlib's check. In the device's name, as in the equipment code it
  // becomes, each run of white space is one underscore, and a control
  // character that is white space, as a tab or a line feed is, is part of a
  // run. The writer writes a character UTF-8 cannot hold, half of a pair of
  // surrogates, as a question mark in an annotation, as the library does.
  const files = new Map<string, Uint8Array>();
  const q = "?".repeat(CONTROLS.length);
  const events = [{ host_time: 0.5, cond: "cue" + CONTROLS }, { host_time: 1.0, kind: "marker" + CONTROLS }];
  const cases: [string, Record<string, unknown>, string][] = [
    ["named", { device_name: "dev" + CONTROLS + "ice", board_revision: "rev" + CONTROLS, montage: { electrode: "electrode" + CONTROLS }, processing: { description: "chain" + CONTROLS } }, "chain" + q],
    ["by_id", { device_name: null, device_id: "dev" + CONTROLS + "ice", signal: "notch" + CONTROLS }, "notch" + q],
  ];
  for (const [name, extra] of cases) {
    for (const started of ["2026-11-25T10:20:30Z", null]) plainRecording(files, `t_controls_${name}_${started === null}`, { ...extra, started_utc: started }, { events });
  }
  const rec = new Collection(files);
  const holds = (b: Uint8Array, text: string): boolean => Buffer.from(b).includes(Buffer.from(text, "utf8"));
  for (const [name, , prefilter] of cases) {
    for (const started of [false, true]) {
      for (const fmt of ["edf", "bdf"]) {
        const where = `${name} ${started} ${fmt}`;
        const b = exportRecording(rec, `t_controls_${name}_${started}`, fmt).bytes;
        assert.equal(edflibRefusal(b), null, where);
        const h = edfHeader(b);
        assert.equal(h.recording.split(" ")[4], "dev??_????_ice", `${where} ${h.recording}`);
        if (name === "named") {
          assert.ok(h.recording.includes(`, rev${q}; raw record`), h.recording);
          assert.deepEqual(h.transducer.slice(0, 2), ["electrode" + q, "electrode" + q], where);
        }
        assert.deepEqual(h.prefilter.slice(0, -1), [prefilter, prefilter], where);
        assert.ok(holds(b, `\x14cue${q}\x14`) && holds(b, `\x14marker${q}\x14`), where);
      }
    }
  }
  // The writer itself, given a control character in each text it writes.
  for (const w of [edf.EDF, edf.BDF]) {
    const b = written(w, {
      signals: [["ch" + CONTROLS, Float64Array.from({ length: 1200 }, (_, k) => -50 + (100 * k) / 1199)]],
      recording: "Startdate 25-NOV-2026 X X dev" + CONTROLS + "ice",
      prefilter: "chain" + CONTROLS,
      transducer: "electrode" + CONTROLS,
      events: [{ t: 0.5, label: "cue" + CONTROLS }, { t: 1.5, label: "lone \ud834 half" }],
      stretches: [["BAD" + CONTROLS, Array.from({ length: 1200 }, (_, k) => k < 10)]],
    });
    assert.equal(edflibRefusal(b), null, w.family);
    const h = edfHeader(b);
    assert.ok(h.labels[0] === "ch" + q && h.transducer[0] === "electrode" + q, JSON.stringify(h));
    assert.equal(h.recording, `Startdate 25-NOV-2026 X X dev${q}ice`);
    assert.ok(holds(b, `\x14cue${q}\x14`) && holds(b, `\x14BAD${q}\x14`) && holds(b, "\x14lone ? half\x14"), w.family);
  }
});

test("every data record of an EDF+ or BDF+ export begins the record duration the header states after the one before it", () => {
  // EDF+C requires each data record to begin where the one before it ends,
  // and EDFlib refuses a file whose record onsets are not exactly the
  // header's record duration apart. At a rate that is not a whole number the
  // header holds the duration rounded to its eight characters, and the
  // onsets followed the rate instead, so at 498.8996 samples a second a
  // record began a microsecond late by the header's count within a few
  // records. Every export of such a recording passes the whole of EDFlib's
  // check, for a known start, an unknown start and a generated signal, with a
  // hole in it too.
  const files = new Map<string, Uint8Array>();
  const holed = [...Array.from({ length: 2500 }, (_, k) => k), ...Array.from({ length: 2500 }, (_, k) => k + 2600)];
  const cases: [string, Record<string, unknown>][] = [["known", KNOWN_START], ["unknown", {}], ["generated", { ...KNOWN_START, ...GENERATED }]];
  for (const [name, extra] of cases) {
    plainRecording(files, `t_onsets_${name}`, { ...extra, fs_effective: 498.8996 }, { n: 5000, fs: 498.8996 });
    plainRecording(files, `t_onsets_${name}_holed`, { ...extra, fs_effective: 498.8996 }, { index: holed, fs: 498.8996 });
  }
  const rec = new Collection(files);
  for (const [name] of cases) {
    for (const label of [`t_onsets_${name}`, `t_onsets_${name}_holed`]) {
      for (const fmt of ["edf", "bdf"]) {
        const b = exportRecording(rec, label, fmt).bytes;
        const h = edfHeader(b);
        assert.ok(h.duration === "1.000201" && h.nRecords > 3, `${label} ${fmt} ${JSON.stringify(h)}`);
        assert.equal(edflibRefusal(b), null, `${label} ${fmt}`);
      }
    }
  }
});

test("a folder on disk exports where the Python library would put the file", () => {
  const work = mkdtempSync(join(tmpdir(), "intomind-capture-"));
  try {
    const dir = new URL(CAPTURES).pathname;
    const s = exportToFile(dir, "clean-v7", "csv", join(work, "out.csv"));
    assert.equal(s["path"], join(work, "out.csv"));
    assert.deepEqual(Object.keys(s).slice(0, 3), ["label", "path", "format"]);
    const again = exportRecording(openCollection(dir), "clean-v7", "csv");
    assert.deepEqual(new Uint8Array(readFileSync(join(work, "out.csv"))), again.bytes);
    const into = exportToFile(dir, "clean-v7", "bdf", work);
    assert.equal(into["path"], join(work, "clean-v7.bdf"));
    assert.throws(() => exportToFile(dir, "../clean-v7", "csv", work), /is not a capture label/);
    assert.throws(() => exportToFile(dir, "clean-v7", "xls", work), /unknown export format "xls"/);
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
});
