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
import { roundHalfEven } from "../src/capture/edf.ts";
import { bytesOf, descr } from "../src/capture/npy.ts";
import { fixed, general, quantize, repr } from "../src/capture/pyfmt.ts";
import { ParseError, dumpsIndented, dumpsSorted, get, loads, pyStr } from "../src/capture/pyjson.ts";
import type { Value } from "../src/capture/pyjson.ts";
import { readNpz } from "../src/capture/recording.ts";

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
    assert.throws(() => exportToFile(dir, "clean-v7", "xls", work), /unknown export format 'xls'/);
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
});
