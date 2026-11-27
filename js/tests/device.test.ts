/**
 * The connection's device operations, against the Python library's own.
 *
 * Each case in contract/device is one operation the library's `Device` ran
 * against a stand-in radio: the device's info and status, the answer it gave
 * each control request, every request the library wrote, in order, and what
 * came back. The same device is played here to the connection, and it has
 * to write the same requests and come back with the same value, or refuse
 * in the same words, read through the name table in `words.ts`.
 */

import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import test from "node:test";

import * as J from "../src/capture/pyjson.ts";
import * as P from "../src/protocol.ts";
import { connect, type Connection } from "../src/web-bluetooth.ts";
import { Browser } from "./browser.ts";
import { inJavaScript } from "./words.ts";

interface Case {
  readonly name: string;
  readonly info: string;
  readonly status: string;
  readonly method: string;
  readonly args: unknown[];
  readonly answers: Record<string, string | string[]>;
  readonly result?: unknown;
  readonly error?: string;
  readonly writes: string[];
}

const FIXTURE = JSON.parse(readFileSync(new URL("../../contract/device/cases.json", import.meta.url), "utf8"));
const CASES: Case[] = FIXTURE.cases;

/** The library's embedding callback, fed one window in two notifications. */
const DELIVERY: {
  readonly info: string;
  readonly status: string;
  readonly answers: Record<string, string>;
  readonly notifications: string[];
  readonly calls_after_each: number[];
  readonly calls: { readonly arguments: number; readonly first_is_the_device: boolean; readonly window: unknown }[];
  readonly writes: string[];
} = FIXTURE.delivery;

/** The library's `collect_embeddings`: one window gathered, the form turned off, the callback put back. */
const COLLECTION: {
  readonly info: string;
  readonly status: string;
  readonly answers: Record<string, string>;
  readonly notifications: string[];
  readonly windows: unknown[];
  readonly writes: string[];
  readonly callback_restored: boolean;
} = FIXTURE.collection;

class FakeCharacteristic extends EventTarget {
  readonly uuid: string;
  value?: DataView;
  readonly writes: Uint8Array[] = [];
  onWrite?: (bytes: Uint8Array) => void;

  constructor(uuid: string, value?: Uint8Array) {
    super();
    this.uuid = uuid;
    if (value !== undefined) this.value = new DataView(value.buffer.slice(0) as ArrayBuffer);
  }

  readValue(): Promise<DataView> {
    assert.ok(this.value !== undefined, `${this.uuid} has nothing to read`);
    return Promise.resolve(this.value);
  }

  writeValueWithResponse(value: BufferSource): Promise<void> {
    return this.#write(value);
  }

  writeValueWithoutResponse(value: BufferSource): Promise<void> {
    return this.#write(value);
  }

  notify(bytes: Uint8Array): void {
    this.value = new DataView(bytes.buffer.slice(0) as ArrayBuffer);
    this.dispatchEvent(new Event("characteristicvaluechanged"));
  }

  #write(value: BufferSource): Promise<void> {
    const bytes = ArrayBuffer.isView(value)
      ? new Uint8Array(value.buffer, value.byteOffset, value.byteLength).slice()
      : new Uint8Array(value).slice();
    this.writes.push(bytes);
    this.onWrite?.(bytes);
    return Promise.resolve();
  }
}

/** The case's device: its info, its status, and its answer to each request, in order. */
async function played(c: Case): Promise<{ link: Connection; control: FakeCharacteristic }> {
  const control = new FakeCharacteristic(P.CONTROL);
  const response = new FakeCharacteristic(P.CONTROL_RESPONSE);
  const characteristics = new Map<string, FakeCharacteristic>([
    [P.DEVICE_INFO, new FakeCharacteristic(P.DEVICE_INFO, P.fromHex(c.info))],
    [P.CONTROL, control],
    [P.CONTROL_RESPONSE, response],
    [P.EEG_DATA, new FakeCharacteristic(P.EEG_DATA)],
    [P.STATUS, new FakeCharacteristic(P.STATUS, P.fromHex(c.status))],
    [P.EMBEDDINGS, new FakeCharacteristic(P.EMBEDDINGS)],
  ]);
  const answers = new Map<number, string[]>(
    Object.entries(c.answers).map(([op, a]) => [Number(op), Array.isArray(a) ? [...a] : [a]]),
  );
  const repeat = new Map<number, string>(
    Object.entries(c.answers).filter(([, a]) => !Array.isArray(a)).map(([op, a]) => [Number(op), a as string]),
  );
  // The device acknowledges a write first and handles what it carried
  // afterwards, so its answer comes on the loop's next turn, once the write
  // has returned and the caller has gone on, as the library's fixture radio
  // answers.
  control.onWrite = (bytes) => {
    const op = bytes[0]!;
    const queued = answers.get(op);
    const answer = repeat.get(op) ?? (queued && queued.length ? queued.shift()! : null);
    setImmediate(() => response.notify(answer === null ? new Uint8Array([op, 0]) : P.fromHex(answer)));
  };
  const browser = new Browser(characteristics, { id: "fake", name: "IntoMind One" });
  return { link: await connect(browser.device), control };
}

const METHOD: Record<string, string> = {
  refresh_config: "refreshConfig",
  set_gain: "setGain",
  set_rate: "setRate",
  set_mode: "setMode",
  set_leadoff: "setLeadoff",
  pipeline_catalog: "pipelineCatalog",
  pipeline: "pipeline",
  set_pipeline: "setPipeline",
  clear_pipeline: "clearPipeline",
  restore_pipeline_default: "restorePipelineDefault",
  indicator: "indicator",
  set_indicator: "setIndicator",
  identify: "identify",
  model_interval: "modelInterval",
  set_model_interval: "setModelInterval",
  get_name: "getName",
  set_name: "setName",
  converter_registers: "converterRegisters",
  model_info: "modelInfo",
  heads: "heads",
  select_head: "selectHead",
  set_predictions: "setPredictions",
  set_embeddings: "setEmbeddings",
  remove_head: "removeHead",
  upload_head: "uploadHead",
};

/** Where a field the library names one way sits in this package, tried in order. */
const FIELD: Record<string, string[]> = {
  state: ["modelStateName", "stateName"],
  family: ["familyName"],
  family_code: ["family"],
  cls: ["className"],
  origin: ["originName"],
};

const camel = (k: string) => k.replace(/_([a-z])/g, (_, ch: string) => ch.toUpperCase());

/** The library's value against this package's, every field the library names. */
function same(py: unknown, js: unknown, where: string): void {
  if (js instanceof Uint8Array) js = Buffer.from(js).toString("hex");
  if (typeof js === "bigint") js = Number(js);
  if (Array.isArray(py)) {
    assert.ok(Array.isArray(js), `${where}: the library has a list, this has ${String(js)}`);
    assert.equal((js as unknown[]).length, py.length, `${where}: lengths`);
    py.forEach((v, i) => same(v, (js as unknown[])[i], `${where}[${i}]`));
    return;
  }
  if (py !== null && typeof py === "object") {
    assert.ok(js !== null && typeof js === "object", `${where}: the library has an object`);
    for (const [k, v] of Object.entries(py)) {
      const o = js as Record<string, unknown>;
      const key = [...(FIELD[k] ?? []), camel(k), k].find((x) => x in o);
      assert.ok(key !== undefined, `${where}: no field for the library's ${k}`);
      // The library names a window's input source. This package's window
      // carries its code, as it has since 1.4.0, read through its own table.
      const value = k === "input_source" && typeof o[key] === "number" ? P.INPUT_SOURCE_NAMES[o[key] as number] : o[key];
      same(v, value, `${where}.${k}`);
    }
    return;
  }
  assert.deepEqual(js ?? null, py, where);
}

for (const c of CASES) {
  test(`${c.method}: ${c.name}`, async () => {
    const { link, control } = await played(c);
    const fn = (link as unknown as Record<string, (...a: unknown[]) => Promise<unknown>>)[METHOD[c.method]!]!;
    assert.ok(typeof fn === "function", `the connection has no ${METHOD[c.method]}`);
    const args = c.method === "set_pipeline"
      ? [(c.args[0] as { kind: number; params: number[] }[]).map((s) => ({ kind: s.kind, params: s.params }))]
      : c.method === "upload_head"
        ? [P.fromHex(c.args[0] as string), ...c.args.slice(1)]
        : c.args;
    let result: unknown;
    let error: Error | null = null;
    try {
      result = await fn.apply(link, args);
    } catch (e) {
      error = e as Error;
    }
    assert.deepEqual(control.writes.map((b) => Buffer.from(b).toString("hex")), c.writes, "the requests written");
    if (c.error !== undefined) {
      assert.ok(error !== null, `the library refused: ${c.error}`);
      const said = [error.message, `${error.name}: ${error.message}`];
      const words = inJavaScript(c.error);
      assert.ok(said.includes(words), `refused as\n  ${said[0]}\nwhere the library says\n  ${words}`);
      return;
    }
    assert.equal(error, null, `the library did not refuse: ${error?.message}`);
    if (c.method === "refresh_config") {
      assert.equal(J.dumps(result as J.Value), J.dumps(J.fromPlain(c.result)));
      return;
    }
    same(c.result ?? null, result, c.method);
  });
}

test("a whole window reaches onEmbedding with the connection, as the library hands its callback the device", async () => {
  const d = DELIVERY;
  const { link, control } = await played({ ...d, method: "set_embeddings", args: ["window"] } as unknown as Case);
  const calls: unknown[][] = [];
  await link.setEmbeddings("window", (...args: unknown[]) => calls.push(args));
  const after: number[] = [];
  for (const n of d.notifications) {
    link.deliver(P.FILL.EMBEDDINGS, P.fromHex(n));
    after.push(calls.length);
  }
  assert.deepEqual(control.writes.map((b) => Buffer.from(b).toString("hex")), d.writes, "the requests written");
  assert.deepEqual(after, d.calls_after_each, "a call only when a window is whole");
  assert.equal(calls.length, d.calls.length);
  d.calls.forEach((py, i) => {
    assert.equal(calls[i]!.length, py.arguments, "the number of arguments");
    assert.equal(calls[i]![0] === link, py.first_is_the_device, "the first argument is the connection");
    same(py.window, calls[i]![1], "window");
  });
});

test("windows of the encoder's output are gathered as the library gathers them, and the form turned off after", async () => {
  const c = COLLECTION;
  const { link, control } = await played({ ...c, method: "collect_embeddings", args: [1] } as unknown as Case);
  const answer = control.onWrite!;
  const window = Buffer.from([P.OPCODES.set_embeddings, P.EMBEDDING_FORMS.window]).toString("hex");
  // The windows the form asks for come once the write has returned, before
  // its answer, on the same turn, as the library's fixture radio sends them.
  control.onWrite = (bytes) => {
    if (Buffer.from(bytes).toString("hex") === window) setImmediate(() => c.notifications.forEach((n) => link.deliver(P.FILL.EMBEDDINGS, P.fromHex(n))));
    answer(bytes);
  };
  const got = await link.collectEmbeddings(1);
  assert.deepEqual(control.writes.map((b) => Buffer.from(b).toString("hex")), c.writes, "the requests written");
  assert.equal(got.length, c.windows.length);
  c.windows.forEach((py, i) => same(py, got[i], `window ${i}`));
  assert.equal(link.onEmbedding === null, c.callback_restored, "the callback put back as it was");
});
