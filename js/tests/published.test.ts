/**
 * A program written against the published 1.4.3 package's recording types
 * still compiles: a packet's device time is a bigint and its mode text, the
 * device's firmware is text and its tick rate a number, and a take's device
 * times are an array, each never null. Checked with the compiler itself, on
 * a program it is handed in memory, against the package's own source.
 */

import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import test from "node:test";
import ts from "typescript";

const ROOT = fileURLToPath(new URL("../", import.meta.url));
const PROGRAM = `${ROOT}tests/written-against-1.4.3.ts`;

/** What a program written against 1.4.3 does with the recording types. */
const WRITTEN = `
import { Take, save, type DeviceFacts, type Packet } from "../src/capture/index.ts";

const take = new Take();
take.packet({ index: 0, deviceTime: 1000n, gain: 24, sampleRateHz: 250, mode: "normal", synthetic: false });
take.sample(0n, 1000n, 1.5, [1, 2], 0, false, true);
const ticks: BigInt64Array = take.deviceTicks;
const first: Packet = take.packets[0]!;
const later: bigint = first.deviceTime + 4000n;
const mode: string = first.mode.toUpperCase();
function rate(facts: DeviceFacts): number {
  return facts.tickHz / 4000;
}
function firmware(facts: DeviceFacts): string {
  return facts.firmware.split(".")[0]!;
}
const facts: DeviceFacts = {
  deviceId: "00000000c0de0001",
  firmware: "1.4.6",
  protocol: "1.4",
  tickHz: 1_000_000,
  vrefUv: 4_500_000,
  adcBits: 24,
  channels: 2,
  clock: null,
  gapsInRun: 0,
};
const saved = save("written-against-1.4.3", take, facts);
export const used = [ticks.length, later, mode, rate(facts), firmware(facts), take.arrays().length, saved.label];
`;

test("a program written against the published 1.4.3 recording types compiles against this package", () => {
  const config = ts.getParsedCommandLineOfConfigFile(`${ROOT}tsconfig.json`, {}, { ...ts.sys, onUnRecoverableConfigFileDiagnostic: () => {} });
  assert.ok(config !== undefined, "the package's compiler settings are read");
  const host = ts.createCompilerHost(config.options);
  const getSourceFile = host.getSourceFile.bind(host);
  host.getSourceFile = (name, version, ...rest) =>
    name === PROGRAM ? ts.createSourceFile(name, WRITTEN, version, true, ts.ScriptKind.TS) : getSourceFile(name, version, ...rest);
  const fileExists = host.fileExists.bind(host);
  host.fileExists = (name) => name === PROGRAM || fileExists(name);
  const program = ts.createProgram({ rootNames: [PROGRAM], options: { ...config.options, noEmit: true }, host });
  const found = ts.getPreEmitDiagnostics(program, program.getSourceFile(PROGRAM));
  const said = found.map((d) => ts.flattenDiagnosticMessageText(d.messageText, "\n"));
  assert.deepEqual(said, [], said.join("\n"));
});
