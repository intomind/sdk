/**
 * The protocol against the contract's own vectors.
 *
 * `contract/conformance.json` is emitted by the firmware from the codec the
 * device runs. Every message in it decodes here to the fields it states,
 * and every malformed one is refused for the reason it states. A drift
 * between this implementation and the device shows up here rather than on a
 * bench.
 */

import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import test from "node:test";

import * as P from "../src/protocol.ts";

// --- the vectors ------------------------------------------------------------

interface Vector {
  name: string;
  kind: string;
  bytes: string;
  fields: Record<string, unknown>;
}

interface Refusal {
  name: string;
  kind: string;
  bytes: string;
  reason: P.Refusal;
}

interface Vectors {
  protocol: string;
  uuids: Array<{ name: string; fill: number; uuid: string }>;
  decode: Vector[];
  refuse: Refusal[];
  scaling: Array<{ raw: number; adc_bits: number; gain: number; vref_uv: number; microvolts: number }>;
  continuity: Array<{
    prev_index: number;
    prev_n_samples: number;
    new_index: number;
    discontinuity_flag: boolean;
    verdict: string;
    /** null for a break, which has no extent. */
    lost: number | null;
  }>;
  embedding: Array<{ floats: number[]; integers: number[] }>;
}

const VECTORS: Vectors = (() => {
  const text = readFileSync(new URL("../../contract/conformance.json", import.meta.url), "utf8");
  const parsed = JSON.parse(text) as Vectors;
  // The contract writes every 64-bit field as a decimal string, so nothing
  // here goes through the rounding a JSON number would apply. Hold that,
  // because the rounding is silent: a reader that parsed a device time as a
  // number would agree with an equally rounded expectation and pass.
  assert.ok(
    !/:\s*-?\d{16,}\s*[,}\]]/.test(text),
    "a 64-bit field is written as a decimal string, never as a JSON number",
  );
  return parsed;
})();

/** A 64-bit field, which the contract writes as a decimal string. */
function u64(field: unknown): bigint {
  assert.equal(typeof field, "string", "a 64-bit field is written as a decimal string");
  return BigInt(field as string);
}

function byKind(kind: string): Vector[] {
  return VECTORS.decode.filter((v) => v.kind === kind);
}

/** A chain as the vectors write it: kind and parameters, nothing named. */
function stagesOf(stages: readonly P.Stage[]): Array<{ kind: number; params: number[] }> {
  return stages.map((s) => ({ kind: s.kind, params: [...s.params] }));
}

const bytesOf = P.fromHex;

/** The device the vectors were taken from, and the only source of its width. */
const DEVICE = P.decodeDeviceInfo(bytesOf(byKind("device_info")[0]!.bytes));

// --- comparison -------------------------------------------------------------

function same(got: unknown, want: unknown, where: string, tol: number): void {
  if (typeof got === "bigint") {
    assert.equal(got, BigInt(want as string | number), where);
    return;
  }
  if (Array.isArray(want)) {
    assert.ok(Array.isArray(got), `${where}: expected a list`);
    assert.equal((got as unknown[]).length, want.length, `${where}: length`);
    want.forEach((w, i) => same((got as unknown[])[i], w, `${where}[${i}]`, tol));
    return;
  }
  if (want !== null && typeof want === "object") {
    for (const [key, w] of Object.entries(want as Record<string, unknown>)) {
      same((got as Record<string, unknown>)[key], w, `${where}.${key}`, tol);
    }
    return;
  }
  if (typeof want === "number" && !Number.isInteger(want)) {
    assert.equal(typeof got, "number", where);
    const slack = tol * Math.max(1, Math.abs(want));
    assert.ok(Math.abs((got as number) - want) <= slack, `${where}: ${got} is not ${want}`);
    return;
  }
  assert.deepStrictEqual(got, want, where);
}

/** Every field the vector states, decoded to what it states. */
function check(vector: Vector, actual: Record<string, unknown>, tol = 1e-9): void {
  for (const [key, want] of Object.entries(vector.fields)) {
    assert.ok(key in actual, `${vector.name}: ${key} is not decoded`);
    same(actual[key], want, `${vector.name}: ${key}`, tol);
  }
}

// --- one decoder per kind ---------------------------------------------------

type Decoder = (bytes: Uint8Array, vector: Vector) => Record<string, unknown>;

const DECODERS: Record<string, Decoder> = {
  device_info(bytes) {
    const info = P.decodeDeviceInfo(bytes);
    return {
      proto_major: info.protocol[0],
      proto_minor: info.protocol[1],
      fw_version: info.firmwareString,
      hw_version: info.hardwareString,
      channel_count: info.channels,
      adc_bits: info.adcBits,
      time_tick_hz: info.tickHz,
      vref_uv: info.vrefUv,
      capabilities: info.capabilities,
      supported_rates: info.supportedRates,
      device_id: info.deviceId,
      extension_present: info.extensionPresent,
      // What this codec emits for this device is the length the contract states.
      info_len: P.encodeDeviceInfo(info).length,
      capabilities_high: info.capabilitiesHigh,
      model_embed_dim: info.modelEmbedDim,
      model_native_sps: info.modelNativeSps,
      model_window_samples: info.modelWindowSamples,
      head_slots: info.headSlots,
      head_max_outputs: info.headMaxOutputs,
      head_slot_bytes: info.headSlotBytes,
      update_chunk_max: info.updateChunkMax,
      app_slot_bytes: info.appSlotBytes,
      weights_image_bytes: info.weightsImageBytes,
    };
  },

  eeg_data(bytes) {
    const p = P.decodePacket(bytes, DEVICE.channels);
    return {
      packet_type: p.packetType,
      discontinuity: p.discontinuity,
      leadoff_active: p.leadoffActive,
      mode: p.mode,
      usb_present: p.usbPresent,
      samples_lost_before: p.samplesLostBefore,
      sample_index: p.index,
      device_time: p.deviceTime,
      n_samples: p.nSamples,
      loff_statp: p.loffStatp,
      gain_code: p.gainCode,
      gain: p.gain,
      rate_code: p.rateCode,
      sps: p.sps,
      samples: p.counts,
      synthetic: p.synthetic,
    };
  },

  control_request(bytes) {
    const r = P.decodeRequest(bytes);
    assert.deepEqual(P.encodeRequest(r.name, r.payload ?? r.arg), bytes, "encoding gives back the same bytes");
    const out: Record<string, unknown> = { opcode: r.opcode, arg: r.arg };
    if (r.payload !== null && r.name === "set_model_interval") {
      out.interval_s = P.decodeModelInterval(new Uint8Array([...r.payload, 0, 0])).intervalS;
    }
    if (r.payload !== null && r.name === "set_name") {
      const parts = P.decodeNameParts(r.payload);
      out.name = parts.name;
      out.adjective = parts.adjective;
    }
    if (r.payload !== null && r.name === "set_pipeline") out.stages = stagesOf(P.decodeChain(r.payload));
    if (r.payload !== null && r.name === "set_prediction_input") {
      const pi = P.decodePredictionInput(r.payload);
      out.source = pi.source;
      out.stages = stagesOf(pi.stages);
    }
    return out;
  },

  pipeline_catalog(bytes) {
    return {
      kinds: P.decodeCatalog(bytes).map((e) => ({
        kind: e.kind,
        class: e.cls,
        n_params: e.nParams,
        max_instances: e.maxInstances,
        name: e.name,
      })),
    };
  },

  chain(bytes) {
    return { stages: stagesOf(P.decodeChain(bytes)) };
  },

  pipeline_state(bytes) {
    const st = P.decodePipelineState(bytes);
    return { origin: st.origin, stages: stagesOf(st.stages) };
  },

  prediction_input(bytes) {
    const pi = P.decodePredictionInput(bytes);
    return { source: pi.source, stages: stagesOf(pi.stages) };
  },

  bias_diagnostic(bytes) {
    const d = P.decodeBiasDiagnostic(bytes);
    return { mean_mv: d.meanMv, sd_mv: d.sdMv, min_mv: d.minMv, max_mv: d.maxMv };
  },

  control_response(bytes) {
    const r = P.decodeResponse(bytes);
    const out: Record<string, unknown> = { opcode: r.opcode, status: r.status, status_name: r.statusName };
    if (r.opcode === P.OPCODES.time_sync && r.ok) out.device_time = P.decodeTimeSync(r.payload);
    if (r.opcode === P.OPCODES.get_indicator && r.ok) {
      out.level = r.payload[0];
      out.level_name = P.INDICATOR_LEVEL_NAMES[r.payload[0]!];
    }
    if (r.opcode === P.OPCODES.get_model_interval && r.ok) {
      const mi = P.decodeModelInterval(r.payload);
      out.interval_s = mi.intervalS;
      out.minimum_s = mi.minimumS;
    }
    if (r.opcode === P.OPCODES.get_name && r.ok) {
      const parts = P.decodeNameParts(r.payload);
      out.name = parts.name;
      out.adjective = parts.adjective;
    }
    return out;
  },

  battery_info(bytes) {
    const b = P.decodeBattery(bytes);
    return { battery_mv: b.millivolts, battery_percent: b.percent, charger_state: b.chargerState };
  },

  boot_info(bytes) {
    const b = P.decodeBootInfo(bytes);
    return {
      active_slot: b.activeSlot,
      boot_reason: b.bootReason,
      boot_reason_name: b.bootReasonName,
      slot_state: b.slotState,
      boot_count: b.bootCount,
    };
  },

  list_heads(bytes) {
    const list = P.decodeHeads(bytes);
    return {
      active_slot: list.activeSlot,
      heads: list.heads.map((h) => ({
        slot: h.slot,
        state: h.state,
        out_dim: h.outDim,
        head_id: h.headId,
        name: h.name,
        encoder_id: h.encoderId,
      })),
    };
  },

  list_head_encoders(bytes) {
    return {
      heads: [...P.decodeHeadEncoders(bytes)].map(([slot, encoderId]) => ({ slot, encoder_id: encoderId })),
    };
  },

  model_info(bytes) {
    const m = P.decodeModelInfo(bytes);
    return {
      model_state: m.modelState,
      active_head: m.activeHead,
      predictions_on: m.predictionsOn ? 1 : 0,
      encoder_id: m.encoderId,
      input_classes: m.inputClasses,
      tokens_per_channel: m.tokensPerChannel,
      pass_ms: m.passMs,
      interval_s: m.intervalS,
      generator: m.generator,
    };
  },

  converter_registers(bytes) {
    const r = P.decodeConverterRegisters(bytes);
    return {
      family: r.family,
      family_name: r.familyName,
      first: r.first,
      count: r.values.length,
      values: Array.from(r.values, (b) => b.toString(16).padStart(2, "0")).join(""),
    };
  },

  embedding(bytes) {
    const e = P.decodeEmbedding(bytes);
    return {
      packet_type: e.packetType,
      gap_in_window: e.gapInWindow,
      duty_reduced: e.dutyReduced,
      leadoff_in_window: e.leadoffInWindow,
      more_parts: e.moreParts,
      embed_dim: e.embedDim,
      first: e.first,
      sample_index: e.index,
      device_time: e.deviceTime,
      window_samples: e.windowSamples,
      encoder_id: e.encoderId,
      input_source: e.inputSource,
      token: e.token ?? 255,
      is_window_embedding: e.token === null,
      values: e.values,
    };
  },

  status(bytes) {
    const s = P.decodeStatus(bytes);
    return {
      state: s.state,
      streaming: s.streaming,
      mode: s.mode,
      synthetic: s.mode === P.MODES.synthetic,
      gain_code: s.gainCode,
      rate_code: s.rateCode,
      charger_state: s.chargerState,
      battery_percent: s.batteryPercent,
      battery_percent_raw: s.batteryPercentRaw,
      loff_statp: s.loffStatp,
      usb_present: s.usbPresent,
      buffer_high_watermark: s.bufferHighWatermark,
      dropped_total: s.droppedTotal,
      buffer_fill: s.bufferFill,
    };
  },

  update_request(bytes) {
    const r = P.decodeUpdateRequest(bytes);
    return { op: r.op, target: r.target, slot: r.slot, total_len: r.totalLen, transfer_id: r.transferId };
  },

  update_response(bytes) {
    const r = P.decodeUpdateResponse(bytes);
    const out: Record<string, unknown> = { op: r.op, status: r.status };
    if (r.op === P.UPDATE_OPS.start && r.ok) {
      const s = P.decodeUpdateStart(r.payload);
      out.target_slot = s.targetSlot;
      out.chunk_max = s.chunkMax;
      out.resume_offset = s.resumeOffset;
    }
    if (r.op === P.UPDATE_OPS.query && r.ok) {
      const q = P.decodeUpdateQuery(r.payload);
      out.state = q.state;
      out.offset = q.offset;
      out.crc32 = q.crc32;
    }
    if (!r.ok) {
      out.verify_result = r.verifyResult;
      out.verify_result_name = r.verifyResultName;
    }
    return out;
  },

  envelope(bytes) {
    const e = P.decodeEnvelope(bytes);
    return {
      target: e.target,
      slot_link: e.slotLink,
      key_id: e.keyId,
      plain_len: e.plainLen,
      total_len: e.totalLen,
    };
  },

  prediction(bytes) {
    const p = P.decodePrediction(bytes);
    return {
      packet_type: p.packetType,
      head_slot: p.headSlot,
      head_id: p.headId,
      sample_index: p.index,
      device_time: p.deviceTime,
      window_samples: p.windowSamples,
      gap_in_window: p.gapInWindow,
      duty_reduced: p.dutyReduced,
      leadoff_in_window: p.leadoffInWindow,
      n_outputs: p.outputs.length,
      outputs: p.outputs,
      input_source: p.inputSource,
    };
  },

  head(bytes, vector) {
    const h = P.decodeHead(bytes);
    const asked = (vector.fields.outputs_for_embedding ?? {}) as { embedding?: number[] };
    return {
      kind: h.kind,
      in_dim: h.inDim,
      out_dim: h.outDim,
      name: h.name,
      head_id: h.headId,
      weights: h.weights,
      bias: h.bias,
      scale: h.scale,
      outputs_for_embedding: {
        embedding: asked.embedding,
        outputs: asked.embedding === undefined ? undefined : P.evaluateHead(h, asked.embedding),
      },
    };
  },
};

/** Kinds whose values are single precision on the wire. */
const FLOAT_KINDS = new Set(["prediction", "head"]);

// --- the tests --------------------------------------------------------------

test("the vectors are for this version", () => {
  assert.equal(VECTORS.protocol, P.VERSION.join("."));
});

test("every identifier matches", () => {
  const named: Record<string, string> = {
    service: P.SERVICE,
    device_info: P.DEVICE_INFO,
    control: P.CONTROL,
    control_response: P.CONTROL_RESPONSE,
    eeg_data: P.EEG_DATA,
    status: P.STATUS,
    update_control: P.UPDATE_CONTROL,
    update_data: P.UPDATE_DATA,
    predictions: P.PREDICTIONS,
    embeddings: P.EMBEDDINGS,
  };
  for (const u of VECTORS.uuids) {
    assert.equal(named[u.name], u.uuid, u.name);
    assert.equal(P.uuid(u.fill), u.uuid, `${u.name} built from its fill`);
  }
  assert.equal(Object.keys(named).length, VECTORS.uuids.length);
});

test("every message decodes to the fields the contract states", () => {
  const seen = new Set<string>();
  for (const vector of VECTORS.decode) {
    const decode = DECODERS[vector.kind];
    assert.ok(decode !== undefined, `no decoder for ${vector.kind}`);
    seen.add(vector.kind);
    check(vector, decode(bytesOf(vector.bytes), vector), FLOAT_KINDS.has(vector.kind) ? 1e-6 : 1e-9);
  }
  assert.equal(seen.size, 22, "every kind the contract carries has a decoder");
  assert.equal(VECTORS.decode.length, 70, "the contract carries seventy messages, and all of them were read");
});

test("every message encodes back to the bytes it came from", () => {
  const encoders: Record<string, (bytes: Uint8Array) => Uint8Array> = {
    device_info: (b) => P.encodeDeviceInfo(P.decodeDeviceInfo(b)),
    eeg_data: (b) => {
      const p = P.decodePacket(b, DEVICE.channels);
      return P.encodePacket({
        index: p.index,
        deviceTime: p.deviceTime,
        gainCode: p.gainCode,
        rateCode: p.rateCode,
        counts: p.counts,
        discontinuity: p.discontinuity,
        leadoffActive: p.leadoffActive,
        mode: p.mode,
        usbPresent: p.usbPresent,
        samplesLostBefore: p.samplesLostBefore,
        loffStatp: p.loffStatp,
        synthetic: p.synthetic,
      });
    },
    control_request: (b) => {
      const r = P.decodeRequest(b);
      return P.encodeRequest(r.name, r.payload ?? r.arg);
    },
    control_response: (b) => {
      const r = P.decodeResponse(b);
      return P.encodeResponse(r.opcode, r.status, r.payload);
    },
    battery_info: (b) => {
      const x = P.decodeBattery(b);
      return P.encodeBattery({ millivolts: x.millivolts, percent: x.percent, chargerState: x.chargerState });
    },
    boot_info: (b) => {
      const x = P.decodeBootInfo(b);
      return P.encodeBootInfo({
        activeSlot: x.activeSlot,
        bootReason: x.bootReason,
        slotState: x.slotState,
        bootCount: x.bootCount,
      });
    },
    list_heads: (b) => {
      const l = P.decodeHeads(b);
      return P.encodeHeads(
        l.activeSlot,
        l.heads.map((h) => ({ slot: h.slot, state: h.state, outDim: h.outDim, headId: h.headId, name: h.name })),
      );
    },
    list_head_encoders: (b) =>
      P.encodeHeadEncoders([...P.decodeHeadEncoders(b)].map(([slot, encoderId]) => ({ slot, encoderId }))),
    model_info: (b) => {
      const m = P.decodeModelInfo(b);
      return P.encodeModelInfo({
        modelState: m.modelState,
        activeHead: m.activeHead,
        predictionsOn: m.predictionsOn,
        encoderId: m.encoderId,
        weightsVersion: m.weightsVersion,
        inputClasses: m.inputClasses,
        tokensPerChannel: b.length >= 17 ? m.tokensPerChannel : undefined,
        passMs: b.length >= 22 ? m.passMs : undefined,
        intervalS: b.length >= 22 ? m.intervalS : undefined,
        generator: b.length >= 22 ? m.generator : undefined,
      });
    },
    converter_registers: (b) => P.encodeConverterRegisters(P.decodeConverterRegisters(b)),
    embedding: (b) => {
      const e = P.decodeEmbedding(b);
      return P.encodeEmbedding({
        index: e.index,
        deviceTime: e.deviceTime,
        windowSamples: e.windowSamples,
        encoderId: e.encoderId,
        inputSource: e.inputSource,
        embedDim: e.embedDim,
        first: e.first,
        token: e.token,
        moreParts: e.moreParts,
        gapInWindow: e.gapInWindow,
        dutyReduced: e.dutyReduced,
        leadoffInWindow: e.leadoffInWindow,
        values: e.values,
      });
    },
    status: (b) => {
      const s = P.decodeStatus(b);
      return P.encodeStatus({
        state: s.state,
        mode: s.mode,
        gainCode: s.gainCode,
        rateCode: s.rateCode,
        chargerState: s.chargerState,
        batteryPercent: s.batteryPercent,
        loffStatp: s.loffStatp,
        usbPresent: s.usbPresent,
        bufferHighWatermark: s.bufferHighWatermark,
        droppedTotal: s.droppedTotal,
        bufferFill: s.bufferFill,
      });
    },
    update_request: (b) => {
      const r = P.decodeUpdateRequest(b);
      return P.encodeUpdateStart(r.target!, r.slot!, r.totalLen!, r.transferId!);
    },
    update_response: (b) => {
      const r = P.decodeUpdateResponse(b);
      return P.encodeUpdateResponse(r.op, r.status, r.payload);
    },
    envelope: (b) => {
      const e = P.decodeEnvelope(b);
      return P.encodeEnvelope({
        target: e.target,
        slotLink: e.slotLink,
        keyId: e.keyId,
        nonce: e.nonce,
        plainLen: e.plainLen,
      });
    },
    prediction: (b) => {
      const p = P.decodePrediction(b);
      return P.encodePrediction({
        headSlot: p.headSlot,
        headId: p.headId,
        index: p.index,
        deviceTime: p.deviceTime,
        windowSamples: p.windowSamples,
        outputs: p.outputs,
        gapInWindow: p.gapInWindow,
        dutyReduced: p.dutyReduced,
        leadoffInWindow: p.leadoffInWindow,
        inputSource: p.inputSource,
      });
    },
    pipeline_catalog: (b) => P.encodeCatalog(P.decodeCatalog(b)),
    chain: (b) => P.encodeChain(P.decodeChain(b)),
    pipeline_state: (b) => {
      const st = P.decodePipelineState(b);
      return P.encodePipelineState(st.origin, st.stages);
    },
    prediction_input: (b) => {
      const pi = P.decodePredictionInput(b);
      return P.encodePredictionInput(pi.source, pi.stages);
    },
    bias_diagnostic: (b) => P.encodeBiasDiagnostic(P.decodeBiasDiagnostic(b)),
    head: (b) => {
      const h = P.decodeHead(b);
      return P.buildHead(h.weights, h.bias, h.scale, { name: h.name, inDim: h.inDim, version: h.version, encoderId: h.encoderId });
    },
  };
  for (const vector of VECTORS.decode) {
    const encode = encoders[vector.kind];
    assert.ok(encode !== undefined, `no encoder for ${vector.kind}`);
    const want = bytesOf(vector.bytes);
    assert.deepEqual(encode(want), want, vector.name);
  }
});

test("every malformed message is refused for its stated reason", () => {
  const decoders: Record<string, (b: Uint8Array) => unknown> = {
    device_info: (b) => P.decodeDeviceInfo(b),
    eeg_data: (b) => P.decodePacket(b, DEVICE.channels),
    control_request: (b) => P.decodeRequest(b),
    control_response: (b) => P.decodeResponse(b),
    envelope: (b) => P.decodeEnvelope(b),
    status: (b) => P.decodeStatus(b),
    battery_info: (b) => P.decodeBattery(b),
    boot_info: (b) => P.decodeBootInfo(b),
    model_info: (b) => P.decodeModelInfo(b),
    list_heads: (b) => P.decodeHeads(b),
    list_head_encoders: (b) => P.decodeHeadEncoders(b),
    prediction: (b) => P.decodePrediction(b),
    update_response: (b) => P.decodeUpdateResponse(b),
    head: (b) => P.decodeHead(b),
    pipeline_catalog: (b) => P.decodeCatalog(b),
    chain: (b) => P.decodeChain(b),
    pipeline_state: (b) => P.decodePipelineState(b),
    prediction_input: (b) => P.decodePredictionInput(b),
    bias_diagnostic: (b) => P.decodeBiasDiagnostic(b),
    converter_registers: (b) => P.decodeConverterRegisters(b),
    embedding: (b) => P.decodeEmbedding(b),
  };
  const expected: Record<P.Refusal, new (...args: never[]) => P.ProtocolError> = {
    truncated: P.Truncated,
    invalid: P.Invalid,
    reserved: P.Reserved,
  };
  let seen = 0;
  for (const vector of VECTORS.refuse) {
    const decode = decoders[vector.kind];
    assert.ok(decode !== undefined, `no decoder for ${vector.kind}`);
    seen += 1;
    assert.throws(
      () => decode(bytesOf(vector.bytes)),
      (error: unknown) => {
        assert.ok(error instanceof P.ProtocolError, `${vector.name}: refused as ${error}`);
        assert.ok(
          error instanceof expected[vector.reason],
          `${vector.name}: refused as ${error.name}, and the contract calls it ${vector.reason}`,
        );
        assert.equal(error.reason, vector.reason, vector.name);
        return true;
      },
      `${vector.name}: accepted, and it is not a message`,
    );
  }
  assert.equal(seen, VECTORS.refuse.length, "every refusal vector has a decoder here");
});

test("a count is worth what the contract says", () => {
  for (const s of VECTORS.scaling) {
    // The contract's scaling vectors are for the converter in this device,
    // and its width comes from the device rather than from this test.
    const got = P.microvolts(s.raw, s.vref_uv, s.gain, DEVICE.adcBits);
    assert.ok(
      Math.abs(got - s.microvolts) <= 1e-12 * Math.max(1, Math.abs(s.microvolts)),
      `${s.raw} at gain ${s.gain}: ${got} is not ${s.microvolts}`,
    );
  }
  // And the device's own numbers give the same answer.
  assert.equal(DEVICE.microvoltsPerCount(24), 0.022351741790771484);
});

test("the timeline is judged by the contract's rule", () => {
  for (const c of VECTORS.continuity) {
    const got = P.continuity(c.prev_index, c.prev_n_samples, c.new_index, c.discontinuity_flag);
    assert.equal(got.verdict, c.verdict, JSON.stringify(c));
    if (got.verdict === "gap") assert.equal(got.lost, c.lost, JSON.stringify(c));
    if (got.verdict === "break") {
      assert.equal(got.lost, null, "a break has no count, and zero would be a different claim");
    }
  }
  // The wrap is exact rather than four billion samples of imaginary loss.
  const wrapped = P.continuity(0xfffffffc, 5, 3, false);
  assert.equal(wrapped.verdict, "gap");
  assert.equal(wrapped.lost, 2);
});

test("an embedding quantizes to the published scale", () => {
  for (const e of VECTORS.embedding) {
    assert.deepEqual(P.quantizeEmbedding(e.floats), e.integers, JSON.stringify(e));
  }
});

test("a head that does not match its own hash is refused", () => {
  const blob = bytesOf(byKind("head")[0]!.bytes);
  const broken = blob.slice();
  broken[P.HEAD_HEADER_LEN] = broken[P.HEAD_HEADER_LEN]! ^ 0x01;
  assert.throws(() => P.decodeHead(broken), P.Invalid);
});

test("capabilities are the device's answer", () => {
  for (const name of ["battery_voltage", "leadoff", "test_signal", "update", "model", "model_ready", "heads"]) {
    assert.ok(DEVICE.can(name), name);
  }
  assert.ok(!DEVICE.can("battery_low_flag"));
  assert.ok(!DEVICE.can("a capability invented after this host was written"));
  assert.deepEqual(DEVICE.rates, [250, 500, 1000]);
  // An older device is not given fields it never reported.
  const older = P.decodeDeviceInfo(bytesOf(byKind("device_info")[1]!.bytes));
  assert.equal(older.hardware, null);
  assert.equal(older.headSlots, 0);
  assert.ok(!older.extensionPresent);
});

test("a region's mains bands, and only those a rate can represent", () => {
  const pairs = (stages: P.Stage[]) => stages.map((s) => [...s.params]);
  assert.deepEqual(pairs(P.mainsBands(60)), [[580, 620], [1180, 1220], [1780, 1820]]);
  assert.deepEqual(pairs(P.mainsBands(50, 250)), [[480, 520], [980, 1020]]);
  assert.deepEqual(pairs(P.mainsBands(60, 250)), [[580, 620]]);
});

test("the IntoMind One's longest head list fits one answer, and the encoders come in their own", () => {
  const full = VECTORS.decode.find((v: { kind: string; name: string }) => v.kind === "list_heads" && v.name.includes("IntoMind One"));
  assert.ok(full !== undefined);
  assert.ok(bytesOf(full.bytes).length + 2 <= 156, "an answer is at most 156 bytes");
  const list = P.decodeHeads(bytesOf(full.bytes));
  assert.equal(list.heads.length, 5);
  assert.ok(list.heads.every((h) => h.encoderId === P.NO_ENCODER_ID), "the list carries no encoder ids");
  const three = VECTORS.decode.find((v: { kind: string }) => v.kind === "list_heads")!;
  const ids = VECTORS.decode.find((v: { kind: string }) => v.kind === "list_head_encoders")!;
  const merged = P.withEncoders(P.decodeHeads(bytesOf(three.bytes)), P.decodeHeadEncoders(bytesOf(ids.bytes)));
  assert.deepEqual(merged.heads.map((h) => h.encoderId), ["0000000000000000", "a7c61680d9a202db", "0000000000000000"]);
});
