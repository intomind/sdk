"""The device fixture: what the Python library's `Device` asks of a device,
and answers, for each operation an application drives on one, which the
JavaScript connection is held to.

    PYTHONPATH=<a checkout of the Python library> python3 contract/device/make.py

A case is one operation on one device: the device's info, its status, and
the answer it gives to each control request, taken from the conformance
vectors so both languages decode the same bytes. The library's
`client.Device` runs the operation through a stand-in for the radio, and
the case records every control request it wrote, in order, and what came
back: the value, or the exception's name and message.

Uploading a head is held to the library up to the transfer: what it
refuses, and what it asks the device before it refuses. The transfer itself
is the package's own, held to its own device in its transfer tests. The
heads uploaded are trained from seeded synthetic embeddings.

Writes cases.json beside this script. Run it again and nothing changes
unless the library does.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import pathlib
import struct
import types
import warnings

warnings.simplefilter("ignore")
HERE = pathlib.Path(__file__).resolve().parent
# Made without a sound: the library's experiments speak their cues and play
# their tones through the host's speakers, so every cue goes nowhere here,
# before anything of the library runs.
from intomind import audio as _audio  # noqa: E402
_audio.play = lambda tone, text="": None

import numpy as np  # noqa: E402

from intomind import client, model as M, protocol as P  # noqa: E402

VECTORS = json.loads((HERE.parent / "conformance.json").read_text())["decode"]


def vector(name: str) -> bytes:
    """A conformance vector's bytes, named by the start of its name."""
    return bytes.fromhex(next(v for v in VECTORS if v["name"].startswith(name))["bytes"])


ONE = vector("device info, the IntoMind One")


def info(*, minor: int = 4, channels: int = 4, caps: int | None = None,
         caps_high: int | None = None, length: int | None = None) -> bytes:
    """The IntoMind One's Device Info from the conformance vectors, with a
    field or two changed."""
    b = bytearray(ONE)
    b[1], b[5] = minor, channels
    if caps is not None:
        b[15:17] = caps.to_bytes(2, "little")
    if caps_high is not None:
        b[76:80] = caps_high.to_bytes(4, "little")
    if length is not None:
        b = b[:length]
        b[27] = length
    return bytes(b)


BARE = info(caps=0, caps_high=0)
NO_UPDATE = info(caps=int.from_bytes(ONE[15:17], "little") & ~P.CAPABILITIES["update"])
#: Synthetic integer embeddings, as wide as the device's model, for heads that are refused before they travel.
ROWS = np.random.default_rng(20261102).integers(-900, 900, (120, P.decode_device_info(ONE).model_embed_dim))
LABELS = [i % 2 for i in range(120)]
ONE_13 = info(minor=3)
RATE_CODE = {sps: code for code, sps in P.RATE_BY_CODE.items()}


def status(*, streaming=False, mode=0, gain=24, rate=500, loff=0, flags=0) -> bytes:
    return struct.pack("<BBBBBBBBHH", 1 if streaming else 0, mode, P.GAIN_BY_CODE.index(gain),
                       RATE_CODE.get(rate, 7), 1, 61, loff, flags, 0, 0)


def ok(op: str, payload: bytes = b"") -> bytes:
    return bytes([P.OPCODES[op], 0]) + payload


def refused(op: str, code: int) -> bytes:
    return bytes([P.OPCODES[op], code])


CONFIG = {
    P.OPCODES["get_pipeline"]: ok("get_pipeline", vector("the chain in force, set by a host")),
    P.OPCODES["get_indicator"]: vector("the lamp's level, reserved"),
    P.OPCODES["get_model_interval"]: vector("the model's interval: thirty seconds in force, four the least"),
    P.OPCODES["get_name"]: vector("the device's name: Ada, Blue"),
}
MODEL = {P.OPCODES["get_model_info"]: ok("get_model_info", vector(
    "a model that is ready, with a head selected and predictions on, 1.3: a pass of 3"))}
CATALOG = {P.OPCODES["get_pipeline_catalog"]: ok("get_pipeline_catalog", vector(
    "the catalog of a device that runs the three filter kinds"))}


class Link:
    """The radio: answers each control write as the case says, and records
    it. The device acknowledges a write first and handles what it carried
    afterwards, so its answer comes on the loop's next turn, once the write
    has returned."""

    def __init__(self, dev, status_bytes: bytes, answers: dict):
        self.dev, self.is_connected = dev, True
        self.status = status_bytes
        self.answers = {op: (list(a) if isinstance(a, list) else a) for op, a in answers.items()}
        self.writes: list[str] = []

    async def write_gatt_char(self, _uuid, payload, response=True):
        payload = bytes(payload)
        self.writes.append(payload.hex())
        op = payload[0]
        a = self.answers.get(op)
        if isinstance(a, list):
            a = a.pop(0) if a else None
        answer = a if a is not None else bytes([op, 0])
        asyncio.get_running_loop().call_soon(self.dev._rsp.put_nowait, answer)

    async def read_gatt_char(self, _uuid):
        return self.status

    async def start_notify(self, *_a):
        return None


def canon(x):
    if dataclasses.is_dataclass(x):
        return {f.name: canon(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, (bytes, bytearray)):
        return bytes(x).hex()
    if isinstance(x, (list, tuple)):
        return [canon(v) for v in x]
    if isinstance(x, dict):
        return {k: canon(v) for k, v in x.items()}
    return x


def case(name: str, info_bytes: bytes, method: str, args: list, *,
         status_bytes: bytes | None = None, answers: dict | None = None) -> dict:
    status_bytes = status(gain=24, rate=500) if status_bytes is None else status_bytes
    answers = answers or {}
    dev = client.Device(types.SimpleNamespace(address="02:00:00:00:00:01", name="IntoMind One", details={}))
    dev.info = P.decode_device_info(info_bytes)
    link = Link(dev, status_bytes, answers)
    dev._client = link
    call = [P.Stage(int(s["kind"]), tuple(s["params"])) for s in args[0]] if method == "set_pipeline" else None
    out = dict(name=name, info=info_bytes.hex(), status=status_bytes.hex(), method=method, args=args,
               answers={str(op): ([b.hex() for b in a] if isinstance(a, list) else a.hex())
                        for op, a in answers.items()})

    async def go():
        if call is not None:
            return await dev.set_pipeline(call)
        if method == "upload_head":
            return await dev.upload_head(bytes.fromhex(args[0]), *args[1:])
        return await getattr(dev, method)(*args)
    try:
        out["result"] = canon(asyncio.run(go()))
    except Exception as e:                               # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    out["writes"] = link.writes
    return out


def cases() -> list[dict]:
    one, bare = ONE, BARE
    hp = [dict(kind=1, params=[5])]
    return [
        case("the configuration of a device that claims everything it reads", one, "refresh_config", [],
             answers=CONFIG),
        case("the configuration of an older device: the status alone", info(minor=2, channels=2,
             caps=P.CAPABILITIES["leadoff"], length=76), "refresh_config", [],
             status_bytes=status(gain=8, rate=250, loff=3)),
        case("a gain the contract defines", one, "set_gain", [24]),
        case("a gain it does not", one, "set_gain", [3]),
        case("a gain the device refuses while streaming", one, "set_gain", [12],
             answers={P.OPCODES["set_gain"]: refused("set_gain", 3)}),
        case("a rate the device declares", one, "set_rate", [500]),
        case("a rate it does not", one, "set_rate", [333]),
        case("the test signal, by name", one, "set_mode", ["test"]),
        case("shorted inputs, by code", one, "set_mode", [2]),
        case("a mode with no name", one, "set_mode", ["loud"]),
        case("the synthetic signal on a device that does not claim it", bare, "set_mode", ["synthetic"]),
        case("lead-off detection on", one, "set_leadoff", [True]),
        case("lead-off detection on a device without it", bare, "set_leadoff", [True]),
        case("the stage catalog", one, "pipeline_catalog", [], answers=CATALOG),
        case("the chain in force", one, "pipeline", [], answers={P.OPCODES["get_pipeline"]: CONFIG[
            P.OPCODES["get_pipeline"]]}),
        case("a chain set, after reading the rate and the catalog", one, "set_pipeline", [hp],
             answers={**CONFIG, **CATALOG}),
        case("a chain the device's rate cannot run", one, "set_pipeline", [[dict(kind=1, params=[2400])]],
             answers={**CONFIG, **CATALOG}),
        case("the natural signal", one, "clear_pipeline", []),
        case("the device's default chain", one, "restore_pipeline_default", []),
        case("a chain on a device that runs none", bare, "pipeline", []),
        case("the lamp's level", one, "indicator", [], answers={P.OPCODES["get_indicator"]: CONFIG[
            P.OPCODES["get_indicator"]]}),
        case("the lamp verbose", one, "set_indicator", ["verbose"]),
        case("a lamp level with no name", one, "set_indicator", ["loud"]),
        case("identify for five seconds", one, "identify", [5]),
        case("identify for longer than the device will", one, "identify", [31]),
        case("the model's cadence", one, "model_interval", [], answers={P.OPCODES["get_model_interval"]: CONFIG[
            P.OPCODES["get_model_interval"]]}),
        case("one window every four seconds", one, "set_model_interval", [4]),
        case("an interval below the least the device keeps", one, "set_model_interval", [1],
             answers={P.OPCODES["set_model_interval"]: refused("set_model_interval", 2)}),
        case("the device's name", one, "get_name", [], answers={P.OPCODES["get_name"]: CONFIG[
            P.OPCODES["get_name"]]}),
        case("the device named", one, "set_name", ["Ada", "Blue"]),
        case("a name too long for the air", one, "set_name", ["Bartholomew", "Magnificent"]),
        case("the converter's registers", one, "converter_registers", [], answers={
            P.OPCODES["get_converter_registers"]: ok("get_converter_registers", vector(
                "the converter's twenty four registers"))}),
        case("the model", one, "model_info", [], answers=MODEL),
        case("the model on a device without one", bare, "model_info", []),
        case("the heads and the encoder each names", one, "heads", [], answers={
            P.OPCODES["list_heads"]: ok("list_heads", vector(
                "three head slots: the built-in one empty, one good, one that failed its hash")),
            P.OPCODES["list_head_encoders"]: ok("list_head_encoders", vector(
                "the encoder each of three heads names: only the good one says"))}),
        case("the heads, the encoders refused", one, "heads", [], answers={
            P.OPCODES["list_heads"]: ok("list_heads", vector(
                "three head slots: the built-in one empty, one good, one that failed its hash")),
            P.OPCODES["list_head_encoders"]: refused("list_head_encoders", 1)}),
        case("the heads of a device before 1.4", ONE_13, "heads", [], answers={
            P.OPCODES["list_heads"]: ok("list_heads", vector(
                "three head slots: the built-in one empty, one good, one that failed its hash"))}),
        case("the head in slot two selected", one, "select_head", [2]),
        case("predictions on", one, "set_predictions", [True]),
        case("predictions on with no head selected", one, "set_predictions", [True],
             answers={P.OPCODES["set_predictions"]: refused("set_predictions", 3)}),
        case("the window embedding asked for", one, "set_embeddings", ["window"], answers=MODEL),
        case("embeddings turned off", one, "set_embeddings", ["off"]),
        case("an embedding form with no name", one, "set_embeddings", ["all"]),
        case("embeddings on a device without them", bare, "set_embeddings", ["window"]),
        case("the head in slot two removed", one, "remove_head", [2]),
        case("a removal the device refuses", one, "remove_head", [1],
             answers={P.OPCODES["remove_head"]: refused("remove_head", 2)}),
        case("a removal on a device without heads", bare, "remove_head", [2]),
        case("a head for another width than the device's model", one, "upload_head",
             [M.train_head(ROWS[:, :76].tolist(), LABELS).hex(), 2]),
        case("a head on a device without heads", bare, "upload_head", [M.train_head(ROWS.tolist(), LABELS).hex(), 2]),
        case("a head that names no encoder, on a device that takes no updates", NO_UPDATE, "upload_head",
             [M.train_head(ROWS.tolist(), LABELS).hex(), 2]),
        case("a head trained beside another encoder, on a device that takes no updates", NO_UPDATE,
             "upload_head", [M.train_head(ROWS.tolist(), LABELS, encoder_id="0123456789abcdef").hex(), 2],
             answers=MODEL),
    ]


def delivery() -> dict:
    """What the library hands its embedding callback: the window embedding
    asked for, then the two notifications that carry one window, through the
    device's own notification handler. Records what each call was given."""
    pieces = [vector("the window embedding of one window"),
              vector("the same window embedding's last twelve values")]
    dev = client.Device(types.SimpleNamespace(address="02:00:00:00:00:01", name="IntoMind One", details={}))
    dev.info = P.decode_device_info(ONE)
    link = Link(dev, status(), MODEL)
    dev._client = link
    calls = []
    asyncio.run(dev.set_embeddings("window", on_window=lambda *args: calls.append(args)))
    after = []
    for piece in pieces:
        dev._on_embedding(None, piece)
        after.append(len(calls))
    return dict(name="a window of the encoder's output, from its two notifications", info=ONE.hex(),
                status=status().hex(), answers={str(op): a.hex() for op, a in MODEL.items()},
                notifications=[b.hex() for b in pieces], calls_after_each=after,
                calls=[dict(arguments=len(c), first_is_the_device=c[0] is dev, window=canon(c[1]))
                       for c in calls],
                writes=link.writes)


def collection() -> dict:
    """What the library's `collect_embeddings` writes and hands back: one
    whole window, sent as two notifications while the window form is on,
    then the form turned off and the callback put back as it was."""
    pieces = [vector("the window embedding of one window"),
              vector("the same window embedding's last twelve values")]
    dev = client.Device(types.SimpleNamespace(address="02:00:00:00:00:01", name="IntoMind One", details={}))
    dev.info = P.decode_device_info(ONE)
    link = Link(dev, status(), MODEL)
    dev._client = link
    answer = link.write_gatt_char

    async def write(uuid, payload, response=True):
        # The windows the form asks for come once the write has returned,
        # before its answer, on the same turn.
        if bytes(payload) == bytes([P.OPCODES["set_embeddings"], P.EMBEDDING_FORMS["window"]]):
            asyncio.get_running_loop().call_soon(lambda: [dev._on_embedding(None, piece) for piece in pieces])
        await answer(uuid, payload, response)

    link.write_gatt_char = write
    got = asyncio.run(dev.collect_embeddings(1))
    return dict(name="one window of the encoder's output gathered, then the form turned off", info=ONE.hex(),
                status=status().hex(), answers={str(op): a.hex() for op, a in MODEL.items()},
                notifications=[b.hex() for b in pieces], windows=canon(got), writes=link.writes,
                callback_restored=dev.on_embedding is None)


def main() -> None:
    out = dict(note="Generated by contract/device/make.py from the Python library. Do not edit.",
               cases=cases(), delivery=delivery(), collection=collection())
    (HERE / "cases.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(out['cases'])} cases")


if __name__ == "__main__":
    main()
