"""The several-devices fixture: the Python library's multi-device `Session`,
driven against stand-in devices on scripted clocks, and everything it did,
which every other implementation's counterpart is held to.

    PYTHONPATH=<a checkout of the Python library> python3 contract/together/make.py

Three parts. The files: `Session.record_to` with devices of different
channel counts streaming at once, the packets each sent in the order they
arrived, and the bytes of every file written, with a program's own callback
on one device, the function that stops writing, a second file, one device
let go with `Session.disconnect`, and a device with more channels than the
header names refused by `start_all` before any device is started.

The runs: `Session.run` with an experiment on two to four devices at once,
each device the IntoMind One's Device Info from the conformance vectors with
a hardware version and a device id that are plainly made up, through the
library's own `client.Device` and experiment code. The host's clocks move
only while a run waits, and by exactly the time it waits. While a device
streams it sends one packet each wait, of counts from a seeded source. None
is a person's. A person's acts are scripted by the wait they come after: a
marker, the Stop button, one device's link dropping, or USB power appearing
on one device, which ends its stream. A device may refuse an opcode, or a
form of embeddings, or be down before the run. Each device's gain in
force, which its Status and packets state, is the gain its run's setup
gives it, but in one run, where the second device's setup gives another
and the run is refused before anything is sent to either. Head training asks each
device for the encoder's output, and each device sends its own windows, as
the experiments fixture sends them, a person's own three states among them,
and the link of a device alone drops while its windows are waited for.
Every experiment is given one parameter it cannot use on two devices, and
refuses it by name before anything is sent to either.

The programs: what a program does from the scan. Devices are on the air,
the person picks some, they are connected into one Session, each through
its own clock exchange, and the program says who wears each. Then one run
on all of them, two runs at once on different devices, one experiment that
hands two groups two experiments at the same moment under one Stop, one
that has one group work while the other rests and then the reverse, a run
naming its devices and the refusals of a device that is not the session's
or is named twice, the refusal of a device another run is recording, the
events of each step in its own devices' recordings, names already taken
numbered, a device connected after its run started, an experiment's own
messages, a device whose link drops in the middle of head training and
comes back for the next run, and a device that refuses to turn embeddings
off until it does not. A person's acts are done outside any experiment, by
the person's own hand. The experiments an application registers for them
are written out here, and every other implementation writes its own the
same.

For each run and program it writes the trace: every clock exchange at a
connection, every command written to each device, every read of a device's
status, every wait with the packets each device sent during it and what the
person did, every delivery of windows, every cue played and every message
each run sent, in order, then each recording the library made, its files,
what the run gave back and the error it raised.

Writes together.json beside this script. Run it again and nothing changes
unless the library does.
"""
from __future__ import annotations

import asyncio
import contextvars
import importlib.util
import json
import pathlib
import random
import shutil
import stat
import struct
import sys
import tempfile
import types

HERE = pathlib.Path(__file__).resolve().parent
# Read as a module, the experiments fixture leaves no compiled copy beside it.
sys.dont_write_bytecode = True
# Made without a sound: the library's experiments speak their cues and play
# their tones through the host's speakers, so every cue goes nowhere here,
# before anything of the library runs.
from intomind import audio as _audio  # noqa: E402
_audio.play = lambda tone, text="": None
# The experiments fixture's clocks, archive stamps, software names, Device
# Info, answers and seeded streams, so both fixtures stand on one footing.
_spec = importlib.util.spec_from_file_location("experiments_fixture", HERE.parent / "experiments" / "make.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)

import intomind  # noqa: E402
from intomind import client, experiments, protocol as P, provenance as prov  # noqa: E402
from intomind import session as S  # noqa: E402
from intomind.session import Session  # noqa: E402

OP = P.OPCODES
CLOCKS = F.CLOCKS
#: The event loop's own wait, kept before anything stands in for the experiments module's.
_YIELD = asyncio.sleep


def _info(base: bytes, n: int) -> bytes:
    """A Device Info with the device id plainly made up: eight bytes ending in n."""
    b = bytearray(base)
    b[19:27] = bytes(7) + bytes([0xA0 + n])
    return bytes(b)


def _eight() -> bytes:
    """The IntoMind One's Device Info with eight channels."""
    b = bytearray(F.INFOS["one"])
    b[5] = 8
    return bytes(b)


INFOS = dict(F.INFOS, eight=_eight())


#: The experiments fixture's stream, USB power included.
Stream = F.Stream


class Link:
    """One device's radio: answers each control write as the device would,
    refuses what the case says it refuses, an opcode or a form of
    embeddings, answers a stop with ok whether or not the device is
    streaming, as the firmware does, and moves the host's clocks across a
    time exchange as the script says. The device acknowledges a write first
    and handles what it carried afterwards, so the windows a write asks for
    and its answer come on the loop's next turn, once the write has
    returned. A link lost after a rep's windows drops once the answer to
    the request that asked for them is given, since nothing reaches the
    host on a link that is down."""

    def __init__(self, run, i):
        self.run, self.i, self.is_connected = run, i, True
        self.drop_after_answer = False

    async def write_gatt_char(self, _uuid, payload, response=True):
        payload = bytes(payload)
        op = payload[0]
        u = self.run.units[self.i]
        if self.run.tracing and not u.connecting:
            self.run.trace.append(["write", self.i, payload.hex()])
        refused = op in u.refuse or (op == OP["set_embeddings"]
                                     and payload[1] in {P.EMBEDDING_FORMS[f] for f in u.refuse_forms})
        if op == OP["time_sync"]:
            after, ticks = u.sync
            CLOCKS.set(after)
            answer = bytes([op, 0]) + int(ticks).to_bytes(8, "little")
        elif refused:
            answer = bytes([op, 1])
        else:
            answer = u.answers.get(op) or bytes([op, 0])
            u.stream.command(op)
        windows = op == OP["set_embeddings"] and payload[1] == P.EMBEDDING_FORMS["window"] and self.is_connected \
            and not refused

        def handled():
            if windows:
                _deliver(self.run, self.i)
            u.dev._rsp.put_nowait(answer)
            if self.drop_after_answer:
                self.is_connected, self.drop_after_answer = False, False
        asyncio.get_running_loop().call_soon(handled)

    async def read_gatt_char(self, _uuid):
        if self.run.tracing:
            self.run.trace.append(["read", self.i, "status"])
        return self.run.units[self.i].status

    async def start_notify(self, *_a):
        return None


RUN = None


def _act(run, a) -> None:
    """One thing a person does: a marker or an event in a run's context, its
    Stop, one device's link dropping, or USB power appearing on one device.
    In a run of the runs part the context is the run's own, and in a program
    it is named."""
    what = a[0]
    if what == "marker":
        (run.ctx if len(a) == 2 else run.ctxs[a[1]]).marker(a[-1])
    elif what == "event":
        run.ctxs[a[1]].event(a[2], a[3], **a[4])
    elif what == "stop":
        (run.ctx if len(a) == 1 else run.ctxs[a[1]]).stop.set()
    elif what == "link_lost":
        run.units[a[1]].link.is_connected = False
    elif what == "usb":
        run.units[a[1]].stream.usb = True


async def _sleep(seconds) -> None:
    """A wait: the clocks move by exactly that, one packet arrives from every
    device streaming with its link up, then whatever the person does after
    this wait, by the person's own hand, and then whatever else is running
    takes its turn."""
    run = RUN
    run.sleeps += 1
    CLOCKS.advance(seconds)
    packets = []
    for i, u in enumerate(run.units):
        if u.link.is_connected and u.stream.streaming:
            h = u.stream.next()
            u.dev._on_data(None, bytes.fromhex(h))
            packets.append([i, h])
    acts = run.actions.get(run.sleeps, [])
    for a in acts:
        if run.program:
            contextvars.Context().run(_act, run, a)
        else:
            _act(run, a)
    run.trace.append(["sleep", seconds, packets, acts])
    if run.program:
        await _YIELD(0)


def _deliver(run, i) -> None:
    """The windows one device sends for one ask of the window form, as the
    experiments fixture sends them, from that device's own seeded source,
    then what happens after this delivery."""
    u = run.units[i]
    w = u.windows
    rep = u.deliveries
    u.deliveries += 1
    dim = u.dev.info.model_embed_dim
    states = w.get("states", 2)

    def lean(j):
        if states == 2:
            return (2000 if rep % 2 else -2000) if j < 12 else 0
        return 2000 if j // 12 == rep % states else 0
    sent = []
    for k in range(w.get("short", {}).get(str(rep), w.get("per_set", 0))):
        CLOCKS.advance(4.0)
        packet = None
        if u.link.is_connected and u.stream.streaming:
            packet = u.stream.next()
            u.dev._on_data(None, bytes.fromhex(packet))
        flags = sum(P.EMBEDDING_FLAGS[f] for f in w.get("flags", {}).get(f"{rep}:{k}", []))
        values = [u.rng.randint(-900, 900) + lean(j) for j in range(dim)]
        body = P._EMBEDDING.pack(P.PACKET_EMBEDDING, flags, dim, 0, u.window_index,
                                 4_000_000_000 + 4_000_000 * u.window_index, 2000,
                                 bytes.fromhex(F.READY[4:12].hex()), 0, P.WINDOW_TOKEN)
        body += struct.pack(f"<{dim}h", *values)
        u.window_index += 1
        u.dev._on_embedding(None, body)
        sent.append([packet, body.hex()])
    acts = w.get("after", {}).get(str(rep), [])
    for a in acts:
        if a[0] == "stop":
            run.ctx.stop.set()
        elif a[0] == "link_lost":
            # Once the device has answered the request that asked for the
            # windows: nothing reaches the host on a link that is down.
            u.link.drop_after_answer = True
        elif a[0] == "usb":
            u.stream.usb = True
    run.trace.append(["embeddings", i, sent, acts])


experiments.asyncio = types.SimpleNamespace(sleep=_sleep, Event=asyncio.Event)
experiments.cue_sound = lambda tone, text="": RUN.trace.append(["cue", tone, text])


class Unit(client.Device):
    """The library's own Device on a stand-in radio. Connected as a person's
    pick is connected, it takes its clock exchange at the host's clocks now,
    as `Device.connect` ends with one, and the trace says when and how."""

    async def connect(self):
        u = self._unit
        before = [CLOCKS.mono, CLOCKS.real]
        after = [before[0] + 0.03, before[1] + 0.03]
        u.sync = (after, u.clock["ticks"])
        u.connecting = True
        try:
            await self.time_sync()
        finally:
            u.connecting = False
        RUN.trace.append(["connect", u.i, before, after, u.clock["ticks"]])
        return self


def _unit(run, i, k, spec, cls=client.Device):
    """One stand-in device: its Device Info, answers, status, clock
    exchange, stream, windows and what it refuses, on its own radio."""
    info_bytes = _info(INFOS[spec.get("info", "one")], i + 1)
    info = P.decode_device_info(info_bytes)
    rate = spec.get("rate", 500)
    gain = spec.get("gain", 24)
    dev = cls(types.SimpleNamespace(address=f"02:00:00:00:00:{i + 1:02x}", name="IntoMind One", details={}))
    dev.info = info
    start = [5000.0 + 1000.0 * k + 0.25 * i, 1_767_225_600.0 + 1000.0 * k + 0.25 * i]
    u = types.SimpleNamespace(
        i=i, dev=dev, info_bytes=info_bytes,
        answers=F.ANSWERS[spec.get("answers", spec.get("info", "one"))],
        status=F.status(gain=gain, rate=rate),
        stream=Stream(100 * (k + 1) + i, channels=info.channels, rate=rate, gain=gain),
        sync=None, link=None, windows=spec.get("windows", {}), deliveries=0,
        rng=random.Random(5000 + 10 * k + i), window_index=100 + 1000 * i,
        streaming=bool(spec.get("streaming")), setup_packets=[], connecting=False,
        refuse={OP[o] for o in spec.get("refuse", [])}, refuse_forms=set(spec.get("refuse_forms", [])),
        down=bool(spec.get("down")),
        clock=dict(before=start, after=[start[0] + 0.03, start[1] + 0.03],
                   ticks=4_000_000_000 + 15_000 + 7_000 * i))
    u.link = Link(run, i)
    dev._client = u.link
    dev._unit = u
    return u


def _described(u) -> dict:
    """A unit as another implementation's stand-in is built from it."""
    return dict(info=u.info_bytes.hex(), answers={str(op): a.hex() for op, a in u.answers.items()},
                status=u.status.hex(), sync=u.clock, streaming=u.streaming, setup_packets=u.setup_packets,
                windows_per_set=u.windows.get("per_set", 0), refuse=sorted(u.refuse),
                refuse_forms=sorted(u.refuse_forms), down=u.down)


def _emit_into(run, who=None):
    """An emit that writes what it hears into the trace: a saved message as
    its name, the devices it concerns and what a head training adds, every
    other message as it is."""
    async def emit(msg):
        if msg.get("type") == "saved":
            kept = dict(type="saved", label=msg["label"])
            if "head" in msg:
                kept.update(head=msg["head"], score=msg["score"], no_head=msg["no_head"])
            if "devices" in msg:
                kept["devices"] = msg["devices"]
        else:
            kept = msg
        run.trace.append(["emit", kept] if who is None else ["emit", who, kept])
    return emit


def _recordings(captures: pathlib.Path, labels) -> list:
    out = []
    for label in labels:
        ev = captures / f"{label}.events.json"
        sums = captures / f"{label}.sha256"
        taught = captures / f"{label}.training.json"
        out.append(dict(
            label=label, meta_json=(captures / f"{label}.meta.json").read_text(),
            events_json=ev.read_text() if ev.exists() else None,
            training_json=taught.read_text() if taught.exists() else None,
            sha256_names=([line.split(None, 1)[1] for line in sums.read_text().splitlines()]
                          if sums.exists() else None),
            arrays=F._digest(captures / f"{label}.npz")))
    return out


def _files(captures: pathlib.Path) -> dict:
    return {p.name: dict(read_only=not (p.stat().st_mode & stat.S_IWUSR))
            for p in (sorted(captures.iterdir()) if captures.exists() else [])}


def _fresh(captures: pathlib.Path) -> None:
    prov.use_captures_dir(captures)
    prov.use_site(None)
    prov.use_repo(None)


# ------------------------------------------------------------------ the runs
async def run_one(spec_run: dict, k: int, captures: pathlib.Path) -> dict:
    global RUN
    _fresh(captures)
    spec = experiments.REGISTRY[spec_run["experiment"]]
    run = types.SimpleNamespace(trace=[], tracing=False, sleeps=0, units=[], ctx=None, ctxs={}, program=False,
                                actions={int(i): a for i, a in spec_run.get("actions", {}).items()})
    RUN = run
    for i, unit in enumerate(spec_run["units"]):
        run.units.append(_unit(run, i, k, unit))
    for u in run.units:
        CLOCKS.set(u.clock["before"])
        u.sync = (u.clock["after"], u.clock["ticks"])
        await u.dev.time_sync()
        if u.streaming:
            await u.dev.start()
            for _ in range(2):
                h = u.stream.next()
                u.dev._on_data(None, bytes.fromhex(h))
                u.setup_packets.append(h)
        if u.down:
            u.link.is_connected = False

    params = F.page_params(spec, spec_run.get("params", {}))
    cfgs = [dict(c) for c in spec_run["cfgs"]]
    emit = _emit_into(run)
    s = Session()
    s.devices = [u.dev for u in run.units]
    ctx = experiments.RunCtx(emit, cfgs[0])
    run.ctx = ctx
    run.tracing = True
    out = dict(labels=None, raised=None)
    try:
        out["labels"] = await s.run(spec_run["experiment"], params, emit, cfgs=cfgs, ctx=ctx)
    except Exception as e:  # noqa: BLE001
        out["raised"] = f"{type(e).__name__}: {e}"
    run.tracing = False
    out["trace"] = run.trace
    out["files"] = _files(captures)
    out["recordings"] = _recordings(captures, out["labels"] or [])
    out["sessions_jsonl"] = ((captures / "SESSIONS.jsonl").read_text()
                             if (captures / "SESSIONS.jsonl").exists() else None)
    out["left"] = [_left(u.dev) for u in run.units]
    return dict(name=spec_run["name"], experiment=spec_run["experiment"], params=params, cfgs=cfgs,
                actions={str(i): a for i, a in run.actions.items()},
                units=[_described(u) for u in run.units], expected=out)


def _left(dev) -> dict:
    """What a run left on a device: whether it streams, by this side's
    account, whether anything still says it is being recorded or keeps
    what was, and the form of embeddings this side holds it to."""
    return dict(streaming=bool(dev.streaming), held=id(dev) in experiments._RECORDING,
                tapped=any(dev._taps.values()), embeddings=dev._embeddings_form)


def runs() -> list[dict]:
    one = dict(info="one")
    ready = dict(info="one", answers="one, model ready")

    def cfg(gain):
        return dict(gain=gain, rate_sps=500, mode="normal", leadoff=None, tier="basic")

    def at(unit, gain):
        """`unit` with `gain` in force, which its Status and packets state."""
        return dict(unit, gain=gain)
    short = dict(trials=2, include_sham=0, hold_s=0.1, rest_s=0.1)
    return [
        dict(name="blinks on two devices", experiment="blink", units=[one, at(one, 12)],
             params=dict(trials=3, include_sham=0, hold_s=0.1, rest_s=0.1), cfgs=[cfg(24), cfg(12)]),
        dict(name="a baseline on three devices, one already streaming", experiment="baseline",
             units=[one, dict(one, streaming=True), at(one, 6)], params=dict(seconds=0.6),
             cfgs=[cfg(24), dict(cfg(24), rld="on"), cfg(6)]),
        dict(name="an open recording on two devices with a marker, stopped", experiment="record",
             units=[one, one], params=dict(markers="a, b", note="two heads"), cfgs=[cfg(24), cfg(24)],
             actions={"2": [["marker", "a"]], "4": [["marker", "b"]], "6": [["stop"]]}),
        dict(name="an open recording on three devices, the second one's link drops", experiment="record",
             units=[one, at(one, 12), one], params=dict(max_seconds=3), cfgs=[cfg(24), cfg(12), cfg(24)],
             actions={"3": [["link_lost", 1]], "7": [["stop"]]}),
        dict(name="alpha blocking on two devices, stopped", experiment="berger", units=[one, one],
             params=dict(block=1, reps=3), cfgs=[cfg(24), cfg(24)], actions={"6": [["stop"]]}),
        dict(name="a head trained on two devices, eyes open and closed", experiment="head_training",
             units=[dict(ready, windows=dict(per_set=17, flags={"1:5": ["gap_in_window"]})),
                    dict(ready, gain=12, windows=dict(per_set=17, flags={"2:3": ["leadoff_in_window"]}))],
             params=dict(pair="eyes", rep_length=13, reps=8), cfgs=[cfg(24), cfg(12)]),
        dict(name="a head trained on two devices, the second one's link drops after its first rep",
             experiment="head_training",
             units=[dict(ready, windows=dict(per_set=17)),
                    dict(ready, windows=dict(per_set=17, after={"0": [["link_lost"]]}))],
             params=dict(rep_length=13, reps=8), cfgs=[cfg(24), cfg(24)]),
        dict(name="head training on two devices, the second one's model not ready", experiment="head_training",
             units=[ready, dict(one, answers="one, model loading")], params={}, cfgs=[cfg(24), cfg(24)]),
        dict(name="a baseline on three devices, the second one's link down when it starts", experiment="baseline",
             units=[one, dict(one, down=True), one], params=dict(seconds=0.5), cfgs=[cfg(24), cfg(24), cfg(24)]),
        dict(name="blinks on two devices, the second refuses to start, so nothing is left armed or started",
             experiment="blink", units=[one, dict(one, refuse=["start_stream"])], params=short,
             cfgs=[cfg(24), cfg(24)]),
        dict(name="an open recording on two devices, USB power ends the second one's stream", experiment="record",
             units=[one, one], params=dict(max_seconds=4), cfgs=[cfg(24), cfg(24)],
             actions={"4": [["usb", 1]], "12": [["stop"]]}),
        dict(name="a baseline on four devices, each saying who wears it", experiment="baseline",
             units=[one, at(one, 12), at(one, 12), at(one, 12)], params=dict(seconds=0.5),
             cfgs=[dict(gain=24, subject_id="S01", notes="behind the left ear"), dict(gain=12, subject_id="S02"),
                   dict(gain=12), dict(gain=12, subject_id=None)]),
        dict(name="alpha blocking on two devices, the first already streaming, which it leaves streaming",
             experiment="berger", units=[dict(one, streaming=True), one], params=dict(block=1, reps=1),
             cfgs=[cfg(24), cfg(24)]),
        dict(name="a custom protocol on two devices, the first already streaming", experiment="protocol",
             units=[dict(one, streaming=True), one], params=dict(steps="Close your eyes | 1", reps=1, lead_in=0),
             cfgs=[cfg(24), cfg(24)]),
        dict(name="eye movements on two devices, the first already streaming", experiment="saccade",
             units=[dict(one, streaming=True), one], params=dict(trials=2, include_sham=0, hold_s=0.1, center_s=0.1),
             cfgs=[cfg(24), cfg(24)]),
        dict(name="jaw clenches on two devices, the first already streaming", experiment="jaw_clench",
             units=[dict(one, streaming=True), one], params=short, cfgs=[cfg(24), cfg(24)]),
        dict(name="head training on two devices, the first already streaming", experiment="head_training",
             units=[dict(ready, streaming=True, windows=dict(per_set=6)), dict(ready, windows=dict(per_set=6))],
             params=dict(pair="eyes", rep_length=2, reps=2), cfgs=[cfg(24), cfg(24)]),
        dict(name="head training on two devices, USB power ends the second one's stream after its second rep",
             experiment="head_training",
             units=[dict(ready, windows=dict(per_set=17)),
                    dict(ready, windows=dict(per_set=17, after={"1": [["usb"]]}))],
             params=dict(pair="eyes", rep_length=13, reps=4), cfgs=[cfg(24), cfg(24)]),
        dict(name="head training on three devices, the second one's link down when it starts",
             experiment="head_training",
             units=[dict(ready, windows=dict(per_set=17)), dict(ready, down=True, windows=dict(per_set=17)),
                    dict(ready, windows=dict(per_set=17))],
             params=dict(pair="eyes", rep_length=13, reps=2), cfgs=[cfg(24), cfg(24), cfg(24)]),
        dict(name="head training on one device, USB power ends its stream after its second rep",
             experiment="head_training", units=[dict(ready, windows=dict(per_set=17, after={"1": [["usb"]]}))],
             params=dict(pair="eyes", rep_length=13, reps=4), cfgs=[cfg(24)]),
        dict(name="head training on two devices on three of a person's own states", experiment="head_training",
             units=[dict(ready, windows=dict(per_set=16, states=3)),
                    dict(ready, gain=12, windows=dict(per_set=16, states=3))],
             params=dict(pair="own", states="Red | Picture the color red.\nBlue | Picture the color blue.\nBlank | Picture nothing.",
                         rep_length=12, reps=6),
             cfgs=[cfg(24), cfg(12)]),
        dict(name="own states written as one line refused on two devices before anything is sent",
             experiment="head_training", units=[dict(ready, windows=dict(per_set=17))] * 2,
             params=dict(pair="own", states="Focus | Look at the dot."), cfgs=[cfg(24), cfg(24)]),
        dict(name="head training on one device whose link drops between reps", experiment="head_training",
             units=[dict(ready, windows=dict(per_set=17, after={"0": [["link_lost"]]}))],
             params=dict(pair="eyes", rep_length=13, reps=4), cfgs=[cfg(24)]),
        dict(name="head training on one device whose link drops while its second rep's windows are waited for",
             experiment="head_training",
             units=[dict(ready, windows=dict(per_set=17, short={"1": 9}, after={"1": [["link_lost"]]}))],
             params=dict(pair="eyes", rep_length=13, reps=4), cfgs=[cfg(24)]),
        *[dict(name=f"{experiment} on two devices given {what} for {param} refuses it by name before anything is sent",
               experiment=experiment, units=[dict(ready, windows=dict(per_set=6))] * 2,
               params={**F.SHORT[experiment], param: value}, cfgs=[cfg(24), cfg(24)])
          for experiment, cases in F.BAD.items() for param, value, what in cases[:1]],
        dict(name="blinks on two devices whose second one's setup gives a gain other than its gain in force, "
                  "refused before anything is sent to either",
             experiment="blink", units=[one, one], params=short, cfgs=[cfg(24), cfg(12)]),
    ]


# -------------------------------------------------------------- the programs
#: An experiment's parameters as a page's form sends them: every default, then what is given.
def _given(name, **given):
    return F.page_params(experiments.REGISTRY[name], given)


#: The steps the programs' experiments hand their groups, by name, with their parameters.
STEPS = dict(
    berger_long=("berger", dict(block=2, reps=10)),
    baseline_long=("baseline", dict(seconds=600)),
    berger_short=("berger", dict(block=1, reps=1)),
    blink_short=("blink", dict(trials=2, include_sham=0, rest_s=0.1, hold_s=0.1)),
    saccade_short=("saccade", dict(trials=2, include_sham=0, hold_s=0.1, center_s=0.1)),
    baseline_quarter=("baseline", dict(seconds=0.25)),
    baseline_one=("baseline", dict(seconds=1)),
)


def _step(key):
    name, given = STEPS[key]
    return experiments.REGISTRY[name]["fn"], _given(name, **given)


async def two_groups(dev, rec, emit, p, ctx):
    """One group in alpha blocking while the other records at rest, at the
    same moment."""
    (berger, pa), (baseline, pb) = _step("berger_long"), _step("baseline_long")
    async with asyncio.TaskGroup() as group:
        first = group.create_task(berger(dev[:2], rec, emit, pa, ctx))
        second = group.create_task(baseline(dev[2:], rec, emit, pb, ctx))
    return first.result() + second.result()


async def swap(dev, rec, emit, p, ctx):
    """Half work while half rest, then they swap."""
    berger, pa = _step("berger_short")
    await emit(dict(type="note", text="the first group works"))
    first = await berger(dev[:2], rec, emit, pa, ctx)
    second = await berger(dev[2:], rec, emit, pa, ctx)
    return first + second


async def in_turn(dev, rec, emit, p, ctx):
    """One device blinks, then the other looks left and right, the
    experiment's own cues between."""
    (blink, pa), (saccade, pb) = _step("blink_short"), _step("saccade_short")
    ctx.cue("begin")
    first = await blink(dev[:1], rec, emit, pa, ctx)
    ctx.cue("swap")
    second = await saccade(dev[1:], rec, emit, pb, ctx)
    return first + second


async def twice(dev, rec, emit, p, ctx):
    """The second device handed to one step twice."""
    baseline, pa = _step("baseline_one")
    return await baseline([dev[1], dev[1]], rec, emit, pa, ctx)


async def again(dev, rec, emit, p, ctx):
    """Two baselines in the same second, so the second's name is taken."""
    baseline, pa = _step("baseline_quarter")
    first = await baseline(dev, rec, emit, pa, ctx)
    second = await baseline(dev, rec, emit, pa, ctx)
    return first + second


async def late(dev, rec, emit, p, ctx):
    """A device connected while the run goes on, recorded beside the run's own."""
    baseline, pa = _step("baseline_one")
    s = RUN.session
    await s.connect(RUN.found[1])
    return await baseline([dev, s.devices[-1]], rec, emit, pa, ctx)


async def comeback(run, s):
    """A device whose link drops in the middle of head training is asked for
    its windows again by the next run, once the link is back, and trains
    its head."""
    await s.connect(run.found[0])
    run.tracing = True
    raised = None
    try:
        await s.run("head_training", _given("head_training", rep_length=40, reps=2), _emit_into(run, "dropped"))
    except ConnectionError as e:
        raised = f"{type(e).__name__}: {e}"
    run.units[0].link.is_connected = True
    names = await s.run("head_training", _given("head_training", rep_length=40, reps=2), _emit_into(run, "back"))
    return dict(raised=raised, names=names)


async def refused_off(run, s):
    """A turn off the device refuses leaves this side believing embeddings
    on, and the next rep still asks for its windows. Once the device takes
    a turn off, the form found is put back."""
    await s.connect(run.found[0])
    run.tracing = True
    first = await s.run("head_training", _given("head_training", rep_length=2, reps=2), _emit_into(run, "refusing"))
    run.units[0].refuse_forms = set()
    second = await s.run("head_training", _given("head_training", rep_length=2, reps=2), _emit_into(run, "taken"))
    return dict(first=first, second=second)


async def outside(dev, rec, emit, p, ctx):
    """Messages of the experiment's own, around one step."""
    baseline, pa = _step("baseline_one")
    await emit(dict(type="note", text="before any step"))
    got = await baseline(dev[1:], rec, emit, pa, ctx)
    await emit(dict(type="note", text="after it", devices=["as the experiment says"]))
    return got


#: The experiments the programs' application registers, as written above.
APPLICATION = dict(two_groups=two_groups, swap=swap, in_turn=in_turn, twice=twice, again=again, late=late,
                   outside=outside)


def _ctx(run, name, cfg):
    run.ctxs[name] = experiments.RunCtx(None, cfg)
    return run.ctxs[name]


async def everyone(run, s):
    """Four heard, three picked, connected into one Session with the wall
    clock stepped between connections, each wearer said, and one run naming
    no devices that one Stop ends."""
    await s.connect(run.found[0])
    for bd in (run.found[2], run.found[3]):
        CLOCKS.real += 2.5                       # the wall clock is stepped between connections
        await s.connect(bd)
    run.tracing = True
    wearers = [dict(gain=24, subject_id=w) for w in ("S01", "S02", "S03")]
    ctx = _ctx(run, "run", dict(gain=24))
    return await s.run("record", _given("record", max_seconds=600), _emit_into(run, "run"), cfgs=wearers, ctx=ctx)


async def two_runs(run, s):
    """Two runs at once naming different devices, each with its own Stop
    and markers."""
    await s.connect(*run.found)
    run.tracing = True
    one, two, three = s.devices
    first, second = _ctx(run, "first", dict(gain=24)), _ctx(run, "second", dict(gain=24))
    return await asyncio.gather(
        s.run("record", _given("record", max_seconds=60), _emit_into(run, "first"),
              cfgs=[dict(subject_id="S01")], ctx=first, devices=[one]),
        s.run("record", _given("record", max_seconds=60), _emit_into(run, "second"),
              cfgs=[dict(subject_id="S02"), dict(subject_id="S03")], ctx=second, devices=[two, three]))


async def groups(run, s):
    """One experiment hands one group alpha blocking and the other a
    resting recording at the same moment, under one Stop."""
    await s.connect(*run.found)
    run.tracing = True
    wearers = [dict(gain=24, subject_id=w) for w in ("S01", "S02", "S03", "S04")]
    ctx = _ctx(run, "run", dict(gain=24))
    return await s.run("two_groups", {}, _emit_into(run, "run"), cfgs=wearers, ctx=ctx)


async def resting(run, s):
    """Every device streams, then one group works while the other rests,
    and then the reverse."""
    await s.connect(*run.found)
    run.tracing = True
    await s.start_all()
    wearers = [dict(gain=24, subject_id=w) for w in ("S01", "S02", "S03", "S04")]
    ctx = _ctx(run, "run", dict(gain=24))
    return await s.run("swap", {}, _emit_into(run, "run"), cfgs=wearers, ctx=ctx)


async def naming(run, s):
    """A run on the devices it names, given as a list, a tuple, a set or one
    device, and the refusals of a device that is not the session's or is
    named twice, before anything is sent."""
    await s.connect(*run.found[:2])
    run.tracing = True
    first, second = s.devices
    stranger = run.units[2].dev
    names = []
    for given in ([second], (second,), {second}, second):
        names.append(await s.run("baseline", _given("baseline", seconds=1), _emit_into(run, "run"),
                                 devices=given))
    refused = []
    for devices in ([first, stranger], [second, first, second]):
        try:
            await s.run("baseline", _given("baseline", seconds=1), _emit_into(run, "refused"), devices=devices)
        except ValueError as e:
            refused.append(f"{type(e).__name__}: {e}")
    return dict(names=names, refused=refused)


async def busy(run, s):
    """A device another run is recording is refused by name, so is one
    handed to a step twice, and once its recording ends it is free again."""
    await s.connect(*run.found)
    run.tracing = True
    one, _two = s.devices
    first_ctx = _ctx(run, "first", dict(gain=24))
    first = asyncio.ensure_future(s.run("record", _given("record", max_seconds=60), _emit_into(run, "first"),
                                        ctx=first_ctx, devices=[one]))
    for _ in range(3):
        await _YIELD(0)                          # the first run is recording the first device
    refused = []
    for name, params in (("baseline", _given("baseline", seconds=1)), ("twice", {})):
        try:
            await s.run(name, params, _emit_into(run, name))
        except RuntimeError as e:
            refused.append(f"{type(e).__name__}: {e}")
    names = await first
    again_ = await s.run("baseline", _given("baseline", seconds=1), _emit_into(run, "again"), devices=[one])
    return dict(refused=refused, names=names, again=again_)


async def turns(run, s):
    """The events each step stamps in its own devices' recordings, and every
    event stamped outside any step in every recording."""
    await s.connect(*run.found)
    run.tracing = True
    ctx = _ctx(run, "run", dict(gain=24))
    return await s.run("in_turn", {}, _emit_into(run, "run"), ctx=ctx)


async def taken(run, s):
    """A name already taken has a number added, on several devices and on one."""
    await s.connect(*run.found)
    run.tracing = True
    several = await s.run("again", {}, _emit_into(run, "several"), cfgs=[dict(gain=24)] * 2)
    one = await s.run("again", {}, _emit_into(run, "one"), devices=[s.devices[0]])
    return dict(several=several, one=one)


async def latecomer(run, s):
    """A device connected after the run started is recorded beside its own."""
    await s.connect(run.found[0])
    run.tracing = True
    return await s.run("late", {}, _emit_into(run, "run"))


async def own_messages(run, s):
    await s.connect(*run.found)
    run.tracing = True
    return await s.run("outside", {}, _emit_into(run, "run"))


PROGRAMS = dict(everyone=everyone, two_runs=two_runs, groups=groups, resting=resting, naming=naming, busy=busy,
                turns=turns, taken=taken, latecomer=latecomer, own_messages=own_messages, comeback=comeback,
                refused_off=refused_off)


async def play(spec_program: dict, k: int, captures: pathlib.Path) -> dict:
    global RUN
    _fresh(captures)
    run = types.SimpleNamespace(trace=[], tracing=False, sleeps=0, units=[], ctx=None, ctxs={}, program=True,
                                actions={int(i): a for i, a in spec_program.get("actions", {}).items()},
                                session=None, found=[])
    RUN = run
    for i, unit in enumerate(spec_program["units"]):
        run.units.append(_unit(run, i, k, unit, cls=Unit))
    run.found = [u.dev._bd for u in run.units]
    on_air = {u.dev._bd.address: u.dev for u in run.units}
    CLOCKS.set([5000.0 + 1000.0 * k, 1_767_225_600.0 + 1000.0 * k])
    saved_device = S.Device
    S.Device = lambda bd: on_air[bd.address]
    added = []
    for name in spec_program.get("application", []):
        experiments.experiment(name, APPLICATION[name].__doc__.split("\n")[0].strip(), [])(APPLICATION[name])
        added.append(name)
    out = dict(gave=None, raised=None)
    try:
        run.session = s = Session()
        try:
            out["gave"] = await PROGRAMS[spec_program["program"]](run, s)
        except Exception as e:  # noqa: BLE001
            out["raised"] = f"{type(e).__name__}: {e}"
    finally:
        S.Device = saved_device
        for name in added:
            experiments.REGISTRY.pop(name, None)
    run.tracing = False
    out["trace"] = run.trace
    out["files"] = _files(captures)
    labels = sorted(p.name[:-len(".meta.json")] for p in captures.iterdir() if p.name.endswith(".meta.json")) \
        if captures.exists() else []
    out["recordings"] = _recordings(captures, labels)
    out["sessions_jsonl"] = ((captures / "SESSIONS.jsonl").read_text()
                             if (captures / "SESSIONS.jsonl").exists() else None)
    out["devices"] = [u.i for u in run.units if any(u.dev is d for d in s.devices)]
    out["left"] = [_left(u.dev) for u in run.units]
    return dict(name=spec_program["name"], program=spec_program["program"],
                application=spec_program.get("application", []),
                steps={key: dict(experiment=n, params=_given(n, **g)) for key, (n, g) in STEPS.items()},
                actions={str(i): a for i, a in run.actions.items()},
                units=[_described(u) for u in run.units], expected=out)


def programs() -> list[dict]:
    one = dict(info="one")
    return [
        dict(name="everyone does the same thing: one run naming no devices, one Stop", program="everyone",
             units=[one] * 4, actions={"9": [["stop", "run"]]}),
        dict(name="two runs at once on different devices, each with its own Stop and markers", program="two_runs",
             units=[one] * 3,
             actions={"3": [["marker", "first", "first marker"]], "4": [["marker", "second", "second marker"]],
                      "12": [["stop", "first"]], "40": [["stop", "second"]]}),
        dict(name="one experiment hands two groups two experiments at once, one Stop ending both", program="groups",
             units=[one] * 4, application=["two_groups"],
             actions={"30": [["marker", "run", "pressed"]], "60": [["stop", "run"]]}),
        dict(name="one group works while the other rests, then the reverse", program="resting", units=[one] * 4,
             application=["swap"]),
        dict(name="a run on the devices it names, and a device not the session's or named twice refused",
             program="naming", units=[one] * 3),
        dict(name="a device another run is recording refused, and free again once that recording ends",
             program="busy", units=[one] * 2, application=["twice"], actions={"20": [["stop", "first"]]}),
        dict(name="each step's events in its own devices' recordings, and every outside event in all",
             program="turns", units=[one] * 2, application=["in_turn"],
             actions={"2": [["marker", "run", "pressed"]], "3": [["event", "run", "button", "a", dict(value=1)]]}),
        dict(name="a name already taken numbered, on several devices and on one", program="taken", units=[one] * 2,
             application=["again"]),
        dict(name="a device connected after its run started, recorded beside the run's own", program="latecomer",
             units=[one] * 2, application=["late"]),
        dict(name="an experiment's own messages around one step", program="own_messages", units=[one] * 2,
             application=["outside"]),
        dict(name="a device whose link drops in the middle of head training and comes back is asked for its windows again",
             program="comeback",
             units=[dict(one, answers="one, model ready", windows=dict(per_set=44, after={"0": [["link_lost"]]}))]),
        dict(name="embeddings asked for again after a turn off the device refused, and the form found put back once it takes one",
             program="refused_off",
             units=[dict(one, answers="one, model ready", windows=dict(per_set=6), refuse_forms=["off"])]),
    ]


# ------------------------------------------------------------------ the files
def _packet(u, *, leadoff=0, synthetic=False, skip=0, discontinuity=False) -> str:
    """One data packet from a device's stream, its lead-off byte, type and
    flags as the case says, `skip` samples left out before it."""
    st = u.stream
    st.index += skip * st.per
    st.t += skip * st.per * st.ticks_per
    rows = [[st.rng.randint(-(1 << 22), 1 << 22) for _ in range(st.channels)] for _ in range(st.per)]
    flags = (F.DISC if (st.flag or discontinuity) else 0) | F.ACTIVE
    head = struct.pack("<BBHIQBBBB", P.PACKET_SYNTHETIC if synthetic else P.PACKET_EEG, flags, leadoff,
                       st.index & 0xFFFFFFFF, st.t, len(rows), 0, P.GAIN_BY_CODE.index(st.gain), F.RATE_CODE[st.rate])
    body = b"".join((c & 0xFFFFFF).to_bytes(3, "big") for row in rows for c in row)
    st.index += st.per
    st.t += st.per * st.ticks_per
    st.flag = 0
    return (head + body).hex()


async def _file_units(run, specs):
    run.units = [_unit(run, i, 90, spec) for i, spec in enumerate(specs)]
    for u in run.units:
        CLOCKS.set(u.clock["before"])
        u.sync = (u.clock["after"], u.clock["ticks"])
        await u.dev.time_sync()


async def the_file(work: pathlib.Path) -> dict:
    """Two devices, four channels and two, recorded into one file."""
    global RUN
    run = types.SimpleNamespace(trace=[], tracing=False, sleeps=0, units=[], ctx=None, ctxs={}, actions={},
                                program=False)
    RUN = run
    await _file_units(run, [dict(info="one"), dict(info="one_12", rate=250, gain=12)])
    s = Session()
    s.devices = [u.dev for u in run.units]
    path = work / "together.csv"
    s.record_to(str(path))
    await s.start_all()
    order = []
    script = [
        (0, {}), (1, {}), (0, dict(leadoff=0x0105)), (1, {}), (0, dict(skip=2)), (1, dict(synthetic=True)),
        (0, {}), (1, dict(leadoff=3)), (0, dict(synthetic=True)), (1, dict(skip=1)), (0, {}), (1, {}),
    ]
    for i, how in script:
        CLOCKS.advance(0.006)
        h = _packet(run.units[i], **how)
        run.units[i].dev._on_data(None, bytes.fromhex(h))
        order.append([i, h])
    await s.stop_all()
    s._file.close()
    return dict(units=[_described(u) for u in run.units], packets=order, csv=path.read_bytes().decode("utf-8"))


async def files(work: pathlib.Path) -> list[dict]:
    """Files written as a program writes them. Each script is a list of
    steps: `record_to` with the file's number, `start_all`, a `packet` from
    one device, a program's own `callback` set on one device's samples,
    `stop` of a file by its number, `disconnect` of one device, `add` of a
    device connected outside the session, and `close`."""
    global RUN
    out = []
    cases = [
        dict(name="a program's own callback, a file stopped, a second file, and a device let go",
             units=[dict(info="one"), dict(info="one_12", rate=250, gain=12)],
             script=[["record_to", 0], ["callback", 0], ["start_all"], ["start_all"],
                     ["packet", 0], ["packet", 1], ["packet", 0], ["packet", 1],
                     ["stop", 0], ["packet", 0], ["packet", 1], ["stop", 0],
                     ["record_to", 1], ["stop", 0], ["packet", 0], ["packet", 1],
                     ["disconnect", 1], ["packet", 0], ["packet", 1], ["packet", 0], ["stop", 1],
                     ["packet", 0]]),
        dict(name="a device with more channels than the file names refused before any device starts",
             units=[dict(info="one"), dict(info="eight", answers="one")],
             script=[["record_to", 0, [0]], ["start_all"]]),
        dict(name="every device written, then the session closed", units=[dict(info="one"), dict(info="one")],
             script=[["record_to", 0], ["start_all"], ["packet", 1], ["packet", 0], ["close"], ["packet", 0]]),
    ]
    for c in cases:
        run = types.SimpleNamespace(trace=[], tracing=False, sleeps=0, units=[], ctx=None, ctxs={}, actions={},
                                    program=False)
        RUN = run
        await _file_units(run, c["units"])
        run.tracing = True
        s = Session()
        s.devices = [u.dev for u in run.units]
        paths, stops, heard, raised = {}, {}, [], None
        try:
            for step in c["script"]:
                what = step[0]
                if what == "record_to":
                    connected = s.devices
                    if len(step) > 2:
                        s.devices = [run.units[i].dev for i in step[2]]
                    paths[step[1]] = work / f"{c['name'][:12].replace(' ', '-')}-{step[1]}.csv"
                    try:
                        stops[step[1]] = s.record_to(str(paths[step[1]]))
                    finally:
                        s.devices = connected
                elif what == "callback":
                    run.units[step[1]].dev.on_sample = lambda _d, smp, i=step[1]: heard.append([i, smp.index])
                elif what == "start_all":
                    await s.start_all()
                elif what == "packet":
                    CLOCKS.advance(0.006)
                    u = run.units[step[1]]
                    h = _packet(u)
                    u.dev._on_data(None, bytes.fromhex(h))
                    run.trace.append(["packet", step[1], h])
                elif what == "stop":
                    stops[step[1]]()
                elif what == "disconnect":
                    await s.disconnect(run.units[step[1]].dev)
                elif what == "close":
                    await s.close()
        except Exception as e:  # noqa: BLE001
            raised = f"{type(e).__name__}: {e}"
        for f in stops.values():
            f()
        out.append(dict(name=c["name"], units=[_described(u) for u in run.units], script=c["script"],
                        expected=dict(trace=run.trace, raised=raised, heard=heard,
                                      csv={str(n): p.read_bytes().decode("utf-8") for n, p in paths.items()},
                                      devices=[u.i for u in run.units if any(u.dev is d for d in s.devices)])))
    return out


def main() -> int:
    work = pathlib.Path(tempfile.mkdtemp(prefix="intomind-together-"))
    try:
        done = []
        for k, r in enumerate(runs()):
            d = work / f"run-{k}"
            done.append(asyncio.run(run_one(r, k, d)))
            if d.exists():
                for p in d.iterdir():
                    p.chmod(0o644)
        played = []
        for k, prog in enumerate(programs()):
            d = work / f"program-{k}"
            played.append(asyncio.run(play(prog, 100 + k, d)))
            if d.exists():
                for p in d.iterdir():
                    p.chmod(0o644)
        file_case = asyncio.run(the_file(work))
        file_cases = asyncio.run(files(work))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    out = dict(library=intomind.__version__, file=file_case, files=file_cases, runs=done, programs=played)
    (HERE / "together.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(done)} runs, {len(played)} programs and {1 + len(file_cases)} files across several devices, "
          f"from intomind {intomind.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
