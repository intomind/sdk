"""The experiments fixture: the Python library's experiments, run against a
stand-in device on scripted clocks, and everything each run did, which every
other implementation's experiments are held to.

    PYTHONPATH=<a checkout of the Python library> python3 contract/experiments/make.py

Each run is one of the library's experiments with small parameters, sent as
a page's form sends them, on the IntoMind One's Device Info from the
conformance vectors with a hardware version that is plainly made up, through
the library's own `client.Device`, recorder and experiment code. The host's
clocks move only while a run waits, and by exactly the time it waits. While
the device streams it sends one packet each wait, of counts from a seeded
source. None is a person's. A person's acts are scripted by the wait they
come after: a marker, the Stop button, or the link dropping.

A device may refuse an opcode, or have its link down before the run, as
the run says, and USB power may appear after a wait or after a delivery of
windows: the next packet says so and is the stream's last, and a stop is
then answered as a device that is not streaming answers it. The device's
gain in force, which its Status and packets state, is the gain the run's
setup gives, but in two runs whose setup gives another, which are refused
before anything records, one of them on a device of firmware 1.4.6 at
gain 1.

The head training asks for the encoder's output. Each time it asks for the
window form, the device sends as many whole windows as the run says, each
four seconds after the last on the host's clocks, each after one packet
while it streams, of seeded values that lean one way in odd reps and the
other in even ones, or, for a run of more states than two, lean on values
of their own by which state the rep cues. Some windows carry a gap or an
electrode off, as the run says, and a person's act can come after a
delivery as after a wait. A rep sent fewer windows than it needs waits for
more in steps of 50 ms, which is where a person presses Stop in the middle
of one, or where the link drops. A person's own states are taught, two and
three of them, and refused before anything is sent when not written one to
a line as 'name | cue words'. Every experiment is given one parameter it
cannot use, and refuses it by name before anything is sent.

For each run it writes the trace: every command written to the device,
every read of its status, every wait with the packets that arrived during it
and what the person did, every cue played and every message the run sent,
in order, then the recording the library made and the error a run raised.
Also the registry as the library lists it, Python's shuffles and draws for a
range of seeds, and the cue tones' WAV files' checksums and lengths.

Writes experiments.json beside this script. Run it again and nothing
changes unless the library does: the clocks are the script's, the time zone
is UTC, and the archives are stamped with a fixed time.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pathlib
import random
import shutil
import stat
import struct
import sys
import tempfile
import time as _time
import types
import warnings
import zipfile

os.environ["TZ"] = "UTC"
_time.tzset()
warnings.simplefilter("ignore")
HERE = pathlib.Path(__file__).resolve().parent
# Made without a sound: the library's experiments speak their cues and play
# their tones through the host's speakers, so every cue goes nowhere here,
# before anything of the library runs.
from intomind import audio as _audio  # noqa: E402
_audio.play = lambda tone, text="": None

import numpy as np  # noqa: E402

import intomind  # noqa: E402
from intomind import audio, client, experiments, protocol as P, provenance as prov, timebase  # noqa: E402

STAMP = (2026, 1, 1, 0, 0, 0)
OP = P.OPCODES
MADE_UP_HARDWARE = (9, 8, 7)


# ---------------------------------------------------------------- the clocks
class Clocks:
    """The host's two clocks, reading what the script says they read."""

    def __init__(self):
        self.mono, self.real = 0.0, 0.0

    def set(self, both) -> None:
        self.mono, self.real = float(both[0]), float(both[1])

    def advance(self, seconds) -> None:
        self.mono += seconds
        self.real += seconds


CLOCKS = Clocks()


def _strftime(fmt, t=None):
    return _time.strftime(fmt, t if t is not None else _time.localtime(CLOCKS.real))


_FAKE = types.SimpleNamespace(
    monotonic=lambda: CLOCKS.mono, time=lambda: CLOCKS.real,
    gmtime=lambda secs=None: _time.gmtime(CLOCKS.real if secs is None else secs),
    localtime=lambda secs=None: _time.localtime(CLOCKS.real if secs is None else secs),
    strftime=_strftime, sleep=_time.sleep, perf_counter=_time.perf_counter)
for _m in (client, experiments, prov, timebase):
    _m.time = _FAKE
# numpy's archive stamps each member with the local time.
zipfile.time = types.SimpleNamespace(
    time=lambda: 0.0, localtime=lambda secs=None: _time.struct_time(STAMP + (3, 1, -1)),
    gmtime=_time.gmtime)
# What the library and its own dependencies are. Another language's recording
# names itself here instead, so a reader compares everything else.
prov.host_versions = lambda: dict(python="fixture", platform="fixture", numpy="fixture",
                                  scipy=None, bleak=None)


# ---------------------------------------------------------------- the device
def _conformance_info() -> bytes:
    vectors = json.loads((HERE.parent / "conformance.json").read_text())["decode"]
    v = next(v for v in vectors if v["name"] == "device info, the IntoMind One")
    b = bytearray(bytes.fromhex(v["bytes"]))
    b[28:31] = bytes(MADE_UP_HARDWARE)
    return bytes(b)


ONE = _conformance_info()


def _one_12() -> bytes:
    """A 1.2 device with two channels that claims lead-off detection and nothing newer."""
    b = bytearray(ONE[:76])
    b[1], b[5], b[27] = 2, 2, 76
    b[15:17] = P.CAPABILITIES["leadoff"].to_bytes(2, "little")
    return bytes(b)


def _one_146() -> bytes:
    """The IntoMind One's Device Info from firmware 1.4.6."""
    b = bytearray(ONE)
    b[2:5] = bytes((1, 4, 6))
    return bytes(b)


INFOS = dict(one=ONE, one_12=_one_12(), one_146=_one_146())
RATE_CODE = {sps: code for code, sps in P.RATE_BY_CODE.items()}
DISC, ACTIVE = P.DATA_FLAGS["discontinuity"], P.DATA_FLAGS["leadoff_active"]
USB = P.DATA_FLAGS["usb_present"]


def status(*, gain=24, rate=500) -> bytes:
    return struct.pack("<BBBBBBBBHH", 1, 0, P.GAIN_BY_CODE.index(gain), RATE_CODE[rate], 1, 61, 0, 0, 0, 0)


HOST_CHAIN = P.encode_chain([P.highpass(0.5), P.lowpass(), P.notch(49, 51)])
VECTORS = json.loads((HERE.parent / "conformance.json").read_text())["decode"]
READY = bytes.fromhex(next(v for v in VECTORS if v["name"].startswith(
    "a model that is ready, with a head selected"))["bytes"])
ANSWERS = dict(
    one={OP["get_pipeline"]: bytes([OP["get_pipeline"], 0, 1]) + HOST_CHAIN,
         OP["get_indicator"]: bytes([OP["get_indicator"], 0, 2]),
         OP["get_model_interval"]: bytes([OP["get_model_interval"], 0]) + struct.pack("<HH", 4, 2),
         OP["get_name"]: bytes([OP["get_name"], 0]) + P.encode_name_parts("Ada", "Blue")},
    one_12={},
)
MODELS = dict(
    ready=READY,
    loading=bytes([1]) + READY[1:],             # state 1: no weights
    every_thirty=READY[:19] + (30).to_bytes(2, "little") + READY[21:],
)
for _name, _model in MODELS.items():
    ANSWERS[f"one, model {_name}"] = {**ANSWERS["one"], OP["get_model_info"]: bytes([OP["get_model_info"], 0]) + _model}



class Stream:
    """What the device sends while it streams: a few samples a packet, of
    seeded counts, on its own clock, starting again at zero with the
    discontinuity flag set whenever it is told to. Once `usb` is set, as
    when USB power appears, the next packet says so and is the stream's
    last."""

    usb = False

    def __init__(self, seed: int, *, channels: int, rate: int, per: int = 3, gain: int = 24,
                 tick_hz: int = 1_000_000, t0: int = 4_000_000_000):
        self.rng = random.Random(seed)
        self.channels, self.rate, self.per, self.gain = channels, rate, per, gain
        self.ticks_per = tick_hz // rate
        self.t, self.index = t0, 0
        self.streaming, self.flag = False, DISC

    def command(self, op: int) -> None:
        if op == OP["reset_epoch"]:
            self.index, self.flag = 0, DISC
        elif op == OP["start_stream"]:
            self.streaming = True
        elif op == OP["stop_stream"]:
            self.streaming = False

    def next(self) -> str:
        rows = [[self.rng.randint(-(1 << 22), 1 << 22) for _ in range(self.channels)] for _ in range(self.per)]
        flags = self.flag | ACTIVE | (USB if self.usb else 0)
        head = struct.pack("<BBHIQBBBB", P.PACKET_EEG, flags, 0, self.index & 0xFFFFFFFF, self.t,
                           len(rows), 0, P.GAIN_BY_CODE.index(self.gain), RATE_CODE[self.rate])
        body = b"".join((c & 0xFFFFFF).to_bytes(3, "big") for row in rows for c in row)
        self.index += self.per
        self.t += self.per * self.ticks_per
        self.flag = 0
        if self.usb:
            self.streaming = False
        return (head + body).hex()


class Link:
    """The radio: answers each control write as the device would, refuses
    what the run says it refuses, answers a stop with ok whether or not the
    device is streaming, as the firmware does, and moves the host's clocks
    across a time exchange as the script says. The device acknowledges a
    write first and handles what it carried afterwards, so the windows a
    write asks for and its answer come on the loop's next turn, once the
    write has returned. A link lost after a rep's windows drops once the
    answer to the request that asked for them is given, since nothing
    reaches the host on a link that is down."""

    def __init__(self, dev, run):
        self.dev, self.run, self.is_connected = dev, run, True
        self.drop_after_answer = False

    async def write_gatt_char(self, _uuid, payload, response=True):
        payload = bytes(payload)
        op = payload[0]
        if self.run.tracing:
            self.run.trace.append(["write", payload.hex()])
        if op == OP["time_sync"]:
            after, ticks = self.run.sync
            CLOCKS.set(after)
            answer = bytes([op, 0]) + int(ticks).to_bytes(8, "little")
        elif op in self.run.refuse:
            answer = bytes([op, 1])
        else:
            answer = self.run.answers.get(op) or bytes([op, 0])
            self.run.stream.command(op)
        windows = op == OP["set_embeddings"] and payload[1] == P.EMBEDDING_FORMS["window"] and self.is_connected \
            and op not in self.run.refuse

        def handled():
            if windows:
                self.run.deliver_windows()
            self.dev._rsp.put_nowait(answer)
            if self.drop_after_answer:
                self.is_connected, self.drop_after_answer = False, False
        asyncio.get_running_loop().call_soon(handled)

    async def read_gatt_char(self, _uuid):
        if self.run.tracing:
            self.run.trace.append(["read", "status"])
        return self.run.status

    async def start_notify(self, *_a):
        return None


# ---------------------------------------------------------------- a run
RUN = None


async def _sleep(seconds) -> None:
    """A run's wait: the clocks move by exactly that, one packet arrives while
    the device streams, and then whatever the person does after this wait."""
    run = RUN
    run.sleeps += 1
    CLOCKS.advance(seconds)
    packets = []
    if run.link.is_connected and run.stream.streaming:
        h = run.stream.next()
        run.dev._on_data(None, bytes.fromhex(h))
        packets.append(h)
    acts = run.actions.get(run.sleeps, [])
    for a in acts:
        if a[0] == "marker":
            run.ctx.marker(a[1])
        elif a[0] == "stop":
            run.ctx.stop.set()
        elif a[0] == "link_lost":
            run.link.is_connected = False
        elif a[0] == "usb":
            run.stream.usb = True
    run.trace.append(["sleep", seconds, packets, acts])


experiments.asyncio = types.SimpleNamespace(sleep=_sleep, Event=asyncio.Event)
experiments.cue_sound = lambda tone, text="": RUN.trace.append(["cue", tone, text])


def page_params(spec: dict, given: dict, drop=()) -> dict:
    """The parameters as a page's form sends them: every default, then what
    was changed, and a whole number written without a fraction, as
    JavaScript writes one."""
    p = {d["name"]: d["default"] for d in spec["params"]}
    p.update(given)
    for k in drop:
        p.pop(k, None)
    return {k: (int(v) if isinstance(v, float) and v.is_integer() else v) for k, v in p.items()}


def _digest(path: pathlib.Path) -> list:
    """Each array in an npz: name, dtype, shape, the sha256 of its elements
    in C order, and whether the file stores it in Fortran order."""
    rows = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            with z.open(name) as f:
                version = np.lib.format.read_magic(f)
                shape, fortran, dtype = np.lib.format._read_array_header(f, version)
            with np.load(path) as npz:
                a = npz[name[:-len(".npy")]]
            rows.append([name[:-len(".npy")], a.dtype.str, list(a.shape),
                         hashlib.sha256(a.tobytes()).hexdigest(), bool(fortran)])
    return rows


def _deliver(run, windows: dict) -> None:
    """The windows the device sends for one ask of the window form: each four
    seconds after the last, after a packet while it streams, of seeded values
    leaning by the rep's parity, some flagged as the run says, and then what
    the person does after this delivery. Where the run gives `varied`, only
    the first `varied` windows sent hold seeded values, and each after them
    holds its lean alone."""
    rep = run.deliveries
    run.deliveries += 1
    info = run.dev.info
    dim = info.model_embed_dim
    states = windows.get("states", 2)

    def lean(j):
        """How far value `j` of each window of this rep leans: the first
        twelve one way in odd reps and the other in even ones, or, with more
        states than two, twelve values of the state's own."""
        if states == 2:
            return (2000 if rep % 2 else -2000) if j < 12 else 0
        return 2000 if j // 12 == rep % states else 0
    sent = []
    for i in range(windows.get("short", {}).get(str(rep), windows.get("per_set", 0))):
        CLOCKS.advance(4.0)
        packet = None
        if run.link.is_connected and run.stream.streaming:
            packet = run.stream.next()
            run.dev._on_data(None, bytes.fromhex(packet))
        flags = sum(P.EMBEDDING_FLAGS[f] for f in windows.get("flags", {}).get(f"{rep}:{i}", []))
        varied = windows.get("varied")
        if varied is not None and run.window_index - 100 >= varied:
            values = [lean(j) for j in range(dim)]
        else:
            values = [run.windows_rng.randint(-900, 900) + lean(j) for j in range(dim)]
        body = P._EMBEDDING.pack(P.PACKET_EMBEDDING, flags, dim, 0, run.window_index,
                                 4_000_000_000 + 4_000_000 * run.window_index, 2000,
                                 bytes.fromhex(READY[4:12].hex()), 0, P.WINDOW_TOKEN)
        body += struct.pack(f"<{dim}h", *values)
        run.window_index += 1
        run.dev._on_embedding(None, body)
        sent.append([packet, body.hex()])
    acts = windows.get("after", {}).get(str(rep), [])
    for a in acts:
        if a[0] == "stop":
            run.ctx.stop.set()
        elif a[0] == "link_lost":
            # Once the device has answered the request that asked for the
            # windows: nothing reaches the host on a link that is down.
            run.link.drop_after_answer = True
        elif a[0] == "usb":
            run.stream.usb = True
    run.trace.append(["embeddings", sent, acts])


async def run_one(spec_run: dict, k: int, captures: pathlib.Path) -> dict:
    global RUN
    prov.use_captures_dir(captures)
    prov.use_site(None)
    prov.use_repo(None)
    spec = experiments.REGISTRY[spec_run["experiment"]]
    info_bytes = INFOS[spec_run.get("info", "one")]
    info = P.decode_device_info(info_bytes)
    rate = spec_run.get("rate", 500)
    run = types.SimpleNamespace(
        trace=[], tracing=False, sleeps=0, actions={int(i): a for i, a in spec_run.get("actions", {}).items()},
        answers=ANSWERS[spec_run.get("answers", spec_run.get("info", "one"))],
        status=status(gain=spec_run.get("gain", 24), rate=rate),
        stream=Stream(k + 1, channels=info.channels, rate=rate, gain=spec_run.get("gain", 24)),
        sync=None, dev=None, link=None, ctx=None, refuse={OP[o] for o in spec_run.get("refuse", [])})
    RUN = run
    windows = spec_run.get("windows", {})
    run.deliveries, run.windows_rng, run.window_index = 0, random.Random(1000 + k), 100
    run.deliver_windows = lambda: _deliver(run, windows)
    dev = client.Device(types.SimpleNamespace(address="02:00:00:00:00:01", name="IntoMind One", details={}))
    dev.info = info
    link = Link(dev, run)
    dev._client = link
    rec = experiments.Recorder()
    dev.on_sample, dev.on_packet = rec.on_sample, rec.on_packet
    run.dev, run.link = dev, link

    start = [5000.0 + 1000.0 * k, 1_767_225_600.0 + 1000.0 * k]
    sync = dict(before=start, after=[start[0] + 0.03, start[1] + 0.03], ticks=4_000_000_000 + 15_000)
    CLOCKS.set(sync["before"])
    run.sync = (sync["after"], sync["ticks"])
    await dev.time_sync()
    setup_packets = []
    if spec_run.get("streaming"):
        await dev.start()
        for _ in range(2):
            h = run.stream.next()
            dev._on_data(None, bytes.fromhex(h))
            setup_packets.append(h)
    if spec_run.get("down"):
        link.is_connected = False

    params = page_params(spec, spec_run.get("params", {}), spec_run.get("drop", ()))
    cfg = dict(spec_run.get("cfg", {}))

    async def emit(msg):
        if msg.get("type") == "saved":
            kept = dict(type="saved", label=msg["label"])
            if "head" in msg:
                kept.update(head=msg["head"], score=msg["score"], no_head=msg["no_head"])
            run.trace.append(["emit", kept])
        else:
            run.trace.append(["emit", msg])

    ctx = experiments.RunCtx(emit, cfg)
    run.ctx = ctx
    run.tracing = True
    out = dict(labels=None, raised=None)
    try:
        out["labels"] = await spec["fn"](dev, rec, emit, params, ctx)
    except Exception as e:  # noqa: BLE001
        out["raised"] = f"{type(e).__name__}: {e}"
    run.tracing = False
    out["trace"] = run.trace
    files = {}
    for p in sorted(captures.iterdir()) if captures.exists() else []:
        files[p.name] = dict(read_only=not (p.stat().st_mode & stat.S_IWUSR))
    out["files"] = files
    recordings = []
    for label in out["labels"] or []:
        ev = captures / f"{label}.events.json"
        sums = captures / f"{label}.sha256"
        taught = captures / f"{label}.training.json"
        recordings.append(dict(
            label=label, meta_json=(captures / f"{label}.meta.json").read_text(),
            events_json=ev.read_text() if ev.exists() else None,
            training_json=taught.read_text() if taught.exists() else None,
            sha256_names=([line.split(None, 1)[1] for line in sums.read_text().splitlines()]
                          if sums.exists() else None),
            arrays=_digest(captures / f"{label}.npz")))
    out["recordings"] = recordings
    out["sessions_jsonl"] = ((captures / "SESSIONS.jsonl").read_text()
                             if (captures / "SESSIONS.jsonl").exists() else None)
    return dict(name=spec_run["name"], experiment=spec_run["experiment"], info=info_bytes.hex(),
                windows_per_set=windows.get("per_set", 0),
                answers={str(op): a.hex() for op, a in run.answers.items()}, status=run.status.hex(),
                sync=sync, streaming=bool(spec_run.get("streaming")), setup_packets=setup_packets,
                refuse=sorted(run.refuse), down=bool(spec_run.get("down")),
                params=params, cfg=cfg, actions={str(i): a for i, a in run.actions.items()}, expected=out)


def _prereg(name: str) -> dict:
    return dict(experiments.REGISTRY[name]["prereg"])


#: Each experiment with the smallest parameters that run it.
SHORT = dict(
    baseline=dict(seconds=1), berger=dict(block=1, reps=1), record=dict(max_seconds=1),
    protocol=dict(steps="Close your eyes | 1", reps=1, lead_in=0),
    blink=dict(trials=2, include_sham=0, hold_s=0.1, rest_s=0.1),
    jaw_clench=dict(trials=2, include_sham=0, hold_s=0.1, rest_s=0.1),
    saccade=dict(trials=2, include_sham=0, hold_s=0.1, center_s=0.1),
    head_training=dict(rep_length=2, reps=2),
)

#: For each experiment, a parameter it cannot use: the parameter, what it
#: is given, and a word for the case.
BAD = dict(
    baseline=[("seconds", "soon", "words")],
    berger=[("block", "long", "words"), ("reps", "8.5", "a fraction"), ("settle", None, "nothing")],
    record=[("max_seconds", "", "an empty string")],
    protocol=[("reps", "x", "a letter"), ("settle", "s", "a letter"), ("lead_in", [6], "a list"),
              ("steps", "Close your eyes | 20\n\nOpen | twenty", "seconds in words"),
              ("steps", "Close your eyes", "no bar"), ("steps", None, "nothing")],
    blink=[("trials", "many", "a word"), ("hold_s", "x", "a letter"), ("rest_s", None, "nothing"),
           ("include_sham", "yes", "a word")],
    jaw_clench=[("trials", "many", "a word"), ("hold_s", "x", "a letter"), ("rest_s", None, "nothing"),
                ("include_sham", "yes", "a word")],
    saccade=[("trials", "many", "a word"), ("hold_s", "x", "a letter"), ("center_s", "x", "a letter"),
             ("include_sham", "yes", "a word")],
    head_training=[("rep_length", "x", "a letter"), ("reps", None, "nothing"), ("pair", "colors", "a pair not offered")],
)

#: Three states a person writes, one to a line.
THREE = ("Melody | Hear a song you know in your head.\n"
         "Alphabet | Say the alphabet backward in your head.\n"
         "\n"
         "Rest | Rest, and think of nothing in particular.\n")

#: States not written one to a line as 'name | cue words', by what is wrong.
WRONG = (
    ("no state", ""),
    ("one state", "Focus | Look at the dot.\n\n"),
    ("a line without a bar", "Focus | Look at the dot.\n\nRest"),
    ("a line without a name", "Focus | Look at the dot.\n  | Rest now."),
    ("a line without cue words", "Focus | Look at the dot.\nRest |  "),
    ("two states of one name", "Rest | Rest now.\nFocus | Look at the dot.\nrest | Rest again."),
)


def runs() -> list[dict]:
    cfg = dict(gain=24, rate_sps=500, mode="normal", leadoff=None, tier="basic")
    return [
        dict(name="baseline from a stopped stream", experiment="baseline",
             params=dict(seconds=1, note="desk"), cfg=cfg),
        dict(name="baseline on the running stream", experiment="baseline", streaming=True, gain=12,
             params=dict(seconds=0.6, tone_start="beep_high", tone_end="none"),
             cfg=dict(rld="on", gain=12, threshold=1.5, tier="advanced", config_before=dict(gain=12, rate_sps=500))),
        dict(name="alpha blocking", experiment="berger",
             params=dict(block=1, reps=2, settle=3, tone_start="chime_up", tone_end="none"),
             cfg=dict(cfg, preregistered_analysis=_prereg("berger"))),
        dict(name="alpha blocking, stopped by the person", experiment="berger",
             params=dict(block=1, reps=3), cfg=cfg, actions={"5": [["stop"]]}),
        dict(name="an open recording with markers, stopped", experiment="record", streaming=True,
             params=dict(markers=" a, b,,c ,", note="eyes open"), cfg=cfg,
             actions={"2": [["marker", "event 1"]], "3": [["marker", "mark"]], "6": [["stop"]]}),
        dict(name="an open recording that reaches its limit", experiment="record",
             params=dict(max_seconds=1), cfg=cfg),
        dict(name="a custom protocol", experiment="protocol",
             params=dict(steps="Close your eyes | 0.5\n\n  Open your eyes|0.75  \r\nLook up | 0.25",
                         reps=2, lead_in=1, settle=2.5, note="three steps"), cfg=cfg),
        dict(name="a custom protocol stopped mid-step", experiment="protocol",
             params=dict(steps="One | 1\nTwo | 1", reps=2, lead_in=1), cfg=cfg, actions={"3": [["stop"]]}),
        dict(name="a step without a bar", experiment="protocol", params=dict(steps="Close your eyes 20"), cfg=cfg),
        dict(name="no steps at all", experiment="protocol", params=dict(steps="  \n\n"), cfg=cfg),
        dict(name="a step whose seconds are not a number", experiment="protocol",
             params=dict(steps="Close | soon"), cfg=cfg),
        dict(name="blinks with shams", experiment="blink", params=dict(trials=4, hold_s=0.1, rest_s=0.1),
             cfg=dict(cfg, preregistered_analysis=_prereg("blink"))),
        dict(name="jaw clenches without shams", experiment="jaw_clench",
             params=dict(trials=3, include_sham=0, hold_s=0.1, rest_s=0.15), cfg=cfg),
        dict(name="eye movements", experiment="saccade", params=dict(trials=3, hold_s=0.2, center_s=0.2),
             cfg=dict(cfg, preregistered_analysis=_prereg("saccade"))),
        dict(name="eye movements, the parameter's older spelling", experiment="saccade",
             params=dict(trials=2, hold_s=0.1, centre_s=0.1), drop=["center_s"], cfg=cfg),
        dict(name="blinks, the link drops", experiment="blink", params=dict(trials=3, hold_s=0.1, rest_s=0.1),
             cfg=cfg, actions={"9": [["link_lost"]]}),
        dict(name="a baseline on a 1.2 device with two channels", experiment="baseline", info="one_12",
             rate=250, gain=12, params=dict(seconds=0.5), cfg=dict(gain=12, rate_sps=250)),
        dict(name="a head trained on eyes open and closed", experiment="head_training",
             answers="one, model ready", params=dict(pair="eyes", rep_length=13, reps=8, note="desk"), cfg=cfg,
             windows=dict(per_set=17, flags={"1:5": ["gap_in_window"], "4:3": ["leadoff_in_window"],
                                              "6:16": ["gap_in_window", "leadoff_in_window"]})),
        dict(name="a head trained on a running stream, mental math", experiment="head_training",
             answers="one, model ready", streaming=True, params=dict(rep_length=13, reps=8), cfg=cfg,
             windows=dict(per_set=17)),
        dict(name="head training stopped after three reps, too few windows for a head",
             experiment="head_training", answers="one, model ready", params=dict(pair="jaw", rep_length=13, reps=8),
             cfg=cfg, windows=dict(per_set=17, after={"2": [["stop"]]})),
        dict(name="head training stopped in the middle of its last rep, a head from what was kept",
             experiment="head_training", answers="one, model ready", params=dict(pair="eyes", rep_length=13, reps=10),
             cfg=cfg, windows=dict(per_set=17, short={"9": 9}), actions={"4": [["stop"]]}),
        dict(name="head training stopped in the middle of its third rep, too few windows for a head",
             experiment="head_training", answers="one, model ready", streaming=True,
             params=dict(pair="jaw", rep_length=13, reps=8), cfg=cfg, windows=dict(per_set=17, short={"2": 7}),
             actions={"4": [["stop"]]}),
        dict(name="head training stopped in the middle of its first rep, one state only",
             experiment="head_training", answers="one, model ready", params=dict(rep_length=13, reps=8), cfg=cfg,
             windows=dict(per_set=17, short={"0": 6}), actions={"2": [["stop"]]}),
        dict(name="head training whose link drops after the second rep", experiment="head_training",
             answers="one, model ready", params=dict(rep_length=13, reps=8), cfg=cfg,
             windows=dict(per_set=17, after={"1": [["link_lost"]]})),
        dict(name="head training on a model that is not ready", experiment="head_training",
             answers="one, model loading", params={}, cfg=cfg),
        dict(name="head training on a model that describes one window in thirty seconds",
             experiment="head_training", answers="one, model every_thirty", params={}, cfg=cfg),
        dict(name="head training with a pair it does not know", experiment="head_training",
             answers="one, model ready", params=dict(pair="colors"), cfg=cfg),
        dict(name="alpha blocking whose stream USB power ends, saved whole and saying so", experiment="berger",
             params=dict(block=1, reps=1), cfg=cfg, actions={"3": [["usb"]]}),
        dict(name="alpha blocking on a device that refuses to start, which leaves nothing armed or started",
             experiment="berger", params=dict(block=1, reps=1), cfg=cfg, refuse=["start_stream"]),
        dict(name="alpha blocking on a device whose link is down, which fails with the error the device raises",
             experiment="berger", params=dict(block=1, reps=1), cfg=cfg, down=True),
        dict(name="head training whose stream USB power ends after its second rep, a head from what it kept",
             experiment="head_training", answers="one, model ready", params=dict(pair="eyes", rep_length=13, reps=4),
             cfg=cfg, windows=dict(per_set=17, after={"1": [["usb"]]})),
        dict(name="a head trained on a person's own two states, written as the eyes pair", experiment="head_training",
             answers="one, model ready",
             params=dict(pair="own", states=("Eyes open | Eyes open.\n"
                                             "Eyes closed | Close your eyes, and keep them closed until the next cue."),
                         rep_length=13, reps=8, note="desk"),
             cfg=cfg, windows=dict(per_set=17, flags={"1:5": ["gap_in_window"], "4:3": ["leadoff_in_window"],
                                                      "6:16": ["gap_in_window", "leadoff_in_window"]})),
        dict(name="a head trained on three of a person's own states, one output for each", experiment="head_training",
             answers="one, model ready", params=dict(pair="own", states=THREE, rep_length=12, reps=9), cfg=cfg,
             windows=dict(per_set=16, states=3)),
        dict(name="three own states on a running stream, stopped before the third has windows",
             experiment="head_training", answers="one, model ready", streaming=True,
             params=dict(pair="own", states=THREE, rep_length=12, reps=9), cfg=cfg,
             windows=dict(per_set=16, states=3, after={"1": [["stop"]]})),
        dict(name="states written wrong are not read with a pair of the library's", experiment="head_training",
             answers="one, model ready", params=dict(pair="eyes", states="not states at all", rep_length=13, reps=2),
             cfg=cfg, windows=dict(per_set=17)),
        *[dict(name=f"own states refused before anything is sent: {what}", experiment="head_training",
               answers="one, model ready", params=dict(pair="own", states=written), cfg=cfg, windows=dict(per_set=17))
          for what, written in WRONG],
        dict(name="head training whose link drops while its second rep's windows are waited for",
             experiment="head_training", answers="one, model ready", params=dict(rep_length=13, reps=8), cfg=cfg,
             windows=dict(per_set=17, short={"1": 9}, after={"1": [["link_lost"]]})),
        *[dict(name=f"{experiment} given {what} for {param} refuses it by name before anything is sent",
               experiment=experiment, answers="one, model ready", params={**SHORT[experiment], param: value}, cfg=cfg,
               windows=dict(per_set=6))
          for experiment, cases in BAD.items() for param, value, what in cases],
        dict(name="a baseline on the running stream whose setup gives a gain other than the gain in force, "
                  "refused before anything records",
             experiment="baseline", streaming=True, params=dict(seconds=0.6), cfg=dict(cfg, gain=12)),
        dict(name="a baseline on a device of firmware 1.4.6 at gain 1 whose setup gives a gain of 24, refused "
                  "in words saying the device samples at gain 1 while lead-off detection is on",
             experiment="baseline", info="one_146", answers="one", gain=1, params=dict(seconds=0.6), cfg=cfg),
        dict(name="head training whose windows hold their state's lean alone, too alike to determine a head",
             experiment="head_training", answers="one, model ready", params=dict(pair="eyes", rep_length=13, reps=8),
             cfg=cfg, windows=dict(per_set=17, varied=0)),
        dict(name="head training whose windows after the 125th hold their state's lean alone, which determine a "
                  "head, but not with a block of them held out to score it",
             experiment="head_training", answers="one, model ready", params=dict(pair="eyes", rep_length=13, reps=8),
             cfg=cfg, windows=dict(per_set=17, varied=125)),
        dict(name="head training given fewer reps than states, refused by name before anything is sent",
             experiment="head_training", answers="one, model ready", params={**SHORT["head_training"], "reps": 1},
             cfg=cfg, windows=dict(per_set=6)),
        dict(name="three own states with two reps, fewer than the states, refused before anything is sent",
             experiment="head_training", answers="one, model ready",
             params=dict(pair="own", states=THREE, rep_length=12, reps=2), cfg=cfg, windows=dict(per_set=16, states=3)),
    ]


def registry() -> list[dict]:
    return [dict(name=v["name"], doc=v["doc"], params=v["params"], enforced=v["enforced"], prereg=v["prereg"])
            for v in experiments.REGISTRY.values()]


def random_cases() -> list[dict]:
    out = []
    for seed in (0, 1, 2, 24601, 31337, 12345, 2 ** 32, 2 ** 32 + 1, 2 ** 64 + 12345, -7, 10 ** 30):
        r = random.Random(seed)
        words = [r.getrandbits(32) for _ in range(10)]
        r = random.Random(seed)
        bits = [r.getrandbits(k) for k in range(33)]
        r = random.Random(seed)
        below = [r._randbelow(n) for n in (1, 2, 3, 7, 100, 2 ** 31 - 1, 2 ** 32 - 1)]
        r = random.Random(seed)
        x = list(range(40))
        r.shuffle(x)
        y = ["blink"] * 30 + ["sham"] * 7
        random.Random(seed).shuffle(y)
        out.append(dict(seed=seed, words=words, bits=bits, below=below, shuffled=x, cues=y))
    return out


def tones() -> list[dict]:
    out = []
    for name in audio.CHOICES:
        p = audio._wav(name)
        if p is None:
            out.append(dict(name=name, sha256=None, length=None))
            continue
        b = p.read_bytes()
        out.append(dict(name=name, sha256=hashlib.sha256(b).hexdigest(), length=len(b)))
    return out


def main() -> int:
    work = pathlib.Path(tempfile.mkdtemp(prefix="intomind-experiments-"))
    try:
        done = []
        for k, r in enumerate(runs()):
            d = work / f"run-{k}"
            done.append(asyncio.run(run_one(r, k, d)))
            if d.exists():
                for p in d.iterdir():
                    p.chmod(0o644)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    out = dict(library=intomind.__version__, registry=registry(), runs=done, random=random_cases(),
               tones=tones(), tone_params=experiments.TONE_PARAMS)
    (HERE / "experiments.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(done)} runs, {len(out['registry'])} experiments, {len(out['random'])} seeds and "
          f"{len(out['tones'])} tones, from intomind {intomind.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
