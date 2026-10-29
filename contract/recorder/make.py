"""The recorder fixture: sessions with a device, run through the Python
library's own device and recorder, and the recordings the library wrote,
which every other implementation's recorder is held to.

    PYTHONPATH=<a checkout of the Python library> python3 contract/recorder/make.py

A session is a script of what a host sees and does: the device's info, time
exchanges with both host clocks read either side, data notifications,
control answers, and when the recorder is armed, stopped and saved. The
library's `client.Device` runs each script through a stand-in for the radio
whose clocks read what the script says, and `experiments.Recorder` writes
the recording. Every signal is generated from a seeded source. None is a
person's.

Writes sessions.json beside this script: each session's script, and what the
library made of it: the clock record after every exchange, the
configuration and the processing it read back, the manifest and events
files as text, the arrays, the checksum file's names, the session log's
line, and which files were made read-only. Also the montage cases, where
the electrodes went and what a manifest makes of it, and processing chains
described in words.

Run it again and nothing changes unless the library does: the clocks are the
script's, and the archives are stamped with a fixed time.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
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

import numpy as np

warnings.simplefilter("ignore")
HERE = pathlib.Path(__file__).resolve().parent

import intomind  # noqa: E402
from intomind import client, experiments, montage as M, placement as PL  # noqa: E402
from intomind import protocol as P, provenance as prov, timebase  # noqa: E402

STAMP = (2026, 1, 1, 0, 0, 0)
OPS = dict(time_sync=P.OPCODES["time_sync"], start=P.OPCODES["start_stream"],
           stop=P.OPCODES["stop_stream"], pipeline=P.OPCODES["get_pipeline"],
           indicator=P.OPCODES["get_indicator"], model_interval=P.OPCODES["get_model_interval"],
           name=P.OPCODES["get_name"], reset_epoch=P.OPCODES["reset_epoch"])


# ---------------------------------------------------------------- the clocks
class Clocks:
    """The host's two clocks, reading what the script says they read."""

    def __init__(self):
        self.mono, self.real = 0.0, 0.0

    def set(self, both) -> None:
        self.mono, self.real = float(both[0]), float(both[1])


CLOCKS = Clocks()
_FAKE = types.SimpleNamespace(
    monotonic=lambda: CLOCKS.mono, time=lambda: CLOCKS.real,
    gmtime=lambda secs=None: _time.gmtime(CLOCKS.real if secs is None else secs),
    strftime=_time.strftime, sleep=_time.sleep, perf_counter=_time.perf_counter)
for _m in (client, experiments, prov, timebase):
    _m.time = _FAKE
# numpy's archive stamps each member with the local time.
zipfile.time = types.SimpleNamespace(
    time=lambda: 0.0, localtime=lambda secs=None: _time.struct_time(STAMP + (3, 1, -1)),
    gmtime=_time.gmtime)
# What the library and its own dependencies are. The recorder of another
# language names itself here instead, so a reader compares everything else.
prov.host_versions = lambda: dict(python="fixture", platform="fixture", numpy="fixture",
                                  scipy=None, bleak=None)


# ---------------------------------------------------------------- the device
def _conformance_info() -> bytes:
    vectors = json.loads((HERE.parent / "conformance.json").read_text())["decode"]
    v = next(v for v in vectors if v["name"] == "device info, the IntoMind One")
    return bytes.fromhex(v["bytes"])


ONE = _conformance_info()


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


ONE_12 = info(minor=2, channels=2, caps=P.CAPABILITIES["leadoff"], length=76)


class Link:
    """The radio: answers each control write as the device would, and
    moves the host's clocks across a time exchange as the script says."""

    def __init__(self, dev):
        self.dev, self.is_connected = dev, True
        self.status = bytes(P.STATUS_LEN)
        self.answers: dict[int, bytes] = {}
        self.sync = None

    async def write_gatt_char(self, _uuid, payload, response=True):
        payload = bytes(payload)
        op = payload[0]
        if op == OPS["time_sync"]:
            after, ticks = self.sync
            CLOCKS.set(after)
            answer = bytes([op, 0]) + int(ticks).to_bytes(8, "little")
        else:
            answer = self.answers.get(op) or bytes([op, 0])
        self.dev._rsp.put_nowait(answer)

    async def read_gatt_char(self, _uuid):
        return self.status

    async def start_notify(self, *_a):
        return None


def device(info_bytes: bytes):
    dev = client.Device(types.SimpleNamespace(address="EE:AE:00:00:00:01", name="IntoMind One", details={}))
    dev.info = P.decode_device_info(info_bytes)
    link = Link(dev)
    dev._client = link
    return dev, link


# ---------------------------------------------------------------- the wire
RATE_CODE = {sps: code for code, sps in P.RATE_BY_CODE.items()}


def packet(index: int, device_time: int, rows, *, flags: int = 0, lost: int = 0, loff: int = 0,
           gain: int = 24, rate: int | None = 500, ptype: int = P.PACKET_EEG, mode: int = 0) -> str:
    """One data notification, as hex."""
    code = RATE_CODE[rate] if rate in RATE_CODE else 7
    head = struct.pack("<BBHIQBBBB", ptype, flags | (mode << P.MODE_SHIFT), lost, index & 0xFFFFFFFF,
                       device_time, len(rows), loff, P.GAIN_BY_CODE.index(gain), code)
    body = b"".join((c & 0xFFFFFF).to_bytes(3, "big") for row in rows for c in row)
    return (head + body).hex()


DISC, ACTIVE, USB = P.DATA_FLAGS["discontinuity"], P.DATA_FLAGS["leadoff_active"], P.DATA_FLAGS["usb_present"]


class Stream:
    """A device streaming at a rate: consecutive packets of generated counts,
    on the device's own clock."""

    def __init__(self, seed: int, *, channels: int = 4, rate: int = 500, per: int = 18,
                 tick_hz: int = 1_000_000, t0: int = 4_000_000_000, index: int = 0):
        self.rng = random.Random(seed)
        self.channels, self.rate, self.per = channels, rate, per
        self.ticks_per = tick_hz // rate
        self.t, self.index = t0, index

    def rows(self, n: int):
        return [[self.rng.randint(-(1 << 22), 1 << 22) for _ in range(self.channels)] for _ in range(n)]

    def next(self, n: int | None = None, **kw) -> str:
        n = self.per if n is None else n
        kw.setdefault("rate", self.rate)
        p = packet(self.index, self.t, self.rows(n), **kw)
        self.index += n
        self.t += n * self.ticks_per
        return p

    def run(self, k: int, **kw) -> list[str]:
        return [self.next(**kw) for _ in range(k)]

    def skip(self, samples: int) -> None:
        self.index += samples
        self.t += samples * self.ticks_per


def status(*, streaming=True, mode=0, gain=24, rate=500, loff=0, flags=0) -> str:
    return struct.pack("<BBBBBBBBHH", 1 if streaming else 0, mode, P.GAIN_BY_CODE.index(gain),
                       RATE_CODE.get(rate, 7), 1, 61, loff, flags, 0, 0).hex()


def answer(op: str, payload: bytes = b"") -> str:
    return (bytes([OPS[op], 0]) + payload).hex()


HOST_CHAIN = P.encode_chain([P.highpass(0.5), P.lowpass(), P.notch(49, 51)])
DEFAULT_500 = P.encode_chain(P.default_chain(500))
CONFIG_ONE = dict(status=status(), pipeline=answer("pipeline", bytes([1]) + HOST_CHAIN),
                  indicator=answer("indicator", bytes([2])),
                  model_interval=answer("model_interval", struct.pack("<HH", 4, 2)),
                  name=answer("name", P.encode_name_parts("Ada", "Blue")))


def syncs(start_mono: float, start_real: float, device_s: float, n: int, *, gap_s: float = 0.15,
          rtt: float = 0.03, skew_ppm: float = 0.0, step_after: int | None = None, step_s: float = 0.0,
          jitter: float = 0.0, seed: int = 0, tick_hz: int = 1_000_000) -> list[dict]:
    """`n` time exchanges `gap_s` apart: both host clocks either side of each,
    and the device's answer, with the device's crystal `skew_ppm` off."""
    rng = random.Random(seed)
    out, real_off = [], start_real - start_mono
    for i in range(n):
        if step_after is not None and i == step_after:
            real_off += step_s
        m1 = start_mono + i * gap_s
        r = rtt + (rng.uniform(-jitter, jitter) if jitter else 0.0)
        m2 = m1 + r
        mid = (m1 + m2) / 2
        dev = device_s + (mid - start_mono) * (1 + skew_ppm * 1e-6)
        out.append(dict(op="sync", before=[m1, m1 + real_off], after=[m2, m2 + real_off],
                        ticks=int(round(dev * tick_hz))))
    return out


# ---------------------------------------------------------------- placement
def _one_layout() -> dict:
    s = [PL.slot("north", "north electrode", name="e1", angle_deg=0),
         PL.slot("east", "east electrode", name="e2", angle_deg=90),
         PL.slot("south", "south electrode", name="e3", angle_deg=180),
         PL.slot("west", "west electrode", name="e4", angle_deg=270),
         PL.slot("center", "center reference", PL.REFERENCE, name="ref"),
         PL.slot("bias", "bias electrode", PL.BIAS, name="drive")]
    ch = [dict(index=i + 1, derivation=M.REFERENTIAL, positive=k, negative="center")
          for i, k in enumerate(("north", "east", "south", "west"))]
    presets = [PL.preset("crown", "on the crown", dict(north="left-crown-side", east="left-crown-side",
                                                         south="left-crown-side", west="left-crown-side",
                                                         center="left-crown-side", bias="left-behind-ear"),
                         rotation_deg=0),
               PL.preset("crown-turned", "on the crown, turned a quarter",
                         dict(north="left-crown-side", east="left-crown-side", south="left-crown-side",
                              west="left-crown-side", center="left-crown-side", bias="right-behind-ear"),
                         rotation_deg=90)]
    return PL.layout("four-around-a-center", "four electrodes around a center reference", s, ch,
                     rotatable=True, presets=presets, electrode="dry gold")


def _pairs_layout() -> dict:
    s = [PL.slot("a1", "left front"), PL.slot("a2", "left back"),
         PL.slot("b1", "right front"), PL.slot("b2", "right back"),
         PL.slot("drive", "bias", PL.BIAS)]
    ch = [dict(index=1, derivation=M.BIPOLAR, positive="a1", negative="a2", label="left pair"),
          dict(index=2, derivation=M.BIPOLAR, positive="b1", negative="b2")]
    return PL.layout("two-pairs", "two bipolar pairs", s, ch)


ONE_LAYOUT, PAIRS_LAYOUT = _one_layout(), _pairs_layout()
ONE_PLACED = dict(layout="four-around-a-center", preset="crown", sites=dict(ONE_LAYOUT["presets"][0]["sites"]),
                  rotation_deg=0, over={}, note="", bias_driven=True)
PAIRS_PLACED = dict(layout="two-pairs", preset=None,
                    sites=dict(a1="left-temple-front", a2="left-temple-back", b1="right-temple-front",
                               b2="right-temple-back", drive="forehead-center"),
                    rotation_deg=None, over=dict(a1=["temporalis"]), note="first try", bias_driven=True)
SITE_MONTAGE = M.build(
    [M.channel(i + 1, M.REFERENTIAL, M.site(n, ten_twenty=t), M.site("ref", description="center"))
     for i, (n, t) in enumerate((("e1", "C3"), ("e2", "C1"), ("e3", "CP3"), ("e4", "CP1")))],
    electrode="dry gold", reference=M.site("ref", description="center"), bias=M.site("mastoid"),
    prose="four around a center", source="the site file")


def montage_cases() -> list[dict]:
    """Where the electrodes went, every way a manifest is told, and what it
    records: `_montage_fields` on each, or the exception it raises."""
    def bad(**kw):
        d = json.loads(json.dumps(PAIRS_PLACED))
        d.update(kw)
        return d

    def layout_with(**kw):
        d = json.loads(json.dumps(PAIRS_LAYOUT))
        d.update(kw)
        return d

    no_ref = json.loads(json.dumps(ONE_LAYOUT))
    no_ref["slots"] = [s for s in no_ref["slots"] if s["role"] != PL.REFERENCE]
    cases = [
        dict(name="a preset on four around a center", site={}, channels=4, layout=ONE_LAYOUT, place=ONE_PLACED),
        dict(name="the same placement on an eight channel device", site={}, channels=8,
             layout=ONE_LAYOUT, place=ONE_PLACED),
        dict(name="unknown channel count", site={}, channels=None, layout=ONE_LAYOUT, place=ONE_PLACED),
        dict(name="turned, bias not driven", site={}, channels=4, layout=ONE_LAYOUT,
             place=dict(ONE_PLACED, preset="crown-turned", rotation_deg=90, bias_driven=False,
                        note="band slipped once")),
        dict(name="two pairs with an override and a note", site={}, channels=2,
             layout=PAIRS_LAYOUT, place=PAIRS_PLACED),
        dict(name="a slot left out", site={}, channels=2, layout=PAIRS_LAYOUT,
             place=bad(sites=dict(a1="left-temple-front", a2="left-temple-back", b1="right-temple-front",
                                  drive="forehead-center"))),
        dict(name="a bias that is not driven need not be placed", site={}, channels=2, layout=PAIRS_LAYOUT,
             place=bad(sites=dict(a1="left-temple-front", a2="left-temple-back", b1="right-temple-front",
                                  b2="right-temple-back"), bias_driven=False)),
        dict(name="not a scalp position", site={}, channels=2, layout=PAIRS_LAYOUT,
             place=bad(sites=dict(PAIRS_PLACED["sites"], a2="left-elbow"))),
        dict(name="not a slot", site={}, channels=2, layout=PAIRS_LAYOUT,
             place=bad(sites=dict(PAIRS_PLACED["sites"], c9="left-temple-back"))),
        dict(name="an override of no slot", site={}, channels=2, layout=PAIRS_LAYOUT,
             place=bad(over=dict(zz=["temporalis"]))),
        dict(name="an override of no region", site={}, channels=2, layout=PAIRS_LAYOUT,
             place=bad(over=dict(a1=["broca"]))),
        dict(name="a rotation on a band that does not turn", site={}, channels=2, layout=PAIRS_LAYOUT,
             place=bad(rotation_deg=45)),
        dict(name="a placement for another layout", site={}, channels=2, layout=PAIRS_LAYOUT,
             place=bad(layout="something-else")),
        dict(name="a layout with no slots", site={}, channels=2, layout=layout_with(slots=[]), place=PAIRS_PLACED),
        dict(name="a layout with no channels", site={}, channels=2, layout=layout_with(channels=[]),
             place=PAIRS_PLACED),
        dict(name="channel numbers with a hole", site={}, channels=2,
             layout=layout_with(channels=[dict(PAIRS_LAYOUT["channels"][0]),
                                          dict(PAIRS_LAYOUT["channels"][1], index=3)]), place=PAIRS_PLACED),
        dict(name="a pole that names no slot", site={}, channels=2,
             layout=layout_with(channels=[dict(PAIRS_LAYOUT["channels"][0], negative="a9"),
                                          dict(PAIRS_LAYOUT["channels"][1])]), place=PAIRS_PLACED),
        dict(name="a pole that is a number", site={}, channels=2,
             layout=layout_with(channels=[dict(PAIRS_LAYOUT["channels"][0], negative=7),
                                          dict(PAIRS_LAYOUT["channels"][1])]), place=PAIRS_PLACED),
        dict(name="a bad derivation", site={}, channels=2,
             layout=layout_with(channels=[dict(PAIRS_LAYOUT["channels"][0], derivation="laplacian"),
                                          dict(PAIRS_LAYOUT["channels"][1])]), place=PAIRS_PLACED),
        dict(name="single-ended with no reference slot", site={}, channels=4, layout=no_ref,
             place=dict(ONE_PLACED, sites={k: v for k, v in ONE_PLACED["sites"].items() if k != "center"})),
        dict(name="a preset with no key", site={}, channels=4,
             layout=dict(ONE_LAYOUT, presets=[dict(ONE_LAYOUT["presets"][0], key="")]), place=ONE_PLACED),
        dict(name="a preset missing a slot", site={}, channels=4,
             layout=dict(ONE_LAYOUT, presets=[dict(ONE_LAYOUT["presets"][0],
                                                   sites=dict(north="left-crown-side"))]), place=ONE_PLACED),
        dict(name="a layout without a placement reads the site", site=dict(montage=SITE_MONTAGE), channels=4,
             layout=ONE_LAYOUT, place=None),
        dict(name="the site's montage", site=dict(montage=SITE_MONTAGE), channels=4, layout=None, place=None),
        dict(name="the site's montage on a two channel device", site=dict(montage=SITE_MONTAGE), channels=2,
             layout=None, place=None),
        dict(name="the site's montage with a hole", channels=4, layout=None, place=None,
             site=dict(montage=dict(SITE_MONTAGE, channels=[dict(SITE_MONTAGE["channels"][0], index=2)]))),
        dict(name="the site's montage against another reference", channels=4, layout=None, place=None,
             site=dict(montage=dict(SITE_MONTAGE, reference=dict(name="other", ten_twenty=None,
                                                                    description=None)))),
        dict(name="the site's montage at a position that is not 10-20", channels=4, layout=None, place=None,
             site=dict(montage=dict(SITE_MONTAGE, channels=[dict(
                 SITE_MONTAGE["channels"][0], positive=dict(name="e1", ten_twenty="C99", description=None))]))),
        dict(name="the site's montage with no channels", channels=4, layout=None, place=None,
             site=dict(montage=dict(SITE_MONTAGE, channels=[]))),
        dict(name="no montage anywhere", site={}, channels=4, layout=None, place=None),
        dict(name="no site at all", site=None, channels=4, layout=None, place=None),
    ]
    for c in cases:
        try:
            c["result"] = prov._montage_fields(c["site"], c["channels"], layout=c["layout"], place=c["place"])
        except Exception as e:  # noqa: BLE001
            c["raises"] = f"{type(e).__name__}: {e}"
    return cases


def chain_cases() -> list[dict]:
    """Processing chains in words, the way a manifest and a header say them."""
    k = P.STAGE_KINDS
    rows = [
        ([], 500), ([P.highpass(0.5)], 500), ([P.lowpass(40)], None), ([P.lowpass()], 500),
        ([P.lowpass()], 250), ([P.lowpass()], 1000), ([P.lowpass()], None), ([P.notch(49, 51)], 500),
        ([P.notch(59.5, 60.5)], 250), (P.default_chain(500), 500), (P.default_chain(250), 250),
        ([P.Stage(9, (1, 2))], 500), ([P.Stage(k["highpass"], ())], 500), ([P.Stage(k["notch"], (500,))], 500),
        ([P.highpass(0.16), P.lowpass(), P.notch(49, 51), P.notch(99, 101)], 500),
        ([P.highpass(1234.5)], 500),
    ]
    return [dict(stages=[s.as_dict() for s in stages], rate=rate, chain=P.encode_chain(stages).hex(),
                 description=P.describe_chain(stages, rate)) for stages, rate in rows]


# ---------------------------------------------------------------- sessions
def sessions() -> list[dict]:
    out = []

    def add(name, script, note):
        out.append(dict(name=name, note=note, script=script))

    s = Stream(1)
    add("demo-minute", [
        dict(op="info", bytes=ONE.hex(), pipeline=answer("pipeline", bytes([0]) + DEFAULT_500)),
        *syncs(1000.0, 1_767_225_600.0, 4000.0, 1),
        dict(op="arm"), dict(op="start"),
        dict(op="data", packets=[s.next(flags=DISC | ACTIVE)] + s.run(29, flags=ACTIVE)),
        dict(op="stop"),
        dict(op="save", label="demo-minute", extra=dict(experiment="demo-record"), events=None,
             now=[1012.0, 1_767_225_612.0]),
    ], "the Python example: one exchange at connect, armed, started, stopped, saved")

    s = Stream(2, t0=5_000_000_000)
    pre = s.run(5, flags=ACTIVE)
    s.index = 0
    s.t += 1_000_000
    drained = [packet(9000 + 18 * i, s.t - 500_000 + 36_000 * i, s.rows(18), flags=ACTIVE) for i in range(2)]
    first = s.next(flags=DISC | ACTIVE)
    body = s.run(12, flags=ACTIVE) + [s.next(flags=ACTIVE, loff=0b0101)] + s.run(11, flags=ACTIVE, loff=1)
    tail = s.run(10, flags=ACTIVE)
    add("record-while-streaming", [
        dict(op="info", bytes=ONE.hex(), pipeline=answer("pipeline", bytes([1]) + HOST_CHAIN)),
        *syncs(2000.0, 1_767_225_700.0, 5000.0, 5, jitter=0.01, seed=2),
        dict(op="start"), dict(op="data", packets=pre),
        dict(op="reset_epoch"), dict(op="data", packets=drained),
        dict(op="arm"), dict(op="data", packets=[first] + body),
        *syncs(2010.0, 1_767_225_710.0, 5010.0, 1, seed=3),
        dict(op="data", packets=tail),
        dict(op="stop"), dict(op="disarm"),
        dict(op="config_after", link_ok=True, **CONFIG_ONE),
        dict(op="save", label="record-while-streaming",
             extra=dict(experiment="record", note="", started_utc="2026-01-01T00:01:41Z", duration_s=11.75),
             events=[dict(t=1.25, host_time=1_767_225_702.5, mono=2002.5, label="event 1"),
                     dict(t=4.5, host_time=1_767_225_705.75, mono=2005.75, label="event 2")],
             config_after="read", now=[2013.0, 1_767_225_713.0]),
    ], "the record experiment: a running stream, reset, drained, armed, markers, read back after")

    s = Stream(3)
    add("cued-with-processing", [
        dict(op="info", bytes=ONE.hex(), pipeline=answer("pipeline", bytes([1]) + HOST_CHAIN)),
        *syncs(3000.0, 1_767_225_800.0, 4000.0, 5, seed=4),
        dict(op="arm"), dict(op="start"),
        dict(op="data", packets=[s.next(flags=DISC | ACTIVE)] + s.run(20, flags=ACTIVE)),
        dict(op="stop"), dict(op="disarm"),
        dict(op="config_after", link_ok=True, **CONFIG_ONE),
        dict(op="processing", pipeline=answer("pipeline", bytes([1]) + HOST_CHAIN)),
        dict(op="save", label="cued-with-processing",
             extra=dict(experiment="protocol", reps=2, settle=3.0, started_utc="2026-01-01T00:03:20Z"),
             with_processing=True, events=[dict(t=0.0, host_time=1_767_225_801.0, mono=3001.0, cue="Close your eyes")],
             config_after="read", now=[3010.0, 1_767_225_810.0]),
    ], "a cued run: the chain the device ran read back and stored as processing and signal")

    s = Stream(4, channels=4)
    a = [s.next(flags=DISC | ACTIVE)] + s.run(6, flags=ACTIVE)
    s.skip(5)
    gap = [s.next(flags=ACTIVE, lost=5)] + s.run(3, flags=ACTIVE)
    copy = gap[-1]
    s.index, s.t = 0, s.t + 2_000_000
    rebase = [s.next(flags=DISC | ACTIVE)] + s.run(4, flags=ACTIVE)
    usb = [s.next(flags=ACTIVE | USB)] + s.run(2, flags=ACTIVE | USB)
    after = s.run(3, flags=ACTIVE | USB)
    add("breaks-copies-and-usb", [
        dict(op="info", bytes=ONE.hex(), pipeline=answer("pipeline", bytes([0]) + DEFAULT_500)),
        *syncs(4000.0, 1_767_225_900.0, 4000.0, 3, seed=5),
        dict(op="arm"), dict(op="start"),
        dict(op="data", packets=a + gap + [copy] + rebase + usb),
        *syncs(4005.0, 1_767_225_905.0, 4005.0, 1, seed=6),
        dict(op="data", packets=after),
        dict(op="stop"),
        dict(op="save", label="breaks-copies-and-usb", extra=dict(experiment="demo-record"), events=None,
             now=[4010.0, 1_767_225_910.0]),
    ], "a lost run, a packet delivered twice, a restarted count, and USB power ending the stream")

    s = Stream(5)
    add("link-lost", [
        dict(op="info", bytes=ONE.hex(), pipeline=answer("pipeline", bytes([0]) + DEFAULT_500)),
        *syncs(5000.0, 1_767_226_000.0, 4000.0, 1),
        dict(op="arm"), dict(op="start"),
        dict(op="data", packets=[s.next(flags=DISC | ACTIVE)] + s.run(9, flags=ACTIVE)),
        dict(op="link_lost"), dict(op="stop"), dict(op="disarm"),
        dict(op="config_after", link_ok=False, **CONFIG_ONE),
        dict(op="save", label="link-lost", extra=dict(experiment="record", note="lost"), events=[],
             config_after="read", aborted="link_lost", now=[5003.0, 1_767_226_003.0]),
    ], "the link drops: kept as aborted, with no checksums")

    s = Stream(6)
    add("synthetic", [
        dict(op="info", bytes=ONE.hex(), pipeline=answer("pipeline", bytes([0]) + DEFAULT_500)),
        *syncs(6000.0, 1_767_226_100.0, 4000.0, 1),
        dict(op="arm"), dict(op="start"),
        dict(op="data", packets=[s.next(flags=DISC, ptype=P.PACKET_SYNTHETIC, mode=3)]
             + s.run(9, ptype=P.PACKET_SYNTHETIC, mode=3)),
        dict(op="stop"),
        dict(op="save", label="synthetic", extra=dict(experiment="demo-record"), events=None,
             now=[6002.0, 1_767_226_102.0]),
    ], "samples the device generated")

    s = Stream(7, channels=2, per=25, rate=250)
    add("two-channel-1-2", [
        dict(op="info", bytes=ONE_12.hex()),
        *syncs(7000.0, 1_767_226_200.0, 4000.0, 2, seed=8),
        dict(op="arm"), dict(op="start"),
        dict(op="data", packets=[s.next(flags=DISC, gain=12)] + s.run(5, gain=12)
             + s.run(3, gain=12, flags=USB) + s.run(2, gain=12, loff=3)),
        dict(op="stop"), dict(op="disarm"),
        dict(op="config_after", link_ok=True, status=status(gain=12, rate=250, loff=3)),
        dict(op="save", label="two-channel-1-2", extra=dict(experiment="record", note=""), events=[],
             config_after="read", now=[7003.0, 1_767_226_203.0]),
    ], "a 1.2 device with two channels and none of the newer reads, whose USB flag is not an end")

    s = Stream(8)
    add("site-declared", [
        dict(op="info", bytes=ONE.hex(), pipeline=answer("pipeline", bytes([0]) + DEFAULT_500)),
        dict(op="site", site=dict(board_revision="B3", mains_hz=50, psu_volts=3.7, firmware_sha256="ab" * 32,
                                  firmware_artifact="app-1.4.5.bin", electrode_montage="four around a center",
                                  subject_id="S01", notes="bench", montage=SITE_MONTAGE)),
        *syncs(8000.0, 1_767_226_300.0, 4000.0, 1),
        dict(op="arm"), dict(op="start"),
        dict(op="data", packets=[s.next(flags=DISC | ACTIVE)] + s.run(7, flags=ACTIVE)),
        dict(op="stop"),
        dict(op="save", label="site-declared",
             extra=dict(experiment="site-check", gain=None, rate_sps=None, fs_effective=None, threshold=1.5,
                        regs_before=dict(reg_a=150, reg_b=96)),
             events=None, regs_after=dict(reg_a=150, reg_b=97, reg_c=0), now=[8002.0, 1_767_226_302.0]),
    ], "a bench that declares its site and montage, registers that changed, and an extra that tries to erase a measurement")

    s = Stream(9)
    add("placed", [
        dict(op="info", bytes=ONE.hex(), pipeline=answer("pipeline", bytes([0]) + DEFAULT_500)),
        *syncs(9000.0, 1_767_226_400.0, 4000.0, 1),
        dict(op="arm"), dict(op="start"),
        dict(op="data", packets=[s.next(flags=DISC | ACTIVE)] + s.run(7, flags=ACTIVE)),
        dict(op="stop"),
        dict(op="save", label="placed",
             extra=dict(experiment="record", electrode_layout=ONE_LAYOUT, placement=ONE_PLACED,
                        config_before=dict(gain=24, rate_sps=500, mode="normal", leadoff=True)),
             events=None, config_after=dict(gain=12, rate_sps=500, mode="normal", leadoff=True),
             now=[9002.0, 1_767_226_402.0]),
    ], "where the electrodes went, and a configuration read before and after")

    s = Stream(10)
    add("placed-wrong", [
        dict(op="info", bytes=ONE.hex(), pipeline=answer("pipeline", bytes([0]) + DEFAULT_500)),
        *syncs(9100.0, 1_767_226_500.0, 4000.0, 1),
        dict(op="arm"), dict(op="start"),
        dict(op="data", packets=[s.next(flags=DISC | ACTIVE)] + s.run(3, flags=ACTIVE)),
        dict(op="stop"),
        dict(op="save", label="placed-wrong",
             extra=dict(experiment="record", electrode_layout=PAIRS_LAYOUT,
                        placement=dict(PAIRS_PLACED, sites=dict(a1="left-temple-front"))),
             events=None, now=[9102.0, 1_767_226_502.0]),
    ], "a placement that leaves electrodes out, recorded as an error")

    s = Stream(11)
    long_before = syncs(10_000.0, 1_767_226_600.0, 4000.0, 10, gap_s=60.0, rtt=0.02, skew_ppm=20.0,
                        jitter=0.002, seed=12)
    during = syncs(10_600.0, 1_767_227_200.0, 4000.0 + 600.0 * (1 + 20e-6), 3, gap_s=30.0, rtt=0.02,
                   skew_ppm=20.0, seed=13)
    add("clock-measured", [
        dict(op="info", bytes=ONE.hex(), pipeline=answer("pipeline", bytes([0]) + DEFAULT_500)),
        *long_before,
        dict(op="arm"), dict(op="start"),
        dict(op="data", packets=[s.next(flags=DISC | ACTIVE)] + s.run(9, flags=ACTIVE)),
        *during,
        dict(op="data", packets=s.run(5, flags=ACTIVE)),
        dict(op="stop"),
        *syncs(10_700.0, 1_767_227_300.0, 4000.0 + 700.0 * (1 + 20e-6), 1, rtt=0.02, seed=14),
        dict(op="save", label="clock-measured", extra=dict(experiment="demo-record"), events=None,
             now=[10_701.0, 1_767_227_301.0]),
    ], "ten minutes of exchanges that measure the crystal, more while the map is held, one after")

    s = Stream(12)
    add("clock-stepped", [
        dict(op="info", bytes=ONE.hex(), pipeline=answer("pipeline", bytes([0]) + DEFAULT_500)),
        *syncs(11_000.0, 1_767_227_400.0, 4000.0, 6, gap_s=1.0, step_after=3, step_s=2.0, seed=15),
        dict(op="arm"), dict(op="start"),
        dict(op="data", packets=[s.next(flags=DISC | ACTIVE)] + s.run(5, flags=ACTIVE)),
        dict(op="stop"),
        dict(op="save", label="clock-stepped", extra=dict(experiment="demo-record"), events=None,
             now=[11_010.0, 1_767_227_412.0]),
    ], "the wall clock steps two seconds between exchanges")

    s = Stream(13)
    add("clock-refused", [
        dict(op="info", bytes=ONE.hex(), pipeline=answer("pipeline", bytes([0]) + DEFAULT_500)),
        *syncs(12_000.0, 1_767_227_500.0, 4000.0, 3, gap_s=0.15, skew_ppm=3000.0, seed=16),
        *syncs(12_000.6, 1_767_227_500.6, 4000.6, 3, gap_s=0.15, rtt=0.08, jitter=0.04, seed=17),
        dict(op="arm"), dict(op="start"),
        dict(op="data", packets=[s.next(flags=DISC | ACTIVE)] + s.run(3, flags=ACTIVE)),
        dict(op="stop"),
        dict(op="save", label="clock-refused", extra=dict(experiment="demo-record"), events=None,
             now=[12_002.0, 1_767_227_502.0]),
    ], "slopes no crystal could have, then slopes too noisy to believe")

    s = Stream(14)
    add("no-exchange", [
        dict(op="info", bytes=ONE.hex(), pipeline=answer("pipeline", bytes([0]) + DEFAULT_500)),
        dict(op="arm"), dict(op="start"),
        dict(op="data", packets=[s.next(flags=DISC | ACTIVE)] + s.run(2, flags=ACTIVE)),
        dict(op="stop"),
        dict(op="save", label="no-exchange", extra=dict(experiment="demo-record"), events=None,
             now=[13_000.0, 1_767_227_600.0]),
    ], "no time exchange at all: the device's own seconds")

    add("nothing-recorded", [
        dict(op="info", bytes=ONE.hex(), pipeline=answer("pipeline", bytes([0]) + DEFAULT_500)),
        *syncs(14_000.0, 1_767_227_700.0, 4000.0, 1),
        dict(op="arm"), dict(op="disarm"),
        dict(op="save", label="nothing-recorded", extra=dict(experiment="demo-record"), events=None,
             now=[14_001.0, 1_767_227_701.0]),
    ], "armed and saved with no samples")

    s = Stream(15)
    add("one-sample", [
        dict(op="info", bytes=ONE.hex(), pipeline=answer("pipeline", bytes([0]) + DEFAULT_500)),
        *syncs(15_000.0, 1_767_227_800.0, 4000.0, 1),
        dict(op="arm"), dict(op="start"),
        dict(op="data", packets=[s.next(1, flags=DISC | ACTIVE)]),
        dict(op="stop"),
        dict(op="save", label="one-sample", extra=dict(experiment="demo-record"), events=None,
             now=[15_001.0, 1_767_227_801.0]),
    ], "one sample")

    s = Stream(16)
    add("unknown-rate", [
        dict(op="info", bytes=ONE.hex(), pipeline=answer("pipeline", bytes([0]) + DEFAULT_500)),
        *syncs(16_000.0, 1_767_227_900.0, 4000.0, 1),
        dict(op="arm"), dict(op="start"),
        dict(op="data", packets=[s.next(flags=DISC | ACTIVE, rate=None)] + s.run(3, flags=ACTIVE, rate=None)),
        dict(op="stop"),
        dict(op="save", label="unknown-rate", extra=dict(experiment="demo-record"), events=None,
             now=[16_001.0, 1_767_227_901.0]),
    ], "packets whose rate code this contract does not define")

    s = Stream(18)
    t0 = s.t
    add("empty-packets", [
        dict(op="info", bytes=ONE.hex(), pipeline=answer("pipeline", bytes([0]) + DEFAULT_500)),
        *syncs(18_000.0, 1_767_228_100.0, 4000.0, 1),
        dict(op="arm"), dict(op="start"),
        dict(op="data", packets=[packet(0, t0, [], flags=DISC | ACTIVE), packet(0, t0 + 1000, s.rows(1), flags=ACTIVE),
                                 packet(1, t0 + 3000, [], flags=ACTIVE)]),
        dict(op="stop"),
        dict(op="save", label="empty-packets", extra=dict(experiment="demo-record"), events=None,
             now=[18_001.0, 1_767_228_101.0]),
    ], "packets that carry no samples around one that carries one: the rate from the packets")

    s = Stream(17, index=0xFFFFFFFF - 40)
    add("count-wraps", [
        dict(op="info", bytes=ONE.hex(), pipeline=answer("pipeline", bytes([0]) + DEFAULT_500)),
        *syncs(17_000.0, 1_767_228_000.0, 4000.0, 1),
        dict(op="arm"),
        dict(op="data", packets=s.run(6, flags=ACTIVE)),
        dict(op="save", label="count-wraps", extra=dict(experiment="demo-record"), events=None,
             now=[17_001.0, 1_767_228_001.0]),
    ], "the device's count passes 2^32 while armed on a running stream")
    return out


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


async def run(session: dict, captures: pathlib.Path) -> dict:
    prov.use_captures_dir(captures)
    prov.use_site(None)
    prov.use_repo(None)
    out = dict(clock=[], config_after=None, processing=None)
    dev = link = rec = None
    for step in session["script"]:
        op = step["op"]
        if op == "info":
            dev, link = device(bytes.fromhex(step["bytes"]))
            if step.get("pipeline"):
                link.answers[OPS["pipeline"]] = bytes.fromhex(step["pipeline"])
            rec = experiments.Recorder()
            dev.on_sample, dev.on_packet = rec.on_sample, rec.on_packet
        elif op == "site":
            prov.use_site(step["site"])
        elif op == "sync":
            CLOCKS.set(step["before"])
            link.sync = (step["after"], step["ticks"])
            await dev.time_sync()
            out["clock"].append(dev.clock_quality())
        elif op == "start":
            await dev.start()
        elif op == "reset_epoch":
            await dev.reset_epoch()
        elif op == "arm":
            rec.arm(dev)
        elif op == "data":
            for h in step["packets"]:
                dev._on_data(None, bytes.fromhex(h))
        elif op == "link_lost":
            link.is_connected = False
        elif op == "stop":
            out["link_ok"] = await experiments._safe_stop(dev)
        elif op == "disarm":
            rec.armed = False
        elif op == "config_after":
            link.status = bytes.fromhex(step["status"])
            for k in ("pipeline", "indicator", "model_interval", "name"):
                if step.get(k):
                    link.answers[OPS[k]] = bytes.fromhex(step[k])
            out["config_after"] = await experiments._config_after(dev, step["link_ok"])
        elif op == "processing":
            if step.get("pipeline"):
                link.answers[OPS["pipeline"]] = bytes.fromhex(step["pipeline"])
            out["processing"] = await experiments._processing(dev)
        elif op == "save":
            extra = dict(step["extra"])
            if step.get("with_processing"):
                proc = out["processing"]
                if proc:
                    extra["processing"] = proc
                extra["signal"] = proc["description"] if proc else "unknown"
            cfg = out["config_after"] if step.get("config_after") == "read" else step.get("config_after")
            CLOCKS.set(step["now"])
            rec.save(dev, step["label"], extra, step.get("events"), regs_after=step.get("regs_after"),
                     aborted=step.get("aborted"), config_after=cfg)
            label = step["label"]
            files = {}
            for p in sorted(captures.iterdir()):
                files[p.name] = dict(read_only=not (p.stat().st_mode & stat.S_IWUSR))
            out["files"] = files
            out["meta_json"] = (captures / f"{label}.meta.json").read_text()
            ev = captures / f"{label}.events.json"
            out["events_json"] = ev.read_text() if ev.exists() else None
            sums = captures / f"{label}.sha256"
            out["sha256_names"] = ([line.split(None, 1)[1] for line in sums.read_text().splitlines()]
                                   if sums.exists() else None)
            out["sessions_jsonl"] = (captures / "SESSIONS.jsonl").read_text()
            out["arrays"] = _digest(captures / f"{label}.npz")
        else:
            raise ValueError(f"no step {op!r}")
    return out


def main() -> int:
    work = pathlib.Path(tempfile.mkdtemp(prefix="intomind-recorder-"))
    try:
        result = dict(library=intomind.__version__, sessions=[], montage=montage_cases(), chains=chain_cases())
        for s in sessions():
            d = work / s["name"]
            s["expected"] = asyncio.run(run(s, d))
            result["sessions"].append(s)
            for p in d.iterdir():
                p.chmod(0o644)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    (HERE / "sessions.json").write_text(json.dumps(result, indent=1, sort_keys=False) + "\n")
    print(f"{len(result['sessions'])} sessions, {len(result['montage'])} montage cases and "
          f"{len(result['chains'])} chains, from intomind {intomind.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
