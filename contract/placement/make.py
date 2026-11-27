"""The placement fixture: where a device was put, said with the whole 10-10
system, and what the Python library makes of it, which every other
implementation's placement is held to.

    PYTHONPATH=<a checkout of the Python library> python3 contract/placement/make.py

Writes expected.json beside this script: every position on offer, each
position by key and what is under it, the refusals in the library's words,
and placements made with one position, with sites or with a preset on two
layouts whose slots are made up and on the wiring preset the library's
profile gives the IntoMind One, each resolved, turned into a montage and
described. Run it again and nothing changes unless the library does.
"""
from __future__ import annotations

import json
import pathlib
import sys
import types

HERE = pathlib.Path(__file__).resolve().parent
# Made without a sound: the library's experiments speak their cues and play
# their tones through the host's speakers, so every cue goes nowhere here,
# before anything of the library runs.
from intomind import audio as _audio  # noqa: E402
_audio.play = lambda tone, text="": None

import intomind  # noqa: E402
from intomind import client, montage as M, placement as PL, protocol as P  # noqa: E402


def attempt(fn) -> dict:
    """What the library returns, or the words it refuses in."""
    try:
        return dict(result=fn())
    except Exception as e:  # noqa: BLE001
        return dict(raises=f"{type(e).__name__}: {e}")


CLUSTER = PL.layout(
    "four-around-a-center", "four electrodes around a center reference",
    [PL.slot("north", "north electrode", name="e1", angle_deg=0),
     PL.slot("east", "east electrode", name="e2", angle_deg=90),
     PL.slot("south", "south electrode", name="e3", angle_deg=180),
     PL.slot("west", "west electrode", name="e4", angle_deg=270),
     PL.slot("center", "center reference", PL.REFERENCE, name="ref"),
     PL.slot("drive", "bias electrode", PL.BIAS, name="drive")],
    [dict(index=i + 1, derivation=M.REFERENTIAL, positive=k, negative="center")
     for i, k in enumerate(("north", "east", "south", "west"))],
    rotatable=True, electrode="dry gold")

PAIRS = PL.layout(
    "two-pairs", "two bipolar pairs",
    [PL.slot("a1", "left front"), PL.slot("a2", "left back"),
     PL.slot("b1", "right front"), PL.slot("b2", "right back"),
     PL.slot("drive", "bias", PL.BIAS)],
    [dict(index=1, derivation=M.BIPOLAR, positive="a1", negative="a2", label="left pair"),
     dict(index=2, derivation=M.BIPOLAR, positive="b1", negative="b2")],
    presets=[PL.preset("codes", "over the central strip",
                       dict(a1="C3", a2="CP3", b1="C4", b2="CP4", drive="Fpz"))])


def _wiring(every: bool) -> dict:
    """The wiring preset the library's profile gives the IntoMind One's own
    Device Info from the conformance vectors, with every capability declared
    when `every`, which brings the bias electrode."""
    vectors = json.loads((HERE.parent / "conformance.json").read_text())["decode"]
    b = bytearray.fromhex(next(v for v in vectors if v["name"] == "device info, the IntoMind One")["bytes"])
    if every:
        b[15:17] = sum(P.CAPABILITIES.values()).to_bytes(2, "little")
    dev = client.Device(types.SimpleNamespace(address="02:00:00:00:00:01", name="IntoMind One", details={}))
    dev.info = P.decode_device_info(bytes(b))
    return dev.profile()["wiring"]["layout"]


WIRED = _wiring(False)
WIRED_BIAS = _wiring(True)

# Every key on offer, every key of the earlier list, and keys that are not.
KEYS = ([s["key"] for s in PL.positions()] + list(PL.SCALP)
        + ["left-elbow", "C99", "cpz", "Other", "", 7, None])

PLACEMENTS = [
    ("one position", CLUSTER, dict(position="CPz")),
    ("one position off the system, with a note", CLUSTER,
     dict(position="other", note="left forearm")),
    ("one position, the bias not driven", CLUSTER,
     dict(position="Iz", bias_driven=False)),
    ("one position, turned", CLUSTER, dict(position="P9", rotation_deg=90)),
    ("one position named the earlier way", CLUSTER, dict(position="left-crown-side")),
    ("one position with an override", CLUSTER,
     dict(position="C3", over={"north": ["motor"]})),
    ("one position and sites", CLUSTER, dict(position="CPz", sites={})),
    ("one position and a preset", PAIRS, dict(position="CPz", preset="codes")),
    ("one position that is not one", CLUSTER, dict(position="C99")),
    ("sites at 10-10 positions", PAIRS,
     dict(sites=dict(a1="F7", a2="FT7", b1="F8", b2="FT8", drive="Fpz"))),
    ("sites named both ways", PAIRS,
     dict(sites=dict(a1="left-temple-front", a2="FT7", b1="F8", b2="right-temple-back",
                     drive="forehead-center"))),
    ("a site off the system", PAIRS,
     dict(sites=dict(a1="other", a2="FT7", b1="F8", b2="FT8"), bias_driven=False,
          note="a1 on the cheek")),
    ("a preset at 10-10 positions", PAIRS, dict(preset="codes")),
    ("an override at a 10-10 position", PAIRS,
     dict(preset="codes", over=dict(b1=["somatosensory", "temporalis"]))),
    ("an override naming no region", PAIRS, dict(preset="codes", over=dict(b1=["broca"]))),
    ("the wiring preset at one position", WIRED, dict(position="CPz")),
    ("the wiring preset off the system, with a note", WIRED,
     dict(position="other", note="behind the left ear")),
    ("the wiring preset with its bias", WIRED_BIAS, dict(position="TP8")),
    ("the wiring preset, the bias not driven", WIRED_BIAS, dict(position="Pz", bias_driven=False)),
]


def placement_cases() -> list[dict]:
    out = []
    for name, lay, kw in PLACEMENTS:
        c = dict(name=name, layout=lay, args=kw)
        c.update(attempt(lambda: PL.placement(lay, **kw)))
        if "result" in c:
            p = c["result"]
            c["resolved"] = PL.resolve(lay, p)
            c["montage"] = PL.to_montage(lay, p)
            c["described"] = PL.describe(lay, p)
            c["described_plain"] = PL.describe(lay, p, extras=False)
        out.append(c)
    return out


def main() -> int:
    result = dict(
        library=intomind.__version__,
        positions=PL.positions(),
        regions=list(PL.REGIONS.values()),
        scalp=[dict(key=k, **attempt(lambda: PL.scalp(k))) for k in KEYS],
        regions_under=[dict(key=k, **attempt(lambda: PL.regions_under(k))) for k in KEYS],
        placements=placement_cases(),
    )
    (HERE / "expected.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"{len(result['positions'])} positions, {len(KEYS)} keys and "
          f"{len(result['placements'])} placements, from intomind {intomind.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
