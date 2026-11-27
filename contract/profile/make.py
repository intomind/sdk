"""The profile fixture: what the Python library says a device can do, and
its instrument layer's answers, which every other implementation is held to.

    PYTHONPATH=<a checkout of the Python library> python3 contract/profile/make.py

`Device.profile` is asked about devices that declare every combination of
capabilities that changes what it says: none, each one alone, the three
signal sources in every combination, a model ready and not, every rate
mask, all of them at once, and the shorter layouts of older devices. Each
device is the IntoMind One's Device Info from the conformance vectors with
its capability words, rate mask and model fields changed, and a hardware
version that is plainly made up, except where the wiring preset is the
question: there the hardware version is the IntoMind One's own from the
same vectors, and other hardware's that has no preset.

The instrument module is asked about contact from lead-off bytes, the
enforced preconditions, and the effective settings a manifest records,
including what each refuses and the words it refuses in.

Writes profile.json beside this script. Run it again and nothing changes
unless the library does.
"""
from __future__ import annotations

import json
import pathlib
import sys
import types
import warnings

warnings.simplefilter("ignore")
HERE = pathlib.Path(__file__).resolve().parent
# Made without a sound: the library's experiments speak their cues and play
# their tones through the host's speakers, so every cue goes nowhere here,
# before anything of the library runs.
from intomind import audio as _audio  # noqa: E402
_audio.play = lambda tone, text="": None

import intomind  # noqa: E402
from intomind import client, instrument as I, protocol as P  # noqa: E402

#: A hardware version no device has, so nothing here repeats a real one.
MADE_UP_HARDWARE = (9, 8, 7)


def _conformance_info() -> bytes:
    vectors = json.loads((HERE.parent / "conformance.json").read_text())["decode"]
    v = next(v for v in vectors if v["name"] == "device info, the IntoMind One")
    return bytes.fromhex(v["bytes"])


ONE = _conformance_info()


def info(*, caps: int = 0, caps_high: int = 0, rates: int = 7, slots: int = 4, embed: int = 96,
         hardware=MADE_UP_HARDWARE, length: int | None = None, channels: int | None = None) -> bytes:
    """The IntoMind One's Device Info with its capability words, rate mask,
    model fields, hardware version and channel count set, and its layout cut
    to `length`."""
    b = bytearray(ONE)
    if channels is not None:
        b[5] = channels
    b[15:17] = caps.to_bytes(2, "little")
    b[17] = rates
    b[28:31] = bytes(hardware)
    b[40:42] = embed.to_bytes(2, "little")
    b[46] = slots
    b[76:80] = caps_high.to_bytes(4, "little")
    if length is not None:
        b = b[:length]
        if length > P.INFO_V01_LEN:
            b[27] = length
        if length == 76:
            b[1] = 2           # a device whose info ends at 76 bytes speaks 1.2
    return bytes(b)


def device(info_bytes: bytes | None):
    dev = client.Device(types.SimpleNamespace(address="02:00:00:00:00:01", name="IntoMind One", details={}))
    dev.info = None if info_bytes is None else P.decode_device_info(info_bytes)
    return dev


def profile_cases() -> list[dict]:
    low = dict(P.CAPABILITIES)
    high = dict(P.CAPABILITIES_HIGH)
    every_low = sum(low.values())
    every_high = sum(high.values())
    cases = [dict(name="no device info", info=None),
             dict(name="nothing declared", info=info())]
    for name, bit in low.items():
        cases.append(dict(name=f"only {name}", info=info(caps=bit)))
    for name, bit in high.items():
        cases.append(dict(name=f"only {name}", info=info(caps_high=bit)))
    for t in (0, 1):
        for s in (0, 1):
            for y in (0, 1):
                caps = (low["test_signal"] if t else 0) | (low["input_short"] if s else 0)
                cases.append(dict(name=f"signal sources: test {t}, short {s}, synthetic {y}",
                                  info=info(caps=caps, caps_high=high["synthetic"] if y else 0)))
    cases += [
        dict(name="a model, ready", info=info(caps=low["model"] | low["model_ready"])),
        dict(name="a model, not ready", info=info(caps=low["model"])),
        dict(name="a model with no slots and no width", info=info(caps=low["model"], slots=0, embed=0)),
        dict(name="ready without a model", info=info(caps=low["model_ready"])),
    ]
    for rates in (0, 1, 2, 3, 4, 5, 6, 7, 0xFF):
        cases.append(dict(name=f"rate mask {rates}", info=info(rates=rates)))
    cases += [
        dict(name="everything", info=info(caps=every_low, caps_high=every_high)),
        dict(name="the IntoMind One's own declaration", info=info(caps=int.from_bytes(ONE[15:17], "little"),
                                                                  caps_high=int.from_bytes(ONE[76:80], "little"))),
        dict(name="everything, on a device whose info ends at 76 bytes",
             info=info(caps=every_low, caps_high=every_high, length=76)),
        dict(name="the original 27 bytes, which report no hardware", info=info(caps=every_low, length=P.INFO_V01_LEN)),
        dict(name="a hardware version of zeros", info=info(caps=low["leadoff"], hardware=(0, 0, 0))),
    ]
    # The wiring preset, chosen by the hardware version and nothing else.
    one_caps = int.from_bytes(ONE[15:17], "little")
    one_high = int.from_bytes(ONE[76:80], "little")
    one_hardware = tuple(ONE[28:31])
    cases += [
        dict(name="the IntoMind One's hardware, which has a wiring preset",
             info=info(caps=one_caps, caps_high=one_high, hardware=one_hardware)),
        dict(name="the IntoMind One's hardware, no bias drive",
             info=info(caps=one_caps & ~low["bias_drive"], caps_high=one_high, hardware=one_hardware)),
        dict(name="the IntoMind One's hardware, everything declared",
             info=info(caps=every_low, caps_high=every_high, hardware=one_hardware)),
        dict(name="the IntoMind One's hardware, one channel",
             info=info(caps=one_caps, caps_high=one_high, hardware=one_hardware, channels=1)),
        dict(name="the IntoMind One's hardware, eight channels",
             info=info(caps=one_caps, caps_high=one_high, hardware=one_hardware, channels=8)),
        dict(name="other hardware, 1.0.0, which has no preset",
             info=info(caps=one_caps, caps_high=one_high, hardware=(1, 0, 0))),
    ]
    for c in cases:
        b = c["info"]
        c["info"] = None if b is None else b.hex()
        c["profile"] = device(b).profile()
    return cases


def _outcome(fn) -> dict:
    try:
        return dict(result=fn())
    except Exception as e:  # noqa: BLE001
        return dict(raises=type(e).__name__, message=str(e))


def contact_cases() -> list[dict]:
    rows = [
        (0, 2), (1, 2), (2, 2), (3, 2), (5, 3), (255, 8), (0, 8), (0b10101010, 8), (0, 9), (0, 1), (0, 0),
        ([], 2), ([], 4), ([0, 0, 0, 0], 4), ([1, 0, 0, 0], 2), ([1, 1, 0, 0], 2), ([1, 1, 1, 0], 2),
        ([0, 1, 2, 3], 4), ([3] * 3 + [0], 2), ([3] * 2 + [0] * 6, 2), ([2] * 25 + [0] * 75, 2),
        ([2] * 26 + [0] * 74, 2), ([1] * 3 + [0] * 9, 2), ([True, False], 2), ([True, True, True, True], 1),
        ([1.5], 2), ([255.9, 0.25], 8), ([0, 300], 2), ([-1], 2), (300, 2), ([0] * 4, 9), ([7] * 10, 3),
    ]
    out = []
    for loff, n in rows:
        out.append(dict(loff=loff, n_channels=n, **_outcome(lambda: I.contact_from_leadoff(loff, n))))
    return out


def channel_cases() -> list[dict]:
    out = []
    for channels, default in ((4, 2), (2, 2), (0, 2), (None, 2), (None, 3), (8, 5)):
        dev = types.SimpleNamespace(info=None if channels is None else types.SimpleNamespace(channels=channels))
        out.append(dict(channels=channels, default=default, result=I.contact_channels(dev, default)))
    out.append(dict(channels="no device", default=2, result=I.contact_channels(None, 2)))
    return out


def enforced_cases() -> list[dict]:
    rows = [
        ("nothing declared", dict(name="baseline", enforced={}), dict(seconds=30)),
        ("enforced of null", dict(name="baseline", enforced=None), {}),
        ("no enforced key", dict(name="baseline"), {}),
        ("required and given", dict(name="x", enforced=dict(consent="required")), dict(consent=True)),
        ("required and missing", dict(name="x", enforced=dict(consent="required")), {}),
        ("required and empty", dict(name="x", enforced=dict(consent="required")), dict(consent="")),
        ("required and zero", dict(name="x", enforced=dict(consent="required")), dict(consent=0)),
        ("a value met", dict(name="x", enforced=dict(mode="normal")), dict(mode="normal")),
        ("a value not met", dict(name="x", enforced=dict(mode="normal")), dict(mode="test")),
        ("a value missing", dict(name="x", enforced=dict(mode="normal")), {}),
        ("a number met by a float", dict(name="x", enforced=dict(n=5)), dict(n=5.0)),
        ("a number not met by text", dict(name="x", enforced=dict(n=5)), dict(n="5")),
        ("a float required", dict(name="x", enforced=dict(n=2.5)), dict(n=2)),
        ("true met by one", dict(name="x", enforced=dict(flag=True)), dict(flag=1)),
        ("null required", dict(name="x", enforced=dict(z=None)), dict(z=0)),
        ("a list required", dict(name="x", enforced=dict(z=[1, 2])), dict(z=[1, 2])),
        ("two keys, the second broken", dict(name="x", enforced=dict(a="required", b="on")), dict(a=1, b="off")),
        ("enforced as a list", dict(name="x", enforced=[1]), {}),
        ("parameters of null", dict(name="x", enforced=dict(a="required")), None),
    ]
    out = []
    for name, spec, params in rows:
        out.append(dict(name=name, spec=spec, params=params,
                        **_outcome(lambda: I.check_enforced(spec, params))))
    return out


def effective_cases() -> list[dict]:
    rows = [
        (dict(gain=24, rate_sps=500, mode="normal", leadoff=None), dict(gain=12, bias="on"),
         dict(seconds=30, note="desk", gain=1.5)),
        ({}, {}, {}),
        (dict(a=1), {}, dict(a=2)),
        (dict(rate_sps=500.0), dict(rate_sps=250), dict(steps=[["Close your eyes", 20.0]])),
    ]
    return [dict(actual=a, applied=b, params=p, result=I.effective_settings({}, a, b, p)) for a, b, p in rows]


def main() -> int:
    out = dict(library=intomind.__version__, profiles=profile_cases(), contact=contact_cases(),
               contact_channels=channel_cases(), min_fraction=I.CONTACT_MIN_FRACTION, tiers=list(I.TIERS),
               enforced=enforced_cases(), effective=effective_cases())
    (HERE / "profile.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(out['profiles'])} profiles, {len(out['contact'])} contact cases, "
          f"{len(out['enforced'])} enforced cases, {len(out['effective'])} effective settings, "
          f"from intomind {intomind.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
