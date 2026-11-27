"""The analysis fixture: synthetic recordings, and what the Python library's
analyses return for each, which every other implementation's analyses are
held to.

    PYTHONPATH=<a checkout of the Python library> python3 contract/analysis/make.py

The Python library is the reference for every number an analysis reports, so
this script asks it, and only it. Every signal here is generated from a
seeded random source and shaped to reach each branch of each analysis: a
saccade that inverts across the two channels and one that does not, blinks
large and small, a clench that lifts the muscle band and trials whose rest
and clench windows are the same samples, closed and open blocks where alpha
moves and where only its reference band does, a wall clock that steps in the
middle of a run, gaps, a flat channel and a railed one, and a game pad's
buttons and sticks beside a recording, stored as arrays. None is a person's.

Writes, beside this script:

    captures/      the recordings, as every IntoMind program writes them
    analysis.json  the library, numpy and scipy versions. Every case, each
                   an analysis, its recordings and its parameters as a page
                   sends them, with what the library returned or the
                   exception it raised. The collection as `list_captures`
                   lists it, newest first. The events and the controller
                   inputs of recordings, each on the same timeline as its
                   samples, as `events` and `inputs` return them. An
                   analysis an application registers with `several`, run
                   as an application runs it: once, handed every selected
                   label in a list, where any other is run on each label
                   separately, and which of the library's own take one. A
                   select held to its choices, as the library holds every
                   select: an entry registered here whose choices are text,
                   and text and numbers, and the library's own selects of 0
                   and 1, each with the values it takes and the words it
                   refuses the others in. The
                   numerics the analyses stand on, each asked of scipy
                   directly over inputs the recordings do not reach: filter
                   design at other rates, forward and backward filtering,
                   Welch's spectrum at transform lengths of every kind, the
                   three tests on each of their methods, and the normal and
                   Student tails

Run it again and nothing changes unless the library does: every random
source is seeded, the archives are stamped with a fixed time, and the files'
modification times are set.
"""
from __future__ import annotations

import json
import os
import pathlib
import platform
import shutil
import warnings
import zipfile

import numpy as np

warnings.simplefilter("ignore")
HERE = pathlib.Path(__file__).resolve().parent
CAPTURES = HERE / "captures"
# Made without a sound: the library's experiments speak their cues and play
# their tones through the host's speakers, so every cue goes nowhere here,
# before anything of the library runs.
from intomind import audio as _audio  # noqa: E402
_audio.play = lambda tone, text="": None

import intomind  # noqa: E402
import scipy  # noqa: E402
from intomind import analysis as A, provenance as prov  # noqa: E402

#: The IntoMind One's reference and converter, as its Device Info reports
#: them, and its gain.
VREF_UV, ADC_BITS, GAIN = 4_500_000, 24, 24
LSB_UV = 2.0 * VREF_UV / (GAIN * (1 << ADC_BITS))
FS = 249.95
T0, M0 = 1767262830.0, 4503695.6
STAMP = (2026, 1, 1, 0, 0, 0)


# ---------------------------------------------------------------- writing
def savez(path: pathlib.Path, arrays: dict) -> None:
    """`np.savez_compressed`, with every member stamped at one fixed time so
    the bytes, and so the checksums, are the same on every run."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, a in arrays.items():
            info = zipfile.ZipInfo(f"{name}.npy", date_time=STAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            with z.open(info, "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(a), allow_pickle=False)


def anchors(step_at: float | None = None, step_s: float = 0.0, span: float = 200.0) -> list:
    """Clock exchanges every five seconds as (realtime, monotonic), the
    realtime reading moved by `step_s` from `step_at` seconds on: a wall
    clock stepped in the middle of the run."""
    out = []
    for i in range(int(span / 5) + 1):
        t = i * 5.0
        jump = step_s if step_at is not None and t >= step_at else 0.0
        out.append([T0 + t + jump, M0 + t])
    return out


def manifest(label: str, n: int, *, channels: int, fs: float = FS, version: int = 7,
             clock_anchors: list | None = None, **extra) -> dict:
    """A manifest as the recorder writes one, with generated values."""
    m = dict(
        manifest_version=version, label=label,
        started_utc="2026-01-01T10:20:30Z", finished_utc="2026-01-01T10:22:30Z",
        captured_unix=T0 + n / fs,
        device_id="00000000c0de0001", firmware_version="1.4.5",
        tick_hz=1_000_000, vref_uv=VREF_UV, adc_bits=ADC_BITS, channels=channels,
        device_proto="1.4", gain=GAIN, rate_sps=250, fs_effective=fs,
        protocol_id="fixture", protocol_params={},
        signal_source="normal", synthetic=False,
        quality=dict(gaps=[], n_gaps=0, epoch_resets=0, samples_received=n,
                     samples_expected=n, samples_lost=0),
        clock=dict(n_syncs=8, skew=1.0000123, offset=4501984.819461971, skew_pinned=False,
                   residual_ms=0.41, span_s=2.5, fitted_against="monotonic",
                   anchors=clock_anchors if clock_anchors is not None else anchors(),
                   frame_mono=M0, wall_clock_steps=[], wall_clock_stepped=False),
        n=n, experiment="fixture",
    )
    m.update(extra)
    return m


def to_counts(uv: np.ndarray) -> np.ndarray:
    """Microvolts, channels by samples, as the converter's counts."""
    return np.round(uv / LSB_UV).astype(np.int64)


def write(label: str, *, uv=None, counts=None, index=None, meta=None, leadoff=None,
          leadoff_active=None, leadoff_judged=None, events=None, arrays_drop=(), checksum=True, fs=FS,
          host=None) -> None:
    """One recording, the arrays laid out as the recorder lays them out.
    `host`, where given, is the host times of its samples."""
    c = to_counts(uv) if counts is None else counts
    nch, n = c.shape
    index = np.arange(n, dtype=np.int64) if index is None else np.asarray(index, np.int64)
    ticks = (index * (1_000_000 / fs)).astype(np.int64)
    host = T0 + index / fs if host is None else np.asarray(host, float)
    arrays = dict(counts=np.ascontiguousarray(c.T).T, index=index, device_ticks=ticks,
                  host_time=host,
                  leadoff=np.zeros(n, np.uint8) if leadoff is None else leadoff,
                  gap_before=np.concatenate(([False], np.diff(index) != 1))[:n])
    if leadoff_active is not None:
        arrays["leadoff_active"] = leadoff_active
    if leadoff_judged is not None:
        arrays["leadoff_judged"] = leadoff_judged
    for k in arrays_drop:
        arrays.pop(k)
    savez(CAPTURES / f"{label}.npz", arrays)
    (CAPTURES / f"{label}.meta.json").write_text(json.dumps(meta, indent=2))
    if events is not None:
        (CAPTURES / f"{label}.events.json").write_text(json.dumps(events, indent=2))
    if checksum:
        prov.write_checksums(label)


def cue(t: float, cond: str, *, step_at=None, step_s=0.0, **kw) -> dict:
    """A cue `t` seconds in, stamped on both clocks as the recorder stamps
    one: the wall clock moved by `step_s` once it has stepped."""
    jump = step_s if step_at is not None and t >= step_at else 0.0
    return dict(cond=cond, kind="cue", host_time=T0 + t + jump, mono=M0 + t, **kw)


# ---------------------------------------------------------------- signals
def background(rng, n: int, channels: int, *, white=2.0, red=1.5, dc=-4200.0, fs=FS) -> np.ndarray:
    """White noise, a red component, and a DC offset per channel, microvolts."""
    out = np.empty((channels, n))
    for c in range(channels):
        w = rng.normal(0, 1, n)
        r = np.empty(n)
        acc = 0.0
        for i in range(n):
            acc = 0.97 * acc + w[i]
            r[i] = acc
        out[c] = dc + 600 * c + white * rng.normal(0, 1, n) + red * r / 4
    return out


def tone(n: int, hz: float, amp: float, phase: float = 0.0, fs=FS) -> np.ndarray:
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * hz * t + phase)


def bump(n: int, at_s: float, width_s: float, amp: float, fs=FS) -> np.ndarray:
    """A half sine `width_s` long, peaking at `amp`, starting `at_s` in."""
    out = np.zeros(n)
    a, w = int(round(at_s * fs)), int(round(width_s * fs))
    out[a:a + w] = amp * np.sin(np.pi * np.arange(w) / w)
    return out


def plateau(n: int, at_s: float, hold_s: float, amp: float, fs=FS) -> np.ndarray:
    """A smooth rise to `amp`, held for `hold_s`, and a smooth return."""
    out = np.zeros(n)
    edge = int(round(0.06 * fs))
    a, h = int(round(at_s * fs)), int(round(hold_s * fs))
    rise = amp * 0.5 * (1 - np.cos(np.pi * np.arange(edge) / edge))
    out[a:a + edge] = rise
    out[a + edge:a + edge + h] = amp
    out[a + edge + h:a + 2 * edge + h] = rise[::-1]
    return out


# ---------------------------------------------------------------- the recordings
def make_recordings() -> None:
    rng = np.random.default_rng(20261102)

    # Four channels at rest, two runs either side of a hole of 37 samples,
    # alpha and the mains line, an electrode off for a while, the fourth
    # channel railed for a moment.
    n = 10_000
    uv = background(rng, n, 4)
    for c in range(4):
        uv[c] += tone(n, 10.0, 15.0, c) + tone(n, 60.0, 4.0, 2 * c) + tone(n, 3.0, 6.0)
    full = ((1 << 23) - 1) * LSB_UV
    uv[3, 4000:4200] = full
    idx = np.concatenate([np.arange(5000), np.arange(5037, 5037 + 5000)])
    lo = np.zeros(n, np.uint8)
    lo[2000:2600] |= 0x01
    lo[7000:7100] |= 0x06
    # Firmware 1.4.6, so its lead-off bits are the device's verdict on contact:
    # an older IntoMind One cannot check electrode contact.
    write("rest-4ch", uv=uv, index=idx, meta=manifest("rest-4ch", n, channels=4, firmware_version="1.4.6"),
          leadoff=lo, leadoff_active=np.ones(n, bool))

    # Contact checked over part of a recording of firmware 1.4.6: detection
    # off for the first 200 samples, then on, with the first 128 samples it
    # ran no verdict, before the device's first judgment. Of the 800 samples
    # checked, channel 1's electrode is on throughout, channel 2's is off in
    # exactly a quarter, channel 3's in one sample more, and channel 4's in
    # all. Where detection was off every bit is clear, and where no verdict
    # was given every bit is set.
    n = 1128
    k = np.arange(n)
    active = k >= 200
    judged = ~((k >= 200) & (k < 328))
    lo = np.zeros(n, np.uint8)
    lo[~judged] = 0x0F
    checked = np.flatnonzero(active & judged)
    lo[checked[:200]] |= 0x02
    lo[checked[:201]] |= 0x04
    lo[checked] |= 0x08
    # Its own random source, so the recordings after it are what they were.
    write("contact-checked-4ch", uv=background(np.random.default_rng(1128), n, 4),
          meta=manifest("contact-checked-4ch", n, channels=4, firmware_version="1.4.6"),
          leadoff=lo, leadoff_active=active, leadoff_judged=judged)

    # Shorter than one spectrum segment, with a flat channel.
    n = 1500
    uv = background(rng, n, 2)
    uv[0] += tone(n, 10.0, 12.0)
    uv[1] = -3000.0
    write("short-2ch", uv=uv, meta=manifest("short-2ch", n, channels=2))

    # Saccades: the two channels deflect in opposite directions, reversing
    # with direction, and a sham cue moves nothing. The sham cues come first,
    # well before any movement, since a slow high-pass spreads each
    # deflection over seconds. The wall clock steps by 0.75 s at 20 s, so
    # only the aligned event times land on the samples.
    conds = ["sham"] * 4 + ["left", "right"] * 6
    times = [3.0 + 4.0 * i for i in range(4)] + [24.0 + 4.0 * i for i in range(12)]
    n = int((times[-1] + 4.0) * FS)
    for label, same_sign, sham_moves in (("saccade-pass", False, False), ("saccade-fail", True, True)):
        uv = background(rng, n, 2, white=1.5, red=0.15)
        for t, c in zip(times, conds):
            if c == "sham" and not sham_moves:
                continue
            s = {"left": 1.0, "right": -1.0, "sham": 0.6}[c]
            d = plateau(n, t + 0.15, 0.5, 90.0 * s)
            uv[0] += d
            uv[1] += d if same_sign else -d
        clock = anchors(step_at=20.0, step_s=0.75)
        ev = [cue(t, c, step_at=20.0, step_s=0.75, trial=i) for i, (t, c) in enumerate(zip(times, conds))]
        write(label, uv=uv, meta=manifest(label, n, channels=2, clock_anchors=clock), events=ev)

    # Blinks: a large bump at a steady latency on both channels, and a sham
    # cue with nothing, then the same with blinks too small to count. The
    # first event predates dual stamping and carries no monotonic reading.
    conds = ["blink"] * 8 + ["sham"] * 4
    order = [0, 8, 1, 2, 9, 3, 4, 10, 5, 6, 11, 7]
    conds = [conds[i] for i in order]
    times = [2.5 + 2.5 * i for i in range(len(conds))]
    n = int((2.5 * len(conds) + 3.0) * FS)
    for label, amp in (("blink-pass", 220.0), ("blink-fail", 60.0)):
        uv = background(rng, n, 2, white=2.0, red=1.0)
        for i, (t, c) in enumerate(zip(times, conds)):
            if c != "blink":
                continue
            lat = 0.22 + 0.02 * ((i * 7) % 5 - 2)
            b = bump(n, t + lat, 0.25, amp)
            uv[0] += b
            uv[1] += 0.8 * b
        ev = [cue(t, c, trial=i) for i, (t, c) in enumerate(zip(times, conds))]
        del ev[0]["mono"]
        write(label, uv=uv, meta=manifest(label, n, channels=2), events=ev)

    # Jaw clench: muscle noise for a second after each cue. Twelve trials,
    # every one a clear win. Twelve where two clench windows repeat their
    # rest window exactly, so two differences are zero. Sixteen with three
    # such trials. Two trials, too few to test.
    for label, n_trials, same in (("jaw-pass", 12, ()), ("jaw-zeros", 12, (3, 8)),
                                  ("jaw-ties", 16, (2, 7, 11)), ("jaw-few", 2, ())):
        times = [2.0 + 3.0 * i for i in range(n_trials)]
        n = int((3.0 * n_trials + 3.0) * FS)
        uv = background(rng, n, 2, white=2.0, red=1.0)
        w = int(1.0 * FS)
        for i, t in enumerate(times):
            a = int(np.searchsorted(T0 + np.arange(n) / FS, T0 + t))
            if i in same:
                uv[:, a:a + w] = uv[:, a - w:a]
            else:
                uv[:, a:a + w] += rng.normal(0, 35.0, (2, w))
        ev = [cue(t, "clench", trial=i) for i, t in enumerate(times)]
        write(label, uv=uv, meta=manifest(label, n, channels=2), events=ev)

    # Eyes closed and open in blocks. In the first, alpha grows with the eyes
    # closed, and two epochs carry an artifact past the rejection threshold.
    # In the second, alpha holds still while the theta in its reference band
    # falls with the eyes closed: a moving denominator.
    for label, reps, block, settle, alpha_closed, theta_closed in (
            ("berger-blocks", 4, 10.0, 2.0, 22.0, 8.0),
            ("berger-denominator", 6, 8.0, 2.0, 7.0, 1.0)):
        conds = [c for _ in range(reps) for c in ("closed", "open")]
        n = int((block * len(conds) + 2.0) * FS)
        uv = background(rng, n, 2, white=1.5, red=0.8)
        ev = []
        for i, c in enumerate(conds):
            t = 1.0 + block * i
            a, b = int(t * FS), int((t + block) * FS)
            m = b - a
            alpha = alpha_closed if c == "closed" else 7.0
            theta = theta_closed if c == "closed" else 8.0
            for ch in range(2):
                uv[ch, a:b] += tone(m, 10.0, alpha, ch) + tone(m, 6.0, theta, 0.5 + ch)
            ev.append(cue(t, c, rep=i // 2))
        if label == "berger-blocks":
            for t in (5.5, 37.0):
                k = int(t * FS)
                uv[:, k:k + 5] += 400.0
        write(label, uv=uv, meta=manifest(label, n, channels=2, block=block, settle=settle), events=ev)

    # Too short to filter, with events.
    n = 12
    uv = background(rng, n, 2)
    write("tiny-events", uv=uv, meta=manifest("tiny-events", n, channels=2),
          events=[cue(0.01, "left"), cue(0.02, "blink"), cue(0.03, "clench")])

    # No events file, no lead-off array, a version 1 manifest without the
    # converter's width, and bytes changed after the checksum was written.
    # Each is refused or listed, so a second of samples is enough.
    n = 300
    write("no-events", uv=background(rng, n, 2), meta=manifest("no-events", n, channels=2))
    write("no-leadoff", uv=background(rng, n, 2), meta=manifest("no-leadoff", n, channels=2),
          arrays_drop=("leadoff",))
    m = manifest("no-adc-bits", n, channels=2, version=1)
    del m["adc_bits"]
    write("no-adc-bits", uv=background(rng, n, 2), meta=m)
    write("bad-checksum", uv=background(rng, n, 2), meta=manifest("bad-checksum", n, channels=2))
    with open(CAPTURES / "bad-checksum.meta.json", "a") as f:
        f.write("\n")

    # What the collection list has to classify: one marked in the legacy
    # ledger, one never checksummed, an aborted run, a file not named like a
    # recording, and a manifest that is not an object.
    write("legacy-marked", uv=background(rng, n, 2), meta=manifest("legacy-marked", n, channels=2),
          checksum=False)
    sums = []
    for suffix in (".npz", ".meta.json"):
        p = CAPTURES / f"legacy-marked{suffix}"
        sums.append(f"{prov.sha256_file(p)}  {p.name}\n")
    (CAPTURES / "LEGACY.sha256").write_text("".join(sums))
    write("unverified", uv=background(rng, n, 2), meta=manifest("unverified", n, channels=2), checksum=False)
    write("aborted-run", uv=background(rng, n, 2),
          meta=manifest("aborted-run", n, channels=2, aborted="the link dropped"), checksum=False)
    (CAPTURES / "bad label.meta.json").write_text(json.dumps(manifest("bad label", n, channels=2)))
    (CAPTURES / "not-an-object.meta.json").write_text("[1, 2, 3]")

    # A game pad's inputs beside a recording: buttons and sticks the kernel
    # stamped, stored as arrays for their rate, with the wall clock stepped by
    # half a second at ten seconds, so only the aligned times land on the
    # samples. Among them a button and an axis the controller module does not
    # name, and three events of other kinds, the last Linux's report that it
    # threw away events before they were read, of type EV_SYN and code
    # SYN_DROPPED, which a recording keeps as a row where it came.
    n = int(20.0 * FS)
    clock = anchors(step_at=10.0, step_s=0.5)
    pressed = [(1, 0x130, 1), (1, 0x130, 0), (3, 0x00, -1200), (3, 0x01, 300), (1, 0x13b, 1), (3, 0x10, 1),
               (1, 0x2c0, 1), (3, 0x08, 17), (4, 0x04, 5), (0, 0x00, 0), (1, 0x13b, 0), (3, 0x00, 0),
               (0, 0x03, 0)]
    times = [1.0 + 1.5 * i for i in range(len(pressed))]
    savez(CAPTURES / "pad-inputs.inputs.npz", dict(
        input_mono=np.asarray([M0 + t for t in times], np.float64),
        input_host_time=np.asarray([T0 + t + (0.5 if t >= 10.0 else 0.0) for t in times], np.float64),
        input_type=np.asarray([t for t, _, _ in pressed], np.int16),
        input_code=np.asarray([c for _, c, _ in pressed], np.int32),
        input_value=np.asarray([v for _, _, v in pressed], np.int32)))
    write("pad-inputs", uv=background(rng, n, 2), meta=manifest("pad-inputs", n, channels=2, clock_anchors=clock),
          events=[cue(2.0, "level 1", step_at=10.0, step_s=0.5), cue(12.0, "level 2", step_at=10.0, step_s=0.5)])

    # A converter of sixteen bits whose counts reach its own full scale, on
    # one sample in fifty on one channel and one in twenty five on the
    # other.
    n = 1000
    c = rng.integers(-3000, 3000, size=(2, n)).astype(np.int64)
    c[0, ::50] = 32767
    c[1, 7::25] = -32767
    write("sixteen-bit", counts=c, meta=manifest("sixteen-bit", n, channels=2, adc_bits=16, vref_uv=2_500_000))

    # Too few samples for a spectrum, none and one, and the fewest that
    # make one, two.
    for label, n in (("no-samples", 0), ("one-sample", 1), ("two-samples", 2)):
        write(label, uv=background(rng, n, 2), meta=manifest(label, n, channels=2))

    # No stretch long enough for a segment and a second more: 1,000 samples
    # and 500 either side of a break, so the spectrum is the first
    # stretch's. Six samples with a break between every two, and stretches
    # of 12 and 10 samples, too short for the filters.
    n = 1500
    write("broken-rest", uv=background(rng, n, 2) + tone(n, 10.0, 12.0),
          index=np.concatenate([np.arange(1000), np.arange(1100, 1600)]), meta=manifest("broken-rest", n, channels=2))
    write("scattered", uv=background(rng, 6, 2), index=np.arange(0, 12, 2), meta=manifest("scattered", 6, channels=2))
    write("short-pieces", uv=background(rng, 22, 2), index=np.concatenate([np.arange(12), np.arange(20, 30)]),
          meta=manifest("short-pieces", 22, channels=2))

    # Jaw clench: twelve trials, as jaw-pass, with a break of 50 samples in
    # the clench window of the sixth, so that trial is left out.
    times = [2.0 + 3.0 * i for i in range(12)]
    n = int((3.0 * 12 + 3.0) * FS)
    uv = background(rng, n, 2, white=2.0, red=1.0)
    w = int(1.0 * FS)
    for t in times:
        a = int(t * FS)
        uv[:, a:a + w] += rng.normal(0, 35.0, (2, w))
    gap, at = 50, int((times[5] + 0.5) * FS)
    idx = np.concatenate([np.arange(at), np.arange(at, n) + gap])
    ev = [cue(t + (gap / FS if t * FS >= at else 0.0), "clench", trial=i) for i, t in enumerate(times)]
    write("jaw-broken", uv=uv, index=idx, meta=manifest("jaw-broken", n, channels=2), events=ev)

    # Eyes closed and open as berger-blocks, with a marker a person stamped
    # in the middle of a block, which starts no block, and cues that carry
    # no rep. Events of no condition closed or open: none that names a
    # condition, and one sham cue.
    conds = [c for _ in range(4) for c in ("closed", "open")]
    n = int((10.0 * len(conds) + 2.0) * FS)
    uv = background(rng, n, 2, white=1.5, red=0.8)
    ev = []
    for i, c in enumerate(conds):
        t = 1.0 + 10.0 * i
        a, b = int(t * FS), int((t + 10.0) * FS)
        for ch in range(2):
            uv[ch, a:b] += tone(b - a, 10.0, 22.0 if c == "closed" else 7.0, ch)
        ev.append(cue(t, c))
    ev.append(dict(cond="mark", kind="marker", host_time=T0 + 15.0, mono=M0 + 15.0))
    write("berger-marked", uv=uv, meta=manifest("berger-marked", n, channels=2, block=10.0, settle=2.0), events=ev)
    n = 2000
    write("unnamed-events", uv=background(rng, n, 2), meta=manifest("unnamed-events", n, channels=2),
          events=[dict(kind="cue", host_time=T0 + 1.0, mono=M0 + 1.0)])
    write("sham-only", uv=background(rng, n, 2), meta=manifest("sham-only", n, channels=2), events=[cue(1.0, "sham")])

    # Slow recordings, ten samples a second, whose first stretch, of 15
    # samples and then of 16, is kept for an epoch or a spectrum segment, and
    # is one sample short of what a 1 Hz high-pass needs and then enough.
    for label, first in (("slow", 15), ("slow-enough", 16)):
        idx = np.concatenate([np.arange(first), np.arange(100, 131)])
        n = len(idx)
        write(label, uv=background(rng, n, 1, fs=10.0), index=idx, fs=10.0,
              meta=manifest(label, n, channels=1, fs=10.0, rate_sps=10, block=10.0, settle=0.0),
              events=[cue(0.0, "closed"), cue(10.0, "open")])

    # A recording of 1 sample, whose rate measures 0 as that of every
    # recording of fewer than 2 samples does, and one of 3,000 samples whose
    # host clock did not advance, so no rate can be measured from it. Each
    # carries a cue of every condition an analysis reads.
    conds = ("closed", "open", "left", "right", "sham", "blink", "clench")
    write("one-sample-no-rate", uv=background(rng, 1, 2),
          meta=manifest("one-sample-no-rate", 1, channels=2, fs_effective=0.0), events=[cue(0.0, c) for c in conds])
    n = 3000
    write("frozen-clock", uv=background(rng, n, 2), host=np.full(n, T0),
          meta=manifest("frozen-clock", n, channels=2, fs_effective=0.0),
          events=[cue(1.0 + i, c) for i, c in enumerate(conds)])

    # Jaw clench, twelve trials as jaw-pass. In jaw-flat the first channel
    # is flat throughout, no power in its rest or its clench windows, and the
    # second is flat but for each clench window, no power in its rest
    # windows. In jaw-silent the first channel is flat throughout and the
    # second holds no clench.
    times = [2.0 + 3.0 * i for i in range(12)]
    n = int((3.0 * 12 + 3.0) * FS)
    w = int(1.0 * FS)
    uv = np.full((2, n), -3000.0)
    for t in times:
        a = int(np.searchsorted(T0 + np.arange(n) / FS, T0 + t))
        uv[1, a:a + w] += rng.normal(0, 35.0, w)
    ev = [cue(t, "clench", trial=i) for i, t in enumerate(times)]
    write("jaw-flat", uv=uv, meta=manifest("jaw-flat", n, channels=2), events=ev)
    uv = background(rng, n, 2, white=2.0, red=1.0)
    uv[0] = -3000.0
    write("jaw-silent", uv=uv, meta=manifest("jaw-silent", n, channels=2), events=ev)

    # Jaw clench at 100 samples a second, so a hi_hz at or above half the
    # rate is taken as 49 Hz: twelve trials, as jaw-pass.
    n = int((3.0 * 12 + 3.0) * 100.0)
    uv = background(rng, n, 2, white=2.0, red=1.0, fs=100.0)
    w = 100
    for t in times:
        a = int(t * 100.0)
        uv[:, a:a + w] += rng.normal(0, 35.0, (2, w))
    write("jaw-slow", uv=uv, fs=100.0, meta=manifest("jaw-slow", n, channels=2, fs=100.0, rate_sps=100), events=ev)

    # Saccades and blinks as saccade-pass and blink-pass, each with a break
    # of 50 samples 0.3 s after one trial's cue and another at a later
    # trial's cue, so those two trials are left out. Each cue is stamped at
    # the host time of the sample it falls on. Their own random source, so
    # every recording before them is what it was.
    srng = np.random.default_rng(907)
    gap = 50
    conds = ["sham"] * 4 + ["left", "right"] * 6
    times = [3.0 + 4.0 * i for i in range(4)] + [24.0 + 4.0 * i for i in range(12)]
    n = int((times[-1] + 4.0) * FS)
    uv = background(srng, n, 2, white=1.5, red=0.15)
    for t, c in zip(times, conds):
        if c != "sham":
            d = plateau(n, t + 0.15, 0.5, 90.0 * (1.0 if c == "left" else -1.0))
            uv[0] += d
            uv[1] -= d
    idx = np.arange(n)
    for at in (int(times[6] * FS) + int(0.3 * FS), int(times[9] * FS)):
        idx[at:] += gap
    ev = [cue(idx[int(t * FS)] / FS, c, trial=i) for i, (t, c) in enumerate(zip(times, conds))]
    write("saccade-broken", uv=uv, index=idx, meta=manifest("saccade-broken", n, channels=2), events=ev)
    conds = ["blink"] * 8 + ["sham"] * 4
    conds = [conds[i] for i in (0, 8, 1, 2, 9, 3, 4, 10, 5, 6, 11, 7)]
    times = [2.5 + 2.5 * i for i in range(len(conds))]
    n = int((2.5 * len(conds) + 3.0) * FS)
    uv = background(srng, n, 2, white=2.0, red=1.0)
    for i, (t, c) in enumerate(zip(times, conds)):
        if c == "blink":
            b = bump(n, t + 0.22 + 0.02 * ((i * 7) % 5 - 2), 0.25, 220.0)
            uv[0] += b
            uv[1] += 0.8 * b
    idx = np.arange(n)
    for at in (int(times[3] * FS) + int(0.3 * FS), int(times[7] * FS)):
        idx[at:] += gap
    ev = [cue(idx[int(t * FS)] / FS, c, trial=i) for i, (t, c) in enumerate(zip(times, conds))]
    write("blink-broken", uv=uv, index=idx, meta=manifest("blink-broken", n, channels=2), events=ev)

    # Stretches of 6,000, 2,200 and 1,200 samples with a break of 100 after
    # each of the first two, so the stretches a spectrum is computed from,
    # each of nperseg samples and a second more, are the first alone for an
    # nperseg of 2,048, all three for 512, the first two for 1,024, and for
    # 8,192, none being that long, the longest.
    lengths = (6000, 2200, 1200)
    idx = np.concatenate([np.arange(s, s + m) for s, m in
                          zip(np.cumsum((0,) + tuple(m + 100 for m in lengths))[:-1], lengths)])
    n = len(idx)
    write("three-stretches", uv=background(np.random.default_rng(6000), n, 2) + tone(n, 10.0, 12.0), index=idx,
          meta=manifest("three-stretches", n, channels=2))


def set_times() -> list[str]:
    """Give every manifest its own modification time, a minute apart in name
    order, so the collection lists in one order on every run, and return
    the labels newest first."""
    names = sorted(p.name for p in CAPTURES.glob("*.meta.json"))
    for i, name in enumerate(names):
        t = 1767262830 + 60 * i
        os.utime(CAPTURES / name, (t, t))
    return [nm[:-len(".meta.json")] for nm in reversed(names)]


# ---------------------------------------------------------------- the cases
EVENTS = ("saccade-pass", "saccade-fail", "blink-pass", "blink-fail", "jaw-pass", "jaw-zeros",
          "jaw-ties", "jaw-few", "berger-blocks", "berger-denominator")

CASES = [
    ("summary", ["rest-4ch"], {}),
    ("summary", ["rest-4ch"], {"line_hz": 50.0, "nperseg": 512, "slope_lo": 3.0, "slope_hi": 30}),
    ("summary", ["short-2ch"], {}),
    ("summary", ["saccade-pass"], {"nperseg": 1024}),
    ("summary", ["berger-blocks"], {}),
    ("summary", ["no-adc-bits"], {}),
    ("summary", ["sixteen-bit"], {"nperseg": 256}),
    ("summary", ["bad-checksum"], {}),
    ("psd", ["rest-4ch"], {}),
    ("psd", ["rest-4ch"], {"filtered": 1}),
    ("psd", ["rest-4ch"], {"filtered": 1, "fmax": 60, "nperseg": 1024, "hp": 0.5, "lp": 30.0,
                            "notch": 50.0, "notch_q": 20.0}),
    ("psd", ["rest-4ch"], {"filtered": 1, "hp": 0, "notch": 0}),
    ("psd", ["short-2ch"], {}),
    ("psd", ["short-2ch"], {"filtered": 1}),
    ("psd", ["rest-4ch", "short-2ch", "saccade-pass"], {}),
    ("psd", ["rest-4ch", "berger-blocks"], {"filtered": 1, "fmax": 45.0}),
    ("psd", ["rest-4ch"], {"filtered": 1, "lp": 200.0}),
    ("psd", ["rest-4ch"], {"filtered": 1, "notch": 130.0}),
    ("psd", ["rest-4ch"], {"filtered": 1, "hp": -1.0}),
    ("contact", ["rest-4ch"], {}),
    ("contact", ["rest-4ch"], {"dc_min_uv": 5000.0, "ac_min_uv": 40}),
    ("contact", ["short-2ch"], {}),
    ("contact", ["contact-checked-4ch"], {}),
    ("contact", ["no-leadoff"], {}),
    ("saccade", ["saccade-pass"], {}),
    ("saccade", ["saccade-fail"], {}),
    ("saccade", ["saccade-pass"], {"pre_s": 0.3, "post_s": 1.2, "hp": 0.2, "lp": 8.0, "notch": 0,
                                   "min_deflection_uv": 120.0, "max_corr": -0.9}),
    ("saccade", ["blink-pass"], {}),
    ("saccade", ["no-events"], {}),
    ("saccade", ["tiny-events"], {}),
    ("blink", ["blink-pass"], {}),
    ("blink", ["blink-fail"], {}),
    ("blink", ["blink-pass"], {"pre_s": 0.4, "post_s": 1.0, "min_pp_uv": 300.0, "min_detect_frac": 0.5,
                               "max_latency_sd_s": 0.01, "sham_max_pp_ratio": 3.0}),
    ("blink", ["saccade-pass"], {}),
    ("blink", ["no-events"], {}),
    ("jaw", ["jaw-pass"], {}),
    ("jaw", ["jaw-zeros"], {}),
    ("jaw", ["jaw-ties"], {}),
    ("jaw", ["jaw-few"], {}),
    ("jaw", ["jaw-pass"], {"window_s": 0.5, "lo_hz": 30.0, "hi_hz": 100.0, "min_ratio": 50.0,
                           "alpha": 0.01}),
    ("jaw", ["blink-pass"], {}),
    ("jaw", ["no-events"], {}),
    ("berger", ["berger-blocks"], {}),
    ("berger", ["berger-denominator"], {}),
    ("berger", ["berger-blocks"], {"allow_pseudoreplication": 1}),
    ("berger", ["berger-denominator"], {"epoch_s": 4.0, "allow_pseudoreplication": 1}),
    ("berger", ["berger-blocks"], {"epoch_s": 1.0, "ppmax": 80.0, "alpha_lo": 9.0, "alpha_hi": 12.0,
                                   "ref_lo": 3.0, "ref_hi": 25.0, "hp": 0.5, "lp": 40.0, "notch": 50.0,
                                   "notch_q": 25.0}),
    ("berger", ["no-events"], {}),
    ("berger", ["saccade-pass"], {}),
    # A spectrum needs two samples.
    ("summary", ["no-samples"], {}),
    ("summary", ["one-sample"], {}),
    ("summary", ["two-samples"], {}),
    ("psd", ["no-samples"], {}),
    ("psd", ["one-sample"], {"filtered": 1, "hp": 0, "lp": 0, "notch": 0}),
    ("psd", ["two-samples"], {}),
    ("psd", ["one-sample", "rest-4ch"], {}),
    ("psd", ["two-samples", "short-2ch"], {}),
    # No spectrum across a break.
    ("summary", ["broken-rest"], {}),
    ("psd", ["broken-rest"], {}),
    ("psd", ["broken-rest"], {"filtered": 1}),
    ("psd", ["scattered"], {}),
    ("psd", ["short-pieces"], {}),
    ("psd", ["short-pieces"], {"filtered": 1}),
    ("jaw", ["jaw-broken"], {}),
    # Settings from which no filter can be made, refused before filtering,
    # and the edges that are not refused.
    ("psd", ["rest-4ch"], {"filtered": 1, "hp": 124.975}),
    ("psd", ["rest-4ch"], {"filtered": 1, "hp": "1"}),
    ("psd", ["rest-4ch"], {"filtered": 1, "notch": -5.0}),
    ("psd", ["rest-4ch"], {"filtered": 1, "notch_q": 0}),
    ("psd", ["rest-4ch"], {"filtered": 1, "notch_q": -1.0}),
    ("psd", ["rest-4ch"], {"filtered": 1, "notch_q": "30"}),
    ("psd", ["rest-4ch"], {"filtered": 1, "notch_q": 0.4}),
    ("psd", ["rest-4ch"], {"filtered": 1, "notch": 60.0, "notch_q": 0.48}),
    ("psd", ["rest-4ch"], {"filtered": 1, "notch": 60.0, "notch_q": 0.481}),
    ("psd", ["rest-4ch"], {"filtered": 1, "hp": 1e-6}),
    ("psd", ["rest-4ch"], {"filtered": 1, "hp": 1e-7}),
    ("psd", ["rest-4ch"], {"filtered": 1, "lp": 1e-7}),
    ("psd", ["rest-4ch"], {"filtered": 1, "notch": 1e-7}),
    ("psd", ["rest-4ch"], {"filtered": 1, "notch": 124.975}),
    ("psd", ["rest-4ch"], {"filtered": 1, "notch": 0, "notch_q": 0}),
    ("psd", ["rest-4ch", "short-2ch"], {"filtered": 1, "lp": 130.0}),
    ("saccade", ["saccade-pass"], {"lp": 300.0}),
    ("blink", ["blink-pass"], {"notch_q": 0}),
    ("berger", ["berger-blocks"], {"hp": 1e-7}),
    ("saccade", ["tiny-events"], {"lp": 300.0}),
    # Too few samples for the filters, with one stage and with a notch alone.
    ("saccade", ["tiny-events"], {"notch": 0, "lp": 0}),
    ("blink", ["tiny-events"], {"hp": 0, "lp": 0}),
    ("blink", ["tiny-events"], {"hp": 0, "lp": 0, "notch": 50.0, "notch_q": 1.0}),
    # A stretch kept for an epoch or a segment, too short for the filters.
    ("berger", ["slow"], {"epoch_s": 0.5, "lp": 0, "notch": 0}),
    ("psd", ["slow"], {"filtered": 1, "nperseg": 4, "lp": 0, "notch": 0}),
    ("berger", ["slow-enough"], {"epoch_s": 0.5, "lp": 0, "notch": 0}),
    ("psd", ["slow-enough"], {"filtered": 1, "nperseg": 4, "lp": 0, "notch": 0}),
    # berger's blocks start only at events of the condition closed or open.
    ("berger", ["berger-marked"], {}),
    ("berger", ["unnamed-events"], {}),
    ("berger", ["sham-only"], {}),
    # A recording of 1 sample whose rate measures 0 is refused for its
    # samples before its rate or any setting is checked; one of more
    # samples whose rate cannot be measured is refused naming that.
    ("summary", ["one-sample-no-rate"], {}),
    ("psd", ["one-sample-no-rate"], {"filtered": 1}),
    ("psd", ["one-sample-no-rate"], {"filtered": 1, "lp": 300.0}),
    ("psd", ["one-sample-no-rate", "rest-4ch"], {"filtered": 1}),
    ("saccade", ["one-sample-no-rate"], {}),
    ("blink", ["one-sample-no-rate"], {"hp": 0, "lp": 0}),
    ("saccade", ["one-sample-no-rate"], {"hp": 0, "lp": 0, "notch": 0}),
    ("berger", ["one-sample-no-rate"], {}),
    ("jaw", ["one-sample-no-rate"], {}),
    ("summary", ["frozen-clock"], {}),
    ("psd", ["frozen-clock"], {}),
    ("psd", ["frozen-clock"], {"filtered": 1, "lp": 300.0}),
    ("saccade", ["frozen-clock"], {}),
    ("blink", ["frozen-clock"], {}),
    ("berger", ["frozen-clock"], {}),
    ("jaw", ["frozen-clock"], {}),
    # Window and duration settings an analysis cannot use, refused naming
    # them, and the least each refusal names, analyzed.
    ("saccade", ["saccade-pass"], {"pre_s": 0}),
    ("saccade", ["saccade-pass"], {"pre_s": 0.004}),
    ("saccade", ["saccade-pass"], {"pre_s": 0.00401}),
    ("saccade", ["saccade-pass"], {"post_s": 0}),
    ("saccade", ["saccade-pass"], {"pre_s": "0.5"}),
    ("blink", ["blink-pass"], {"pre_s": -1.0}),
    ("blink", ["blink-pass"], {"post_s": None}),
    ("blink", ["blink-pass"], {"pre_s": 0.00401, "post_s": 0.00401}),
    ("jaw", ["jaw-pass"], {"window_s": 0.128}),
    ("jaw", ["jaw-pass"], {"window_s": 0.129}),
    ("berger", ["berger-blocks"], {"epoch_s": 0}),
    ("berger", ["berger-blocks"], {"epoch_s": 0.00401}),
    ("berger", ["berger-blocks"], {"settle": -1.0}),
    ("berger", ["berger-blocks"], {"settle": 10.0}),
    ("berger", ["berger-blocks"], {"settle": 0}),
    # A time whose samples at the rate are past the largest number a 64-bit
    # float holds, refused naming it, and one whose samples a float holds,
    # analyzed with no trial or epoch that long.
    ("saccade", ["saccade-pass"], {"pre_s": 1e308}),
    ("saccade", ["saccade-pass"], {"pre_s": -1e308}),
    ("saccade", ["saccade-pass"], {"pre_s": 1e300}),
    ("blink", ["blink-pass"], {"post_s": 1e308}),
    ("jaw", ["jaw-pass"], {"window_s": 1e308}),
    ("jaw", ["jaw-pass"], {"window_s": 1e300}),
    ("berger", ["berger-blocks"], {"epoch_s": 1e308}),
    ("berger", ["berger-blocks"], {"epoch_s": 1e300}),
    ("summary", ["rest-4ch"], {"nperseg": 1}),
    ("summary", ["rest-4ch"], {"nperseg": 2}),
    ("psd", ["rest-4ch"], {"nperseg": 1.5}),
    ("psd", ["rest-4ch"], {"nperseg": "2048"}),
    ("psd", ["rest-4ch", "short-2ch"], {"filtered": 1, "nperseg": 0}),
    # Rest power of 0: a channel with clench power meets power_ratio, and
    # one with none in either window does not.
    ("jaw", ["jaw-flat"], {}),
    ("jaw", ["jaw-silent"], {}),
    # Every setting is held to the range the registry gives it. Those whose
    # range needs no recording, refused and at their bounds; a band's low
    # edge below its high edge; and berger's and jaw's bands at the
    # recording's rate.
    ("summary", ["rest-4ch"], {"line_hz": None}),
    ("summary", ["rest-4ch"], {"line_hz": 0}),
    ("summary", ["rest-4ch"], {"slope_lo": 0}),
    ("summary", ["rest-4ch"], {"slope_lo": 40.0, "slope_hi": 30.0}),
    ("psd", ["rest-4ch"], {"filtered": 2}),
    ("psd", ["rest-4ch"], {"filtered": "1"}),
    ("psd", ["rest-4ch"], {"fmax": 0}),
    ("saccade", ["saccade-pass"], {"max_corr": 2}),
    ("saccade", ["saccade-pass"], {"max_corr": -1}),
    ("saccade", ["saccade-pass"], {"min_deflection_uv": -1.0}),
    ("blink", ["blink-pass"], {"min_detect_frac": 1.5}),
    ("blink", ["blink-pass"], {"min_detect_frac": 1}),
    ("blink", ["blink-pass"], {"max_latency_sd_s": -0.1}),
    ("blink", ["blink-pass"], {"sham_max_pp_ratio": -1}),
    ("blink", ["blink-pass"], {"min_pp_uv": "100"}),
    ("jaw", ["jaw-pass"], {"alpha": 1.5}),
    ("jaw", ["jaw-pass"], {"alpha": 1}),
    ("jaw", ["jaw-pass"], {"min_ratio": -1}),
    ("jaw", ["jaw-pass"], {"lo_hz": 200.0}),
    ("jaw", ["jaw-pass"], {"lo_hz": 0}),
    ("jaw", ["jaw-slow"], {}),
    ("jaw", ["jaw-slow"], {"lo_hz": 49.0}),
    ("jaw", ["jaw-slow"], {"lo_hz": 48.0}),
    ("jaw", ["jaw-slow"], {"lo_hz": 45.0, "hi_hz": 48.0}),
    ("berger", ["berger-blocks"], {"ppmax": -1}),
    ("berger", ["berger-blocks"], {"alpha_lo": 13.0}),
    ("berger", ["berger-blocks"], {"ref_lo": 125.0, "ref_hi": 200.0}),
    # With no low-pass, so the reference band holds the recording's power
    # rather than what rounding leaves past a 45 Hz low-pass.
    ("berger", ["berger-blocks"], {"ref_lo": 120.0, "ref_hi": 200.0, "lp": 0}),
    ("berger", ["berger-blocks"], {"allow_pseudoreplication": 2}),
    ("psd", ["rest-4ch"], {"filtered": 1, "notch_q": None}),
    # The slow recordings' stretches, with bands below half their rate.
    ("berger", ["slow"], {"epoch_s": 0.5, "lp": 0, "notch": 0, "alpha_lo": 2.0, "alpha_hi": 4.0, "ref_lo": 1.0,
                          "ref_hi": 4.5}),
    ("berger", ["slow-enough"], {"epoch_s": 0.5, "lp": 0, "notch": 0, "alpha_lo": 2.0, "alpha_hi": 4.0,
                                 "ref_lo": 1.0, "ref_hi": 4.5}),
    # A trial whose samples, from pre_s before its cue to post_s after it,
    # hold a break is left out, a break at its cue among them.
    ("saccade", ["saccade-broken"], {}),
    ("blink", ["blink-broken"], {}),
    # A summary's n_runs and kept_frac count the stretches its spectrum was
    # computed from, for the nperseg given.
    ("summary", ["three-stretches"], {}),
    ("summary", ["three-stretches"], {"nperseg": 512}),
    ("summary", ["three-stretches"], {"nperseg": 1024}),
    ("summary", ["three-stretches"], {"nperseg": 8192}),
    # A recording of no samples has no offset or level to judge, and one of
    # one sample keeps the verdict its offset gives.
    ("contact", ["no-samples"], {}),
    ("contact", ["one-sample"], {}),
]


def numerics() -> dict:
    """scipy's own answers for the routines the analyses use, on seeded
    inputs chosen to reach each of their paths."""
    from scipy import signal, special, stats
    rng = np.random.default_rng(20261103)
    out = {}
    designs = [(0.1, "hp", 249.95), (1.0, "hp", 249.95), (45.0, "lp", 249.95), (10.0, "lp", 500.0),
               (0.5, "hp", 1000.0), (100.0, "lp", 1000.0), (200.0, "lp", 500.0), (30.0, "hp", 250.0)]
    out["butter"] = [dict(hz=hz, btype=bt, fs=fs, sos=signal.butter(4, hz, bt, fs=fs, output="sos").tolist())
                     for hz, bt, fs in designs]
    out["iirnotch"] = [dict(hz=hz, q=q, fs=fs, ba=[c.tolist() for c in signal.iirnotch(hz, q, fs)])
                       for hz, q, fs in ((60.0, 30.0, 249.95), (50.0, 20.0, 500.0), (60.0, 5.0, 1000.0))]
    x = rng.normal(0, 10, 700).cumsum() * 0.1 + rng.normal(0, 3, 700) + 50.0
    out["filtering"] = dict(
        x=x.tolist(),
        sosfiltfilt=[dict(hz=hz, btype=bt, fs=fs, y=signal.sosfiltfilt(signal.butter(4, hz, bt, fs=fs, output="sos"), x).tolist())
                     for hz, bt, fs in designs[:5]],
        filtfilt=[dict(hz=hz, q=q, fs=fs, y=signal.filtfilt(*signal.iirnotch(hz, q, fs), x).tolist())
                  for hz, q, fs in ((60.0, 30.0, 249.95), (50.0, 20.0, 500.0))])
    welch = []
    for n, nperseg, fs in ((1000, 256, 250.0), (997, 997, 249.95), (1200, 600, 500.0), (777, 210, 250.0),
                           (2310, 1155, 250.0), (513, 513, 100.0), (4096, 2048, 249.95), (60, 7, 10.0)):
        y = rng.normal(0, 5, n) + 20 * np.sin(2 * np.pi * 10 * np.arange(n) / fs)
        f, pxx = signal.welch(y, fs=fs, nperseg=nperseg)
        welch.append(dict(x=y.tolist(), fs=fs, nperseg=nperseg, f=f.tolist(), p=pxx.tolist()))
    out["welch"] = welch
    tests = []
    def wil(a, b, why):
        r = stats.wilcoxon(a, b, alternative="greater")
        tests.append(dict(test="wilcoxon", why=why, x=list(map(float, a)), y=list(map(float, b)),
                          statistic=float(r.statistic), p=float(r.pvalue)))
    for k in (3, 5, 9, 15, 30, 50):
        a = rng.normal(1.0, 1.0, k); wil(a, np.zeros(k), f"exact, {k} pairs")
    a = rng.normal(0.5, 1.0, 60); wil(a, np.zeros(60), "normal, past fifty pairs")
    a = np.round(rng.normal(1.0, 1.0, 10), 1); a[3] = a[5] = 0.7; wil(a, np.zeros(10), "every sign flip, ties")
    a = rng.normal(1.0, 1.0, 13); b = a.copy(); b[2:4] = rng.normal(0, 1, 2); b[7] = a[7]; wil(a, b, "every sign flip, a zero")
    a = np.round(rng.normal(1.0, 1.0, 25), 1); wil(a, np.zeros(25), "normal, ties past thirteen")
    a = rng.normal(0.0, 1.0, 8); wil(a, a, "every difference zero")
    for a, b, why in ((rng.normal(1, 1, 4), rng.normal(0, 1, 4), "four pairs"),
                      (rng.normal(1, 2, 12), rng.normal(0, 1, 12), "twelve pairs"),
                      (np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0]), "every difference the same"),
                      (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), "no difference")):
        r = stats.ttest_rel(a, b)
        tests.append(dict(test="ttest_rel", why=why, x=a.tolist(), y=b.tolist(), statistic=float(r.statistic),
                          p=float(r.pvalue)))
    for a, b, why in ((rng.normal(1, 1, 5), rng.normal(0, 1, 7), "exact, five and seven"),
                      (rng.normal(1, 1, 3), rng.normal(0, 1, 12), "exact, three and twelve"),
                      (rng.normal(0.5, 1, 8), rng.normal(0, 1, 9), "exact, eight and nine"),
                      (rng.normal(1, 1, 15), rng.normal(0, 1, 15), "normal, fifteen each"),
                      (np.round(rng.normal(1, 1, 6), 0), np.round(rng.normal(0, 1, 7), 0), "normal, ties"),
                      (rng.normal(0, 1, 40), rng.normal(0, 1, 31), "normal, forty and thirty one")):
        r = stats.mannwhitneyu(a, b, alternative="greater")
        tests.append(dict(test="mannwhitneyu", why=why, x=a.tolist(), y=b.tolist(), statistic=float(r.statistic),
                          p=float(r.pvalue)))
    out["tests"] = tests
    z = [float(v) for v in np.arange(-9.0, 9.01, 0.25)] + [-38.0, -20.0, 1e-8, -1e-8, 0.0, 0.70710678, 1.4142135]
    out["ndtr"] = [[v, float(special.ndtr(v))] for v in z]
    out["stdtr"] = [[df, t, float(special.stdtr(df, t))] for df in (1, 2, 3, 5, 10, 30, 101)
                    for t in (-40.0, -9.0, -4.0, -2.5, -1.0, -0.3, 0.0, 0.3, 1.0, 2.5, 6.0)]
    return out


def run(name: str, labels: list[str], params: dict) -> dict:
    """One case as the Command Center runs it: several recordings overlay
    their spectra, everything else analyzes the first."""
    case = dict(analysis=name, labels=labels, params=params)
    try:
        if name == "psd" and len(labels) > 1:
            out = A.compare(labels, **params)
        else:
            out = A.ANALYSES[name]["fn"](labels[0], **params)
        case["result"] = json.loads(json.dumps(out, default=float))
    except Exception as e:
        case["refused"] = dict(type=type(e).__name__, message=str(e))
    return case


def plain(v):
    return json.loads(json.dumps(v, default=float))


def events_and_inputs() -> dict:
    """The library's `events` and `inputs` for recordings with cues, with a
    game pad's inputs, and with neither, `inputs` filtered by every kind."""
    events = {label: plain(A.events(label))
              for label in ("saccade-pass", "blink-pass", "berger-blocks", "pad-inputs", "no-events", "rest-4ch")}
    inputs = []
    for label, kinds in (("pad-inputs", None), ("pad-inputs", ["button"]), ("pad-inputs", ["axis"]),
                         ("pad-inputs", ["button", "axis", "other"]), ("pad-inputs", ["other"]), ("pad-inputs", []),
                         ("no-events", None), ("rest-4ch", ["button"])):
        got = A.inputs(label) if kinds is None else A.inputs(label, kinds=kinds)
        inputs.append(dict(label=label, kinds=kinds, result=plain(got)))
    return dict(events=events, inputs=inputs)


def several() -> dict:
    """An analysis an application registers with `several`, run as an
    application runs it, as the registry says: once, handed every selected
    label in a list, where any other is run on each label separately."""
    handed = []

    def lengths(labels, scale=1.0):
        handed.append(list(labels))
        return dict(analysis="lengths", labels=list(labels),
                    samples=[int(scale * A.load(label).counts.shape[1]) for label in labels])

    entry = dict(fn=lengths, needs_events=False, several=True, doc="How long each selected recording is",
                 params=[dict(name="scale", type="number", default=1.0)])
    A.ANALYSES["lengths"] = entry
    try:
        cases = []
        for name, labels, params in (("lengths", ["rest-4ch", "short-2ch"], dict(scale=2.0)),
                                     ("lengths", ["jaw-few"], {}),
                                     ("contact", ["short-2ch", "rest-4ch"], {})):
            spec = A.ANALYSES[name]
            if spec.get("several"):
                results = [spec["fn"](list(labels), **params)]
            else:
                results = [spec["fn"](label, **params) for label in labels]
            cases.append(dict(analysis=name, labels=labels, params=params, results=plain(results)))
        takes_one = [k for k, v in A.ANALYSES.items() if not v.get("several")]
    finally:
        del A.ANALYSES["lengths"]
    return dict(registered=dict(name="lengths", doc=entry["doc"], needs_events=False, several=True,
                                params=entry["params"]),
                handed=handed, cases=cases, takes_one=takes_one)


def selects() -> dict:
    """A select held to its choices by the library's own check, which reads
    an entry from the registry: an entry registered here whose choices are
    text, and text and numbers, with what each setting is, as the library
    says it in a refusal, and the library's own selects of 0 and 1. Each
    case is the settings given, and null where the library takes them or
    the exception it refuses them with. A value is one of the choices by
    Python's equality: a number where a choice is that number, true where
    one is 1, and text only where a choice is the same text."""
    entry = dict(fn=lambda label, side="left", depth=1: (side, depth), needs_events=False, doc="Which side is read",
                 params=[dict(name="side", type="select", default="left", choices=["left", "right"]),
                         dict(name="depth", type="select", default=1, choices=[1, "auto", 2.5])])
    means = dict(side="the side that is read", depth="how deep the reading goes")
    given = [("sides", g) for g in (
        dict(side="left"), dict(side="right"), dict(depth=1), dict(depth=True), dict(depth=1.0), dict(depth="auto"),
        dict(depth=2.5), dict(side="right", depth="auto"),
        dict(side="both"), dict(side="Left"), dict(side=""), dict(side="it's"), dict(side=0), dict(side=1),
        dict(side=True), dict(side=None), dict(side=["left"]), dict(side={"left": 1}),
        dict(depth="1"), dict(depth="Auto"), dict(depth=2), dict(depth=False), dict(depth=None), dict(depth="2.5"),
        dict(side="both", depth="1"), dict(side="left", depth=3))]
    for name, analysis in (("filtered", "psd"), ("allow_pseudoreplication", "berger")):
        given += [(analysis, {name: v}) for v in (0, 1, False, True, 0.0, 1.0, "1", "yes", 2, 0.5, -1, None, [1])]
    A.ANALYSES["sides"] = entry
    A._MEANS.update(means)
    try:
        cases = []
        for analysis, g in given:
            case = dict(analysis=analysis, given=g, refused=None)
            try:
                A._check_settings("rest-4ch", analysis, g)
            except Exception as e:
                case["refused"] = dict(type=type(e).__name__, message=str(e))
            cases.append(case)
        named = sorted({name for _analysis, g in given for name in g})
        said = {name: A._MEANS[name] for name in named}
    finally:
        del A.ANALYSES["sides"]
        for name in means:
            del A._MEANS[name]
    return dict(label="rest-4ch", registered=dict(name="sides", doc=entry["doc"], needs_events=False,
                                                  params=entry["params"]),
                means=said, cases=cases)


def main() -> None:
    if CAPTURES.exists():
        shutil.rmtree(CAPTURES)
    CAPTURES.mkdir(parents=True)
    prov.use_captures_dir(CAPTURES)
    make_recordings()
    order = set_times()
    cases = [run(*c) for c in CASES]
    listed = plain(A.list_captures())
    out = dict(
        library=intomind.__version__, numpy=np.__version__, scipy=scipy.__version__,
        python=platform.python_version(), machine=platform.machine(),
        registry=[dict(name=k, doc=v["doc"], needs_events=v["needs_events"], params=v.get("params", []))
                  for k, v in A.ANALYSES.items()],
        cases=cases,
        list_captures=dict(labels=order, result=listed),
        several=several(),
        selects=selects(),
        **events_and_inputs(),
        numerics=numerics(),
    )
    (HERE / "analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    refused = sum("refused" in c for c in cases)
    print(f"{len(cases)} cases, {refused} refused, analyzed by intomind {out['library']}, "
          f"numpy {out['numpy']}, scipy {out['scipy']}")


if __name__ == "__main__":
    main()
