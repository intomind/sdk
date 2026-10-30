"""The recordings fixture: synthetic recordings, and what the Python library
exports from each, which every other implementation's readers and writers
are held to.

    PYTHONPATH=<a checkout of the Python library> python3 contract/recordings/make.py

The Python library is the reference for every file a recording becomes, so
this script asks it, and only it, what each export is. Every signal here is
generated from a seeded random source. None is a person's recording.

Writes, beside this script:

    captures/      the recordings, as every IntoMind program writes them
    expected.json  for each recording and format, the summary the Python
                   library returns without its path, and the file's sha256
                   for bdf, edf, csv and tsv, its arrays for npz, its
                   variables for mat, or the refusal
    numbers.json   floats as Python formats them, JSON as it reads and
                   writes it, and strings as its repr writes them, which
                   every byte above depends on

and, where each implementation compiles it in, the table of characters
Python's repr writes as escapes:

    crates/intomind-capture/src/unprintable.rs
    js/src/capture/unprintable.ts

Run it again and nothing changes unless the library does: the archives are
stamped with a fixed time and every random source is seeded.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import platform
import random
import shutil
import struct
import sys
import tempfile
import unicodedata
import warnings
import zipfile
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR

import numpy as np

warnings.simplefilter("ignore")
HERE = pathlib.Path(__file__).resolve().parent
CAPTURES = HERE / "captures"

import intomind  # noqa: E402
from intomind import export as E, provenance as prov  # noqa: E402

#: The IntoMind One's reference and converter, as its Device Info reports
#: them.
VREF_UV, ADC_BITS = 4_500_000, 24
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


def manifest(label: str, n: int, *, channels: int, fs: float = 249.95, gain: int = 24,
             version: int = 7, **extra) -> dict:
    """A manifest as the recorder writes one, with generated values."""
    m = dict(
        manifest_version=version, label=label,
        started_utc="2026-01-01T10:20:30Z", finished_utc="2026-01-01T10:20:33Z",
        captured_unix=1767262833.25,
        device_id="00000000c0de0001", firmware_version="1.4.5",
        firmware_sha256=None, firmware_artifact=None,
        tick_hz=1_000_000, vref_uv=VREF_UV, adc_bits=ADC_BITS, channels=channels,
        device_proto="1.4", board_revision="unknown", electrode_montage=None,
        montage=None, montage_error=None, montage_matches_device=None, montage_source=None,
        electrode_layout=None, placement=None, mains_hz=None, psu_volts=None, subject_id=None,
        regs_before=None, regs_after=None, regs_diff={}, config_after=None,
        device_config_changed_by_run=False, gain=gain, rate_sps=250, fs_effective=fs,
        protocol_id="fixture", protocol_params={},
        signal_chain=dict(adapter=dict(name="intomind.ble", version="1", first_party=True,
                                       transport="bluetooth-le"),
                          filtering="none", filters=None),
        signal_source="normal", synthetic=False,
        quality=dict(gaps=[], n_gaps=0, epoch_resets=0, samples_received=n,
                     samples_expected=n, samples_lost=0),
        gaps_announced=0,
        clock=dict(n_syncs=8, skew=1.0000123, offset=4501984.819461971, skew_pinned=False,
                   residual_ms=0.41, max_half_rtt_ms=15.588760375976562, span_s=2.5,
                   fitted_against="monotonic",
                   anchors=[[1767262830.0 + i * 0.3, 4503695.6 + i * 0.3] for i in range(8)],
                   frame_mono=4503695.6, wall_clock_steps=[], wall_clock_stepped=False),
        git_commit=None, git_dirty=None,
        host_versions=dict(python="3.13.2", platform="Linux", numpy="2.2.4", scipy="1.15.2", bleak=None),
        n=n, fw="1.4.5", experiment="fixture",
    )
    if version < 7:
        m["manifest_version"] = version
    m.update(extra)
    return m


def signal(rng, n: int, channels: int, *, dc=-4200.0, amp=30.0, fs=250.0) -> np.ndarray:
    """Channels by samples of counts: a 10 Hz rhythm, a slower drift and
    noise, at a DC offset, in microvolts turned into counts."""
    lsb = 2.0 * VREF_UV / (24 * (1 << ADC_BITS))
    t = np.arange(n) / fs
    rows = []
    for c in range(channels):
        uv = (dc + 500 * c + amp * np.sin(2 * np.pi * 10 * t + c)
              + 5 * np.sin(2 * np.pi * 0.7 * t) + rng.normal(0, 1.5, n))
        rows.append(np.round(uv / lsb))
    return np.array(rows, dtype=np.int64)


def write(label: str, *, counts, index=None, ticks="from-index", host=None, meta=None,
          leadoff=None, gap_before=None, leadoff_active=None, events=None,
          fortran=True, checksum=True, extra_arrays=None, fs=249.95) -> None:
    """One recording, the arrays laid out as the recorder lays them out."""
    nch, n = counts.shape
    index = np.arange(n, dtype=np.int64) if index is None else np.asarray(index, np.int64)
    if isinstance(ticks, str):
        ticks = (index * (1_000_000 / fs)).astype(np.int64)
    host = (1767262830.0 + index / fs) if host is None else np.asarray(host, np.float64)
    # The recorder builds the counts sample by sample and transposes them,
    # which numpy stores in Fortran order.
    c = np.ascontiguousarray(counts.T).T if fortran else np.ascontiguousarray(counts)
    arrays = dict(counts=c, index=index)
    if ticks is not None:
        arrays["device_ticks"] = np.asarray(ticks, np.int64)
    arrays["host_time"] = host
    arrays["leadoff"] = np.zeros(n, np.uint8) if leadoff is None else leadoff
    arrays["gap_before"] = np.zeros(n, bool) if gap_before is None else gap_before
    if leadoff_active is not None:
        arrays["leadoff_active"] = leadoff_active
    for k, v in (extra_arrays or {}).items():
        if v is None:
            arrays.pop(k, None)
        else:
            arrays[k] = v
    savez(CAPTURES / f"{label}.npz", arrays)
    (CAPTURES / f"{label}.meta.json").write_text(json.dumps(meta, indent=2))
    if events is not None:
        (CAPTURES / f"{label}.events.json").write_text(json.dumps(events, indent=2))
    if checksum:
        prov.write_checksums(label)


def cue_events(times, conds) -> list:
    return [dict(kind="cue", cond=c, host_time=1767262830.0 + t, mono=4503695.6 + t)
            for t, c in zip(times, conds)]


# ---------------------------------------------------------------- the cases
def make_recordings() -> None:
    rng = np.random.default_rng(20260101)
    n = 600

    # The common case: four channels, a measured rate, contact checked, an
    # electrode off on two channels for a while, cues.
    lab = "clean-v7"
    lo = np.zeros(n, np.uint8); lo[120:180] |= 0x01; lo[300:330] |= 0x04
    write(lab, counts=signal(rng, n, 4), meta=manifest(lab, n, channels=4, device_name="Ada's IntoMind One",
          processing=dict(origin="default", stages=[], description="notch 60 Hz, high-pass 0.5 Hz"),
          signal="notch 60 Hz, high-pass 0.5 Hz"),
          leadoff=lo, leadoff_active=np.ones(n, bool),
          events=cue_events([0.0, 0.8, 1.6, 2.39], ["rest", "eyes closed", "rest", "end"]))

    # A hole of 37 samples, and a last record to complete. A version 6
    # manifest, made before detection was kept.
    lab = "hole-and-tail"
    idx = np.concatenate([np.arange(300), np.arange(337, 337 + 223)])
    write(lab, counts=signal(rng, len(idx), 2), index=idx,
          meta=manifest(lab, len(idx), channels=2, version=6),
          gap_before=np.isin(np.arange(len(idx)), [300]))

    # A hole longer than a record, and one that ends on a record's edge.
    lab = "big-gap"
    idx = np.concatenate([np.arange(100), np.arange(100 + 600, 100 + 600 + 150), np.arange(1100, 1350)])
    write(lab, counts=signal(rng, len(idx), 2), index=idx, meta=manifest(lab, len(idx), channels=2, fs=250.0),
          fs=250.0)

    # Samples numbered from before the recording, then the count reset.
    lab = "restart-stub"
    idx = np.concatenate([np.arange(48_000, 48_010), np.arange(590)])
    write(lab, counts=signal(rng, len(idx), 2), index=idx, meta=manifest(lab, len(idx), channels=2),
          events=cue_events([0.5, 1.0], ["a", "b"]))

    # A count that steps back after the first second: a break of no known
    # length, which BDF and EDF refuse.
    lab = "step-back-later"
    idx = np.concatenate([np.arange(400), np.arange(390, 590)])
    write(lab, counts=signal(rng, len(idx), 2), index=idx, meta=manifest(lab, len(idx), channels=2))

    # Detection on for half the recording.
    lab = "leadoff-partly"
    lo = np.zeros(n, np.uint8); lo[50:90] = 0x03; lo[200:260] = 0x08; lo[400:450] = 0x02
    act = np.zeros(n, bool); act[:300] = True
    write(lab, counts=signal(rng, n, 4), meta=manifest(lab, n, channels=4), leadoff=lo, leadoff_active=act)

    # Detection off throughout: nothing was checked, whatever the byte says.
    lab = "leadoff-off"
    write(lab, counts=signal(rng, n, 4), meta=manifest(lab, n, channels=4),
          leadoff=np.full(n, 0x0F, np.uint8), leadoff_active=np.zeros(n, bool))

    # A signal the device generated.
    lab = "synthetic"
    write(lab, counts=signal(rng, n, 4, dc=0.0, amp=200.0),
          meta=manifest(lab, n, channels=4, synthetic=True, signal_source="synthetic"),
          leadoff_active=np.ones(n, bool))

    # Eight channels, every one with its own electrode off once, and ten
    # channels, past what the lead-off byte covers.
    for lab, ch in (("eight-channels", 8), ("ten-channels", 10)):
        lo = np.zeros(n, np.uint8)
        for c in range(min(ch, 8)):
            lo[40 * c + 10:40 * c + 30] |= 1 << c
        write(lab, counts=signal(rng, n, ch), meta=manifest(lab, n, channels=ch), leadoff=lo,
              leadoff_active=np.ones(n, bool))

    # One channel, and counts stored channel by channel.
    write("one-channel", counts=signal(rng, n, 1), meta=manifest("one-channel", n, channels=1))
    write("c-order", counts=signal(rng, n, 3), meta=manifest("c-order", n, channels=3), fortran=False)

    # A flat channel, a signal far below zero, and one spanning eight
    # millivolts: the ranges the header has to hold.
    flat = signal(rng, n, 2); flat[0, :] = flat[0, 0]
    write("flat", counts=flat, meta=manifest("flat", n, channels=2))
    write("far-below-zero", counts=signal(rng, n, 2, dc=-150_000.4, amp=0.3),
          meta=manifest("far-below-zero", n, channels=2))
    write("wide-range", counts=signal(rng, n, 2, dc=0.0, amp=4000.0), meta=manifest("wide-range", n, channels=2))
    # One microvolt a count, and a signal reaching 32125 either way: the EDF
    # range comes out at exactly -32767.5 to 32767.5, so every sample
    # digitizes to an exact half and rounds to the even neighbour.
    halves = np.array([np.round(32125 * np.sin(np.linspace(0, 7, n) + c)) for c in range(2)], dtype=np.int64)
    halves[:, 0], halves[:, 1] = (-32125, 32125), (32125, -32125)
    write("exact-halves", counts=halves,
          meta=manifest("exact-halves", n, channels=2, gain=1, vref_uv=1 << 23, adc_bits=24))
    # What a recorder armed and stopped before a packet came keeps: every
    # array empty, the counts with them, which reads as no channels.
    savez(CAPTURES / "no-samples.npz", dict(
        counts=np.array([], np.int64), index=np.array([], np.int64), device_ticks=np.array([], np.int64),
        host_time=np.array([], np.float64), leadoff=np.array([], np.uint8), gap_before=np.array([], bool),
        leadoff_active=np.array([], bool)))
    (CAPTURES / "no-samples.meta.json").write_text(json.dumps(manifest(
        "no-samples", 0, channels=4, fs=0.0, gain=None, rate_sps=None, signal_source=None, synthetic=None), indent=2))
    prov.write_checksums("no-samples")

    # Every packet delivered twice, as two subscriptions recorded it.
    base = signal(rng, n, 2)
    t = (np.arange(n) * 4000).astype(np.int64)
    blocks = [slice(i, i + 25) for i in range(0, n, 25)]
    dup_c = np.hstack([base[:, s] for s in blocks for _ in (0, 1)])
    dup_t = np.hstack([t[s] for s in blocks for _ in (0, 1)])
    dup_i = np.hstack([np.arange(n)[s] for s in blocks for _ in (0, 1)])
    write("duplicated", counts=dup_c, index=dup_i, ticks=dup_t, host=1767262830.0 + dup_t / 1e6,
          meta=manifest("duplicated", dup_c.shape[1], channels=2, fs=250.0))

    # Repeated device times with different counts: not duplicate delivery.
    c = signal(rng, n, 2)
    tt = (np.arange(n) * 4000).astype(np.int64); tt[100] = tt[99]
    write("dup-different", counts=c, ticks=tt, meta=manifest("dup-different", n, channels=2))

    # A clock fitted at an impossible skew, with and without the exchanges
    # needed to refit it.
    m = manifest("bad-skew", n, channels=2)
    m["clock"].update(skew=1.6986, offset=53373.097512857, span_s=0.62)
    tt = (np.arange(n) * 4000).astype(np.int64)
    write("bad-skew", counts=signal(rng, n, 2), ticks=tt, host=1767262830.0 + tt / 1e6 * 1.6986, meta=m)
    m = manifest("bad-skew-no-anchors", n, channels=2)
    m["clock"].update(skew=0.41, anchors=[[1767262830.0, 4503695.6]])
    write("bad-skew-no-anchors", counts=signal(rng, n, 2), meta=m)

    # A stored rate the host clock contradicts, as a reset once caused.
    write("poisoned-rate", counts=signal(rng, n, 2), meta=manifest("poisoned-rate", n, channels=2, fs=162.0))

    # No device clock, and a lead-off array that does not line up.
    write("no-ticks", counts=signal(rng, n, 2), ticks=None, meta=manifest("no-ticks", n, channels=2))
    write("misaligned-leadoff", counts=signal(rng, n, 2), meta=manifest("misaligned-leadoff", n, channels=2),
          leadoff=np.zeros(n - 7, np.uint8), leadoff_active=np.ones(n, bool))

    # Markers of every shape a file may hold.
    ev = cue_events([0.1, 0.2], ["start", 7]) + [
        dict(kind="cue", cond=None, host_time=1767262830.3),
        dict(kind="cue", cond={"step": "light", "hz": 15.0}, host_time=1767262830.4),
        dict(kind="button", host_time=1767262830.5),
        dict(cond="no time"),
        dict(cond="as text", host_time="1767262830.6"),
        dict(cond="before", host_time=1767262829.0),
        dict(cond="after", host_time=1767262840.0),
        dict(cond=True, host_time=1767262830.7),
    ] + [dict(cond=f"many {i}", host_time=1767262831.0 + i * 0.001) for i in range(40)]
    write("events-odd", counts=signal(rng, n, 2), meta=manifest("events-odd", n, channels=2), events=ev)

    # Text outside ASCII, a long processing description, and an electrode.
    write("non-ascii", counts=signal(rng, n, 2), meta=manifest(
        "non-ascii", n, channels=2, device_name="Zoë's IntoMind One 🧠",
        processing=dict(origin="host", stages=[], description="band-pass 1 to 45 Hz, then a notch at 50 Hz, "
                        "then a moving average over eight samples, as the café's desk set it"),
        montage=dict(electrode="gold cup, conductive paste", source="site")))

    # Integers where floats are usual, and floats where integers are.
    write("int-fields", counts=signal(rng, n, 2), meta=manifest("int-fields", n, channels=2, fs=250, gain=24.0,
                                                                  vref_uv=4500000.0))

    # A start the parser reads leniently, and one it cannot read.
    write("start-lenient", counts=signal(rng, n, 2), meta=manifest("start-lenient", n, channels=2,
                                                                     started_utc="2026-1-5T3:4:5z"))
    write("start-invalid", counts=signal(rng, n, 2), meta=manifest("start-invalid", n, channels=2,
                                                                     started_utc="2026-02-30T00:00:00Z"))

    # One sample.
    write("one-sample", counts=signal(rng, 1, 2), meta=manifest("one-sample", 1, channels=2))

    # A name with dots, dashes and underscores.
    write("a.label-with_marks", counts=signal(rng, n, 2), meta=manifest("a.label-with_marks", n, channels=2))

    # What is refused: a version 1 manifest without the converter's width, an
    # aborted run, a damaged file, and a rate that cannot be known.
    m = manifest("v1-no-width", n, channels=2, version=1)
    for k in ("adc_bits", "channels", "device_proto"):
        m.pop(k)
    write("v1-no-width", counts=signal(rng, n, 2), meta=m)
    write("aborted", counts=signal(rng, n, 2), meta=manifest("aborted", n, channels=2, aborted="link_lost"),
          checksum=False)
    write("damaged", counts=signal(rng, n, 2), meta=manifest("damaged", n, channels=2))
    p = CAPTURES / "damaged.meta.json"
    p.write_text(p.read_text().replace('"fixture"', '"edited"', 1))
    write("no-rate", counts=signal(rng, n, 2), host=np.full(n, 1767262830.0),
          meta=manifest("no-rate", n, channels=2, fs=0.0))

    # Recorded before checksums: marked in the legacy ledger, and not marked.
    write("legacy", counts=signal(rng, n, 2), meta=manifest("legacy", n, channels=2), checksum=False)
    lines = [f"{prov.sha256_file(CAPTURES / f'legacy{s}')}  legacy{s}" for s in (".npz", ".meta.json")]
    (CAPTURES / "LEGACY.sha256").write_text("\n".join(sorted(lines)) + "\n")
    write("unverified", counts=signal(rng, n, 2), meta=manifest("unverified", n, channels=2), checksum=False)


# ---------------------------------------------------------------- expected
def npz_digest(path: pathlib.Path) -> list:
    with np.load(path) as z:
        return [[k, z[k].dtype.str, list(z[k].shape), hashlib.sha256(z[k].tobytes()).hexdigest()] for k in z.files]


def mat_digest(path: pathlib.Path) -> dict:
    import h5py
    b = path.read_bytes()
    out = dict(header_tail=b[116:128].hex(), variables=[])
    with h5py.File(path, "r") as f:
        for k in f:
            d = f[k]
            attrs = [[a, d.attrs[a].dtype.str, repr(d.attrs[a].item() if hasattr(d.attrs[a], "item") else d.attrs[a])]
                     for a in sorted(d.attrs)]
            out["variables"].append([k, d.dtype.str, list(d.shape), hashlib.sha256(d[()].tobytes()).hexdigest(), attrs])
    return out


def make_expected() -> dict:
    out = dict(library=intomind.__version__, recordings={})
    work = pathlib.Path(tempfile.mkdtemp(prefix="intomind-fixture-"))
    try:
        labels = sorted(p.name[:-len(".meta.json")] for p in CAPTURES.glob("*.meta.json"))
        for label in labels:
            rec = {}
            for fmt in E.FORMATS:
                p = work / f"{label}.{fmt}"
                try:
                    s = E.export(label, fmt, p)
                except Exception as e:  # what is refused, and the words it is refused in
                    rec[fmt] = dict(refused=str(e))
                    continue
                row = dict(summary={k: v for k, v in s.items() if k != "path"})
                if fmt in ("bdf", "edf", "csv", "tsv"):
                    row["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
                elif fmt == "npz":
                    row["arrays"] = npz_digest(p)
                else:
                    row["mat"] = mat_digest(p)
                rec[fmt] = row
            out["recordings"][label] = rec
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out


# ---------------------------------------------------------------- numbers
def bits(x: float) -> str:
    return struct.pack("<d", x).hex()


def make_numbers() -> dict:
    rnd = random.Random(20260101)
    xs = [0.0, -0.0, 1e-5, 1e-4, 1e15, 1e16, 1e17, 5e-324, 2.2250738585072014e-308, 1.7976931348623157e308,
          15.5887603759765625, 0.1, 0.125, 0.375, 2.5, 0.0078125, -4231.1234567891, 498.8034567891234,
          99999999.5, 9.99999999, 1 / 3, 2 / 3, 123456789.0, 12345678.0, -150000.4, 4500000.0]
    for _ in range(1200):
        x = struct.unpack("<d", rnd.getrandbits(64).to_bytes(8, "little"))[0]
        if math.isfinite(x):
            xs.append(x)
    for _ in range(1200):
        xs.append(round(rnd.uniform(-1, 1) * 10 ** rnd.uniform(-12, 12), rnd.randint(0, 12)))
    for _ in range(800):
        xs.append(rnd.randint(-10 ** 6, 10 ** 6) / 2 ** rnd.randint(0, 40))
    for _ in range(500):
        xs.append(rnd.uniform(-10000, 10000) * (1 + rnd.uniform(-0.01, 0.01)))

    def quant(x, places, rounding):
        try:
            return f"{Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-places), rounding=rounding):f}"
        except InvalidOperation:
            return None
    small = [x for x in xs if abs(x) < 1e30]
    docs = [
        '{"b": 1, "a": [1.5, -0, 2e-05, "é😀", null, true], "n": NaN, "i": -Infinity, "b": 2}',
        '{"text": "tab\\tquote\\"slash\\\\/ \\u0001 \\u007f \\u00ff \\ud834\\udd1e", "e": 1E400, "z": -0.0, "big": 123456789012345678901234567890}',
        '[1, 2.0, 3e10, 4.5e-7, 1e16, 1e-5, 0.0001, 100000000000000000000.0]',
        '{"nested": {"z": [], "a": {}, "m": [{"y": 1, "x": 2}]}, "": "empty key"}',
    ]
    texts = ["", "plain", "it's", 'say "hi"', 'both \' and "', "back\\slash", "tab\tnew\nline\rcr",
             "\x00\x01\x1f\x7f", "\x80\x9f\xa0\xad\xff", "é ü ñ", "zero\u200bwidth", "\ufeffbom",
             "line\u2028para\u2029", "emoji \U0001f600", "\U000e0001 tag", "\U0010ffff end", "\u0378 unassigned"]
    for lo, hi in unprintable():
        for cp in {lo - 1, lo, hi, hi + 1}:
            if 0 <= cp <= sys.maxunicode and not 0xD800 <= cp <= 0xDFFF:
                texts.append(f"a{chr(cp)}b")
    return dict(
        note="Floats are given as the little-endian hex of their bits, so none is rounded on the way in.",
        strings=[[t, repr(t)] for t in texts],
        repr=[[bits(x), repr(x)] for x in xs],
        fixed=[[bits(x), p, f"{x:.{p}f}"] for x in xs if abs(x) < 1e22 for p in (rnd.randint(0, 12),)],
        general=[[bits(x), p, f"{x:.{p}g}"] for x in xs for p in (8, 6)],
        quantize=[[bits(x), p, quant(x, p, ROUND_FLOOR), quant(x, p, ROUND_CEILING)]
                  for x in small for p in (rnd.randint(0, 7),)],
        json=[[d, json.dumps(json.loads(d), sort_keys=True), json.dumps(json.loads(d), indent=2)] for d in docs],
    )


def unprintable() -> list[tuple[int, int]]:
    """Every run of characters `str.isprintable` refuses, which `repr`
    writes as escapes rather than as themselves."""
    runs, start = [], None
    for cp in range(sys.maxunicode + 2):
        bad = cp <= sys.maxunicode and not chr(cp).isprintable()
        if bad and start is None:
            start = cp
        elif not bad and start is not None:
            runs.append((start, cp - 1))
            start = None
    return runs


def write_tables() -> None:
    """The table each implementation compiles in, from this Python."""
    runs = unprintable()
    whence = (f"Python {platform.python_version()}, whose characters are those of Unicode "
              f"{unicodedata.unidata_version}")
    rows = lambda per: [", ".join(f"({a}, {b})" if per else f"[{a}, {b}]" for a, b in runs[i:i + 8])
                        for i in range(0, len(runs), 8)]
    root = HERE.parent.parent
    (root / "crates" / "intomind-capture" / "src" / "unprintable.rs").write_text(
        "//! The characters Python's `repr` writes as escapes rather than as\n"
        "//! themselves: what `str.isprintable` refuses, as runs of code points.\n"
        f"//! Written by contract/recordings/make.py from {whence}.\n\n"
        "/// Each run, first and last code point, in order.\n"
        "pub(crate) const UNPRINTABLE: &[(u32, u32)] = &[\n"
        + "".join(f"    {r},\n" for r in rows(True)) + "];\n")
    (root / "js" / "src" / "capture" / "unprintable.ts").write_text(
        "/**\n * The characters Python's `repr` writes as escapes rather than as\n"
        " * themselves: what `str.isprintable` refuses, as runs of code points.\n"
        f" * Written by contract/recordings/make.py from {whence}.\n */\n\n"
        "/** Each run, first and last code point, in order. */\n"
        "export const UNPRINTABLE: readonly (readonly [number, number])[] = [\n"
        + "".join(f"  {r},\n" for r in rows(False)) + "];\n")


def main() -> None:
    if CAPTURES.exists():
        shutil.rmtree(CAPTURES)
    CAPTURES.mkdir(parents=True)
    prov.use_captures_dir(CAPTURES)
    make_recordings()
    expected = make_expected()
    (HERE / "expected.json").write_text(json.dumps(expected, indent=1, sort_keys=True) + "\n")
    (HERE / "numbers.json").write_text(json.dumps(make_numbers(), indent=0) + "\n")
    write_tables()
    n = len(expected["recordings"])
    print(f"{n} recordings, exported by intomind {expected['library']}")


if __name__ == "__main__":
    main()
