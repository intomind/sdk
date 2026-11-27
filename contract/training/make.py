"""The training fixture: the heads the Python library trains, and its words
when it refuses, which every other implementation is held to.

    PYTHONPATH=<a checkout of the Python library> python3 contract/training/make.py

`model.train_head` trains heads from seeded synthetic embeddings: the web
demos' 150 windows of 76 (seed 20261020), given as the device's integers
and as the encoder's floats, with one output and several, the ridge the
library chooses and one given, names from empty to past the sixteen bytes a
head holds, cut inside a character included, an encoder id written several
ways and none, and other shapes besides. For each it records the head's
bytes, the ridge, and the value every weight, bias and scale held before it
was rounded, so an implementation whose arithmetic differs from numpy's in
the last bits can show that a byte it rounds the other way sits on a
rounding boundary. `model.gap_ridge` and `train_head` are also asked what
they refuse, and record the words they refuse in.

Whether embeddings are integers or floats is numpy's type of the whole
input, so the cases include whole numbers given as floats, the same numbers
given as integers, and integers with one row of floats. The demos' case is
also trained from the library's own embedding windows: its rows sent as the
device sends them, two notifications a window, decoded and assembled by the
library, and trained from `EmbeddingWindow.embedding`, the lists of ints a
host collects. The notifications' SHA-256 is recorded, so another
implementation can show it decoded the same bytes.

The library's head writer is asked about what it does with values a head
cannot hold as given: weights and biases that are not whole numbers, or not
numbers, or do not fit, scales too large for four bytes, encoder ids written
several ways or empty, a version it does not write, and several of these at
once, which shows the order it checks them in. Its hex reader is asked about
characters where a digit has to be. What each writes, or its words refusing,
is recorded.

`model.blocked_folds` scores heads on contiguous blocks of windows they
did not train on: the demos' case at the default and other fold counts,
windows too few to hold any out, a block whose training windows lack a
state, and what it refuses. Its answer is counts, and a fraction of them.
It scores the encoder's floats as well: each fold's head is trained as
`train_head` decides the type of the windows it trains on, and each held-out
window is quantized when it holds a float, as `P.evaluate_head` decides it
for one window, so the cases include the demos' case as floats, integers
with one row of floats, and a held-out window that is not a number. It
scores more than two states as well, labeled 0 to k-1, each block by a head
with one output for each state, a window right when its own state's output
is the largest, as the device picks: three states and four, in blocks that
take the states in turn, each state leaning on values of its own, a block
whose training windows lack one of them, two states given as booleans and
as floats, which score exactly as whole numbers do, and the labels it
refuses, past the states, not whole, or below zero. A head with one output
for each of three states is trained from one-hot targets too, named by
every state joined by a slash.

Every signal is synthetic. The encoder ids are plainly made up.

Writes training.json beside this script. Run it again and nothing changes
unless the library, or the numerical libraries under it, do.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import pathlib
import platform
import warnings

import numpy as np

warnings.simplefilter("ignore")
HERE = pathlib.Path(__file__).resolve().parent
# Made without a sound: the library's experiments speak their cues and play
# their tones through the host's speakers, so every cue goes nowhere here,
# before anything of the library runs.
from intomind import audio as _audio  # noqa: E402
_audio.play = lambda tone, text="": None

import intomind  # noqa: E402
from intomind import model as M, protocol as P  # noqa: E402

#: An encoder id no encoder has, so nothing here repeats a real one.
MADE_UP_ENCODER = "0123456789abcdef"


def special(v):
    """A value as JSON can carry it: NaN and the infinities as words."""
    if isinstance(v, float) and not math.isfinite(v):
        return "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")
    if isinstance(v, list):
        return [special(x) for x in v]
    return v


def said(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"


def demo():
    """The web demos' training case: 150 windows of 76, centered on each
    window as the encoder's last normalization leaves them, and a label from
    a random direction plus noise."""
    rng = np.random.default_rng(20261020)
    n, d = 150, 76
    x = rng.normal(0.0, 0.6, (n, d))
    x -= x.mean(axis=1, keepdims=True)
    direction = rng.normal(size=d)
    score = x @ direction + rng.normal(0.0, 1.5, n)
    y = (score > np.median(score)).astype(float)
    return x, y, score


def wide():
    """240 windows of 96, made the same way from their own seed."""
    rng = np.random.default_rng(20261102)
    n, d = 240, 96
    x = rng.normal(0.0, 0.5, (n, d))
    x -= x.mean(axis=1, keepdims=True)
    direction = rng.normal(size=d)
    score = x @ direction + rng.normal(0.0, 1.0, n)
    return x, (score > np.median(score)).astype(float)


def loose():
    """40 windows of 6 integers with no near-dead direction, so the ridge
    comes from two real eigenvalues, and continuous targets."""
    rng = np.random.default_rng(20261103)
    x = rng.integers(-3000, 3001, (40, 6))
    y = x @ rng.normal(size=6) / 1000.0 + rng.normal(0.0, 0.5, 40)
    return x, y


def several_states(n, d, states, seed, lean=3000):
    """`n` integer windows of `d` values, in blocks of five that take the
    states in turn, each state leaning on eight values of its own."""
    rng = np.random.default_rng(seed)
    labels = [(i // 5) % states for i in range(n)]
    rows = [(rng.integers(-900, 900, d) + lean * ((np.arange(d) // 8) == y)).tolist() for y in labels]
    return rows, labels


def whole():
    """40 windows of 6 small whole numbers, which quantize without saturating
    when taken as floats, and continuous targets."""
    rng = np.random.default_rng(20261104)
    x = rng.integers(-7, 8, (40, 6))
    y = x @ rng.normal(size=6) + rng.normal(0.0, 0.5, 40)
    return x, y


def ints(x):
    return [[int(v) for v in row] for row in x]


def floats(x):
    return [[float(v) for v in row] for row in x]


def column(y):
    return [float(v) for v in y]


def internals(embeddings, targets, *, ridge=None, integer=True):
    """`train_head`'s own steps, kept apart so that what each weight, bias
    and scale held before rounding can be written down. The head they make
    is checked against `train_head`'s, so these are its steps."""
    e = np.asarray(embeddings, dtype=np.float64)
    if not integer:
        e = np.array([P.quantize_embedding(row) for row in e], dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if ridge is None:
        ridge = M.gap_ridge(e)
    design = np.hstack([e, np.ones((len(e), 1))])
    penalty = ridge * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y).T
    weights_f, intercepts = coefficients[:, :-1], coefficients[:, -1]
    scales, over, bias_over = [], [], []
    for o in range(weights_f.shape[0]):
        peak = np.abs(weights_f[o]).max()
        s = float(peak / 127.0) if peak > 0 else 1.0
        scales.append(s)
        over.append([float(v) for v in weights_f[o] / s])
        bias_over.append(float(intercepts[o] / s))
    return float(ridge), scales, over, bias_over


#: The inputs several cases share, written once and named by the cases.
INPUTS: dict[str, list] = {}


def value(v):
    """A case's input: a name in `INPUTS`, or the input itself."""
    return INPUTS[v] if isinstance(v, str) else v


def trained(label, embeddings, targets, *, integer=True, **options):
    """One head the library trains, and everything it held on the way."""
    named_e, named_y = embeddings, targets
    embeddings, targets = value(embeddings), value(targets)
    blob = M.train_head(embeddings, targets, **options)
    ridge, scales, over, bias_over = internals(embeddings, targets, ridge=options.get("ridge"),
                                               integer=integer)
    rows = [[int(v) for v in np.clip(np.rint(np.asarray(r)), -127, 127)] for r in over]
    bias = [int(round(b)) for b in bias_over]
    again = P.build_head(rows, bias, scales, name=options.get("name", ""),
                         encoder_id=options.get("encoder_id"))
    assert again == blob, f"{label}: the steps written down are not train_head's"
    return dict(case=label, embeddings=named_e, integer=integer, targets=named_y,
                options={k: options.get(k) for k in ("name", "ridge", "encoder_id")},
                ridge=ridge, head=blob.hex(), scale=scales, weights_over_scale=over,
                bias_over_scale=bias_over)


def refused(label, embeddings, targets, *, integer=True, **options):
    """One input the library refuses, and the words it refuses in."""
    try:
        M.train_head(embeddings, targets, **options)
    except Exception as e:  # noqa: BLE001 - every refusal, as the library raises it
        return dict(case=label, embeddings=special(embeddings), integer=integer,
                    targets=special(targets),
                    options={k: options.get(k) for k in ("name", "ridge", "encoder_id")},
                    refusal=said(e))
    raise AssertionError(f"{label}: the library trained a head")


def folds_case(label, rows, labels, **options):
    """One held-out score the library gives, or its words refusing."""
    named_r, named_y = special(rows), labels
    try:
        got = M.blocked_folds(value(rows), value(labels), **options)
    except Exception as e:  # noqa: BLE001 - every refusal, as the library raises it
        return dict(case=label, rows=named_r, labels=named_y, options=options, refusal=said(e))
    return dict(case=label, rows=named_r, labels=named_y, options=options, score=got)


def ridge_case(label, quantized):
    named, quantized = quantized, value(quantized)
    try:
        return dict(case=label, quantized=special(named), ridge=special(M.gap_ridge(quantized)))
    except Exception as e:  # noqa: BLE001
        return dict(case=label, quantized=special(named), refusal=said(e))


#: The embedding notification's header, as the protocol lays it out.
HEADER = struct.Struct("<BBBBIQH8sBB")


def notifications(rows, *, encoder_id: str, split: int, index_step: int, time_step: int,
                  window_samples: int) -> list[bytes]:
    """Each row as the device sends a window embedding: `split` values a
    notification, every part but the last flagged that more follow."""
    out = []
    for w, row in enumerate(rows):
        for first in range(0, len(row), split):
            part = row[first:first + split]
            flags = P.EMBEDDING_FLAGS["more_parts"] if first + split < len(row) else 0
            head = HEADER.pack(P.PACKET_EMBEDDING, flags, len(row), first, index_step * w, time_step * w,
                               window_samples, bytes.fromhex(encoder_id), 0, P.WINDOW_TOKEN)
            out.append(head + struct.pack(f"<{len(part)}h", *part))
    return out


def from_windows(label, rows, targets, *, name: str, channels: int, tokens_per_channel: int, **sent):
    """A head trained from the library's own embedding windows, assembled
    from the notifications that carry `rows`."""
    notes = notifications(value(rows), **sent)
    assembler = P.EmbeddingAssembler(channels, tokens_per_channel, form="window")
    windows = [w for w in (assembler.feed(P.decode_embedding(b)) for b in notes) if w is not None]
    embeddings = [w.embedding for w in windows]
    assert embeddings == value(rows), f"{label}: the windows are not the rows sent"
    assert all(type(v) is int for e in embeddings for v in e), f"{label}: a window holds a value that is not an int"
    case = trained(label, rows, targets, name=name, encoder_id=windows[0].encoder_id)
    assert M.train_head(embeddings, value(targets), name=name, encoder_id=windows[0].encoder_id).hex() == case["head"]
    case.update(channels=channels, tokens_per_channel=tokens_per_channel, notifications=len(notes),
                notifications_sha256=hashlib.sha256(b"".join(notes)).hexdigest(), sent=sent)
    return case


def head(label, weights, bias, scale, **options):
    """What the library's `build_head` writes from these values, or its
    words refusing them."""
    case = dict(case=label, weights=special(weights), bias=special(bias), scale=special(scale),
                options={k: options.get(k) for k in ("name", "in_dim", "encoder_id", "version")})
    try:
        case["head"] = P.build_head(weights, bias, scale, **options).hex()
    except Exception as e:  # noqa: BLE001 - every refusal, as the library raises it
        case["refusal"] = said(e)
    return case


def hex_read(text: str):
    """`bytes.fromhex`, which the library reads every hex string with."""
    try:
        return dict(text=text, bytes=bytes.fromhex(text).hex())
    except Exception as e:  # noqa: BLE001
        return dict(text=text, refusal=said(e))


def written(v, level=0) -> str:
    """JSON as `json.dumps` writes it, indented, with a list of plain
    values on one line, so a window is one line rather than seventy-six."""
    pad = " " * level
    if isinstance(v, dict):
        if not v:
            return "{}"
        inner = ",\n".join(f"{pad} {json.dumps(k)}: {written(x, level + 1)}" for k, x in v.items())
        return "{\n" + inner + "\n" + pad + "}"
    if isinstance(v, list) and any(isinstance(x, (dict, list)) for x in v):
        inner = ",\n".join(f"{pad} {written(x, level + 1)}" for x in v)
        return "[\n" + inner + "\n" + pad + "]"
    return json.dumps(v)


def main() -> int:
    x, y, score = demo()
    q = ints(P.quantize_embedding(row) for row in x)
    xw, yw = wide()
    qw = ints(P.quantize_embedding(row) for row in xw)
    xl, yl = loose()
    xn, yn = whole()
    small = [[1, 2], [3, 4], [5, 7], [2, 9]]
    rows3, labels3 = several_states(160, 30, 3, 6)
    rows4, labels4 = several_states(200, 24, 4, 9)
    sixty, _sixty_labels = several_states(60, 24, 3, 7)
    INPUTS.update({
        "three states": rows3,
        "three state labels": labels3,
        "three state targets": [[1.0 if y == s else 0.0 for s in range(3)] for y in labels3],
        "four states": rows4,
        "four state labels": labels4,
        "three states, sixty windows": sixty,
        "demo": q,
        "demo, unquantized": floats(x),
        "demo labels": column(y),
        "demo, three outputs": [[float(a), float(1.0 - a), float(s / 10.0)] for a, s in zip(y, score)],
        "demo score": column(score / 10.0),
        "wide": qw,
        "wide labels": column(yw),
        "loose": ints(xl),
        "loose targets": column(yl),
        "whole numbers": ints(xn),
        "whole numbers, as floats": floats(xn),
        "whole number targets": column(yn),
        "whole numbers with one row of floats": ints(xn[:-1]) + [[v + 0.25 for v in floats(xn[-1:])[0]]],
    })

    heads = [
        trained("the demos' case, as the device's integers", "demo", "demo labels",
                name="fixture", encoder_id=MADE_UP_ENCODER),
        trained("the demos' case, as the encoder's floats", "demo, unquantized", "demo labels", integer=False,
                name="fixture", encoder_id=MADE_UP_ENCODER),
        trained("three outputs, one of them continuous", "demo", "demo, three outputs",
                name="three", encoder_id=MADE_UP_ENCODER),
        trained("a ridge given", "demo", "demo labels", ridge=25.0, name="ridge 25"),
        trained("no name and no encoder", "demo", "demo labels"),
        trained("a name of exactly sixteen bytes", "demo", "demo labels", name="sixteen byte nam"),
        trained("a name longer than a head holds", "demo", "demo labels",
                name="a name much longer than sixteen bytes"),
        trained("a name cut inside a two byte character", "demo", "demo labels", name="abcdefghijklmnoé"),
        trained("a name cut inside a three byte character", "demo", "demo labels",
                name="日本語の名前です"),
        trained("an encoder id with spaces and capitals", "demo", "demo labels",
                encoder_id="01 23 45 67 89 AB CD EF"),
        trained("an encoder id around tabs and a newline", "demo", "demo labels",
                encoder_id="\t0123456789abcdef\n"),
        trained("an empty encoder id", "demo", "demo labels", encoder_id=""),
        trained("continuous targets", "demo", "demo score"),
        trained("240 windows of 96", "wide", "wide labels", name="wide"),
        trained("integers with no near-dead direction", "loose", "loose targets"),
        trained("one weight, with a ridge given", [[1], [2], [4]], [1.0, 2.0, 3.0], ridge=2.0),
        trained("small whole numbers, as integers", "whole numbers", "whole number targets"),
        trained("small whole numbers, as floats", "whole numbers, as floats", "whole number targets",
                integer=False),
        trained("integers with one row of floats", "whole numbers with one row of floats",
                "whole number targets", integer=False),
        trained("three states, one output for each, named by every state", "three states", "three state targets",
                name="Melody/Alphabet", encoder_id=MADE_UP_ENCODER),
    ]

    windows = from_windows("the demos' case, from the library's own embedding windows", "demo", "demo labels",
                           name="windows", channels=4, tokens_per_channel=8, encoder_id=MADE_UP_ENCODER,
                           split=38, index_step=1000, time_step=4000, window_samples=2000)

    one = [[1, -2, 3]]
    biggest = struct.unpack("<f", b"\xff\xff\x7f\x7f")[0]        # the largest float four bytes hold
    halfway = biggest + 2.0**103                                   # rounds to infinity in four bytes
    heads_written = [
        *(head(f"a bias of {b!r}", one, [b], [0.5], name="bias")
          for b in (2**31 - 1, 2**31, -2**31, -2**31 - 1, 2147483647.9, -2147483648.9, 2147483648.0, 1e12,
                    1.5, -1.5, float("inf"), float("-inf"), float("nan"))),
        head("weights that are not whole numbers", [[1.5, -1.5, 0.5]], [0], [1.0]),
        head("weights just inside the range", [[127.9, -128.9, 0]], [0], [1.0]),
        head("weights cut toward zero", [[-0.5, 0.25, -127.5]], [0], [1.0]),
        head("a weight of 128", [[128.0, 0, 0]], [0], [1.0]),
        head("a weight of -129", [[-129.0, 0, 0]], [0], [1.0]),
        head("a weight that is not a number", [[float("nan"), 0, 0]], [0], [1.0]),
        head("an infinite weight", [[float("inf"), 0, 0]], [0], [1.0]),
        head("a weight out of range before one that is not a number", [[200, float("nan"), 0]], [0], [1.0]),
        head("a weight that is not a number before one out of range", [[float("nan"), 200, 0]], [0], [1.0]),
        head("an infinite weight in the second row", [[1, 2, 3], [float("inf"), 0, 0]], [0, 0], [1.0, 1.0]),
        *(head(f"a scale of {s!r}", one, [0], [s])
          for s in (biggest, math.nextafter(halfway, 0.0), halfway, 1e39, -1e39, float("inf"), float("-inf"),
                    float("nan"), -0.0, 1e-46, 1.5)),
        head("an empty encoder id", one, [0], [1.0], encoder_id=""),
        head("no encoder id", one, [0], [1.0]),
        head("an empty encoder id, format 1", one, [0], [1.0], encoder_id="", version=1),
        head("an encoder id of zeros with spaces, format 1", one, [0], [1.0],
             encoder_id="00 00 00 00 00 00 00 00", version=1),
        head("an encoder id, format 1", one, [0], [1.0], encoder_id=MADE_UP_ENCODER, version=1),
        head("an encoder id around spaces", one, [0], [1.0], encoder_id=" 01 23 45 67 89 ab cd ef "),
        head("an encoder id too short", one, [0], [1.0], encoder_id="abcd"),
        head("format 1", one, [0], [1.0], version=1),
        head("format 3", one, [0], [1.0], version=3),
        head("a weight out of range and an encoder id that is not hex", [[200, 0, 0]], [0], [1.0],
             encoder_id="zz"),
        head("an encoder id too short and format 3", one, [0], [1.0], encoder_id="abcd", version=3),
        head("no outputs and format 3", [], [], [], version=3),
        head("a bias and a scale that do not fit", one, [2**31], [1e39]),
        head("rows of different widths and an encoder id that is not hex", [[1, 0, 0], [1, 0]], [0, 0],
             [1.0, 1.0], encoder_id="zz"),
        head("format 3 and a bias that does not fit", one, [2**31], [1.0], version=3),
        head("two biases for one output and a weight out of range", [[200, 0, 0]], [0, 0], [1.0]),
        head("a width given that the rows do not have", one, [0], [1.0], in_dim=4),
    ]
    hexes = [hex_read(t) for t in ("1g", "g1", "abc", "zz", "0 0", " 01 ", "01\t23\n", "a", "a ", "", "  ",
                                   "0123456789ABCDEF", "00\x0b11\x0c22\r33", "0\u00a011", "0011223344556\u00e977")]

    refusals = [
        refused("rows of different lengths", [[1, 2], [3], [4, 5]], [1, 2, 3]),
        refused("a row holding a list", [[1, 2], [3, [4]], [4, 5]], [1, 2, 3], integer=False),
        refused("a row and a number", [[1, 2], 3, [4, 5]], [1, 2, 3], integer=False),
        refused("one row, not a list of rows", [1, 2, 3], [1, 2, 3], integer=False),
        refused("no windows", [], [], integer=False),
        refused("rows of rows", [[[1, 2]], [[3, 4]], [[5, 6]]], [1, 2, 3], integer=False),
        refused("more targets than windows", small[:3], [1, 2]),
        refused("no more windows than weights", small[:2], [1, 2]),
        refused("windows of no values", [[], [], []], [1, 2, 3], integer=False),
        refused("windows of no values, a ridge given", [[], [], []], [1, 2, 3], integer=False,
                ridge=1.0),
        refused("one value a window, the ridge chosen", [[1], [2], [4]], [1, 2, 3]),
        refused("a float embedding that is not a number", [[0.1, float("nan")], [0.2, 0.3], [0.4, 0.5]],
                [1, 2, 3], integer=False),
        refused("a target that is not a number", small, [1.0, float("nan"), 3.0, 4.0]),
        refused("an infinite target", small, [1.0, float("inf"), 3.0, 4.0]),
        refused("embeddings that are all zero", [[0, 0], [0, 0], [0, 0]], [1, 2, 3]),
        refused("targets with no outputs", small, [[], [], [], []]),
        refused("target rows of different lengths", small, [[1], [2, 3], [4], [5]]),
        refused("one number for all the targets", small, 5.0),
        refused("targets one level too deep", small, [[[1]], [[2]], [[3]], [[4]]]),
        refused("targets one level too deep, as wide as the windows", small,
                np.ones((4, 4, 1)).tolist()),
        refused("an encoder id too short", small, [1, 2, 3, 4], encoder_id="abcd"),
        refused("an encoder id of odd length", small, [1, 2, 3, 4], encoder_id="abc"),
        refused("an encoder id that is not hex", small, [1, 2, 3, 4], encoder_id="zz112233445566gg"),
        refused("an encoder id with a space inside a byte", small, [1, 2, 3, 4],
                encoder_id="0 0112233445566 77"),
        refused("an encoder id with a letter outside ASCII", small, [1, 2, 3, 4],
                encoder_id="0011223344556é77"),
        refused("a bias too large for its four bytes", small, [1e12, 1e12, 1e12, 1e12 + 1]),
        refused("a scale too large for a float", small, [1e300, 2e300, 3.5e300, 4e300]),
    ]

    labels = [int(v) for v in y]
    INPUTS["demo labels, as a list"] = labels
    INPUTS["demo labels, twenty of one state first"] = [1] * 20 + [0] * 130
    INPUTS["whole number labels"] = [int(v > np.median(yn)) for v in yn]
    folds = [
        folds_case("the demos' case", "demo", "demo labels, as a list", encoder_id=MADE_UP_ENCODER),
        folds_case("the demos' case, three folds", "demo", "demo labels, as a list", folds=3),
        folds_case("the demos' case, more folds than twenty", "demo", "demo labels, as a list", folds=25),
        folds_case("the demos' case, one fold", "demo", "demo labels, as a list", folds=1),
        folds_case("a block whose training windows lack a state", "demo", "demo labels, twenty of one state first"),
        folds_case("too few windows to hold any out", [r[:8] for r in q[:9]], labels[:9]),
        folds_case("the folds grow until each trains on more windows than values", [r[:8] for r in q[:10]],
                   [0, 1] * 5),
        folds_case("more labels than windows", small, [0, 1, 0, 1, 0]),
        folds_case("no windows", [], []),
        folds_case("a label that is neither state", small, [0, 1, 2, 1]),
        folds_case("no folds", small, [0, 1, 0, 1], folds=0),
        folds_case("the demos' case, as the encoder's floats", "demo, unquantized", "demo labels, as a list",
                   encoder_id=MADE_UP_ENCODER),
        folds_case("integers with one row of floats", "whole numbers with one row of floats", "whole number labels"),
        folds_case("a held-out window that is not a number", [[float("nan"), 0.5]] + small + [[4, 1], [6, 3], [1, 8]],
                   [0, 1, 0, 1, 0, 1, 0, 1]),
        folds_case("three states, scored by the largest of one output each", "three states", "three state labels",
                   encoder_id=MADE_UP_ENCODER),
        folds_case("three states, three folds", "three states", "three state labels", folds=3),
        folds_case("four states", "four states", "four state labels"),
        folds_case("a block whose training windows lack one of three states", "three states, sixty windows",
                   [2] * 12 + [0, 1] * 24),
        folds_case("three states, the labels given as floats", "three states", [float(y) for y in labels3]),
        folds_case("two states, the labels given as booleans", "demo", [bool(v) for v in labels]),
        folds_case("two states, the labels given as floats", "demo", [float(v) for v in labels]),
        folds_case("a label past the states", "three states, sixty windows", [0, 1, 3] * 20),
        folds_case("a label that is not a whole number", "three states, sixty windows", [0, 1, 2] * 19 + [0.5, 1, 2]),
        folds_case("a label below zero", "three states, sixty windows", [0, 1, 2] * 19 + [-1, 1, 2]),
        folds_case("two states, the second labeled 2", small, [0, 2, 0, 2]),
    ]

    ridges = [
        ridge_case("the demos' case", "demo"),
        ridge_case("the demos' case, unquantized", "demo, unquantized"),
        ridge_case("integers with no near-dead direction", "loose"),
        ridge_case("a value that is not a number", [[1.0, float("nan")], [2.0, 3.0], [4.0, 1.0]]),
        ridge_case("an infinite value", [[1.0, float("inf")], [2.0, 3.0], [4.0, 1.0]]),
        ridge_case("one row", [1, 2, 3]),
        ridge_case("nothing", []),
        ridge_case("rows of different lengths", [[1, 2], [3]]),
        ridge_case("a row and a number", [[1, 2], 3]),
        ridge_case("rows of no values", [[], []]),
        ridge_case("rows of one value", [[1], [2]]),
        ridge_case("a number", 5.0),
        ridge_case("rows of rows", [[[1, 2]], [[3, 4]], [[5, 6]]]),
    ]

    out = dict(
        library=intomind.__version__,
        numpy=np.__version__,
        python=platform.python_version(),
        inputs=INPUTS,
        heads=heads,
        windows=windows,
        refusals=refusals,
        gap_ridge=ridges,
        build_head=heads_written,
        fromhex=hexes,
        blocked_folds=folds,
    )
    (HERE / "training.json").write_text(written(out) + "\n")
    print(f"{len(heads)} heads and one from windows, {len(refusals)} refusals, {len(ridges)} ridge cases, "
          f"{len(heads_written)} heads written, {len(hexes)} hex strings, {len(folds)} held-out scores")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
