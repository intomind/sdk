# intomind-capture

What an IntoMind recording is on disk. `recording` reads the recordings
every IntoMind program writes, the Command Center and the Python library
included: the samples in `name.npz`, the manifest in `name.meta.json`, the
markers in `name.events.json`, and the checksums in `name.sha256`, written
last. `export` writes any such recording as npz, edf, bdf, csv, tsv or mat,
with the summary the Python library gives: bdf, edf, csv and tsv byte for
byte the files the library writes, and npz and mat with the same arrays.

```rust
use intomind_capture::{export, recording::Collection};

let recordings = Collection::open(Collection::default_dir());
let summary = export::export(&recordings, "my-recording", "bdf", None)?;
```

From firmware 1.4.6 an IntoMind device's lead-off bits are no verdict on
contact until its first judgment of a stream, 0.512 s in or later, and a
recording then holds `leadoff_judged`, false for each sample whose bits came
before it, which `Recording::leadoff_judged` gives. No export calls those
samples attached: csv and tsv leave their lead-off empty, edf and bdf label
the stretch `LEADOFF_NOT_JUDGED`, npz and mat carry `leadoff_judged`, and
each summary counts a sample as checked only where its bits are a verdict.

An IntoMind device whose firmware is older than 1.4.6 cannot check
electrode contact: its lead-off detection does not work, and every
electrode reads off whether it touches skin or not. A recording of one
that holds no `leadoff_judged` is read with every sample taken with
detection on as no verdict, from the firmware its manifest names, and
`Recording::repairs` says so. No export calls its electrodes off: each
summary's `contact` says "not checked", csv and tsv have no lead-off
column, and edf and bdf carry no lead-off label.

docs.intomind.com/files.html states what a recording and every export
promise. The crate also describes a flat-file format, format version 1:
a directory holding the samples in `samples.i32`, rows of little-endian
signed 32-bit integers, one for each channel, the events in `events.json`,
and the manifest in `capture.json`, as JSON.

## License

Free software under the GNU Affero General Public License, version 3, in
`LICENSE`. We also license it commercially, on fair terms shaped by your use
case: write to contact@intomind.com. See `LICENSING.md` in
[github.com/intomind/sdk](https://github.com/intomind/sdk).

IntoMind is a trademark of IntoMind, Inc. The license covers the code, not
the name.
