# intomind-capture

What an IntoMind recording is on disk. `recording` reads the recordings
every IntoMind program writes, the Command Center and the Python library
included: the samples in `name.npz`, the manifest in `name.meta.json`, the
markers in `name.events.json`, and the checksums in `name.sha256`, written
last. `export` writes any such recording as npz, edf, bdf, csv, tsv or mat,
each the same file the Python library writes, with the same summary.

```rust
use intomind_capture::{export, recording::Collection};

let recordings = Collection::open(Collection::default_dir());
let summary = export::export(&recordings, "my-recording", "bdf", None)?;
```

docs.intomind.com/files.html states what a recording and every export
promise. The flat-file format this crate was first published with is kept
as it was: a directory holding the samples in one file of native-endian
integers, the events in another, and a JSON manifest.

## License

Free software under the GNU Affero General Public License, version 3, in
`LICENSE`. We also license it commercially, on fair terms shaped by your use
case: write to contact@intomind.com. See `LICENSING.md` in
[github.com/intomind/sdk](https://github.com/intomind/sdk).

IntoMind is a trademark of IntoMind, Inc. The license covers the code, not
the name.
