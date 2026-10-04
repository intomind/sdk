# intomind-capture

What an IntoMind recording is on disk: a directory holding the samples in
one flat file of native-endian integers, the events in another, and a JSON
manifest that says what they are and how they were made. Nothing is
compressed or packed, so a recording can be read in fifty years with a hex
editor and the manifest beside it. A gap is written down: the sample file
has no silent splices.

## License

Free software under the GNU Affero General Public License, version 3, in
`LICENSE`. We also license it commercially, on fair terms shaped by your use
case: write to contact@intomind.com. See `LICENSING.md` in
[github.com/intomind/sdk](https://github.com/intomind/sdk).

IntoMind is a trademark of IntoMind, Inc. The license covers the code, not
the name.
