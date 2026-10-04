# intomind-pipeline

The processing chain an IntoMind device runs on its signal, as a library.
The same code the device runs: the catalog of stage kinds, the rules a
chain must meet at a sample rate, the filter design, and the per-sample
run. A host uses it to check a chain before sending it, to explain a
refusal in words, to compose the device's default for a rate, and to
reproduce on a recording exactly what the device did to a stream.

`no_std`, one dependency for the floating-point library.

## License

Free software under the GNU Affero General Public License, version 3, in
`LICENSE`. We also license it commercially, on fair terms shaped by your use
case: write to contact@intomind.com. See `LICENSING.md` in
[github.com/intomind/sdk](https://github.com/intomind/sdk).

IntoMind is a trademark of IntoMind, Inc. The license covers the code, not
the name.
