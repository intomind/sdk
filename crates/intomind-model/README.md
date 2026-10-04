# intomind-model

The encoder that runs on an IntoMind device, as a library: the same 8-bit
weights and floating-point arithmetic the device uses, `no_std`, allocating
nothing. Given the weights file a device runs, a host produces the same
embedding the device does.

The encoder's weights are not published. A device sends its embeddings, and
heads are trained on those: see
[docs.intomind.com/model.html](https://docs.intomind.com/model.html).

## License

Free software under the GNU Affero General Public License, version 3, in
`LICENSE`. We also license it commercially, on fair terms shaped by your use
case: write to contact@intomind.com. See `LICENSING.md` in
[github.com/intomind/sdk](https://github.com/intomind/sdk).

IntoMind is a trademark of IntoMind, Inc. The license covers the code, not
the name.
