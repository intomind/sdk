# intomind

The IntoMind instrument client: one device as a state machine. It does no
input and no output. It is handed the bytes that arrived and hands back the
bytes to send, so it works over whatever Bluetooth stack and runtime you
already have, and can be tested without either.

The guide is at [docs.intomind.com/rust.html](https://docs.intomind.com/rust.html).

## License

Free software under the GNU Affero General Public License, version 3, in
`LICENSE`. We also license it commercially, on fair terms shaped by your use
case: write to contact@intomind.com. See `LICENSING.md` in
[github.com/intomind/sdk](https://github.com/intomind/sdk).

IntoMind is a trademark of IntoMind, Inc. The license covers the code, not
the name.
