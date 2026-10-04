# intomind-protocol

The IntoMind BLE protocol as code: every packet, control exchange and
characteristic of the wire, encoded and decoded. `no_std`, with no
dependencies. The device's firmware and the host libraries build this same
code, so an encode on one side and a decode on the other cannot drift
apart.

The protocol itself is at
[docs.intomind.com/protocol.html](https://docs.intomind.com/protocol.html).

## License

Free software under the GNU Affero General Public License, version 3, in
`LICENSE`. We also license it commercially, on fair terms shaped by your use
case: write to contact@intomind.com. See `LICENSING.md` in
[github.com/intomind/sdk](https://github.com/intomind/sdk).

IntoMind is a trademark of IntoMind, Inc. The license covers the code, not
the name.
