# @intomind/sdk

The IntoMind instrument client for JavaScript. One package, every IntoMind
device, in a browser and in Node.

```ts
import { requestAndConnect } from "@intomind/sdk";

const device = await requestAndConnect({
  onEvent(event) {
    if (event.type === "samples") plot(event.batch);
    if (event.type === "gap") mark(event.gap);
  },
});
await device.send(device.session.startStream());
```

It owns the protocol spoken to a device, the session that turns
notifications into events, the clock that puts those events on the host's
timeline, and one transport for the browser. It knows nothing about any
particular device beyond what that device tells it, and nothing about any
application built on top of it.

## What is in here

| | |
|---|---|
| `src/protocol.ts` | the wire, and nothing else |
| `src/session.ts` | one device as a state machine: bytes in, events out |
| `src/timebase.ts` | device time onto host time, and what it refuses to believe |
| `src/web-bluetooth.ts` | the browser transport, and the only file that touches a radio |
| `src/index.ts` | the public surface |

The protocol, the session, and the timebase do no input and no output.
They take bytes that arrived and hand back bytes to send, so they run over
whatever Bluetooth stack you already have, on whatever runtime you already
use, and they can be tested without either.

```ts
import { Session, protocol } from "@intomind/sdk";

const session = new Session();
session.onDeviceInfo(await readDeviceInfoSomehow());
for (const event of session.onNotification(protocol.FILL.EEG_DATA, notification)) {
  // ...
}
```

## Capabilities are the device's answer, never this package's assumption

A device declares what it is over the protocol, and everything follows
from that declaration. The channel count, the converter's resolution and
reference, the rates it offers, whether it can detect lead-off, whether it
carries a model: all of it is asked and none of it is assumed.

```ts
session.requireInfo().channels   // what the device said
session.can("model")             // a capability bit, not a version number
session.setLeadoff(true)         // throws NotCapable if the device never claimed it
```

A table of specific hardware in this package is a defect. A capability the
device does not claim is not asked for, and asking anyway throws rather
than putting a byte on the wire the device will refuse.

## The timeline is never silently repaired

Every sample carries the device's own index and the device time of its own
conversion. A forward step in that index is a loss with an exact count. A
step that is not forward is a break whose extent is not a number, and this
package says so rather than returning a count that would be a fiction.

```ts
if (event.type === "gap") {
  event.gap.samplesLost;   // a number, or null for a break
}
```

A gap is emitted before the samples that follow it, so a caller writing
samples down writes the gap down first. Nothing splices.

## The contract is tested, not assumed

`contract/conformance.json` is emitted by the firmware from the codec the
device itself runs. Every message this package decodes, it decodes to the
fields that file states, and every malformed one it refuses for the reason
that file states: truncated, invalid, or reserved, which are three
different errors because the contract treats them as three different
things.

```
npm test
```

No device, no network, no build step. That is the command, and there is no
other. It needs Node 22.18 or newer, which runs the TypeScript as it is
written. `npm run build` produces the JavaScript that is published, which
runs on Node 20 and in any current browser.

## Time

A device's clock is a different crystal from the host's. The fit between
them is a line, and one exchange cannot tell a clock difference from a slow
answer, so one exchange fixes an offset and claims no rate. A skew beyond a
thousand parts per million, or a slope whose standard error is worse than a
hundred parts per million, is refused rather than believed. The offset is
kept, because the offset is the part that was measured.

```ts
await device.timeSync();
device.session.timebase?.fit;            // skewUsed says whether the rate is real
device.session.hostTime(batch.deviceTime);
```

## The model

A device can carry an encoder that turns a window of signal into an
embedding, and a head that turns that embedding into a few outputs. A head
is weights, never code.

```ts
import { buildHead, quantizeEmbedding, evaluateHead, decodeHead } from "@intomind/sdk";

const blob = buildHead(weights, bias, scale, { name: "focus" });
evaluateHead(decodeHead(blob), quantizeEmbedding(embedding));
```

The embedding is quantized to a fixed scale that is part of the contract,
so what a head produces on a host is what it produces on the device,
because the device evaluates exactly those integers.

## The browser

`web-bluetooth.ts` is the only file that reaches `navigator.bluetooth`.
Web Bluetooth needs a secure context and a user gesture, so
`requestDevice` has to be called from a click. Everything else in this
package works with no browser at all, which is how the tests run.

## License

Free software under the GNU Affero General Public License, version 3. A
commercial license is available for building a closed product on this code
or running a modified service without publishing the changes. Write to
contact@intomind.com. See `LICENSING.md`, and `CONTRIBUTING.md` before a
first pull request.

IntoMind is a trademark of IntoMind, Inc. The license covers the code, not
the name.
