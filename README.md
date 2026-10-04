# IntoMind SDK

Build on an IntoMind device, in the language you already use.

| | |
|---|---|
| `crates/intomind-protocol` | the wire, and nothing else |
| `crates/intomind` | one device as a state machine, over any transport |
| `crates/intomind-capture` | what a recording is on disk |
| `crates/intomind-model` | the device's encoder, the arithmetic exactly as the device does it |
| `crates/intomind-pipeline` | the device's processing chain, so a host can check one before sending it and reproduce what the device did |
| `js/` | the same, in TypeScript, for browsers and Node |

Python users want `intomind` on PyPI, which is the instrument library and
carries the recorder, the provenance, the exports, and the analyses as
well. This repository is the protocol and the model for everyone else.

## One contract, several implementations, held together by tests

`contract/conformance.json` is emitted by the device firmware from the
codec the device itself runs. It carries 66 messages with the fields those
bytes mean, 56 malformed ones with the reason each is not a message, the
scaling from counts to microvolts, and the continuity rule.

Every implementation here decodes those vectors to the stated fields and
refuses the malformed ones for the stated reason. So does the Python
library, and so does the firmware. An implementation that drifts fails a
test rather than a bench session.

Two things about the file are worth knowing before you write a fifth
implementation. A 64-bit field is written as a decimal string, because a
JSON number cannot hold a device time exactly and a reader that parses one
as a number agrees with an equally rounded expectation. And a count that
does not exist is null: a break in the timeline has no extent, and zero
would claim that nothing was lost.

## Your head, on your device

A head is a small model that turns the device's embedding into whatever
you are measuring. You train it on your own recordings and put it on the
device, and the device runs it there.

The device checks a head for its own hash and for its shape, never for a
signature of ours. Our firmware is signed and the device refuses anything
else in its place. Your head is not signed, and the device runs it.

Every package here can build one and carry one, without a Python runtime
and without doing the input and output for you:

```rust
let mut transfer = Transfer::head(&session, 1, &blob)?;
loop {
    match transfer.step(&blob) {
        Step::Send(command) => { write(&command); transfer.on_answer(&answer()?, &blob)?; }
        Step::Data { from, to, .. } => { write_data(&blob[from..to]); transfer.sent(to - from); }
        Step::Verified(_) => break,
    }
}
```

Nothing is activated by a transfer. What lands is verified where it
landed, and putting it in force is a separate act.

## No transport is forced on you

The Rust client does no input and no output. It is handed bytes that
arrived and it hands back bytes to send, so it works over whatever
Bluetooth stack you already have, on whatever runtime you already use, and
it can be tested without either.

```rust
let mut session = Session::new();
session.on_device_info(&read(DEVICE_INFO)?)?;
let command = session.start_stream();
write(command.characteristic, &command.bytes)?;

for event in session.on_notification(uuid_fill::EEG_DATA, &notification) {
    match event {
        Event::Samples(batch) => { /* batch.microvolts(row, channel, &info) */ }
        Event::Gap(gap) => { /* written down, never smoothed over */ }
        _ => {}
    }
}
```

## Capabilities are the device's answer

What a device can do is the bits it reports, never a version number and
never a table of hardware here. A device that does not claim a capability
is not asked for it, and asking is refused before anything reaches the
wire.

## The device processes its signal, and says so

A device that claims the processing chain preprocesses its own signal: a
chain of stages, filters first. When a chain is in force only its output
streams, and the chain is readable at any time, so a recording can say
exactly what produced its samples. There is no raw-or-filtered flag
anywhere, because a flag says nothing about what a signal is. The rules a
chain must meet are the device's, in `intomind-pipeline`, so a chain that
cannot run at the device's rate is refused here with its reason before
anything is sent.

## The timeline is never silently repaired

A forward step in the sample index is a loss with an exact count. A step
that is not forward is a break whose extent is not a number, and this
reports it as unknown rather than as zero, because zero missing samples
is a different statement. The subtraction is signed, so the index's own
wrap is exact.

A break is announced before the samples that follow it, so a caller
writing samples down writes the break down first and can never splice.

## The model

A device can carry an encoder that turns four seconds of signal into an
embedding. `intomind-model` is that encoder: given the weights file a
device runs, a host produces the same numbers. The encoder's weights are
not published, and the device sends its embeddings, so heads are trained
on those. The crate is `no_std` and allocates nothing, which is how the
same crate runs on the device.

A head turns an embedding into outputs. Heads are weights, never code,
and what a head produces on a host is what it produces on the device
because the device evaluates exactly those integers.

## License

Free software under the GNU Affero General Public License, version 3. We
also license it commercially, on fair terms shaped by your use case: write
to contact@intomind.com. See `LICENSING.md`, and `CONTRIBUTING.md` before a
first pull request.

IntoMind is a trademark of IntoMind, Inc. The license covers the code, not
the name.
