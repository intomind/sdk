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
timeline, and the browser's connection to an IntoMind device. It
knows nothing about any particular device beyond what that device tells
it, except a wiring preset for IntoMind hardware version 1.1.7, since the
protocol gives a device no way to report how its electrodes are wired, and
nothing about any application built on top of it.

## What is in here

| | |
|---|---|
| `src/protocol.ts` | the wire, and nothing else |
| `src/session.ts` | one device as a state machine: bytes in, events out |
| `src/timebase.ts` | device time onto host time, and what it refuses to believe |
| `src/web-bluetooth.ts` | the browser's connection to an IntoMind device, over Web Bluetooth |
| `src/instrument.ts` | what a device can do, drawn from what it declares, and contact from its lead-off bits |
| `src/experiments.ts` | the experiments, run as the Python library runs them, with the same cues in the same order |
| `src/audio.ts` | the cue tones, the library's own sounds, and speech in the browser's voice |
| `src/analysis.ts` | the analyses, run on a recording as the Python library runs them, with the same report, and a recording's cues and controller inputs on the same timeline as its samples |
| `src/model.ts` | training a head, as the Python library trains one |
| `src/placement.ts` | where a device sat: every 10-10 position, what is under it, every region a person may name instead, and the placement a recording stores |
| `src/daisy-chain.ts` | several devices on one host, as the Python library's multi-device session: on one clock, streamed into one file, and any experiment run on any of them, alone, in groups or all at once, each device saved as its own recording |
| `src/capture/` | recordings: the recorder, the reader and the six exports, as the Python library makes them |
| `src/capture/node.ts` | recordings on disk, in the folder the Python library uses, for Node |
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

A table of specific hardware in this package is a defect, with one
exception: the wiring preset for IntoMind hardware version 1.1.7, since the
protocol gives a device no way to report how its electrodes are wired. A
device that reports any other hardware version gets no preset. A capability the device does not
claim is not asked for, and asking anyway throws rather than putting a
byte on the wire the device will refuse.

## The timeline is never silently repaired

Every sample carries the device's own index and the device time of its own
conversion. A forward step in that index is a loss with an exact count. A
step that is not forward is a break whose extent is not known, and this
package says so rather than returning a count that would be a fiction.

```ts
if (event.type === "gap") {
  event.gap.samplesLost;   // a number, or null for a break
}
```

A gap is emitted before the samples that follow it, so a caller writing
samples down writes the gap down first. Nothing splices.

A recording's rate is measured over the
stretches where its index climbs by one, so samples that were lost are
never read as slow sampling.

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

The tests need no device, no network beyond the machine's own loopback,
and no build step, and `npm test` runs every one of them. The tests need Node 22.18 or newer, which runs the TypeScript
as it is written. `npm run build` produces the JavaScript that is
published, compiled to ES2023. Its package.json names Node 20 or newer.

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

## Recordings

The session records as the Python library records, and
`@intomind/sdk/capture` reads recordings and exports them in the Python
library's six formats: BDF, EDF, CSV and TSV byte for byte the files Python
writes, NPZ and MATLAB holding the same arrays.

Lead-off is in every format, and only where the recording says detection
was running and the bits are a verdict. The device's lead-off is a byte,
one bit for each of up to eight channels. A `Recording`'s
`leadoff` gives its words as bytes, and null for a recording with a word
that holds a value a byte cannot, never cutting a word down to a byte,
and its `leadoffWords` gives every word at its full width.

From firmware 1.4.6 the device judges each electrode over a block of
0.512 s of samples, and until its first judgment of a stream every lead-off
bit reads 0, which says neither that an electrode is attached nor that one
is off. A sample the device loses starts its block over, and each packet
carries the bits of its last sample. So in a stream the package started
with detection on, no sample carries a verdict until a packet whose last
sample is the round(0.512 * rate)th or later, counted from the stream's
first sample, or from the first sample after a break in the sample index
before then: a batch says from which row its bits are one in `judgedFrom`
and `leadoffJudged(row)`. A recording one of whose samples carries no
verdict holds `leadoff_judged`, one flag a sample, which a `Recording`
gives as `leadoffJudged`. No format calls those samples attached: `csv`
and `tsv` leave their lead-off columns empty there, `edf` and `bdf` label
the stretch `LEADOFF_NOT_JUDGED`, a daisy chain's file leaves its
`leadoff` empty, and every summary's `contact` counts a sample as checked
only where detection was running and its bits are a verdict.

An IntoMind device whose firmware is older than 1.4.6 cannot check
electrode contact: its lead-off detection does not work, and every
electrode reads off whether it touches skin or not. On such a device
`setLeadoff(true)` is refused before anything is sent to the device, with
a `SessionError` whose message ends "This device's firmware (1.4.5) cannot
check electrode contact. Update it to 1.4.6 or later.", naming the
device's own version, and the profile's `leadoff` control is `supported: false`, with
that sentence in `why` for an application to show in place of a contact
check. `setLeadoff(false)` is not refused. Where detection is on all the
same, as another program can leave it, no row of such a device's batches
with detection on is a verdict, `leadoffJudged` says so, and a recording
holds `leadoff_judged`. A recording of such a device that holds no
`leadoff_judged` is read the same way, from the firmware its manifest
names, and the recording `Collection.load` gives says so in `repairs`.
Either way no export calls an electrode off: `contact` says "not checked",
the text formats have no lead-off column, and `bdf` and `edf` carry no
lead-off label.

```ts
for (let i = 0; i < 5; i++) await device.timeSync();
device.session.armRecorder();
await device.request(device.session.resetEpoch());
await device.request(device.session.startStream());
// ...
await device.request(device.session.stopStream());
const saved = device.session.saveRecording("my-recording");  // each file's name and bytes
```

## The analyses

`@intomind/sdk/analysis` runs the Python library's analyses on a
recording, with the same report, key for key, and lists them in `ANALYSES`
as the library's registry does. An analysis takes one recording's label
first, and an application runs it on each selected recording separately.
An application adds its own to `ANALYSES`, and one it registers with
`several: true` takes a list of the selected recordings' labels first
instead, and is run once, on all of them together. Every analysis of the
package's takes one label. `events` and `inputs` put a recording's cues
and its controller events on the same timeline as its samples, `host_time`
the aligned instant and the time stamped kept as `host_time_recorded`, as
the library's do. The `summary` analysis gives each channel's
`saturation_pct`, the share of its counts beyond 99 percent of the
converter's full scale, from the width the recording states in `adc_bits`,
24 bits where it states none.

```ts
import { analysis, capture } from "@intomind/sdk";

const recordings = new capture.Collection(files);
const spec = analysis.ANALYSES.get("berger")!;
const report = spec.fn(recordings, "berger_20260101-102030", { epoch_s: 2 });
const cues = analysis.events(recordings, "berger_20260101-102030");
```

## Several devices

A `DaisyChain` is the Python library's multi-device `Session`. Connect the
devices a person picked from what the browser found, say who wears each,
and run any registered experiment on any of them: alone, in groups, or all
at once, each device saved as its own recording. Nothing is connected that
was not chosen.

```ts
import { Connection, DaisyChain, experiments } from "@intomind/sdk";

const chain = new DaisyChain();
await chain.connect(...picked);                                  // the devices a person picked
for (const d of chain.devices) if (d instanceof Connection) await d.calibrateClock();   // each on the chain's one reference point
const wearers = [{ gain: 24, subject_id: "S01" }, { gain: 24, subject_id: "S02" }, { gain: 24, subject_id: "S03" }];

// Everyone does the same thing: one run, every device, one Stop.
const ctx = new experiments.RunCtx(show, { gain: 24 });
const names = await chain.run("record", { max_seconds: 600 }, show, { cfgs: wearers, ctx });
// ctx.stop.set() ends it, and each device is saved as its own recording

// Two runs at once, on different devices, each with its own Stop and markers.
const [a, b, c] = chain.devices as [Connection, Connection, Connection];
const [first, second] = await Promise.all([
  chain.run("baseline", { seconds: 60 }, show, { devices: [a] }),
  chain.run("record", { max_seconds: 600 }, show, { devices: [b, c] }),
]);

// One experiment that hands each group its own at the same moment, or in turn.
const berger = experiments.REGISTRY.get("berger")!.fn;
const baseline = experiments.REGISTRY.get("baseline")!.fn;
experiments.experiment("groups", "Alpha blocking beside a resting group.", [], async (dev, emit, p, ctx) => {
  const all = Array.isArray(dev) ? dev : [dev];              // a run of one device hands the device itself
  const [x, y] = await Promise.all([berger(all.slice(0, 2), emit, p, ctx), baseline(all.slice(2), emit, p, ctx)]);
  return [...x, ...y];
});
experiments.experiment("swap", "One group works while the other rests, then the reverse.", [], async (dev, emit, p, ctx) => {
  const all = Array.isArray(dev) ? dev : [dev];
  const x = await berger(all.slice(0, 2), emit, p, ctx);     // the others wait, left as they are
  const y = await berger(all.slice(2), emit, p, ctx);
  return [...x, ...y];
});
```

A run naming no devices uses every device of the chain, and `devices`
names any of them, as an array, a set, or one connection on its own. A
device that is not the chain's, or is named twice, is refused by name
before anything is sent. An experiment's function is handed the connection
itself on a run of one, and a new array of the connections on a run of
several. Its code hands any group of them to the package's experiments,
passing on the `emit` and `ctx` it was handed, and every one of them takes
one connection or several. Several runs go at once on different devices,
each with its own context, so its own `ctx.stop`, markers and events, and
setting a run's `ctx.stop` ends every step of that run.

An experiment of one's own that cues trials, one cue a trial with a rest
before each, is built from the step the package's blink and jaw clench
protocols use, `experiments.cuedTrials`, as the Python library's are built
from its `_cued_trials`, with `experiments.label` naming the recording as
the package's experiments name theirs. The trials run in the order given,
which the recording keeps as `order`, so shuffle them first with a fixed
seed: a subject who can predict the next cue is a different subject, and a
fixed seed lets the order be told again. Give `restText`, since a cue that
starts a state needs a cue that ends it.

```ts
import { capture, experiments } from "@intomind/sdk";

experiments.experiment(
  "my_protocol",
  "One line describing what it measures.",
  [
    { name: "trials", type: "number", default: 20 },
    { name: "hold_s", type: "number", default: 2.0 },
    { name: "note", type: "text", default: "" },
    ...experiments.TONE_PARAMS,
  ],
  async (dev, emit, p, ctx) => {
    const conds = Array.from({ length: Math.floor(Number(p.trials) / 2) }, () => ["go", "sham"]).flat();
    shuffle(conds, 1234);                    // your own shuffle, with a fixed seed, so the order can be told again
    return experiments.cuedTrials(dev, emit, p, ctx, {
      label: experiments.label(ctx, "my"),
      conds,
      texts: { go: "Do the thing", sham: "Stay still" },
      restText: "Rest",
      holdS: Number(p.hold_s),
      restS: 3.0,
      intro: "My protocol. Do the thing when told, rest when told.",
      extra: capture.fromPlain({ experiment: "my_protocol" }),
    });
  },
  { prereg: { name: "my_protocol", window_s: 1.0, min_ratio: 5.0 } },
);
```

What a step does to a device is the library's. It starts the streams of
the devices it records that were not running, and stops only those it
started, so a device that was streaming when the step began is still
streaming when it ends, and nothing records it after. A device another
step or another run is recording is refused, in words that name it, before
anything is sent to any device of the step, since a recording restarts the
count of a device that can restart it. A setup in `cfgs` that gives an IntoMind device
a gain other than the gain in force, which the device's Status states, is
refused in words naming both gains before anything records, since the
recording carries the gain its packets carry. A parameter of the
package's experiments that is not given, that cannot be read as the number
or the whole number it needs, or whose text the experiment cannot read,
such as a line of `steps` with no "|", is refused in words that name it,
before anything is sent to any device, so no stream is started and no
device is held for it. Infinity and NaN are refused as any other value
that is not a number is, a whole number given with a fraction is cut to its
whole part, and no number is checked against a range. A step whose
recording cannot be saved still stops the streams it started and lets its
devices go. A device
whose link is down when a step starts is not recorded, and the step says
so and goes on with the others, unless every device's link is down: then
the step fails with the error raised by the first device it writes to. One
whose link drops has its recording saved with `aborted` set to
"link_lost" while the others carry on, and one that ends its stream itself
because USB power appeared is saved with every sample up to that moment,
with `ended_by_usb` in its manifest. On one device there is no other to
carry on with, so head training whose link drops ends at once, whether the
link drops between reps or while a rep's windows are waited for: it saves
its recording with `aborted` set to "link_lost" and what it kept beside it,
then throws a `SessionError` whose message begins `ConnectionError`. A device
that refuses to start leaves nothing armed or started by the step.
JavaScript cannot end a promise from outside, so a step that fails does
not end the others beside it the way a Python task group does: set the
run's `ctx.stop` to end them, and each saves what it recorded.

Every recording a run makes carries `run`, the experiment's registered
name and when the run started in UTC to the millisecond, the same in every
one, and `together`, the names of the others its step saved at the same
moment. Each device's setup in `cfgs`, such as who wears it, lands in its
own manifest. A cue a step stamps is written into the recordings of the
devices that step handles, and every other event, a person's marker among
them, into every recording the run saves after it is stamped. A run on one
device saves its recording under the name its step gives it. A run of
several adds each device's id to that name, and where the name gives a
gain, puts in it the gain the device's own setup gives, or `na` where its
setup gives none. A name a run of the same chain already saved has `_2`, `_3` added. A
run gives back the name of every recording it saved, and every message it
sends carries `devices`, the id of each device it concerns. The line the
recordings folder's session log gains for a recording a run made says in
`run` which run made it.

Every device connected through a chain places its samples on the chain's
one reference point, the first clock anchor any of them took, even where
the wall clock stepped between their connections. Two IntoMind devices'
samples taken at the same moment carry nearly the same host time: apart by
up to the round trips of their clock exchanges, plus however far each
device's clock has drifted from the clock map in force when its stream
started. Each recording lists that anchor among its own, so its events
still line up with its samples. `recordTo(write)` opens one file, byte for
byte what the library's `record_to` writes, and gives back a function that
stops writing. Every sample of each device `startAll` started is written
into it while it is open, each value under the heading that names it.
`startAll` refuses a device with more channels than the file's header
names before any device starts, and
`disconnect(device)` lets one device go while the others carry on.
`contract/together` holds all of this to the library, and the programs it
runs include each way of running a chain shown above.

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

`@intomind/sdk/model` trains a head as the Python library does, from the
embeddings a device sends and one target per window.

```ts
import { trainHead } from "@intomind/sdk/model";

const windows = await link.collectEmbeddings(60);                   // while streaming
const rows = windows.map((w) => w.embedding ?? []);                 // every window carries one: the form asked for was "window"
const blob = trainHead(rows, targets, { name: "focus", encoderId: windows[0]?.encoderId ?? null });
```

`blockedFolds(rows, labels)` says how well such a head tells two or more
states apart on windows it never saw, as the library's
`model.blocked_folds` does: for k states the labels are 0 to k-1, two
states are scored by whether the head's one output is above one half, and
more by which of one output for each state is the largest.
The experiments' `head_training` runs the whole of it, as the library's
does, for a pair of states it offers or, with `pair` set to `own`, for two
or more a person writes in `states`, one to a line as `name | cue words`:
reps cued aloud in turn, each `rep_length` windows long, the device's
embeddings labeled by the state cued, a head trained on them and scored
that way. More than two states train one output for each. States not
written so are refused, saying which line
and why, before anything is sent to any device, and so are fewer `reps`
than states, which would leave a state never cued, in words saying how
many are needed. The recording's `states`
keeps each state's name and the cue words a person heard, in the order of
their labels, and the training data saved beside it names the states its
head tells apart. It leaves the head with the person, who uploads it with
`uploadHead` to the slot they choose.

## The browser

`web-bluetooth.ts` is the only file that reaches `navigator.bluetooth`.
Web Bluetooth needs a secure context and a user gesture, so
`requestDevice` has to be called from a click. Everything else in this
package works with no browser at all, which is how the tests run.

A `Connection` drives the device as the Python library's `Device` does,
method for method: `refreshConfig`, `setGain`, `setRate`, `setMode`,
`setLeadoff`, the chain (`pipeline`, `setPipeline`, `clearPipeline`,
`restorePipelineDefault`, `pipelineCatalog`), the lamp (`indicator`,
`setIndicator`, `identify`), the name (`getName`, `setName`), the model
(`modelInfo`, `heads`, `selectHead`, `setPredictions`, `modelInterval`,
`setModelInterval`, `setEmbeddings` with `onEmbedding`, `collectEmbeddings`,
`uploadHead`, `removeHead`),
`converterRegisters`, `calibrateClock`, `start` and `stop`. Each writes the
requests the library writes, in its order, and refuses in its words, held
to `contract/device`.

Control exchanges on one connection take turns, so each takes its own
answer, as the library's do. When an earlier exchange of an opcode stopped
waiting, the next answer of that opcode may be its late one, so it is held:
dropped when another of the opcode follows, and taken as the exchange's own
when none does by its timeout. An exchange whose answer never comes costs
the next exchange of the same opcode its full timeout, and nothing more.
An answer that does not come in time, to a control request or to an
exchange with the update service, a progress check's last asking among
them, throws `Unanswered`, an error of its own and not a `SessionError`,
so a program tells a device that stopped answering from a refusal, a link
that is down or a write the system turned away, each of which keeps its own
class. A transfer that stops on a link that is up waits up to five seconds
for the device's answer to ABORT before it throws.

A browser does not say how large a write its link carries, so a transfer
starts at 244 bytes, the largest the contract allows. A data write the
browser refuses while the link is up, or a progress check that shows the
device's count not moving, its checksum not matching ours over as many of
the file's first bytes, or the device failing the transfer at its first 32
bytes with a checksum that is not ours over the writes that carried them,
starts the file again at 101 bytes and then at 20: ABORT, START again, and
the data from the count the device's answer to START gives, which is where
its progress then stands. Progress is reported, as the Python library
reports it, after each progress check that finds the device holding what was
sent, and for that answer to START. Only when 20 fails too does the transfer end, in the error it ends
in on any link, and a link that goes down during that ABORT ends it in the
error that stopped the larger size. A later transfer on the same connection
starts at the size the last one finished at. A program that drives a
`Transfer` itself tells it the largest write its system reports, where above
20, with `limitWrites(n)`, and a data write the system refused with
`writeRefused()`, ends a transfer whose link goes down during that ABORT in
the write's own error or in `smallerBecause`, and gives a later transfer on
the same connection the last one's `largestWrite`.

`await link.activate()` puts an image a transfer verified in force, as the
library's `activate` does. It waits five seconds for the device's answer,
and only an answer of ok resolves. It throws `Refused` with the device's
status in words when the device refuses, and `Unanswered` when no answer
comes in that time, a link that drops before the answer comes among them,
since only the answer says the device accepted.

A listener that throws has its error handed to the runtime's
`reportError` where it has one, as a browser reports an uncaught error,
and to `console.error` where it has not, and it stops neither the other
listeners, nor the windows being put together, nor the package's own
consumers: a
run's recorder, a chain's file and the windows the package gathers hear a
device by a path of their own, before any callback a program set, so a
callback set in the middle of a run cuts nothing off. Embeddings are left
as they were found: `collectEmbeddings` and head training turn them on, or
widen the form to carry what they gather, only when it does not already,
put back the form they found, and never touch `onEmbedding`. The form the
package believes is set never keeps it from asking for a form it needs:
after a request to set the form failed, the form needed is asked for
again, and a form found that could not be put back, as when the link had
dropped, is put back by the next consumer to let go. `onUsbStop` is
called each time the device ends its stream because USB power appeared. A
`Connection`'s own `connect()` connects it again to the same device after
its link drops, keeping its listeners, `onEmbedding`, `onUsbStop` and its
session, which reads the device's info again where the device states
different info, as after a firmware update. The `onDisconnect` the
connection was made with hears every drop, until a later `connect` to the
same device, a new connection, takes it off, so each drop is reported
once, by the newest connection.

## License

Free software under the GNU Affero General Public License, version 3. A
commercial license is available for building a closed product on this code
or running a modified service without publishing the changes. Write to
contact@intomind.com. See `LICENSING.md`, and `CONTRIBUTING.md` before a
first pull request.

IntoMind is a trademark of IntoMind, Inc. The license covers the code, not
the name.
