/**
 * A JavaScript example the documentation shows, type checked: the release
 * gate holds the page's block to these lines.
 */

declare const chain: import("@intomind/sdk").DaisyChain;        // the devices the page connected

import { Connection, experiments } from "@intomind/sdk";

const show = (message: experiments.Message) => console.log(message);   // hears what a run says
const [first, second, third] = chain.devices;
if (!first || !second || !third) throw new Error("connect three devices first");
for (const device of chain.devices) if (device instanceof Connection) await device.calibrateClock();

const parts: Uint8Array[] = [];
const stop = chain.recordTo((chunk) => parts.push(chunk));   // every device's samples, the file record_to writes in Python
await chain.startAll();
// ...
await chain.stopAll();
stop();                                                       // stops writing

const names = await chain.run("berger", { block: 20, reps: 8, settle: 3 }, show,
                              { cfgs: [{ subject_id: "S01" }, { subject_id: "S02" }, { subject_id: "S03" }] });

const rest = new experiments.RunCtx(show);                    // two runs at once, each with its own Stop and markers
const work = new experiments.RunCtx(show);
const saved = (await Promise.all([
  chain.run("baseline", { seconds: 60 }, show, { devices: first, ctx: rest }),
  chain.run("record", { max_seconds: 600, markers: "start,end" }, show, { devices: [second, third], ctx: work }),
])).flat();
// work.stop.set() ends the second run alone

await chain.disconnect(third);                                // the others carry on
