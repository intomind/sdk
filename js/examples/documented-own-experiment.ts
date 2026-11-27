/**
 * A JavaScript example the documentation shows, type checked: the release
 * gate holds the page's block to these lines.
 */

import { experiments } from "@intomind/sdk";

const berger = experiments.REGISTRY.get("berger")!.fn;
const baseline = experiments.REGISTRY.get("baseline")!.fn;
experiments.experiment("groups", "Alpha blocking on two devices beside a resting group.",
  [{ name: "block", type: "number", default: 20 }, { name: "reps", type: "number", default: 8 },
   { name: "settle", type: "number", default: 3 }, { name: "seconds", type: "number", default: 60 },
   ...experiments.TONE_PARAMS],
  async (dev, emit, p, ctx) => {
    const all = Array.isArray(dev) ? dev : [dev];                          // a run of one device hands the device itself
    const rest = all.slice(2);                                             // the resting group, none on a run of one or two
    const [x, y] = await Promise.all([berger(all.slice(0, 2), emit, p, ctx), rest.length > 0 ? baseline(rest, emit, p, ctx) : []]);
    return [...x, ...y];
  });
