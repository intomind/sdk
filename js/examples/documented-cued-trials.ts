/**
 * A JavaScript example the documentation shows, type checked: the release
 * gate holds the page's block to these lines.
 */

declare function shuffle(order: string[], seed: number): void;   // your own shuffle

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
