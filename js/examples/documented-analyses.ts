/**
 * A JavaScript example the documentation shows, type checked: the release
 * gate holds the page's block to these lines.
 */

declare const recordings: import("@intomind/sdk/capture").Collection;   // the recordings to analyze

import { analysis, capture } from "@intomind/sdk";

const cues = analysis.events(recordings, "berger_20260101-102030");       // on the same timeline as the samples
const presses = analysis.inputs(recordings, "berger_20260101-102030");    // a controller's events, when the recording has them

analysis.ANALYSES.set("cue_count", {
  fn: (recordings: capture.Collection, labels: readonly string[]) =>
    capture.fromPlain({ cues: labels.map((label) => analysis.events(recordings, label).length) }),
  needsEvents: true, doc: "How many cues each recording holds.", params: [], several: true,
});
