/**
 * A JavaScript example the documentation shows, type checked: the release
 * gate holds the page's block to these lines.
 */

declare const link: import("@intomind/sdk").Connection;          // the link the page opened

import { blockedFolds, trainHead } from "@intomind/sdk/model";

const windows = await link.collectEmbeddings(60);                   // while streaming: 60 windows, at least 4 minutes at a model interval of 0
const labels = windows.map((_w, i) => (i < 30 ? 0 : 1));            // the first half one state, the rest the other
const rows = windows.map((w) => w.embedding ?? []);                 // every window carries one: the form asked for was "window"
const encoderId = windows[0]?.encoderId ?? null;                   // the encoder the windows came from
const head = trainHead(rows, labels, { name: "focus", encoderId });
const score = blockedFolds(rows, labels);                           // on windows it never saw, or null with too few
await link.uploadHead(head, 2, { select: true });
