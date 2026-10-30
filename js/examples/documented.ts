/**
 * Every JavaScript example the documentation shows, type checked.
 *
 * The examples are here rather than only in prose so that each is checked
 * against the package it documents (`npm run typecheck`). The release gate
 * holds the documentation's blocks to this file line for line, so what the
 * page shows is what compiled. The declarations between the blocks stand in
 * for what a page has at hand: the link it opened, a head's bytes, a
 * recording's files.
 */

// Connect and stream.
import { requestAndConnect } from "@intomind/sdk";

document.querySelector("#connect")?.addEventListener("click", async () => {
  const link = await requestAndConnect();              // the browser lists the devices nearby

  link.on((event) => {
    if (event.type === "samples") {
      const batch = event.batch;
      for (let row = 0; row < batch.rows; row++) {
        const uv = batch.microvolts(row, 0, link.info);  // channel 1, in microvolts
      }
    } else if (event.type === "gap") {
      console.log("lost", event.gap.samplesLost);        // null when the device restarted its count
    }
  });

  const answer = await link.request(link.session.startStream());
  if (!answer.ok) console.log("refused:", answer.statusName);
});

declare const link: Awaited<ReturnType<typeof requestAndConnect>>;
declare const blob: Uint8Array;
declare const deviceInfoBytes: Uint8Array;
declare const notificationBytes: Uint8Array;
declare const files: Map<string, Uint8Array>;

// Commands.
import { protocol as P } from "@intomind/sdk";

const s = link.session;
await link.request(s.setRate(250));                     // refused while streaming
await link.request(s.setIndicator("verbose"));          // "silent", "reserved" or "verbose"
await link.request(s.identify(5));                      // the status light winks for five seconds
await link.request(s.setName("Ada", "Blue"));           // "Ada's Blue IntoMind One"
await link.request(s.setPipeline([P.highpass(0.5), ...P.mainsBands(60, 500), P.lowpass()], 500));
await link.request(s.setMode("synthetic"));             // the generated signal, at 500 per second

// Run your own head.
import { Transfer } from "@intomind/sdk";

const transfer = Transfer.head(link.session, 2, blob);   // slot 2: slot 1 holds the open sex head
await link.transfer(transfer, blob, ({ taken, total }) => console.log(`${taken} of ${total}`));
await link.request(link.session.selectHead(2));
await link.request(link.session.setPredictions(true));

// Node.
import { Session, FILL } from "@intomind/sdk";

const session = new Session();
session.onDeviceInfo(deviceInfoBytes);                  // read from the Device Info characteristic
for (const event of session.onNotification(FILL.EEG_DATA, notificationBytes)) {
  // the same events as in a browser
}
const command = session.startStream();                  // write command.bytes to its characteristic

// Record.
for (let i = 0; i < 5; i++) await link.timeSync();   // a few exchanges before the stream

link.session.armRecorder();
await link.request(link.session.resetEpoch());
await link.request(link.session.startStream());
// the batches arrive as before
await link.request(link.session.stopStream());

const saved = link.session.saveRecording("my-recording", {}, { extra: { experiment: "my-experiment" } });
// saved.files holds each file's name and bytes: the npz, the manifest and the checksums

// Recordings and exports, in a browser.
import { Collection, exportRecording } from "@intomind/sdk/capture";

const recordings = new Collection(files);                        // file name -> bytes
const bdf = exportRecording(recordings, "my-recording", "bdf");  // also "edf", "npz", "csv", "tsv" and "mat"
// bdf.bytes is the file, and bdf.summary what the Python API reports for it

// Recordings and exports, in Node.
import { defaultDir, exportToFile, openCollection } from "@intomind/sdk/capture/node";

const summary = exportToFile(defaultDir(), "my-recording", "bdf");  // the folder the Python API uses
const recording = openCollection().load("my-recording");            // refused if its checksums do not match

export {};
