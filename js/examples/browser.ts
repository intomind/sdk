/**
 * The example the documentation shows, compiled.
 *
 * It is here rather than only in prose so that it is type checked against
 * the package it documents. An example that no longer compiles is a
 * broken promise to whoever copies it.
 */
import { requestAndConnect } from "../src/index.ts";

export async function start(): Promise<void> {
  // The browser asks the user to pick a device. It must start from a click.
  const link = await requestAndConnect();

  link.on((event) => {
    if (event.type === "samples") {
      const uv = event.batch.microvolts(0, 0, link.info);
      void uv;
    }
    if (event.type === "gap") {
      // written down, never smoothed over
      void event.gap;
    }
  });

  await link.send(link.session.startStream());
}

/** Putting your own head on the device, from a browser. */
export async function uploadHead(blob: Uint8Array): Promise<void> {
  const { Transfer, requestAndConnect } = await import("../src/index.ts");
  const link = await requestAndConnect();
  const transfer = Transfer.head(link.session, 1, blob);
  await link.transfer(transfer, blob, ({ taken, total }) => {
    void taken;
    void total;
  });
  await link.send(link.session.selectHead(1));
  await link.send(link.session.setPredictions(true));
}
