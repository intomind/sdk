/**
 * A JavaScript example the documentation shows, type checked: the release
 * gate holds the page's block to these lines.
 */

import { requestAndConnect } from "@intomind/sdk";

const link = await requestAndConnect({ onDisconnect: () => console.log("the link dropped") });   // hears every drop
link.onUsbStop = (connection) => console.log("USB was plugged in, so the stream ended");       // called each time USB power ends a stream
await link.connect();                                   // the same connection again, its listeners and session kept
