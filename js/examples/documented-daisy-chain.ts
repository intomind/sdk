/**
 * A JavaScript example the documentation shows, type checked: the release
 * gate holds the page's block to these lines.
 */

import { DaisyChain, requestDevice } from "@intomind/sdk";

const chain = new DaisyChain();
document.querySelector("#add")?.addEventListener("click", async () => {
  await chain.connect(await requestDevice());           // one more device, beside the others
});
