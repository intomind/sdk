/**
 * The package's own path to a connection: what its own consumers hear, and
 * what it knows of the embeddings it found on.
 *
 * A run's recorder, a daisy chain's file and the windows the package
 * gathers for itself each hear a connection here, before any listener a
 * program added with `on`, before `onUsbStop` and before `onEmbedding`, as
 * the Python library's own consumers hear a `Device` by a path of their
 * own. A program never sets anything here, so a callback it sets during a
 * run cuts no recording off, and nothing the package puts back is put over
 * what the program set. One consumer that throws is reported, and the
 * others still hear everything.
 *
 * Internal: nothing here is part of the package's surface.
 */

import { EMBEDDING_FORMS, Invalid, type EmbeddingWindow } from "./protocol.ts";
import * as J from "./capture/pyjson.ts";
import { stateGain, statedGain } from "./capture/record.ts";
import { SessionError, type Batch } from "./session.ts";
import type { Connection } from "./web-bluetooth.ts";

/** What the package's own consumers hear: every batch of samples, and every whole window of the encoder's output. */
export interface Heard {
  /** A batch of samples, as the connection's session passed it on. */
  readonly batch: Batch;
  /** A whole window of the encoder's output. */
  readonly window: EmbeddingWindow;
}

/** While the package's own consumers need embeddings on, as the library's `_lent`. */
export interface Lent {
  /** The form they found on. */
  found: string;
  /** How many times the form had been set when the package last looked or set it. */
  seen: number;
  /** How many of them need embeddings now. */
  users: number;
}

/** What the package keeps of one connection for itself. */
export interface Own {
  /** Its own consumers of every batch of samples. */
  readonly batch: ((batch: Batch) => void)[];
  /** Its own consumers of every whole window. */
  readonly window: ((window: EmbeddingWindow) => void)[];
  /** The form of embeddings this side holds the device to be sending, as the library's `_embeddings_form`. */
  form: string;
  /** How many times the form has been set, as the library's `_form_changes`. */
  changes: number;
  /**
   * Whether the form this side believes may not be the device's, as the
   * library's `_embeddings_in_doubt`: the last request to set it failed, so
   * it may or may not have reached the device. While in doubt, a form the
   * package needs is asked for even when this side believes it is on.
   */
  inDoubt: boolean;
  /** While its own consumers need embeddings on, what lets them be left as found. */
  lent: Lent | null;
  /** The gain the device was last set to or last said, as the library's `Device._gain`, which starts at 6: kept by its session, so a recording of the session alone carries it too. */
  gain: number;
  /** The rate the last packet stated, or 0 for a code the contract does not define, as the library's `_rate_wire`. Null before any packet. */
  rateWire: number | null;
  /** Where the last packet's signal came from, in words, as the library's `_mode_wire`. Null before any packet. */
  modeWire: string | null;
  /**
   * Whether `transfer` is carrying something to the device now, as the
   * library's `Device._transferring`. A device writing a page of its flash
   * holds eight incoming writes and throws away what arrives beyond them,
   * and a request thrown away is never answered, so a daisy chain's
   * `resyncLoop` asks nothing of a device while this is set.
   */
  transferring: boolean;
}

const OWN = new WeakMap<Connection, Own>();

/** What the package keeps of `link`, made the first time it is asked for. */
export function own(link: Connection): Own {
  let o = OWN.get(link);
  if (o === undefined) {
    o = {
      batch: [],
      window: [],
      form: "off",
      changes: 0,
      inDoubt: false,
      lent: null,
      get gain(): number {
        return statedGain(link.session);
      },
      set gain(g: number) {
        stateGain(link.session, g);
      },
      rateWire: null,
      modeWire: null,
      transferring: false,
    };
    OWN.set(link, o);
  }
  return o;
}

/** A device as a warning about it names it: its name, or the browser's identifier for it when it has none. */
function nameOf(link: Connection): string {
  return link.device.name || link.device.id || "a device";
}

/** What each kind is called where a consumer that threw is reported. */
const WORDS: { readonly [K in keyof Heard]: string } = { batch: "batch of samples", window: "window" };

/**
 * Add `fn`, one of the package's own consumers, to hear every `kind` from
 * `link`, and give back how to take it off.
 */
export function tap<K extends keyof Heard>(link: Connection, kind: K, fn: (item: Heard[K]) => void): () => void {
  const list = own(link)[kind] as ((item: Heard[K]) => void)[];
  list.push(fn);
  return () => {
    const i = list.indexOf(fn);
    if (i >= 0) list.splice(i, 1);
  };
}

/**
 * Hand one item to each of the package's own consumers of its kind. One
 * that throws is reported, as the library warns of one, and the others
 * still hear it.
 */
export function hand<K extends keyof Heard>(link: Connection, kind: K, item: Heard[K]): void {
  const o = OWN.get(link);
  if (o === undefined) return;
  for (const fn of [...(o[kind] as ((item: Heard[K]) => void)[])]) {
    try {
      fn(item);
    } catch (e) {
      const err = e instanceof Error ? e : new Error(String(e));
      console.warn(
        `${nameOf(link)}: one of the package's own consumers of each ${WORDS[kind]} threw ${err.name}: ${err.message}, ` +
          `and the others still hear every ${WORDS[kind]}`,
      );
    }
  }
}

/** What each form of embeddings carries. */
const CARRIES: Readonly<Record<string, readonly string[]>> = {
  off: [],
  window: ["window"],
  tokens: ["tokens"],
  both: ["window", "tokens"],
};

/** The least form of embeddings that carries all that `have` and `want` carry between them, as the library's `_covering`. */
function covering(have: string, want: string): string {
  const need = new Set([...(CARRIES[have] ?? []), ...(CARRIES[want] ?? [])]);
  for (const [form, parts] of Object.entries(CARRIES)) {
    if (parts.length === need.size && parts.every((p) => need.has(p))) return form;
  }
  return want;
}

/**
 * Hand every whole window from `link` that carries `form` to `take`, one of
 * the package's own consumers, with embeddings on in a form that carries
 * `form`, and give back a function that undoes it, as the library's
 * `_embeddings_for` does. A window that lacks it, as one the device sent in
 * the form it had before it was widened, is passed over for `take`, and
 * still reaches `onEmbedding`.
 *
 * Embeddings are left as they were found. They are turned on, or widened
 * to carry `form` as well, only when they do not already, and once no
 * consumer of the package's needs them any more the form found is put
 * back, unless something else has set the form since. A form that was on is
 * never turned off, and `onEmbedding` is never touched.
 *
 * The form the package believes is set never keeps it from asking for a
 * form it needs: after a request to set the form failed, the form needed is
 * asked for even where the package believes it is set. A form found that could not be
 * put back, as when the link had dropped, is still owed, and is put back
 * by the next consumer to let go, unless something else sets the form
 * first.
 */
export async function embeddingsFor(
  link: Connection,
  form: string,
  take: (window: EmbeddingWindow) => void,
): Promise<() => Promise<void>> {
  if (!link.session.can("embeddings")) {
    throw new SessionError("RuntimeError: this device does not send its encoder's output: it does not claim the embeddings capability");
  }
  if (!(form in EMBEDDING_FORMS)) {
    throw new Invalid(`ValueError: the form is one of [${Object.keys(EMBEDDING_FORMS).map((k) => J.quoted(k)).join(", ")}], not ${J.shown(form)}`);
  }
  const o = own(link);
  let lent = o.lent;
  if (lent === null) {
    lent = o.lent = { found: o.form, seen: o.changes, users: 0 };
  } else if (lent.seen !== o.changes) {
    // The program has set the form since: what is on is its own.
    lent.found = o.form;
    lent.seen = o.changes;
  }
  const mine = lent;
  mine.users += 1;
  const need = CARRIES[form] ?? [];
  const off = tap(link, "window", (w) => {
    if ((need.includes("window") && w.embedding === null) || (need.includes("tokens") && w.tokens === null)) return;
    take(w);
  });

  const undo = async (): Promise<void> => {
    off();
    mine.users -= 1;
    if (mine.users > 0 || o.lent !== mine) return;
    if (mine.seen === o.changes && o.form !== mine.found) {
      try {
        await link.setEmbeddings(mine.found);
      } catch {
        // The form found is still owed, so what was found is kept for the
        // next consumer to put back.
        return;
      }
    }
    o.lent = null;
  };

  const before = o.form;
  const want = covering(before, form);
  if (want !== before || o.inDoubt) {
    try {
      await link.setEmbeddings(want);
    } catch (e) {
      // The request may have reached the device, so the form it had is
      // asked for again, and this consumer is let go.
      try {
        if (want !== before) {
          await link.setEmbeddings(before);
          mine.seen = o.changes;
        }
      } catch {
        // Nothing more to say to a device that refused twice.
      } finally {
        await undo();
      }
      throw e;
    }
    mine.seen = o.changes;
  }
  return undo;
}
