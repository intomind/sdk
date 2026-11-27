/**
 * Cue tones and spoken cues, as the Python library's `audio` module plays
 * them. Every cue point is configurable, silence included.
 *
 * A tone is the library's own sound, sample for sample, held to the
 * library's files by the fixture in `contract/experiments`. In a browser a
 * tone plays through the page's audio output and speech through the
 * browser's own voice, where the library plays through Linux's `aplay` and
 * speaks through `spd-say`. Nothing here waits for a sound to finish and
 * nothing here throws: a computer that cannot play or speak stays silent,
 * as the library does where those programs are missing, because a missing
 * sound device must never abort a recording. An event stamped right after
 * the call, as each experiment of the package's stamps its cue, comes before
 * the cue is heard, by a delay that has not been measured.
 */

import { CHOICES, RATE, wav } from "./tones.ts";

export { CHOICES, RATE, TONES } from "./tones.ts";

/** What a browser's speech offers, as far as a cue uses it. */
interface SpeechLike {
  speak(utterance: unknown): void;
}

/** What a browser's audio offers, as far as a cue uses it. */
interface AudioLike {
  readonly state: string;
  readonly destination: unknown;
  resume(): Promise<void>;
  createBuffer(channels: number, length: number, rate: number): { getChannelData(channel: number): Float32Array };
  createBufferSource(): { buffer: unknown; connect(to: unknown): void; start(): void };
}

type Global = {
  speechSynthesis?: SpeechLike;
  SpeechSynthesisUtterance?: new (text: string) => { rate: number };
  AudioContext?: new () => AudioLike;
};

/**
 * How fast the browser speaks a cue, against its normal rate. The library
 * asks `spd-say` for -20 on its scale of -100 to 100, a fifth of the way
 * toward its slowest, and the browser's voice is slowed by the same fifth.
 */
const SPEECH_RATE = 0.8;

let context: AudioLike | null = null;

/** The one audio context every tone plays through, made on first use. Null where there is none. */
function audio(): AudioLike | null {
  if (context !== null) return context;
  const Ctx = (globalThis as Global).AudioContext;
  if (Ctx === undefined) return null;
  context = new Ctx();
  return context;
}

/**
 * The tone names this module plays, and whether this computer can play a
 * tone and speak text: `aplay` and `speech`, the names the library gives
 * them, so one page reads both.
 */
export function available(): { tones: string[]; aplay: boolean; speech: boolean } {
  const g = globalThis as Global;
  return {
    tones: [...CHOICES],
    aplay: g.AudioContext !== undefined,
    speech: g.speechSynthesis !== undefined && g.SpeechSynthesisUtterance !== undefined,
  };
}

/**
 * Fire a cue: `none` or nothing is silent, `speech` speaks `text` and is
 * silent when `text` is empty, any other name in `TONES` plays its tone, and
 * a name not in `TONES` is silent. Returns at once and never throws.
 */
export function play(tone: unknown, text = ""): void {
  if (!tone || tone === "none") return;
  try {
    const g = globalThis as Global;
    if (tone === "speech") {
      if (text && g.speechSynthesis !== undefined && g.SpeechSynthesisUtterance !== undefined) {
        const utterance = new g.SpeechSynthesisUtterance(text);
        utterance.rate = SPEECH_RATE;
        g.speechSynthesis.speak(utterance);
      }
      return;
    }
    const bytes = typeof tone === "string" ? wav(tone) : null;
    const ctx = bytes === null ? null : audio();
    if (bytes === null || ctx === null) return;
    const n = (bytes.length - 44) / 2;
    const buffer = ctx.createBuffer(1, n, RATE);
    const samples = buffer.getChannelData(0);
    const v = new DataView(bytes.buffer, bytes.byteOffset + 44, 2 * n);
    for (let i = 0; i < n; i++) samples[i] = v.getInt16(2 * i, true) / 32768;
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    source.start();
    // A browser holds audio back until a person has acted on the page, and
    // a run is started by one, so this only ever wakes a context it paused.
    if (ctx.state === "suspended") ctx.resume().catch(() => undefined);
  } catch {
    // A missing sound device must never abort a recording.
  }
}
