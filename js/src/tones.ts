/**
 * The cue tones themselves, as the Python library's `audio` module makes
 * them: the tone table, and each tone's WAV file, sample for sample what the
 * library writes before it plays one. Internal, as the library's `_wav` is:
 * `audio` publishes the table and plays the tones, and the package's tests
 * hold these bytes to the library's files.
 */

/** The tones' sample rate, in samples a second. */
export const RATE = 44100;

/** Each tone as its segments, frequency in hertz and length in seconds. A frequency of zero is a pause, and no segment at all is silence. `speech` is spoken text instead. */
export const TONES: Readonly<Record<string, readonly (readonly [number, number])[]>> = {
  none: [],
  speech: [],
  beep_low: [[440.0, 0.18]],
  beep_high: [[880.0, 0.18]],
  chime_up: [
    [660.0, 0.12],
    [990.0, 0.16],
  ],
  chime_down: [
    [990.0, 0.12],
    [660.0, 0.16],
  ],
  click: [[1500.0, 0.035]],
  double: [
    [880.0, 0.09],
    [0.0, 0.06],
    [880.0, 0.09],
  ],
  long_low: [[330.0, 0.5]],
};

/** Every cue a cue point may be set to, in the library's order. */
export const CHOICES: readonly string[] = Object.keys(TONES);

const made = new Map<string, Uint8Array>();

/**
 * The tone `name` as the WAV file the library writes for it: 16-bit mono at
 * 44.1 kHz, each segment under the library's envelope, which starts and
 * ends a segment in silence and falls back to silence a quarter, a half and
 * three quarters of the way into it, so a segment sounds as four short
 * swells. Null for silence, speech, or a name
 * that is not a tone.
 */
export function wav(name: string): Uint8Array | null {
  const segs = Object.hasOwn(TONES, name) ? TONES[name]! : [];
  if (segs.length === 0) return null;
  const cached = made.get(name);
  if (cached !== undefined) return cached.slice();
  const frames: number[] = [];
  for (const [freq, dur] of segs) {
    const n = Math.trunc(RATE * dur);
    for (let i = 0; i < n; i++) {
      let s = 0.0;
      if (freq > 0) {
        const env = Math.min(1.0, 0.5 * (1 - Math.cos((2 * Math.PI * Math.min(i, n - i)) / Math.max(1, n * 0.25))));
        s = env * 0.6 * Math.sin((2 * Math.PI * freq * i) / RATE);
      }
      frames.push(Math.trunc(s * 32767));
    }
  }
  const out = new Uint8Array(44 + 2 * frames.length);
  const v = new DataView(out.buffer);
  const ascii = (at: number, s: string): void => {
    for (let k = 0; k < s.length; k++) out[at + k] = s.charCodeAt(k);
  };
  ascii(0, "RIFF");
  v.setUint32(4, 36 + 2 * frames.length, true);
  ascii(8, "WAVEfmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, RATE, true);
  v.setUint32(28, RATE * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  ascii(36, "data");
  v.setUint32(40, 2 * frames.length, true);
  frames.forEach((f, k) => v.setInt16(44 + 2 * k, f, true));
  made.set(name, out);
  return out.slice();
}
