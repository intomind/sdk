/**
 * Python's `random.Random`, as far as the experiments use it: seeded with an
 * integer, `getrandbits`, and `shuffle`, giving the same numbers in the same
 * order as Python does.
 *
 * The generator is the Mersenne Twister, MT19937, seeded as CPython seeds it
 * from an integer: the absolute value split into 32-bit words, the lowest
 * first, through `init_by_array`. `shuffle` is Python's own loop over
 * `_randbelow`, which draws as many bits as the bound has and draws again
 * while the draw is too large. A cue order shuffled here with a seed is the
 * order the Python library shuffles with that seed, which matters because
 * the order is part of a recording.
 */

const N = 624;
const M = 397;
const MATRIX_A = 0x9908b0df;
const UPPER_MASK = 0x80000000;
const LOWER_MASK = 0x7fffffff;

/** A Mersenne Twister in the state CPython's `random.Random(seed)` starts in. */
export class Random {
  readonly #mt = new Uint32Array(N);
  #index = N;

  /** Seeded with an integer as Python seeds one: every bit of its absolute value is used. */
  constructor(seed: number | bigint) {
    this.seed(seed);
  }

  /** Seed again, as `Random.seed(a)` does for an integer `a`. */
  seed(a: number | bigint): void {
    if (typeof a === "number" && !Number.isSafeInteger(a)) {
      throw new TypeError(`a seed here is an integer, not ${a}`);
    }
    let n = BigInt(a);
    if (n < 0n) n = -n;
    const key: number[] = [];
    for (; n > 0n; n >>= 32n) key.push(Number(n & 0xffffffffn));
    if (key.length === 0) key.push(0);
    this.#initByArray(key);
  }

  #initGenrand(s: number): void {
    const mt = this.#mt;
    mt[0] = s >>> 0;
    for (let i = 1; i < N; i++) {
      const prev = mt[i - 1]! ^ (mt[i - 1]! >>> 30);
      mt[i] = Math.imul(1812433253, prev) + i;
    }
    this.#index = N;
  }

  #initByArray(key: readonly number[]): void {
    const mt = this.#mt;
    this.#initGenrand(19650218);
    let i = 1;
    let j = 0;
    for (let k = Math.max(N, key.length); k > 0; k--) {
      const prev = mt[i - 1]! ^ (mt[i - 1]! >>> 30);
      mt[i] = (mt[i]! ^ Math.imul(prev, 1664525)) + key[j]! + j;
      i++;
      j++;
      if (i >= N) {
        mt[0] = mt[N - 1]!;
        i = 1;
      }
      if (j >= key.length) j = 0;
    }
    for (let k = N - 1; k > 0; k--) {
      const prev = mt[i - 1]! ^ (mt[i - 1]! >>> 30);
      mt[i] = (mt[i]! ^ Math.imul(prev, 1566083941)) - i;
      i++;
      if (i >= N) {
        mt[0] = mt[N - 1]!;
        i = 1;
      }
    }
    // The most significant bit set, so the state is never all zeros.
    mt[0] = 0x80000000;
  }

  /** The next 32 bits of the generator, as CPython's `genrand_uint32` takes them. */
  #next(): number {
    const mt = this.#mt;
    if (this.#index >= N) {
      let kk = 0;
      for (; kk < N - M; kk++) {
        const y = (mt[kk]! & UPPER_MASK) | (mt[kk + 1]! & LOWER_MASK);
        mt[kk] = mt[kk + M]! ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0);
      }
      for (; kk < N - 1; kk++) {
        const y = (mt[kk]! & UPPER_MASK) | (mt[kk + 1]! & LOWER_MASK);
        mt[kk] = mt[kk + (M - N)]! ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0);
      }
      const y = (mt[N - 1]! & UPPER_MASK) | (mt[0]! & LOWER_MASK);
      mt[N - 1] = mt[M - 1]! ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0);
      this.#index = 0;
    }
    let y = mt[this.#index++]!;
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  /** `getrandbits(k)` for 0 to 32 bits: a draw below 2 to the `k`, from the top bits of one word. */
  getrandbits(k: number): number {
    if (!Number.isInteger(k) || k < 0) throw new RangeError("number of bits must be non-negative");
    if (k > 32) throw new RangeError("this port draws at most 32 bits at once");
    if (k === 0) return 0;
    return this.#next() >>> (32 - k);
  }

  /** `_randbelow(n)`: a draw below `n`, from as many bits as `n` has, drawn again while too large. */
  randbelow(n: number): number {
    if (!Number.isInteger(n) || n <= 0 || n > 0xffffffff) throw new RangeError(`a bound here is a positive 32-bit integer, not ${n}`);
    const k = 32 - Math.clz32(n);
    let r = this.getrandbits(k);
    while (r >= n) r = this.getrandbits(k);
    return r;
  }

  /** Shuffle `x` in place, as `random.Random.shuffle` does. */
  shuffle<T>(x: T[]): void {
    for (let i = x.length - 1; i > 0; i--) {
      // Pick an element in x[0..i] to exchange with x[i].
      const j = this.randbelow(i + 1);
      const t = x[i]!;
      x[i] = x[j]!;
      x[j] = t;
    }
  }
}
