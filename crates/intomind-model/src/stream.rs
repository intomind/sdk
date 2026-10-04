//! Feeding the model its own copy of the stream.
//!
//! The raw stream is never resampled, so the model keeps a private copy on
//! its native grid. Samples arrive one at a time from acquisition, and
//! this holds the most recent window of them.
//!
//! Only the rates the instrument offers are handled, each by the obvious
//! exact ratio, because a general resampler would cost more than it is
//! worth for three fixed cases. Going down averages the pair rather than
//! dropping one of them, which is a two tap filter and keeps the fold from
//! the very top of the band out of the answer.

/// How the device's rate maps onto the model's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ratio {
    /// The device is already on the native grid.
    Same,
    /// Two device samples make one model sample.
    Halve,
    /// One device sample makes two model samples.
    Double,
}

impl Ratio {
    /// The ratio for a device rate against the model's native rate, or
    /// `None` for a rate the model cannot be fed.
    pub fn of(device_sps: u32, native_sps: u32) -> Option<Ratio> {
        if device_sps == native_sps {
            Some(Ratio::Same)
        } else if device_sps == 2 * native_sps {
            Some(Ratio::Halve)
        } else if 2 * device_sps == native_sps {
            Some(Ratio::Double)
        } else {
            None
        }
    }
}

/// The most recent `CAP` samples per channel, on the model's grid, and a
/// count of every sample since the reset, so a window is named by the index
/// of its first sample and copied out by index.
///
/// `CH` channels, windows of `N` samples, a ring of `CAP` samples. The ring
/// is the window plus the slack that covers the time a caller takes to copy
/// a window out, channel by channel with yields between, while samples keep
/// arriving; nothing is ever dropped on the way in, and the window a caller
/// names stays intact for `CAP - N` samples after it completes.
pub struct Feeder<const CH: usize, const N: usize, const CAP: usize> {
    buf: [[f32; CAP]; CH],
    /// Samples written since the last reset. The next goes at `count % CAP`.
    count: u64,
    ratio: Ratio,
    held: [f32; CH],
    have_held: bool,
}

impl<const CH: usize, const N: usize, const CAP: usize> Default for Feeder<CH, N, CAP> {
    fn default() -> Self {
        Self::new(Ratio::Same)
    }
}

impl<const CH: usize, const N: usize, const CAP: usize> Feeder<CH, N, CAP> {
    const RING_HOLDS_A_WINDOW: () = assert!(CAP >= N && N > 0);

    /// An empty feeder at this device-to-model rate ratio.
    pub const fn new(ratio: Ratio) -> Self {
        let () = Self::RING_HOLDS_A_WINDOW;
        Feeder { buf: [[0.0; CAP]; CH], count: 0, ratio, held: [0.0; CH], have_held: false }
    }

    /// Change the rate the device is running at. Whatever was gathered
    /// under the old rate is discarded, because a window that spans a rate
    /// change is not a window of anything.
    pub fn set_ratio(&mut self, ratio: Ratio) {
        if ratio != self.ratio {
            self.ratio = ratio;
            self.reset();
        }
    }

    /// Discard everything gathered and start counting from zero again.
    pub fn reset(&mut self) {
        self.count = 0;
        self.have_held = false;
    }

    /// Samples taken since the reset, on the model's grid.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Whether a whole window has been gathered.
    pub fn ready(&self) -> bool {
        self.count >= N as u64
    }

    /// Samples held, up to a whole window.
    pub fn len(&self) -> usize {
        self.count.min(N as u64) as usize
    }

    /// Whether no samples have been taken since the reset.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The first sample of the newest complete window, once there is one.
    pub fn newest_start(&self) -> Option<u64> {
        self.ready().then(|| self.count - N as u64)
    }

    /// Whether the window starting at `start` is complete and still whole
    /// in the ring.
    pub fn holds(&self, start: u64) -> bool {
        start + N as u64 <= self.count && self.count - start <= CAP as u64
    }

    /// The window starting at `start`, to hand to the encoder, if the ring
    /// holds it.
    pub fn at(&self, start: u64) -> Option<WindowAt<'_, CH, N, CAP>> {
        self.holds(start).then_some(WindowAt { feeder: self, start })
    }

    fn write(&mut self, sample: &[f32; CH]) {
        let at = (self.count % CAP as u64) as usize;
        for c in 0..CH {
            self.buf[c][at] = sample[c];
        }
        self.count += 1;
    }

    /// Take one acquired sample, in microvolts.
    pub fn push(&mut self, sample: &[f32; CH]) {
        match self.ratio {
            Ratio::Same => self.write(sample),
            Ratio::Halve => {
                if self.have_held {
                    let mut mean = [0.0f32; CH];
                    for c in 0..CH {
                        mean[c] = 0.5 * (self.held[c] + sample[c]);
                    }
                    self.write(&mean);
                    self.have_held = false;
                } else {
                    self.held = *sample;
                    self.have_held = true;
                }
            }
            Ratio::Double => {
                if self.have_held {
                    let mut mid = [0.0f32; CH];
                    for c in 0..CH {
                        mid[c] = 0.5 * (self.held[c] + sample[c]);
                    }
                    self.write(&mid);
                } else {
                    // Nothing to interpolate from yet, so the first sample
                    // stands alone rather than being invented twice.
                    self.have_held = true;
                }
                self.write(sample);
                self.held = *sample;
            }
        }
    }

    /// Copy one channel of the window starting at `start` out, oldest
    /// sample first. The caller has checked `holds(start)`; two copies at
    /// most, where the ring wraps.
    pub fn read_channel_at(&self, c: usize, start: u64, out: &mut [f32]) {
        let n = N.min(out.len());
        let from = (start % CAP as u64) as usize;
        let first = (CAP - from).min(n);
        out[..first].copy_from_slice(&self.buf[c][from..from + first]);
        out[first..n].copy_from_slice(&self.buf[c][..n - first]);
    }

    /// Read one channel of the newest window out, oldest sample first.
    fn channel_into(&self, c: usize, out: &mut [f32]) {
        let start = self.count.saturating_sub(N as u64);
        self.read_channel_at(c, start, out);
    }

    /// Copy the newest window out, channel major, oldest sample first.
    /// Returns false when a whole window has not been gathered yet.
    pub fn window_into(&self, out: &mut [f32]) -> bool {
        if !self.ready() || out.len() < CH * N {
            return false;
        }
        for c in 0..CH {
            self.channel_into(c, &mut out[c * N..(c + 1) * N]);
        }
        true
    }
}

/// The newest window, for hosts and tests that only ever want that.
impl<const CH: usize, const N: usize, const CAP: usize> crate::blob::Window for Feeder<CH, N, CAP> {
    fn channels(&self) -> usize {
        CH
    }
    fn read_channel(&self, channel: usize, out: &mut [f32]) {
        self.channel_into(channel, out);
    }
}

/// One named window in the ring, the form the device hands to a pass.
pub struct WindowAt<'a, const CH: usize, const N: usize, const CAP: usize> {
    feeder: &'a Feeder<CH, N, CAP>,
    start: u64,
}

impl<const CH: usize, const N: usize, const CAP: usize> crate::blob::Window for WindowAt<'_, CH, N, CAP> {
    fn channels(&self) -> usize {
        CH
    }
    fn read_channel(&self, channel: usize, out: &mut [f32]) {
        self.feeder.read_channel_at(channel, self.start, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rates_the_instrument_offers_all_map() {
        assert_eq!(Ratio::of(500, 500), Some(Ratio::Same));
        assert_eq!(Ratio::of(1000, 500), Some(Ratio::Halve));
        assert_eq!(Ratio::of(250, 500), Some(Ratio::Double));
        assert_eq!(Ratio::of(333, 500), None);
        assert_eq!(Ratio::of(0, 500), None);
    }

    #[test]
    fn a_window_reads_out_oldest_first_and_wraps() {
        let mut f: Feeder<2, 4, 4> = Feeder::new(Ratio::Same);
        let mut out = [0.0f32; 8];
        assert!(!f.window_into(&mut out), "an unfilled window is not a window");
        for i in 0..4 {
            f.push(&[i as f32, 10.0 + i as f32]);
        }
        assert!(f.ready());
        assert!(f.window_into(&mut out));
        assert_eq!(out, [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0]);
        // Two more push the oldest out.
        f.push(&[4.0, 14.0]);
        f.push(&[5.0, 15.0]);
        f.window_into(&mut out);
        assert_eq!(out, [2.0, 3.0, 4.0, 5.0, 12.0, 13.0, 14.0, 15.0]);
    }

    #[test]
    fn halving_averages_the_pair_rather_than_dropping_one() {
        let mut f: Feeder<1, 4, 4> = Feeder::new(Ratio::Halve);
        for i in 0..8 {
            f.push(&[i as f32]);
        }
        let mut out = [0.0f32; 4];
        assert!(f.window_into(&mut out));
        assert_eq!(out, [0.5, 2.5, 4.5, 6.5]);
        assert_eq!(f.len(), 4);
    }

    #[test]
    fn doubling_puts_a_midpoint_between_each_pair() {
        let mut f: Feeder<1, 8, 8> = Feeder::new(Ratio::Double);
        for i in 0..5 {
            f.push(&[(i * 2) as f32]);
        }
        // The first sample stands alone, then each later one is preceded
        // by the midpoint of it and its predecessor: nine values out of
        // five in, of which the window keeps the last eight.
        assert_eq!(f.len(), 8);
        let mut out = [0.0f32; 8];
        assert!(f.window_into(&mut out));
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn a_rate_change_throws_away_the_window_it_spans() {
        let mut f: Feeder<1, 4, 4> = Feeder::new(Ratio::Same);
        for i in 0..4 {
            f.push(&[i as f32]);
        }
        assert!(f.ready());
        f.set_ratio(Ratio::Halve);
        assert!(!f.ready(), "a window that spans a rate change is not a window of anything");
        f.set_ratio(Ratio::Halve);
        assert!(!f.ready(), "setting the same ratio again is not a reason to discard anything");
    }

    #[test]
    fn a_window_is_named_by_its_first_sample_and_survives_the_slack() {
        use crate::blob::Window;
        let mut f: Feeder<1, 4, 6> = Feeder::new(Ratio::Same);
        for i in 0..8 {
            f.push(&[i as f32]);
        }
        assert_eq!(f.count(), 8);
        assert_eq!(f.newest_start(), Some(4));
        // Complete and still whole: starts 2, 3, 4. Start 1 has been
        // written over; start 5 is not complete yet.
        assert!(!f.holds(1) && f.holds(2) && f.holds(3) && f.holds(4) && !f.holds(5));
        let mut out = [0.0f32; 4];
        f.at(2).unwrap().read_channel(0, &mut out);
        assert_eq!(out, [2.0, 3.0, 4.0, 5.0], "the copy crosses the ring's wrap");
        f.at(4).unwrap().read_channel(0, &mut out);
        assert_eq!(out, [4.0, 5.0, 6.0, 7.0]);
        assert!(f.at(1).is_none() && f.at(5).is_none());
        // Two more samples and the window at 2 is gone, the one at 4 stays.
        f.push(&[8.0]);
        f.push(&[9.0]);
        assert!(!f.holds(2) && f.holds(4) && f.holds(6));
        // A reset starts the count over.
        f.reset();
        assert!(f.is_empty() && f.newest_start().is_none());
    }
}
