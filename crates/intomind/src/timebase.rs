//! Device time onto a host clock.
//!
//! A device's clock is monotonic and free running. A host's is a
//! different crystal. The fit between them is a line, and the point of
//! taking several exchanges rather than one is that a single exchange
//! cannot tell a clock difference from a slow answer.
//!
//! The fit is refused rather than believed when it is absurd. Two
//! crystals do not differ by a thousand parts per million, and a
//! measurement that says they do is measuring something else, usually a
//! link that stalled in the middle of an exchange.

/// Beyond this, a skew is not two crystals disagreeing.
pub const MAX_SKEW_PPM: f64 = 1000.0;
/// And it has to be measured rather than drawn through noise.
pub const MAX_SKEW_STDERR_PPM: f64 = 100.0;

/// One time exchange, read on both clocks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exchange {
    /// Host time before the request, in seconds.
    pub before: f64,
    /// Host time after the answer.
    pub after: f64,
    /// The device's own time, in its ticks, captured when it read the request.
    pub device_ticks: u64,
}

impl Exchange {
    /// The middle of the window the device's answer must lie in.
    pub fn host(&self) -> f64 {
        (self.before + self.after) / 2.0
    }

    /// How wide that window is. Half the round trip, which is the
    /// uncertainty in this one exchange.
    pub fn uncertainty(&self) -> f64 {
        (self.after - self.before) / 2.0
    }
}

/// The line through the exchanges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fit {
    /// host = device_seconds * (1 + skew) + offset.
    pub offset_s: f64,
    /// Clock skew, in parts per million. Zero when `skew_used` is false.
    pub skew_ppm: f64,
    /// Exchanges this fit was computed from.
    pub exchanges: usize,
    /// How far the exchanges sit from the line, in microseconds.
    pub residual_us: f64,
    /// Whether the skew was measured well enough to be used.
    pub skew_used: bool,
}

/// Device time onto host time, and the exchanges it was learned from.
#[derive(Debug, Clone, Default)]
pub struct Timebase {
    exchanges: Vec<Exchange>,
    tick_hz: f64,
    fit: Option<Fit>,
}

impl Timebase {
    /// `tick_hz` is the device's own, from Device Info.
    pub fn new(tick_hz: u32) -> Timebase {
        Timebase { exchanges: Vec::new(), tick_hz: tick_hz as f64, fit: None }
    }

    /// Take one exchange. The fit is redone from everything so far.
    pub fn observe(&mut self, exchange: Exchange) -> Option<Fit> {
        self.exchanges.push(exchange);
        self.refit();
        self.fit
    }

    /// Every exchange taken so far.
    pub fn exchanges(&self) -> &[Exchange] {
        &self.exchanges
    }

    /// The current fit, once an exchange has been taken.
    pub fn fit(&self) -> Option<Fit> {
        self.fit
    }

    /// Host time for a device time, in seconds. None until there is a fit.
    pub fn host_time(&self, device_ticks: u64) -> Option<f64> {
        let f = self.fit?;
        let d = device_ticks as f64 / self.tick_hz;
        Some(d * (1.0 + f.skew_ppm * 1e-6) + f.offset_s)
    }

    fn refit(&mut self) {
        let n = self.exchanges.len();
        if n == 0 {
            self.fit = None;
            return;
        }
        let xs: Vec<f64> = self.exchanges.iter().map(|e| e.device_ticks as f64 / self.tick_hz).collect();
        let ys: Vec<f64> = self.exchanges.iter().map(|e| e.host()).collect();
        if n == 1 {
            // One exchange fixes an offset and says nothing about rate.
            // Claiming a skew from it would be claiming a measurement that
            // was never made.
            self.fit = Some(Fit { offset_s: ys[0] - xs[0], skew_ppm: 0.0, exchanges: 1,
                                  residual_us: self.exchanges[0].uncertainty() * 1e6, skew_used: false });
            return;
        }
        let mean_x = xs.iter().sum::<f64>() / n as f64;
        let mean_y = ys.iter().sum::<f64>() / n as f64;
        let sxx: f64 = xs.iter().map(|x| (x - mean_x).powi(2)).sum();
        let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - mean_x) * (y - mean_y)).sum();
        let slope = if sxx > 0.0 { sxy / sxx } else { 1.0 };
        let intercept = mean_y - slope * mean_x;
        let residuals: Vec<f64> = xs.iter().zip(&ys).map(|(x, y)| y - (slope * x + intercept)).collect();
        let rss: f64 = residuals.iter().map(|r| r * r).sum();
        let residual_us = (rss / n as f64).sqrt() * 1e6;
        let skew_ppm = (slope - 1.0) * 1e6;
        // The standard error of the slope, in the same units.
        let stderr_ppm = if n > 2 && sxx > 0.0 {
            ((rss / (n as f64 - 2.0)) / sxx).sqrt() * 1e6
        } else {
            f64::INFINITY
        };
        let believable = skew_ppm.abs() <= MAX_SKEW_PPM && stderr_ppm <= MAX_SKEW_STDERR_PPM;
        self.fit = Some(if believable {
            Fit { offset_s: intercept, skew_ppm, exchanges: n, residual_us, skew_used: true }
        } else {
            // Keep the offset, which is measured well, and drop the rate,
            // which is not. A wrong rate is worse than no rate: it grows.
            Fit { offset_s: mean_y - mean_x, skew_ppm: 0.0, exchanges: n, residual_us, skew_used: false }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exchanges(skew_ppm: f64, offset: f64, n: usize, round_trip: f64, tick_hz: f64) -> Vec<Exchange> {
        (0..n)
            .map(|i| {
                let device_s = i as f64 * 10.0;
                let host = device_s * (1.0 + skew_ppm * 1e-6) + offset;
                Exchange { before: host - round_trip / 2.0, after: host + round_trip / 2.0,
                           device_ticks: (device_s * tick_hz) as u64 }
            })
            .collect()
    }

    #[test]
    fn one_exchange_fixes_an_offset_and_claims_no_rate() {
        let mut tb = Timebase::new(1_000_000);
        let f = tb.observe(exchanges(20.0, 1000.0, 1, 0.004, 1e6)[0]).unwrap();
        assert!(!f.skew_used, "a rate cannot be measured from one point");
        assert_eq!(f.skew_ppm, 0.0);
        assert!((f.offset_s - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn several_exchanges_find_the_rate() {
        let mut tb = Timebase::new(1_000_000);
        for e in exchanges(20.0, 1000.0, 8, 0.004, 1e6) {
            tb.observe(e);
        }
        let f = tb.fit().unwrap();
        assert!(f.skew_used);
        assert!((f.skew_ppm - 20.0).abs() < 0.5, "{}", f.skew_ppm);
        assert!((f.offset_s - 1000.0).abs() < 1e-3);
        // An hour of device time lands where the rate says it does.
        let at_hour = tb.host_time(3_600 * 1_000_000).unwrap();
        assert!((at_hour - (1000.0 + 3600.0 * 1.000_02)).abs() < 0.01, "{at_hour}");
    }

    #[test]
    fn a_skew_that_two_crystals_cannot_have_is_refused() {
        let mut tb = Timebase::new(1_000_000);
        for e in exchanges(5_000.0, 0.0, 8, 0.004, 1e6) {
            tb.observe(e);
        }
        let f = tb.fit().unwrap();
        assert!(!f.skew_used, "five thousand parts per million is not two crystals");
        assert_eq!(f.skew_ppm, 0.0, "the offset is kept and the rate is dropped");
    }

    #[test]
    fn a_rate_drawn_through_noise_is_refused() {
        // One exchange whose answer came back late drags a line through
        // nothing. The fit says so rather than growing that error.
        let mut tb = Timebase::new(1_000_000);
        let mut es = exchanges(20.0, 1000.0, 5, 0.004, 1e6);
        es[2].before += 0.5;
        es[2].after += 0.5;
        for e in es {
            tb.observe(e);
        }
        let f = tb.fit().unwrap();
        assert!(!f.skew_used, "a fit this noisy is not a measurement of rate");
        assert!(f.residual_us > 1000.0);
    }

    #[test]
    fn nothing_is_mapped_before_anything_is_measured() {
        let tb = Timebase::new(1_000_000);
        assert!(tb.fit().is_none());
        assert!(tb.host_time(0).is_none());
    }
}
