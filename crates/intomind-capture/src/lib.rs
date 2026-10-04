//! The capture format.
//!
//! A recording is a directory. The samples are one flat file of native
//! endian integers, the events are another, and a manifest in JSON says
//! what they are and how they were made. Nothing is compressed and
//! nothing is packed into a container, because the point of this format
//! is that somebody can read it in fifty years with a hex editor and the
//! manifest beside them.
//!
//! Three rules hold it together:
//!
//! 1. A gap is written down. The sample file has no silent splices. Every
//!    break in the timeline is a row in the gap list with the index range
//!    and the device time range it covers.
//! 2. The manifest is written last, with the checksums in it. A run that
//!    dies half way leaves a directory with no manifest, which is not a
//!    capture, rather than a capture that is quietly short.
//! 3. Nothing is inferred. A field whose value was not measured is absent,
//!    never guessed and never defaulted to something plausible.

use serde::{Deserialize, Serialize};

pub const MANIFEST: &str = "capture.json";
pub const SAMPLES: &str = "samples.i32";
pub const EVENTS: &str = "events.json";
pub const FORMAT: u32 = 1;

/// What a capture says about itself.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub format: u32,
    /// When the recording started, as the host saw it. RFC 3339.
    pub started: String,
    pub device: Device,
    pub signal: Signal,
    pub samples: Samples,
    /// Breaks in the timeline, in order. Empty means the recording is
    /// continuous, which is a claim this format lets a reader check.
    pub gaps: Vec<Gap>,
    pub clock: Clock,
    /// Who recorded it and with what. Absent fields were not declared,
    /// which is not the same as being unknown to the person who was there.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub site: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operator: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    pub software: Software,
    pub checksums: Checksums,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Device {
    /// As the device reported it, never as the host assumed it.
    pub device_id: String,
    pub name: String,
    pub protocol: String,
    pub firmware: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hardware: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Signal {
    pub channels: usize,
    pub sample_rate_hz: u32,
    pub gain: u8,
    /// Microvolts per count, so a reader never has to know the gain rule.
    pub microvolts_per_count: f64,
    /// 0 normal, 1 the internal test signal, 2 inputs shorted.
    pub mode: u8,
    pub leadoff: bool,
    /// Where the electrodes were, when that was declared.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub electrodes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Samples {
    /// Rows in the sample file. One row is `channels` little-endian
    /// signed 32 bit integers, in converter counts.
    pub count: u64,
    pub dtype: String,
    pub byte_order: String,
    /// The device's own sample index for the first row.
    pub first_index: u32,
    /// The device time of the first row, in the device's ticks.
    pub first_device_time: u64,
    pub device_tick_hz: u32,
}

/// A break in the timeline, written down rather than papered over.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gap {
    /// Row in the sample file the gap precedes.
    pub at_row: u64,
    /// Device sample index either side of it.
    pub last_index_before: u32,
    pub first_index_after: u32,
    /// Samples missing, when the contract could say. A re-base has no
    /// count, and the field is absent rather than zero, because zero
    /// missing samples is a different statement from an unknown extent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub samples_lost: Option<u32>,
    pub device_time_before: u64,
    pub device_time_after: u64,
    /// "loss" or "rebase".
    pub kind: String,
}

/// How device time was put on the host's clock.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Clock {
    /// host_time = device_time * (1 + skew) + offset, in seconds.
    pub offset_s: f64,
    pub skew_ppm: f64,
    /// How many exchanges the fit came from, and how well they agreed.
    pub syncs: usize,
    pub residual_us: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Software {
    pub name: String,
    pub version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
}

/// Written last, over the files as they finally are.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Checksums {
    pub algorithm: String,
    pub samples: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub events: Option<String>,
}

impl Manifest {
    /// The device time of a row, from the first row and the rate, walking
    /// the gaps. Exact, because the gaps are written down.
    pub fn device_time_of_row(&self, row: u64) -> Option<u64> {
        if row >= self.samples.count {
            return None;
        }
        let per_sample = self.samples.device_tick_hz as f64 / self.signal.sample_rate_hz as f64;
        let mut base_row = 0u64;
        let mut base_time = self.samples.first_device_time;
        for gap in &self.gaps {
            if gap.at_row > row {
                break;
            }
            base_time = gap.device_time_after;
            base_row = gap.at_row;
        }
        Some(base_time + ((row - base_row) as f64 * per_sample) as u64)
    }

    /// The host time of a row, in seconds, through the clock fit.
    pub fn host_time_of_row(&self, row: u64) -> Option<f64> {
        let t = self.device_time_of_row(row)? as f64 / self.samples.device_tick_hz as f64;
        Some(t * (1.0 + self.clock.skew_ppm * 1e-6) + self.clock.offset_s)
    }

    /// Whether the recording claims to be continuous.
    pub fn continuous(&self) -> bool {
        self.gaps.is_empty()
    }

    /// Samples the device says never arrived. A re-base contributes
    /// nothing, because its extent is not a number.
    pub fn samples_lost(&self) -> u32 {
        self.gaps.iter().filter_map(|g| g.samples_lost).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> Manifest {
        Manifest {
            format: FORMAT,
            started: "2026-09-19T12:00:00Z".into(),
            device: Device {
                device_id: "a0a1a2a3a4a5a6a7".into(),
                name: "IntoMind-1-A6A7".into(),
                protocol: "1.0".into(),
                firmware: "1.0.0".into(),
                hardware: Some("1.1.5".into()),
                build_id: Some("cd1ebee5118d7231".into()),
            },
            signal: Signal {
                channels: 4,
                sample_rate_hz: 500,
                gain: 24,
                microvolts_per_count: 0.022_351_741_790_771_484,
                mode: 0,
                leadoff: false,
                electrodes: vec![],
            },
            samples: Samples {
                count: 5_000,
                dtype: "int32".into(),
                byte_order: "little".into(),
                first_index: 0,
                first_device_time: 1_000_000,
                device_tick_hz: 1_000_000,
            },
            gaps: vec![],
            clock: Clock { offset_s: 1_758_000_000.0, skew_ppm: 12.5, syncs: 6, residual_us: 180.0 },
            site: None,
            operator: None,
            note: None,
            software: Software { name: "intomind".into(), version: "1.0.0".into(), revision: None },
            checksums: Checksums { algorithm: "sha256".into(), samples: "00".repeat(32), events: None },
        }
    }

    #[test]
    fn a_manifest_round_trips_through_json() {
        let m = manifest();
        let text = serde_json::to_string_pretty(&m).unwrap();
        assert_eq!(serde_json::from_str::<Manifest>(&text).unwrap(), m);
        // A field that was not declared is absent, not null and not a
        // plausible default.
        assert!(!text.contains("\"site\""), "{text}");
        assert!(!text.contains("\"electrodes\""));
    }

    #[test]
    fn row_times_follow_the_rate_when_there_are_no_gaps() {
        let m = manifest();
        assert!(m.continuous());
        assert_eq!(m.device_time_of_row(0), Some(1_000_000));
        assert_eq!(m.device_time_of_row(1), Some(1_002_000));
        assert_eq!(m.device_time_of_row(500), Some(2_000_000));
        assert_eq!(m.device_time_of_row(5_000), None, "a row past the end has no time");
        let t0 = m.host_time_of_row(0).unwrap();
        let t1 = m.host_time_of_row(500).unwrap();
        assert!((t1 - t0 - 1.000_012_5).abs() < 1e-6, "one second of device time, stretched by the fitted skew");
    }

    #[test]
    fn a_gap_moves_the_clock_and_is_never_smoothed_over() {
        let mut m = manifest();
        // Three hundred samples never arrived after row 1000.
        m.gaps.push(Gap {
            at_row: 1_000,
            last_index_before: 999,
            first_index_after: 1_300,
            samples_lost: Some(300),
            device_time_before: 1_000_000 + 999 * 2_000,
            device_time_after: 1_000_000 + 1_300 * 2_000,
            kind: "loss".into(),
        });
        assert!(!m.continuous());
        assert_eq!(m.samples_lost(), 300);
        // The row before the gap is where the rate says it is, and the row
        // after it is where the device said it was, not where counting
        // rows would have put it.
        assert_eq!(m.device_time_of_row(999), Some(1_000_000 + 999 * 2_000));
        assert_eq!(m.device_time_of_row(1_000), Some(1_000_000 + 1_300 * 2_000));
        assert_eq!(m.device_time_of_row(1_001), Some(1_000_000 + 1_301 * 2_000));

        // A re-base has no extent, and saying zero would be a different
        // claim from saying unknown.
        m.gaps.push(Gap {
            at_row: 2_000,
            last_index_before: 2_299,
            first_index_after: 0,
            samples_lost: None,
            device_time_before: 1_000_000 + 2_300 * 2_000,
            device_time_after: 9_000_000,
            kind: "rebase".into(),
        });
        assert_eq!(m.samples_lost(), 300);
        assert_eq!(m.device_time_of_row(2_000), Some(9_000_000));
        let text = serde_json::to_string(&m).unwrap();
        assert!(text.contains("\"samples_lost\":300"));
        assert_eq!(text.matches("samples_lost").count(), 1, "the unknown extent is absent, never zero");
    }
}
