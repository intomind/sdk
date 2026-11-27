//! The published names of a recording and its exports keep the signatures
//! they were published with.
//!
//! This crate is published, and a published version stays downloadable, so
//! its surface sees additions and nothing else: a name is never narrowed,
//! changed or removed under a program that calls it. Each name here is held
//! to its signature as a program outside the crate sees it, so a change to
//! one fails to compile, and a reader of this file sees the surface whole.

use std::path::{Path, PathBuf};

use intomind_capture::export;
use intomind_capture::json::Value;
use intomind_capture::recording::{check_label, Collection, Error, Recording, Verification};

#[test]
fn every_published_name_keeps_its_signature() {
    // A recordings folder and what it reads.
    let _: fn(PathBuf) -> Collection = Collection::open;
    let _: fn() -> PathBuf = Collection::default_dir;
    let _: fn(&Collection) -> &Path = Collection::dir;
    let _: fn(&Collection, &str, &str) -> Result<PathBuf, Error> = Collection::path;
    let _: fn(&Collection, &str) -> Result<Verification, Error> = Collection::verify;
    let _: fn(&Collection, &str) -> Result<Recording, Error> = Collection::load;
    let _: fn(&Collection, &str) -> Result<Recording, Error> = Collection::load_as_recorded;
    let _: fn(&str) -> Result<&str, Error> = check_label;

    // One recording, as published in 1.4.
    let _: fn(&Recording) -> &str = Recording::label;
    let _: fn(&Recording) -> usize = Recording::channels;
    let _: fn(&Recording) -> usize = Recording::samples;
    let _: fn(&Recording, usize) -> Vec<i64> = Recording::counts;
    let _: fn(&Recording) -> Vec<i64> = Recording::index;
    let _: fn(&Recording) -> Vec<f64> = Recording::host_time;
    let _: fn(&Recording) -> Option<Vec<i64>> = Recording::device_ticks;
    let _: fn(&Recording) -> Option<Vec<u8>> = Recording::leadoff;
    let _: fn(&Recording) -> Option<Vec<bool>> = Recording::gap_before;
    let _: fn(&Recording) -> Option<Vec<bool>> = Recording::leadoff_active;
    let _: fn(&Recording) -> &str = Recording::manifest_json;
    let _: fn(&Recording) -> &[String] = Recording::repairs;
    let _: fn(&Recording) -> bool = Recording::synthetic;
    let _: fn(&Recording) -> f64 = Recording::fs;
    let _: fn(&Recording) -> Result<f64, Error> = Recording::lsb_uv;
    let _: fn(&Recording, usize) -> Result<Vec<f64>, Error> = Recording::uv;

    // The exports.
    let _: fn(&Collection, &str, &str, Option<&Path>) -> Result<Value, Error> = export::export;
    let _: [&str; 6] = export::FORMATS;
}
