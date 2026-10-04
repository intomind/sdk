//! Nothing in this repository names what it must not name.
//!
//! These packages are published. A part number, an unreleased project, a
//! person, or somebody's directory layout, left in a comment, goes out
//! with the crate and cannot be taken back: a published version stays
//! downloadable forever, and yanking it does not remove the code.
//!
//! The file list comes from git rather than from a list written here,
//! because a list written here goes stale the first time somebody adds a
//! file, and the check then passes over exactly the new work it was
//! meant to read.
//!
//! The converter (the ADS1299 family) and the radio module (an MDBT50Q,
//! which is an nRF52840) are named in the product's public datasheet and
//! may be named here (ruled 2026-09-26). No other part is named, or needs
//! to be, and `intomind-protocol` is copied from the firmware, so this
//! check reads every re-sync.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Patterns that give nothing away by being written down: an address, a
/// path on somebody's machine. A pattern, what it is, and a made-up
/// example it must catch.
const PUBLIC: &[(&str, &str, &str)] = &[
    (r"(?i)[\w.+-]+@gmail\.com", "a personal address", "write to someone@gmail.com"),
    (r"(?i)/home/[a-z]", "a path on somebody's machine", "/home/someone/project"),
];

/// The rest are themselves what they protect: unreleased devices and
/// projects, part numbers, register names, people, private repositories.
/// Written here they would be published by the very check that keeps them
/// out, so they live on the maintainers' machines, in the file
/// INTOMIND_PRIVATE_WORDS names or ~/.config/intomind/private-words.tsv:
/// one pattern per line, then what it is, "fold" or "exact", and an
/// example the check must catch, separated by tabs. Without the file the
/// check fails and says why: a check that quietly does not run is not a
/// check.
fn private_words() -> Vec<(String, String, String)> {
    let path = std::env::var_os("INTOMIND_PRIVATE_WORDS").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var_os("HOME").expect("HOME is set")).join(".config/intomind/private-words.tsv")
    });
    let text = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!(
            "the private word list is not on this machine ({}), so what this repository may publish cannot be checked. Set INTOMIND_PRIVATE_WORDS to it.",
            path.display()
        )
    });
    let mut out = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f.len(), 4, "a line of the private word list is not four fields: {line:?}");
        let pattern = if f[2] == "fold" { format!("(?i){}", f[0]) } else { f[0].to_string() };
        out.push((pattern, f[1].to_string(), f[3].to_string()));
    }
    assert!(!out.is_empty(), "{} holds no patterns", path.display());
    out
}

/// Every pattern, public and private.
fn banned() -> Vec<(String, String, String)> {
    let mut all: Vec<(String, String, String)> =
        PUBLIC.iter().map(|(p, w, e)| (p.to_string(), w.to_string(), e.to_string())).collect();
    all.extend(private_words());
    all
}

/// The wire field is spelled this way in the contract every
/// implementation is held to, so it is our word and not a borrowed one.
/// Renaming it would change the contract and hide nothing: the gain
/// ladder and the converter width say as much to a reader who is
/// looking. Listed so the choice is deliberate.
const ALLOWED: &[&str] = &["loff_statp"];

/// This file, which writes the public patterns' made-up examples and is
/// therefore held to the private words alone.
const THIS_FILE: &str = "crates/intomind/tests/published.rs";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn tracked() -> Vec<String> {
    let out = Command::new("git")
        .arg("ls-files")
        .current_dir(root())
        .output()
        .expect("git is on the path");
    assert!(out.status.success(), "git ls-files failed");
    String::from_utf8(out.stdout).unwrap().lines().map(str::to_owned).collect()
}

/// Every banned name in this text, with what it is.
fn scan(text: &str, patterns: &[(String, String, String)]) -> Vec<String> {
    let mut hits = Vec::new();
    for (pattern, why, _example) in patterns {
        let re = regex_lite::Regex::new(pattern).unwrap_or_else(|e| panic!("the pattern {pattern:?} does not compile: {e}"));
        for m in re.find_iter(text) {
            if ALLOWED.iter().any(|a| a.eq_ignore_ascii_case(m.as_str())) {
                continue;
            }
            hits.push(format!("names {:?}, which is {why}", m.as_str()));
        }
    }
    hits
}

#[test]
fn no_tracked_file_names_what_it_must_not() {
    let root = root();
    let files = tracked();
    assert!(files.len() > 20, "git listed almost nothing, so this read almost nothing");
    let (all, private) = (banned(), private_words());
    let mut problems: Vec<String> = Vec::new();
    let mut read = 0;
    for f in &files {
        let path: &Path = &root.join(f);
        let Ok(text) = std::fs::read_to_string(path) else { continue };
        read += 1;
        let patterns = if f == THIS_FILE { &private } else { &all };
        for hit in scan(&text, patterns) {
            let line = format!("{f}: {hit}");
            if !problems.contains(&line) {
                problems.push(line);
            }
        }
    }
    assert!(read > 20, "almost nothing was read, so this proves almost nothing");
    assert!(problems.is_empty(), "\n  {}", problems.join("\n  "));
}

#[test]
fn the_check_can_see_a_name() {
    // A gate never seen to fail is not a gate: every pattern catches its
    // own example.
    let all = banned();
    for (pattern, why, example) in &all {
        let hits = scan(example, &all);
        assert!(hits.iter().any(|h| h.contains(why.as_str())), "{example:?} went unseen by {pattern:?}: {hits:?}");
    }
}

#[test]
fn the_contracts_own_field_is_not_a_finding() {
    assert!(scan("the packet carries loff_statp, one bit per channel", &banned()).is_empty());
}

#[test]
fn this_file_names_nothing_it_protects() {
    // No file is exempt, this one included: the words it protects are read
    // from the private list, never written here.
    let text = std::fs::read_to_string(root().join(THIS_FILE)).unwrap();
    assert!(scan(&text, &private_words()).is_empty(), "this check publishes what it protects");
    assert!(tracked().iter().any(|f| f == THIS_FILE), "this file is not tracked, so the name above is stale");
}
