//! Where a device was put, held to the Python library's placement: every
//! position it offers, each key and what is under it, the refusals word for
//! word, and the placements it made with one position, with sites or with a
//! preset, each resolved, turned into a montage and described here as the
//! library did in `contract/placement`.

use std::path::Path;

use crate::montage::Fail;
use crate::placement;
use crate::pyjson::{self, Value};

fn fixture() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contract/placement/expected.json");
    pyjson::loads(&std::fs::read_to_string(path).expect("the placement fixture is in the repository")).expect("JSON")
}

fn cases<'a>(f: &'a Value, key: &str) -> &'a [Value] {
    match f.get(key) {
        Some(Value::Array(a)) => a,
        _ => panic!("no {key} in the fixture"),
    }
}

/// What the library returned or the words it raised, against what this did.
fn compare(name: &str, want: &Value, got: Result<Value, Fail>, problems: &mut Vec<String>) {
    let said = |e: &Fail| match e {
        Fail::Placement(m) => format!("PlacementError: {m}"),
        other => format!("{other:?}"),
    };
    match (got, want.get("result"), want.get("raises")) {
        (Ok(v), Some(w), _) if pyjson::dumps(&v) == pyjson::dumps(w) => {}
        (Ok(v), Some(w), _) => {
            problems.push(format!("{name}: {} differs from {}", pyjson::dumps(&v), pyjson::dumps(w)))
        }
        (Err(e), _, Some(w)) if said(&e) == w.py_str() => {}
        (Err(e), _, w) => problems.push(format!("{name}: raised {} where the library did {w:?}", said(&e))),
        (Ok(v), None, w) => problems.push(format!("{name}: {} where the library raised {w:?}", pyjson::dumps(&v))),
    }
}

#[test]
fn every_position_the_library_offers_reads_here_as_it_offers_it() {
    let f = fixture();
    let offered = cases(&f, "positions");
    assert_eq!(offered.len(), 78);
    for want in offered {
        let got = placement::scalp(want.field("key")).expect("a position on offer");
        assert_eq!(pyjson::dumps(&got), pyjson::dumps(want));
    }
}

#[test]
fn every_position_and_every_key_that_is_not_one_reads_as_the_library_reads_it() {
    let f = fixture();
    let mut problems = Vec::new();
    let scalp = cases(&f, "scalp");
    assert!(scalp.len() >= 110, "{}", scalp.len());
    for c in scalp {
        let key = c.field("key");
        compare(&format!("scalp({})", key.py_repr()), c, placement::scalp(key), &mut problems);
    }
    for c in cases(&f, "regions_under") {
        let key = c.field("key");
        let got = placement::regions_under(key).map(Value::Array);
        compare(&format!("regions_under({})", key.py_repr()), c, got, &mut problems);
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

#[test]
fn a_placement_the_library_made_is_resolved_and_described_as_the_library_does_it() {
    let f = fixture();
    let mut problems = Vec::new();
    let made: Vec<&Value> = cases(&f, "placements").iter().filter(|c| c.get("result").is_some()).collect();
    assert!(made.len() >= 11, "{}", made.len());
    for c in made {
        let name = c.field("name").py_str();
        let (lay, p) = (c.field("layout"), c.field("result"));
        let mut same = |what: &str, got: Result<Value, Fail>, want: &Value| match got {
            Ok(v) if pyjson::dumps(&v) == pyjson::dumps(want) => {}
            other => problems.push(format!("{name}, {what}: {other:?} differs from {}", pyjson::dumps(want))),
        };
        same("resolved", placement::resolve(lay, p), c.field("resolved"));
        same("montage", placement::to_montage(lay, p), c.field("montage"));
        same("described", placement::describe(lay, p, true).map(Value::Str), c.field("described"));
        same(
            "described without extras",
            placement::describe(lay, p, false).map(Value::Str),
            c.field("described_plain"),
        );
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}
