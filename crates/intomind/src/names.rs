//! Telling devices apart in a list a person picks from.

use std::collections::HashMap;

/// Labels for a list a person picks devices from, in the list's order.
///
/// Each device's own advertised name, and when two in the list share one,
/// a number after the later ones. The number is the host's and lasts as
/// long as the list. The device knows nothing of it, since two devices
/// cannot agree on which is second without a host that sees them both.
/// Pass a device heard before its name arrived as whatever the list shows
/// for it, its product name for instance.
///
/// ```
/// let labels = intomind::display_names(&["Ada's IntoMind One", "Blue IntoMind One", "Ada's IntoMind One"]);
/// assert_eq!(labels, ["Ada's IntoMind One", "Blue IntoMind One", "Ada's IntoMind One 2"]);
/// ```
pub fn display_names<S: AsRef<str>>(names: &[S]) -> Vec<String> {
    let mut seen: HashMap<&str, usize> = HashMap::new();
    names
        .iter()
        .map(|n| {
            let name = n.as_ref();
            let count = seen.entry(name).or_insert(0);
            *count += 1;
            if *count == 1 {
                name.to_string()
            } else {
                format!("{name} {count}")
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_later_of_a_shared_name_are_numbered() {
        let labels = display_names(&["Ada's IntoMind One", "IntoMind One", "Ada's IntoMind One", "IntoMind One", "Ada's IntoMind One"]);
        assert_eq!(labels, ["Ada's IntoMind One", "IntoMind One", "Ada's IntoMind One 2", "IntoMind One 2", "Ada's IntoMind One 3"]);
        assert!(display_names::<&str>(&[]).is_empty());
    }
}
