//! The example the documentation shows, compiled.
//!
//! It is here rather than only in prose so that it is built against the
//! crate it documents. `cargo test` builds every example, so a rename
//! breaks the build rather than somebody's first hour with the crate.
//!
//! The two functions standing in for a Bluetooth stack are the point of
//! the design: this crate does no input and no output, so the example
//! compiles without one.

use intomind::protocol::device_info::capability;
use intomind::{protocol::uuid_fill, Event, Exchange, Session, Timebase};

/// Whatever Bluetooth stack you already have, reading a characteristic.
fn read(_characteristic: u16) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    unimplemented!("your stack")
}

/// And writing one.
fn write(_characteristic: u16, _bytes: &[u8], _with_response: bool)
    -> Result<(), Box<dyn std::error::Error>> {
    unimplemented!("your stack")
}

fn documented(notification: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let mut session = Session::new();
    session.on_device_info(&read(uuid_fill::DEVICE_INFO)?)?;

    let command = session.start_stream();
    write(command.characteristic, &command.bytes, command.with_response)?;

    for event in session.on_notification(uuid_fill::EEG_DATA, notification) {
        match event {
            Event::Samples(batch) => {
                let info = *session.info()?;
                for row in 0..batch.rows() {
                    let _uv = batch.microvolts(row, 0, &info);
                }
            }
            Event::Gap(gap) => {
                // Written down, never smoothed over. `samples_lost` is None
                // for a re-base, whose extent is not a number.
                let _ = gap.samples_lost;
            }
            _ => {}
        }
    }

    if session.can(capability::MODEL) { /* ... */ }
    let _ = session.set_leadoff(true)?; // refused unless the device claims it

    let info = *session.info()?;
    let mut tb = Timebase::new(info.time_tick_hz);
    tb.observe(Exchange { before: 0.0, after: 0.0, device_ticks: 0 });
    let _ = tb.host_time(0);
    Ok(())
}

fn main() {
    // Nothing here runs: the two stack functions are unimplemented. What
    // matters is that it compiles against the crate as it actually is.
    let _ = documented;
}
