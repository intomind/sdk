//! Every Rust example the documentation shows, compiled.
//!
//! The examples are here rather than only in prose so that each is built
//! against the crate it documents: `cargo test` builds every example, so a
//! rename breaks the build rather than somebody's first hour with the
//! crate. The release gate holds the documentation's blocks to this file
//! line for line, so what the page shows is what compiled.
//!
//! The stand-ins for a Bluetooth stack are the point of the design: this
//! crate does no input and no output, so every example compiles without one.
#![allow(dead_code, unused_variables, unused_imports, unused_mut, clippy::all)]

/// Connect and stream.
mod stream {
    use intomind::protocol::uuid_fill;
    use intomind::{Event, Session};

    fn stream(stack: &mut impl Stack) -> Result<(), Box<dyn std::error::Error>> {
        let mut session = Session::new();
        session.on_device_info(&stack.read(uuid_fill::DEVICE_INFO)?)?;
        let info = *session.info()?;

        let start = session.start_stream();
        stack.write(start.characteristic, &start.bytes, start.with_response)?;

        while let Some((characteristic, bytes)) = stack.next_notification() {
            for event in session.on_notification(characteristic, &bytes) {
                match event {
                    Event::Samples(batch) => {
                        for row in 0..batch.rows() {
                            let uv = batch.microvolts(row, 0, &info); // channel 1, in microvolts
                            println!("{} {uv:.2}", batch.index + row as u32);
                        }
                    }
                    // None when the device restarted its count.
                    Event::Gap(gap) => println!("gap {:?}", gap.samples_lost),
                    Event::Duplicate { .. } => {} // another program is listening: dropped
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// What your Bluetooth library provides.
    trait Stack {
        fn read(&mut self, fill: u16) -> Result<Vec<u8>, Box<dyn std::error::Error>>;
        fn write(&mut self, fill: u16, bytes: &[u8], with_response: bool) -> Result<(), Box<dyn std::error::Error>>;
        fn next_notification(&mut self) -> Option<(u16, Vec<u8>)>;
    }
}

/// Commands.
mod commands {
    use intomind::Session;

    fn commands(session: &Session) -> Result<(), Box<dyn std::error::Error>> {
        use intomind::protocol::control::{indicator, mode};
        use intomind::protocol::device_info::capability;

        if session.can(capability::LEADOFF) {
            let command = session.set_leadoff(true)?;
        }
        // Every batch then says whether detection was running, in `batch.leadoff_active`:
        // a lead-off byte of zero is an attached electrode only when it was.
        let verbose = session.set_indicator(indicator::VERBOSE)?;          // silent, reserved, verbose
        let identify = session.identify(5)?;                               // the status light winks for five seconds
        let name = session.set_name("Ada", "Blue", "IntoMind One")?;       // refused unless it fits 29 bytes
        let synthetic = session.set_mode(mode::SYNTHETIC)?;                // the generated signal
        Ok(())
    }
}

/// Processing on the device.
mod processing {
    use intomind::protocol::pipeline::{kind, Chain, Stage};
    use intomind::{Command, Error, Session};

    fn my_chain(session: &Session) -> Result<Command, Error> {
        let mut chain = Chain::NATURAL;                    // no stages
        chain.push(Stage::one(kind::HIGH_PASS, 5))?;       // 0.5 Hz, in tenths of a hertz
        chain.push(Stage::two(kind::NOTCH, 580, 620))?;    // 58 to 62 Hz
        chain.push(Stage::one(kind::LOW_PASS, 0))?;        // the automatic corner
        session.set_pipeline(&chain, 500)                  // refused, with the reason, if it cannot run at 500
    }
}

/// Embeddings.
mod embeddings {
    use intomind::Session;

    fn embeddings(session: &Session) -> Result<(), Box<dyn std::error::Error>> {
        use intomind::embeddings::{Assembler, Form};

        let ask = session.set_embeddings(Some(Form::Window))?;
        let mut windows = Assembler::new(4, 20, Form::Window);  // channels, tokens per channel
        // for each Event::Embedding(part): if let Some(window) = windows.feed(part) { ... }
        Ok(())
    }
}

/// Run your own head.
mod head {
    use intomind::{Command, Session};

    /// Whatever Bluetooth stack you already have, writing a command.
    fn write(command: &Command) {
        let _ = command;
    }

    /// And a chunk of data, with no response.
    fn write_data(bytes: &[u8]) {
        let _ = bytes;
    }

    /// The update service's answer, as the stack delivers it.
    fn update_answer() -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        unimplemented!("your stack")
    }

    fn head(session: &Session, blob: Vec<u8>) -> Result<(), Box<dyn std::error::Error>> {
        use intomind::{Step, Transfer};

        let mut transfer = Transfer::head(&session, 2, &blob)?;   // slot 2: slot 1 holds the open sex head
        loop {
            match transfer.step(&blob) {
                Step::Send(command) => {
                    write(&command);
                    transfer.on_answer(&update_answer()?, &blob)?;
                }
                Step::Data { from, to, .. } => {
                    write_data(&blob[from..to]);
                    transfer.sent(to - from);
                }
                Step::Verified(_) => break,
            }
        }
        Ok(())
    }
}

/// The timebase.
mod timebase {
    use intomind::protocol::device_info::DeviceInfo;

    fn timebase(info: &DeviceInfo, before: f64, after: f64, device_ticks: u64) {
        use intomind::{Exchange, Timebase};

        let mut timebase = Timebase::new(info.time_tick_hz);
        timebase.observe(Exchange { before, after, device_ticks });
        let seconds = timebase.host_time(device_ticks);
        let _ = seconds;
    }
}

/// Record.
mod record {
    use intomind::Session;

    fn record(session: &mut Session, answer: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        use intomind::capture::record::{Run, Setup};
        use intomind::capture::recording::Collection;
        use intomind::HostClocks;

        // Each time exchange: send session.time_sync() and wait for its answer.
        let before = HostClocks::now();
        session.on_time_sync_clocks(&answer, before, HostClocks::now())?;

        session.arm_recorder();             // then send reset_epoch() and start_stream()
        // Every notification goes to session.on_notification, as before.
        // Send stop_stream() at the end.
        let mut run = Run::default();
        run.extra.set("experiment", "my-experiment".into());
        let recordings = Collection::open(Collection::default_dir());
        let manifest = session.save_recording(&recordings, "my-recording", &Setup::default(), &run)?;
        let _ = manifest;
        Ok(())
    }
}

/// Recordings and exports.
mod exports {
    fn exports() -> Result<(), Box<dyn std::error::Error>> {
        use intomind::capture::{export, recording::Collection};

        let recordings = Collection::open(Collection::default_dir());
        let summary = export::export(&recordings, "my-recording", "bdf", None)?;   // also "edf", "npz", "csv", "tsv" and "mat"
        println!("{} samples, largest error {} µV", summary["samples"], summary["max_error_uv"]);

        let recording = recordings.load("my-recording")?;                       // refused if its checksums do not match
        let ch1 = recording.uv(0)?;
        let _ = ch1;
        Ok(())
    }
}

fn main() {
    // Nothing here runs: the stand-ins are unimplemented. What matters is
    // that every example compiles against the crate as it actually is.
}
