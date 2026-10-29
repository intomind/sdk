//! The zip container of an `.npz`, read and written.
//!
//! An `.npz` is a zip archive with one `.npy` file per array, which is what
//! numpy's `savez_compressed` writes. Only what that needs is here: stored
//! and deflated members, the ZIP64 records a large archive carries, and a
//! CRC-32 checked on every member read.

use std::fmt;

/// Why an archive could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ZipError(pub(crate) String);

impl fmt::Display for ZipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

fn bad(what: impl Into<String>) -> ZipError {
    ZipError(what.into())
}

const CRC_TABLE: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
};

/// CRC-32 as zip computes it, IEEE 802.3.
pub(crate) fn crc32(data: &[u8]) -> u32 {
    let mut c = !0u32;
    for &b in data {
        c = CRC_TABLE[((c ^ b as u32) & 0xff) as usize] ^ (c >> 8);
    }
    !c
}

fn u16_at(b: &[u8], at: usize) -> Result<u16, ZipError> {
    b.get(at..at + 2).map(|s| u16::from_le_bytes([s[0], s[1]])).ok_or_else(|| bad("the archive ends early"))
}

fn u32_at(b: &[u8], at: usize) -> Result<u32, ZipError> {
    b.get(at..at + 4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]])).ok_or_else(|| bad("the archive ends early"))
}

fn u64_at(b: &[u8], at: usize) -> Result<u64, ZipError> {
    b.get(at..at + 8)
        .map(|s| u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]))
        .ok_or_else(|| bad("the archive ends early"))
}

/// Every member of an archive, by name, in the order the central directory
/// lists them, each decompressed and checked against its CRC.
pub(crate) fn read(archive: &[u8]) -> Result<Vec<(String, Vec<u8>)>, ZipError> {
    // The end of central directory record, searched for from the end, past
    // any comment.
    let eocd = (0..archive.len().saturating_sub(21))
        .rev()
        .take(65_557)
        .find(|&i| archive[i..].starts_with(&[0x50, 0x4b, 0x05, 0x06]))
        .ok_or_else(|| bad("not a zip archive: no end of central directory"))?;
    let mut entries = u16_at(archive, eocd + 10)? as u64;
    let mut cd_size = u32_at(archive, eocd + 12)? as u64;
    let mut cd_offset = u32_at(archive, eocd + 16)? as u64;
    if entries == 0xffff || cd_size == 0xffff_ffff || cd_offset == 0xffff_ffff {
        // ZIP64: the locator sits just before the record above.
        let loc = eocd.checked_sub(20).ok_or_else(|| bad("a ZIP64 archive without its locator"))?;
        if u32_at(archive, loc)? != 0x0706_4b50 {
            return Err(bad("a ZIP64 archive without its locator"));
        }
        let rec = u64_at(archive, loc + 8)? as usize;
        if u32_at(archive, rec)? != 0x0606_4b50 {
            return Err(bad("a ZIP64 archive without its end record"));
        }
        entries = u64_at(archive, rec + 32)?;
        cd_size = u64_at(archive, rec + 40)?;
        cd_offset = u64_at(archive, rec + 48)?;
    }
    let mut out = Vec::with_capacity(entries as usize);
    let mut p = cd_offset as usize;
    let end = (cd_offset + cd_size) as usize;
    for _ in 0..entries {
        if p >= end || u32_at(archive, p)? != 0x0201_4b50 {
            return Err(bad("a central directory entry is missing"));
        }
        let flags = u16_at(archive, p + 8)?;
        let method = u16_at(archive, p + 10)?;
        let crc = u32_at(archive, p + 16)?;
        let mut csize = u32_at(archive, p + 20)? as u64;
        let mut usize_ = u32_at(archive, p + 24)? as u64;
        let name_len = u16_at(archive, p + 28)? as usize;
        let extra_len = u16_at(archive, p + 30)? as usize;
        let comment_len = u16_at(archive, p + 32)? as usize;
        let mut local = u32_at(archive, p + 42)? as u64;
        let name_bytes = archive.get(p + 46..p + 46 + name_len).ok_or_else(|| bad("the archive ends early"))?;
        let name = if flags & 0x800 != 0 {
            String::from_utf8(name_bytes.to_vec()).map_err(|_| bad("a member name that is not UTF-8"))?
        } else {
            // cp437 in principle. numpy's names are ASCII.
            name_bytes.iter().map(|&c| c as char).collect()
        };
        // The ZIP64 extra field holds, in order, whichever of the sizes and the
        // offset did not fit 32 bits.
        let mut x = p + 46 + name_len;
        let x_end = x + extra_len;
        while x + 4 <= x_end {
            let id = u16_at(archive, x)?;
            let len = u16_at(archive, x + 2)? as usize;
            if id == 0x0001 {
                let mut q = x + 4;
                if usize_ == 0xffff_ffff {
                    usize_ = u64_at(archive, q)?;
                    q += 8;
                }
                if csize == 0xffff_ffff {
                    csize = u64_at(archive, q)?;
                    q += 8;
                }
                if local == 0xffff_ffff {
                    local = u64_at(archive, q)?;
                }
            }
            x += 4 + len;
        }
        p = x_end + comment_len;

        let lh = local as usize;
        if u32_at(archive, lh)? != 0x0403_4b50 {
            return Err(bad(format!("{name}: the member's header is missing")));
        }
        let data_at = lh + 30 + u16_at(archive, lh + 26)? as usize + u16_at(archive, lh + 28)? as usize;
        let raw = archive
            .get(data_at..data_at + csize as usize)
            .ok_or_else(|| bad(format!("{name}: the archive ends early")))?;
        let data = match method {
            0 => raw.to_vec(),
            8 => miniz_oxide::inflate::decompress_to_vec_with_limit(raw, usize_ as usize)
                .map_err(|_| bad(format!("{name}: the compressed data is damaged")))?,
            m => return Err(bad(format!("{name}: compression method {m} is not one numpy writes"))),
        };
        if data.len() as u64 != usize_ {
            return Err(bad(format!("{name}: {} bytes where the archive says {usize_}", data.len())));
        }
        if crc32(&data) != crc {
            return Err(bad(format!("{name}: the CRC does not match")));
        }
        out.push((name, data));
    }
    Ok(out)
}

/// The date and time fields of a zip header for a moment in UTC.
fn dos_time(unix: u64) -> (u16, u16) {
    let days = (unix / 86_400) as i64;
    let secs = unix % 86_400;
    // Civil date from days since 1970-01-01.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    if y < 1980 {
        return (0, (1 << 5) | 1);
    }
    let time = (((secs / 3600) << 11) | ((secs % 3600 / 60) << 5) | ((secs % 60) / 2)) as u16;
    let date = (((y - 1980) as u16) << 9) | ((m as u16) << 5) | d as u16;
    (time, date)
}

/// An archive of `members`, each deflated as numpy's `savez_compressed`
/// deflates them, stamped with `unix_time`.
pub(crate) fn write(members: &[(String, Vec<u8>)], unix_time: u64) -> Vec<u8> {
    let (time, date) = dos_time(unix_time);
    let mut out = Vec::new();
    let mut central = Vec::new();
    let mut count = 0u64;
    for (name, data) in members {
        let packed = miniz_oxide::deflate::compress_to_vec(data, 6);
        let crc = crc32(data);
        let offset = out.len() as u64;
        let zip64 = data.len() as u64 >= 0xffff_ffff || packed.len() as u64 >= 0xffff_ffff || offset >= 0xffff_ffff;
        let version: u16 = if zip64 { 45 } else { 20 };
        let name_b = name.as_bytes();

        // Local header. Sizes that do not fit go in its ZIP64 field.
        out.extend(0x0403_4b50u32.to_le_bytes());
        out.extend(version.to_le_bytes());
        out.extend(0u16.to_le_bytes()); // flags
        out.extend(8u16.to_le_bytes()); // deflate
        out.extend(time.to_le_bytes());
        out.extend(date.to_le_bytes());
        out.extend(crc.to_le_bytes());
        if zip64 {
            out.extend(0xffff_ffffu32.to_le_bytes());
            out.extend(0xffff_ffffu32.to_le_bytes());
        } else {
            out.extend((packed.len() as u32).to_le_bytes());
            out.extend((data.len() as u32).to_le_bytes());
        }
        out.extend((name_b.len() as u16).to_le_bytes());
        out.extend((if zip64 { 20u16 } else { 0 }).to_le_bytes());
        out.extend(name_b);
        if zip64 {
            out.extend(0x0001u16.to_le_bytes());
            out.extend(16u16.to_le_bytes());
            out.extend((data.len() as u64).to_le_bytes());
            out.extend((packed.len() as u64).to_le_bytes());
        }
        out.extend(&packed);

        // Central directory entry.
        central.extend(0x0201_4b50u32.to_le_bytes());
        central.extend((version | (3 << 8)).to_le_bytes()); // made on unix
        central.extend(version.to_le_bytes());
        central.extend(0u16.to_le_bytes());
        central.extend(8u16.to_le_bytes());
        central.extend(time.to_le_bytes());
        central.extend(date.to_le_bytes());
        central.extend(crc.to_le_bytes());
        // Order in the ZIP64 field: uncompressed size, compressed size, offset.
        let mut extra = Vec::new();
        let (us, cs, off) = if zip64 {
            extra.extend((data.len() as u64).to_le_bytes());
            extra.extend((packed.len() as u64).to_le_bytes());
            extra.extend(offset.to_le_bytes());
            (0xffff_ffffu32, 0xffff_ffffu32, 0xffff_ffffu32)
        } else {
            (data.len() as u32, packed.len() as u32, offset as u32)
        };
        central.extend(cs.to_le_bytes());
        central.extend(us.to_le_bytes());
        central.extend((name_b.len() as u16).to_le_bytes());
        central.extend((if zip64 { 4 + extra.len() as u16 } else { 0 }).to_le_bytes());
        central.extend(0u16.to_le_bytes()); // comment
        central.extend(0u16.to_le_bytes()); // disk
        central.extend(0u16.to_le_bytes()); // internal attributes
        central.extend((0o600u32 << 16).to_le_bytes()); // external attributes, as numpy's members get
        central.extend(off.to_le_bytes());
        central.extend(name_b);
        if zip64 {
            central.extend(0x0001u16.to_le_bytes());
            central.extend((extra.len() as u16).to_le_bytes());
            central.extend(&extra);
        }
        count += 1;
    }
    let cd_offset = out.len() as u64;
    let cd_size = central.len() as u64;
    out.extend(&central);
    let big = count >= 0xffff || cd_offset >= 0xffff_ffff || cd_size >= 0xffff_ffff;
    if big {
        let rec = out.len() as u64;
        out.extend(0x0606_4b50u32.to_le_bytes());
        out.extend(44u64.to_le_bytes());
        out.extend(45u16.to_le_bytes());
        out.extend(45u16.to_le_bytes());
        out.extend(0u32.to_le_bytes());
        out.extend(0u32.to_le_bytes());
        out.extend(count.to_le_bytes());
        out.extend(count.to_le_bytes());
        out.extend(cd_size.to_le_bytes());
        out.extend(cd_offset.to_le_bytes());
        out.extend(0x0706_4b50u32.to_le_bytes());
        out.extend(0u32.to_le_bytes());
        out.extend(rec.to_le_bytes());
        out.extend(1u32.to_le_bytes());
    }
    out.extend(0x0605_4b50u32.to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out.extend(0u16.to_le_bytes());
    let n16 = if big { 0xffff } else { count as u16 };
    out.extend(n16.to_le_bytes());
    out.extend(n16.to_le_bytes());
    out.extend((if big { 0xffff_ffff } else { cd_size as u32 }).to_le_bytes());
    out.extend((if big { 0xffff_ffff } else { cd_offset as u32 }).to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_archive_reads_back_as_written() {
        let members =
            vec![("a.npy".to_string(), b"hello hello hello".to_vec()), ("b.npy".to_string(), vec![0u8; 100_000])];
        let z = write(&members, 1_792_986_217);
        assert_eq!(read(&z).unwrap(), members);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn a_damaged_member_is_refused() {
        let mut z = write(&[("a.npy".to_string(), b"some bytes to compress".to_vec())], 0);
        z[40] ^= 0xff;
        assert!(read(&z).is_err());
    }

    #[test]
    fn dates_are_civil() {
        // 2026-10-26 03:43:37 UTC
        let (t, d) = dos_time(1_792_986_217);
        assert_eq!((d >> 9) + 1980, 2026);
        assert_eq!((d >> 5) & 0xf, 10);
        assert_eq!(d & 0x1f, 26);
        assert_eq!(t >> 11, 3);
    }
}
