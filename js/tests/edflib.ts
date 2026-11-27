/**
 * EDFlib, the library EDFbrowser and pyedflib read files with, refuses to
 * open an EDF+ or BDF+ file that breaks any rule below. They are its header
 * check, edflib_check_edf_file in edflib.c, rule for rule and in its order,
 * then its check of the annotation signal in every data record,
 * edflib_get_annotations, as the Python library's suite writes them again,
 * so the suite here holds every export to them with nothing installed.
 */

const MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];

function printable(b: Uint8Array): boolean {
  return b.every((c) => c >= 32 && c <= 126);
}

function isDigit(c: string | undefined): boolean {
  return c !== undefined && c >= "0" && c <= "9";
}

/** edflib_is_integer_number: a sign, then digits, then only spaces. */
function integer(text: string): boolean {
  if (text === "") return false;
  let digit = 0;
  let space = false;
  for (const c of text.slice(text[0] === "+" || text[0] === "-" ? 1 : 0)) {
    if (c === " ") {
      if (!digit) return false;
      space = true;
    } else if (isDigit(c) && !space) {
      digit++;
    } else {
      return false;
    }
  }
  return digit > 0;
}

/** edflib_is_number: a sign, digits with one dot, an exponent, then only spaces. */
function number(text: string): boolean {
  if (text === "") return false;
  let i = text[0] === "+" || text[0] === "-" ? 1 : 0;
  let digit = 0;
  let space = false;
  let dot = false;
  while (i < text.length) {
    const c = text[i]!;
    if (c === "e" || c === "E") {
      if (!digit) return false;
      i++;
      if (i === text.length) return false;
      if (text[i] === "+" || text[i] === "-") i++;
      digit = 0;
      for (const d of text.slice(i)) {
        if (d === " ") {
          if (!digit) return false;
          space = true;
        } else if (isDigit(d) && !space) {
          digit++;
        } else {
          return false;
        }
      }
      return digit > 0;
    }
    if (c === " ") {
      if (!digit) return false;
      space = true;
    } else if ((isDigit(c) || c === ".") && !space) {
      if (c === ".") {
        if (dot) return false;
        dot = true;
      } else {
        digit++;
      }
    } else {
      return false;
    }
    i++;
  }
  return digit > 0;
}

/** edflib_atoi_nonlocalized: spaces, a sign, then the digits up to the first other character. */
function atoi(text: string): number {
  let s = text.replace(/^ +/, "");
  const sign = s[0] === "-" ? -1 : 1;
  if (s[0] === "+" || s[0] === "-") s = s.slice(1);
  let digits = "";
  for (const c of s) {
    if (!isDigit(c)) break;
    digits += c;
  }
  return sign * Number(digits || "0");
}

/** A time as EDFlib reads it, edflib_get_long_time: in units of 100 ns. */
function ticks(text: string): bigint {
  const neg = text[0] === "-";
  if (text[0] === "+" || text[0] === "-") text = text.slice(1);
  const dot = text.indexOf(".");
  const whole = dot < 0 ? text : text.slice(0, dot);
  const frac = dot < 0 ? "" : text.slice(dot + 1);
  let value = 0n;
  [...whole].reverse().forEach((c, k) => {
    value += BigInt(c.charCodeAt(0) - 48) * 10n ** BigInt(7 + k);
  });
  [...frac.slice(0, 7)].forEach((c, k) => {
    value += BigInt(c.charCodeAt(0) - 48) * 10n ** BigInt(6 - k);
  });
  return neg ? -value : value;
}

/** edflib_is_onset_number: a sign, then digits with one dot, the dot neither first nor last. */
function onsetNumber(text: string): boolean {
  if (text.length < 2 || (text[0] !== "+" && text[0] !== "-") || text[1] === "." || text[text.length - 1] === ".") return false;
  const body = text.slice(1);
  return body.split(".").length <= 2 && [...body].every((c) => isDigit(c) || c === ".");
}

/** edflib_is_duration_number: digits with one dot, the dot neither first nor last. */
function durationNumber(text: string): boolean {
  if (text === "" || text[0] === "." || text[text.length - 1] === ".") return false;
  return text.split(".").length <= 2 && [...text].every((c) => isDigit(c) || c === ".");
}

/**
 * The first of EDFlib's rules the file `b` breaks, named, or null for a file
 * EDFlib opens. Its header check first, then, unless `headerOnly`, its check
 * of the annotation signal in every data record, which it runs as it opens
 * an EDF+ or BDF+ file.
 */
export function edflibRefusal(b: Uint8Array, headerOnly = false): string | null {
  const latin1 = (x: Uint8Array): string => Array.from(x, (c) => String.fromCharCode(c)).join("");
  const at = (o: number, n: number): string => latin1(b.subarray(o, o + n));
  const head = b.subarray(0, 256);
  if (head.length < 256) return "a header of fewer than 256 bytes";
  // The version.
  const bdf = head[0] === 0xff;
  if (bdf) {
    if (!printable(head.subarray(1, 8)) || latin1(head.subarray(1, 8)) !== "BIOSEMI") return "the version field";
  } else if (!printable(head.subarray(0, 8)) || latin1(head.subarray(0, 8)) !== "0       ") {
    return "the version field";
  }
  // The patient and recording fields, each printable.
  if (!printable(head.subarray(8, 88))) return "the patient field holds a character that is not printable";
  if (!printable(head.subarray(88, 168))) return "the recording field holds a character that is not printable";
  // The start date, dd.mm.yy.
  const d = at(168, 8);
  if (!printable(head.subarray(168, 176)) || d[2] !== "." || d[5] !== "." || ![0, 1, 3, 4, 6, 7].every((k) => isDigit(d[k]))) {
    return "the start date field is not dd.mm.yy";
  }
  const day = Number(d.slice(0, 2));
  const month = Number(d.slice(3, 5));
  const yy = Number(d.slice(6, 8));
  if (day < 1 || day > 31 || month < 1 || month > 12) return "the start date field holds no day or month";
  const year = yy + (yy > 84 ? 1900 : 2000);
  // The start time, hh.mm.ss.
  const t = at(176, 8);
  if (!printable(head.subarray(176, 184)) || t[2] !== "." || t[5] !== "." || ![0, 1, 3, 4, 6, 7].every((k) => isDigit(t[k]))) {
    return "the start time field is not hh.mm.ss";
  }
  if (Number(t.slice(0, 2)) > 23 || Number(t.slice(3, 5)) > 59 || Number(t.slice(6, 8)) > 59) return "the start time field holds no time of day";
  // The number of signals, and of header bytes.
  if (!printable(head.subarray(252, 256)) || !integer(at(252, 4))) return "the number of signals is not a whole number";
  const ns = atoi(at(252, 4));
  if (ns < 1 || ns > 4096) return "the number of signals is not 1 to 4096";
  if (!printable(head.subarray(184, 192)) || !integer(at(184, 8)) || atoi(at(184, 8)) !== 256 * (ns + 1)) {
    return "the number of header bytes is not 256 for each signal and 256 more";
  }
  // The reserved field.
  const reserved = at(192, 44);
  if (!printable(head.subarray(192, 236))) return "the reserved field holds a character that is not printable";
  const family = bdf ? "BDF" : "EDF";
  const plus = reserved === `${family}+C`.padEnd(44) || reserved === `${family}+D`.padEnd(44);
  if (!bdf && !plus && reserved !== " ".repeat(44)) return "the reserved field is neither EDF+C, EDF+D nor empty";
  const discontinuous = plus && reserved[4] === "D";
  // The number of data records and their duration.
  if (!printable(head.subarray(236, 244)) || !integer(at(236, 8)) || atoi(at(236, 8)) < 1) {
    return "the number of data records is not a whole number of at least 1";
  }
  const nRecords = atoi(at(236, 8));
  if (!printable(head.subarray(244, 252)) || !number(at(244, 8)) || Number(at(244, 8).trim()) < -0.000001) {
    return "the data record duration is not a number of at least 0";
  }
  const duration = ticks(at(244, 8).split(" ")[0]!);
  const sig = b.subarray(256, 256 * (ns + 1));
  if (sig.length < 256 * ns) return "a header cut short";
  const col = (o: number, w: number): Uint8Array[] => Array.from({ length: ns }, (_, i) => sig.subarray(o * ns + i * w, o * ns + (i + 1) * w));
  const blank = (x: Uint8Array): boolean => x.every((c) => c === 0x20);
  // The labels, and the annotation signal.
  const labels = col(0, 16);
  if (!labels.every(printable)) return "a label holds a character that is not printable";
  const annot = labels.map((x) => plus && latin1(x) === `${family} Annotations `);
  if (plus && !annot.some((a) => a)) return `an ${family}+ file with no ${family} Annotations signal`;
  if ((!annot.every((a) => a) || !plus) && duration < 1n) return "a data record duration of 0 in a file of signals";
  // The transducers, dimensions, physical and digital ranges, prefilters,
  // samples per record and reserved fields of every signal.
  for (const [i, x] of col(16, 80).entries()) {
    if (!printable(x) || (annot[i] && !blank(x))) return `signal ${i + 1}'s transducer field`;
  }
  if (!col(96, 8).every(printable)) return "a physical dimension holds a character that is not printable";
  const pmin = col(104, 8);
  for (const [i, x] of pmin.entries()) {
    if (!printable(x) || !number(latin1(x))) return `signal ${i + 1}'s physical minimum is not a number`;
  }
  for (const [i, x] of col(112, 8).entries()) {
    if (!printable(x) || !number(latin1(x))) return `signal ${i + 1}'s physical maximum is not a number`;
    if (Number(latin1(x).trim()) === Number(latin1(pmin[i]!).trim())) return `signal ${i + 1}'s physical maximum is its minimum`;
  }
  const [low, high] = bdf ? [-0x800000, 0x7fffff] : [-0x8000, 0x7fff];
  const dmin = col(120, 8);
  for (const [i, x] of dmin.entries()) {
    if (!printable(x) || !integer(latin1(x))) return `signal ${i + 1}'s digital minimum is not a whole number`;
    const v = atoi(latin1(x));
    if ((annot[i] && v !== low) || v < low || v > high) return `signal ${i + 1}'s digital minimum`;
  }
  for (const [i, x] of col(128, 8).entries()) {
    if (!printable(x) || !integer(latin1(x))) return `signal ${i + 1}'s digital maximum is not a whole number`;
    const v = atoi(latin1(x));
    if ((annot[i] && v !== high) || v < low || v > high) return `signal ${i + 1}'s digital maximum`;
    if (v < atoi(latin1(dmin[i]!)) + 1) return `signal ${i + 1}'s digital maximum is not above its minimum`;
  }
  for (const [i, x] of col(136, 80).entries()) {
    if (!printable(x) || (annot[i] && !blank(x))) return `signal ${i + 1}'s prefilter field`;
  }
  const nsr: number[] = [];
  for (const [i, x] of col(216, 8).entries()) {
    if (!printable(x) || !integer(latin1(x)) || atoi(latin1(x)) < 1) return `signal ${i + 1}'s samples per data record`;
    nsr.push(atoi(latin1(x)));
  }
  const sampleBytes = bdf ? 3 : 2;
  const recordBytes = nsr.reduce((a, v) => a + v, 0) * sampleBytes;
  if (recordBytes > (bdf ? 15 : 10) * 1024 * 1024) return "a data record of more bytes than EDFlib reads";
  for (const [i, x] of col(224, 32).entries()) {
    if ((!bdf || plus) && !blank(x)) return `signal ${i + 1}'s reserved field is not empty`;
    if (bdf && !plus && !printable(x)) return `signal ${i + 1}'s reserved field holds a character that is not printable`;
  }
  if (plus) {
    // The patient field: code, sex, birthdate and name, each X where not known.
    const p = at(8, 80) + "\0".repeat(48);
    let dot = p.slice(0, 80).includes(" ") ? p.indexOf(" ") + 1 : 1;
    if (dot > 73 || dot < 2) return "the patient field";
    if (p[dot + 2] !== "X" && dot > 65) return "the patient field";
    if (!"MFX".includes(p[dot]!)) return "the patient field's sex";
    dot += 1;
    if (p[dot] !== " ") return "the patient field";
    if (p[dot + 1] === "X") {
      if (p[dot + 2] !== " " || p[dot + 3] === " ") return "the patient field";
    } else {
      if (p[dot + 12] !== " " || p[dot + 13] === " ") return "the patient field";
      const born = p.slice(dot + 1, dot + 12);
      const bd = Number(born.slice(0, 2));
      if (born[2] !== "-" || born[6] !== "-" || ![0, 1, 7, 8, 9, 10].every((k) => isDigit(born[k])) || bd < 1 || bd > 31 || !MONTHS.includes(born.slice(3, 6))) {
        return "the patient field's birthdate";
      }
    }
    // The recording field: Startdate, the start date as DD-MMM-YYYY or X,
    // then the administration, technician and equipment codes, each one or
    // more characters, separated by single spaces.
    const r = at(88, 80) + "\0".repeat(48);
    if (r.slice(0, 10) !== "Startdate ") return "the recording field does not begin with Startdate";
    let k: number;
    let date = "";
    if (r[10] === "X") {
      if (r[11] !== " " || r[12] === " ") return "the recording field's start date";
      k = 12;
    } else {
      if (r[21] !== " " || r[22] === " ") return "the recording field's start date";
      k = 22;
      date = r.slice(10, 21);
      const dd = Number(date.slice(0, 2));
      if (date[2] !== "-" || date[6] !== "-" || ![0, 1, 7, 8, 9, 10].every((j) => isDigit(date[j])) || dd < 1 || dd > 31 || !MONTHS.includes(date.slice(3, 6))) {
        return "the recording field's start date is not DD-MMM-YYYY";
      }
    }
    let spaces = 0;
    for (let i = k; i < 80; i++) {
      if (i > 78) return "the recording field holds fewer than five subfields";
      if (r[i] === " ") {
        spaces++;
        if (r[i + 1] === " ") return "the recording field holds two spaces where one separates its subfields";
      }
      if (spaces > 1) break;
    }
    if (r[10] !== "X") {
      if (day !== Number(date.slice(0, 2)) || month !== MONTHS.indexOf(date.slice(3, 6)) + 1 || yy !== Number(date.slice(9, 11))) {
        return "the recording field's start date is not the header's";
      }
      if (year !== Number(date.slice(7, 11))) return "the recording field's year is not the header's";
    }
  }
  // The file's size.
  if (b.length !== 256 * (ns + 1) + recordBytes * nRecords) return "the file's size is not its header and its data records";
  if (headerOnly || !plus) return null;
  // Every data record's annotation signal: zeros at its end, its first list
  // the record's onset, exactly one duration after the last record's in a
  // continuous file, then lists of onset, duration and texts.
  let elapsed = 0n;
  const annots = annot.flatMap((a, i) => (a ? [i] : []));
  for (let rec = 0; rec < nRecords; rec++) {
    const base = 256 * (ns + 1) + rec * recordBytes;
    for (const [which, a] of annots.entries()) {
      const start = base + nsr.slice(0, a).reduce((s, v) => s + v, 0) * sampleBytes;
      const blob = b.subarray(start, start + nsr[a]! * sampleBytes);
      if (blob[blob.length - 1] !== 0) return `data record ${rec + 1}'s annotations do not end in a zero byte`;
      if (which === 0) {
        const cut = blob.subarray(0, blob.length - 2).indexOf(0x14);
        if (cut < 0 || blob[cut + 1] !== 0x14) return `data record ${rec + 1} begins with no list of its onset`;
        const onset = latin1(blob.subarray(0, cut));
        if (!onsetNumber(onset)) return `data record ${rec + 1}'s onset is not a number`;
        const t0 = ticks(onset);
        if (rec === 0 && !(t0 >= 0n && t0 < 10_000_000n)) return "the first data record's onset is not 0 to 1 s";
        if (rec && !discontinuous && t0 - elapsed !== duration) {
          return `data record ${rec + 1}'s onset is not the data record duration after data record ${rec}'s`;
        }
        if (rec && discontinuous && t0 - elapsed < duration) return `data record ${rec + 1} begins before data record ${rec} ends`;
        elapsed = t0;
      }
      let text: number[] = [];
      let zero = 0;
      let onsetRead = false;
      let durRead = false;
      let durOpen = false;
      let inTal = 0;
      let inRecord = 0;
      for (let k = 0; k < blob.length; k++) {
        const c = blob[k]!;
        if (c === 0) {
          if (!zero) {
            if (k && blob[k - 1] !== 0x14) return `data record ${rec + 1}: a list of annotations not closed by 0x14`;
            text = [];
            onsetRead = false;
            durRead = false;
            durOpen = false;
            inTal = 0;
          }
          zero++;
          continue;
        }
        if (zero > 1) return `data record ${rec + 1}: a list of annotations after its padding`;
        zero = 0;
        if (c === 0x14 || c === 0x15) {
          if (c === 0x15) {
            if (durRead || durOpen || onsetRead || inTal) return `data record ${rec + 1}: a duration not right after an onset`;
            durOpen = true;
          }
          if (c === 0x14 && onsetRead && !durOpen) {
            if ((which || inRecord) && text.slice(0, 512).some((x) => (x < 32 || x === 127) && x !== 9 && x !== 10 && x !== 13)) {
              return `data record ${rec + 1}: an annotation holds a character that is not printable`;
            }
            inTal++;
            inRecord++;
            text = [];
            continue;
          }
          if (!onsetRead) {
            if (!onsetNumber(String.fromCharCode(...text))) return `data record ${rec + 1}: an onset that is not a number`;
            onsetRead = true;
            text = [];
            continue;
          }
          if (durOpen) {
            if (!durationNumber(String.fromCharCode(...text))) return `data record ${rec + 1}: a duration that is not a number`;
            durRead = true;
            durOpen = false;
            text = [];
            continue;
          }
        }
        text.push(c);
      }
    }
  }
  return null;
}
