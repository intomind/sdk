/**
 * Recordings in a folder on disk, for Node.
 *
 * The rest of `capture` reads and writes bytes, so it runs in a browser.
 * This is the one file of it that touches a disk: it hands a folder's files
 * to a `Collection`, and writes an export where the Python library would.
 */

import { mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, sep } from "node:path";
import process from "node:process";

import { CaptureError } from "./errors.ts";
import { exportRecording, fileName, formatOf } from "./export.ts";
import type { ExportOptions } from "./export.ts";
import { Collection } from "./recording.ts";

/** The folder the Python library uses when nothing else is said: `INTOMIND_CAPTURES`, or `~/.local/share/intomind/captures`. */
export function defaultDir(): string {
  const d = process.env["INTOMIND_CAPTURES"];
  if (d !== undefined && d !== "") return d;
  const home = process.env["HOME"] ?? process.env["USERPROFILE"] ?? "";
  return join(home, ".local", "share", "intomind", "captures");
}

/** The recordings folder at `dir`, or at `defaultDir()`. */
export function openCollection(dir: string = defaultDir()): Collection {
  return new Collection(
    (name) => {
      const p = join(dir, name);
      try {
        if (!statSync(p).isFile()) return null;
        return new Uint8Array(readFileSync(p));
      } catch (e) {
        const code = (e as { code?: string }).code;
        if (code === "ENOENT" || code === "ENOTDIR") return null;
        throw new CaptureError(`${p}: ${(e as Error).message}`);
      }
    },
    { dir, separator: sep },
  );
}

function isDir(p: string): boolean {
  try {
    return statSync(p).isDirectory();
  } catch {
    return false;
  }
}

function mkdir(d: string): void {
  try {
    mkdirSync(d, { recursive: true });
  } catch (e) {
    throw new CaptureError(`${d}: ${(e as Error).message}`);
  }
}

/**
 * Export `label` from the recordings folder `dir` as `format`, into `path`:
 * a file, a folder to put it in, or, when absent, `exports` beside the
 * recordings folder. The summary is the Python library's, key for key, with
 * the file's path.
 */
export function exportToFile(
  dir: string,
  label: string,
  format: string,
  path?: string | null,
  options: ExportOptions = {},
): Record<string, unknown> {
  const fmt = formatOf(format);
  const out = exportRecording(openCollection(dir), label, fmt, options);
  const file = fileName(label, fmt);
  let target: string;
  if (path === undefined || path === null) {
    const parent = dirname(dir) === dir ? "." : dirname(dir);
    const d = join(parent, "exports");
    mkdir(d);
    target = join(d, file);
  } else if (isDir(path)) {
    target = join(path, file);
  } else {
    const parent = dirname(path);
    if (parent !== "" && parent !== ".") mkdir(parent);
    target = path;
  }
  try {
    writeFileSync(target, out.bytes);
  } catch (e) {
    throw new CaptureError(`${target}: ${(e as Error).message}`);
  }
  const { label: name, ...rest } = out.summary;
  return { label: name, path: target, ...rest };
}
