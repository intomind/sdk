/**
 * Recordings in a folder on disk, for Node.
 *
 * The rest of `capture` reads and writes bytes, so it runs in a browser.
 * This is the one file of it that touches a disk: it hands a folder's files
 * to a `Collection`, writes an export where the Python library would, and
 * writes a recording into a folder as the library's recorder does.
 */

import { spawnSync } from "node:child_process";
import { appendFileSync, chmodSync, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, resolve, sep } from "node:path";
import process from "node:process";

import { CaptureError } from "./errors.ts";
import { exportRecording, fileName, formatOf } from "./export.ts";
import type { ExportOptions } from "./export.ts";
import { trim } from "./pyjson.ts";
import { checksums } from "./record.ts";
import type { GitState, Saved } from "./record.ts";
import { Collection, checkLabel } from "./recording.ts";

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

function write(p: string, bytes: Uint8Array): void {
  try {
    writeFileSync(p, bytes);
  } catch (e) {
    throw new CaptureError(`${p}: ${(e as Error).message}`);
  }
}

/**
 * Write a recording `saved` made into the recordings folder `dir`, as the
 * Python library's recorder writes one: the samples, the manifest and the
 * events; then, for a run that finished, the checksums and the files made
 * read-only; and last a line in the folder's session log. The checksums are
 * taken over the files as they are on disk once written, as the library
 * takes them, so a file already there under this name is counted with them.
 */
export function writeRecording(dir: string, saved: Saved): void {
  const label = checkLabel(saved.label);
  mkdir(dir);
  for (const [name, bytes] of saved.files) if (name !== `${label}.sha256`) write(join(dir, name), bytes);
  if (saved.finalized) {
    const present: [string, Uint8Array][] = [];
    for (const suffix of [".npz", ".meta.json", ".events.json", ".inputs.npz"]) {
      const p = join(dir, `${label}${suffix}`);
      if (existsSync(p)) present.push([`${label}${suffix}`, new Uint8Array(readFileSync(p))]);
    }
    write(join(dir, `${label}.sha256`), checksums(present));
    // Read-only once finalized, as the library leaves a recording. Best
    // effort, never fatal.
    for (const suffix of [".npz", ".meta.json", ".events.json", ".inputs.npz", ".sha256"]) {
      const p = join(dir, `${label}${suffix}`);
      try {
        if (existsSync(p)) chmodSync(p, 0o444);
      } catch {
        // as the library: a recording that cannot be frozen is still saved
      }
    }
  }
  const log = join(dir, "SESSIONS.jsonl");
  try {
    appendFileSync(log, saved.sessionLine);
  } catch (e) {
    throw new CaptureError(`${log}: ${(e as Error).message}`);
  }
}

/**
 * The commit of the source tree at `repo` and whether it has changes, as the
 * Python library's `git_state` reads them for the application that declared
 * the tree: each from git, or null where git cannot say within five seconds.
 */
export function gitState(repo: string): GitState {
  const at = resolve(repo);
  const git = (...args: string[]): string | null => {
    const r = spawnSync("git", ["-C", at, ...args], {
      encoding: "utf8",
      timeout: 5000,
      stdio: ["ignore", "pipe", "ignore"],
    });
    if (r.error !== undefined || r.status !== 0) return null;
    return trim(r.stdout);
  };
  const commit = git("rev-parse", "HEAD");
  const status = git("status", "--porcelain");
  return { commit, dirty: status === null ? null : status !== "" };
}
