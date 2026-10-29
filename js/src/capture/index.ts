/**
 * What an IntoMind recording is, read and exported in a browser or in Node.
 *
 * A recording is the set of files every IntoMind program writes under one
 * name: `name.npz` with the samples, `name.meta.json` with the manifest,
 * `name.events.json` with the markers, and `name.sha256`, written last.
 * `Collection` reads them as the Python library reads them, its checks and
 * its two corrections included, and `exportRecording` writes any of them as
 * npz, edf, bdf, csv, tsv or mat, each byte for byte or array for array
 * what the Python library writes, with the same summary.
 * docs.intomind.com/files.html states what both promise.
 *
 * Nothing here reads or writes a disk: a collection is handed its files'
 * bytes, and an export hands back the file's bytes. `capture/node` does both
 * on a folder in Node.
 *
 * ```ts
 * import { Collection, exportRecording } from "@intomind/sdk/capture";
 *
 * const recordings = new Collection(files);       // file name -> bytes
 * const bdf = exportRecording(recordings, "my-recording", "bdf");
 * console.log(bdf.summary.samples, "samples, largest error", bdf.summary.max_error_uv, "µV");
 * ```
 */

export { CaptureError } from "./errors.ts";
export { Collection, Recording, checkLabel } from "./recording.ts";
export type { CollectionOptions, Parts, Source, Verification } from "./recording.ts";
export { FORMATS, exportRecording, fileName, formatOf } from "./export.ts";
export type { ExportOptions, Exported, Format } from "./export.ts";
export { toPlain } from "./pyjson.ts";
export type { Value } from "./pyjson.ts";
export type { Data, NpyArray } from "./npy.ts";
