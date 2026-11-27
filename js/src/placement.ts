/**
 * Where a device was put on a head, as the Python library's `placement`
 * module says it: every position of the 10-10 system on offer, and "other"
 * for anywhere else, each position with what is under it, and the placement
 * a recording stores.
 *
 * A device with a wiring preset in its profile goes on a head at one
 * position, so a person picks only that:
 *
 *   const wiring = session.profile().wiring;
 *   const where = placement(wiring.layout, null, { position: "CPz" });
 *   // record with run.extra = { electrode_layout: wiring.layout, placement: where }
 *
 * Each does exactly what its Python counterpart does, held to the library's
 * own output in `contract/placement`. What the library refuses is refused
 * here in its words, with the package's errors: `Invalid` for a value,
 * `TypeError` for a type.
 */

import * as PL from "./capture/placement.ts";
import { Fail } from "./capture/montage.ts";
import * as J from "./capture/pyjson.ts";
import type { Json, Value } from "./capture/pyjson.ts";
import { Invalid } from "./protocol.ts";

/** What the library raises, as the package's error with the library's words. */
function refused(e: unknown): unknown {
  if (!(e instanceof Fail)) return e;
  if (e.kind === "placement") return new Invalid(`PlacementError: ${e.shown}`);
  if (e.kind === "montage") return new Invalid(`MontageError: ${e.shown}`);
  if (e.shown.startsWith("KeyError") || e.shown.startsWith("ValueError")) return new Invalid(e.shown);
  return new TypeError(e.shown);
}

function asked<T>(run: () => T): T {
  try {
    return run();
  } catch (e) {
    throw refused(e);
  }
}

/**
 * Every region a position can be over, by key, in the library's order: the
 * library's `REGIONS`. What `regionsUnder` fills in is drawn from these, and
 * a placement's `over` names them when a person changes what is under a slot.
 */
export const REGIONS: ReadonlyMap<string, { readonly key: string; readonly human: string; readonly kind: string }> = new Map(
  PL.REGIONS.map(([key, human, kind]) => [key, Object.freeze({ key, human, kind })]),
);

/** Every position on offer, in head order, front to back, then "other": the library's `positions()`. */
export function positions(): Value[] {
  return PL.positions();
}

/** One position by key, a 10-10 code or "other": the library's `scalp()`. */
export function scalp(key: Json): Value {
  return asked(() => PL.scalp(J.fromPlain(key)));
}

/** What is under a position, each region marked as inferred, and carrying its share and source where it is under a 10-10 position and the study measured it there: the library's `regions_under()`. */
export function regionsUnder(key: Json): Value[] {
  return asked(() => PL.regionsUnder(J.fromPlain(key)));
}

/** What `placement` takes after the layout and the sites, by the library's keywords. */
export interface PlacementOptions {
  /** The name of a preset the layout declares, which supplies the sites. */
  readonly preset?: Json;
  /** How far the layout was turned, in degrees, as Python's `rotation_deg`. */
  readonly rotationDeg?: Json;
  /** Regions named by hand for slots, in place of what is under their positions. */
  readonly over?: Json;
  /** Free text, never parsed: what fits no vocabulary, such as where an electrode at "other" sat. */
  readonly note?: Json;
  /** Whether the bias was driven, as Python's `bias_driven`. True when not given. */
  readonly biasDriven?: boolean;
  /** One position for every electrode, a 10-10 code or "other". */
  readonly position?: Json;
}

/**
 * Where a layout was put, on this head, this time: the library's
 * `placement()`. `sites` maps slot key to position and may be null when
 * `preset` names one the layout declares. `position` puts every electrode at
 * one position, leaving out a bias that is not driven, and takes neither
 * sites nor a preset.
 */
export function placement(layout: Json, sites: Json = null, options: PlacementOptions = {}): Value {
  const some = (v: Json | undefined): Value | undefined => (v === undefined ? undefined : J.fromPlain(v));
  return asked(() =>
    PL.placement(J.fromPlain(layout), J.fromPlain(sites), {
      preset: some(options.preset),
      rotationDeg: some(options.rotationDeg),
      over: some(options.over),
      note: some(options.note),
      biasDriven: options.biasDriven === undefined ? undefined : J.bool(options.biasDriven),
      position: some(options.position),
    }),
  );
}
