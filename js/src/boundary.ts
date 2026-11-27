import { Connection } from "./web-bluetooth.ts";

/** Whether `x` is a browser connection to an IntoMind device, for a module that names the class as a type alone. */
export function isConnection(x: unknown): x is Connection {
  return x instanceof Connection;
}

/** Whether `source` says it can do what `capability` names. */
export function able(source: Connection, capability: string): boolean {
  return Boolean(source.can(capability));
}
