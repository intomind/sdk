/**
 * Web Bluetooth as Chrome behaves with a real device, in front of a device
 * made of nothing.
 *
 * A test's stand-in characteristics are the device's own ends of them: what
 * reading one gives, what writing to one does, and what the device notifies
 * on one, which it says by firing `characteristicvaluechanged` on it. A page
 * never holds them. It holds the objects the browser hands it, and those
 * keep to the rules Chrome keeps with a real device:
 *
 * - the browser holds one subscription to each characteristic of a device,
 *   bound to the object that last truly subscribed, and what the device
 *   notifies reaches that object's listeners and no other's;
 * - `startNotifications` on another object of the same characteristic, while
 *   that subscription lives, resolves at once and binds nothing;
 * - a link that drops, from the device's side or from this one, leaves the
 *   subscription alive, and `stopNotifications` on an object of a link that
 *   dropped is refused;
 * - each link hands out new objects, its service and every characteristic;
 * - `stopNotifications` on an object of the link that is up ends the
 *   subscription, whichever object held it;
 * - a read or a write on an object of a link that is down, or of one that
 *   dropped, is refused, and never reaches the device;
 * - nothing the device notifies reaches a page while the link is down.
 *
 * So a page that subscribes on a new link's objects without ending what the
 * browser still holds hears the device on the old objects, and on the new
 * ones not at all. A test may also have the browser refuse a
 * characteristic's next subscription, as a GATT operation that fails on a
 * link that is up is refused.
 */

import type {
  BluetoothDeviceLike,
  BluetoothRemoteGATTCharacteristicLike,
  BluetoothRemoteGATTServerLike,
  BluetoothRemoteGATTServiceLike,
} from "../src/web-bluetooth.ts";

/** The device's own end of a characteristic, as a test's stand-in plays it. */
export interface End extends EventTarget {
  readonly uuid: string;
  readonly value?: DataView;
  readValue(): Promise<DataView>;
  writeValueWithResponse(value: BufferSource): Promise<void>;
  writeValueWithoutResponse(value: BufferSource): Promise<void>;
}

/** What Chrome says of an operation on a link that is down. */
function down(): DOMException {
  return new DOMException("GATT Server is disconnected. Cannot perform GATT operations. (Re)connect first with `device.gatt.connect`.", "NetworkError");
}

/** What Chrome says of a GATT operation that failed on a link that is up. */
function failed(): DOMException {
  return new DOMException("GATT operation failed for unknown reason.", "NetworkError");
}

/** The browser's rules, as the objects it hands out keep to them. */
interface Rules {
  start(c: Handed): Promise<Handed>;
  stop(c: Handed): Promise<Handed>;
  /** Whether `c` was handed out on the link that is up, so a read or a write on it reaches the device. */
  live(c: Handed): boolean;
}

/** A characteristic as the browser hands it to a page, on one link. */
class Handed extends EventTarget implements BluetoothRemoteGATTCharacteristicLike {
  readonly uuid: string;
  value?: DataView;
  /** The link it was handed out on. */
  readonly link: number;
  readonly #end: End;
  readonly #rules: Rules;

  constructor(end: End, link: number, rules: Rules) {
    super();
    this.uuid = end.uuid;
    this.link = link;
    this.#end = end;
    this.#rules = rules;
  }

  readValue(): Promise<DataView> {
    if (!this.#rules.live(this)) return Promise.reject(down());
    return this.#end.readValue();
  }

  writeValueWithResponse(value: BufferSource): Promise<void> {
    if (!this.#rules.live(this)) return Promise.reject(down());
    return this.#end.writeValueWithResponse(value);
  }

  writeValueWithoutResponse(value: BufferSource): Promise<void> {
    if (!this.#rules.live(this)) return Promise.reject(down());
    return this.#end.writeValueWithoutResponse(value);
  }

  startNotifications(): Promise<this> {
    return this.#rules.start(this) as Promise<this>;
  }

  stopNotifications(): Promise<this> {
    return this.#rules.stop(this) as Promise<this>;
  }
}

/** One device as the browser holds it: the device a page is handed, its links, and the subscriptions the browser keeps. */
export class Browser {
  /** The device as the chooser hands it to a page. */
  readonly device: BluetoothDeviceLike;
  readonly #ends: ReadonlyMap<string, End>;
  readonly #service: string | undefined;
  /** By characteristic, the object the browser's subscription to it is bound to. */
  readonly #held = new Map<string, Handed>();
  /** The device ends the browser hears notifications on. */
  readonly #hearing = new WeakSet<End>();
  /** The characteristics whose next subscription is refused. */
  readonly #refusing = new Set<string>();
  #connected = false;
  /** The link that is up, or the last that was, numbered as each opens. */
  #link = 0;
  /** The objects the link that is up has handed out, by characteristic. */
  #handed = new Map<string, Handed>();
  #links = 0;

  /**
   * The browser in front of a device whose characteristics are `ends`, by
   * their UUIDs, looked up as a page asks for each, so a test may add one
   * before connecting. Where `about.service` names the device's service, a
   * page that asks for another is told there is none.
   */
  constructor(ends: ReadonlyMap<string, End>, about: { readonly id: string; readonly name: string; readonly service?: string }) {
    this.#ends = ends;
    this.#service = about.service;
    const browser = this;
    const rules: Rules = {
      start: (c) => {
        if (!this.#live(c)) return Promise.reject(down());
        if (this.#refusing.delete(c.uuid)) return Promise.reject(failed());
        if (!this.#held.has(c.uuid)) this.#held.set(c.uuid, c);
        return Promise.resolve(c);
      },
      stop: (c) => {
        if (!this.#live(c)) return Promise.reject(down());
        this.#held.delete(c.uuid);
        return Promise.resolve(c);
      },
      live: (c) => this.#live(c),
    };
    const characteristic = (link: number, uuid: string): Promise<BluetoothRemoteGATTCharacteristicLike> => {
      if (!this.#connected || link !== this.#link) return Promise.reject(down());
      const end = this.#ends.get(uuid);
      if (end === undefined) return Promise.reject(new DOMException(`No Characteristics matching UUID ${uuid} found in Service.`, "NotFoundError"));
      let c = this.#handed.get(uuid);
      if (c === undefined) {
        c = new Handed(end, link, rules);
        this.#handed.set(uuid, c);
        this.#hear(uuid, end);
      }
      return Promise.resolve(c);
    };
    let service: BluetoothRemoteGATTServiceLike | null = null;
    const server: BluetoothRemoteGATTServerLike = {
      get connected() {
        return browser.#connected;
      },
      connect() {
        if (!browser.#connected) {
          browser.#connected = true;
          browser.#link += 1;
          browser.#links += 1;
          browser.#handed = new Map();
          const link = browser.#link;
          service = { getCharacteristic: (uuid) => characteristic(link, uuid) };
        }
        return Promise.resolve(server);
      },
      disconnect() {
        browser.#lose();
      },
      getPrimaryService(uuid) {
        if (!browser.#connected || service === null) return Promise.reject(down());
        if (browser.#service !== undefined && uuid !== browser.#service) {
          return Promise.reject(new DOMException(`No Services matching UUID ${uuid} found in Device.`, "NotFoundError"));
        }
        return Promise.resolve(service);
      },
    };
    this.device = Object.assign(new EventTarget(), { id: about.id, name: about.name, gatt: server });
  }

  /** How many links have been opened to the device. */
  get links(): number {
    return this.#links;
  }

  /** Whether the browser holds a subscription to the characteristic `uuid`. */
  held(uuid: string): boolean {
    return this.#held.has(uuid);
  }

  /** The next `startNotifications` on the characteristic `uuid` is refused, and binds nothing, as a GATT operation that fails on a link that is up. */
  refuseNextStart(uuid: string): void {
    this.#refusing.add(uuid);
  }

  /** The device drops the link, as one that restarts, is switched off or goes out of range does. */
  drop(): void {
    this.#lose();
  }

  /**
   * Bring a link the device dropped back up as it was, its objects and all,
   * which no browser does by itself: the Python library's fixtures bring a
   * device back so between runs, where a page would connect it again.
   */
  back(): void {
    this.#connected = true;
  }

  /** Whether `c` was handed out on the link that is up. */
  #live(c: Handed): boolean {
    return this.#connected && c.link === this.#link;
  }

  /** The link is down: the device is told gone, and every subscription is kept. */
  #lose(): void {
    if (!this.#connected) return;
    this.#connected = false;
    this.device.dispatchEvent(new Event("gattserverdisconnected"));
  }

  /** What the device notifies on `end` while the link is up reaches the object the browser's subscription to it is bound to, and no other. */
  #hear(uuid: string, end: End): void {
    if (this.#hearing.has(end)) return;
    this.#hearing.add(end);
    end.addEventListener("characteristicvaluechanged", () => {
      if (!this.#connected) return;
      const c = this.#held.get(uuid);
      if (c === undefined || end.value === undefined) return;
      c.value = end.value;
      c.dispatchEvent(new Event("characteristicvaluechanged"));
    });
  }
}
