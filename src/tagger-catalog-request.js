// One page's concurrent mount/poll reads share a flight, not a persistent status cache.
// A force refresh behind a normal flight queues one fresh read; it never authorizes models.
export function createCatalogRequest(load) {
  let active = null;
  return function get(force = false) {
    if (active) {
      if (force && !active.force) return active.promise.catch(() => {}).then(() => get(true));
      return active.promise;
    }
    const ticket = { force, promise: null };
    active = ticket;
    ticket.promise = Promise.resolve().then(() => load(force)).finally(() => { if (active === ticket) active = null; });
    return ticket.promise;
  };
}

// Same-origin GET normally has no Origin header. The local explicit action
// uses POST, so the established browser write gate sees the real Origin.
// LAN refresh always remains an ordinary read, never a hidden force request.
export function catalogReadIntent(force, localBrowser) {
  return force && localBrowser
    ? { path: "/models?refresh=true", options: { method: "POST" } }
    : { path: "/models", options: undefined };
}
