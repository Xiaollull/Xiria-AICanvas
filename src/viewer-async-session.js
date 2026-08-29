export class ViewerAsyncSession {
  constructor() {
    this.mounted = true;
    this.active = false;
    this.session = 0;
    this.sequence = 0;
    this.latest = new Map();
    this.dropTail = Promise.resolve();
    this.dropReleases = new Set();
  }

  reset(active) {
    for (const release of this.dropReleases || []) release();
    this.dropReleases?.clear();
    this.dropTail = Promise.resolve();
    this.latest.clear();
    this.session += 1;
    this.active = active;
    return this.session;
  }

  mount() {
    this.mounted = true;
    return this.reset(false);
  }

  beginSession() {
    return this.reset(true);
  }

  request(kind, { session = this.session, latest = false, key = kind } = {}) {
    const token = { kind, key, session, id: ++this.sequence, latest };
    if (kind === "drop") {
      token.previousDrop = this.dropTail;
      token.releaseDrop = null;
      this.dropTail = new Promise((resolve) => { token.releaseDrop = resolve; this.dropReleases.add(resolve); });
    }
    if (latest) this.latest.set(key, token.id);
    return token;
  }

  isCurrent(token) {
    return Boolean(token && this.mounted && this.active && token.session === this.session && (!token.latest || this.latest.get(token.key) === token.id));
  }

  async waitForDropTurn(token) {
    if (token?.kind === "drop") await token.previousDrop;
    return this.isCurrent(token);
  }

  releaseDrop(token) {
    token?.releaseDrop?.();
    if (token?.releaseDrop) this.dropReleases.delete(token.releaseDrop);
    token.releaseDrop = null;
  }

  close() {
    this.reset(false);
  }

  unmount() {
    this.reset(false);
    this.mounted = false;
  }
}

export function viewerOpenPlan(source) {
  return { decode: Boolean(source), empty: !source };
}

export function createViewerRafScheduler(api = globalThis) {
  let pendingId = null;
  const request = api?.requestAnimationFrame;
  const cancel = api?.cancelAnimationFrame;
  return {
    cancel() {
      if (pendingId !== null && typeof cancel === "function") cancel.call(api, pendingId);
      pendingId = null;
    },
    schedule(callback) {
      this.cancel();
      if (typeof request !== "function") return null;
      pendingId = request.call(api, () => {
        pendingId = null;
        callback();
      });
      return pendingId;
    },
    get pendingId() { return pendingId; },
  };
}
