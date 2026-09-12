export class ViewerAsyncSession {
  constructor() {
    this.mounted = true;
    this.active = false;
    this.session = 0;
    this.sequence = 0;
    this.latest = new Map();
    this.revision = 0;
    this.appendEpoch = 0;
    this.operations = new Map();
    this.dropTail = Promise.resolve();
    this.dropReleases = new Set();
    this.dropEpoch = 0;
    this.replacementToken = null;
    this.appendTokens = new Set();
    this.abortableRequests = new Set();
    this.latestAbortable = new Map();
  }

  abortReplacement() {
    this.replacementToken?.controller?.abort();
    this.replacementToken = null;
  }

  resetAppendQueue() {
    this.appendEpoch += 1;
    this.dropEpoch += 1;
    for (const token of this.appendTokens) token.controller?.abort();
    this.appendTokens.clear();
    for (const release of this.dropReleases) release();
    this.dropReleases.clear();
    this.dropTail = Promise.resolve();
  }

  reset(active) {
    this.abortReplacement();
    for (const token of this.operations?.values() || []) token.controller?.abort();
    this.operations?.clear();
    this.resetAppendQueue();
    for (const token of this.abortableRequests) token.controller?.abort();
    this.abortableRequests.clear();
    this.latestAbortable.clear();
    this.latest.clear();
    this.session += 1;
    this.revision += 1;
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

  request(kind, { session = this.session, latest = false, key = kind, abortable = false } = {}) {
    const token = { kind, key, session, id: ++this.sequence, latest };
    if (abortable && typeof AbortController !== "undefined") {
      this.latestAbortable.get(key)?.controller?.abort();
      token.controller = new AbortController();
      token.signal = token.controller.signal;
      this.abortableRequests.add(token);
      if (latest) this.latestAbortable.set(key, token);
    }
    if (kind === "drop") {
      token.dropEpoch = this.dropEpoch;
      token.previousDrop = this.dropTail;
      token.releaseDrop = null;
      this.dropTail = new Promise((resolve) => { token.releaseDrop = resolve; this.dropReleases.add(resolve); });
    }
    if (latest) this.latest.set(key, token.id);
    return token;
  }

  invalidate(key) {
    this.latest.set(key, ++this.sequence);
  }

  beginReplacement(kind, { session = this.session } = {}) {
    this.revise({ invalidateReplacement: true, invalidateAppends: true });
    const token = this.request(kind, { session, latest: true, key: "replace" });
    token.revision = this.revision;
    token.revisionBound = true;
    token.controller = typeof AbortController !== "undefined" ? new AbortController() : null;
    token.signal = token.controller?.signal;
    this.replacementToken = token;
    return token;
  }

  beginAppend(kind) {
    this.revise({ invalidateReplacement: true, invalidateAppends: false });
    const token = this.request(kind, { latest: false });
    token.appendEpoch = this.appendEpoch;
    token.canvasAppend = true;
    token.controller = typeof AbortController !== "undefined" ? new AbortController() : null;
    token.signal = token.controller?.signal;
    this.appendTokens.add(token);
    return token;
  }

  revise({ invalidateReplacement = true, invalidateAppends = true } = {}) {
    this.revision += 1;
    if (invalidateReplacement) {
      this.invalidate("replace");
      this.abortReplacement();
    }
    if (invalidateAppends) this.resetAppendQueue();
    for (const token of this.abortableRequests) token.controller?.abort();
    this.abortableRequests.clear();
    this.latestAbortable.clear();
    for (const token of this.operations.values()) token.controller?.abort();
    this.operations.clear();
    return this.revision;
  }

  beginOperation(kind, { group = kind, abortable = true } = {}) {
    if (this.operations.has(group)) return null;
    const token = this.request(kind, { latest: true, key: `operation:${group}` });
    token.operationGroup = group;
    token.revision = this.revision;
    token.controller = abortable && typeof AbortController !== "undefined" ? new AbortController() : null;
    token.signal = token.controller?.signal;
    this.operations.set(group, token);
    return token;
  }

  isOperationCurrent(token) {
    return Boolean(token && this.operations.get(token.operationGroup) === token && token.revision === this.revision && this.isCurrent(token));
  }

  isOperationOwned(token) {
    return Boolean(token && this.operations.get(token.operationGroup) === token && token.revision === this.revision && this.isOwned(token));
  }

  finishOperation(token) {
    if (!token || this.operations.get(token.operationGroup) !== token) return false;
    this.operations.delete(token.operationGroup);
    token.finished = true;
    return true;
  }

  isOwned(token) {
    return Boolean(token && !token.finished && this.mounted && this.active && token.session === this.session
      && (!token.latest || this.latest.get(token.key) === token.id)
      && (!token.revisionBound || token.revision === this.revision)
      && (!token.canvasAppend || token.appendEpoch === this.appendEpoch));
  }

  isCurrent(token) {
    return Boolean(this.isOwned(token) && !token.signal?.aborted);
  }

  async waitForDropTurn(token) {
    if (token?.kind === "drop") await token.previousDrop;
    return token?.dropEpoch === this.dropEpoch && this.isCurrent(token);
  }

  releaseDrop(token) {
    token?.releaseDrop?.();
    if (token?.releaseDrop) this.dropReleases.delete(token.releaseDrop);
    token.releaseDrop = null;
    this.finishAppend(token);
  }

  finishAppend(token) {
    if (!token) return false;
    const existed = this.appendTokens.delete(token);
    token.finished = true;
    token.controller = null;
    token.signal = null;
    return existed;
  }

  finishRequest(token) {
    if (!token) return false;
    const existed = this.abortableRequests.delete(token);
    token.finished = true;
    if (this.latestAbortable.get(token.key) === token) this.latestAbortable.delete(token.key);
    token.controller = null;
    token.signal = null;
    return existed;
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
