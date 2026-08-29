/**
 * Latest-wins guard for the dedicated LoRA page's merged ui-state write.
 * A lock, unmount or newer local/incoming state invalidates every continuation
 * before it may issue a GET or PUT. It intentionally never mutates LoRA data.
 */
export function createLoraPersistenceEpochGuard() {
  let epoch = 0;
  let mounted = true;
  let controller = null;
  const invalidate = () => {
    epoch += 1;
    controller?.abort();
    controller = null;
    return epoch;
  };
  return {
    nextEpoch: invalidate,
    invalidate,
    mount() { mounted = true; },
    unmount() { mounted = false; invalidate(); },
    isCurrent(token, { locked = false, syncReady = false } = {}) {
      return mounted && token === epoch && locked !== true && syncReady === true;
    },
    controllerFor(token, admission) {
      if (!this.isCurrent(token, admission())) return null;
      const next = new AbortController();
      controller = next;
      return next;
    },
    release(candidate) { if (controller === candidate) controller = null; },
  };
}

/** Executes every async persistence boundary under the same latest-wins gate. */
export async function runLoraPersistenceEpoch({ guard, epoch, admission, get, prepare, put } = {}) {
  const current = () => guard?.isCurrent(epoch, admission?.());
  if (!current()) return { written: false, stale: true };
  const controller = guard.controllerFor(epoch, admission);
  if (!controller) return { written: false, stale: true };
  try {
    const state = await get(controller.signal);
    if (!current()) return { written: false, stale: true };
    const request = await prepare(state);
    if (!current()) return { written: false, stale: true };
    const response = await put(request, controller.signal);
    if (!current()) return { written: false, stale: true };
    return { written: response?.ok === true, stale: false };
  } catch (error) {
    if (!current() || error?.name === "AbortError") return { written: false, stale: true };
    return { written: false, stale: false, error };
  } finally {
    guard.release(controller);
  }
}
