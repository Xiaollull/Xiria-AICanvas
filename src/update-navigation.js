export const UPDATE_RESTART_SESSION_KEY = "xirai-update-restart-return";

/** Statuses in which a task owns the update state and nothing else may take it over.
 *
 * Both the server and the updater page gate on this. It is one list rather than a literal repeated
 * at each gate because a status that is missing from one of them is not a visible bug: it is a
 * second update starting on top of a running one.
 */
export const UPDATE_BUSY_STATUSES = ["uploading", "downloading", "preparing", "applying", "repairing"];

/** Statuses that already own an archive, so starting a fresh one has to be refused. */
export const UPDATE_OCCUPIED_STATUSES = [...UPDATE_BUSY_STATUSES, "ready", "complete"];

export function updateBusy(status) {
  return UPDATE_BUSY_STATUSES.includes(status);
}

export function markUpdateRestart(storage, now = Date.now()) {
  try {
    storage.setItem(UPDATE_RESTART_SESSION_KEY, String(now));
    return true;
  } catch {
    return false;
  }
}

export function clearUpdateRestart(storage) {
  try {
    storage.removeItem(UPDATE_RESTART_SESSION_KEY);
  } catch {}
}

export function updateRestartPending(storage, now = Date.now(), maximumAgeMs = 5 * 60 * 1000) {
  try {
    const startedAt = Number(storage.getItem(UPDATE_RESTART_SESSION_KEY));
    if (Number.isFinite(startedAt) && startedAt > 0 && now - startedAt <= maximumAgeMs) return true;
    storage.removeItem(UPDATE_RESTART_SESSION_KEY);
  } catch {}
  return false;
}

export async function waitForUpdatedApplication({
  checkHealth,
  returnHome,
  wait = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds)),
  now = Date.now,
  timeoutMs = 90000,
  initialDelayMs = 1800,
  retryDelayMs = 600,
}) {
  await wait(initialDelayMs);
  const deadline = now() + timeoutMs;
  while (now() < deadline) {
    try {
      if (await checkHealth()) {
        returnHome();
        return true;
      }
    } catch {}
    await wait(retryDelayMs);
  }
  throw new Error("应用重启超时，请手动双击 Start-XirAI");
}
