export const UPDATE_RESTART_SESSION_KEY = "xirai-update-restart-return";

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
