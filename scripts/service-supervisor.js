/**
 * When the supervisor brings the service back, and when it gives up.
 *
 * `start.mjs` is a supervisor for a long-running server, and the person it serves is usually
 * nowhere near the terminal it was started from -- a browser on another machine, a phone, a remote
 * instance reached over SSH. So the service exiting is not the same event as the user asking it to
 * stop, and treating every exit as final is what turned a single crash into an unreachable machine
 * with nobody there to notice.
 *
 * The exits that mean something specific -- switch to the configurator, come back from it, restart
 * on request -- are handled by their own codes before this is asked. What reaches here is either a
 * deliberate stop or a crash.
 */

// Restarts are counted inside a moving window, so a service that runs for an hour and then falls
// over is a fresh incident rather than the next strike against an old one.
export const SERVICE_RESTART_WINDOW_MS = 60_000;
export const SERVICE_RESTART_LIMIT = 5;

/**
 * A stop the user asked for. `SIGINT` is Ctrl-C, `SIGTERM` is a deploy script or a service manager
 * asking politely; bringing the service back up in either case would be disobeying an instruction,
 * not surviving a fault.
 */
const DELIBERATE_SIGNALS = new Set(["SIGINT", "SIGTERM", "SIGQUIT"]);

/**
 * Whether to start the main service again after it exited, and why.
 *
 * `history` is the timestamps of earlier restarts; the caller keeps it and passes the pruned list
 * back in. A crash loop -- a broken install, a port that never frees -- stops after
 * `SERVICE_RESTART_LIMIT` attempts inside the window, because restarting forever would bury the
 * error message that says what is actually wrong under an endless scroll of attempts.
 */
export function serviceRestartDecision({ code, signal, shuttingDown = false, history = [], now = Date.now() }) {
  const recent = (Array.isArray(history) ? history : []).filter((at) => Number.isFinite(at) && now - at < SERVICE_RESTART_WINDOW_MS);
  if (shuttingDown) return { restart: false, reason: "shutdown", history: recent };
  if (signal && DELIBERATE_SIGNALS.has(signal)) return { restart: false, reason: "signalled", history: recent };
  const crashed = Boolean(signal) || (Number.isInteger(code) ? code !== 0 : true);
  if (!crashed) return { restart: false, reason: "exited", history: recent };
  if (recent.length >= SERVICE_RESTART_LIMIT) return { restart: false, reason: "crash-loop", history: recent };
  return { restart: true, reason: "crashed", history: [...recent, now], attempt: recent.length + 1 };
}

/** What to tell the user, in the terminal they may or may not be watching. */
export function serviceRestartMessage(decision, { code, signal }) {
  const cause = signal ? `信号 ${signal}` : `退出代码 ${code}`;
  if (decision.restart) {
    return `\n主服务意外退出（${cause}），正在自动重启（第 ${decision.attempt} 次，${SERVICE_RESTART_WINDOW_MS / 1000} 秒内最多 ${SERVICE_RESTART_LIMIT} 次）...`;
  }
  if (decision.reason === "crash-loop") {
    return `\n主服务在 ${SERVICE_RESTART_WINDOW_MS / 1000} 秒内连续退出 ${SERVICE_RESTART_LIMIT} 次（${cause}），已停止自动重启。请查看上方错误信息。`;
  }
  return "";
}
