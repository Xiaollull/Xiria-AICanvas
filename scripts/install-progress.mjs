import { spawn } from "node:child_process";
import { watch } from "node:fs";
import { stat } from "node:fs/promises";
import path from "node:path";

import { createSpeedometer } from "./download.mjs";

// Most of what a first-time setup downloads never passes through `download.mjs`. The PyTorch wheel
// does, but its CUDA runtime dependencies, the whole backend requirement set and the npm tree are
// fetched by uv and npm — and neither prints anything at all between "Resolved 42 packages" and
// "Prepared 42 packages", which on a slow link is twenty silent minutes. The configurator showed
// that as a progress bar frozen at 0%, indistinguishable from a hang.
//
// This module gives those phases something true to report: whatever the installer does say is
// parsed, and the bytes it writes are metered directly from the filesystem.

const SIZE_UNITS = {
  b: 1,
  kb: 1000, mb: 1000 ** 2, gb: 1000 ** 3, tb: 1000 ** 4,
  k: 1000, m: 1000 ** 2, g: 1000 ** 3, t: 1000 ** 4,
  kib: 1024, mib: 1024 ** 2, gib: 1024 ** 3, tib: 1024 ** 4,
};

/** Bytes from a printed size. pip and tqdm label powers of 1000 `MB`; uv labels powers of 1024
 *  `MiB`. Both conventions are kept rather than picking one and misreporting the other. */
export function parseSize(value, unit = "B") {
  const factor = SIZE_UNITS[String(unit || "B").toLowerCase()];
  const size = Number(value);
  return Number.isFinite(size) && factor ? Math.round(size * factor) : 0;
}

const ANSI = /\x1b\[[0-?]*[ -/]*[@-~]/g;

/** One line of installer output, as a progress fact or nothing.
 *
 * Four producers write to this stream and each says something different:
 *   uv    `Downloading torch (846.2MiB)` … `Resolved 42 packages in 1.2s`
 *   pip   `Downloading torch-2.9.0-cp312-win_amd64.whl (2.6 GB)`
 *   tqdm  `model.safetensors:  45%|####  | 1.20G/2.70G [00:12<00:14, 42.3MB/s]`  (huggingface_hub)
 *   npm   `added 512 packages in 41s`
 */
export function parseInstallerProgress(line) {
  const text = String(line ?? "").replace(ANSI, "").trim();
  if (!text) return null;

  // huggingface_hub renders one bar per file and rewrites it in place, so this is both the name and
  // the only byte-accurate reading available for the model runtime configs.
  const bar = text.match(/^(.*?):\s*(\d+)%\|[^|]*\|\s*([\d.]+)\s*([KMGT]?i?B?)\s*\/\s*([\d.]+)\s*([KMGT]?i?B?)(?:\s*\[[^\]]*?,\s*([\d.]+)\s*([KMGT]?i?B?)\/s[^\]]*\])?/i);
  if (bar) {
    return {
      kind: "bytes",
      name: bar[1].trim() || "下载中",
      currentBytes: parseSize(bar[3], bar[4]),
      totalBytes: parseSize(bar[5], bar[6]),
      speedBps: bar[7] ? parseSize(bar[7], bar[8]) : 0,
    };
  }

  const downloading = text.match(/^(?:Downloading|正在下载)\s+(\S+?)(?:\s*\(([\d.]+)\s*([KMGT]?i?B)\))?$/i);
  if (downloading) {
    let name = downloading[1];
    try {
      name = decodeURIComponent(path.posix.basename(new URL(name).pathname)) || name;
    } catch {
      name = path.basename(name);
    }
    return { kind: "file", name, totalBytes: downloading[2] ? parseSize(downloading[2], downloading[3]) : 0 };
  }

  const downloaded = text.match(/^Downloaded\s+(\S+)$/i);
  if (downloaded) return { kind: "file-complete", name: downloaded[1] };

  // uv and npm both end each phase with a count, which is the only total either one ever states.
  const stage = text.match(/^(Resolved|Prepared|Installed|Uninstalled|Audited|added|removed|changed)\s+(\d+)\s+packages?\b/i);
  if (stage) return { kind: "stage", stage: stage[1].toLowerCase(), packages: Number(stage[2]) };

  // pip's own transfer line, still produced by the uninstall and repair paths.
  const pip = text.match(/^\s*Downloading\s+(\S+)\s+\(([\d.]+)\s*(kB|MB|GB|KiB|MiB|GiB)\)/i);
  if (pip) return { kind: "file", name: path.basename(pip[1]), totalBytes: parseSize(pip[2], pip[3]) };

  return null;
}

/** A byte meter for directories an installer writes into.
 *
 * Re-walking uv's cache would cost more than the download it is measuring — it holds tens of
 * thousands of extracted files. A recursive watch instead reports only what actually changed, so the
 * cost tracks the activity rather than the size of the tree. Where the platform refuses to watch
 * (inotify limits on a large cache, most often), `available` is false and the caller falls back to
 * what the installer prints.
 */
export function createChangeMeter(directories, { watcher = watch, statFile = stat, limit = 16384 } = {}) {
  const changed = new Set();
  const sizes = new Map();
  const handles = [];
  let available = false;

  for (const directory of directories) {
    try {
      const handle = watcher(directory, { recursive: true, persistent: false }, (_event, filename) => {
        if (!filename || changed.size >= limit) return;
        changed.add(path.join(directory, filename.toString()));
      });
      handle.on?.("error", () => {});
      handles.push(handle);
      available = true;
    } catch {
      // An unwatchable directory is not a failure: the phase still runs, it just reports less.
    }
  }

  return {
    available,
    async sample() {
      const pending = [...changed];
      changed.clear();
      for (const target of pending) {
        // Installing PyTorch creates tens of thousands of files. Past the bound the meter stops
        // learning about new ones and only keeps the ones it already tracks current, so a progress
        // indicator can never grow into a memory problem of its own.
        if (sizes.size >= limit && !sizes.has(target)) continue;
        try {
          const info = await statFile(target);
          if (info.isFile()) sizes.set(target, info.size);
        } catch {
          // Written and moved away between the event and the stat — uv stages every archive.
        }
      }
      let bytes = 0;
      for (const size of sizes.values()) bytes += size;
      return { bytes, files: sizes.size };
    },
    close() {
      for (const handle of handles) {
        try {
          handle.close();
        } catch {}
      }
      handles.length = 0;
    },
  };
}

/** Build the runner the configurator uses for every installer it shells out to.
 *
 * `spawnSync` cannot report anything: it blocks the whole process, so between the moment uv starts
 * and the moment it exits there is nothing to emit — which is precisely the window the user is
 * watching, and the reason the download panel used to sit frozen for twenty minutes. Running the
 * child asynchronously buys three things: the panel keeps moving, a mirror that has gone silent can
 * be told apart from one that is merely slow, and the installer's own output is read rather than
 * only forwarded.
 *
 * Everything it touches is injected so the stall, timeout and stream handling can be tested without
 * a network, a Python environment or a real installer.
 */
export function createInstallerRunner({
  emit,
  cwd,
  env = process.env,
  concurrency = 1,
  activeTask = () => undefined,
  spawnProcess = spawn,
  createMeter = createChangeMeter,
  speedometer = createSpeedometer,
  clock = Date.now,
  stdout = process.stdout,
  stderr = process.stderr,
  intervalMs = 1000,
  killGraceMs = 10000,
  log = (message) => console.log(message),
}) {
  return function runInstaller({ command, args, options = {}, taskId, label, meterDirectories = [], stallMs = 0, timeoutMs = 0 }) {
    log(`\n> ${command} ${args.join(" ")}`);
    const meter = createMeter(meterDirectories);
    const speed = speedometer();
    const startedAt = clock();
    let baselineBytes = null;
    let observedBytes = 0;
    let lastActivityAt = clock();
    let currentName = label;
    let currentTotal = 0;
    let currentBytes = 0;
    let fileBaselineBytes = 0;
    let packages = 0;
    let stalled = false;
    let killed = false;

    return new Promise((resolve) => {
      const child = spawnProcess(command, args, {
        cwd,
        env,
        stdio: ["ignore", "pipe", "pipe"],
        windowsHide: true,
        ...options,
      });

      const absorb = (line) => {
        const event = parseInstallerProgress(line);
        if (!event) return;
        if (event.kind === "file") {
          currentName = event.name;
          currentTotal = event.totalBytes;
          currentBytes = 0;
          // uv names a file and its size but never its progress. The filesystem meter supplies that,
          // counted from here so the reading belongs to this file rather than to the whole phase.
          fileBaselineBytes = observedBytes;
        } else if (event.kind === "bytes") {
          currentName = event.name;
          currentTotal = event.totalBytes;
          currentBytes = event.currentBytes;
        } else if (event.kind === "stage") {
          packages = event.packages;
        }
      };

      // Whole lines only. The configurator's own events go to this same stdout, and a forwarded
      // chunk that stopped mid-line would leave `@@XIRAI_SETUP@@{...}` glued to the end of it — the
      // GUI matches that marker at the start of a line, so the event would be lost as log text.
      const consume = (stream, target) => {
        let buffer = "";
        stream.setEncoding?.("utf8");
        stream.on("data", (chunk) => {
          lastActivityAt = clock();
          buffer += chunk;
          const lines = buffer.split(/[\r\n]+/);
          buffer = lines.pop() || "";
          if (!lines.length) return;
          target.write(`${lines.join("\n")}\n`);
          for (const line of lines) absorb(line);
        });
        stream.on("end", () => {
          if (!buffer) return;
          target.write(`${buffer}\n`);
          absorb(buffer);
          buffer = "";
        });
      };
      consume(child.stdout, stdout);
      consume(child.stderr, stderr);

      const ticker = setInterval(async () => {
        const sample = meter.available ? await meter.sample() : { bytes: 0 };
        if (meter.available) {
          baselineBytes ??= sample.bytes;
          const written = Math.max(0, sample.bytes - baselineBytes);
          if (written > observedBytes) lastActivityAt = clock();
          observedBytes = written;
        }
        // A named file with a stated size is the better readout; the filesystem meter is what keeps
        // the panel alive for everything else, and it reports no total because none exists.
        const named = currentTotal > 0;
        const reported = named
          ? Math.min(currentTotal, currentBytes || Math.max(0, observedBytes - fileBaselineBytes))
          : observedBytes;
        speed.record(named && currentBytes ? currentBytes : observedBytes, clock());
        emit("download", {
          id: taskId || activeTask(),
          name: currentName || label,
          currentBytes: reported,
          totalBytes: named ? currentTotal : 0,
          speedBps: speed.speed(clock()),
          connections: concurrency,
          // No installer states a total for the phase as a whole, so the panel is told to show that
          // it is moving rather than to pretend a percentage it does not have.
          indeterminate: !named,
          packages,
          elapsedSeconds: Math.round((clock() - startedAt) / 1000),
        });
        const expired = Boolean(timeoutMs) && clock() - startedAt > timeoutMs;
        // Once, not once a second: the condition stays true until the child actually exits, and
        // re-signalling every tick would bury the log in duplicate warnings.
        if (!killed && (expired || (stallMs && clock() - lastActivityAt > stallMs))) {
          killed = true;
          stalled = !expired;
          emit("warning", {
            message: expired
              ? `${label} 超过 ${Math.round(timeoutMs / 1000)} 秒仍未完成，已停止`
              : `${label} 已经 ${Math.round(stallMs / 1000)} 秒没有任何进展，正在结束当前线路并改用其他镜像`,
          });
          child.kill();
          // A wedged download can ignore the first signal; do not leave setup waiting on it forever.
          setTimeout(() => child.kill("SIGKILL"), killGraceMs).unref?.();
        }
      }, intervalMs);

      const settle = (status) => {
        clearInterval(ticker);
        meter.close();
        resolve({ status, stalled, bytes: observedBytes });
      };
      child.on("error", (error) => {
        emit("warning", { message: `${label} 无法启动：${error.message}` });
        settle(null);
      });
      child.on("close", (code) => settle(code == null ? null : code));
    });
  };
}
