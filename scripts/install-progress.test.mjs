import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

import { createChangeMeter, createInstallerRunner, parseInstallerProgress, parseSize } from "./install-progress.mjs";

const readSource = (file) => readFile(new URL(`../${file}`, import.meta.url), "utf8");

test("printed sizes keep the unit convention the tool that wrote them uses", () => {
  // pip and tqdm label powers of 1000; uv labels powers of 1024. Reading both as the same thing
  // would misreport one of them by 7% at gigabyte scale.
  assert.equal(parseSize("2.6", "GB"), 2_600_000_000);
  assert.equal(parseSize("846.2", "MiB"), 887_305_011);
  assert.equal(parseSize("512", "kB"), 512_000);
  assert.equal(parseSize("1.5", "G"), 1_500_000_000);
  assert.equal(parseSize("12", "not-a-unit"), 0);
  assert.equal(parseSize("nonsense", "MB"), 0);
});

test("uv's transfer lines become a named file with a size", () => {
  assert.deepEqual(parseInstallerProgress("Downloading torch (846.2MiB)"), {
    kind: "file",
    name: "torch",
    totalBytes: parseSize("846.2", "MiB"),
  });
  // uv says nothing about size for a small wheel; the name alone is still worth showing.
  assert.deepEqual(parseInstallerProgress("Downloading filelock"), { kind: "file", name: "filelock", totalBytes: 0 });
  assert.deepEqual(parseInstallerProgress("Downloaded torch"), { kind: "file-complete", name: "torch" });
});

test("the package counts uv and npm print are the only totals either one states", () => {
  assert.deepEqual(parseInstallerProgress("Resolved 42 packages in 1.23s"), { kind: "stage", stage: "resolved", packages: 42 });
  assert.deepEqual(parseInstallerProgress("Prepared 42 packages in 30.12s"), { kind: "stage", stage: "prepared", packages: 42 });
  assert.deepEqual(parseInstallerProgress("Installed 1 package in 15ms"), { kind: "stage", stage: "installed", packages: 1 });
  assert.deepEqual(parseInstallerProgress("added 512 packages in 41s"), { kind: "stage", stage: "added", packages: 512 });
});

test("pip's own wording still reads, because uninstall and repair paths still use pip", () => {
  assert.deepEqual(
    parseInstallerProgress("  Downloading torch-2.9.0-cp312-cp312-win_amd64.whl (2.6 GB)"),
    { kind: "file", name: "torch-2.9.0-cp312-cp312-win_amd64.whl", totalBytes: 2_600_000_000 },
  );
});

test("a huggingface_hub bar is the one byte-accurate reading the config prefetch produces", () => {
  const event = parseInstallerProgress("model.safetensors:  45%|####      | 1.20G/2.70G [00:12<00:14, 42.3MB/s]");
  assert.equal(event.kind, "bytes");
  assert.equal(event.name, "model.safetensors");
  assert.equal(event.currentBytes, 1_200_000_000);
  assert.equal(event.totalBytes, 2_700_000_000);
  assert.equal(event.speedBps, 42_300_000);
});

test("colour codes and ordinary log noise are not mistaken for progress", () => {
  assert.deepEqual(parseInstallerProgress("[2mResolved[0m 7 packages in 90ms"), {
    kind: "stage", stage: "resolved", packages: 7,
  });
  assert.equal(parseInstallerProgress(""), null);
  assert.equal(parseInstallerProgress(null), null);
  assert.equal(parseInstallerProgress("warning: `uv pip install` is experimental"), null);
  assert.equal(parseInstallerProgress(" + torch==2.9.0"), null);
});

function fakeWatcher(registry) {
  return (directory, options, listener) => {
    if (registry.unwatchable?.has(directory)) throw new Error("ENOSPC");
    const handle = new EventEmitter();
    handle.close = () => registry.closed.push(directory);
    registry.listeners.push({ directory, options, listener });
    return handle;
  };
}

test("the byte meter counts only what the installer actually touched", async () => {
  const registry = { listeners: [], closed: [] };
  const files = new Map();
  const meter = createChangeMeter(["/cache", "/venv"], {
    watcher: fakeWatcher(registry),
    statFile: async (target) => {
      if (!files.has(target)) throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
      return { isFile: () => true, size: files.get(target) };
    },
  });
  assert.equal(meter.available, true);
  // Recursive is the whole point: re-walking a uv cache of tens of thousands of extracted files
  // every second would cost more than the download it is measuring.
  assert.equal(registry.listeners[0].options.recursive, true);
  assert.equal(registry.listeners[0].options.persistent, false);

  const [cache, venv] = registry.listeners;
  files.set(path.join("/cache", "wheel.whl"), 400);
  cache.listener("rename", "wheel.whl");
  assert.deepEqual(await meter.sample(), { bytes: 400, files: 1 });

  // The same file growing is re-stated, not added twice.
  files.set(path.join("/cache", "wheel.whl"), 900);
  cache.listener("change", "wheel.whl");
  files.set(path.join("/venv", "torch", "_C.pyd"), 100);
  venv.listener("rename", path.join("torch", "_C.pyd"));
  assert.deepEqual(await meter.sample(), { bytes: 1000, files: 2 });

  // uv stages every archive and moves it away; a file that vanished before the stat is skipped
  // rather than ending the phase.
  cache.listener("rename", "staging-tmp");
  assert.deepEqual(await meter.sample(), { bytes: 1000, files: 2 });

  meter.close();
  assert.deepEqual(registry.closed, ["/cache", "/venv"]);
});

test("a directory the platform refuses to watch degrades to no meter rather than to a crash", async () => {
  const registry = { listeners: [], closed: [], unwatchable: new Set(["/cache"]) };
  const meter = createChangeMeter(["/cache"], {
    watcher: fakeWatcher(registry),
    statFile: async () => ({ isFile: () => true, size: 1 }),
  });
  // inotify limits on a large cache are a real outcome on Linux; the phase still runs and simply
  // reports what the installer prints.
  assert.equal(meter.available, false);
  assert.deepEqual(await meter.sample(), { bytes: 0, files: 0 });
  meter.close();
});

test("watch events stop accumulating once the bound is reached", async () => {
  const registry = { listeners: [], closed: [] };
  const meter = createChangeMeter(["/cache"], {
    watcher: fakeWatcher(registry),
    statFile: async () => ({ isFile: () => true, size: 10 }),
    limit: 3,
  });
  for (let index = 0; index < 50; index += 1) registry.listeners[0].listener("rename", `file-${index}`);
  assert.deepEqual(await meter.sample(), { bytes: 30, files: 3 });
  meter.close();
});

test("a silent mirror is treated as a failed one, but only when there is somewhere to go", async () => {
  const setup = await readSource("scripts/setup.mjs");
  assert.match(setup, /const installStallMs = /);
  // The stall watchdog is armed per attempt, and deliberately not on the last route: killing a slow
  // but living install when there is no alternative left is strictly worse than waiting.
  assert.match(setup, /stallMs: stall \? installStallMs : 0/);
  assert.match(setup, /stall: index \+ 1 < routeNames\.length/);
  assert.match(setup, /result\?\.stalled \? "长时间无响应" : "失败"/);
});

test("mirrors are ranked on one large body, so the reading is bandwidth and not round-trip time", async () => {
  const setup = await readSource("scripts/setup.mjs");
  assert.match(setup, /const BANDWIDTH_PROBE = \{ pip: "torch\/", npm: "react" \}/);
  assert.match(setup, /sampleBytes: 1024 \*\* 2, sampleMs: 2000/);
  // The four-second latency probe this replaced measured who answered first, which is not the
  // question when the answer decides where gigabytes come from.
  assert.doesNotMatch(setup, /function probeSource\(/);
});

test("a parsed log line adds to the download panel instead of replacing it", async () => {
  const gui = await readSource("scripts/setup-gui.mjs");
  // The reader this replaced rebuilt `currentDownload` from scratch whenever it matched, wiping the
  // route, mode and connection count the panel shows beside the bar — so a recognised line made the
  // display worse than an unrecognised one.
  assert.match(gui, /import \{ parseInstallerProgress \} from "\.\/install-progress\.mjs"/);
  const reader = gui.slice(gui.indexOf("function parseProgressLine"), gui.indexOf("async function saveSetupMarker"));
  assert.match(reader, /const event = parseInstallerProgress\(line\)/);
  for (const branch of ['event.kind === "file"', 'event.kind === "bytes"', 'event.kind === "stage"']) {
    assert.ok(reader.includes(branch), `missing branch: ${branch}`);
  }
  // Three spreads, one per branch, each preserving what the event does not carry.
  assert.equal((reader.match(/\.\.\.state\.currentDownload,/g) || []).length, 3);
  assert.match(reader, /scheduleSetupStatePersistence\(\)/);
});

test("a phase with no stated total shows that it is moving, not that it is at zero", async () => {
  const ui = await readSource("scripts/setup-ui.html");
  assert.match(ui, /\.bar\.indeterminate i \{[^}]*animation:barSlide/);
  assert.match(ui, /@media \(prefers-reduced-motion:reduce\) \{ \.bar\.indeterminate i \{[^}]*animation:none/);
  const render = ui.slice(ui.indexOf("const download = state.currentDownload"), ui.indexOf("const logs = state.logs"));
  assert.match(render, /const indeterminate = !download\.totalBytes && \(download\.indeterminate === true \|\| download\.currentBytes > 0\)/);
  assert.match(render, /elements\.downloadBarTrack\.className = indeterminate \? "bar indeterminate" : "bar"/);
  assert.match(render, /indeterminate \? "进行中" : `\$\{downloadPercent\}%`/);
  // A multi-source transfer names every mirror carrying it rather than only the one that won the
  // benchmark.
  assert.match(render, /sources\.length > 1/);
  assert.match(render, /elements\.downloadEta\.textContent/);
});

test("the setup GUI carries the fields the multi-source engine reports", async () => {
  const gui = await readSource("scripts/setup-gui.mjs");
  const handler = gui.slice(gui.indexOf('} else if (event.type === "download")'), gui.indexOf('} else if (event.type === "warning")'));
  for (const field of ["indeterminate", "etaSeconds", "packages", "sources"]) {
    assert.ok(handler.includes(field), `the download event drops ${field}`);
  }
});

function fakeChild() {
  const child = new EventEmitter();
  child.stdout = new EventEmitter();
  child.stderr = new EventEmitter();
  child.signals = [];
  child.kill = (signal = "SIGTERM") => child.signals.push(signal);
  return child;
}

function collector() {
  const written = [];
  return { written, write: (text) => written.push(text) };
}

function runnerHarness(t, { meterBytes = () => 0, clock, ...overrides } = {}) {
  const events = [];
  const child = fakeChild();
  const out = collector();
  const err = collector();
  const runInstaller = createInstallerRunner({
    emit: (type, payload) => events.push({ type, ...payload }),
    cwd: "/project",
    concurrency: 12,
    spawnProcess: () => child,
    createMeter: () => ({ available: true, sample: async () => ({ bytes: meterBytes() }), close() {} }),
    clock,
    stdout: out,
    stderr: err,
    intervalMs: 5,
    killGraceMs: 5,
    log: () => {},
    ...overrides,
  });
  // A failed assertion must not leave the ticker running: the child would never close, the runner's
  // promise would never settle, and the test runner would wait on a handle that never clears.
  t.after(() => child.emit("close", 0));
  return { runInstaller, child, events, out, err };
}

const settle = (ms = 40) => new Promise((resolve) => setTimeout(resolve, ms));

test("installer output reaches the log only as whole lines", async (t) => {
  // The configurator's own `@@XIRAI_SETUP@@` events share this stdout. A chunk forwarded mid-line
  // would leave the next event glued to its tail, and the GUI matches that marker only at the start
  // of a line — the task, route and progress events for the whole phase would vanish into the log.
  const { runInstaller, child, out } = runnerHarness(t);
  const pending = runInstaller({ command: "uv", args: ["pip", "install"], label: "后端依赖" });
  child.stdout.emit("data", "Resolved 42 packages in 1.2s\nDownloa");
  assert.deepEqual(out.written, ["Resolved 42 packages in 1.2s\n"]);
  child.stdout.emit("data", "ding torch (846.2MiB)\n");
  assert.deepEqual(out.written, ["Resolved 42 packages in 1.2s\n", "Downloading torch (846.2MiB)\n"]);
  // A last line with no terminator is still flushed rather than swallowed.
  child.stdout.emit("data", "Prepared 42 packages in 30s");
  child.stdout.emit("end");
  assert.equal(out.written.at(-1), "Prepared 42 packages in 30s\n");
  child.emit("close", 0);
  assert.deepEqual(await pending, { status: 0, stalled: false, bytes: 0 });
});

test("a phase with nothing to report still reports the bytes it is writing", async (t) => {
  // uv prints nothing between "Resolved" and "Prepared". On a slow link that is twenty minutes of
  // silence, which the panel used to render as a bar frozen at 0%.
  let bytes = 0;
  const { runInstaller, child, events } = runnerHarness(t, { meterBytes: () => bytes });
  const pending = runInstaller({ command: "uv", args: [], label: "后端依赖", meterDirectories: ["/cache"] });
  // The first sample is the baseline: only what the phase writes from here on is its own.
  await settle(15);
  bytes = 40 * 1024 ** 2;
  await settle();
  const indeterminate = events.filter((event) => event.type === "download").at(-1);
  assert.equal(indeterminate.indeterminate, true);
  assert.equal(indeterminate.totalBytes, 0);
  assert.ok(indeterminate.currentBytes > 0, "the meter reported no bytes");
  assert.equal(indeterminate.name, "后端依赖");

  // Once uv names a file and its size, the reading is attributed to that file rather than to
  // everything the phase has written so far.
  child.stdout.emit("data", "Downloading torch (100MB)\n");
  bytes = 40 * 1024 ** 2 + 30 * 1000 ** 2;
  await settle();
  const named = events.filter((event) => event.type === "download").at(-1);
  assert.equal(named.indeterminate, false);
  assert.equal(named.name, "torch");
  assert.equal(named.totalBytes, 100 * 1000 ** 2);
  assert.equal(named.currentBytes, 30 * 1000 ** 2);
  child.emit("close", 0);
  await pending;
});

test("a mirror that goes silent is ended once and reported as stalled", async (t) => {
  let now = 0;
  const { runInstaller, child, events } = runnerHarness(t, { clock: () => now });
  const pending = runInstaller({
    command: "uv", args: [], label: "PyTorch 2.9.0", meterDirectories: ["/cache"], stallMs: 1000,
  });
  now = 500;
  await settle();
  assert.deepEqual(child.signals, [], "ended a mirror that was still within its grace period");

  now = 5000;
  await settle();
  // Ended once, not once per tick: the condition stays true until the child actually exits.
  assert.equal(child.signals.filter((signal) => signal === "SIGTERM").length, 1);
  assert.equal(events.filter((event) => event.type === "warning").length, 1);
  assert.match(events.find((event) => event.type === "warning").message, /没有任何进展/);

  child.emit("close", null);
  const result = await pending;
  // `stalled` is what tells the caller to try the next mirror rather than to give up.
  assert.equal(result.stalled, true);
  assert.equal(result.status, null);
});

test("output or bytes keep a slow mirror alive; only silence ends it", async (t) => {
  let now = 0;
  let bytes = 0;
  const { runInstaller, child } = runnerHarness(t, { clock: () => now, meterBytes: () => bytes });
  const pending = runInstaller({
    command: "uv", args: [], label: "后端依赖", meterDirectories: ["/cache"], stallMs: 1000,
  });
  for (const step of [600, 1200, 1800, 2400]) {
    now = step;
    bytes += 1024 ** 2;
    child.stdout.emit("data", `still working at ${step}\n`);
    await settle(15);
  }
  // Four grace periods have passed, and the link never stopped delivering. A slow mirror is not a
  // dead one, and killing it would throw away everything it has fetched.
  assert.deepEqual(child.signals, []);
  child.emit("close", 0);
  assert.equal((await pending).stalled, false);
});

test("a hard timeout is not a stalled mirror, so it does not send the caller to the next one", async (t) => {
  let now = 0;
  const { runInstaller, child, events } = runnerHarness(t, { clock: () => now });
  const pending = runInstaller({ command: "python", args: [], label: "模型运行配置", timeoutMs: 1000 });
  now = 5000;
  await settle();
  // SIGTERM once, with the SIGKILL follow-up behind it for a child that ignores the first signal.
  assert.equal(child.signals.filter((signal) => signal === "SIGTERM").length, 1);
  child.emit("close", null);
  const result = await pending;
  assert.equal(result.stalled, false);
  assert.match(events.find((event) => event.type === "warning").message, /仍未完成/);
});

test("an installer that cannot start is reported rather than thrown", async (t) => {
  const { runInstaller, child, events } = runnerHarness(t);
  const pending = runInstaller({ command: "missing", args: [], label: "xformers" });
  child.emit("error", new Error("spawn ENOENT"));
  const result = await pending;
  assert.equal(result.status, null);
  assert.match(events.find((event) => event.type === "warning").message, /无法启动/);
});

test("the configurator builds its runner from this module rather than inlining one", async () => {
  const setup = await readSource("scripts/setup.mjs");
  assert.match(setup, /import \{ createInstallerRunner \} from "\.\/install-progress\.mjs"/);
  assert.match(setup, /const runInstaller = createInstallerRunner\(\{\s*\n\s*emit,/);
  // Every uv install is awaited: a forgotten `await` would hand the caller a pending promise, whose
  // `status` is undefined — reported as a failed install of software that is still downloading.
  assert.doesNotMatch(setup, /(?:=|return)\s*uvInstall\(uvExecutable/);
  assert.equal((setup.match(/await uvInstall\(uvExecutable/g) || []).length, 6);
  for (const phase of ["node_modules", "uvCacheDirectory\\(\\)", "configuredHuggingFaceHome"]) {
    assert.match(setup, new RegExp(`meterDirectories: \\[[^\\]]*${phase}`));
  }
});
