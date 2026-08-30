import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { networkInterfaces } from "node:os";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { writeDiagnosticLog } from "./diagnostics.mjs";
import { loadLocalEnv, findPython, isolatedPythonEnv, runPython } from "./python.mjs";
import { viteInvocation } from "./node-tools.mjs";
import { getSetupMarkerPath, readSetupMarker, writeSetupMarker } from "./setup-state.mjs";
import { detectNvidiaDriverApi, detectNvidiaSmi, mergeNvidiaDetection } from "./nvidia.mjs";
import { buildSetupArguments, parseCudaVariants, parseTorchWheelVersions, buildSetupCatalog, recommendRepairPlan, validateSetupConfiguration } from "./setup-options.mjs";
import { parseInstallerProgress } from "./install-progress.mjs";
import { applyInstallerProxyPolicy, createInstallerFetch } from "./proxy-policy.mjs";

const projectRoot = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
Object.assign(process.env, loadLocalEnv(projectRoot));
// Captured before the policy strips this process: the direct-connection rule covers installer
// downloads, not the application the wizard hands over to. XiriaCanvas AI has its own proxy
// handling for model downloads, and it must still see the proxy the user configured.
const applicationEnvironment = { ...process.env };
const proxyPolicy = applyInstallerProxyPolicy();
const fetchWithProxy = createInstallerFetch(proxyPolicy);
const uiPath = path.join(projectRoot, "scripts", "setup-ui.html");
const logoPath = path.join(projectRoot, "public", "xiriacanvas-logo.svg");
const stateDirectory = path.resolve(projectRoot, process.env.XIRAI_STATE_DIR || "state-cache");
const uiStatePath = path.join(stateDirectory, "ui-state.json");
const setupProgressPath = path.join(stateDirectory, "setup-progress.json");
const eventPrefix = "@@XIRAI_SETUP@@";
const setupMarkerPath = getSetupMarkerPath(projectRoot);
const webPort = Number(process.env.WEB_PORT || 7709);
const webHost = process.env.WEB_HOST || "0.0.0.0";
if (!Number.isInteger(webPort) || webPort < 1 || webPort > 65535) throw new Error("WEB_PORT 必须是 1-65535 的整数");


function accessUrls() {
  const addresses = new Set([`http://localhost:${webPort}/`]);
  for (const interfaces of Object.values(networkInterfaces())) {
    for (const item of interfaces || []) {
      if (item.family === "IPv4" && !item.internal) addresses.add(`http://${item.address}:${webPort}/`);
    }
  }
  return [...addresses];
}

const forwardedArguments = process.argv.slice(2).filter((argument) => {
  const key = argument.replace(/^--/, "").split("=", 2)[0];
  return !["no-open", "return-to-app", "supervised", "exit-on-complete", "torch", "torch-version", "without-xformers", "with-rtx-vsr", "without-rtx-vsr", "refresh-selection"].includes(key);
});
const noOpen = process.argv.includes("--no-open");
const returnToApp = process.argv.includes("--return-to-app");
const supervised = process.argv.includes("--supervised");
const exitOnComplete = process.argv.includes("--exit-on-complete");
const diagnoseOnly = forwardedArguments.includes("--diagnose");

const existingSetupMarker = readSetupMarker(projectRoot);
const state = {
  status: "awaiting-choice",
  phase: "选择环境配置方式",
  message: "配置器不会在确认前安装或修改环境",
  catalogStatus: "loading",
  machine: null,
  recommended: null,
  configuration: existingSetupMarker ? {
    installRtxVsr: existingSetupMarker.selection?.rtxVsr === true,
    installTriton: existingSetupMarker.selection?.triton === true,
    installSageAttention: existingSetupMarker.selection?.sageAttention === true,
  } : null,
  features: existingSetupMarker?.features || {},
  repair: null,
  repairAttempted: false,
  autoRepairScheduled: false,
  tasks: [],
  selection: null,
  downloads: [],
  currentDownload: { name: "等待选择", currentBytes: 0, totalBytes: 0, speedBps: 0, connections: 0, route: "等待测速", mode: "自动选择" },
  logs: [],
  warnings: [],
  startedAt: Date.now(),
  updatedAt: Date.now(),
  configured: Boolean(existingSetupMarker),
  restarting: false,
  returnToApp,
  accessUrls: accessUrls(),
};

let setupCatalog = null;
let setupProcess;
let server;
let appProcess;
let pendingEvent = Promise.resolve();
let setupFailureLogged = false;
let setupPersistTimer;
let setupStateWrite = Promise.resolve();

// Migration source only: the default the withdrawn 2026-08-15 visual pass shipped. See src/theme.js.
const withdrawnDefaultTheme = { overall: "#D96846", accent: "#D96846", ink: "#111112", disc: "#FFF9F2", guide: "#E3C7BD" };
const defaultTheme = { overall: "#8B7CFF", accent: "#8B7CFF", ink: "#111112", disc: "#FFFFFF", guide: "#C9C5E8", light: false };

function normalizedThemeHex(value, fallback) {
  const match = typeof value === "string" ? value.trim().match(/^#?([0-9a-f]{6})$/i) : null;
  return match ? `#${match[1].toUpperCase()}` : fallback;
}

async function readPersistedTheme() {
  try {
    const saved = JSON.parse(await readFile(uiStatePath, "utf8"))?.theme;
    if (!saved || typeof saved !== "object") return defaultTheme;
    const legacyDefault = normalizedThemeHex(saved.overall, defaultTheme.overall) === "#D6FF3F"
      && normalizedThemeHex(saved.accent, defaultTheme.accent) === "#D6FF3F";
    const withdrawnDefault = normalizedThemeHex(saved.overall, defaultTheme.overall) === withdrawnDefaultTheme.overall
      && normalizedThemeHex(saved.accent, defaultTheme.accent) === withdrawnDefaultTheme.accent
      && normalizedThemeHex(saved.ink, defaultTheme.ink) === withdrawnDefaultTheme.ink
      && normalizedThemeHex(saved.disc, defaultTheme.disc) === withdrawnDefaultTheme.disc
      && normalizedThemeHex(saved.guide, defaultTheme.guide) === withdrawnDefaultTheme.guide;
    if (legacyDefault || withdrawnDefault) return { ...defaultTheme, light: saved.light === true || saved.monochrome === true };
    return {
      overall: normalizedThemeHex(saved.overall, defaultTheme.overall),
      accent: normalizedThemeHex(saved.accent, defaultTheme.accent),
      ink: normalizedThemeHex(saved.ink, defaultTheme.ink),
      disc: normalizedThemeHex(saved.disc, defaultTheme.disc),
      guide: normalizedThemeHex(saved.guide, defaultTheme.guide),
      light: saved.light === true || saved.monochrome === true,
    };
  } catch {
    return defaultTheme;
  }
}

function themeFromPalette(value) {
  const colors = typeof value === "string" ? value.split("-") : [];
  if (colors.length !== 4 || colors.some((color) => !/^[0-9a-f]{6}$/i.test(color))) return null;
  return {
    ...defaultTheme,
    accent: `#${colors[0].toUpperCase()}`,
    ink: `#${colors[1].toUpperCase()}`,
    disc: `#${colors[2].toUpperCase()}`,
    guide: `#${colors[3].toUpperCase()}`,
  };
}

function themedLogo(svg, theme) {
  return svg
    .replaceAll("#8b7cff", "__XIRIA_LOGO_ACCENT__")
    .replaceAll("#c9c5e8", "__XIRIA_LOGO_GUIDE__")
    .replaceAll("#111112", "__XIRIA_LOGO_INK__")
    .replace('<circle cx="512" cy="512" r="434" fill="#ffffff"', '<circle cx="512" cy="512" r="434" fill="__XIRIA_LOGO_DISC__"')
    .replaceAll("__XIRIA_LOGO_ACCENT__", theme.accent.toLowerCase())
    .replaceAll("__XIRIA_LOGO_GUIDE__", theme.guide.toLowerCase())
    .replaceAll("__XIRIA_LOGO_INK__", theme.ink.toLowerCase())
    .replaceAll("__XIRIA_LOGO_DISC__", theme.disc.toLowerCase());
}

function updateState(updates = {}) {
  Object.assign(state, updates, { updatedAt: Date.now() });
  scheduleSetupStatePersistence();
}

function appendLog(text, stream = "stdout") {
  const clean = text.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "").trim();
  if (!clean) return;
  state.logs.push({ text: clean, stream, at: Date.now() });
  if (state.logs.length > 500) state.logs.splice(0, state.logs.length - 500);
  state.updatedAt = Date.now();
  scheduleSetupStatePersistence();
}

function persistSetupState() {
  const snapshot = { schema: 1, state: { ...state, logs: state.logs.slice(-500) } };
  const content = `${JSON.stringify(snapshot, null, 2)}\n`;
  setupStateWrite = setupStateWrite
    .then(async () => {
      await mkdir(stateDirectory, { recursive: true });
      const temporaryPath = path.join(stateDirectory, `${process.pid}-${Date.now()}-setup.tmp`);
      try {
        await writeFile(temporaryPath, content, { encoding: "utf8", flag: "wx" });
        await rename(temporaryPath, setupProgressPath);
      } finally {
        await rm(temporaryPath, { force: true });
      }
    })
    .catch((error) => console.warn(`Unable to save setup progress: ${error.message}`));
  return setupStateWrite;
}

function scheduleSetupStatePersistence() {
  if (setupPersistTimer) return;
  setupPersistTimer = setTimeout(() => {
    setupPersistTimer = null;
    void persistSetupState();
  }, 500);
}

async function restoreSetupState() {
  try {
    const saved = JSON.parse(await readFile(setupProgressPath, "utf8"));
    if (!saved || saved.schema !== 1 || !saved.state || typeof saved.state !== "object") return;
    if (!["running", "error", "catalog-error"].includes(saved.state.status)) return;
    Object.assign(state, saved.state, {
      configured: Boolean(readSetupMarker(projectRoot)),
      returnToApp,
      accessUrls: accessUrls(),
      updatedAt: Date.now(),
    });
    if (["running", "catalog-error"].includes(state.status)) {
      state.status = "error";
      state.phase = "配置器已重新打开";
      state.message = "之前的配置进程已停止；安装器会保留已完成的断点，点击继续即可恢复。";
    }
  } catch (error) {
    if (error.code !== "ENOENT") console.warn(`Unable to restore setup progress: ${error.message}`);
  }
}

function recordSetupFailure(message, details = {}) {
  if (setupFailureLogged) return;
  setupFailureLogged = true;
  writeDiagnosticLog(projectRoot, {
    kind: "setup-failure",
    message,
    details: {
      phase: state.phase,
      selection: state.selection,
      machine: state.machine,
      warnings: state.warnings.slice(-20),
      recent_output: state.logs.slice(-120).map((entry) => `[${entry.stream}] ${entry.text}`).join("\n"),
      ...details,
    },
  });
}

/** Progress facts recovered from plain installer output.
 *
 * Only the two files `download.mjs` fetches ever emit structured `download` events; everything uv,
 * npm and huggingface_hub download arrives here as ordinary log text. The old reader recognised
 * pip's wording only — which setup no longer uses — and replaced the whole `currentDownload` record
 * whenever it did match, wiping the route, mode and connection count that the panel shows beside it.
 */
function parseProgressLine(line) {
  const event = parseInstallerProgress(line);
  if (!event) return;
  if (event.kind === "file") {
    state.currentDownload = {
      ...state.currentDownload,
      name: event.name,
      currentBytes: 0,
      totalBytes: event.totalBytes,
      indeterminate: !event.totalBytes,
    };
  } else if (event.kind === "bytes") {
    state.currentDownload = {
      ...state.currentDownload,
      name: event.name,
      currentBytes: event.currentBytes,
      totalBytes: event.totalBytes,
      speedBps: event.speedBps || state.currentDownload.speedBps || 0,
      indeterminate: !event.totalBytes,
    };
  } else if (event.kind === "stage") {
    state.currentDownload = { ...state.currentDownload, packages: event.packages, stage: event.stage };
  } else {
    return;
  }
  state.updatedAt = Date.now();
  scheduleSetupStatePersistence();
}

async function saveSetupMarker(event) {
  if (diagnoseOnly) return;
  writeSetupMarker(projectRoot, {
    complete: true,
    product: "XiriaCanvas AI",
    completed_at: new Date().toISOString(),
    selection: {
      ...state.selection,
      rtxVsr: state.configuration?.installRtxVsr === true,
      rtxVsrAvailable: state.features?.rtxVsr?.available === true,
      triton: state.configuration?.installTriton === true,
      tritonAvailable: state.features?.triton?.available === true,
      sageAttention: state.configuration?.installSageAttention === true,
      sageAttentionAvailable: state.features?.sageattention?.available === true,
    },
    features: state.features,
    message: event.message,
  });
  state.configured = true;
}

async function handleEvent(event) {
  if (event.type === "task-list") {
    state.tasks = event.tasks.map((task) => ({ ...task, status: "pending", detail: "" }));
    updateState({ status: "running", phase: "检测机器环境" });
  } else if (event.type === "task") {
    const task = state.tasks.find((item) => item.id === event.id);
    if (task) Object.assign(task, { status: event.status, detail: event.detail || task.detail });
    if (event.status === "running") {
      updateState({ status: "running", phase: event.label, message: event.detail || "正在自动处理，请保持窗口开启" });
      state.currentDownload = { name: event.label, currentBytes: 0, totalBytes: 0, speedBps: 0, connections: 0, route: "正在测速", mode: "自动选择" };
    }
  } else if (event.type === "machine") {
    updateState({ machine: event });
  } else if (event.type === "selection") {
    updateState({
      selection: event,
      phase: event.resumed ? "已恢复版本选择" : "已选择合适版本",
      message: `${event.resumed ? "断点恢复 · " : ""}PyTorch ${event.torch} · ${event.variant} · ${event.source}`,
    });
  } else if (event.type === "plan") {
    updateState({ downloads: event.downloads });
  } else if (event.type === "plan-add") {
    if (!state.downloads.some((item) => item.id === event.download.id)) state.downloads.push(event.download);
    state.updatedAt = Date.now();
  } else if (event.type === "download") {
    state.currentDownload = {
      ...state.currentDownload,
      name: event.name,
      currentBytes: Number(event.currentBytes || 0),
      totalBytes: Number(event.totalBytes || 0),
      speedBps: Number(event.speedBps || 0),
      connections: Number(event.connections ?? state.currentDownload.connections ?? 0),
      route: event.route || state.currentDownload.route,
      // A phase with no stated total still has to look alive; the panel switches to a moving bar
      // with a byte counter rather than sitting at 0%.
      indeterminate: event.indeterminate === true || (!event.totalBytes && !event.cached),
      etaSeconds: Number(event.etaSeconds || 0),
      packages: Number(event.packages || 0),
      sources: Array.isArray(event.sources) ? event.sources : state.currentDownload.sources || [],
    };
    state.updatedAt = Date.now();
  } else if (event.type === "route") {
    state.currentDownload = {
      ...state.currentDownload,
      route: event.label || "已选择线路",
      mode: event.mode || (event.connections > 1 ? `并发 ×${event.connections}` : "单连接"),
      connections: Number(event.connections || 0),
      speedBps: Number(event.speedBps || 0),
      sources: Array.isArray(event.sources) ? event.sources : [],
    };
    const task = state.tasks.find((item) => item.id === event.id);
    if (task && task.status === "running") task.detail = `${state.currentDownload.route} · ${state.currentDownload.mode}`;
    state.updatedAt = Date.now();
  } else if (event.type === "warning") {
    state.warnings.push(event.message);
    appendLog(`提示：${event.message}`, "warning");
  } else if (event.type === "feature") {
    state.features[event.id === "rtx-vsr" ? "rtxVsr" : event.id] = event;
    state.updatedAt = Date.now();
  } else if (event.type === "error") {
    const failedSelection = {
      variant: state.selection?.variant || state.configuration?.cudaVariant,
      torch: state.selection?.torch || state.configuration?.torchVersion,
      repairAttempted: state.configuration?.repairAttempted === true,
    };
    const repair = event.repairable && setupCatalog && state.configuration
      ? recommendRepairPlan(setupCatalog, failedSelection)
      : null;
    updateState({
      status: "error",
      phase: "配置失败",
      message: event.message,
      repair,
      lastError: event,
      autoRepairScheduled: Boolean(repair && state.configuration?.autoRepair && !state.configuration?.repairAttempted),
    });
    recordSetupFailure(event.message, { event });
  } else if (event.type === "complete") {
    await saveSetupMarker(event);
    updateState({ status: "complete", phase: "配置完成", message: event.message });
    state.currentDownload = { name: "全部完成", currentBytes: 1, totalBytes: 1 };
  }
}

function consumeStream(stream, streamName) {
  let buffer = "";
  stream.setEncoding("utf8");
  stream.on("data", (chunk) => {
    buffer += chunk;
    const lines = buffer.split(/[\r\n]+/);
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (line.startsWith(eventPrefix)) {
        try {
          const event = JSON.parse(line.slice(eventPrefix.length));
          pendingEvent = pendingEvent.then(() => handleEvent(event));
        } catch {
          appendLog(line, streamName);
        }
      } else {
        parseProgressLine(line);
        appendLog(line, streamName);
      }
    }
  });
  stream.on("end", () => {
    if (buffer) appendLog(buffer, streamName);
  });
}

async function fetchText(url) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetchWithProxy(url, {
      headers: { "User-Agent": "XirAI-Setup/0.1" },
      redirect: "follow",
      signal: controller.signal,
    });
    return response.ok ? await response.text() : "";
  } catch (error) {
    // An explicit proxy choice is a contract: surfacing its failure is safer
    // than turning it into an indistinguishable empty catalog response.
    if (proxyPolicy.useProxy) throw error;
    return "";
  } finally {
    clearTimeout(timeout);
  }
}

function readPythonTag(python) {
  const result = runPython(python, ["-I", "-c", "import sys;print(f'cp{sys.version_info.major}{sys.version_info.minor}')"], {
    env: isolatedPythonEnv(process.env),
    stdio: "pipe",
    encoding: "utf8",
    timeout: 10000,
  });
  return result.status === 0 ? result.stdout.trim() : "cp312";
}

async function loadSetupCatalog() {
  const recoveringProgress = state.status === "error" && Boolean(state.configuration);
  state.catalogStatus = "loading";
  if (!recoveringProgress) updateState({ phase: "正在查询可用 CUDA / PyTorch 版本", message: "从官方 wheel 目录查询中，不安装任何文件" });
  try {
    let hardware = detectNvidiaSmi();
    const python = findPython(projectRoot);
    let pythonTag = "cp312";
    if (python) {
      hardware = mergeNvidiaDetection(hardware, detectNvidiaDriverApi(python, process.env));
      pythonTag = readPythonTag(python);
    }

    const rootHtml = await fetchText("https://download.pytorch.org/whl/");
    const variants = parseCudaVariants(rootHtml);
    if (!variants.length) throw new Error("无法读取官方 PyTorch CUDA wheel 目录，请检查网络、代理或稍后重试");

    const entries = await Promise.all(variants.map(async (variant) => {
      const html = await fetchText(`https://download.pytorch.org/whl/${variant}/torch/`);
      const versions = parseTorchWheelVersions(html, {
        variant,
        pythonTag,
        platform: process.platform,
        architecture: process.arch,
      });
      return [variant, versions];
    }));
    const variantVersions = Object.fromEntries(entries.filter(([, versions]) => versions.length));
    if (!Object.keys(variantVersions).length) throw new Error(`官方目录中没有适用于 ${pythonTag} / ${process.platform} / ${process.arch} 的稳定 PyTorch CUDA wheel`);

    setupCatalog = buildSetupCatalog({ hardware, pythonTag, variantVersions });
    state.catalogStatus = "ready";
    state.machine ||= hardware;
    state.recommended = setupCatalog.recommended;
    if (!recoveringProgress) {
      updateState({
        status: "awaiting-choice",
        phase: "选择环境配置方式",
        message: hardware.available
          ? `${hardware.name || "NVIDIA GPU"} · 驱动 CUDA 上限 ${hardware.cuda?.text || "确认中"}`
          : "未检测到 NVIDIA 驱动，自动模式不可用",
      });
    }
  } catch (error) {
    setupCatalog = null;
    state.catalogStatus = "error";
    updateState({ status: "catalog-error", phase: "无法查询可用版本", message: error.message });
    recordSetupFailure(error.message, { stage: "catalog" });
  }
}

function startSetup(args = [], configuration = state.configuration) {
  if (setupProcess && setupProcess.exitCode == null) return false;
  state.status = "running";
  setupFailureLogged = false;
  state.phase = "检测机器环境";
  state.message = "正在读取断点并检测系统、Python、显卡和网络";
  state.logs = [];
  state.warnings = [];
  state.tasks = [];
  state.downloads = [];
  state.machine = null;
  state.selection = null;
  state.repair = null;
  state.autoRepairScheduled = false;
  state.repairAttempted = Boolean(configuration?.repairAttempted);
  state.currentDownload = { name: "环境检测", currentBytes: 0, totalBytes: 0, speedBps: 0, connections: 0, route: "等待测速", mode: "自动选择" };
  state.startedAt = Date.now();
  state.updatedAt = Date.now();
  void persistSetupState();

  const allArgs = [path.join(projectRoot, "scripts", "setup.mjs"), "--events", ...forwardedArguments, ...args];
  setupProcess = spawn(process.execPath, allArgs, {
    cwd: projectRoot,
    env: { ...process.env, XIRAI_SETUP_EVENTS: "1", FORCE_COLOR: "0" },
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"],
  });
  consumeStream(setupProcess.stdout, "stdout");
  consumeStream(setupProcess.stderr, "stderr");
  setupProcess.on("error", (error) => {
    updateState({ status: "error", phase: "无法启动配置", message: error.message });
    recordSetupFailure(error.message, { stage: "spawn" });
  });
  setupProcess.on("exit", async (code) => {
    await pendingEvent;
    if (code === 0 && state.status !== "complete") {
      updateState({ status: "complete", phase: "配置完成", message: "环境检测、下载、安装和构建均已完成" });
    } else if (code !== 0 && state.status !== "error") {
      updateState({ status: "error", phase: "配置失败", message: `配置进程退出，错误码 ${code ?? "未知"}` });
      recordSetupFailure(`配置进程退出，错误码 ${code ?? "未知"}`, { stage: "exit", exit_code: code ?? null });
    }
    if (code !== 0 && state.autoRepairScheduled && state.repair && !state.repairAttempted) {
      startRecommendedRepair();
      return;
    }
    if (exitOnComplete && state.status === "complete") setTimeout(() => server?.close(() => process.exit(code || 0)), 1000);
  });
  return true;
}

function startRecommendedRepair() {
  const repairPlan = state.repair;
  if (!repairPlan) return false;
  const installRtxVsr = state.configuration?.installRtxVsr === true;
  const configuration = {
    mode: "manual",
    cudaVariant: repairPlan.variant,
    torchVersion: repairPlan.version,
    installXformers: false,
    installRtxVsr,
    // A repair changes the Torch build, so the optional accelerators are
    // reinstalled and revalidated against it rather than assumed to still fit.
    installTriton: state.configuration?.installTriton === true,
    installSageAttention: state.configuration?.installSageAttention === true,
    autoRepair: false,
    repair: true,
    repairAttempted: true,
  };
  state.configuration = configuration;
  state.repair = null;
  state.autoRepairScheduled = false;
  return startSetup(buildSetupArguments(configuration), configuration);
}

function openBrowser(url) {
  let command;
  let args;
  if (process.platform === "win32") {
    command = "cmd.exe";
    args = ["/d", "/s", "/c", "start", "", url];
  } else if (process.platform === "darwin") {
    command = "open";
    args = [url];
  } else {
    command = "xdg-open";
    args = [url];
  }
  const child = spawn(command, args, { detached: true, stdio: "ignore", windowsHide: true });
  child.on("error", () => console.log(`请在浏览器打开：${url}`));
  child.unref();
}

function startMainService() {
  if (!state.configured || (appProcess && appProcess.exitCode == null)) return false;
  state.restarting = true;
  state.phase = "正在重启服务";
  state.message = "主页面准备完成后将自动进入";
  state.updatedAt = Date.now();
  const vite = viteInvocation(projectRoot);
  server.close(() => {
    appProcess = spawn(vite.command, vite.args, {
      cwd: projectRoot,
      env: applicationEnvironment,
      windowsHide: true,
      stdio: "inherit",
    });
    appProcess.on("error", (error) => console.error(`主服务启动失败：${error.message}`));
    appProcess.on("exit", (code) => {
      if (code) console.error(`主服务退出，错误码：${code}`);
      process.exit(code || 0);
    });
  });
  return true;
}

function sendJson(response, statusCode, payload) {
  response.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
  });
  response.end(JSON.stringify(payload));
}

function readBody(request, maxBytes = 16384) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let total = 0;
    request.on("data", (chunk) => {
      total += chunk.length;
      if (total > maxBytes) {
        request.destroy();
        reject(new Error("Request body too large"));
        return;
      }
      chunks.push(chunk);
    });
    request.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    request.on("error", reject);
  });
}

const setupStateReady = restoreSetupState();

server = createServer(async (request, response) => {
  await setupStateReady;
  const url = new URL(request.url, "http://127.0.0.1");
  if (request.method === "POST" && request.headers.origin) {
    try {
      if (new URL(request.headers.origin).host !== request.headers.host) {
        sendJson(response, 403, { error: "Cross-origin setup control is not allowed" });
        return;
      }
    } catch {
      sendJson(response, 403, { error: "Invalid request origin" });
      return;
    }
  }
  if (request.method === "GET" && url.pathname === "/") {
    response.writeHead(302, { Location: "/config" });
    response.end();
    return;
  }
  if (request.method === "GET" && (url.pathname === "/config" || url.pathname === "/config/")) {
    try {
      const html = await readFile(uiPath);
      response.writeHead(200, { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" });
      response.end(html);
    } catch (error) {
      response.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
      response.end(error.message);
    }
    return;
  }
  if (request.method === "GET" && ["/xiriacanvas-logo.svg", "/xiriacanvas-favicon.svg"].includes(url.pathname)) {
    try {
      const requestedTheme = themeFromPalette(url.searchParams.get("palette"));
      const [logo, theme] = await Promise.all([readFile(logoPath, "utf8"), requestedTheme || readPersistedTheme()]);
      response.writeHead(200, { "Content-Type": "image/svg+xml; charset=utf-8", "Cache-Control": "no-store" });
      response.end(themedLogo(logo, theme));
    } catch (error) {
      response.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
      response.end(error.message);
    }
    return;
  }
  if (request.method === "GET" && url.pathname === "/api/theme") {
    sendJson(response, 200, { theme: await readPersistedTheme() });
    return;
  }
  if (request.method === "GET" && url.pathname === "/api/state") {
    sendJson(response, 200, state);
    return;
  }
  if (request.method === "GET" && url.pathname === "/api/options") {
    if (state.catalogStatus === "ready") {
      sendJson(response, 200, {
        status: "ready",
        hardware: state.machine,
        pythonTag: setupCatalog?.pythonTag || "cp312",
        recommended: state.recommended,
        cudaOptions: setupCatalog?.cudaOptions || [],
      });
    } else if (state.catalogStatus === "error") {
      sendJson(response, 503, { status: "error", error: state.message });
    } else {
      sendJson(response, 202, { status: state.catalogStatus });
    }
    return;
  }
  if (request.method === "POST" && url.pathname === "/api/options/retry") {
    if (state.status !== "awaiting-choice" && state.status !== "catalog-error") {
      sendJson(response, 409, { error: "当前状态不允许重新查询" });
      return;
    }
    setupCatalog = null;
    state.catalogStatus = "loading";
    state.repair = null;
    state.repairAttempted = false;
    state.configuration = null;
    sendJson(response, 202, { status: "loading" });
    loadSetupCatalog();
    return;
  }
  if (request.method === "POST" && url.pathname === "/api/start") {
    if (!["awaiting-choice", "error"].includes(state.status)) {
      sendJson(response, 409, { error: "当前状态不允许启动配置" });
      return;
    }
    let body;
    try {
      body = JSON.parse(await readBody(request));
    } catch {
      sendJson(response, 400, { error: "无法解析配置请求" });
      return;
    }
    const validation = validateSetupConfiguration(body, setupCatalog);
    if (!validation.ok) {
      sendJson(response, 400, { error: validation.error });
      return;
    }
    const config = validation.configuration;
    state.configuration = config;
    state.repairAttempted = Boolean(config.repairAttempted);

    if (!startSetup(buildSetupArguments(config), config)) {
      sendJson(response, 409, { error: "配置进程已在运行" });
      return;
    }
    sendJson(response, 202, { status: "running" });
    return;
  }
  if (request.method === "POST" && url.pathname === "/api/retry") {
    if (state.status !== "error" || !state.configuration) {
      sendJson(response, 409, { error: "当前没有可继续的配置任务" });
      return;
    }
    if (!startSetup(buildSetupArguments(state.configuration), state.configuration)) {
      sendJson(response, 409, { error: "配置进程已在运行" });
      return;
    }
    sendJson(response, 202, { status: "running" });
    return;
  }
  if (request.method === "POST" && url.pathname === "/api/repair") {
    if (state.status !== "error" || !state.repair) {
      sendJson(response, 409, { error: "当前没有可修复的错误" });
      return;
    }
    if (!startRecommendedRepair()) {
      sendJson(response, 409, { error: "配置进程已在运行" });
      return;
    }
    sendJson(response, 202, { status: "running" });
    return;
  }
  if (request.method === "POST" && url.pathname === "/api/close") {
    if (setupProcess && setupProcess.exitCode == null) {
      sendJson(response, 202, { status: "continuing", message: "配置会继续在后台运行；重新打开配置页即可查看进度。" });
      return;
    }
    if (returnToApp && state.configured) {
      sendJson(response, 202, { status: "restarting", url: "/" });
      if (supervised) {
        updateState({ restarting: true, phase: "正在返回主页面", message: "工作区状态将在主页面恢复" });
        setTimeout(() => server.close(() => process.exit(76)), 100);
      } else {
        setTimeout(startMainService, 100);
      }
    } else {
      sendJson(response, 200, { status: "closing" });
      setTimeout(() => server.close(() => process.exit(0)), 100);
    }
    return;
  }
  if (request.method === "POST" && url.pathname === "/api/restart") {
    if (!state.configured || state.status !== "complete") {
      sendJson(response, 409, { error: "配置尚未完成" });
      return;
    }
    sendJson(response, 202, { status: "restarting", url: "/" });
    if (supervised) {
      updateState({ restarting: true, phase: "正在重启服务", message: "主页面准备完成后将自动进入" });
      setTimeout(() => server.close(() => process.exit(76)), 100);
    } else {
      setTimeout(startMainService, 100);
    }
    return;
  }
  response.writeHead(404);
  response.end();
});

server.on("error", (error) => {
  if (error.code === "EADDRINUSE") {
    console.error(`端口 ${webPort} 已被占用。请关闭已有 XirAI 服务后重试。`);
    process.exit(1);
  }
  throw error;
});

server.listen(webPort, webHost, () => {
  const url = `http://localhost:${webPort}/config`;
  console.log(`XirAI 可视化配置器：${url}`);
  for (const accessUrl of state.accessUrls) console.log(`访问地址：${accessUrl}config`);
  if (!noOpen) openBrowser(url);
  void setupStateReady.finally(() => setTimeout(loadSetupCatalog, 100));
});

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.once(signal, () => {
    setupProcess?.kill(signal);
    appProcess?.kill(signal);
    server.close(() => process.exit(0));
  });
}
