import { spawnSync } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { chmodSync, existsSync, lstatSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { benchmarkRoutes, downloadFile } from "./download.mjs";
import { createInstallerRunner } from "./install-progress.mjs";
import { writeDiagnosticLog } from "./diagnostics.mjs";
import { findPython, getVenvPythonPath, isolatedPythonEnv, loadLocalEnv, runPython, verifyProjectVenv } from "./python.mjs";
import { npmInvocation, viteInvocation } from "./node-tools.mjs";
import { detectNvidiaDriverApi, detectNvidiaSmi, mergeNvidiaDetection } from "./nvidia.mjs";
import { resolveRtxVsrChoice, rtxVsrEligibility, RTX_VSR_INDEX, RTX_VSR_PRODUCT_SOURCE, RTX_VSR_VERSION } from "./rtx-vsr.mjs";
import { installOptionalAccelerator, resolveAcceleratorChoice, sageAttentionSmokeSource, SAGE_ATTENTION_REQUIREMENT, tritonPackageName, tritonSmokeSource } from "./anima-accelerators.mjs";
import { readSetupMarker, readSetupResume, writeSetupMarker, writeSetupResume } from "./setup-state.mjs";
import { availableTorchVersions, resolveLatestStableTorch, torchInternals } from "./torch.mjs";
import { resolveUvTarget, uvDownloadRoutes, uvVersionMatches } from "./uv-bootstrap.mjs";
import { configuredModelDirectories } from "./model-paths.mjs";
import { requireBundledAnimaTokenizers } from "./anima-models.mjs";
import { applyInstallerProxyPolicy, createInstallerFetch, npmInstallerEnvironment } from "./proxy-policy.mjs";

const projectRoot = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
Object.assign(process.env, loadLocalEnv(projectRoot));
const proxyPolicy = applyInstallerProxyPolicy();
const fetchWithProxy = createInstallerFetch(proxyPolicy);

const argumentsSet = new Map(process.argv.slice(2).map((argument) => {
  const [key, value = "true"] = argument.replace(/^--/, "").split("=", 2);
  return [key, value];
}));
const requestedSource = argumentsSet.get("source") || process.env.XIRAI_PACKAGE_SOURCE || "auto";
const requestedTorch = argumentsSet.get("torch") || process.env.XIRAI_TORCH_VARIANT || "auto";
const requestedTorchVersion = argumentsSet.get("torch-version") || process.env.XIRAI_TORCH_VERSION || null;
const recreateVenv = argumentsSet.has("recreate-venv");
const skipNode = argumentsSet.has("skip-node");
const diagnoseOnly = argumentsSet.has("diagnose");
const refreshSelection = argumentsSet.has("refresh-selection");
const updateRepair = argumentsSet.has("update-repair");
const installXformers = !argumentsSet.has("without-xformers");
let rtxVsrChoice = null;
let rtxVsrChoiceError = null;
try {
  rtxVsrChoice = resolveRtxVsrChoice(process.argv.slice(2));
} catch (error) {
  rtxVsrChoiceError = error.message;
}
const requestedRtxVsr = rtxVsrChoice ?? (readSetupMarker(projectRoot)?.selection?.rtxVsr === true);
const rtxVsrExplicitlyDisabled = rtxVsrChoice === false;
let acceleratorChoiceError = null;
let tritonChoice = null;
let sageAttentionChoice = null;
try {
  tritonChoice = resolveAcceleratorChoice(process.argv.slice(2), "triton");
  sageAttentionChoice = resolveAcceleratorChoice(process.argv.slice(2), "sageattention");
} catch (error) {
  acceleratorChoiceError = error.message;
}
const requestedSageAttention = sageAttentionChoice ?? (readSetupMarker(projectRoot)?.selection?.sageAttention === true);
// SageAttention 1.x is a pure-Triton package, so asking for it asks for Triton.
const requestedTriton = (tritonChoice ?? (readSetupMarker(projectRoot)?.selection?.triton === true)) || requestedSageAttention;
const eventMode = argumentsSet.has("events") || process.env.XIRAI_SETUP_EVENTS === "1";
const eventPrefix = "@@XIRAI_SETUP@@";
const bootstrapDownloads = [];
const downloadConnections = Math.max(1, Math.min(16, Number(argumentsSet.get("connections") || process.env.XIRAI_DOWNLOAD_CONNECTIONS || 8)) || 8);
const packageConcurrency = Math.max(1, Math.min(32, Number(process.env.XIRAI_PACKAGE_CONCURRENCY || 12)) || 12);
// How long an installer may produce neither output nor bytes before the mirror it is talking to is
// treated as dead. Generous on purpose: resolution, sdist builds and antivirus scans are all quiet,
// and killing a working install is worse than waiting. Only used when there is somewhere to switch.
const installStallMs = Math.max(60, Number(process.env.XIRAI_INSTALL_STALL_SECONDS || 300) || 300) * 1000;
let resumeState = readSetupResume(projectRoot) || { schema: 1, product: "XiriaCanvas AI", completed: {} };
let activeTask;

const tasks = [
  { id: "machine", label: "检测机器环境" },
  { id: "python", label: "准备独立 Python 环境" },
  { id: "source", label: "检测下载线路" },
  { id: "select", label: "选择最新兼容 PyTorch / CUDA Runtime" },
  { id: "tools", label: "配置 Python 基础工具" },
  { id: "torch", label: "下载并安装 PyTorch" },
  { id: "backend", label: "下载后端依赖（含 Ultralytics YOLO）" },
  { id: "rtx-vsr", label: "配置可选 NVIDIA RTX VSR" },
  { id: "xformers", label: "安装 xformers 加速" },
  { id: "triton", label: "安装 Triton 编译器" },
  { id: "sageattention", label: "安装 SageAttention 注意力内核" },
  { id: "node", label: "下载前端依赖" },
  { id: "verify", label: "验证环境并构建项目" },
];

const sources = {
  official: { label: "官方源", pip: "https://pypi.org/simple", npm: "https://registry.npmjs.org", torch: "https://download.pytorch.org/whl", hf: "https://huggingface.co" },
  aliyun: { label: "阿里云", pip: "https://mirrors.aliyun.com/pypi/simple", npm: "https://registry.npmmirror.com", torch: "https://mirrors.aliyun.com/pytorch-wheels", hf: "https://hf-mirror.com" },
  tsinghua: { label: "清华镜像", pip: "https://pypi.tuna.tsinghua.edu.cn/simple", npm: "https://registry.npmmirror.com", torch: "https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels", hf: "https://hf-mirror.com" },
};

function emit(type, payload = {}) {
  if (eventMode) process.stdout.write(`${eventPrefix}${JSON.stringify({ type, ...payload })}\n`);
}

function startTask(id, detail = "") {
  activeTask = id;
  const task = tasks.find((item) => item.id === id);
  emit("task", { id, status: "running", label: task?.label || id, detail });
}

function completeTask(id, detail = "") {
  const task = tasks.find((item) => item.id === id);
  emit("task", { id, status: "complete", label: task?.label || id, detail });
}

function skipTask(id, detail = "") {
  const task = tasks.find((item) => item.id === id);
  emit("task", { id, status: "skipped", label: task?.label || id, detail });
}

function warn(message) {
  emit("warning", { message });
  if (!eventMode) console.warn(`提示：${message}`);
}

function fail(message, details = {}) {
  writeDiagnosticLog(projectRoot, {
    kind: "setup-failure",
    message,
    details: {
      task: activeTask || null,
      requested_source: requestedSource,
      requested_torch: requestedTorch,
      requested_torch_version: requestedTorchVersion,
      requested_rtx_vsr: requestedRtxVsr,
      ...details,
    },
  });
  emit("error", { message, ...details });
  console.error(`\n配置失败：${message}`);
  process.exit(1);
}

function digest(value) {
  return createHash("sha256").update(value).digest("hex");
}

function persistResume(updates = {}) {
  if (diagnoseOnly) return;
  resumeState = { ...resumeState, ...updates, schema: 1, product: "XiriaCanvas AI", updatedAt: new Date().toISOString() };
  writeSetupResume(projectRoot, resumeState);
}

function checkpointTask(id, fingerprint, status = "complete") {
  persistResume({
    completed: {
      ...(resumeState.completed || {}),
      [id]: { status, fingerprint, environmentId: resumeState.environmentId, at: new Date().toISOString() },
    },
  });
}

function taskCheckpoint(id, fingerprint, statuses = ["complete"]) {
  const checkpoint = resumeState.completed?.[id];
  return checkpoint?.environmentId === resumeState.environmentId
    && checkpoint?.fingerprint === fingerprint
    && statuses.includes(checkpoint?.status);
}

function pythonCheck(python, code, environment) {
  return runPython(python, ["-I", "-c", code], {
    cwd: projectRoot,
    env: environment,
    stdio: "ignore",
    timeout: 30000,
  }).status === 0;
}

function xformersCudaSmoke(expectedTorchVersion) {
  return [
    "import torch,xformers",
    "from xformers.ops import memory_efficient_attention",
    `assert torch.__version__==${JSON.stringify(expectedTorchVersion)}`,
    "assert xformers._cpp_lib._cpp_library_load_exception is None",
    "query=torch.randn((1,32,1,64),device='cuda',dtype=torch.float16)",
    "output=memory_efficient_attention(query,query,query)",
    "torch.cuda.synchronize()",
    "assert output.shape==query.shape and torch.isfinite(output).all().item()",
  ].join(";");
}

function readTorchDeviceMetadata(python, environment) {
  const script = [
    "import json,torch",
    "available=torch.cuda.is_available()",
    "device=torch.cuda.current_device() if available else None",
    "capability=torch.cuda.get_device_capability(device) if available else None",
    "print(json.dumps({'cudaAvailable':available,'torchVersion':torch.__version__,'torchCudaVersion':torch.version.cuda,'deviceName':torch.cuda.get_device_name(device) if available else None,'computeCapability':'.'.join(map(str,capability)) if capability else None}))",
  ].join(";");
  const result = runPython(python, ["-I", "-c", script], {
    cwd: projectRoot,
    env: environment,
    stdio: "pipe",
    encoding: "utf8",
    timeout: 30000,
  });
  if (result.status !== 0) return null;
  try {
    return JSON.parse(result.stdout.trim());
  } catch {
    return null;
  }
}

function probeRtxVsrAdapter(python, environment) {
  const script = [
    "import sys,time",
    "sys.path.insert(0,'backend')",
    "import rtx_vsr",
    "status=rtx_vsr.status(refresh=True)",
    "deadline=time.monotonic()+180",
    "while status.get('probing') and time.monotonic()<deadline:",
    " time.sleep(0.1)",
    " status=rtx_vsr.status()",
    "assert status.get('available'), status.get('reason') or 'RTX VSR adapter is unavailable'",
  ].join("\n");
  return runPython(python, ["-I", "-c", script], {
    cwd: projectRoot,
    env: environment,
    timeout: 200000,
  }).status === 0;
}

function detectNvidiaDriverModel(deviceName) {
  if (process.platform !== "win32") return null;
  const result = spawnSync(process.env.NVIDIA_SMI_PATH || "nvidia-smi", [
    "--query-gpu=name,driver_model.current", "--format=csv,noheader,nounits",
  ], { encoding: "utf8", windowsHide: true, timeout: 10000 });
  if (result.status !== 0) return null;
  const devices = String(result.stdout || "").split(/\r?\n/).map((line) => {
    const separator = line.lastIndexOf(",");
    return separator < 0 ? null : { name: line.slice(0, separator).trim(), model: line.slice(separator + 1).trim().toUpperCase() };
  }).filter(Boolean);
  const matching = devices.find((device) => device.name === deviceName) || (devices.length === 1 ? devices[0] : null);
  return ["TCC", "WDDM"].includes(matching?.model) ? matching.model : null;
}

function run(command, args, options = {}) {
  console.log(`\n> ${command} ${args.join(" ")}`);
  const result = spawnSync(command, args, { cwd: projectRoot, env: process.env, stdio: "inherit", windowsHide: true, ...options });
  if (result.status !== 0) fail(`${command} 退出，错误码 ${result.status ?? "未知"}`);
  return result;
}

function tryRun(command, args, options = {}) {
  console.log(`\n> ${command} ${args.join(" ")}`);
  return spawnSync(command, args, { cwd: projectRoot, env: process.env, stdio: "inherit", windowsHide: true, ...options });
}

function uvCacheDirectory() {
  return path.resolve(projectRoot, process.env.XIRAI_CACHE_DIR || ".cache", "uv-cache");
}

// Every installer the configurator shells out to runs through this: it streams the child's output,
// meters the bytes the phase writes, and treats a mirror that has gone silent as a failed one. The
// body lives in `install-progress.mjs` so each of those behaviours is testable without a network.
const runInstaller = createInstallerRunner({
  emit,
  cwd: projectRoot,
  concurrency: packageConcurrency,
  activeTask: () => activeTask,
});

function uniqueSources(kind) {
  const names = requestedSource === "auto"
    ? Object.keys(sources)
    : [...new Set([requestedSource, "official"])];
  const seen = new Set();
  return names.filter((name) => {
    const url = sources[name][kind];
    if (seen.has(url)) return false;
    seen.add(url);
    return true;
  });
}

// The resource every mirror is measured on. Probing whichever package the phase happens to need
// measured the wrong thing: `pip/`'s index page is a few kilobytes, which is over before the link
// leaves slow start, so the ranking came down to who answered first rather than who transfers
// fastest. These two are large on every mirror and present on all of them, so the same body is
// compared across candidates and the reading is bandwidth rather than round-trip time.
const BANDWIDTH_PROBE = { pip: "torch/", npm: "react" };

function packageProbeUrl(source, kind, packageName) {
  const root = source[kind].replace(/\/$/, "");
  if (kind === "pip") return `${root}/${(BANDWIDTH_PROBE.pip || packageName).replace(/_/g, "-")}`;
  return `${root}/${BANDWIDTH_PROBE.npm || packageName}`;
}

async function choosePackageSource(taskId, kind, packageName, mode) {
  const candidates = uniqueSources(kind).map((name) => ({
    id: name,
    label: sources[name].label,
    url: packageProbeUrl(sources[name], kind, packageName),
  }));
  const ranked = requestedSource === "auto"
    ? await benchmarkRoutes(candidates, { fetcher: fetchWithProxy, sampleBytes: 1024 ** 2, sampleMs: 2000, timeoutMs: 9000 })
    : candidates.map((route) => ({ ...route, ok: true, latencyMs: 0, speedBps: 0 }));
  const selected = ranked.find((route) => route.ok);
  if (!selected) fail(`${packageName} 的所有下载线路均不可用`);
  const selectedSource = sources[selected.id];
  const routeMode = mode || (kind === "pip" ? `包级并发 ×${packageConcurrency}` : "并行资源下载");
  const measured = selected.speedBps ? ` · ${(selected.speedBps / 1024 ** 2).toFixed(1)} MB/s` : "";
  emit("route", {
    id: taskId,
    label: selected.label,
    url: selectedSource[kind],
    latencyMs: selected.latencyMs,
    speedBps: selected.speedBps,
    connections: kind === "pip" ? packageConcurrency : 1,
    mode: routeMode,
  });
  console.log(`\n${packageName} 自动选择线路：${selected.label}${selected.latencyMs ? ` · ${selected.latencyMs}ms` : ""}${measured} · ${routeMode}`);
  return { name: selected.id, source: selectedSource, ranked };
}

function uvFetch(url, options = {}) {
  const githubToken = process.env.UV_GITHUB_TOKEN;
  if (!githubToken) return fetchWithProxy(url, options);
  const trustedHosts = new Set(["github.com", "api.github.com"]);
  for (const base of [process.env.UV_INSTALLER_GHE_BASE_URL, process.env.UV_INSTALLER_GITHUB_BASE_URL].filter(Boolean)) {
    try {
      trustedHosts.add(new URL(base).hostname);
    } catch {}
  }
  if (!trustedHosts.has(new URL(url).hostname)) return fetchWithProxy(url, options);
  return fetchWithProxy(url, {
    ...options,
    headers: { ...options.headers, Authorization: `Bearer ${githubToken}` },
  });
}

function commandOutput(command, args) {
  return spawnSync(command, args, { cwd: projectRoot, encoding: "utf8", windowsHide: true, timeout: 10000 });
}

async function bootstrapUv() {
  const toolsDirectory = path.resolve(projectRoot, process.env.XIRAI_CACHE_DIR || ".cache", "tools", "uv");
  let target;
  try {
    target = resolveUvTarget({
      platform: process.platform,
      architecture: process.arch,
      glibcVersion: process.report?.getReport()?.header?.glibcVersionRuntime || null,
    });
  } catch (error) {
    fail(error.message);
  }
  const extractDirectory = path.join(toolsDirectory, "current");
  const executable = path.join(extractDirectory, target.executable);
  const metadataPath = path.join(extractDirectory, "xirai-uv-install.json");
  if (existsSync(executable) && existsSync(metadataPath)) {
    try {
      const metadata = JSON.parse(readFileSync(metadataPath, "utf8"));
      const version = commandOutput(executable, ["--version"]);
      if (metadata.version === target.version
        && metadata.target === target.target
        && metadata.sha256 === target.sha256
        && version.status === 0
        && uvVersionMatches(version.stdout, target.version)) return executable;
    } catch {}
  }

  mkdirSync(toolsDirectory, { recursive: true });
  const planItem = { id: "python", label: `Python 引导工具 uv ${target.version}`, detail: `${target.target} · 固定哈希校验` };
  bootstrapDownloads.push(planItem);
  emit("plan-add", { download: planItem });
  const archivePath = path.join(toolsDirectory, `${target.version}-${target.assetName}`);
  try {
    await downloadFile({
      routes: uvDownloadRoutes(target, process.env),
      destination: archivePath,
      expectedSha256: target.sha256,
      connections: downloadConnections,
      thresholdBytes: 8 * 1024 ** 2,
      fetcher: uvFetch,
      rankRoutes: true,
      onRoute: (route) => emit("route", { id: "python", ...route, mode: route.connections > 1 ? `分片下载 ×${route.connections}` : "单连接下载" }),
      onProgress: (progress) => emit("download", { name: target.assetName, ...progress }),
      onWarning: (message) => emit("warning", { message }),
    });
  } catch (error) {
    fail(`uv 下载失败：${error.message}`);
  }

  const stagingDirectory = path.join(toolsDirectory, `staging-${process.pid}-${Date.now()}`);
  const backupDirectory = path.join(toolsDirectory, `previous-${process.pid}`);
  rmSync(stagingDirectory, { recursive: true, force: true });
  mkdirSync(stagingDirectory, { recursive: true });
  let result;
  if (target.assetName.endsWith(".zip")) {
    result = spawnSync("powershell.exe", [
      "-NoProfile", "-NonInteractive", "-Command",
      "Expand-Archive -LiteralPath $env:XIRAI_UV_ARCHIVE -DestinationPath $env:XIRAI_UV_DEST -Force",
    ], {
      env: { ...process.env, XIRAI_UV_ARCHIVE: archivePath, XIRAI_UV_DEST: stagingDirectory },
      stdio: "inherit",
      windowsHide: true,
    });
  } else if (/\.tar\.[a-z0-9]+$/i.test(target.assetName)) {
    result = spawnSync("tar", ["xf", archivePath, "--no-same-owner", "--strip-components", "1", "-C", stagingDirectory], { stdio: "inherit" });
  } else {
    fail(`不支持的 uv 安装包扩展名：${target.assetName}`);
  }
  if (result.status !== 0) fail("uv 安装包解压失败");
  for (const binary of target.binaries) {
    const binaryPath = path.join(stagingDirectory, binary);
    if (!existsSync(binaryPath)) fail(`uv 安装包缺少 ${binary}`);
    if (process.platform !== "win32") chmodSync(binaryPath, 0o755);
  }
  const stagedExecutable = path.join(stagingDirectory, target.executable);
  const stagedVersion = commandOutput(stagedExecutable, ["--version"]);
  if (stagedVersion.status !== 0 || !uvVersionMatches(stagedVersion.stdout, target.version)) {
    fail(`uv 解压版本不匹配，预期 ${target.version}`);
  }
  writeFileSync(path.join(stagingDirectory, "xirai-uv-install.json"), `${JSON.stringify({
    version: target.version,
    target: target.target,
    assetName: target.assetName,
    binaries: target.binaries,
    sha256: target.sha256,
    unmanaged: true,
    modifyPath: false,
  }, null, 2)}\n`, "utf8");

  rmSync(backupDirectory, { recursive: true, force: true });
  if (existsSync(extractDirectory)) renameSync(extractDirectory, backupDirectory);
  try {
    renameSync(stagingDirectory, extractDirectory);
  } catch (error) {
    if (existsSync(backupDirectory) && !existsSync(extractDirectory)) renameSync(backupDirectory, extractDirectory);
    throw error;
  }
  rmSync(backupDirectory, { recursive: true, force: true });
  return executable;
}

function uvInstall(uvExecutable, python, args, indexUrl, environment, { taskId = activeTask, label = "Python 依赖", stall = false } = {}) {
  return runInstaller({
    command: uvExecutable,
    args: ["pip", "install", "--no-config", "--python", python.command, "--index-url", indexUrl, ...args],
    taskId,
    label,
    // uv streams every wheel into its own cache and then unpacks it into the environment, so those
    // two trees between them account for every byte the phase fetches.
    meterDirectories: [uvCacheDirectory(), path.join(projectRoot, ".venv")],
    // Only worth arming when there is another mirror to move to; on the last route, waiting is
    // strictly better than killing a slow but living install.
    stallMs: stall ? installStallMs : 0,
    options: {
      env: {
        ...environment,
        UV_CACHE_DIR: uvCacheDirectory(),
        UV_CONCURRENT_DOWNLOADS: String(packageConcurrency),
        UV_HTTP_RETRIES: "4",
        UV_HTTP_TIMEOUT: "120",
      },
    },
  });
}

/** Install through the mirror that measures fastest, and leave it the moment it stops delivering.
 *
 * Choosing a mirror once and committing the whole phase to it is what made the configurator "stick
 * to a single source": the ranking is taken before a byte is installed, and a mirror that answers a
 * probe quickly is regularly not the one that sustains a rate for the next twenty minutes. Failure
 * was the only thing that moved it on — a mirror that simply stopped delivering held the install
 * open indefinitely. Silence now counts as failure too.
 */
async function uvInstallFastest({ taskId, packageName, probePackage = packageName, uvExecutable, python, args, environment }) {
  const choice = await choosePackageSource(taskId, "pip", probePackage);
  const routeNames = [...new Set([
    choice.name,
    ...choice.ranked.filter((route) => route.ok).map((route) => route.id),
    "official",
  ])];
  let result;
  for (let index = 0; index < routeNames.length; index += 1) {
    const name = routeNames[index];
    const route = sources[name];
    if (index > 0) {
      emit("warning", { message: `${packageName} 当前线路${result?.stalled ? "长时间无响应" : "失败"}，切换到 ${route.label}` });
      emit("route", {
        id: taskId,
        label: route.label,
        url: route.pip,
        connections: packageConcurrency,
        mode: `包级并发 ×${packageConcurrency}`,
      });
    }
    result = await uvInstall(uvExecutable, python, args, route.pip, environment, {
      taskId,
      label: packageName,
      stall: index + 1 < routeNames.length,
    });
    if (result.status === 0) return result;
  }
  return result;
}

function decodeHtml(value) {
  return value.replace(/&amp;/g, "&").replace(/&#x2B;/gi, "+").replace(/&#43;/g, "+");
}

function torchSourceName(indexUrl) {
  return Object.entries(sources).find(([, candidate]) => indexUrl.startsWith(candidate.torch))?.[0] || "custom";
}

function wheelMatches(filename, version, pythonTag) {
  const decoded = decodeURIComponent(filename).replace(/\+/g, "+");
  if (!decoded.startsWith(`torch-${version}-${pythonTag}-${pythonTag}-`)) return false;
  if (process.platform === "win32") {
    return decoded.endsWith(process.arch === "arm64" ? "-win_arm64.whl" : "-win_amd64.whl");
  }
  if (process.platform === "linux") {
    const architecture = process.arch === "arm64" ? "aarch64" : "x86_64";
    return decoded.endsWith(`-${architecture}.whl`);
  }
  return false;
}

async function fetchText(url) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetchWithProxy(url, {
      headers: { "User-Agent": "XirAI-Setup/0.1" },
      redirect: "follow",
      signal: controller.signal,
    });
    return response.ok ? await response.text() : "";
  } catch (error) {
    if (proxyPolicy.useProxy) throw error;
    return "";
  } finally {
    clearTimeout(timeout);
  }
}

async function findTorchWheel(python, torchPlan, indexes) {
  const versionResult = runPython(python, ["-I", "-c", "import sys;print(f'cp{sys.version_info.major}{sys.version_info.minor}')"], {
    cwd: projectRoot,
    env: isolatedPythonEnv(process.env),
    stdio: "pipe",
    encoding: "utf8",
  });
  if (versionResult.status !== 0) return null;
  const pythonTag = versionResult.stdout.trim();
  const pageRequests = indexes.flatMap((indexUrl) => ["torch/", ""].map(async (suffix) => {
    const pageUrl = `${indexUrl.replace(/\/$/, "")}/${suffix}`;
    const html = await fetchText(pageUrl);
    return html ? { indexUrl, pageUrl, html } : null;
  }));
  const pages = (await Promise.all(pageRequests)).filter(Boolean);

  let filename;
  let digest;
  const explicitRoutes = [];
  for (const page of pages) {
    for (const match of page.html.matchAll(/<a\s+[^>]*href=["']([^"']+)["'][^>]*>([^<]+)<\/a>/gi)) {
      const linkText = decodeHtml(match[2].trim());
      const linkFilename = path.posix.basename(new URL(decodeHtml(match[1]), page.pageUrl).pathname);
      const candidate = linkText.endsWith(".whl") ? linkText : linkFilename;
      if (!wheelMatches(candidate, torchPlan.version, pythonTag)) continue;
      filename ||= decodeURIComponent(linkFilename);
      const url = new URL(decodeHtml(match[1]), page.pageUrl);
      digest ||= url.hash.match(/^#sha256=([a-f0-9]{64})$/i)?.[1];
      url.hash = "";
      explicitRoutes.push({
        id: torchSourceName(page.indexUrl),
        label: sources[torchSourceName(page.indexUrl)]?.label || "自定义线路",
        url: url.href,
        indexUrl: page.indexUrl,
      });
    }
  }
  if (!filename) return null;

  const routes = [...explicitRoutes];
  for (const indexUrl of indexes) {
    const sourceName = torchSourceName(indexUrl);
    const root = indexUrl.replace(/\/$/, "");
    const encodedFilename = filename.replace(/\+/g, "%2B");
    for (const url of [`${root}/${encodedFilename}`, `${root}/torch/${encodedFilename}`]) {
      routes.push({
        id: sourceName,
        label: sources[sourceName]?.label || "自定义线路",
        url,
        indexUrl,
      });
    }
  }
  const seen = new Set();
  return {
    filename,
    digest,
    routes: routes
      .filter((route) => {
        if (seen.has(route.url)) return false;
        seen.add(route.url);
        return true;
      })
      .sort((first, second) => indexes.indexOf(first.indexUrl) - indexes.indexOf(second.indexUrl)),
  };
}

emit("task-list", { tasks });
console.log(`XiriaCanvas AI 一键环境配置 (${process.platform}/${process.arch})`);

startTask("machine");
const [nodeMajor, nodeMinor] = process.versions.node.split(".").map(Number);
if (nodeMajor < 22 || (nodeMajor === 22 && nodeMinor < 21)) fail("需要 Node.js 22.21 或更高版本");
if (proxyPolicy.useProxy && !proxyPolicy.configured) fail("已显式启用代理，但未设置 HTTP_PROXY、HTTPS_PROXY 或 ALL_PROXY");
if (!sources[requestedSource] && requestedSource !== "auto") fail(`未知下载源：${requestedSource}`);
if (rtxVsrChoiceError) fail(rtxVsrChoiceError);
if (acceleratorChoiceError) fail(acceleratorChoiceError);
if (requestedTorchVersion && !/^\d+\.\d+\.\d+(?:\+[a-z0-9.]+)?$/i.test(requestedTorchVersion)) fail(`无效 PyTorch 版本：${requestedTorchVersion}`);
if (requestedTorch === "auto" && requestedTorchVersion) fail("--torch-version 只在手动 --torch=cuXXX 或 --torch=cpu 下生效，自动模式不能限定精确版本");
if (requestedTorchVersion && !requestedTorchVersion.toLowerCase().endsWith(`+${requestedTorch.toLowerCase()}`)) fail(`${requestedTorchVersion} 本地后缀与 ${requestedTorch} 不一致`);
let nvidiaHardware = detectNvidiaSmi();
const detectionPython = findPython(projectRoot);
if (detectionPython) nvidiaHardware = mergeNvidiaDetection(nvidiaHardware, detectNvidiaDriverApi(detectionPython, process.env));
emit("machine", {
  platform: process.platform,
  architecture: process.arch,
  node: process.versions.node,
  nvidia: nvidiaHardware.available,
  gpu: nvidiaHardware.name,
  driver: nvidiaHardware.driverVersion,
  cuda: nvidiaHardware.cuda?.text || null,
});
completeTask("machine", `${process.platform}/${process.arch} · Node ${process.versions.node} · ${nvidiaHardware.available ? `${nvidiaHardware.name || "NVIDIA GPU"} · 驱动 CUDA 上限 ${nvidiaHardware.cuda?.text || "待确认"}` : "未检测到 NVIDIA 驱动"}`);

startTask("python");
const venvRoot = path.join(projectRoot, ".venv");
const venvPythonPath = getVenvPythonPath(projectRoot);
if (!diagnoseOnly && existsSync(venvRoot)) {
  const currentVenv = verifyProjectVenv(projectRoot);
  if (recreateVenv || !currentVenv.ok) {
    if (lstatSync(venvRoot).isSymbolicLink()) fail(".venv 是外部链接，为保护其他软件环境，请手动移除该链接");
    const backupPath = path.join(projectRoot, `.venv.backup-${new Date().toISOString().replace(/[:.]/g, "-")}`);
    renameSync(venvRoot, backupPath);
    console.warn(`旧环境已安全保留为 ${path.basename(backupPath)}`);
    emit("warning", { message: `旧环境已备份为 ${path.basename(backupPath)}` });
  }
}

const managedPythonEnvironment = {
  ...process.env,
  UV_CACHE_DIR: path.resolve(projectRoot, process.env.XIRAI_CACHE_DIR || ".cache", "uv-cache"),
  UV_PYTHON_INSTALL_DIR: path.resolve(projectRoot, process.env.XIRAI_CACHE_DIR || ".cache", "python"),
  UV_PYTHON_PREFERENCE: "only-managed",
  UV_NO_CONFIG: "1",
};

let basePython = findPython(projectRoot, { includeVenv: !recreateVenv });
if (!basePython && !diagnoseOnly) {
  const uvExecutable = await bootstrapUv();
  console.log("未检测到 Python，使用项目内 uv 自动下载 Python 3.12...");
  run(uvExecutable, ["venv", "--no-config", "--seed", "--python", "3.12", venvRoot], { env: managedPythonEnvironment });
  basePython = findPython(projectRoot);
}
if (!basePython) {
  const hint = process.platform === "win32"
    ? "未找到 Python 3.10-3.13。请安装 Python 3.12 或 uv 后再次双击配置器"
    : "未找到 Python 3.10-3.13。请安装 python3、python3-venv，或 uv 后再次运行配置器";
  fail(hint);
}

if (!diagnoseOnly && !existsSync(venvPythonPath)) {
  // A non-zero exit is not fatal by itself. Debian and Ubuntu ship ensurepip in a separate
  // python3-venv package, so on a stock image this command lays down the directory and the
  // interpreter symlink and *then* exits non-zero — which is precisely the case the uv repair
  // below was written to handle. Stopping here would send the user off to `apt install
  // python3-venv` for something the bundled uv fixes without any system package at all.
  runPython(basePython, ["-I", "-m", "venv", venvRoot], { cwd: projectRoot, env: isolatedPythonEnv(process.env) });
}
// A venv without pip is the shape `python -m venv` leaves behind when ensurepip is missing — the
// Debian and Ubuntu split of python3-venv is the usual cause. It creates the directory layout and
// the interpreter symlink first and only then fails, so the *next* run finds what looks like a
// working environment and breaks much later, somewhere that says nothing about the real cause.
// uv can seed a complete environment without any system package, so repair it rather than sending
// the user away to install one.
if (!diagnoseOnly) {
  // Only a `.venv` interpreter counts as seeded: when the creation above left nothing behind,
  // `findPython` would otherwise answer with the system interpreter, and a system pip would then
  // pass the check and leave the project with no environment at all.
  const seededPython = existsSync(venvPythonPath) ? findPython(projectRoot) : null;
  if (!seededPython || seededPython.source !== ".venv" || !pythonCheck(seededPython, "import ensurepip,pip", isolatedPythonEnv(process.env))) {
    emit("warning", { message: "项目 Python 环境缺少 pip（Linux 上通常是未安装 python3-venv），正在使用项目内 uv 重建完整环境" });
    if (existsSync(venvRoot)) {
      const backupPath = path.join(projectRoot, `.venv.backup-${new Date().toISOString().replace(/[:.]/g, "-")}`);
      renameSync(venvRoot, backupPath);
      console.warn(`不完整的环境已保留为 ${path.basename(backupPath)}`);
    }
    const uvExecutable = await bootstrapUv();
    run(uvExecutable, ["venv", "--no-config", "--seed", "--python", "3.12", venvRoot], { env: managedPythonEnvironment });
  }
}
const venvPython = diagnoseOnly ? basePython : findPython(projectRoot);
if (!venvPython || (!diagnoseOnly && venvPython.source !== ".venv")) fail("项目 Python 环境不可用");
if (!diagnoseOnly) {
  const verifiedVenv = verifyProjectVenv(projectRoot);
  if (!verifiedVenv.ok) fail(`${verifiedVenv.error}。请移除 .venv 后重试`);
}
const pythonVersionResult = runPython(venvPython, ["-I", "-c", "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')"], {
  cwd: projectRoot,
  env: isolatedPythonEnv(process.env),
  stdio: "pipe",
  encoding: "utf8",
});
if (pythonVersionResult.status !== 0) fail("无法读取项目 Python 版本");
const pythonVersion = pythonVersionResult.stdout.trim();
nvidiaHardware = mergeNvidiaDetection(nvidiaHardware, detectNvidiaDriverApi(venvPython, process.env));
emit("machine", {
  platform: process.platform,
  architecture: process.arch,
  node: process.versions.node,
  nvidia: nvidiaHardware.available,
  gpu: nvidiaHardware.name,
  driver: nvidiaHardware.driverVersion,
  cuda: nvidiaHardware.cuda?.text || null,
  computeCapability: nvidiaHardware.computeCapability || null,
});
completeTask("machine", `${process.platform}/${process.arch} · Node ${process.versions.node} · ${nvidiaHardware.available ? `${nvidiaHardware.name || "NVIDIA GPU"}${nvidiaHardware.computeCapability ? ` · 计算能力 sm_${nvidiaHardware.computeCapability.replace(".", "")}` : ""} · 驱动 CUDA 上限 ${nvidiaHardware.cuda?.text || "未知"}` : "未检测到 NVIDIA 驱动"}`);
if (!diagnoseOnly) {
  const environmentIdPath = path.join(venvRoot, ".xirai-environment-id");
  const environmentId = existsSync(environmentIdPath) ? readFileSync(environmentIdPath, "utf8").trim() : randomUUID();
  if (!existsSync(environmentIdPath)) writeFileSync(environmentIdPath, `${environmentId}\n`, "utf8");
  if (resumeState.environmentId !== environmentId) {
    resumeState = { ...resumeState, environmentId, completed: {} };
    persistResume();
  }
}
completeTask("python", `${venvPython.source} · ${venvPython.command}`);

const selectionFingerprint = digest(JSON.stringify({
  selectionPolicy: 4,
  platform: process.platform,
  architecture: process.arch,
  pythonVersion,
  nvidia: nvidiaHardware.available,
  gpu: nvidiaHardware.name,
  driver: nvidiaHardware.driverVersion,
  driverCuda: nvidiaHardware.cuda?.text || null,
  computeCapability: nvidiaHardware.computeCapability || null,
  requestedTorch,
  requestedTorchVersion,
  torchIndex: process.env.XIRAI_TORCH_INDEX || null,
}));
const cachedSelection = !refreshSelection
  && resumeState.selection?.fingerprint === selectionFingerprint
  && /^\d+\.\d+\.\d+(?:\+[a-z0-9.]+)?$/i.test(resumeState.selection?.torchPlan?.version || "")
  && /^(cpu|cu\d+)$/.test(resumeState.selection?.torchPlan?.variant || "")
  ? resumeState.selection.torchPlan
  : null;

startTask("source");
let sourceName = requestedSource;
let sourceDetail = "";
if (cachedSelection && sourceName === "auto") {
  sourceName = "official";
} else if (sourceName === "auto") {
  // This picks the Hugging Face endpoint and the order the PyTorch indexes are tried in, so it has
  // to be a bandwidth measurement. The four-second latency probe it replaced regularly named a
  // mirror that answers instantly and then transfers at a fraction of the official host's rate.
  const measured = await benchmarkRoutes(Object.entries(sources).map(([name, candidate]) => ({
    id: name,
    label: candidate.label,
    url: packageProbeUrl(candidate, "pip"),
  })), { fetcher: fetchWithProxy, sampleBytes: 1024 ** 2, sampleMs: 2000, timeoutMs: 9000 });
  const fastest = measured.find((route) => route.ok);
  sourceName = fastest?.id || "official";
  if (fastest) {
    sourceDetail = ` ${(fastest.speedBps / 1024 ** 2).toFixed(1)} MB/s · ${fastest.latencyMs}ms`;
  }
}
const source = sources[sourceName];
if (cachedSelection && requestedSource === "auto") {
  skipTask("source", "断点恢复 · 下载时逐项测速");
} else {
  completeTask("source", requestedSource === "auto"
    ? `逐项自动测速 · 初始 ${source.label}${sourceDetail}`
    : `固定 ${source.label}`);
}

startTask("select");
const huggingFaceHome = path.resolve(projectRoot, process.env.XIRAI_CACHE_DIR || ".cache", "huggingface");
const configuredHuggingFaceHome = process.env.HF_HOME || huggingFaceHome;
const installEnvironment = isolatedPythonEnv({
  ...process.env,
  HF_HOME: configuredHuggingFaceHome,
  HF_HUB_CACHE: process.env.HF_HUB_CACHE || path.join(configuredHuggingFaceHome, "hub"),
  // A measured mirror route means huggingface.co is the slow or blocked one; try
  // its mirror first instead of always stalling on the official host.
  HF_ENDPOINT: process.env.HF_ENDPOINT || source.hf,
  HF_HUB_ETAG_TIMEOUT: process.env.HF_HUB_ETAG_TIMEOUT || "15",
  HF_HUB_DOWNLOAD_TIMEOUT: process.env.HF_HUB_DOWNLOAD_TIMEOUT || "120",
});
const torchRoots = [...new Set([
  process.env.XIRAI_TORCH_INDEX,
  ...(requestedSource === "auto" ? uniqueSources("torch").map((name) => sources[name].torch) : [source.torch]),
  sources.official.torch,
].filter(Boolean))];
let torchPlan;
if (cachedSelection) {
  torchPlan = cachedSelection;
} else if (updateRepair && requestedTorch !== "auto" && requestedTorchVersion) {
  const requestedCuda = requestedTorch === "cpu" ? "None" : torchInternals.cudaVariantVersion(requestedTorch)?.text;
  const installedSelectionCheck = requestedCuda && [
    "import torch",
    `assert torch.__version__==${JSON.stringify(requestedTorchVersion)}`,
    `assert str(torch.version.cuda)==${JSON.stringify(requestedCuda)}`,
    ...(requestedTorch === "cpu" ? [] : [
      "assert torch.cuda.is_available()",
      "assert (torch.ones(1,device='cuda')+1).item()==2",
    ]),
  ].join(";");
  if (installedSelectionCheck && pythonCheck(venvPython, installedSelectionCheck, installEnvironment)) {
    torchPlan = {
      version: requestedTorchVersion,
      variant: requestedTorch,
      indexUrl: `${sources.official.torch}/${requestedTorch}`,
      indexRoot: sources.official.torch,
    };
  }
} else if (requestedTorch === "auto") {
  if (!nvidiaHardware.available) {
    fail("自动模式未检测到 NVIDIA GPU/驱动。无需安装系统 CUDA Toolkit，但必须先安装可用的 NVIDIA 显卡驱动；仅诊断时可显式使用 --torch=cpu");
  }
  if (!nvidiaHardware.cuda) {
    fail("已检测到 NVIDIA GPU，但无法读取驱动支持的 CUDA 上限。请更新 NVIDIA 驱动或检查 nvidia-smi 后重试，配置器不会静默安装 CPU 版 PyTorch");
  }
  torchPlan = resolveLatestStableTorch(venvPython, torchRoots, [source.pip, sources.official.pip], nvidiaHardware, installEnvironment);
} else {
  if (!/^(cpu|cu\d+)$/.test(requestedTorch)) fail(`无效 PyTorch 类型：${requestedTorch}`);
  if (requestedTorch === "cpu" && nvidiaHardware.available) {
    emit("warning", { message: "检测到 NVIDIA GPU，但已按用户显式设置选择 CPU 版 PyTorch；自动模式不会这样回退" });
  }
  if (requestedTorch !== "cpu" && !nvidiaHardware.available) {
    fail(`${requestedTorch} 需要 NVIDIA GPU 和可用驱动；不需要另行安装系统 CUDA Toolkit`);
  }
  if (requestedTorch !== "cpu" && !nvidiaHardware.cuda) {
    fail(`无法读取 NVIDIA 驱动支持的 CUDA 上限，不能安全安装 ${requestedTorch}`);
  }
  if (requestedTorch !== "cpu" && !torchInternals.cudaVariantIsCompatible(requestedTorch, nvidiaHardware.cuda)) {
    fail(`${requestedTorch} 超过当前 NVIDIA 驱动支持的 CUDA ${nvidiaHardware.cuda?.text || "未知"} 上限`);
  }
  // The driver may allow a runtime the card itself cannot execute: CUDA 13 wheels carry no kernels
  // below sm_75, so a Tesla V100 installs cu130 successfully and then fails every real operation.
  if (requestedTorch !== "cpu" && !torchInternals.cudaVariantSupportsDevice(requestedTorch, nvidiaHardware.computeCapability)) {
    const minimum = torchInternals.minimumComputeCapability(requestedTorch);
    fail(`${requestedTorch} 的 wheel 不包含 ${nvidiaHardware.name || "当前 GPU"}（计算能力 sm_${String(nvidiaHardware.computeCapability).replace(".", "")}）的内核，最低需要 sm_${minimum.major}${minimum.minor}；请改用较低的 CUDA 版本`);
  }
  const manualPlans = [];
  for (const root of torchRoots) {
    const directVariant = torchInternals.directRootVariant(root);
    if (directVariant && directVariant !== requestedTorch) continue;
    const indexUrl = directVariant ? root.replace(/\/$/, "") : `${root.replace(/\/$/, "")}/${requestedTorch}`;
    const versions = availableTorchVersions(venvPython, indexUrl, installEnvironment)
      .filter((version) => !requestedTorchVersion || version === requestedTorchVersion);
    for (const version of versions) manualPlans.push({ version, variant: requestedTorch, indexUrl, indexRoot: root });
  }
  torchPlan = torchInternals.uniquePlans(manualPlans)[0];
}
if (!torchPlan && updateRepair && requestedTorch !== "auto" && requestedTorchVersion) {
  const manualPlans = [];
  for (const root of torchRoots) {
    const directVariant = torchInternals.directRootVariant(root);
    if (directVariant && directVariant !== requestedTorch) continue;
    const indexUrl = directVariant ? root.replace(/\/$/, "") : `${root.replace(/\/$/, "")}/${requestedTorch}`;
    const versions = availableTorchVersions(venvPython, indexUrl, installEnvironment)
      .filter((version) => version === requestedTorchVersion);
    for (const version of versions) manualPlans.push({ version, variant: requestedTorch, indexUrl, indexRoot: root });
  }
  torchPlan = torchInternals.uniquePlans(manualPlans)[0];
}
if (!torchPlan) fail(requestedTorchVersion
  ? `没有找到 PyTorch ${requestedTorchVersion} / ${requestedTorch} 对应当前系统、Python 和架构的 wheel`
  : `没有找到与当前系统、Python、架构和 NVIDIA 驱动兼容的稳定 PyTorch CUDA wheel${nvidiaHardware.computeCapability ? `（GPU 计算能力 sm_${nvidiaHardware.computeCapability.replace(".", "")}，只有为该架构编译的 CUDA wheel 可用）` : ""}`);
if (!cachedSelection) {
  persistResume({ selection: { fingerprint: selectionFingerprint, torchPlan } });
  completeTask("select", `PyTorch ${torchPlan.version} · ${torchPlan.variant} · 一次性选择最新稳定兼容组合`);
} else {
  skipTask("select", `断点恢复 · PyTorch ${torchPlan.version} · ${torchPlan.variant}`);
}
if (torchPlan.variant === "cpu") {
  writeDiagnosticLog(projectRoot, {
    kind: "cpu-mode-selected",
    message: "The environment configuration selected CPU-only PyTorch.",
    details: {
      torch_version: torchPlan.version,
      torch_variant: torchPlan.variant,
      requested_torch: requestedTorch,
      gpu_detected: nvidiaHardware.available,
      gpu_name: nvidiaHardware.name || null,
    },
  });
}
let expectedCuda = torchPlan.variant === "cpu" ? "None" : torchInternals.cudaVariantVersion(torchPlan.variant)?.text;
if (!expectedCuda) fail(`无法解析 PyTorch CUDA Runtime 标签：${torchPlan.variant}`);
emit("selection", {
  source: requestedSource === "auto" ? "逐项自动选路" : source.label,
  driverCuda: nvidiaHardware.cuda?.text || null,
  computeCapability: nvidiaHardware.computeCapability || null,
  cudaRuntime: expectedCuda,
  gpu: nvidiaHardware.name,
  torch: torchPlan.version,
  variant: torchPlan.variant,
  python: venvPython.command,
  xformers: installXformers,
  rtxVsr: requestedRtxVsr,
  resumed: Boolean(cachedSelection),
});

const backendPackages = readFileSync(path.join(projectRoot, "backend", "requirements.txt"), "utf8")
  .split(/\r?\n/)
  .map((line) => line.trim())
  .filter((line) => line && !line.startsWith("#"));
const setupCacheDirectory = path.resolve(projectRoot, process.env.XIRAI_CACHE_DIR || ".cache", "setup");
mkdirSync(setupCacheDirectory, { recursive: true });
const npmConfigPath = path.join(setupCacheDirectory, "npmrc");
writeFileSync(npmConfigPath, "# XiriaCanvas setup-owned npm boundary\nproxy=null\nhttps-proxy=null\n", { encoding: "utf8", mode: 0o600 });
const npmEnvironment = npmInstallerEnvironment(process.env, {
  useProxy: proxyPolicy.useProxy,
  userConfig: npmConfigPath,
});
const torchConstraintPath = path.join(setupCacheDirectory, "torch-constraint.txt");
writeFileSync(torchConstraintPath, `torch==${torchPlan.version}\n`, "utf8");
const downloadPlan = [
  ...bootstrapDownloads,
  { id: "tools", label: "Python 基础工具", detail: "pip / setuptools / wheel" },
  { id: "torch", label: `PyTorch ${torchPlan.version}`, detail: `${torchPlan.variant} · 含匹配的 torchvision · 自动测速 · 大文件分片 ×${downloadConnections}` },
  { id: "backend", label: "后端依赖（含 Ultralytics YOLO）", detail: `${backendPackages.join(" · ")} · 包级并发 ×${packageConcurrency}` },
  ...(requestedRtxVsr ? [{ id: "rtx-vsr", label: `NVIDIA RTX VSR ${RTX_VSR_VERSION}`, detail: "专有可选组件 · 仅合格环境安装" }] : []),
  ...(installXformers ? [{ id: "xformers", label: "xformers", detail: `匹配 ${torchPlan.variant}，默认安装` }] : []),
  ...(requestedTriton ? [{ id: "triton", label: tritonPackageName(), detail: "可选 · Transformer 编译加速的 Inductor 后端" }] : []),
  ...(requestedSageAttention ? [{ id: "sageattention", label: "SageAttention", detail: "可选 · Anima 自注意力 INT8 内核，依赖 Triton" }] : []),
  ...(!skipNode ? [{ id: "node", label: "前端依赖", detail: "React · Vite · Lucide · Undici" }] : []),
];
emit("plan", { downloads: downloadPlan });

if (diagnoseOnly) {
  for (const id of ["tools", "torch", "backend", "rtx-vsr", "xformers", "node", "verify"]) skipTask(id, "仅检测模式");
  emit("complete", { message: "环境检测完成" });
  process.exit(0);
}

const uvExecutable = await bootstrapUv();

const modelPathConfig = JSON.parse(readFileSync(path.join(projectRoot, "models", "model-paths.json"), "utf8"));
for (const directory of configuredModelDirectories(modelPathConfig, projectRoot)) mkdirSync(directory, { recursive: true });

startTask("tools");
const toolsFingerprint = digest(`pip-setuptools-wheel:${pythonVersion}`);
let result;
if (pythonCheck(venvPython, "import pip,setuptools,wheel", installEnvironment)) {
  if (!taskCheckpoint("tools", toolsFingerprint)) checkpointTask("tools", toolsFingerprint);
  skipTask("tools", "断点恢复 · 已安装并验证");
} else {
  result = await uvInstallFastest({
    taskId: "tools",
    packageName: "Python 基础工具",
    probePackage: "pip",
    uvExecutable,
    python: venvPython,
    args: ["--upgrade", "pip", "setuptools", "wheel"],
    environment: installEnvironment,
  });
  if (result.status !== 0) fail("Python 基础工具安装失败");
  checkpointTask("tools", toolsFingerprint);
  completeTask("tools");
}

startTask("torch", `PyTorch ${torchPlan.version} · ${torchPlan.variant}`);

function indexesForTorchPlan(plan) {
  const forVariant = (root) => {
    const directVariant = torchInternals.directRootVariant(root);
    if (directVariant && directVariant !== plan.variant) return [];
    return [directVariant ? root.replace(/\/$/, "") : `${root.replace(/\/$/, "")}/${plan.variant}`];
  };
  return [...new Set([
    // An explicitly configured index is the user's answer to "where do I download PyTorch from",
    // and it has to outrank the index the plan happened to be resolved against — otherwise setting
    // XIRAI_TORCH_INDEX to a fast mirror still downloads gigabytes from the slow default whenever
    // the default is where the newest wheel was found.
    ...(process.env.XIRAI_TORCH_INDEX ? forVariant(process.env.XIRAI_TORCH_INDEX) : []),
    plan.indexUrl,
    ...torchRoots.flatMap(forVariant),
  ].filter(Boolean))];
}

/** torchvision ships compiled ops that bind to one exact PyTorch build.
 *
 * Nothing in `backend/requirements.txt` names it — it arrives as an Ultralytics dependency, from
 * whichever index resolved the backend packages — so it can land as a build compiled against a
 * different CUDA runtime than the torch beside it. That mismatch is silent until the first call
 * into its C++ ops, where it surfaces as `operator torchvision::nms does not exist` and takes
 * `transformers.AutoImageProcessor` and every Diffusers import down with it. So it is installed
 * from the torch plan's own index, and verified by running one of those ops.
 */
function checkForTorchvision(plan) {
  const cudaRuntime = plan.variant === "cpu" ? "None" : torchInternals.cudaVariantVersion(plan.variant)?.text;
  if (!cudaRuntime) return null;
  return [
    "import torch,torchvision",
    `assert torch.__version__==${JSON.stringify(plan.version)}`,
    // torchvision reports its CUDA build as a packed integer from 0.28 onward — 12060 for 12.6 —
    // where it used to be a dotted string and where torch still is one. Compared raw, a correctly
    // matched pair reads as a mismatch: torchvision 0.28.0+cu126 beside torch 2.13.0+cu126 failed
    // this assert on every mirror in turn and ended the whole run with "no matching torchvision",
    // having in fact installed the right wheel. Both forms are normalised to major.minor first.
    "_cuda=lambda v:(lambda t: f'{int(t)//1000}.{(int(t)%1000)//10}' if t.isdigit() and len(t)>=4 else t)(str(v))",
    `assert _cuda(torchvision.version.cuda)==_cuda(${JSON.stringify(cudaRuntime)}), f'torchvision built for CUDA {torchvision.version.cuda}, torch for {torch.version.cuda}'`,
    "from torchvision.ops import nms",
    "assert nms(torch.tensor([[0.,0.,1.,1.],[0.,0.,1.,1.]]),torch.tensor([0.9,0.5]),0.5).numel()==1",
  ].join(";");
}

function checkForTorchPlan(plan) {
  const cudaRuntime = plan.variant === "cpu" ? "None" : torchInternals.cudaVariantVersion(plan.variant)?.text;
  if (!cudaRuntime) return null;
  return [
    "import torch",
    `assert torch.__version__==${JSON.stringify(plan.version)}`,
    `assert str(torch.version.cuda)==${JSON.stringify(cudaRuntime)}`,
    ...(plan.variant === "cpu" ? [] : [
      "assert torch.cuda.is_available()",
      "capability=torch.cuda.get_device_capability(0)",
      "assert (torch.ones(1,device='cuda')+1).item()==2",
      "torch.cuda.synchronize()",
      "print(f'CUDA device verified: {torch.cuda.get_device_name(0)} sm_{capability[0]}{capability[1]}')",
    ]),
  ].join(";");
}

async function installTorchCandidate(plan) {
  const indexes = indexesForTorchPlan(plan);
  const planFingerprint = digest(`${selectionFingerprint}:${plan.version}:${plan.variant}`);
  const downloadFingerprint = digest(`${planFingerprint}:${requestedSource}:${process.env.XIRAI_TORCH_INDEX || ""}`);
  let wheel = resumeState.torchDownload?.fingerprint === downloadFingerprint ? resumeState.torchDownload.wheel : null;
  if (!wheel) {
    wheel = await findTorchWheel(venvPython, plan, indexes);
    if (wheel) persistResume({ torchDownload: { fingerprint: downloadFingerprint, wheel } });
  }
  let installed = false;
  if (wheel) {
    const wheelDirectory = path.resolve(projectRoot, process.env.XIRAI_CACHE_DIR || ".cache", "downloads");
    const wheelPath = path.join(wheelDirectory, wheel.filename);
    try {
      await downloadFile({
        routes: wheel.routes,
        destination: wheelPath,
        expectedSha256: wheel.digest,
        connections: downloadConnections,
        fetcher: fetchWithProxy,
        rankRoutes: requestedSource === "auto",
        onRoute: (route) => {
          const mode = route.cached ? "已缓存 · 无需下载" : route.connections > 1 ? `分片下载 ×${route.connections}` : "单连接下载";
          emit("route", { id: "torch", ...route, mode });
          console.log(`\nPyTorch 自动选择线路：${route.label} · ${mode}`);
        },
        onProgress: (progress) => emit("download", { name: wheel.filename, ...progress, route: progress.cached ? "本地缓存" : progress.route }),
        onWarning: (message) => emit("warning", { message }),
      });
      result = await uvInstallFastest({
        taskId: "torch",
        packageName: "PyTorch 运行依赖",
        probePackage: "filelock",
        uvExecutable,
        python: venvPython,
        args: [wheelPath],
        environment: installEnvironment,
      });
      installed = result.status === 0;
    } catch (error) {
      emit("warning", { message: `PyTorch ${plan.version} ${plan.variant} 分片下载失败：${error.message}，改用安装器下载` });
    }
  }
  if (!installed) {
    for (const indexUrl of indexes) {
      console.log(`\n正在从 ${indexUrl} 安装 PyTorch ${plan.version}`);
      emit("route", { id: "torch", label: sources[torchSourceName(indexUrl)]?.label || "备用线路", url: indexUrl, connections: packageConcurrency, mode: `包级并发 ×${packageConcurrency}` });
      result = await uvInstall(uvExecutable, venvPython, [`torch==${plan.version}`], indexUrl, installEnvironment, {
        taskId: "torch",
        label: `PyTorch ${plan.version}`,
        stall: indexUrl !== indexes[indexes.length - 1],
      });
      if (result.status === 0) {
        installed = true;
        break;
      }
      emit("warning", { message: `PyTorch 线路${result.stalled ? "长时间无响应" : "失败"}，正在切换：${indexUrl}` });
    }
  }
  return { installed, verified: installed && pythonCheck(venvPython, checkForTorchPlan(plan), installEnvironment) };
}

const torchCheck = checkForTorchPlan(torchPlan);
if (!torchCheck) fail(`无法验证 PyTorch CUDA Runtime 标签：${torchPlan.variant}`);
const torchvisionCheck = checkForTorchvision(torchPlan);
const torchIndexes = indexesForTorchPlan(torchPlan);
const torchFingerprint = digest(`${selectionFingerprint}:${torchPlan.version}:${torchPlan.variant}:torchvision-v1`);

async function installTorchvision() {
  for (const indexUrl of torchIndexes) {
    console.log(`\n正在从 ${indexUrl} 安装匹配 ${torchPlan.variant} 的 torchvision`);
    emit("route", { id: "torch", label: sources[torchSourceName(indexUrl)]?.label || "备用线路", url: indexUrl, connections: packageConcurrency, mode: `包级并发 ×${packageConcurrency}` });
    const attempt = await uvInstall(uvExecutable, venvPython, [
      "--constraint", path.relative(projectRoot, torchConstraintPath), "torchvision",
    ], indexUrl, installEnvironment, {
      taskId: "torch",
      label: "torchvision",
      stall: indexUrl !== torchIndexes[torchIndexes.length - 1],
    });
    if (attempt.status === 0 && pythonCheck(venvPython, torchvisionCheck, installEnvironment)) return true;
    emit("warning", { message: `torchvision 线路${attempt.stalled ? "长时间无响应" : "失败或与 PyTorch " + torchPlan.version + " 不匹配"}，正在切换：${indexUrl}` });
  }
  return false;
}

if (pythonCheck(venvPython, torchCheck, installEnvironment) && pythonCheck(venvPython, torchvisionCheck, installEnvironment)) {
  if (!taskCheckpoint("torch", torchFingerprint)) checkpointTask("torch", torchFingerprint);
  emit("route", { id: "torch", label: "本地环境", connections: 0, mode: "已验证 · 无需下载" });
  skipTask("torch", `已验证 PyTorch ${torchPlan.version} · ${torchPlan.variant} · torchvision 匹配`);
} else {
  if (!pythonCheck(venvPython, torchCheck, installEnvironment)) {
    const outcome = await installTorchCandidate(torchPlan);
    if (!outcome.installed) fail(`PyTorch ${torchPlan.version} · ${torchPlan.variant} 从镜像和官方源均安装失败`);
    if (!outcome.verified) fail(
      `PyTorch ${torchPlan.version} · ${torchPlan.variant} 已安装，但未能在当前 GPU 上完成 CUDA 实际运算`,
      { repairable: true, failureKind: "cuda-verification" },
    );
  }
  // Ahead of the backend dependencies, so Ultralytics finds a matching torchvision already in
  // place instead of resolving one against whichever index answered fastest.
  if (!await installTorchvision()) fail(`没有找到与 PyTorch ${torchPlan.version} · ${torchPlan.variant} 匹配的 torchvision`);
  checkpointTask("torch", torchFingerprint);
  completeTask("torch", `PyTorch ${torchPlan.version} · ${torchPlan.variant} · CUDA 实际运算通过 · torchvision 匹配`);
}
// The backend install must not be free to swap the build that was just verified: `--constraint`
// is what stops Ultralytics' loose torchvision requirement from resolving to another one.
const installedTorchvision = runPython(venvPython, ["-I", "-c", "import torchvision;print(torchvision.__version__)"], {
  cwd: projectRoot,
  env: installEnvironment,
  stdio: "pipe",
  encoding: "utf8",
  timeout: 30000,
});
if (installedTorchvision.status === 0 && installedTorchvision.stdout.trim()) {
  writeFileSync(torchConstraintPath, `torch==${torchPlan.version}\ntorchvision==${installedTorchvision.stdout.trim()}\n`, "utf8");
}

startTask("backend");
const pipelineConfigsScript = path.join(projectRoot, "backend", "pipeline_configs.py");
const backendFingerprint = digest(`${pythonVersion}:${torchPlan.version}:${backendPackages.join("\n")}:${readFileSync(pipelineConfigsScript, "utf8")}`);
const backendCheck = "import accelerate,diffusers,fastapi,huggingface_hub,numpy,onnxruntime,peft,PIL,psutil,pydantic,safetensors,scipy,spandrel,spandrel_extra_arches,transformers,ultralytics,uvicorn;from diffusers import AutoencoderKLQwenImage,AutoencoderKLWan,CosmosTransformer3DModel;from diffusers.loaders.single_file_utils import convert_cosmos_transformer_checkpoint_to_diffusers,convert_wan_vae_to_diffusers;from transformers import Qwen3Config,Qwen3Model,Qwen3VLTextModel,T5TokenizerFast";
const backendDependenciesReady = pythonCheck(venvPython, backendCheck, installEnvironment);
const checkPipelineConfigs = (scope = []) => runPython(venvPython, [pipelineConfigsScript, "--check", ...scope, "--installed", projectRoot], {
  cwd: projectRoot,
  env: installEnvironment,
  stdio: "ignore",
  timeout: 30000,
}).status === 0;
const backendCheckpointReady = taskCheckpoint("backend", backendFingerprint, ["complete", "partial"]);
const pipelineConfigsReady = backendDependenciesReady && checkPipelineConfigs();
if (backendCheckpointReady && backendDependenciesReady && pipelineConfigsReady) {
  emit("route", { id: "backend", label: "本地环境", connections: 0, mode: "断点恢复 · 无需下载" });
  skipTask("backend", "断点恢复 · 后端依赖与模型运行配置已就绪");
} else {
  if (!backendCheckpointReady || !backendDependenciesReady) {
    result = await uvInstallFastest({
      taskId: "backend",
      packageName: "后端依赖",
      probePackage: "pillow",
      uvExecutable,
      python: venvPython,
      args: ["--upgrade", "--constraint", path.relative(projectRoot, torchConstraintPath), "-r", path.join("backend", "requirements.txt")],
      environment: installEnvironment,
    });
    if (result.status !== 0) fail("后端依赖安装失败");
  }
  let pipelineConfigsDeferred = false;
  if (!checkPipelineConfigs()) {
    emit("route", { id: "backend", label: "Hugging Face / HF-Mirror", connections: 1, mode: "下载本地模型运行配置" });
    // `huggingface_hub` draws a per-file progress bar, so this phase can report real bytes rather
    // than only the cache growing underneath it.
    result = await runInstaller({
      command: venvPython.command,
      args: [...venvPython.args, pipelineConfigsScript, "--download", "--installed", projectRoot],
      taskId: "backend",
      label: "模型运行配置",
      meterDirectories: [configuredHuggingFaceHome],
      timeoutMs: 300000,
      options: { env: installEnvironment },
    });
    // The runtime configs are an offline prefetch, not a dependency: generation
    // checks them itself and reports a clear message. An unreachable Hugging Face
    // must not discard an otherwise complete environment.
    pipelineConfigsDeferred = result.status !== 0;
    if (pipelineConfigsDeferred) {
      warn(checkPipelineConfigs(["--required"])
        ? "模型运行配置下载失败（Hugging Face 与 HF-Mirror 均不可达）；已安装的模型不受影响，下次配置或联网后会自动补齐"
        : "模型运行配置下载失败（Hugging Face 与 HF-Mirror 均不可达）；已安装的底模在补齐前无法生成，可在 .env 设置 HF_ENDPOINT，或显式启用 --use-proxy 后重新配置");
    }
  }
  checkpointTask("backend", backendFingerprint, pipelineConfigsDeferred ? "partial" : "complete");
  completeTask("backend", pipelineConfigsDeferred ? "后端依赖已就绪 · 模型运行配置待联网补齐" : "后端依赖与离线模型运行配置已就绪");
}
if (!pythonCheck(venvPython, torchCheck, installEnvironment)) {
  fail("后端依赖安装改变了已验证的 PyTorch/CUDA 组合，已停止配置以避免错误回退", { repairable: true, failureKind: "dependency-changed-torch" });
}
if (!pythonCheck(venvPython, torchvisionCheck, installEnvironment)) {
  fail("后端依赖安装替换了与 PyTorch 匹配的 torchvision，已停止配置以避免 Ultralytics 与 Diffusers 导入失败", { repairable: true, failureKind: "dependency-changed-torch" });
}

let rtxVsrAvailable = false;
let rtxVsrReason = requestedRtxVsr ? "尚未验证" : "用户未选择";
if (requestedRtxVsr) {
  startTask("rtx-vsr", `NVIDIA RTX VSR ${RTX_VSR_VERSION} · 专有可选组件`);
  const torchDevice = readTorchDeviceMetadata(venvPython, installEnvironment);
  const eligibility = rtxVsrEligibility({
    platform: process.platform,
    architecture: process.arch,
    pythonVersion,
    driverVersion: nvidiaHardware.driverVersion,
    driverModel: detectNvidiaDriverModel(torchDevice?.deviceName),
    cudaAvailable: torchDevice?.cudaAvailable === true,
    torchCudaVersion: torchDevice?.torchCudaVersion,
    computeCapability: torchDevice?.computeCapability,
    isWsl: process.platform === "linux" && (Boolean(process.env.WSL_DISTRO_NAME) || /microsoft/i.test(os.release())),
  });
  if (eligibility.warning) emit("warning", { message: eligibility.warning });
  if (!eligibility.supported) {
    rtxVsrReason = eligibility.reason;
    emit("warning", { message: `已请求 RTX VSR，但当前环境不符合条件：${eligibility.reason}；未下载可选包` });
    skipTask("rtx-vsr", `${eligibility.reason} · 未下载`);
  } else {
    console.log(`\nRTX VSR API / 产品来源：${RTX_VSR_PRODUCT_SOURCE}`);
    const packageCheck = `import importlib.metadata;assert importlib.metadata.version('nvidia-vfx')==${JSON.stringify(RTX_VSR_VERSION)}`;
    let packageReady = pythonCheck(venvPython, packageCheck, installEnvironment);
    if (!packageReady) {
      emit("route", { id: "rtx-vsr", label: "NVIDIA Python Package Index", url: RTX_VSR_INDEX, connections: packageConcurrency, mode: "仅二进制 wheel · 不解析依赖" });
      result = await uvInstall(uvExecutable, venvPython, [
        "--only-binary", ":all:", "--no-deps", `nvidia-vfx==${RTX_VSR_VERSION}`,
      ], RTX_VSR_INDEX, installEnvironment, { taskId: "rtx-vsr", label: `nvidia-vfx ${RTX_VSR_VERSION}` });
      packageReady = result.status === 0 && pythonCheck(venvPython, packageCheck, installEnvironment);
    }
    const torchUnchanged = pythonCheck(venvPython, torchCheck, installEnvironment);
    if (!torchUnchanged) {
      rtxVsrReason = "PyTorch 完整性复核失败";
      emit("warning", { message: "可选 NVIDIA RTX VSR 安装后 PyTorch/CUDA 完整性验证未通过，继续进入基础环境的最终验证" });
      skipTask("rtx-vsr", "PyTorch 完整性复核失败 · 基础环境继续");
    } else if (!packageReady) {
      rtxVsrReason = "可选包安装或版本验证失败";
      emit("warning", { message: `可选 NVIDIA RTX VSR 包 nvidia-vfx==${RTX_VSR_VERSION} 安装或版本验证失败，继续完成基础环境配置` });
      skipTask("rtx-vsr", "可选包安装失败 · 基础环境继续");
    } else {
      if (probeRtxVsrAdapter(venvPython, installEnvironment)) {
        rtxVsrAvailable = true;
        rtxVsrReason = null;
        completeTask("rtx-vsr", `nvidia-vfx ${RTX_VSR_VERSION} · 实际 VSR 探针通过`);
      } else {
        rtxVsrReason = "实际 VSR 探针失败";
        emit("warning", { message: "可选 NVIDIA RTX VSR 实际探针失败，继续完成基础环境配置" });
        skipTask("rtx-vsr", "实际探针失败 · 基础环境继续");
      }
    }
  }
} else {
  const packageInstalled = pythonCheck(venvPython, "import importlib.metadata;importlib.metadata.version('nvidia-vfx')", installEnvironment);
  const supportedPackageInstalled = pythonCheck(venvPython, `import importlib.metadata;assert importlib.metadata.version('nvidia-vfx')==${JSON.stringify(RTX_VSR_VERSION)}`, installEnvironment);
  if (packageInstalled && rtxVsrExplicitlyDisabled) {
    result = runPython(venvPython, ["-m", "pip", "--isolated", "uninstall", "-y", "nvidia-vfx"], {
      cwd: projectRoot,
      env: installEnvironment,
    });
    if (result.status !== 0 || pythonCheck(venvPython, "import importlib.metadata;importlib.metadata.version('nvidia-vfx')", installEnvironment)) {
      fail("已选择关闭 RTX VSR，但无法移除项目环境中的 nvidia-vfx");
    }
    skipTask("rtx-vsr", "已移除专有可选组件");
  } else if (supportedPackageInstalled) {
    rtxVsrAvailable = probeRtxVsrAdapter(venvPython, installEnvironment);
    rtxVsrReason = rtxVsrAvailable ? null : "保留的可选运行时未通过实际 VSR 探针";
    skipTask("rtx-vsr", rtxVsrAvailable ? "未指定 RTX VSR 选项 · 保留并验证现有运行时" : "保留现有运行时 · 实际探针失败");
  } else if (packageInstalled) {
    rtxVsrReason = `已安装的 nvidia-vfx 版本不是受支持的 ${RTX_VSR_VERSION}`;
    skipTask("rtx-vsr", "未指定 RTX VSR 选项 · 保留现有但不兼容的运行时");
  } else {
    skipTask("rtx-vsr", "未选择专有可选组件 · 未下载");
  }
}
emit("feature", {
  id: "rtx-vsr",
  requested: requestedRtxVsr,
  available: rtxVsrAvailable,
  version: rtxVsrAvailable ? RTX_VSR_VERSION : null,
  reason: rtxVsrReason,
});

if (installXformers) {
  startTask("xformers", "默认安装，失败时自动使用 PyTorch 原生注意力");
  const xformersFingerprint = digest(`${torchFingerprint}:xformers:cuda-smoke-v2`);
  const xformersCheck = xformersCudaSmoke(torchPlan.version);
  if (taskCheckpoint("xformers", xformersFingerprint) && pythonCheck(venvPython, xformersCheck, installEnvironment)) {
    emit("route", { id: "xformers", label: "本地环境", connections: 0, mode: "断点恢复 · 无需下载" });
    skipTask("xformers", "断点恢复 · xformers 已安装");
  } else if (taskCheckpoint("xformers", xformersFingerprint, ["skipped"])) {
    skipTask("xformers", "断点恢复 · 当前版本不兼容，继续使用原生注意力");
  } else {
    const xformersIndexes = torchIndexes.map((indexUrl) => ({
      id: torchSourceName(indexUrl),
      label: sources[torchSourceName(indexUrl)]?.label || "自定义线路",
      url: `${indexUrl.replace(/\/$/, "")}/xformers/`,
      indexUrl,
    }));
    const rankedXformers = requestedSource === "auto"
      ? await benchmarkRoutes(xformersIndexes, { fetcher: fetchWithProxy, sampleBytes: 96 * 1024, timeoutMs: 7000 })
      : xformersIndexes.map((route) => ({ ...route, ok: true }));
    result = { status: 1 };
    const usableXformers = rankedXformers.filter((item) => item.ok);
    for (const [index, route] of usableXformers.entries()) {
      emit("route", { id: "xformers", label: route.label, url: route.indexUrl, connections: packageConcurrency, mode: `包级并发 ×${packageConcurrency}` });
      result = await uvInstall(uvExecutable, venvPython, ["xformers", "--no-deps"], route.indexUrl, installEnvironment, {
        taskId: "xformers",
        label: "xformers",
        stall: index + 1 < usableXformers.length,
      });
      if (result.status === 0) break;
      emit("warning", { message: `xformers 线路${result.stalled ? "长时间无响应" : "失败"}，正在切换：${route.label}` });
    }
    if (result.status === 0) {
      if (!pythonCheck(venvPython, xformersCheck, installEnvironment)) {
        runPython(venvPython, ["-m", "pip", "--isolated", "uninstall", "-y", "xformers"], { cwd: projectRoot, env: installEnvironment });
        emit("warning", { message: "xformers 与当前 PyTorch 不兼容，已安全移除并启用原生注意力" });
        checkpointTask("xformers", xformersFingerprint, "skipped");
        skipTask("xformers", "不兼容，已使用 PyTorch 原生注意力");
      } else {
        checkpointTask("xformers", xformersFingerprint);
        completeTask("xformers");
      }
    } else {
      emit("warning", { message: "没有兼容的 xformers wheel，继续使用 PyTorch 原生注意力" });
      checkpointTask("xformers", xformersFingerprint, "skipped");
      skipTask("xformers", "当前平台无兼容 wheel");
    }
  }
} else {
  try {
    runPython(venvPython, ["-m", "pip", "--isolated", "uninstall", "-y", "xformers"], { cwd: projectRoot, env: installEnvironment, stdio: "ignore" });
  } catch {}
  skipTask("xformers", "已通过参数关闭");
}

// Optional Anima accelerators. Each unlocks a Performance choice and nothing
// more, so a failure here is never a setup failure: uninstall and carry on.
// The body lives in `anima-accelerators.mjs` so each branch is unit-testable.
const acceleratorHooks = {
  startTask,
  completeTask,
  skipTask,
  emit,
  taskCheckpoint,
  checkpointTask,
  verify: (source) => pythonCheck(venvPython, source, installEnvironment),
  uninstall: (names) => {
    try {
      runPython(venvPython, ["-m", "pip", "--isolated", "uninstall", "-y", ...names], { cwd: projectRoot, env: installEnvironment, stdio: "ignore" });
    } catch {}
  },
  // Deliberately not `uvInstallFastest`: it probes through `choosePackageSource`,
  // which calls `fail()` when no route answers — ending the whole run for a
  // package the product does not need. It also probes by the first argument, so a
  // requirement specifier like `sageattention<2` becomes a URL that cannot exist.
  // Walk the configured pip routes instead and report the last status.
  install: async (args) => {
    const routeNames = [...new Set([...uniqueSources("pip"), "official"])];
    let result = { status: 1 };
    for (let index = 0; index < routeNames.length; index += 1) {
      const route = sources[routeNames[index]];
      emit("route", { id: activeTask, label: route.label, url: route.pip, connections: packageConcurrency, mode: `包级并发 ×${packageConcurrency}` });
      result = await uvInstall(uvExecutable, venvPython, args, route.pip, installEnvironment, {
        taskId: activeTask,
        label: args[0],
        stall: index + 1 < routeNames.length,
      });
      if (result.status === 0) return result;
      if (index < routeNames.length - 1) emit("warning", { message: `${args[0]} 线路${result.stalled ? "长时间无响应" : "失败"}，正在切换：${route.label}` });
    }
    return result;
  },
};

// Which of torch's own declared requirements apply to this machine. The markers decide it — the
// same wheel line requires Triton on Linux and not on Windows — so they are evaluated by the
// interpreter that will run the environment rather than guessed at here.
function torchRequiredPackages() {
  const script = [
    "import json",
    "import importlib.metadata as metadata",
    "from packaging.requirements import Requirement",
    "names=set()",
    "for raw in metadata.requires('torch') or []:",
    " requirement=Requirement(raw)",
    " if requirement.marker is None or requirement.marker.evaluate():",
    "  names.add(requirement.name)",
    "print(json.dumps(sorted(names)))",
  ].join("\n");
  const result = runPython(venvPython, ["-I", "-c", script], {
    cwd: projectRoot,
    env: installEnvironment,
    stdio: "pipe",
    encoding: "utf8",
    timeout: 30000,
  });
  if (result.status !== 0) return [];
  try {
    const names = JSON.parse(result.stdout.trim());
    return Array.isArray(names) ? names : [];
  } catch {
    return [];
  }
}

const torchRequirements = torchRequiredPackages();

const tritonPackage = tritonPackageName();
const actualTriton = await installOptionalAccelerator({
  ...acceleratorHooks,
  id: "triton",
  requested: requestedTriton,
  packageArgs: [tritonPackage],
  smokeSource: tritonSmokeSource(torchPlan.version),
  fingerprint: digest(`${torchFingerprint}:triton:${tritonPackage}:inductor-smoke-v1`),
  uninstallNames: ["triton-windows", "triton"],
  // On Linux the PyTorch wheel requires Triton outright. Removing it because the optional
  // accelerator was not selected leaves `pip check` reporting a broken install and fails the whole
  // configuration — so it stays installed, and simply stays unused.
  protectedNames: torchRequirements,
  retainedLabel: "未选择，但 PyTorch 依赖 Triton，保留安装且不启用编译加速",
  offLabel: "未选择，Transformer 编译加速保持关闭",
  startDetail: "可选：为 Anima Transformer 编译加速提供 Inductor 后端",
  incompatibleMessage: "Triton 与当前 PyTorch 不兼容，已安全移除；Transformer 编译加速保持关闭",
  unavailableMessage: "没有兼容的 Triton wheel，Transformer 编译加速保持关闭",
});

const actualSageAttention = await installOptionalAccelerator({
  ...acceleratorHooks,
  id: "sageattention",
  // Sage is pure Triton; without a working Triton the package cannot run at all.
  requested: requestedSageAttention && actualTriton,
  packageArgs: [SAGE_ATTENTION_REQUIREMENT, "--no-deps"],
  smokeSource: sageAttentionSmokeSource(),
  fingerprint: digest(`${torchFingerprint}:sageattention:cuda-smoke-v1`),
  uninstallNames: ["sageattention"],
  offLabel: requestedSageAttention && !actualTriton ? "Triton 不可用，SageAttention 已跳过" : "未选择，注意力保持 PyTorch SDPA",
  startDetail: "可选：Anima 自注意力的 INT8 量化内核",
  incompatibleMessage: "SageAttention 未通过真实 CUDA 验证，已安全移除；注意力保持 PyTorch SDPA",
  unavailableMessage: "没有兼容的 SageAttention wheel，注意力保持 PyTorch SDPA",
});

emit("feature", { id: "triton", requested: requestedTriton, available: actualTriton, package: tritonPackage });
emit("feature", { id: "sageattention", requested: requestedSageAttention, available: actualSageAttention });

if (!skipNode) {
  startTask("node");
  const packageLockPath = path.join(projectRoot, "package-lock.json");
  const nodeFingerprint = digest(readFileSync(existsSync(packageLockPath) ? packageLockPath : path.join(projectRoot, "package.json"), "utf8"));
  if (taskCheckpoint("node", nodeFingerprint) && existsSync(path.join(projectRoot, "node_modules", "vite", "package.json"))) {
    emit("route", { id: "node", label: "本地环境", connections: 0, mode: "断点恢复 · 无需下载" });
    skipTask("node", "断点恢复 · 前端依赖已安装");
  } else {
    const nodeChoice = await choosePackageSource("node", "npm", "react", "npm 并行资源下载");
    // Created up front so the byte meter has something to watch from the first sample; npm is happy
    // to fill a directory that already exists.
    mkdirSync(path.join(projectRoot, "node_modules"), { recursive: true });
    const npm = npmInvocation();
    const npmCommand = existsSync(packageLockPath) ? "ci" : "install";
    const nodeRoutes = [...new Set([nodeChoice.name, ...nodeChoice.ranked.filter((route) => route.ok).map((route) => route.id), "official"] )];
    for (let index = 0; index < nodeRoutes.length; index += 1) {
      const route = sources[nodeRoutes[index]];
      if (index > 0) {
        emit("warning", { message: `npm 当前线路${result?.stalled ? "长时间无响应" : "失败"}，切换到 ${route.label}` });
        emit("route", { id: "node", label: route.label, url: route.npm, connections: 1, mode: "npm 并行资源下载" });
      }
      result = await runInstaller({
        command: npm.command,
        args: [...npm.args, npmCommand, "--registry", route.npm, "--replace-registry-host=always"],
        taskId: "node",
        label: "前端依赖",
        // npm reports nothing at all on a non-interactive stream until it is finished, so the tree
        // it is filling is the only place the progress can come from.
        meterDirectories: [path.join(projectRoot, "node_modules")],
        stallMs: index + 1 < nodeRoutes.length ? installStallMs : 0,
        options: { env: npmEnvironment },
      });
      if (result.status === 0) break;
    }
    if (result.status !== 0) fail("前端依赖安装失败");
    checkpointTask("node", nodeFingerprint);
    completeTask("node");
  }
} else {
  skipTask("node", "已通过参数跳过");
}

startTask("verify");
try {
  await requireBundledAnimaTokenizers(projectRoot);
} catch (error) {
  fail(error.message);
}
const verification = [
   "import torch,torchvision,fastapi,diffusers,transformers,psutil,numpy,onnxruntime,scipy,ultralytics;from diffusers import AutoencoderKLQwenImage,AutoencoderKLWan,CosmosTransformer3DModel;from transformers import Qwen3Config,Qwen3Model,Qwen3VLTextModel,T5TokenizerFast",
  "from torchvision.ops import nms",
  "assert nms(torch.tensor([[0.,0.,1.,1.],[0.,0.,1.,1.]]),torch.tensor([0.9,0.5]),0.5).numel()==1",
  "print('Python environment ready')",
  "print('PyTorch:', torch.__version__)",
  "print('torchvision:', torchvision.__version__)",
  "print('CUDA available:', torch.cuda.is_available())",
  "print('CUDA runtime:', torch.version.cuda)",
  "print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')",
].join(";");
let verified = runPython(venvPython, ["-I", "-c", verification], { cwd: projectRoot, env: installEnvironment });
if (verified.status !== 0) fail("环境导入验证失败");

const cudaVerification = [
  "import torch",
  `expected_version=${JSON.stringify(torchPlan.version)}`,
  `expected=${JSON.stringify(expectedCuda)}`,
  "assert torch.__version__ == expected_version, f'Expected PyTorch {expected_version}, got {torch.__version__}'",
  "actual=str(torch.version.cuda)",
  "assert actual == expected, f'Expected CUDA runtime {expected}, got {actual}'",
  "if expected != 'None':",
  " assert torch.cuda.is_available(), 'CUDA build installed but no usable CUDA device was found'",
  " assert (torch.ones(1,device='cuda') + 1).item() == 2, 'CUDA operation failed'",
].join("\n");
verified = runPython(venvPython, ["-I", "-c", cudaVerification], { cwd: projectRoot, env: installEnvironment });
if (verified.status !== 0) fail("PyTorch/CUDA 实际运算验证失败", { repairable: true, failureKind: "cuda-verification" });
verified = runPython(venvPython, ["-m", "pip", "--isolated", "check"], { cwd: projectRoot, env: installEnvironment });
if (verified.status !== 0) fail("Python 依赖一致性检查失败");

const freeze = runPython(venvPython, ["-m", "pip", "--isolated", "freeze"], { cwd: projectRoot, env: installEnvironment, stdio: "pipe", encoding: "utf8" });
if (freeze.status === 0) {
  const stateCache = path.resolve(projectRoot, process.env.XIRAI_CACHE_DIR || ".cache");
  mkdirSync(stateCache, { recursive: true });
  writeFileSync(path.join(stateCache, "environment-lock.txt"), freeze.stdout, "utf8");
}

const vite = viteInvocation(projectRoot, ["build"]);
result = tryRun(vite.command, vite.args);
if (result.status !== 0) fail("环境已安装，但项目构建失败");
completeTask("verify", "环境验证通过 · 项目构建通过");
const actualXformers = pythonCheck(venvPython, xformersCudaSmoke(torchPlan.version), installEnvironment);
writeSetupMarker(projectRoot, {
  complete: true,
  product: "XiriaCanvas AI",
  completed_at: new Date().toISOString(),
  selection: {
    type: "selection",
    source: requestedSource === "auto" ? "逐项自动选路" : source.label,
    driverCuda: nvidiaHardware.cuda?.text || null,
    computeCapability: nvidiaHardware.computeCapability || null,
    cudaRuntime: expectedCuda,
    gpu: nvidiaHardware.name || null,
    torch: torchPlan.version,
    variant: torchPlan.variant,
    python: venvPython.command,
    xformers: actualXformers,
    rtxVsr: requestedRtxVsr,
    rtxVsrAvailable,
    triton: requestedTriton,
    tritonAvailable: actualTriton,
    sageAttention: requestedSageAttention,
    sageAttentionAvailable: actualSageAttention,
    resumed: Boolean(cachedSelection),
  },
  features: {
    rtxVsr: { requested: requestedRtxVsr, available: rtxVsrAvailable, version: rtxVsrAvailable ? RTX_VSR_VERSION : null, reason: rtxVsrReason },
    triton: { requested: requestedTriton, available: actualTriton, package: tritonPackage },
    sageAttention: { requested: requestedSageAttention, available: actualSageAttention },
  },
  message: updateRepair ? "离线更新后的环境自动修复与验证已完成" : "环境验证与项目构建均已通过",
});
emit("complete", { message: "配置完成，环境验证与项目构建均已通过" });
console.log("\n配置完成：环境验证与项目构建均已通过。运行 npm run dev 启动 XiriaCanvas AI。");
