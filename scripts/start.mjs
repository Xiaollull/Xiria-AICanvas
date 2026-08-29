import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync } from "node:fs";
import { networkInterfaces } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { writeDiagnosticLog } from "./diagnostics.mjs";
import { recoverInterruptedUpdate } from "./archive-update.mjs";
import { getVenvPythonPath, isolatedPythonEnv, loadLocalEnv } from "./python.mjs";
import { readSetupMarker } from "./setup-state.mjs";
import { viteInvocation } from "./node-tools.mjs";
import { acquireOfflineUpdateLock } from "./offline-update-lock.mjs";
import { requireBundledAnimaTokenizers } from "./anima-models.mjs";

const projectRoot = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
Object.assign(process.env, loadLocalEnv(projectRoot));

const noOpen = process.argv.includes("--no-open");
const exitAfterReady = process.argv.includes("--exit-after-ready");
const webPort = Number(process.env.WEB_PORT || 7709);
const mainUrl = `http://localhost:${webPort}/`;
const browserProfile = path.resolve(projectRoot, process.env.XIRAI_BROWSER_PROFILE || "state-cache/browser-profile");
const stateDirectory = path.resolve(projectRoot, process.env.XIRAI_STATE_DIR || "state-cache");
const workspaceId = createHash("sha256")
  .update(process.platform === "win32" ? path.normalize(projectRoot).toLowerCase() : path.normalize(projectRoot))
  .digest("hex");
const inferenceProtocol = 34;

function fail(message) {
  writeDiagnosticLog(projectRoot, {
    kind: "startup-failure",
    message,
    details: { node_version: process.version, platform: process.platform, architecture: process.arch },
  });
  console.error(`\n启动失败：${message}`);
  console.error("请先双击 Setup-XirAI 配置环境，或运行：npm run setup\n");
  process.exit(1);
}

function openBrowser(url) {
  let command;
  let args;
  if (process.platform === "win32") {
    const candidates = [
      process.env["PROGRAMFILES(X86)"] && path.join(process.env["PROGRAMFILES(X86)"], "Microsoft", "Edge", "Application", "msedge.exe"),
      process.env.PROGRAMFILES && path.join(process.env.PROGRAMFILES, "Microsoft", "Edge", "Application", "msedge.exe"),
      process.env.LOCALAPPDATA && path.join(process.env.LOCALAPPDATA, "Microsoft", "Edge", "Application", "msedge.exe"),
      process.env.PROGRAMFILES && path.join(process.env.PROGRAMFILES, "Google", "Chrome", "Application", "chrome.exe"),
    ].filter(Boolean);
    command = candidates.find((candidate) => existsSync(candidate));
    if (command) {
      mkdirSync(browserProfile, { recursive: true });
      args = [`--app=${url}`, `--user-data-dir=${browserProfile}`, "--disable-extensions", "--no-first-run", "--no-default-browser-check"];
    } else {
      command = "cmd.exe";
      args = ["/d", "/s", "/c", "start", "", url];
    }
  } else if (process.platform === "darwin") {
    command = "open";
    args = [url];
  } else {
    command = "xdg-open";
    args = [url];
  }
  const browser = spawn(command, args, { detached: true, stdio: "ignore", windowsHide: true });
  browser.on("error", () => console.log(`请在浏览器打开：${url}`));
  browser.unref();
}

function lanUrls() {
  const urls = [];
  for (const interfaces of Object.values(networkInterfaces())) {
    for (const item of interfaces || []) {
      if (item.family === "IPv4" && !item.internal) urls.push(`http://${item.address}:${webPort}/`);
    }
  }
  return [...new Set(urls)];
}

async function getHealth(timeout = 1500) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(`http://127.0.0.1:${webPort}/api/inference/health`, { cache: "no-store", signal: controller.signal });
    return response.ok ? await response.json() : null;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

const [nodeMajor, nodeMinor] = process.versions.node.split(".").map(Number);
if (nodeMajor < 22 || (nodeMajor === 22 && nodeMinor < 21)) fail("需要 Node.js 22.21 或更高版本");
const existingHealth = await getHealth();
if (existingHealth?.status === "ready" && existingHealth.protocol === inferenceProtocol && existingHealth.workspace_id === workspaceId) {
  console.log(`XiriaCanvas AI 已经运行：${mainUrl}`);
  if (!noOpen) openBrowser(mainUrl);
  process.exit(0);
}
let recoveryLock;
try {
  recoveryLock = await acquireOfflineUpdateLock({ stateDirectory, projectRoot, operation: "startup-recovery" });
  const recovery = await recoverInterruptedUpdate({
    projectRoot,
    transactionPath: path.join(stateDirectory, "manual-update-transaction.json"),
    report: (message) => console.log(message),
    lock: recoveryLock,
  });
  if (recovery.rolledBack) console.log("未完成的离线更新已自动回滚，正在启动原版本。");
  const environmentRecovery = await recoverInterruptedUpdate({
    projectRoot,
    transactionPath: path.join(stateDirectory, "manual-update-environment-transaction.json"),
    report: (message) => console.log(message),
    lock: recoveryLock,
  });
  if (environmentRecovery.rolledBack) console.log("未完成的离线环境修复已自动回滚。");
} catch (error) {
  fail(`离线更新中断恢复失败：${error.message}`);
} finally {
  await recoveryLock?.release().catch(() => {});
}
if (!readSetupMarker(projectRoot)) fail("尚未完成环境配置，或配置完成标记无效");
try {
  await requireBundledAnimaTokenizers(projectRoot);
} catch (error) {
  fail(error.message);
}

const pythonPath = getVenvPythonPath(projectRoot);
const pythonCheck = spawnSync(pythonPath, ["-I", "-c", "import torch,fastapi,diffusers,transformers,ultralytics,uvicorn,psutil,numpy,scipy,spandrel,spandrel_extra_arches;from diffusers import AutoencoderKLQwenImage,CosmosTransformer3DModel;from transformers import Qwen3Config,Qwen3Model,T5TokenizerFast"], {
  cwd: projectRoot,
  env: isolatedPythonEnv(process.env),
  stdio: "ignore",
  windowsHide: true,
  timeout: 60000,
});
if (pythonCheck.status !== 0) fail("项目 Python 环境缺失、损坏或依赖未安装完整");

let vite;
try {
  vite = viteInvocation(projectRoot);
} catch (error) {
  fail(error.message);
}

console.log("配置检查通过，正在启动 XiriaCanvas AI...");
console.log(`本机访问：${mainUrl}`);
for (const url of lanUrls()) console.log(`局域网访问：${url}`);

let activeChild;
let shuttingDown = false;

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.once(signal, () => {
    shuttingDown = true;
    activeChild?.kill(signal);
  });
}

function childExit(child) {
  return new Promise((resolve, reject) => {
    child.once("error", reject);
    child.once("exit", (code, signal) => resolve({ code, signal }));
  });
}

async function runMainService(openOnReady) {
  const child = spawn(vite.command, vite.args, {
    cwd: projectRoot,
    env: { ...process.env, XIRAI_SERVICE_SUPERVISOR: "1" },
    stdio: "inherit",
    windowsHide: true,
  });
  activeChild = child;
  const exited = childExit(child);
  let ready = false;
  for (let attempt = 0; attempt < 180; attempt += 1) {
    if (child.exitCode != null || child.signalCode != null) break;
    await new Promise((resolve) => setTimeout(resolve, 500));
    const health = await getHealth();
    if (health?.status === "ready" && health.protocol === inferenceProtocol && health.workspace_id === workspaceId) {
      ready = true;
      console.log(`\n启动完成：${mainUrl}`);
      if (openOnReady && !noOpen) openBrowser(mainUrl);
      break;
    }
  }
  if (!ready) {
    child.kill();
    throw new Error("主页面或推理后端未能在 90 秒内启动");
  }
  if (exitAfterReady) child.kill("SIGTERM");
  return exited;
}

async function runConfigurator() {
  console.log("\n正在切换到环境配置器，当前创作状态已保存在浏览器中...");
  const child = spawn(process.execPath, [path.join(projectRoot, "scripts", "setup-gui.mjs"), "--no-open", "--return-to-app", "--supervised"], {
    cwd: projectRoot,
    env: process.env,
    stdio: "inherit",
    windowsHide: true,
  });
  activeChild = child;
  return childExit(child);
}

let service = "main";
let openOnReady = true;
while (!shuttingDown) {
  try {
    const result = service === "main" ? await runMainService(openOnReady) : await runConfigurator();
    activeChild = null;
    if (shuttingDown) break;
    if (service === "main" && result.code === 75) {
      service = "config";
      openOnReady = false;
      continue;
    }
    if (service === "main" && result.code === 77) {
      service = "main";
      openOnReady = false;
      continue;
    }
    if (service === "config" && result.code === 76) {
      service = "main";
      openOnReady = false;
      continue;
    }
    process.exit(result.signal ? 1 : (result.code ?? 0));
  } catch (error) {
    fail(error.message);
  }
}
