import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { existsSync, lstatSync } from "node:fs";
import { cp, link, mkdir, mkdtemp, readdir, readFile, rm } from "node:fs/promises";
import { createServer } from "node:net";
import os from "node:os";
import path from "node:path";
import { isolatedPythonEnv, verifyProjectVenv, withoutUpdateSecrets } from "./python.mjs";
import { getSetupMarkerPath, readSetupMarker, writeSetupMarker } from "./setup-state.mjs";
import { createEnvironmentBackupOwnership, removeEnvironmentBackup, removeEnvironmentOwnershipMarker, renameWithRetry, restoreEnvironmentBackup, writeEnvironmentBackupOwnership } from "./offline-update-temp.mjs";
import { assertModelPathsUsable } from "./model-paths.mjs";
import { createHttpFetch } from "./node-tools.mjs";

const REQUIRED_MODEL_MANIFESTS = [
  "models/model-paths.json",
  "models/recommended-models.json",
  "models/yolo-models.json",
  "models/background-removal-models.json",
];
const INFERENCE_PROTOCOL = 39;

// The updater that applies an update is the *previous* release's code, loaded when its server
// started. Validating the new tree in that process judged it by the old release's constants: every
// protocol bump failed the old `INFERENCE_PROTOCOL` assertion with a bare AssertionError and rolled
// the update back (34→35, 35→36, 36→39). Each tree therefore ships this entry point, and the updater
// runs the copy inside the tree it just installed, so a release is validated by its own rules.
export const UPDATE_VALIDATION_CLI = "scripts/update-validate-cli.mjs";
export const UPDATE_VALIDATION_EVENT_PREFIX = "@@XIRAI_UPDATE_VALIDATION@@";
export const UPDATE_VALIDATION_CONTRACT = 1;
const UPDATE_VALIDATION_TIMEOUT_MS = 20 * 60 * 1000;
// Windows keeps a deleted file's name busy for a moment after its last handle closes (antivirus and
// the indexer reopen fresh files too), so a recursive delete of `.venv` right after its Python
// processes exit fails with EPERM/EBUSY. `rm` retries those codes itself when asked to.
const REMOVE_RETRIES = { recursive: true, force: true, maxRetries: 10, retryDelay: 200 };

// The validation backend is reachable only on loopback, and the shutdown call carries a one-time
// token in a header. Global fetch honours NODE_USE_ENV_PROXY, so both would be handed to whatever
// proxy the user has configured; this adapter has its own agent and reads no proxy settings.
const localFetch = createHttpFetch();

export class UpdateValidationError extends Error {
  constructor(kind, message, options = {}) {
    super(message, options);
    this.name = "UpdateValidationError";
    this.kind = kind;
  }
}

function validationFailure(kind, label, error) {
  if (error instanceof UpdateValidationError) return error;
  return new UpdateValidationError(kind, `${label}失败：${error.message}`, { cause: error });
}

function terminateProcessTree(child) {
  if (!child?.pid) {
    child?.kill?.();
    return;
  }
  if (process.platform === "win32") {
    spawn("taskkill.exe", ["/PID", String(child.pid), "/T", "/F"], { windowsHide: true, stdio: "ignore" }).unref();
    return;
  }
  try { process.kill(-child.pid, "SIGTERM"); } catch { child.kill(); }
  setTimeout(() => { try { process.kill(-child.pid, "SIGKILL"); } catch {} }, 5000).unref();
}

export function runValidationCommand(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    let stdout = "";
    let stderr = "";
    let settled = false;
    let timedOut = false;
    const child = (options.spawnProcess || spawn)(command, args, {
      cwd: options.cwd,
      env: options.env || process.env,
      windowsHide: true,
      detached: process.platform !== "win32",
      stdio: ["ignore", "pipe", "pipe"],
    });
    const finish = (callback, value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      callback(value);
    };
    const timer = setTimeout(() => {
      timedOut = true;
      terminateProcessTree(child);
    }, options.timeout || 120000);
    child.stdout?.on("data", (chunk) => {
      const text = chunk.toString("utf8");
      stdout = `${stdout}${text}`.slice(-24000);
      options.onStdout?.(text);
    });
    child.stderr?.on("data", (chunk) => {
      const text = chunk.toString("utf8");
      stderr = `${stderr}${text}`.slice(-24000);
      options.onStderr?.(text);
    });
    child.once("error", (error) => finish(reject, error));
    child.once("close", (code) => {
      if (settled) return;
      if (timedOut) finish(reject, new Error(`${options.label || path.basename(command)}超时，进程树已终止`));
      else if (code === 0) finish(resolve, { stdout, stderr });
      else finish(reject, new Error(`${options.label || path.basename(command)}退出，错误码 ${code ?? "未知"}${stderr.trim() ? `：${stderr.trim()}` : ""}`));
    });
  });
}

async function availableLoopbackPort() {
  const server = createServer();
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const port = server.address().port;
  await new Promise((resolve) => server.close(resolve));
  return port;
}

async function probeInferenceService(projectRoot, python, environment) {
  const port = await availableLoopbackPort();
  const shutdownToken = randomUUID();
  const workspaceId = `update-validation-${randomUUID()}`;
  let output = "";
  let closed = false;
  let exitCode;
  const child = spawn(python.command, [...python.args, path.join(projectRoot, "backend", "inference_server.py")], {
    cwd: projectRoot,
    env: {
      ...environment,
      INFERENCE_HOST: "127.0.0.1",
      INFERENCE_PORT: String(port),
      INFERENCE_SHUTDOWN_TOKEN: shutdownToken,
      INFERENCE_WORKSPACE_ID: workspaceId,
      PYTHONUNBUFFERED: "1",
    },
    windowsHide: true,
    detached: process.platform !== "win32",
    stdio: ["ignore", "pipe", "pipe"],
  });
  const exited = new Promise((resolve) => {
    child.once("error", (error) => {
      output = `${output}\n${error.message}`.slice(-24000);
      closed = true;
      exitCode = "spawn-error";
      resolve();
    });
    child.once("close", (code, signal) => {
      closed = true;
      exitCode = signal || code;
      resolve();
    });
  });
  const capture = (chunk) => { output = `${output}${chunk.toString("utf8")}`.slice(-24000); };
  child.stdout.on("data", capture);
  child.stderr.on("data", capture);
  const deadline = Date.now() + 120000;
  try {
    while (Date.now() < deadline && !closed) {
      await new Promise((resolve) => setTimeout(resolve, 500));
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 2000);
      try {
        const response = await localFetch(`http://127.0.0.1:${port}/api/inference/health`, { signal: controller.signal });
        if (response.ok) {
          const health = await response.json();
          if (health.status !== "ready" || health.protocol !== INFERENCE_PROTOCOL || health.workspace_id !== workspaceId) {
            throw new Error("后端健康响应不属于本次离线更新验证");
          }
          await localFetch(`http://127.0.0.1:${port}/api/inference/shutdown`, {
            method: "POST",
            headers: { "X-Shutdown-Token": shutdownToken },
          });
          // Grace timers are unref'd: once the backend has exited they must not hold the validator
          // process open for the rest of their 30 seconds.
          await Promise.race([exited, new Promise((resolve) => setTimeout(resolve, 30000).unref())]);
          if (!closed) terminateProcessTree(child);
          return health;
        }
      } catch {
        // The service is still importing or binding its validation port.
      } finally {
        clearTimeout(timer);
      }
    }
    throw new Error(`后端未能启动（${exitCode ?? "超时"}）${output.trim() ? `：${output.trim()}` : ""}`);
  } finally {
    if (!closed) {
      terminateProcessTree(child);
      await Promise.race([exited, new Promise((resolve) => setTimeout(resolve, 5000).unref())]);
    }
  }
}

async function validateModelManifests(projectRoot) {
  for (const relativePath of REQUIRED_MODEL_MANIFESTS) {
    try {
      const value = JSON.parse(await readFile(path.join(projectRoot, relativePath), "utf8"));
      if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("根节点必须是对象");
      if (relativePath.endsWith("model-paths.json")) {
        assertModelPathsUsable(value, projectRoot);
      } else if (relativePath.endsWith("recommended-models.json")) {
        if (!Number.isInteger(value.schema) || !Array.isArray(value.artifacts) || !Array.isArray(value.staticFamilies) || !Array.isArray(value.civitaiFamilies)) {
          throw new Error("缺少推荐模型目录数组");
        }
        const ids = value.artifacts.map((item) => item?.id);
        if (ids.some((id) => typeof id !== "string" || !id) || new Set(ids).size !== ids.length) throw new Error("推荐模型 ID 无效或重复");
      } else if (!Number.isInteger(value.schema) || !Array.isArray(value.models)) {
        throw new Error("缺少 schema 或 models 数组");
      }
    } catch (error) {
      throw validationFailure("program", `${relativePath} 校验`, error);
    }
  }
}

function warningDetail(error) {
  const text = String(error?.message || error || "").trim();
  return text.length > 600 ? `…${text.slice(-600)}` : text;
}

/**
 * Validate an installed tree. Blocking checks are the ones a program update can break: the backend
 * sources, the shipped manifests, the backend loading and serving, and the frontend build — plus
 * the bare prerequisites without which nothing runs (a project `.venv`, the setup marker and the
 * backend's imports). Everything else describes the machine rather than the release: runtime
 * configs that are only fetched once a checkpoint family is installed, `pip check` advisories, a
 * torch build that drifted from the marker, CUDA or RTX VSR availability. An update cannot change
 * any of those while `requirements.txt` is unchanged, the setup configurator itself finishes with
 * only a warning for the runtime configs, and blocking on them rolled back every update on any
 * machine that lacked one (both known installations did). They are returned as `warnings`.
 */
export async function validateUpdatedProject({
  projectRoot,
  environment = process.env,
  runCommand = runValidationCommand,
  verifyVenv = verifyProjectVenv,
  probeBackend = probeInferenceService,
  report,
} = {}) {
  const warnings = [];
  const advise = async (check, message, action) => {
    try {
      await action();
    } catch (error) {
      warnings.push({ check, message: `${message}：${warningDetail(error)}` });
    }
  };
  report?.({ phase: "environment", progress: 2, message: "正在检查项目 Python 环境" });
  const python = verifyVenv(projectRoot);
  if (!python.ok) throw new UpdateValidationError("environment", `更新后 Python 环境验证失败：${python.error}`);
  const pythonEnvironment = isolatedPythonEnv(environment);
  const executePython = async (script, label, kind = "environment", timeout = 120000) => {
    try {
      await runCommand(python.python.command, ["-I", "-c", script], {
        cwd: projectRoot,
        env: pythonEnvironment,
        label,
        timeout,
      });
    } catch (error) {
      throw validationFailure(kind, label, error);
    }
  };

  report?.({ phase: "environment", progress: 10, message: "正在检查后端源码与运行清单" });
  await executePython([
    "import ast,pathlib",
    "files=list(pathlib.Path('backend').rglob('*.py'))",
    "assert files, 'backend Python files are missing'",
    "[ast.parse(file.read_text(encoding='utf-8'), filename=str(file)) for file in files]",
  ].join(";"), "Python 源码", "program");
  await validateModelManifests(projectRoot);

  report?.({ phase: "environment", progress: 24, message: "正在验证 Python 与后端依赖" });
  await executePython(
    "import accelerate,diffusers,fastapi,huggingface_hub,numpy,onnxruntime,peft,PIL,psutil,pydantic,safetensors,scipy,spandrel,spandrel_extra_arches,torch,transformers,ultralytics,uvicorn;from diffusers import AutoencoderKLQwenImage,CosmosTransformer3DModel;from diffusers.loaders.single_file_utils import convert_cosmos_transformer_checkpoint_to_diffusers,convert_wan_vae_to_diffusers;from transformers import Qwen3Config,Qwen3Model,T5TokenizerFast",
    "Python 后端依赖",
  );
  await advise("python-dependency-consistency", "Python 依赖一致性检查有提示（不影响本次更新）", () => runCommand(python.python.command, ["-m", "pip", "--isolated", "check"], {
    cwd: projectRoot,
    env: pythonEnvironment,
    label: "Python 依赖一致性",
    timeout: 120000,
  }));

  report?.({ phase: "environment", progress: 38, message: "正在检查本地模型运行配置" });
  await advise("runtime-configs", "已安装底模的本地运行配置不完整（不影响本次更新；联网后重新运行环境配置器即可补齐，补齐前对应底模无法生成）", () => runCommand(python.python.command, [path.join(projectRoot, "backend", "pipeline_configs.py"), "--check", "--required", "--installed", projectRoot], {
    cwd: projectRoot,
    env: pythonEnvironment,
    label: "模型运行配置",
    timeout: 120000,
  }));

  const marker = readSetupMarker(projectRoot, environment);
  if (!marker) throw new UpdateValidationError("environment", "更新后配置完成标记缺失或无效");
  const selection = marker?.selection || {};
  const torchScript = [
    "import torch",
    ...(selection.torch ? [`assert torch.__version__==${JSON.stringify(selection.torch)}, f'Expected ${selection.torch}, got {torch.__version__}'`] : []),
    ...(selection.cudaRuntime ? [`assert str(torch.version.cuda)==${JSON.stringify(selection.cudaRuntime)}, f'Expected CUDA ${selection.cudaRuntime}, got {torch.version.cuda}'`] : []),
    ...(selection.variant && selection.variant !== "cpu" ? [
      "assert torch.cuda.is_available(), 'CUDA is unavailable'",
      "assert (torch.ones(1,device='cuda')+1).item()==2, 'CUDA operation failed'",
      "torch.cuda.synchronize()",
    ] : []),
  ].join(";");
  report?.({ phase: "environment", progress: 50, message: "正在执行 PyTorch / CUDA 实际运算" });
  await advise("torch-cuda", "PyTorch / CUDA 检查未通过（不影响本次更新；生成时可能无法使用 GPU，可重新运行环境配置器修复）", () => executePython(torchScript, "PyTorch / CUDA", "environment"));

  report?.({ phase: "environment", progress: 62, message: "正在加载后端服务模块" });
  await executePython(
    `import sys;sys.path.insert(0,'backend');import inference_server;assert inference_server.INFERENCE_PROTOCOL==${INFERENCE_PROTOCOL};inference_server.anima_tokenizer_sources()`,
    "后端服务加载",
    "program",
    180000,
  );

  if (selection.rtxVsr === true && (selection.rtxVsrAvailable === true || selection.rtxVsrAvailable == null)) {
    report?.({ phase: "environment", progress: 66, message: "正在验证已启用的 RTX VSR 运行时" });
    await advise("rtx-vsr", "已启用的 RTX VSR 运行时当前不可用（不影响本次更新）", () => executePython([
      "import sys,time",
      "sys.path.insert(0,'backend')",
      "import rtx_vsr",
      "status=rtx_vsr.status(refresh=True)",
      "deadline=time.monotonic()+180",
      "while status.get('probing') and time.monotonic()<deadline:",
      " time.sleep(0.1)",
      " status=rtx_vsr.status()",
      "assert status.get('available'), status.get('reason') or 'RTX VSR adapter is unavailable'",
    ].join("\n"), "RTX VSR 生产适配器", "environment", 200000));
  }

  report?.({ phase: "environment", progress: 70, message: "正在启动并探测后端服务" });
  try {
    await probeBackend(projectRoot, python.python, pythonEnvironment);
  } catch (error) {
    throw validationFailure("program", "后端服务启动", error);
  }

  const viteEntry = path.join(projectRoot, "node_modules", "vite", "bin", "vite.js");
  if (!existsSync(viteEntry)) throw new UpdateValidationError("program", "前端环境验证失败：Vite 尚未安装");
  const validationOutput = await mkdtemp(path.join(os.tmpdir(), "xirai-update-build-"));
  report?.({ phase: "environment", progress: 78, message: "正在执行前端生产构建" });
  try {
    await runCommand(process.execPath, [viteEntry, "build", "--outDir", validationOutput, "--emptyOutDir"], {
      cwd: projectRoot,
      env: environment,
      label: "前端生产构建",
      timeout: 180000,
    });
  } catch (error) {
    throw validationFailure("program", "前端生产构建", error);
  } finally {
    await rm(validationOutput, { recursive: true, force: true });
  }
  if (!readSetupMarker(projectRoot, environment)) {
    throw new UpdateValidationError("environment", "更新验证期间配置完成标记失效");
  }
  report?.({
    phase: "environment",
    progress: 100,
    message: warnings.length ? `程序验证通过；另有 ${warnings.length} 项环境提示` : "更新后的环境与后端验证通过",
  });
  return { verified: true, warnings };
}

function validationWarnings(value) {
  if (!Array.isArray(value)) return [];
  return value
    .filter((item) => item && typeof item.check === "string" && typeof item.message === "string")
    .slice(0, 20)
    .map((item) => ({ check: item.check.slice(0, 64), message: item.message.slice(0, 2000) }));
}

/**
 * Validate the tree at `projectRoot` with the validator that tree ships. A tree that predates the
 * entry point (an offline downgrade to an older release) falls back to this process's validator,
 * which is what every update did before.
 */
export async function validateProjectTree({
  projectRoot,
  environment = process.env,
  runCommand = runValidationCommand,
  validateInProcess = validateUpdatedProject,
  report,
} = {}) {
  const entry = path.join(projectRoot, ...UPDATE_VALIDATION_CLI.split("/"));
  if (!existsSync(entry)) return validateInProcess({ projectRoot, environment, report });
  let pending = "";
  let outcome;
  const consume = (text) => {
    pending += text;
    const lines = pending.split(/\r?\n/);
    pending = lines.pop() || "";
    for (const line of lines) {
      if (!line.startsWith(UPDATE_VALIDATION_EVENT_PREFIX)) continue;
      let event;
      try {
        event = JSON.parse(line.slice(UPDATE_VALIDATION_EVENT_PREFIX.length));
      } catch {
        continue;
      }
      if (event?.contract !== UPDATE_VALIDATION_CONTRACT) continue;
      if (event.type === "progress") {
        report?.({ phase: "environment", progress: Number(event.progress) || 0, message: String(event.message || "") });
      } else if (event.type === "result" || event.type === "error") {
        outcome = event;
      }
    }
  };
  try {
    // The new tree's validator is the code this update installs and the next start runs anyway.
    // It still gets the update secrets stripped, as every other child of the update API does.
    await runCommand(process.execPath, [entry, projectRoot], {
      cwd: projectRoot,
      env: withoutUpdateSecrets(environment),
      label: "更新后验证",
      timeout: UPDATE_VALIDATION_TIMEOUT_MS,
      onStdout: consume,
    });
  } catch (error) {
    if (outcome?.type === "error") {
      throw new UpdateValidationError(outcome.kind === "environment" ? "environment" : "program", String(outcome.message || "更新后验证失败"));
    }
    throw validationFailure("program", "更新后验证", error);
  }
  if (outcome?.type !== "result" || outcome.verified !== true) {
    throw new UpdateValidationError("program", "更新后验证未返回有效结果");
  }
  return { verified: true, warnings: validationWarnings(outcome.warnings) };
}

export function repairArguments(marker, fallback = false) {
  const selection = marker?.selection || {};
  const args = ["--events", "--update-repair"];
  if (!fallback && /^(cpu|cu\d+)$/.test(selection.variant || "") && /^\d+\.\d+\.\d+(?:\+[a-z0-9.]+)?$/i.test(selection.torch || "")) {
    args.push(`--torch=${selection.variant}`, `--torch-version=${selection.torch}`);
  } else {
    args.push("--torch=auto", "--refresh-selection", "--without-xformers");
  }
  if (!fallback && selection.xformers === false) args.push("--without-xformers");
  args.push(selection.rtxVsr === true ? "--with-rtx-vsr" : "--without-rtx-vsr");
  return args;
}

async function runRepairAttempt({ projectRoot, environment, marker, fallback, runCommand, report }) {
  let pending = "";
  const consume = (text) => {
    pending += text;
    const lines = pending.split(/\r?\n/);
    pending = lines.pop() || "";
    for (const line of lines) {
      if (!line.startsWith("@@XIRAI_SETUP@@")) continue;
      try {
        const event = JSON.parse(line.slice("@@XIRAI_SETUP@@".length));
        if (event.type === "task" && event.status === "running") {
          report?.({ phase: "repair", progress: 20, message: `正在自动修复：${event.label}${event.detail ? ` · ${event.detail}` : ""}` });
        } else if (event.type === "download") {
          const progress = event.totalBytes ? Math.round(event.currentBytes / event.totalBytes * 100) : 0;
          report?.({ phase: "repair", progress: 20 + progress * 0.6, message: `正在自动修复：${event.name || "下载依赖"}` });
        } else if (event.type === "warning") {
          report?.({ phase: "repair", progress: 45, message: event.message });
        } else if (event.type === "error") {
          consume.failureKind = event.failureKind || "setup";
        }
      } catch {
        // Malformed progress output does not invalidate the repair process.
      }
    }
  };
  try {
    await runCommand(process.execPath, [path.join(projectRoot, "scripts", "setup.mjs"), ...repairArguments(marker, fallback), "--skip-node"], {
      cwd: projectRoot,
      env: { ...environment, XIRAI_SETUP_EVENTS: "1", FORCE_COLOR: "0" },
      label: fallback ? "环境兼容回退修复" : "环境自动修复",
      timeout: 45 * 60 * 1000,
      onStdout: consume,
    });
  } catch (error) {
    error.failureKind = consume.failureKind || error.failureKind;
    throw error;
  }
}

const UNLINKABLE_CODES = new Set(["EXDEV", "EPERM", "EACCES", "ENOTSUP", "EOPNOTSUPP", "ENOSYS", "EINVAL"]);

/**
 * Rebuild `source` at `destination` with every regular file hard-linked rather than copied.
 *
 * The repair used to byte-copy the whole `.venv` (6.9 GB on the cloud instance) before touching it,
 * and a failed update did that twice. A hard link costs only a directory entry. It is safe for the
 * backup because nothing the repair runs writes into an existing file: pip unlinks a destination
 * before writing it (so a loaded shared object keeps its old pages), uv links or copies fresh
 * files, and Python replaces `.pyc` files by rename. Every write therefore lands on a new inode in
 * `destination` and the original stays byte-identical in `source`. A volume that cannot link (FAT,
 * some network shares) falls back to copying, which is what every repair did before.
 */
export async function mirrorEnvironmentTree(source, destination, { linkFile = link } = {}) {
  let linked = 0;
  let copied = 0;
  let linking = true;
  await mkdir(destination);
  const pending = [[source, destination]];
  while (pending.length) {
    const [from, to] = pending.pop();
    for (const entry of await readdir(from, { withFileTypes: true })) {
      const sourcePath = path.join(from, entry.name);
      const targetPath = path.join(to, entry.name);
      if (entry.isDirectory()) {
        await mkdir(targetPath);
        pending.push([sourcePath, targetPath]);
        continue;
      }
      if (entry.isFile() && linking) {
        try {
          await linkFile(sourcePath, targetPath);
          linked += 1;
          continue;
        } catch (error) {
          // EMLINK is one file at its link limit; anything else means this volume cannot link.
          if (error.code !== "EMLINK" && !UNLINKABLE_CODES.has(error.code)) throw error;
          if (error.code !== "EMLINK") linking = false;
        }
      }
      // Symbolic links (POSIX `bin/python`, `lib64`) are recreated verbatim, files copied.
      await cp(sourcePath, targetPath, { dereference: false, errorOnExist: true, force: false, preserveTimestamps: true, verbatimSymlinks: true });
      if (entry.isFile()) copied += 1;
    }
  }
  return { linked, copied };
}

export async function repairUpdatedEnvironment({
  projectRoot,
  environment = process.env,
  runCommand = runValidationCommand,
  report,
  onBackup,
} = {}) {
  const marker = readSetupMarker(projectRoot, environment);
  const previousMarker = marker ? structuredClone(marker) : null;
  const markerPath = getSetupMarkerPath(projectRoot, environment);
  const venvRoot = path.join(projectRoot, ".venv");
  const backupOwnership = createEnvironmentBackupOwnership(projectRoot);
  const backupRoot = backupOwnership.path;
  const originalExisted = existsSync(venvRoot);
  let originalMoved = false;
  let finished = false;
  const rollback = async () => {
    if (finished) return;
    if (originalMoved) {
      await restoreEnvironmentBackup({ projectRoot, record: backupOwnership, venvRoot });
    } else if (!originalExisted) {
      await rm(venvRoot, REMOVE_RETRIES);
    } else {
      await removeEnvironmentOwnershipMarker({ projectRoot, record: backupOwnership, directory: venvRoot });
    }
    if (previousMarker) writeSetupMarker(projectRoot, previousMarker, environment);
    else await rm(markerPath, { force: true });
    finished = true;
  };
  const commit = async () => {
    if (finished) return;
    if (originalMoved) await removeEnvironmentBackup({ projectRoot, record: backupOwnership });
    finished = true;
  };
  report?.({ phase: "repair", progress: 0, message: "环境检查未通过，正在自动修复 Python 与后端依赖" });
  if (originalExisted && lstatSync(venvRoot).isSymbolicLink()) {
    throw new UpdateValidationError("environment", ".venv 是外部链接，不能自动替换");
  }
  try {
    if (existsSync(venvRoot)) {
      await onBackup?.(backupOwnership);
      await writeEnvironmentBackupOwnership(projectRoot, venvRoot, backupOwnership);
      await renameWithRetry(venvRoot, backupRoot);
      originalMoved = true;
      report?.({ phase: "repair", progress: 5, message: "正在为原 Python 环境建立硬链接快照（不复制文件内容）" });
      await mirrorEnvironmentTree(backupRoot, venvRoot);
      await rm(path.join(venvRoot, ".xirai-offline-update-owner.json"), { force: true });
    }
    await runRepairAttempt({ projectRoot, environment, marker, fallback: false, runCommand, report });
  } catch (firstError) {
    try {
      if (!["cuda-verification", "dependency-changed-torch"].includes(firstError.failureKind)
        || !marker?.selection?.variant || marker.selection.variant === "cpu") throw firstError;
      await rm(venvRoot, REMOVE_RETRIES);
      report?.({ phase: "repair", progress: 10, message: "PyTorch / CUDA 修复失败，正在尝试兼容回退方案" });
      await runRepairAttempt({ projectRoot, environment, marker, fallback: true, runCommand, report });
    } catch (fallbackError) {
      try {
        await rollback();
      } catch (rollbackError) {
        const failure = new UpdateValidationError("environment", `环境自动修复失败且旧 Python 环境回滚不完整：${rollbackError.message}`, { cause: fallbackError });
        failure.rollbackIncomplete = true;
        throw failure;
      }
      throw new UpdateValidationError("environment", `环境自动修复失败：${fallbackError.message}`, { cause: fallbackError });
    }
  }
  report?.({ phase: "repair", progress: 100, message: "环境自动修复完成，正在重新验证" });
  return { repaired: true, commit, rollback, backupRoot: originalMoved ? backupRoot : null, backupOwnership: originalMoved ? backupOwnership : null };
}

export async function ensureUpdatedProjectReady({ forceRepair = false, validate, repair, report } = {}) {
  let repaired = false;
  let transaction;
  const rollbackOrThrow = async (error) => {
    if (!transaction?.rollback) throw error;
    try {
      await transaction.rollback();
    } catch (rollbackError) {
      const failure = new UpdateValidationError("environment", `${error.message}；Python 环境回滚不完整：${rollbackError.message}`, { cause: error });
      failure.rollbackIncomplete = true;
      throw failure;
    }
    throw error;
  };
  if (forceRepair) {
    transaction = await repair();
    repaired = true;
  }
  let validation;
  try {
    validation = await validate();
  } catch (error) {
    if (error?.kind !== "environment" || repaired) {
      await rollbackOrThrow(error);
    }
    report?.({ phase: "repair", progress: 0, message: "检测到环境异常，正在自动修复" });
    transaction = await repair();
    repaired = true;
    try {
      validation = await validate();
    } catch (validationError) {
      await rollbackOrThrow(validationError);
    }
  }
  return {
    verified: true,
    repaired,
    ...(validation?.warnings?.length ? { warnings: validation.warnings } : {}),
    ...(transaction?.commit ? { commit: transaction.commit } : {}),
    ...(transaction?.rollback ? { rollback: transaction.rollback } : {}),
    ...(transaction?.backupOwnership ? { environmentBackup: transaction.backupOwnership } : {}),
  };
}
