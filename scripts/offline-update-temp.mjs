import { randomUUID } from "node:crypto";
import { lstat, mkdtemp, readFile, realpath, rename, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

export const OFFLINE_UPDATE_OWNER_FILE = ".xirai-offline-update-owner.json";
const PRODUCT = "XiriaCanvas AI";

function pathsEqual(first, second) {
  const left = path.resolve(first);
  const right = path.resolve(second);
  return process.platform === "win32" ? left.toLowerCase() === right.toLowerCase() : left === right;
}

function isWithin(root, candidate, allowRoot = false) {
  const relative = path.relative(path.resolve(root), path.resolve(candidate));
  if (!relative) return allowRoot;
  return !path.isAbsolute(relative) && !relative.split(path.sep).some((segment) => segment === "..");
}

function allowedParents(projectRoot) {
  return [...new Set([os.tmpdir(), path.dirname(projectRoot)].map((item) => path.resolve(item)))];
}

function validateRecord(projectRoot, record, prefix, kind) {
  if (!record || record.schema !== 1 || typeof record.path !== "string" || typeof record.token !== "string"
    || typeof record.projectRoot !== "string" || record.product !== PRODUCT
    || !record.token || record.kind !== kind || !pathsEqual(record.projectRoot, projectRoot)) {
    throw new Error(`离线更新${kind}目录所有权信息无效`);
  }
  const resolved = path.resolve(record.path);
  if (!path.basename(resolved).startsWith(prefix)
    || !allowedParents(projectRoot).some((parent) => pathsEqual(path.dirname(resolved), parent))
    || isWithin(projectRoot, resolved, true)) {
    throw new Error(`离线更新${kind}目录路径无效：${resolved}`);
  }
  return resolved;
}

export async function createOfflineUpdateTemp({ projectRoot, prefix, kind } = {}) {
  const root = path.resolve(projectRoot || "");
  if (!projectRoot || !prefix || !kind) throw new Error("缺少离线更新临时目录参数");
  let lastError;
  for (const parent of allowedParents(root)) {
    let directory;
    try {
      directory = await mkdtemp(path.join(parent, prefix));
      const canonical = await realpath(directory);
      if (isWithin(root, canonical, true)) {
        await rm(canonical, { recursive: true, force: true });
        continue;
      }
      const record = {
        schema: 1,
        product: PRODUCT,
        projectRoot: root,
        path: canonical,
        token: randomUUID(),
        kind,
      };
      await writeFile(path.join(canonical, OFFLINE_UPDATE_OWNER_FILE), `${JSON.stringify(record)}\n`, { encoding: "utf8", flag: "wx" });
      return record;
    } catch (error) {
      if (directory) await rm(directory, { recursive: true, force: true }).catch(() => {});
      lastError = error;
    }
  }
  throw new Error(`无法在项目外创建离线更新临时目录${lastError ? `：${lastError.message}` : ""}`);
}

export async function assertOfflineUpdateTemp({ projectRoot, record, prefix, kind, requireExists = true } = {}) {
  const root = path.resolve(projectRoot || "");
  const resolved = validateRecord(root, record, prefix, kind);
  let stats;
  try {
    stats = await lstat(resolved);
  } catch (error) {
    if (!requireExists && error.code === "ENOENT") return resolved;
    throw error;
  }
  if (!stats.isDirectory() || stats.isSymbolicLink()) throw new Error(`离线更新${kind}目录不是安全目录：${resolved}`);
  const canonical = await realpath(resolved);
  if (!pathsEqual(resolved, canonical)) throw new Error(`离线更新${kind}目录包含路径重定向：${resolved}`);

  let marker;
  try {
    marker = JSON.parse(await readFile(path.join(canonical, OFFLINE_UPDATE_OWNER_FILE), "utf8"));
  } catch (error) {
    throw new Error(`无法验证离线更新${kind}目录所有权：${error.message}`);
  }
  if (marker.schema !== record.schema || marker.product !== PRODUCT || marker.token !== record.token
    || marker.kind !== kind || !pathsEqual(marker.projectRoot, root) || !pathsEqual(marker.path, canonical)) {
    throw new Error(`离线更新${kind}目录所有权不匹配：${canonical}`);
  }
  return canonical;
}

export async function removeOfflineUpdateTemp(options) {
  const target = await assertOfflineUpdateTemp({ ...options, requireExists: false });
  await rm(target, { recursive: true, force: true });
}

export function createEnvironmentBackupOwnership(projectRoot) {
  const root = path.resolve(projectRoot || "");
  if (!projectRoot) throw new Error("缺少离线更新项目根目录");
  return {
    schema: 1,
    product: PRODUCT,
    projectRoot: root,
    path: path.join(root, `.venv.update-backup-${randomUUID()}`),
    token: randomUUID(),
    kind: "environment",
  };
}

function validateEnvironmentRecord(projectRoot, record) {
  const root = path.resolve(projectRoot || "");
  if (!record || record.schema !== 1 || record.product !== PRODUCT || record.kind !== "environment"
    || typeof record.projectRoot !== "string" || typeof record.path !== "string" || typeof record.token !== "string"
    || !record.token || !pathsEqual(record.projectRoot, root)) {
    throw new Error("离线更新 Python 环境备份所有权信息无效");
  }
  const target = path.resolve(record.path);
  if (!pathsEqual(path.dirname(target), root) || !path.basename(target).startsWith(".venv.update-backup-")) {
    throw new Error("离线更新 Python 环境备份路径无效");
  }
  return target;
}

export async function writeEnvironmentBackupOwnership(projectRoot, directory, record) {
  validateEnvironmentRecord(projectRoot, record);
  await writeFile(path.join(directory, OFFLINE_UPDATE_OWNER_FILE), `${JSON.stringify(record)}\n`, { encoding: "utf8", flag: "wx" });
}

export async function assertEnvironmentBackup({ projectRoot, record, requireExists = true } = {}) {
  const target = validateEnvironmentRecord(projectRoot, record);
  let stats;
  try {
    stats = await lstat(target);
  } catch (error) {
    if (!requireExists && error.code === "ENOENT") return null;
    throw error;
  }
  if (!stats.isDirectory() || stats.isSymbolicLink() || !pathsEqual(await realpath(target), target)) {
    throw new Error("离线更新 Python 环境备份不是安全目录");
  }
  let marker;
  try {
    marker = JSON.parse(await readFile(path.join(target, OFFLINE_UPDATE_OWNER_FILE), "utf8"));
  } catch (error) {
    throw new Error(`无法验证离线更新 Python 环境备份所有权：${error.message}`);
  }
  if (marker.schema !== record.schema || marker.product !== PRODUCT || marker.kind !== "environment"
    || marker.token !== record.token || !pathsEqual(marker.projectRoot, projectRoot) || !pathsEqual(marker.path, target)) {
    throw new Error("离线更新 Python 环境备份所有权不匹配");
  }
  return target;
}

export async function removeEnvironmentBackup({ projectRoot, record } = {}) {
  const target = await assertEnvironmentBackup({ projectRoot, record, requireExists: false });
  if (target) await rm(target, { recursive: true, force: true });
}

export async function restoreEnvironmentBackup({ projectRoot, record, venvRoot } = {}) {
  const target = await assertEnvironmentBackup({ projectRoot, record, requireExists: false });
  if (!target) return false;
  await rm(venvRoot, { recursive: true, force: true });
  await rename(target, venvRoot);
  await removeEnvironmentOwnershipMarker({ projectRoot, record, directory: venvRoot });
  return true;
}

export async function removeEnvironmentOwnershipMarker({ projectRoot, record, directory } = {}) {
  validateEnvironmentRecord(projectRoot, record);
  const markerPath = path.join(directory, OFFLINE_UPDATE_OWNER_FILE);
  try {
    const marker = JSON.parse(await readFile(markerPath, "utf8"));
    if (marker.token === record.token && marker.product === PRODUCT && marker.kind === "environment") {
      await rm(markerPath, { force: true });
    }
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
}
