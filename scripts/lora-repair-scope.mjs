import path from "node:path";
import { lstat, readdir, realpath, stat } from "node:fs/promises";
import { parseSharedRef } from "../src/shared-model-refs.js";
import { libraryError, validRelative } from "./lora-library.mjs";

// P10.2 repair scope. The server owns normalization: a source directory filters
// repair rows (it may be missing), a target directory constrains candidates and
// must exist, and coverage records exactly which root prefixes were observed so
// a narrowed scan never claims whole-root completeness.

const fold = (value, insensitive) => insensitive ? value.toLowerCase() : value;
export { withinPrefix, scanCovers } from "./lora-repair-ranking.mjs";
import { withinPrefix, repairRelative as repairRelativeOf } from "./lora-repair-ranking.mjs";
/** Entries: [{ engine, rootId, prefix }] from resolveRepairDirectory. */
export function inRepairScope(entries, item, insensitive) {
  return entries.some((entry) => entry.engine === item.engine && entry.rootId === item.rootId && withinPrefix(repairRelativeOf(item.value), entry.prefix, insensitive));
}
export function fullRootScan(scan) { return scan.roots.every((root) => !root.narrowed); }

export function normalizeDirectoryText(value) {
  return String(value ?? "").trim().replaceAll("\\", "/").replace(/\/{2,}/g, "/").replace(/^\.\//, "").replace(/\/+$/, "");
}
const contained = (root, file, insensitive) => {
  const relative = path.relative(fold(root, insensitive), fold(file, insensitive));
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
};

/**
 * Resolve one user directory (engine-relative, project-relative, a server
 * absolute path inside a registered root, `shared:<id>/dir`, or a picked
 * `{ rootId, path }`) to registered `engine + rootId + prefix` entries for the
 * selected engines. Never switches engines, never opens arbitrary disk paths.
 */
export async function resolveRepairDirectory(input, { roots, engines, projectRoot, role, platform = process.platform }) {
  const insensitive = platform === "win32";
  const fail = (message) => { throw libraryError(message, 400); };
  const picked = input && typeof input === "object" && !Array.isArray(input);
  if (input !== undefined && input !== null && !picked && typeof input !== "string") fail("目录参数无效");
  if (picked && typeof input.path !== "string") fail("目录参数无效");
  if (String(picked ? input.path : input ?? "").length > 1000) fail("目录路径过长");
  const text = picked ? normalizeDirectoryText(input.path) : normalizeDirectoryText(input);
  if (!picked && !text) return null;
  if (/^(?:\/\/|\\\\)/.test(String(picked ? input.path : input).trim())) fail("不接受网络或设备路径；请选择已登记的 LoRA 目录");
  const selected = roots.filter((root) => engines.includes(root.engine));
  let matches = [];
  const relativeTo = (root, absolute) => path.relative(root.directory, absolute).split(path.sep).join("/");
  if (picked) {
    if (typeof input.rootId !== "string") fail("目录选择缺少根目录");
    matches = selected.filter((root) => root.id === input.rootId).map((root) => ({ root, prefix: text }));
    if (!matches.length) fail(roots.some((root) => root.id === input.rootId) ? "该根目录不适用于所选引擎" : "未知的根目录");
  } else if (text.startsWith("shared:")) {
    const parsed = parseSharedRef(`${text}/x`);
    if (!parsed) fail("共享目录引用格式无效");
    matches = selected.filter((root) => root.id === parsed.rootId).map((root) => ({ root, prefix: parsed.relativePath.split("/").slice(0, -1).join("/") }));
    if (!matches.length) fail("该共享目录未对所选引擎启用");
  } else if (path.win32.isAbsolute(text) && /^[a-z]:/i.test(text) || path.posix.isAbsolute(text) || text.startsWith("models/")) {
    const absolute = text.startsWith("models/") ? path.resolve(projectRoot, text) : path.resolve(text);
    const owners = [];
    for (const root of roots) {
      let base = root.directory;
      try { base = await realpath(root.directory); } catch {}
      for (const candidate of new Set([root.directory, base])) if (contained(candidate, absolute, insensitive)) { owners.push({ root, prefix: normalizeDirectoryText(relativeTo({ directory: candidate }, absolute)) }); break; }
    }
    matches = owners.filter((owner) => engines.includes(owner.root.engine));
    if (!matches.length) {
      const others = [...new Set(owners.map((owner) => owner.root.engine))];
      fail(others.length ? `该目录属于 ${others.join(" / ")} 的 LoRA 根目录，不属于 ${engines.length > 1 ? "所选引擎" : engines[0]}` : "该目录不在已登记的 LoRA 根目录内");
    }
  } else {
    matches = selected.filter((root) => root.id === `local-${root.engine}`).map((root) => ({ root, prefix: text }));
    if (!matches.length) fail("所选引擎没有本地 LoRA 根目录");
  }
  const entries = [];
  for (const { root, prefix } of matches) {
    if (prefix && !validRelative(prefix)) fail("目录路径无效（不允许 .. 或越出根目录）");
    if (root.scopeDirectories && !root.scopeDirectories.some((scope) => withinPrefix(prefix, scope, insensitive))) fail("该目录不在共享目录的 LoRA 范围内");
    let exists = false;
    if (root.enabled !== false) {
      const absolute = path.join(root.directory, ...(prefix ? prefix.split("/") : []));
      try {
        for (let cursor = root.directory, parts = prefix ? prefix.split("/") : []; parts.length; parts = parts.slice(1)) {
          cursor = path.join(cursor, parts[0]);
          if ((await lstat(cursor)).isSymbolicLink()) fail("目录包含链接或联接点；不扫描其目标");
        }
        const [base, resolved] = [await realpath(root.directory), await realpath(absolute)];
        if (!contained(base, resolved, insensitive)) fail("目录越出已登记的根目录");
        if (!(await stat(resolved)).isDirectory()) fail("该路径不是文件夹");
        await readdir(resolved);
        exists = true;
      } catch (error) {
        if (error.statusCode) throw error;
        if (error.code !== "ENOENT" && error.code !== "ENOTDIR") fail(`目录不可读取（${error.code || "unreadable"}）`);
      }
    }
    if (role === "target" && root.enabled === false) fail("目标所在的共享目录已停用");
    if (role === "target" && !exists) fail("目标目录不存在或不可读取");
    entries.push({ engine: root.engine, rootId: root.id, prefix, exists, shared: Boolean(root.shared), label: root.shared ? root.label || "共享目录" : "本地", enabled: root.enabled !== false });
  }
  return { input: picked ? { rootId: input.rootId, path: text } : text, entries };
}

/** Lightweight folder navigation: directory names only, no file opened or hashed. */
export async function listRepairDirectories(roots, engines, { maxEntries = 3000, maxDepth = 8 } = {}) {
  const result = [];
  for (const root of roots.filter((item) => engines.includes(item.engine))) {
    const entry = { engine: root.engine, rootId: root.id, shared: Boolean(root.shared), label: root.shared ? root.label || "共享目录" : "本地", enabled: root.enabled !== false, directories: [], truncated: false };
    result.push(entry);
    if (!entry.enabled) continue;
    const walk = async (relative, depth) => {
      if (depth > maxDepth) return;
      if (entry.directories.length >= maxEntries) { entry.truncated = true; return; }
      let children;
      try { children = await readdir(path.join(root.directory, ...(relative ? relative.split("/") : [])), { withFileTypes: true }); } catch { return; }
      for (const child of children.filter((item) => item.isDirectory()).sort((a, b) => a.name.localeCompare(b.name))) {
        if (entry.directories.length >= maxEntries) { entry.truncated = true; return; }
        const next = relative ? `${relative}/${child.name}` : child.name;
        entry.directories.push(next);
        await walk(next, depth + 1);
      }
    };
    for (const scope of root.scopeDirectories || [""]) { if (scope) entry.directories.push(scope); await walk(scope, 1); }
  }
  return result;
}
