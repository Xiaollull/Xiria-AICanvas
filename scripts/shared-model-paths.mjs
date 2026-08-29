import { createHash } from "node:crypto";
import { readdir, readFile, realpath, stat, writeFile } from "node:fs/promises";
import path from "node:path";

import {
  MAX_SHARED_ROOTS,
  SHARED_DIRECTORY_KINDS,
  SHARED_KIND_EXTENSIONS,
  SHARED_KIND_LABELS,
  SHARED_ROOT_ID_LENGTH,
  SHARED_ROOT_KINDS,
  classifyDirectoryName,
  formatSharedRef,
  normalizeSharedEngines,
  normalizeSharedLabel,
  normalizeSharedRoots,
  parseSharedRef,
  sharedKindSupported,
  sharedRootIdentityPath,
} from "../src/shared-model-refs.js";

export {
  SHARED_DIRECTORY_KINDS,
  SHARED_KIND_LABELS,
  classifyDirectoryName,
  formatSharedRef,
  parseSharedRef,
  sharedKindSupported,
};

// Pointing the scanner at a drive root would otherwise walk the whole disk
// while the request hangs. These caps make the worst case bounded and let the
// report say "stopped early" instead of never answering.
export const SHARED_SCAN_MAX_FILES = 20000;
export const SHARED_SCAN_MAX_DEPTH = 6;

// Whether this host's filesystem considers two spellings of a path to be one
// folder. Mirrored by `SHARED_PATHS_CASE_INSENSITIVE` in the Python twin, which
// must reach the same answer on the same machine or a shared LoRA registered by
// the control plane would not resolve at generation time.
export const SHARED_PATHS_CASE_INSENSITIVE = process.platform === "win32";

export function sharedRootId(absolutePath, { caseInsensitive = SHARED_PATHS_CASE_INSENSITIVE } = {}) {
  const identity = sharedRootIdentityPath(absolutePath, { caseInsensitive });
  return createHash("sha256").update(identity, "utf8").digest("hex").slice(0, SHARED_ROOT_ID_LENGTH);
}

export function sharedPathConfigFile(projectRoot) {
  return path.join(projectRoot, "models", "shared-paths.json");
}

export async function readSharedRoots(projectRoot) {
  let raw;
  try {
    raw = await readFile(sharedPathConfigFile(projectRoot), "utf8");
  } catch (error) {
    // No file means the feature was never used. That is the default state and
    // must never read as an error, or a fresh install would show a red banner.
    if (error.code === "ENOENT") return [];
    throw error;
  }
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw Object.assign(new Error("共享目录配置文件不是有效的 JSON"), { statusCode: 500 });
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw Object.assign(new Error("共享目录配置根节点必须是对象"), { statusCode: 500 });
  }
  return normalizeSharedRoots(parsed);
}

export async function writeSharedRoots(projectRoot, roots) {
  const normalized = normalizeSharedRoots({ roots });
  const body = { version: 1, roots: normalized };
  await writeFile(sharedPathConfigFile(projectRoot), `${JSON.stringify(body, null, 2)}\n`, "utf8");
  return normalized;
}

function isFilesystemRoot(resolved) {
  return path.dirname(resolved) === resolved;
}

// Registering a folder is the one moment the user hands this app read access to
// somewhere outside the project, so the checks that matter happen here rather
// than at every later read.
export async function resolveSharedRootPath(projectRoot, candidate) {
  const requested = String(candidate ?? "").trim();
  if (!requested) throw Object.assign(new Error("请输入目录路径"), { statusCode: 400 });
  if (!path.isAbsolute(requested)) {
    // `path.isAbsolute` follows the host, so the example has to as well: a
    // Linux user told to type `F:\...` has been handed a path their own
    // filesystem would reject.
    const example = process.platform === "win32" ? "F:\\AI\\sd-webui\\models\\Lora" : "/home/you/stable-diffusion-webui/models/Lora";
    throw Object.assign(new Error(`共享目录必须是绝对路径，例如 ${example}`), { statusCode: 400 });
  }
  let resolved;
  try {
    resolved = await realpath(path.resolve(requested));
  } catch {
    throw Object.assign(new Error("目录不存在或无法访问"), { statusCode: 400 });
  }
  let entryStat;
  try {
    entryStat = await stat(resolved);
  } catch {
    throw Object.assign(new Error("目录不存在或无法访问"), { statusCode: 400 });
  }
  if (!entryStat.isDirectory()) throw Object.assign(new Error("共享路径必须是文件夹，不能是文件"), { statusCode: 400 });
  if (isFilesystemRoot(resolved)) {
    throw Object.assign(new Error("不能共享磁盘根目录，请选择具体的模型文件夹"), { statusCode: 400 });
  }
  return resolved;
}

// A folder already covered by `models/model-paths.json` is not an error, but
// sharing it lists every file twice under two different references. The caller
// surfaces this as a warning so the choice stays the user's.
export function sharedRootOverlapsProject(projectRoot, resolved) {
  const modelsRoot = path.resolve(projectRoot, "models");
  const relative = path.relative(modelsRoot, resolved);
  return !relative || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

async function scanKindDirectory(directory, extensions, budget, maxDepth = SHARED_SCAN_MAX_DEPTH) {
  const allow = new Set(extensions);
  let files = 0;
  let bytes = 0;
  let truncated = false;

  const walk = async (current, depth) => {
    if (truncated || depth > maxDepth) return;
    let entries;
    try {
      entries = await readdir(current, { withFileTypes: true });
    } catch {
      // An unreadable subfolder is reported as nothing found rather than
      // failing the whole scan; foreign trees routinely contain locked caches.
      return;
    }
    for (const entry of entries) {
      if (truncated) return;
      const entryPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        await walk(entryPath, depth + 1);
        continue;
      }
      if (!entry.isFile() || !allow.has(path.extname(entry.name).toLowerCase())) continue;
      if (budget.scanned >= SHARED_SCAN_MAX_FILES) {
        truncated = true;
        return;
      }
      budget.scanned += 1;
      files += 1;
      try {
        bytes += (await stat(entryPath)).size;
      } catch {
        // Counted but unsized: a file that vanished mid-scan still existed.
      }
    }
  };

  await walk(directory, 0);
  return { files, bytes, truncated };
}

function kindEntry(directory, kind, relative, scan) {
  return {
    directory: relative,
    kind,
    label: SHARED_KIND_LABELS[kind] || kind,
    supported: sharedKindSupported(kind),
    files: scan.files,
    bytes: scan.bytes,
    truncated: scan.truncated,
    absolutePath: directory,
  };
}

// The "verify and categorise" half of the model-folder entry point. A folder
// name is only a hypothesis — `vae` proves nothing until the scan finds files
// the loader could actually read — so every reported kind carries its count.
export async function inspectSharedDirectory(projectRoot, requestedPath, requestedKind = "auto") {
  const kind = SHARED_ROOT_KINDS.includes(requestedKind) ? requestedKind : "auto";
  const resolved = await resolveSharedRootPath(projectRoot, requestedPath);
  const budget = { scanned: 0 };
  const entries = [];
  const warnings = [];

  if (kind === "auto") {
    let children;
    try {
      children = await readdir(resolved, { withFileTypes: true });
    } catch {
      throw Object.assign(new Error("无法读取目录内容"), { statusCode: 400 });
    }
    for (const child of children) {
      if (!child.isDirectory()) continue;
      const childKind = classifyDirectoryName(child.name);
      if (!childKind) continue;
      const childPath = path.join(resolved, child.name);
      const scan = await scanKindDirectory(childPath, SHARED_KIND_EXTENSIONS[childKind], budget);
      if (!scan.files) continue;
      entries.push(kindEntry(childPath, childKind, child.name, scan));
    }
    // A leaf folder the user reached through the model entry point still has to
    // work: no recognised subfolder means the folder itself may be the library.
    // Its own name decides how deep we are willing to look. `Lora/` earns a
    // recursive scan because its categories are subfolders; an unrecognised
    // name only counts files sitting directly inside it, so pointing at a whole
    // tool install cannot quietly declare the entire tree to be checkpoints.
    if (!entries.length) {
      const selfKind = classifyDirectoryName(path.basename(resolved));
      const kindGuess = selfKind || "checkpoints";
      const scan = await scanKindDirectory(resolved, SHARED_KIND_EXTENSIONS[kindGuess], budget, selfKind ? SHARED_SCAN_MAX_DEPTH : 0);
      if (scan.files) entries.push(kindEntry(resolved, kindGuess, ".", scan));
    }
  } else {
    const scan = await scanKindDirectory(resolved, SHARED_KIND_EXTENSIONS[kind], budget);
    if (scan.files) entries.push(kindEntry(resolved, kind, ".", scan));
  }

  if (!entries.length) warnings.push("该目录下没有找到可识别的模型文件");
  if (sharedRootOverlapsProject(projectRoot, resolved)) {
    warnings.push("该目录已是本项目的本地模型目录，共享后同一文件会重复出现");
  }
  const unsupported = entries.filter((entry) => !entry.supported);
  if (unsupported.length) {
    warnings.push(`${unsupported.map((entry) => entry.label).join("、")} 已识别，但当前版本尚未接入生成流程`);
  }
  if (entries.some((entry) => entry.truncated)) {
    warnings.push(`目录过大，仅统计前 ${SHARED_SCAN_MAX_FILES} 个文件`);
  }

  return {
    path: resolved,
    id: sharedRootId(resolved),
    kind,
    entries: entries.sort((first, second) => second.files - first.files),
    warnings,
    totals: {
      files: entries.reduce((total, entry) => total + entry.files, 0),
      bytes: entries.reduce((total, entry) => total + entry.bytes, 0),
    },
  };
}

export function sharedRootDraft(inspection, { label, engines } = {}) {
  return {
    id: inspection.id,
    path: inspection.path,
    kind: inspection.kind,
    label: normalizeSharedLabel(label, inspection.path),
    enabled: true,
    engines: normalizeSharedEngines(engines),
  };
}

export function upsertSharedRoot(roots, draft) {
  const normalized = normalizeSharedRoots({ roots });
  const existing = normalized.findIndex((root) => root.id === draft.id);
  if (existing >= 0) {
    const next = [...normalized];
    next[existing] = { ...normalized[existing], ...draft };
    return next;
  }
  if (normalized.length >= MAX_SHARED_ROOTS) {
    throw Object.assign(new Error(`最多只能共享 ${MAX_SHARED_ROOTS} 个目录`), { statusCode: 400 });
  }
  return [...normalized, draft];
}

// Which directories inside a root hold a given kind. `auto` roots are
// re-inspected by name so a folder the user adds to their other tool later
// shows up without re-registering anything here.
export async function sharedKindDirectories(root, kind) {
  const resolved = await realpath(root.path).catch(() => "");
  if (!resolved) return [];
  if (root.kind === kind) return [{ directory: resolved, prefix: "" }];
  if (root.kind !== "auto") return [];
  let children;
  try {
    children = await readdir(resolved, { withFileTypes: true });
  } catch {
    return [];
  }
  const matches = [];
  for (const child of children) {
    if (child.isDirectory() && classifyDirectoryName(child.name) === kind) {
      matches.push({ directory: path.join(resolved, child.name), prefix: child.name });
    }
  }
  // A leaf folder registered through the model entry point classifies itself.
  if (!matches.length && classifyDirectoryName(path.basename(resolved)) === kind) {
    matches.push({ directory: resolved, prefix: "" });
  }
  return matches;
}

export const SHARED_CATEGORY_ID = "shared";
export const SHARED_CATEGORY_LABEL = "共享";
export const SHARED_ROOT_FOLDER_LABEL = "根目录";

// The LoRA manager's fifth category. It keeps a flat `models` array so the
// existing consumers (category counts, the analytics roll-up, mount lookups)
// keep working untouched, and adds the `roots` tree that the Shared view uses
// to show the folder structure the user actually has on disk.
export function shapeSharedLoraCategory(listings) {
  const roots = [];
  const flat = [];
  for (const listing of listings) {
    const folders = new Map();
    for (const model of listing.models) {
      flat.push(model);
      const segments = model.name.split("/");
      const folder = segments.length > 1 ? segments.slice(0, -1).join("/") : "";
      if (!folders.has(folder)) folders.set(folder, { name: folder, label: folder || SHARED_ROOT_FOLDER_LABEL, models: [] });
      folders.get(folder).models.push(model);
    }
    const ordered = [...folders.values()].sort((first, second) => {
      // Loose files first, then folders alphabetically, so the tree opens on
      // the shallowest thing the user is most likely looking for.
      if (!first.name !== !second.name) return first.name ? 1 : -1;
      return first.name.localeCompare(second.name, "zh-CN");
    });
    for (const folder of ordered) folder.models.sort((first, second) => first.name.localeCompare(second.name, "zh-CN"));
    roots.push({
      id: listing.root.id,
      label: listing.root.label,
      path: listing.root.path,
      kind: listing.root.kind,
      folders: ordered,
      files: listing.models.length,
      bytes: listing.models.reduce((total, model) => total + (model.size || 0), 0),
    });
  }
  return {
    id: SHARED_CATEGORY_ID,
    label: SHARED_CATEGORY_LABEL,
    directory: roots.length === 1 ? roots[0].path : `${roots.length} 个共享目录`,
    shared: true,
    roots,
    models: flat,
  };
}

// Resolving a `shared:` reference back to a real file. This is the boundary the
// runtime trusts, so containment is re-checked against the realpath: a symlink
// inside somebody else's model folder must not become a way out of it.
export async function resolveSharedFile(roots, value, { extensions } = {}) {
  const parsed = parseSharedRef(value);
  if (!parsed) throw Object.assign(new Error("共享模型引用格式无效"), { statusCode: 400 });
  const root = normalizeSharedRoots({ roots }).find((entry) => entry.id === parsed.rootId);
  if (!root) throw Object.assign(new Error("共享目录未注册或已被移除"), { statusCode: 404 });
  if (!root.enabled) throw Object.assign(new Error("该共享目录已停用"), { statusCode: 403 });
  if (extensions && !extensions.has(path.extname(parsed.relativePath).toLowerCase())) {
    throw Object.assign(new Error("共享模型文件类型不受支持"), { statusCode: 400 });
  }
  let rootReal;
  let fileReal;
  try {
    rootReal = await realpath(root.path);
    fileReal = await realpath(path.resolve(rootReal, parsed.relativePath));
  } catch {
    throw Object.assign(new Error("共享模型文件不存在"), { statusCode: 404 });
  }
  const relative = path.relative(rootReal, fileReal);
  if (!relative || relative.startsWith("..") || path.isAbsolute(relative)) {
    throw Object.assign(new Error("共享模型文件不在已注册的共享目录内"), { statusCode: 403 });
  }
  if (!(await stat(fileReal)).isFile()) {
    throw Object.assign(new Error("共享模型引用的不是文件"), { statusCode: 400 });
  }
  return fileReal;
}
