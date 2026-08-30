import { cp, lstat, mkdir, readdir, readFile, realpath, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

export const RELEASE_MANIFEST_FILE = ".xirai-release-manifest.json";
export const RELEASE_MANIFEST_SCHEMA = 1;
export const RELEASE_MANAGED_DIRECTORIES = ["src", "backend", "scripts", "public", "assistant"];
export const RELEASE_MANAGED_FILES = [
  "index.html", "vite.config.js", "package.json", "package-lock.json", "README.md", "README.zh-CN.md",
  "Start-XirAI.bat", "Start-XirAI.sh", "Setup-XirAI.bat", "Setup-XirAI.sh", "XirAI-Setup.desktop",
  "XirAI-Start.desktop", ".env.example", ".gitattributes", ".gitignore",
  "models/model-paths.json", "models/recommended-models.json", "models/yolo-models.json",
  "models/background-removal-models.json", "models/README.md",
];
export const RELEASE_REQUIRED_FILES = ["package.json", "package-lock.json", "vite.config.js", "index.html", "backend/requirements.txt"];
export const RELEASE_REQUIRED_DIRECTORIES = ["src", "backend", "scripts", "assistant"];

const ROOT_FILES = new Set(RELEASE_MANAGED_FILES);
const DIRECTORY_ROOTS = new Set(RELEASE_MANAGED_DIRECTORIES);
const FORBIDDEN_SEGMENTS = new Set([".git", ".venv", "node_modules", "dist", "outputs", "state-cache", ".cache", "logs", "plugins", ".claude", "__pycache__"]);
const FORBIDDEN_FILE = /(^|\/)(?:\.?[^/]*\.test\.[^/]+|test_[^/]+)$/i;
const HANDOFF_FILE = /(^|\/)(?:handoff|HANDOFF)(?:[._-]|$)/;

function portable(relativePath) {
  return relativePath.replace(/\\/g, "/");
}

function comparePortablePaths(left, right) {
  const first = portable(left);
  const second = portable(right);
  return first < second ? -1 : first > second ? 1 : 0;
}

function safeRelativePath(relativePath) {
  if (typeof relativePath !== "string" || !relativePath || relativePath.includes("\0")) return false;
  const normalized = portable(relativePath);
  return !normalized.startsWith("/") && !/^[A-Za-z]:/.test(normalized)
    && !normalized.split("/").some((part) => !part || part === "." || part === "..");
}

async function lstatOrNull(target, options) {
  try {
    return await lstat(target, options);
  } catch (error) {
    if (error.code === "ENOENT") return null;
    throw error;
  }
}

function comparablePath(target) {
  const resolved = path.resolve(target);
  return process.platform === "win32" ? resolved.toLowerCase() : resolved;
}

function sameOrNestedPath(root, candidate) {
  const first = comparablePath(root);
  const second = comparablePath(candidate);
  return second === first || second.startsWith(first.endsWith(path.sep) ? first : `${first}${path.sep}`);
}

function pathsOverlap(first, second) {
  return sameOrNestedPath(first, second) || sameOrNestedPath(second, first);
}

function assertNoStagingOverlap(root, destination) {
  if (pathsOverlap(root, destination)) {
    throw new Error("Release staging 目录不能与项目目录相同、位于其内部或包含项目目录");
  }
}

function samePath(first, second) {
  return comparablePath(first) === comparablePath(second);
}

async function canonicalUnlinkedDirectory(target, description) {
  const resolved = path.resolve(target);
  const parsed = path.parse(resolved);
  const segments = resolved.slice(parsed.root.length).split(path.sep).filter(Boolean);
  let current = parsed.root;

  for (const segment of segments) {
    current = path.join(current, segment);
    const stats = await lstatOrNull(current);
    if (!stats || !stats.isDirectory() || stats.isSymbolicLink()) {
      throw new Error(`${description} 必须是不存在符号链接或目录联接的已有目录`);
    }
    const canonical = await realpath(current);
    if (!samePath(current, canonical)) {
      throw new Error(`${description} 不能经过符号链接、目录联接或路径别名`);
    }
  }

  return realpath(resolved);
}

function directoryIdentity(stats) {
  return { dev: stats.dev.toString(), ino: stats.ino.toString() };
}

function sameDirectoryIdentity(first, second) {
  return first.dev === second.dev && first.ino === second.ino;
}

async function captureOwnedStagingDirectory(destination, expectedCanonicalPath) {
  const stats = await lstat(destination, { bigint: true });
  if (!stats.isDirectory() || stats.isSymbolicLink()) {
    throw new Error("新建的 Release staging 路径不是安全目录");
  }
  const canonicalPath = await realpath(destination);
  if (!samePath(canonicalPath, expectedCanonicalPath)) {
    throw new Error("新建的 Release staging 目录发生了路径重定向");
  }
  return { path: destination, canonicalPath, identity: directoryIdentity(stats) };
}

async function assertOwnedStagingDirectory(ownership) {
  const stats = await lstatOrNull(ownership.path, { bigint: true });
  if (!stats) return false;
  if (!stats.isDirectory() || stats.isSymbolicLink()
    || !sameDirectoryIdentity(ownership.identity, directoryIdentity(stats))) {
    throw new Error("Release staging 目录所有权已变化，拒绝递归清理");
  }
  const canonicalPath = await realpath(ownership.path);
  if (!samePath(canonicalPath, ownership.canonicalPath)) {
    throw new Error("Release staging 目录路径已重定向，拒绝递归清理");
  }
  return true;
}

async function createOwnedStagingDirectory(root, stagingDirectory) {
  const destination = path.resolve(stagingDirectory);
  assertNoStagingOverlap(root, destination);

  const existing = await lstatOrNull(destination);
  if (existing) {
    const canonicalExisting = await realpath(destination).catch(() => null);
    if (canonicalExisting) assertNoStagingOverlap(root, canonicalExisting);
    throw new Error("Release staging 目录必须不存在；不会递归删除调用方提供的已有路径");
  }

  const parent = path.dirname(destination);
  const canonicalParent = await canonicalUnlinkedDirectory(parent, "Release staging 父目录");
  const canonicalDestination = path.join(canonicalParent, path.basename(destination));
  assertNoStagingOverlap(root, canonicalDestination);

  await mkdir(destination);
  return captureOwnedStagingDirectory(destination, canonicalDestination);
}

async function removeOwnedStagingDirectory(ownership) {
  if (await assertOwnedStagingDirectory(ownership)) {
    await rm(ownership.path, { recursive: true, force: false });
  }
}

export function isReleaseManagedFile(relativePath) {
  if (!safeRelativePath(relativePath)) return false;
  const normalized = portable(relativePath);
  const parts = normalized.split("/");
  if (parts.some((part) => FORBIDDEN_SEGMENTS.has(part)) || parts[0] === "models" && !ROOT_FILES.has(normalized)) return false;
  if (normalized === "opencode.json" || parts[0] === ".claude" || /\.py[co]$/i.test(normalized) || FORBIDDEN_FILE.test(normalized) || HANDOFF_FILE.test(normalized)) return false;
  return ROOT_FILES.has(normalized) || (DIRECTORY_ROOTS.has(parts[0]) && parts.length > 1);
}

function assertManifestPaths(files) {
  if (!Array.isArray(files) || !files.length || files.some((item) => !isReleaseManagedFile(item))) {
    throw new Error("Release manifest 包含未允许的文件");
  }
  const sorted = [...files].sort(comparePortablePaths);
  if (new Set(sorted).size !== sorted.length || sorted.some((item, index) => item !== files[index])) {
    throw new Error("Release manifest 文件列表必须唯一且按路径排序");
  }
  for (const required of RELEASE_REQUIRED_FILES) {
    if (!sorted.includes(required)) throw new Error(`Release manifest 缺少关键文件：${required}`);
  }
  return sorted;
}

async function regularFile(target, description) {
  const stats = await lstat(target).catch((error) => error.code === "ENOENT" ? null : Promise.reject(error));
  if (!stats || !stats.isFile() || stats.isSymbolicLink()) throw new Error(`${description} 不是安全普通文件`);
}

async function collectFiles(root, relative = "") {
  const directory = path.join(root, ...relative.split("/").filter(Boolean));
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries.sort((left, right) => comparePortablePaths(left.name, right.name))) {
    const child = relative ? `${relative}/${entry.name}` : entry.name;
    const traversable = entry.isDirectory() && !entry.isSymbolicLink()
      && (relative === "" && (DIRECTORY_ROOTS.has(entry.name) || entry.name === "models") || Boolean(relative));
    if (!isReleaseManagedFile(child)) {
      if (traversable) files.push(...await collectFiles(root, child));
      continue;
    }
    const source = path.join(root, ...child.split("/"));
    if (entry.isSymbolicLink() || (!entry.isDirectory() && !entry.isFile())) throw new Error(`Release 文件类型不安全：${child}`);
    if (entry.isDirectory()) files.push(...await collectFiles(root, child));
    else files.push(child);
  }
  return files;
}

async function collectStrictPackageFiles(root, relative = "") {
  const directory = path.join(root, ...relative.split("/").filter(Boolean));
  const files = [];
  for (const entry of (await readdir(directory, { withFileTypes: true })).sort((left, right) => comparePortablePaths(left.name, right.name))) {
    const child = relative ? `${relative}/${entry.name}` : entry.name;
    if (entry.isSymbolicLink() || (!entry.isDirectory() && !entry.isFile())) throw new Error(`Release package 文件类型不安全：${child}`);
    if (entry.isDirectory()) {
      const allowedDirectory = relative === "" && (DIRECTORY_ROOTS.has(entry.name) || entry.name === "models")
        || Boolean(relative) && (DIRECTORY_ROOTS.has(relative.split("/")[0]) || relative === "models");
      if (!allowedDirectory) throw new Error(`Release package 包含未知目录：${child}`);
      files.push(...await collectStrictPackageFiles(root, child));
    } else if (isReleaseManagedFile(child)) {
      files.push(child);
    } else if (child !== RELEASE_MANIFEST_FILE) {
      throw new Error(`Release package 包含未知或禁止文件：${child}`);
    }
  }
  return files;
}

export async function createReleaseManifest({ projectRoot } = {}) {
  if (!projectRoot) throw new Error("缺少 projectRoot");
  const root = path.resolve(projectRoot);
  for (const directory of RELEASE_REQUIRED_DIRECTORIES) {
    const stats = await lstat(path.join(root, directory)).catch((error) => error.code === "ENOENT" ? null : Promise.reject(error));
    if (!stats || !stats.isDirectory() || stats.isSymbolicLink()) throw new Error(`Release package 缺少关键目录：${directory}`);
  }
  const files = (await collectFiles(root)).sort(comparePortablePaths);
  assertManifestPaths(files);
  return { schema: RELEASE_MANIFEST_SCHEMA, product: "XiriaCanvas AI", packageType: "release", files };
}

export async function validateReleasePackageDirectory({ packageRoot } = {}) {
  if (!packageRoot) throw new Error("缺少 packageRoot");
  const root = path.resolve(packageRoot);
  const manifestPath = path.join(root, RELEASE_MANIFEST_FILE);
  await regularFile(manifestPath, "Release manifest");
  let manifest;
  try { manifest = JSON.parse(await readFile(manifestPath, "utf8")); } catch { throw new Error("Release manifest 不是有效 JSON"); }
  if (manifest?.schema !== RELEASE_MANIFEST_SCHEMA || manifest.product !== "XiriaCanvas AI" || manifest.packageType !== "release") {
    throw new Error("Release manifest 标识无效");
  }
  const files = assertManifestPaths(manifest.files);
  for (const relativePath of files) await regularFile(path.join(root, ...relativePath.split("/")), `Release 文件 ${relativePath}`);
  const actual = (await collectStrictPackageFiles(root)).sort(comparePortablePaths);
  if (JSON.stringify(actual) !== JSON.stringify(files)) throw new Error("Release package 包含未知文件或缺少 manifest 文件");
  return { ...manifest, files };
}

export async function stageReleasePackage({ projectRoot, stagingDirectory, wrapperName = "XiriaCanvas-AI-release" } = {}) {
  if (!projectRoot || !stagingDirectory) throw new Error("stageReleasePackage 需要 projectRoot 和 stagingDirectory");
  if (!safeRelativePath(wrapperName) || portable(wrapperName).includes("/")) throw new Error("Release wrapper 名称无效");
  const requestedRoot = path.resolve(projectRoot);
  const root = await realpath(requestedRoot).catch((error) => {
    if (error.code === "ENOENT") throw new Error("Release 项目目录不存在");
    throw error;
  });
  const rootStats = await lstat(root);
  if (!rootStats.isDirectory() || rootStats.isSymbolicLink()) throw new Error("Release 项目目录无效");
  const manifest = await createReleaseManifest({ projectRoot: root });
  let ownership;
  try {
    ownership = await createOwnedStagingDirectory(root, stagingDirectory);
    const destination = ownership.path;
    const packageRoot = path.join(destination, wrapperName);
    await mkdir(packageRoot);
    for (const relativePath of manifest.files) {
      const source = path.join(root, ...relativePath.split("/"));
      const target = path.join(packageRoot, ...relativePath.split("/"));
      await mkdir(path.dirname(target), { recursive: true });
      await cp(source, target, { force: false, errorOnExist: true, dereference: false, preserveTimestamps: true });
    }
    await writeFile(path.join(packageRoot, RELEASE_MANIFEST_FILE), `${JSON.stringify(manifest, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
    await validateReleasePackageDirectory({ packageRoot });
    return { stagingDirectory: destination, packageRoot, wrapperName, manifest };
  } catch (error) {
    if (ownership) {
      try {
        await removeOwnedStagingDirectory(ownership);
      } catch (cleanupError) {
        throw new Error(`${error.message}；暂存目录未清理：${cleanupError.message}`, { cause: error });
      }
    }
    throw error;
  }
}

async function runCli() {
  const [projectRoot, stagingDirectory, wrapperName] = process.argv.slice(2);
  if (!projectRoot || !stagingDirectory) throw new Error("用法：node scripts/release-package.mjs <projectRoot> <stagingDirectory> [wrapperName]");
  const result = await stageReleasePackage({ projectRoot, stagingDirectory, ...(wrapperName ? { wrapperName } : {}) });
  process.stdout.write(`${result.packageRoot}\n`);
}

if (process.argv[1] && samePath(
  fileURLToPath(import.meta.url),
  fileURLToPath(pathToFileURL(path.resolve(process.argv[1]))),
)) {
  runCli().catch((error) => { process.stderr.write(`${error.message}\n`); process.exitCode = 1; });
}
