import { spawn } from "node:child_process";
import { chmod, cp, lstat, mkdir, readFile, readdir, realpath, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { downloadFile } from "./download.mjs";
import { assertEnvironmentBackup, assertOfflineUpdateTemp, createOfflineUpdateTemp, OFFLINE_UPDATE_OWNER_FILE, removeEnvironmentBackup, removeEnvironmentOwnershipMarker, removeOfflineUpdateTemp, restoreEnvironmentBackup } from "./offline-update-temp.mjs";
import { assertOfflineUpdateLock } from "./offline-update-lock.mjs";
import { getSetupMarkerPath, readSetupMarker, writeSetupMarker } from "./setup-state.mjs";
import { RELEASE_MANAGED_DIRECTORIES, RELEASE_MANAGED_FILES, RELEASE_MANIFEST_FILE, validateReleasePackageDirectory } from "./release-package.mjs";

const SEVEN_ZIP_URL = "https://github.com/ip7z/7zip/releases/download/26.02/7zr.exe";
const SEVEN_ZIP_SHA256 = "56b8cc9f4971cef253644fafe54063ed7fdca551d4dee0f8c6baa81b855acd72";
const SEVEN_ZIP_WINDOWS_PACKAGES = {
  x64: {
    url: "https://github.com/ip7z/7zip/releases/download/26.02/7z2602-x64.exe",
    sha256: "6745fa76dc2ea031596d8678f6f6b99c3c1b435b4164a63485adbbc7b8d82ef0",
  },
  arm64: {
    url: "https://github.com/ip7z/7zip/releases/download/26.02/7z2602-arm64.exe",
    sha256: "7c6fde79ed5e11b81c7bb6573b7962d3b6322aa5fce69c33ed19f672b55173ab",
  },
  ia32: {
    url: "https://github.com/ip7z/7zip/releases/download/26.02/7z2602.exe",
    sha256: "17d894c17b04984b6ffcc1b31926b39c42d315cd861c3adbf7f34bd941d529ac",
  },
};
const SEVEN_ZIP_LINUX_PACKAGES = {
  x64: {
    url: "https://github.com/ip7z/7zip/releases/download/26.02/7z2602-linux-x64.tar.xz",
    sha256: "41aaba7b1235304ab5aa0624530c67ae829496cd29e875925271efdccc28c03e",
  },
  arm64: {
    url: "https://github.com/ip7z/7zip/releases/download/26.02/7z2602-linux-arm64.tar.xz",
    sha256: "70ea6cc737ae1495ea2d7eb20ef3120fe579bd3f1a83a9d2362b62ec5bde2bba",
  },
};
const MAX_COMMAND_OUTPUT = 64 * 1024 * 1024;
const NODE_DEPENDENCY_ERROR = "更新包包含前端依赖变化；为避免运行中的 Node 模块在 Windows 上被部分替换，请使用完整安装包更新";

const ARCHIVE_EXTENSIONS = [
  ".tar.bz2",
  ".tar.gz",
  ".tar.xz",
  ".tar.zst",
  ".tbz2",
  ".tgz",
  ".txz",
  ".tzst",
  ".7z",
  ".rar",
  ".tar",
  ".tbz",
  ".zip",
];

const MANAGED_DIRECTORIES = RELEASE_MANAGED_DIRECTORIES;
const MANAGED_FILES = RELEASE_MANAGED_FILES;
const MANAGED_MODEL_MANIFESTS = new Set([
  "models/recommended-models.json",
  "models/model-paths.json",
  "models/yolo-models.json",
  "models/background-removal-models.json",
  "models/README.md",
]);
const MANAGED_ITEM_KINDS = new Map([
  ...MANAGED_DIRECTORIES.map((relativePath) => [relativePath, "directory"]),
  ...MANAGED_FILES.map((relativePath) => [relativePath, "file"]),
]);

const FORBIDDEN_TOP_LEVEL_ITEMS = new Set([
  ".venv",
  "node_modules",
  "models",
  "outputs",
  "logs",
  "state-cache",
  ".cache",
  ".env",
  "Ultralytics",
  // Installed plugin folders are user-owned program code. An update must never replace, remove,
  // migrate, or execute anything under `plugins/`, so it can never enter an update plan even if a
  // future change adds it to the managed lists by mistake.
  "plugins",
  ".git",
  "dist",
]);

class ArchiveSecurityError extends Error {
  constructor(message) {
    super(message);
    this.name = "ArchiveSecurityError";
  }
}

function emit(report, event) {
  try {
    report?.({ ...event, progress: Math.max(0, Math.min(100, Math.round(event.progress))) });
  } catch {
    // A UI reporting failure must not interrupt or roll back an update.
  }
}

async function lstatOrNull(target) {
  try {
    return await lstat(target);
  } catch (error) {
    if (error.code === "ENOENT") return null;
    throw error;
  }
}

function isWithin(root, candidate, allowRoot = false) {
  const relative = path.relative(path.resolve(root), path.resolve(candidate));
  if (!relative) return allowRoot;
  if (path.isAbsolute(relative)) return false;
  return !relative.split(path.sep).some((segment) => segment === "..");
}

function resolveWithin(root, relativePath) {
  validateArchiveMemberPath(relativePath);
  const resolved = path.resolve(root, ...relativePath.replace(/\\/g, "/").split("/"));
  if (!isWithin(root, resolved)) throw new ArchiveSecurityError(`路径越界：${relativePath}`);
  return resolved;
}

function pathsEqual(first, second) {
  const left = path.resolve(first);
  const right = path.resolve(second);
  return process.platform === "win32" ? left.toLowerCase() === right.toLowerCase() : left === right;
}

async function fileIdentity(target) {
  const stats = await lstat(target, { bigint: true });
  if (!stats.isFile() || stats.isSymbolicLink()) return null;
  return {
    dev: stats.dev.toString(),
    ino: stats.ino.toString(),
    size: stats.size.toString(),
    mtimeNs: stats.mtimeNs.toString(),
  };
}

function sameFileIdentity(first, second) {
  return Boolean(first && second
    && first.dev === second.dev
    && first.ino === second.ino
    && first.size === second.size
    && first.mtimeNs === second.mtimeNs);
}

function archiveExtension(filename) {
  if (typeof filename !== "string") return null;
  const lower = path.basename(filename).toLowerCase();
  return ARCHIVE_EXTENSIONS.find((extension) => lower.endsWith(extension)) || null;
}

export function archiveExtensionAllowed(filename) {
  return archiveExtension(filename) !== null;
}

function archiveKind(filename) {
  const extension = archiveExtension(filename);
  if (extension === ".zip") return "zip";
  if (extension === ".7z") return "7z";
  if (extension === ".rar") return "rar";
  return extension ? "tar" : null;
}

function managedSevenZipPackage(platform = process.platform, architecture = process.arch) {
  const packages = platform === "win32" ? SEVEN_ZIP_WINDOWS_PACKAGES : platform === "linux" ? SEVEN_ZIP_LINUX_PACKAGES : null;
  const packageInfo = packages?.[architecture];
  if (!packageInfo) throw new Error(`7-Zip 不支持当前平台或架构：${platform}/${architecture}`);
  return packageInfo;
}

function validateArchiveMemberPath(memberPath) {
  if (typeof memberPath !== "string" || !memberPath) {
    throw new ArchiveSecurityError("更新归档包含空路径");
  }
  if (/[\0-\x1f\x7f]/.test(memberPath)) {
    throw new ArchiveSecurityError("更新归档路径包含非法控制字符");
  }

  const portablePath = memberPath.replace(/\\/g, "/");
  if (portablePath.startsWith("/") || portablePath.startsWith("//") || /^[A-Za-z]:/.test(portablePath)) {
    throw new ArchiveSecurityError(`更新归档包含绝对路径：${memberPath}`);
  }

  const segments = portablePath.split("/");
  if (segments.some((segment) => segment === "..")) {
    throw new ArchiveSecurityError(`更新归档包含路径穿越：${memberPath}`);
  }
  if (segments.some((segment) => segment.includes(":"))) {
    throw new ArchiveSecurityError(`更新归档路径包含不安全的冒号：${memberPath}`);
  }
  if (process.platform === "win32" && segments.some((segment) => /[ .]$/.test(segment))) {
    throw new ArchiveSecurityError(`更新归档路径在 Windows 上不安全：${memberPath}`);
  }
  if (process.platform === "win32" && segments.some((segment) => {
    const basename = segment.split(".")[0];
    return /^(con|prn|aux|nul|com[1-9]|lpt[1-9])$/i.test(basename);
  })) {
    throw new ArchiveSecurityError(`更新归档路径使用了 Windows 保留名称：${memberPath}`);
  }

  return path.posix.normalize(portablePath);
}

function validateArchiveMembers(members) {
  if (!members.length) throw new Error("更新归档为空");
  const seen = new Set();
  for (const member of members) {
    const normalized = validateArchiveMemberPath(member).replace(/\/$/, "");
    if (normalized === ".") continue;
    const identity = process.platform === "win32" ? normalized.toLowerCase() : normalized;
    if (seen.has(identity)) throw new ArchiveSecurityError(`更新归档包含重复路径：${member}`);
    seen.add(identity);
  }
}

function releaseWrapperFromMembers(members) {
  const markerMembers = members.filter((member) => {
    const normalized = member.replace(/\\/g, "/").replace(/\/$/, "");
    return normalized === RELEASE_MANIFEST_FILE || normalized.endsWith(`/${RELEASE_MANIFEST_FILE}`);
  });
  if (!markerMembers.length) return null;
  if (markerMembers.length !== 1) throw new ArchiveSecurityError("Release 归档包含多个 manifest");
  const normalized = markerMembers[0].replace(/\\/g, "/").replace(/\/$/, "");
  const [wrapper, ...rest] = normalized.split("/");
  if (!wrapper || rest.join("/") !== RELEASE_MANIFEST_FILE || members.some((member) => {
    const candidate = member.replace(/\\/g, "/").replace(/\/$/, "");
    return candidate && candidate !== wrapper && !candidate.startsWith(`${wrapper}/`);
  })) throw new ArchiveSecurityError("Release 归档必须恰有一层 wrapper");
  return wrapper;
}

function commandError(command, args, code, stderr) {
  const detail = stderr.trim().slice(-2000);
  return new Error(`${path.basename(command)} ${args[0] || ""} 执行失败（退出码 ${code}）${detail ? `：${detail}` : ""}`);
}

function rollbackIncompleteError(message, cause) {
  const error = new Error(message, { cause });
  error.rollbackIncomplete = true;
  return error;
}

function runCommand(command, args, { cwd, maxOutput = MAX_COMMAND_OUTPUT } = {}) {
  return new Promise((resolve, reject) => {
    let stdout = Buffer.alloc(0);
    let stderr = Buffer.alloc(0);
    let outputExceeded = false;
    const child = spawn(command, args, {
      cwd,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    });

    const append = (current, chunk) => {
      if (current.length + chunk.length > maxOutput) {
        outputExceeded = true;
        child.kill();
        return current;
      }
      return Buffer.concat([current, chunk]);
    };

    child.stdout.on("data", (chunk) => {
      stdout = append(stdout, chunk);
    });
    child.stderr.on("data", (chunk) => {
      stderr = append(stderr, chunk);
    });
    child.once("error", reject);
    child.once("close", (code) => {
      if (outputExceeded) {
        reject(new Error("归档成员列表过大，已停止处理"));
      } else if (code !== 0) {
        reject(commandError(command, args, code, stderr.toString("utf8")));
      } else {
        resolve({ stdout: stdout.toString("utf8"), stderr: stderr.toString("utf8") });
      }
    });
  });
}

async function commandAvailable(command, args) {
  try {
    await runCommand(command, args, { maxOutput: 2 * 1024 * 1024 });
    return true;
  } catch {
    return false;
  }
}

async function findTar() {
  return await commandAvailable("tar", ["--version"]) ? { kind: "tar", command: "tar" } : null;
}

async function findSystemSevenZip() {
  for (const command of ["7z", "7za", "7zr"]) {
    if (await commandAvailable(command, ["i"])) return { kind: "7zip", command };
  }
  return null;
}

async function ensureSafeDirectory(root, relativePath) {
  let current = root;
  for (const segment of relativePath.split("/")) {
    current = resolveWithin(root, path.relative(root, path.join(current, segment)));
    const stats = await lstatOrNull(current);
    if (stats) {
      if (!stats.isDirectory() || stats.isSymbolicLink()) {
        throw new ArchiveSecurityError(`工具缓存路径不安全：${relativePath}`);
      }
    } else {
      await mkdir(current);
    }
    const canonical = await realpath(current);
    if (!isWithin(root, canonical)) {
      throw new ArchiveSecurityError(`工具缓存路径越界：${relativePath}`);
    }
  }
  return current;
}

async function ensureDownloadedWindowsSevenZip(projectRoot, report) {
  const toolDirectory = await ensureSafeDirectory(projectRoot, ".cache/tools/7zip");
  const bootstrap = resolveWithin(toolDirectory, "7zr.exe");
  const existing = await lstatOrNull(bootstrap);
  if (existing && (!existing.isFile() || existing.isSymbolicLink())) {
    throw new ArchiveSecurityError("7zr.exe 工具缓存不是安全的普通文件");
  }
  emit(report, { phase: "tools", progress: 5, message: "正在准备安全归档工具" });
  await downloadFile({
    routes: [{ id: "7zip-official", label: "7-Zip 官方源", url: SEVEN_ZIP_URL }],
    destination: bootstrap,
    expectedSha256: SEVEN_ZIP_SHA256,
    connections: 1,
    onProgress: ({ currentBytes, totalBytes }) => {
      const ratio = totalBytes ? currentBytes / totalBytes : 0;
      emit(report, { phase: "tools", progress: 5 + ratio * 10, message: "正在下载并校验 7zr.exe" });
    },
    onWarning: (message) => emit(report, { phase: "tools", progress: 8, message }),
  });
  const canonical = await realpath(bootstrap);
  if (!pathsEqual(bootstrap, canonical) || !isWithin(projectRoot, canonical)) {
    throw new ArchiveSecurityError("7zr.exe 工具缓存路径越界");
  }

  const portableDirectory = resolveWithin(toolDirectory, "portable");
  const portableExecutable = resolveWithin(portableDirectory, "7z.exe");
  const portablePackage = managedSevenZipPackage();
  const portableArchive = resolveWithin(toolDirectory, "7zip-portable-package.exe");
  const existingPortableArchive = await lstatOrNull(portableArchive);
  if (existingPortableArchive && (!existingPortableArchive.isFile() || existingPortableArchive.isSymbolicLink())) {
    throw new ArchiveSecurityError("便携 7-Zip 工具包缓存不是安全的普通文件");
  }
  emit(report, { phase: "tools", progress: 12, message: "正在准备完整 7-Zip 命令组件" });
  await downloadFile({
    routes: [{ id: "7zip-portable-official", label: "7-Zip 官方源", url: portablePackage.url }],
    destination: portableArchive,
    expectedSha256: portablePackage.sha256,
    connections: 1,
    onProgress: ({ currentBytes, totalBytes }) => {
      const ratio = totalBytes ? currentBytes / totalBytes : 0;
      emit(report, { phase: "tools", progress: 12 + ratio * 6, message: "正在下载并校验便携 7-Zip 命令" });
    },
    onWarning: (message) => emit(report, { phase: "tools", progress: 15, message }),
  });
  const canonicalPortableArchive = await realpath(portableArchive);
  if (!pathsEqual(portableArchive, canonicalPortableArchive) || !isWithin(projectRoot, canonicalPortableArchive)) {
    throw new ArchiveSecurityError("便携 7-Zip 工具包缓存路径越界");
  }

  // Rebuild the executable from the checksum-verified package instead of trusting a mutable cached binary.
  await rm(portableDirectory, { recursive: true, force: true });
  await ensureSafeDirectory(toolDirectory, "portable");
  // The official installer is a 7z self-extracting archive. Extract it as data; never execute the installer.
  await runCommand(bootstrap, ["x", "-y", "-bd", "-bso0", "-bsp0", `-o${portableDirectory}`, portableArchive]);
  await inspectExtractedTree(portableDirectory);
  if (!await commandAvailable(portableExecutable, ["i"])) {
    throw new Error(`官方便携 7-Zip 不包含当前架构 ${process.arch} 的命令组件`);
  }
  return { kind: "7zip", command: portableExecutable, downloaded: true };
}

async function ensureDownloadedLinuxSevenZip(projectRoot, report) {
  const toolDirectory = await ensureSafeDirectory(projectRoot, ".cache/tools/7zip");
  const portableDirectory = resolveWithin(toolDirectory, "portable");
  const portableExecutable = resolveWithin(portableDirectory, "7zz");
  const portablePackage = managedSevenZipPackage();
  const portableArchive = resolveWithin(toolDirectory, `7zip-linux-${process.arch}.tar.xz`);
  const existingPortableArchive = await lstatOrNull(portableArchive);
  if (existingPortableArchive && (!existingPortableArchive.isFile() || existingPortableArchive.isSymbolicLink())) {
    throw new ArchiveSecurityError("Linux 7-Zip 工具包缓存不是安全的普通文件");
  }
  const tar = await findTar();
  if (!tar) throw new Error("Linux 更新器需要系统 tar 来准备安全归档工具");

  emit(report, { phase: "tools", progress: 5, message: "正在准备安全归档工具" });
  await downloadFile({
    routes: [{ id: "7zip-linux-official", label: "7-Zip 官方源", url: portablePackage.url }],
    destination: portableArchive,
    expectedSha256: portablePackage.sha256,
    connections: 1,
    onProgress: ({ currentBytes, totalBytes }) => {
      const ratio = totalBytes ? currentBytes / totalBytes : 0;
      emit(report, { phase: "tools", progress: 5 + ratio * 15, message: "正在下载并校验 Linux 7-Zip 命令" });
    },
    onWarning: (message) => emit(report, { phase: "tools", progress: 10, message }),
  });
  const canonicalPortableArchive = await realpath(portableArchive);
  if (!pathsEqual(portableArchive, canonicalPortableArchive) || !isWithin(projectRoot, canonicalPortableArchive)) {
    throw new ArchiveSecurityError("Linux 7-Zip 工具包缓存路径越界");
  }

  // The official Linux package is data only. Rebuild the command directory from the verified archive.
  await rm(portableDirectory, { recursive: true, force: true });
  await ensureSafeDirectory(toolDirectory, "portable");
  const portableSource = tarArchiveLocation(portableArchive);
  await runCommand(tar.command, ["-xf", portableSource.name, "-C", tarDirectoryArgument(portableDirectory)], {
    cwd: portableSource.cwd,
  });
  await inspectExtractedTree(portableDirectory);
  await chmod(portableExecutable, 0o755);
  if (!await commandAvailable(portableExecutable, ["i"])) {
    throw new Error(`官方 Linux 7-Zip 不包含当前架构 ${process.arch} 的命令组件`);
  }
  return { kind: "7zip", command: portableExecutable, downloaded: true };
}

async function ensureDownloadedSevenZip(projectRoot, report) {
  if (process.platform === "win32") return ensureDownloadedWindowsSevenZip(projectRoot, report);
  if (process.platform === "linux") return ensureDownloadedLinuxSevenZip(projectRoot, report);
  throw new Error(`当前系统不支持自动准备 7-Zip：${process.platform}`);
}

function tarMembers(output) {
  return output.split(/\r?\n/).map((line) => line.replace(/\r$/, "")).filter(Boolean);
}

function assertTarHasNoLinks(verboseOutput) {
  for (const line of verboseOutput.split(/\r?\n/)) {
    const listing = line.trimStart();
    if (/^[lh]/.test(listing) || listing.includes(" -> ") || listing.includes(" link to ")) {
      throw new ArchiveSecurityError("更新归档包含符号链接或硬链接");
    }
  }
}

function parseSevenZipListing(output) {
  const blocks = [];
  let fields = {};
  const flush = () => {
    if (Object.keys(fields).length) blocks.push(fields);
    fields = {};
  };

  for (const line of output.split(/\r?\n/)) {
    if (!line.trim()) {
      flush();
      continue;
    }
    const separator = line.indexOf(" = ");
    if (separator < 0) continue;
    const key = line.slice(0, separator);
    if (key === "Path" && fields.Path) flush();
    fields[key] = line.slice(separator + 3);
  }
  flush();

  const members = [];
  for (const block of blocks) {
    if (!block.Path) continue;
    if (block.Type && block.Folder === undefined && block.Attributes === undefined && block.Size === undefined) continue;
    if (block["Symbolic Link"] || block["Hard Link"] || /\bl[rwx-]{9}\b/i.test(block.Attributes || "")) {
      throw new ArchiveSecurityError("更新归档包含符号链接或硬链接");
    }
    members.push(block.Path);
  }
  return members;
}

/** How to name an archive to `tar` so that either Windows implementation opens it.
 *
 * GNU tar reads an archive name containing a colon as `host:path` and tries to reach a remote
 * machine, so an absolute Windows path fails with "Cannot connect to D: resolve failed". Windows
 * ships bsdtar in System32, but Git for Windows also ships GNU tar and puts it earlier on PATH on
 * plenty of developer machines — and on GitHub's Windows runners. `--force-local` is not a
 * portable answer: bsdtar rejects the option outright, and GNU tar built on MSYS still will not
 * take backslashes as separators. Naming the archive relative to a working directory keeps the
 * colon out of the argument, which both implementations accept.
 */
function tarArchiveLocation(archivePath) {
  const resolved = path.resolve(archivePath);
  return { cwd: path.dirname(resolved), name: path.basename(resolved) };
}

/** A `-C` target both implementations accept: absolute, but never backslash-separated. */
function tarDirectoryArgument(directory) {
  return path.resolve(directory).split(path.sep).join("/");
}

async function listArchive(tool, archivePath) {
  if (tool.kind === "tar") {
    const { cwd, name } = tarArchiveLocation(archivePath);
    const [{ stdout }, { stdout: verboseOutput }] = await Promise.all([
      runCommand(tool.command, ["-tf", name], { cwd }),
      runCommand(tool.command, ["-tvf", name], { cwd }),
    ]);
    assertTarHasNoLinks(verboseOutput);
    const members = tarMembers(stdout);
    validateArchiveMembers(members);
    return members;
  }

  const { stdout } = await runCommand(tool.command, ["l", "-slt", "-ba", "-sccUTF-8", archivePath]);
  const members = parseSevenZipListing(stdout);
  validateArchiveMembers(members);
  return members;
}

async function selectArchiveTool(projectRoot, archivePath, kind, report) {
  const [tar, sevenZip] = await Promise.all([findTar(), findSystemSevenZip()]);
  const tarPreferred = kind === "zip" || kind === "tar";
  const preferred = tarPreferred ? [tar, sevenZip] : [sevenZip];
  const errors = [];

  for (const tool of preferred.filter(Boolean)) {
    try {
      const members = await listArchive(tool, archivePath);
      return { tool, members };
    } catch (error) {
      if (error instanceof ArchiveSecurityError) throw error;
      errors.push(error.message);
    }
  }

  if (["win32", "linux"].includes(process.platform)) {
    const downloaded = await ensureDownloadedSevenZip(projectRoot, report);
    try {
      const members = await listArchive(downloaded, archivePath);
      return { tool: downloaded, members };
    } catch (error) {
      if (error instanceof ArchiveSecurityError) throw error;
      errors.push(error.message);
    }
  }

  if (!tarPreferred && tar) {
    try {
      const members = await listArchive(tar, archivePath);
      return { tool: tar, members };
    } catch (error) {
      if (error instanceof ArchiveSecurityError) throw error;
      errors.push(error.message);
    }
  }

  const details = errors.length ? `：${errors.join("；")}` : "";
  throw new Error(`无法读取更新归档，请确认归档完整且系统 tar/7z 可用${details}`);
}

async function extractArchive(tool, archivePath, destination) {
  if (tool.kind === "tar") {
    const { cwd, name } = tarArchiveLocation(archivePath);
    await runCommand(tool.command, ["-xf", name, "-C", tarDirectoryArgument(destination)], { cwd });
    return;
  }
  await runCommand(tool.command, ["x", "-y", "-bd", "-bso0", "-bsp0", `-o${destination}`, archivePath]);
}

async function inspectExtractedTree(root) {
  const rootStats = await lstat(root);
  if (!rootStats.isDirectory() || rootStats.isSymbolicLink()) {
    throw new ArchiveSecurityError("更新暂存目录无效");
  }

  const pending = [{ absolute: root, relative: "" }];
  while (pending.length) {
    const current = pending.pop();
    for (const entry of await readdir(current.absolute, { withFileTypes: true })) {
      const relative = current.relative ? `${current.relative}/${entry.name}` : entry.name;
      validateArchiveMemberPath(relative);
      const absolute = resolveWithin(root, relative);
      const stats = await lstat(absolute);
      if (stats.isSymbolicLink()) throw new ArchiveSecurityError(`更新归档包含符号链接：${relative}`);
      if (stats.isDirectory()) {
        pending.push({ absolute, relative });
      } else if (!stats.isFile()) {
        throw new ArchiveSecurityError(`更新归档包含不支持的特殊文件：${relative}`);
      } else if (stats.nlink > 1) {
        throw new ArchiveSecurityError(`更新归档包含硬链接：${relative}`);
      }
    }
  }
}

async function entryHasType(target, type) {
  const stats = await lstatOrNull(target);
  return Boolean(stats && !stats.isSymbolicLink() && (type === "file" ? stats.isFile() : stats.isDirectory()));
}

async function validProjectRoot(candidate) {
  const requiredFiles = ["package.json", "package-lock.json", "vite.config.js"];
  const requiredDirectories = ["src", "backend", "scripts"];
  const checks = await Promise.all([
    ...requiredFiles.map((item) => entryHasType(path.join(candidate, item), "file")),
    ...requiredDirectories.map((item) => entryHasType(path.join(candidate, item), "directory")),
  ]);
  if (checks.some((result) => !result)) return false;
  try {
    const packageJson = JSON.parse(await readFile(path.join(candidate, "package.json"), "utf8"));
    return packageJson.name === "xiriacanvas-ai";
  } catch {
    return false;
  }
}

async function identifyProjectRoot(stagingDirectory) {
  if (await validProjectRoot(stagingDirectory)) return stagingDirectory;
  const entries = (await readdir(stagingDirectory, { withFileTypes: true }))
    .filter((entry) => entry.name !== OFFLINE_UPDATE_OWNER_FILE);
  if (entries.length === 1 && entries[0].isDirectory() && !entries[0].isSymbolicLink()) {
    const wrappedRoot = path.join(stagingDirectory, entries[0].name);
    if (await validProjectRoot(wrappedRoot)) return wrappedRoot;
  }
  throw new Error("更新归档不是有效的 XiriaCanvas AI 项目，或包含超过一层包裹目录");
}

function normalizeDependencyMap(value) {
  if (value === undefined) return { present: false, entries: {} };
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return {
    present: true,
    entries: Object.fromEntries(Object.entries(value).sort(([first], [second]) => first.localeCompare(second))),
  };
}

function canonicalize(value) {
  if (Array.isArray(value)) return value.map(canonicalize);
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(Object.entries(value)
    .sort(([first], [second]) => first.localeCompare(second))
    .map(([key, item]) => [key, canonicalize(item)]));
}

function lockEnvironment(lock) {
  const root = lock?.packages?.[""];
  if (!root || typeof root !== "object" || Array.isArray(root)) return null;
  const rootEnvironment = {};
  for (const key of ["dependencies", "devDependencies", "optionalDependencies", "peerDependencies", "peerDependenciesMeta", "engines"]) {
    const normalized = normalizeDependencyMap(root[key]);
    if (!normalized) return null;
    rootEnvironment[key] = normalized;
  }
  const packages = Object.fromEntries(Object.entries(lock.packages || {})
    .filter(([key]) => key !== "")
    .sort(([first], [second]) => first.localeCompare(second))
    .map(([key, value]) => [key, canonicalize(value)]));
  return canonicalize({
    lockfileVersion: lock.lockfileVersion ?? null,
    root: rootEnvironment,
    packages,
    dependencies: lock.dependencies || {},
  });
}

function normalizeRequirements(contents) {
  return contents.split(/\r?\n/).map((line) => {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) return "";
    return trimmed.replace(/\s+#.*$/, "").trim();
  }).filter(Boolean).sort();
}

function dependencyEnvironmentEqual(oldLock, newLock, oldRequirements, newRequirements) {
  const oldEnvironment = lockEnvironment(oldLock);
  const newEnvironment = lockEnvironment(newLock);
  if (!oldEnvironment || !newEnvironment) return false;
  return JSON.stringify(oldEnvironment) === JSON.stringify(newEnvironment)
    && JSON.stringify(normalizeRequirements(oldRequirements)) === JSON.stringify(normalizeRequirements(newRequirements));
}

async function dependencyChangesRequired(projectRoot, packageRoot) {
  try {
    const [oldLockText, newLockText, oldRequirements, newRequirements] = await Promise.all([
      readFile(path.join(projectRoot, "package-lock.json"), "utf8"),
      readFile(path.join(packageRoot, "package-lock.json"), "utf8"),
      readFile(path.join(projectRoot, "backend", "requirements.txt"), "utf8"),
      readFile(path.join(packageRoot, "backend", "requirements.txt"), "utf8"),
    ]);
    const oldEnvironment = lockEnvironment(JSON.parse(oldLockText));
    const newEnvironment = lockEnvironment(JSON.parse(newLockText));
    if (!oldEnvironment || !newEnvironment) throw new Error(NODE_DEPENDENCY_ERROR);
    if (JSON.stringify(oldEnvironment) !== JSON.stringify(newEnvironment)) throw new Error(NODE_DEPENDENCY_ERROR);
    return JSON.stringify(normalizeRequirements(oldRequirements)) !== JSON.stringify(normalizeRequirements(newRequirements));
  } catch (error) {
    if (error.message === NODE_DEPENDENCY_ERROR) throw error;
    throw new Error(NODE_DEPENDENCY_ERROR);
  }
}

async function createUpdatePlan(projectRoot, packageRoot) {
  const plan = [];
  for (const relativePath of MANAGED_DIRECTORIES) {
    const sourceStats = await lstatOrNull(resolveWithin(packageRoot, relativePath));
    if (sourceStats) {
      if (!sourceStats.isDirectory() || sourceStats.isSymbolicLink()) {
        throw new ArchiveSecurityError(`更新包中的 ${relativePath} 不是安全目录`);
      }
      plan.push({ relativePath, kind: "directory", action: "replace" });
    } else if (await lstatOrNull(resolveWithin(projectRoot, relativePath))) {
      plan.push({ relativePath, kind: "directory", action: "remove" });
    }
  }

  for (const relativePath of MANAGED_FILES) {
    const sourceStats = await lstatOrNull(resolveWithin(packageRoot, relativePath));
    if (sourceStats) {
      if (!sourceStats.isFile() || sourceStats.isSymbolicLink()) {
        throw new ArchiveSecurityError(`更新包中的 ${relativePath} 不是安全文件`);
      }
      plan.push({ relativePath, kind: "file", action: "replace" });
    } else if (await lstatOrNull(resolveWithin(projectRoot, relativePath))) {
      plan.push({ relativePath, kind: "file", action: "remove" });
    }
  }

  for (const item of plan) {
    const topLevel = item.relativePath.replace(/\\/g, "/").split("/")[0];
    if (FORBIDDEN_TOP_LEVEL_ITEMS.has(topLevel) && !MANAGED_MODEL_MANIFESTS.has(item.relativePath.replace(/\\/g, "/"))) {
      throw new ArchiveSecurityError(`更新计划试图修改受保护项目：${topLevel}`);
    }
  }
  return plan;
}

async function canonicalProjectRoot(projectRoot) {
  if (!projectRoot) throw new Error("缺少项目根目录");
  const canonical = await realpath(path.resolve(projectRoot));
  const stats = await lstat(canonical);
  if (!stats.isDirectory()) throw new Error("项目根目录无效");
  return canonical;
}

export async function prepareUpdate({ projectRoot, archivePath, report } = {}) {
  const root = await canonicalProjectRoot(projectRoot);
  if (!archivePath || !archiveExtensionAllowed(archivePath)) {
    throw new Error("不支持的更新归档格式");
  }

  const archiveInput = path.resolve(archivePath);
  const archiveStats = await lstatOrNull(archiveInput);
  if (!archiveStats || !archiveStats.isFile() || archiveStats.isSymbolicLink()) {
    throw new Error("更新归档不存在或不是普通文件");
  }
  const canonicalArchive = await realpath(archiveInput);
  const initialArchiveIdentity = await fileIdentity(canonicalArchive);
  const kind = archiveKind(archivePath);
  let stagingDirectory;
  let stagingOwnership;

  emit(report, { phase: "inspect", progress: 0, message: "正在检查更新归档" });
  try {
    const selected = await selectArchiveTool(root, canonicalArchive, kind, report);
    const releaseWrapper = releaseWrapperFromMembers(selected.members);
    if (!sameFileIdentity(initialArchiveIdentity, await fileIdentity(canonicalArchive))) {
      throw new ArchiveSecurityError("更新归档在安全检查期间发生变化，请重新选择文件");
    }
    emit(report, {
      phase: "inspect",
      progress: 20,
      message: `已安全检查 ${selected.members.length} 个归档项目`,
      memberCount: selected.members.length,
      tool: selected.tool.kind,
    });

    stagingOwnership = await createOfflineUpdateTemp({ projectRoot: root, prefix: "xirai-update-stage-", kind: "stage" });
    stagingDirectory = stagingOwnership.path;
    const extractionDirectory = path.join(stagingDirectory, "contents");
    await mkdir(extractionDirectory);
    emit(report, { phase: "extract", progress: 30, message: "正在项目目录外解压更新包" });
    await extractArchive(selected.tool, canonicalArchive, extractionDirectory);
    await assertOfflineUpdateTemp({ projectRoot: root, record: stagingOwnership, prefix: "xirai-update-stage-", kind: "stage" });
    if (!sameFileIdentity(initialArchiveIdentity, await fileIdentity(canonicalArchive))) {
      throw new ArchiveSecurityError("更新归档在解压期间发生变化，请重新选择文件");
    }

    emit(report, { phase: "validate", progress: 55, message: "正在校验解压结果" });
    await inspectExtractedTree(extractionDirectory);
    const packageRoot = await identifyProjectRoot(extractionDirectory);
    const releaseManifest = releaseWrapper ? await validateReleasePackageDirectory({ packageRoot }) : null;

    emit(report, { phase: "dependencies", progress: 72, message: "正在比较新旧环境依赖" });
    const environmentRepairRequired = await dependencyChangesRequired(root, packageRoot);

    emit(report, { phase: "plan", progress: 88, message: "正在生成安全更新计划" });
    const plan = await createUpdatePlan(root, packageRoot);
    const prepared = {
      schema: 1,
      projectRoot: root,
      archivePath: canonicalArchive,
      archiveIdentity: initialArchiveIdentity,
      stagingDirectory,
      stagingOwnership,
      packageRoot,
      archiveTool: { kind: selected.tool.kind, command: selected.tool.command },
      environmentRepairRequired,
      releaseManifest,
      plan,
    };
    emit(report, {
      phase: "ready",
      progress: 100,
      message: environmentRepairRequired ? "更新包已准备完成；替换后将自动修复环境" : "更新包已准备完成",
      plan,
      environmentRepairRequired,
    });
    return prepared;
  } catch (error) {
    if (stagingOwnership) {
      await removeOfflineUpdateTemp({ projectRoot: root, record: stagingOwnership, prefix: "xirai-update-stage-", kind: "stage" }).catch(() => {});
    }
    throw error;
  }
}

async function copyEntry(source, destination) {
  const stats = await lstat(source);
  await mkdir(path.dirname(destination), { recursive: true });
  await cp(source, destination, {
    recursive: stats.isDirectory(),
    dereference: false,
    errorOnExist: true,
    force: false,
    preserveTimestamps: true,
    verbatimSymlinks: true,
  });
}

async function writeTransactionJournal(transactionPath, transaction) {
  if (!transactionPath) return;
  await mkdir(path.dirname(transactionPath), { recursive: true });
  const temporaryPath = `${transactionPath}.${process.pid}.tmp`;
  try {
    await writeFile(temporaryPath, `${JSON.stringify(transaction, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
    await rename(temporaryPath, transactionPath);
  } finally {
    await rm(temporaryPath, { force: true });
  }
}

async function assertSafeManagedTarget(root, relativePath) {
  const segments = relativePath.replace(/\\/g, "/").split("/");
  let current = root;
  for (const segment of segments) {
    current = path.join(current, segment);
    const stats = await lstatOrNull(current);
    if (!stats) break;
    if (stats.isSymbolicLink()) throw new ArchiveSecurityError(`受管路径不能经过符号链接或目录联接：${relativePath}`);
    const canonical = await realpath(current);
    if (!isWithin(root, canonical, true)) throw new ArchiveSecurityError(`受管路径越界：${relativePath}`);
  }
}

async function validatePrepared(root, prepared) {
  if (!prepared || prepared.schema !== 1) throw new Error("更新暂存信息无效");
  if (!pathsEqual(root, prepared.projectRoot)) throw new Error("更新暂存信息不属于当前项目");

  const stagingDirectory = path.resolve(prepared.stagingDirectory || "");
  const packageRoot = path.resolve(prepared.packageRoot || "");
  if (!pathsEqual(prepared.stagingOwnership?.path, stagingDirectory)) throw new ArchiveSecurityError("更新暂存目录所有权路径不匹配");
  await assertOfflineUpdateTemp({ projectRoot: root, record: prepared.stagingOwnership, prefix: "xirai-update-stage-", kind: "stage" });
  if (isWithin(root, stagingDirectory, true) || !isWithin(stagingDirectory, packageRoot, true)) {
    throw new ArchiveSecurityError("更新暂存目录越界");
  }
  const [stagingStats, packageStats] = await Promise.all([lstat(stagingDirectory), lstat(packageRoot)]);
  if (!stagingStats.isDirectory() || stagingStats.isSymbolicLink()
    || !packageStats.isDirectory() || packageStats.isSymbolicLink()) {
    throw new ArchiveSecurityError("更新暂存目录无效");
  }
  const [canonicalStaging, canonicalPackage] = await Promise.all([
    realpath(stagingDirectory),
    realpath(packageRoot),
  ]);
  if (!pathsEqual(stagingDirectory, canonicalStaging)
    || !pathsEqual(packageRoot, canonicalPackage)
    || !isWithin(canonicalStaging, canonicalPackage, true)) {
    throw new ArchiveSecurityError("更新暂存目录包含不安全的路径重定向");
  }

  await inspectExtractedTree(packageRoot);
  if (!await validProjectRoot(packageRoot)) throw new Error("更新暂存项目根已失效");
  if (prepared.releaseManifest) await validateReleasePackageDirectory({ packageRoot });
  const environmentRepairRequired = await dependencyChangesRequired(root, packageRoot);
  return {
    stagingDirectory,
    stagingOwnership: prepared.stagingOwnership,
    packageRoot,
    archivePath: path.resolve(prepared.archivePath || ""),
    archiveIdentity: prepared.archiveIdentity,
    environmentRepairRequired,
    plan: await createUpdatePlan(root, packageRoot),
  };
}

async function backUpPlan(root, backupRoot, plan) {
  const manifest = [];
  for (const item of plan) {
    await assertSafeManagedTarget(root, item.relativePath);
    const source = resolveWithin(root, item.relativePath);
    const stats = await lstatOrNull(source);
    const record = { ...item, existed: Boolean(stats) };
    if (stats) await copyEntry(source, resolveWithin(backupRoot, item.relativePath));
    manifest.push(record);
  }
  return manifest;
}

async function restoreBackup(root, backupRoot, manifest) {
  const errors = [];
  for (const item of [...manifest].reverse()) {
    const target = resolveWithin(root, item.relativePath);
    try {
      await assertSafeManagedTarget(root, item.relativePath);
      await rm(target, { recursive: true, force: true });
      if (item.existed) {
        const source = resolveWithin(backupRoot, item.relativePath);
        const sourceStats = await lstat(source);
        if (sourceStats.isSymbolicLink()
          || (item.kind === "file" ? !sourceStats.isFile() : !sourceStats.isDirectory())) {
          throw new ArchiveSecurityError(`回滚备份中的 ${item.relativePath} 类型无效`);
        }
        await copyEntry(source, target);
      }
    } catch (error) {
      errors.push(`${item.relativePath}: ${error.message}`);
    }
  }
  if (errors.length) throw new Error(errors.join("；"));
}

function validateRecoveryManifest(manifest) {
  if (!Array.isArray(manifest)) throw new Error("离线更新恢复清单无效");
  const seen = new Set();
  return manifest.map((item) => {
    const expectedKind = MANAGED_ITEM_KINDS.get(item?.relativePath);
    if (!expectedKind || item.kind !== expectedKind || !["replace", "remove"].includes(item.action)
      || typeof item.existed !== "boolean" || seen.has(item.relativePath)) {
      throw new Error("离线更新恢复清单包含无效或重复项目");
    }
    seen.add(item.relativePath);
    return {
      relativePath: item.relativePath,
      kind: item.kind,
      action: item.action,
      existed: item.existed,
    };
  });
}

export async function applyPreparedUpdate({ projectRoot, prepared, report, validate, transactionPath } = {}) {
  const root = await canonicalProjectRoot(projectRoot);
  transactionPath = transactionPath ? path.resolve(transactionPath) : null;
  if (transactionPath && typeof validate !== "function") {
    throw new Error("离线更新事务必须提供完整的应用后验证");
  }
  emit(report, { phase: "validate", progress: 0, message: "正在重新校验更新暂存" });
  const validated = await validatePrepared(root, prepared);
  const backupOwnership = await createOfflineUpdateTemp({ projectRoot: root, prefix: "xirai-update-backup-", kind: "backup" });
  const backupRoot = backupOwnership.path;
  let manifest;
  let validation;
  let transaction;

  try {
    emit(report, { phase: "backup", progress: 10, message: "正在项目目录外创建回滚备份" });
    manifest = await backUpPlan(root, backupRoot, validated.plan);
    transaction = {
      schema: 1,
      product: "XiriaCanvas AI",
      projectRoot: root,
      ownerPid: process.pid,
      phase: "applying",
      backupRoot,
      backupOwnership,
      manifest,
      environmentBackup: null,
      setupMarker: readSetupMarker(root),
      createdAt: new Date().toISOString(),
    };
    await writeTransactionJournal(transactionPath, transaction);

    const count = Math.max(validated.plan.length, 1);
    for (let index = 0; index < validated.plan.length; index += 1) {
      const item = validated.plan[index];
      await assertSafeManagedTarget(root, item.relativePath);
      const target = resolveWithin(root, item.relativePath);
      emit(report, {
        phase: "apply",
        progress: 20 + (index / count) * 70,
        message: `${item.action === "remove" ? "正在移除" : "正在更新"} ${item.relativePath}`,
        item,
      });
      await rm(target, { recursive: true, force: true });
      if (item.action === "replace") {
        await copyEntry(resolveWithin(validated.packageRoot, item.relativePath), target);
      }
    }
    if (validate) {
      emit(report, { phase: "verify", progress: 92, message: "正在验证更新后的项目" });
      validation = await validate({
        projectRoot: root,
        plan: validated.plan,
        environmentRepairRequired: validated.environmentRepairRequired,
        setEnvironmentBackup: async (environmentBackup) => {
          transaction.environmentBackup = environmentBackup || null;
          await writeTransactionJournal(transactionPath, transaction);
        },
      });
      if (validation?.verified !== true) throw new Error("更新后的环境与后端未返回有效验证结果");
    }
  } catch (error) {
    let environmentRollbackError;
    if (validation?.rollback) {
      try {
        await validation.rollback();
      } catch (rollbackError) {
        environmentRollbackError = rollbackError;
      }
    }
    if (!manifest) {
      await removeOfflineUpdateTemp({ projectRoot: root, record: backupOwnership, prefix: "xirai-update-backup-", kind: "backup" }).catch(() => {});
      throw error;
    }
    emit(report, { phase: "rollback", progress: 90, message: "应用失败，正在自动回滚" });
    try {
      await restoreBackup(root, backupRoot, manifest);
    } catch (rollbackError) {
      throw rollbackIncompleteError(`更新应用失败且自动回滚不完整，备份保留在 ${backupRoot}：${rollbackError.message}`, error);
    }
    if (environmentRollbackError) {
      throw rollbackIncompleteError(`更新应用失败；程序文件已回滚，但 Python 环境回滚不完整，恢复日志与备份已保留：${environmentRollbackError.message}`, error);
    }
    if (error.rollbackIncomplete) {
      throw rollbackIncompleteError(`更新应用失败且 Python 环境回滚不完整，恢复日志与备份已保留：${error.message}`, error);
    }
    if (transactionPath) {
      transaction.phase = "rolled-back";
      await writeTransactionJournal(transactionPath, transaction);
    }
    let backupRemoved = true;
    await removeOfflineUpdateTemp({ projectRoot: root, record: backupOwnership, prefix: "xirai-update-backup-", kind: "backup" })
      .catch(() => { backupRemoved = false; });
    if (transactionPath && backupRemoved) await rm(transactionPath, { force: true }).catch(() => {});
    throw new Error(`更新应用失败，已自动回滚：${error.message}`, { cause: error });
  }

  if (transactionPath) {
    transaction.phase = "committed";
    await writeTransactionJournal(transactionPath, transaction);
  }
  let environmentCleanupWarning;
  if (validation?.commit) {
    try {
      await validation.commit();
    } catch (error) {
      environmentCleanupWarning = `${validation.environmentBackup?.path || validation.environmentBackup || ".venv update backup"}: ${error.message}`;
    }
  }

  emit(report, { phase: "cleanup", progress: 95, message: "更新成功，正在清理临时文件" });
  const cleanupTargets = [backupRoot, validated.stagingDirectory];
  const cleanupResults = await Promise.allSettled([
    removeOfflineUpdateTemp({ projectRoot: root, record: backupOwnership, prefix: "xirai-update-backup-", kind: "backup" }),
    removeOfflineUpdateTemp({ projectRoot: root, record: validated.stagingOwnership, prefix: "xirai-update-stage-", kind: "stage" }),
  ]);
  const cleanupWarnings = cleanupResults.flatMap((result, index) => result.status === "rejected"
    ? [`${cleanupTargets[index]}: ${result.reason.message}`]
    : []);
  if (environmentCleanupWarning) cleanupWarnings.push(environmentCleanupWarning);
  if (transactionPath && !environmentCleanupWarning && cleanupResults[0].status !== "rejected") {
    await rm(transactionPath, { force: true }).catch((error) => cleanupWarnings.push(`${transactionPath}: ${error.message}`));
  }
  if (validated.archivePath) {
    try {
      const currentIdentity = await fileIdentity(validated.archivePath).catch((error) => {
        if (error.code === "ENOENT") return null;
        throw error;
      });
      if (sameFileIdentity(validated.archiveIdentity, currentIdentity)
        && archiveExtensionAllowed(validated.archivePath)) {
        await rm(validated.archivePath, { force: true });
      }
    } catch (error) {
      cleanupWarnings.push(`${validated.archivePath}: ${error.message}`);
    }
  }

  emit(report, {
    phase: "complete",
    progress: 100,
    message: cleanupWarnings.length ? "更新已完成，但部分临时文件未能清理" : "更新已安全应用",
    cleanupWarnings,
  });
  return {
    applied: true,
    plan: validated.plan,
    cleanupWarnings,
    environmentRepairRequired: validated.environmentRepairRequired,
    validation,
  };
}

export async function recoverInterruptedUpdate({ projectRoot, transactionPath, report, lock } = {}) {
  const root = await canonicalProjectRoot(projectRoot);
  assertOfflineUpdateLock(lock, root);
  if (!transactionPath) return { recovered: false };
  const journalPath = path.resolve(transactionPath);
  let transaction;
  try {
    transaction = JSON.parse(await readFile(journalPath, "utf8"));
  } catch (error) {
    if (error.code === "ENOENT") return { recovered: false };
    throw new Error(`无法读取离线更新恢复日志：${error.message}`);
  }
  if (transaction?.schema !== 1 || transaction.product !== "XiriaCanvas AI" || !pathsEqual(transaction.projectRoot, root)) {
    throw new Error("离线更新恢复日志不属于当前项目");
  }
  if (["environment-repair", "environment-committed"].includes(transaction.phase)) {
    const environmentBackup = transaction.environmentBackup;
    if (!environmentBackup) throw new Error("离线更新环境恢复日志缺少 Python 备份");
    await assertEnvironmentBackup({ projectRoot: root, record: environmentBackup, requireExists: false });
    if (transaction.phase === "environment-committed") {
      const cleanupWarnings = [];
      await removeEnvironmentBackup({ projectRoot: root, record: environmentBackup })
        .catch((error) => cleanupWarnings.push(`${environmentBackup.path}: ${error.message}`));
      if (!cleanupWarnings.length) await rm(journalPath, { force: true });
      return {
        recovered: true,
        rolledBack: false,
        ...(cleanupWarnings.length ? { cleanupWarnings } : {}),
      };
    } else {
      const venvRoot = path.join(root, ".venv");
      const restored = await restoreEnvironmentBackup({ projectRoot: root, record: environmentBackup, venvRoot });
      if (!restored) await removeEnvironmentOwnershipMarker({ projectRoot: root, record: environmentBackup, directory: venvRoot });
      if (transaction.setupMarker) writeSetupMarker(root, transaction.setupMarker);
      else await rm(getSetupMarkerPath(root), { force: true });
    }
    await rm(journalPath, { force: true });
    return { recovered: true, rolledBack: true };
  }
  const manifest = validateRecoveryManifest(transaction.manifest);
  const backupRoot = path.resolve(transaction.backupRoot || "");
  if (!pathsEqual(transaction.backupOwnership?.path, backupRoot)) throw new Error("离线更新程序备份所有权路径不匹配");
  await assertOfflineUpdateTemp({
    projectRoot: root,
    record: transaction.backupOwnership,
    prefix: "xirai-update-backup-",
    kind: "backup",
    requireExists: !["committed", "rolled-back"].includes(transaction.phase),
  });
  const environmentBackup = transaction.environmentBackup || null;
  if (environmentBackup) await assertEnvironmentBackup({ projectRoot: root, record: environmentBackup, requireExists: false });
  if (transaction.phase === "committed") {
    report?.("检测到已提交更新，正在清理残留备份");
    const cleanupWarnings = [];
    await removeOfflineUpdateTemp({ projectRoot: root, record: transaction.backupOwnership, prefix: "xirai-update-backup-", kind: "backup" })
      .catch((error) => cleanupWarnings.push(`${backupRoot}: ${error.message}`));
    if (environmentBackup) {
      await removeEnvironmentBackup({ projectRoot: root, record: environmentBackup })
        .catch((error) => cleanupWarnings.push(`${environmentBackup.path}: ${error.message}`));
    }
    if (!cleanupWarnings.length) await rm(journalPath, { force: true });
    return {
      recovered: true,
      rolledBack: false,
      ...(cleanupWarnings.length ? { cleanupWarnings } : {}),
    };
  }
  if (transaction.phase === "rolled-back") {
    const cleanupWarnings = [];
    await removeOfflineUpdateTemp({ projectRoot: root, record: transaction.backupOwnership, prefix: "xirai-update-backup-", kind: "backup" })
      .catch((error) => cleanupWarnings.push(`${backupRoot}: ${error.message}`));
    if (!cleanupWarnings.length) await rm(journalPath, { force: true });
    return {
      recovered: true,
      rolledBack: true,
      ...(cleanupWarnings.length ? { cleanupWarnings } : {}),
    };
  }
  if (transaction.phase !== "applying") {
    throw new Error("离线更新恢复日志状态无效");
  }
  report?.("检测到未完成的离线更新，正在自动回滚");
  await restoreBackup(root, backupRoot, manifest);
  if (environmentBackup) {
    const venvRoot = path.join(root, ".venv");
    const restored = await restoreEnvironmentBackup({ projectRoot: root, record: environmentBackup, venvRoot });
    if (!restored) await removeEnvironmentOwnershipMarker({ projectRoot: root, record: environmentBackup, directory: venvRoot });
  }
  if (transaction.setupMarker) {
    writeSetupMarker(root, transaction.setupMarker);
  } else {
    await rm(getSetupMarkerPath(root), { force: true });
  }
  transaction.phase = "rolled-back";
  await writeTransactionJournal(journalPath, transaction);
  const cleanupWarnings = [];
  await removeOfflineUpdateTemp({ projectRoot: root, record: transaction.backupOwnership, prefix: "xirai-update-backup-", kind: "backup" })
    .catch((error) => cleanupWarnings.push(`${backupRoot}: ${error.message}`));
  if (!cleanupWarnings.length) await rm(journalPath, { force: true });
  return {
    recovered: true,
    rolledBack: true,
    ...(cleanupWarnings.length ? { cleanupWarnings } : {}),
  };
}

export const archiveUpdateInternals = {
  FORBIDDEN_TOP_LEVEL_ITEMS,
  MANAGED_FILES,
  MANAGED_DIRECTORIES,
  createUpdatePlan,
  dependencyChangesRequired,
  dependencyEnvironmentEqual,
  identifyProjectRoot,
  managedSevenZipPackage,
  normalizeRequirements,
  parseSevenZipListing,
  validateArchiveMemberPath,
  validateArchiveMembers,
};
