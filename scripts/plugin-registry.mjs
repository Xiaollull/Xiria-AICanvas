import { createHash } from "node:crypto";
import { constants as fsConstants } from "node:fs";
import { lstat, open, readdir, realpath } from "node:fs/promises";
import path from "node:path";

// Discovery-only plugin port.
//
// This module reads and validates `plugins/<id>/plugin.json` and nothing else. It never loads,
// executes, serves, stats, or opens plugin program files: the Node control plane and the FastAPI
// data plane both run with the user's full privileges (GPU, model weights, outputs, shell), so
// loading plugin code here would be arbitrary code execution rather than an extension API.
// Everything below exists to stabilise the directory, manifest, and path-safety contracts that a
// future sandboxed execution host would have to satisfy before any plugin code may run.

export const PLUGIN_HOST_API_VERSION = 1;
export const PLUGIN_MANIFEST_SCHEMA_VERSION = 1;
export const PLUGIN_REGISTRY_SCHEMA_VERSION = 1;
export const PLUGIN_EXECUTION_SUPPORT = "not-supported";
export const PLUGIN_DIRECTORY_NAME = "plugins";
export const PLUGIN_MANIFEST_FILENAME = "plugin.json";
export const PLUGIN_MANIFEST_MAXIMUM_BYTES = 64 * 1024;

const PLUGIN_ID_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const PLUGIN_ID_MINIMUM_LENGTH = 3;
const PLUGIN_ID_MAXIMUM_LENGTH = 64;
const SEMVER_PATTERN = /^(?:0|[1-9]\d{0,8})\.(?:0|[1-9]\d{0,8})\.(?:0|[1-9]\d{0,8})$/;
const WINDOWS_RESERVED_NAME_PATTERN = /^(?:con|prn|aux|nul|com[0-9]|lpt[0-9])$/;
const DISPLAY_NAME_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
const PERMISSION_PATTERN = /^[a-z0-9]+(?:[-.:][a-z0-9]+)*$/;
const HOST_API_BOUND_MAXIMUM = 1000;
const MAXIMUM_ENTRYPOINT_LENGTH = 200;
const MAXIMUM_PERMISSIONS = 16;
const MAXIMUM_CONTRIBUTIONS = 16;

export const PLUGIN_STATES = Object.freeze(["discovered", "invalid", "incompatible", "blocked"]);

// Stable machine-readable codes. Never widen these into free text: diagnostics are returned to the
// browser and must not carry absolute paths, manifest contents, or stack traces.
export const PLUGIN_DIAGNOSTIC_CODES = Object.freeze([
  "plugins_root_unavailable",
  "plugins_root_unsafe",
  "invalid_plugin_id",
  "duplicate_id",
  "manifest_missing",
  "manifest_unreadable",
  "manifest_too_large",
  "manifest_not_utf8",
  "manifest_not_json",
  "manifest_changed_during_read",
  "unsafe_reparse_point",
  "invalid_manifest",
  "id_folder_mismatch",
  "unsupported_schema_version",
  "invalid_version",
  "invalid_entrypoint",
  "host_api_incompatible",
  "permissions_not_supported",
  "plugin_state_unreadable",
  "plugin_unavailable",
]);

class PluginDiagnosticError extends Error {
  constructor(code) {
    super(code);
    this.name = "PluginDiagnosticError";
    this.code = code;
  }
}

function fail(code) {
  throw new PluginDiagnosticError(code);
}

function ownProperty(value, key) {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function plainObject(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function assertExactKeys(value, allowed) {
  if (!plainObject(value)) fail("invalid_manifest");
  for (const key of Object.keys(value)) {
    // `JSON.parse` exposes `__proto__` as an own key, so this allow-list also fences prototype
    // pollution and any future field that tries to smuggle in executable semantics.
    if (!allowed.has(key)) fail("invalid_manifest");
  }
}

function boundedText(value, { minimum, maximum }) {
  if (typeof value !== "string") fail("invalid_manifest");
  // Control characters would corrupt terminal and JSON diagnostics downstream.
  if (/[\0-\x1f\x7f]/.test(value)) fail("invalid_manifest");
  if (value.length < minimum || value.length > maximum) fail("invalid_manifest");
  return value;
}

export function validPluginId(value) {
  if (typeof value !== "string") return false;
  if (value.length < PLUGIN_ID_MINIMUM_LENGTH || value.length > PLUGIN_ID_MAXIMUM_LENGTH) return false;
  if (!PLUGIN_ID_PATTERN.test(value)) return false;
  // A folder called `nul` or `com1` cannot be created or opened reliably on Windows.
  return !WINDOWS_RESERVED_NAME_PATTERN.test(value);
}

/**
 * A bounded ASCII label for a folder that cannot be reported under its own name. Folder names are
 * not secrets, but the `id` field is an ASCII-bounded contract, so anything outside that shape is
 * reported through a stable digest instead of being echoed back verbatim.
 */
export function displayFolderName(value) {
  if (typeof value === "string" && DISPLAY_NAME_PATTERN.test(value)) return value;
  const digest = createHash("sha256").update(typeof value === "string" ? value : "", "utf8").digest("hex");
  return `unsupported-${digest.slice(0, 12)}`;
}

export function validPluginVersion(value) {
  return typeof value === "string" && SEMVER_PATTERN.test(value);
}

export function hostApiCompatible(range, hostVersion = PLUGIN_HOST_API_VERSION) {
  if (!plainObject(range)) return false;
  const { min, max } = range;
  if (!Number.isInteger(min) || !Number.isInteger(max)) return false;
  if (min < 1 || max < min || max > HOST_API_BOUND_MAXIMUM) return false;
  return hostVersion >= min && hostVersion <= max;
}

// Entrypoints are validated metadata. The host records them so a future sandboxed loader has a
// stable contract to read, and deliberately never stats or opens them in this version.
export function validEntrypointPath(value) {
  if (typeof value !== "string") return false;
  if (!value || value.length > MAXIMUM_ENTRYPOINT_LENGTH) return false;
  if (/[\0-\x1f\x7f]/.test(value)) return false;
  if (value.includes("\\")) return false;
  if (value.includes(":")) return false;
  if (value.startsWith("/")) return false;
  if (/^[A-Za-z]:/.test(value)) return false;
  const segments = value.split("/");
  if (segments.some((segment) => segment === "" || segment === "." || segment === "..")) return false;
  if (segments.some((segment) => /[ .]$/.test(segment))) return false;
  return !segments.some((segment) => WINDOWS_RESERVED_NAME_PATTERN.test(segment.split(".")[0].toLowerCase()));
}

function parseDeveloper(value) {
  assertExactKeys(value, new Set(["name", "homepage"]));
  const developer = { name: boundedText(value.name, { minimum: 1, maximum: 64 }) };
  if (!ownProperty(value, "homepage")) return developer;
  const homepage = boundedText(value.homepage, { minimum: 1, maximum: 2048 });
  let parsed;
  try {
    parsed = new URL(homepage);
  } catch {
    fail("invalid_manifest");
  }
  // `https:` only, no embedded credentials. The host never fetches this value; the restriction
  // exists so the UI can never be talked into rendering a `javascript:` or `file:` link.
  if (parsed.protocol !== "https:" || !parsed.hostname || parsed.username || parsed.password) fail("invalid_manifest");
  developer.homepage = parsed.href;
  return developer;
}

function parseContributionList(value) {
  if (!Array.isArray(value) || value.length > MAXIMUM_CONTRIBUTIONS) fail("invalid_manifest");
  const seen = new Set();
  return value.map((entry) => {
    assertExactKeys(entry, new Set(["id", "title"]));
    const id = boundedText(entry.id, { minimum: 1, maximum: 64 });
    if (!PLUGIN_ID_PATTERN.test(id)) fail("invalid_manifest");
    if (seen.has(id)) fail("invalid_manifest");
    seen.add(id);
    return { id, title: boundedText(entry.title, { minimum: 1, maximum: 64 }) };
  });
}

function parseContributes(value) {
  assertExactKeys(value, new Set(["panels", "commands"]));
  return {
    panels: ownProperty(value, "panels") ? parseContributionList(value.panels) : [],
    commands: ownProperty(value, "commands") ? parseContributionList(value.commands) : [],
  };
}

function parseEntrypoints(value) {
  assertExactKeys(value, new Set(["frontend", "backend"]));
  const entrypoints = {};
  for (const key of ["frontend", "backend"]) {
    if (!ownProperty(value, key)) continue;
    if (!validEntrypointPath(value[key])) fail("invalid_entrypoint");
    entrypoints[key] = value[key];
  }
  return entrypoints;
}

function parsePermissions(value) {
  if (!Array.isArray(value) || value.length > MAXIMUM_PERMISSIONS) fail("invalid_manifest");
  return value.map((entry) => {
    const permission = boundedText(entry, { minimum: 1, maximum: 64 });
    if (!PERMISSION_PATTERN.test(permission)) fail("invalid_manifest");
    return permission;
  });
}

/**
 * Parse and validate one manifest document against schema v1. Throws `PluginDiagnosticError` with a
 * stable code. The result contains only fields this host understands; nothing is passed through.
 */
export function parsePluginManifest(document, folderName) {
  assertExactKeys(document, new Set([
    "schemaVersion",
    "id",
    "version",
    "hostApi",
    "permissions",
    "name",
    "description",
    "developer",
    "entrypoints",
    "contributes",
  ]));

  for (const key of ["schemaVersion", "id", "version", "hostApi", "permissions"]) {
    if (!ownProperty(document, key)) fail("invalid_manifest");
  }

  if (!Number.isInteger(document.schemaVersion)) fail("invalid_manifest");
  if (document.schemaVersion !== PLUGIN_MANIFEST_SCHEMA_VERSION) fail("unsupported_schema_version");

  if (!validPluginId(document.id)) fail("invalid_manifest");
  if (document.id !== folderName) fail("id_folder_mismatch");
  if (!validPluginVersion(document.version)) fail("invalid_version");

  assertExactKeys(document.hostApi, new Set(["min", "max"]));
  if (!ownProperty(document.hostApi, "min") || !ownProperty(document.hostApi, "max")) fail("invalid_manifest");
  const { min, max } = document.hostApi;
  if (!Number.isInteger(min) || !Number.isInteger(max)) fail("invalid_manifest");
  if (min < 1 || max < min || max > HOST_API_BOUND_MAXIMUM) fail("invalid_manifest");

  const permissions = parsePermissions(document.permissions);

  return {
    schemaVersion: document.schemaVersion,
    id: document.id,
    version: document.version,
    hostApi: { min, max },
    permissions,
    name: ownProperty(document, "name") ? boundedText(document.name, { minimum: 1, maximum: 64 }) : document.id,
    description: ownProperty(document, "description") ? boundedText(document.description, { minimum: 0, maximum: 280 }) : "",
    developer: ownProperty(document, "developer") ? parseDeveloper(document.developer) : null,
    entrypoints: ownProperty(document, "entrypoints") ? parseEntrypoints(document.entrypoints) : {},
    contributes: ownProperty(document, "contributes") ? parseContributes(document.contributes) : { panels: [], commands: [] },
  };
}

export function decodePluginManifest(bytes, folderName) {
  if (bytes.byteLength > PLUGIN_MANIFEST_MAXIMUM_BYTES) fail("manifest_too_large");
  // A UTF-8 BOM is rejected rather than stripped so the manifest byte range stays exactly the range
  // a future signature would have to cover.
  if (bytes.byteLength >= 3 && bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf) fail("manifest_not_utf8");
  let text;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    fail("manifest_not_utf8");
  }
  let document;
  try {
    document = JSON.parse(text);
  } catch {
    fail("manifest_not_json");
  }
  if (!plainObject(document)) fail("manifest_not_json");
  return parsePluginManifest(document, folderName);
}

function samePath(first, second) {
  return process.platform === "win32" ? first.toLowerCase() === second.toLowerCase() : first === second;
}

/**
 * The default reparse-point probe.
 *
 * Node reports Windows symlinks and directory junctions through `lstat`, but it cannot see other
 * reparse tags (AppExecLink, deduplication, cloud placeholders), so `lstat` alone is not a complete
 * answer. `assertOrdinaryEntry` therefore also requires `realpath()` to return the exact same path,
 * which catches every reparse point that actually redirects somewhere else. Tests inject their own
 * probe to exercise the rejection branch decisively on any platform.
 */
export function defaultReparsePointProbe(_absolutePath, stats) {
  return stats.isSymbolicLink();
}

async function assertOrdinaryEntry(absolutePath, { kind, probeReparsePoint, code = "unsafe_reparse_point" }) {
  const stats = await lstat(absolutePath);
  if (stats.isSymbolicLink()) fail(code);
  if (kind === "directory" ? !stats.isDirectory() : !stats.isFile()) fail(code);
  if (probeReparsePoint(absolutePath, stats)) fail(code);
  // A successful `realpath()` is not permission. Equality is: it proves no segment of this exact
  // path is a link, junction, or mount point that redirects elsewhere.
  let canonical;
  try {
    canonical = await realpath(absolutePath);
  } catch {
    fail(code);
  }
  if (!samePath(canonical, absolutePath)) fail(code);
  return stats;
}

function fileIdentity(stats) {
  return `${stats.dev}:${stats.ino}:${stats.size}:${stats.mtimeNs}`;
}

async function readManifestBytes(manifestPath, probeReparsePoint) {
  let before;
  try {
    before = await lstat(manifestPath, { bigint: true });
  } catch (error) {
    if (error.code === "ENOENT") fail("manifest_missing");
    fail("manifest_unreadable");
  }
  if (before.isSymbolicLink() || !before.isFile()) fail("unsafe_reparse_point");
  if (probeReparsePoint(manifestPath, before)) fail("unsafe_reparse_point");
  if (before.size > BigInt(PLUGIN_MANIFEST_MAXIMUM_BYTES)) fail("manifest_too_large");

  // `O_NOFOLLOW` closes the lstat-then-open window where the platform has it; Windows does not, so
  // the identity comparisons below are what make the check decisive there.
  const flags = fsConstants.O_RDONLY | (fsConstants.O_NOFOLLOW || 0);
  let handle;
  try {
    handle = await open(manifestPath, flags);
  } catch (error) {
    if (error.code === "ENOENT") fail("manifest_missing");
    if (error.code === "ELOOP") fail("unsafe_reparse_point");
    fail("manifest_unreadable");
  }
  try {
    // Every check from here on runs against the open handle, so the bytes we parse provably come
    // from the inode we validated rather than from whatever the path points at now.
    const opened = await handle.stat({ bigint: true });
    if (!opened.isFile()) fail("unsafe_reparse_point");
    if (opened.size > BigInt(PLUGIN_MANIFEST_MAXIMUM_BYTES)) fail("manifest_too_large");
    if (fileIdentity(opened) !== fileIdentity(before)) fail("manifest_changed_during_read");
    const bytes = await handle.readFile();
    const after = await handle.stat({ bigint: true });
    if (fileIdentity(after) !== fileIdentity(before)) fail("manifest_changed_during_read");
    if (BigInt(bytes.byteLength) !== before.size) fail("manifest_changed_during_read");
    return bytes;
  } catch (error) {
    if (error instanceof PluginDiagnosticError) throw error;
    fail("manifest_unreadable");
  } finally {
    await handle.close().catch(() => {});
  }
}

async function inspectPluginFolder(pluginsRoot, folderName, probeReparsePoint) {
  const folderPath = path.join(pluginsRoot, folderName);
  await assertOrdinaryEntry(folderPath, { kind: "directory", probeReparsePoint });
  const bytes = await readManifestBytes(path.join(folderPath, PLUGIN_MANIFEST_FILENAME), probeReparsePoint);
  // Re-verify the canonical parent after the read: a folder swapped for a junction mid-read must
  // invalidate the manifest that was just parsed.
  await assertOrdinaryEntry(folderPath, { kind: "directory", probeReparsePoint });
  return decodePluginManifest(bytes, folderName);
}

function invalidEntry(id, code) {
  return {
    id,
    name: id,
    description: "",
    version: null,
    enabled: false,
    state: "invalid",
    compatible: false,
    execution: PLUGIN_EXECUTION_SUPPORT,
    contributions: { panels: [], commands: [] },
    diagnostics: [code],
  };
}

function manifestEntry(manifest, enabledIds) {
  const compatible = hostApiCompatible(manifest.hostApi);
  const blocked = compatible && manifest.permissions.length > 0;
  const diagnostics = [];
  // Precedence is fixed: an incompatible host API is reported before an unsupported permission
  // declaration, so one manifest never produces two different states across hosts.
  if (!compatible) diagnostics.push("host_api_incompatible");
  else if (blocked) diagnostics.push("permissions_not_supported");
  const state = compatible ? (blocked ? "blocked" : "discovered") : "incompatible";
  return {
    id: manifest.id,
    name: manifest.name,
    // Bounded, control-character-free, and validated at parse time. Exposed so the settings page can
    // describe a plugin without the browser ever reading a manifest.
    description: manifest.description,
    version: manifest.version,
    // A stored user preference, never an execution grant: `execution` stays `"not-supported"` and
    // nothing runs either way. Health always wins, so a plugin that became invalid while enabled
    // reports `false` without losing the stored intent — repairing the manifest restores the choice.
    enabled: state === "discovered" && enabledIds.has(manifest.id),
    state,
    compatible,
    execution: PLUGIN_EXECUTION_SUPPORT,
    contributions: {
      panels: manifest.contributes.panels.map((panel) => ({ ...panel })),
      commands: manifest.contributes.commands.map((command) => ({ ...command })),
    },
    diagnostics,
  };
}

function deepFreeze(value) {
  if (Array.isArray(value)) {
    value.forEach(deepFreeze);
    return Object.freeze(value);
  }
  if (plainObject(value)) {
    Object.values(value).forEach(deepFreeze);
    return Object.freeze(value);
  }
  return value;
}

function compareText(first, second) {
  return first < second ? -1 : first > second ? 1 : 0;
}

function registrySnapshot(plugins, diagnostics) {
  const sorted = [...plugins].sort((first, second) => compareText(first.id, second.id));
  const sortedDiagnostics = [...diagnostics].sort((first, second) => (
    compareText(first.id || "", second.id || "") || compareText(first.code, second.code)
  ));
  return deepFreeze({
    schemaVersion: PLUGIN_REGISTRY_SCHEMA_VERSION,
    hostApiVersion: PLUGIN_HOST_API_VERSION,
    execution: PLUGIN_EXECUTION_SUPPORT,
    plugins: sorted,
    diagnostics: sortedDiagnostics,
  });
}

/**
 * Resolve one plugin folder under the same path-safety contract discovery uses, and return its
 * absolute path. Exported so folder actions never re-implement the rule: the plugins root and the
 * folder must both be ordinary directories whose `realpath()` equals the inspected path.
 *
 * Throws `PluginDiagnosticError` with a stable code. This function only inspects; it never writes.
 */
export async function assertSafePluginFolder({ projectRoot, id, probeReparsePoint = defaultReparsePointProbe } = {}) {
  if (!validPluginId(id)) fail("invalid_plugin_id");
  let canonicalRoot;
  try {
    canonicalRoot = await realpath(path.resolve(projectRoot));
  } catch {
    fail("plugins_root_unavailable");
  }
  const pluginsRoot = pluginsRootFor(canonicalRoot);
  await assertOrdinaryEntry(pluginsRoot, { kind: "directory", probeReparsePoint, code: "plugins_root_unsafe" });
  const folderPath = path.join(pluginsRoot, id);
  // A validated id cannot contain a separator, so this cannot escape; the check is kept explicit so
  // the containment guarantee is visible rather than inferred.
  if (path.dirname(folderPath) !== pluginsRoot) fail("unsafe_reparse_point");
  await assertOrdinaryEntry(folderPath, { kind: "directory", probeReparsePoint });
  return folderPath;
}

export function pluginsRootFor(projectRoot) {
  // Fixed by construction: there is no environment variable, manifest field, API parameter, or UI
  // control that can move the plugin root away from `<projectRoot>/plugins`.
  return path.join(projectRoot, PLUGIN_DIRECTORY_NAME);
}

/**
 * Discover every plugin under `<projectRoot>/plugins`.
 *
 * Never throws: an unreadable or unsafe root degrades to an empty, fail-closed registry so the Vite
 * startup path and every other control-plane route stay unaffected.
 */
export async function discoverPlugins({ projectRoot, probeReparsePoint = defaultReparsePointProbe, enabledIds = [], stateReadable = true } = {}) {
  const diagnostics = [];
  const enabled = enabledIds instanceof Set ? enabledIds : new Set(enabledIds);
  // An unreadable or future-schema preference file disables everything and is reported once at
  // registry scope, rather than being silently rewritten.
  if (!stateReadable) diagnostics.push({ id: null, code: "plugin_state_unreadable" });
  let canonicalRoot;
  try {
    // Canonicalise the project root first so a project that legitimately lives under a symlinked
    // parent (WSL mounts, macOS `/tmp`) does not make every plugin look unsafe.
    canonicalRoot = await realpath(path.resolve(projectRoot));
  } catch {
    return registrySnapshot([], [...diagnostics, { id: null, code: "plugins_root_unavailable" }]);
  }

  const pluginsRoot = pluginsRootFor(canonicalRoot);
  let entries;
  try {
    await assertOrdinaryEntry(pluginsRoot, { kind: "directory", probeReparsePoint, code: "plugins_root_unsafe" });
    // Direct children only. The registry never recurses into a plugin folder.
    entries = await readdir(pluginsRoot, { withFileTypes: true });
  } catch (error) {
    // A missing plugin root is the normal state; it is reported as an empty registry and is never
    // created implicitly.
    if (error.code === "ENOENT") return registrySnapshot([], diagnostics);
    if (error instanceof PluginDiagnosticError) return registrySnapshot([], [...diagnostics, { id: null, code: error.code }]);
    return registrySnapshot([], [...diagnostics, { id: null, code: "plugins_root_unavailable" }]);
  }

  const candidates = [];
  for (const entry of entries) {
    // Root-level files (including `plugins/README.md`) are not plugins and are ignored.
    if (entry.isFile()) continue;
    if (!entry.isDirectory() && !entry.isSymbolicLink()) continue;
    candidates.push({ name: entry.name, link: entry.isSymbolicLink() });
  }

  // Windows folder names are case-insensitive, so a case-only collision would be one folder there
  // and two here. Fail every colliding candidate closed rather than picking a winner.
  const collisions = new Map();
  for (const candidate of candidates) {
    const key = candidate.name.toLowerCase();
    collisions.set(key, (collisions.get(key) || 0) + 1);
  }

  const plugins = [];
  for (const candidate of candidates) {
    const label = displayFolderName(candidate.name);
    if (collisions.get(candidate.name.toLowerCase()) > 1) {
      plugins.push(invalidEntry(label, "duplicate_id"));
      diagnostics.push({ id: label, code: "duplicate_id" });
      continue;
    }
    if (!validPluginId(candidate.name)) {
      plugins.push(invalidEntry(label, "invalid_plugin_id"));
      diagnostics.push({ id: label, code: "invalid_plugin_id" });
      continue;
    }
    if (candidate.link) {
      plugins.push(invalidEntry(candidate.name, "unsafe_reparse_point"));
      diagnostics.push({ id: candidate.name, code: "unsafe_reparse_point" });
      continue;
    }
    let manifest;
    try {
      manifest = await inspectPluginFolder(pluginsRoot, candidate.name, probeReparsePoint);
    } catch (error) {
      // One broken plugin is isolated; it never hides its healthy neighbours.
      const code = error instanceof PluginDiagnosticError ? error.code : "plugin_unavailable";
      plugins.push(invalidEntry(candidate.name, code));
      diagnostics.push({ id: candidate.name, code });
      continue;
    }
    const entry = manifestEntry(manifest, enabled);
    plugins.push(entry);
    for (const code of entry.diagnostics) diagnostics.push({ id: entry.id, code });
  }

  // Backstop: identifiers in the snapshot must be unique so a client can key on them. Reaching this
  // branch would mean two folders collapsed to one label, so the whole registry fails closed.
  const seenIds = new Set();
  for (const entry of plugins) {
    if (seenIds.has(entry.id)) return registrySnapshot([], [...diagnostics, { id: null, code: "duplicate_id" }]);
    seenIds.add(entry.id);
  }
  return registrySnapshot(plugins, diagnostics);
}

/**
 * Lazy, single-flight registry reader. Concurrent requests share one directory scan; the next
 * request after that scan settles starts a fresh one, so a newly added folder appears without a
 * restart. Nothing is scanned until the first `read()`, so Vite startup never touches the disk.
 */
export function createPluginRegistry({ projectRoot, probeReparsePoint = defaultReparsePointProbe, loadState } = {}) {
  // Without a state loader the registry behaves exactly as before: nothing is enabled and no
  // preference file is consulted, which keeps this module usable on its own.
  const readState = loadState || (async () => ({ enabled: [], readable: true }));
  let inFlight = null;
  const scan = async () => {
    const state = await readState();
    return discoverPlugins({ projectRoot, probeReparsePoint, enabledIds: state.enabled, stateReadable: state.readable });
  };
  return {
    read() {
      if (!inFlight) inFlight = scan().finally(() => { inFlight = null; });
      return inFlight;
    },
  };
}

function decodedRequestPath(rawUrl) {
  if (typeof rawUrl !== "string" || !rawUrl) return null;
  let raw = rawUrl.split("#")[0].split("?")[0];
  // Absolute-form request targets carry a scheme and authority. Parsing the target by hand rather
  // than through `new URL(url, base)` is deliberate: the URL parser reads `//plugins/x.js` as the
  // host `plugins`, which would hand the static layer a path this check had already cleared.
  const scheme = /^[A-Za-z][A-Za-z0-9+.-]*:\/\//.exec(raw);
  if (scheme) {
    const authorityAndPath = raw.slice(scheme[0].length);
    const separator = authorityAndPath.indexOf("/");
    raw = separator === -1 ? "/" : authorityAndPath.slice(separator);
  }
  let decoded;
  try {
    // Decode before matching: Vite resolves `/%70lugins/x.js` to `plugins/x.js` on disk.
    decoded = decodeURIComponent(raw);
  } catch {
    return null;
  }
  if (/[\0-\x1f\x7f]/.test(decoded)) return null;
  // `\`, `//`, and `/` all reach the same file, so they are folded before the prefix comparison.
  const normalized = decoded.replace(/\\/g, "/").replace(/\/{2,}/g, "/");
  return path.posix.normalize(normalized.startsWith("/") ? normalized : `/${normalized}`);
}

/**
 * True when a request would expose anything under `plugins/`, including the `/@fs/` absolute-path
 * form that bypasses a plain `/plugins/` prefix check. Malformed percent-encoding fails closed.
 */
export function servesPluginContent(rawUrl, projectRoot) {
  const resolved = decodedRequestPath(rawUrl);
  if (resolved === null) return true;
  const windows = process.platform === "win32";
  const candidate = windows ? resolved.toLowerCase() : resolved;
  const prefix = `/${PLUGIN_DIRECTORY_NAME}`;
  if (candidate === prefix || candidate.startsWith(`${prefix}/`)) return true;
  if (!candidate.startsWith("/@fs/")) return false;
  // `/@fs/` paths arrive posix-style with any drive letter inline (`/@fs/D:/project/plugins/x`).
  const stripDrive = (value) => value.replace(/^\/([A-Za-z]:)/, "$1");
  const target = stripDrive(windows ? resolved.slice("/@fs".length).toLowerCase() : resolved.slice("/@fs".length));
  const pluginsRoot = pluginsRootFor(path.resolve(projectRoot)).split(path.sep).join("/");
  const root = stripDrive(windows ? pluginsRoot.toLowerCase() : pluginsRoot);
  return target === root || target.startsWith(`${root}/`);
}

export const pluginRegistryInternals = {
  PluginDiagnosticError,
  decodedRequestPath,
  fileIdentity,
  readManifestBytes,
};
