import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { createHash, randomUUID } from "node:crypto";
import { execFileSync, spawn } from "node:child_process";
import { createReadStream } from "node:fs";
import { mkdir, open, readFile, readdir, realpath, rename, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { EnvHttpProxyAgent, fetch as undiciFetch } from "undici";
import { applyPreparedUpdate, archiveExtensionAllowed, prepareUpdate } from "./scripts/archive-update.mjs";
import { appVersion } from "./scripts/app-version.mjs";
import { downloadFile } from "./scripts/download.mjs";
import {
  checksumRoutes,
  hasCustomReleaseFeed,
  parseChecksumFile,
  parseRelease,
  releaseDownloadRoutes,
  missingReleaseMeaning,
  releaseFeedUrl,
  releaseRepositoryUrl,
  trustedGithubUrl,
  updateAuthorizationHeaders,
  updateAvailable,
} from "./scripts/release-feed.mjs";
import { getLogsDirectory, writeDiagnosticLog } from "./scripts/diagnostics.mjs";
import { findPython, isolatedPythonEnv, loadLocalEnv, verifyProjectVenv } from "./scripts/python.mjs";
import { getSetupMarkerPath, readSetupMarker } from "./scripts/setup-state.mjs";
import { ensureUpdatedProjectReady, repairUpdatedEnvironment, validateUpdatedProject } from "./scripts/update-validation.mjs";
import { createUpdateRestartHandoff } from "./scripts/update-restart.mjs";
import { updateBusy } from "./src/update-navigation.js";
import { acquireOfflineUpdateLock } from "./scripts/offline-update-lock.mjs";
import { createOfflineUpdateTemp, removeOfflineUpdateTemp } from "./scripts/offline-update-temp.mjs";
import { configuredModelDirectory, defaultLoraCategories, groupLoraModels, mergeModelPaths } from "./scripts/model-paths.mjs";
import { readPngMetadataChunks } from "./scripts/png-text-chunks.mjs";
import { interpretImageMetadata, matchModelName } from "./src/image-metadata.js";
import {
  formatSharedRef,
  inspectSharedDirectory,
  parseSharedRef,
  readSharedRoots,
  resolveSharedFile,
  sharedKindDirectories,
  sharedRootDraft,
  shapeSharedLoraCategory,
  upsertSharedRoot,
  writeSharedRoots,
} from "./scripts/shared-model-paths.mjs";
import { ANIMA_RUNTIME_ARTIFACTS, animaRuntimeArtifactStatuses, bundledAnimaTokenizerDirectory, discoverAnimaModels } from "./scripts/anima-models.mjs";
import { discoverFluxModels } from "./scripts/flux-models.mjs";
import { discoverFlux2Models } from "./scripts/flux2-models.mjs";
import { discoverKrea2Models } from "./scripts/krea2-models.mjs";
import { createConfiguratorHandoff } from "./scripts/configurator-handoff.mjs";
import { guardedInferenceResponse, inferenceIdentityGate } from "./scripts/inference-identity-gate.mjs";
import { PLUGIN_DIAGNOSTIC_CODES, createPluginRegistry, pluginsRootFor, servesPluginContent } from "./scripts/plugin-registry.mjs";
import { applyPluginEnabled, pluginStatePathFor, pluginToggleAdmission, readPluginState, writePluginState } from "./scripts/plugin-state.mjs";
import { removePluginFolder, revealPluginFolder } from "./scripts/plugin-actions.mjs";
import { descriptionNeedsVersionIdentity, loraMetadataCacheValid, plainTextFromHtml, readLoraFileMetadata, reviewLoraPrompts, TRIGGER_REVIEW_SCHEMA } from "./scripts/lora-metadata.mjs";
import { appendDownloadQueueState, filterPendingRecommendedArtifacts, itemStatusIsTerminal } from "./scripts/model-download-queue.mjs";
import { assistantReadiness, mergeAssistantSettings, readAssistantProfileStore, readAssistantSettings, redactAssistantProfileStore, redactAssistantSettings, assistantSettingsPath, writeAssistantProfileStore, writeAssistantSettings } from "./scripts/assistant-settings.mjs";
import {
  MAXIMUM_PROFILES,
  activateAssistantProfile,
  assistantProfileAt,
  createAssistantProfile,
  duplicateAssistantProfile,
  removeAssistantProfile,
  updateAssistantProfile,
  validProfileId,
} from "./src/assistant-profiles.js";
import {
  MAXIMUM_USER_PERSONAS,
  createUserPersona,
  deleteUserPersona,
  readPersonaDirectory,
  readUserPersona,
  saveUserPersona,
  selectPersona,
} from "./scripts/assistant-personas.mjs";
import { validUserPersonaId, validateAssistantPersona } from "./src/assistant-persona.js";
import { createSession, deleteSession, listSessions, readSession, saveSession } from "./scripts/assistant-sessions.mjs";
import { buildChatMessages, buildChatRequestBody, providerErrorMessage, validSessionId } from "./src/ai-assistant-protocol.js";
import { chatCompletionsUrl, modelsUrl, parseModelList, strengthPayload } from "./src/ai-assistant-providers.js";

const projectRoot = path.dirname(fileURLToPath(import.meta.url));
Object.assign(process.env, loadLocalEnv(projectRoot));
const modelPathConfig = path.join(projectRoot, "models", "model-paths.json");
const recommendedModelCatalogPath = path.join(projectRoot, "models", "recommended-models.json");
const yoloCatalogPath = path.join(projectRoot, "models", "yolo-models.json");
const backgroundRemovalCatalogPath = path.join(projectRoot, "models", "background-removal-models.json");
const logoPath = path.join(projectRoot, "public", "xiriacanvas-logo.svg");
const checkpointExtensions = new Set([".safetensors", ".ckpt"]);
const loraExtensions = new Set([".safetensors", ".ckpt", ".pt", ".pth"]);
const animaLoraExtensions = new Set([".safetensors"]);
const modelExtensions = new Set([...checkpointExtensions, ...loraExtensions]);
// FLUX.1, FLUX.2 and Krea 2 also load a GGUF diffusion model, so one has to be matchable by name
// when a saved image is read back — not only selectable in the picker.
const diffusionModelExtensions = new Set([...checkpointExtensions, ".gguf"]);
const checkpointEnginePathKeys = { SD: "sd", iL: "illustrious" };
// The two native engines accept only .safetensors LoRAs: their fusion reads tensors directly
// rather than through a pickle loader.
const nativeEngines = new Set(["Anima", "Flux", "Flux2", "Krea2"]);
const loraEnginePathKeys = { SD: "sd", iL: "illustrious", Anima: "anima", Flux: "flux", Flux2: "flux2", Krea2: "krea2" };
const loraCategories = defaultLoraCategories;
const downloadableModelKinds = new Set(["checkpoint", "diffusion_model", "text_encoder", "lora", "vae", "yolo", "upscaler", "embedding", "config"]);
const downloadableModelExtensions = {
  checkpoint: new Set([".safetensors", ".ckpt"]),
  diffusion_model: new Set([".safetensors", ".gguf", ".ckpt", ".bin"]),
  text_encoder: new Set([".safetensors", ".gguf", ".bin", ".pt", ".pth"]),
  lora: loraExtensions,
  vae: new Set([".safetensors", ".ckpt", ".pt", ".pth", ".bin"]),
  yolo: new Set([".pt", ".onnx", ".pth"]),
  upscaler: new Set([".pth", ".pt", ".ckpt", ".safetensors"]),
  embedding: new Set([".pt", ".bin", ".safetensors"]),
  config: new Set([".json", ".yaml", ".yml"]),
};
const maximumModelDownloadBytes = 128 * 1024 ** 3;
const webSecurityHeaders = {
  "Content-Security-Policy": [
    "default-src 'self'",
    "base-uri 'self'",
    "connect-src 'self' ws: wss:",
    "font-src 'self' data:",
    "form-action 'self'",
    "frame-ancestors 'none'",
    "frame-src 'none'",
    "img-src 'self' data: blob:",
    "media-src 'self' data: blob:",
    "object-src 'none'",
    "script-src 'self' 'unsafe-inline'",
    "style-src 'self' 'unsafe-inline'",
    "worker-src 'self' blob:",
  ].join("; "),
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Resource-Policy": "same-origin",
  "Referrer-Policy": "no-referrer",
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY",
};
const civitaiDomains = ["civitai.com", "civitai.red"];
const allowedPreviewDomains = [...civitaiDomains, "img.genur.art"];
const previewContentTypes = {
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".webp": "image/webp",
};
const defaultLogoTheme = { overall: "#8B7CFF", accent: "#8B7CFF", ink: "#111112", disc: "#FFFFFF", guide: "#C9C5E8" };

function parsePort(name, fallback) {
  const value = Number(process.env[name] || fallback);
  if (!Number.isInteger(value) || value < 1 || value > 65535) throw new Error(`${name} must be an integer between 1 and 65535`);
  return value;
}

function resolveProjectPath(value, fallback) {
  if (value != null && !String(value).trim()) throw new Error("Runtime directory paths must not be empty");
  return path.resolve(projectRoot, value == null ? fallback : String(value).trim());
}

const webHost = process.env.WEB_HOST || "0.0.0.0";
const webPort = parsePort("WEB_PORT", 7709);
const inferenceHost = process.env.INFERENCE_HOST || "127.0.0.1";
const inferencePort = parsePort("INFERENCE_PORT", 8718);
const inferenceConnectHost = inferenceHost === "0.0.0.0" ? "127.0.0.1" : inferenceHost === "::" ? "::1" : inferenceHost;
const inferenceUrlHost = inferenceConnectHost.includes(":") ? `[${inferenceConnectHost}]` : inferenceConnectHost;
const inferenceTarget = `http://${inferenceUrlHost}:${inferencePort}`;
const cacheDirectory = resolveProjectPath(process.env.XIRAI_CACHE_DIR, ".cache");
const stateDirectory = resolveProjectPath(process.env.XIRAI_STATE_DIR, "state-cache");
const uiStatePath = path.join(stateDirectory, "ui-state.json");
const manualUpdateStatePath = path.join(stateDirectory, "manual-update-state.json");
const manualUpdateTransactionPath = path.join(stateDirectory, "manual-update-transaction.json");
const manualUpdateEnvironmentTransactionPath = path.join(stateDirectory, "manual-update-environment-transaction.json");
const pluginStatePath = pluginStatePathFor(stateDirectory);
const logsDirectory = getLogsDirectory(projectRoot);
const runtimeWatchDirectories = [
  path.join(projectRoot, "models"),
  cacheDirectory,
  stateDirectory,
  resolveProjectPath(process.env.XIRAI_OUTPUT_DIR, "outputs"),
  logsDirectory,
  path.join(projectRoot, ".venv"),
  // User-owned plugin folders are never part of the module graph. Ignoring them keeps a dropped-in
  // plugin from churning the dev watcher.
  pluginsRootFor(projectRoot),
];
const setupMarkerPath = getSetupMarkerPath(projectRoot);
const setupComplete = Boolean(readSetupMarker(projectRoot));
const loraCacheDirectory = path.join(cacheDirectory, "lora-metadata");
const loraMetadataRequests = new Map();
const loraPreviewRequests = new Map();
const workspaceId = createHash("sha256")
  .update(process.platform === "win32" ? path.normalize(projectRoot).toLowerCase() : path.normalize(projectRoot))
  .digest("hex");
const inferenceProtocol = 34;
export const inferenceWorkspaceId = workspaceId;
const maximumUpdateArchiveBytes = 4 * 1024 ** 3;
let stopInferenceForUpdate = () => Promise.resolve();

function normalizedThemeHex(value, fallback) {
  const match = typeof value === "string" ? value.trim().match(/^#?([0-9a-f]{6})$/i) : null;
  return match ? `#${match[1].toUpperCase()}` : fallback;
}

async function readPersistedLogoTheme() {
  try {
    const saved = JSON.parse(await readFile(uiStatePath, "utf8"))?.theme;
    if (!saved || typeof saved !== "object") return defaultLogoTheme;
    const overall = normalizedThemeHex(saved.overall, defaultLogoTheme.overall);
    const accent = normalizedThemeHex(saved.accent, defaultLogoTheme.accent);
    // The vermilion tuple is the default the withdrawn 2026-08-15 visual pass shipped. A record
    // that matches it in all five roles was written by that default rather than chosen, so it
    // follows the default back; anything else — including a vermilion a user picked — is kept.
    const withdrawnDefault = overall === "#D96846"
      && accent === "#D96846"
      && normalizedThemeHex(saved.ink, defaultLogoTheme.ink) === "#111112"
      && normalizedThemeHex(saved.disc, defaultLogoTheme.disc) === "#FFF9F2"
      && normalizedThemeHex(saved.guide, defaultLogoTheme.guide) === "#E3C7BD";
    if (overall === "#D6FF3F" && accent === "#D6FF3F" || withdrawnDefault) return defaultLogoTheme;
    return {
      overall,
      accent,
      ink: normalizedThemeHex(saved.ink, defaultLogoTheme.ink),
      disc: normalizedThemeHex(saved.disc, defaultLogoTheme.disc),
      guide: normalizedThemeHex(saved.guide, defaultLogoTheme.guide),
    };
  } catch {
    return defaultLogoTheme;
  }
}

function logoThemeFromPalette(value) {
  const colors = typeof value === "string" ? value.split("-") : [];
  if (colors.length !== 4 || colors.some((color) => !/^[0-9a-f]{6}$/i.test(color))) return null;
  return {
    ...defaultLogoTheme,
    accent: `#${colors[0].toUpperCase()}`,
    ink: `#${colors[1].toUpperCase()}`,
    disc: `#${colors[2].toUpperCase()}`,
    guide: `#${colors[3].toUpperCase()}`,
  };
}

function renderThemedLogo(svg, theme) {
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

async function themedLogoMiddleware(request, response, next) {
  const url = new URL(request.url, "http://localhost");
  if (request.method !== "GET" || !["/xiriacanvas-logo.svg", "/xiriacanvas-favicon.svg"].includes(url.pathname)) {
    next();
    return;
  }
  try {
    const requestedTheme = logoThemeFromPalette(url.searchParams.get("palette"));
    const [svg, theme] = await Promise.all([readFile(logoPath, "utf8"), requestedTheme || readPersistedLogoTheme()]);
    response.statusCode = 200;
    response.setHeader("Content-Type", "image/svg+xml; charset=utf-8");
    response.setHeader("Cache-Control", "no-store");
    response.end(renderThemedLogo(svg, theme));
  } catch (error) {
    response.statusCode = 500;
    response.end(error.message || "Logo is unavailable");
  }
}

function themedLogoPlugin() {
  return {
    name: "persisted-theme-logo",
    configureServer(server) { server.middlewares.use(themedLogoMiddleware); },
    configurePreviewServer(server) { server.middlewares.use(themedLogoMiddleware); },
  };
}

function isPathInside(root, candidate) {
  const relative = path.relative(root, candidate);
  return relative === "" || (!relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative));
}

function ignoreRuntimeWatchPath(candidate) {
  const resolved = path.resolve(candidate);
  return runtimeWatchDirectories.some((directory) => isPathInside(directory, resolved));
}

function setupGatePlugin() {
  const middleware = (request, response, next) => {
    if (readSetupMarker(projectRoot)) {
      next();
      return;
    }
    const url = new URL(request.url, "http://localhost");
    if (["/xiriacanvas-logo.svg", "/xiriacanvas-favicon.svg"].includes(url.pathname)) {
      next();
      return;
    }
    if (url.pathname === "/config" || url.pathname === "/config/") {
      response.statusCode = 503;
      response.setHeader("Content-Type", "text/html; charset=utf-8");
      response.end("<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\"><title>XirAI 尚未配置</title><body style=\"margin:0;display:grid;place-items:center;min-height:100vh;color:#eee;background:#090a09;font-family:sans-serif\"><main><h1>尚未完成环境配置</h1><p>请运行 <code>npm run setup</code> 或双击 <code>Setup-XirAI.bat</code>。</p></main></body></html>");
      return;
    }
    if (url.pathname.startsWith("/api/")) {
      response.statusCode = 503;
      response.setHeader("Content-Type", "application/json; charset=utf-8");
      response.end(JSON.stringify({ error: "Setup is not complete", config_url: "/config" }));
      return;
    }
    response.statusCode = 302;
    response.setHeader("Location", "/config");
    response.end();
  };
  return {
    name: "setup-gate",
    enforce: "pre",
    configureServer(server) { server.middlewares.use(middleware); },
    configurePreviewServer(server) { server.middlewares.use(middleware); },
  };
}

function getProxyUrl() {
  const environmentProxy = process.env.HTTPS_PROXY || process.env.https_proxy
    || process.env.HTTP_PROXY || process.env.http_proxy
    || process.env.ALL_PROXY || process.env.all_proxy;
  if (environmentProxy) return environmentProxy;
  if (process.platform !== "win32") return null;

  try {
    const output = execFileSync("reg.exe", ["query", "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings"], {
      encoding: "utf8",
      windowsHide: true,
    });
    const enabled = output.match(/ProxyEnable\s+REG_DWORD\s+0x([0-9a-f]+)/i)?.[1];
    if (enabled !== "1") return null;
    const value = output.match(/ProxyServer\s+REG_SZ\s+(.+)$/mi)?.[1]?.trim();
    if (!value) return null;
    const proxy = value.includes("=")
      ? Object.fromEntries(value.split(";").map((item) => item.split("=", 2))).https || Object.fromEntries(value.split(";").map((item) => item.split("=", 2))).http
      : value;
    return proxy ? `${proxy.includes("://") ? "" : "http://"}${proxy}` : null;
  } catch {
    return null;
  }
}

let activeProxySignature;
let activeProxyDispatcher;
let proxyCheckedAt = 0;

function getProxyDispatcher() {
  const now = Date.now();
  if (now - proxyCheckedAt < 2000) return activeProxyDispatcher;
  proxyCheckedAt = now;
  const fallbackProxy = getProxyUrl();
  const httpProxy = process.env.HTTP_PROXY || process.env.http_proxy || process.env.ALL_PROXY || process.env.all_proxy || fallbackProxy;
  const httpsProxy = process.env.HTTPS_PROXY || process.env.https_proxy || process.env.ALL_PROXY || process.env.all_proxy || fallbackProxy;
  const noProxy = process.env.NO_PROXY || process.env.no_proxy || "";
  const currentSignature = JSON.stringify({ httpProxy, httpsProxy, noProxy });
  if (currentSignature === activeProxySignature) return activeProxyDispatcher;
  activeProxyDispatcher?.destroy();
  activeProxySignature = currentSignature;
  activeProxyDispatcher = new EnvHttpProxyAgent({ httpProxy, httpsProxy, noProxy });
  return activeProxyDispatcher;
}

// How long one backend-identity observation may be shared across requests.
const GATE_PROBE_TTL_MS = 1000;

export function inferenceBackendPlugin({ testOnly = false, probeBackend: injectedProbeBackend, gateProbeTtlMs = GATE_PROBE_TTL_MS } = {}) {
  let inferenceProcess;
  let backendMonitor;
  let launchingBackend = false;
  let shuttingDown = false;
  let shutdownPromise;
  let signalHandlersRegistered = false;
  let backendState = { status: "starting", phase: "Starting inference service" };
  let consoleSequence = 0;
  let consoleCommand;
  const consoleEntries = [];

  const appendConsoleEntry = (source, stream, message) => {
    // A lone carriage return moves a terminal's cursor; the drawer has no cursor, and rendering
    // one inside a <pre> costs a blank line. Progress redraws arrive separated by exactly that
    // character, so it becomes the newline it stands in for rather than being dropped.
    const text = String(message || "").replace(/\r\n?/g, "\n");
    if (!text) return;
    consoleEntries.push({ id: ++consoleSequence, at: new Date().toISOString(), source, stream, message: text.slice(0, 12000) });
    if (consoleEntries.length > 1200) consoleEntries.splice(0, consoleEntries.length - 1200);
  };

  const requireLocalConsoleRequest = (request) => {
    const address = request.socket.remoteAddress || "";
    if (!new Set(["127.0.0.1", "::1", "::ffff:127.0.0.1"]).has(address)) {
      throw Object.assign(new Error("Console commands are available only from this computer"), { statusCode: 403 });
    }
    requireSameOrigin(request);
  };

  const runConsoleCommand = (command) => {
    const id = randomUUID();
    const executable = process.platform === "win32" ? (process.env.ComSpec || "cmd.exe") : "/bin/sh";
    const args = process.platform === "win32" ? ["/d", "/s", "/c", command] : ["-lc", command];
    appendConsoleEntry("terminal", "command", `> ${command}\n`);
    const child = spawn(executable, args, {
      cwd: projectRoot,
      env: process.env,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    });
    consoleCommand = { id, child };
    let timedOut = false;
    const timeout = setTimeout(() => {
      timedOut = true;
      appendConsoleEntry("terminal", "error", "Command exceeded the 60 second limit and was stopped.\n");
      child.kill();
    }, 60000);
    child.stdout.on("data", (data) => appendConsoleEntry("terminal", "stdout", data.toString("utf8")));
    child.stderr.on("data", (data) => appendConsoleEntry("terminal", "stderr", data.toString("utf8")));
    child.on("error", (error) => appendConsoleEntry("terminal", "error", `${error.message}\n`));
    child.on("close", (code, signal) => {
      clearTimeout(timeout);
      if (consoleCommand?.id === id) consoleCommand = undefined;
      appendConsoleEntry("terminal", timedOut || code ? "error" : "system", timedOut
        ? "Command stopped.\n"
        : code ? `Command exited with code ${code}${signal ? ` (${signal})` : ""}.\n` : `Command completed${signal ? ` (${signal})` : ""}.\n`);
    });
    return id;
  };

  const registerConsoleMiddleware = (server) => {
    server.middlewares.use(async (request, response, next) => {
      const url = new URL(request.url, "http://localhost");
      if (url.pathname !== "/api/console" && url.pathname !== "/api/console/commands") {
        next();
        return;
      }
      response.setHeader("Content-Type", "application/json; charset=utf-8");
      response.setHeader("Cache-Control", "no-store");
      try {
        requireLocalConsoleRequest(request);
        if (url.pathname === "/api/console") {
          if (request.method !== "GET") throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
          const after = Math.max(0, Number(url.searchParams.get("after") || 0) || 0);
          response.statusCode = 200;
          response.end(JSON.stringify({
            entries: consoleEntries.filter((entry) => entry.id > after),
            latest: consoleSequence,
            command_running: Boolean(consoleCommand),
          }));
          return;
        }
        if (request.method !== "POST") throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
        if (consoleCommand) throw Object.assign(new Error("Another console command is still running"), { statusCode: 409 });
        const payload = await readJsonRequest(request);
        const command = typeof payload.command === "string" ? payload.command.trim() : "";
        if (!command) throw Object.assign(new Error("Command is required"), { statusCode: 400 });
        if (command.length > 4096) throw Object.assign(new Error("Command is too long"), { statusCode: 413 });
        const id = runConsoleCommand(command);
        response.statusCode = 202;
        response.end(JSON.stringify({ status: "running", id }));
      } catch (error) {
        response.statusCode = error.statusCode || 500;
        response.end(JSON.stringify({ error: error.message || "Console request failed" }));
      }
    });
  };

  const directProbeBackend = injectedProbeBackend || (async () => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 1200);
    try {
      const response = await undiciFetch(`${inferenceTarget}/api/inference/health`, { signal: controller.signal });
      if (!response.ok) return null;
      return await response.json();
    } catch {
      return null;
    } finally {
      clearTimeout(timer);
    }
  });
  const probeBackend = async () => {
    try {
      return await directProbeBackend();
    } catch {
      return null;
    }
  };

  // The identity gate below runs on every proxied request, and the health
  // endpoint it calls is not cheap: it walks the ADetailer, upscaler and Anima
  // model trees on each call — around 300 ms against a populated `models/`
  // directory. Every gallery image therefore paid a full directory scan before
  // its bytes started moving, and a card with a thumbnail strip multiplied that
  // by the number of images. The gate was the latency, not the images.
  //
  // The gate still runs on every request; what is shared is the *observation*.
  // Concurrent requests join one in-flight probe instead of starting a stampede,
  // and a successful result is reused for GATE_PROBE_TTL_MS. Backend identity
  // only changes when the process is replaced, which cannot happen and go
  // unnoticed inside that window — the first request after it expires re-probes.
  // Only an accepted identity is ever cached, so a backend that disappears or
  // comes back mismatched is caught on the very next request and recovery is
  // never delayed by a stale negative.
  let gateProbeAt = 0;
  let gateProbeHealth = null;
  let gateProbeInFlight = null;

  const probeBackendForGate = () => {
    if (gateProbeHealth && Date.now() - gateProbeAt < gateProbeTtlMs) return Promise.resolve(gateProbeHealth);
    if (gateProbeInFlight) return gateProbeInFlight;
    gateProbeInFlight = probeBackend().then((health) => {
      // Only an observation the gate *accepts* may be shared. A reachable but
      // mismatched backend is a failure here, and caching it would keep the gate
      // shut for the rest of the window — exactly across a restart, when the
      // identity flips from wrong to right and recovery has to be immediate.
      const shareable = inferenceIdentityGate(health, inferenceProtocol, workspaceId).allowed;
      gateProbeHealth = shareable ? health : null;
      gateProbeAt = shareable ? Date.now() : 0;
      gateProbeInFlight = null;
      return health;
    }).catch(() => {
      gateProbeHealth = null;
      gateProbeAt = 0;
      gateProbeInFlight = null;
      return null;
    });
    return gateProbeInFlight;
  };

  // A restart changes identity, so the shared observation cannot outlive one.
  const invalidateGateProbe = () => {
    gateProbeAt = 0;
    gateProbeHealth = null;
  };

  const waitUntilReady = async () => {
    for (let attempt = 0; attempt < 180 && !shuttingDown; attempt += 1) {
      const health = await probeBackend();
      if (health?.status === "ready") {
        backendState = health;
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
    if (!shuttingDown && backendState.status === "starting") {
      backendState = { status: "error", error: "Inference service did not become ready within 90 seconds" };
      writeDiagnosticLog(projectRoot, {
        kind: "inference-startup-failure",
        message: backendState.error,
        details: { inference_host: inferenceHost, inference_port: inferencePort },
      });
    }
  };

  const registerHealthMiddleware = (server) => {
    server.middlewares.use(async (request, response, next) => {
      const url = new URL(request.url, "http://localhost");
      if (url.pathname !== "/api/inference/health") {
        next();
        return;
      }
      const liveHealth = await probeBackend();
      if (liveHealth?.status === "ready" && liveHealth.protocol === inferenceProtocol && liveHealth.workspace_id === workspaceId) {
        backendState = liveHealth;
      } else if (backendState.status === "ready") {
        backendState = { status: "starting", phase: "Restarting inference service" };
        invalidateGateProbe();
      }
      response.statusCode = 200;
      response.setHeader("Content-Type", "application/json; charset=utf-8");
      response.setHeader("Cache-Control", "no-store");
      response.end(JSON.stringify({
        ...backendState,
        node_version: process.version,
        web_port: webPort,
        inference_port: inferencePort,
      }));
    });
  };

  const registerInferenceIdentityGate = (server) => {
    server.middlewares.use(async (request, response, next) => {
      const url = new URL(request.url, "http://localhost");
      // Health is Vite's aggregate/direct probe endpoint and must never route
      // through this gate, otherwise identity verification deadlocks itself.
      if (!url.pathname.startsWith("/api/inference/") || url.pathname === "/api/inference/health") {
        next();
        return;
      }
      const health = await probeBackendForGate();
      const gate = inferenceIdentityGate(health, inferenceProtocol, workspaceId);
      if (gate.allowed) {
        backendState = health;
        next();
        return;
      }
      backendState = {
        status: "error",
        error: "Inference backend identity is unavailable",
        identity_reason: gate.reason,
      };
      const blocked = guardedInferenceResponse(url.pathname, health, inferenceProtocol, workspaceId);
      response.statusCode = blocked.statusCode;
      response.setHeader("Content-Type", "application/json; charset=utf-8");
      response.setHeader("Cache-Control", "no-store");
      response.end(JSON.stringify(blocked.body));
    });
  };

  const terminate = () => {
    if (shutdownPromise) return shutdownPromise;
    shuttingDown = true;
    invalidateGateProbe();
    if (!inferenceProcess || inferenceProcess.exitCode !== null || inferenceProcess.signalCode !== null) return Promise.resolve();
    shutdownPromise = new Promise((resolve) => {
      const forceTimer = setTimeout(() => {
        if (inferenceProcess.exitCode === null) inferenceProcess.kill();
      }, 30000);
      inferenceProcess.once("exit", () => {
        clearTimeout(forceTimer);
        resolve();
      });
      void undiciFetch(`${inferenceTarget}/api/inference/shutdown`, {
        method: "POST",
        headers: { "X-Shutdown-Token": inferenceProcess.shutdownToken },
      }).catch(() => inferenceProcess.kill());
    });
    return shutdownPromise;
  };
  stopInferenceForUpdate = terminate;

  const registerSignalHandlers = () => {
    if (signalHandlersRegistered) return;
    signalHandlersRegistered = true;
    for (const signal of ["SIGINT", "SIGTERM"]) {
      process.once(signal, () => void terminate());
    }
  };

  const launchBackend = async (server) => {
    if (shuttingDown || launchingBackend || (inferenceProcess && inferenceProcess.exitCode === null && inferenceProcess.signalCode === null)) return;
    launchingBackend = true;
    try {
    const existingHealth = await probeBackend();
    if (existingHealth?.status === "ready") {
      backendState = existingHealth.protocol === inferenceProtocol && existingHealth.workspace_id === workspaceId
        ? existingHealth
        : { status: "error", error: "Inference port is already used by another XiriaCanvas AI workspace" };
      return;
    }

    backendState = { status: "starting", phase: "Importing PyTorch and Diffusers" };
    invalidateGateProbe();
    const shutdownToken = randomUUID();
    const environment = {
      ...process.env,
      INFERENCE_HOST: inferenceHost,
      INFERENCE_PORT: String(inferencePort),
      INFERENCE_SHUTDOWN_TOKEN: shutdownToken,
      INFERENCE_WORKSPACE_ID: workspaceId,
      PYTHONUNBUFFERED: "1",
      // Piped, Python encodes stdout with the system code page — gbk on a Chinese Windows — while
      // this end decodes every chunk as UTF-8. Anything outside ASCII then arrives as replacement
      // characters, and a multi-byte one eats more columns than it draws, which is what pushed the
      // progress bar's closing edge further right the fuller it got.
      PYTHONIOENCODING: "utf-8",
    };
    environment.XIRAI_CACHE_DIR ||= cacheDirectory;
    environment.HF_HOME ||= path.join(cacheDirectory, "huggingface");
    environment.HF_HUB_CACHE ||= path.join(environment.HF_HOME, "hub");
    const startupProxyUrl = getProxyUrl();
    if (startupProxyUrl) {
      environment.HTTP_PROXY = environment.HTTP_PROXY || startupProxyUrl;
      environment.HTTPS_PROXY = environment.HTTPS_PROXY || startupProxyUrl;
      const noProxy = new Set((environment.NO_PROXY || environment.no_proxy || "").split(",").map((entry) => entry.trim()).filter(Boolean));
      for (const host of ["127.0.0.1", "localhost", "::1"]) noProxy.add(host);
      environment.NO_PROXY = [...noProxy].join(",");
    }
    const venvInfo = verifyProjectVenv(projectRoot);
    let python;
    if (venvInfo.ok) {
      python = venvInfo.python;
    } else {
      python = findPython(projectRoot);
      if (!python) {
        backendState = { status: "error", error: `${venvInfo.error}. Run npm run setup first.` };
        writeDiagnosticLog(projectRoot, { kind: "inference-startup-failure", message: backendState.error, details: { venv_error: venvInfo.error } });
        return;
      }
      console.warn(`[inference] ${venvInfo.error}; the project may not be fully isolated from other Python installations.`);
    }
    const isolatedEnvironment = isolatedPythonEnv(environment);
    inferenceProcess = spawn(python.command, [...python.args, path.join(projectRoot, "backend", "inference_server.py")], {
      cwd: projectRoot,
      env: isolatedEnvironment,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    });
    appendConsoleEntry("inference", "system", `Starting inference service with ${python.command}.\n`);
    inferenceProcess.shutdownToken = shutdownToken;
    registerSignalHandlers();
    inferenceProcess.stdout.on("data", (data) => {
      const text = data.toString("utf8");
      // A progress redraw opens with a carriage return so it lands back at column zero and paints
      // over the line it is replacing. Prefixing that chunk would leave the label sitting exactly
      // where the bar is about to be drawn, so it is forwarded as the backend wrote it.
      process.stdout.write(text.startsWith("\r") ? text : `[inference] ${text}`);
      appendConsoleEntry("inference", "stdout", text);
    });
    inferenceProcess.stderr.on("data", (data) => {
      process.stderr.write(`[inference] ${data}`);
      appendConsoleEntry("inference", "stderr", data.toString("utf8"));
    });
    inferenceProcess.on("error", (error) => {
      backendState = { status: "error", error: error.message };
      writeDiagnosticLog(projectRoot, { kind: "inference-startup-failure", message: error.message, details: { stage: "spawn" } });
      appendConsoleEntry("inference", "error", `Failed to start: ${error.message}\n`);
      console.error(`[inference] Failed to start: ${error.message}`);
    });
    inferenceProcess.on("exit", (code, signal) => {
      appendConsoleEntry("inference", shuttingDown ? "system" : "error", `Inference service exited (${signal || code}).\n`);
      if (!shuttingDown && backendState.status !== "ready") {
        backendState = { status: "error", error: `Inference service exited before startup (${signal || code})` };
        writeDiagnosticLog(projectRoot, { kind: "inference-startup-failure", message: backendState.error, details: { exit_code: code ?? null, signal: signal ?? null } });
      } else if (!shuttingDown) {
        backendState = { status: "error", error: `Inference service stopped unexpectedly (${signal || code})` };
        writeDiagnosticLog(projectRoot, { kind: "inference-stopped", message: backendState.error, details: { exit_code: code ?? null, signal: signal ?? null } });
      }
    });
    void waitUntilReady();
    } finally {
      launchingBackend = false;
    }
  };

  const start = async (server) => {
    registerConsoleMiddleware(server);
    registerHealthMiddleware(server);
    registerInferenceIdentityGate(server);
    if (testOnly) return;
    registerSignalHandlers();
    await launchBackend(server);
    backendMonitor = setInterval(async () => {
      if (shuttingDown || launchingBackend) return;
      const health = await probeBackend();
      if (health?.status === "ready" && health.protocol === inferenceProtocol && health.workspace_id === workspaceId) {
        backendState = health;
        return;
      }
      if (backendState.status === "ready") backendState = { status: "starting", phase: "Restarting inference service" };
      if (!inferenceProcess || inferenceProcess.exitCode !== null || inferenceProcess.signalCode !== null) {
        appendConsoleEntry("inference", "system", "Inference health check failed; restarting service.\n");
        await launchBackend(server);
      }
    }, 2000);
    backendMonitor.unref?.();
    server.httpServer?.once("close", () => {
      clearInterval(backendMonitor);
      void terminate().finally(() => {
        activeProxyDispatcher?.destroy();
        activeProxyDispatcher = undefined;
      });
    });
  };

  return {
    name: "inference-backend",
    configureServer: start,
    configurePreviewServer: start,
  };
}

function getSidecarBase(modelPath) {
  return modelPath.slice(0, -path.extname(modelPath).length);
}

function getLoraCachePath(modelPath) {
  const modelId = path.relative(path.join(projectRoot, "models"), modelPath).split(path.sep).join("/");
  const key = createHash("sha256").update(modelId).digest("hex");
  return path.join(loraCacheDirectory, key);
}

async function existingLoraPreviewFile(modelPath, metadata) {
  const candidates = [];
  if (metadata?.previewFile && path.basename(metadata.previewFile) === metadata.previewFile) {
    candidates.push(path.join(getLoraCachePath(modelPath), metadata.previewFile), path.join(path.dirname(modelPath), metadata.previewFile));
  }
  for (const extension of Object.keys(previewContentTypes)) {
    candidates.push(path.join(getLoraCachePath(modelPath), `preview${extension}`));
    candidates.push(`${getSidecarBase(modelPath)}.preview${extension}`);
  }
  for (const candidate of candidates) {
    try {
      if ((await stat(candidate)).isFile()) return path.basename(candidate);
    } catch {
      // Continue through cache and legacy model-side preview conventions.
    }
  }
  return null;
}

function loraReviewKind(requestedPath) {
  const folder = String(requestedPath || "").replaceAll("\\", "/").split("/").filter(Boolean)[0]?.toLowerCase();
  return ["character", "style", "concept", "other"].includes(folder) ? folder : "other";
}

async function readLoraMetadata(modelPath, reviewKind = "other") {
  const candidates = [
    path.join(getLoraCachePath(modelPath), "metadata.json"),
    `${getSidecarBase(modelPath)}.civitai.info`,
  ];
  for (const candidate of candidates) {
    try {
      let metadata = JSON.parse(await readFile(candidate, "utf8"));
      const previewFile = await existingLoraPreviewFile(modelPath, metadata);
      if (previewFile) metadata = { ...metadata, previewFile, previewError: undefined };
      const needsVersionIdentity = descriptionNeedsVersionIdentity(metadata.modelDescription || metadata.description) && typeof metadata.versionIsLatest !== "boolean";
      if (!needsVersionIdentity && metadata.detailSchema === 1 && (metadata.triggerReviewSchema !== TRIGGER_REVIEW_SCHEMA || metadata.triggerReviewKind !== reviewKind || !Object.hasOwn(metadata.promptReview || {}, "versionScopeKind"))) {
        metadata = { ...metadata, ...reviewedLoraMetadata(metadata, metadata.localMetadata, reviewKind) };
        await writeLoraMetadata(modelPath, metadata);
      }
      return metadata;
    } catch {
      // Continue to the legacy sidecar or report no cached metadata.
    }
  }
  return null;
}

async function findModels(directory, displayRoot = directory, includeMetadata = false, extensions = modelExtensions, valueRoot = displayRoot) {
  let entries;
  try {
    entries = await readdir(directory, { withFileTypes: true });
  } catch (error) {
    if (error.code === "ENOENT") return [];
    throw error;
  }
  const models = [];

  for (const entry of entries) {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      models.push(...await findModels(entryPath, displayRoot, includeMetadata, extensions, valueRoot));
    } else if (entry.isFile() && extensions.has(path.extname(entry.name).toLowerCase())) {
      const fileStat = await stat(entryPath);
      const model = {
        name: path.relative(displayRoot, entryPath).split(path.sep).join("/"),
        value: path.relative(valueRoot, entryPath).split(path.sep).join("/"),
        size: fileStat.size,
        modifiedAt: fileStat.mtimeMs,
      };
      if (includeMetadata) model.metadata = await readLoraMetadata(entryPath, loraReviewKind(model.value));
      models.push(model);
    }
  }

  return models;
}

// Files discovered in a shared root, named relative to that root and carrying
// a `shared:<id>/` value so nothing downstream has to know where they live.
async function findSharedModels(root, kind, extensions, includeMetadata) {
  const directories = await sharedKindDirectories(root, kind);
  const models = [];
  for (const { directory, prefix } of directories) {
    const found = await findModels(directory, directory, includeMetadata, extensions, directory);
    for (const model of found) {
      const relative = prefix ? `${prefix}/${model.value}` : model.value;
      models.push({ ...model, name: relative, value: formatSharedRef(root.id, relative), shared: true, rootId: root.id, rootLabel: root.label });
    }
  }
  models.sort((first, second) => first.name.localeCompare(second.name, "zh-CN"));
  return models;
}

async function sharedListingsFor(engine, kind, extensions, includeMetadata) {
  let roots;
  try {
    roots = await readSharedRoots(projectRoot);
  } catch {
    // A damaged shared config must not take the whole model listing down with
    // it; the local library is the thing the user cannot work without.
    return [];
  }
  const listings = [];
  for (const root of roots) {
    if (!root.enabled) continue;
    if (kind === "loras" && !root.engines.includes(engine)) continue;
    const models = await findSharedModels(root, kind, extensions, includeMetadata);
    if (models.length) listings.push({ root, models });
  }
  return listings;
}

async function getConfiguredDirectory(engine, type) {
  const enginePathKeys = type === "checkpoints" ? checkpointEnginePathKeys : type === "loras" ? loraEnginePathKeys : null;
  const pathKey = enginePathKeys?.[engine];
  // The engine decides which directory is being listed, so there is no sensible default — but the
  // refusal has to say what was asked for and what would have worked, or a caller that simply
  // forgot the parameter reads it as a broken model library.
  if (!pathKey) {
    const supported = enginePathKeys ? Object.keys(enginePathKeys).join(", ") : "";
    throw Object.assign(new Error(supported
      ? `Unsupported model engine ${engine ? `"${engine}"` : "(missing)"} for ${type}; expected one of: ${supported}`
      : `Unsupported model type "${type}"`), { statusCode: 400 });
  }
  const config = JSON.parse(await readFile(modelPathConfig, "utf8"));
  let directory;
  try {
    directory = configuredModelDirectory(config, type, projectRoot, pathKey);
  } catch (error) {
    throw Object.assign(error, { statusCode: 400 });
  }
  const modelsRoot = await realpath(path.resolve(projectRoot, "models"));
  if (!isPathInside(modelsRoot, directory)) {
    throw Object.assign(new Error("Configured model path must stay inside the project models directory"), { statusCode: 403 });
  }
  try {
    await stat(directory);
  } catch (error) {
    if (error.code === "ENOENT") await mkdir(directory, { recursive: true });
    else throw error;
  }
  const resolvedDirectory = await realpath(directory);
  if (!isPathInside(modelsRoot, resolvedDirectory)) {
    throw Object.assign(new Error("Configured model path must stay inside the project models directory"), { statusCode: 403 });
  }
  return resolvedDirectory;
}

async function validateLoraPath(engine, requestedPath) {
  if (!loraEnginePathKeys[engine]) {
    throw Object.assign(
      new Error(`Unsupported LoRA engine ${engine ? `"${engine}"` : "(missing)"}; expected one of: ${Object.keys(loraEnginePathKeys).join(", ")}`),
      { statusCode: 400 },
    );
  }
  const extensions = nativeEngines.has(engine) ? animaLoraExtensions : loraExtensions;
  if (typeof requestedPath !== "string" || path.isAbsolute(requestedPath) || !extensions.has(path.extname(requestedPath).toLowerCase())) {
    throw Object.assign(new Error("Invalid LoRA file"), { statusCode: 400 });
  }
  // Metadata and previews for a shared LoRA are cached inside this project
  // (`getLoraCachePath` keys on the path, not its location), so lookups work
  // without ever writing into the folder the user only lent us for reading.
  if (parseSharedRef(requestedPath)) {
    const sharedRoots = await readSharedRoots(projectRoot);
    return resolveSharedFile(sharedRoots.filter((root) => root.engines.includes(engine)), requestedPath, { extensions });
  }
  const loraRoot = await getConfiguredDirectory(engine, "loras");
  let modelPath;
  try {
    modelPath = await realpath(path.resolve(loraRoot, requestedPath));
  } catch {
    throw Object.assign(new Error("LoRA file does not exist"), { statusCode: 404 });
  }
  if (!isPathInside(loraRoot, modelPath) || modelPath === loraRoot || !(await stat(modelPath)).isFile()) {
    throw Object.assign(new Error("LoRA file is outside the configured model directory"), { statusCode: 403 });
  }
  return modelPath;
}

function hashFile(modelPath) {
  return new Promise((resolve, reject) => {
    const hash = createHash("sha256");
    const stream = createReadStream(modelPath);
    stream.on("error", reject);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("end", () => resolve(hash.digest("hex")));
  });
}

async function fetchWithTimeout(url, options = {}, timeout = 10000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    return await undiciFetch(url, { ...options, dispatcher: getProxyDispatcher(), signal: controller.signal });
  } catch (error) {
    const cause = error.cause?.message;
    throw new Error(cause ? `${error.message}: ${cause}` : error.message);
  } finally {
    clearTimeout(timer);
  }
}

async function fetchDownload(url, options = {}, {
  fetcher = undiciFetch,
  environment = process.env,
} = {}) {
  try {
    const headers = new Headers(options.headers || {});
    const carriesUpdateToken = headers.has("authorization");
    if (carriesUpdateToken && !trustedGithubUrl(url, environment)) {
      throw new Error("拒绝向非受信任地址发送更新令牌");
    }
    let currentUrl = url;
    for (let redirects = 0; redirects <= 5; redirects += 1) {
      const response = await fetcher(currentUrl, {
        ...options,
        headers,
        redirect: "manual",
        ...(fetcher === undiciFetch ? { dispatcher: getProxyDispatcher() } : {}),
      });
      const location = response.headers.get("location");
      if (!location || !new Set([301, 302, 303, 307, 308]).has(response.status)) return response;
      if (redirects === 5) throw new Error("更新包下载重定向次数过多");
      const nextUrl = new URL(location, currentUrl).href;
      const parsedNextUrl = new URL(nextUrl);
      if (parsedNextUrl.protocol !== "https:" || parsedNextUrl.username || parsedNextUrl.password) {
        throw new Error("更新包下载拒绝不安全的重定向地址");
      }
      // GitHub's API redirects private assets to a short-lived object URL. That URL needs no PAT;
      // stripping here is explicit rather than relying on a fetch implementation's redirect rules.
      if (headers.has("authorization") && !trustedGithubUrl(nextUrl, environment)) headers.delete("authorization");
      void response.body?.cancel().catch(() => {});
      currentUrl = nextUrl;
    }
    throw new Error("更新包下载重定向次数过多");
  } catch (error) {
    const cause = error.cause?.message;
    throw new Error(cause ? `${error.message}: ${cause}` : error.message);
  }
}

const maximumReleaseFeedBytes = 1024 ** 2;
const maximumReleaseSidecarBytes = 16 * 1024;

/** Fetch and consume a small update resource under one deadline.
 *
 * `fetchWithTimeout` is intentionally header-oriented for several unrelated metadata callers. The
 * release feed and checksum are security inputs, so their deadline remains armed while every body
 * chunk is read and a declared or actual oversized body is rejected before it reaches JSON/hash
 * parsing. Redirects are manual so an Authorization header cannot follow GitHub to object storage.
 */
async function fetchBoundedUpdateBody(url, {
  headers = {},
  timeoutMs,
  maximumBytes,
  fetcher = undiciFetch,
  environment = process.env,
  label,
} = {}) {
  const controller = new AbortController();
  let rejectDeadline;
  const deadline = new Promise((_, reject) => { rejectDeadline = reject; });
  const timer = setTimeout(() => {
    controller.abort();
    rejectDeadline(new Error(`${label}响应超时`));
  }, timeoutMs);
  const requestHeaders = new Headers(headers);
  let currentUrl = url;
  let response;
  let reader;
  try {
    if (requestHeaders.has("authorization") && !trustedGithubUrl(currentUrl, environment)) {
      throw new Error("拒绝向非受信任地址发送更新令牌");
    }
    for (let redirects = 0; redirects <= 5; redirects += 1) {
      response = await Promise.race([
        fetcher(currentUrl, {
          headers: requestHeaders,
          redirect: "manual",
          signal: controller.signal,
          ...(fetcher === undiciFetch ? { dispatcher: getProxyDispatcher() } : {}),
        }),
        deadline,
      ]);
      const location = response.headers.get("location");
      if (!location || !new Set([301, 302, 303, 307, 308]).has(response.status)) break;
      if (redirects === 5) throw new Error(`${label}重定向次数过多`);
      const nextUrl = new URL(location, currentUrl);
      if (nextUrl.protocol !== "https:" || nextUrl.username || nextUrl.password) {
        throw new Error(`${label}拒绝不安全的重定向地址`);
      }
      if (requestHeaders.has("authorization") && !trustedGithubUrl(nextUrl, environment)) {
        requestHeaders.delete("authorization");
      }
      void Promise.resolve(response.body?.cancel()).catch(() => {});
      currentUrl = nextUrl.href;
    }
    if (!response) throw new Error(`${label}没有返回响应`);
    const declared = response.headers.get("content-length");
    if (declared != null && (!/^\d+$/.test(declared) || BigInt(declared) > BigInt(maximumBytes))) {
      throw new Error(`${label}内容超过 ${maximumBytes} 字节限制`);
    }
    if (!response.body) return { response, body: Buffer.alloc(0), finalUrl: currentUrl };
    reader = response.body.getReader();
    const chunks = [];
    let bytes = 0;
    while (true) {
      const { done, value } = await Promise.race([reader.read(), deadline]);
      if (done) break;
      bytes += value.byteLength;
      if (bytes > maximumBytes) throw new Error(`${label}内容超过 ${maximumBytes} 字节限制`);
      chunks.push(Buffer.from(value));
    }
    return { response, body: Buffer.concat(chunks, bytes), finalUrl: currentUrl };
  } catch (error) {
    if (controller.signal.aborted && !/超时/.test(error.message || "")) throw new Error(`${label}响应超时`);
    throw error;
  } finally {
    clearTimeout(timer);
    if (reader) void reader.cancel().catch(() => {});
    else void response?.body?.cancel().catch(() => {});
  }
}

// Exported for deterministic network-boundary tests; production callers use updateApiPlugin.
export const onlineUpdateNetworkInternals = { fetchBoundedUpdateBody, fetchDownload };

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function fetchJsonWithRetry(url, attempts = 3, timeout = 15000) {
  let lastError;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      const response = await fetchWithTimeout(url, {
        headers: { Accept: "application/json", "User-Agent": "XirAI/0.1" },
      }, timeout);
      if (response.ok) return { data: await response.json(), notFound: false };
      if (response.status === 404) return { data: null, notFound: true };
      const retryable = response.status === 429 || response.status >= 500;
      if (!retryable) throw new Error(`${new URL(url).hostname} returned HTTP ${response.status}`);
      const retryAfter = Number(response.headers.get("retry-after"));
      lastError = new Error(`${new URL(url).hostname} returned HTTP ${response.status}`);
      if (attempt < attempts - 1) await delay(Number.isFinite(retryAfter) ? Math.min(retryAfter * 1000, 30000) : 2 ** attempt * 1000);
    } catch (error) {
      lastError = error;
      if (attempt < attempts - 1) await delay(2 ** attempt * 1000);
    }
  }
  throw lastError || new Error("Remote metadata request failed");
}

async function queryCivitaiDomain(domain, hash) {
  const versionResult = await fetchJsonWithRetry(`https://${domain}/api/v1/model-versions/by-hash/${hash}`, 1, 6000);
  if (versionResult.notFound) throw Object.assign(new Error(`${domain} did not find this hash`), { notFound: true });
  return { domain, data: versionResult.data, source: "civitai" };
}

async function enrichCivitaiVersion(result) {
  const version = result.data;
  if (version.modelId) {
    try {
      const domains = [result.domain, ...civitaiDomains.filter((domain) => domain !== result.domain)];
      const model = await Promise.any(domains.map(async (domain) => {
        const modelResult = await fetchJsonWithRetry(`https://${domain}/api/v1/models/${version.modelId}`, 1, 4000);
        if (modelResult.notFound) throw Object.assign(new Error(`${domain} did not find model ${version.modelId}`), { notFound: true });
        return modelResult.data;
      }));
      if (model) {
        version.model = { ...model, ...version.model };
        version.creator = version.creator || model.creator;
        version.tags = version.tags || model.tags;
      }
    } catch {
      // Version data is sufficient when model-level enrichment is unavailable.
    }
  }

  return result;
}

async function queryCivitai(hash) {
  const attempts = civitaiDomains.map(async (domain) => {
    return queryCivitaiDomain(domain, hash);
  });

  try {
    return await enrichCivitaiVersion(await Promise.any(attempts));
  } catch (error) {
    const errors = error.errors ?? [];
    if (errors.some((item) => item.notFound)) return null;
    throw new Error(errors.map((item) => item.message).join("; ") || "Civitai lookup failed");
  }
}

function normalizeCivArchivePayload(payload) {
  const data = payload?.data && typeof payload.data === "object" ? payload.data : payload;
  const modelBlock = data?.model && typeof data.model === "object" ? data.model : {};
  const version = data?.version || modelBlock.version || data;
  const files = version?.files || modelBlock.files || data?.files || [];
  const firstFile = Array.isArray(files) ? files[0] : files;
  const modelId = version?.modelId || version?.model_id || firstFile?.modelId || firstFile?.model_id || modelBlock.id || data?.modelId;
  const versionId = version?.id || version?.modelVersionId || version?.model_version_id || firstFile?.modelVersionId || firstFile?.model_version_id;
  if (!versionId && !modelId) return null;

  return {
    ...version,
    id: versionId,
    modelId,
    baseModel: version?.baseModel || version?.base_model,
    trainedWords: Array.isArray(version?.trainedWords) ? version.trainedWords : version?.trigger ? [version.trigger].flat() : [],
    files: Array.isArray(files) ? files : [files],
    images: Array.isArray(version?.images) ? version.images : version?.images ? [version.images] : [],
    model: {
      name: modelBlock.name || data?.name || version?.model?.name,
      type: modelBlock.type || data?.type || version?.model?.type,
      description: modelBlock.description || data?.description,
    },
    creator: version?.creator || {
      username: modelBlock.creator_username || data?.creator_username || "",
      image: modelBlock.creator_image || data?.creator_image || "",
    },
  };
}

async function queryCivArchive(hash) {
  const result = await fetchJsonWithRetry(`https://civarchive.com/api/sha256/${hash.toLowerCase()}`, 1, 6000);
  if (result.notFound) return null;
  const data = normalizeCivArchivePayload(result.data);
  return data ? { domain: "civarchive.com", data, source: "civarchive" } : null;
}

async function queryLoraMetadata(hash) {
  const archiveRequest = queryCivArchive(hash).then((result) => ({ result, error: null })).catch((error) => ({ result: null, error }));
  let civitaiError;
  try {
    const civitaiResult = await queryCivitai(hash);
    if (civitaiResult) return civitaiResult;
  } catch (error) {
    civitaiError = error;
  }

  try {
    const archive = await archiveRequest;
    if (archive.result) return archive.result;
    if (archive.error) {
      if (!civitaiError) throw archive.error;
      throw new Error(`Civitai: ${civitaiError.message}; CivArchive: ${archive.error.message}`);
    }
  } catch (archiveError) {
    throw archiveError;
  }

  if (civitaiError) throw civitaiError;
  return null;
}

function normalizeCivitaiMetadata(result, hash) {
  const { data, domain } = result;
  const modelVersions = Array.isArray(data.model?.modelVersions) ? data.model.modelVersions : [];
  const latestVersion = [...modelVersions].sort((first, second) => Date.parse(second.publishedAt || second.createdAt || 0) - Date.parse(first.publishedAt || first.createdAt || 0))[0];
  const sourceUrl = result.source === "civitai" && data.modelId
    ? `https://${domain}/models/${data.modelId}?modelVersionId=${data.id}`
    : "https://civarchive.com";
  const versionDescription = plainTextFromHtml(data.description);
  const modelDescription = plainTextFromHtml(data.model?.description);
  const description = [modelDescription, versionDescription]
    .filter((value, index, values) => value && values.indexOf(value) === index)
    .join("\n\n");
  return {
    status: "found",
    hash,
    sourceDomain: domain,
    sourceUrl,
    metadataSource: result.source,
    modelId: data.modelId,
    versionId: data.id,
    versionIsLatest: latestVersion ? Number(latestVersion.id) === Number(data.id) : undefined,
    modelName: data.model?.name || "Unknown model",
    modelType: data.model?.type || "LoRA",
    versionName: data.name || "Unknown version",
    baseModel: data.baseModel || "Unknown base model",
    trainedWords: Array.isArray(data.trainedWords) ? data.trainedWords.slice(0, 80) : [],
    stats: {
      downloads: data.stats?.downloadCount ?? 0,
      likes: data.stats?.thumbsUpCount ?? 0,
    },
    creator: data.creator?.username || data.model?.creator?.username || "",
    tags: Array.isArray(data.tags) ? data.tags.slice(0, 20) : Array.isArray(data.model?.tags) ? data.model.tags.slice(0, 20) : [],
    description,
    versionDescription,
    modelDescription,
    queriedAt: new Date().toISOString(),
    previewFile: null,
  };
}

function selectCivitaiPreviews(images) {
  const candidates = Array.isArray(images) ? images.filter((image) => image?.type === "image" && image.url) : [];
  return candidates.sort((first, second) => {
    const firstSafe = Number(first.nsfwLevel || 0) < 4 ? 0 : 1;
    const secondSafe = Number(second.nsfwLevel || 0) < 4 ? 0 : 1;
    return firstSafe - secondSafe || Number(first.nsfwLevel || 0) - Number(second.nsfwLevel || 0);
  });
}

function optimizeCivitaiPreviewUrl(imageUrl) {
  const parsedUrl = new URL(imageUrl);
  if (parsedUrl.hostname.endsWith(".civitai.com") && parsedUrl.pathname.includes("/original=true/")) {
    parsedUrl.pathname = parsedUrl.pathname.replace("/original=true/", "/width=450,optimized=true/");
  }
  return parsedUrl.toString();
}

async function fetchPreviewBuffer(imageUrl) {
  const parsedUrl = new URL(imageUrl);
  const isAllowedHost = allowedPreviewDomains.some((domain) => parsedUrl.hostname === domain || parsedUrl.hostname.endsWith(`.${domain}`));
  if (parsedUrl.protocol !== "https:" || !isAllowedHost) throw new Error("Invalid preview URL");
  const response = await fetchWithTimeout(parsedUrl, { headers: { "User-Agent": "XirAI/0.1" } }, 30000);
  if (!response.ok) throw new Error(`Preview download returned HTTP ${response.status}`);
  const finalUrl = new URL(response.url);
  const isAllowedFinalHost = allowedPreviewDomains.some((domain) => finalUrl.hostname === domain || finalUrl.hostname.endsWith(`.${domain}`));
  if (finalUrl.protocol !== "https:" || !isAllowedFinalHost) throw new Error("Preview redirected to an unsupported host");
  const declaredSize = Number(response.headers.get("content-length") || 0);
  if (declaredSize > 25 * 1024 * 1024) throw new Error("Preview image is larger than 25 MB");
  const buffer = Buffer.from(await response.arrayBuffer());
  if (buffer.length > 25 * 1024 * 1024) throw new Error("Preview image is larger than 25 MB");
  return { buffer, contentType: response.headers.get("content-type")?.split(";")[0].toLowerCase() };
}

async function downloadCivitaiPreview(imageUrl, modelPath) {
  if (!imageUrl) return null;
  const optimizedUrl = optimizeCivitaiPreviewUrl(imageUrl);
  let downloaded;
  try {
    downloaded = await fetchPreviewBuffer(optimizedUrl);
  } catch (error) {
    if (optimizedUrl === imageUrl) throw error;
    downloaded = await fetchPreviewBuffer(imageUrl);
  }
  const extension = Object.entries(previewContentTypes).find(([, type]) => type === downloaded.contentType)?.[0] || ".jpg";
  const modelCacheDirectory = getLoraCachePath(modelPath);
  await mkdir(modelCacheDirectory, { recursive: true });
  const previewPath = path.join(modelCacheDirectory, `preview${extension}`);
  const temporaryPath = path.join(modelCacheDirectory, `${randomUUID()}.tmp`);
  try {
    await writeFile(temporaryPath, downloaded.buffer, { flag: "wx" });
    await rm(previewPath, { force: true });
    await rename(temporaryPath, previewPath);
  } finally {
    await rm(temporaryPath, { force: true });
  }
  return path.basename(previewPath);
}

async function writeLoraMetadata(modelPath, metadata) {
  const modelCacheDirectory = getLoraCachePath(modelPath);
  await mkdir(modelCacheDirectory, { recursive: true });
  const metadataPath = path.join(modelCacheDirectory, "metadata.json");
  const temporaryPath = path.join(modelCacheDirectory, `${randomUUID()}.tmp`);
  try {
    await writeFile(temporaryPath, `${JSON.stringify(metadata, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
    await rm(metadataPath, { force: true });
    await rename(temporaryPath, metadataPath);
  } finally {
    await rm(temporaryPath, { force: true });
  }
}

async function ensureLoraPreview(modelPath, metadata, reviewKind = "other") {
  if (metadata.previewFile || !metadata.previewUrl) return metadata;
  const key = `${modelPath}:${metadata.hash || metadata.previewUrl}`;
  if (loraPreviewRequests.has(key)) return loraPreviewRequests.get(key);
  const request = (async () => {
    const previewFile = await downloadCivitaiPreview(metadata.previewUrl, modelPath);
    const latest = await readLoraMetadata(modelPath, reviewKind);
    if (latest?.hash && metadata.hash && latest.hash !== metadata.hash) return latest;
    const updated = { ...(latest || metadata), previewFile, previewError: undefined };
    await writeLoraMetadata(modelPath, updated);
    return updated;
  })();
  loraPreviewRequests.set(key, request);
  try {
    return await request;
  } finally {
    if (loraPreviewRequests.get(key) === request) loraPreviewRequests.delete(key);
  }
}

function normalizedSha256(value) {
  const normalized = typeof value === "string" ? value.toLowerCase() : "";
  return /^[a-f0-9]{64}$/.test(normalized) ? normalized : undefined;
}

function safeModelFilename(value, kind) {
  const name = typeof value === "string" ? path.basename(value.trim()) : "";
  if (!name || name !== value?.trim() || name.length > 180 || !downloadableModelExtensions[kind]?.has(path.extname(name).toLowerCase())) {
    throw Object.assign(new Error("模型文件名或格式与所选类型不匹配"), { statusCode: 400 });
  }
  return name;
}

function providerApiKey(value) {
  if (value == null || value === "") return "";
  if (typeof value !== "string" || value.length > 1024 || /[\r\n]/.test(value)) {
    throw Object.assign(new Error("API 密钥格式无效"), { statusCode: 400 });
  }
  return value.trim();
}

function providerHeaders(apiKey) {
  return apiKey ? { Authorization: `Bearer ${apiKey}` } : {};
}

function isAllowedProviderHost(hostname, domains) {
  return domains.some((domain) => hostname === domain || hostname.endsWith(`.${domain}`));
}

async function fetchProviderJson(url, headers, timeout = 30000) {
  const response = await fetchWithTimeout(url, {
    headers: { Accept: "application/json", "User-Agent": "XirAI/0.1", ...headers },
  }, timeout);
  if (!response.ok) throw Object.assign(new Error(`远程模型信息请求失败（HTTP ${response.status}）`), { statusCode: response.status >= 400 && response.status < 500 ? 400 : 502 });
  return response.json();
}

function selectDownloadableFile(files, kind, requestedPath = "") {
  const eligible = (Array.isArray(files) ? files : []).map((file) => {
    const sourceName = file?.name || file?.rfilename || file?.path || file?.Path || file?.Name;
    if (typeof sourceName !== "string") return null;
    let name;
    try {
      name = safeModelFilename(sourceName.split("/").pop(), kind);
    } catch {
      return null;
    }
    return { ...file, name, sourceName };
  }).filter(Boolean);
  const exact = requestedPath && eligible.find((file) => file.sourceName === requestedPath || file.sourceName.replace(/^\//, "") === requestedPath.replace(/^\//, ""));
  if (exact) return exact;
  const primary = eligible.find((file) => file.primary || file.Primary || file.default || file.Default);
  if (primary) return primary;
  if (eligible.length === 1) return eligible[0];
  if (!eligible.length) throw Object.assign(new Error("链接中没有找到符合所选模型类型的文件"), { statusCode: 400 });
  throw Object.assign(new Error("该模型仓库包含多个可下载文件，请粘贴文件详情页或直接下载链接"), { statusCode: 400 });
}

function parseModelLink(value) {
  if (typeof value !== "string" || value.length > 4096) throw Object.assign(new Error("请输入有效的模型链接"), { statusCode: 400 });
  let url;
  try {
    url = new URL(value.trim());
  } catch {
    throw Object.assign(new Error("模型链接格式无效"), { statusCode: 400 });
  }
  if (url.protocol !== "https:") throw Object.assign(new Error("模型链接必须使用 HTTPS"), { statusCode: 400 });
  const hostname = url.hostname.toLowerCase();
  if (isAllowedProviderHost(hostname, ["huggingface.co", "hf-mirror.com"])) return { provider: "huggingface", url };
  if (isAllowedProviderHost(hostname, [...civitaiDomains])) return { provider: "civitai", url };
  if (isAllowedProviderHost(hostname, ["modelscope.cn"])) return { provider: "modelscope", url };
  throw Object.assign(new Error("仅支持 Civitai、Hugging Face 和 ModelScope 的模型链接"), { statusCode: 400 });
}

function parseHuggingFaceLocation(url) {
  const parts = url.pathname.split("/").filter(Boolean).map(decodeURIComponent);
  const marker = parts.findIndex((part) => part === "resolve" || part === "blob");
  if (parts.length < 2 || (marker >= 0 && marker < 2)) throw Object.assign(new Error("Hugging Face 模型链接格式无效"), { statusCode: 400 });
  const repository = parts.slice(0, 2).join("/");
  if (!/^[\w.-]+\/[\w.-]+$/.test(repository)) throw Object.assign(new Error("Hugging Face 仓库标识无效"), { statusCode: 400 });
  return {
    repository,
    revision: marker >= 0 ? parts[marker + 1] : (url.searchParams.get("revision") || "main"),
    filePath: marker >= 0 ? parts.slice(marker + 2).join("/") : "",
  };
}

function encodeRemotePath(value) {
  return value.split("/").map(encodeURIComponent).join("/");
}

async function resolveHuggingFaceDownload(url, kind, apiKey) {
  const location = parseHuggingFaceLocation(url);
  const headers = providerHeaders(apiKey);
  const apiHosts = url.hostname.toLowerCase() === "hf-mirror.com"
    ? ["hf-mirror.com", "huggingface.co"]
    : ["huggingface.co", "hf-mirror.com"];
  let model;
  let apiError;
  for (const host of apiHosts) {
    try {
      model = await fetchProviderJson(
        `https://${host}/api/models/${encodeRemotePath(location.repository)}?blobs=true`,
        host === "huggingface.co" ? headers : {},
      );
      break;
    } catch (error) {
      apiError = error;
    }
  }
  if (!model) throw apiError;
  const file = selectDownloadableFile(model.siblings, kind, location.filePath);
  const revision = location.revision || "main";
  const relative = `${location.repository}/resolve/${encodeURIComponent(revision)}/${encodeRemotePath(file.sourceName)}?download=true`;
  const expectedSha256 = normalizedSha256(file.lfs?.oid || file.sha256 || file.hashes?.SHA256);
  return {
    provider: "Hugging Face",
    filename: file.name,
    expectedSha256,
    routes: [
      { id: "huggingface", label: "Hugging Face", url: `https://huggingface.co/${relative}`, headers },
      { id: "hf-mirror", label: "HF-Mirror", url: `https://hf-mirror.com/${relative}` },
    ],
  };
}

function civitaiVersionFromUrl(url) {
  const fromQuery = url.searchParams.get("modelVersionId");
  if (fromQuery && /^\d+$/.test(fromQuery)) return fromQuery;
  const match = url.pathname.match(/\/api\/download\/models\/(\d+)/) || url.pathname.match(/\/model-versions\/(\d+)/);
  return match?.[1] || "";
}

async function resolveCivitaiDownload(url, kind, apiKey) {
  const headers = providerHeaders(apiKey);
  const domain = isAllowedProviderHost(url.hostname, ["civitai.red"]) ? "civitai.red" : "civitai.com";
  let versionId = civitaiVersionFromUrl(url);
  let version;
  if (!versionId) {
    const modelMatch = url.pathname.match(/\/models\/(\d+)/);
    if (!modelMatch) throw Object.assign(new Error("请使用 Civitai 模型版本或文件下载链接"), { statusCode: 400 });
    const model = await fetchProviderJson(`https://${domain}/api/v1/models/${modelMatch[1]}`, headers);
    version = Array.isArray(model.modelVersions) ? model.modelVersions[0] : null;
    versionId = String(version?.id || "");
  }
  if (!versionId) throw Object.assign(new Error("Civitai 模型页面没有可用版本"), { statusCode: 400 });
  if (!version) version = await fetchProviderJson(`https://${domain}/api/v1/model-versions/${versionId}`, headers);
  const file = selectDownloadableFile(version.files, kind);
  const downloadUrl = typeof file.downloadUrl === "string" ? file.downloadUrl : `https://${domain}/api/download/models/${versionId}`;
  const downloadHost = new URL(downloadUrl).hostname.toLowerCase();
  if (!isAllowedProviderHost(downloadHost, [...civitaiDomains])) throw Object.assign(new Error("Civitai 返回了不受支持的下载地址"), { statusCode: 502 });
  const alternateDomain = downloadHost.endsWith("civitai.red") ? "civitai.com" : "civitai.red";
  const alternateUrl = new URL(downloadUrl);
  alternateUrl.hostname = alternateDomain;
  return {
    provider: "Civitai",
    filename: file.name,
    expectedSha256: normalizedSha256(file.hashes?.SHA256 || file.hashes?.sha256 || file.sha256),
    routes: [
      { id: downloadHost, label: downloadHost, url: downloadUrl, headers },
      { id: alternateDomain, label: alternateDomain, url: alternateUrl.toString(), headers },
    ],
  };
}

function parseModelScopeLocation(url) {
  const parts = url.pathname.split("/").filter(Boolean).map(decodeURIComponent);
  const marker = parts.findIndex((part) => part === "resolve");
  const modelIndex = parts.indexOf("models");
  const modelParts = marker >= 0 ? parts.slice(modelIndex + 1, marker) : parts.slice(modelIndex + 1);
  if (modelIndex < 0 || modelParts.length < 2) throw Object.assign(new Error("ModelScope 模型链接格式无效"), { statusCode: 400 });
  return {
    repository: modelParts.slice(0, 2).join("/"),
    revision: marker >= 0 ? parts[marker + 1] : (url.searchParams.get("Revision") || url.searchParams.get("revision") || "master"),
    filePath: marker >= 0 ? parts.slice(marker + 2).join("/") : "",
  };
}

async function resolveModelScopeDownload(url, kind, apiKey) {
  const location = parseModelScopeLocation(url);
  const headers = providerHeaders(apiKey);
  const apiUrl = new URL(`https://modelscope.cn/api/v1/models/${encodeRemotePath(location.repository)}/repo/files`);
  apiUrl.searchParams.set("Revision", location.revision);
  apiUrl.searchParams.set("Recursive", "true");
  const listing = await fetchProviderJson(apiUrl, headers);
  const files = listing.Data?.Files || listing.data?.files || listing.Files || listing.files || [];
  const file = selectDownloadableFile(files, kind, location.filePath);
  const filePath = file.sourceName;
  return {
    provider: "ModelScope",
    filename: file.name,
    expectedSha256: normalizedSha256(file.Sha256 || file.sha256 || file.Hashes?.SHA256),
    routes: [{
      id: "modelscope",
      label: "ModelScope",
      url: `https://modelscope.cn/models/${encodeRemotePath(location.repository)}/resolve/${encodeURIComponent(location.revision)}/${encodeRemotePath(filePath)}`,
      headers,
    }],
  };
}

async function getAuxiliaryModelDirectory(configKey) {
  const config = JSON.parse(await readFile(modelPathConfig, "utf8"));
  let directory;
  try {
    directory = configuredModelDirectory(config, configKey, projectRoot);
  } catch (error) {
    throw Object.assign(error, { statusCode: 500 });
  }
  const modelsRoot = await realpath(path.resolve(projectRoot, "models"));
  if (!isPathInside(modelsRoot, directory)) throw Object.assign(new Error("模型目录必须位于项目 models 文件夹中"), { statusCode: 403 });
  await mkdir(directory, { recursive: true });
  const resolvedDirectory = await realpath(directory);
  if (!isPathInside(modelsRoot, resolvedDirectory)) throw Object.assign(new Error("模型目录必须位于项目 models 文件夹中"), { statusCode: 403 });
  return resolvedDirectory;
}

async function effectiveModelPaths() {
  const config = mergeModelPaths(JSON.parse(await readFile(modelPathConfig, "utf8")));
  const result = {
    checkpoints: {},
    loras: {},
  };
  for (const engine of Object.values(checkpointEnginePathKeys)) {
    result.checkpoints[engine] = path.relative(projectRoot, configuredModelDirectory(config, "checkpoints", projectRoot, engine)).split(path.sep).join("/");
  }
  for (const engine of Object.values(loraEnginePathKeys)) {
    result.loras[engine] = path.relative(projectRoot, configuredModelDirectory(config, "loras", projectRoot, engine)).split(path.sep).join("/");
  }
  for (const key of ["vae", "diffusion_models", "text_encoders", "embeddings", "upscalers", "configs", "yolo", "background_removal"]) {
    result[key] = path.relative(projectRoot, configuredModelDirectory(config, key, projectRoot)).split(path.sep).join("/");
  }
  return result;
}

async function getDownloadDestination(payload, filename) {
  const kind = payload.kind;
  if (!downloadableModelKinds.has(kind)) throw Object.assign(new Error("请选择模型类型"), { statusCode: 400 });
  const safeName = safeModelFilename(filename, kind);
  if (kind === "lora" && nativeEngines.has(payload.engine) && !animaLoraExtensions.has(path.extname(safeName).toLowerCase())) {
    throw Object.assign(new Error(`${payload.engine} LoRA 仅支持 .safetensors 文件`), { statusCode: 400 });
  }
  let directory;
  if (kind === "checkpoint") directory = await getConfiguredDirectory(payload.engine, "checkpoints");
  else if (kind === "diffusion_model") directory = await getAuxiliaryModelDirectory("diffusion_models");
  else if (kind === "text_encoder") directory = await getAuxiliaryModelDirectory("text_encoders");
  else if (kind === "lora") {
    const category = loraCategories.find((item) => item.id === payload.category);
    if (!category) throw Object.assign(new Error("请选择 LoRA 分类"), { statusCode: 400 });
    const root = await getConfiguredDirectory(payload.engine, "loras");
    directory = path.join(root, category.directory);
    await mkdir(directory, { recursive: true });
    directory = await realpath(directory);
    if (!isPathInside(root, directory)) throw Object.assign(new Error("LoRA 分类目录无效"), { statusCode: 403 });
  } else if (kind === "yolo") {
    directory = await getAuxiliaryModelDirectory("yolo");
  } else if (kind === "upscaler") {
    directory = await getAuxiliaryModelDirectory("upscalers");
  } else {
    directory = await getAuxiliaryModelDirectory({ vae: "vae", embedding: "embeddings", config: "configs" }[kind]);
  }
  const destination = path.resolve(directory, safeName);
  if (!isPathInside(directory, destination) || destination === directory) throw Object.assign(new Error("模型文件路径无效"), { statusCode: 403 });
  return destination;
}

async function refreshLoraMetadata(modelPath, fileStat, cachedMetadata, reviewKind) {
  const cachedHash = cachedMetadata?.detailSchema === 1
    && Number(cachedMetadata.fileSize) === Number(fileStat.size)
    && Number(cachedMetadata.modifiedAt) === Number(fileStat.mtimeMs)
    && normalizedSha256(cachedMetadata.hash);
  const [hash, localResult] = await Promise.all([
    cachedHash || hashFile(modelPath),
    readLoraFileMetadata(modelPath).then((metadata) => ({ metadata, error: "" })).catch((error) => ({ metadata: null, error: error.message || "无法解析本地模型元数据" })),
  ]);
  const localMetadata = localResult.metadata || { format: path.extname(modelPath).slice(1).toUpperCase(), parsed: false, fields: [], topTags: [], triggerWords: [] };
  let result;
  let remoteError = "";
  try {
    result = await queryLoraMetadata(hash);
  } catch (error) {
    remoteError = error.message || "Civitai lookup failed";
  }
  const preserveCachedRemote = !result && remoteError && cachedMetadata?.status === "found" && cachedMetadata.hash === hash;
  const metadata = result
    ? normalizeCivitaiMetadata(result, hash)
    : preserveCachedRemote
      ? { ...cachedMetadata, queriedAt: new Date().toISOString() }
      : { status: remoteError ? "local_only" : "not_found", hash, fileSize: fileStat.size, modifiedAt: fileStat.mtimeMs, queriedAt: new Date().toISOString(), previewFile: null };
  metadata.fileSize = fileStat.size;
  metadata.modifiedAt = fileStat.mtimeMs;
  metadata.detailSchema = 1;
  metadata.previewSchema = 1;
  metadata.localMetadata = localMetadata;
  if (localResult.error) metadata.localMetadataError = localResult.error;
  metadata.description ||= localMetadata.description;
  Object.assign(metadata, reviewedLoraMetadata(metadata, localMetadata, reviewKind));
  if (remoteError) metadata.remoteError = remoteError;
  if (result) {
    const previewCandidates = selectCivitaiPreviews(result.data.images);
    metadata.previewUrl = previewCandidates[0]?.url || null;
    metadata.previewFile = cachedMetadata?.hash === hash ? cachedMetadata.previewFile || null : null;
    if (!metadata.previewUrl && !metadata.previewFile) metadata.previewError = "No preview images were provided by Civitai";
  }
  await writeLoraMetadata(modelPath, metadata);
  return metadata;
}

function reviewedLoraMetadata(metadata, localMetadata = metadata?.localMetadata, reviewKind = "other") {
  const promptReview = reviewLoraPrompts({
    trainedWords: metadata.trainedWords,
    description: metadata.description,
    modelDescription: metadata.modelDescription,
    versionDescription: metadata.versionDescription,
    versionName: metadata.versionName,
    versionIsLatest: metadata.versionIsLatest,
    localMetadata,
    reviewKind,
  });
  return {
    triggerReviewSchema: TRIGGER_REVIEW_SCHEMA,
    triggerReviewKind: reviewKind,
    triggerGroups: promptReview.groups,
    promptReview: { reviewedSources: promptReview.reviewedSources, ignoredSegments: promptReview.ignoredSegments, acceptedGroups: promptReview.acceptedGroups, versionScope: promptReview.versionScope || "", versionScopeKind: promptReview.versionScopeKind || "" },
  };
}

async function cacheLoraMetadata(modelPath, { refresh = false, reviewKind = "other" } = {}) {
  const fileStat = await stat(modelPath);
  const cachedMetadata = await readLoraMetadata(modelPath, reviewKind);
  if (!refresh && loraMetadataCacheValid(cachedMetadata, fileStat, reviewKind)) return cachedMetadata;
  const cacheIdentityMatches = cachedMetadata?.detailSchema === 1
    && Number(cachedMetadata.fileSize) === Number(fileStat.size)
    && Number(cachedMetadata.modifiedAt) === Number(fileStat.mtimeMs);
  const needsVersionIdentity = descriptionNeedsVersionIdentity(cachedMetadata?.modelDescription || cachedMetadata?.description) && typeof cachedMetadata?.versionIsLatest !== "boolean";
  if (!refresh && cacheIdentityMatches && !needsVersionIdentity && (cachedMetadata.triggerReviewSchema !== TRIGGER_REVIEW_SCHEMA || cachedMetadata.triggerReviewKind !== reviewKind || !Object.hasOwn(cachedMetadata.promptReview || {}, "versionScopeKind"))) {
    const reviewedMetadata = { ...cachedMetadata, ...reviewedLoraMetadata(cachedMetadata, cachedMetadata.localMetadata, reviewKind) };
    await writeLoraMetadata(modelPath, reviewedMetadata);
    return reviewedMetadata;
  }
  const key = `${modelPath}:${fileStat.size}:${fileStat.mtimeMs}`;
  if (loraMetadataRequests.has(key)) return loraMetadataRequests.get(key);
  const request = refreshLoraMetadata(modelPath, fileStat, cachedMetadata, reviewKind);
  loraMetadataRequests.set(key, request);
  try {
    return await request;
  } finally {
    if (loraMetadataRequests.get(key) === request) loraMetadataRequests.delete(key);
  }
}

async function readJsonRequest(request, maxBytes = 64 * 1024) {
  const chunks = [];
  let length = 0;
  for await (const chunk of request) {
    length += chunk.length;
    if (length > maxBytes) throw Object.assign(new Error("Request body is too large"), { statusCode: 413 });
    chunks.push(chunk);
  }
  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    throw Object.assign(new Error("Invalid JSON request body"), { statusCode: 400 });
  }
}

function requireSameOrigin(request) {
  if (!request.headers.origin) return;
  let origin;
  try {
    origin = new URL(request.headers.origin);
  } catch {
    throw Object.assign(new Error("Invalid request origin"), { statusCode: 403 });
  }
  if (origin.host !== request.headers.host) {
    throw Object.assign(new Error("Cross-origin model downloads are not allowed"), { statusCode: 403 });
  }
}

function requireLocalRequest(request, message = "This operation is available only from this computer") {
  const address = request.socket.remoteAddress || "";
  if (!new Set(["127.0.0.1", "::1", "::ffff:127.0.0.1"]).has(address)) {
    throw Object.assign(new Error(message), { statusCode: 403 });
  }
  requireSameOrigin(request);
}

function validLogFilename(value) {
  return typeof value === "string" && value === path.basename(value) && value.endsWith(".log") && value.length <= 180;
}

async function listDiagnosticLogs() {
  await mkdir(logsDirectory, { recursive: true });
  const entries = await readdir(logsDirectory, { withFileTypes: true });
  const logs = await Promise.all(entries.filter((entry) => entry.isFile() && validLogFilename(entry.name)).map(async (entry) => {
    const info = await stat(path.join(logsDirectory, entry.name));
    return { name: entry.name, bytes: info.size, modified_at: info.mtime.toISOString() };
  }));
  return logs.sort((first, second) => second.modified_at.localeCompare(first.modified_at));
}

async function logApi(request, response, next) {
  const url = new URL(request.url, "http://localhost");
  if (url.pathname !== "/api/logs") {
    next();
    return;
  }
  response.setHeader("Content-Type", "application/json; charset=utf-8");
  response.setHeader("Cache-Control", "no-store");
  try {
    requireLocalRequest(request, "Diagnostic logs are available only from this computer");
    if (request.method === "GET") {
      const name = url.searchParams.get("name");
      if (!name) {
        response.statusCode = 200;
        response.end(JSON.stringify({ directory: "logs", logs: await listDiagnosticLogs() }));
        return;
      }
      if (!validLogFilename(name)) throw Object.assign(new Error("Invalid log filename"), { statusCode: 400 });
      const logPath = path.join(logsDirectory, name);
      const info = await stat(logPath);
      if (!info.isFile() || info.size > 2 * 1024 * 1024) throw Object.assign(new Error("Log file is unavailable or too large"), { statusCode: 404 });
      response.statusCode = 200;
      response.end(JSON.stringify({ name, content: await readFile(logPath, "utf8") }));
      return;
    }
    if (request.method === "DELETE") {
      const logs = await listDiagnosticLogs();
      await Promise.all(logs.map((log) => rm(path.join(logsDirectory, log.name), { force: true })));
      response.statusCode = 200;
      response.end(JSON.stringify({ deleted: logs.length }));
      return;
    }
    if (request.method === "POST") {
      const payload = await readJsonRequest(request);
      const message = typeof payload.message === "string" ? payload.message.slice(0, 2000) : "Client-side generation failure";
      const stage = typeof payload.stage === "string" ? payload.stage.slice(0, 80) : "client";
      const details = payload.details && typeof payload.details === "object" && !Array.isArray(payload.details) ? payload.details : {};
      writeDiagnosticLog(projectRoot, {
        kind: "generation-client-failure",
        message,
        details: { stage, ...details },
      });
      response.statusCode = 201;
      response.end(JSON.stringify({ status: "logged" }));
      return;
    }
    throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
  } catch (error) {
    response.statusCode = error.code === "ENOENT" ? 404 : (error.statusCode || 500);
    response.end(JSON.stringify({ error: error.message || "Diagnostic log request failed" }));
  }
}

function logApiPlugin() {
  return {
    name: "local-diagnostic-log-api",
    configureServer(server) { server.middlewares.use(logApi); },
    configurePreviewServer(server) { server.middlewares.use(logApi); },
  };
}

async function uiStateApi(request, response, next) {
  const url = new URL(request.url, "http://localhost");
  if (url.pathname !== "/api/ui-state") {
    next();
    return;
  }
  response.setHeader("Content-Type", "application/json; charset=utf-8");
  response.setHeader("Cache-Control", "no-store");
  try {
    requireSameOrigin(request);
    if (request.method === "GET") {
      let state = null;
      try {
        const saved = JSON.parse(await readFile(uiStatePath, "utf8"));
        if (saved && typeof saved === "object" && !Array.isArray(saved)) state = saved;
      } catch (error) {
        if (error.code !== "ENOENT") throw Object.assign(new Error("Saved interface state is invalid"), { statusCode: 500 });
      }
      response.statusCode = 200;
      response.end(JSON.stringify({ state }));
      return;
    }
    if (request.method === "PUT") {
      const state = await readJsonRequest(request);
      if (!state || typeof state !== "object" || Array.isArray(state)) {
        throw Object.assign(new Error("Saved interface state must be an object"), { statusCode: 400 });
      }
      await mkdir(stateDirectory, { recursive: true });
      const temporaryPath = path.join(stateDirectory, `${randomUUID()}.tmp`);
      try {
        await writeFile(temporaryPath, `${JSON.stringify(state, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
        await rename(temporaryPath, uiStatePath);
      } finally {
        await rm(temporaryPath, { force: true });
      }
      response.statusCode = 204;
      response.end();
      return;
    }
    throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
  } catch (error) {
    response.statusCode = error.statusCode || 500;
    response.end(JSON.stringify({ error: error.message || "Saved interface state request failed" }));
  }
}

function uiStateApiPlugin() {
  return {
    name: "local-ui-state-api",
    configureServer(server) { server.middlewares.use(uiStateApi); },
    configurePreviewServer(server) { server.middlewares.use(uiStateApi); },
  };
}

// AI assistant proxy.
//
// The browser never talks to the AI provider directly. Routing through the control plane keeps the
// API key on disk instead of in the page, sidesteps the CORS policies that would block most
// providers from a browser origin, and reuses the proxy dispatcher the rest of the app already uses
// for outbound traffic. The upstream URL is user-supplied by design — pointing at a local runtime
// such as Ollama on 127.0.0.1 is a first-class case — so no host blocklist is applied here; the
// routes are instead restricted to local, same-origin callers.
const ASSISTANT_CHAT_TIMEOUT_MS = 240000;
const ASSISTANT_TEST_TIMEOUT_MS = 30000;
const ASSISTANT_SESSION_ROUTE_PATTERN = /^\/api\/assistant\/sessions\/([^/]+)$/;
const ASSISTANT_PERSONA_ROUTE_PATTERN = /^\/api\/assistant\/personas\/([^/]+)$/;
const ASSISTANT_PROFILE_ROUTE_PATTERN = /^\/api\/assistant\/profiles\/([^/]+)$/;
const ASSISTANT_PROFILE_ACTION_PATTERN = /^\/api\/assistant\/profiles\/([^/]+)\/(activate|duplicate)$/;

function assistantJson(response, statusCode, payload) {
  response.statusCode = statusCode;
  response.setHeader("Content-Type", "application/json; charset=utf-8");
  response.setHeader("Cache-Control", "no-store");
  response.end(JSON.stringify(payload));
}

// Transport failures are the "wrong URL" half of the graceful-failure contract: they must read as
// an actionable sentence, not as a Node error code.
function assistantNetworkMessage(error) {
  if (error?.name === "AbortError" || error?.name === "TimeoutError") return "请求超时或已被取消。";
  const code = error?.cause?.code || error?.code || "";
  if (code === "ENOTFOUND" || code === "EAI_AGAIN") return "无法解析服务地址的域名，请检查 URL 是否填写正确。";
  if (code === "ECONNREFUSED") return "服务地址拒绝连接，请确认服务已启动且端口正确。";
  if (code === "ETIMEDOUT" || code === "UND_ERR_CONNECT_TIMEOUT") return "连接服务地址超时，请检查网络或代理设置。";
  if (code === "ECONNRESET" || code === "UND_ERR_SOCKET") return "连接被重置，请检查网络或代理设置。";
  if (typeof code === "string" && code.startsWith("CERT_") || code === "UNABLE_TO_VERIFY_LEAF_SIGNATURE" || code === "DEPTH_ZERO_SELF_SIGNED_CERT") {
    return "HTTPS 证书校验失败，请检查服务地址。";
  }
  return "无法连接到 AI 服务，请检查服务地址与网络。";
}

async function assistantUpstream(settings, messages, { stream, signal }) {
  const endpoint = chatCompletionsUrl(settings.baseUrl);
  const headers = {
    "Content-Type": "application/json",
    Accept: stream ? "text/event-stream" : "application/json",
  };
  // Providers that need no credential (a local runtime) must not receive an empty bearer header.
  if (settings.apiKey) headers.Authorization = `Bearer ${settings.apiKey}`;
  const body = buildChatRequestBody({
    messages,
    model: settings.model,
    // Resolved per provider *and* model, so a reasoning model is never sent `temperature`.
    strength: strengthPayload(settings.provider, settings.model, settings.strength),
    stream,
  });
  return undiciFetch(endpoint, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
    dispatcher: getProxyDispatcher(),
    signal,
  });
}

async function assistantUpstreamFailure(upstream) {
  const text = await upstream.text().catch(() => "");
  let parsed = null;
  try {
    parsed = JSON.parse(text);
  } catch {
    parsed = null;
  }
  return providerErrorMessage(parsed, upstream.status);
}

// Dependencies are injectable so the route contract can be exercised against a temporary state
// directory and a stub provider, without a network call or a real API key.
export function assistantApiPlugin({
  stateDirectory: stateDirectoryOverride,
  projectRoot: projectRootOverride,
  upstreamRequest,
} = {}) {
  const assistantStateDirectory = stateDirectoryOverride || stateDirectory;
  const assistantProjectRoot = projectRootOverride || projectRoot;
  const callUpstream = upstreamRequest || assistantUpstream;

  const assistantApi = async (request, response, next) => {
  let pathname;
  try {
    pathname = new URL(request.url, "http://localhost").pathname;
  } catch {
    pathname = null;
  }
  if (!pathname || !pathname.startsWith("/api/assistant")) {
    next();
    return;
  }
  try {
    requireLocalRequest(request, "AI 助手仅允许本机访问");
    const settingsFile = assistantSettingsPath(assistantStateDirectory);

    if (pathname === "/api/assistant/settings" && request.method === "GET") {
      const stored = await readAssistantSettings(settingsFile);
      assistantJson(response, 200, { settings: redactAssistantSettings(stored), corrupt: stored.corrupt === true });
      return;
    }

    if (pathname === "/api/assistant/settings" && request.method === "PUT") {
      const stored = await readAssistantSettings(settingsFile);
      const merged = mergeAssistantSettings(stored, await readJsonRequest(request));
      const readiness = assistantReadiness(merged);
      // An invalid configuration is refused rather than stored, so the file on disk is always one
      // the client can actually use. The form keeps the user's typed values, so nothing is lost.
      if (!readiness.ready) {
        assistantJson(response, 400, { error: "配置无效，请修正标注的字段。", errors: readiness.errors });
        return;
      }
      await writeAssistantSettings(assistantStateDirectory, readiness.settings);
      assistantJson(response, 200, { settings: redactAssistantSettings(readiness.settings) });
      return;
    }

    // Profile management. Every response carries the whole redacted store rather than the one row
    // that changed: the settings page renders a list plus an editor from the same state, so a
    // partial reply would mean a second round trip before the page could redraw either.
    if (pathname === "/api/assistant/profiles") {
      if (request.method === "GET") {
        const store = await readAssistantProfileStore(settingsFile);
        assistantJson(response, 200, { ...redactAssistantProfileStore(store), corrupt: store.corrupt === true });
        return;
      }
      if (request.method === "POST") {
        const store = await readAssistantProfileStore(settingsFile);
        if (store.profiles.length >= MAXIMUM_PROFILES) {
          assistantJson(response, 409, { error: `最多只能保存 ${MAXIMUM_PROFILES} 套配置，请先删除不再使用的配置。` });
          return;
        }
        const seed = await readJsonRequest(request).catch(() => null);
        // Deliberately unvalidated: a profile is created empty and configured afterwards, so
        // refusing it for a missing API key would make "new profile" impossible.
        const created = await writeAssistantProfileStore(assistantStateDirectory, createAssistantProfile(store, seed));
        assistantJson(response, 201, { ...redactAssistantProfileStore(created), createdId: created.activeId });
        return;
      }
      throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
    }

    const profileActionMatch = ASSISTANT_PROFILE_ACTION_PATTERN.exec(pathname);
    if (profileActionMatch) {
      const [, profileId, action] = profileActionMatch;
      if (request.method !== "POST") throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
      const store = await readAssistantProfileStore(settingsFile);
      if (!validProfileId(profileId) || !assistantProfileAt(store, profileId)) {
        assistantJson(response, 404, { error: "配置不存在。" });
        return;
      }
      if (action === "duplicate" && store.profiles.length >= MAXIMUM_PROFILES) {
        assistantJson(response, 409, { error: `最多只能保存 ${MAXIMUM_PROFILES} 套配置，请先删除不再使用的配置。` });
        return;
      }
      const next = action === "activate"
        ? activateAssistantProfile(store, profileId)
        : duplicateAssistantProfile(store, profileId);
      const saved = await writeAssistantProfileStore(assistantStateDirectory, next);
      assistantJson(response, 200, { ...redactAssistantProfileStore(saved), createdId: action === "duplicate" ? saved.activeId : "" });
      return;
    }

    const profileMatch = ASSISTANT_PROFILE_ROUTE_PATTERN.exec(pathname);
    if (profileMatch) {
      const profileId = profileMatch[1];
      const store = await readAssistantProfileStore(settingsFile);
      const existing = validProfileId(profileId) ? assistantProfileAt(store, profileId) : null;
      if (!existing) {
        assistantJson(response, 404, { error: "配置不存在。" });
        return;
      }
      if (request.method === "PUT") {
        const body = await readJsonRequest(request);
        const renaming = Object.prototype.hasOwnProperty.call(body || {}, "name");
        const patch = renaming ? { name: body.name } : {};
        if (body?.settings) {
          // Key presence is resolved against *this* profile's stored secret, not the active one,
          // so saving a profile you are not currently using cannot inherit another's key.
          const readiness = assistantReadiness(mergeAssistantSettings(existing.settings, body.settings));
          if (!readiness.ready) {
            assistantJson(response, 400, { error: "配置无效，请修正标注的字段。", errors: readiness.errors });
            return;
          }
          patch.settings = readiness.settings;
        } else if (!renaming) {
          assistantJson(response, 400, { error: "请求没有需要更新的内容。" });
          return;
        }
        const saved = await writeAssistantProfileStore(assistantStateDirectory, updateAssistantProfile(store, profileId, patch));
        assistantJson(response, 200, redactAssistantProfileStore(saved));
        return;
      }
      if (request.method === "DELETE") {
        // The store always holds at least one profile; an empty list would leave the chat surface
        // with no configuration to read and the page with no row to edit.
        if (store.profiles.length <= 1) {
          assistantJson(response, 409, { error: "至少需要保留一套配置。" });
          return;
        }
        const saved = await writeAssistantProfileStore(assistantStateDirectory, removeAssistantProfile(store, profileId));
        assistantJson(response, 200, { ...redactAssistantProfileStore(saved), deleted: profileId });
        return;
      }
      throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
    }

    // The character library. Shipped characters are read-only — an update replaces that directory
    // wholesale — so editing one is refused and copying it is the offered path instead. Every
    // mutation replies with the refreshed library so the interface redraws from one round trip.
    const personaLibrary = () => readPersonaDirectory(assistantProjectRoot, assistantStateDirectory);

    if (pathname === "/api/assistant/personas") {
      if (request.method === "GET") {
        const { personas, diagnostics, available } = await personaLibrary();
        assistantJson(response, 200, { personas, diagnostics, available });
        return;
      }
      if (request.method === "POST") {
        const body = await readJsonRequest(request).catch(() => ({}));
        // `fromId` copies an existing character — the only way to start from a built-in.
        let seed = body;
        if (typeof body?.fromId === "string" && body.fromId) {
          const source = (await personaLibrary()).personas.find((persona) => persona.id === body.fromId);
          if (!source) {
            assistantJson(response, 404, { error: "角色不存在。" });
            return;
          }
          const { id, builtIn, ...copyable } = source;
          seed = { ...copyable, name: typeof body.name === "string" && body.name.trim() ? body.name : source.name };
        }
        const readiness = validateAssistantPersona(seed);
        if (!readiness.valid) {
          assistantJson(response, 400, { error: "角色配置无效，请修正标注的字段。", errors: readiness.errors });
          return;
        }
        let created;
        try {
          created = await createUserPersona(assistantStateDirectory, readiness.persona);
        } catch (error) {
          if (error.statusCode !== 409) throw error;
          assistantJson(response, 409, { error: error.message, errors: [{ field: "name", code: "persona_limit_reached", message: error.message }] });
          return;
        }
        const { personas, diagnostics, available } = await personaLibrary();
        assistantJson(response, 201, { persona: created, createdId: created.id, personas, diagnostics, available });
        return;
      }
      throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
    }

    const personaMatch = ASSISTANT_PERSONA_ROUTE_PATTERN.exec(pathname);
    if (personaMatch) {
      const personaId = personaMatch[1];
      if (request.method !== "PUT" && request.method !== "DELETE") {
        throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
      }
      // A shipped character has no file in the state directory, so this same check covers "unknown"
      // and "read-only" — but they are answered differently, because "复制后再修改" is only useful
      // advice when the character actually exists.
      if (!validUserPersonaId(personaId) || !(await readUserPersona(assistantStateDirectory, personaId))) {
        const known = (await personaLibrary()).personas.some((persona) => persona.id === personaId);
        assistantJson(response, known ? 409 : 404, known
          ? { error: "内置角色不可修改或删除，请复制一份后再编辑。", errors: [{ field: "name", code: "persona_readonly", message: "内置角色不可修改或删除。" }] }
          : { error: "角色不存在。" });
        return;
      }
      if (request.method === "PUT") {
        const readiness = validateAssistantPersona({ ...await readJsonRequest(request), id: personaId });
        if (!readiness.valid) {
          assistantJson(response, 400, { error: "角色配置无效，请修正标注的字段。", errors: readiness.errors });
          return;
        }
        const saved = await saveUserPersona(assistantStateDirectory, personaId, readiness.persona);
        if (!saved) {
          assistantJson(response, 404, { error: "角色不存在。" });
          return;
        }
        const { personas, diagnostics, available } = await personaLibrary();
        assistantJson(response, 200, { persona: saved, personas, diagnostics, available });
        return;
      }
      await deleteUserPersona(assistantStateDirectory, personaId);
      const { personas, diagnostics, available } = await personaLibrary();
      assistantJson(response, 200, { deleted: personaId, personas, diagnostics, available });
      return;
    }

    if (pathname === "/api/assistant/models" && request.method === "POST") {
      const settings = await readAssistantSettings(settingsFile);
      // Listing does not need a model name — that is the thing being looked up — so only the
      // endpoint and credential errors block it.
      const blocking = assistantReadiness(settings).errors.filter((entry) => entry.field === "baseUrl" || entry.field === "apiKey");
      if (blocking.length) {
        assistantJson(response, 400, { error: "请先填写服务地址与 API Key。", errors: blocking });
        return;
      }
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), ASSISTANT_TEST_TIMEOUT_MS);
      try {
        const headers = { Accept: "application/json" };
        if (settings.apiKey) headers.Authorization = `Bearer ${settings.apiKey}`;
        const upstream = await undiciFetch(modelsUrl(settings.baseUrl), {
          method: "GET",
          headers,
          dispatcher: getProxyDispatcher(),
          signal: controller.signal,
        });
        if (!upstream.ok) {
          assistantJson(response, 502, { error: await assistantUpstreamFailure(upstream), status: upstream.status });
          return;
        }
        const models = parseModelList(await upstream.json());
        assistantJson(response, 200, { models });
      } catch (error) {
        assistantJson(response, 502, { error: assistantNetworkMessage(error) });
      } finally {
        clearTimeout(timeout);
      }
      return;
    }

    if (pathname === "/api/assistant/sessions") {
      if (request.method === "GET") {
        assistantJson(response, 200, { sessions: await listSessions(assistantStateDirectory) });
        return;
      }
      if (request.method === "POST") {
        // A seed lets the client hand over the pre-session transcript once, during migration.
        const created = await createSession(assistantStateDirectory, await readJsonRequest(request).catch(() => null));
        assistantJson(response, 201, { session: created });
        return;
      }
      throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
    }

    const sessionMatch = ASSISTANT_SESSION_ROUTE_PATTERN.exec(pathname);
    if (sessionMatch) {
      const sessionId = sessionMatch[1];
      // Rejected before the id can reach `path.join`, so a traversal sequence never becomes a path.
      if (!validSessionId(sessionId)) {
        assistantJson(response, 400, { error: "会话标识无效。" });
        return;
      }
      if (request.method === "GET") {
        const session = await readSession(assistantStateDirectory, sessionId);
        if (!session) {
          assistantJson(response, 404, { error: "会话不存在。" });
          return;
        }
        assistantJson(response, 200, { session });
        return;
      }
      if (request.method === "PUT") {
        const saved = await saveSession(assistantStateDirectory, sessionId, await readJsonRequest(request));
        if (!saved) {
          assistantJson(response, 404, { error: "会话不存在。" });
          return;
        }
        assistantJson(response, 200, { session: saved });
        return;
      }
      if (request.method === "DELETE") {
        await deleteSession(assistantStateDirectory, sessionId);
        assistantJson(response, 200, { deleted: sessionId });
        return;
      }
      throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
    }

    if ((pathname === "/api/assistant/chat" || pathname === "/api/assistant/test") && request.method === "POST") {
      const probe = pathname === "/api/assistant/test";
      const payload = await readJsonRequest(request);
      const settings = await readAssistantSettings(settingsFile);
      const readiness = assistantReadiness(settings);
      if (!readiness.ready) {
        assistantJson(response, 400, { error: "AI 助手尚未完成配置。", errors: readiness.errors });
        return;
      }
      const { personas } = await readPersonaDirectory(assistantProjectRoot, assistantStateDirectory);
      const persona = selectPersona(personas, settings.personaId);
      const messages = probe
        ? [{ role: "user", content: "ping" }]
        : buildChatMessages({ persona, history: payload.history, knowledge: [] });
      if (!messages.some((message) => message.role === "user")) {
        assistantJson(response, 400, { error: "没有可发送的对话内容。" });
        return;
      }
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), probe ? ASSISTANT_TEST_TIMEOUT_MS : ASSISTANT_CHAT_TIMEOUT_MS);
      // A user who closes the panel mid-answer should stop paying for tokens immediately.
      const abortOnDisconnect = () => controller.abort();
      request.on("aborted", abortOnDisconnect);
      response.on("close", abortOnDisconnect);
      let upstream;
      try {
        upstream = await callUpstream(settings, messages, { stream: !probe, signal: controller.signal });
      } catch (error) {
        clearTimeout(timeout);
        assistantJson(response, 502, { error: assistantNetworkMessage(error) });
        return;
      }
      if (!upstream.ok) {
        clearTimeout(timeout);
        assistantJson(response, 502, { error: await assistantUpstreamFailure(upstream), status: upstream.status });
        return;
      }
      if (probe) {
        clearTimeout(timeout);
        assistantJson(response, 200, { ok: true, model: settings.model, provider: settings.provider });
        return;
      }
      response.statusCode = 200;
      response.setHeader("Content-Type", "text/event-stream; charset=utf-8");
      response.setHeader("Cache-Control", "no-cache, no-transform");
      response.setHeader("Connection", "keep-alive");
      // Defeats reverse-proxy response buffering, which would otherwise hold the whole answer back
      // and defeat the point of streaming.
      response.setHeader("X-Accel-Buffering", "no");
      try {
        for await (const chunk of upstream.body) {
          if (response.writableEnded) break;
          response.write(chunk);
        }
        if (!response.writableEnded) response.end();
      } catch (error) {
        // Headers are already sent, so a late failure has to be reported inside the stream. The
        // client decoder recognises an `error` payload frame.
        if (!response.writableEnded) {
          response.write(`data: ${JSON.stringify({ error: { message: assistantNetworkMessage(error) } })}\n\n`);
          response.end();
        }
      } finally {
        clearTimeout(timeout);
      }
      return;
    }

    throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
  } catch (error) {
    if (response.headersSent) {
      if (!response.writableEnded) response.end();
      return;
    }
    assistantJson(response, error.statusCode || 500, { error: error.message || "AI 助手请求失败" });
  }
  };

  return {
    name: "local-assistant-api",
    configureServer(server) { server.middlewares.use(assistantApi); },
    configurePreviewServer(server) { server.middlewares.use(assistantApi); },
  };
}

const PLUGIN_ROUTE_PATTERN = /^\/api\/plugins\/([^/]+)(?:\/(reveal))?$/;

function pluginRegistryApi({ readRegistry, setPluginEnabled, revealPlugin, removePlugin }) {
  return async (request, response, next) => {
    let pathname;
    try {
      pathname = new URL(request.url, "http://localhost").pathname;
    } catch {
      pathname = null;
    }
    const pluginMatch = pathname && PLUGIN_ROUTE_PATTERN.exec(pathname);
    if (pathname === "/api/plugins" || pluginMatch) {
      response.setHeader("Content-Type", "application/json; charset=utf-8");
      response.setHeader("Cache-Control", "no-store");
      try {
        if (pluginMatch) {
          // Managing a plugin is a local decision about a folder on this machine, so every write
          // takes the same loopback plus same-origin check as the console and update routes.
          requireLocalRequest(request, "Plugin management is available only from this computer");
          let id;
          try {
            id = decodeURIComponent(pluginMatch[1]);
          } catch {
            throw Object.assign(new Error("Plugin was not found"), { statusCode: 404, publicCode: "plugin_not_found" });
          }
          if (pluginMatch[2] === "reveal") {
            if (request.method !== "POST") throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
            response.statusCode = 200;
            response.end(JSON.stringify(await revealPlugin(id)));
            return;
          }
          if (request.method === "DELETE") {
            response.statusCode = 200;
            response.end(JSON.stringify(await removePlugin(id)));
            return;
          }
          if (request.method !== "PUT") throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
          const payload = await readJsonRequest(request);
          if (typeof payload?.enabled !== "boolean") {
            throw Object.assign(new Error("Field enabled must be a boolean"), { statusCode: 400, publicCode: "invalid_request" });
          }
          response.statusCode = 200;
          response.end(JSON.stringify(await setPluginEnabled(id, payload.enabled)));
          return;
        }
        requireSameOrigin(request);
        if (request.method !== "GET") throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
        response.statusCode = 200;
        response.end(JSON.stringify(await readRegistry()));
      } catch (error) {
        // A registry failure returns a stable message: never a path, manifest content, or stack.
        response.statusCode = error.statusCode || 500;
        response.end(JSON.stringify({
          error: error.statusCode ? error.message : "Plugin registry is unavailable",
          ...(error.publicCode ? { code: error.publicCode } : {}),
        }));
      }
      return;
    }
    // Plugin folders hold user program code that this version never executes, imports, or serves.
    // Registering in `configureServer` puts this ahead of Vite's transform, static, `/@fs/`, and
    // SPA fallback middlewares, so no plugin byte can reach the browser through any of them.
    if (servesPluginContent(request.url, projectRoot)) {
      response.statusCode = 404;
      response.setHeader("Content-Type", "application/json; charset=utf-8");
      response.setHeader("Cache-Control", "no-store");
      response.end(JSON.stringify({ error: "Plugin files are not served" }));
      return;
    }
    next();
  };
}

function pluginSnapshot(enabled, readable = true) {
  return createPluginRegistry({ projectRoot, loadState: async () => ({ enabled, readable }) }).read();
}

async function requirePluginEntry(id) {
  const state = await readPluginState({ statePath: pluginStatePath });
  const snapshot = await pluginSnapshot(state.enabled, state.readable);
  const entry = snapshot.plugins.find((plugin) => plugin.id === id);
  if (!entry) throw Object.assign(new Error("Plugin was not found"), { statusCode: 404, publicCode: "plugin_not_found" });
  return { entry, state };
}

function pluginActionFailure(error, fallback) {
  // Registry diagnostics are already stable machine codes; anything else is reported generically so
  // no path, stack, or filesystem detail reaches the browser.
  if (error.statusCode) return error;
  const code = typeof error.code === "string" && PLUGIN_DIAGNOSTIC_CODES.includes(error.code) ? error.code : null;
  return Object.assign(new Error(fallback), { statusCode: code ? 409 : 500, ...(code ? { publicCode: code } : {}) });
}

async function setPluginEnabledState(id, enabled) {
  const { entry, state } = await requirePluginEntry(id);
  if (!state.readable) {
    // Never rewrite a preference file this build cannot parse; the user's data outranks the toggle.
    throw Object.assign(new Error("Saved plugin state is invalid; repair or remove state-cache/plugins.json"), {
      statusCode: 409, publicCode: "plugin_state_unreadable",
    });
  }
  const admission = pluginToggleAdmission(entry, enabled);
  if (!admission.allowed) {
    throw Object.assign(new Error(admission.message), { statusCode: admission.statusCode, publicCode: admission.code });
  }
  const nextEnabled = applyPluginEnabled(state.enabled, id, enabled);
  await writePluginState({ stateDirectory, statePath: pluginStatePath, enabled: nextEnabled });
  return pluginSnapshot(nextEnabled);
}

async function revealPluginFolderAction(id) {
  // Any listed plugin may be opened, including an invalid one: looking at the folder is exactly what
  // a user needs when a manifest fails to parse.
  await requirePluginEntry(id);
  try {
    await revealPluginFolder({ projectRoot, id });
  } catch (error) {
    throw pluginActionFailure(error, "Could not open the plugin folder");
  }
  return { id, revealed: true };
}

async function removePluginAction(id) {
  const { state } = await requirePluginEntry(id);
  try {
    await removePluginFolder({ projectRoot, id });
  } catch (error) {
    throw pluginActionFailure(error, "Could not remove the plugin folder");
  }
  // Drop the stored preference too, so a later folder with the same id is not silently pre-enabled.
  // A failure here must not resurrect the deleted folder, so the removal still counts as done.
  let enabled = state.enabled;
  if (state.readable && state.enabled.includes(id)) {
    enabled = applyPluginEnabled(state.enabled, id, false);
    await writePluginState({ stateDirectory, statePath: pluginStatePath, enabled }).catch(() => { enabled = state.enabled; });
  }
  return pluginSnapshot(enabled, state.readable);
}

export function pluginRegistryApiPlugin({
  readRegistry: injectedReadRegistry,
  setPluginEnabled: injectedSetPluginEnabled,
  revealPlugin: injectedRevealPlugin,
  removePlugin: injectedRemovePlugin,
} = {}) {
  const registry = createPluginRegistry({
    projectRoot,
    loadState: () => readPluginState({ statePath: pluginStatePath }),
  });
  const middleware = pluginRegistryApi({
    readRegistry: injectedReadRegistry || (() => registry.read()),
    setPluginEnabled: injectedSetPluginEnabled || setPluginEnabledState,
    revealPlugin: injectedRevealPlugin || revealPluginFolderAction,
    removePlugin: injectedRemovePlugin || removePluginAction,
  });
  return {
    name: "local-plugin-registry-api",
    configureServer(server) { server.middlewares.use(middleware); },
    configurePreviewServer(server) { server.middlewares.use(middleware); },
  };
}

async function readYoloCatalog() {
  let payload;
  try {
    payload = JSON.parse(await readFile(yoloCatalogPath, "utf8"));
  } catch {
    throw Object.assign(new Error("YOLO model catalog is unavailable"), { statusCode: 500 });
  }
  if (!Array.isArray(payload.models) || typeof payload.repository !== "string" || typeof payload.revision !== "string") {
    throw Object.assign(new Error("YOLO model catalog is invalid"), { statusCode: 500 });
  }
  const models = payload.models.map((model) => ({
    name: model?.name,
    label: model?.label || model?.name,
    description: model?.description || "",
  })).filter((model) => typeof model.name === "string"
    && path.basename(model.name) === model.name
    && path.extname(model.name).toLowerCase() === ".pt");
  if (models.length !== payload.models.length) throw Object.assign(new Error("YOLO model catalog contains an invalid filename"), { statusCode: 500 });
  return { repository: payload.repository, revision: payload.revision, models };
}

async function readBackgroundRemovalCatalog() {
  let payload;
  try {
    payload = JSON.parse(await readFile(backgroundRemovalCatalogPath, "utf8"));
  } catch {
    throw Object.assign(new Error("透明背景模型清单不可用"), { statusCode: 500 });
  }
  if (payload?.schema !== 3 || !Array.isArray(payload.models) || !payload.models.length) {
    throw Object.assign(new Error("透明背景模型清单无效"), { statusCode: 500 });
  }
  const models = payload.models.map((model) => ({ ...model })).filter((model) => (
    typeof model.id === "string"
    && typeof model.repository === "string"
    && typeof model.revision === "string"
    && typeof model.filename === "string" && path.basename(model.filename) === model.filename
    && path.extname(model.filename).toLowerCase() === ".onnx"
    && typeof model.source_path === "string" && !model.source_path.includes("..")
    && typeof model.sha256 === "string" && /^[a-f0-9]{64}$/i.test(model.sha256)
    && Number.isSafeInteger(model.size) && model.size > 0 && model.size <= 1024 * 1024 ** 2
    && (model.modelscope_repository === undefined || typeof model.modelscope_repository === "string")
    && (model.modelscope_revision === undefined || typeof model.modelscope_revision === "string")
  ));
  if (models.length !== payload.models.length || new Set(models.map((model) => model.id)).size !== models.length) {
    throw Object.assign(new Error("透明背景模型清单包含无效或重复模型"), { statusCode: 500 });
  }
  return { ...payload, models: models.sort((first, second) => (second.priority || 0) - (first.priority || 0)) };
}

function backgroundRemovalDownloadRoutes(model) {
  const sourcePath = model.source_path.split("/").map(encodeURIComponent).join("/");
  const relativePath = `${model.repository}/resolve/${model.revision}/${sourcePath}?download=true`;
  const routes = [
    { id: "huggingface", label: "Hugging Face", url: `https://huggingface.co/${relativePath}` },
    { id: "hf-mirror", label: "HF-Mirror", url: `https://hf-mirror.com/${relativePath}` },
  ];
  if (model.modelscope_repository) {
    const repository = model.modelscope_repository.split("/").map(encodeURIComponent).join("/");
    const revision = encodeURIComponent(model.modelscope_revision || "master");
    routes.push({ id: "modelscope", label: "ModelScope", url: `https://modelscope.cn/models/${repository}/resolve/${revision}/${sourcePath}` });
  }
  return routes;
}

function yoloDownloadRoutes(catalog, model) {
  const relativePath = `${catalog.repository}/resolve/${catalog.revision}/${encodeURIComponent(model.name)}?download=true`;
  return [
    { id: "huggingface", label: "Hugging Face", url: `https://huggingface.co/${relativePath}` },
    { id: "hf-mirror", label: "HF-Mirror", url: `https://hf-mirror.com/${relativePath}` },
  ];
}

function writeDownloadEvent(response, event) {
  response.write(`${JSON.stringify(event)}\n`);
}

let yoloDownloadActive = false;
let backgroundRemovalDownloadActive = false;
let backgroundRemovalDownloadJob = null;
let activeModelDownloadJob = null;
let storedModelDownloadJob = null;
let recommendedCatalogCache;
let recommendedCatalogCachedAt = 0;
const recommendedHashCache = new Map();
const recommendedCatalogCacheLifetime = 10 * 60 * 1000;
const recommendedCivitaiTimeout = 5000;
const modelDownloadJobPath = path.join(stateDirectory, "model-download-job.json");
const backgroundRemovalDownloadJobPath = path.join(stateDirectory, "background-removal-download-job.json");
const activeDownloadStatuses = new Set(["queued", "resolving", "downloading", "metadata"]);
const backgroundRemovalActiveStatuses = new Set(["queued", "downloading"]);

async function writeBackgroundRemovalDownloadJob() {
  if (!backgroundRemovalDownloadJob) return;
  await mkdir(stateDirectory, { recursive: true });
  const temporaryPath = path.join(stateDirectory, `${randomUUID()}.tmp`);
  try {
    await writeFile(temporaryPath, `${JSON.stringify({ schema: 1, ...backgroundRemovalDownloadJob }, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
    await rename(temporaryPath, backgroundRemovalDownloadJobPath);
  } finally {
    await rm(temporaryPath, { force: true });
  }
}

async function loadBackgroundRemovalDownloadJob() {
  try {
    const saved = JSON.parse(await readFile(backgroundRemovalDownloadJobPath, "utf8"));
    if (!saved || saved.schema !== 1 || typeof saved.id !== "string" || typeof saved.status !== "string") return;
    const { schema: _schema, ...job } = saved;
    backgroundRemovalDownloadJob = job;
    if (backgroundRemovalActiveStatuses.has(job.status)) {
      backgroundRemovalDownloadJob = {
        ...job,
        status: "error",
        active: false,
        message: "下载服务已重启；断点文件已保留，请重新启动下载以续传。",
        updatedAt: Date.now(),
      };
      await writeBackgroundRemovalDownloadJob();
    }
  } catch (error) {
    if (error.code !== "ENOENT") console.warn(`Unable to restore background-removal download state: ${error.message}`);
  }
}

const backgroundRemovalDownloadJobReady = loadBackgroundRemovalDownloadJob();
let backgroundRemovalPersistTimer = null;

function updateBackgroundRemovalDownloadJob(updates, immediate = false) {
  if (!backgroundRemovalDownloadJob) return;
  backgroundRemovalDownloadJob = { ...backgroundRemovalDownloadJob, ...updates, updatedAt: Date.now() };
  if (immediate) {
    if (backgroundRemovalPersistTimer) clearTimeout(backgroundRemovalPersistTimer);
    backgroundRemovalPersistTimer = null;
    void writeBackgroundRemovalDownloadJob().catch((error) => console.warn(`Unable to save background-removal download state: ${error.message}`));
  } else if (!backgroundRemovalPersistTimer) {
    backgroundRemovalPersistTimer = setTimeout(() => {
      backgroundRemovalPersistTimer = null;
      void writeBackgroundRemovalDownloadJob().catch((error) => console.warn(`Unable to save background-removal download state: ${error.message}`));
    }, 500);
  }
}

function modelDownloadJobIsActive(job) {
  return Boolean(job && (activeDownloadStatuses.has(job.state?.status) || job.workerActive || job.workerScheduled || job.pendingBatches?.length));
}

function publicModelDownloadJob(job) {
  if (!job?.state) return null;
  return {
    ...job.state,
    retryAvailable: Boolean(job.batches?.length && job.state.items?.some((item) => item.status === "error")),
    jobId: job.id,
    source: job.source,
    active: modelDownloadJobIsActive(job),
    createdAt: job.createdAt,
    updatedAt: job.updatedAt,
  };
}

async function writeModelDownloadJob(job) {
  await mkdir(stateDirectory, { recursive: true });
  const temporaryPath = path.join(stateDirectory, `${randomUUID()}.tmp`);
  const payload = {
    schema: 1,
    id: job.id,
    source: job.source,
    createdAt: job.createdAt,
    updatedAt: job.updatedAt,
    state: job.state,
  };
  try {
    await writeFile(temporaryPath, `${JSON.stringify(payload, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
    await rename(temporaryPath, modelDownloadJobPath);
  } finally {
    await rm(temporaryPath, { force: true });
  }
}

async function loadModelDownloadJob() {
  try {
    const saved = JSON.parse(await readFile(modelDownloadJobPath, "utf8"));
    if (!saved || saved.schema !== 1 || typeof saved.id !== "string" || !saved.state || typeof saved.state !== "object") return;
    storedModelDownloadJob = {
      id: saved.id,
      source: saved.source || "manual",
      createdAt: Number(saved.createdAt) || Date.now(),
      updatedAt: Number(saved.updatedAt) || Date.now(),
      state: saved.state,
      persistTimer: null,
      persistPromise: null,
    };
    const items = storedModelDownloadJob.state.items || [];
    const allComplete = items.length > 0 && items.every((item) => item.status === "complete");
    if (allComplete || modelDownloadJobIsActive(storedModelDownloadJob)) {
      storedModelDownloadJob.state = {
        ...storedModelDownloadJob.state,
        status: allComplete ? "complete" : "error",
        speedBps: 0,
        connections: 0,
        message: allComplete ? `${items.length} 个模型均已下载完成` : "下载服务已重启；已保留断点文件，重新启动同一模型下载即可续传。",
      };
      storedModelDownloadJob.updatedAt = Date.now();
      await writeModelDownloadJob(storedModelDownloadJob);
    }
  } catch (error) {
    if (error.code !== "ENOENT") console.warn(`Unable to restore model download state: ${error.message}`);
  }
}

const modelDownloadJobReady = loadModelDownloadJob();

function persistModelDownloadJob(job, immediate = false) {
  job.updatedAt = Date.now();
  if (immediate) {
    if (job.persistTimer) clearTimeout(job.persistTimer);
    job.persistTimer = null;
    job.persistPromise = (job.persistPromise || Promise.resolve())
      .then(() => writeModelDownloadJob(job))
      .catch((error) => console.warn(`Unable to save model download state: ${error.message}`));
    return job.persistPromise;
  }
  if (job.persistTimer) return;
  job.persistTimer = setTimeout(() => {
    job.persistTimer = null;
    void persistModelDownloadJob(job, true);
  }, 750);
}

function modelDownloadItemLabel(item) {
  try {
    const parsed = new URL(item);
    return `${parsed.host}${parsed.pathname}`;
  } catch {
    return String(item).slice(0, 240);
  }
}

async function enqueueModelDownloadBatch({ source, kind, engine, items, run }) {
  await modelDownloadJobReady;
  let job = activeModelDownloadJob;
  let isNew = false;
  if (!job || (!modelDownloadJobIsActive(job) && !job.workerActive && !job.workerScheduled)) {
    isNew = true;
    const now = Date.now();
    job = {
      id: randomUUID(),
      source,
      createdAt: now,
      updatedAt: now,
      persistTimer: null,
      persistPromise: null,
      pendingBatches: [],
      batches: [],
      workerActive: false,
      workerScheduled: false,
      state: {
        status: "queued",
        kind,
        engine: engine || "",
        message: "下载任务已交给本地服务，可安全关闭或刷新页面。",
        currentBytes: 0,
        totalBytes: 0,
        speedBps: 0,
        connections: 0,
        route: "",
        totalModels: 0,
        completedModels: 0,
        failedModels: 0,
        items: [],
        targets: [],
      },
    };
    activeModelDownloadJob = job;
    storedModelDownloadJob = job;
  }
  const wasActive = !isNew && (modelDownloadJobIsActive(job) || job.workerActive || job.workerScheduled);
  const acceptedItems = items.filter((item) => !item.sha256 || !job.state.items.some((queued) => queued.sha256 === item.sha256 && queued.status !== "complete"));
  if (!acceptedItems.length) return { job, addedCount: 0 };
  const appended = appendDownloadQueueState(job.state, { source, kind, engine, items: acceptedItems });
  job.state = {
    ...appended.state,
    status: wasActive && activeDownloadStatuses.has(job.state.status) ? job.state.status : "queued",
    message: wasActive ? `已追加 ${acceptedItems.length} 个项目，当前任务完成后自动继续` : job.state.message,
  };
  job.pendingBatches ||= [];
  job.batches ||= [];
  const batch = { startIndex: appended.startIndex, itemCount: acceptedItems.length, items: acceptedItems, run };
  job.batches.push(batch);
  job.pendingBatches.push(batch);
  if (job.source !== source) job.source = "mixed";
  await persistModelDownloadJob(job, true);
  queueModelDownloadWorker(job);
  return { job, addedCount: acceptedItems.length };
}

async function createCompletedModelDownloadJob({ source, kind, engine, message, targets = [] }) {
  await modelDownloadJobReady;
  const now = Date.now();
  const job = {
    id: randomUUID(),
    source,
    createdAt: now,
    updatedAt: now,
    persistTimer: null,
    persistPromise: null,
    state: {
      status: "complete",
      kind,
      engine: engine || "",
      message,
      currentBytes: 0,
      totalBytes: 0,
      speedBps: 0,
      connections: 0,
      route: "",
      totalModels: 0,
      completedModels: 0,
      failedModels: 0,
      items: [],
      targets,
    },
  };
  storedModelDownloadJob = job;
  await persistModelDownloadJob(job, true);
  return job;
}

function updateModelDownloadJob(job, event) {
  const current = job.state;
  const modelIndex = event.model_index || event.index || current.modelIndex;
  const modelItem = (current.items || []).find((item) => item.index === modelIndex);
  const staleItemEvent = modelItem && itemStatusIsTerminal(modelItem.status) && !["starting", "complete", "error"].includes(event.type);
  if (staleItemEvent) return;
  const updateItem = (changes) => (current.items || []).map((item) => item.index === modelIndex ? { ...item, ...changes } : item);
  if (event.type === "starting") {
    job.state = { ...current, status: "resolving", message: `准备处理 ${current.totalModels - (current.completedModels || 0) - (current.failedModels || 0)} 个队列项目` };
  } else if (event.type === "model") {
    job.state = { ...current, status: "resolving", kind: modelItem?.kind || current.kind, engine: modelItem?.engine || "", modelIndex, currentBytes: 0, totalBytes: 0, speedBps: 0, connections: 0, route: "", filename: "", destination: "", message: `正在准备 ${modelIndex} / ${current.totalModels}`, items: updateItem({ status: "resolving" }) };
  } else if (event.type === "resolving") {
    job.state = { ...current, status: "resolving", provider: event.provider, modelIndex, message: event.message, items: updateItem({ status: "resolving" }) };
  } else if (event.type === "resolved") {
    job.state = { ...current, status: "downloading", provider: event.provider || current.provider, modelIndex, filename: event.filename, destination: event.destination, verified: event.verified, message: "正在测速可用下载线路...", items: updateItem({ status: "downloading", filename: event.filename, destination: event.destination }) };
  } else if (event.type === "route") {
    job.state = { ...current, status: "downloading", modelIndex, route: event.label, connections: event.connections || 0, message: event.cached ? "已验证本地同版本模型" : `已选择 ${event.label} 下载线路`, items: updateItem({ route: event.label }) };
  } else if (event.type === "progress") {
    job.state = { ...current, status: "downloading", modelIndex, currentBytes: event.current_bytes || 0, totalBytes: event.total_bytes || 0, speedBps: event.speed_bps || 0, connections: event.connections || 0, route: event.route || current.route, message: event.cached ? "本地模型已可用" : "正在下载模型...", items: updateItem({ currentBytes: event.current_bytes || 0, totalBytes: event.total_bytes || 0 }) };
  } else if (event.type === "metadata") {
    job.state = { ...current, status: "metadata", modelIndex, message: event.message || "正在同步 LoRA 信息...", metadataStatus: event.status, metadataName: event.model_name, items: updateItem({ status: "metadata" }) };
  } else if (event.type === "warning") {
    job.state = { ...current, warning: event.message, message: event.message };
  } else if (event.type === "model-complete") {
    job.state = { ...current, completedModels: (current.completedModels || 0) + 1, filename: event.filename || current.filename, destination: event.destination || current.destination, cached: Boolean(event.cached), message: event.cached ? "模型已存在且校验通过" : "模型已下载到对应目录", items: updateItem({ status: "complete", filename: event.filename || current.filename, destination: event.destination || current.destination }) };
  } else if (event.type === "model-error") {
    job.state = { ...current, failedModels: (current.failedModels || 0) + 1, warning: event.error, message: `第 ${modelIndex} 个模型下载失败，继续后续任务`, items: updateItem({ status: "error", error: event.error }) };
  } else if (event.type === "complete") {
    job.state = { ...current, status: event.failed_models ? "partial" : "complete", completedModels: event.completed_models ?? current.completedModels, failedModels: event.failed_models ?? current.failedModels, message: event.failed_models ? `${event.completed_models} 个模型完成，${event.failed_models} 个失败` : `${event.completed_models} 个模型均已下载完成` };
  } else if (event.type === "error") {
    job.state = { ...current, status: "error", message: event.error || "模型下载失败" };
  }
  persistModelDownloadJob(job, ["complete", "error"].includes(event.type));
}

function queueModelDownloadWorker(job) {
  if (job.workerScheduled || job.workerActive) return;
  job.workerScheduled = true;
  setImmediate(() => { void runModelDownloadQueue(job); });
}

async function runModelDownloadQueue(job) {
  if (job.workerActive) return;
  job.workerScheduled = false;
  job.workerActive = true;
  const response = createModelDownloadJobResponse(job);
  try {
    while (job.pendingBatches?.length) {
      const batch = job.pendingBatches.shift();
      try {
        await batch.run(job, response, batch.startIndex, batch.retryIndexes, batch.retryInput, batch.items);
      } catch (error) {
        for (let offset = 0; offset < batch.itemCount; offset += 1) {
          const modelIndex = batch.startIndex + offset;
          const item = job.state.items.find((candidate) => candidate.index === modelIndex);
          if (!itemStatusIsTerminal(item?.status)) updateModelDownloadJob(job, { type: "model-error", model_index: modelIndex, error: error.message || "模型下载失败" });
        }
      }
    }
    const failedModels = job.state.failedModels || 0;
    job.state = {
      ...job.state,
      status: failedModels ? "partial" : "complete",
      active: false,
      speedBps: 0,
      connections: 0,
      message: failedModels ? `${job.state.completedModels || 0} 个模型完成，${failedModels} 个失败` : `${job.state.completedModels || 0} 个模型均已下载完成`,
    };
  } finally {
    if (job.pendingBatches?.length) {
      job.workerActive = false;
      job.state = { ...job.state, status: "queued", message: `${job.pendingBatches.length} 个新增批次正在等待处理` };
      await persistModelDownloadJob(job, true);
      queueModelDownloadWorker(job);
    } else {
      await persistModelDownloadJob(job, true);
      job.workerActive = false;
      if (job.pendingBatches?.length) {
        job.state = { ...job.state, status: "queued", message: `${job.pendingBatches.length} 个新增批次正在等待处理` };
        activeModelDownloadJob = job;
        await persistModelDownloadJob(job, true);
        queueModelDownloadWorker(job);
      } else if (activeModelDownloadJob?.id === job.id) {
        activeModelDownloadJob = null;
      }
    }
  }
}

async function sendModelDownloadJob(response) {
  await modelDownloadJobReady;
  response.statusCode = 200;
  response.setHeader("Content-Type", "application/json; charset=utf-8");
  response.setHeader("Cache-Control", "no-store");
  response.end(JSON.stringify({ job: publicModelDownloadJob(activeModelDownloadJob || storedModelDownloadJob) }));
}

async function retryModelDownloads(request, response) {
  if (request.method !== "POST") throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
  requireLocalRequest(request, "模型下载重试仅允许在本机发起");
  const retryInput = await readJsonRequest(request);
  await modelDownloadJobReady;
  const job = activeModelDownloadJob || storedModelDownloadJob;
  const failedIndexes = new Set((job?.state?.items || []).filter((item) => item.status === "error").map((item) => item.index));
  if (!failedIndexes.size) throw Object.assign(new Error("当前没有可重试的失败文件"), { statusCode: 409 });
  if (!job?.batches?.length) throw Object.assign(new Error("下载服务已重启，请重新提交原链接以从保留的断点继续"), { statusCode: 409 });

  const retryBatches = [];
  for (const batch of job.batches) {
    const retryIndexes = new Set([...failedIndexes].filter((index) => index >= batch.startIndex && index < batch.startIndex + batch.itemCount));
    if (retryIndexes.size) retryBatches.push({ ...batch, retryIndexes, retryInput });
  }
  if (!retryBatches.length) throw Object.assign(new Error("失败文件的重试上下文已不可用，请重新提交原链接"), { statusCode: 409 });

  job.state = {
    ...job.state,
    status: modelDownloadJobIsActive(job) ? job.state.status : "queued",
    failedModels: Math.max(0, (job.state.failedModels || 0) - failedIndexes.size),
    warning: "",
    message: `已将 ${failedIndexes.size} 个失败文件重新加入队列，将从现有断点继续`,
    items: job.state.items.map((item) => failedIndexes.has(item.index)
      ? { ...item, status: "waiting", error: "", currentBytes: 0, totalBytes: 0 }
      : item),
  };
  job.pendingBatches ||= [];
  job.pendingBatches.push(...retryBatches);
  activeModelDownloadJob = job;
  storedModelDownloadJob = job;
  await persistModelDownloadJob(job, true);
  queueModelDownloadWorker(job);
  response.statusCode = 202;
  response.setHeader("Content-Type", "application/json; charset=utf-8");
  response.setHeader("Cache-Control", "no-store");
  response.end(JSON.stringify({ job: publicModelDownloadJob(job), retried_models: failedIndexes.size }));
}

function createModelDownloadJobResponse(job) {
  return {
    setHeader() {},
    flushHeaders() {},
    write(value) {
      for (const line of String(value).split("\n")) {
        if (!line.trim()) continue;
        try {
          updateModelDownloadJob(job, JSON.parse(line));
        } catch (error) {
          console.warn(`Unable to record model download event: ${error.message}`);
        }
      }
    },
    end() {},
  };
}

async function readRecommendedModelCatalog() {
  const catalog = JSON.parse(await readFile(recommendedModelCatalogPath, "utf8"));
  if (catalog?.schema !== 1 || !Array.isArray(catalog.civitaiFamilies) || !Array.isArray(catalog.staticFamilies) || !Array.isArray(catalog.artifacts)) {
    throw Object.assign(new Error("推荐模型目录格式无效"), { statusCode: 500 });
  }
  const ids = new Set();
  for (const artifact of catalog.artifacts) {
    if (!artifact || typeof artifact.id !== "string" || !artifact.id || ids.has(artifact.id)) {
      throw Object.assign(new Error("推荐模型目录包含无效或重复的资源 ID"), { statusCode: 500 });
    }
    if (!downloadableModelKinds.has(artifact.role) || !Number.isSafeInteger(artifact.size) || artifact.size <= 0 || !normalizedSha256(artifact.sha256)) {
      throw Object.assign(new Error(`推荐资源 ${artifact.id} 缺少有效的类型、大小或 SHA-256`), { statusCode: 500 });
    }
    try {
      safeModelFilename(artifact.filename, artifact.role);
      new URL(artifact.url);
    } catch {
      throw Object.assign(new Error(`推荐资源 ${artifact.id} 的文件名或 URL 无效`), { statusCode: 500 });
    }
    ids.add(artifact.id);
  }
  return catalog;
}

function catalogArtifactsById(ids, artifactMap, familyId, field) {
  if (!Array.isArray(ids)) throw Object.assign(new Error(`推荐系列 ${familyId} 的 ${field} 配置无效`), { statusCode: 500 });
  return ids.map((id) => {
    const artifact = artifactMap.get(id);
    if (!artifact) throw Object.assign(new Error(`推荐系列 ${familyId} 引用了不存在的资源 ${id}`), { statusCode: 500 });
    return artifact;
  });
}

function publicCatalogArtifact(artifact) {
  return {
    id: artifact.id,
    role: artifact.role,
    label: artifact.label,
    filename: artifact.filename,
    size: artifact.size || 0,
    sha256: normalizedSha256(artifact.sha256),
    url: artifact.url,
    detail: artifact.detail || "",
    provider: artifact.provider || (artifact.url?.includes("civitai.") ? "Civitai" : "Hugging Face"),
    requiresHuggingfaceKey: Boolean(artifact.requiresHuggingfaceKey),
  };
}

function primaryCivitaiArtifact(family, version) {
  const file = (Array.isArray(version.files) ? version.files : []).find((item) => item?.primary)
    || (Array.isArray(version.files) ? version.files : []).find((item) => item?.type === "Model")
    || version.files?.[0];
  let filename;
  try {
    filename = file?.name ? safeModelFilename(file.name, family.role) : "";
  } catch {
    filename = "";
  }
  return {
    id: `${family.id}-${version.id}`,
    role: family.role,
    label: version.name || `Version ${version.id}`,
    filename,
    size: Number(file?.sizeKB || 0) * 1024,
    sha256: normalizedSha256(file?.hashes?.SHA256 || file?.sha256),
    url: `https://civitai.red/models/${family.modelId}/${family.slug}?modelVersionId=${version.id}`,
    provider: "Civitai",
    versionId: Number(version.id),
    baseModel: version.baseModel || "",
  };
}

async function fetchCivitaiRecommendedFamily(family) {
  let response;
  try {
    response = await Promise.any(civitaiDomains.map((domain) => fetchProviderJson(`https://${domain}/api/v1/models/${family.modelId}`, {}, recommendedCivitaiTimeout)));
  } catch {
    throw new Error(`${family.name} 在线版本暂时不可用`);
  }
  const allowed = new Set(family.baseModels);
  const versions = (Array.isArray(response.modelVersions) ? response.modelVersions : [])
    .filter((version) => allowed.has(version.baseModel))
    .map((version) => primaryCivitaiArtifact(family, version))
    .filter((artifact) => artifact.versionId);
  if (!versions.length) throw new Error(`${family.name} 没有匹配 ${family.baseModels.join(" / ")} 的版本`);
  return versions;
}

function fallbackCivitaiVersions(family) {
  return family.versions.map(([versionId, name]) => ({
    id: `${family.id}-${versionId}`,
    role: family.role,
    label: name,
    filename: "",
    size: 0,
    url: `https://civitai.red/models/${family.modelId}/${family.slug}?modelVersionId=${versionId}`,
    provider: "Civitai",
    versionId,
    baseModel: family.baseModels[0],
  }));
}

async function buildRecommendedCatalog(forceRefresh = false) {
  if (!forceRefresh && recommendedCatalogCache && Date.now() - recommendedCatalogCachedAt < recommendedCatalogCacheLifetime) return recommendedCatalogCache;
  const source = await readRecommendedModelCatalog();
  const artifactMap = new Map(source.artifacts.map((artifact) => [artifact.id, publicCatalogArtifact(artifact)]));
  const civitaiFamilies = await Promise.all(source.civitaiFamilies.map(async (family) => {
    let models;
    let remoteError = "";
    if (forceRefresh) {
      try {
        models = await fetchCivitaiRecommendedFamily(family);
      } catch (error) {
        models = fallbackCivitaiVersions(family);
        remoteError = error.message;
      }
    } else {
      models = fallbackCivitaiVersions(family);
    }
    return {
      id: family.id,
      group: family.group,
      name: family.name,
      description: family.description,
      provider: "Civitai",
      sourceUrl: `https://civitai.red/models/${family.modelId}/${family.slug}`,
      models,
      textEncoders: catalogArtifactsById(family.textEncoders || [], artifactMap, family.id, "textEncoders"),
      vaes: catalogArtifactsById(family.vaes || [], artifactMap, family.id, "vaes"),
      remoteError,
      requiresCivitaiKey: true,
    };
  }));
  const staticFamilies = source.staticFamilies.map((family) => ({
      ...family,
      models: catalogArtifactsById(family.models, artifactMap, family.id, "models"),
      textEncoders: catalogArtifactsById(family.textEncoders, artifactMap, family.id, "textEncoders"),
      vaes: catalogArtifactsById(family.vaes, artifactMap, family.id, "vaes"),
    }));
  recommendedCatalogCache = { schema: 1, remoteRefreshed: forceRefresh, families: [...civitaiFamilies, ...staticFamilies] };
  recommendedCatalogCachedAt = Date.now();
  return recommendedCatalogCache;
}

async function findFilesBySize(directory, sizes) {
  let entries;
  try {
    entries = await readdir(directory, { withFileTypes: true });
  } catch (error) {
    if (error.code === "ENOENT") return [];
    throw error;
  }
  const matches = [];
  for (const entry of entries) {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) matches.push(...await findFilesBySize(entryPath, sizes));
    else if (entry.isFile()) {
      const fileStat = await stat(entryPath);
      if (sizes.has(fileStat.size)) matches.push({ path: entryPath, size: fileStat.size, mtimeMs: fileStat.mtimeMs, ctimeMs: fileStat.ctimeMs });
    }
  }
  return matches;
}

async function cachedRecommendedFileHash(file) {
  const key = process.platform === "win32" ? file.path.toLowerCase() : file.path;
  const cached = recommendedHashCache.get(key);
  if (cached?.size === file.size && cached?.mtimeMs === file.mtimeMs && cached?.ctimeMs === file.ctimeMs) return cached.sha256;
  const sha256 = await hashFile(file.path);
  recommendedHashCache.set(key, { size: file.size, mtimeMs: file.mtimeMs, ctimeMs: file.ctimeMs, sha256 });
  return sha256;
}

async function recommendedArtifactInstallations(catalog) {
  const artifacts = catalog.families.flatMap((family) => [...family.models, ...family.textEncoders, ...family.vaes]);
  const expected = new Map(artifacts.filter((artifact) => artifact.sha256 && artifact.size).map((artifact) => [artifact.sha256, artifact]));
  const byRole = new Map();
  for (const artifact of expected.values()) {
    const role = artifact.role;
    if (!byRole.has(role)) byRole.set(role, new Set());
    byRole.get(role).add(artifact.size);
  }
  const installed = new Map();
  for (const [role, sizes] of byRole) {
    const directory = role === "checkpoint"
      ? await getConfiguredDirectory("iL", "checkpoints")
      : await getAuxiliaryModelDirectory(role === "diffusion_model" ? "diffusion_models" : role === "text_encoder" ? "text_encoders" : role === "upscaler" ? "upscalers" : role === "config" ? "configs" : "vae");
    const roleArtifacts = [...expected.values()].filter((artifact) => artifact.role === role);
    const candidates = role === "config"
      ? (await Promise.all(roleArtifacts.map(async (artifact) => {
          try {
            const file = path.join(directory, artifact.filename);
            const fileStat = await stat(file);
            return fileStat.isFile() && fileStat.size === artifact.size
              ? { path: file, size: fileStat.size, mtimeMs: fileStat.mtimeMs, ctimeMs: fileStat.ctimeMs }
              : null;
          } catch {
            return null;
          }
        }))).filter(Boolean)
      : await findFilesBySize(directory, sizes);
    for (const file of candidates) {
      const sha256 = await cachedRecommendedFileHash(file);
      if (expected.has(sha256)) installed.set(sha256, path.relative(projectRoot, file.path).split(path.sep).join("/"));
    }
  }
  return installed;
}

async function sendRecommendedCatalog(url, response) {
  const remoteRefreshed = url.searchParams.get("refresh") === "1";
  const catalog = await buildRecommendedCatalog(remoteRefreshed);
  const checkInstalled = url.searchParams.get("installed") === "1";
  const installed = checkInstalled ? await recommendedArtifactInstallations(catalog) : new Map();
  const families = catalog.families.map((family) => ({
    ...family,
    models: family.models.map((artifact) => ({ ...artifact, installed: Boolean(artifact.sha256 && installed.has(artifact.sha256)), installedPath: installed.get(artifact.sha256) || "" })),
    textEncoders: family.textEncoders.map((artifact) => ({ ...artifact, installed: Boolean(artifact.sha256 && installed.has(artifact.sha256)), installedPath: installed.get(artifact.sha256) || "" })),
    vaes: family.vaes.map((artifact) => ({ ...artifact, installed: Boolean(artifact.sha256 && installed.has(artifact.sha256)), installedPath: installed.get(artifact.sha256) || "" })),
  }));
  response.statusCode = 200;
  response.setHeader("Cache-Control", "no-store");
  response.end(JSON.stringify({ schema: catalog.schema, families, installed_checked: checkInstalled, remote_refreshed: catalog.remoteRefreshed === true }));
}

function recommendedArtifactById(catalog, familyId, artifactId, field) {
  const family = catalog.families.find((item) => item.id === familyId);
  if (!family) throw Object.assign(new Error("推荐模型不存在"), { statusCode: 400 });
  const artifact = family[field].find((item) => item.id === artifactId);
  if (!artifact) throw Object.assign(new Error("所选推荐模型版本无效"), { statusCode: 400 });
  return { family, artifact };
}

async function resolveRecommendedArtifact(artifact, apiKeys) {
  if (artifact.provider === "Civitai") {
    if (!apiKeys.civitai) throw Object.assign(new Error("Civitai 推荐模型需要先填写 Civitai API Key，并开启可访问 Civitai 的 VPN 代理"), { statusCode: 400 });
    return resolveCivitaiDownload(new URL(artifact.url), artifact.role, apiKeys.civitai);
  }
  if (artifact.provider === "GitHub") {
    const url = new URL(artifact.url);
    if (!isAllowedProviderHost(url.hostname.toLowerCase(), ["github.com", "githubusercontent.com"])) {
      throw Object.assign(new Error("推荐超分模型下载地址无效"), { statusCode: 500 });
    }
    return {
      provider: "GitHub",
      filename: safeModelFilename(artifact.filename, artifact.role),
      expectedSha256: artifact.sha256,
      routes: [
        { id: "ghfast", label: "GHFast · SHA-256", url: `https://ghfast.top/${artifact.url}` },
        { id: "ghproxy", label: "GHProxy · SHA-256", url: `https://ghproxy.net/${artifact.url}` },
        { id: "github", label: "GitHub Release", url: artifact.url },
      ],
    };
  }
  const location = parseHuggingFaceLocation(new URL(artifact.url));
  const headers = providerHeaders(apiKeys.huggingface);
  const relative = `${location.repository}/resolve/${encodeURIComponent(location.revision)}/${encodeRemotePath(location.filePath)}?download=true`;
  return {
    provider: "Hugging Face",
    filename: safeModelFilename(artifact.filename || path.posix.basename(location.filePath), artifact.role),
    expectedSha256: artifact.sha256,
    routes: [
      { id: "huggingface", label: "Hugging Face", url: `https://huggingface.co/${relative}`, headers },
      { id: "hf-mirror", label: "HF-Mirror", url: `https://hf-mirror.com/${relative}` },
    ],
  };
}

async function downloadRecommendedModels(request, response) {
  if (request.method !== "POST") throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
  requireLocalRequest(request, "推荐模型下载仅允许在本机发起");
  const payload = await readJsonRequest(request);
  const catalog = await buildRecommendedCatalog();
  const selected = recommendedArtifactById(catalog, payload.family_id, payload.model_id, "models");
  const artifacts = [selected.artifact];
  if (selected.family.textEncoders.length) artifacts.push(recommendedArtifactById(catalog, payload.family_id, payload.text_encoder_id, "textEncoders").artifact);
  if (selected.family.vaes.length) artifacts.push(recommendedArtifactById(catalog, payload.family_id, payload.vae_id, "vaes").artifact);
  const uniqueArtifacts = artifacts.filter((artifact, index, all) => all.findIndex((item) => item.sha256 ? item.sha256 === artifact.sha256 : item.id === artifact.id) === index);
  const installed = await recommendedArtifactInstallations({ families: [{ models: uniqueArtifacts, textEncoders: [], vaes: [] }] });
  const existingQueueJob = activeModelDownloadJob || (storedModelDownloadJob?.batches?.length ? storedModelDownloadJob : null);
  const pendingArtifacts = filterPendingRecommendedArtifacts(uniqueArtifacts, installed, existingQueueJob?.state?.items || []);
  const apiKeys = {
    civitai: providerApiKey(payload.civitai_key),
    huggingface: providerApiKey(payload.huggingface_key),
  };
  if (pendingArtifacts.some((artifact) => artifact.provider === "Civitai") && !apiKeys.civitai) {
    throw Object.assign(new Error("Civitai 推荐模型需要先填写 Civitai API Key，并开启可访问 Civitai 的 VPN 代理"), { statusCode: 400 });
  }
  if (((selected.family.requiresHuggingfaceKey && pendingArtifacts.includes(selected.artifact)) || pendingArtifacts.some((artifact) => artifact.requiresHuggingfaceKey)) && !apiKeys.huggingface) {
    throw Object.assign(new Error("该 Hugging Face 仓库受许可协议保护，请先接受仓库条款并填写 Hugging Face Token"), { statusCode: 400 });
  }
  const connections = Math.max(1, Math.min(32, Number(payload.connections) || 8));
  const engine = selected.family.group === "Illustrious" ? "iL" : selected.family.group === "Anima" ? "Anima" : "";
  let job;
  let addedModels = 0;
  if (pendingArtifacts.length) {
    const queued = await enqueueModelDownloadBatch({
      source: "recommended",
      kind: selected.artifact.role,
      engine,
      items: pendingArtifacts.map((artifact) => ({ id: artifact.id, label: artifact.label, sha256: artifact.sha256, artifact })),
      run: (queuedJob, queuedResponse, startIndex, retryIndexes, retryInput, batchItems) => runRecommendedModelDownloadBatch(queuedJob, queuedResponse, startIndex, { selected, artifacts: batchItems.map((item) => item.artifact), apiKeys, connections }, retryIndexes, retryInput),
    });
    job = queued.job;
    addedModels = queued.addedCount;
  } else if (existingQueueJob?.state?.items?.some((item) => item.sha256 && uniqueArtifacts.some((artifact) => artifact.sha256 === item.sha256) && item.status !== "complete")) {
    job = existingQueueJob;
  } else if (modelDownloadJobIsActive(activeModelDownloadJob) || activeModelDownloadJob?.workerActive || activeModelDownloadJob?.workerScheduled) {
    job = activeModelDownloadJob;
  } else {
    job = await createCompletedModelDownloadJob({ source: "recommended", kind: selected.artifact.role, engine, message: "所选模型和依赖均已通过本地 SHA-256 校验", targets: [{ kind: selected.artifact.role, engine }] });
  }
  response.statusCode = addedModels ? 202 : 200;
  response.setHeader("Content-Type", "application/json; charset=utf-8");
  response.setHeader("Cache-Control", "no-store");
  response.end(JSON.stringify({ job: publicModelDownloadJob(job), added_models: addedModels, skipped_models: uniqueArtifacts.length - addedModels }));
}

async function runRecommendedModelDownloadBatch(job, response, startIndex, { selected, artifacts, apiKeys, connections }, retryIndexes, retryInput) {
  const effectiveApiKeys = retryInput ? {
    civitai: providerApiKey(retryInput.civitai_key),
    huggingface: providerApiKey(retryInput.huggingface_key),
  } : apiKeys;
  writeDownloadEvent(response, { type: "starting" });
  for (const [offset, artifact] of artifacts.entries()) {
      const modelIndex = startIndex + offset;
      if (retryIndexes && !retryIndexes.has(modelIndex)) continue;
      writeDownloadEvent(response, { type: "model", index: modelIndex, total_models: job.state.totalModels });
      try {
        writeDownloadEvent(response, { type: "resolving", model_index: modelIndex, provider: artifact.provider, message: `正在解析 ${artifact.label}...` });
        const resolved = await resolveRecommendedArtifact(artifact, effectiveApiKeys);
        const destination = await getDownloadDestination({ kind: artifact.role, engine: selected.family.group === "Illustrious" ? "iL" : undefined }, resolved.filename);
        writeDownloadEvent(response, { type: "resolved", model_index: modelIndex, provider: resolved.provider, filename: resolved.filename, destination: path.relative(projectRoot, destination).split(path.sep).join("/"), verified: Boolean(resolved.expectedSha256) });
        const result = await downloadFile({
          routes: resolved.routes,
          destination,
          expectedSha256: artifact.sha256 || resolved.expectedSha256,
          connections,
          thresholdBytes: 8 * 1024 ** 2,
          maximumBytes: maximumModelDownloadBytes,
          existingFilePolicy: artifact.sha256 || resolved.expectedSha256 ? "reuse" : "error",
          fetcher: fetchDownload,
          rankRoutes: true,
          onRoute: (route) => writeDownloadEvent(response, { type: "route", model_index: modelIndex, label: route.label, cached: Boolean(route.cached), latency_ms: route.latencyMs, speed_bps: route.speedBps, connections: route.connections }),
          onProgress: (progress) => writeDownloadEvent(response, { type: "progress", model_index: modelIndex, current_bytes: progress.currentBytes, total_bytes: progress.totalBytes, speed_bps: progress.speedBps, connections: progress.connections, cached: Boolean(progress.cached), route: progress.route }),
          onWarning: (message) => writeDownloadEvent(response, { type: "warning", model_index: modelIndex, message }),
        });
        writeDownloadEvent(response, { type: "model-complete", model_index: modelIndex, filename: resolved.filename, destination: path.relative(projectRoot, destination).split(path.sep).join("/"), cached: result.cached });
      } catch (error) {
        writeDownloadEvent(response, { type: "model-error", model_index: modelIndex, error: error.message || "推荐模型下载失败" });
      }
  }
}

async function downloadRecommendedYoloModels(request, response) {
  if (request.method !== "POST") throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
  requireSameOrigin(request);
  await readJsonRequest(request);
  const catalog = await readYoloCatalog();
  if (yoloDownloadActive) throw Object.assign(new Error("The recommended YOLO model set is already downloading"), { statusCode: 409 });

  yoloDownloadActive = true;
  response.statusCode = 200;
  response.setHeader("Content-Type", "application/x-ndjson; charset=utf-8");
  response.setHeader("Cache-Control", "no-store");
  response.setHeader("X-Accel-Buffering", "no");
  response.flushHeaders?.();
  try {
    const yoloDirectory = await getAuxiliaryModelDirectory("yolo");
    writeDownloadEvent(response, { type: "starting", total_models: catalog.models.length });
    const results = [];
    for (const [index, model] of catalog.models.entries()) {
      writeDownloadEvent(response, {
        type: "model",
        index: index + 1,
        total_models: catalog.models.length,
        name: model.name,
        label: model.label,
      });
      const result = await downloadFile({
        routes: yoloDownloadRoutes(catalog, model),
        destination: path.join(yoloDirectory, model.name),
        connections: Math.max(1, Math.min(16, Number(process.env.XIRAI_DOWNLOAD_CONNECTIONS || 8) || 8)),
        thresholdBytes: 8 * 1024 ** 2,
        fetcher: fetchDownload,
        rankRoutes: true,
        onRoute: (route) => writeDownloadEvent(response, {
          type: "route",
          model_index: index + 1,
          label: route.label,
          cached: Boolean(route.cached),
          latency_ms: route.latencyMs,
          speed_bps: route.speedBps,
          connections: route.connections,
        }),
        onProgress: (progress) => writeDownloadEvent(response, {
          type: "progress",
          model_index: index + 1,
          current_bytes: progress.currentBytes,
          total_bytes: progress.totalBytes,
          speed_bps: progress.speedBps,
          connections: progress.connections,
          cached: Boolean(progress.cached),
          route: progress.route,
        }),
        onWarning: (message) => writeDownloadEvent(response, { type: "warning", model_index: index + 1, message }),
      });
      results.push(result);
    }
    writeDownloadEvent(response, {
      type: "complete",
      total_models: catalog.models.length,
      cached_models: results.filter((result) => result.cached).length,
    });
  } catch (error) {
    writeDownloadEvent(response, { type: "error", error: error.message || "YOLO model download failed" });
  } finally {
    yoloDownloadActive = false;
    response.end();
  }
}

async function runBackgroundRemovalModelDownload(model) {
  try {
    const backgroundRemovalDirectory = await getAuxiliaryModelDirectory("background_removal");
    const result = await downloadFile({
      routes: backgroundRemovalDownloadRoutes(model),
      destination: path.join(backgroundRemovalDirectory, model.filename),
      expectedSha256: model.sha256,
      maximumBytes: 1024 * 1024 ** 2,
      connections: 8,
      thresholdBytes: 2 * 1024 ** 2,
      fetcher: fetchDownload,
      rankRoutes: true,
      onRoute: (route) => updateBackgroundRemovalDownloadJob({
        status: route.cached ? "complete" : "downloading",
        active: !route.cached,
        route: route.label,
        connections: route.connections || 0,
        message: route.cached ? `${model.label} 已安装` : `已选择 ${route.label} · 8 路下载`,
      }, Boolean(route.cached)),
      onProgress: (progress) => updateBackgroundRemovalDownloadJob({
        status: progress.cached ? "complete" : "downloading",
        active: !progress.cached,
        currentBytes: progress.currentBytes || 0,
        totalBytes: progress.totalBytes || model.size,
        speedBps: progress.speedBps || 0,
        connections: progress.connections || 0,
        route: progress.route || backgroundRemovalDownloadJob?.route || "",
        message: progress.cached ? `${model.label} 已安装` : `正在下载 ${model.label}`,
      }),
      onWarning: (message) => updateBackgroundRemovalDownloadJob({ warning: message, message }),
    });
    updateBackgroundRemovalDownloadJob({
      status: "complete",
      active: false,
      cached: Boolean(result.cached),
      currentBytes: model.size,
      totalBytes: model.size,
      speedBps: 0,
      message: result.cached ? `${model.label} 已安装` : `${model.label} 下载完成`,
    }, true);
  } catch (error) {
    updateBackgroundRemovalDownloadJob({ status: "error", active: false, speedBps: 0, message: error.message || "透明背景模型下载失败" }, true);
  } finally {
    backgroundRemovalDownloadActive = false;
  }
}

async function backgroundRemovalDownloadApi(request, response) {
  if (request.method === "GET") {
    await backgroundRemovalDownloadJobReady;
    response.statusCode = 200;
    response.setHeader("Content-Type", "application/json; charset=utf-8");
    response.setHeader("Cache-Control", "no-store");
    response.end(JSON.stringify({ job: backgroundRemovalDownloadJob }));
    return;
  }
  if (request.method !== "POST") throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
  requireLocalRequest(request, "透明背景模型下载仅允许在本机发起");
  if (backgroundRemovalDownloadActive) throw Object.assign(new Error("透明背景模型正在下载"), { statusCode: 409 });
  backgroundRemovalDownloadActive = true;
  try {
    const input = await readJsonRequest(request);
    const catalog = await readBackgroundRemovalCatalog();
    const requestedId = typeof input?.model_id === "string" ? input.model_id : catalog.models[0].id;
    const model = catalog.models.find((item) => item.id === requestedId);
    if (!model) throw Object.assign(new Error("请求的透明背景模型不在受信清单中"), { statusCode: 422 });
    await backgroundRemovalDownloadJobReady;
    const now = Date.now();
    backgroundRemovalDownloadJob = {
      id: randomUUID(),
      status: "queued",
      active: true,
      modelId: model.id,
      label: model.label,
      filename: model.filename,
      currentBytes: 0,
      totalBytes: model.size,
      speedBps: 0,
      connections: 0,
      route: "",
      message: "下载任务已交给本地服务，可安全关闭或刷新页面。",
      createdAt: now,
      updatedAt: now,
    };
    await writeBackgroundRemovalDownloadJob();
    response.statusCode = 202;
    response.setHeader("Content-Type", "application/json; charset=utf-8");
    response.setHeader("Cache-Control", "no-store");
    response.end(JSON.stringify({ job: backgroundRemovalDownloadJob }));
    setImmediate(() => { void runBackgroundRemovalModelDownload(model); });
  } catch (error) {
    backgroundRemovalDownloadActive = false;
    throw error;
  }
}

async function downloadModel(request, response) {
  if (request.method !== "POST") throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
  requireLocalRequest(request, "携带 API 密钥的模型下载仅允许在本机发起");
  const payload = await readJsonRequest(request);
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) throw Object.assign(new Error("模型下载请求无效"), { statusCode: 400 });
  const urls = Array.isArray(payload.urls) ? payload.urls.filter((url) => typeof url === "string" && url.trim()) : [];
  if (!urls.length) throw Object.assign(new Error("请至少输入一个模型链接"), { statusCode: 400 });
  if (urls.length > 64) throw Object.assign(new Error("单次最多支持 64 个链接"), { statusCode: 400 });
  const kind = payload.kind;
  if (!downloadableModelKinds.has(kind)) throw Object.assign(new Error("请选择模型类型"), { statusCode: 400 });
  // A native engine's own components are downloaded by kind (diffusion_model / text_encoder /
  // vae) into the shared directories, so naming an engine only ever describes a LoRA destination.
  if (nativeEngines.has(payload.engine) && kind !== "lora") throw Object.assign(new Error(`${payload.engine} 仅支持下载 LoRA`), { statusCode: 400 });
  if (kind === "checkpoint" && !checkpointEnginePathKeys[payload.engine]) throw Object.assign(new Error("请选择兼容的底模类型"), { statusCode: 400 });
  if (kind === "lora" && !loraEnginePathKeys[payload.engine]) throw Object.assign(new Error("请选择兼容的底模类型"), { statusCode: 400 });
  const queued = await enqueueModelDownloadBatch({
    source: "manual",
    kind,
    engine: payload.engine,
    items: urls.map((url) => ({ label: modelDownloadItemLabel(url) })),
    run: (queuedJob, queuedResponse, startIndex, retryIndexes, retryInput) => runModelDownloadBatch(queuedJob, queuedResponse, startIndex, { payload, urls, kind }, retryIndexes, retryInput),
  });
  const job = queued.job;
  response.statusCode = 202;
  response.setHeader("Content-Type", "application/json; charset=utf-8");
  response.setHeader("Cache-Control", "no-store");
  response.end(JSON.stringify({ job: publicModelDownloadJob(job), added_models: queued.addedCount, skipped_models: urls.length - queued.addedCount }));
}

async function runModelDownloadBatch(job, response, startIndex, { payload, urls, kind }, retryIndexes, retryInput) {
  try {
    const effectivePayload = retryInput ? {
      ...payload,
      api_key: retryInput.api_key,
      civitai_key: retryInput.civitai_key,
      huggingface_key: retryInput.huggingface_key,
      modelscope_key: retryInput.modelscope_key,
    } : payload;
    providerApiKey(effectivePayload.api_key);
    const civitaiKey = providerApiKey(effectivePayload.civitai_key);
    const huggingfaceKey = providerApiKey(effectivePayload.huggingface_key);
    const modelscopeKey = providerApiKey(effectivePayload.modelscope_key);
    const connections = Math.max(1, Math.min(32, Number(effectivePayload.connections) || 8));

    writeDownloadEvent(response, { type: "starting" });

    for (const [offset, url] of urls.entries()) {
      const modelIndex = startIndex + offset;
      if (retryIndexes && !retryIndexes.has(modelIndex)) continue;
      writeDownloadEvent(response, { type: "model", index: modelIndex, total_models: job.state.totalModels });

      try {
        const source = parseModelLink(url);
        const keyForProvider = source.provider === "huggingface" ? huggingfaceKey
          : source.provider === "civitai" ? civitaiKey
          : modelscopeKey;
        writeDownloadEvent(response, { type: "resolving", model_index: modelIndex, provider: source.provider, message: `正在解析 ${source.provider} 模型信息...` });

        const resolved = source.provider === "huggingface"
          ? await resolveHuggingFaceDownload(source.url, kind, keyForProvider)
          : source.provider === "civitai"
            ? await resolveCivitaiDownload(source.url, kind, keyForProvider)
            : await resolveModelScopeDownload(source.url, kind, keyForProvider);

        const destination = await getDownloadDestination(payload, resolved.filename);
        writeDownloadEvent(response, {
          type: "resolved",
          model_index: modelIndex,
          provider: resolved.provider,
          filename: resolved.filename,
          destination: path.relative(projectRoot, destination).split(path.sep).join("/"),
          verified: Boolean(resolved.expectedSha256),
        });

        const result = await downloadFile({
          routes: resolved.routes,
          destination,
          expectedSha256: resolved.expectedSha256,
          connections,
          thresholdBytes: 8 * 1024 ** 2,
          maximumBytes: maximumModelDownloadBytes,
          existingFilePolicy: resolved.expectedSha256 ? "reuse" : "error",
          fetcher: fetchDownload,
          rankRoutes: true,
          onRoute: (route) => writeDownloadEvent(response, {
            type: "route",
            model_index: modelIndex,
            label: route.label,
            cached: Boolean(route.cached),
            latency_ms: route.latencyMs,
            speed_bps: route.speedBps,
            connections: route.connections,
          }),
          onProgress: (progress) => writeDownloadEvent(response, {
            type: "progress",
            model_index: modelIndex,
            current_bytes: progress.currentBytes,
            total_bytes: progress.totalBytes,
            speed_bps: progress.speedBps,
            connections: progress.connections,
            cached: Boolean(progress.cached),
            route: progress.route,
          }),
          onWarning: (message) => writeDownloadEvent(response, { type: "warning", model_index: modelIndex, message }),
        });

        if (kind === "lora") {
          writeDownloadEvent(response, { type: "metadata", model_index: modelIndex, message: "正在查询 LoRA 来源和触发词..." });
          try {
            const metadata = await cacheLoraMetadata(destination, { reviewKind: payload.category });
            writeDownloadEvent(response, { type: "metadata", model_index: modelIndex, status: metadata.status, model_name: metadata.modelName, message: metadata.status === "found" ? "LoRA 信息已同步到管理器" : "未找到公开 LoRA 信息，模型已下载" });
          } catch (error) {
            writeDownloadEvent(response, { type: "warning", model_index: modelIndex, message: `模型已下载，但 LoRA 信息查询失败：${error.message}` });
          }
        }

        writeDownloadEvent(response, {
          type: "model-complete",
          model_index: modelIndex,
          filename: resolved.filename,
          destination: path.relative(projectRoot, destination).split(path.sep).join("/"),
          cached: result.cached,
        });
      } catch (error) {
        writeDownloadEvent(response, {
          type: "model-error",
          model_index: modelIndex,
          error: error.message || "模型下载出现未知错误",
        });
        // Continue with remaining URLs.
      }
    }

  } catch (error) {
    throw Object.assign(error, { message: error.message || "模型下载失败" });
  }
}

async function lookupLora(request, response) {
  if (request.method !== "POST") throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
  const payload = await readJsonRequest(request);
  const modelPath = await validateLoraPath(payload.engine, payload.path);
  const metadata = await cacheLoraMetadata(modelPath, { refresh: payload.refresh === true, reviewKind: loraReviewKind(payload.path) });
  response.statusCode = 200;
  response.end(JSON.stringify({ metadata }));
}

async function sendLoraPreview(url, response) {
  const modelPath = await validateLoraPath(url.searchParams.get("engine"), url.searchParams.get("path"));
  const reviewKind = loraReviewKind(url.searchParams.get("path"));
  let metadata = await readLoraMetadata(modelPath, reviewKind);
  if (metadata?.status === "found" && !metadata.previewFile && !metadata.previewUrl) metadata = await cacheLoraMetadata(modelPath, { reviewKind });
  if (!metadata?.previewFile && metadata?.previewUrl) metadata = await ensureLoraPreview(modelPath, metadata, reviewKind);
  if (!metadata?.previewFile || path.basename(metadata.previewFile) !== metadata.previewFile) {
    throw Object.assign(new Error("No cached preview image"), { statusCode: 404 });
  }
  const roots = [getLoraCachePath(modelPath), path.dirname(modelPath)];
  let previewPath;
  for (const root of roots) {
    try {
      const candidate = await realpath(path.join(root, metadata.previewFile));
      const resolvedRoot = await realpath(root);
      if (isPathInside(resolvedRoot, candidate) && candidate !== resolvedRoot) {
        previewPath = candidate;
        break;
      }
    } catch {
      // Continue to the legacy model-side preview location.
    }
  }
  if (!previewPath) throw Object.assign(new Error("No cached preview image"), { statusCode: 404 });
  const extension = path.extname(previewPath).toLowerCase();
  if (!previewContentTypes[extension]) throw Object.assign(new Error("Invalid preview image type"), { statusCode: 400 });
  const buffer = await readFile(previewPath);
  response.statusCode = 200;
  response.setHeader("Content-Type", previewContentTypes[extension]);
  response.setHeader("Cache-Control", "private, max-age=3600");
  response.end(buffer);
}

const imageInfoExtensions = new Set([".png", ".jpg", ".jpeg", ".webp"]);
// One navigation step should not re-walk every model directory, but the answer
// must not go stale while the user is adding folders either.
const MODEL_CATALOG_TTL_MS = 10000;
const IMAGE_INFO_MAX_UPLOAD_BYTES = 64 * 1024 * 1024;
const IMAGE_INFO_MAX_DIRECTORY_FILES = 2000;
let modelCatalogCache = null;

// `name` is what a producer's metadata is matched against; `value` is what a
// picker here would need to select the same file, and for a shared model the two
// differ — the name is the path inside the shared root, the value is the
// reference that addresses it. `engine` says which root it came from, because a
// checkpoint is only selectable while that engine is the active one.
function catalogEntry(origin, name, label, { value = name, engine = "" } = {}) {
  const normalized = name.split("\\").join("/").toLowerCase();
  const base = normalized.split("/").pop();
  return { origin, path: normalized, base, stem: base.replace(/\.[^.]+$/, ""), label, value, engine };
}

// Everything the workspace could load, flattened for name matching. Local
// entries come first inside each origin, but the ordering that decides a match
// is enforced in `matchModelName` rather than here.
async function buildModelCatalogs() {
  if (modelCatalogCache && Date.now() - modelCatalogCache.at < MODEL_CATALOG_TTL_MS) return modelCatalogCache.value;
  const checkpoints = [];
  const loras = [];

  for (const engine of ["SD", "iL"]) {
    try {
      const directory = await getConfiguredDirectory(engine, "checkpoints");
      const relative = path.relative(projectRoot, directory).split(path.sep).join("/");
      for (const model of await findModels(directory, directory, false, checkpointExtensions)) {
        checkpoints.push(catalogEntry("local", model.value, `${relative} · ${engine}`, { engine }));
      }
    } catch {
      // A missing or misconfigured engine root must not stop the other engines
      // from answering; the reader degrades to "not found" for that one.
    }
  }
  try {
    const directory = await getAuxiliaryModelDirectory("diffusion_models");
    const relative = path.relative(projectRoot, directory).split(path.sep).join("/");
    for (const model of await findModels(directory, directory, false, diffusionModelExtensions)) {
      checkpoints.push(catalogEntry("local", model.value, `${relative} · Anima`, { engine: "Anima" }));
    }
  } catch {}
  for (const engine of Object.keys(loraEnginePathKeys)) {
    try {
      const directory = await getConfiguredDirectory(engine, "loras");
      const relative = path.relative(projectRoot, directory).split(path.sep).join("/");
      const extensions = nativeEngines.has(engine) ? animaLoraExtensions : loraExtensions;
      for (const model of await findModels(directory, directory, false, extensions)) {
        loras.push(catalogEntry("local", model.value, `${relative} · ${engine}`, { engine }));
      }
    } catch {}
  }

  let roots = [];
  try {
    roots = await readSharedRoots(projectRoot);
  } catch {
    // A damaged shared config costs the shared half of matching, not the local.
  }
  for (const root of roots) {
    if (!root.enabled) continue;
    for (const model of await findSharedModels(root, "checkpoints", checkpointExtensions, false)) {
      checkpoints.push(catalogEntry("shared", model.name, root.label, { value: model.value }));
    }
    for (const model of await findSharedModels(root, "loras", loraExtensions, false)) {
      loras.push(catalogEntry("shared", model.name, root.label, { value: model.value }));
    }
  }

  const value = { checkpoints, loras };
  modelCatalogCache = { at: Date.now(), value };
  return value;
}

// A directory the user typed. Same trust shape as a shared root — read-only,
// must be a real, specific folder — but never registered or remembered.
async function resolveImageDirectory(candidate) {
  const requested = String(candidate ?? "").trim();
  if (!requested) throw Object.assign(new Error("请输入目录路径"), { statusCode: 400 });
  if (!path.isAbsolute(requested)) throw Object.assign(new Error("目录必须是绝对路径"), { statusCode: 400 });
  let resolved;
  try {
    resolved = await realpath(path.resolve(requested));
  } catch {
    throw Object.assign(new Error("目录不存在或无法访问"), { statusCode: 400 });
  }
  if (!(await stat(resolved)).isDirectory()) throw Object.assign(new Error("该路径不是文件夹"), { statusCode: 400 });
  if (path.dirname(resolved) === resolved) throw Object.assign(new Error("请选择具体的图片文件夹，而不是磁盘根目录"), { statusCode: 400 });
  return resolved;
}

async function resolveImageFile(directory, name) {
  const filename = String(name ?? "");
  // A name is a single entry in the listed folder. Anything with a separator
  // in it is trying to leave, whatever it resolves to.
  if (!filename || path.basename(filename) !== filename) throw Object.assign(new Error("图片文件名无效"), { statusCode: 400 });
  if (!imageInfoExtensions.has(path.extname(filename).toLowerCase())) throw Object.assign(new Error("不支持的图片格式"), { statusCode: 400 });
  const root = await resolveImageDirectory(directory);
  let resolved;
  try {
    resolved = await realpath(path.join(root, filename));
  } catch {
    throw Object.assign(new Error("图片不存在"), { statusCode: 404 });
  }
  if (!isPathInside(root, resolved) || !(await stat(resolved)).isFile()) {
    throw Object.assign(new Error("图片不在所选文件夹内"), { statusCode: 403 });
  }
  return resolved;
}

function decodeImageUpload(value) {
  const text = String(value ?? "");
  const base64 = text.startsWith("data:") ? text.slice(text.indexOf(",") + 1) : text;
  if (!base64) throw Object.assign(new Error("图片内容为空"), { statusCode: 400 });
  const buffer = Buffer.from(base64, "base64");
  if (!buffer.length) throw Object.assign(new Error("图片内容无法解码"), { statusCode: 400 });
  return buffer;
}

// Attach a resolution verdict to every model name the metadata mentions:
// project directory first, then registered shared folders, then missing.
async function resolveImageInfoModels(info) {
  if (!info || info.status !== "ok") return info;
  const catalogs = await buildModelCatalogs();
  return {
    ...info,
    checkpointMatch: info.checkpoint ? matchModelName(info.checkpoint, catalogs.checkpoints) : null,
    loras: (info.loras || []).map((item) => ({ ...item, match: matchModelName(item.name, catalogs.loras) })),
    animaAssets: info.animaAssets ? {
      text_encoder: { name: info.animaAssets.text_encoder, match: matchModelName(info.animaAssets.text_encoder, catalogs.checkpoints) },
      vae: { name: info.animaAssets.vae, match: matchModelName(info.animaAssets.vae, catalogs.checkpoints) },
    } : null,
  };
}

async function readImageInfoBuffer(buffer, name) {
  const chunks = readPngMetadataChunks(buffer);
  if (chunks === null) {
    // Not a PNG at all. Reporting that is more useful than "no metadata",
    // which would imply the file was checked and found bare.
    return { name, status: "unsupported", source: "unknown", sourceLabel: "非 PNG 文件", bytes: buffer.length, raw: {} };
  }
  const info = interpretImageMetadata(chunks);
  return { ...(await resolveImageInfoModels(info)), name, bytes: buffer.length };
}

async function imageInfoApi(request, response, url) {
  if (url.pathname === "/api/image-info/preview") {
    if (request.method !== "GET") throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
    const file = await resolveImageFile(url.searchParams.get("directory"), url.searchParams.get("name"));
    const buffer = await readFile(file);
    response.statusCode = 200;
    response.setHeader("Content-Type", previewContentTypes[path.extname(file).toLowerCase()] || "application/octet-stream");
    response.setHeader("Cache-Control", "private, max-age=60");
    response.end(buffer);
    return;
  }

  response.setHeader("Content-Type", "application/json; charset=utf-8");
  response.setHeader("Cache-Control", "no-store");
  if (request.method !== "POST") throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });

  if (url.pathname === "/api/image-info/scan") {
    const payload = await readJsonRequest(request);
    const directory = await resolveImageDirectory(payload.directory);
    const entries = await readdir(directory, { withFileTypes: true });
    const files = [];
    for (const entry of entries) {
      // Non-recursive on purpose: a batch is "this folder of renders", and
      // walking a whole output tree would be a different, much slower promise.
      if (!entry.isFile() || !imageInfoExtensions.has(path.extname(entry.name).toLowerCase())) continue;
      if (files.length >= IMAGE_INFO_MAX_DIRECTORY_FILES) break;
      const fileStat = await stat(path.join(directory, entry.name));
      files.push({ name: entry.name, size: fileStat.size, modifiedAt: fileStat.mtimeMs });
    }
    files.sort((first, second) => first.name.localeCompare(second.name, "zh-CN", { numeric: true }));
    response.statusCode = 200;
    response.end(JSON.stringify({
      directory,
      files,
      truncated: entries.filter((entry) => entry.isFile() && imageInfoExtensions.has(path.extname(entry.name).toLowerCase())).length > files.length,
    }));
    return;
  }

  if (url.pathname === "/api/image-info/read") {
    const payload = await readJsonRequest(request, IMAGE_INFO_MAX_UPLOAD_BYTES);
    if (payload.directory) {
      const file = await resolveImageFile(payload.directory, payload.name);
      response.statusCode = 200;
      response.end(JSON.stringify({ info: await readImageInfoBuffer(await readFile(file), path.basename(file)) }));
      return;
    }
    const buffer = decodeImageUpload(payload.image);
    response.statusCode = 200;
    response.end(JSON.stringify({ info: await readImageInfoBuffer(buffer, String(payload.name || "")) }));
    return;
  }
  throw Object.assign(new Error("Not found"), { statusCode: 404 });
}

// Registering, inspecting and forgetting shared folders. Deliberately separate
// from `/api/model-paths`: that one describes where this project keeps its own
// files and is the only thing downloads may write into. These are read-only
// borrowings from other tools and never a download destination.
async function sharedPathsApi(request, response, url) {
  if (url.pathname === "/api/shared-paths/inspect") {
    if (request.method !== "POST") throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
    const payload = await readJsonRequest(request);
    const inspection = await inspectSharedDirectory(projectRoot, payload.path, payload.kind);
    const roots = await readSharedRoots(projectRoot);
    response.statusCode = 200;
    response.end(JSON.stringify({ inspection, registered: roots.some((root) => root.id === inspection.id) }));
    return;
  }
  if (request.method === "GET") {
    response.statusCode = 200;
    response.end(JSON.stringify({ roots: await readSharedRoots(projectRoot) }));
    return;
  }
  if (request.method === "POST") {
    const payload = await readJsonRequest(request);
    const inspection = await inspectSharedDirectory(projectRoot, payload.path, payload.kind);
    if (!inspection.entries.length) {
      throw Object.assign(new Error("该目录下没有找到可识别的模型文件"), { statusCode: 400 });
    }
    const roots = upsertSharedRoot(await readSharedRoots(projectRoot), sharedRootDraft(inspection, payload));
    response.statusCode = 200;
    response.end(JSON.stringify({ roots: await writeSharedRoots(projectRoot, roots), inspection }));
    return;
  }
  if (request.method === "PUT") {
    const payload = await readJsonRequest(request);
    const current = await readSharedRoots(projectRoot);
    const target = current.find((root) => root.id === payload.id);
    if (!target) throw Object.assign(new Error("共享目录未注册"), { statusCode: 404 });
    const updated = current.map((root) => root.id !== payload.id ? root : {
      ...root,
      enabled: payload.enabled === undefined ? root.enabled : payload.enabled !== false,
      label: payload.label === undefined ? root.label : payload.label,
      engines: payload.engines === undefined ? root.engines : payload.engines,
    });
    response.statusCode = 200;
    response.end(JSON.stringify({ roots: await writeSharedRoots(projectRoot, updated) }));
    return;
  }
  if (request.method === "DELETE") {
    const id = url.searchParams.get("id");
    const current = await readSharedRoots(projectRoot);
    if (!current.some((root) => root.id === id)) throw Object.assign(new Error("共享目录未注册"), { statusCode: 404 });
    // Forgetting a root never touches the folder itself — it only stops this
    // workspace from reading it.
    response.statusCode = 200;
    response.end(JSON.stringify({ roots: await writeSharedRoots(projectRoot, current.filter((root) => root.id !== id)) }));
    return;
  }
  throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
}

async function modelApi(request, response, next) {
  const url = new URL(request.url, "http://localhost");
  const apiPaths = ["/api/models", "/api/loras", "/api/model-paths", "/api/shared-paths", "/api/shared-paths/inspect", "/api/image-info/scan", "/api/image-info/read", "/api/image-info/preview", "/api/lora-lookup", "/api/lora-preview", "/api/yolo/download", "/api/background-removal/download", "/api/model-download", "/api/model-download/job", "/api/model-download/retry", "/api/recommended-models", "/api/recommended-download"];
  if (!apiPaths.includes(url.pathname)) {
    next();
    return;
  }

  try {
    if (url.pathname === "/api/model-download/job") {
      if (request.method !== "GET") throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
      await sendModelDownloadJob(response);
      return;
    }
    if (url.pathname === "/api/model-download/retry") {
      await retryModelDownloads(request, response);
      return;
    }
    if (url.pathname === "/api/yolo/download") {
      await downloadRecommendedYoloModels(request, response);
      return;
    }
    if (url.pathname === "/api/background-removal/download") {
      await backgroundRemovalDownloadApi(request, response);
      return;
    }
    if (url.pathname === "/api/model-download") {
      await downloadModel(request, response);
      return;
    }
    if (url.pathname === "/api/recommended-download") {
      await downloadRecommendedModels(request, response);
      return;
    }
    if (url.pathname === "/api/lora-preview") {
      if (request.method !== "GET") throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
      await sendLoraPreview(url, response);
      return;
    }
    // Dispatched before the blanket JSON headers below: the preview route
    // answers with image bytes and sets its own content type.
    if (url.pathname.startsWith("/api/image-info/")) {
      await imageInfoApi(request, response, url);
      return;
    }

    response.setHeader("Content-Type", "application/json; charset=utf-8");
    response.setHeader("Cache-Control", "no-store");
    if (url.pathname === "/api/recommended-models") {
      if (request.method !== "GET") throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
      await sendRecommendedCatalog(url, response);
      return;
    }
    if (url.pathname === "/api/model-paths") {
      if (request.method !== "GET") throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
      response.statusCode = 200;
      response.end(JSON.stringify({ paths: await effectiveModelPaths() }));
      return;
    }
    if (url.pathname === "/api/shared-paths" || url.pathname === "/api/shared-paths/inspect") {
      await sharedPathsApi(request, response, url);
      return;
    }
    if (url.pathname === "/api/lora-lookup") {
      await lookupLora(request, response);
      return;
    }
    if (request.method !== "GET") throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });

    const engine = url.searchParams.get("engine");
    const isLoraRequest = url.pathname === "/api/loras";
    if (!isLoraRequest && engine === "Anima") {
      const [diffusionDirectory, textEncoderDirectory, vaeDirectory] = await Promise.all([
        getAuxiliaryModelDirectory("diffusion_models"),
        getAuxiliaryModelDirectory("text_encoders"),
        getAuxiliaryModelDirectory("vae"),
      ]);
      const discovered = await discoverAnimaModels({ diffusion_model: diffusionDirectory, text_encoder: textEncoderDirectory, vae: vaeDirectory });
      const relativeDirectory = (directory) => path.relative(projectRoot, directory).split(path.sep).join("/");
      const tokenizerDirectory = bundledAnimaTokenizerDirectory(projectRoot);
      const runtimePath = (filename) => path.posix.join(relativeDirectory(tokenizerDirectory), filename);
      const runtimeStatuses = await animaRuntimeArtifactStatuses(tokenizerDirectory, cachedRecommendedFileHash);
      response.statusCode = 200;
      response.end(JSON.stringify({
        engine: "Anima",
        model_type: "split",
        assets: {
          diffusion_model: { directory: relativeDirectory(diffusionDirectory), models: discovered.diffusion_model },
          text_encoder: { directory: relativeDirectory(textEncoderDirectory), models: discovered.text_encoder },
          vae: { directory: relativeDirectory(vaeDirectory), models: discovered.vae },
        },
        runtime: {
          qwen_tokenizer: { ...runtimeStatuses.qwen_tokenizer, path: runtimePath(ANIMA_RUNTIME_ARTIFACTS.qwen_tokenizer.filename) },
          qwen_tokenizer_config: { ...runtimeStatuses.qwen_tokenizer_config, path: runtimePath(ANIMA_RUNTIME_ARTIFACTS.qwen_tokenizer_config.filename) },
          t5_tokenizer: { ...runtimeStatuses.t5_tokenizer, path: runtimePath(ANIMA_RUNTIME_ARTIFACTS.t5_tokenizer.filename) },
        },
      }));
      return;
    }
    if (!isLoraRequest && engine === "Flux") {
      // FLUX.1 mounts four files out of three shared directories, and its two text encoders live
      // side by side in one of them. They are listed as separate assets rather than one list the
      // user has to sort out, because the file's own tensors already say which is which.
      const [diffusionDirectory, textEncoderDirectory, vaeDirectory] = await Promise.all([
        getAuxiliaryModelDirectory("diffusion_models"),
        getAuxiliaryModelDirectory("text_encoders"),
        getAuxiliaryModelDirectory("vae"),
      ]);
      const discovered = await discoverFluxModels({
        diffusion_model: diffusionDirectory,
        text_encoder: textEncoderDirectory,
        text_encoder_2: textEncoderDirectory,
        vae: vaeDirectory,
      });
      const relativeDirectory = (directory) => path.relative(projectRoot, directory).split(path.sep).join("/");
      const tokenizerDirectory = bundledAnimaTokenizerDirectory(projectRoot);
      const runtimeStatuses = await animaRuntimeArtifactStatuses(tokenizerDirectory, cachedRecommendedFileHash);
      response.statusCode = 200;
      response.end(JSON.stringify({
        engine: "Flux",
        model_type: "split",
        assets: {
          diffusion_model: { directory: relativeDirectory(diffusionDirectory), models: discovered.diffusion_model },
          text_encoder: { directory: relativeDirectory(textEncoderDirectory), models: discovered.text_encoder },
          text_encoder_2: { directory: relativeDirectory(textEncoderDirectory), models: discovered.text_encoder_2 },
          vae: { directory: relativeDirectory(vaeDirectory), models: discovered.vae },
        },
        runtime: {
          // The same pinned T5 v1.1 tokenizer both native engines read; it is bundled once.
          t5_tokenizer: {
            ...runtimeStatuses.t5_tokenizer,
            path: path.posix.join(relativeDirectory(tokenizerDirectory), ANIMA_RUNTIME_ARTIFACTS.t5_tokenizer.filename),
          },
        },
      }));
      return;
    }
    if (!isLoraRequest && engine === "Flux2") {
      // FLUX.2 mounts three files: one diffusion model, one language model and one autoencoder.
      // The language model is listed as `text_encoder` like Anima's, because there is no second
      // one — the whole conditioning comes from three intermediate layers of that one model.
      const [diffusionDirectory, textEncoderDirectory, vaeDirectory] = await Promise.all([
        getAuxiliaryModelDirectory("diffusion_models"),
        getAuxiliaryModelDirectory("text_encoders"),
        getAuxiliaryModelDirectory("vae"),
      ]);
      const discovered = await discoverFlux2Models({
        diffusion_model: diffusionDirectory,
        text_encoder: textEncoderDirectory,
        vae: vaeDirectory,
      });
      const relativeDirectory = (directory) => path.relative(projectRoot, directory).split(path.sep).join("/");
      const tokenizerDirectory = bundledAnimaTokenizerDirectory(projectRoot);
      const runtimeStatuses = await animaRuntimeArtifactStatuses(tokenizerDirectory, cachedRecommendedFileHash);
      response.statusCode = 200;
      response.end(JSON.stringify({
        engine: "Flux2",
        model_type: "split",
        assets: {
          diffusion_model: { directory: relativeDirectory(diffusionDirectory), models: discovered.diffusion_model },
          text_encoder: { directory: relativeDirectory(textEncoderDirectory), models: discovered.text_encoder },
          vae: { directory: relativeDirectory(vaeDirectory), models: discovered.vae },
        },
        runtime: {
          // [klein] reads the same pinned Qwen tokenizer Anima does. [dev]'s Mistral tokenizer is
          // not listed because it is not an installed resource: ComfyUI packs it into the text
          // encoder checkpoint itself.
          qwen_tokenizer: {
            ...runtimeStatuses.qwen_tokenizer,
            path: path.posix.join(relativeDirectory(tokenizerDirectory), ANIMA_RUNTIME_ARTIFACTS.qwen_tokenizer.filename),
          },
        },
      }));
      return;
    }
    if (!isLoraRequest && engine === "Krea2") {
      // Krea 2 mounts three files: one single-stream DiT, one Qwen3-VL-4B and Wan 2.1's
      // autoencoder. Like FLUX.2 it lists the language model as `text_encoder`, because there is
      // no second one — the whole conditioning is twelve intermediate layers of that one model.
      const [diffusionDirectory, textEncoderDirectory, vaeDirectory] = await Promise.all([
        getAuxiliaryModelDirectory("diffusion_models"),
        getAuxiliaryModelDirectory("text_encoders"),
        getAuxiliaryModelDirectory("vae"),
      ]);
      const discovered = await discoverKrea2Models({
        diffusion_model: diffusionDirectory,
        text_encoder: textEncoderDirectory,
        vae: vaeDirectory,
      });
      const relativeDirectory = (directory) => path.relative(projectRoot, directory).split(path.sep).join("/");
      const tokenizerDirectory = bundledAnimaTokenizerDirectory(projectRoot);
      const runtimeStatuses = await animaRuntimeArtifactStatuses(tokenizerDirectory, cachedRecommendedFileHash);
      response.statusCode = 200;
      response.end(JSON.stringify({
        engine: "Krea2",
        model_type: "split",
        assets: {
          diffusion_model: { directory: relativeDirectory(diffusionDirectory), models: discovered.diffusion_model },
          text_encoder: { directory: relativeDirectory(textEncoderDirectory), models: discovered.text_encoder },
          vae: { directory: relativeDirectory(vaeDirectory), models: discovered.vae },
        },
        runtime: {
          // Qwen3-VL reads text through Qwen2.5's table, which is the same pinned resource Anima
          // and FLUX.2 [klein] already use; Krea 2 installs nothing new.
          qwen_tokenizer: {
            ...runtimeStatuses.qwen_tokenizer,
            path: path.posix.join(relativeDirectory(tokenizerDirectory), ANIMA_RUNTIME_ARTIFACTS.qwen_tokenizer.filename),
          },
        },
      }));
      return;
    }
    const directory = await getConfiguredDirectory(engine, isLoraRequest ? "loras" : "checkpoints");

    if (isLoraRequest) {
      const extensions = nativeEngines.has(engine) ? animaLoraExtensions : loraExtensions;
      const models = await findModels(directory, directory, true, extensions, directory);
      // A local folder literally named `shared:` would collide with the shared
      // namespace. Windows cannot create one, but the guard is cheap and the
      // failure it prevents is a silently unresolvable mount.
      const local = models.filter((model) => !parseSharedRef(model.value));
      const categories = groupLoraModels(local, directory, projectRoot);
      for (const category of categories) category.models.sort((first, second) => first.name.localeCompare(second.name, "zh-CN"));
      const sharedListings = await sharedListingsFor(engine, "loras", extensions, true);
      // The shared category sits after the four defaults, exactly where the
      // user expects a fifth folder to appear.
      if (sharedListings.length) categories.push(shapeSharedLoraCategory(sharedListings));
      response.statusCode = 200;
      response.end(JSON.stringify({ engine, directory: path.relative(projectRoot, directory).split(path.sep).join("/"), categories }));
      return;
    }

    const models = await findModels(directory, directory, false, checkpointExtensions);
    models.sort((first, second) => first.name.localeCompare(second.name, "zh-CN"));
    const sharedCheckpoints = (await sharedListingsFor(engine, "checkpoints", checkpointExtensions, false)).flatMap((listing) => listing.models);
    response.statusCode = 200;
    response.end(JSON.stringify({
      engine,
      directory: path.relative(projectRoot, directory).split(path.sep).join("/"),
      models: [...models.filter((model) => !parseSharedRef(model.value)), ...sharedCheckpoints],
    }));
  } catch (error) {
    response.statusCode = error.statusCode || 500;
    if (!response.getHeader("Content-Type")) response.setHeader("Content-Type", "application/json; charset=utf-8");
    response.end(JSON.stringify({ error: error.message }));
  }
}

function modelApiPlugin() {
  return {
    name: "local-model-api",
    configureServer(server) {
      server.middlewares.use(modelApi);
    },
    configurePreviewServer(server) {
      server.middlewares.use(modelApi);
    },
  };
}

export function updateApiPlugin({
  fetcher: updateFetcher = undiciFetch,
  environment: updateEnvironment = process.env,
  releaseTimeoutMs = 15000,
  sidecarTimeoutMs = 15000,
  repositoryTimeoutMs = 10000,
} = {}) {
  // The feed provenance and PAT allow-list must not change between check and download. Snapshot the
  // update settings once for this server instance instead of re-reading a mutable process.env after
  // a release has already crossed the check boundary.
  const releaseEnvironment = Object.freeze({ ...updateEnvironment });
  let preparedUpdate;
  let activeTask;
  let uploadDirectory;
  let uploadOwnership;
  let operationLock;
  // The last answer the release feed gave. Held in memory only: a check is cheap, and a cached
  // "update available" that outlived the release it named would offer a download that 404s.
  let onlineRelease = null;
  let onlineCheckedAt = null;
  let onlineCheckGeneration = 0;
  let updateState = {
    status: "idle",
    phase: "idle",
    progress: 0,
    message: "请选择更新归档",
    restart_required: false,
    environment_ready: false,
    repair_available: false,
  };
  let updatePersistTimer;
  let updateStateWrite = Promise.resolve();

  const persistUpdateState = (immediate = false) => {
    if (!immediate) {
      if (updatePersistTimer) return;
      updatePersistTimer = setTimeout(() => {
        updatePersistTimer = null;
        void persistUpdateState(true);
      }, 500);
      return;
    }
    if (updatePersistTimer) clearTimeout(updatePersistTimer);
    updatePersistTimer = null;
    const payload = {
      schema: 1,
      product: "XiriaCanvas AI",
      projectRoot,
      state: updateState,
      prepared: preparedUpdate || null,
      upload: uploadOwnership || null,
    };
    updateStateWrite = updateStateWrite
      .then(async () => {
        await mkdir(stateDirectory, { recursive: true });
        const temporaryPath = path.join(stateDirectory, `${randomUUID()}.tmp`);
        try {
          await writeFile(temporaryPath, `${JSON.stringify(payload, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
          await rename(temporaryPath, manualUpdateStatePath);
        } finally {
          await rm(temporaryPath, { force: true });
        }
      })
      .catch((error) => console.warn(`Unable to save manual update state: ${error.message}`));
    return updateStateWrite;
  };

  const restoreUpdateState = async () => {
    try {
      const saved = JSON.parse(await readFile(manualUpdateStatePath, "utf8"));
      const savedRoot = typeof saved?.projectRoot === "string" ? path.resolve(saved.projectRoot) : "";
      const sameRoot = process.platform === "win32"
        ? savedRoot.toLowerCase() === path.resolve(projectRoot).toLowerCase()
        : savedRoot === path.resolve(projectRoot);
      if (!saved || saved.schema !== 1 || saved.product !== "XiriaCanvas AI" || !sameRoot
        || !saved.state || typeof saved.state !== "object") return;
      updateState = { ...updateState, ...saved.state };
      preparedUpdate = saved.prepared && typeof saved.prepared === "object" ? saved.prepared : undefined;
      uploadOwnership = saved.upload && typeof saved.upload === "object" ? saved.upload : undefined;
      uploadDirectory = uploadOwnership?.path;
      if ((updateState.restart_required || updateState.environment_ready) && !readSetupMarker(projectRoot)) {
        updateState = {
          ...updateState,
          status: "error",
          phase: "error",
          message: "环境配置完成标记缺失或无效，需要修复后才能重启",
          restart_required: false,
          environment_ready: false,
          repair_available: true,
        };
        await persistUpdateState(true);
      }
      if (updateBusy(updateState.status)) {
        // A server restart cannot safely resume an in-flight archive operation.
        let stateRecoveryLock;
        try {
          stateRecoveryLock = await acquireOfflineUpdateLock({ stateDirectory, projectRoot, operation: "state-recovery" });
        } catch (error) {
          if (error.code !== "UPDATE_BUSY") throw error;
          updateState = {
            ...updateState,
            status: "error",
            phase: "error",
            progress: 0,
            message: "另一进程仍在执行离线更新，本实例不会恢复或清理其文件。",
            restart_required: false,
            environment_ready: false,
          };
          preparedUpdate = undefined;
          uploadDirectory = undefined;
          uploadOwnership = undefined;
          return;
        }
        try {
        updateState = {
          ...updateState,
          status: "error",
          phase: "error",
          progress: 0,
          message: "更新服务已重启；已完成的更新包校验可重新开始，未完成的上传需要重新选择归档。",
          restart_required: false,
          environment_ready: false,
          repair_available: updateState.status === "repairing",
        };
        const abandonedStage = preparedUpdate?.stagingOwnership;
        const abandonedUpload = uploadOwnership;
        preparedUpdate = undefined;
        uploadDirectory = undefined;
        uploadOwnership = undefined;
        await Promise.all([
          abandonedStage && removeOfflineUpdateTemp({ projectRoot, record: abandonedStage, prefix: "xirai-update-stage-", kind: "stage" }),
          abandonedUpload && removeOfflineUpdateTemp({ projectRoot, record: abandonedUpload, prefix: "xirai-update-upload-", kind: "upload" }),
        ].filter(Boolean).map((operation) => operation.catch(() => {})));
        await persistUpdateState(true);
        } finally {
          await stateRecoveryLock.release().catch(() => {});
        }
      }
    } catch (error) {
      if (error.code !== "ENOENT") console.warn(`Unable to restore manual update state: ${error.message}`);
    }
  };
  const updateStateReady = restoreUpdateState();

  const publicOnlineRelease = () => (onlineRelease ? {
    version: onlineRelease.version,
    prerelease: onlineRelease.prerelease,
    published_at: onlineRelease.publishedAt,
    notes: onlineRelease.notes.slice(0, 4000),
    asset_name: onlineRelease.asset.name,
    asset_bytes: onlineRelease.asset.bytes,
    // Advertised metadata is not verification. This is true only after the required sidecar has
    // been fetched, bounded, parsed for this exact filename, and reconciled with any API digest.
    verified: Boolean(onlineRelease.resolvedChecksum),
    update_available: updateAvailable(onlineRelease.version, appVersion),
  } : null);
  const publicState = () => ({
    ...updateState,
    prepared_items: preparedUpdate?.plan?.map((item) => item.relativePath) || [],
    prepared_plan: preparedUpdate?.plan?.map(({ relativePath, kind, action }) => ({ relativePath, kind, action })) || [],
    environment_repair_required: Boolean(preparedUpdate?.environmentRepairRequired),
    maximum_bytes: maximumUpdateArchiveBytes,
    current_version: appVersion,
    online_checked_at: onlineCheckedAt,
    online_release: publicOnlineRelease(),
  });
  const setState = (updates, immediate = false) => {
    updateState = { ...updateState, ...updates, updated_at: new Date().toISOString() };
    return persistUpdateState(immediate);
  };
  const report = (event) => setState({ phase: event.phase, progress: event.progress, message: event.message });

  /** Whether the configured repository can be read, which is what tells a 404 apart.
   *
   * Asked only for the built-in feed; `missingReleaseMeaning` decides what the answer means.
   */
  const releaseSourceExists = async () => {
    try {
      const repositoryUrl = releaseRepositoryUrl(releaseEnvironment);
      const { response: probe } = await fetchBoundedUpdateBody(repositoryUrl, {
        headers: {
          Accept: "application/vnd.github+json",
          "User-Agent": "XiriaCanvas-AI",
          ...updateAuthorizationHeaders(repositoryUrl, { environment: releaseEnvironment }),
        },
        timeoutMs: repositoryTimeoutMs,
        maximumBytes: maximumReleaseFeedBytes,
        fetcher: updateFetcher,
        environment: releaseEnvironment,
        label: "发布仓库探测",
      });
      return probe.ok;
    } catch {
      return false;
    }
  };

  /** Resolve the required checksum sidecar before a release can enter the download trust set. */
  const fetchReleaseChecksum = async (release) => {
    const customFeed = hasCustomReleaseFeed(releaseEnvironment);
    const [route] = checksumRoutes(release, {
      environment: releaseEnvironment,
      allowToken: !customFeed,
    });
    if (!route) throw Object.assign(new Error("发布信息缺少必需的 SHA-256 校验和文件"), { statusCode: 502 });
    let result;
    try {
      result = await fetchBoundedUpdateBody(route.url, {
        headers: { "User-Agent": "XiriaCanvas-AI", ...route.headers },
        timeoutMs: sidecarTimeoutMs,
        maximumBytes: maximumReleaseSidecarBytes,
        fetcher: updateFetcher,
        environment: releaseEnvironment,
        label: "校验和文件",
      });
    } catch (error) {
      throw Object.assign(new Error(`无法获取发布校验和：${error.message}`), { statusCode: 502 });
    }
    if (!result.response.ok) {
      throw Object.assign(new Error(`无法获取发布校验和：服务器返回 HTTP ${result.response.status}`), { statusCode: 502 });
    }
    const checksum = parseChecksumFile(result.body.toString("utf8"), release.asset.name);
    if (!checksum) {
      throw Object.assign(new Error("发布校验和文件格式无效或文件名不匹配"), { statusCode: 502 });
    }
    if (release.asset.advertisedSha256 && release.asset.advertisedSha256 !== checksum) {
      throw Object.assign(new Error("发布资产摘要与校验和文件不一致"), { statusCode: 502 });
    }
    return checksum;
  };

  /** Reads the release feed and remembers what it offers. */
  const checkForRelease = async () => {
    const generation = ++onlineCheckGeneration;
    // A failed refresh must revoke the prior release immediately. Otherwise /download would still
    // install the last successful answer after the current source had become malformed or hostile.
    onlineRelease = null;
    onlineCheckedAt = null;
    let feedUrl;
    let customFeed;
    try {
      feedUrl = releaseFeedUrl(releaseEnvironment);
      customFeed = hasCustomReleaseFeed(releaseEnvironment);
    } catch (error) {
      throw Object.assign(error, { statusCode: error.statusCode || 500 });
    }
    let feedResult;
    let authenticatedFeed = false;
    try {
      const authorizationHeaders = updateAuthorizationHeaders(feedUrl, {
        environment: releaseEnvironment,
        allowToken: !customFeed,
      });
      authenticatedFeed = Object.keys(authorizationHeaders)
        .some((name) => name.toLowerCase() === "authorization");
      feedResult = await fetchBoundedUpdateBody(feedUrl, {
        headers: {
          Accept: "application/vnd.github+json",
          "User-Agent": "XiriaCanvas-AI",
          // A custom feed is untrusted data even when its hostname happens to be github.com.
          ...authorizationHeaders,
        },
        timeoutMs: releaseTimeoutMs,
        maximumBytes: maximumReleaseFeedBytes,
        fetcher: updateFetcher,
        environment: releaseEnvironment,
        label: "发布信息",
      });
    } catch (error) {
      throw Object.assign(new Error(`无法连接更新服务器：${error.message}`), { statusCode: 502 });
    }
    const feedResponse = feedResult.response;
    if (feedResponse.status === 404) {
      // A repository that exists but has published nothing answers 404 exactly as a private or
      // misspelled one does. That is not a failure to report: it means there is no newer version,
      // which is the ordinary state of a project between releases. Only when the repository itself
      // cannot be read is the configuration actually wrong.
      const meaning = missingReleaseMeaning({
        customFeed,
        repositoryReachable: customFeed ? false : await releaseSourceExists(),
      });
      if (meaning === "no-release") {
        if (generation === onlineCheckGeneration) {
          onlineRelease = null;
          onlineCheckedAt = new Date().toISOString();
        }
        return;
      }
      throw Object.assign(new Error("未找到发布源，请确认仓库地址与 Release 是否公开"), { statusCode: 502 });
    }
    if (feedResponse.status === 403 || feedResponse.status === 429) {
      throw Object.assign(new Error("更新服务器暂时限制了请求频率，请稍后再试"), { statusCode: 503 });
    }
    if (!feedResponse.ok) {
      throw Object.assign(new Error(`更新服务器返回 HTTP ${feedResponse.status}`), { statusCode: 502 });
    }
    let payload;
    try {
      payload = JSON.parse(feedResult.body.toString("utf8"));
    } catch {
      throw Object.assign(new Error("更新服务器返回的内容不是有效的发布信息"), { statusCode: 502 });
    }
    let release;
    try {
      release = parseRelease(payload);
    } catch (error) {
      throw Object.assign(new Error(`发布信息不可用于在线更新：${error.message}`), { statusCode: 502 });
    }
    const resolvedChecksum = await fetchReleaseChecksum(release);
    // Concurrent check requests may finish out of order. Only the newest initiated check may name
    // what /download is allowed to fetch; an older success cannot overwrite a newer failure/result.
    if (generation === onlineCheckGeneration) {
      onlineRelease = {
        ...release,
        resolvedChecksum,
        allowToken: !customFeed,
        // This is derived from the request that actually authenticated, not token presence alone.
        privateContext: authenticatedFeed,
      };
      onlineCheckedAt = new Date().toISOString();
    }
  };
  const validationReport = (event) => report({
    ...event,
    progress: Math.min(99, 92 + Math.round((event.progress || 0) * 0.07)),
  });
  const validateAndRepair = ({ forceRepair = false, setEnvironmentBackup } = {}) => ensureUpdatedProjectReady({
    forceRepair,
    report: validationReport,
    validate: () => validateUpdatedProject({ projectRoot, environment: process.env, report: validationReport }),
    repair: () => repairUpdatedEnvironment({ projectRoot, environment: process.env, report: validationReport, onBackup: setEnvironmentBackup }),
  });
  const writeRepairTransaction = async (transaction) => {
    await mkdir(path.dirname(manualUpdateEnvironmentTransactionPath), { recursive: true });
    const temporaryPath = `${manualUpdateEnvironmentTransactionPath}.${process.pid}.${randomUUID()}.tmp`;
    try {
      await writeFile(temporaryPath, `${JSON.stringify(transaction, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
      await rename(temporaryPath, manualUpdateEnvironmentTransactionPath);
    } finally {
      await rm(temporaryPath, { force: true });
    }
  };
  const unsafeUpdateTransactionPending = async () => {
    for (const transactionPath of [manualUpdateTransactionPath, manualUpdateEnvironmentTransactionPath]) {
      try {
        const transaction = JSON.parse(await readFile(transactionPath, "utf8"));
        if (!["committed", "environment-committed", "rolled-back"].includes(transaction?.phase)) return true;
      } catch (error) {
        if (error.code !== "ENOENT") return true;
      }
    }
    return false;
  };
  const validateAndRepairStandalone = async ({ forceRepair = false } = {}) => {
    let transaction;
    let result;
    try {
      result = await validateAndRepair({
        forceRepair,
        setEnvironmentBackup: async (environmentBackup) => {
          transaction = {
            schema: 1,
            product: "XiriaCanvas AI",
            projectRoot,
            phase: "environment-repair",
            environmentBackup,
            setupMarker: readSetupMarker(projectRoot),
            createdAt: new Date().toISOString(),
          };
          await writeRepairTransaction(transaction);
        },
      });
      if (transaction) {
        transaction.phase = "environment-committed";
        try {
          await writeRepairTransaction(transaction);
        } catch (journalError) {
          try {
            await result.rollback?.();
          } catch (rollbackError) {
            const failure = new Error(`环境已验证，但提交恢复日志失败且旧环境回滚不完整：${rollbackError.message}`, { cause: journalError });
            failure.rollbackIncomplete = true;
            throw failure;
          }
          await rm(manualUpdateEnvironmentTransactionPath, { force: true }).catch(() => {});
          throw journalError;
        }
      }
      let cleanupWarning;
      try {
        await result.commit?.();
      } catch (error) {
        cleanupWarning = error.message;
      }
      if (transaction && !cleanupWarning) await rm(manualUpdateEnvironmentTransactionPath, { force: true });
      if (cleanupWarning) result.cleanupWarnings = [cleanupWarning];
      return result;
    } catch (error) {
      if (transaction && !error.rollbackIncomplete) await rm(manualUpdateEnvironmentTransactionPath, { force: true }).catch(() => {});
      throw error;
    }
  };
  const acquireTaskLock = async (operation) => {
    operationLock = await acquireOfflineUpdateLock({ stateDirectory, projectRoot, operation });
  };
  const releaseTaskLock = async () => {
    const current = operationLock;
    operationLock = undefined;
    await current?.release();
  };
  const cleanupPrepared = async () => {
    const stagingOwnership = preparedUpdate?.stagingOwnership;
    const currentUploadOwnership = uploadOwnership;
    preparedUpdate = undefined;
    uploadDirectory = undefined;
    uploadOwnership = undefined;
    await Promise.all([
      stagingOwnership && removeOfflineUpdateTemp({ projectRoot, record: stagingOwnership, prefix: "xirai-update-stage-", kind: "stage" }),
      currentUploadOwnership && removeOfflineUpdateTemp({ projectRoot, record: currentUploadOwnership, prefix: "xirai-update-upload-", kind: "upload" }),
    ].filter(Boolean).map((operation) => operation.catch(() => {})));
    await persistUpdateState(true);
  };
  const failTask = async (error) => {
    writeDiagnosticLog(projectRoot, {
      kind: "manual-update-failure",
      message: error.message,
      details: { phase: updateState.phase },
    });
    await cleanupPrepared();
    await setState({ status: "error", phase: "error", progress: 0, message: error.message, environment_ready: false, repair_available: false }, true);
  };

  const register = (server) => {
    server.middlewares.use(async (request, response, next) => {
      const url = new URL(request.url, "http://localhost");
      if (!url.pathname.startsWith("/api/system/update")) {
        next();
        return;
      }
      response.setHeader("Content-Type", "application/json; charset=utf-8");
      response.setHeader("Cache-Control", "no-store");
      let acquiredTaskLock = false;
      try {
        await updateStateReady;
        requireLocalRequest(request, "手动更新只能在运行程序的本机执行");
        if (url.pathname === "/api/system/update" && request.method === "GET") {
          response.statusCode = 200;
          response.end(JSON.stringify(publicState()));
          return;
        }
        if (url.pathname === "/api/system/update" && request.method === "DELETE") {
          if (activeTask || operationLock || updateBusy(updateState.status)) throw Object.assign(new Error("更新任务正在运行，暂时不能离开"), { statusCode: 409 });
          await cleanupPrepared();
          setState({ status: "idle", phase: "idle", source: undefined, progress: 0, message: "请选择更新归档", filename: undefined, bytes: undefined, restart_required: false, environment_ready: false, repair_available: false });
          response.statusCode = 200;
          response.end(JSON.stringify(publicState()));
          return;
        }
        if (url.pathname === "/api/system/update/check" && request.method === "POST") {
          if (activeTask || operationLock) throw Object.assign(new Error("更新任务正在运行，暂时不能检查新版本"), { statusCode: 409 });
          await checkForRelease();
          response.statusCode = 200;
          response.end(JSON.stringify(publicState()));
          return;
        }
        if (url.pathname === "/api/system/update/download" && request.method === "POST") {
          if (activeTask || operationLock || !["idle", "error"].includes(updateState.status)) throw Object.assign(new Error("已有更新包正在处理"), { statusCode: 409 });
          // The check is the only thing that may name what gets downloaded: a client cannot hand
          // this endpoint a URL of its own.
          if (!onlineRelease) throw Object.assign(new Error("请先检查更新"), { statusCode: 409 });
          if (!updateAvailable(onlineRelease.version, appVersion)) throw Object.assign(new Error("当前已是最新版本"), { statusCode: 409 });
          const release = onlineRelease;
          const filename = release.asset.name;
          if (filename !== path.basename(filename) || !archiveExtensionAllowed(filename)) {
            throw Object.assign(new Error("发布的更新包格式不受支持"), { statusCode: 502 });
          }
          if (release.asset.bytes > maximumUpdateArchiveBytes) throw Object.assign(new Error("更新包不能超过 4 GB"), { statusCode: 413 });
          await acquireTaskLock("online-download");
          acquiredTaskLock = true;
          await cleanupPrepared();
          uploadOwnership = await createOfflineUpdateTemp({ projectRoot, prefix: "xirai-update-upload-", kind: "upload" });
          uploadDirectory = uploadOwnership.path;
          const archivePath = path.join(uploadDirectory, filename);
          setState({
            status: "downloading",
            phase: "download",
            source: "online",
            progress: 0,
            message: `正在下载 ${release.version} 更新包`,
            filename,
            bytes: 0,
            restart_required: false,
            environment_ready: false,
            repair_available: false,
          });
          activeTask = (async () => {
            const checksum = release.resolvedChecksum;
            const routes = releaseDownloadRoutes(release, {
              checksum,
              environment: releaseEnvironment,
              allowToken: release.allowToken === true,
              privateContext: release.privateContext === true,
            });
            if (!routes.length) throw new Error("发布信息中没有已验证、可下载的更新包");
            await downloadFile({
              routes,
              destination: archivePath,
              expectedSha256: checksum || undefined,
              maximumBytes: maximumUpdateArchiveBytes,
              sizeHint: release.asset.bytes,
              connections: 8,
              fetcher: (downloadUrl, options) => fetchDownload(downloadUrl, options, {
                fetcher: updateFetcher,
                environment: releaseEnvironment,
              }),
              // Every admitted release is checksum-pinned; ranking may therefore compare mirrors
              // without allowing any sampled bytes to bypass the final digest check.
              rankRoutes: Boolean(checksum),
              existingFilePolicy: "replace",
              onRoute: (route) => setState({ message: `正在通过 ${route.label} 下载 ${release.version} 更新包` }),
              onProgress: (progress) => setState({
                bytes: progress.currentBytes || 0,
                progress: progress.totalBytes ? Math.round((progress.currentBytes || 0) / progress.totalBytes * 100) : 0,
              }),
              onWarning: (warning) => setState({ message: warning.message || `正在下载 ${release.version} 更新包` }),
            });
            setState({ status: "preparing", phase: "inspect", progress: 0, message: "正在解析更新包" });
            const prepared = await prepareUpdate({ projectRoot, archivePath, report });
            preparedUpdate = prepared;
            await setState({
              status: "ready",
              phase: "ready",
              progress: 100,
              message: prepared.environmentRepairRequired
                ? `${release.version} 更新包校验通过；替换后将自动修复环境`
                : `${release.version} 更新包校验通过，准备应用`,
              environment_ready: false,
              repair_available: false,
            }, true);
          })().catch(failTask).finally(async () => {
            activeTask = undefined;
            await releaseTaskLock().catch(() => {});
          });
          acquiredTaskLock = false;
          response.statusCode = 202;
          response.end(JSON.stringify(publicState()));
          return;
        }
        if (url.pathname === "/api/system/update/archive" && request.method === "POST") {
          if (activeTask || operationLock || !["idle", "error"].includes(updateState.status)) throw Object.assign(new Error("已有更新包正在处理"), { statusCode: 409 });
          const encodedName = String(request.headers["x-archive-name"] || "");
          let filename;
          try {
            filename = decodeURIComponent(encodedName);
          } catch {
            throw Object.assign(new Error("更新包文件名无效"), { statusCode: 400 });
          }
          if (!filename || filename !== path.basename(filename) || !archiveExtensionAllowed(filename)) {
            throw Object.assign(new Error("请选择 ZIP、7Z、RAR、TAR、TAR.GZ 或 TAR.XZ 更新包"), { statusCode: 400 });
          }
          const declaredBytes = Number(request.headers["content-length"] || 0);
          if (declaredBytes > maximumUpdateArchiveBytes) throw Object.assign(new Error("更新包不能超过 4 GB"), { statusCode: 413 });
          await acquireTaskLock("archive-preparation");
          acquiredTaskLock = true;
          await cleanupPrepared();
          uploadOwnership = await createOfflineUpdateTemp({ projectRoot, prefix: "xirai-update-upload-", kind: "upload" });
          uploadDirectory = uploadOwnership.path;
          const archivePath = path.join(uploadDirectory, filename);
          const destination = await open(archivePath, "wx");
          let bytes = 0;
          setState({ status: "uploading", phase: "upload", source: "local", progress: 0, message: "正在接收更新包", filename, bytes: 0, restart_required: false, environment_ready: false, repair_available: false });
          try {
            for await (const chunk of request) {
              if (request.aborted) throw new Error("更新包上传中断，请重新选择归档");
              bytes += chunk.length;
              if (bytes > maximumUpdateArchiveBytes) throw Object.assign(new Error("更新包不能超过 4 GB"), { statusCode: 413 });
              let offset = 0;
              while (offset < chunk.length) {
                const { bytesWritten } = await destination.write(chunk, offset, chunk.length - offset);
                if (!bytesWritten) throw new Error("无法写入更新包");
                offset += bytesWritten;
              }
              setState({ bytes, progress: declaredBytes ? Math.round(bytes / declaredBytes * 100) : 0, message: "正在接收更新包" });
            }
            await destination.sync();
          } catch (error) {
            await destination.close().catch(() => {});
            await cleanupPrepared();
            await setState({ status: "error", phase: "upload", progress: 0, message: error.message || "更新包上传中断，请重新选择归档", restart_required: false }, true);
            await releaseTaskLock();
            throw error;
          }
          await destination.close();
          if (!bytes) {
            await cleanupPrepared();
            await setState({ status: "error", phase: "upload", progress: 0, message: "更新包为空，请重新选择归档", restart_required: false }, true);
            await releaseTaskLock();
            throw Object.assign(new Error("更新包为空"), { statusCode: 400 });
          }
          setState({ status: "preparing", phase: "inspect", progress: 0, message: "正在解析更新包", filename, bytes });
          activeTask = prepareUpdate({ projectRoot, archivePath, report })
            .then(async (prepared) => {
              preparedUpdate = prepared;
              await setState({
                status: "ready",
                phase: "ready",
                progress: 100,
                message: prepared.environmentRepairRequired ? "更新包校验通过；替换后将自动修复环境" : "更新包校验通过，准备应用",
                environment_ready: false,
                repair_available: false,
              }, true);
            })
            .catch(failTask)
            .finally(async () => {
              activeTask = undefined;
              await releaseTaskLock().catch(() => {});
            });
          acquiredTaskLock = false;
          response.statusCode = 202;
          response.end(JSON.stringify(publicState()));
          return;
        }
        if (url.pathname === "/api/system/update/apply" && request.method === "POST") {
          if (activeTask || operationLock || updateState.status !== "ready" || !preparedUpdate) throw Object.assign(new Error("更新包尚未准备完成"), { statusCode: 409 });
          await acquireTaskLock("archive-apply");
          acquiredTaskLock = true;
          setState({ status: "applying", phase: "shutdown", progress: 0, message: "正在停止推理服务" });
          activeTask = (async () => {
            const rollbackEnvironmentRepairRequired = Boolean(preparedUpdate.environmentRepairRequired);
            await stopInferenceForUpdate();
            setState({ restart_required: false, environment_ready: false, repair_available: false });
            await server.watcher?.close();
            const result = await applyPreparedUpdate({
              projectRoot,
              prepared: preparedUpdate,
              transactionPath: manualUpdateTransactionPath,
              report,
              validate: ({ environmentRepairRequired, setEnvironmentBackup }) => validateAndRepair({ forceRepair: environmentRepairRequired, setEnvironmentBackup }),
            });
            const completedUploadDirectory = uploadDirectory;
            const completedUploadOwnership = uploadOwnership;
            preparedUpdate = undefined;
            uploadDirectory = undefined;
            uploadOwnership = undefined;
            if (completedUploadDirectory && completedUploadOwnership) {
              await removeOfflineUpdateTemp({ projectRoot, record: completedUploadOwnership, prefix: "xirai-update-upload-", kind: "upload" }).catch(() => {});
            }
            await setState({
              status: "complete",
              phase: "complete",
              progress: 100,
              message: result.validation?.repaired
                ? "程序文件更新完成，环境已自动修复并验证通过"
                : "程序文件更新完成，环境与后端验证通过",
              updated_items: result.plan.map((item) => item.relativePath),
              removed_items: result.plan.filter((item) => item.action === "remove").map((item) => item.relativePath),
              restart_required: true,
              environment_ready: true,
              environment_repaired: Boolean(result.validation?.repaired),
              repair_available: false,
            }, true);
          })().catch(async (error) => {
            writeDiagnosticLog(projectRoot, {
              kind: "manual-update-failure",
              message: error.message,
              details: { phase: updateState.phase, rolled_back: !error.rollbackIncomplete, rollback_incomplete: Boolean(error.rollbackIncomplete) },
            });
            await cleanupPrepared();
            if (error.rollbackIncomplete) {
              await setState({
                status: "error",
                phase: "error",
                progress: 0,
                message: `${error.message}；为避免在混合状态下运行，已禁止重启，请关闭程序后重新启动以执行恢复`,
                restart_required: false,
                environment_ready: false,
                repair_available: false,
              }, true);
              return;
            }
            try {
              await setState({ status: "repairing", phase: "environment", progress: 92, message: "更新已回滚，正在检查原环境", restart_required: false, environment_ready: false }, true);
              const recovery = await validateAndRepairStandalone({ forceRepair: rollbackEnvironmentRepairRequired });
              if (!readSetupMarker(projectRoot)) throw new Error("原环境验证完成后配置标记失效");
              if (await unsafeUpdateTransactionPending()) throw new Error("回滚事务尚未完成，拒绝标记为可重启");
              await setState({
                status: "error",
                phase: "error",
                progress: 0,
                message: `${error.message}；原程序环境${recovery.repaired ? "已自动修复并" : "已"}验证通过，可安全重启返回`,
                restart_required: true,
                environment_ready: true,
                environment_repaired: recovery.repaired,
                repair_available: false,
              }, true);
            } catch (recoveryError) {
              await setState({
                status: "error",
                phase: "error",
                progress: 0,
                message: `${error.message}；更新已回滚，但环境仍不可运行：${recoveryError.message}`,
                restart_required: false,
                environment_ready: false,
                repair_available: true,
              }, true);
            }
          }).finally(async () => {
            activeTask = undefined;
            await releaseTaskLock().catch(() => {});
          });
          acquiredTaskLock = false;
          response.statusCode = 202;
          response.end(JSON.stringify(publicState()));
          return;
        }
        if (url.pathname === "/api/system/update/repair" && request.method === "POST") {
          if (activeTask || operationLock || !updateState.repair_available || updateState.environment_ready) throw Object.assign(new Error("当前不需要环境修复"), { statusCode: 409 });
          await acquireTaskLock("environment-repair");
          acquiredTaskLock = true;
          setState({ status: "repairing", phase: "repair", progress: 92, message: "正在重新自动修复环境", repair_available: false });
          activeTask = validateAndRepairStandalone({ forceRepair: true })
            .then(async () => {
              if (!readSetupMarker(projectRoot)) throw new Error("环境修复后配置标记无效");
              return setState({
                status: "error",
                phase: "error",
                progress: 0,
                message: "环境自动修复并验证通过，可安全重启返回主页面",
                restart_required: true,
                environment_ready: true,
                environment_repaired: true,
                repair_available: false,
              }, true);
            })
            .catch(async (error) => setState({
              status: "error",
              phase: "error",
              progress: 0,
              message: `环境自动修复失败：${error.message}`,
              restart_required: false,
              environment_ready: false,
              repair_available: true,
            }, true))
            .finally(async () => {
              activeTask = undefined;
              await releaseTaskLock().catch(() => {});
            });
          acquiredTaskLock = false;
          response.statusCode = 202;
          response.end(JSON.stringify(publicState()));
          return;
        }
        if (url.pathname === "/api/system/update/restart" && request.method === "POST") {
          if (activeTask || operationLock || !updateState.restart_required || !updateState.environment_ready || !readSetupMarker(projectRoot)) {
            throw Object.assign(new Error("环境尚未验证通过，暂时不能重启返回主页面"), { statusCode: 409 });
          }
          if (await unsafeUpdateTransactionPending()) {
            throw Object.assign(new Error("离线更新事务尚未安全结束，暂时不能重启"), { statusCode: 409 });
          }
          setState({
            status: "idle",
            phase: "idle",
            progress: 0,
            message: "请选择更新归档",
            filename: undefined,
            bytes: undefined,
            prepared_items: undefined,
            updated_items: undefined,
            restart_required: false,
            environment_ready: false,
            environment_repaired: undefined,
            repair_available: false,
          });
          await persistUpdateState(true);
          response.statusCode = 202;
          response.end(JSON.stringify({ status: "restarting", url: "/" }));
          setTimeout(async () => {
            const supervised = process.env.XIRAI_SERVICE_SUPERVISOR === "1";
            const handOffRestart = createUpdateRestartHandoff({ supervised, projectRoot });
            const forceRestart = setTimeout(handOffRestart, 5000);
            try {
              await server.close();
            } finally {
              clearTimeout(forceRestart);
              handOffRestart();
            }
          }, 150);
          return;
        }
        throw Object.assign(new Error("Method not allowed"), { statusCode: 405 });
      } catch (error) {
        if (acquiredTaskLock) await releaseTaskLock().catch(() => {});
        response.statusCode = error.statusCode || 500;
        response.end(JSON.stringify({ error: error.message || "更新请求失败" }));
      }
    });
  };
  return {
    name: "local-archive-update-api",
    configureServer: register,
    configurePreviewServer: register,
  };
}

function systemApiPlugin() {
  const supervised = process.env.XIRAI_SERVICE_SUPERVISOR === "1";
  const register = (server) => {
    server.middlewares.use((request, response, next) => {
      const url = new URL(request.url, "http://localhost");
      if (url.pathname !== "/api/system/reconfigure") {
        next();
        return;
      }
      response.setHeader("Content-Type", "application/json; charset=utf-8");
      response.setHeader("Cache-Control", "no-store");
      if (request.method !== "POST") {
        response.statusCode = 405;
        response.end(JSON.stringify({ error: "Method not allowed" }));
        return;
      }
      if (request.headers.origin) {
        try {
          if (new URL(request.headers.origin).host !== request.headers.host) {
            response.statusCode = 403;
            response.end(JSON.stringify({ error: "Cross-origin system control is not allowed" }));
            return;
          }
        } catch {
          response.statusCode = 403;
          response.end(JSON.stringify({ error: "Invalid request origin" }));
          return;
        }
      }

      response.statusCode = 202;
      response.end(JSON.stringify({ status: "switching", url: "/config" }));
      setTimeout(async () => {
        const handOffConfigurator = createConfiguratorHandoff({ supervised, projectRoot });
        const forceSwitch = setTimeout(handOffConfigurator, 5000);
        try {
          await server.close();
        } finally {
          clearTimeout(forceSwitch);
          handOffConfigurator();
        }
      }, 150);
    });
  };
  return {
    name: "system-control-api",
    configureServer: register,
    configurePreviewServer: register,
  };
}

export default defineConfig({
  plugins: [setupGatePlugin(), themedLogoPlugin(), react(), ...(setupComplete ? [logApiPlugin(), uiStateApiPlugin(), assistantApiPlugin(), pluginRegistryApiPlugin(), modelApiPlugin(), updateApiPlugin(), systemApiPlugin(), inferenceBackendPlugin()] : [])],
  cacheDir: path.join(cacheDirectory, "vite"),
  define: {
    // The browser may be on a different machine than the folders the user is
    // typing paths for, so the path hints have to come from the host. See
    // src/host-platform.js.
    __XIRAI_HOST_PLATFORM__: JSON.stringify(process.platform),
    // Read from package.json at build time so the About panel and the update check can never
    // disagree about which release is running. See scripts/app-version.mjs.
    __XIRAI_APP_VERSION__: JSON.stringify(appVersion),
  },
  build: {
    rolldownOptions: {
      output: {
        // The React runtime changes only when the dependency is upgraded, so it is kept out of the
        // application chunk: an ordinary UI edit then leaves the largest single asset cached instead
        // of invalidating it. `minSize` is below the runtime's own size so the group always forms.
        codeSplitting: {
          minSize: 20000,
          groups: [
            { name: "vendor-react", test: /node_modules[\\/](react|react-dom|scheduler)[\\/]/ },
            // Icons are one module per glyph. Without a group, every glyph two routes happen to share
            // becomes its own sub-kilobyte chunk; one shared icon chunk replaces that fan-out.
            { name: "vendor-icons", test: /node_modules[\\/]lucide-react[\\/]/, minSize: 0 },
          ],
        },
      },
    },
  },
  server: {
    host: webHost,
    port: webPort,
    strictPort: true,
    headers: webSecurityHeaders,
    watch: {
      ignored: ignoreRuntimeWatchPath,
    },
    proxy: {
      "/api/inference": { target: inferenceTarget, changeOrigin: true },
    },
  },
  preview: {
    host: webHost,
    port: webPort,
    strictPort: true,
    headers: webSecurityHeaders,
    proxy: {
      "/api/inference": { target: inferenceTarget, changeOrigin: true },
    },
  },
});
