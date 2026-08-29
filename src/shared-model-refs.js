// Shared model roots let the workspace read model files that live in another
// tool's install (ComfyUI, sd-webui, ...) instead of copying tens of gigabytes
// into `models/`. They are a deliberately separate contract from
// `models/model-paths.json`, which stays project-relative and stays the only
// thing downloads may ever write into. A shared root is read-only by design.
//
// This module is the part both planes and the browser agree on: how a shared
// file is named on the wire, how a root is identified, and which folder names
// map to which kind of model. It touches no filesystem so it can be imported
// from the UI, from the Node control plane, and mirrored by the Python twin
// (`backend/shared_model_paths.py`).

export const SHARED_REF_PREFIX = "shared:";
export const SHARED_ROOT_ID_LENGTH = 12;
export const MAX_SHARED_ROOTS = 24;

// Model files are referenced relative to their root everywhere in this app —
// in `ui-state.json`, in saved PNG parameter blocks, in gallery records. An
// absolute path on the wire would leak the user's disk layout into all three
// and would defeat the containment checks that expect a relative path. So a
// shared file keeps that shape and carries its root in an explicit namespace.
export function formatSharedRef(rootId, relativePath) {
  const relative = String(relativePath ?? "").split("\\").join("/").replace(/^\/+/, "");
  if (!isSharedRootId(rootId) || !relative) return "";
  return `${SHARED_REF_PREFIX}${rootId}/${relative}`;
}

export function isSharedRef(value) {
  return typeof value === "string" && value.startsWith(SHARED_REF_PREFIX);
}

export function parseSharedRef(value) {
  if (!isSharedRef(value)) return null;
  const body = value.slice(SHARED_REF_PREFIX.length);
  const separator = body.indexOf("/");
  if (separator <= 0) return null;
  const rootId = body.slice(0, separator);
  const relativePath = body.slice(separator + 1);
  if (!isSharedRootId(rootId) || !relativePath) return null;
  // `..` can only ever be an escape attempt: the listing never produces one.
  const segments = relativePath.split("/");
  if (segments.some((segment) => !segment || segment === "." || segment === "..")) return null;
  return { rootId, relativePath };
}

export function isSharedRootId(value) {
  return typeof value === "string" && new RegExp(`^[0-9a-f]{${SHARED_ROOT_ID_LENGTH}}$`).test(value);
}

// A root id is derived from its path rather than issued from a counter, so
// removing a root and adding the same folder back resolves every LoRA the user
// had already mounted. The normalisation below is the half of that derivation
// that has to agree byte-for-byte with the Python twin; the hash itself is done
// where a crypto primitive is available.
//
// Spelling differences that both platforms agree are the same folder — `\` vs
// `/`, repeated separators, a trailing slash — are folded here.
export function normalizeSharedRootPath(value) {
  const text = String(value ?? "").trim().split("\\").join("/");
  const collapsed = text.replace(/\/{2,}/g, "/");
  return collapsed.length > 1 ? collapsed.replace(/\/+$/, "") : collapsed;
}

// Case is the one difference the two platforms disagree about. Windows treats
// `F:\AI` and `f:\ai` as one folder, so the id must survive a retype; Linux
// treats `/srv/Lora` and `/srv/lora` as two real directories, and folding them
// would give both the same id — the second registration would then be
// discarded as a duplicate of the first. So the caller that knows which
// filesystem it is standing on decides, and only the id derivation asks.
export function sharedRootIdentityPath(value, { caseInsensitive = false } = {}) {
  const normalized = normalizeSharedRootPath(value);
  return caseInsensitive ? normalized.toLowerCase() : normalized;
}

export const SHARED_ROOT_KINDS = ["auto", "loras", "checkpoints"];

// What a folder called `Lora` or `upscale_models` actually holds. Names come
// from the layouts this feature exists to read: sd-webui (`Stable-diffusion`,
// `Lora`, `VAE`, `ESRGAN`), ComfyUI (`checkpoints`, `loras`, `upscale_models`,
// `unet`), and this project's own `models/` tree. Matching is case-insensitive
// because sd-webui capitalises where ComfyUI does not.
export const SHARED_DIRECTORY_KINDS = {
  checkpoints: ["checkpoints", "checkpoint", "stable-diffusion", "stablediffusion", "sd", "illustrious", "ckpt"],
  loras: ["loras", "lora", "locon", "lycoris", "lokr"],
  vae: ["vae"],
  text_encoders: ["text_encoders", "text_encoder", "clip"],
  diffusion_models: ["diffusion_models", "unet"],
  embeddings: ["embeddings", "embedding", "textual_inversion", "textualinversion"],
  upscalers: ["upscalers", "upscale_models", "esrgan", "realesrgan", "swinir", "scunet", "ldsr"],
  yolo: ["yolo", "ultralytics", "adetailer"],
  controlnet: ["controlnet", "controlnets"],
  hypernetworks: ["hypernetworks", "hypernetwork"],
};

// Detection is deliberately wider than support. Reporting a ControlNet folder
// as "found, not usable yet" is honest; silently dropping it would leave the
// user wondering why the scan missed half their library.
export const SHARED_RUNTIME_KINDS = ["checkpoints", "loras"];

export const SHARED_KIND_LABELS = {
  checkpoints: "底模",
  loras: "LoRA",
  vae: "VAE",
  text_encoders: "文本编码器",
  diffusion_models: "扩散模型",
  embeddings: "嵌入",
  upscalers: "超分模型",
  yolo: "YOLO 检测",
  controlnet: "ControlNet",
  hypernetworks: "Hypernetwork",
};

export const SHARED_KIND_EXTENSIONS = {
  checkpoints: [".safetensors", ".ckpt"],
  loras: [".safetensors", ".ckpt", ".pt", ".pth"],
  vae: [".safetensors", ".ckpt", ".pt"],
  text_encoders: [".safetensors", ".ckpt", ".pt", ".bin"],
  diffusion_models: [".safetensors", ".ckpt", ".gguf"],
  embeddings: [".safetensors", ".pt", ".bin"],
  upscalers: [".pth", ".pt", ".ckpt", ".safetensors"],
  yolo: [".pt", ".onnx"],
  controlnet: [".safetensors", ".ckpt", ".pth"],
  hypernetworks: [".pt", ".safetensors"],
};

export function classifyDirectoryName(name) {
  const folded = String(name ?? "").trim().toLowerCase();
  if (!folded) return "";
  for (const [kind, aliases] of Object.entries(SHARED_DIRECTORY_KINDS)) {
    if (aliases.includes(folded)) return kind;
  }
  return "";
}

export function sharedKindSupported(kind) {
  return SHARED_RUNTIME_KINDS.includes(kind);
}

export const SHARED_LORA_ENGINES = ["SD", "iL", "Anima", "Flux", "Flux2", "Krea2"];

// Full engine sets this field has previously shipped with. A root saved while one of these was the
// whole roster was never scoped to those engines — it simply predates the ones added since, so it
// reads as "not configured" rather than as "deliberately excludes Flux". Without this a folder
// registered before Flux existed would hide its Flux LoRAs, and there is no UI to re-tick.
const LEGACY_SHARED_LORA_ENGINE_SETS = [
  ["SD", "iL", "Anima"],
  ["SD", "iL", "Anima", "Flux"],
  ["SD", "iL", "Anima", "Flux", "Flux2"],
];

function sameEngineSet(first, second) {
  return first.length === second.length && first.every((engine, index) => engine === second[index]);
}

export function normalizeSharedEngines(value) {
  if (!Array.isArray(value)) return [...SHARED_LORA_ENGINES];
  const kept = SHARED_LORA_ENGINES.filter((engine) => value.includes(engine));
  if (LEGACY_SHARED_LORA_ENGINE_SETS.some((legacy) => sameEngineSet(kept, legacy))) return [...SHARED_LORA_ENGINES];
  // An empty list would silently hide the folder the user just registered, so
  // it reads as "not configured" rather than as "attached to nothing".
  return kept.length ? kept : [...SHARED_LORA_ENGINES];
}

export function normalizeSharedLabel(value, fallbackPath) {
  const label = String(value ?? "").trim().replace(/\s+/g, " ").slice(0, 60);
  if (label) return label;
  const segments = normalizeSharedRootPath(fallbackPath).split("/").filter(Boolean);
  return segments[segments.length - 1] || "共享目录";
}

// One shared root as the rest of the app sees it. `id` and `path` are the only
// fields the resolver trusts; everything else is presentation or filtering.
export function normalizeSharedRoot(entry) {
  if (!entry || typeof entry !== "object" || Array.isArray(entry)) return null;
  const path = String(entry.path ?? "").trim();
  if (!path) return null;
  const id = isSharedRootId(entry.id) ? entry.id : "";
  if (!id) return null;
  const kind = SHARED_ROOT_KINDS.includes(entry.kind) ? entry.kind : "auto";
  return {
    id,
    path,
    kind,
    label: normalizeSharedLabel(entry.label, path),
    enabled: entry.enabled !== false,
    engines: normalizeSharedEngines(entry.engines),
  };
}

export function normalizeSharedRoots(config) {
  const source = Array.isArray(config?.roots) ? config.roots : [];
  const roots = [];
  const seen = new Set();
  for (const entry of source) {
    const root = normalizeSharedRoot(entry);
    if (!root || seen.has(root.id)) continue;
    seen.add(root.id);
    roots.push(root);
    if (roots.length >= MAX_SHARED_ROOTS) break;
  }
  return roots;
}

export function findSharedRoot(roots, rootId) {
  return normalizeSharedRoots({ roots }).find((root) => root.id === rootId) || null;
}

// The LoRA manager groups local files by their first path segment. A shared
// file has no first segment of its own until its root is stripped, so the
// browser needs this to build the tree without re-deriving the ref format.
export function sharedRefSegments(value) {
  const parsed = parseSharedRef(value);
  return parsed ? parsed.relativePath.split("/") : [];
}
