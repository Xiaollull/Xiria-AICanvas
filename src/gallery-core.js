// Card settings shape, gallery transport and dialog lifecycle. Shared by the lazily loaded
// gallery page and the add-to-gallery dialog, which stays with the generate page.
import { useEffect, useRef } from "react";
import { SAMPLER_NAMES as SAMPLERS, SCHEDULER_NAMES as SCHEDULERS } from "./sampling-options.js";
import { normalizeGalleryHires, normalizeUint64Seed } from "./hires-settings.js";
import { normalizeADetailerStage } from "./adetailer-units.js";

export const DEFAULT_SETTINGS = {
  model: "SD",
  checkpoint: "",
  diffusionModel: "",
  textEncoder: "",
  vae: "",
  positive: "",
  negative: "",
  steps: 28,
  cfg: 6.5,
  denoise: 1,
  imagesPerBatch: 1,
  batchCount: 1,
  seed: "847291",
  seedMode: "fixed",
  sampler: "dpmpp_2m",
  scheduler: "karras",
  guidance: "none",
  pag: { scale: 0.3, appliedLayers: "mid" },
  size: { width: 1024, height: 1024 },
  processPreview: true,
  backgroundRemovalModel: "",
  hires: { enabled: false, expanded: false, model: "", seedMode: "inherit", seed: "", scale: 1, denoise: 0.35, steps: 20, cfg: 7, tileSize: 192, tileOverlap: 16, executionMode: "full_frame", sampler: null, scheduler: null, tileWidth: "auto", tileHeight: "auto", padding: 32, maskBlur: 8, seamMode: "none", uniformTiles: true, tiledDecode: true },
  adetailer: normalizeADetailerStage({ enabled: false, expanded: false }),
  rtx: { enabled: false, expanded: false, scale: 2, quality: "ultra" },
  postprocessOrder: ["hires", "adetailer", "rtx"],
  loras: [],
  // Which LoRA combinations were switched on for this generation, and the prefix
  // they contributed. This is a *record of the run*, not the group library:
  // `loraGroupsByEngine` describes the whole workspace and is excluded, but
  // without these two fields a card could not show the prompt that actually
  // produced the image, only the part the user typed.
  loraGroups: [],
  loraGroupPrompt: "",
};

const CARD_GROUP_ID = /^[0-9a-z-]{1,64}$/;

/** Bounded, self-healing copy of the groups that were enabled for a generation. */
export function normalizeCardLoraGroups(value) {
  if (!Array.isArray(value)) return [];
  const seen = new Set();
  const groups = [];
  for (const entry of value) {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) continue;
    const id = typeof entry.id === "string" && CARD_GROUP_ID.test(entry.id) ? entry.id : "";
    if (!id || seen.has(id) || groups.length >= 32) continue;
    seen.add(id);
    const name = typeof entry.name === "string" ? entry.name.replace(/\s+/g, " ").trim().slice(0, 60) : "";
    groups.push({
      id,
      name: name || id,
      presetPrompt: typeof entry.presetPrompt === "string" ? entry.presetPrompt.slice(0, 2000).trim() : "",
    });
  }
  return groups;
}

export const GUIDANCE = [["none", "无（None）"], ["pag", "PAG（扰动注意力引导）"], ["cfg_zero_star", "CFG-Zero*（零星 CFG）"]];

export function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

export function normalizedSettings(value, fallback = DEFAULT_SETTINGS, { hiresSourceKind = "persisted_card" } = {}) {
  const source = value && typeof value === "object" ? value : {};
  const base = fallback && typeof fallback === "object" ? fallback : DEFAULT_SETTINGS;
  const model = ["SD", "iL", "Anima"].includes(source.model)
    ? source.model
    : ["SD", "iL", "Anima"].includes(base.model) ? base.model : "SD";
  const normalized = {
    ...clone(DEFAULT_SETTINGS),
    ...clone(base),
    ...clone(source),
    model,
    seed: normalizeUint64Seed(source.seed, normalizeUint64Seed(base.seed, DEFAULT_SETTINGS.seed)),
    seedMode: ["fixed", "random", "increment", "decrement"].includes(source.seedMode)
      ? source.seedMode
      : ["fixed", "random", "increment", "decrement"].includes(base.seedMode) ? base.seedMode : DEFAULT_SETTINGS.seedMode,
    checkpoint: typeof source.checkpoint === "string" ? source.checkpoint : typeof base.checkpoint === "string" ? base.checkpoint : "",
    diffusionModel: typeof source.diffusionModel === "string" ? source.diffusionModel : typeof base.diffusionModel === "string" ? base.diffusionModel : "",
    textEncoder: typeof source.textEncoder === "string" ? source.textEncoder : typeof base.textEncoder === "string" ? base.textEncoder : "",
    vae: typeof source.vae === "string" ? source.vae : typeof base.vae === "string" ? base.vae : "",
    sampler: SAMPLERS.includes(source.sampler) ? source.sampler : SAMPLERS.includes(base.sampler) ? base.sampler : model === "Anima" ? "euler" : "dpmpp_2m",
    scheduler: SCHEDULERS.includes(source.scheduler) ? source.scheduler : SCHEDULERS.includes(base.scheduler) ? base.scheduler : model === "Anima" ? "simple" : "karras",
    guidance: GUIDANCE.some(([id]) => id === source.guidance)
      ? source.guidance
      : GUIDANCE.some(([id]) => id === base.guidance) ? base.guidance : "none",
    pag: {
      scale: Math.max(0, Math.min(5, Number.isFinite(Number(source.pag?.scale)) ? Number(source.pag.scale) : Number.isFinite(Number(base.pag?.scale)) ? Number(base.pag.scale) : DEFAULT_SETTINGS.pag.scale)),
      appliedLayers: ["mid", "all"].includes(source.pag?.appliedLayers)
        ? source.pag.appliedLayers
        : ["mid", "all"].includes(base.pag?.appliedLayers) ? base.pag.appliedLayers : DEFAULT_SETTINGS.pag.appliedLayers,
    },
    size: { ...DEFAULT_SETTINGS.size, ...base.size, ...source.size },
    hires: normalizeGalleryHires(model, source.hires, base.hires, { ...DEFAULT_SETTINGS.hires, samplers: SAMPLERS, schedulers: SCHEDULERS }, { sourceKind: hiresSourceKind }),
    adetailer: normalizeADetailerStage({ ...base.adetailer, ...source.adetailer }),
    rtx: { ...DEFAULT_SETTINGS.rtx, ...base.rtx, ...source.rtx },
    postprocessOrder: Array.isArray(source.postprocessOrder) ? [...source.postprocessOrder] : Array.isArray(base.postprocessOrder) ? [...base.postprocessOrder] : [...DEFAULT_SETTINGS.postprocessOrder],
    loras: (Array.isArray(source.loras) ? clone(source.loras) : Array.isArray(base.loras) ? clone(base.loras) : []).slice(0, 16),
    loraGroups: normalizeCardLoraGroups(Array.isArray(source.loraGroups) ? source.loraGroups : base.loraGroups),
    loraGroupPrompt: typeof source.loraGroupPrompt === "string"
      ? source.loraGroupPrompt.slice(0, 4000).trim()
      : typeof base.loraGroupPrompt === "string" ? base.loraGroupPrompt.slice(0, 4000).trim() : "",
    imageSeeds: Array.isArray(source.imageSeeds) ? [...source.imageSeeds] : Array.isArray(base.imageSeeds) ? [...base.imageSeeds] : [],
    imageHiresSeedModes: Array.isArray(source.imageHiresSeedModes) ? [...source.imageHiresSeedModes] : Array.isArray(base.imageHiresSeedModes) ? [...base.imageHiresSeedModes] : [],
    imageHiresSeeds: Array.isArray(source.imageHiresSeeds) ? [...source.imageHiresSeeds] : Array.isArray(base.imageHiresSeeds) ? [...base.imageHiresSeeds] : [],
  };
  delete normalized.promptPresets;
  // A Gallery card is a single reproducible model/list record, never a
  // workspace persistence transport. Retire any malformed/old injected map.
  delete normalized.mountedLorasByEngine;
  // Same reasoning: the group *library* describes the workspace, so a card can
  // never carry one back in and rewrite the user's saved combinations.
  delete normalized.loraGroupsByEngine;
  return normalized;
}

export async function galleryRequest(path = "", options = {}) {
  const response = await fetch(`/api/inference/gallery${path}`, {
    cache: "no-store",
    ...options,
    headers: options.body ? { "Content-Type": "application/json", ...options.headers } : options.headers,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(payload.detail) ? payload.detail.map((item) => item.msg || String(item)).join("；") : payload.detail;
    throw new Error(detail || payload.error || `画廊请求失败（HTTP ${response.status}）`);
  }
  return payload;
}

export function displayTitle(card, maxLength = 96) {
  const value = card?.title?.trim() || card?.settings?.positive?.trim() || "No image";
  const normalized = value.replace(/\s+/g, " ");
  return normalized.length > maxLength ? `${normalized.slice(0, maxLength).trimEnd()}…` : normalized;
}

export function useDialogLifecycle(open, onClose, focusReturnSelector = "") {
  const dialogRef = useRef(null);
  const closeRef = useRef(onClose);
  const focusReturnSelectorRef = useRef(focusReturnSelector);
  closeRef.current = onClose;
  focusReturnSelectorRef.current = focusReturnSelector;
  useEffect(() => {
    if (!open) return undefined;
    const previousOverflow = document.body.style.overflow;
    const previousFocus = document.activeElement;
    document.body.style.overflow = "hidden";
    const focusableSelector = 'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';
    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        closeRef.current();
        return;
      }
      if (event.key !== "Tab") return;
      const focusable = [...(dialogRef.current?.querySelectorAll(focusableSelector) || [])]
        .filter((element) => !element.hidden && element.getClientRects().length > 0);
      if (!focusable.length) {
        event.preventDefault();
        dialogRef.current?.focus();
        return;
      }
      const first = focusable[0];
      const last = focusable.at(-1);
      if (event.shiftKey && (document.activeElement === first || !dialogRef.current?.contains(document.activeElement))) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && (document.activeElement === last || !dialogRef.current?.contains(document.activeElement))) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    const focusFrame = window.requestAnimationFrame(() => {
      const initialFocus = dialogRef.current?.querySelector("[autofocus], [data-dialog-autofocus]") || dialogRef.current?.querySelector(focusableSelector);
      (initialFocus || dialogRef.current)?.focus();
    });
    return () => {
      window.cancelAnimationFrame(focusFrame);
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
      if (previousFocus instanceof HTMLElement && previousFocus.isConnected) previousFocus.focus();
      else if (focusReturnSelectorRef.current) document.querySelector(focusReturnSelectorRef.current)?.focus();
    };
  }, [open]);
  return dialogRef;
}
