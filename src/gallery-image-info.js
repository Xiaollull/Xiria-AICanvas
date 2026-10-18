// A picture added to a gallery card, read back into the card's own settings.
//
// The card editor reads each new picture's generation record through the route the image reader
// uses (`/api/image-info/read`), so the same producers are understood — this program's own PNGs,
// SD-WebUI/Forge infotext, ComfyUI graphs — and every model name arrives with the same verdict.
// This file turns such a record into card settings with the reader's own apply rules: only a model
// that resolves to exactly one installed file is written, a LoRA already loaded by a `<lora:…>` tag
// in the prompt is not mounted a second time, only the post-processing stages that can run here are
// written (and only as a chain that passes the run checks), and everything left out is itemised so
// the editor can say so.
//
// Pure on purpose: no fetch, no React. The editor owns reading and writing.

import { buildImageInfoApplyPlan, imageInfoApplyAllFields, imageInfoSampling } from "./image-info-apply.js";
import { buildPostprocessChainPlan } from "./image-info-postprocess.js";
import { normalizeResampleChain } from "./resample-chain.js";
import { effectiveCanvasMaxEdge } from "./canvas-limit.js";
import { engineGuidance, isSplitEngine } from "./engine-settings.js";
import { SAMPLER_NAMES, SCHEDULER_NAMES } from "./sampling-options.js";
import { isMossEngine, mossSamplerFor, mossSchedulerFor, normalizeMossSampling } from "./moss-engine.js";

/** At most this many pictures of one addition are read; the rest keep the settings they get. */
export const GALLERY_IMAGE_INFO_READ_LIMIT = 12;

const GROUP_LABELS = {
  prompts: "提示词",
  sampling: "采样参数",
  canvas: "画布",
  model: "底模",
  loras: "LoRA",
  chain: "后处理链",
};

function textOf(value) {
  return typeof value === "string" ? value : value === null || value === undefined ? "" : String(value);
}

function resolved(match) {
  return Boolean(match) && (match.status === "local" || match.status === "shared") ? textOf(match.match?.value) : "";
}

/** Whether a read record carries anything a card can be filled from. */
export function imageInfoHasGenerationData(info) {
  return Boolean(info) && info.status === "ok" && imageInfoApplyAllFields(info).length > 0;
}

/** The seed a picture records for itself, as the decimal string cards store, or "". */
export function imageInfoRecordedSeed(info) {
  if (!info || info.status !== "ok") return "";
  const seed = imageInfoSampling(info).values.seed;
  return typeof seed === "string" && /^\d+$/.test(seed) ? seed : "";
}

// The first folder under the LoRA root is what the manager calls the category; a root file has none.
function loraCategory(value) {
  const parts = textOf(value).replace(/^shared:[^/]+\//, "").split("/");
  return parts.length > 1 ? parts[0] : "根目录";
}

function cardLora(item) {
  const weight = Number.isFinite(Number(item.weight)) ? Number(item.weight) : 1;
  return {
    value: item.value,
    name: item.name,
    category: loraCategory(item.value),
    weight,
    // Tenths unless the recorded weight needs hundredths, so the stepper starts where the value is.
    precision: Number.isInteger(Math.round(weight * 100) / 10) ? 1 : 2,
    enabled: true,
  };
}

/**
 * The card settings a picture's record describes, merged over `current`, or `null` when the record
 * holds nothing to fill from.
 *
 * Everything the record states replaces the card's value, including what it states by omission: a
 * picture with no LoRA and no post-processing leaves the card with none, because the card describes
 * that picture and not whatever the workspace held when the editor opened. What the record does not
 * cover — batch counts, denoise, guidance — stays as it was.
 */
export function galleryCardFromImageInfo(current, info, {
  canvasMaxEdge,
  upscalers = [],
  upscalerRuntimeReady = true,
  detectors = [],
  detectorRuntimeReady = true,
} = {}) {
  if (!imageInfoHasGenerationData(info)) return null;
  const fieldIds = imageInfoApplyAllFields(info);
  // The canvas bound depends on the engine the picture lands in, which the plan decides, so a
  // picture from another engine is planned once more against that engine's own bound.
  let plan = buildImageInfoApplyPlan(info, fieldIds, "t2i", { engine: current.model, maximumEdge: effectiveCanvasMaxEdge(canvasMaxEdge, current.model) });
  if (plan.targetEngine && plan.targetEngine !== current.model) {
    plan = buildImageInfoApplyPlan(info, fieldIds, "t2i", { engine: current.model, maximumEdge: effectiveCanvasMaxEdge(canvasMaxEdge, plan.targetEngine) });
  }
  const engine = plan.targetEngine || current.model;
  const overlay = plan.overlay;
  const next = { ...current };
  if (engine !== current.model) {
    // The same reset the editor's own engine picker performs: another engine's files and LoRAs
    // cannot be carried across, and guidance follows what the new engine accepts.
    Object.assign(next, {
      model: engine,
      checkpoint: "",
      diffusionModel: "",
      textEncoder: "",
      textEncoder2: "",
      vae: "",
      loras: [],
      sampler: isMossEngine(engine) ? mossSamplerFor(current.sampler) : SAMPLER_NAMES.includes(current.sampler) ? current.sampler : isSplitEngine(engine) ? "euler" : "dpmpp_2m",
      scheduler: isMossEngine(engine) ? mossSchedulerFor(current.scheduler) : SCHEDULER_NAMES.includes(current.scheduler) ? current.scheduler : isSplitEngine(engine) ? "simple" : "karras",
      guidance: engineGuidance(engine, current.guidance),
    });
    if (isMossEngine(engine)) next.mossSampling = normalizeMossSampling(current.mossSampling);
    else delete next.mossSampling;
  }
  const filled = [];
  if (plan.groups.includes("prompts")) {
    next.positive = textOf(overlay.positive);
    next.negative = textOf(overlay.negative);
    filled.push(GROUP_LABELS.prompts);
  }
  if (plan.groups.includes("sampling")) {
    for (const key of ["steps", "cfg", "sampler", "scheduler", "seed", "seedMode", "mossSampling"]) if (key in overlay) next[key] = overlay[key];
    // A picture that records no CLIP skip ran the family default, so the card says nothing either.
    next.clipSkip = "clipSkip" in overlay ? overlay.clipSkip : null;
    filled.push(GROUP_LABELS.sampling);
  }
  if (plan.groups.includes("canvas")) {
    next.size = { ...overlay.size };
    filled.push(GROUP_LABELS.canvas);
  }
  if (plan.groups.includes("model")) {
    if (overlay.checkpoint) next.checkpoint = overlay.checkpoint;
    if (overlay.diffusionModel) next.diffusionModel = overlay.diffusionModel;
    // A native Anima picture also names its text encoder and VAE; each is written only when it
    // resolves to one installed file, like the diffusion model itself.
    if (isSplitEngine(engine) && info.animaAssets) {
      const encoder = resolved(info.animaAssets.text_encoder?.match);
      const vae = resolved(info.animaAssets.vae?.match);
      if (encoder) next.textEncoder = encoder;
      if (vae) next.vae = vae;
    }
    filled.push(GROUP_LABELS.model);
  }
  // Only LoRAs of the card's engine can be mounted on it; the plan already itemised the rest.
  next.loras = (overlay.loras || []).filter((item) => !item.engine || item.engine === engine).map(cardLora);
  if (next.loras.length) filled.push(`${GROUP_LABELS.loras} ${next.loras.length}`);

  const unapplied = [];
  const notes = [...plan.notes];
  const emptyChain = () => normalizeResampleChain([], { engine, samplers: SAMPLER_NAMES, schedulers: SCHEDULER_NAMES });
  if (plan.postprocess) {
    const chainPlan = buildPostprocessChainPlan(plan.postprocess, {
      engine,
      upscalers,
      upscalerRuntimeReady,
      detectors,
      detectorRuntimeReady,
      baseSteps: next.steps,
      size: next.size,
      order: "hires-first",
    });
    unapplied.push(...chainPlan.unapplied);
    notes.push(...chainPlan.notes);
    if (chainPlan.ready) {
      next.postprocessChain = chainPlan.chain;
      filled.push(GROUP_LABELS.chain);
    } else {
      next.postprocessChain = emptyChain();
    }
  } else {
    next.postprocessChain = emptyChain();
  }

  return {
    settings: next,
    engine,
    engineChanged: engine !== current.model,
    filled,
    missing: plan.missing,
    skipped: plan.skipped,
    unapplied,
    notes,
    promptLoras: plan.promptLoras || [],
    seed: imageInfoRecordedSeed(info),
    source: textOf(info.sourceLabel || info.source),
  };
}
