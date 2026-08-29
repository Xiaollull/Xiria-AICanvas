// Turning a picture's metadata back into workspace settings.
//
// The reader already knows what produced an image; this decides what of that
// can be *re-used*. Two things make it more than a copy. First, a producer
// writes its own vocabulary — "DPM++ 2M Karras", "1024x1280" — and only values
// this workspace actually offers may be applied, so an unrecognised sampler is
// reported as skipped rather than written and silently ignored later. Second,
// a model name is only a name: whether the file exists here is a separate
// question, and the answer decides both what is applied and what the user is
// told afterwards.
//
// Pure on purpose: no React, no fetch. `App` owns the writing.

import { SAMPLER_NAMES, SCHEDULER_NAMES } from "./sampling-options.js";

export const IMAGE_INFO_APPLY_TARGETS = [
  { id: "t2i", label: "文生图" },
  { id: "i2i", label: "图生图" },
];

// Only the prompt is on by default: it is the one field that is always safe to
// take, always what the user came for, and never has to be reconciled against
// what is installed.
export const IMAGE_INFO_APPLY_DEFAULT_FIELDS = ["prompts"];

const MAXIMUM_EDGE = 2048;
const MINIMUM_EDGE = 64;

function textOf(value) {
  return typeof value === "string" ? value : value === null || value === undefined ? "" : String(value);
}

function parameterValue(info, label) {
  const row = (info?.parameters || []).find((item) => textOf(item?.label).toLowerCase() === label.toLowerCase());
  return row ? textOf(row.value).trim() : "";
}

// `Number("")` is 0, so a blank has to be rejected before the finite check —
// otherwise a record that never mentioned CFG would apply a CFG of zero.
function finiteNumber(value) {
  const text = textOf(value).trim();
  if (!text) return null;
  const parsed = Number(text);
  return Number.isFinite(parsed) ? parsed : null;
}

function snapEdge(value) {
  const parsed = finiteNumber(value);
  if (parsed === null) return null;
  const snapped = Math.round(parsed / 64) * 64;
  return Math.max(MINIMUM_EDGE, Math.min(MAXIMUM_EDGE, snapped || MINIMUM_EDGE));
}

// A producer's own spelling only counts when this workspace offers the same
// one. Applying "DPM++ 2M Karras" as a sampler name would store a value no
// picker can show and no backend can run.
function knownName(value, catalog) {
  const wanted = textOf(value).trim();
  if (!wanted) return "";
  const match = catalog.find((name) => name.toLowerCase() === wanted.toLowerCase());
  return match || "";
}

export function imageInfoSampling(info) {
  const sampler = knownName(parameterValue(info, "Sampler"), SAMPLER_NAMES);
  const scheduler = knownName(parameterValue(info, "Scheduler"), SCHEDULER_NAMES);
  const steps = finiteNumber(parameterValue(info, "Steps"));
  const cfg = finiteNumber(parameterValue(info, "CFG scale"));
  const seed = parameterValue(info, "Seed");
  const values = {};
  const skipped = [];
  if (steps !== null) values.steps = Math.max(1, Math.min(100, Math.round(steps)));
  if (cfg !== null) values.cfg = Math.max(0, Math.min(30, cfg));
  if (sampler) values.sampler = sampler;
  else if (parameterValue(info, "Sampler")) skipped.push(`采样器 ${parameterValue(info, "Sampler")}`);
  if (scheduler) values.scheduler = scheduler;
  else if (parameterValue(info, "Scheduler")) skipped.push(`调度器 ${parameterValue(info, "Scheduler")}`);
  if (/^\d+$/.test(seed)) {
    values.seed = seed;
    // A recorded seed is only worth taking if it is then held: leaving the mode
    // on random would roll a new one on the next run and reproduce nothing.
    values.seedMode = "fixed";
  }
  return { values, skipped };
}

export function imageInfoCanvas(info) {
  const size = parameterValue(info, "Size") || parameterValue(info, "Requested size");
  const [width, height] = size.split(/[x×]/i).map((part) => snapEdge(part.trim()));
  return width && height ? { width, height } : null;
}

function loraRows(info) {
  return (info?.loras || []).map((item) => ({
    name: textOf(item?.name),
    weight: Number.isFinite(Number(item?.weight)) ? Math.max(-5, Math.min(5, Number(item.weight))) : 1,
    match: item?.match || null,
  }));
}

// A model resolved to a file here can be applied; one that resolved to nothing
// cannot, and saying so is the whole point of the red mark afterwards.
function resolvedValue(match) {
  return match && match.status !== "missing" && match.status !== "unknown" ? textOf(match.match?.value) : "";
}

// Which engine's model root the file was found under. A checkpoint is only
// selectable while that engine is active, and a LoRA mounted into another
// engine's scope would be pruned by the next catalogue scan, so the caller has
// to be able to tell.
function resolvedEngine(match) {
  return textOf(match?.match?.engine);
}

export function imageInfoApplyFields(info) {
  if (!info || info.status !== "ok") return [];
  const fields = [];
  if (info.positive || info.negative) {
    fields.push({
      id: "prompts",
      label: "提示词",
      detail: "正向与负向提示词",
      summary: info.positive ? `${info.positive.slice(0, 40)}${info.positive.length > 40 ? "…" : ""}` : "仅负向提示词",
      shared: false,
      missing: [],
    });
  }
  const sampling = imageInfoSampling(info);
  if (Object.keys(sampling.values).length) {
    fields.push({
      id: "sampling",
      label: "采样参数",
      detail: "步数、CFG、采样器、调度器与 Seed",
      summary: Object.entries(sampling.values)
        .filter(([key]) => key !== "seedMode")
        .map(([key, value]) => `${key} ${value}`)
        .join(" · "),
      shared: false,
      missing: [],
      skipped: sampling.skipped,
    });
  }
  const canvas = imageInfoCanvas(info);
  if (canvas) {
    fields.push({ id: "canvas", label: "画布大小", detail: "宽度与高度", summary: `${canvas.width} × ${canvas.height}`, shared: false, missing: [] });
  }
  if (info.checkpoint) {
    const value = resolvedValue(info.checkpointMatch);
    fields.push({
      id: "model",
      label: "底模",
      detail: "两个生图页共用同一个模型状态",
      summary: info.checkpoint,
      shared: true,
      missing: value ? [] : [info.checkpoint],
    });
  }
  const loras = loraRows(info);
  if (loras.length) {
    fields.push({
      id: "loras",
      label: `LoRA`,
      detail: "两个生图页共用同一份挂载",
      summary: `${loras.length} 个`,
      shared: true,
      entries: loras.map((item) => ({ name: item.name, weight: item.weight, missing: !resolvedValue(item.match) })),
      missing: loras.filter((item) => !resolvedValue(item.match)).map((item) => item.name),
    });
  }
  return fields;
}

export function imageInfoApplyAllFields(info) {
  return imageInfoApplyFields(info).map((field) => field.id);
}

export function imageInfoDefaultFields(info) {
  const available = new Set(imageInfoApplyAllFields(info));
  return IMAGE_INFO_APPLY_DEFAULT_FIELDS.filter((id) => available.has(id));
}

/**
 * What `App` has to write, and what the user has to be told.
 *
 * `overlay` is a partial workspace record — the same shape a gallery card
 * carries — so the caller merges it over the current workspace and the existing
 * apply path does the rest. `imageToImage` is the same information in the
 * image-to-image page's own field names, because that page owns its prompt,
 * sampling and canvas separately while sharing the model and the LoRA mount.
 */
export function buildImageInfoApplyPlan(info, fieldIds, target = "t2i") {
  const selected = new Set(Array.isArray(fieldIds) ? fieldIds : []);
  const available = imageInfoApplyFields(info).filter((field) => selected.has(field.id));
  const toImageToImage = target === "i2i";
  const overlay = {};
  const imageToImage = {};
  const groups = [];
  const missing = [];
  const skipped = [];

  for (const field of available) {
    if (field.id === "prompts") {
      const values = { positive: textOf(info.positive), negative: textOf(info.negative) };
      if (toImageToImage) Object.assign(imageToImage, values);
      else Object.assign(overlay, values);
      groups.push("prompts");
    }
    if (field.id === "sampling") {
      const { values, skipped: unusable } = imageInfoSampling(info);
      skipped.push(...unusable);
      if (toImageToImage) Object.assign(imageToImage, values);
      else Object.assign(overlay, values);
      groups.push("sampling");
    }
    if (field.id === "canvas") {
      const canvas = imageInfoCanvas(info);
      // The image-to-image page sizes from its source picture unless it is told
      // to use an explicit canvas, so applying a size has to say so as well.
      if (toImageToImage) Object.assign(imageToImage, { sizeMode: "custom", width: canvas.width, height: canvas.height });
      else overlay.size = canvas;
      groups.push("canvas");
    }
    if (field.id === "model") {
      const value = resolvedValue(info.checkpointMatch);
      if (value) {
        overlay.checkpoint = value;
        // Naming the engine is what lets the existing apply path switch to it;
        // without it the checkpoint would be looked for in the wrong catalogue.
        const engine = resolvedEngine(info.checkpointMatch);
        if (engine) overlay.model = engine;
        groups.push("model");
      } else missing.push({ kind: "checkpoint", name: info.checkpoint });
    }
    if (field.id === "loras") {
      const rows = loraRows(info);
      const usable = rows.filter((item) => resolvedValue(item.match));
      for (const item of rows) {
        if (!resolvedValue(item.match)) missing.push({ kind: "lora", name: item.name });
      }
      // The mounted library is reconciled against the scanned catalogue, so a
      // LoRA that is not installed cannot be mounted at all — it would be
      // pruned on the next scan. Only the found ones are mounted; the rest are
      // reported, which is what the red mark is for.
      if (usable.length) {
        overlay.loras = usable.map((item) => ({
          value: resolvedValue(item.match),
          name: item.name,
          weight: item.weight,
          enabled: true,
          engine: resolvedEngine(item.match),
        }));
        groups.push("loras");
      }
    }
  }

  return {
    target: toImageToImage ? "i2i" : "t2i",
    overlay,
    imageToImage,
    groups,
    missing,
    skipped,
    applied: available.map((field) => field.id),
  };
}

export function imageInfoApplySummary(plan) {
  if (!plan) return "";
  const parts = [];
  if (plan.groups.length) parts.push(`已应用 ${plan.groups.length} 项`);
  else parts.push("没有可应用的项目");
  if (plan.missing.length) parts.push(`${plan.missing.length} 个模型本地缺失`);
  if (plan.skipped.length) parts.push(`${plan.skipped.length} 项本工作区不支持`);
  return parts.join(" · ");
}
