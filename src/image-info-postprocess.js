// A picture's recorded post-processing, turned into a chain this workspace can run — or an honest
// list of why it cannot.
//
// The reader's `flags` only say that a Hires pass or an ADetailer pass happened. Applying them needs
// every number the producer wrote, mapped field by field onto the chain the generate pages already
// run, and anything that has no equivalent here named rather than dropped. Three producers:
//
//   * this project's PNGs carry the request chain itself (`postprocess_chain`), stage for stage and
//     in execution order; older ones carry a `hires` / `adetailer` result block, migrated explicitly;
//   * A1111/Forge infotext writes flat `Hires …` and `ADetailer … 2nd` settings. It never says which
//     of the two ran first, so the plan uses Hires then ADetailer — what A1111 itself does — and
//     says that the order was assumed, so the user can reorder it;
//   * a ComfyUI graph wires upscale and detailer nodes in whatever way the workflow chose. Nothing
//     here is inferred from node names, so a graph contributes no stages, only an explanation.
//
// Pure: no React, no fetch. Which upscalers and detectors are installed, and what the target engine
// can run, are passed in by the caller that knows.

import { ADETAILER_UNIT_DEFAULTS, ADETAILER_UNIT_LIMIT, NATIVE_STEP_ENGINES, adetailerStageIssue } from "./adetailer-units.js";
import { LATENT_ALGORITHM_IDS, RESAMPLE_UNIT_DEFAULTS, normalizeResampleChain, resampleChainIssue } from "./resample-chain.js";
import { SAMPLER_NAMES, SCHEDULER_NAMES } from "./sampling-options.js";

function textOf(value) {
  return typeof value === "string" ? value : value === null || value === undefined ? "" : String(value);
}

function isPlainObject(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function numberOf(value) {
  const text = textOf(value).trim();
  if (!text) return null;
  const parsed = Number(text);
  return Number.isFinite(parsed) ? parsed : null;
}

function truthy(value) {
  return /^(true|1|yes)$/i.test(textOf(value).trim());
}

function parameterMap(info) {
  const map = new Map();
  for (const item of Array.isArray(info?.parameters) ? info.parameters : []) {
    const label = textOf(item?.label).trim();
    if (label) map.set(label.toLowerCase(), textOf(item?.value).trim());
  }
  return map;
}

// A1111 numbers a repeated ADetailer unit with an ordinal suffix: `ADetailer model`, `ADetailer model 2nd`.
const ADETAILER_KEY = /^adetailer (.+?)(?: (\d+)(?:st|nd|rd|th))?$/;

function adetailerGroups(parameters) {
  const units = new Map();
  for (const [label, value] of parameters) {
    const match = ADETAILER_KEY.exec(label);
    if (!match || match[1] === "version") continue;
    const index = Number(match[2] || 1);
    if (!units.has(index)) units.set(index, new Map());
    units.get(index).set(match[1], value);
  }
  return [...units.entries()].sort((first, second) => first[0] - second[0]);
}

// A1111's built-in upscaler names, as the files they are normally installed as.
const UPSCALER_ALIASES = new Map([
  ["resrgan4x", "realesrganx4plus"],
  ["resrgan4xanime6b", "realesrganx4plusanime6b"],
  ["resrgangeneral4xv3", "realesrgeneralx4v3"],
  ["resrgangeneralwdn4xv3", "realesrgeneralwdnx4v3"],
  ["resrgananimevideo", "realesranimevideov3"],
  ["esrgan4x", "esrgan4x"],
  ["swinir4x", "swinir4x"],
  ["ldsr", "ldsr"],
  ["scunet", "scunetgan"],
  ["scunetpsnr", "scunetpsnr"],
]);

function upscalerKey(value) {
  return textOf(value).toLowerCase().replace(/\.(pth|pt|safetensors|ckpt|bin)$/, "").replace(/[^a-z0-9]+/g, "");
}

const LATENT_UPSCALERS = new Map([
  ["latent", "latent"],
  ["latent (antialiased)", "latent-antialias"],
  ["latent (bicubic)", "latent-bicubic"],
  ["latent (bicubic antialiased)", "latent-bicubic-antialias"],
  ["latent (nearest)", "latent-nearest"],
  ["latent (nearest-exact)", "latent-nearest-exact"],
]);

/**
 * What a recorded upscaler name means here: a latent mode, a pixel resize, an installed model, or
 * nothing (`missing`). Model names are compared by file stem with punctuation ignored, and A1111's
 * built-in labels are translated to the files they ship as.
 */
export function resolveUpscalerName(name, catalog = []) {
  const raw = textOf(name).trim();
  const lowered = raw.toLowerCase();
  if (!raw) return { kind: "none" };
  if (LATENT_UPSCALERS.has(lowered)) return { kind: "latent", latentAlgorithm: LATENT_UPSCALERS.get(lowered) };
  if (lowered === "lanczos") return { kind: "resize", resizeAlgorithm: "lanczos" };
  if (lowered === "nearest") return { kind: "resize", resizeAlgorithm: "nearest" };
  if (lowered === "none") return { kind: "resize", resizeAlgorithm: "lanczos", note: "A1111 的 None 放大器是直接缩放，按 Lanczos 还原" };
  const wanted = upscalerKey(raw);
  const alias = UPSCALER_ALIASES.get(wanted);
  const models = (Array.isArray(catalog) ? catalog : []).filter((model) => model && model.compatible !== false);
  const found = models.filter((model) => {
    const keys = [model.id, model.local_id, model.name, model.label].map(upscalerKey).filter(Boolean);
    return keys.includes(wanted) || (alias && keys.includes(alias));
  });
  if (found.length === 1) return { kind: "model", upscaler: found[0].id, label: found[0].label || found[0].name || found[0].id };
  if (found.length > 1) return { kind: "ambiguous", candidates: found.map((model) => model.id) };
  return { kind: "missing" };
}

/** An installed detector a recorded ADetailer model name refers to, by exact file name. */
export function resolveDetectorName(name, detectors = []) {
  const wanted = textOf(name).trim().split(/[\\/]/).pop().toLowerCase();
  if (!wanted || wanted === "none") return "";
  const found = (Array.isArray(detectors) ? detectors : []).filter((model) => textOf(model?.name || model?.value).split(/[\\/]/).pop().toLowerCase() === wanted);
  return found.length === 1 ? found[0].value : "";
}

function knownName(value, catalog) {
  const wanted = textOf(value).trim().toLowerCase();
  return catalog.find((name) => name.toLowerCase() === wanted) || "";
}

function a1111Hires(parameters, baseSize) {
  const present = ["hires upscale", "hires upscaler", "hires steps", "hires resize-1", "hires resize-2"].some((key) => parameters.has(key));
  if (!present) return null;
  const issues = [];
  const notes = [];
  let scale = numberOf(parameters.get("hires upscale"));
  const resizeWidth = numberOf(parameters.get("hires resize-1"));
  const resizeHeight = numberOf(parameters.get("hires resize-2"));
  if (resizeWidth && resizeHeight) {
    if (baseSize?.width && baseSize?.height) {
      const horizontal = resizeWidth / baseSize.width;
      const vertical = resizeHeight / baseSize.height;
      if (Math.abs(horizontal - vertical) > 0.01) {
        issues.push(`Hires 指定了非等比目标尺寸 ${resizeWidth}×${resizeHeight}，无法用倍率还原`);
      } else scale = Math.round(horizontal * 100) / 100;
    } else issues.push("Hires 指定了目标尺寸，但图片未记录基础尺寸，无法换算倍率");
  }
  if (scale === null) issues.push("Hires 未记录放大倍率");
  else if (scale < 1 || scale > 4) issues.push(`Hires 倍率 ${scale} 超出本工作区 1～4 的范围`);
  const denoise = numberOf(parameters.get("denoising strength"));
  if (denoise === null) issues.push("Hires 未记录重绘强度（Denoising strength）");
  const steps = numberOf(parameters.get("hires steps"));
  const cfg = numberOf(parameters.get("hires cfg scale"));
  const upscalerName = parameters.get("hires upscaler") || "";
  const sampler = parameters.get("hires sampler");
  const scheduler = parameters.get("hires schedule type");
  const unit = {
    enabled: true,
    upscalerName,
    scale: scale ?? RESAMPLE_UNIT_DEFAULTS.scale,
    denoise: denoise ?? RESAMPLE_UNIT_DEFAULTS.denoise,
    // A1111 writes 0 (or nothing) for "the same as the first pass", which is what null means here.
    steps: steps && steps > 0 ? Math.round(steps) : null,
    cfg,
    sampler: null,
    scheduler: null,
    seedMode: "inherit",
  };
  if (sampler && !/^use same/i.test(sampler)) {
    const known = knownName(sampler, SAMPLER_NAMES);
    if (known) unit.sampler = known;
    else notes.push(`Hires 采样器 ${sampler} 在本工作区没有同名选项，沿用主采样器`);
  }
  if (scheduler && !/^use same/i.test(scheduler)) {
    const known = knownName(scheduler, SCHEDULER_NAMES);
    if (known) unit.scheduler = known;
    else notes.push(`Hires 调度器 ${scheduler} 在本工作区没有同名选项，沿用主调度器`);
  }
  if (parameters.get("hires prompt") || parameters.get("hires negative prompt")) notes.push("Hires 独立提示词不支持，重绘沿用主提示词");
  if (parameters.get("hires checkpoint") && !/^use same/i.test(parameters.get("hires checkpoint"))) issues.push(`Hires 独立底模 ${parameters.get("hires checkpoint")} 不支持`);
  for (const [label, value] of parameters) {
    if (/^hires module \d+$/.test(label) && value && !/^use same/i.test(value)) notes.push(`Hires 模块设置 ${value} 不支持，已忽略`);
  }
  return { unit, issues, notes };
}

function a1111Adetailer(parameters, hasMainControlNet) {
  const groups = adetailerGroups(parameters);
  if (!groups.length) return null;
  const issues = [];
  const notes = [];
  const units = [];
  for (const [index, fields] of groups) {
    const label = `ADetailer ${index}`;
    const model = fields.get("model") || "";
    if (!model || /^none$/i.test(model)) continue;
    if (units.length >= ADETAILER_UNIT_LIMIT) {
      issues.push(`${label}：超出 ${ADETAILER_UNIT_LIMIT} 个单元的上限，未应用`);
      continue;
    }
    const value = (key) => numberOf(fields.get(key));
    const unit = {
      enabled: true,
      detectorName: model,
      confidence: value("confidence") ?? ADETAILER_UNIT_DEFAULTS.confidence,
      maxDetections: ADETAILER_UNIT_DEFAULTS.maxDetections,
      maskMinRatio: value("mask min ratio") ?? ADETAILER_UNIT_DEFAULTS.maskMinRatio,
      maskMaxRatio: value("mask max ratio") ?? ADETAILER_UNIT_DEFAULTS.maskMaxRatio,
      dilateErode: value("dilate erode") ?? ADETAILER_UNIT_DEFAULTS.dilateErode,
      maskBlur: value("mask blur") ?? ADETAILER_UNIT_DEFAULTS.maskBlur,
      padding: value("inpaint padding") ?? ADETAILER_UNIT_DEFAULTS.padding,
      denoise: value("denoising strength") ?? ADETAILER_UNIT_DEFAULTS.denoise,
      useSteps: truthy(fields.get("use separate steps")),
      steps: value("steps") ?? ADETAILER_UNIT_DEFAULTS.steps,
      useCfg: truthy(fields.get("use separate cfg scale")),
      cfg: value("cfg scale") ?? ADETAILER_UNIT_DEFAULTS.cfg,
      prompt: fields.get("prompt") || "",
      negativePrompt: fields.get("negative prompt") || "",
    };
    const topK = value("mask only top k largest");
    if (topK !== null) {
      if (topK <= 0) notes.push(`${label}：记录为不限检测数量，按本工作区上限 8 处理`);
      unit.maxDetections = topK <= 0 ? 8 : Math.min(8, Math.round(topK));
    }
    if (fields.has("denoising strength") === false) notes.push(`${label}：未记录重绘强度，使用默认 ${ADETAILER_UNIT_DEFAULTS.denoise}`);
    if (fields.has("inpaint only masked") && !truthy(fields.get("inpaint only masked"))) issues.push(`${label}：整图重绘（inpaint only masked = False）不支持，本工作区只重绘遮罩区域`);
    const controlNet = fields.get("controlnet model");
    if (controlNet && !/^none$/i.test(controlNet) && !(/^passthrough$/i.test(controlNet) && !hasMainControlNet)) {
      issues.push(`${label}：ControlNet ${controlNet} 不支持，ControlNet 部分未复现`);
    }
    for (const [key, text] of [["use separate checkpoint", "独立底模"], ["use separate vae", "独立 VAE"], ["use separate sampler", "独立采样器"], ["use separate noise multiplier", "独立噪声倍率"], ["use separate clip skip", "独立 CLIP skip"], ["use inpaint width height", "独立重绘尺寸"]]) {
      if (truthy(fields.get(key))) notes.push(`${label}：${text}不支持，已沿用主设置`);
    }
    for (const [key, text] of [["x offset", "水平偏移"], ["y offset", "垂直偏移"]]) {
      if (value(key)) notes.push(`${label}：遮罩${text} ${fields.get(key)} 不支持，已忽略`);
    }
    if (fields.get("mask merge invert") && !/^none$/i.test(fields.get("mask merge invert"))) notes.push(`${label}：遮罩合并/反转 ${fields.get("mask merge invert")} 不支持，已忽略`);
    units.push(unit);
  }
  if (!units.length) return null;
  return { units, issues, notes };
}

function backendChainToFrontend(chain) {
  // The PNG holds the request's own chain in wire shape; this is `resampleChainPayload` read back.
  return chain.filter(isPlainObject).map((stage) => {
    if (stage.type === "resample") {
      return {
        type: "resample",
        enabled: stage.enabled === true,
        name: textOf(stage.name),
        upscaler: textOf(stage.upscaler),
        latentAlgorithm: LATENT_ALGORITHM_IDS.includes(stage.latent_algorithm) ? stage.latent_algorithm : null,
        scale: stage.scale,
        resizeAlgorithm: stage.resize_algorithm,
        denoise: stage.denoise,
        steps: stage.steps ?? null,
        cfg: stage.cfg ?? null,
        sampler: stage.sampler ?? null,
        scheduler: stage.scheduler ?? null,
        seedMode: stage.seed_mode,
        seed: stage.seed ?? "",
        tileSize: stage.tile_size,
        tileOverlap: stage.tile_overlap,
        executionMode: stage.execution_mode,
        padding: stage.padding,
        maskBlur: stage.mask_blur,
        uniformTiles: stage.uniform_tiles,
        tiledDecode: stage.tiled_decode,
        exactOutput: stage.exact_output,
      };
    }
    if (stage.type === "adetailer") {
      return {
        type: "adetailer",
        enabled: stage.enabled === true,
        units: (Array.isArray(stage.units) ? stage.units : []).filter(isPlainObject).map((unit) => ({
          enabled: true,
          detector: textOf(unit.detector),
          confidence: unit.confidence,
          maxDetections: unit.max_detections,
          maskMinRatio: unit.mask_min_ratio,
          maskMaxRatio: unit.mask_max_ratio,
          dilateErode: unit.dilate_erode,
          maskBlur: unit.mask_blur,
          padding: unit.padding,
          denoise: unit.denoise,
          useSteps: unit.use_steps === true,
          steps: unit.steps,
          useCfg: unit.use_cfg === true,
          cfg: unit.cfg,
          prompt: textOf(unit.prompt),
          negativePrompt: textOf(unit.negative_prompt),
        })),
      };
    }
    if (stage.type === "rtx") return { type: "rtx", enabled: stage.enabled === true, scale: stage.scale, quality: stage.quality };
    return null;
  }).filter(Boolean);
}

/**
 * The post-processing a picture recorded, as a candidate plan that has not yet met this machine.
 *
 * `stages` are frontend-shaped chain entries in their recorded (or assumed) order; a stage whose
 * model has to be found here carries the recorded name (`upscalerName`, `detectorName`) instead of
 * an ID. `issues` block the stage they name; `notes` explain a value that was translated or dropped
 * without changing what runs.
 */
export function imageInfoPostprocessCandidate(info) {
  if (!info || info.status !== "ok") return null;
  const flags = Array.isArray(info.flags) ? info.flags : [];
  const flagged = (id) => flags.some((flag) => flag.id === id && flag.enabled);
  if (info.source === "xiria") {
    const record = isPlainObject(info.postprocess) ? info.postprocess : {};
    if (Array.isArray(record.chain) && record.chain.some((stage) => stage?.enabled)) {
      return { source: "xiria", order: "recorded", stages: backendChainToFrontend(record.chain).filter((stage) => stage.enabled), issues: [], notes: [], recordedChain: true };
    }
    if (isPlainObject(record.hires) || isPlainObject(record.adetailer)) {
      // A picture from before the chain: its result blocks name the model and the headline numbers
      // and nothing else, so exactly those are migrated and the rest stay at their defaults.
      const stages = [];
      const notes = ["该图片早于后处理链格式：只还原旧记录中写明的模型与主要数值"];
      const order = Array.isArray(record.stages) ? record.stages : ["hires", "adetailer"];
      for (const type of order) {
        if (type === "hires" && isPlainObject(record.hires)) {
          const hires = record.hires;
          stages.push({ type: "resample", enabled: true, upscaler: textOf(hires.upscaler || hires.model), scale: hires.scale, denoise: hires.denoise, steps: hires.steps ?? null, cfg: hires.cfg ?? null });
        }
        if (type === "adetailer" && isPlainObject(record.adetailer) && textOf(record.adetailer.detector)) {
          const detail = record.adetailer;
          stages.push({ type: "adetailer", enabled: true, units: [{ enabled: true, detector: textOf(detail.detector), confidence: detail.confidence, denoise: detail.denoise, padding: detail.padding, maskBlur: detail.mask_blur }] });
        }
      }
      return { source: "xiria", order: Array.isArray(record.stages) ? "recorded" : "assumed", stages, issues: stages.length ? [] : ["旧记录没有可还原的后处理设置"], notes };
    }
    return null;
  }
  if (info.source === "comfyui") {
    if (!flagged("hires") && !flagged("adetailer")) return null;
    return {
      source: "comfyui",
      order: "unknown",
      stages: [],
      issues: ["ComfyUI 工作流中的放大与细节修复节点无法沿连线可靠换算为本工作区的后处理链，未应用"],
      notes: [],
    };
  }
  const parameters = parameterMap(info);
  const size = /^(\d+)\s*[x×]\s*(\d+)$/i.exec(parameters.get("size") || "");
  const baseSize = size ? { width: Number(size[1]), height: Number(size[2]) } : null;
  const hasMainControlNet = [...parameters.keys()].some((key) => key.startsWith("controlnet"));
  const hires = a1111Hires(parameters, baseSize);
  const adetailer = a1111Adetailer(parameters, hasMainControlNet);
  const issues = [];
  const notes = [];
  if (hasMainControlNet) issues.push("主流程 ControlNet 不支持，未复现");
  if (flagged("refiner")) issues.push("Refiner 不支持，未复现");
  if (!hires && !adetailer) return issues.length ? { source: info.source, order: "unknown", stages: [], issues, notes } : null;
  const stages = [];
  if (hires) {
    stages.push({ type: "resample", ...hires.unit, blocked: hires.issues.length > 0 });
    issues.push(...hires.issues);
    notes.push(...hires.notes);
  }
  if (adetailer) {
    stages.push({ type: "adetailer", enabled: true, units: adetailer.units });
    issues.push(...adetailer.issues);
    notes.push(...adetailer.notes);
  }
  return { source: info.source, order: hires && adetailer ? "assumed" : "recorded", stages, issues, notes };
}

/**
 * The chain one candidate becomes on this machine, validated as a whole.
 *
 * `ready` is true only when every stage that survived mapping can run: the upscaler and detectors
 * are installed, the engine can run each stage, and the chain passes the same checks a run does.
 * `applied` lists what will run; `unapplied` lists what will not and why. Nothing is enabled empty,
 * no model is invented, and nothing is reduced to make it fit.
 */
export function buildPostprocessChainPlan(candidate, {
  engine,
  upscalers = [],
  upscalerRuntimeReady = true,
  detectors = [],
  detectorRuntimeReady = true,
  baseSteps = 28,
  size = null,
  order = "hires-first",
  features = null,
} = {}) {
  const unapplied = [...(candidate?.issues || [])];
  const notes = [...(candidate?.notes || [])];
  const applied = [];
  if (!candidate || !candidate.stages?.length) {
    return { ready: false, chain: null, applied, unapplied: unapplied.length ? unapplied : ["图片没有可应用的后处理设置"], notes, order: candidate?.order || "unknown" };
  }
  let stages = candidate.stages.map((stage) => ({ ...stage }));
  if (candidate.order === "assumed" && order === "ade-first") {
    stages = [...stages.filter((stage) => stage.type === "adetailer"), ...stages.filter((stage) => stage.type !== "adetailer")];
  }
  const kept = [];
  for (const stage of stages) {
    if (stage.type === "resample") {
      if (stage.blocked) {
        unapplied.push("Hires 未应用：记录的数据不足以还原");
        continue;
      }
      if (features && features.hires !== true) {
        unapplied.push(`Hires 未应用：${engine} 当前运行时未声明重采样能力`);
        continue;
      }
      const unit = { ...stage };
      delete unit.blocked;
      if (unit.upscalerName !== undefined) {
        const resolved = resolveUpscalerName(unit.upscalerName, upscalerRuntimeReady ? upscalers : []);
        delete unit.upscalerName;
        if (resolved.kind === "missing" || resolved.kind === "ambiguous") {
          unapplied.push(resolved.kind === "missing"
            ? `Hires 未应用：放大器 ${stage.upscalerName} 未安装或不可用`
            : `Hires 未应用：放大器 ${stage.upscalerName} 对应多个本地模型`);
          continue;
        }
        if (resolved.kind === "latent") {
          if (engine !== "SD" && engine !== "iL") {
            unapplied.push(`Hires 未应用：潜空间放大（${stage.upscalerName}）仅支持 SD/iL`);
            continue;
          }
          Object.assign(unit, { upscaler: "", latentAlgorithm: resolved.latentAlgorithm });
        } else if (resolved.kind === "resize") {
          Object.assign(unit, { upscaler: "", latentAlgorithm: null, resizeAlgorithm: resolved.resizeAlgorithm });
          if (resolved.note) notes.push(resolved.note);
        } else if (resolved.kind === "model") {
          Object.assign(unit, { upscaler: resolved.upscaler, latentAlgorithm: null });
        } else {
          Object.assign(unit, { upscaler: "", latentAlgorithm: null, resizeAlgorithm: "lanczos" });
        }
      }
      kept.push(unit);
      applied.push(`Hires ${Number(unit.scale).toFixed(2).replace(/\.?0+$/, "")}×${unit.upscaler ? ` · ${unit.upscaler}` : unit.latentAlgorithm ? ` · ${unit.latentAlgorithm}` : ""} · 重绘 ${unit.denoise}`);
    } else if (stage.type === "adetailer") {
      if (features && features.adetailer !== true) {
        unapplied.push(`ADetailer 未应用：${engine} 当前运行时未声明 ADetailer 能力`);
        continue;
      }
      if (!detectorRuntimeReady) {
        unapplied.push("ADetailer 未应用：检测运行时尚未就绪");
        continue;
      }
      const units = [];
      for (const [index, unit] of (stage.units || []).entries()) {
        const copy = { ...unit };
        if (copy.detectorName !== undefined) {
          copy.detector = resolveDetectorName(copy.detectorName, detectors);
          if (!copy.detector) {
            unapplied.push(`ADetailer ${index + 1} 未应用：检测模型 ${unit.detectorName} 未安装`);
            continue;
          }
          delete copy.detectorName;
        } else if (copy.detector && !detectors.some((model) => model.value === copy.detector)) {
          unapplied.push(`ADetailer ${index + 1} 未应用：检测模型 ${copy.detector} 未安装`);
          continue;
        }
        units.push(copy);
      }
      if (!units.length) {
        unapplied.push("ADetailer 未应用：没有可运行的检测单元");
        continue;
      }
      kept.push({ type: "adetailer", enabled: true, units });
      applied.push(`ADetailer ${units.length} 个单元`);
    } else if (stage.type === "rtx") {
      if (features && features.rtx !== true) {
        unapplied.push(`RTX VSR 未应用：${engine} 当前运行时未声明 RTX VSR 能力`);
        continue;
      }
      kept.push(stage);
      applied.push(`RTX VSR ${stage.scale}×`);
    }
  }
  if (!kept.length) return { ready: false, chain: null, applied: [], unapplied, notes, order: candidate.order };
  const chain = normalizeResampleChain(kept.map((stage) => ({ ...stage, enabled: true })), { samplers: SAMPLER_NAMES, schedulers: SCHEDULER_NAMES, engine });
  const chainIssue = resampleChainIssue(chain, {
    baseSteps,
    engine,
    nativeEngines: NATIVE_STEP_ENGINES,
    upscalers: upscalerRuntimeReady ? upscalers : [],
    size,
  });
  if (chainIssue) return { ready: false, chain: null, applied: [], unapplied: [...unapplied, `后处理链无法运行：${chainIssue}`], notes, order: candidate.order };
  const detail = chain.find((stage) => stage.type === "adetailer");
  if (detail?.enabled) {
    const issue = adetailerStageIssue(detail, baseSteps, (value) => detectors.some((model) => model.value === value), engine);
    if (issue) return { ready: false, chain: null, applied: [], unapplied: [...unapplied, `后处理链无法运行：${issue}`], notes, order: candidate.order };
  }
  return { ready: true, chain, applied, unapplied, notes, order: candidate.order };
}
