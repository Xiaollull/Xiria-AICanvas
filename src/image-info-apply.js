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

import { clipSkipRange, isSplitEngine } from "./engine-settings.js";
import { extractPromptLoras } from "./image-metadata.js";
import { imageInfoPostprocessCandidate } from "./image-info-postprocess.js";
import { SAMPLER_NAMES, SCHEDULER_NAMES } from "./sampling-options.js";
import { describeMossSampling, isMossEngine, mossSamplingFromPayload } from "./moss-engine.js";
import { mossLayoutFromRecord } from "./moss-layout.js";

export const IMAGE_INFO_APPLY_TARGETS = [
  { id: "t2i", label: "文生图" },
  { id: "i2i", label: "图生图" },
];

// Only the prompt is on by default: it is the one field that is always safe to
// take, always what the user came for, and never has to be reconciled against
// what is installed.
export const IMAGE_INFO_APPLY_DEFAULT_FIELDS = ["prompts"];

// The ceiling a recorded size is clamped to. It is the configurable canvas bound (2048 unless the
// user raised it), passed in by the caller; this is only the fallback.
const DEFAULT_MAXIMUM_EDGE = 2048;
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

function snapEdge(value, maximum = DEFAULT_MAXIMUM_EDGE) {
  const parsed = finiteNumber(value);
  if (parsed === null) return null;
  const snapped = Math.round(parsed / 64) * 64;
  return Math.max(MINIMUM_EDGE, Math.min(maximum, snapped || MINIMUM_EDGE));
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
  const fixSteps = parameterValue(info, "Img2img fix steps").toLowerCase();
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
  if (fixSteps === "true" || fixSteps === "false") values.fixSteps = fixSteps === "true";
  const clipSkip = finiteNumber(parameterValue(info, "Clip skip"));
  if (clipSkip !== null && Number.isInteger(clipSkip)) values.clipSkip = clipSkip;
  return { values, skipped };
}

/**
 * The CLIP skip a target engine will actually run for a recorded one, or why none applies.
 *
 * iL's towers cannot read shallower than the penultimate layer, so a recorded 1 becomes 2 there —
 * which is what Forge ran too — and the note says so rather than showing 1 and running 2.
 */
export function imageInfoClipSkip(recorded, engine) {
  const range = clipSkipRange(engine);
  if (!Number.isInteger(recorded)) return { value: null, note: "" };
  if (!range) return { value: null, note: `CLIP skip ${recorded}（${engine || "当前引擎"} 不使用 CLIP skip）` };
  const value = Math.max(range.minimum, Math.min(range.maximum, recorded));
  const note = value !== recorded ? `CLIP skip ${recorded} → ${value}（${engine} 的有效范围 ${range.minimum}～${range.maximum}）` : "";
  return { value, note };
}

export function imageInfoCanvas(info, maximumEdge = DEFAULT_MAXIMUM_EDGE) {
  // The canvas is what was sampled. This program's own record writes the finished picture's size
  // as "Size" and adds "Requested size" exactly when post-processing changed it; taking "Size"
  // there would upscale a Hires picture twice once its chain is applied too. A1111 and ComfyUI
  // records state the sampled size as "Size" and have no "Requested size".
  const size = parameterValue(info, "Requested size") || parameterValue(info, "Size");
  const [width, height] = size.split(/[x×]/i).map((part) => snapEdge(part.trim(), maximumEdge));
  return width && height ? { width, height } : null;
}

function loraRows(info) {
  return (info?.loras || []).map((item) => ({
    name: textOf(item?.name),
    ...inspectLoraIdentity(item).fields,
    weight: Number.isFinite(Number(item?.weight)) ? Math.max(-5, Math.min(5, Number(item.weight))) : 1,
    match: item?.match || null,
  }));
}

// A model resolved to exactly one file here can be applied. One that resolved to nothing, or to
// several same-named files, cannot: an ambiguous name is never mounted as its first candidate.
function resolvedValue(match) {
  return match && (match.status === "local" || match.status === "shared") ? textOf(match.match?.value) : "";
}

/** Why a recorded LoRA will not be mounted on `engine`, or "" when it will. */
export function loraUnappliedReason(match, engine) {
  if (!match || match.status === "unknown") return "未记录名称";
  if (match.status === "missing") return "本地与共享目录中都没有找到";
  if (match.status === "ambiguous") return `有歧义：${match.count || match.candidates?.length || 2} 个同名文件，无法确定是哪一个`;
  const owner = textOf(match.match?.engine);
  if (engine && owner && owner !== engine) return `属于 ${owner} 引擎，不能挂载到 ${engine}`;
  if (engine && match.status === "shared" && Array.isArray(match.match?.engines) && match.match.engines.length && !match.match.engines.includes(engine)) {
    return `该共享目录未对 ${engine} 启用`;
  }
  return "";
}

/** The label a recorded model is shown by: `Style_九V6.0\matcha.safetensors` once resolved, else the recorded name. */
export function matchLabel(match, fallback = "") {
  if (match && (match.status === "local" || match.status === "shared")) return textOf(match.display) || textOf(match.match?.display) || textOf(match.name) || fallback;
  return textOf(match?.name) || fallback;
}

// A LoRA named by a `<lora:…>` tag is compared by its file stem: the tag may spell a folder
// (`style/matcha`) where the record names the file (`matcha`).
function loraStem(name) {
  return textOf(name).split(/[\\/]/).pop().replace(/\.(?:safetensors|ckpt|pt|pth|bin)$/i, "").trim().toLowerCase();
}

/** The LoRAs a prompt loads by itself, through its own `<lora:…>` tags, as file stems. */
export function promptTagLoraStems(...prompts) {
  return new Set(prompts.flatMap((prompt) => extractPromptLoras(prompt).map((item) => loraStem(item.name))).filter(Boolean));
}

/**
 * Point every `<lora:…>` tag of one file stem at an exact relative ID, weights untouched.
 *
 * The server resolves a tag by itself and refuses a name several different files share, so a tag
 * whose file the reader chose by folder is rewritten to name that file: the choice shown is the one
 * that runs. Only a local ID can be written — a shared reference carries a `:` a tag cannot hold.
 */
export function retargetPromptLora(prompt, stem, reference) {
  const target = textOf(reference).replace(/\.(?:safetensors|ckpt|pt|pth|bin)$/i, "");
  if (!stem || !target || target.includes(":")) return textOf(prompt);
  return textOf(prompt).replace(/<(lora|lyco):([^:>]+)((?::[^>]*)?)>/gi, (whole, kind, name, rest) => (loraStem(name) === stem ? `<${kind}:${target}${rest}>` : whole));
}

// Which engine's model root the file was found under. A checkpoint is only
// selectable while that engine is active, and a LoRA mounted into another
// engine's scope would be pruned by the next catalogue scan, so the caller has
// to be able to tell.
function resolvedEngine(match) {
  return textOf(match?.match?.engine);
}

/** How a recorded post-processing chain reads before anything is resolved: a one-line summary. */
function postprocessSummary(candidate) {
  const parts = [];
  for (const stage of candidate.stages) {
    if (stage.type === "resample") parts.push(`Hires ${stage.scale}×${stage.upscalerName ? ` ${stage.upscalerName}` : stage.upscaler ? ` ${stage.upscaler.split("/").pop()}` : ""}`);
    if (stage.type === "adetailer") parts.push(`ADetailer ×${(stage.units || []).length}`);
    if (stage.type === "rtx") parts.push(`RTX ${stage.scale}×`);
  }
  const order = candidate.order === "assumed" ? "（顺序未记录，默认先 Hires 后 ADetailer）" : "";
  return parts.length ? `${parts.join(" → ")}${order}` : candidate.issues[0] || "无可应用阶段";
}

export function imageInfoApplyFields(info) {
  // An XYZ grid is a comparison sheet, never one picture's settings: nothing of it is applied, and a
  // cell is reached only by choosing one (QPlan2/P1 Phase E).
  if (!info || info.status !== "ok" || info.source === "xyz_grid") return [];
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
      summary: matchLabel(info.checkpointMatch, info.checkpoint),
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
      entries: loras.map((item) => ({
        name: item.name,
        label: matchLabel(item.match, item.name),
        weight: item.weight,
        status: item.match?.status || "unknown",
        missing: !resolvedValue(item.match),
        reason: resolvedValue(item.match) ? "" : loraUnappliedReason(item.match, ""),
      })),
      missing: loras.filter((item) => !resolvedValue(item.match)).map((item) => item.name),
    });
  }
  const postprocess = imageInfoPostprocessCandidate(info);
  if (postprocess) {
    fields.push({
      id: "chain",
      label: "后处理链",
      detail: "Hires 与 ADetailer 等阶段，按记录的数值与顺序",
      summary: postprocessSummary(postprocess),
      shared: false,
      missing: [],
      candidate: postprocess,
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
export function buildImageInfoApplyPlan(info, fieldIds, target = "t2i", { engine = "", maximumEdge = DEFAULT_MAXIMUM_EDGE } = {}) {
  const selected = new Set(Array.isArray(fieldIds) ? fieldIds : []);
  const available = imageInfoApplyFields(info).filter((field) => selected.has(field.id));
  const toImageToImage = target === "i2i";
  const overlay = {};
  const imageToImage = {};
  const groups = [];
  const missing = [];
  const skipped = [];
  const notes = [];
  const promptLoras = [];
  const retargeted = [];
  // Which engine the written values land in: the picture's own model's engine when that group is
  // applied and resolves, otherwise the engine that is selected now. Everything engine-specific —
  // which LoRAs may mount, what CLIP skip means — is decided against it.
  const modelSelected = available.some((field) => field.id === "model") && resolvedValue(info?.checkpointMatch);
  const targetEngine = (modelSelected && resolvedEngine(info.checkpointMatch)) || engine;
  let postprocess = null;

  for (const field of available) {
    if (field.id === "prompts") {
      const values = { positive: textOf(info.positive), negative: textOf(info.negative) };
      // A MOSS layout picture's prompt is compiled JSON. Its layout goes back into layout mode and the
      // text box keeps what it holds; the prefix line came from LoRA groups and presets, which are
      // not rewritten from a picture.
      const restored = info.mossLayout ? mossLayoutFromRecord(info.mossLayout) : null;
      if (restored && !toImageToImage && isMossEngine(targetEngine)) {
        overlay.negative = values.negative;
        overlay.mossLayout = restored.layout;
        if (restored.prefix.trim()) notes.push(`布局前缀「${restored.prefix.trim().slice(0, 80)}」来自当时的 LoRA 组与预设，未写入`);
      } else {
        if (restored) skipped.push("MOSS 坐标布局仅适用于 MOSS 文生图；已按编译后的文本提示词写入");
        if (toImageToImage) Object.assign(imageToImage, values);
        else Object.assign(overlay, values);
      }
      groups.push("prompts");
    }
    if (field.id === "sampling") {
      const { values, skipped: unusable } = imageInfoSampling(info);
      skipped.push(...unusable);
      const sampling = { ...values };
      if ("clipSkip" in sampling) {
        const { value, note } = imageInfoClipSkip(sampling.clipSkip, targetEngine);
        if (value === null) delete sampling.clipSkip;
        else sampling.clipSkip = value;
        if (note) (value === null ? skipped : notes).push(note);
      }
      if (toImageToImage) Object.assign(imageToImage, sampling);
      else {
        delete sampling.fixSteps;
        Object.assign(overlay, sampling);
      }
      // A MOSS picture's shift and skip layers belong to its sampling; they land only on a MOSS
      // workspace, since no other engine has the block (and image-to-image does not run MOSS).
      if (info.mossSampling) {
        if (!toImageToImage && isMossEngine(targetEngine)) overlay.mossSampling = mossSamplingFromPayload(info.mossSampling);
        else skipped.push(`MOSS 采样设置（${describeMossSampling(mossSamplingFromPayload(info.mossSampling))}）仅适用于 MOSS 文生图`);
      }
      groups.push("sampling");
    }
    if (field.id === "canvas") {
      const canvas = imageInfoCanvas(info, maximumEdge);
      const recorded = imageInfoCanvas(info, Number.MAX_SAFE_INTEGER);
      if (recorded && (recorded.width !== canvas.width || recorded.height !== canvas.height)) {
        notes.push(`画布 ${recorded.width} × ${recorded.height} 超出当前画布上限 ${maximumEdge}，已按上限写入 ${canvas.width} × ${canvas.height}`);
      }
      // The image-to-image page sizes from its source picture unless it is told
      // to use an explicit canvas, so applying a size has to say so as well.
      if (toImageToImage) Object.assign(imageToImage, { sizeMode: "custom", width: canvas.width, height: canvas.height });
      else overlay.size = canvas;
      groups.push("canvas");
    }
    if (field.id === "model") {
      const value = resolvedValue(info.checkpointMatch);
      if (value) {
        // Naming the engine is what lets the existing apply path switch to it;
        // without it the checkpoint would be looked for in the wrong catalogue.
        const modelEngine = resolvedEngine(info.checkpointMatch);
        if (modelEngine) overlay.model = modelEngine;
        // A split engine mounts a diffusion model, not a checkpoint, and its apply
        // check validates `diffusionModel`. Writing the file into `checkpoint` left
        // that check looking at whatever diffusion model the workspace already had.
        if (isSplitEngine(modelEngine)) overlay.diffusionModel = value;
        else overlay.checkpoint = value;
        groups.push("model");
      } else {
        const ambiguous = info.checkpointMatch?.status === "ambiguous";
        missing.push({
          kind: "checkpoint",
          name: info.checkpoint,
          reason: ambiguous ? `有歧义：${info.checkpointMatch.count || 2} 个同名文件` : "本地与共享目录中都没有找到",
          reasonKind: ambiguous ? "ambiguous" : "missing",
        });
      }
    }
    if (field.id === "loras") {
      const rows = loraRows(info);
      const usable = [];
      // A prompt applied with its `<lora:…>` tags already loads those LoRAs when it runs. Mounting
      // the same files as well is refused by the server (an inline tag may not repeat a manual
      // mount), so a tagged LoRA travels with the prompt and only the others are mounted.
      const tagged = selected.has("prompts") && available.some((entry) => entry.id === "prompts")
        ? promptTagLoraStems(info.positive, info.negative)
        : new Set();
      const carried = [];
      for (const item of rows) {
        if (tagged.has(loraStem(item.name)) && resolvedValue(item.match)) {
          if (item.match.relink?.originalValue && item.match.relink.originalValue !== item.match.relink.currentValue) {
            if (item.match.relink.promptRewriteAllowed) retargeted.push({ exactSource: item.match.relink.originalValue, reference: item.match.relink.currentValue });
            else { missing.push({ kind: "lora", name: item.name, reasonKind: "missing", reason: "旧精确标签尚未授权改写；请在修复表明确确认或手动修改，未自动更改 Prompt。" }); continue; }
          }
          carried.push(matchLabel(item.match, item.name));
          // A file chosen among same-named ones by the folder rule is named exactly in the tag.
          if (item.match.chosenBy) retargeted.push({ stem: loraStem(item.name), reference: resolvedValue(item.match) });
          continue;
        }
        const reason = loraUnappliedReason(item.match, targetEngine);
        const status = item.match?.status;
        const reasonKind = status === "ambiguous" ? "ambiguous" : status === "local" || status === "shared" ? "engine" : "missing";
        if (reason) missing.push({ kind: "lora", name: item.name, label: matchLabel(item.match, item.name), reason, reasonKind });
        else usable.push(item);
      }
      // The mounted library is reconciled against the scanned catalogue, so a
      // LoRA that is not installed — or cannot be told apart from a same-named
      // one, or belongs to another engine — cannot be mounted: it would be pruned
      // on the next scan or mount the wrong file. Only unique matches on the
      // target engine are mounted; every other row is itemised.
      if (usable.length) {
        overlay.loras = usable.map((item) => ({
          value: resolvedValue(item.match),
          ...inspectLoraIdentity(item).fields,
          name: item.name,
          weight: item.weight,
          enabled: true,
          engine: resolvedEngine(item.match),
        }));
        groups.push("loras");
      }
      if (carried.length) {
        promptLoras.push(...carried);
        notes.push(`${carried.length} 个 LoRA 由提示词中的 <lora:…> 标签加载，不再重复挂载`);
      }
      const prompts = toImageToImage ? imageToImage : overlay;
      let rewritten = 0;
      for (const { stem, reference, exactSource } of retargeted) {
        let changed = false;
        for (const key of ["positive", "negative"]) {
          if (typeof prompts[key] !== "string") continue;
          const next = exactSource ? rewriteExactLoraTags(prompts[key], exactSource, reference).text : retargetPromptLora(prompts[key], stem, reference);
          changed ||= next !== prompts[key];
          prompts[key] = next;
        }
        if (changed) rewritten += 1;
      }
      if (rewritten) notes.push(`${rewritten} 个同名 LoRA 标签已改写为所选文件的相对路径，生成时加载的就是这里显示的文件`);
    }
    if (field.id === "chain") {
      // Resolved against installed models and validated by the caller, which knows the catalogue;
      // it becomes `postprocessChain` only if the whole chain can run.
      postprocess = field.candidate;
      groups.push("chain");
    }
  }

  return {
    target: toImageToImage ? "i2i" : "t2i",
    targetEngine,
    overlay,
    imageToImage,
    groups,
    missing,
    skipped,
    notes,
    postprocess,
    promptLoras,
    unapplied: [],
    applied: available.map((field) => field.id),
  };
}

export function imageInfoApplySummary(plan) {
  if (!plan) return "";
  const parts = [];
  if (plan.groups.length) parts.push(`已应用 ${plan.groups.length} 项`);
  else parts.push("没有可应用的项目");
  // Three different reasons a model was left out, counted apart: absent here, several same-named
  // files, or a file that belongs to another engine.
  const count = (kind) => plan.missing.filter((item) => (item.reasonKind || "missing") === kind).length;
  if (count("missing")) parts.push(`${count("missing")} 个模型本地缺失`);
  if (count("ambiguous")) parts.push(`${count("ambiguous")} 个模型有歧义`);
  if (count("engine")) parts.push(`${count("engine")} 个模型属于其他引擎`);
  if (plan.unapplied?.length) parts.push(`${plan.unapplied.length} 个后处理项未应用`);
  if (plan.skipped.length) parts.push(`${plan.skipped.length} 项本工作区不支持`);
  return parts.join(" · ");
}
import { inspectLoraIdentity } from "./lora-identity.js";
import { rewriteExactLoraTags } from "./lora-relink.js";
