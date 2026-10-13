// QPlan2/P1 Phase E: one XYZ cell meets Basic Generation, the Gallery and the Viewer through the
// owners those already have. A cell's settings are its FROZEN REQUEST — what actually ran — turned
// into the workspace record `applyGallerySettings` and the Gallery take; nothing here writes a
// setting, picks a field for the user or reads the current draft. Pure on purpose: no fetch, no React.
import { PROMPT_BLOCK_SEPARATOR, activePromptPresetBlocks, composeGenerationPrompt, joinPromptBlocks } from "./prompt-presets.js";
import { backendChainToFrontend } from "./image-info-postprocess.js";

/** The groups a cell offers, in the Gallery's own vocabulary. Prompt alone is preselected. */
export const XYZ_APPLY_GROUPS = Object.freeze([
  ["prompts", "提示词", "正向与负向提示词（不含组合预设词）"],
  ["model", "模型组件", "底模或完整模型组件"],
  ["loras", "LoRA 组合", "本格实际挂载的有序列表与权重"],
  ["sampling", "采样参数", "步数、CFG、引导、Seed、采样器与调度器"],
  ["canvas", "画布大小", "宽度与高度"],
  ["chain", "后处理链", "本格实际运行的后处理链"],
  ["auxiliary", "其他选项", "透明背景模型"],
]);
export const XYZ_APPLY_DEFAULT_GROUPS = Object.freeze(["prompts"]);

const text = (value) => (typeof value === "string" ? value : value === null || value === undefined ? "" : String(value));
const blockList = (blocks) => (Array.isArray(blocks) ? blocks : []).map((block) => text(block).trim()).filter(Boolean);

/**
 * `body` without the preset blocks Basic would add around it, when — and only when — they frame it
 * exactly (`joinPromptBlocks([...leading, rest, ...trailing]) === body`). Anything else is left whole.
 */
export function stripPresetBlocks(body, { leading = [], trailing = [] } = {}) {
  const whole = text(body).trim(), lead = blockList(leading), tail = blockList(trailing);
  if (!lead.length && !tail.length) return { text: whole, removed: false };
  let rest = whole;
  const prefix = lead.join(PROMPT_BLOCK_SEPARATOR), suffix = tail.join(PROMPT_BLOCK_SEPARATOR);
  if (prefix) {
    if (rest === prefix) rest = "";
    else if (rest.startsWith(prefix + PROMPT_BLOCK_SEPARATOR)) rest = rest.slice(prefix.length + PROMPT_BLOCK_SEPARATOR.length);
    else return { text: whole, removed: false };
  }
  if (suffix) {
    if (rest === suffix) rest = "";
    else if (rest.endsWith(PROMPT_BLOCK_SEPARATOR + suffix)) rest = rest.slice(0, rest.length - suffix.length - PROMPT_BLOCK_SEPARATOR.length);
    else return { text: whole, removed: false };
  }
  return joinPromptBlocks([...lead, rest, ...tail]) === whole ? { text: rest.trim(), removed: true } : { text: whole, removed: false };
}

/**
 * What Apply writes into Basic's prompt boxes for one cell, and the prompt Basic would then send.
 *
 * `parts` is the server's `prompt_parts` (the cell's combination words already split off when they
 * lead its frozen prompt). `context` is Basic as it stands: `groupPrompt` (the words its enabled
 * combinations add — after the apply, when the caller knows LoRAs are applied too) and the prompt
 * preset library with the ids switched on. Preset text is removed only when Basic's switched-on
 * presets frame the cell text exactly, so they are not added twice; `reproduces` says whether
 * Basic, as it stands, would send exactly the cell's prompt.
 */
export function xyzBasicPrompt(parts, context = {}) {
  const records = Array.isArray(context.records) ? context.records : [];
  const positiveIds = Array.isArray(context.activePositive) ? context.activePositive : [];
  const negativeIds = Array.isArray(context.activeNegative) ? context.activeNegative : [];
  const groupPrompt = text(context.groupPrompt);
  const positive = stripPresetBlocks(parts?.rest ?? parts?.prompt, activePromptPresetBlocks(records, positiveIds, "positive"));
  const negative = stripPresetBlocks(parts?.negative_prompt, activePromptPresetBlocks(records, negativeIds, "negative"));
  const preview = {
    positive: composeGenerationPrompt({ groupPrompt, records, activeIds: positiveIds, type: "positive", prompt: positive.text }),
    negative: composeGenerationPrompt({ records, activeIds: negativeIds, type: "negative", prompt: negative.text }),
  };
  const reproduces = {
    positive: preview.positive === joinPromptBlocks([parts?.prompt]),
    negative: preview.negative === joinPromptBlocks([parts?.negative_prompt]),
  };
  const notes = [];
  if (parts?.removed) notes.push(parts.lead_source === "profile" ? "已去除本格 LoRA 方案的组合预设词：基础生图只按自己启用的组合加入组合词" : "已去除本格开头的 LoRA 组合预设词：基础生图会按当前启用的组合在生成时加入");
  else if (parts?.expected_lead) notes.push("本格提示词不以实验的组合预设词开头（提示词轴或精确替换改写过），按原文写入");
  if (positive.removed || negative.removed) notes.push("已去除基础生图当前启用的提示词预设文本，生成时由预设再次加入");
  if (!reproduces.positive || !reproduces.negative) notes.push("按基础生图当前的组合与预设，生成时的实际提示词与本格不同；见下方对照");
  return { positive: positive.text, negative: negative.text, removedLead: parts?.removed ? text(parts.lead) : "", leadSource: parts?.removed ? parts.lead_source : null,
    presetsRemoved: { positive: positive.removed, negative: negative.removed }, preview, reproduces, notes };
}

/** The cell's frozen request as a workspace/Gallery settings record. Seeds stay decimal strings. */
export function xyzCellSettings(cell, prompt) {
  const request = cell?.request || {};
  const pag = request.pag || {};
  return {
    model: text(request.engine),
    checkpoint: text(request.checkpoint),
    diffusionModel: text(request.diffusion_model),
    textEncoder: text(request.text_encoder),
    textEncoder2: text(request.text_encoder_2),
    vae: text(request.vae),
    positive: text(prompt?.positive),
    negative: text(prompt?.negative),
    steps: request.steps,
    cfg: request.cfg,
    denoise: request.denoise ?? 1,
    seed: text(request.seed),
    seedMode: "fixed",
    sampler: text(request.sampler),
    scheduler: text(request.scheduler),
    guidance: text(request.guidance) || "none",
    pag: { scale: pag.scale, appliedLayers: pag.applied_layers },
    size: { width: request.width, height: request.height },
    clipSkip: Number.isInteger(request.clip_skip) ? request.clip_skip : null,
    // Instance identities exactly as admitted for this cell; a weight-0 target is not in the list.
    loras: structuredClone(Array.isArray(cell?.lora_references) ? cell.lora_references : []),
    postprocessChain: backendChainToFrontend(Array.isArray(request.postprocess_chain) ? request.postprocess_chain : []),
    backgroundRemovalModel: text(request.background_removal_model),
    // A record of the run, never the group library: the words that led the prompt and what ran.
    loraGroups: [],
    loraGroupPrompt: text(prompt?.removedLead),
    composedPrompt: text(cell?.prompt_parts?.prompt ?? request.prompt),
    composedNegative: text(request.negative_prompt),
  };
}

function modelSummary(settings) {
  if (settings.checkpoint) return settings.checkpoint;
  return [settings.diffusionModel, settings.textEncoder, settings.textEncoder2, settings.vae].filter(Boolean).join(" · ") || "未记录模型";
}

/** One row per group with what applying it would write; nothing is preselected here. */
export function xyzApplyRows(settings) {
  const chain = settings.postprocessChain.filter((stage) => stage.enabled);
  const summaries = {
    prompts: settings.positive ? settings.positive.replace(/\s+/g, " ").slice(0, 120) : "（空正向提示词）",
    model: `${settings.model} · ${modelSummary(settings)}`,
    loras: settings.loras.length ? settings.loras.map((item) => `${item.name || item.value} ${item.weight}`).join("、") : "本格不挂载 LoRA；应用会清空基础生图当前引擎的挂载",
    sampling: `${settings.steps} 步 · CFG ${settings.cfg} · ${settings.sampler} / ${settings.scheduler} · Seed ${settings.seed}${settings.guidance && settings.guidance !== "none" ? ` · ${settings.guidance}` : ""}${settings.clipSkip ? ` · CLIP skip ${settings.clipSkip}` : ""}`,
    canvas: `${settings.size.width} × ${settings.size.height}`,
    chain: chain.length ? chain.map((stage) => stage.type === "resample" ? `重采样 ${stage.scale ?? 1}x` : stage.type === "adetailer" ? "ADetailer" : "RTX VSR").join(" → ") : "本格未运行后处理；应用会清空基础生图的后处理链",
    auxiliary: settings.backgroundRemovalModel || "本格未使用透明背景；应用会关闭透明背景",
  };
  return XYZ_APPLY_GROUPS.map(([id, label, description]) => ({ id, label, description, summary: summaries[id] }));
}

/** The one output AddToGalleryDialog shows for a cell; its seed is the cell's frozen seed. */
export function xyzCellGalleryOutput(cell) {
  const output = cell?.output || {};
  const seed = text(cell?.request?.seed ?? cell?.seed);
  return {
    asset_id: output.asset_id || cell?.asset_id || "",
    url: cell?.thumbnail_url || output.asset_url || cell?.asset_url || "",
    batch_index: 1, image_index: 1, seed, base_seed: seed,
    width: output.width ?? cell?.width, height: output.height ?? cell?.height,
  };
}

/** The Viewer source for a picture this service published (a cell original or a grid export). */
export function xyzViewerSource({ assetId, name }) {
  return assetId ? { assetId, name: text(name) || "XYZ.png" } : null;
}
