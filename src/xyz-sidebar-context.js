import { XYZ_CAPABILITIES } from "./xyz-draft.js";
import { validXYZSeed } from "./xyz-axis-editor.js";
import { inspectMountedLoras } from "./lora-state.js";
import { inspectLoraIdentity } from "./lora-identity.js";
import { captureXYZPromptSource } from "./xyz-prompt-source.js";
import { mossRequestFields } from "./moss-engine.js";

/** UI adapters only. Never normalize a saved draft, rebind an identity, or write Basic. */
export function xyzCatalogChoices(payload) {
  const cap = XYZ_CAPABILITIES.engines[payload?.engine];
  const fields = cap?.model_fields || (payload?.model_type === "split" ? ["diffusion_model", "text_encoder", "text_encoder_2", "vae"] : ["checkpoint"]);
  return Object.fromEntries(fields.map((field) => [field, (field === "checkpoint" ? payload?.models || [] : payload?.assets?.[field]?.models || []).map((item) => ({ value: item.value, label: item.name || item.value }))]));
}

export function xyzBaselineFromSettings(engine, settings, { mounts, groups, groupPrompt, composePrompt, chainPayload, clipPayload, output = {}, backgroundRemovalModel = null, libraryRevision, promptRecords }) {
  const cap = XYZ_CAPABILITIES.engines[engine];
  const names = { checkpoint: "checkpoint", diffusion_model: "diffusionModel", text_encoder: "textEncoder", text_encoder_2: "textEncoder2", vae: "vae" };
  const baseline = structuredClone({ parameters: {
    engine, ...Object.fromEntries(cap.model_fields.map((field) => [field, settings[names[field]] || ""])),
    prompt: composePrompt("positive", settings.positive, settings.activePromptPresets, groupPrompt),
    negative_prompt: cap.negative_prompt ? composePrompt("negative", settings.negative, settings.activePromptPresets, "") : "",
    steps: settings.steps, cfg: settings.cfg, denoise: settings.denoise, width: settings.size.width, height: settings.size.height,
    seed: String(settings.seed), sampler: settings.sampler, scheduler: settings.scheduler, guidance: settings.guidance,
    pag: { scale: settings.pag?.scale ?? 0.3, applied_layers: settings.pag?.appliedLayers ?? "mid" },
    postprocess_chain: chainPayload, background_removal_model: backgroundRemovalModel, ...clipPayload, ...output,
    // MOSS freezes its versioned sampling block into the baseline; other engines carry none.
    ...mossRequestFields(engine, settings.mossSampling),
   }, seed_mode: settings.seedMode === "random" ? "random" : "fixed", seed_source:{schema_version:1,mode:settings.seedMode}, loras: mounts, group_snapshots: groups,
    prompt_without_groups: composePrompt("positive", settings.positive, settings.activePromptPresets, ""), group_prompt: groupPrompt, library_revision: libraryRevision });
  return promptRecords ? captureXYZPromptSource(baseline, { positive: settings.positive, negative: settings.negative, active: settings.activePromptPresets, records: promptRecords }) : baseline;
}

export function patchXYZParameters(draft, patch) {
  if (Object.hasOwn(patch, "engine")) throw new Error("Switch draft owner, not baseline.engine");
  const next = structuredClone(draft);
  Object.assign(next.baseline.parameters, patch);
  if (Object.hasOwn(patch, "prompt") || Object.hasOwn(patch, "negative_prompt")) delete next.baseline.prompt_source;
  if (Object.hasOwn(patch, "prompt")) {
    // Editing actual composed text explicitly detaches the old group prefix; never compose twice.
    next.baseline.prompt_without_groups = patch.prompt;
    next.baseline.group_prompt = "";
  }
  return next;
}

export function xyzStrongLora(item) {
  return Boolean(item?.instanceId && item.identitySchema === 1 && /^[a-f0-9]{64}$/.test(item.contentHash || "") && Number.isInteger(item.contentRevision) && item.contentRevision > 0 && !inspectLoraIdentity(item).fatal);
}

export function editXYZMounts(draft, action) {
  const next = structuredClone(draft), mounts = next.baseline.loras;
  if (action.kind === "add") {
    if (!xyzStrongLora(action.item)) throw new Error("候选缺少强实例 / 内容修订，不能按同名文件挂载。");
    if (mounts.some((item) => item.instanceId === action.item.instanceId)) throw new Error("该实例已挂载。");
    if (mounts.length >= 16) throw new Error("最多挂载 16 个 LoRA；现有列表未改动。");
    // Project a NEW catalog choice through the existing mount adapter. Catalog-only metadata must
    // not leak into strict LoraReference. Never run normalization over the saved background list.
    const parsed = inspectMountedLoras([{ ...action.item, weight: 1, precision: 1, enabled: true }]);
    if (parsed.fatal || parsed.items.length !== 1) throw new Error("候选身份无法安全读取。");
    mounts.push(structuredClone(parsed.items[0]));
  } else {
    const index = action.index;
    if (!Number.isInteger(index) || !mounts[index]) throw new Error("挂载项不存在。");
    if (action.kind === "remove") mounts.splice(index, 1);
    else if (action.kind === "move") {
      const to = index + action.delta;
      if (to < 0 || to >= mounts.length) return next;
      [mounts[index], mounts[to]] = [mounts[to], mounts[index]];
    } else if (action.kind === "patch") {
      if (Object.keys(action.patch).some((key) => !["weight", "precision", "enabled"].includes(key))) throw new Error("Identity cannot be edited");
      if (Object.hasOwn(action.patch, "weight") && (action.patch.weight === "" || !Number.isFinite(Number(action.patch.weight)) || Math.abs(Number(action.patch.weight)) > 5)) throw new Error("权重需为 −5 到 5；不会静默截断。");
      Object.assign(mounts[index], action.patch);
    } else throw new Error("Unknown mount action");
  }
  return next;
}

/** A library group is copied into a frozen experiment snapshot, never toggled in the library. */
export function copyXYZGroup(draft, group) {
  if (group.members.length > 16 || group.members.some((item) => !xyzStrongLora(item))) throw new Error("组合有未解析身份或超过 16 项；原始基线未改动。");
  const next = structuredClone(draft);
  next.baseline.loras = structuredClone(group.members);
  const index = next.baseline.group_snapshots.findIndex((item) => item.id === group.id);
  if (index === -1) next.baseline.group_snapshots.push(structuredClone(group));
  else next.baseline.group_snapshots[index] = structuredClone(group);
  // Keep the actual prompt fixed. Preset text is opt-in at the axis profile, not added here.
  return next;
}

export function xyzSidebarIssues(draft) {
  if (!draft) return [];
  const p = draft.baseline.parameters, issues = [];
  if (!Number.isInteger(p.steps) || p.steps < 1) issues.push("基线步数需为正整数");
  if (p.cfg === "" || !Number.isFinite(Number(p.cfg))) issues.push("基线 CFG 需为有限数值");
  if (draft.baseline.seed_mode === "fixed" && !validXYZSeed(p.seed)) issues.push("基线种子需为 uint64 文本");
  for (const field of ["width", "height"]) if (!Number.isInteger(p[field]) || p[field] < 64 || p[field] % 64 || p[field] > XYZ_CAPABILITIES.engines[draft.engine].max_edge) issues.push(`基线${field === "width" ? "宽度" : "高度"}需为至少 64 的整数 / 64 像素倍数，且不超过引擎上限`);
  return issues;
}
