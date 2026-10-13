import capabilities from "./xyz-capabilities.json" with { type: "json" };
import { generationMode } from "./generation-modes.js";
import { validXYZPromptSource } from "./xyz-prompt-source.js";
import { validXYZPostprocessSource } from "./xyz-postprocess-source.js";
import { validXYZSeedSource, validXYZAxisSource, xyzSampleSeeds, projectXYZEditor } from './xyz-editor-source.js';
import { xyzLocalFieldIssues } from './xyz-local-validation.js';

export const XYZ_CAPABILITIES = capabilities;
export const emptyXYZDrafts = () => ({ schema_version: 1, by_engine: {} });
const clone = (value) => structuredClone(value);
const object = (value) => value && typeof value === "object" && !Array.isArray(value);

/** Never coerce/clamp/drop a bad or future draft, nor mistake a plan for a draft. */
export function inspectXYZDrafts(raw, selectedMode = "basic") {
  if (raw === undefined && generationMode(selectedMode)) return { container: emptyXYZDrafts(), raw: null, error: "", fatal: false };
  const valid = object(raw) && raw.schema_version === 1 && object(raw.by_engine) && generationMode(selectedMode)
    && Object.entries(raw.by_engine).every(([engine, draft]) => capabilities.engines[engine] && object(draft)
      && draft.kind === "xyz_draft" && draft.schema_version === 1 && draft.engine === engine
      && object(draft.baseline) && object(draft.baseline.parameters) && draft.baseline.parameters.engine === engine
      && typeof draft.baseline.parameters.prompt === "string" && typeof draft.baseline.parameters.negative_prompt === "string"
      && object(draft.baseline.parameters.pag) && Array.isArray(draft.baseline.parameters.postprocess_chain)
      && Array.isArray(draft.baseline.loras) && draft.baseline.loras.every(object) && Array.isArray(draft.baseline.group_snapshots)
      && (draft.baseline.prompt_source === undefined || validXYZPromptSource(draft.baseline.prompt_source))
       && (draft.baseline.postprocess_source === undefined || validXYZPostprocessSource(draft.baseline.postprocess_source))
       && (draft.baseline.seed_source === undefined || validXYZSeedSource(draft.baseline.seed_source))
      && object(draft.axes) && Object.keys(draft.axes).every((key) => ["x", "y", "z"].includes(key))
      && ["x", "y", "z"].every((key) => {
        const axis = draft.axes[key];
         return axis === null || axis === undefined || object(axis) && typeof axis.field === "string"
           && (axis.editor_source === undefined || validXYZAxisSource(axis.editor_source))
          && Array.isArray(axis.values) && axis.values.every((entry) => object(entry) && Object.hasOwn(entry, "value"))
          && (!axis.range || object(axis.range) && ["start", "end", "step"].every((part) => typeof axis.range[part] === "string"));
      }));
  return valid ? { container: clone(raw), raw: null, error: "", fatal: false }
    : { container: emptyXYZDrafts(), raw: clone(raw), error: "XYZ 草稿版本或结构无法安全读取；保留原始数据，已暂停自动保存。执行记录不能作为新草稿恢复。", fatal: true };
}

/** Caller supplies the very same composed words/chain as Basic generate, but no RNG. */
export function createXYZDraft(baseline) {
  return { schema_version: 1, kind: "xyz_draft", engine: baseline.parameters.engine, baseline: clone(baseline),
    axes: { x: null, y: null, z: null }, repeat_count: 1, sample_seeds: null, requested_concurrency: 1, inherit_postprocess: false, group_by_resources: false };
}
export function groupXYZProfile(group, { includePresetText = false, mode = "keep_background" } = {}) {
  return { mode, members: clone(group.members), test_targets: group.members.map((member) => member.instanceId),
    includePresetText, preset_prompt: group.presetPrompt || "", name: group.name, id: group.id };
}
export function withXYZDraft(container, engine, draft) {
  if (draft.engine !== engine || draft.kind !== "xyz_draft") throw new Error("XYZ engine/draft mismatch");
  return { ...container, by_engine: { ...container.by_engine, [engine]: clone(draft) } };
}
export function syncXYZBaseline(draft, baseline, fields) {
  if (draft.engine !== baseline.parameters.engine) throw new Error("XYZ baseline engine mismatch");
  const allowed = ["models", "prompts", "sampling", "canvas", "loras", "postprocess", "output"];
  if (!fields.length || fields.some((field) => !allowed.includes(field))) throw new Error("Select explicit baseline fields");
  const next = clone(draft), source = baseline.parameters, target = next.baseline.parameters;
  const copyFields = (keys) => keys.forEach((key) => { if (Object.hasOwn(source, key)) target[key] = clone(source[key]); else delete target[key]; });
  if (fields.includes("models")) copyFields(["checkpoint", "diffusion_model", "text_encoder", "text_encoder_2", "vae"]);
  if (fields.includes("prompts")) {
    copyFields(["prompt", "negative_prompt", "background_removal_model"]);
    for (const key of ["prompt_without_groups", "group_prompt", "group_snapshots"]) next.baseline[key] = clone(baseline[key]);
    if (baseline.prompt_source) next.baseline.prompt_source = clone(baseline.prompt_source);
    else delete next.baseline.prompt_source;
  }
  if (fields.includes("sampling")) { copyFields(["steps", "cfg", "denoise", "seed", "sampler", "scheduler", "guidance", "pag", "clip_skip"]); next.baseline.seed_mode = baseline.seed_mode; if (baseline.seed_source) next.baseline.seed_source = clone(baseline.seed_source); else delete next.baseline.seed_source; }
  if (fields.includes("canvas")) copyFields(["width", "height"]);
  if (fields.includes("loras")) { next.baseline.loras = clone(baseline.loras); next.baseline.library_revision = baseline.library_revision; }
  if (fields.includes("postprocess")) {
    copyFields(["postprocess_chain"]);
    if (baseline.postprocess_source) next.baseline.postprocess_source = clone(baseline.postprocess_source);
    else delete next.baseline.postprocess_source;
  }
  if (fields.includes("output")) copyFields(["output_template"]);
  return next;
}

export function xyzAxisOptions(engine) {
  const cap = capabilities.engines[engine];
  return [
    ["steps", "步数"], ["cfg", cap.negative_prompt ? "CFG" : "蒸馏引导"], ["sampler", "采样器"], ["scheduler", "调度器"], ["seed", "种子 · uint64"],
    ...cap.model_fields.filter((key) => key !== "text_encoder" && key !== "text_encoder_2").map((key) => [key, key === "checkpoint" ? "Checkpoint" : key === "vae" ? "VAE" : "扩散模型"]),
    ["components", "完整模型组件方案"], ["prompt", "正向文本"], ["prompt_replace", "正向精确替换"],
    ...(cap.negative_prompt ? [["negative_prompt", "负向文本"], ["negative_replace", "负向精确替换"]] : []),
    ...(cap.clip_skip ? [["clip_skip", "CLIP skip"]] : []),
    ...(cap.guidance.length > 1 ? [["guidance", "引导类型"]] : []),
    ...(cap.guidance.includes("pag") ? [["pag.scale", "PAG 强度 · 需 PAG"]] : []),
    ["lora_weight", "LoRA 实例权重"],
  ].map(([value, label]) => ({ value, label }));
}

// UI-only cardinality plus typed input guards. NOT resource admission or a resolved plan.
// Exact decimal ranges use BigInt, and resource limits match the Python owner.
export function decimalRangeValues(range) {
  const parts = [range.start, range.end, range.step];
  if (parts.some((value) => typeof value !== "string" || value.length > 64 || !/^-?(?:\d+(?:\.\d{1,16})?|\.\d{1,16})$/.test(value))) throw new Error("范围需要有限十进制文本（最多 16 位小数）");
  const places = Math.max(...parts.map((value) => value.split(".")[1]?.length || 0));
  const scale = 10n ** BigInt(places);
  const ints = parts.map((value) => { const [whole, fraction = ""] = value.replace("-", "").split("."); return (BigInt(whole || "0") * scale + BigInt(fraction.padEnd(places, "0") || "0")) * (value.startsWith("-") ? -1n : 1n); });
  const [start, end, step] = ints;
  if (!step || (end - start) * step < 0n) throw new Error("步长不能为 0，且必须朝向终点");
  const count = (end - start) / step + 1n;
  if (count > BigInt(capabilities.limits.axis_values)) throw new Error("每轴最多 256 项；请减少范围");
  return Array.from({ length: Number(count) }, (_, i) => {
    const number = start + BigInt(i) * step, absolute = number < 0n ? -number : number;
    const fraction = places ? `.${String(absolute % scale).padStart(places, "0")}` : "";
    return `${number < 0n ? "-" : ""}${absolute / scale}${fraction}`;
  });
}
export function xyzLocalSummary(draft) {
  try {
    if (!Number.isInteger(draft.repeat_count) || draft.repeat_count < 1 || draft.repeat_count > 32) throw new Error("重复样本必须为 1–32 整数");
    if (!Number.isInteger(draft.requested_concurrency) || draft.requested_concurrency < 1 || draft.requested_concurrency > 10) throw new Error("请求并发必须为 1–10 整数；不会自动调整");
    const expanded = {};
    const dimensions = Object.fromEntries(["x", "y", "z"].map((key) => {
      const axis = draft.axes[key];
      const count = !axis ? 1 : axis.range ? decimalRangeValues(axis.range).length : axis.values.length;
      expanded[key] = !axis ? [] : axis.range ? decimalRangeValues(axis.range) : axis.values.map(e=>e.value);
      if (count < 1 || count > 256) throw new Error(`${key.toUpperCase()} 轴需要 1–256 项`);
      return [key, count];
    }));
    const total = dimensions.x * dimensions.y * dimensions.z * draft.repeat_count;
    if (total > 1024) throw new Error("最多 1024 格；不截断组合，请减少轴项或重复次数");
    let issues = xyzLocalFieldIssues(draft,expanded,null);
    if (!issues.length) try { issues = xyzLocalFieldIssues(draft,expanded,projectXYZEditor(draft)); } catch(error) {issues=[error.message];}
    return { dimensions, total, valid:!issues.length, issues, error:issues[0] || "", authoritative: false, seedsResolved: false };
  } catch (error) { return { total: null, valid:false, issues:[error.message], error: error.message, authoritative: false, seedsResolved: false }; }
}
