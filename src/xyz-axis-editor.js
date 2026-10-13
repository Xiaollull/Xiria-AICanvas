import capabilities from "./xyz-capabilities.json" with { type: "json" };

export const XYZ_UINT64_MAX = 18446744073709551615n;
const clone = (value) => structuredClone(value);

export function xyzStructuredText(value) {
  if (typeof value === "string") return value;
  try { return JSON.stringify(value, null, 2) ?? ""; } catch { return String(value ?? ""); }
}

/** Invalid text stays a plain raw string so it round-trips without an invented wrapper object. */
export function parseXYZStructuredText(text) {
  try { return { value: JSON.parse(text), valid: true }; }
  catch { return { value: text, valid: false }; }
}

export function validXYZSeed(value) {
  const text = String(value ?? "");
  if (!/^\d{1,20}$/.test(text)) return false;
  try { return BigInt(text) <= XYZ_UINT64_MAX; } catch { return false; }
}

export function validXYZFiniteDecimal(value) {
  if (typeof value !== "string" && typeof value !== "number") return false;
  const text = String(value);
  return text.length > 0 && text.length <= 64 && /^-?(?:\d+(?:\.\d{0,16})?|\.\d{1,16})$/.test(text) && Number.isFinite(Number(text));
}

export function initialXYZAxisValue(field, draft) {
  const parameters = draft?.baseline?.parameters || {};
  if (field === "components") return Object.fromEntries([
    ["engine", draft.engine],
    ...((capabilities.engines[draft.engine]?.model_fields || []).filter((key) => parameters[key]).map((key) => [key, parameters[key]])),
  ]);
  if (field === "prompt_replace" || field === "negative_replace") return { find: "", replace: "", expected_matches: 1 };
  if (field === "lora_profile") return { mode: "no_test", members: [], test_targets: [], includePresetText: false, preset_prompt: "" };
  if (field === "lora_model") return clone(draft?.baseline?.loras?.[0] || { value: "", weight: "1", identitySchema: 1, instanceId: "", contentHash: "", contentRevision: 1 });
  if (field === "lora_weight" || field === "lora_multiplier") return "1";
  if (field === "pag.scale") return String(parameters.pag?.scale ?? "");
  if (field === "seed") return String(parameters.seed ?? "0");
  return String(parameters[field] ?? "");
}

export function initialXYZRange(field, draft) {
  const value = initialXYZAxisValue(field, draft);
  const start = validXYZFiniteDecimal(value) ? String(value) : "1";
  const step = field === "pag.scale" ? "0.1" : "1";
  const places = Math.max(start.split(".")[1]?.length || 0, step.split(".")[1]?.length || 0);
  const scale = 10n ** BigInt(places);
  const scaled = text => {const [whole,fraction='']=text.replace('-','').split('.');return (BigInt(whole||'0')*scale+BigInt(fraction.padEnd(places,'0')||'0'))*(text.startsWith('-')?-1n:1n);};
  const n=scaled(start)+2n*scaled(step),absolute=n<0n?-n:n;
  const end=`${n<0n?'-':''}${absolute/scale}${places?'.'+String(absolute%scale).padStart(places,'0'):''}`;
  return { start, end, step };
}

export function xyzComponentFields(engine) {
  return capabilities.engines[engine]?.model_fields || [];
}

const runtimeGuidanceReason = (status) => status === "unavailable"
  ? "运行时明确报告当前引擎不可用"
  : "当前引擎的运行时能力未知；读取到明确能力前不会放行";
const RUNTIME_GUIDANCE_MODES = new Set(["pag", "cfg_zero_star"]);
const PAG_IMPLEMENTATIONS = {
  SD: "diffusers_pag",
  iL: "diffusers_pag",
  Anima: "native_cosmos_identity_self_attention",
};

/** Interpret the published inference-health contract shared by Basic and XYZ guidance. */
export function runtimeGuidanceHealthStatus(health, engine, mode) {
  const key = mode === "pag" ? "pag" : "cfg_zero_star";
  const capability = health?.guidance?.[key];
  if (!capability || typeof capability.available !== "boolean") {
    return { status: "unknown", reason: `${mode === "pag" ? "PAG" : "CFG-Zero*"} 运行时能力尚未提供` };
  }
  if (capability.available === false) return { status: "unavailable", reason: `${mode === "pag" ? "PAG" : "CFG-Zero*"} 运行时明确不可用` };
  if (!Array.isArray(capability.engines)) return { status: "unknown", reason: `${mode === "pag" ? "PAG" : "CFG-Zero*"} 引擎支持列表尚未提供` };
  if (!capability.engines.includes(engine)) return { status: "unavailable", reason: `当前引擎 ${engine} 不在运行时支持列表中` };
  if (mode === "pag") {
    const feature = health?.engines?.[engine]?.features?.pag;
    if (feature === false) return { status: "unavailable", reason: `当前引擎 ${engine} 未启用 PAG` };
    if (feature === true || capability.implementations?.[engine] === PAG_IMPLEMENTATIONS[engine]) return { status: "available", reason: "" };
    return { status: "unknown", reason: `当前引擎 ${engine} 缺少明确 PAG 实现或特性证明` };
  }
  return { status: "available", reason: "" };
}

function guidanceHealthStatus(health, engine, mode) {
  if (!capabilities.engines[engine]?.guidance?.includes(mode)) {
    return { status: "unavailable", reason: `当前引擎 ${engine} 不支持 ${mode === "pag" ? "PAG" : "CFG-Zero*"}` };
  }
  return runtimeGuidanceHealthStatus(health, engine, mode);
}

/** Build an engine-tagged snapshot from the same current health fields used by Basic guidance. */
export function xyzRuntimeGuidanceForEngine(health, engine) {
  return {
    engine,
    modes: {
      none: { status: "available", reason: "" },
      pag: guidanceHealthStatus(health, engine, "pag"),
      cfg_zero_star: guidanceHealthStatus(health, engine, "cfg_zero_star"),
    },
  };
}

export function xyzGuidanceRuntimeStatus(runtimeGuidance, engine, mode) {
  if (mode === "none") return { status: "available", reason: "" };
  if (!runtimeGuidance || runtimeGuidance.engine !== engine) {
    return { status: "unknown", reason: "运行时能力快照不属于当前引擎；等待当前引擎能力" };
  }
  const status = runtimeGuidance.modes?.[mode];
  if (!status || !["available", "unavailable", "unknown"].includes(status.status)) {
    return { status: "unknown", reason: `${mode} 运行时能力尚未确认` };
  }
  return status;
}

export function xyzGuidanceOption(runtimeGuidance, engine, mode, label) {
  const status = xyzGuidanceRuntimeStatus(runtimeGuidance, engine, mode);
  return {
    value: mode,
    label: status.status === "available" ? label : `${label} · ${status.status === "unknown" ? "能力未知" : "当前不可用"}`,
    ...(status.status !== "available" ? { disabled: true, disabledReason: `${status.reason || runtimeGuidanceReason(status.status)}；保留已保存值，预检会阻止` } : {}),
  };
}

export function xyzAxisRuntimeIssues(draft, runtimeGuidance) {
  if (!draft?.axes || !draft.engine) return [];
  const issues = [];
  const engineGuidance = capabilities.engines[draft.engine]?.guidance || [];
  for (const axisKey of ["x", "y", "z"]) {
    const axis = draft.axes[axisKey];
    if (!axis) continue;
    if (axis.field === "pag.scale") {
      if (!engineGuidance.includes("pag")) issues.push(`${axisKey.toUpperCase()} 轴 PAG 强度当前引擎不支持；已保存定义保留`);
      else {
        const status = xyzGuidanceRuntimeStatus(runtimeGuidance, draft.engine, "pag");
        if (status.status !== "available") issues.push(`${axisKey.toUpperCase()} 轴 PAG 强度在当前运行时不可用：${status.reason}`);
      }
    }
    if (axis.field === "guidance") for (const [index, entry] of (Array.isArray(axis.values) ? axis.values : []).entries()) {
      const mode = entry?.value;
      if (mode === "none") continue;
      if (!RUNTIME_GUIDANCE_MODES.has(mode) || !engineGuidance.includes(mode)) {
        issues.push(`${axisKey.toUpperCase()} 轴第 ${index + 1} 项引导值「${String(mode)}」当前引擎不支持或无法识别；原值保留`);
        continue;
      }
      const status = xyzGuidanceRuntimeStatus(runtimeGuidance, draft.engine, mode);
      if (status.status !== "available") issues.push(`${axisKey.toUpperCase()} 轴第 ${index + 1} 项「${mode === "pag" ? "PAG" : "CFG-Zero*"}」在当前运行时不可用：${status.reason}`);
    }
  }
  return issues;
}

export function xyzGuidanceStatusReason(status) {
  return status.status === "unavailable" ? status.reason || runtimeGuidanceReason("unavailable")
    : status.status === "unknown" ? status.reason || runtimeGuidanceReason("unknown") : "";
}
