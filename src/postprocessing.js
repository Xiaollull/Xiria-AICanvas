export const POSTPROCESS_STAGE_IDS = ["hires", "adetailer", "rtx"];
export const MAX_RTX_OUTPUT_EDGE = 8192;
export const MAX_RTX_OUTPUT_PIXELS = 32 * 1024 * 1024;
export const RTX_TARGET_LIMIT_REASON = "RTX VSR target exceeds the safe 8192-edge / 32-megapixel limit";

export function normalizePostprocessOrder(value, legacyHiresOrder) {
  const source = Array.isArray(value)
    ? value
    : legacyHiresOrder === "after_adetailer"
      ? ["adetailer", "hires", "rtx"]
      : POSTPROCESS_STAGE_IDS;
  const normalized = [];
  for (const stage of [...source, ...POSTPROCESS_STAGE_IDS]) {
    if (POSTPROCESS_STAGE_IDS.includes(stage) && !normalized.includes(stage)) normalized.push(stage);
  }
  return normalized;
}

export function enabledPostprocessStages(order, settings) {
  return normalizePostprocessOrder(order).filter((stage) => settings[stage]?.enabled === true);
}

export function bankersRound(value) {
  if (!Number.isFinite(value)) throw new TypeError("bankersRound requires a finite number");
  const sign = value < 0 ? -1 : 1;
  const absolute = Math.abs(value);
  const lower = Math.floor(absolute);
  const fraction = absolute - lower;
  const rounded = fraction < 0.5
    ? lower
    : fraction > 0.5
      ? lower + 1
      : lower % 2 === 0 ? lower : lower + 1;
  return rounded === 0 ? 0 : sign * rounded;
}

function rtxTargetSize(size, scale) {
  return {
    width: Math.max(8, bankersRound(Math.trunc(size.width * scale) / 8) * 8),
    height: Math.max(8, bankersRound(Math.trunc(size.height * scale) / 8) * 8),
  };
}

function rtxTargetIsSafe(size) {
  return Number.isFinite(size.width)
    && Number.isFinite(size.height)
    && Math.max(size.width, size.height) <= MAX_RTX_OUTPUT_EDGE
    && size.width * size.height <= MAX_RTX_OUTPUT_PIXELS;
}

export function postprocessTargetSize(size, order, settings) {
  let current = { width: size.width, height: size.height };
  const trace = [];
  let valid = true;
  let reason = "";

  for (const stage of enabledPostprocessStages(order, settings)) {
    const input = { ...current };
    if (stage === "hires") {
      current = {
        width: Math.ceil(current.width * settings.hires.scale / 64) * 64,
        height: Math.ceil(current.height * settings.hires.scale / 64) * 64,
      };
    } else if (stage === "rtx") {
      current = rtxTargetSize(current, settings.rtx.scale);
    }

    const stageValid = settings.rtx?.enabled !== true || rtxTargetIsSafe(current);
    const stageReason = stageValid ? "" : RTX_TARGET_LIMIT_REASON;
    trace.push({ stage, input, output: { ...current }, valid: stageValid, reason: stageReason });
    if (!stageValid && valid) {
      valid = false;
      reason = stageReason;
    }
  }

  return { ...current, trace, valid, reason };
}
