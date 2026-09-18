// The safety envelope every post-processing stage is held to, and the rounding the RTX stage does.
//
// The stage list itself lives in `resample-chain.js`: with a variable number of resample units the
// order is the chain rather than a list of stage names, so what is left here is the arithmetic both
// modules need and neither owns.

export const MAX_RTX_OUTPUT_EDGE = 8192;
export const MAX_RTX_OUTPUT_PIXELS = 32 * 1024 * 1024;
export const RTX_TARGET_LIMIT_REASON = "后处理目标超出 8192 边 / 32MP 安全上限";

/**
 * Round half to even, the way Python's `round` does.
 *
 * Every chain size is computed in both places — here, to show the card, and in the backend, to run
 * the stage. JavaScript's `Math.round` breaks ties upward and Python's breaks them to even, so a
 * size ending in exactly .5 would differ between the preview and the result.
 */
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
