// Diffusers image-to-image and inpaint retain only `int(total × strength)` updates. Post-processing
// exposes the retained update count instead, so both Resample and ADetailer use this one admission
// calculation. Keeping it dependency-free lets their model modules share the exact floating-point
// boundary without creating a resample-chain <-> adetailer-units import cycle.

export const DIFFUSERS_SCHEDULE_STEP_LIMIT = 1000;

/**
 * Return the smallest schedule total that makes Diffusers execute `steps`, or null when no total
 * fits the bounded quotient-centred search used by the backend.
 */
export function diffusersScheduleTotal(steps, denoise) {
  const requested = Math.floor(Number(steps));
  const strength = Number(denoise);
  if (!Number.isFinite(requested) || requested < 1 || !Number.isFinite(strength) || strength <= 0) return null;
  if (requested > DIFFUSERS_SCHEDULE_STEP_LIMIT) return null;
  if (strength >= 1) return requested;
  const estimatedTotal = Math.ceil(requested / strength);
  if (!Number.isFinite(estimatedTotal)) return null;
  const first = Math.max(requested, estimatedTotal - 2);
  const last = Math.min(DIFFUSERS_SCHEDULE_STEP_LIMIT, estimatedTotal + 2);
  for (let total = first; total <= last; total += 1) {
    if (Math.trunc(total * strength) === requested) return total;
  }
  return null;
}
