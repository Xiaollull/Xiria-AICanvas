// CLIP skip: which layer of the CLIP text towers an SD/iL prompt is read from, counted from the last.
//
// It exists where a CLIP stack encodes the prompt and nowhere else, so the four native engines —
// language models and T5 — have no range and never send the field. SD reads its final layer by
// default. SDXL/iL reads the penultimate layer of both towers, which is also the shallowest either
// can be read at (Forge's floor), so iL starts at 2: a 1 there would run as 2, and a control that
// shows one number while the job runs another is exactly what this avoids.

const CLIP_SKIP_RANGES = Object.freeze({
  SD: Object.freeze({ minimum: 1, maximum: 12, default: 1 }),
  iL: Object.freeze({ minimum: 2, maximum: 12, default: 2 }),
});

/** The CLIP skip an engine offers, or `null` for the engines that have no CLIP stack to skip in. */
export function clipSkipRange(engine) {
  return CLIP_SKIP_RANGES[engine] || null;
}

/** A stored or typed CLIP skip, as the value `engine` will actually run — `null` where it has none. */
export function normalizeClipSkip(engine, value) {
  const range = clipSkipRange(engine);
  if (!range) return null;
  const numeric = Number(value);
  if (value === null || value === undefined || value === "" || typeof value === "boolean" || !Number.isFinite(numeric)) return range.default;
  return Math.min(range.maximum, Math.max(range.minimum, Math.round(numeric)));
}

/** The request field a job carries: the effective value for SD/iL, nothing for the other engines. */
export function clipSkipPayload(engine, value) {
  const range = clipSkipRange(engine);
  return range ? { clip_skip: normalizeClipSkip(engine, value) } : {};
}
