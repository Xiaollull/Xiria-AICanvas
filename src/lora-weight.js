// Mounted LoRA weight arithmetic.
//
// Split out of the components so the three ways a weight can change — dragging
// the slider, typing a number, clicking a stepper — agree on rounding and on the
// range, and so each is unit-tested rather than verified by dragging things.

export const WEIGHT_MINIMUM = -5;
export const WEIGHT_MAXIMUM = 5;
export const WEIGHT_SLIDER_STEP = 0.01;
export const WEIGHT_STEP_DELTA = 0.1;

export function clampWeight(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  return Math.max(WEIGHT_MINIMUM, Math.min(WEIGHT_MAXIMUM, numeric));
}

/**
 * Kills binary float drift: 1.1 - 0.1 must be 1, not 0.9999999999999999. The
 * `|| 0` also folds -0 back to 0, which Math.ceil produces on the way up from a
 * small negative weight and which would otherwise be stored in the workspace.
 */
function round(value, places) {
  const factor = 10 ** places;
  return Math.round(value * factor) / factor || 0;
}

/**
 * Slider semantics, unchanged from the original control: a plain drag moves in
 * tenths and carries the hundredth digit the user set earlier, so dragging does
 * not silently discard a 1.25 down to 1.2. Holding Shift drags in hundredths.
 */
export function sliderWeight(current, raw, fine = false) {
  const value = clampWeight(raw);
  if (fine) return { weight: round(value, 2), precision: 2 };
  const currentWeight = clampWeight(current?.weight);
  const hundredth = (current?.precision ?? 1) >= 2 ? Math.abs(Math.round(currentWeight * 100)) % 10 : 0;
  const sign = value < 0 ? -1 : 1;
  const tenths = Math.round(Math.abs(value) * 10) * 10;
  return { weight: sign * Math.min(500, tenths + hundredth) / 100, precision: current?.precision ?? 1 };
}

/**
 * One stepper click. Snaps onto the tenths grid first, so stepping up from a
 * hand-typed 1.234 reaches 1.3 rather than 1.334 — the button exists to give
 * round numbers, which is the whole reason to prefer it over the text field.
 */
export function steppedWeight(current, direction) {
  const sign = direction < 0 ? -1 : 1;
  const steps = clampWeight(current?.weight) / WEIGHT_STEP_DELTA;
  const snapped = Math.round(steps);
  // 1.0 / 0.1 is 9.999999999999998, so an exact-grid test has to tolerate the
  // representation error or every stepper click from a round number is a no-op.
  const onGrid = Math.abs(steps - snapped) < 1e-6;
  const target = onGrid ? snapped + sign : (sign > 0 ? Math.ceil(steps) : Math.floor(steps));
  return { weight: clampWeight(round(target * WEIGHT_STEP_DELTA, 1)), precision: 1 };
}

/**
 * Free-text entry. Intermediate states a user must be able to type on the way to
 * a number — "", "-", "1.", "-0." — are reported as incomplete rather than
 * invalid, so the field can hold them instead of snapping the value back and
 * making the box impossible to edit.
 */
// "", "-", ".", "-.", "1.", "-12." — a number is being typed but is not one yet.
const INCOMPLETE_WEIGHT = /^-?(?:\d+\.|\.|)$/;
const COMPLETE_WEIGHT = /^-?(?:\d+(?:\.\d+)?|\.\d+)$/;

export function parseWeightInput(text) {
  const raw = typeof text === "string" ? text.trim() : "";
  // Order matters: a lone "-" fails the number test, and reporting it invalid is
  // what made the old field impossible to edit — it wrote the previous value
  // back on the first keystroke of a negative weight.
  if (INCOMPLETE_WEIGHT.test(raw)) return { state: "incomplete", weight: null };
  if (!COMPLETE_WEIGHT.test(raw)) return { state: "invalid", weight: null };
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) return { state: "invalid", weight: null };
  return { state: "valid", weight: round(clampWeight(parsed), 4) };
}

/**
 * The one way a mounted weight is written on screen.
 *
 * `precision` records *how* the weight was last set — tenths grid, hundredths
 * locked, hand-typed — which is a fact about the input, not about the number.
 * Formatting with it rounded 0.75 to "0.8" in the summaries while the manager
 * showed 0.75, and neither 0.8 nor the rounding was what the request would
 * carry: `toFixed` reads the binary value, so 0.25 rose to "0.3" while 0.15 fell
 * to "0.1". A weight is shown as the number that will be sent. `toFixed(4)`
 * only removes representation noise, and `Number` strips the trailing zeros it
 * leaves behind so 0.3 does not read as "0.3000".
 */
export function formatWeight(value) {
  return String(Number(clampWeight(value).toFixed(4)));
}

/** How the weight was last set, for the hint under the entry's name. */
export function weightPrecisionLabel(item, shiftPressed) {
  if (shiftPressed) return "精调 0.01";
  if ((item?.precision ?? 1) >= 4) return "手动精度 0.0001";
  return item?.precision === 2 ? "百分位已锁定" : "步进 0.1";
}
