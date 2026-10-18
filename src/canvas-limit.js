// The canvas bound: how large a sampling canvas the pickers offer.
//
// It is one square number — X means "up to X × X" — because what grows with the canvas is the
// area the sampler works on, and a single number is what the grid, the inputs and the presets can
// all share. The canvas itself stays rectangular: a 1024 × 4096 canvas is inside a 4096 bound.
//
// 2048 is the default because it is the canvas every record before protocol 39 was held to, so an
// untouched workspace offers exactly what it always did. 8192 is the hard safety ceiling of the
// wire, not a promise: whether a given card can sample a canvas that large is decided by the
// server's resource admission when the job is submitted, before anything is queued.

export const DEFAULT_CANVAS_MAX_EDGE = 2048;
export const MIN_CANVAS_BOUND = 1024;
export const HARD_CANVAS_MAX_EDGE = 8192;
export const CANVAS_BOUND_STEP = 64;
export const CANVAS_BOUND_PRESETS = Object.freeze([1024, 2048, 3072, 4096, 6144, 8192]);
// Each native runtime refuses a canvas past its own edge; SD/iL go to the ceiling.
const ENGINE_CANVAS_CEILING = Object.freeze({ SD: 8192, iL: 8192, Anima: 4096, Flux: 4096, Flux2: 4096, Krea2: 4096, MOSS: 4096 });

/**
 * A stored or typed bound, and whether it had to be repaired.
 *
 * A value that is not a number, or off the 64-pixel grid, or outside 1024–8192 is a damaged
 * preference, not a request: it falls back to the default and says so, rather than being bent into
 * a bound nobody chose.
 */
export function normalizeCanvasMaxEdge(value) {
  if (value === undefined || value === null) return { value: DEFAULT_CANVAS_MAX_EDGE, warning: "" };
  const numeric = Number(value);
  if (typeof value === "boolean" || !Number.isInteger(numeric) || numeric < MIN_CANVAS_BOUND || numeric > HARD_CANVAS_MAX_EDGE || numeric % CANVAS_BOUND_STEP !== 0) {
    return { value: DEFAULT_CANVAS_MAX_EDGE, warning: `画布上限设置无效（${String(value).slice(0, 24)}），已恢复默认 ${DEFAULT_CANVAS_MAX_EDGE} × ${DEFAULT_CANVAS_MAX_EDGE}` };
  }
  return { value: numeric, warning: "" };
}

/** A typed bound snapped onto the grid and the range, for an input that is being edited. */
export function snapCanvasMaxEdge(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return DEFAULT_CANVAS_MAX_EDGE;
  const snapped = Math.round(numeric / CANVAS_BOUND_STEP) * CANVAS_BOUND_STEP;
  return Math.min(HARD_CANVAS_MAX_EDGE, Math.max(MIN_CANVAS_BOUND, snapped));
}

export function engineCanvasCeiling(engine) {
  return ENGINE_CANVAS_CEILING[engine] || HARD_CANVAS_MAX_EDGE;
}

/** The largest edge a picker should offer for `engine` under `maxEdge`. */
export function effectiveCanvasMaxEdge(maxEdge, engine) {
  return Math.min(normalizeCanvasMaxEdge(maxEdge).value, engineCanvasCeiling(engine));
}

/** Why a sampling canvas cannot run as chosen, or "" when it can (resource admission aside). */
export function canvasEdgeIssue(size, { maxEdge = DEFAULT_CANVAS_MAX_EDGE, engine = "" } = {}) {
  const width = Number(size?.width);
  const height = Number(size?.height);
  if (!Number.isFinite(width) || !Number.isFinite(height) || width < 64 || height < 64) return "画布尺寸至少为 64 × 64";
  if (width % 64 !== 0 || height % 64 !== 0) return "画布尺寸必须为 64 的倍数";
  const bound = normalizeCanvasMaxEdge(maxEdge).value;
  if (width > bound || height > bound) return `画布 ${width} × ${height} 超出当前画布上限 ${bound} × ${bound}；请重新选择画布或在设置中提高上限`;
  const ceiling = engineCanvasCeiling(engine);
  if (engine && (width > ceiling || height > ceiling)) return `${engine} 的画布边长不能超过 ${ceiling}`;
  return "";
}

/**
 * Which chosen canvases a lower bound would exclude, so lowering it can be refused with a reason
 * instead of cutting those canvases down.
 */
export function canvasBoundConflicts(nextBound, canvases) {
  const bound = normalizeCanvasMaxEdge(nextBound).value;
  return (Array.isArray(canvases) ? canvases : []).filter((entry) => entry && (Number(entry.width) > bound || Number(entry.height) > bound));
}
