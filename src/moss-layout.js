// MOSS-Image coordinate layouts as the browser edits them (QPlan2/MOSS/P2).
//
// A MOSS layout is prompt text: one line of JSON with a `summary` and a flat `items` list whose boxes are
// pixels around the canvas centre (`abs_centralized`, `[y1, x1, y2, x2]`). The editor keeps the official
// readable draft instead — top-left `[x1, y1, x2, y2]` on the generation canvas — and compiles it here
// exactly as `backend/moss_layout.py` does (and that module, OpenMOSS's own `compile_prompt.py`), so the
// request's `prompt` is byte-for-byte what the server recompiles. `scripts/moss-layout-parity.mjs` writes
// the fixture both suites assert.
//
// Editor-only facts (ids, hidden, locked) live on the items of the editor state and never reach the draft.

export const MOSS_LAYOUT_VERSION = 1;
export const MOSS_LAYOUT_MAX_ITEMS = 256;
export const MOSS_LAYOUT_MAX_PREFIX = 4000;
export const MOSS_LAYOUT_MAX_TEXT = 4000;
export const MOSS_LAYOUT_MAX_TOKENS = 4096;
// The meter turns amber from here: a few long captions away from the window.
export const MOSS_LAYOUT_TOKEN_WARNING = 3600;
export const MOSS_LAYOUT_MIN_BOX = 16;
export const MOSS_LAYOUT_GRID = 16;
export const MOSS_LAYOUT_LABELS = Object.freeze(["object", "character", "text", "panel"]);
export const MOSS_LAYOUT_LABEL_NAMES = Object.freeze({ object: "物体", character: "人物", text: "文字", panel: "画面" });
export const MOSS_LAYOUT_CANVAS_KINDS = Object.freeze(["single", "multi_panel", "collage"]);
export const MOSS_LAYOUT_CANVAS_NAMES = Object.freeze({ single: "单图", multi_panel: "多格", collage: "拼贴" });
export const MOSS_LAYOUT_HANDLES = Object.freeze(["tl", "top", "tr", "right", "br", "bottom", "bl", "left"]);
// Which edges a handle moves: -1 the left/top edge, 1 the right/bottom one, 0 neither on that axis.
export const MOSS_LAYOUT_HANDLE_AXES = Object.freeze({
  tl: { x: -1, y: -1 }, top: { x: 0, y: -1 }, tr: { x: 1, y: -1 }, right: { x: 1, y: 0 },
  br: { x: 1, y: 1 }, bottom: { x: 0, y: 1 }, bl: { x: -1, y: 1 }, left: { x: -1, y: 0 },
});

const ROOT_FIELDS = ["width", "height", "bbox_coord_space", "bbox_format", "summary", "items"];
const SUMMARY_FIELDS = ["caption", "meta", "canvas", "palette"];
const ITEM_FIELDS = ["label", "caption", "bbox", "text", "tags", "palette"];
// Python's `str.isspace` set, so "blank" and `.strip()` mean here what they mean on the server.
const PY_SPACE = "\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const PY_BLANK = new RegExp(`^[${PY_SPACE}]*$`);
const PY_STRIP = new RegExp(`^[${PY_SPACE}]+|[${PY_SPACE}]+$`, "g");

const isPlainObject = (value) => Boolean(value) && typeof value === "object" && !Array.isArray(value);
const finiteNumber = (value) => typeof value === "number" && Number.isFinite(value);
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

export function pyStrip(text) {
  return String(text ?? "").replace(PY_STRIP, "");
}

/** Why a string is not acceptable layout text, or "" when it is. */
export function layoutTextProblem(value) {
  if (typeof value !== "string" || PY_BLANK.test(value)) return "不能为空";
  if (value.length > MOSS_LAYOUT_MAX_TEXT) return `不能超过 ${MOSS_LAYOUT_MAX_TEXT} 个字符`;
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(index + 1);
      if (next >= 0xdc00 && next <= 0xdfff) { index += 1; continue; }
      return "含有不成对的代理字符";
    }
    if (code >= 0xdc00 && code <= 0xdfff) return "含有不成对的代理字符";
    if (code < 0x20 && code !== 0x0a && code !== 0x09) return "含有控制字符（只允许换行与制表符）";
  }
  return "";
}

// Python's `round(value, 9)` before floor/ceil, so an edge a hair past an integer encloses the same way.
const round9 = (value) => Number(value.toFixed(9));

function enclose([x1, y1, x2, y2], width, height) {
  const halfWidth = Math.floor(width / 2), halfHeight = Math.floor(height / 2);
  return [
    Math.floor(round9(y1)) - halfHeight,
    Math.floor(round9(x1)) - halfWidth,
    Math.ceil(round9(y2)) - halfHeight,
    Math.ceil(round9(x2)) - halfWidth,
  ];
}

/** The official centred `[y1, x1, y2, x2]` for a top-left `[x1, y1, x2, y2]` box. */
export function centredBox(box, width, height) {
  return enclose(box, width, height);
}

function boxProblem(box, width, height) {
  if (!Array.isArray(box) || box.length !== 4) return "必须是 [x1, y1, x2, y2]";
  if (!box.every(finiteNumber)) return "必须是四个有限数值";
  const [x1, y1, x2, y2] = box;
  if (!(x1 < x2 && y1 < y2)) return "宽和高必须大于 0";
  if (width === null || height === null) return "";
  if (!(x1 >= 0 && x2 <= width && y1 >= 0 && y2 <= height)) return `超出 ${width}×${height} 画布`;
  const result = enclose(box, width, height);
  if (!(result[0] < result[2] && result[1] < result[3])) return "太小，导出精度下为空";
  return "";
}

function stringListIssues(value, path, add) {
  if (!Array.isArray(value)) { add(path, "必须是数组"); return; }
  value.forEach((entry, index) => {
    const problem = layoutTextProblem(entry);
    if (problem) add(`${path}[${index}]`, problem);
  });
}

/** Every problem in an official draft as `{path, message}`; empty when the compiler accepts it. */
export function mossLayoutDraftIssues(draft) {
  const issues = [];
  const add = (path, message) => issues.push({ path, message });
  if (!isPlainObject(draft)) { add("", "布局草稿必须是对象"); return issues; }
  const unknown = Object.keys(draft).filter((key) => !ROOT_FIELDS.includes(key)).sort();
  const missing = ROOT_FIELDS.filter((key) => !(key in draft)).sort();
  if (unknown.length) add("", `未知字段 ${unknown.join(", ")}`);
  if (missing.length) add("", `缺少字段 ${missing.join(", ")}`);
  let canvasOk = "width" in draft && "height" in draft;
  for (const name of ["width", "height"]) {
    const value = draft[name];
    if (name in draft && !(Number.isInteger(value) && value > 0 && value % 16 === 0)) {
      add(name, "必须是 16 的正整数倍");
      canvasOk = false;
    }
  }
  if ("bbox_coord_space" in draft && draft.bbox_coord_space !== "abs") add("bbox_coord_space", "必须是 \"abs\"（左上角像素）");
  if ("bbox_format" in draft && draft.bbox_format !== "xyxy") add("bbox_format", "必须是 \"xyxy\"");
  if ("summary" in draft) {
    const summary = draft.summary;
    if (!isPlainObject(summary)) add("summary", "必须是对象");
    else {
      const extra = Object.keys(summary).filter((key) => !SUMMARY_FIELDS.includes(key)).sort();
      if (extra.length) add("summary", `未知字段 ${extra.join(", ")}`);
      if (!("caption" in summary)) add("summary.caption", "全局画面描述必填");
      else {
        const problem = layoutTextProblem(summary.caption);
        if (problem) add("summary.caption", problem);
      }
      if ("canvas" in summary && !MOSS_LAYOUT_CANVAS_KINDS.includes(summary.canvas)) add("summary.canvas", "必须是 single、multi_panel 或 collage");
      for (const key of ["meta", "palette"]) if (key in summary) stringListIssues(summary[key], `summary.${key}`, add);
    }
  }
  if ("items" in draft) {
    const items = draft.items;
    if (!Array.isArray(items) || !items.length) add("items", "至少需要一个框");
    else if (items.length > MOSS_LAYOUT_MAX_ITEMS) add("items", `最多 ${MOSS_LAYOUT_MAX_ITEMS} 个框`);
    else items.forEach((item, index) => itemIssues(item, `items[${index}]`, canvasOk ? draft.width : null, canvasOk ? draft.height : null, add));
  }
  return issues;
}

function itemIssues(item, path, width, height, add) {
  if (!isPlainObject(item)) { add(path, "必须是对象"); return; }
  const unknown = Object.keys(item).filter((key) => !ITEM_FIELDS.includes(key)).sort();
  const missing = ["bbox", "caption", "label"].filter((key) => !(key in item));
  if (unknown.length) add(path, `未知字段 ${unknown.join(", ")}`);
  if (missing.length) add(path, `缺少字段 ${missing.join(", ")}`);
  if ("label" in item && !MOSS_LAYOUT_LABELS.includes(item.label)) add(`${path}.label`, `必须是 ${MOSS_LAYOUT_LABELS.join("、")} 之一（当前 ${JSON.stringify(item.label)}）`);
  if ("caption" in item) {
    const problem = layoutTextProblem(item.caption);
    if (problem) add(`${path}.caption`, problem);
  }
  if (item.label === "text" && !("text" in item)) add(`${path}.text`, "文字框必须填写要渲染的文字");
  if ("text" in item) {
    const problem = layoutTextProblem(item.text);
    if (problem) add(`${path}.text`, problem);
  }
  for (const key of ["tags", "palette"]) if (key in item) stringListIssues(item[key], `${path}.${key}`, add);
  if ("bbox" in item) {
    const problem = boxProblem(item.bbox, width, height);
    if (problem) add(`${path}.bbox`, problem);
  }
}

export class MossLayoutError extends Error {
  constructor(issue) {
    super(issue.path ? `${issue.path}: ${issue.message}` : issue.message);
    this.path = issue.path;
  }
}

/** The model-facing JSON for a valid draft — what the official tool prints for it. */
export function mossLayoutJson(draft) {
  const issues = mossLayoutDraftIssues(draft);
  if (issues.length) throw new MossLayoutError(issues[0]);
  const items = draft.items.map((item) => Object.fromEntries(Object.entries(item).map(([key, value]) => [key, key === "bbox" ? enclose(value, draft.width, draft.height) : value])));
  return JSON.stringify({ summary: draft.summary, items, bbox_coord_space: "abs_centralized", bbox_format: "yxyx" });
}

export function joinMossLayoutPrefix(prefix, body) {
  const head = pyStrip(prefix);
  return head ? `${head}\n${body}` : body;
}

/** The prompt for a draft at its own canvas: the layout JSON, after the prefix line if any. */
export function compileMossLayoutDraft(draft, prefix = "") {
  return joinMossLayoutPrefix(prefix, mossLayoutJson(draft));
}

/** The same draft on a `width × height` canvas, boxes scaled per axis in top-left space. */
export function scaleMossLayoutDraft(draft, width, height) {
  const issues = mossLayoutDraftIssues(draft);
  if (issues.length) throw new MossLayoutError(issues[0]);
  if (draft.width === width && draft.height === height) return draft;
  const scaled = ([x1, y1, x2, y2]) => [
    clamp(x1 * width / draft.width, 0, width), clamp(y1 * height / draft.height, 0, height),
    clamp(x2 * width / draft.width, 0, width), clamp(y2 * height / draft.height, 0, height),
  ];
  const replaced = {
    width, height,
    items: draft.items.map((item) => Object.fromEntries(Object.entries(item).map(([key, value]) => [key, key === "bbox" ? scaled(value) : value]))),
  };
  return Object.fromEntries(Object.entries(draft).map(([key, value]) => [key, key in replaced ? replaced[key] : value]));
}

// ---------------------------------------------------------------------------------------------------
// Editor state: what the workspace stores and the Studio edits.

let itemSequence = 0;
/** A local id; `crypto.randomUUID` is unavailable on plain-HTTP hosts, and ids never leave the page. */
export function newMossLayoutItemId() {
  itemSequence += 1;
  return `box-${Date.now().toString(36)}-${itemSequence.toString(36)}`;
}

const cleanStrings = (value) => (Array.isArray(value) ? value : [])
  .filter((entry) => typeof entry === "string")
  .map((entry) => entry.slice(0, MOSS_LAYOUT_MAX_TEXT))
  .slice(0, 64);
const cleanText = (value) => typeof value === "string" ? value.replace(/\r\n?/g, "\n").slice(0, MOSS_LAYOUT_MAX_TEXT) : "";

function normalizeItem(value, width, height) {
  const source = isPlainObject(value) ? value : {};
  const box = Array.isArray(source.box) && source.box.length === 4 && source.box.every(finiteNumber)
    ? source.box
    : [0, 0, Math.min(width, 256), Math.min(height, 256)];
  return {
    id: typeof source.id === "string" && source.id ? source.id.slice(0, 80) : newMossLayoutItemId(),
    label: MOSS_LAYOUT_LABELS.includes(source.label) ? source.label : "object",
    caption: cleanText(source.caption),
    text: cleanText(source.text),
    tags: cleanStrings(source.tags),
    palette: cleanStrings(source.palette),
    box: clampBox(box, width, height, 1),
    hidden: source.hidden === true,
    locked: source.locked === true,
  };
}

export function emptyMossLayout(width = 1024, height = 1024) {
  return { enabled: false, width, height, summary: { caption: "", meta: [], canvas: "single", palette: [] }, items: [] };
}

/** A stored or restored layout state, repaired field by field; `size` is the canvas it must sit on. */
export function normalizeMossLayout(value, size = null) {
  const source = isPlainObject(value) ? value : {};
  const width = Number.isInteger(source.width) && source.width > 0 ? source.width : size?.width ?? 1024;
  const height = Number.isInteger(source.height) && source.height > 0 ? source.height : size?.height ?? 1024;
  const summary = isPlainObject(source.summary) ? source.summary : {};
  const layout = {
    enabled: source.enabled === true,
    width, height,
    summary: {
      caption: cleanText(summary.caption),
      meta: cleanStrings(summary.meta),
      canvas: MOSS_LAYOUT_CANVAS_KINDS.includes(summary.canvas) ? summary.canvas : "",
      palette: cleanStrings(summary.palette),
    },
    items: (Array.isArray(source.items) ? source.items : []).slice(0, MOSS_LAYOUT_MAX_ITEMS).map((item) => normalizeItem(item, width, height)),
  };
  return size && (size.width !== width || size.height !== height) ? rescaleMossLayout(layout, size.width, size.height) : layout;
}

const nonEmpty = (list) => list.filter((entry) => !PY_BLANK.test(entry));

/** The official draft for the visible items, keys in the order the README and examples use. */
export function mossLayoutDraft(layout) {
  const summary = {};
  const meta = nonEmpty(layout.summary.meta), palette = nonEmpty(layout.summary.palette);
  if (meta.length) summary.meta = meta;
  if (layout.summary.canvas) summary.canvas = layout.summary.canvas;
  summary.caption = layout.summary.caption;
  if (palette.length) summary.palette = palette;
  const items = layout.items.filter((item) => !item.hidden).map((item) => {
    const entry = { label: item.label, caption: item.caption };
    if (item.label === "text") entry.text = item.text;
    const tags = nonEmpty(item.tags), colours = nonEmpty(item.palette);
    if (tags.length) entry.tags = tags;
    if (colours.length) entry.palette = colours;
    entry.bbox = [...item.box];
    return entry;
  });
  return { width: layout.width, height: layout.height, bbox_coord_space: "abs", bbox_format: "xyxy", summary, items };
}

/** Issues of the layout's draft, each also naming the editor item it belongs to. */
export function mossLayoutIssues(layout) {
  const visible = layout.items.filter((item) => !item.hidden);
  return mossLayoutDraftIssues(mossLayoutDraft(layout)).map((issue) => {
    const match = /^items\[(\d+)\]/.exec(issue.path);
    return match ? { ...issue, itemId: visible[Number(match[1])]?.id ?? null } : { ...issue, itemId: null };
  });
}

/** `{prompt, draft, issues}` for the current layout; `prompt` is null while the draft is invalid. */
export function compileMossLayout(layout, prefix = "") {
  const draft = mossLayoutDraft(layout);
  const issues = mossLayoutIssues(layout);
  return { draft, issues, prompt: issues.length ? null : compileMossLayoutDraft(draft, prefix) };
}

/** The wire block (`GenerateInput.moss_layout`, version 1). */
export function mossLayoutPayload(layout, prefix = "") {
  return { version: MOSS_LAYOUT_VERSION, prefix: String(prefix ?? ""), draft: mossLayoutDraft(layout) };
}

/** Editor state from an official readable draft (validated or not), on the draft's own canvas. */
export function mossLayoutFromDraft(draft, { enabled = true } = {}) {
  const summary = isPlainObject(draft?.summary) ? draft.summary : {};
  return normalizeMossLayout({
    enabled,
    width: draft?.width, height: draft?.height,
    summary,
    items: (Array.isArray(draft?.items) ? draft.items : []).map((item) => ({ ...item, box: item?.bbox })),
  });
}

/** The PNG record's `moss_layout` back as editor state, or null when it is not a version-1 record. */
export function mossLayoutFromRecord(record) {
  if (!isPlainObject(record) || record.version !== MOSS_LAYOUT_VERSION || !isPlainObject(record.draft)) return null;
  return { layout: mossLayoutFromDraft(record.draft), prefix: typeof record.prefix === "string" ? record.prefix : "" };
}

// ---------------------------------------------------------------------------------------------------
// Geometry.

/** A box inside the canvas with at least `minimum` pixels per side, edges ordered. */
export function clampBox(box, width, height, minimum = MOSS_LAYOUT_MIN_BOX) {
  let [x1, y1, x2, y2] = box.map(Number);
  if (x2 < x1) [x1, x2] = [x2, x1];
  if (y2 < y1) [y1, y2] = [y2, y1];
  const side = (low, high, limit) => {
    const size = Math.min(limit, Math.max(Math.min(minimum, limit), high - low));
    const start = clamp(low, 0, limit - size);
    return [start, start + size];
  };
  [x1, x2] = side(Math.round(x1), Math.round(x2), width);
  [y1, y2] = side(Math.round(y1), Math.round(y2), height);
  return [x1, y1, x2, y2];
}

/** A box moved by `(dx, dy)`, kept whole inside the canvas. */
export function moveBox(box, dx, dy, width, height) {
  const [x1, y1, x2, y2] = box;
  const w = x2 - x1, h = y2 - y1;
  const left = clamp(Math.round(x1 + dx), 0, width - w), top = clamp(Math.round(y1 + dy), 0, height - h);
  return [left, top, left + w, top + h];
}

/** A box resized from one handle; `keepRatio` holds the start box's aspect. */
export function resizeBox(box, handle, dx, dy, width, height, { keepRatio = false, minimum = MOSS_LAYOUT_MIN_BOX } = {}) {
  const axes = MOSS_LAYOUT_HANDLE_AXES[handle];
  if (!axes) return box;
  let [x1, y1, x2, y2] = box;
  const ratio = (x2 - x1) / Math.max(1, y2 - y1);
  if (axes.x < 0) x1 = clamp(x1 + dx, 0, x2 - minimum);
  if (axes.x > 0) x2 = clamp(x2 + dx, x1 + minimum, width);
  if (axes.y < 0) y1 = clamp(y1 + dy, 0, y2 - minimum);
  if (axes.y > 0) y2 = clamp(y2 + dy, y1 + minimum, height);
  if (keepRatio && axes.x && axes.y) {
    // The axis dragged further leads; the other follows, then both shrink to the room the fixed corner has.
    let w = x2 - x1, h = y2 - y1;
    if (w / Math.max(1, h) > ratio) h = w / ratio; else w = h * ratio;
    const fit = Math.min(1, (axes.x < 0 ? x2 : width - x1) / w, (axes.y < 0 ? y2 : height - y1) / h);
    w *= fit; h *= fit;
    if (axes.x < 0) x1 = x2 - w; else x2 = x1 + w;
    if (axes.y < 0) y1 = y2 - h; else y2 = y1 + h;
  }
  return clampBox([x1, y1, x2, y2], width, height, minimum);
}

/** The box spanned by a drag from `start` to `end` (canvas pixels). */
export function boxFromDrag(start, end, width, height) {
  return clampBox([Math.min(start.x, end.x), Math.min(start.y, end.y), Math.max(start.x, end.x), Math.max(start.y, end.y)], width, height);
}

/** The value snapped to the nearest target within `threshold`, and the target used (or null). */
export function snapValue(value, targets, threshold) {
  let best = null;
  for (const target of targets) {
    const distance = Math.abs(value - target);
    if (distance <= threshold && (best === null || distance < Math.abs(value - best))) best = target;
  }
  return best === null ? { value, target: null } : { value: best, target: best };
}

/** The shift that brings the nearest of `edges` onto a target within `threshold`, with that target; or null. */
export function snapDelta(edges, targets, threshold) {
  let best = null;
  for (const edge of edges) {
    for (const target of targets) {
      const delta = target - edge;
      if (Math.abs(delta) <= threshold && (best === null || Math.abs(delta) < Math.abs(best.delta))) best = { delta, target };
    }
  }
  return best;
}

/** Snap targets on one axis: the grid, the canvas edges and centre, and the other boxes' edges. */
export function snapTargets(layout, axis, exclude, { grid = MOSS_LAYOUT_GRID, useGrid = true } = {}) {
  const excluded = new Set([].concat(exclude ?? []));
  const limit = axis === "x" ? layout.width : layout.height;
  const targets = new Set([0, limit, Math.round(limit / 2)]);
  if (useGrid) for (let value = 0; value <= limit; value += grid) targets.add(value);
  for (const item of layout.items) {
    if (excluded.has(item.id) || item.hidden) continue;
    const [x1, y1, x2, y2] = item.box;
    for (const value of axis === "x" ? [x1, x2, Math.round((x1 + x2) / 2)] : [y1, y2, Math.round((y1 + y2) / 2)]) targets.add(value);
  }
  return [...targets];
}

/** The layout on another canvas: every box scaled per axis and rounded, nothing else changed. */
export function rescaleMossLayout(layout, width, height) {
  if (layout.width === width && layout.height === height) return layout;
  const sx = width / layout.width, sy = height / layout.height;
  return {
    ...layout, width, height,
    items: layout.items.map((item) => ({
      ...item,
      box: clampBox([item.box[0] * sx, item.box[1] * sy, item.box[2] * sx, item.box[3] * sy], width, height, 1),
    })),
  };
}

/** Align/distribute the given items; `mode` is left/hcenter/right/top/vcenter/bottom/hspace/vspace. */
export function alignMossLayoutItems(layout, ids, mode) {
  const chosen = layout.items.filter((item) => ids.includes(item.id) && !item.locked);
  if (!chosen.length) return layout;
  const canvasOnly = chosen.length === 1;
  const bounds = canvasOnly
    ? [0, 0, layout.width, layout.height]
    : [Math.min(...chosen.map((i) => i.box[0])), Math.min(...chosen.map((i) => i.box[1])), Math.max(...chosen.map((i) => i.box[2])), Math.max(...chosen.map((i) => i.box[3]))];
  const moved = new Map();
  if (mode === "hspace" || mode === "vspace") {
    if (chosen.length < 3) return layout;
    const horizontal = mode === "hspace";
    const sorted = [...chosen].sort((a, b) => horizontal ? a.box[0] - b.box[0] : a.box[1] - b.box[1]);
    const total = sorted.reduce((sum, item) => sum + (horizontal ? item.box[2] - item.box[0] : item.box[3] - item.box[1]), 0);
    const span = horizontal ? bounds[2] - bounds[0] : bounds[3] - bounds[1];
    const gap = (span - total) / (sorted.length - 1);
    let cursor = horizontal ? bounds[0] : bounds[1];
    for (const item of sorted) {
      const size = horizontal ? item.box[2] - item.box[0] : item.box[3] - item.box[1];
      const start = Math.round(cursor);
      moved.set(item.id, horizontal ? [start, item.box[1], start + size, item.box[3]] : [item.box[0], start, item.box[2], start + size]);
      cursor += size + gap;
    }
  } else {
    for (const item of chosen) {
      const [x1, y1, x2, y2] = item.box;
      const w = x2 - x1, h = y2 - y1;
      let left = x1, top = y1;
      if (mode === "left") left = bounds[0];
      if (mode === "right") left = bounds[2] - w;
      if (mode === "hcenter") left = Math.round((bounds[0] + bounds[2] - w) / 2);
      if (mode === "top") top = bounds[1];
      if (mode === "bottom") top = bounds[3] - h;
      if (mode === "vcenter") top = Math.round((bounds[1] + bounds[3] - h) / 2);
      moved.set(item.id, [left, top, left + w, top + h]);
    }
  }
  return { ...layout, items: layout.items.map((item) => moved.has(item.id) ? { ...item, box: clampBox(moved.get(item.id), layout.width, layout.height, 1) } : item) };
}

/** A new editor item; without a box it sits in the middle third of the canvas. */
export function newMossLayoutItem(layout, label = "object", box = null) {
  const width = layout.width, height = layout.height;
  const fallback = [Math.round(width / 3), Math.round(height / 3), Math.round(width * 2 / 3), Math.round(height * 2 / 3)];
  return normalizeItem({ label, box: box ?? (label === "panel" ? [0, 0, width, height] : fallback) }, width, height);
}

/** Copies of the given items, offset and inserted after the last of them; returns the copies' ids too. */
export function duplicateMossLayoutItems(layout, ids, offset = MOSS_LAYOUT_GRID) {
  const chosen = layout.items.filter((item) => ids.includes(item.id));
  if (!chosen.length) return { layout, ids: [] };
  const copies = chosen.map((item) => ({
    ...item, id: newMossLayoutItemId(), locked: false, hidden: false,
    tags: [...item.tags], palette: [...item.palette],
    box: moveBox(item.box, offset, offset, layout.width, layout.height),
  }));
  const last = Math.max(...chosen.map((item) => layout.items.indexOf(item)));
  const items = [...layout.items.slice(0, last + 1), ...copies, ...layout.items.slice(last + 1)].slice(0, MOSS_LAYOUT_MAX_ITEMS);
  return { layout: { ...layout, items }, ids: copies.filter((copy) => items.includes(copy)).map((copy) => copy.id) };
}

/** The item moved to `index` in the list (list order is JSON order, which is reading order). */
export function moveMossLayoutItem(layout, id, index) {
  const from = layout.items.findIndex((item) => item.id === id);
  if (from < 0) return layout;
  const to = clamp(Math.round(index), 0, layout.items.length - 1);
  if (to === from) return layout;
  const items = [...layout.items];
  const [item] = items.splice(from, 1);
  items.splice(to, 0, item);
  return { ...layout, items };
}

/** Visible items under a canvas point, topmost (last in the list) first. */
export function mossLayoutItemsAt(layout, x, y) {
  return layout.items.filter((item) => !item.hidden && x >= item.box[0] && x <= item.box[2] && y >= item.box[1] && y <= item.box[3]).reverse();
}

/** One item with `patch` merged in. */
export function patchMossLayoutItem(layout, id, patch) {
  return { ...layout, items: layout.items.map((item) => item.id === id ? { ...item, ...patch } : item) };
}

// ---------------------------------------------------------------------------------------------------
// Import / templates.

/**
 * Editor state from text the user pastes or loads: a readable draft, or an official compiled layout
 * (`abs_centralized` / `yxyx`, optionally after a prefix line), placed on the current canvas.
 * Returns `{layout, prefix, note}` or `{error}`.
 */
export function parseMossLayoutImport(text, width, height) {
  const source = String(text ?? "").trim();
  const start = source.indexOf("{");
  if (start < 0) return { error: "没有找到 JSON 对象。粘贴可读草稿或官方编译后的布局 JSON。" };
  const prefix = source.slice(0, start).trim();
  let value;
  try {
    value = JSON.parse(source.slice(start));
  } catch {
    return { error: "JSON 无法解析：请检查括号、引号与逗号。" };
  }
  if (!isPlainObject(value)) return { error: "布局必须是 JSON 对象。" };
  if ("elements" in value || "scene_type" in value || "gen_brief" in value) {
    return { error: "这是 MOSS 的场景图（scene graph）格式，布局编辑器只编辑 summary/items 格式。请在文本模式中直接使用它。" };
  }
  if (value.bbox_coord_space === "abs" || ("width" in value && "height" in value)) {
    const layout = { ...mossLayoutFromDraft(value), enabled: true };
    const issues = mossLayoutIssues(layout);
    const placed = rescaleMossLayout(layout, width, height);
    const resized = layout.width !== width || layout.height !== height;
    return { layout: placed, prefix, note: [resized ? `草稿画布 ${layout.width}×${layout.height} 已按比例缩放到 ${width}×${height}` : "", issues.length ? `${issues.length} 处待修正` : ""].filter(Boolean).join("；") };
  }
  if (value.bbox_coord_space === "abs_centralized" || Array.isArray(value.items)) {
    if (value.bbox_format !== undefined && value.bbox_format !== "yxyx") return { error: `不支持的 bbox_format ${JSON.stringify(value.bbox_format)}；官方格式为 yxyx。` };
    if (!Array.isArray(value.items)) return { error: "缺少 items 数组。" };
    const halfWidth = Math.floor(width / 2), halfHeight = Math.floor(height / 2);
    let outside = 0;
    const items = value.items.map((item) => {
      const box = Array.isArray(item?.bbox) && item.bbox.length === 4 && item.bbox.every(finiteNumber)
        ? [item.bbox[1] + halfWidth, item.bbox[0] + halfHeight, item.bbox[3] + halfWidth, item.bbox[2] + halfHeight]
        : null;
      if (box && (box[0] < 0 || box[1] < 0 || box[2] > width || box[3] > height)) outside += 1;
      const { bbox: _bbox, ...rest } = isPlainObject(item) ? item : {};
      return { ...rest, box: box ?? undefined };
    });
    const layout = normalizeMossLayout({ enabled: true, width, height, summary: isPlainObject(value.summary) ? value.summary : {}, items });
    return {
      layout, prefix,
      note: [prefix ? "前缀文字未导入（前缀来自 LoRA 组与预设）" : "", outside ? `${outside} 个框超出当前 ${width}×${height} 画布，已收回画布内——官方坐标按生成尺寸计算，请确认尺寸` : ""].filter(Boolean).join("；"),
    };
  }
  return { error: "无法识别：需要 summary/items 布局。" };
}

/** The README example (OpenMOSS/MOSS-Image, 1024×1024), as a readable draft. */
export const MOSS_LAYOUT_README_DRAFT = Object.freeze({
  width: 1024, height: 1024, bbox_coord_space: "abs", bbox_format: "xyxy",
  summary: { meta: ["poster"], canvas: "single", caption: "A minimalist travel poster with a cream background, a blue mountain illustration, and the title EXPLORE at the top." },
  items: [
    { label: "text", caption: "Large dark-blue uppercase title centered at the top.", text: "EXPLORE", bbox: [128, 96, 896, 224] },
    { label: "object", caption: "A flat blue mountain illustration in the lower center.", bbox: [224, 320, 800, 896] },
  ],
});

export const MOSS_LAYOUT_TEMPLATES = Object.freeze([
  { id: "blank", name: "空白画布", description: "只保留画布尺寸与画布类型" },
  { id: "readme-poster", name: "官方示例 · 旅行海报", description: "README 中的 EXPLORE 海报（1024×1024，按当前尺寸缩放）", draft: MOSS_LAYOUT_README_DRAFT },
]);

export function mossLayoutFromTemplate(id, width, height) {
  const template = MOSS_LAYOUT_TEMPLATES.find((entry) => entry.id === id);
  if (!template) return null;
  if (!template.draft) return { ...emptyMossLayout(width, height), enabled: true };
  return rescaleMossLayout({ ...mossLayoutFromDraft(template.draft), enabled: true }, width, height);
}

/** Item counts by label and the issue count, for the prompt card. */
export function mossLayoutStats(layout) {
  const counts = Object.fromEntries(MOSS_LAYOUT_LABELS.map((label) => [label, 0]));
  for (const item of layout.items) if (!item.hidden) counts[item.label] += 1;
  return { counts, visible: layout.items.filter((item) => !item.hidden).length, total: layout.items.length };
}
