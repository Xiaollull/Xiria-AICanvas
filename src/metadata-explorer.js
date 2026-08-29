// Cutting a metadata payload into blocks a person can read.
//
// The reader's summary answers "what made this picture". This module answers the
// other question — "what is actually in the file" — and for a ComfyUI export the
// answer is a 40 KB API graph plus a 120 KB editor workflow. Printed as one
// `<pre>` that is a wall nobody scrolls, so everything here exists to cut a
// payload along the seams its author wrote it with: one block per `{}`.
//
// Nothing here is filesystem- or DOM-aware, so the same functions that feed the
// dialog are the ones the tests exercise.

import { comfyNodeLoras, parseLooseJson } from "./image-metadata.js";

// A grouping for the eye, not a claim about semantics: it exists so 67 nodes
// arrive as seven short lists instead of one long one. First match wins, so the
// order below is the priority order — `Image Save` is an output before it is an
// image, and a prompt editor whose class name ends in `...WithoutLora` is a
// prompt node before it is a LoRA one.
export const NODE_CATEGORIES = [
  { id: "sampling", label: "采样", match: /sampler|guider|sigmas|noise|seed/i },
  { id: "prompt", label: "提示词", match: /encode|prompt|text|string|conditioning|tag/i },
  { id: "lora", label: "LoRA", match: /lora|lycoris/i },
  { id: "model", label: "模型", match: /loader|checkpoint|unet/i },
  { id: "output", label: "输出", match: /save|preview|output|show|compare/i },
  { id: "latent", label: "画布", match: /latent|resolution|empty/i },
  { id: "image", label: "图像", match: /image|upscale|mask|detailer|inpaint|controlnet|scale|decode|vae/i },
  { id: "other", label: "其他", match: /(?:)/ },
];

const CATEGORY_FALLBACK = NODE_CATEGORIES[NODE_CATEGORIES.length - 1].id;
// Long enough to recognise a prompt, short enough that a card stays a card.
const FIELD_PREVIEW_LIMIT = 220;

export function nodeCategory(node) {
  // A stack plugin's class name says "prompt" as loudly as it says "lora", so
  // the widgets it carries are the more reliable signal than the name.
  if (comfyNodeLoras(node).length) return "lora";
  const type = String(node?.class_type || "");
  return NODE_CATEGORIES.find((category) => category.match.test(type))?.id || CATEGORY_FALLBACK;
}

// A1111 and its forks write one flat settings line, so an image that ran four
// ADetailer passes and a Hires pass arrives as thirty-odd cells whose names are
// nine tenths repeated prefix — `ADetailer denoising strength 2nd` next to
// `ADetailer denoising strength 3rd`. The prefix is the grouping: lifting it
// into a heading turns the wall back into the modules the producer meant, and
// leaves each cell reading as the one thing it actually says.
export const PARAMETER_BASE_GROUP = "base";
export const PARAMETER_GROUPS = [
  { id: "adetailer", label: "ADetailer", match: /^ADetailer\b/i },
  { id: "hires", label: "Hires.fix", match: /^Hires\b/i },
  { id: "controlnet", label: "ControlNet", match: /^ControlNet\b/i },
  { id: "refiner", label: "Refiner", match: /^Refiner\b/i },
];
// `2nd`, `3rd`, `4th` — how A1111 numbers a repeated unit. A bare trailing digit
// is not one: `Hires Module 1` is the widget's own name.
const PARAMETER_UNIT = /\s(\d+)(?:st|nd|rd|th)$/i;

export function groupParameters(list, baseLabel = "基础参数") {
  const groups = new Map();
  for (const item of Array.isArray(list) ? list : []) {
    const label = String(item?.label ?? "");
    const group = PARAMETER_GROUPS.find((entry) => entry.match.test(label));
    const unit = group ? Number(PARAMETER_UNIT.exec(label)?.[1] ?? 1) : 1;
    const id = group ? group.id : PARAMETER_BASE_GROUP;
    const key = `${id}:${unit}`;
    if (!groups.has(key)) groups.set(key, { id, key, unit, label: group ? group.label : baseLabel, items: [] });
    const short = group ? label.replace(PARAMETER_UNIT, "").replace(group.match, "").trim() : label;
    groups.get(key).items.push({ label: short || label, value: item?.value });
  }

  const order = [PARAMETER_BASE_GROUP, ...PARAMETER_GROUPS.map((entry) => entry.id)];
  const units = new Map();
  for (const group of groups.values()) units.set(group.id, (units.get(group.id) || 0) + 1);
  return [...groups.values()]
    .sort((first, second) => order.indexOf(first.id) - order.indexOf(second.id) || first.unit - second.unit)
    // A group that ran once is just itself; only a repeated one needs numbering.
    .map((group) => ({ ...group, label: units.get(group.id) > 1 ? `${group.label} · ${group.unit}` : group.label }));
}

export function isComfyGraph(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  return Object.values(value).some((node) => node && typeof node === "object" && typeof node.class_type === "string");
}

function isLink(value) {
  return Array.isArray(value) && value.length === 2 && typeof value[1] === "number";
}

function clip(text) {
  return text.length > FIELD_PREVIEW_LIMIT ? `${text.slice(0, FIELD_PREVIEW_LIMIT)}…` : text;
}

// One row of a node card. `kind` decides how it draws: a link is a jump to
// another block, a long string gets an expander, everything else is one line.
export function describeField(name, value) {
  if (isLink(value)) return { name, kind: "link", target: String(value[0]), slot: value[1], text: `#${value[0]} · ${value[1]}` };
  if (typeof value === "string") {
    return { name, kind: "text", text: value, preview: clip(value), truncated: value.length > FIELD_PREVIEW_LIMIT, chars: value.length };
  }
  if (value === null || value === undefined) return { name, kind: "value", text: "null" };
  if (typeof value === "object") {
    const json = JSON.stringify(value);
    return { name, kind: "text", text: json, preview: clip(json), truncated: json.length > FIELD_PREVIEW_LIMIT, chars: json.length };
  }
  return { name, kind: "value", text: String(value) };
}

// One block per node, in the order the file lists them. `roles` comes from the
// summary pass so a card can say "this is where the negative prompt came from",
// which is the one thing a plugin-built graph makes hard to see.
export function buildGraphModules(graph, roles = {}) {
  const modules = [];
  for (const [id, node] of Object.entries(graph || {})) {
    if (!node || typeof node !== "object" || typeof node.class_type !== "string") continue;
    modules.push({
      id,
      classType: node.class_type,
      title: typeof node._meta?.title === "string" && node._meta.title !== node.class_type ? node._meta.title : "",
      category: nodeCategory(node),
      roles: roles[id] || [],
      fields: Object.entries(node.inputs || {}).map(([name, value]) => describeField(name, value)),
    });
  }
  // Numeric where ComfyUI's ids are numeric, which is every graph it writes.
  return modules.sort((first, second) => String(first.id).localeCompare(String(second.id), "en", { numeric: true }));
}

export function countByCategory(modules) {
  const counts = new Map();
  for (const item of modules) counts.set(item.category, (counts.get(item.category) || 0) + 1);
  return counts;
}

export function matchesQuery(module, query) {
  const wanted = String(query || "").trim().toLowerCase();
  if (!wanted) return true;
  if (`#${module.id}`.includes(wanted)) return true;
  if (module.classType.toLowerCase().includes(wanted) || module.title.toLowerCase().includes(wanted)) return true;
  return module.fields.some((field) => field.name.toLowerCase().includes(wanted) || String(field.text).toLowerCase().includes(wanted));
}

// The generic half: a payload that is JSON but not a graph — the editor's own
// `workflow` chunk, a plugin's settings blob — still gets folded per `{}` rather
// than printed whole. Children are described on demand so a 120 KB workflow
// costs one row until somebody opens it.
export function jsonBranch(value) {
  if (Array.isArray(value)) return { kind: "array", size: value.length, entries: value.map((item, index) => [String(index), item]) };
  if (value && typeof value === "object") {
    const entries = Object.entries(value);
    return { kind: "object", size: entries.length, entries };
  }
  return { kind: "scalar", size: 0, entries: [] };
}

export function jsonSummary(value) {
  const branch = jsonBranch(value);
  if (branch.kind === "array") return `[ ${branch.size} 项 ]`;
  if (branch.kind === "object") return `{ ${branch.size} 字段 }`;
  if (typeof value === "string") return clip(JSON.stringify(value));
  return String(value);
}

// What a single PNG text chunk turns out to be. Detection is by shape: a
// producer is free to put a graph under any keyword, and several do.
export function describeChunk(keyword, value) {
  const text = typeof value === "string" ? value : String(value ?? "");
  const trimmed = text.trim();
  const json = trimmed.startsWith("{") || trimmed.startsWith("[") ? parseLooseJson(text) : null;
  const graph = isComfyGraph(json) ? json : null;
  return {
    keyword,
    text,
    chars: text.length,
    json: graph ? null : json,
    graph,
    kind: graph ? "graph" : json && typeof json === "object" ? "json" : "text",
  };
}

export function describeChunks(raw) {
  return Object.entries(raw || {}).map(([keyword, value]) => describeChunk(keyword, value));
}
