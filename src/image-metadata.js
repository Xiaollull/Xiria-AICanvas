// Reading generation metadata back out of a PNG somebody else's tool wrote.
//
// Three producers matter and they disagree about everything except the file
// format: this project writes one JSON blob under `parameters`, A1111/SD-WebUI
// writes a human-readable text block under the same key, and ComfyUI writes an
// entire node graph under `prompt`. So detection has to look at the *shape* of
// what it found, never at the key alone.
//
// This module is deliberately filesystem-free and takes already-decoded text
// chunks, so the same interpreters run over an uploaded file and over a batch
// directory. Chunk extraction lives in `scripts/png-text-chunks.mjs`.

export const METADATA_SOURCES = {
  xiria: "XiriaCanvas AI",
  comfyui: "ComfyUI",
  a1111: "SD-WebUI / A1111",
  novelai: "NovelAI",
  unknown: "未知来源",
};

export const IMAGE_INFO_STATUS = { ok: "ok", empty: "empty", unsupported: "unsupported" };

function textOf(value) {
  return typeof value === "string" ? value : "";
}

// Python's `json.dumps` spells non-finite floats `NaN`, `Infinity` and
// `-Infinity`, none of which JSON allows and none of which `JSON.parse` accepts.
// ComfyUI serialises whatever a custom node left in a widget, so one stray value
// would otherwise cost the entire graph. Rewriting them to `null` outside string
// literals loses nothing a reader could have used anyway.
const NON_FINITE_LITERALS = ["NaN", "-Infinity", "Infinity"];

export function parseLooseJson(text) {
  if (typeof text !== "string") return null;
  try {
    return JSON.parse(text);
  } catch {
    // Fall through to the repair pass; a genuinely broken payload fails twice.
  }
  let repaired = "";
  let index = 0;
  let quoted = false;
  while (index < text.length) {
    const character = text[index];
    if (quoted) {
      // A backslash escapes the next character, `\"` very much included.
      const width = character === "\\" ? 2 : 1;
      repaired += text.slice(index, index + width);
      if (character === '"') quoted = false;
      index += width;
      continue;
    }
    if (character === '"') {
      quoted = true;
      repaired += character;
      index += 1;
      continue;
    }
    // Outside a string, JSON has no bare words other than `true`/`false`/`null`,
    // so a match here can only be the literal we are looking for.
    const literal = NON_FINITE_LITERALS.find((candidate) => text.startsWith(candidate, index));
    if (literal) {
      repaired += "null";
      index += literal.length;
      continue;
    }
    repaired += character;
    index += 1;
  }
  try {
    return JSON.parse(repaired);
  } catch {
    return null;
  }
}

function parseJsonObject(value) {
  if (typeof value !== "string" || !value.trim().startsWith("{")) return null;
  const parsed = parseLooseJson(value);
  return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : null;
}

function pushParameter(list, label, value) {
  if (value === null || value === undefined || value === "") return;
  list.push({ label, value: String(value) });
}

// `<lora:name:0.8>` in a prompt is the only place A1111 records a LoRA weight;
// `Lora hashes` carries names but never strengths.
export function extractPromptLoras(prompt) {
  const loras = [];
  const pattern = /<(?:lora|lyco):([^:>]+)(?::(-?[\d.]+))?(?::(-?[\d.]+))?>/gi;
  let match = pattern.exec(textOf(prompt));
  while (match) {
    loras.push({ name: match[1].trim(), weight: match[2] === undefined ? 1 : Number(match[2]) });
    match = pattern.exec(textOf(prompt));
  }
  return loras;
}

// A1111 settings lines quote any value that itself contains commas, so a naive
// split shreds `Lora hashes: "a: 1, b: 2"` into nonsense. A JSON value — the
// `Civitai resources` block a published image carries — brings its own quotes
// *and* commas, so quote state alone is not enough either: its inner commas sit
// outside any quote pair and split the array into fragments that then read back
// as settings of their own. Bracket depth is what actually separates them.
export function splitSettingsLine(line) {
  const pairs = [];
  const text = textOf(line);
  let current = "";
  let quoted = false;
  let depth = 0;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (quoted) {
      current += character;
      if (character === "\\") { current += text[index + 1] ?? ""; index += 1; continue; }
      if (character === '"') quoted = false;
      continue;
    }
    if (character === '"') quoted = true;
    else if (character === "[" || character === "{") depth += 1;
    else if (character === "]" || character === "}") depth = Math.max(0, depth - 1);
    else if (character === "," && !depth) {
      pairs.push(current);
      current = "";
      continue;
    }
    current += character;
  }
  pairs.push(current);
  const settings = new Map();
  for (const pair of pairs) {
    const separator = pair.indexOf(":");
    if (separator <= 0) continue;
    const key = pair.slice(0, separator).trim();
    const value = pair.slice(separator + 1).trim().replace(/^"|"$/g, "");
    if (key) settings.set(key, value);
  }
  return settings;
}

function looksLikeSettingsLine(line) {
  // Two or more `Key: value` pairs on one line is what separates the trailing
  // settings block from a prompt that merely happens to contain a colon.
  return /^[A-Za-z][\w \-/()]*:\s*[^,]*(,\s*[A-Za-z][\w \-/()]*:)/.test(line.trim());
}

export function parseA1111(text) {
  const lines = textOf(text).split(/\r?\n/);
  let settingsIndex = -1;
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    if (looksLikeSettingsLine(lines[index])) { settingsIndex = index; break; }
  }
  const settings = settingsIndex >= 0 ? splitSettingsLine(lines[settingsIndex]) : new Map();
  const promptBlock = (settingsIndex >= 0 ? lines.slice(0, settingsIndex) : lines).join("\n");
  const negativeAt = promptBlock.search(/(^|\n)Negative prompt:/);
  const positive = (negativeAt >= 0 ? promptBlock.slice(0, negativeAt) : promptBlock).trim();
  const negative = negativeAt >= 0 ? promptBlock.slice(negativeAt).replace(/(^|\n)Negative prompt:/, "").trim() : "";

  const parameters = [];
  for (const [label, value] of settings) {
    // These four are rendered as their own fields or mined below; repeating a
    // JSON blob in the parameter grid is noise, not information.
    if (label === "Model" || label === "Lora hashes" || label === "TI hashes" || label === "Civitai resources") continue;
    pushParameter(parameters, label, value);
  }

  const loras = extractPromptLoras(positive);
  if (!loras.length && settings.has("Lora hashes")) {
    for (const entry of settings.get("Lora hashes").split(",")) {
      const name = entry.split(":")[0]?.trim();
      if (name) loras.push({ name, weight: null });
    }
  }
  if (!loras.length && settings.has("Civitai resources")) {
    // Civitai's publish block lists every resource the image used, base model
    // included, so only the rows that say they are LoRAs count.
    const resources = parseLooseJson(settings.get("Civitai resources"));
    for (const entry of Array.isArray(resources) ? resources : []) {
      if (!entry || typeof entry !== "object" || String(entry.type || "").toLowerCase() !== "lora") continue;
      const name = textOf(entry.modelName || entry.name);
      if (name) loras.push({ name, weight: Number.isFinite(Number(entry.weight)) ? Number(entry.weight) : null });
    }
  }

  const flags = [
    { id: "hires", label: "Hires.fix", enabled: settings.has("Hires upscale") || settings.has("Hires upscaler") || settings.has("Hires steps"), detail: [settings.get("Hires upscaler"), settings.get("Hires upscale") && `x${settings.get("Hires upscale")}`].filter(Boolean).join(" · ") },
    { id: "adetailer", label: "ADetailer", enabled: [...settings.keys()].some((key) => key.startsWith("ADetailer")), detail: settings.get("ADetailer model") || "" },
    { id: "refiner", label: "Refiner", enabled: settings.has("Refiner"), detail: settings.get("Refiner") || "" },
    { id: "controlnet", label: "ControlNet", enabled: [...settings.keys()].some((key) => key.startsWith("ControlNet")), detail: "" },
  ];

  return {
    source: "a1111",
    engine: "",
    checkpoint: settings.get("Model") || "",
    checkpointHash: settings.get("Model hash") || "",
    positive,
    negative,
    parameters,
    loras,
    flags,
  };
}

// ComfyUI's `prompt` chunk is the API-format graph: node id -> {class_type,
// inputs}. An input is either a literal or a `[nodeId, slot]` link, so reading
// a prompt means following the KSampler's own positive/negative wiring rather
// than guessing which CLIPTextEncode came first.
//
// Custom nodes make that much harder than it sounds, and a workflow assembled
// from them is the normal case rather than the exception. Three habits break
// naive readers, so everything below is written around them:
//
//   * a prompt-editor plugin keeps its text under whatever input name it likes.
//     WeiLin's editor calls it `positive` even on the node feeding the sampler's
//     *negative*, so the input name cannot be trusted to say which role it fills;
//   * a value that looks typed-in is often linked in from somewhere else — the
//     seed from a `Seed (rgthree)` node, the size from a resolution picker;
//   * a LoRA stack plugin keeps the whole stack inside one widget instead of one
//     `LoraLoader` per LoRA, and leaves disconnected stacks on the canvas.
const COMFY_CHECKPOINT_CLASSES = ["CheckpointLoaderSimple", "CheckpointLoader", "UNETLoader", "UnetLoaderGGUF", "ImageOnlyCheckpointLoader"];
const COMFY_SAMPLER_CLASSES = ["KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced"];
const COMFY_LORA_CLASSES = ["LoraLoader", "LoraLoaderModelOnly", "LoraLoaderTagsQuery"];
// Text widgets, most specific first. `positive`/`negative` come last because
// they are how prompt-editor plugins spell "the text this node holds", and
// reading them first would let a node's label outrank its real widget.
const COMFY_TEXT_KEYS = ["text", "string", "value", "prompt", "text_g", "populated_text", "positive", "negative"];
// Prompt editors expose one optional text inlet beside their own widget and
// concatenate the two before encoding. Joining them is the only reading that
// reproduces the string the sampler actually saw.
const COMFY_TEXT_PREFIX_KEY = "opt_text";
// Where a plugin keeps an entire stack. WeiLin writes a JSON array as text,
// LoraManager writes `{ __value__: [...] }` beside `<lora:name:weight>` tags.
const COMFY_LORA_STACK_KEYS = ["lora_str", "lora_stack", "loras"];

function comfyNodes(graph) {
  const nodes = [];
  for (const [id, node] of Object.entries(graph)) {
    if (node && typeof node === "object" && typeof node.class_type === "string") nodes.push({ id, ...node });
  }
  return nodes;
}

function isLink(value) {
  return Array.isArray(value) && value.length === 2 && typeof value[1] === "number";
}

// `role` decides which of two same-shaped widgets wins on a node that carries
// both, which is the only thing that keeps a negative prompt from reading back
// as the positive one.
function resolveComfyText(graph, value, role = "", depth = 0) {
  if (typeof value === "string") return value;
  if (!isLink(value) || depth > 6) return "";
  const node = graph[value[0]];
  if (!node || typeof node !== "object") return "";
  const keys = role ? [role, ...COMFY_TEXT_KEYS] : COMFY_TEXT_KEYS;
  for (const key of keys) {
    const own = node.inputs?.[key];
    const text = typeof own === "string" ? own : isLink(own) ? resolveComfyText(graph, own, role, depth + 1) : "";
    if (!text) continue;
    const prefix = resolveComfyText(graph, node.inputs?.[COMFY_TEXT_PREFIX_KEY], role, depth + 1);
    return prefix ? `${prefix.replace(/[,\s]+$/, "")}, ${text}` : text;
  }
  return "";
}

// A reroute or a display node forwards its input unchanged, so a value routed
// through one is a link further along. They are recognised by name rather than
// by "has exactly one link", which `GetImageSize` also satisfies while
// forwarding nothing a size could be read from.
const COMFY_PASSTHROUGH_CLASS = /reroute|passthrough|show\s*any|showanything|display|primitive/i;

// Following a link by *name* is what stops `width` from picking up `height` off
// a node that publishes both.
function resolveComfyValue(graph, value, name, depth = 0) {
  if (!isLink(value)) return value;
  if (depth > 6) return undefined;
  const node = graph[value[0]];
  const inputs = node?.inputs;
  if (!inputs || typeof inputs !== "object") return undefined;
  for (const key of [name, "value", "seed", "number", "int", "float"]) {
    if (key && inputs[key] !== undefined) return resolveComfyValue(graph, inputs[key], name, depth + 1);
  }
  if (COMFY_PASSTHROUGH_CLASS.test(node.class_type || "")) {
    for (const input of Object.values(inputs)) {
      if (isLink(input)) return resolveComfyValue(graph, input, name, depth + 1);
    }
  }
  return undefined;
}

// The size is whichever node in the sampler's latent chain declares one. A
// `GetImageSize` reading a live image declares none, and printing its link id
// as a size would be worse than printing nothing.
function findComfyLatentId(graph, value, depth = 0) {
  if (!isLink(value) || depth > 8) return "";
  const inputs = graph[value[0]]?.inputs;
  if (!inputs || typeof inputs !== "object") return "";
  if (inputs.width !== undefined && inputs.height !== undefined) return value[0];
  for (const key of ["samples", "latent", "latent_image"]) {
    const found = findComfyLatentId(graph, inputs[key], depth + 1);
    if (found) return found;
  }
  return "";
}

function linkTargetId(value) {
  return isLink(value) ? String(value[0]) : "";
}

// Every node the sampler's `model` input can reach. A workflow that has been
// iterated on keeps disconnected LoRA stacks lying around, and reporting those
// as "used" is worse than reporting none.
function comfyModelChain(graph, sampler) {
  const reached = new Set();
  const walk = (value, depth) => {
    if (!isLink(value) || depth > 24 || reached.has(value[0])) return;
    const node = graph[value[0]];
    if (!node || typeof node !== "object") return;
    reached.add(value[0]);
    for (const input of Object.values(node.inputs || {})) walk(input, depth + 1);
  };
  walk(sampler?.inputs?.model, 0);
  return reached;
}

// One row of somebody's stack. Every plugin spells the three things that matter
// — is it on, which file, how strong — differently, and none of them removes a
// row the user merely switched off.
function comfyStackEntry(entry) {
  if (!entry || typeof entry !== "object" || Array.isArray(entry)) return null;
  if (entry.active === false || entry.hidden === true || entry.on === false) return null;
  // The `lora` field carries the extension the model catalogue matches on;
  // `name` is the editor's own shorthand and is the last resort.
  const name = textOf(entry.lora || entry.display_name || entry.name || entry.lora_name);
  if (!name) return null;
  const weight = Number(entry.weight ?? entry.strength ?? entry.strength_model ?? 1);
  return { name, weight: Number.isFinite(weight) ? weight : null };
}

// rgthree's Power Lora Loader gives each row its own widget — `lora_1`,
// `lora_2`, … — instead of one widget holding a list, so the stack has to be
// gathered from the input names rather than read out of a single value.
const COMFY_LORA_ROW_KEY = /^lora[_ -]?(\d+)$/i;

function comfyRowLoras(inputs) {
  const rows = [];
  for (const [key, value] of Object.entries(inputs || {})) {
    const match = COMFY_LORA_ROW_KEY.exec(key);
    if (!match || isLink(value)) continue;
    const entry = comfyStackEntry(typeof value === "string" ? parseLooseJson(value) : value);
    if (entry) rows.push({ order: Number(match[1]), entry });
  }
  return rows.sort((first, second) => first.order - second.order).map((row) => row.entry);
}

function comfyStackLoras(inputs) {
  for (const key of COMFY_LORA_STACK_KEYS) {
    const raw = inputs?.[key];
    if (isLink(raw)) continue;
    const list = Array.isArray(raw) ? raw : Array.isArray(raw?.__value__) ? raw.__value__ : parseLooseJson(textOf(raw));
    if (!Array.isArray(list)) continue;
    const stack = list.map(comfyStackEntry).filter(Boolean);
    if (stack.length) return stack;
  }
  return comfyRowLoras(inputs);
}

export function comfyNodeLoras(node) {
  const inputs = node?.inputs || {};
  const stack = comfyStackLoras(inputs);
  if (stack.length) return stack;
  const name = textOf(inputs.lora_name);
  if (name) return [{ name, weight: Number(inputs.strength_model ?? inputs.strength ?? 1) }];
  // `<lora:name:weight>` tags only count on a node that says it loads LoRAs; the
  // same text inside a prompt is a tag the graph never acted on.
  return /lora/i.test(node?.class_type || "") ? extractPromptLoras(inputs.text) : [];
}

function dedupeLoras(list) {
  const seen = new Set();
  return list.filter((item) => {
    const key = item.name.split("\\").join("/").toLowerCase();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

// One resolution pass over a graph, shared by the summary and by the node
// explorer that shows which node each summary field was read out of. Doing it
// twice would let the two drift, and "where did this prompt come from" is the
// question a plugin-built workflow makes hardest to answer.
export function readComfyGraph(graph) {
  const nodes = comfyNodes(graph);
  const sampler = nodes.find((node) => COMFY_SAMPLER_CLASSES.includes(node.class_type));
  const chain = comfyModelChain(graph, sampler);
  // A loader is whatever names a checkpoint file, whichever pack it came from:
  // `Checkpoint Loader with Name (Image Saver)` loads the same weights as
  // `CheckpointLoaderSimple` and a class list can only ever chase the packs.
  const isCheckpoint = (node) => COMFY_CHECKPOINT_CLASSES.includes(node.class_type)
    || typeof node.inputs?.ckpt_name === "string"
    || typeof node.inputs?.unet_name === "string";
  const checkpointNode = nodes.find((node) => chain.has(node.id) && isCheckpoint(node)) || nodes.find(isCheckpoint);

  const latentId = findComfyLatentId(graph, sampler?.inputs?.latent_image)
    || nodes.find((node) => node.class_type?.startsWith("EmptyLatentImage") || node.class_type === "EmptySD3LatentImage")?.id
    || "";

  const loraSources = nodes.map((node) => ({ node, loras: comfyNodeLoras(node) })).filter((entry) => entry.loras.length);
  const wired = loraSources.filter((entry) => chain.has(entry.node.id));

  return {
    nodes,
    sampler,
    checkpointNode,
    latent: latentId ? graph[latentId] : null,
    latentId,
    loraSources: wired.length ? wired : loraSources,
    positiveId: linkTargetId(sampler?.inputs?.positive),
    negativeId: linkTargetId(sampler?.inputs?.negative),
  };
}

// Which summary field each node supplied, keyed by node id. The explorer badges
// nodes with this so a workflow whose prompt lives in a custom widget still says
// out loud where the reader found it.
export function comfyNodeRoles(graph) {
  const read = readComfyGraph(graph);
  const roles = {};
  const mark = (id, label) => {
    if (!id || !graph?.[id]) return;
    roles[id] = roles[id] ? [...roles[id], label] : [label];
  };
  mark(read.sampler?.id, "采样");
  mark(read.positiveId, "正向提示词");
  mark(read.negativeId, "负向提示词");
  mark(read.checkpointNode?.id, "底模");
  mark(read.latentId, "尺寸");
  for (const entry of read.loraSources) mark(entry.node.id, "LoRA");
  return roles;
}

export function parseComfyUi(graph) {
  const { nodes, sampler, checkpointNode, latent, loraSources } = readComfyGraph(graph);

  const positive = sampler ? resolveComfyText(graph, sampler.inputs?.positive, "positive") : "";
  const negative = sampler ? resolveComfyText(graph, sampler.inputs?.negative, "negative") : "";
  // A graph with no sampler still usually has text encoders worth showing.
  const encoders = nodes.filter((node) => node.class_type === "CLIPTextEncode");
  const fallbackPositive = positive || textOf(encoders[0]?.inputs?.text);
  const fallbackNegative = negative || (encoders.length > 1 ? textOf(encoders[1]?.inputs?.text) : "");

  const parameters = [];
  if (sampler?.inputs) {
    const setting = (key) => resolveComfyValue(graph, sampler.inputs[key], key);
    pushParameter(parameters, "Steps", setting("steps"));
    pushParameter(parameters, "CFG scale", setting("cfg"));
    pushParameter(parameters, "Sampler", setting("sampler_name"));
    pushParameter(parameters, "Scheduler", setting("scheduler"));
    pushParameter(parameters, "Seed", setting("seed") ?? setting("noise_seed"));
    pushParameter(parameters, "Denoise", setting("denoise"));
  }
  const width = resolveComfyValue(graph, latent?.inputs?.width, "width");
  const height = resolveComfyValue(graph, latent?.inputs?.height, "height");
  if (width !== undefined && height !== undefined) pushParameter(parameters, "Size", `${width}x${height}`);

  const loras = dedupeLoras(loraSources.flatMap((entry) => entry.loras));

  const classes = new Set(nodes.map((node) => node.class_type));
  const has = (predicate) => [...classes].some(predicate);
  const flags = [
    { id: "hires", label: "Hires / 放大", enabled: has((name) => name.includes("Upscale") || name.includes("UltimateSD")), detail: [...classes].filter((name) => name.includes("Upscale")).join(" · ") },
    { id: "adetailer", label: "细节修复", enabled: has((name) => name.includes("FaceDetailer") || name.includes("Impact") || name.includes("ADetailer")), detail: "" },
    { id: "controlnet", label: "ControlNet", enabled: has((name) => name.includes("ControlNet")), detail: "" },
    { id: "inpaint", label: "重绘 / Inpaint", enabled: has((name) => name.includes("Inpaint") || name.includes("VAEEncodeForInpaint")), detail: "" },
  ];

  return {
    source: "comfyui",
    engine: "",
    checkpoint: textOf(checkpointNode?.inputs?.ckpt_name || checkpointNode?.inputs?.unet_name),
    checkpointHash: "",
    positive: fallbackPositive,
    negative: fallbackNegative,
    parameters,
    loras,
    flags,
    nodeCount: nodes.length,
  };
}

// This project's own outputs. The key set is written by `inference_server.py`
// alongside the image; see `08-DATA-PERSISTENCE.md`.
export function parseXiria(parameters, chunks = {}) {
  const list = [];
  pushParameter(list, "Steps", parameters.steps);
  pushParameter(list, "CFG scale", parameters.cfg);
  pushParameter(list, "Sampler", parameters.sampler);
  pushParameter(list, "Scheduler", parameters.scheduler);
  pushParameter(list, "Seed", parameters.seed);
  if (parameters.base_seed && parameters.base_seed !== parameters.seed) pushParameter(list, "Base seed", parameters.base_seed);
  pushParameter(list, "Size", parameters.output_width && parameters.output_height ? `${parameters.output_width}x${parameters.output_height}` : "");
  if (parameters.width !== parameters.output_width || parameters.height !== parameters.output_height) {
    pushParameter(list, "Requested size", parameters.width && parameters.height ? `${parameters.width}x${parameters.height}` : "");
  }
  pushParameter(list, "Denoise", parameters.denoise);
  pushParameter(list, "Guidance", parameters.guidance?.mode || parameters.guidance?.label);
  pushParameter(list, "Elapsed", parameters.elapsed_seconds ? `${parameters.elapsed_seconds}s` : "");
  pushParameter(list, "Job", parameters.job_id);

  const anima = parameters.engine === "Anima";
  const flags = [
    { id: "hires", label: "Hires.fix", enabled: Boolean(parameters.hires), detail: parameters.hires?.upscaler || parameters.hires?.mode || "" },
    { id: "adetailer", label: "ADetailer", enabled: Boolean(parameters.adetailer), detail: parameters.adetailer?.detector || "" },
    { id: "rtx", label: "RTX VSR", enabled: Boolean(parameters.rtx), detail: parameters.rtx?.quality ? `Q${parameters.rtx.quality}` : "" },
    { id: "img2img", label: "图生图", enabled: Boolean(parameters.source_image), detail: parameters.source_image?.resize_mode || "" },
    { id: "transparent", label: "透明背景", enabled: Boolean(parameters.transparent_background), detail: parameters.background_removal_model || "" },
  ];

  return {
    source: "xiria",
    engine: textOf(parameters.engine),
    checkpoint: textOf(anima ? parameters.diffusion_model : parameters.checkpoint),
    checkpointHash: textOf(parameters.checkpoint_sha256),
    positive: textOf(chunks.prompt),
    negative: textOf(chunks.negative_prompt),
    parameters: list,
    loras: Array.isArray(parameters.loras) ? parameters.loras.map((item) => ({ name: textOf(item.name), weight: Number(item.weight) })) : [],
    flags,
    animaAssets: anima ? { text_encoder: textOf(parameters.text_encoder), vae: textOf(parameters.vae) } : null,
  };
}

// A ComfyUI workflow that ends in an Image Saver node writes an A1111-style
// `parameters` block beside the graph — the producer's own summary of the run,
// and the only place some values are stated plainly when the graph routes them
// through slider and selector nodes a reader cannot interpret. The graph stays
// authoritative wherever it resolved something; the sidecar fills the gaps.
function mergeComfySidecar(comfy, parametersText) {
  const text = textOf(parametersText).trim();
  if (!text || text.startsWith("{")) return comfy;
  const sidecar = parseA1111(text);
  const known = new Set(comfy.parameters.map((item) => item.label));
  return {
    ...comfy,
    checkpoint: comfy.checkpoint || sidecar.checkpoint,
    checkpointHash: comfy.checkpointHash || sidecar.checkpointHash,
    positive: comfy.positive || sidecar.positive,
    negative: comfy.negative || sidecar.negative,
    parameters: [...comfy.parameters, ...sidecar.parameters.filter((item) => !known.has(item.label))],
    loras: comfy.loras.length ? comfy.loras : sidecar.loras,
  };
}

// One place decides which producer wrote the file, by shape rather than by key.
export function interpretImageMetadata(chunks) {
  const entries = chunks && typeof chunks === "object" ? chunks : {};
  const keys = Object.keys(entries);
  if (!keys.length) return { status: IMAGE_INFO_STATUS.empty, source: "unknown", sourceLabel: METADATA_SOURCES.unknown, raw: {} };

  const parametersJson = parseJsonObject(entries.parameters);
  if (parametersJson && (parametersJson.job_id !== undefined || parametersJson.engine !== undefined)) {
    return { status: IMAGE_INFO_STATUS.ok, ...parseXiria(parametersJson, entries), sourceLabel: METADATA_SOURCES.xiria, raw: entries };
  }

  const comfyGraph = parseJsonObject(entries.prompt);
  if (comfyGraph && comfyNodes(comfyGraph).length) {
    return { status: IMAGE_INFO_STATUS.ok, ...mergeComfySidecar(parseComfyUi(comfyGraph), entries.parameters), sourceLabel: METADATA_SOURCES.comfyui, raw: entries };
  }

  if (typeof entries.parameters === "string" && entries.parameters.trim()) {
    return { status: IMAGE_INFO_STATUS.ok, ...parseA1111(entries.parameters), sourceLabel: METADATA_SOURCES.a1111, raw: entries };
  }

  const novelai = parseJsonObject(entries.Comment);
  if (novelai && textOf(entries.Software).includes("NovelAI")) {
    const list = [];
    for (const label of ["steps", "scale", "sampler", "seed", "strength", "noise"]) pushParameter(list, label, novelai[label]);
    return {
      status: IMAGE_INFO_STATUS.ok,
      source: "novelai",
      sourceLabel: METADATA_SOURCES.novelai,
      engine: "",
      checkpoint: textOf(entries.Source),
      checkpointHash: "",
      positive: textOf(novelai.prompt || entries.Description),
      negative: textOf(novelai.uc),
      parameters: list,
      loras: [],
      flags: [],
      raw: entries,
    };
  }

  // Text chunks exist but none of them describe a generation. Saying "no
  // metadata" here would be a lie, and showing nothing would be useless.
  return {
    status: IMAGE_INFO_STATUS.ok,
    source: "unknown",
    sourceLabel: METADATA_SOURCES.unknown,
    engine: "",
    checkpoint: "",
    checkpointHash: "",
    positive: "",
    negative: "",
    parameters: keys.map((key) => ({ label: key, value: textOf(entries[key]).slice(0, 2000) })),
    loras: [],
    flags: [],
    raw: entries,
  };
}

// Model matching, shared by checkpoints and LoRAs. A producer records a bare
// filename (`animagine.safetensors`) or a path relative to *its* root
// (`SDXL/animagine.safetensors`), so matching walks from most to least exact
// and reports which of those it managed.
export function matchModelName(name, catalog) {
  const wanted = textOf(name).trim();
  if (!wanted) return { status: "unknown", name: "" };
  const normalized = wanted.split("\\").join("/").toLowerCase();
  const base = normalized.split("/").pop();
  const stem = base.replace(/\.[^.]+$/, "");

  for (const [predicate, precision] of [
    [(entry) => entry.path === normalized, "exact"],
    [(entry) => entry.base === base, "filename"],
    [(entry) => entry.stem === stem, "name"],
  ]) {
    // Local before shared at every precision level: a file the project owns
    // outranks the same filename borrowed from another tool.
    const local = catalog.find((entry) => entry.origin === "local" && predicate(entry));
    if (local) return { status: "local", name: wanted, precision, match: local };
    const shared = catalog.find((entry) => entry.origin === "shared" && predicate(entry));
    if (shared) return { status: "shared", name: wanted, precision, match: shared };
  }
  return { status: "missing", name: wanted };
}
