export const PROMPT_PRESET_SCHEMA_VERSION = 1;
export const PROMPT_PRESET_BUILTIN_SEED_VERSION = 1;
export const PROMPT_PRESET_POSITIONS = Object.freeze(["start", "middle", "end"]);
export const PROMPT_PRESET_TYPES = Object.freeze(["positive", "negative"]);

export const BUILTIN_PROMPT_PRESETS = Object.freeze([
  Object.freeze({ id: "builtin-positive-cinematic-portrait-v1", name: "电影级人像", content: "cinematic portrait, 85mm lens, soft rim light, shallow depth of field", position: "end", type: "positive", order: 0, version: 1 }),
  Object.freeze({ id: "builtin-positive-japanese-illustration-v1", name: "日系插画", content: "anime illustration, delicate linework, soft color palette, expressive lighting", position: "end", type: "positive", order: 1, version: 1 }),
  Object.freeze({ id: "builtin-positive-product-photography-v1", name: "产品摄影", content: "premium product photography, studio lighting, clean backdrop, sharp details", position: "end", type: "positive", order: 2, version: 1 }),
  Object.freeze({ id: "builtin-positive-concept-art-v1", name: "概念艺术", content: "concept art, epic scale, environmental storytelling, dramatic atmosphere", position: "end", type: "positive", order: 3, version: 1 }),
  Object.freeze({ id: "builtin-negative-general-v1", name: "通用负面", content: "worst quality, low quality, blurry, watermark, text, jpeg artifacts", position: "end", type: "negative", order: 0, version: 1 }),
  Object.freeze({ id: "builtin-negative-figure-repair-v1", name: "人物修复", content: "bad anatomy, malformed hands, extra fingers, asymmetrical eyes, distorted face", position: "end", type: "negative", order: 1, version: 1 }),
  Object.freeze({ id: "builtin-negative-anime-cleanup-v1", name: "动漫净化", content: "photorealistic, 3d render, noisy background, muddy colors, rough lines", position: "end", type: "negative", order: 2, version: 1 }),
]);

const cloneRecord = (record) => ({ ...record });

export function seededPromptPresetContainer() {
  return {
    schemaVersion: PROMPT_PRESET_SCHEMA_VERSION,
    builtinSeedVersion: PROMPT_PRESET_BUILTIN_SEED_VERSION,
    records: BUILTIN_PROMPT_PRESETS.map(cloneRecord),
  };
}

export function emptyPromptPresetContainer() {
  return {
    schemaVersion: PROMPT_PRESET_SCHEMA_VERSION,
    builtinSeedVersion: PROMPT_PRESET_BUILTIN_SEED_VERSION,
    records: [],
  };
}

export function promptPresetNameKey(value) {
  return String(value ?? "")
    .trim()
    .normalize("NFKC")
    .toLocaleLowerCase("und")
    .replaceAll("ß", "ss")
    .replaceAll("ς", "σ")
    .replaceAll("ſ", "s");
}

function cleanDraft(draft) {
  return {
    name: typeof draft?.name === "string" ? draft.name.trim() : "",
    content: typeof draft?.content === "string" ? draft.content.trim() : "",
    position: PROMPT_PRESET_POSITIONS.includes(draft?.position) ? draft.position : "end",
    type: PROMPT_PRESET_TYPES.includes(draft?.type) ? draft.type : "positive",
  };
}

export function validatePromptPresetDraft(draft, records = [], editingId = null) {
  const value = cleanDraft(draft);
  const errors = {};
  if (!value.name) errors.name = "请输入预设名称";
  else if (value.name.length > 48) errors.name = "名称不能超过 48 个字符";
  if (!value.content) errors.content = "请输入 Prompt 内容";
  else if (value.content.length > 2000) errors.content = "Prompt 内容不能超过 2000 个字符";
  if (!PROMPT_PRESET_TYPES.includes(draft?.type)) errors.type = "请选择正向或负向预设";
  if (!PROMPT_PRESET_POSITIONS.includes(draft?.position)) errors.position = "请选择有效的插入位置";
  if (!errors.name) {
    const key = promptPresetNameKey(value.name);
    const duplicate = records.some((record) => record?.id !== editingId && record?.type === value.type && promptPresetNameKey(record?.name) === key);
    if (duplicate) errors.name = "同一 Prompt 类型中已存在同名预设";
  }
  return { valid: Object.keys(errors).length === 0, errors, value };
}

function validRecordShape(record) {
  if (!record || typeof record !== "object" || Array.isArray(record)) return false;
  if (typeof record.id !== "string" || !record.id.trim() || record.id.length > 128) return false;
  if (typeof record.name !== "string" || !record.name.trim() || record.name.trim().length > 48) return false;
  if (typeof record.content !== "string" || !record.content.trim() || record.content.trim().length > 2000) return false;
  if (!PROMPT_PRESET_POSITIONS.includes(record.position) || !PROMPT_PRESET_TYPES.includes(record.type)) return false;
  if (!Number.isSafeInteger(record.order) || record.order < 0) return false;
  return Number.isSafeInteger(record.version) && record.version >= 1;
}

function orderedRecords(records) {
  const typeRank = new Map(PROMPT_PRESET_TYPES.map((type, index) => [type, index]));
  const indexed = records.map((record, index) => ({ record, index }));
  indexed.sort((first, second) => typeRank.get(first.record.type) - typeRank.get(second.record.type)
    || first.record.order - second.record.order
    || first.index - second.index
    || first.record.id.localeCompare(second.record.id));
  const nextOrder = { positive: 0, negative: 0 };
  return indexed.map(({ record }) => ({ ...record, order: nextOrder[record.type]++ }));
}

export function sortPromptPresetRecords(records, type = null) {
  const ordered = orderedRecords(Array.isArray(records) ? records.filter((record) => PROMPT_PRESET_TYPES.includes(record?.type)) : []);
  return type === null ? ordered : ordered.filter((record) => record.type === type);
}

function fatalNormalization(raw, error) {
  return {
    ok: false,
    fatal: true,
    container: emptyPromptPresetContainer(),
    raw,
    error,
    warning: "",
    invalidRecordCount: 0,
    shouldPersist: false,
    seeded: false,
  };
}

export function normalizePromptPresetContainer(raw, { missing = raw === undefined } = {}) {
  if (missing) {
    return {
      ok: true,
      fatal: false,
      container: seededPromptPresetContainer(),
      raw: undefined,
      error: "",
      warning: "",
      invalidRecordCount: 0,
      shouldPersist: true,
      seeded: true,
    };
  }
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return fatalNormalization(raw, "Prompt 预设库已损坏：容器格式无效。已停止预设操作并保留原始数据。");
  }
  if (!Number.isSafeInteger(raw.schemaVersion) || raw.schemaVersion < 1) {
    return fatalNormalization(raw, "Prompt 预设库已损坏：schemaVersion 无效。已停止预设操作并保留原始数据。");
  }
  if (raw.schemaVersion > PROMPT_PRESET_SCHEMA_VERSION) {
    return fatalNormalization(raw, `Prompt 预设库来自更高版本（v${raw.schemaVersion}），当前程序仅支持 v${PROMPT_PRESET_SCHEMA_VERSION}。已停止预设操作以避免覆盖。`);
  }
  if (raw.builtinSeedVersion !== undefined && (!Number.isSafeInteger(raw.builtinSeedVersion) || raw.builtinSeedVersion < 0)) {
    return fatalNormalization(raw, "Prompt 预设库已损坏：builtinSeedVersion 无效。已停止预设操作并保留原始数据。");
  }
  if (raw.builtinSeedVersion > PROMPT_PRESET_BUILTIN_SEED_VERSION) {
    return fatalNormalization(raw, `Prompt 预设种子来自更高版本（v${raw.builtinSeedVersion}）。已停止预设操作以避免覆盖。`);
  }
  if (!Array.isArray(raw.records)) {
    return fatalNormalization(raw, "Prompt 预设库已损坏：records 不是数组。已停止预设操作并保留原始数据。");
  }

  const accepted = [];
  const ids = new Set();
  const names = new Set();
  let invalidRecordCount = 0;
  for (const candidate of raw.records) {
    if (!validRecordShape(candidate)) {
      invalidRecordCount += 1;
      continue;
    }
    const record = {
      id: candidate.id.trim(),
      name: candidate.name.trim(),
      content: candidate.content.trim(),
      position: candidate.position,
      type: candidate.type,
      order: candidate.order,
      version: candidate.version,
    };
    const nameKey = `${record.type}:${promptPresetNameKey(record.name)}`;
    if (ids.has(record.id) || names.has(nameKey)) {
      invalidRecordCount += 1;
      continue;
    }
    ids.add(record.id);
    names.add(nameKey);
    accepted.push(record);
  }
  const records = orderedRecords(accepted);
  const builtinSeedVersion = PROMPT_PRESET_BUILTIN_SEED_VERSION;
  const changed = invalidRecordCount > 0
    || raw.builtinSeedVersion !== builtinSeedVersion
    || JSON.stringify(records) !== JSON.stringify(raw.records);
  return {
    ok: true,
    fatal: false,
    container: { schemaVersion: PROMPT_PRESET_SCHEMA_VERSION, builtinSeedVersion, records },
    raw,
    error: "",
    warning: invalidRecordCount ? `已隔离 ${invalidRecordCount} 条非法 Prompt 预设记录；错误信息不包含预设内容。` : "",
    invalidRecordCount,
    shouldPersist: changed,
    seeded: false,
  };
}

function secureUuid(cryptoProvider = globalThis.crypto) {
  if (!cryptoProvider) throw new Error("当前环境不提供安全随机数，无法创建 Prompt 预设 ID");
  if (typeof cryptoProvider.randomUUID === "function") {
    const id = cryptoProvider.randomUUID();
    if (/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(id)) return id.toLowerCase();
  }
  if (typeof cryptoProvider.getRandomValues !== "function") throw new Error("当前环境不提供安全随机数，无法创建 Prompt 预设 ID");
  const bytes = new Uint8Array(16);
  cryptoProvider.getRandomValues(bytes);
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = [...bytes].map((value) => value.toString(16).padStart(2, "0"));
  return `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex.slice(6, 8).join("")}-${hex.slice(8, 10).join("")}-${hex.slice(10).join("")}`;
}

export function createPromptPresetId(existingIds = [], cryptoProvider = globalThis.crypto) {
  const occupied = new Set(existingIds);
  for (let attempt = 0; attempt < 32; attempt += 1) {
    const id = secureUuid(cryptoProvider);
    if (!occupied.has(id)) return id;
  }
  throw new Error("无法生成唯一的 Prompt 预设 ID");
}

function assertContainer(container) {
  const normalized = normalizePromptPresetContainer(container, { missing: false });
  if (normalized.fatal) throw new Error(normalized.error);
  return normalized.container;
}

export function createPromptPreset(container, draft, { cryptoProvider = globalThis.crypto } = {}) {
  const current = assertContainer(container);
  const validation = validatePromptPresetDraft(draft, current.records);
  if (!validation.valid) return { container: current, record: null, validation };
  const record = {
    id: createPromptPresetId(current.records.map((item) => item.id), cryptoProvider),
    ...validation.value,
    order: current.records.filter((item) => item.type === validation.value.type).length,
    version: 1,
  };
  const records = orderedRecords([...current.records, record]);
  return { container: { ...current, records }, record: records.find((item) => item.id === record.id), validation };
}

export function updatePromptPreset(container, id, draft) {
  const current = assertContainer(container);
  const existing = current.records.find((record) => record.id === id);
  if (!existing) throw new Error("Prompt 预设不存在或已被删除");
  const validation = validatePromptPresetDraft(draft, current.records, id);
  if (!validation.valid) return { container: current, record: existing, validation };
  const typeChanged = existing.type !== validation.value.type;
  const updated = {
    ...existing,
    ...validation.value,
    order: typeChanged ? current.records.filter((record) => record.type === validation.value.type).length : existing.order,
    version: existing.version + 1,
  };
  const records = orderedRecords(current.records.map((record) => record.id === id ? updated : record));
  return { container: { ...current, records }, record: records.find((record) => record.id === id), validation };
}

export function deletePromptPreset(container, id) {
  const current = assertContainer(container);
  if (!current.records.some((record) => record.id === id)) return current;
  return { ...current, records: orderedRecords(current.records.filter((record) => record.id !== id)) };
}

export function reorderPromptPreset(container, id, targetIndex, targetType = null) {
  const current = assertContainer(container);
  const moving = current.records.find((record) => record.id === id);
  if (!moving) throw new Error("Prompt 预设不存在或已被删除");
  const type = PROMPT_PRESET_TYPES.includes(targetType) ? targetType : moving.type;
  const groups = Object.fromEntries(PROMPT_PRESET_TYPES.map((groupType) => [groupType, sortPromptPresetRecords(current.records, groupType).filter((record) => record.id !== id)]));
  const destination = groups[type];
  const index = Math.max(0, Math.min(destination.length, Math.trunc(Number(targetIndex) || 0)));
  destination.splice(index, 0, { ...moving, type, version: moving.version + 1 });
  const records = PROMPT_PRESET_TYPES.flatMap((groupType) => groups[groupType].map((record, order) => ({ ...record, order })));
  return { ...current, records };
}

const CJK_PATTERN = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/u;

function stripLeftBoundary(value) {
  return value.replace(/[ \t]*(?:(?:,|，)[ \t]*)+$/u, "").replace(/[ \t]+$/u, "");
}

function stripRightBoundary(value) {
  return value.replace(/^(?:[ \t]*(?:,|，))+[ \t]*/u, "").replace(/^[ \t]+/u, "");
}

function nearestSeparator(value, fromEnd) {
  const separators = [...value.matchAll(/\r?\n|,|，/gu)];
  if (!separators.length) return "";
  return fromEnd ? separators.at(-1)[0] : separators[0][0];
}

function boundarySeparator(rawLeft, rawRight, left, right) {
  if (!left || !right || /\r?\n$/u.test(left) || /^\r?\n/u.test(right)) return "";
  const explicitLeft = rawLeft.match(/(,|，)[ \t]*$/u)?.[1];
  const explicitRight = rawRight.match(/^[ \t]*(,|，)/u)?.[1];
  if (explicitLeft === "，" || explicitRight === "，") return "，";
  if (explicitLeft === "," || explicitRight === ",") return ", ";
  if (/[。；：！？、]$/u.test(left) || /^[。；：！？、）】》」』]/u.test(right)) return "";
  if (/[.;:!?]$/u.test(left)) return " ";
  if (/^[.;:!?\)\]\}]/u.test(right) || /[\(\[\{]$/u.test(left)) return "";
  const leftStyle = nearestSeparator(rawLeft, true);
  const rightStyle = nearestSeparator(rawRight, false);
  if (/\r?\n/u.test(leftStyle) || (!leftStyle && /\r?\n/u.test(rightStyle))) return "\n";
  if (leftStyle === "，" || rightStyle === "，") return "，";
  const leftHint = left.slice(-24);
  const rightHint = right.slice(0, 24);
  return CJK_PATTERN.test(leftHint) || CJK_PATTERN.test(rightHint) ? "，" : ", ";
}

function validSelection(selection, length) {
  if (!selection || !Number.isInteger(selection.start) || !Number.isInteger(selection.end)) return null;
  if (selection.start < 0 || selection.end < selection.start || selection.end > length) return null;
  return { start: selection.start, end: selection.end };
}

export function insertPromptPreset(text, presetContent, position = "end", selection = null) {
  const source = String(text ?? "");
  const content = String(presetContent ?? "").trim();
  if (!content) return { text: source, caret: Math.max(0, Math.min(source.length, position === "start" ? 0 : source.length)) };
  let start = position === "start" ? 0 : source.length;
  let end = start;
  if (position === "middle") {
    const selected = validSelection(selection, source.length);
    start = selected ? selected.start : Math.floor(source.length / 2);
    end = selected ? selected.end : start;
  }
  const rawLeft = source.slice(0, start);
  const rawRight = source.slice(end);
  const left = stripLeftBoundary(rawLeft);
  const insertion = stripRightBoundary(stripLeftBoundary(content));
  const right = stripRightBoundary(rawRight);
  const before = boundarySeparator(rawLeft, content, left, insertion);
  const after = boundarySeparator(content, rawRight, insertion, right);
  const inserted = `${left}${before}${insertion}`;
  return { text: `${inserted}${after}${right}`, caret: inserted.length };
}
