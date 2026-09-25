// The user's prompt preset library: groups of prompt cards, positive and negative.
//
// Schema 2 (2026-09-25) gives every card a group. A positive or negative module holds any number
// of groups the user names — characters, art style, quality tags — and each group holds cards in
// the user's order. A card may carry a preview picture, stored as a file by the control plane and
// referenced here only by its content-hash id, because this container lives in the workspace
// document that is rewritten on every change. A schema 1 library, which had no groups, is read into
// one group per type exactly once; every card keeps its id, so the per-engine switches that name
// cards (`activePromptPresets`) survive the move.
//
// Pure on purpose: no React, no fetch. `App` owns the writing.

export const PROMPT_PRESET_SCHEMA_VERSION = 2;
export const PROMPT_PRESET_BUILTIN_SEED_VERSION = 1;
export const PROMPT_PRESET_POSITIONS = Object.freeze(["start", "middle", "end"]);
export const PROMPT_PRESET_TYPES = Object.freeze(["positive", "negative"]);
export const PROMPT_PRESET_NAME_LIMIT = 48;
export const PROMPT_PRESET_GROUP_NAME_LIMIT = 32;
export const PROMPT_PRESET_GROUP_LIMIT = 64;
export const PROMPT_PRESET_PREVIEW_PATTERN = /^[0-9a-f]{32}$/;
const PREVIEW_EDGE_LIMIT = 8192;

export const BUILTIN_PROMPT_PRESET_GROUPS = Object.freeze([
  Object.freeze({ id: "builtin-group-positive-style-v1", type: "positive", name: "风格", order: 0, version: 1 }),
  Object.freeze({ id: "builtin-group-negative-general-v1", type: "negative", name: "通用", order: 0, version: 1 }),
]);

export const BUILTIN_PROMPT_PRESETS = Object.freeze([
  Object.freeze({ id: "builtin-positive-cinematic-portrait-v1", name: "电影级人像", content: "cinematic portrait, 85mm lens, soft rim light, shallow depth of field", position: "end", type: "positive", groupId: "builtin-group-positive-style-v1", order: 0, version: 1 }),
  Object.freeze({ id: "builtin-positive-japanese-illustration-v1", name: "日系插画", content: "anime illustration, delicate linework, soft color palette, expressive lighting", position: "end", type: "positive", groupId: "builtin-group-positive-style-v1", order: 1, version: 1 }),
  Object.freeze({ id: "builtin-positive-product-photography-v1", name: "产品摄影", content: "premium product photography, studio lighting, clean backdrop, sharp details", position: "end", type: "positive", groupId: "builtin-group-positive-style-v1", order: 2, version: 1 }),
  Object.freeze({ id: "builtin-positive-concept-art-v1", name: "概念艺术", content: "concept art, epic scale, environmental storytelling, dramatic atmosphere", position: "end", type: "positive", groupId: "builtin-group-positive-style-v1", order: 3, version: 1 }),
  Object.freeze({ id: "builtin-negative-general-v1", name: "通用负面", content: "worst quality, low quality, blurry, watermark, text, jpeg artifacts", position: "end", type: "negative", groupId: "builtin-group-negative-general-v1", order: 0, version: 1 }),
  Object.freeze({ id: "builtin-negative-figure-repair-v1", name: "人物修复", content: "bad anatomy, malformed hands, extra fingers, asymmetrical eyes, distorted face", position: "end", type: "negative", groupId: "builtin-group-negative-general-v1", order: 1, version: 1 }),
  Object.freeze({ id: "builtin-negative-anime-cleanup-v1", name: "动漫净化", content: "photorealistic, 3d render, noisy background, muddy colors, rough lines", position: "end", type: "negative", groupId: "builtin-group-negative-general-v1", order: 2, version: 1 }),
]);

// Where a schema 1 library's cards go, and where a card whose group vanished is put back. Fixed ids,
// so reading the same stored library twice gives the same container.
const MIGRATED_GROUP = Object.freeze({ positive: "migrated-group-positive-v1", negative: "migrated-group-negative-v1" });
const MIGRATED_GROUP_NAME = "常用预设";
const RECOVERED_GROUP = Object.freeze({ positive: "recovered-group-positive-v1", negative: "recovered-group-negative-v1" });
const RECOVERED_GROUP_NAME = "未分组";

const TYPE_RANK = new Map(PROMPT_PRESET_TYPES.map((type, index) => [type, index]));

export function seededPromptPresetContainer() {
  return {
    schemaVersion: PROMPT_PRESET_SCHEMA_VERSION,
    builtinSeedVersion: PROMPT_PRESET_BUILTIN_SEED_VERSION,
    groups: BUILTIN_PROMPT_PRESET_GROUPS.map((group) => ({ ...group })),
    records: BUILTIN_PROMPT_PRESETS.map((record) => ({ ...record })),
  };
}

export function emptyPromptPresetContainer() {
  return {
    schemaVersion: PROMPT_PRESET_SCHEMA_VERSION,
    builtinSeedVersion: PROMPT_PRESET_BUILTIN_SEED_VERSION,
    groups: [],
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

/** A preview reference as a card may hold it, or `null` for anything else. */
export function normalizePromptPresetPreview(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  if (!PROMPT_PRESET_PREVIEW_PATTERN.test(value.id || "")) return null;
  const edge = (number) => (Number.isSafeInteger(number) && number > 0 && number <= PREVIEW_EDGE_LIMIT ? number : null);
  const width = edge(value.width);
  const height = edge(value.height);
  return width && height ? { id: value.id, width, height } : { id: value.id };
}

/** Where the control plane serves a card's preview picture. */
export function promptPresetPreviewUrl(preview) {
  const id = typeof preview === "string" ? preview : preview?.id;
  return PROMPT_PRESET_PREVIEW_PATTERN.test(id || "") ? `/api/prompt-presets/preview?id=${id}` : "";
}

/** Every preview id the library still points at: what the preview store must keep. */
export function referencedPromptPresetPreviews(container) {
  const ids = new Set();
  for (const record of Array.isArray(container?.records) ? container.records : []) {
    const preview = normalizePromptPresetPreview(record?.preview);
    if (preview) ids.add(preview.id);
  }
  return ids;
}

// ---------------------------------------------------------------------------------------------
// Ordering. The container keeps its arrays in display order: groups by type then position, cards
// by type, then their group's position, then their own. Every write re-derives both, so `order`
// is always the dense index within a type (groups) or within a group (cards).

function orderedContainer(base, groups, records) {
  const indexedGroups = groups.map((group, index) => ({ group, index }));
  indexedGroups.sort((first, second) => TYPE_RANK.get(first.group.type) - TYPE_RANK.get(second.group.type)
    || first.group.order - second.group.order
    || first.index - second.index);
  const groupOrder = { positive: 0, negative: 0 };
  const orderedGroups = indexedGroups.map(({ group }) => ({ ...group, order: groupOrder[group.type]++ }));
  const groupRank = new Map(orderedGroups.map((group, index) => [group.id, index]));
  const indexedRecords = records.map((record, index) => ({ record, index }));
  indexedRecords.sort((first, second) => (groupRank.get(first.record.groupId) ?? 0) - (groupRank.get(second.record.groupId) ?? 0)
    || first.record.order - second.record.order
    || first.index - second.index);
  const cardOrder = new Map();
  const orderedRecords = indexedRecords.map(({ record }) => {
    const order = cardOrder.get(record.groupId) || 0;
    cardOrder.set(record.groupId, order + 1);
    return { ...record, order };
  });
  return {
    schemaVersion: PROMPT_PRESET_SCHEMA_VERSION,
    builtinSeedVersion: base?.builtinSeedVersion ?? PROMPT_PRESET_BUILTIN_SEED_VERSION,
    groups: orderedGroups,
    records: orderedRecords,
  };
}

/** The groups of one type, in the user's order. */
export function sortPromptPresetGroups(container, type = null) {
  const groups = Array.isArray(container?.groups) ? container.groups : [];
  return type === null ? groups : groups.filter((group) => group.type === type);
}

/** The cards of one group, in the user's order. */
export function promptPresetGroupCards(container, groupId) {
  return (Array.isArray(container?.records) ? container.records : []).filter((record) => record.groupId === groupId);
}

/**
 * The cards of one type, in display order: group by group. The records array of a normalized
 * container is already in that order, so this only filters it.
 */
export function sortPromptPresetRecords(records, type = null) {
  const list = Array.isArray(records) ? records.filter((record) => PROMPT_PRESET_TYPES.includes(record?.type)) : [];
  return type === null ? list : list.filter((record) => record.type === type);
}

// ---------------------------------------------------------------------------------------------
// Reading a stored library.

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
    migrated: false,
  };
}

function validGroupShape(group) {
  if (!group || typeof group !== "object" || Array.isArray(group)) return false;
  if (typeof group.id !== "string" || !group.id.trim() || group.id.length > 128) return false;
  if (!PROMPT_PRESET_TYPES.includes(group.type)) return false;
  if (typeof group.name !== "string" || !group.name.trim() || group.name.trim().length > PROMPT_PRESET_GROUP_NAME_LIMIT) return false;
  if (!Number.isSafeInteger(group.order) || group.order < 0) return false;
  return group.version === undefined || (Number.isSafeInteger(group.version) && group.version >= 1);
}

function validRecordShape(record, { requireGroup }) {
  if (!record || typeof record !== "object" || Array.isArray(record)) return false;
  if (typeof record.id !== "string" || !record.id.trim() || record.id.length > 128) return false;
  if (typeof record.name !== "string" || !record.name.trim() || record.name.trim().length > PROMPT_PRESET_NAME_LIMIT) return false;
  if (typeof record.content !== "string" || !record.content.trim()) return false;
  if (!PROMPT_PRESET_POSITIONS.includes(record.position) || !PROMPT_PRESET_TYPES.includes(record.type)) return false;
  if (!Number.isSafeInteger(record.order) || record.order < 0) return false;
  if (requireGroup && (typeof record.groupId !== "string" || !record.groupId.trim() || record.groupId.length > 128)) return false;
  return Number.isSafeInteger(record.version) && record.version >= 1;
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
      migrated: false,
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
  const legacy = raw.schemaVersion === 1;
  if (!legacy && !Array.isArray(raw.groups)) {
    return fatalNormalization(raw, "Prompt 预设库已损坏：groups 不是数组。已停止预设操作并保留原始数据。");
  }

  let invalidRecordCount = 0;
  let repaired = false;
  const groups = [];
  const groupIds = new Set();
  const groupNames = new Set();
  if (legacy) {
    // Schema 1 had no groups: every type gets the one its cards move into.
    for (const type of PROMPT_PRESET_TYPES) {
      groups.push({ id: MIGRATED_GROUP[type], type, name: MIGRATED_GROUP_NAME, order: 0, version: 1 });
      groupIds.add(MIGRATED_GROUP[type]);
    }
  } else {
    for (const candidate of raw.groups) {
      if (!validGroupShape(candidate)) {
        invalidRecordCount += 1;
        continue;
      }
      const group = { id: candidate.id.trim(), type: candidate.type, name: candidate.name.trim(), order: candidate.order, version: candidate.version ?? 1 };
      const nameKey = `${group.type}:${promptPresetNameKey(group.name)}`;
      if (groupIds.has(group.id) || groupNames.has(nameKey)) {
        invalidRecordCount += 1;
        continue;
      }
      groupIds.add(group.id);
      groupNames.add(nameKey);
      groups.push(group);
    }
  }
  const groupType = new Map(groups.map((group) => [group.id, group.type]));

  const records = [];
  const ids = new Set();
  const names = new Set();
  for (const candidate of raw.records) {
    if (!validRecordShape(candidate, { requireGroup: !legacy })) {
      invalidRecordCount += 1;
      continue;
    }
    let groupId = legacy ? MIGRATED_GROUP[candidate.type] : candidate.groupId.trim();
    // A card whose group is gone, or whose group is of the other type, keeps its words: it goes to
    // its type's recovery group rather than being dropped.
    if (groupType.get(groupId) !== candidate.type) {
      groupId = RECOVERED_GROUP[candidate.type];
      if (!groupIds.has(groupId)) {
        groups.push({ id: groupId, type: candidate.type, name: RECOVERED_GROUP_NAME, order: Number.MAX_SAFE_INTEGER, version: 1 });
        groupIds.add(groupId);
        groupType.set(groupId, candidate.type);
      }
      repaired = true;
    }
    const record = {
      id: candidate.id.trim(),
      name: candidate.name.trim(),
      content: candidate.content.trim(),
      position: candidate.position,
      type: candidate.type,
      groupId,
      order: candidate.order,
      version: candidate.version,
    };
    const preview = normalizePromptPresetPreview(candidate.preview);
    if (preview) record.preview = preview;
    else if (candidate.preview !== undefined) repaired = true;
    const nameKey = `${groupId}:${promptPresetNameKey(record.name)}`;
    if (ids.has(record.id) || names.has(nameKey)) {
      invalidRecordCount += 1;
      continue;
    }
    ids.add(record.id);
    names.add(nameKey);
    records.push(record);
  }
  const container = orderedContainer(raw, groups, records);
  container.builtinSeedVersion = PROMPT_PRESET_BUILTIN_SEED_VERSION;
  const changed = legacy
    || repaired
    || invalidRecordCount > 0
    || raw.builtinSeedVersion !== PROMPT_PRESET_BUILTIN_SEED_VERSION
    || JSON.stringify(container.groups) !== JSON.stringify(raw.groups)
    || JSON.stringify(container.records) !== JSON.stringify(raw.records);
  return {
    ok: true,
    fatal: false,
    container,
    raw,
    error: "",
    warning: invalidRecordCount ? `已隔离 ${invalidRecordCount} 条非法 Prompt 预设记录；错误信息不包含预设内容。` : "",
    invalidRecordCount,
    shouldPersist: changed,
    seeded: false,
    migrated: legacy,
  };
}

// ---------------------------------------------------------------------------------------------
// Ids.

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

// ---------------------------------------------------------------------------------------------
// Groups.

export function validatePromptPresetGroupName(name, container, { type, editingId = null } = {}) {
  const value = typeof name === "string" ? name.trim() : "";
  if (!value) return { valid: false, error: "请输入分组名称", value };
  if (value.length > PROMPT_PRESET_GROUP_NAME_LIMIT) return { valid: false, error: `分组名称不能超过 ${PROMPT_PRESET_GROUP_NAME_LIMIT} 个字符`, value };
  const key = promptPresetNameKey(value);
  const duplicate = sortPromptPresetGroups(container, type).some((group) => group.id !== editingId && promptPresetNameKey(group.name) === key);
  if (duplicate) return { valid: false, error: "同一模块中已有同名分组", value };
  return { valid: true, error: "", value };
}

export function createPromptPresetGroup(container, { type, name }, { cryptoProvider = globalThis.crypto } = {}) {
  const current = assertContainer(container);
  if (!PROMPT_PRESET_TYPES.includes(type)) return { container: current, group: null, error: "请选择正向或负向模块" };
  if (sortPromptPresetGroups(current, type).length >= PROMPT_PRESET_GROUP_LIMIT) return { container: current, group: null, error: `每个模块最多 ${PROMPT_PRESET_GROUP_LIMIT} 个分组` };
  const validation = validatePromptPresetGroupName(name, current, { type });
  if (!validation.valid) return { container: current, group: null, error: validation.error };
  const id = createPromptPresetId([...current.groups.map((group) => group.id), ...current.records.map((record) => record.id)], cryptoProvider);
  const next = orderedContainer(current, [...current.groups, { id, type, name: validation.value, order: sortPromptPresetGroups(current, type).length, version: 1 }], current.records);
  return { container: next, group: next.groups.find((group) => group.id === id), error: "" };
}

export function renamePromptPresetGroup(container, id, name) {
  const current = assertContainer(container);
  const group = current.groups.find((item) => item.id === id);
  if (!group) throw new Error("分组不存在或已被删除");
  const validation = validatePromptPresetGroupName(name, current, { type: group.type, editingId: id });
  if (!validation.valid) return { container: current, group, error: validation.error };
  if (validation.value === group.name) return { container: current, group, error: "" };
  const groups = current.groups.map((item) => (item.id === id ? { ...item, name: validation.value, version: (item.version || 1) + 1 } : item));
  const next = orderedContainer(current, groups, current.records);
  return { container: next, group: next.groups.find((item) => item.id === id), error: "" };
}

/** Removes a group with every card in it; says which cards and previews went with it. */
export function deletePromptPresetGroup(container, id) {
  const current = assertContainer(container);
  if (!current.groups.some((group) => group.id === id)) return { container: current, removedRecordIds: [], removedPreviewIds: [] };
  const removed = current.records.filter((record) => record.groupId === id);
  const next = orderedContainer(current, current.groups.filter((group) => group.id !== id), current.records.filter((record) => record.groupId !== id));
  return {
    container: next,
    removedRecordIds: removed.map((record) => record.id),
    removedPreviewIds: removed.map((record) => record.preview?.id).filter(Boolean),
  };
}

/** Moves a group to `targetIndex` among its type's other groups. */
export function reorderPromptPresetGroup(container, id, targetIndex) {
  const current = assertContainer(container);
  const moving = current.groups.find((group) => group.id === id);
  if (!moving) throw new Error("分组不存在或已被删除");
  const siblings = sortPromptPresetGroups(current, moving.type).filter((group) => group.id !== id);
  const index = Math.max(0, Math.min(siblings.length, Math.trunc(Number(targetIndex) || 0)));
  siblings.splice(index, 0, moving);
  const order = new Map(siblings.map((group, position) => [group.id, position]));
  const groups = current.groups.map((group) => (order.has(group.id) ? { ...group, order: order.get(group.id) } : group));
  return orderedContainer(current, groups, current.records);
}

// ---------------------------------------------------------------------------------------------
// Cards.

function cleanDraft(draft) {
  return {
    name: typeof draft?.name === "string" ? draft.name.trim() : "",
    content: typeof draft?.content === "string" ? draft.content.trim() : "",
    position: PROMPT_PRESET_POSITIONS.includes(draft?.position) ? draft.position : "end",
    groupId: typeof draft?.groupId === "string" ? draft.groupId : "",
    preview: normalizePromptPresetPreview(draft?.preview),
  };
}

export function validatePromptPresetDraft(draft, container, editingId = null) {
  const value = cleanDraft(draft);
  const errors = {};
  const group = (Array.isArray(container?.groups) ? container.groups : []).find((item) => item.id === value.groupId);
  if (!value.name) errors.name = "请输入卡片名称";
  else if (value.name.length > PROMPT_PRESET_NAME_LIMIT) errors.name = `名称不能超过 ${PROMPT_PRESET_NAME_LIMIT} 个字符`;
  // A card is a piece of a prompt and is held to the same rule as the prompt itself: it needs
  // content, and nothing caps how much. The name is a label and keeps its limit.
  if (!value.content) errors.content = "请输入 Prompt 内容";
  if (!group) errors.groupId = "请选择分组";
  if (!PROMPT_PRESET_POSITIONS.includes(draft?.position)) errors.position = "请选择有效的拼接位置";
  if (!errors.name && group) {
    const key = promptPresetNameKey(value.name);
    const duplicate = promptPresetGroupCards(container, group.id).some((record) => record.id !== editingId && promptPresetNameKey(record.name) === key);
    if (duplicate) errors.name = "同一分组中已存在同名卡片";
  }
  return { valid: Object.keys(errors).length === 0, errors, value: { ...value, type: group?.type || "positive" } };
}

function withPreview(record, preview) {
  const next = { ...record };
  if (preview) next.preview = preview;
  else delete next.preview;
  return next;
}

export function createPromptPreset(container, draft, { cryptoProvider = globalThis.crypto } = {}) {
  const current = assertContainer(container);
  const validation = validatePromptPresetDraft(draft, current);
  if (!validation.valid) return { container: current, record: null, validation };
  const { preview, ...fields } = validation.value;
  const id = createPromptPresetId([...current.groups.map((group) => group.id), ...current.records.map((item) => item.id)], cryptoProvider);
  const record = withPreview({
    id,
    ...fields,
    order: promptPresetGroupCards(current, fields.groupId).length,
    version: 1,
  }, preview);
  const next = orderedContainer(current, current.groups, [...current.records, record]);
  return { container: next, record: next.records.find((item) => item.id === id), validation };
}

export function updatePromptPreset(container, id, draft) {
  const current = assertContainer(container);
  const existing = current.records.find((record) => record.id === id);
  if (!existing) throw new Error("Prompt 预设不存在或已被删除");
  const validation = validatePromptPresetDraft(draft, current, id);
  if (!validation.valid) return { container: current, record: existing, validation };
  const { preview, ...fields } = validation.value;
  const moved = existing.groupId !== fields.groupId;
  const updated = withPreview({
    ...existing,
    ...fields,
    order: moved ? promptPresetGroupCards(current, fields.groupId).length : existing.order,
    version: existing.version + 1,
  }, preview);
  const next = orderedContainer(current, current.groups, current.records.map((record) => (record.id === id ? updated : record)));
  return { container: next, record: next.records.find((record) => record.id === id), validation };
}

export function deletePromptPreset(container, id) {
  const current = assertContainer(container);
  if (!current.records.some((record) => record.id === id)) return current;
  return orderedContainer(current, current.groups, current.records.filter((record) => record.id !== id));
}

/**
 * Moves a card to `targetIndex` among the other cards of `targetGroupId` (its own group when
 * omitted). A card takes the type of the group it lands in.
 */
export function reorderPromptPreset(container, id, targetIndex, targetGroupId = null) {
  const current = assertContainer(container);
  const moving = current.records.find((record) => record.id === id);
  if (!moving) throw new Error("Prompt 预设不存在或已被删除");
  const group = current.groups.find((item) => item.id === (targetGroupId || moving.groupId));
  if (!group) throw new Error("分组不存在或已被删除");
  const clash = promptPresetGroupCards(current, group.id).some((record) => record.id !== id && promptPresetNameKey(record.name) === promptPresetNameKey(moving.name));
  if (clash) throw new Error(`分组“${group.name}”中已有同名卡片“${moving.name}”`);
  const siblings = promptPresetGroupCards(current, group.id).filter((record) => record.id !== id);
  const index = Math.max(0, Math.min(siblings.length, Math.trunc(Number(targetIndex) || 0)));
  const changedGroup = moving.groupId !== group.id;
  siblings.splice(index, 0, { ...moving, groupId: group.id, type: group.type, version: changedGroup ? moving.version + 1 : moving.version });
  const order = new Map(siblings.map((record, position) => [record.id, position]));
  const placed = new Map(siblings.map((record) => [record.id, record]));
  const records = current.records.map((record) => (placed.has(record.id) ? { ...placed.get(record.id), order: order.get(record.id) } : record));
  return orderedContainer(current, current.groups, records);
}

// ---------------------------------------------------------------------------------------------
// Composition.

// Every contributing piece of a prompt is its own paragraph. A preset that is switched on is not
// written into the box -- it is joined onto what the user typed only when the job is submitted --
// so the blank line is what keeps the two readable as separate thoughts.
export const PROMPT_BLOCK_SEPARATOR = "\n\n";

export function joinPromptBlocks(blocks) {
  return (Array.isArray(blocks) ? blocks : [])
    .map((block) => String(block ?? "").trim())
    .filter(Boolean)
    .join(PROMPT_BLOCK_SEPARATOR);
}

export function normalizeActivePromptPresetIds(value, records = []) {
  const known = new Set((Array.isArray(records) ? records : []).map((record) => record?.id).filter((id) => typeof id === "string"));
  const seen = new Set();
  // A preset that has since been deleted stops being active rather than lingering as a dangling id.
  return (Array.isArray(value) ? value : [])
    .filter((id) => typeof id === "string" && known.has(id) && !seen.has(id) && seen.add(id))
    .slice(0, 200);
}

export function togglePromptPresetId(ids, id) {
  const list = Array.isArray(ids) ? ids : [];
  return list.includes(id) ? list.filter((candidate) => candidate !== id) : [...list, id];
}

// Where a switched-on preset sits relative to what the user typed. Without a caret to insert at,
// "middle" no longer means anything between two halves of the text: it is a second block ahead of
// the typed prompt, after the ones marked as the beginning.
const PRESET_BLOCK_ORDER = Object.freeze({ start: 0, middle: 1, end: 2 });

/** The switched-on cards of one type as prompt blocks, in group order then card order. */
export function activePromptPresetBlocks(records, activeIds, type = "positive") {
  const active = new Set(normalizeActivePromptPresetIds(activeIds, records));
  const chosen = sortPromptPresetRecords(records, type).filter((record) => active.has(record.id));
  const ordered = [...chosen].sort((first, second) => (PRESET_BLOCK_ORDER[first.position] ?? 2) - (PRESET_BLOCK_ORDER[second.position] ?? 2));
  return {
    leading: ordered.filter((record) => record.position !== "end").map((record) => record.content),
    trailing: ordered.filter((record) => record.position === "end").map((record) => record.content),
  };
}

// The whole prompt as the job receives it. The LoRA groups' words come first whatever else is on --
// they name what the mounted weights were trained to answer to, so they lead -- then the presets
// switched on for the beginning, then what the user typed, then the presets marked for the end.
export function composeGenerationPrompt({ groupPrompt = "", records = [], activeIds = [], type = "positive", prompt = "" } = {}) {
  const { leading, trailing } = activePromptPresetBlocks(records, activeIds, type);
  return joinPromptBlocks([groupPrompt, ...leading, prompt, ...trailing]);
}
