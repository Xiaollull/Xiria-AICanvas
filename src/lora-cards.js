import { READY_LORA_ENGINES, engineScopeKey } from "./lora-model-scope.js";

// User-owned presentation for LoRA files and LoRA combinations.
//
// Everything in this store is decoration: a title, a prompt the user keeps by
// hand, notes, tags and images. None of it is read when a job is submitted, so a
// damaged card can never stop generation — the worst it can do is show a file
// name instead of a nicer one.
//
// It is deliberately *not* the metadata cache. `.cache/lora-metadata` holds what
// Civitai and the safetensors header say about a file, and "重新解析" rewrites it
// wholesale; this holds what the user says, so it lives in `state-cache` beside
// the workspace where an update preserves it and a refresh cannot reach it.
//
// Cards are scoped per engine like the mounted map and the groups, because a
// relative path only names a file together with the engine directory it is
// resolved against.

export const LORA_CARDS_SCHEMA_VERSION = 1;
export const MAXIMUM_CARD_TITLE = 80;
export const MAXIMUM_CARD_PROMPT = 2000;
export const MAXIMUM_CARD_NOTE = 600;
export const MAXIMUM_CARD_TAGS = 8;
export const MAXIMUM_CARD_TAG = 24;
export const MAXIMUM_SHOWCASE = 12;
export const MAXIMUM_CARDS_PER_ENGINE = 4000;

/** Assets are named by a truncated content hash, so the id is the whole check. */
export const CARD_ASSET_PATTERN = /^[0-9a-f]{32}$/;
const GROUP_ID_PATTERN = /^[0-9a-z-]{1,64}$/;

function isPlainObject(value) { return Boolean(value) && typeof value === "object" && !Array.isArray(value); }

function cleanText(value, limit) {
  if (typeof value !== "string") return "";
  return value.replace(/\r\n?/g, "\n").replace(/\n{3,}/g, "\n\n").slice(0, limit).trim();
}

function cleanLine(value, limit) {
  return typeof value === "string" ? value.replace(/\s+/g, " ").trim().slice(0, limit) : "";
}

function cleanAsset(value) {
  return typeof value === "string" && CARD_ASSET_PATTERN.test(value) ? value : "";
}

function cleanTags(value) {
  if (!Array.isArray(value)) return [];
  const seen = new Set();
  const tags = [];
  for (const entry of value) {
    const tag = cleanLine(entry, MAXIMUM_CARD_TAG);
    if (!tag || seen.has(tag) || tags.length >= MAXIMUM_CARD_TAGS) continue;
    seen.add(tag);
    tags.push(tag);
  }
  return tags;
}

function cleanShowcase(value) {
  if (!Array.isArray(value)) return [];
  const seen = new Set();
  const images = [];
  for (const entry of value) {
    const id = cleanAsset(typeof entry === "string" ? entry : entry?.id);
    if (!id || seen.has(id) || images.length >= MAXIMUM_SHOWCASE) continue;
    seen.add(id);
    images.push(id);
  }
  return images;
}

export function emptyLoraCard() {
  return { title: "", prompt: "", note: "", tags: [], cover: "" };
}

export function emptyGroupCard() {
  return { note: "", showcase: [] };
}

export function normalizeLoraCard(value) {
  if (!isPlainObject(value)) return emptyLoraCard();
  return {
    title: cleanLine(value.title, MAXIMUM_CARD_TITLE),
    prompt: cleanText(value.prompt, MAXIMUM_CARD_PROMPT),
    note: cleanText(value.note, MAXIMUM_CARD_NOTE),
    tags: cleanTags(value.tags),
    cover: cleanAsset(value.cover),
  };
}

export function normalizeGroupCard(value) {
  if (!isPlainObject(value)) return emptyGroupCard();
  return {
    note: cleanText(value.note, MAXIMUM_CARD_NOTE),
    // The first showcase image is the group's cover. Keeping one ordered list
    // instead of a list plus a cover pointer means there is no state where the
    // cover names an image the gallery no longer holds.
    showcase: cleanShowcase(value.showcase),
  };
}

export function isBlankLoraCard(card) {
  const normalized = normalizeLoraCard(card);
  return !normalized.title && !normalized.prompt && !normalized.note && !normalized.tags.length && !normalized.cover;
}

export function isBlankGroupCard(card) {
  const normalized = normalizeGroupCard(card);
  return !normalized.note && !normalized.showcase.length;
}

export function emptyLoraCardStore() {
  return {
    schemaVersion: LORA_CARDS_SCHEMA_VERSION,
    byEngine: Object.fromEntries(READY_LORA_ENGINES.map((engine) => [engine, { loras: {}, groups: {} }])),
  };
}

function normalizeScope(value) {
  const scope = { loras: {}, groups: {} };
  let rejected = 0;
  const loras = isPlainObject(value?.loras) ? value.loras : {};
  for (const [key, entry] of Object.entries(loras)) {
    if (typeof key !== "string" || !key || key.length > 500 || key.includes("\0") || Object.keys(scope.loras).length >= MAXIMUM_CARDS_PER_ENGINE) {
      rejected += 1;
      continue;
    }
    const card = normalizeLoraCard(entry);
    // A card that says nothing is the same as no card, and storing it would let
    // the file grow by one row every time a dialog is opened and closed.
    if (isBlankLoraCard(card)) continue;
    scope.loras[key] = card;
  }
  const groups = isPlainObject(value?.groups) ? value.groups : {};
  for (const [key, entry] of Object.entries(groups)) {
    if (typeof key !== "string" || !GROUP_ID_PATTERN.test(key) || Object.keys(scope.groups).length >= MAXIMUM_CARDS_PER_ENGINE) {
      rejected += 1;
      continue;
    }
    const card = normalizeGroupCard(entry);
    if (isBlankGroupCard(card)) continue;
    scope.groups[key] = card;
  }
  return { scope, rejected };
}

/**
 * A damaged card store is never fatal. Unlike the mounted map — where losing an
 * entry loses part of a generation — everything here can be typed again, so the
 * safe response is to drop what cannot be read and keep the rest.
 */
export function normalizeLoraCardStore(value) {
  if (value === undefined || value === null) return { store: emptyLoraCardStore(), rejected: 0, reset: false };
  if (!isPlainObject(value) || !Number.isInteger(value.schemaVersion) || value.schemaVersion > LORA_CARDS_SCHEMA_VERSION || !isPlainObject(value.byEngine)) {
    return { store: emptyLoraCardStore(), rejected: 0, reset: true };
  }
  const store = emptyLoraCardStore();
  let rejected = 0;
  for (const engine of READY_LORA_ENGINES) {
    const normalized = normalizeScope(value.byEngine[engine]);
    rejected += normalized.rejected;
    store.byEngine[engine] = normalized.scope;
  }
  return { store, rejected, reset: false };
}

export function loraCardFor(store, engine, value) {
  const scope = engineScopeKey(engine);
  return normalizeLoraCard(scope && typeof value === "string" ? store?.byEngine?.[scope]?.loras?.[value] : null);
}

export function groupCardFor(store, engine, groupId) {
  const scope = engineScopeKey(engine);
  return normalizeGroupCard(scope && typeof groupId === "string" ? store?.byEngine?.[scope]?.groups?.[groupId] : null);
}

function withScopeEntry(store, engine, kind, key, card, blank) {
  const scope = engineScopeKey(engine);
  const normalized = normalizeLoraCardStore(store).store;
  if (!scope || typeof key !== "string" || !key) return normalized;
  const entries = { ...normalized.byEngine[scope][kind] };
  if (blank) delete entries[key];
  else entries[key] = card;
  normalized.byEngine[scope] = { ...normalized.byEngine[scope], [kind]: entries };
  return normalized;
}

/**
 * Patches are shallow and partial: only the keys present are touched, so a
 * dialog that edits the title cannot silently clear a cover it never showed.
 */
export function withLoraCard(store, engine, value, patch) {
  const current = loraCardFor(store, engine, value);
  const card = normalizeLoraCard({ ...current, ...(isPlainObject(patch) ? patch : {}) });
  return withScopeEntry(store, engine, "loras", value, card, isBlankLoraCard(card));
}

export function withGroupCard(store, engine, groupId, patch) {
  if (typeof groupId !== "string" || !GROUP_ID_PATTERN.test(groupId)) return normalizeLoraCardStore(store).store;
  const current = groupCardFor(store, engine, groupId);
  const card = normalizeGroupCard({ ...current, ...(isPlainObject(patch) ? patch : {}) });
  return withScopeEntry(store, engine, "groups", groupId, card, isBlankGroupCard(card));
}

/** Cards for groups that no longer exist, so deleting a combination takes its images with it. */
export function withoutGroupCard(store, engine, groupId) {
  return withScopeEntry(store, engine, "groups", groupId, null, true);
}

/**
 * Every asset id the store still points at. An image file whose id is not in
 * here belongs to nobody and is deleted on the next write, which is what keeps
 * `state-cache/lora-cards/assets` from growing by one file per replaced cover.
 */
export function referencedCardAssets(store) {
  const normalized = normalizeLoraCardStore(store).store;
  const referenced = new Set();
  for (const engine of READY_LORA_ENGINES) {
    for (const card of Object.values(normalized.byEngine[engine].loras)) {
      if (card.cover) referenced.add(card.cover);
    }
    for (const card of Object.values(normalized.byEngine[engine].groups)) {
      for (const id of card.showcase) referenced.add(id);
    }
  }
  return referenced;
}

export function sameLoraCardStore(first, second) {
  return JSON.stringify(normalizeLoraCardStore(first).store) === JSON.stringify(normalizeLoraCardStore(second).store);
}

/** The URL an uploaded card image is served from. */
export function cardAssetUrl(id) {
  return CARD_ASSET_PATTERN.test(id || "") ? `/api/lora-cards/asset?id=${id}` : "";
}

/**
 * The preview the metadata lookup downloaded, if it found one. Both mount
 * surfaces built this string themselves and had to be kept in step by hand.
 */
export function loraPreviewSourceUrl(engine, value, metadata) {
  if (metadata?.status !== "found" || !engine || typeof value !== "string" || !value) return "";
  return `/api/lora-preview?engine=${encodeURIComponent(engine)}&path=${encodeURIComponent(value)}&v=${encodeURIComponent(metadata.queriedAt)}`;
}

/** The trigger words a lookup found, flattened into something a prompt box can hold. */
export function metadataPromptText(metadata) {
  const groups = Array.isArray(metadata?.triggerGroups) ? metadata.triggerGroups : [];
  const words = [];
  const seen = new Set();
  for (const group of groups) {
    for (const word of Array.isArray(group?.words) ? group.words : []) {
      const trimmed = typeof word === "string" ? word.trim() : "";
      if (!trimmed || seen.has(trimmed)) continue;
      seen.add(trimmed);
      words.push(trimmed);
    }
  }
  return words.join(", ");
}

/**
 * What a card should actually show, given the user's edits, the mounted entry and
 * whatever the lookup found. The user always wins; the lookup fills the gaps; the
 * file name is the floor. Both mount surfaces and the hover preview read this, so
 * a LoRA looks the same everywhere it appears.
 */
export function loraCardPresentation({ engine, item, card, metadata } = {}) {
  const custom = normalizeLoraCard(card);
  const fallbackPrompt = metadataPromptText(metadata);
  const previewUrl = loraPreviewSourceUrl(engine, item?.value, metadata);
  const coverUrl = custom.cover ? cardAssetUrl(custom.cover) : previewUrl;
  return {
    title: custom.title || item?.name || item?.value || "",
    fileName: item?.name || item?.value || "",
    renamed: Boolean(custom.title) && custom.title !== (item?.name || ""),
    prompt: custom.prompt || fallbackPrompt,
    promptSource: custom.prompt ? "custom" : fallbackPrompt ? "metadata" : "none",
    note: custom.note,
    tags: custom.tags,
    coverUrl,
    coverSource: custom.cover ? "custom" : previewUrl ? "metadata" : "none",
    customized: !isBlankLoraCard(custom),
  };
}

export function groupCardPresentation({ group, card } = {}) {
  const custom = normalizeGroupCard(card);
  return {
    name: group?.name || "",
    note: custom.note,
    showcase: custom.showcase,
    coverUrl: cardAssetUrl(custom.showcase[0] || ""),
    customized: !isBlankGroupCard(custom),
  };
}
