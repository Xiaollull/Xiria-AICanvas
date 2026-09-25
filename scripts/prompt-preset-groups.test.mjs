// Prompt preset library schema 2: groups of cards, preview references, and the preview store.
import assert from "node:assert/strict";
import { mkdtemp, readdir, rm, utimes } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import {
  BUILTIN_PROMPT_PRESETS,
  PROMPT_PRESET_SCHEMA_VERSION,
  composeGenerationPrompt,
  createPromptPreset,
  createPromptPresetGroup,
  deletePromptPresetGroup,
  normalizePromptPresetContainer,
  promptPresetGroupCards,
  promptPresetPreviewUrl,
  referencedPromptPresetPreviews,
  renamePromptPresetGroup,
  reorderPromptPreset,
  reorderPromptPresetGroup,
  seededPromptPresetContainer,
  sortPromptPresetGroups,
  sortPromptPresetRecords,
  updatePromptPreset,
  validatePromptPresetDraft,
} from "../src/prompt-presets.js";
import {
  promptPresetPreviewDirectory,
  prunePromptPresetPreviews,
  readPromptPresetPreview,
  referencedPreviewsInUiState,
  writePromptPresetPreview,
} from "./prompt-preset-previews.mjs";

const PREVIEW = "0123456789abcdef0123456789abcdef";
// A 1×1 PNG: the smallest picture the store's type check accepts.
const PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

let sequence = 0;
const crypto = { randomUUID: () => `00000000-0000-4000-8000-${String(++sequence).padStart(12, "0")}` };

test("a fresh library is seeded with one group per module holding the built-in cards", () => {
  const container = seededPromptPresetContainer();
  assert.equal(container.schemaVersion, 2);
  assert.deepEqual(container.groups.map((group) => [group.type, group.name]), [["positive", "风格"], ["negative", "通用"]]);
  assert.equal(container.records.length, BUILTIN_PROMPT_PRESETS.length);
  assert.ok(container.records.every((record) => container.groups.some((group) => group.id === record.groupId && group.type === record.type)));
  const normalized = normalizePromptPresetContainer(container);
  assert.equal(normalized.shouldPersist, false);
});

test("a schema 1 library moves into one group per module once, every card keeping its id and order", () => {
  const legacy = {
    schemaVersion: 1,
    builtinSeedVersion: 1,
    records: [
      { id: "mine-b", name: "Second", content: "b", position: "start", type: "positive", order: 1, version: 3 },
      { id: "mine-a", name: "First", content: "a", position: "end", type: "positive", order: 0, version: 1 },
      { id: "neg", name: "Neg", content: "n", position: "end", type: "negative", order: 0, version: 1 },
    ],
  };
  const read = normalizePromptPresetContainer(legacy);
  assert.equal(read.migrated, true);
  assert.equal(read.shouldPersist, true);
  assert.equal(read.container.schemaVersion, PROMPT_PRESET_SCHEMA_VERSION);
  assert.deepEqual(read.container.groups.map((group) => [group.type, group.name]), [["positive", "常用预设"], ["negative", "常用预设"]]);
  assert.deepEqual(sortPromptPresetRecords(read.container.records, "positive").map((record) => [record.id, record.order, record.version]), [["mine-a", 0, 1], ["mine-b", 1, 3]]);
  // Reading the migrated library again changes nothing and asks for no write.
  const again = normalizePromptPresetContainer(read.container);
  assert.deepEqual(again.container, read.container);
  assert.equal(again.shouldPersist, false);
});

test("damaged schema 2 content is repaired or isolated, never silently lost or trusted", () => {
  const raw = {
    schemaVersion: 2,
    builtinSeedVersion: 1,
    groups: [
      { id: "g1", type: "positive", name: "角色", order: 0, version: 1 },
      { id: "g1", type: "positive", name: "重复 id", order: 1, version: 1 },
      { id: "g2", type: "positive", name: "角色", order: 1, version: 1 },
    ],
    records: [
      { id: "c1", name: "Card", content: "x", position: "end", type: "positive", groupId: "g1", order: 0, version: 1, preview: { id: PREVIEW, width: 512, height: 768 } },
      { id: "c2", name: "Card", content: "y", position: "end", type: "positive", groupId: "g1", order: 1, version: 1 },
      { id: "c3", name: "Orphan", content: "z", position: "end", type: "positive", groupId: "gone", order: 0, version: 1, preview: { id: "not-a-hash" } },
    ],
  };
  const read = normalizePromptPresetContainer(raw);
  assert.equal(read.fatal, false);
  assert.equal(read.invalidRecordCount, 3);
  assert.deepEqual(read.container.groups.map((group) => group.name), ["角色", "未分组"]);
  assert.deepEqual(read.container.records.map((record) => [record.id, record.groupId]), [["c1", "g1"], ["c3", "recovered-group-positive-v1"]]);
  assert.deepEqual(read.container.records[0].preview, { id: PREVIEW, width: 512, height: 768 });
  assert.equal("preview" in read.container.records[1], false);
  assert.equal(normalizePromptPresetContainer({ ...raw, schemaVersion: 3 }).fatal, true);
  assert.equal(normalizePromptPresetContainer({ schemaVersion: 2, records: [] }).fatal, true);
});

test("groups are created, renamed, reordered and deleted with their cards", () => {
  let container = seededPromptPresetContainer();
  const characters = createPromptPresetGroup(container, { type: "positive", name: "角色" }, { cryptoProvider: crypto });
  assert.equal(characters.error, "");
  container = characters.container;
  assert.equal(createPromptPresetGroup(container, { type: "positive", name: " 角色 " }, { cryptoProvider: crypto }).error, "同一模块中已有同名分组");
  assert.equal(createPromptPresetGroup(container, { type: "negative", name: "角色" }, { cryptoProvider: crypto }).error, "");
  container = reorderPromptPresetGroup(container, characters.group.id, 0);
  assert.deepEqual(sortPromptPresetGroups(container, "positive").map((group) => group.name), ["角色", "风格"]);
  const renamed = renamePromptPresetGroup(container, characters.group.id, "人物");
  assert.equal(renamed.group.name, "人物");
  container = renamed.container;
  const card = createPromptPreset(container, { name: "Card 1", content: "1girl", position: "start", groupId: characters.group.id, preview: { id: PREVIEW, width: 10, height: 20 } }, { cryptoProvider: crypto });
  container = card.container;
  const removed = deletePromptPresetGroup(container, characters.group.id);
  assert.deepEqual(removed.removedRecordIds, [card.record.id]);
  assert.deepEqual(removed.removedPreviewIds, [PREVIEW]);
  assert.equal(removed.container.records.some((record) => record.groupId === characters.group.id), false);
});

test("cards live in a group, move between groups and keep their preview", () => {
  let container = seededPromptPresetContainer();
  const style = sortPromptPresetGroups(container, "positive")[0];
  const quality = createPromptPresetGroup(container, { type: "positive", name: "质量" }, { cryptoProvider: crypto });
  container = quality.container;
  assert.equal(validatePromptPresetDraft({ name: "x", content: "y", position: "end", groupId: "" }, container).errors.groupId, "请选择分组");
  // Names are unique within a group, not across them.
  assert.equal(validatePromptPresetDraft({ name: "日系插画", content: "y", position: "end", groupId: style.id }, container).errors.name, "同一分组中已存在同名卡片");
  assert.equal(validatePromptPresetDraft({ name: "日系插画", content: "y", position: "end", groupId: quality.group.id }, container).valid, true);
  const created = createPromptPreset(container, { name: "masterpiece", content: "masterpiece, best quality", position: "start", groupId: quality.group.id, preview: { id: PREVIEW, width: 512, height: 512 } }, { cryptoProvider: crypto });
  container = created.container;
  assert.deepEqual(created.record.preview, { id: PREVIEW, width: 512, height: 512 });
  assert.equal(created.record.type, "positive");
  // Drag a style card into the quality group, first position.
  const moved = reorderPromptPreset(container, "builtin-positive-concept-art-v1", 0, quality.group.id);
  assert.deepEqual(promptPresetGroupCards(moved, quality.group.id).map((record) => record.name), ["概念艺术", "masterpiece"]);
  assert.deepEqual(promptPresetGroupCards(moved, style.id).map((record) => record.order), [0, 1, 2]);
  // Editing away the preview drops it; moving by edit appends to the new group.
  const edited = updatePromptPreset(moved, created.record.id, { name: "masterpiece", content: "masterpiece", position: "end", groupId: style.id, preview: null });
  assert.equal("preview" in edited.record, false);
  assert.equal(edited.record.order, promptPresetGroupCards(edited.container, style.id).length - 1);
  // A same-named card cannot be dropped into a group that already has one.
  const twin = createPromptPreset(edited.container, { name: "电影级人像", content: "q", position: "end", groupId: quality.group.id }, { cryptoProvider: crypto });
  assert.throws(() => reorderPromptPreset(twin.container, twin.record.id, 0, style.id), /同名卡片/);
});

test("switched-on cards compose in group order, then card order, around what was typed", () => {
  let container = seededPromptPresetContainer();
  const quality = createPromptPresetGroup(container, { type: "positive", name: "质量" }, { cryptoProvider: crypto });
  container = reorderPromptPresetGroup(quality.container, quality.group.id, 0);
  const lead = createPromptPreset(container, { name: "Q", content: "masterpiece", position: "start", groupId: quality.group.id }, { cryptoProvider: crypto });
  container = lead.container;
  const active = ["builtin-positive-concept-art-v1", lead.record.id, "builtin-positive-cinematic-portrait-v1"];
  assert.equal(
    composeGenerationPrompt({ records: container.records, activeIds: active, type: "positive", prompt: "1girl" }),
    ["masterpiece", "1girl", "cinematic portrait, 85mm lens, soft rim light, shallow depth of field", "concept art, epic scale, environmental storytelling, dramatic atmosphere"].join("\n\n"),
  );
  assert.deepEqual([...referencedPromptPresetPreviews({ records: [{ preview: { id: PREVIEW } }, { preview: { id: "bad" } }, {}] })], [PREVIEW]);
  assert.equal(promptPresetPreviewUrl(PREVIEW), `/api/prompt-presets/preview?id=${PREVIEW}`);
  assert.equal(promptPresetPreviewUrl("../x"), "");
});

test("the preview store keeps pictures by content hash and prunes only old unreferenced ones", async () => {
  const state = await mkdtemp(path.join(tmpdir(), "preset-previews-"));
  try {
    const stored = await writePromptPresetPreview(state, PNG);
    assert.match(stored.id, /^[0-9a-f]{32}$/);
    assert.equal((await writePromptPresetPreview(state, PNG)).id, stored.id);
    const read = await readPromptPresetPreview(state, stored.id);
    assert.equal(read.contentType, "image/png");
    await assert.rejects(readPromptPresetPreview(state, "../../ui-state"), /Invalid preview id/);
    await assert.rejects(writePromptPresetPreview(state, "data:image/png;base64,SGVsbG8gd29ybGQh"), /does not match/);
    // Unreferenced but fresh: an upload into a card that is still being edited stays.
    assert.deepEqual(await prunePromptPresetPreviews(state, new Set()), []);
    // Referenced: kept however old.
    const file = path.join(promptPresetPreviewDirectory(state), `${stored.id}.png`);
    const old = new Date(Date.now() - 2 * 60 * 60 * 1000);
    await utimes(file, old, old);
    assert.deepEqual(await prunePromptPresetPreviews(state, new Set([stored.id])), []);
    // Unreferenced and past the grace period: collected.
    assert.deepEqual(await prunePromptPresetPreviews(state, new Set()), [stored.id]);
    assert.deepEqual(await readdir(promptPresetPreviewDirectory(state)), []);
    // A library this build cannot read protects its pictures.
    assert.equal(referencedPreviewsInUiState({ workspace: { promptPresets: { schemaVersion: 3, records: [] } } }), null);
    assert.deepEqual([...referencedPreviewsInUiState({ workspace: { promptPresets: { schemaVersion: 2, groups: [], records: [{ preview: { id: PREVIEW } }] } } })], [PREVIEW]);
  } finally {
    await rm(state, { recursive: true, force: true });
  }
});
