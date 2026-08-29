import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import {
  BUILTIN_PROMPT_PRESETS,
  createPromptPreset,
  createPromptPresetId,
  deletePromptPreset,
  emptyPromptPresetContainer,
  insertPromptPreset,
  normalizePromptPresetContainer,
  PROMPT_PRESET_BUILTIN_SEED_VERSION,
  PROMPT_PRESET_SCHEMA_VERSION,
  reorderPromptPreset,
  seededPromptPresetContainer,
  sortPromptPresetRecords,
  updatePromptPreset,
  validatePromptPresetDraft,
} from "../src/prompt-presets.js";

const draft = (overrides = {}) => ({ name: "自定义", content: "custom prompt", position: "end", type: "positive", ...overrides });
const cryptoWithUuids = (...ids) => ({ randomUUID: () => ids.shift(), getRandomValues: () => { throw new Error("unexpected fallback"); } });

test("v1 seed contains the seven existing presets with stable IDs and complete records", () => {
  const seeded = seededPromptPresetContainer();
  assert.equal(seeded.schemaVersion, PROMPT_PRESET_SCHEMA_VERSION);
  assert.equal(seeded.builtinSeedVersion, PROMPT_PRESET_BUILTIN_SEED_VERSION);
  assert.equal(seeded.records.length, 7);
  assert.deepEqual(seeded.records.map(({ id, name, type, position, order, version }) => ({ id, name, type, position, order, version })), [
    { id: "builtin-positive-cinematic-portrait-v1", name: "电影级人像", type: "positive", position: "end", order: 0, version: 1 },
    { id: "builtin-positive-japanese-illustration-v1", name: "日系插画", type: "positive", position: "end", order: 1, version: 1 },
    { id: "builtin-positive-product-photography-v1", name: "产品摄影", type: "positive", position: "end", order: 2, version: 1 },
    { id: "builtin-positive-concept-art-v1", name: "概念艺术", type: "positive", position: "end", order: 3, version: 1 },
    { id: "builtin-negative-general-v1", name: "通用负面", type: "negative", position: "end", order: 0, version: 1 },
    { id: "builtin-negative-figure-repair-v1", name: "人物修复", type: "negative", position: "end", order: 1, version: 1 },
    { id: "builtin-negative-anime-cleanup-v1", name: "动漫净化", type: "negative", position: "end", order: 2, version: 1 },
  ]);
  assert.equal(new Set(BUILTIN_PROMPT_PRESETS.map((record) => record.id)).size, 7);
  for (const record of seeded.records) assert.equal(typeof record.content, "string");
});

test("missing library seeds once while a valid empty library remains empty and deleted builtins never revive", () => {
  const missing = normalizePromptPresetContainer(undefined);
  assert.equal(missing.seeded, true);
  assert.equal(missing.shouldPersist, true);
  assert.equal(missing.container.records.length, 7);

  const empty = normalizePromptPresetContainer(emptyPromptPresetContainer(), { missing: false });
  assert.equal(empty.ok, true);
  assert.equal(empty.seeded, false);
  assert.deepEqual(empty.container.records, []);

  const withoutBuiltin = deletePromptPreset(seededPromptPresetContainer(), BUILTIN_PROMPT_PRESETS[0].id);
  const reloaded = normalizePromptPresetContainer(withoutBuiltin, { missing: false });
  assert.equal(reloaded.container.records.some((record) => record.id === BUILTIN_PROMPT_PRESETS[0].id), false);
  assert.equal(reloaded.container.records.length, 6);

  const editedBuiltin = updatePromptPreset(seededPromptPresetContainer(), BUILTIN_PROMPT_PRESETS[1].id, {
    ...BUILTIN_PROMPT_PRESETS[1],
    name: "我的日系插画",
    content: "user edited content",
    position: "start",
  }).container;
  const editedReloaded = normalizePromptPresetContainer(editedBuiltin, { missing: false });
  assert.equal(editedReloaded.container.records.find((record) => record.id === BUILTIN_PROMPT_PRESETS[1].id).content, "user edited content");
  assert.equal(editedReloaded.container.records.find((record) => record.id === BUILTIN_PROMPT_PRESETS[1].id).version, 2);
});

test("corrupt and future containers fail closed, retain raw identity, and refuse persistence", () => {
  const cases = [
    null,
    [],
    { schemaVersion: 1, builtinSeedVersion: 1, records: "bad" },
    { schemaVersion: 0, builtinSeedVersion: 1, records: [] },
    { schemaVersion: 2, builtinSeedVersion: 1, records: [] },
    { schemaVersion: 1, builtinSeedVersion: 2, records: [] },
  ];
  for (const raw of cases) {
    const result = normalizePromptPresetContainer(raw, { missing: false });
    assert.equal(result.ok, false);
    assert.equal(result.fatal, true);
    assert.equal(result.raw, raw);
    assert.equal(result.shouldPersist, false);
    assert.match(result.error, /Prompt 预设/);
    assert.deepEqual(result.container.records, []);
  }
});

test("illegal records are quarantined without content in warning and valid order is normalized stably", () => {
  const raw = {
    schemaVersion: 1,
    builtinSeedVersion: 1,
    records: [
      { id: "p-late", name: "Late", content: "SECRET-LATE", position: "end", type: "positive", order: 8, version: 3 },
      { id: "bad-order", name: "Bad", content: "SECRET-BAD", position: "end", type: "positive", order: -1, version: 1 },
      { id: "n-first", name: "Negative", content: "SECRET-NEGATIVE", position: "start", type: "negative", order: 9, version: 1 },
      { id: "p-first", name: "First", content: "SECRET-FIRST", position: "middle", type: "positive", order: 2, version: 2 },
      { id: "duplicate-name", name: "Ｆｉｒｓｔ", content: "SECRET-DUPLICATE", position: "end", type: "positive", order: 5, version: 1 },
      { id: "p-late", name: "Duplicate id", content: "SECRET-ID", position: "end", type: "negative", order: 1, version: 1 },
      { id: "bad-content", name: "Content", content: "", position: "end", type: "negative", order: 1, version: 1 },
    ],
  };
  const normalized = normalizePromptPresetContainer(raw, { missing: false });
  assert.equal(normalized.ok, true);
  assert.equal(normalized.invalidRecordCount, 4);
  assert.equal(normalized.shouldPersist, true);
  assert.deepEqual(normalized.container.records.map(({ id, order }) => ({ id, order })), [
    { id: "p-first", order: 0 },
    { id: "p-late", order: 1 },
    { id: "n-first", order: 0 },
  ]);
  assert.match(normalized.warning, /4 条/);
  for (const secret of ["SECRET-BAD", "SECRET-DUPLICATE", "SECRET-ID", "SECRET-CONTENT"]) assert.doesNotMatch(normalized.warning, new RegExp(secret));
});

test("draft validation enforces limits and same-type NFKC/casefold duplicate names", () => {
  const records = [{ id: "one", name: "Ｃａｆｅ", content: "x", position: "end", type: "positive", order: 0, version: 1 }];
  assert.equal(validatePromptPresetDraft(draft({ name: "cafe" }), records).valid, false);
  assert.equal(validatePromptPresetDraft(draft({ name: "cafe", type: "negative" }), records).valid, true);
  assert.equal(validatePromptPresetDraft(draft({ name: "cafe" }), records, "one").valid, true);
  assert.match(validatePromptPresetDraft(draft({ name: "x".repeat(49) }), []).errors.name, /48/);
  assert.match(validatePromptPresetDraft(draft({ content: "x".repeat(2001) }), []).errors.content, /2000/);
  assert.equal(validatePromptPresetDraft(draft({ position: "sideways" }), []).valid, false);
  assert.equal(validatePromptPresetDraft(draft({ name: "STRASSE" }), [{ ...records[0], name: "Straße" }]).valid, false);
  assert.equal(validatePromptPresetDraft(draft({ name: "ΟΣ" }), [{ ...records[0], name: "ος" }]).valid, false);
});

test("CRUD preserves IDs, increments versions, puts type moves at group end, and never revives deleted seeds", () => {
  const cryptoProvider = cryptoWithUuids("123e4567-e89b-42d3-a456-426614174000");
  const created = createPromptPreset(seededPromptPresetContainer(), draft({ position: "middle" }), { cryptoProvider });
  assert.equal(created.validation.valid, true);
  assert.equal(created.record.id, "123e4567-e89b-42d3-a456-426614174000");
  assert.equal(created.record.version, 1);
  assert.equal(created.record.order, 4);

  const edited = updatePromptPreset(created.container, created.record.id, draft({ name: "移到负向", type: "negative", position: "start" }));
  assert.equal(edited.record.id, created.record.id);
  assert.equal(edited.record.version, 2);
  assert.equal(edited.record.type, "negative");
  assert.equal(edited.record.order, 3);
  assert.equal(sortPromptPresetRecords(edited.container.records, "negative").at(-1).id, created.record.id);

  const reordered = reorderPromptPreset(edited.container, created.record.id, 0, "positive");
  const moved = reordered.records.find((record) => record.id === created.record.id);
  assert.equal(moved.type, "positive");
  assert.equal(moved.order, 0);
  assert.equal(moved.version, 3);

  const removed = deletePromptPreset(reordered, created.record.id);
  assert.equal(removed.records.some((record) => record.id === created.record.id), false);
  assert.equal(normalizePromptPresetContainer(removed, { missing: false }).container.records.some((record) => record.id === created.record.id), false);
});

test("invalid duplicate edits do not mutate the container", () => {
  const seeded = seededPromptPresetContainer();
  const existing = seeded.records[0];
  const result = updatePromptPreset(seeded, existing.id, draft({ name: seeded.records[1].name }));
  assert.equal(result.validation.valid, false);
  assert.deepEqual(result.container, seeded);
});

test("IDs prefer randomUUID, retry collisions, and use a secure UUIDv4 fallback without Date/name", () => {
  const preferred = createPromptPresetId([], cryptoWithUuids("123e4567-e89b-42d3-a456-426614174000"));
  assert.equal(preferred, "123e4567-e89b-42d3-a456-426614174000");
  const collision = createPromptPresetId([preferred], cryptoWithUuids(preferred, "123e4567-e89b-42d3-b456-426614174001"));
  assert.equal(collision, "123e4567-e89b-42d3-b456-426614174001");
  const fallback = createPromptPresetId([], {
    getRandomValues(bytes) {
      bytes.set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
      return bytes;
    },
  });
  assert.equal(fallback, "00010203-0405-4607-8809-0a0b0c0d0e0f");
  assert.throws(() => createPromptPresetId([], null), /安全随机数/);
  const source = fs.readFileSync(new URL("../src/prompt-presets.js", import.meta.url), "utf8");
  assert.doesNotMatch(source, /Date\.now|new Date|draft\.name.*id|name.*Date/);
});

test("start/end insertion normalizes only the adjoining English or Chinese punctuation", () => {
  assert.deepEqual(insertPromptPreset("cat, dog", "sun", "start"), { text: "sun, cat, dog", caret: 3 });
  assert.deepEqual(insertPromptPreset("cat, dog", "sun", "end"), { text: "cat, dog, sun", caret: 13 });
  assert.deepEqual(insertPromptPreset("主体，光线", "电影感", "start"), { text: "电影感，主体，光线", caret: 3 });
  assert.deepEqual(insertPromptPreset("主体，光线", "电影感", "end"), { text: "主体，光线，电影感", caret: 9 });
  assert.equal(insertPromptPreset("cat, ", ", sun", "end").text, "cat, sun");
  assert.equal(insertPromptPreset("主体，，", "，电影感", "end").text, "主体，电影感");
});

test("middle uses selection replacement, caret insertion, invalid-selection logical midpoint, and exact caret", () => {
  assert.deepEqual(insertPromptPreset("cat, old, dog", "new", "middle", { start: 5, end: 8 }), { text: "cat, new, dog", caret: 8 });
  assert.deepEqual(insertPromptPreset("cat, dog", "new", "middle", { start: 3, end: 3 }), { text: "cat, new, dog", caret: 8 });
  assert.deepEqual(insertPromptPreset("abcdefg", "X", "middle"), { text: "abc, X, defg", caret: 6 });
  assert.deepEqual(insertPromptPreset("abcdefg", "X", "middle", { start: -1, end: 80 }), { text: "abc, X, defg", caret: 6 });
  assert.deepEqual(insertPromptPreset("😀abc", "X", "middle"), { text: "😀, X, abc", caret: 5 });
});

test("newline, mixed punctuation, weights, parentheses, and internal newlines remain local and lossless", () => {
  assert.equal(insertPromptPreset("first line\nsecond line", "inserted", "end").text, "first line\nsecond line\ninserted");
  assert.equal(insertPromptPreset("first line\nsecond line", "inserted", "start").text, "inserted\nfirst line\nsecond line");
  assert.equal(insertPromptPreset("主体，soft light", "(face:1.25)\nno crop", "end").text, "主体，soft light，(face:1.25)\nno crop");
  assert.equal(insertPromptPreset("cinematic portrait.", "soft light", "end").text, "cinematic portrait. soft light");
  assert.equal(insertPromptPreset("主体。", "电影感", "end").text, "主体。电影感");
  assert.equal(insertPromptPreset("(portrait)", "soft light", "middle", { start: 9, end: 9 }).text, "(portrait, soft light)");
  assert.equal(insertPromptPreset("((masterpiece)), [style], \\(literal\\)", "(eyes:1.2)", "end").text, "((masterpiece)), [style], \\(literal\\), (eyes:1.2)");
  assert.equal(insertPromptPreset("a,,  b, c", "b", "end").text, "a,,  b, c, b", "does not globally rewrite or semantically deduplicate");
});

test("App source pairs every new state setter and enforces running, persistence, selection, ARIA, modal, delete, and Gallery exclusion contracts", () => {
  const app = fs.readFileSync(new URL("../src/App.jsx", import.meta.url), "utf8");
  // The card settings schema and the dialog lifecycle live in gallery-core; the page and the
  // add-to-gallery dialog import them, so the preset library must stay absent from all three.
  const gallery = fs.readFileSync(new URL("../src/gallery-core.js", import.meta.url), "utf8");
  const gallerySources = ["src/gallery-core.js", "src/Gallery.jsx", "src/GalleryPage.jsx"]
    .map((file) => fs.readFileSync(new URL(`../${file}`, import.meta.url), "utf8"));
  const css = fs.readFileSync(new URL("../src/styles.css", import.meta.url), "utf8");
  for (const [state, setter] of [
    ["promptPresets", "setPromptPresets"],
    ["promptPresetLibraryError", "setPromptPresetLibraryError"],
    ["promptPresetLibraryWarning", "setPromptPresetLibraryWarning"],
    ["promptPresetDialog", "setPromptPresetDialog"],
    ["promptPresetDelete", "setPromptPresetDelete"],
  ]) {
    assert.match(app, new RegExp(`const \\[${state}, ${setter}\\] = useState`), `${state} state/setter pairing`);
  }
  assert.match(app, /const \[menuId, setMenuId\] = useState\(""\)/, "PresetBox menu state/setter pairing");
  assert.match(app, /loadWorkspaceState[\s\S]{0,7000}normalizePromptPresetContainer/);
  assert.equal((app.match(/normalizePromptPresetContainer\(/g) || []).length, 1, "loadWorkspaceState is the only App normalization call");
  assert.match(app, /workspaceSnapshot\.current = \{[\s\S]{0,500}promptPresets/);
  assert.match(app, /\[theme,[\s\S]{0,500}promptPresets[\s\S]{0,1200}uiStateReady\]/);
  assert.match(app, /const saveBeforeExit = \(\) => persistUiState\(true\)[\s\S]{0,120}pagehide/);
  assert.match(app, /promptPresetLibraryError[\s\S]{0,400}shouldPersistPromptPresets/);
  assert.match(app, /positivePromptRef/);
  assert.match(app, /negativePromptRef/);
  assert.match(app, /onSelect=\{\(event\) => recordPromptSelection/);
  assert.match(app, /onClick=\{\(event\) => recordPromptSelection/);
  assert.match(app, /onKeyUp=\{\(event\) => recordPromptSelection/);
  assert.match(app, /onFocus=\{\(event\) => recordPromptSelection/);
  assert.match(app, /revision/);
  assert.match(app, /requestAnimationFrame[\s\S]{0,500}setSelectionRange/);
  assert.match(app, /const applyPreset[\s\S]{0,250}status === "running"[\s\S]{0,1800}setPositive\(\(current\)/);
  for (const handler of ["openPromptPresetDialog", "savePromptPreset", "requestDeletePromptPreset", "confirmDeletePromptPreset"]) {
    assert.match(app, new RegExp(`const ${handler}[\\s\\S]{0,180}status === "running"`), `${handler} fails closed while running`);
  }
  assert.match(app, /jobs\/active[\s\S]{0,500}setStatus\("running"\)/, "active-job recovery enters the same running lock");
  assert.match(app, /aria-expanded=\{open\}[\s\S]{0,100}aria-controls=\{listId\}/);
  assert.match(app, /aria-haspopup="menu"[\s\S]{0,100}aria-expanded=/);
  assert.match(app, /role="menu"/);
  assert.match(app, /role="menuitem"/);
  assert.match(app, /role="dialog"[\s\S]{0,100}aria-modal="true"/);
  assert.match(app, /<input autoFocus data-dialog-autofocus/);
  assert.match(gallery, /querySelector\("\[autofocus\], \[data-dialog-autofocus\]"\)/);
  assert.match(app, /预设删除后不会自动恢复/);
  assert.match(app, /草稿尚未保存/);
  assert.match(app, /生成任务运行期间只能查看/);
  assert.match(app, /prompt-preset-main[\s\S]{0,500}prompt-preset-more/);
  assert.match(app, /data-prompt-preset-focus-fallback/);
  assert.match(gallery, /focusReturnSelector[\s\S]{0,1800}document\.querySelector\(focusReturnSelectorRef\.current\)\?\.focus\(\)/);
  assert.doesNotMatch(app.match(/workspaceSnapshot\.current = \{[\s\S]{0,700}\};/)?.[0] || "", /caret|selection/i, "caret and selection are session-only");
  for (const source of gallerySources) assert.doesNotMatch(source, /promptPresets:/, "Gallery settings schema must not declare the preset library");
  assert.match(gallery, /delete normalized\.mountedLorasByEngine/, "Gallery cards must never retain the workspace LoRA map");
  assert.match(app, /delete source\.mountedLorasByEngine/, "Gallery Apply must reject a full workspace LoRA map");
  assert.match(gallery, /delete normalized\.promptPresets|\{ promptPresets, \.\.\.gallerySettings \}/);
  assert.match(app, /const source = savedSettings[\s\S]{0,180}delete source\.promptPresets/);
  // The property guarded here is that prompt presets never reach a gallery card.
  // Both card-producing props now route through `galleryCardSettings`, which
  // calls the stripping helper — the exclusion is unchanged, its caller is not.
  assert.match(app, /const galleryCardSettings = [\s\S]{0,240}gallerySettingsWithoutPromptPresets\(source\)/);
  assert.match(app, /currentSettings=\{galleryCardSettings\(workspaceSnapshot\.current\)\}/);
  assert.match(app, /settings=\{generatedSettings \? galleryCardSettings\(generatedSettings, \{ record: true \}\) : galleryCardSettings\(workspaceSnapshot\.current\)\}/);
  assert.ok(!/(?:currentSettings|settings)=\{gallerySettingsWithoutPromptPresets\(/.test(app),
    "no card-producing prop may bypass galleryCardSettings");
  for (const selector of [".prompt-preset-grid", ".prompt-preset-card", ".prompt-preset-menu", ".prompt-preset-backdrop", ".prompt-preset-dialog", ".prompt-preset-segmented", ".prompt-preset-error"]) assert.match(css, new RegExp(selector.replace(".", "\\.")));
  assert.match(css, /@media \(max-width: 720px\)[\s\S]*\.prompt-preset-grid \{ grid-template-columns: 1fr;/);
});
