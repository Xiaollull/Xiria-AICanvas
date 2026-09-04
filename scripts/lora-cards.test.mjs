import assert from "node:assert/strict";
import { mkdtemp, readFile, readdir, rm, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";

import {
  CARD_ASSET_PATTERN,
  MAXIMUM_CARD_NOTE,
  MAXIMUM_CARD_PROMPT,
  MAXIMUM_CARD_TAGS,
  MAXIMUM_CARD_TITLE,
  MAXIMUM_SHOWCASE,
  cardAssetUrl,
  emptyLoraCardStore,
  groupCardFor,
  groupCardPresentation,
  isBlankLoraCard,
  loraCardFor,
  loraCardPresentation,
  loraPreviewSourceUrl,
  metadataPromptText,
  normalizeLoraCardStore,
  referencedCardAssets,
  sameLoraCardStore,
  withGroupCard,
  withLoraCard,
  withoutGroupCard,
} from "../src/lora-cards.js";
import {
  MAXIMUM_ASSET_BYTES,
  cardAssetId,
  decodeCardImage,
  loraCardPaths,
  pruneCardAssets,
  readCardAsset,
  readLoraCardStore,
  sniffCardImageType,
  writeCardAsset,
  writeLoraCardStore,
} from "./lora-cards.mjs";

const PNG = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13, 0, 0]);
const JPEG = Buffer.concat([Buffer.from([0xff, 0xd8, 0xff, 0xe0]), Buffer.alloc(10, 1)]);
const WEBP = Buffer.concat([Buffer.from("RIFF"), Buffer.alloc(4), Buffer.from("WEBP"), Buffer.alloc(4)]);
const dataUrl = (type, buffer) => `data:${type};base64,${buffer.toString("base64")}`;

async function withState(run) {
  const root = await mkdtemp(path.join(tmpdir(), "lora-cards-"));
  try {
    return await run(root);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}

test("an empty store has a scope for every engine that can mount a LoRA", () => {
  const store = emptyLoraCardStore();
  for (const engine of ["SD", "iL", "Anima", "Flux", "Flux2", "Krea2"]) {
    assert.deepEqual(store.byEngine[engine], { loras: {}, groups: {} });
  }
});

test("a card is scoped to its engine, because a relative path alone names no file", () => {
  const store = withLoraCard(emptyLoraCardStore(), "SD", "Style\\a.safetensors", { title: "夏日" });
  assert.equal(loraCardFor(store, "SD", "Style\\a.safetensors").title, "夏日");
  assert.equal(loraCardFor(store, "Anima", "Style\\a.safetensors").title, "");
});

test("a patch touches only the keys it carries", () => {
  let store = withLoraCard(emptyLoraCardStore(), "SD", "a.safetensors", { title: "夏日", cover: "a".repeat(32) });
  store = withLoraCard(store, "SD", "a.safetensors", { note: "权重 0.6 起" });
  const card = loraCardFor(store, "SD", "a.safetensors");
  // A dialog that edits one field must not clear a cover it never showed.
  assert.equal(card.title, "夏日");
  assert.equal(card.cover, "a".repeat(32));
  assert.equal(card.note, "权重 0.6 起");
});

test("a card emptied back out is removed rather than stored as a blank row", () => {
  let store = withLoraCard(emptyLoraCardStore(), "SD", "a.safetensors", { title: "夏日" });
  assert.deepEqual(Object.keys(store.byEngine.SD.loras), ["a.safetensors"]);
  store = withLoraCard(store, "SD", "a.safetensors", { title: "" });
  assert.deepEqual(Object.keys(store.byEngine.SD.loras), []);
  assert.ok(isBlankLoraCard(loraCardFor(store, "SD", "a.safetensors")));
});

test("stored text is bounded and cleaned", () => {
  const store = withLoraCard(emptyLoraCardStore(), "SD", "a.safetensors", {
    title: "  多  余   空格  ",
    prompt: `${"x".repeat(MAXIMUM_CARD_PROMPT + 50)}`,
    note: "a\r\n\n\n\nb",
    tags: ["  风格 ", "风格", "", "b", "c", "d", "e", "f", "g", "h", "i"],
  });
  const card = loraCardFor(store, "SD", "a.safetensors");
  assert.equal(card.title, "多 余 空格");
  assert.equal(card.prompt.length, MAXIMUM_CARD_PROMPT);
  assert.equal(card.note, "a\n\nb");
  assert.equal(card.tags.length, MAXIMUM_CARD_TAGS, "duplicates and blanks drop out, then the cap applies");
  assert.equal(card.tags[0], "风格");
});

test("a cover is only ever an asset id", () => {
  for (const bad of ["../../etc/passwd", "a".repeat(31), "A".repeat(32), "", null, 7]) {
    const store = withLoraCard(emptyLoraCardStore(), "SD", "a.safetensors", { title: "x", cover: bad });
    assert.equal(loraCardFor(store, "SD", "a.safetensors").cover, "", `${String(bad)} is not an asset id`);
  }
  assert.ok(CARD_ASSET_PATTERN.test("0123456789abcdef0123456789abcdef"));
});

test("a group's first effect image is its cover, so the cover can never dangle", () => {
  const [first, second] = ["1".repeat(32), "2".repeat(32)];
  let store = withGroupCard(emptyLoraCardStore(), "SD", "g-abc", { showcase: [first, second] });
  assert.equal(groupCardPresentation({ group: { name: "夜景" }, card: groupCardFor(store, "SD", "g-abc") }).coverUrl, cardAssetUrl(first));
  // "Set as cover" is a reorder, so removing an image cannot orphan a pointer.
  store = withGroupCard(store, "SD", "g-abc", { showcase: [second, first] });
  assert.equal(groupCardPresentation({ group: { name: "夜景" }, card: groupCardFor(store, "SD", "g-abc") }).coverUrl, cardAssetUrl(second));
});

test("group showcases reject duplicates and stop at the cap", () => {
  const ids = Array.from({ length: MAXIMUM_SHOWCASE + 4 }, (_, index) => String(index % 10).repeat(32));
  const store = withGroupCard(emptyLoraCardStore(), "SD", "g-abc", { showcase: [...ids, ids[0], "nope"] });
  const showcase = groupCardFor(store, "SD", "g-abc").showcase;
  assert.equal(showcase.length, new Set(showcase).size);
  assert.ok(showcase.length <= MAXIMUM_SHOWCASE);
});

test("only a well-formed group id can hold a card", () => {
  const store = withGroupCard(emptyLoraCardStore(), "SD", "../g", { note: "x" });
  assert.deepEqual(Object.keys(store.byEngine.SD.groups), []);
});

test("deleting a combination takes its card with it", () => {
  let store = withGroupCard(emptyLoraCardStore(), "SD", "g-abc", { showcase: ["1".repeat(32)] });
  store = withoutGroupCard(store, "SD", "g-abc");
  assert.deepEqual(Object.keys(store.byEngine.SD.groups), []);
  assert.equal(referencedCardAssets(store).size, 0);
});

test("a damaged store is dropped rather than treated as fatal", () => {
  // Unlike the mounted map, nothing here is on the generation path: everything a
  // reset loses can be typed again, so refusing to save would be the worse harm.
  for (const bad of ["nope", 5, [], { schemaVersion: "1" }, { schemaVersion: 99, byEngine: {} }]) {
    const result = normalizeLoraCardStore(bad);
    assert.equal(result.reset, true);
    assert.deepEqual(result.store, emptyLoraCardStore());
  }
  const partial = normalizeLoraCardStore({
    schemaVersion: 1,
    byEngine: { SD: { loras: { "a.safetensors": { title: "留下" }, "": { title: "丢掉" } }, groups: { "bad id": { note: "丢掉" } } } },
  });
  assert.equal(partial.reset, false);
  assert.equal(partial.rejected, 2);
  assert.equal(partial.store.byEngine.SD.loras["a.safetensors"].title, "留下");
});

test("store equality ignores how a value was spelled", () => {
  const first = withLoraCard(emptyLoraCardStore(), "SD", "a.safetensors", { title: "夏日" });
  const second = withLoraCard(emptyLoraCardStore(), "SD", "a.safetensors", { title: "  夏日  ", tags: [] });
  assert.ok(sameLoraCardStore(first, second));
  assert.ok(!sameLoraCardStore(first, emptyLoraCardStore()));
});

test("referenced assets are exactly the ones a card still points at", () => {
  let store = withLoraCard(emptyLoraCardStore(), "SD", "a.safetensors", { cover: "a".repeat(32) });
  store = withGroupCard(store, "Anima", "g-abc", { showcase: ["b".repeat(32), "c".repeat(32)] });
  assert.deepEqual([...referencedCardAssets(store)].sort(), ["a".repeat(32), "b".repeat(32), "c".repeat(32)]);
  store = withLoraCard(store, "SD", "a.safetensors", { cover: "d".repeat(32) });
  assert.ok(!referencedCardAssets(store).has("a".repeat(32)), "a replaced cover stops being referenced");
});

test("presentation prefers the user, then the lookup, then the file", () => {
  const item = { value: "Style\\a.safetensors", name: "a.safetensors" };
  const metadata = { status: "found", queriedAt: "2026-01-01T00:00:00Z", triggerGroups: [{ words: ["summer", "sunlight"] }] };

  const bare = loraCardPresentation({ engine: "SD", item, card: null, metadata: null });
  assert.equal(bare.title, "a.safetensors");
  assert.equal(bare.prompt, "");
  assert.equal(bare.coverSource, "none");
  assert.equal(bare.customized, false);

  const looked = loraCardPresentation({ engine: "SD", item, card: null, metadata });
  assert.equal(looked.prompt, "summer, sunlight");
  assert.equal(looked.promptSource, "metadata");
  assert.equal(looked.coverSource, "metadata");
  assert.equal(looked.coverUrl, loraPreviewSourceUrl("SD", item.value, metadata));

  const owned = loraCardPresentation({ engine: "SD", item, card: { title: "夏日光线", prompt: "summer", cover: "a".repeat(32) }, metadata });
  assert.equal(owned.title, "夏日光线");
  assert.equal(owned.renamed, true);
  assert.equal(owned.fileName, "a.safetensors");
  assert.equal(owned.promptSource, "custom");
  assert.equal(owned.coverUrl, cardAssetUrl("a".repeat(32)));
  assert.equal(owned.customized, true);
});

test("the preview URL is built in one place and carries the lookup's timestamp", () => {
  const metadata = { status: "found", queriedAt: "2026-01-01T00:00:00Z" };
  assert.equal(
    loraPreviewSourceUrl("SD", "Style\\a.safetensors", metadata),
    "/api/lora-preview?engine=SD&path=Style%5Ca.safetensors&v=2026-01-01T00%3A00%3A00Z",
  );
  assert.equal(loraPreviewSourceUrl("SD", "a.safetensors", { status: "not_found" }), "");
  assert.equal(loraPreviewSourceUrl("", "a.safetensors", metadata), "");
});

test("trigger words flatten into a prompt without repeating themselves", () => {
  assert.equal(metadataPromptText({ triggerGroups: [{ words: ["a", " b "] }, { words: ["b", "c"] }] }), "a, b, c");
  assert.equal(metadataPromptText(null), "");
});

test("image bytes are identified by content, not by what the upload claimed", () => {
  assert.equal(sniffCardImageType(PNG), "image/png");
  assert.equal(sniffCardImageType(JPEG), "image/jpeg");
  assert.equal(sniffCardImageType(WEBP), "image/webp");
  assert.equal(sniffCardImageType(Buffer.from("<html><body>not an image</body>")), "");
  assert.equal(decodeCardImage(dataUrl("image/png", PNG)).extension, ".png");
  // A file that says PNG and carries something else would later be served under
  // a name that misdescribes it, so it is refused instead of stored.
  assert.throws(() => decodeCardImage(dataUrl("image/png", JPEG)), /does not match/);
  assert.throws(() => decodeCardImage(dataUrl("image/gif", PNG)), /PNG, JPEG or WebP/);
  assert.throws(() => decodeCardImage("https://example.com/a.png"), /data URL/);
  assert.throws(() => decodeCardImage("data:image/png;base64,   \n  "), /empty/);
});

test("an oversized image is refused before it is materialised", () => {
  const encoded = "A".repeat(Math.ceil((MAXIMUM_ASSET_BYTES + 1024) * 4 / 3));
  assert.throws(() => decodeCardImage(`data:image/png;base64,${encoded}`), (error) => error.statusCode === 413);
});

test("the same picture uploaded twice reuses one file", async () => {
  await withState(async (root) => {
    const first = await writeCardAsset(root, dataUrl("image/png", PNG));
    const second = await writeCardAsset(root, dataUrl("image/png", PNG));
    assert.equal(first.id, second.id);
    assert.equal(first.id, cardAssetId(PNG));
    assert.deepEqual(await readdir(loraCardPaths(root).assetsDirectory), [`${first.id}.png`]);
    const read = await readCardAsset(root, first.id);
    assert.deepEqual(read.buffer, PNG);
    assert.equal(read.contentType, "image/png");
  });
});

test("an asset id that is not 32 hex characters never reaches the filesystem", async () => {
  await withState(async (root) => {
    for (const bad of ["../../cards", "a".repeat(31), "", null]) {
      await assert.rejects(() => readCardAsset(root, bad), (error) => error.statusCode === 400);
    }
    await assert.rejects(() => readCardAsset(root, "a".repeat(32)), (error) => error.statusCode === 404);
  });
});

test("a saved store round-trips and a missing one is simply empty", async () => {
  await withState(async (root) => {
    assert.deepEqual((await readLoraCardStore(root)).store, emptyLoraCardStore());
    const store = withLoraCard(emptyLoraCardStore(), "SD", "a.safetensors", { title: "夏日" });
    await writeLoraCardStore(root, store);
    const reloaded = await readLoraCardStore(root);
    assert.ok(sameLoraCardStore(reloaded.store, store));
    assert.equal(reloaded.reset, false);
  });
});

test("an unreadable store is reported rather than thrown, and does not lose the images", async () => {
  await withState(async (root) => {
    const { directory, indexPath } = loraCardPaths(root);
    await mkdir(directory, { recursive: true });
    await writeFile(indexPath, "{ not json");
    const result = await readLoraCardStore(root);
    assert.equal(result.reset, true);
    assert.deepEqual(result.store, emptyLoraCardStore());
  });
});

test("saving reclaims images nothing points at any more", async () => {
  await withState(async (root) => {
    const kept = await writeCardAsset(root, dataUrl("image/png", PNG));
    const dropped = await writeCardAsset(root, dataUrl("image/jpeg", JPEG));
    const store = withLoraCard(emptyLoraCardStore(), "SD", "a.safetensors", { cover: kept.id });
    await writeLoraCardStore(root, store);
    assert.deepEqual(await pruneCardAssets(root, store), [dropped.id]);
    assert.deepEqual(await readdir(loraCardPaths(root).assetsDirectory), [`${kept.id}.png`]);
  });
});

test("pruning leaves alone anything that is not a card image", async () => {
  await withState(async (root) => {
    const { assetsDirectory } = loraCardPaths(root);
    await mkdir(assetsDirectory, { recursive: true });
    // This walks a directory inside the user's own state folder; it collects the
    // store's leftovers and must never become a general-purpose cleaner.
    await writeFile(path.join(assetsDirectory, "notes.txt"), "keep me");
    await writeFile(path.join(assetsDirectory, "photo.png"), PNG);
    assert.deepEqual(await pruneCardAssets(root, emptyLoraCardStore()), []);
    assert.deepEqual((await readdir(assetsDirectory)).sort(), ["notes.txt", "photo.png"]);
  });
});

test("the store lives in state-cache, where an update preserves it and a metadata refresh cannot reach it", async () => {
  const paths = loraCardPaths(path.join("project", "state-cache"));
  assert.equal(path.basename(path.dirname(paths.directory)), "state-cache");
  assert.equal(path.basename(paths.indexPath), "cards.json");
  const config = await readFile(new URL("../vite.config.js", import.meta.url), "utf8");
  assert.match(config, /readLoraCardStore\(stateDirectory\)/);
  assert.match(config, /writeLoraCardStore\(stateDirectory, payload\?\.store\)/);
  // The lookup cache is rewritten wholesale by "重新解析"; the card store is a
  // different directory entirely so that cannot touch it.
  assert.match(config, /const loraCacheDirectory = path\.join\(cacheDirectory, "lora-metadata"\)/);
});

test("the card routes require a same-origin request and are registered before the model API", async () => {
  const config = await readFile(new URL("../vite.config.js", import.meta.url), "utf8");
  const handler = config.slice(config.indexOf("async function loraCardsApi"), config.indexOf("function loraCardsApiPlugin"));
  assert.match(handler, /requireSameOrigin\(request\)/);
  assert.match(handler, /max-age=31536000, immutable/, "an image named by its own hash is immutable");
  assert.match(config, /uiStateApiPlugin\(\), loraCardsApiPlugin\(\)/);
});

test("card sizes stay small enough that the whole store fits one request", () => {
  assert.ok(MAXIMUM_CARD_TITLE <= 120);
  assert.ok(MAXIMUM_CARD_NOTE <= 1000);
  assert.ok(MAXIMUM_ASSET_BYTES <= 8 * 1024 * 1024);
});
