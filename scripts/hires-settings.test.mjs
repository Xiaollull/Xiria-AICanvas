import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  MAX_UINT64_SEED,
  galleryOutputSeedSettings,
  generationHiresSeedSettings,
  hiresEffectiveSteps,
  hiresSeedPayload,
  normalizeGalleryHires,
  normalizeHiresSeed,
  normalizeUint64Seed,
  resolvedGalleryOutputHiresSeed,
  secureRandomUint64Seed,
} from "../src/hires-settings.js";

const readSource = (file) => readFile(new URL(`../${file}`, import.meta.url), "utf8");

function sourceBetween(source, start, end) {
  const startIndex = source.indexOf(start);
  assert.notEqual(startIndex, -1, `missing source anchor: ${start}`);
  const endIndex = source.indexOf(end, startIndex + start.length);
  assert.notEqual(endIndex, -1, `missing source anchor: ${end}`);
  return source.slice(startIndex, endIndex);
}

const COMFY_HIRES_SEED = "885289963651097";
const MAX_SEED_TEXT = "18446744073709551615";
const cryptoSourceFor = (high, low) => ({ getRandomValues: (target) => { target[0] = high; target[1] = low; return target; } });

const defaults = { executionMode: "full_frame", tileWidth: "auto", tileHeight: "auto", padding: 32, maskBlur: 8, seamMode: "none", uniformTiles: true, tiledDecode: true, samplers: ["euler", "euler_ancestral"], schedulers: ["normal", "simple"] };

test("Gallery Hires migration reads executionMode only from persisted card source", () => {
  const fallbackWithNewDefault = { executionMode: "full_frame", sampler: "euler", scheduler: "simple" };
  assert.equal(normalizeGalleryHires("Anima", { steps: 27 }, fallbackWithNewDefault, defaults).executionMode, "usdu_tiled");
  assert.equal(normalizeGalleryHires("SD", { steps: 27 }, fallbackWithNewDefault, defaults).executionMode, "full_frame");
  assert.equal(normalizeGalleryHires("iL", { executionMode: "usdu_tiled" }, fallbackWithNewDefault, defaults).executionMode, "full_frame");
  assert.equal(normalizeGalleryHires("Anima", { executionMode: "full_frame" }, fallbackWithNewDefault, defaults).executionMode, "full_frame");
  assert.equal(normalizeGalleryHires("Anima", { executionMode: "usdu_tiled" }, fallbackWithNewDefault, defaults).executionMode, "usdu_tiled");
});

test("new Gallery Anima cards inherit an explicit workspace Hires mode", () => {
  assert.equal(normalizeGalleryHires("Anima", {}, { executionMode: "full_frame" }, defaults, { sourceKind: "workspace_inheritance" }).executionMode, "full_frame");
  assert.equal(normalizeGalleryHires("Anima", {}, { executionMode: "usdu_tiled" }, defaults, { sourceKind: "workspace_inheritance" }).executionMode, "usdu_tiled");
});

test("Gallery Hires preserves independent inherit or override settings and fixed tiled defaults", () => {
  const inherited = normalizeGalleryHires("Anima", {}, {}, defaults);
  assert.equal(inherited.sampler, null);
  assert.equal(inherited.scheduler, null);
  assert.deepEqual({ tileWidth: inherited.tileWidth, tileHeight: inherited.tileHeight, padding: inherited.padding, maskBlur: inherited.maskBlur, seamMode: inherited.seamMode, uniformTiles: inherited.uniformTiles, tiledDecode: inherited.tiledDecode }, { tileWidth: "auto", tileHeight: "auto", padding: 32, maskBlur: 8, seamMode: "none", uniformTiles: true, tiledDecode: true });
  const overridden = normalizeGalleryHires("Anima", { sampler: "euler_ancestral", scheduler: "normal" }, {}, defaults);
  assert.equal(overridden.sampler, "euler_ancestral");
  assert.equal(overridden.scheduler, "normal");
  const explicit = normalizeGalleryHires("Anima", { executionMode: "full_frame", sampler: "euler_ancestral", scheduler: "normal", padding: 48, maskBlur: 12, uniformTiles: false, tiledDecode: false }, {}, defaults);
  assert.deepEqual({ executionMode: explicit.executionMode, sampler: explicit.sampler, scheduler: explicit.scheduler, padding: explicit.padding, maskBlur: explicit.maskBlur, uniformTiles: explicit.uniformTiles, tiledDecode: explicit.tiledDecode }, { executionMode: "full_frame", sampler: "euler_ancestral", scheduler: "normal", padding: 48, maskBlur: 12, uniformTiles: false, tiledDecode: false });
});

test("normalizeUint64Seed keeps the whole unsigned 64-bit range without Number precision loss", () => {
  assert.equal(MAX_UINT64_SEED, 18446744073709551615n);
  assert.equal(normalizeUint64Seed("0"), "0");
  assert.equal(normalizeUint64Seed(MAX_SEED_TEXT), MAX_SEED_TEXT);
  assert.equal(normalizeUint64Seed(COMFY_HIRES_SEED), COMFY_HIRES_SEED);
  assert.equal(normalizeUint64Seed("1015878324182247"), "1015878324182247");
  assert.equal(normalizeUint64Seed("0000885289963651097"), COMFY_HIRES_SEED);
  assert.equal(normalizeUint64Seed(MAX_UINT64_SEED), MAX_SEED_TEXT);
  assert.equal(normalizeUint64Seed(0), "0");
  assert.equal(normalizeUint64Seed(847291), "847291");

  // Values above 2^53 must survive as text; unsafe Number inputs must not be silently rounded.
  assert.equal(normalizeUint64Seed("9007199254740993"), "9007199254740993");
  assert.notEqual(String(Number("9007199254740993")), "9007199254740993");
  assert.equal(normalizeUint64Seed(2 ** 53 + 1), null);
  assert.equal(normalizeUint64Seed(1.5), null);

  for (const rejected of ["18446744073709551616", "-1", "", " 12", "12 ", "0x10", "1e3", "abc", null, undefined, true, false, {}, [], -1, NaN, Infinity]) {
    assert.equal(normalizeUint64Seed(rejected), null, `expected rejection for ${String(rejected)}`);
  }
  assert.equal(normalizeUint64Seed("18446744073709551616", "fallback"), "fallback");
  assert.equal(normalizeUint64Seed(undefined, ""), "");
});

test("secureRandomUint64Seed composes one lossless uint64 from two OS-backed words", () => {
  assert.equal(secureRandomUint64Seed(cryptoSourceFor(0, 0)), "0");
  assert.equal(secureRandomUint64Seed(cryptoSourceFor(0xffffffff, 0xffffffff)), MAX_SEED_TEXT);
  assert.equal(secureRandomUint64Seed(cryptoSourceFor(0, 1)), "1");
  assert.equal(secureRandomUint64Seed(cryptoSourceFor(1, 0)), "4294967296");
  assert.equal(secureRandomUint64Seed(cryptoSourceFor(0x0003252a, 0xa1ce8019)), COMFY_HIRES_SEED);
  for (let attempt = 0; attempt < 32; attempt += 1) {
    const seed = secureRandomUint64Seed();
    assert.equal(normalizeUint64Seed(seed), seed);
  }
  assert.throws(() => secureRandomUint64Seed(null), /Secure random seed generation is unavailable/);
  assert.throws(() => secureRandomUint64Seed({}), /Secure random seed generation is unavailable/);
});

test("normalizeHiresSeed keeps fixed seeds exact and clears stale text for inherit and random", () => {
  assert.deepEqual(normalizeHiresSeed("fixed", COMFY_HIRES_SEED), { seedMode: "fixed", seed: COMFY_HIRES_SEED });
  assert.deepEqual(normalizeHiresSeed("fixed", MAX_SEED_TEXT), { seedMode: "fixed", seed: MAX_SEED_TEXT });
  assert.deepEqual(normalizeHiresSeed("fixed", "0"), { seedMode: "fixed", seed: "0" });

  // An invalid or missing persisted fixed seed must fall back to inherit instead of a lossy value.
  for (const invalid of [null, undefined, "", "-1", "18446744073709551616", "abc", 2 ** 53 + 1]) {
    assert.deepEqual(normalizeHiresSeed("fixed", invalid), { seedMode: "inherit", seed: "" }, `expected inherit fallback for ${String(invalid)}`);
  }

  // inherit/random never carry seed text forward.
  assert.deepEqual(normalizeHiresSeed("inherit", COMFY_HIRES_SEED), { seedMode: "inherit", seed: "" });
  assert.deepEqual(normalizeHiresSeed("random", COMFY_HIRES_SEED), { seedMode: "random", seed: "" });
  assert.deepEqual(normalizeHiresSeed(undefined, COMFY_HIRES_SEED), { seedMode: "inherit", seed: "" });
  assert.deepEqual(normalizeHiresSeed("bogus", COMFY_HIRES_SEED), { seedMode: "inherit", seed: "" });
});

test("generation Hires seed payload sends mode always and a lossless fixed seed only in fixed mode", () => {
  assert.deepEqual(generationHiresSeedSettings({ seedMode: "fixed", seed: COMFY_HIRES_SEED }), { seedMode: "fixed", seed: COMFY_HIRES_SEED });
  assert.deepEqual(generationHiresSeedSettings({ seedMode: "fixed", seed: MAX_SEED_TEXT }), { seedMode: "fixed", seed: MAX_SEED_TEXT });
  assert.deepEqual(generationHiresSeedSettings({ seedMode: "fixed", seed: "" }), { seedMode: "fixed", seed: "0" });
  assert.deepEqual(generationHiresSeedSettings({ seedMode: "inherit", seed: COMFY_HIRES_SEED }), { seedMode: "inherit", seed: "" });
  assert.deepEqual(generationHiresSeedSettings({ seedMode: "random", seed: COMFY_HIRES_SEED }), { seedMode: "random", seed: "" });

  assert.deepEqual(hiresSeedPayload({ seedMode: "fixed", seed: COMFY_HIRES_SEED }), { mode: "fixed", seed: COMFY_HIRES_SEED });
  assert.deepEqual(hiresSeedPayload({ seedMode: "fixed", seed: MAX_SEED_TEXT }), { mode: "fixed", seed: MAX_SEED_TEXT });
  assert.deepEqual(hiresSeedPayload({ seedMode: "inherit", seed: COMFY_HIRES_SEED }), { mode: "inherit" });
  assert.deepEqual(hiresSeedPayload({ seedMode: "random", seed: COMFY_HIRES_SEED }), { mode: "random" });
  assert.deepEqual(hiresSeedPayload({ seedMode: "fixed", seed: "18446744073709551616" }), { mode: "inherit" });
  assert.deepEqual(hiresSeedPayload(undefined), { mode: "inherit" });
  for (const mode of ["inherit", "random"]) {
    assert.equal("seed" in hiresSeedPayload({ seedMode: mode, seed: COMFY_HIRES_SEED }), false);
  }
  assert.equal(JSON.stringify(hiresSeedPayload(generationHiresSeedSettings({ seedMode: "fixed", seed: MAX_SEED_TEXT }))), `{"mode":"fixed","seed":"${MAX_SEED_TEXT}"}`);
});

test("resolved Gallery output Hires seed converts random outputs to their fixed resolved seed", () => {
  const fallback = { seedMode: "fixed", seed: "42" };
  assert.deepEqual(resolvedGalleryOutputHiresSeed({ hires_seed_mode: "random", hires_seed: COMFY_HIRES_SEED }, fallback), { seedMode: "fixed", seed: COMFY_HIRES_SEED });
  assert.deepEqual(resolvedGalleryOutputHiresSeed({ hires_seed_mode: "fixed", hires_seed: MAX_SEED_TEXT }, fallback), { seedMode: "fixed", seed: MAX_SEED_TEXT });
  assert.deepEqual(resolvedGalleryOutputHiresSeed({ hires_seed_mode: "inherit", hires_seed: "1015878324182247" }, fallback), { seedMode: "inherit", seed: "" });

  // Hires-disabled outputs carry a null Hires seed and fall back to the workspace setting.
  assert.deepEqual(resolvedGalleryOutputHiresSeed({ hires_seed_mode: "fixed", hires_seed: null }, fallback), fallback);
  assert.deepEqual(resolvedGalleryOutputHiresSeed({ hires_seed_mode: "random", hires_seed: null }, { seedMode: "random", seed: "" }), { seedMode: "random", seed: "" });
  assert.deepEqual(resolvedGalleryOutputHiresSeed({}, { seedMode: "inherit", seed: "" }), { seedMode: "inherit", seed: "" });
  assert.deepEqual(resolvedGalleryOutputHiresSeed(undefined, fallback), fallback);
  assert.deepEqual(resolvedGalleryOutputHiresSeed({ hires_seed_mode: "bogus", hires_seed: COMFY_HIRES_SEED }, { seedMode: "fixed", seed: "42" }), { seedMode: "fixed", seed: COMFY_HIRES_SEED });
});

test("galleryOutputSeedSettings freezes per-image base and Hires seeds without leaking across outputs", () => {
  const normalized = { seed: "7", seedMode: "random", imagesPerBatch: 4, batchCount: 3, hires: { enabled: true, seedMode: "inherit", seed: "" } };
  const first = { base_seed: "1015878324182247", seed: "1015878324182247", hires_seed_mode: "random", hires_seed: COMFY_HIRES_SEED };
  const second = { base_seed: "2", seed: "2", hires_seed_mode: "fixed", hires_seed: MAX_SEED_TEXT };
  const third = { base_seed: "3", seed: "3", hires_seed_mode: "inherit", hires_seed: "3" };
  const items = [first, second, third];

  const applied = galleryOutputSeedSettings(normalized, first, items);
  assert.equal(applied.seed, "1015878324182247");
  assert.equal(applied.seedMode, "fixed");
  assert.deepEqual({ seedMode: applied.hires.seedMode, seed: applied.hires.seed }, { seedMode: "fixed", seed: COMFY_HIRES_SEED });
  assert.equal(applied.hires.enabled, true);
  assert.deepEqual(applied.imageSeeds, ["1015878324182247", "2", "3"]);
  assert.deepEqual(applied.imageHiresSeedModes, ["fixed", "fixed", "inherit"]);
  assert.deepEqual(applied.imageHiresSeeds, [COMFY_HIRES_SEED, MAX_SEED_TEXT, ""]);
  assert.equal(applied.imagesPerBatch, 1);
  assert.equal(applied.batchCount, 1);
  assert.equal(galleryOutputSeedSettings(normalized, first, items, true).imagesPerBatch, 3);

  // Selecting a different output must not inherit the previous output's Hires seed.
  const secondApplied = galleryOutputSeedSettings(normalized, second, items);
  assert.deepEqual({ seedMode: secondApplied.hires.seedMode, seed: secondApplied.hires.seed }, { seedMode: "fixed", seed: MAX_SEED_TEXT });
  const thirdApplied = galleryOutputSeedSettings(normalized, third, items);
  assert.deepEqual({ seedMode: thirdApplied.hires.seedMode, seed: thirdApplied.hires.seed }, { seedMode: "inherit", seed: "" });
  assert.notEqual(applied.imageHiresSeeds, secondApplied.imageHiresSeeds);
  assert.equal(normalized.hires.seedMode, "inherit");
  assert.equal(normalized.seed, "7");

  assert.deepEqual(galleryOutputSeedSettings(normalized, first, undefined).imageHiresSeeds, []);
  assert.equal(galleryOutputSeedSettings(normalized, { base_seed: null, seed: null }, []).seed, "7");
});

test("Gallery Hires migration defaults legacy cards to inherit and preserves fixed Hires seeds", () => {
  const legacy = normalizeGalleryHires("Anima", { steps: 12 }, {}, defaults);
  assert.deepEqual({ seedMode: legacy.seedMode, seed: legacy.seed }, { seedMode: "inherit", seed: "" });

  const fixed = normalizeGalleryHires("Anima", { seedMode: "fixed", seed: COMFY_HIRES_SEED }, {}, defaults);
  assert.deepEqual({ seedMode: fixed.seedMode, seed: fixed.seed }, { seedMode: "fixed", seed: COMFY_HIRES_SEED });
  const maxFixed = normalizeGalleryHires("SD", { seedMode: "fixed", seed: MAX_SEED_TEXT }, {}, defaults);
  assert.deepEqual({ seedMode: maxFixed.seedMode, seed: maxFixed.seed }, { seedMode: "fixed", seed: MAX_SEED_TEXT });

  const corrupted = normalizeGalleryHires("Anima", { seedMode: "fixed", seed: "18446744073709551616" }, {}, defaults);
  assert.deepEqual({ seedMode: corrupted.seedMode, seed: corrupted.seed }, { seedMode: "inherit", seed: "" });

  const random = normalizeGalleryHires("Anima", { seedMode: "random", seed: COMFY_HIRES_SEED }, {}, defaults);
  assert.deepEqual({ seedMode: random.seedMode, seed: random.seed }, { seedMode: "random", seed: "" });

  const inheritedFromWorkspace = normalizeGalleryHires("Anima", {}, { seedMode: "fixed", seed: COMFY_HIRES_SEED }, defaults, { sourceKind: "workspace_inheritance" });
  assert.deepEqual({ seedMode: inheritedFromWorkspace.seedMode, seed: inheritedFromWorkspace.seed }, { seedMode: "fixed", seed: COMFY_HIRES_SEED });
});

test("workspace and Gallery Hires seed controls stay locked and frozen through generation and reconnect", async () => {
  // Gallery.jsx keeps the add-to-gallery dialog; the curation page it used to hold is its own chunk.
  const [app, gallery, galleryPage] = await Promise.all([
    readSource("src/App.jsx"),
    readSource("src/Gallery.jsx"),
    readSource("src/GalleryPage.jsx"),
  ]);
  const generate = sourceBetween(app, "  const generate = async () => {", "  const releaseLoadedModel = async () => {");
  const restore = sourceBetween(app, "function loadWorkspaceState(saved)", "function reconcileModels");

  assert.match(app, /import \{[^}]*\bgenerationHiresSeedSettings\b[^}]*\bhiresSeedPayload\b[^}]*\} from "\.\/hires-settings"/);
  assert.match(app, /seedMode: "inherit"/);
  assert.match(restore, /const savedHiresSeed = normalizeHiresSeed\(savedHires\.seedMode, savedHires\.seed\)/);

  // The payload carries the frozen resolution, and generated settings are deep-cloned at job start.
  assert.match(generate, /const generationHiresSeed = generationHiresSeedSettings\(hires\)/);
  assert.match(generate, /const generationHires = \{ \.\.\.hires, \.\.\.generationHiresSeed \}/);
  assert.match(generate, /setGeneratedSettings\(JSON\.parse\(JSON\.stringify\(\{[\s\S]*?hires: generationHires,/);
  assert.match(generate, /\.\.\.hiresSeedPayload\(generationHiresSeed\)/);
  assert.doesNotMatch(generate, /seed: Number\(|parseInt\(hires\.seed/);

  // Locking: hires seed controls follow hiresControlsLocked, which the reconnect path also sets.
  assert.match(app, /const hiresControlsLocked = !hires\.enabled \|\| status === "running"/);
  assert.match(app, /Hires Seed 模式<WorkspaceSelect[^>]*value=\{hires\.seedMode\} disabled=\{hiresControlsLocked\}/);
  assert.match(app, /hires-seed-field[\s\S]*?value=\{hires\.seed\} disabled=\{hiresControlsLocked\}/);
  assert.match(app, /hires-seed-field[\s\S]*?<button type="button" title="生成固定 Hires Seed" disabled=\{hiresControlsLocked\}/);
  assert.match(app, /无法恢复生成任务[\s\S]*?generationLocked\.current = true;[\s\S]*?setStatus\("running"\)/);
  assert.match(app, /const hiresSeedReady = hires\.seedMode !== "fixed" \|\| normalizeUint64Seed\(hires\.seed\) !== null/);

  // Gallery routes every persisted and per-image Hires seed through the lossless helpers.
  assert.match(gallery, /import \{[^}]*\bgalleryOutputSeedSettings\b[^}]*\} from "\.\/hires-settings"/);
  assert.match(gallery, /return galleryOutputSeedSettings\(normalized, output, selectedItems, combined\)/);
  assert.match(galleryPage, /imageHiresSeedModes: images\.map\(\(image\) => normalizeHiresSeed\(/);
  assert.match(galleryPage, /imageHiresSeeds: images\.map\(\(image\) => normalizeHiresSeed\(/);
  for (const source of [gallery, galleryPage]) assert.doesNotMatch(source, /Number\(image\.hiresSeed\)|parseInt\(image\.hiresSeed/);
});

test("the Hires pass is counted the way the family actually runs it", () => {
  // Diffusers image-to-image runs int(steps × denoise) updates.
  assert.equal(hiresEffectiveSteps({ steps: 20, denoise: 0.35 }, "SD"), 7);
  assert.equal(hiresEffectiveSteps({ steps: 10, denoise: 0.05 }, "iL"), 0);
  // Native Anima follows Comfy — a longer schedule with the last `steps + 1` sigmas kept — so it
  // runs every requested step whatever the denoise. Multiplying refused a configuration the run
  // would have executed, and understated the steps the progress bar was counting against.
  assert.equal(hiresEffectiveSteps({ steps: 10, denoise: 0.05 }, "Anima"), 10);
  assert.equal(hiresEffectiveSteps({ steps: 20, denoise: 1 }, "Anima"), 20);
  // Nothing configured yet is zero, not NaN: the caller compares it against 1 to block a run.
  assert.equal(hiresEffectiveSteps(undefined, "SD"), 0);
  assert.equal(hiresEffectiveSteps({ steps: 20 }, "SD"), 0);
});
