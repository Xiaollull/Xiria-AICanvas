import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  MAX_UINT64_SEED,
  galleryOutputSeedSettings,
  normalizeHiresSeed,
  normalizeUint64Seed,
  resolvedGalleryOutputHiresSeed,
  secureRandomUint64Seed,
  supportsUsduTiled,
} from "../src/hires-settings.js";
import {
  normalizeResampleUnit,
  resampleUnitEffectiveSteps,
} from "../src/resample-chain.js";

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
  const firstUnit = { enabled: true, seedMode: "inherit", seed: "" };
  const normalized = { seed: "7", seedMode: "random", imagesPerBatch: 4, batchCount: 3 };
  const first = { base_seed: "1015878324182247", seed: "1015878324182247", hires_seed_mode: "random", hires_seed: COMFY_HIRES_SEED };
  const second = { base_seed: "2", seed: "2", hires_seed_mode: "fixed", hires_seed: MAX_SEED_TEXT };
  const third = { base_seed: "3", seed: "3", hires_seed_mode: "inherit", hires_seed: "3" };
  const items = [first, second, third];

  const applied = galleryOutputSeedSettings(normalized, first, items, false, firstUnit);
  assert.equal(applied.seed, "1015878324182247");
  assert.equal(applied.seedMode, "fixed");
  assert.deepEqual({ seedMode: applied.resampleSeed.seedMode, seed: applied.resampleSeed.seed }, { seedMode: "fixed", seed: COMFY_HIRES_SEED });
  assert.deepEqual(applied.imageSeeds, ["1015878324182247", "2", "3"]);
  assert.deepEqual(applied.imageHiresSeedModes, ["fixed", "fixed", "inherit"]);
  assert.deepEqual(applied.imageHiresSeeds, [COMFY_HIRES_SEED, MAX_SEED_TEXT, ""]);
  assert.equal(applied.imagesPerBatch, 1);
  assert.equal(applied.batchCount, 1);
  assert.equal(galleryOutputSeedSettings(normalized, first, items, true, firstUnit).imagesPerBatch, 3);

  // Selecting a different output must not inherit the previous output's Hires seed.
  const secondApplied = galleryOutputSeedSettings(normalized, second, items, false, firstUnit);
  assert.deepEqual({ seedMode: secondApplied.resampleSeed.seedMode, seed: secondApplied.resampleSeed.seed }, { seedMode: "fixed", seed: MAX_SEED_TEXT });
  const thirdApplied = galleryOutputSeedSettings(normalized, third, items, false, firstUnit);
  assert.deepEqual({ seedMode: thirdApplied.resampleSeed.seedMode, seed: thirdApplied.resampleSeed.seed }, { seedMode: "inherit", seed: "" });
  assert.notEqual(applied.imageHiresSeeds, secondApplied.imageHiresSeeds);
  assert.equal(firstUnit.seedMode, "inherit");
  assert.equal(normalized.seed, "7");

  assert.deepEqual(galleryOutputSeedSettings(normalized, first, undefined, false, firstUnit).imageHiresSeeds, []);
  assert.equal(galleryOutputSeedSettings(normalized, { base_seed: null, seed: null }, [], false, firstUnit).seed, "7");
});

test("workspace and Gallery resample seeds freeze per job while the next queued task stays editable", async () => {
  // Gallery.jsx keeps the add-to-gallery dialog; the curation page it used to hold is its own chunk.
  const [app, gallery, galleryPage, chain] = await Promise.all([
    readSource("src/App.jsx"),
    readSource("src/Gallery.jsx"),
    readSource("src/GalleryPage.jsx"),
    readSource("src/resample-chain.js"),
  ]);
  const generate = sourceBetween(app, "  const generate = async () => {", "  const releaseLoadedModel = async () => {");
  const restore = sourceBetween(app, "function loadWorkspaceState(saved)", "function reconcileModels");

  // A saved workspace's Hires seed is still read through the lossless helper on the way in.
  assert.match(restore, /const savedHiresSeed = normalizeHiresSeed\(savedHires\.seedMode, savedHires\.seed\)/);
  assert.match(restore, /postprocessChain: migrateResampleChain\(/);

  // The payload carries each unit's own resolution, and generated settings are deep-cloned at job
  // start so a queued edit cannot reach a job already running.
  assert.match(generate, /const submittedSettings = JSON\.parse\(JSON\.stringify\(\{[\s\S]*?postprocessChain,/);
  assert.match(generate, /postprocess_chain: resampleChainPayload\(postprocessChain, \{ engine: model \}\)/);
  assert.doesNotMatch(generate, /seed: Number\(|parseInt\(/);
  // A fixed seed with nothing typed in it never reaches the service as an invalid number.
  assert.match(chain, /const fixed = seedMode === "fixed" && normalizeUint64Seed\(seed\) !== null;/);

  // A running task owns its frozen payload; only a model-switch transaction locks the next draft.
  assert.match(app, /const hiresControlsLocked = modelSwitching;/);
  assert.match(app, /locked=\{hiresControlsLocked\}/);
  assert.match(app, /无法恢复生成任务[\s\S]*?generationLocked\.current = true;[\s\S]*?setStatus\("running"\)/);

  // Gallery routes every persisted and per-image resample seed through the lossless helpers.
  assert.match(gallery, /import \{[^}]*\bgalleryOutputSeedSettings\b[^}]*\} from "\.\/hires-settings"/);
  assert.match(gallery, /return galleryOutputSeedSettings\(normalized, output, selectedItems, combined, firstUnit\)/);
  assert.match(galleryPage, /imageHiresSeedModes: images\.map\(\(image\) => normalizeHiresSeed\(/);
  assert.match(galleryPage, /imageHiresSeeds: images\.map\(\(image\) => normalizeHiresSeed\(/);
  for (const source of [gallery, galleryPage]) assert.doesNotMatch(source, /Number\(image\.hiresSeed\)|parseInt\(image\.hiresSeed/);
});

test("a unit's redraw is counted the way the family actually runs it", () => {
  const native = ["Anima", "Flux", "Flux2", "Krea2"];
  const unit = (overrides) => normalizeResampleUnit({ enabled: true, ...overrides });
  // Diffusers image-to-image runs int(steps x denoise) updates.
  assert.equal(resampleUnitEffectiveSteps(unit({ steps: 20, denoise: 0.35 }), 28, "SD", native), 7);
  assert.equal(resampleUnitEffectiveSteps(unit({ steps: 10, denoise: 0.05 }), 28, "iL", native), 0);
  // Native engines follow Comfy -- a longer schedule with the last `steps + 1` sigmas kept -- so
  // they run every requested step whatever the denoise.
  assert.equal(resampleUnitEffectiveSteps(unit({ steps: 10, denoise: 0.05 }), 28, "Anima", native), 10);
  // A unit that leaves its steps to inherit is counted against the request's own.
  assert.equal(resampleUnitEffectiveSteps(unit({ denoise: 0.5 }), 28, "SD", native), 14);
  assert.equal(resampleUnitEffectiveSteps(unit({ denoise: 0.5 }), 28, "Anima", native), 28);
  // A unit that does not redraw spends no steps at all.
  assert.equal(resampleUnitEffectiveSteps(unit({ scale: 2, denoise: 0 }), 28, "SD", native), 0);
});
test("tiled Hires is offered by engine capability, not by naming one engine", () => {
  // Both engines prepare their conditioning and sigma schedule once and hand the same pair to
  // every tile. The Flux engines are guidance distilled and expose no such surface.
  assert.equal(supportsUsduTiled("Anima"), true);
  assert.equal(supportsUsduTiled("Krea2"), true);
  for (const engine of ["SD", "iL", "Flux", "Flux2", "", null, undefined]) {
    assert.equal(supportsUsduTiled(engine), false, `${engine} must not offer tiled Hires`);
  }
});

test("the workspace, the gallery and image-to-image all ask the same predicate", async () => {
  const [app, gallery, i2iPage, chain, panel] = await Promise.all([
    readSource("src/App.jsx"),
    readSource("src/GalleryPage.jsx"),
    readSource("src/ImageToImagePage.jsx"),
    readSource("src/resample-chain.js"),
    readSource("src/ResampleChainPanel.jsx"),
  ]);
  // The chain module and the editor both ask the predicate; the pages read the answer off them.
  for (const [name, source] of [["GalleryPage", gallery], ["ImageToImagePage", i2iPage], ["resample-chain", chain], ["ResampleChainPanel", panel]]) {
    assert.match(source, /supportsUsduTiled/, `${name} must derive tiled refinement from the shared predicate`);
  }
  // The capability must never be re-derived from an engine name beside a usdu_tiled decision.
  assert.doesNotMatch(app, /=== "Anima"[^\n]*usdu_tiled/);
  assert.doesNotMatch(gallery, /isAnima[^\n]{0,40}usdu_tiled/);
  assert.doesNotMatch(i2iPage, /=== "Anima"[^\n]*usdu_tiled/);
  assert.doesNotMatch(chain, /\banima\b[^\n]{0,20}executionMode === "usdu_tiled"/);
});
