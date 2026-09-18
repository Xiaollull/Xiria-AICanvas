import assert from "node:assert/strict";
import test from "node:test";
import {
  MAX_RTX_OUTPUT_EDGE,
  MAX_RTX_OUTPUT_PIXELS,
  bankersRound,
} from "../src/postprocessing.js";
import {
  RESAMPLE_UNIT_LIMIT,
  addResampleUnit,
  chainSizeLine,
  chainStage,
  createResampleUnit,
  enabledChainStages,
  migrateResampleChain,
  moveChainStage,
  normalizeResampleChain,
  removeChainStage,
  resampleChainIssue,
  resampleChainPayload,
  resampleChainPlan,
  resampleUnitNumber,
  resampleUnitSizes,
  resampleUnits,
  scaledSize,
  updateChainStage,
} from "../src/resample-chain.js";

const unit = (overrides = {}) => createResampleUnit({ scale: 1.5, denoise: 0.35, ...overrides });
const types = (chain) => chain.map((stage) => stage.type);

test("a chain always offers exactly one ADetailer stage and one RTX stage", () => {
  // Both are single-instance. A saved chain that lost one still shows it, switched off, rather than
  // taking a control off the page with no way to get it back.
  assert.deepEqual(types(normalizeResampleChain([])), ["adetailer", "rtx"]);
  assert.deepEqual(types(normalizeResampleChain([{ type: "rtx", enabled: true }])), ["rtx", "adetailer"]);
  const duplicated = normalizeResampleChain([
    { type: "adetailer", enabled: true }, { type: "adetailer", enabled: false }, { type: "rtx" }, { type: "rtx" },
  ]);
  assert.deepEqual(types(duplicated), ["adetailer", "rtx"]);
  assert.equal(chainStage(duplicated, "adetailer").enabled, true);
});

test("a chain holds at most eight resample units", () => {
  const many = normalizeResampleChain(Array.from({ length: 12 }, () => ({ type: "resample", enabled: true, scale: 1.1 })));
  assert.equal(resampleUnits(many).length, RESAMPLE_UNIT_LIMIT);
  let chain = normalizeResampleChain([]);
  for (let index = 0; index < RESAMPLE_UNIT_LIMIT + 3; index += 1) chain = addResampleUnit(chain);
  assert.equal(resampleUnits(chain).length, RESAMPLE_UNIT_LIMIT);
});

test("the three old blocks migrate to one unit in the order they were configured", () => {
  const chain = migrateResampleChain({
    hires: { enabled: true, model: "x4.pth", scale: 2, denoise: 0.4, steps: 18, cfg: 5, seedMode: "fixed", seed: "17" },
    adetailer: { enabled: true, units: [{ detector: "face.pt" }] },
    rtx: { enabled: true, scale: 2, quality: "high" },
    postprocessOrder: ["rtx", "hires", "adetailer"],
  });
  assert.deepEqual(types(chain), ["rtx", "resample", "adetailer"]);
  const migrated = resampleUnits(chain)[0];
  // Field for field, including the rename the block's `model` went through.
  assert.equal(migrated.upscaler, "x4.pth");
  assert.equal(migrated.scale, 2);
  assert.equal(migrated.denoise, 0.4);
  assert.equal(migrated.steps, 18);
  assert.equal(migrated.cfg, 5);
  assert.deepEqual({ seedMode: migrated.seedMode, seed: migrated.seed }, { seedMode: "fixed", seed: "17" });
  assert.equal(chainStage(chain, "rtx").quality, "high");
  assert.equal(chainStage(chain, "adetailer").units[0].detector, "face.pt");
});

test("migration is idempotent, so reopening a migrated record inserts nothing", () => {
  const first = migrateResampleChain({ hires: { enabled: true, model: "x4.pth", scale: 2 } });
  const second = migrateResampleChain({ postprocessChain: first });
  const third = migrateResampleChain({ postprocessChain: second, hires: { enabled: true, model: "x4.pth", scale: 2 } });
  assert.equal(resampleUnits(first).length, 1);
  assert.equal(resampleUnits(second).length, 1);
  assert.equal(resampleUnits(third).length, 1);
  assert.deepEqual(types(second), types(third));
});

test("a disabled Hires block migrates to a disabled unit rather than disappearing", () => {
  const chain = migrateResampleChain({ hires: { enabled: false, model: "x4.pth", scale: 1.5 } });
  assert.equal(resampleUnits(chain).length, 1);
  assert.equal(resampleUnits(chain)[0].enabled, false);
  assert.deepEqual(enabledChainStages(chain), []);
});

test("a unit's target is the exact product and never rounded up to 64", () => {
  // The acceptance example: 1088 x 1.5 = 1632, which is not a multiple of 64 and runs anyway.
  assert.deepEqual(scaledSize({ width: 1088, height: 1088 }, 1.5), { width: 1632, height: 1632 });
  assert.deepEqual(scaledSize({ width: 1024, height: 768 }, 1.5), { width: 1536, height: 1152 });
  // Half-to-even, the way Python's `round` breaks ties, so the preview and the run agree.
  assert.deepEqual(scaledSize({ width: 5, height: 7 }, 1.5), { width: 8, height: 10 });
});

test("the redraw canvas follows the engine's grid while the target stays put", () => {
  const redrawing = unit({ scale: 1.5, denoise: 0.3 });
  const source = { width: 1090, height: 1090 };
  for (const [engine, expected] of [["SD", 1632], ["Anima", 1632], ["Flux", 1632], ["Krea2", 1632]]) {
    const sizes = resampleUnitSizes(source, redrawing, engine);
    assert.deepEqual(sizes.target, { width: 1635, height: 1635 }, engine);
    assert.deepEqual(sizes.redraw, { width: expected, height: expected }, engine);
  }
  // Without a redraw there is no latent grid to satisfy, so the exact target is what is produced.
  const resizeOnly = unit({ scale: 1.5, denoise: 0 });
  assert.deepEqual(resampleUnitSizes(source, resizeOnly, "Anima").output, { width: 1635, height: 1635 });
  // `exactOutput` asks for the target back after the engine rounded it.
  assert.deepEqual(resampleUnitSizes(source, unit({ scale: 1.5, denoise: 0.3, exactOutput: true }), "SD").output,
    { width: 1635, height: 1635 });
});

test("the plan walks the chain and every stage starts where the last one finished", () => {
  const chain = [
    unit({ scale: 1.5, denoise: 0.3 }),
    { ...createResampleUnit({ scale: 1.5, denoise: 0 }) },
    { type: "adetailer", id: "ade", enabled: true, units: [] },
    { type: "rtx", id: "rtx", enabled: true, scale: 2, quality: "ultra" },
  ];
  const plan = resampleChainPlan({ width: 1024, height: 1024 }, chain, "SD");
  assert.deepEqual(plan.trace.map((entry) => entry.type), ["resample", "resample", "adetailer", "rtx"]);
  assert.deepEqual(plan.trace[0].output, { width: 1536, height: 1536 });
  assert.deepEqual(plan.trace[1].output, { width: 2304, height: 2304 });
  assert.deepEqual(plan.trace[2].output, { width: 2304, height: 2304 });
  assert.deepEqual({ width: plan.width, height: plan.height }, { width: 4608, height: 4608 });
  for (let index = 1; index < plan.trace.length; index += 1) {
    assert.deepEqual(plan.trace[index].input, plan.trace[index - 1].output);
  }
  assert.match(chainSizeLine({ width: 1024, height: 1024 }, chain, "SD"), /1024 × 1024 → 1536 × 1536 → 2304 × 2304/);
});

test("a disabled stage is skipped by the plan but keeps its card number", () => {
  const chain = [unit({ enabled: false, scale: 2 }), unit({ scale: 1.5, denoise: 0 })];
  const plan = resampleChainPlan({ width: 1024, height: 1024 }, chain, "SD");
  assert.deepEqual(plan.trace.map((entry) => entry.unitNumber), [2]);
  assert.equal(resampleUnitNumber(chain, chain[1]), 2);
});

test("the plan reports when an engine's grid moved a unit off its target", () => {
  const chain = [unit({ scale: 1.5, denoise: 0.3 })];
  const rounded = resampleChainPlan({ width: 1090, height: 1090 }, chain, "Anima");
  assert.equal(rounded.trace[0].rounded, true);
  assert.deepEqual(rounded.trace[0].target, { width: 1635, height: 1635 });
  assert.deepEqual(rounded.trace[0].redraw, { width: 1632, height: 1632 });
  const exact = resampleChainPlan({ width: 1024, height: 1024 }, chain, "Anima");
  assert.equal(exact.trace[0].rounded, false);
});

test("the envelope and the engine's own refinement limit both refuse a chain", () => {
  const overEnvelope = resampleChainPlan({ width: 4100, height: 1024 }, [{ type: "rtx", id: "rtx", enabled: true, scale: 2 }], "SD");
  assert.equal(overEnvelope.valid, false);
  assert.match(overEnvelope.reason, /8192/);
  const exact = resampleChainPlan({ width: 4096, height: 2048 }, [{ type: "rtx", id: "rtx", enabled: true, scale: 2 }], "SD");
  assert.deepEqual({ width: exact.width, height: exact.height }, { width: MAX_RTX_OUTPUT_EDGE, height: 4096 });
  assert.equal(exact.width * exact.height, MAX_RTX_OUTPUT_PIXELS);
  assert.equal(exact.valid, true);
  // A native engine refuses a refinement canvas past 4096 even well inside the envelope, and the
  // message names the unit rather than leaving the user to find it.
  const nativeOverflow = resampleChainPlan({ width: 2048, height: 2048 }, [unit({ scale: 2.5, denoise: 0.3 })], "Anima");
  assert.equal(nativeOverflow.valid, false);
  assert.match(nativeOverflow.reason, /重采样 1/);
});

test("add, remove and reorder never mutate the chain they were given", () => {
  const original = normalizeResampleChain([{ type: "resample", enabled: true, scale: 2 }]);
  const added = addResampleUnit(original);
  assert.equal(resampleUnits(original).length, 1);
  assert.equal(resampleUnits(added).length, 2);
  // A new unit lands after the last resample unit rather than at the end, so it does not jump past
  // an ADetailer or RTX stage the user placed deliberately.
  assert.deepEqual(types(added), ["resample", "resample", "adetailer", "rtx"]);
  const moved = moveChainStage(added, added[0].id, 2);
  assert.deepEqual(types(moved), ["resample", "adetailer", "resample", "rtx"]);
  assert.deepEqual(types(added), ["resample", "resample", "adetailer", "rtx"]);
  const removed = removeChainStage(added, added[1].id);
  assert.equal(resampleUnits(removed).length, 1);
  // The two single-instance stages are not removable.
  assert.equal(removeChainStage(added, chainStage(added, "rtx").id), added);
  const updated = updateChainStage(added, added[0].id, { scale: 3 });
  assert.equal(updated[0].scale, 3);
  assert.equal(added[0].scale, 2);
});

test("the payload is the chain in order, without the ids the browser needs", () => {
  const chain = [
    unit({ upscaler: "x4.pth", scale: 2, denoise: 0.4, steps: 12, cfg: 4, seedMode: "fixed", seed: "17", resizeAlgorithm: "bicubic" }),
    { type: "adetailer", id: "ade", enabled: false, units: [] },
    { type: "rtx", id: "rtx", enabled: true, scale: 2, quality: "high" },
  ];
  const payload = resampleChainPayload(chain, { engine: "SD" });
  assert.deepEqual(payload.map((entry) => entry.type), ["resample", "adetailer", "rtx"]);
  assert.equal(payload[0].id, undefined);
  assert.equal(payload[0].expanded, undefined);
  assert.deepEqual(payload[0], {
    type: "resample", enabled: true, name: "", upscaler: "x4.pth", scale: 2, resize_algorithm: "bicubic",
    denoise: 0.4, steps: 12, cfg: 4, sampler: null, scheduler: null, seed_mode: "fixed", seed: "17",
    tile_size: 192, tile_overlap: 16, execution_mode: "full_frame", tile_width: "auto", tile_height: "auto",
    padding: 32, mask_blur: 8, seam_mode: "none", uniform_tiles: true, tiled_decode: true, exact_output: false,
  });
  assert.deepEqual(payload[2], { type: "rtx", enabled: true, scale: 2, quality: "high" });
});

test("a unit with no model sends null rather than an empty string", () => {
  const payload = resampleChainPayload([unit({ upscaler: "" })], { engine: "SD" });
  assert.equal(payload[0].upscaler, null);
});

test("USDU tiling is only sent for the engines that can run it", () => {
  const tiled = [unit({ executionMode: "usdu_tiled" })];
  assert.equal(resampleChainPayload(tiled, { engine: "Anima" })[0].execution_mode, "usdu_tiled");
  assert.equal(resampleChainPayload(tiled, { engine: "Flux" })[0].execution_mode, "full_frame");
});

test("an unfinished fixed seed falls back to inherit instead of reaching the service", () => {
  const payload = resampleChainPayload([unit({ seedMode: "fixed", seed: "" })], { engine: "SD" });
  assert.equal(payload[0].seed_mode, "inherit");
  assert.equal(payload[0].seed, undefined);
  // A gallery restore pins the unit to the seed the output recorded.
  const pinned = resampleChainPayload([unit({ seedMode: "random" })], {
    engine: "SD",
    seedOverrides: { [unit().id]: { seedMode: "fixed", seed: "9" } },
  });
  assert.equal(pinned[0].seed_mode, "random");
});

test("the run blocker names the unit and the first thing to fix", () => {
  const options = { baseSteps: 20, engine: "SD", nativeEngines: ["Anima", "Flux", "Flux2", "Krea2"] };
  assert.equal(resampleChainIssue([unit()], options), "");
  assert.match(resampleChainIssue([unit(), unit({ seedMode: "fixed", seed: "" })], options), /重采样 2/);
  assert.match(resampleChainIssue([unit({ denoise: 0.01, steps: 1 })], options), /有效步骤/);
  assert.match(resampleChainIssue([unit({ tileSize: 32, tileOverlap: 17 })], options), /分块重叠/);
  assert.match(
    resampleChainIssue([unit({ upscaler: "missing.pth" })], { ...options, upscalers: [{ id: "other.pth", compatible: true }] }),
    /超分模型不可用/,
  );
  // A native engine executes every requested step, so the multiplied count must not refuse it.
  assert.equal(resampleChainIssue([unit({ denoise: 0.01, steps: 1 })], { ...options, engine: "Anima" }), "");
});

test("banker rounding matches Python ties-to-even behaviour", () => {
  assert.equal(bankersRound(0.5), 0);
  assert.equal(bankersRound(-0.5), 0);
  assert.equal(bankersRound(2.5), 2);
  assert.equal(bankersRound(3.5), 4);
  assert.equal(bankersRound(-2.5), -2);
  assert.equal(bankersRound(-3.5), -4);
  assert.equal(bankersRound(2.500001), 3);
  assert.equal(bankersRound(2.499999), 2);
  const ties = resampleChainPlan({ width: 10, height: 14 }, [{ type: "rtx", id: "rtx", enabled: true, scale: 2 }], "SD");
  assert.deepEqual({ width: ties.width, height: ties.height }, { width: 16, height: 32 });
});
