import assert from "node:assert/strict";
import test from "node:test";

import {
  ADETAILER_UNIT_LIMIT,
  activeADetailerUnits,
  adetailerPageIndex,
  adetailerPayload,
  adetailerStepUnitId,
  adetailerStageIssue,
  adetailerSummary,
  adetailerTotalSteps,
  adetailerUnitLabel,
  adetailerUnitSteps,
  normalizeADetailerStage,
  normalizeADetailerUnits,
} from "../src/adetailer-units.js";

test("a stage stored before units existed becomes unit one", () => {
  // The flat shape *was* the unit, and its `enabled` described the stage. Reading
  // it onto the unit would produce a unit that stays switched off after the user
  // turns the stage back on.
  const migrated = normalizeADetailerStage({
    enabled: false,
    expanded: true,
    detector: "face_yolov8n.pt",
    denoise: 0.55,
    prompt: "detailed face",
  });

  assert.equal(migrated.enabled, false);
  assert.equal(migrated.expanded, true);
  assert.equal(migrated.units.length, ADETAILER_UNIT_LIMIT);
  assert.equal(migrated.units[0].enabled, true);
  assert.equal(migrated.units[0].detector, "face_yolov8n.pt");
  assert.equal(migrated.units[0].denoise, 0.55);
  assert.equal(migrated.units[0].prompt, "detailed face");

  // Every slot exists from the start, so nothing has to be added by hand; only
  // the first is on, because six enabled passes per render is not a default.
  const fresh = normalizeADetailerStage(undefined);
  assert.equal(fresh.units.length, ADETAILER_UNIT_LIMIT);
  assert.deepEqual(fresh.units.map((unit) => unit.enabled), [true, false, false, false, false, false]);
  assert.deepEqual(normalizeADetailerStage({ units: [] }).units.map((unit) => unit.enabled), fresh.units.map((unit) => unit.enabled));
  // A stored pair keeps its own switches and the slots after it arrive off.
  const stored = normalizeADetailerStage({ units: [{ detector: "face.pt" }, { detector: "hand.pt", enabled: false }] });
  assert.deepEqual(stored.units.map((unit) => unit.enabled), [true, false, false, false, false, false]);
  assert.deepEqual(stored.units.map((unit) => unit.detector), ["face.pt", "hand.pt", "", "", "", ""]);
});

test("normalising is idempotent, so an unchanged stage reports no change", () => {
  const stage = normalizeADetailerStage({ units: [{ detector: "face.pt" }, { detector: "hand.pt" }] });
  assert.deepEqual(normalizeADetailerStage(stage), stage);
  assert.deepEqual(normalizeADetailerStage(JSON.parse(JSON.stringify(stage))), stage);
  assert.deepEqual(stage.units.map((unit) => unit.id), ["ade-1", "ade-2", "ade-3", "ade-4", "ade-5", "ade-6"]);

  // Ids address a unit so a control belongs to it rather than to a position;
  // two stored lists merged into one must not collide on a React key.
  const collided = normalizeADetailerUnits({ units: [{ id: "ade-1", detector: "a" }, { id: "ade-1", detector: "b" }] });
  assert.equal(new Set(collided.map((unit) => unit.id)).size, ADETAILER_UNIT_LIMIT);
  // An id already spoken for is skipped rather than reused.
  const partial = normalizeADetailerUnits({ units: [{ id: "ade-2", detector: "a" }, { detector: "b" }] });
  assert.deepEqual(partial.map((unit) => unit.id), ["ade-2", "ade-1", "ade-3", "ade-4", "ade-5", "ade-6"]);
});

test("out-of-range settings are repaired rather than refused", () => {
  const [unit] = normalizeADetailerUnits({ units: [{
    detector: 42,
    confidence: 9,
    maxDetections: 99,
    // An inverted range is a range, not an error.
    maskMinRatio: 0.8,
    maskMaxRatio: 0.2,
    dilateErode: -900,
    maskBlur: 4000,
    padding: -5,
    denoise: 0,
    steps: 0,
    cfg: 90,
    prompt: 7,
  }] });

  assert.equal(unit.detector, "");
  assert.equal(unit.confidence, 1);
  assert.equal(unit.maxDetections, 8);
  assert.equal(unit.maskMinRatio, 0.2);
  assert.equal(unit.maskMaxRatio, 0.8);
  assert.equal(unit.dilateErode, -128);
  assert.equal(unit.maskBlur, 64);
  assert.equal(unit.padding, 0);
  // Zero denoise would ask the pass to change nothing.
  assert.equal(unit.denoise, 0.05);
  assert.equal(unit.steps, 1);
  assert.equal(unit.cfg, 30);
  assert.equal(unit.prompt, "");

  // The ceiling is about not buying an hour of GPU time by accident.
  const many = normalizeADetailerUnits({ units: Array.from({ length: 20 }, (_, index) => ({ detector: `m${index}.pt` })) });
  assert.equal(many.length, ADETAILER_UNIT_LIMIT);
  assert.deepEqual(many.map((unit) => unit.detector), ["m0.pt", "m1.pt", "m2.pt", "m3.pt", "m4.pt", "m5.pt"]);
});

test("the wire carries the run plan, in the order it will run", () => {
  const stage = normalizeADetailerStage({
    enabled: true,
    units: [
      { detector: "face.pt", denoise: 0.45, prompt: "  a face  ", negativePrompt: "  blur  " },
      { detector: "off.pt", enabled: false },
      { detector: "hand.pt", denoise: 0.3, useCfg: true, cfg: 4 },
    ],
  });

  const payload = adetailerPayload(stage);
  assert.equal(payload.enabled, true);
  // A unit switched off in the editor has no representation on the other side.
  assert.deepEqual(payload.units.map((unit) => unit.detector), ["face.pt", "hand.pt"]);
  assert.deepEqual(payload.units.map((unit) => unit.denoise), [0.45, 0.3]);
  assert.equal(payload.units[0].prompt, "a face");
  assert.equal(payload.units[0].negative_prompt, "blur");
  assert.equal(payload.units[1].use_cfg, true);
  assert.equal(payload.units[1].cfg, 4);
  // Snake case throughout, matching `ADetailerUnitInput`.
  assert.deepEqual(Object.keys(payload.units[0]).filter((key) => /[A-Z]/.test(key)), []);
  assert.equal(activeADetailerUnits(stage).length, 2);
});

test("a switched-off stage sends no units, so it cannot refuse the generation", () => {
  // The default first unit is enabled and has no detector yet. Sending it beside `enabled: false`
  // handed the request validator a unit that cannot run, and the whole generation was refused with
  // "ADetailer detector is required" for a stage the user never turned on.
  const untouched = normalizeADetailerStage({});
  assert.equal(untouched.enabled, false);
  assert.equal(untouched.units[0].enabled, true);
  assert.equal(untouched.units[0].detector, "");

  const payload = adetailerPayload(untouched, "SD");
  assert.equal(payload.enabled, false);
  assert.deepEqual(payload.units, []);

  // Turning the stage on sends the run plan again, unchanged.
  const configured = normalizeADetailerStage({ enabled: true, units: [{ detector: "face.pt" }] });
  assert.deepEqual(adetailerPayload(configured, "SD").units.map((unit) => unit.detector), ["face.pt"]);
});

test("a stage that cannot run says which unit is at fault", () => {
  const steps = 20;
  const stage = (units) => normalizeADetailerStage({ enabled: true, units });

  assert.equal(adetailerStageIssue(stage([{ detector: "face.pt" }]), steps), "");
  assert.match(adetailerStageIssue(stage([{ detector: "" }]), steps), /ADetailer 1 尚未选择检测模型/);
  assert.match(adetailerStageIssue(stage([{ detector: "face.pt" }, { detector: "" }]), steps), /ADetailer 2 尚未选择检测模型/);
  // The number is the *slot*, which is the page number the user can open. Counting
  // only the enabled units would send them to page 2 for a fault on page 4.
  assert.match(
    adetailerStageIssue(stage([{ detector: "face.pt" }, { enabled: false }, { enabled: false }, { detector: "", enabled: true }]), steps),
    /ADetailer 4 尚未选择检测模型/,
  );
  assert.match(
    adetailerStageIssue(stage([{ detector: "face.pt" }, { detector: "gone.pt" }]), steps, (detector) => detector === "face.pt"),
    /ADetailer 2 的检测模型不可用/,
  );
  // Diffusers runs `int(steps × denoise)` updates, so a low denoise can round to
  // nothing and the pass would fail where it would have started working.
  assert.match(adetailerStageIssue(stage([{ detector: "face.pt", denoise: 0.05 }]), 10), /有效步数至少需要 1 步/);
  assert.match(adetailerStageIssue(stage([{ detector: "face.pt", enabled: false }]), steps), /至少需要启用一个/);
});

test("step arithmetic and the labels a user reads", () => {
  assert.equal(adetailerUnitSteps({ denoise: 0.4 }, 20), 8);
  assert.equal(adetailerUnitSteps({ denoise: 0.4, useSteps: true, steps: 30 }, 20), 12);
  assert.equal(adetailerUnitSteps({ denoise: 0.05 }, 10), 0);

  const stage = normalizeADetailerStage({ enabled: true, units: [{ detector: "a.pt", denoise: 0.5 }, { detector: "b.pt", denoise: 0.25 }] });
  assert.equal(adetailerTotalSteps(stage, 20), 15);

  // Native Anima refinement keeps the last `steps + 1` sigmas of a longer schedule, so it executes
  // every requested step whatever the denoise. Multiplying there promised 2 steps and delivered 20 —
  // which is how the job's counter climbed past the total it was reporting.
  assert.equal(adetailerUnitSteps({ denoise: 0.1 }, 20, "Anima"), 20);
  assert.equal(adetailerUnitSteps({ denoise: 0.1, useSteps: true, steps: 30 }, 20, "Anima"), 30);
  assert.equal(adetailerTotalSteps(stage, 20, "Anima"), 40);
  // A denoise that rounds to nothing is a blocker on Diffusers and a non-issue on Anima, because
  // there the pass would have run its 10 steps.
  const faint = normalizeADetailerStage({ enabled: true, units: [{ detector: "face.pt", denoise: 0.05 }] });
  assert.match(adetailerStageIssue(faint, 10), /有效步数至少需要 1 步/);
  assert.equal(adetailerStageIssue(faint, 10, undefined, "Anima"), "");

  assert.equal(adetailerUnitLabel(0, 1), "ADetailer");
  assert.equal(adetailerUnitLabel(1, 3), "ADetailer 2");
  assert.equal(adetailerSummary(normalizeADetailerStage({ enabled: true, units: [{ detector: "sub/face.pt" }] })), "face.pt · 0.30");
  assert.match(adetailerSummary(stage), /a\.pt 等 2 个模型依次执行/);
  assert.equal(adetailerSummary(normalizeADetailerStage({ enabled: true, units: [{ detector: "a.pt", enabled: false }] })), "未启用单元");
});

test("the pager tracks a unit rather than a position", () => {
  const units = normalizeADetailerUnits({ units: [{ detector: "a.pt" }, { detector: "b.pt" }] });
  const [first, second] = units;

  assert.equal(adetailerPageIndex(units, second.id), 1);
  // A page id that names no unit — a stale one, or none opened yet — has to
  // resolve to a page that exists, so a unit is always on screen.
  assert.equal(adetailerPageIndex(units, "ade-gone"), 0);
  assert.equal(adetailerPageIndex(units, ""), 0);
  assert.equal(adetailerPageIndex([], first.id), 0);
});

test("the arrows stop at the first and last slot", () => {
  const units = normalizeADetailerUnits({ units: [{ detector: "a.pt" }] });
  const first = units[0];
  const last = units[ADETAILER_UNIT_LIMIT - 1];

  assert.equal(adetailerStepUnitId(units, first.id, 1), units[1].id);
  assert.equal(adetailerStepUnitId(units, units[1].id, -1), first.id);
  assert.equal(adetailerStepUnitId(units, first.id, -1), first.id);
  assert.equal(adetailerStepUnitId(units, last.id, 1), last.id);
  assert.equal(adetailerStepUnitId([], first.id, 1), "");
});
