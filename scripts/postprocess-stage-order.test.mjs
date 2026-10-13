import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createAdetailerStage, createResampleUnit, createRtxStage, moveChainStage, resampleChainPayload } from "../src/resample-chain.js";
import { IMAGE_TO_IMAGE_DEFAULTS, imageToImageRequestBody } from "../src/image-to-image.js";

const read = (name) => readFile(new URL(`../src/${name}`, import.meta.url), "utf8");

test("Basic and I2I render shared reorder actions for ADetailer/RTX and pass their execution locks", async () => {
  const app = await read("App.jsx");
  const panel = await read("ResampleChainPanel.jsx");
  const image = await read("ImageToImagePage.jsx");
  const renderStages = app.slice(app.indexOf("renderStage={(stage, index, actions) => (stage.type === \"adetailer\""), app.indexOf("</ResampleChainPanel>", app.indexOf("renderStage={(stage, index, actions) => (stage.type === \"adetailer\"")));
  const imageStages = image.slice(image.indexOf("renderStage={(stage, index, actions) => (stage.type === \"adetailer\""), image.indexOf("</ResampleChainPanel>", image.indexOf("renderStage={(stage, index, actions) => (stage.type === \"adetailer\"")));

  assert.equal((renderStages.match(/actions=\{actions\}/g) || []).length, 2, "both Basic auxiliary stage headers receive the existing order controls");
  assert.equal((imageStages.match(/>\{actions\}<\/StageHeader>/g) || []).length, 2, "both I2I auxiliary stage headers receive the shared order controls");
  assert.match(app, /actionsLocked=\{hiresControlsLocked \|\| status === "running"\}/);
  assert.match(image, /running=\{editingLocked\} actionsLocked=\{running\}/);
  assert.match(image, /actionsLocked=\{actionsLocked \|\| running\}/);
  assert.match(panel, /actionsLocked = locked/);
  assert.match(panel, /const canAdd = !actionsLocked && !addDisabled/);
  assert.match(panel, /disabled=\{!canMoveUp\}/);
  assert.match(panel, /disabled=\{!canMoveDown\}/);
  assert.match(panel, /disabled=\{actionsLocked\}/);
  assert.match(panel, /if \(!canAdd\) return;/);
  assert.match(panel, /if \(!canMoveUp\) return;/);
  assert.match(panel, /if \(!canMoveDown\) return;/);
  assert.match(panel, /if \(actionsLocked\) return;/);
  assert.match(panel, /onClick=\{addUnit\}/);
});

test("Basic stage moves preserve settings and request payload follows the visible chain", () => {
  const original = [
    createResampleUnit({ id: "resample-A", enabled: false, expanded: true, name: "kept name", scale: 1.7, seedMode: "fixed", seed: "18446744073709551615" }, { engine: "SD" }),
    createAdetailerStage({ id: "adetailer-A", enabled: true, expanded: true, units: [{ id: "ade-unit-A", enabled: false, detector: "kept-detector.pt", prompt: "kept prompt" }] }),
    createRtxStage({ id: "rtx-A", enabled: false, expanded: true, scale: 1.35, quality: "high" }),
  ];
  const before = structuredClone(original);
  const movedRtx = moveChainStage(original, "rtx-A", -1);
  assert.deepEqual(movedRtx.map((stage) => stage.id), ["resample-A", "rtx-A", "adetailer-A"]);
  assert.deepEqual(movedRtx.map((stage) => stage.type), ["resample", "rtx", "adetailer"]);
  assert.deepEqual(resampleChainPayload(movedRtx, { engine: "SD" }).map((stage) => stage.type), ["resample", "rtx", "adetailer"]);
  assert.deepEqual(
    Object.fromEntries(movedRtx.map((stage) => [stage.id, [stage.enabled, stage.expanded]])),
    Object.fromEntries(before.map((stage) => [stage.id, [stage.enabled, stage.expanded]])),
  );
  assert.deepEqual(movedRtx[0], before[0]);
  assert.deepEqual(movedRtx[1], before[2]);
  assert.deepEqual(movedRtx[2], before[1]);

  const movedAdetailer = moveChainStage(movedRtx, "adetailer-A", -1);
  assert.deepEqual(movedAdetailer.map((stage) => stage.id), ["resample-A", "adetailer-A", "rtx-A"]);
  assert.deepEqual(resampleChainPayload(movedAdetailer, { engine: "SD" }).map((stage) => stage.type), ["resample", "adetailer", "rtx"]);
  assert.deepEqual(movedAdetailer[1].units, before[1].units);
  assert.deepEqual(movedAdetailer[2], before[2]);
  assert.deepEqual(original, before, "reordering never mutates the caller's configured chain");
});

test("Basic and I2I request payloads preserve four distinct stage-order sequences", () => {
  const original = [
    createResampleUnit({ id: "resample-A", enabled: false, expanded: true, scale: 1.7, seedMode: "fixed", seed: "18446744073709551615" }, { engine: "SD" }),
    createAdetailerStage({ id: "adetailer-A", enabled: true, units: [{ id: "ade-unit-A", enabled: false, detector: "kept-detector.pt", prompt: "kept prompt" }] }),
    createRtxStage({ id: "rtx-A", enabled: false, scale: 1.35, quality: "high" }),
  ];
  const beforeById = Object.fromEntries(original.map((stage) => [stage.id, stage]));
  const sequences = [
    original,
    moveChainStage(original, "rtx-A", -1),
    moveChainStage(moveChainStage(original, "rtx-A", -1), "resample-A", 1),
    moveChainStage(moveChainStage(moveChainStage(original, "rtx-A", -1), "resample-A", 1), "adetailer-A", -1),
    moveChainStage(moveChainStage(moveChainStage(moveChainStage(original, "rtx-A", -1), "resample-A", 1), "adetailer-A", -1), "rtx-A", 1),
  ];
  const expected = [
    ["resample", "adetailer", "rtx"],
    ["resample", "rtx", "adetailer"],
    ["rtx", "resample", "adetailer"],
    ["rtx", "adetailer", "resample"],
    ["adetailer", "rtx", "resample"],
  ];

  sequences.forEach((chain, index) => {
    assert.deepEqual(chain.map((stage) => stage.type), expected[index]);
    assert.deepEqual(resampleChainPayload(chain, { engine: "SD" }).map((stage) => stage.type), expected[index]);
    assert.deepEqual(
      Object.fromEntries(chain.map((stage) => [stage.id, stage])),
      beforeById,
      "ordering keeps each stage, unit ID, enable state and parameter object intact",
    );
    const i2iBody = imageToImageRequestBody({
      engine: "SD",
      checkpoint: "fixture-model.safetensors",
      source: { name: "fixture.png", width: 512, height: 512 },
      settings: { ...IMAGE_TO_IMAGE_DEFAULTS, postprocessChain: chain },
      seed: "23",
    });
    assert.deepEqual(i2iBody.postprocess_chain.map((stage) => stage.type), expected[index]);
  });
  assert.deepEqual(original, sequences[0], "test setup remains immutable across all orderings");
});
