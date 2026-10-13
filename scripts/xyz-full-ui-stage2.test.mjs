import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createXYZDraft, inspectXYZDrafts, syncXYZBaseline, XYZ_CAPABILITIES } from "../src/xyz-draft.js";
import { xyzWireDraft } from "../src/xyz-prompt-source.js";
import { captureXYZPostprocessSource, editXYZPostprocess, xyzPostprocessSource } from "../src/xyz-postprocess-source.js";
import { createResampleUnit, createAdetailerStage, createRtxStage, resampleChainPayload, moveChainStage } from "../src/resample-chain.js";

function fixture(engine = "SD") {
  const chain = [createResampleUnit({ id: "r1", enabled: true, seedMode: "random", scale: 1.5 }, { engine }), createAdetailerStage({ id: "ade", enabled: false }), createRtxStage({ id: "rtx", enabled: false })];
  return createXYZDraft(captureXYZPostprocessSource({ parameters: { engine, prompt: "exact", negative_prompt: "", pag: {}, width: 1024, height: 1408, seed: "18446744073709551615", postprocess_chain: resampleChainPayload(chain, { engine }) }, loras: [], group_snapshots: [], seed_mode: "fixed" }, chain));
}
test("chain UI capture and wire never normalize the saved stage values or write the owner", () => {
  for (const engine of ["SD", "iL", "Anima", "Flux", "Flux2", "Krea2"]) {
    const draft = fixture(engine), before = structuredClone(draft), source = xyzPostprocessSource(draft);
    source.chain[0].expanded = false;
    const edited = editXYZPostprocess(draft, source.chain);
    assert.deepEqual(edited.baseline.parameters.postprocess_chain, draft.baseline.parameters.postprocess_chain);
    assert.deepEqual(draft, before);
    assert.equal(inspectXYZDrafts({ schema_version: 1, by_engine: { [engine]: edited } }, "xyz_plot").fatal, false);
    assert.ok(!Object.hasOwn(xyzWireDraft(edited).baseline, "postprocess_source"));
  }
});
test("reorder keeps UI stage ids, random/fixed/inherit seeds and off/on disabled ADE slots", () => {
  const draft = fixture(), chain = xyzPostprocessSource(draft).chain;
  chain[1].units[0].enabled = false;
  chain[1].units[4] = { ...chain[1].units[4], enabled: true, detector: "hands.pt", prompt: "slot five", cfg: 8.25 };
  let edited = editXYZPostprocess(draft, moveChainStage(chain, "r1", 2));
  assert.deepEqual(edited.baseline.postprocess_source.chain.map(s => s.id), ["ade", "rtx", "r1"]);
  assert.equal(edited.baseline.parameters.postprocess_chain[2].seed_mode, "random");
  assert.equal(edited.baseline.parameters.postprocess_chain[0].units[0].prompt, "slot five");
  let next = xyzPostprocessSource(edited).chain; next[0].units[4].enabled = false;
  edited = editXYZPostprocess(edited, next);
  assert.equal(edited.baseline.parameters.postprocess_chain[0].units.length, 0);
  assert.equal(edited.baseline.postprocess_source.chain[0].units[4].cfg, 8.25);
  next = xyzPostprocessSource(edited).chain; next[0].units[4].enabled = true; next[2].seedMode = "fixed"; next[2].seed = "18446744073709551615";
  edited = editXYZPostprocess(edited, next);
  assert.equal(edited.baseline.parameters.postprocess_chain[0].units[0].cfg, 8.25);
  assert.equal(edited.baseline.parameters.postprocess_chain[2].seed, "18446744073709551615");
  next = xyzPostprocessSource(edited).chain; next[2].seedMode = "inherit";
  edited = editXYZPostprocess(edited, next); assert.ok(!Object.hasOwn(edited.baseline.parameters.postprocess_chain[2], "seed"));
});
test("legacy missing/unknown wire fields and invalid raw values survive unrelated stage edits", () => {
  const draft = fixture(); delete draft.baseline.postprocess_source;
  draft.baseline.parameters.postprocess_chain[0].scale = "bad";
  draft.baseline.parameters.postprocess_chain[0].future_field = { raw: "untouched" };
  draft.baseline.parameters.postprocess_chain.push({ type: "unknown", raw: "unknown future stage" });
  const before = structuredClone(draft), source = xyzPostprocessSource(draft);
  assert.equal(source.chain[0].scale, "bad"); assert.equal(source.chain.at(-1).legacy, true);
  source.chain[0].expanded = true; source.chain[1].expanded = true;
  const edited = editXYZPostprocess(draft, source.chain);
  assert.deepEqual(edited.baseline.parameters.postprocess_chain, before.baseline.parameters.postprocess_chain);
  assert.deepEqual(draft, before);
  assert.equal(inspectXYZDrafts({ schema_version: 1, by_engine: { SD: edited } }, "xyz_plot").fatal, false, "UI legacy warnings can persist; backend still validates original unknown wire stage");
});
test("empty/invalid fixed resample seed stays fixed/raw; no inherited fallback", () => {
  const draft = fixture(), chain = xyzPostprocessSource(draft).chain;
  chain[0].seedMode = "fixed"; chain[0].seed = "not a seed";
  const edited = editXYZPostprocess(draft, chain);
  assert.equal(edited.baseline.parameters.postprocess_chain[0].seed_mode, "fixed");
  assert.equal(edited.baseline.parameters.postprocess_chain[0].seed, "not a seed");
});
test("explicit post Sync updates matching UI source only, frozen older draft and axes unchanged", () => {
  const draft = fixture(), baseline = fixture().baseline;
  draft.axes.x = { field: "steps", values: [{ value: "20", label: "exact" }], range: null };
  baseline.postprocess_source.chain[0].scale = 2;
  baseline.parameters.postprocess_chain[0].scale = 2;
  const before = structuredClone(draft), synced = syncXYZBaseline(draft, baseline, ["postprocess"]);
  assert.equal(synced.baseline.postprocess_source.chain[0].scale, 2);
  assert.deepEqual(synced.axes, before.axes); assert.deepEqual(draft, before);
  delete baseline.postprocess_source;
  assert.ok(!Object.hasOwn(syncXYZBaseline(synced, baseline, ["postprocess"]).baseline, "postprocess_source"));
});
test("unknown post-source versions fail closed instead of callback swallowing", () => {
  const draft = fixture(); draft.baseline.postprocess_source.schema_version = 2;
  const raw = { schema_version: 1, by_engine: { SD: draft } }, result = inspectXYZDrafts(raw, "xyz_plot");
  assert.equal(result.fatal, true); assert.deepEqual(result.raw, raw);
});
test("ADE editing merges only changed per-unit fields, keeping ordinary frozen wire semantics", () => {
  const draft = fixture(), chain = xyzPostprocessSource(draft).chain;
  // A native mother can keep UI negative text while freezing an empty ordinary wire negative.
  chain[1].units[0].negativePrompt = "UI-only saved negative";
  draft.baseline.postprocess_source.chain = structuredClone(chain);
  draft.baseline.parameters.postprocess_chain[1].units = [{ ...resampleChainPayload([{ ...chain[1], enabled: true }], { engine: "SD" })[0].units[0], negative_prompt: "" }];
  draft.baseline.parameters.postprocess_chain[1].units[0].future_raw = { exact: "unknown wire kept for validation" };
  const before = structuredClone(draft);chain[1].units[0].cfg = 12;
  const edited = editXYZPostprocess(draft, chain), wire = edited.baseline.parameters.postprocess_chain[1].units[0];
  assert.equal(wire.cfg, 12);assert.equal(wire.negative_prompt, "");assert.deepEqual(wire.future_raw,before.baseline.parameters.postprocess_chain[1].units[0].future_raw);
  assert.deepEqual(draft,before);
});
test("first ADE enable materializes captured disabled UI slots even when mother wire units were empty", () => {
  for(const engine of ["SD","iL","Anima","Flux","Flux2","Krea2"]) {
    const draft=fixture(engine),chain=xyzPostprocessSource(draft).chain;
    assert.deepEqual(draft.baseline.parameters.postprocess_chain[1].units,[]);
    chain[1].units[0].detector="fixture-face.pt";chain[1].units[0].negativePrompt="stored native negative";
    // Simulate capturing a fully configured but switched-off ordinary stage, then only enable it.
    draft.baseline.postprocess_source.chain=structuredClone(chain);chain[1].enabled=true;
    const edited=editXYZPostprocess(draft,chain);
    assert.equal(edited.baseline.parameters.postprocess_chain[1].units[0].detector,"fixture-face.pt");
    assert.equal(edited.baseline.parameters.postprocess_chain[1].units[0].negative_prompt,XYZ_CAPABILITIES.engines[engine].negative_prompt?"stored native negative":"");
    assert.equal(edited.baseline.postprocess_source.chain[1].units[0].negativePrompt,"stored native negative");
  }
});
test("current actual owners share mother sampling/ADE/RTX and place unique canvas right", async () => {
  const read = (name) => readFile(new URL(`../src/${name}`, import.meta.url), "utf8");
  const app = await read("App.jsx"), sidebar = await read("XYZBaselineSidebar.jsx"), workspace = await read("XYZPlotWorkspace.jsx"), axis = await read("XYZAxisEditor.jsx");
  assert.match(app, /<SamplerPair/); assert.match(sidebar, /<SamplerPair/);
  assert.match(app, /<ADetailerUnitControls/); assert.match(app, /<RTXControls/);
  assert.match(sidebar, /<XYZPostprocessPanel/); assert.doesNotMatch(sidebar, /<SizeGrid|<details className="xyz-sidebar-canvas"/);
  assert.match(workspace, /<XYZCanvasPanel/); assert.doesNotMatch(workspace, /<strong>XYZ 对比生成|xyz-baseline-summary/);
  assert.match(workspace, /JSON\.stringify\(xyzWireDraft\(draft\)\)/, "retry/inflight identity excludes the stripped UI-only provenance, not actual axes/parameters/policy");
  assert.doesNotMatch(axis, /axisKey !== "z"|if \(!axis\) return <details/);
});
