import test from "node:test";
import assert from "node:assert/strict";
import { readFile, mkdtemp, mkdir, rm } from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { GENERATION_MODES, stepGenerationMode } from "../src/generation-modes.js";
import { createXYZDraft, groupXYZProfile, emptyXYZDrafts, inspectXYZDrafts, withXYZDraft, syncXYZBaseline, decimalRangeValues, xyzLocalSummary, xyzAxisOptions } from "../src/xyz-draft.js";
import { rebasePublishedWorkspace } from "../src/lora-library-client.js";
import { LoraPublishedStore } from "./lora-published-store.mjs";

const baseline = (engine = "SD") => ({ parameters: { engine, prompt: "group\n\ncat, sky", negative_prompt: "bad", checkpoint: "model.safetensors", steps: 20, cfg: 6.5, width: 512, height: 512, seed: "18446744073709551615", sampler: "euler", scheduler: "simple", guidance: "none", pag: { scale: 0.3, applied_layers: "mid" }, postprocess_chain: [] }, seed_mode: "random", prompt_without_groups: "cat, sky", group_prompt: "group", group_snapshots: [], loras: [], library_revision: 0 });

test("registry cycles both directions; XYZ lazy, executor state truthful for the current phase", () => {
  // QPlan2/P2 registers a third mode beside Basic and XYZ; the cycle is basic -> xyz_plot -> lora_mix_search -> basic.
  assert.deepEqual(GENERATION_MODES.map((mode) => mode.id), ["basic", "xyz_plot", "lora_mix_search"]);
  assert.equal(stepGenerationMode("basic", 1), "xyz_plot");
  assert.equal(stepGenerationMode("xyz_plot", 1), "lora_mix_search");
  assert.equal(stepGenerationMode("lora_mix_search", 1), "basic");
  assert.equal(stepGenerationMode("xyz_plot", -1), "basic");
  assert.equal(stepGenerationMode("basic", -1), "lora_mix_search");
  assert.equal(stepGenerationMode("lora_mix_search", -1), "xyz_plot");
  // Phase B made the service executor real (sequential); see xyz-phase-b.test.mjs.
  assert.equal(GENERATION_MODES[1].executorReady, true);
  assert.match(GENERATION_MODES[1].phase, /^E /); // Phase E: Viewer/Gallery/apply integration
  assert.equal(typeof GENERATION_MODES[1].component, "function");
});
test("draft captures clone; six engines isolated; no basic settings writes", () => {
  const source = baseline(), before = structuredClone(source);
  const draft = createXYZDraft(source); draft.baseline.parameters.prompt = "XYZ only";
  let map = emptyXYZDrafts();
  for (const engine of ["SD", "iL", "Anima", "Flux", "Flux2", "Krea2"]) map = withXYZDraft(map, engine, createXYZDraft(baseline(engine)));
  assert.deepEqual(source, before);
  const next = withXYZDraft(map, "SD", draft);
  assert.equal(next.by_engine.SD.baseline.parameters.prompt, "XYZ only");
  assert.equal(next.by_engine.iL.baseline.parameters.prompt, before.parameters.prompt);
  assert.deepEqual(inspectXYZDrafts(JSON.parse(JSON.stringify(next)), "xyz_plot").container, next);
  assert.equal(draft.inherit_postprocess, false);
  assert.equal(draft.requested_concurrency, 1);
});
test("missing legacy state migrates empty; future/corrupt/plan preserved fail closed", () => {
  assert.deepEqual(inspectXYZDrafts(undefined).container, emptyXYZDrafts());
  for (const raw of [null, [], { schema_version: 2, by_engine: {} }, { schema_version: 1, by_engine: { SD: { ...createXYZDraft(baseline()), kind: "xyz_plan" } } }]) {
    const result = inspectXYZDrafts(raw);
    assert.equal(result.fatal, true); assert.deepEqual(result.raw, raw);
  }
  assert.equal(inspectXYZDrafts(undefined, "unknown-mode").fatal, true);
});
test("explicit sync preserves axes and unrelated baseline fields without aliasing", () => {
  const draft = createXYZDraft(baseline()); draft.axes.x = { field: "steps", values: [{ value: "1", label: "one" }] };
  const before = structuredClone(draft), fresh = baseline(); fresh.parameters.steps = 60; fresh.parameters.prompt = "new words";
  const result = syncXYZBaseline(draft, fresh, ["prompts"]);
  assert.equal(result.baseline.parameters.prompt, "new words");
  assert.equal(result.baseline.parameters.steps, 20);
  assert.deepEqual(result.axes, before.axes); assert.deepEqual(draft, before);
  assert.throws(() => syncXYZBaseline(draft, baseline("Flux"), ["models"]));
  assert.throws(() => syncXYZBaseline(draft, fresh, ["unknown"]));
});
test("group profile is a snapshot and preset words default fixed, explicit opt-in", () => {
  const group = { id: "g-one", name: "group", presetPrompt: "words", members: [{ instanceId: "instance", weight: 1 }] };
  const profile = groupXYZProfile(group);
  assert.equal(profile.includePresetText, false);
  assert.equal(groupXYZProfile(group, { includePresetText: true }).includePresetText, true);
  profile.members[0].weight = 2;
  assert.equal(group.members[0].weight, 1);
});
test("exact Decimal frontend range counts and bounds; comma Prompt not split", () => {
  assert.deepEqual(decimalRangeValues({ start: "0.1", end: "0.3", step: "0.1" }), ["0.1", "0.2", "0.3"]);
  assert.deepEqual(decimalRangeValues({ start: "0.3", end: "0.1", step: "-0.1" }), ["0.3", "0.2", "0.1"]);
  for (const range of [{ start: "0", end: "1", step: "0" }, { start: "0", end: "1", step: "-1" }, { start: "0", end: "1000", step: "1" }]) assert.throws(() => decimalRangeValues(range));
  const draft = createXYZDraft(baseline()); draft.axes.x = { field: "prompt", values: [{ value: "cat, blue, sky" }] }; draft.repeat_count = 2; draft.requested_concurrency = 10;
  const summary = xyzLocalSummary(draft);
  assert.equal(summary.total, 2); assert.equal(summary.authoritative, false); assert.equal(summary.seedsResolved, false);
  draft.axes.x.values = []; assert.ok(xyzLocalSummary(draft).error);
});
test("axis visibility follows six runtime capabilities not health flags", () => {
  for (const engine of ["SD", "iL", "Anima", "Flux", "Flux2", "Krea2"]) {
    const fields = xyzAxisOptions(engine).map((item) => item.value);
    assert.equal(fields.includes("clip_skip"), ["SD", "iL"].includes(engine));
    assert.equal(fields.includes("negative_prompt"), !["Flux", "Flux2"].includes(engine));
    assert.equal(fields.includes("pag.scale"), ["SD", "iL", "Anima"].includes(engine));
    assert.equal(fields.includes("denoise"), false);
    assert.equal(fields.includes("lora_weight"), true, `${engine} keeps the LoRA weight axis`);
    for (const retired of ["lora_model", "lora_profile", "lora_multiplier"]) assert.equal(fields.includes(retired), false, `${engine}: ${retired} stays retired`);
  }
});
test("P10 late rebase preserves locally edited draft while refreshing basic", () => {
  const map = withXYZDraft(emptyXYZDrafts(), "SD", createXYZDraft(baseline()));
  const base = { workspace: { positive: "old", xyzByEngine: map } }, local = structuredClone(base), fresh = structuredClone(base);
  local.workspace.xyzByEngine.by_engine.SD.axes.x = { field: "prompt", values: [{ value: "my, experiment" }] };
  fresh.workspace.positive = "published";
  const merged = rebasePublishedWorkspace(base, local, fresh);
  assert.deepEqual(merged.workspace.xyzByEngine, local.workspace.xyzByEngine);
  assert.equal(merged.workspace.positive, "published");
  assert.equal(merged.workspace.xyzByEngine.by_engine.SD.baseline.parameters.prompt, base.workspace.xyzByEngine.by_engine.SD.baseline.parameters.prompt);
});
test("actual published-store CAS retains XYZ across stale basic-only autosave", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "xyz-phase-a-store-"));
  try {
    await mkdir(path.join(root, "state"));
    const store = new LoraPublishedStore({ stateDirectory: path.join(root, "state") });
    const initial = await store.save("uiState", { workspace: { positive: "old" } });
    const next = { workspace: { positive: "old", xyzByEngine: withXYZDraft(emptyXYZDrafts(), "SD", createXYZDraft(baseline())) } };
    await store.save("uiState", next, { revision: 0, version: initial.version });
    const result = await store.save("uiState", { workspace: { positive: "late basic" } }, { revision: 0, version: initial.version });
    assert.deepEqual(result.value.workspace.xyzByEngine, next.workspace.xyzByEngine);
    assert.equal(result.value.workspace.positive, "late basic");
    assert.equal((await store.read("uiState")).value.workspace.xyzByEngine.by_engine.SD.kind, "xyz_draft");
  } finally { await rm(root, { recursive: true, force: true }); }
});
test("App selector placement, lazy workspace, shortcut guard and persistence gates", async () => {
  const app = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8"), workspace = await readFile(new URL("../src/XYZPlotWorkspace.jsx", import.meta.url), "utf8");
  assert.ok(app.indexOf("<GenerationModeSwitch") > app.indexOf('className="model-segment"'));
  assert.ok(app.indexOf("<GenerationModeSwitch") < app.indexOf("<ModelAssetPicker"));
  assert.match(app, /const XYZPlotWorkspace = lazy\(generationMode\("xyz_plot"\)\.component\)/);
  assert.match(app, /const generate = async \(\) => \{\s*if \(selectedGenerationMode !== "basic"\) return/);
  assert.match(app, /xyzPersistenceBlockedRef\.current/);
  assert.match(workspace, /useDialogLifecycle\(true, onClose\)/);
  // Phase B: the start button submits through xyz-client only (asserted in xyz-phase-b.test.mjs).
  assert.doesNotMatch(workspace, /fetch\(|\/api\/inference\/jobs|randomSeed\(/);
});
