import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { xyzBaselineFromSettings, patchXYZParameters, editXYZMounts, copyXYZGroup, xyzCatalogChoices, xyzSidebarIssues, xyzStrongLora } from "../src/xyz-sidebar-context.js";
import { createXYZDraft, withXYZDraft, emptyXYZDrafts } from "../src/xyz-draft.js";
import { engineSettingsDefaults, SETTINGS_ENGINES } from "../src/engine-settings.js";
import { composeGroupPrompt } from "../src/lora-groups.js";
import { composeGenerationPrompt } from "../src/prompt-presets.js";
import { xyzApi } from "../src/xyz-client.js";

const member = (id = "a", weight = "0.8137") => ({ name: "same name", value: `${id}.safetensors`, instanceId: `00000000-0000-4000-8000-${String(Number(id) || id.charCodeAt(0)).padStart(12, "0")}`, identitySchema: 1, contentHash: "a".repeat(64), contentRevision: 7, weight, precision: 4, enabled: true, originalReference: { value: "old-name" } });
const groups = [{ id: "g-one", name: "group", presetPrompt: "group words", members: [member()], enabled: true }];
const source = (engine) => ({ ...engineSettingsDefaults(engine), checkpoint: "old checkpoint", diffusionModel: "old diffusion", positive: `Basic ${engine}`, negative: "Basic negative", seed: "18446744073709551615", seedMode: "fixed" });
function baseline(engine = "SD") {
  return xyzBaselineFromSettings(engine, source(engine), { mounts: [member()], groups, groupPrompt: composeGroupPrompt(groups, ""),
    composePrompt: (type, prompt, ids, groupPrompt) => composeGenerationPrompt({ groupPrompt, records: [], activeIds: [], type, prompt }),
    chainPayload: [], clipPayload: engine === "SD" ? { clip_skip: 1 } : {}, libraryRevision: 4 });
}

test("all six Basic and i2i records survive independent sidebar edits and roundtrip", () => {
  const basic = Object.fromEntries(SETTINGS_ENGINES.map((engine) => [engine, source(engine)])), before = structuredClone(basic);
  let drafts = emptyXYZDrafts();
  for (const engine of SETTINGS_ENGINES) {
    let draft = createXYZDraft(baseline(engine));
    draft = patchXYZParameters(draft, { steps: 41, cfg: 4.25, seed: "0", prompt: "XYZ only", width: 768 });
    draft = editXYZMounts(draft, { kind: "patch", index: 0, patch: { weight: "-0.1250", precision: 4, enabled: false } });
    drafts = withXYZDraft(drafts, engine, draft);
  }
  assert.deepEqual(basic, before);
  for (const engine of SETTINGS_ENGINES) { assert.equal(drafts.by_engine[engine].baseline.parameters.engine, engine); assert.equal(drafts.by_engine[engine].baseline.parameters.seed, "0"); }
  assert.equal(baseline("Flux").parameters.negative_prompt, "");
  assert.equal(baseline("Flux").parameters.text_encoder_2, "");
});
test("stored-engine prompt adapter is identical to existing composition and clones all snapshots", () => {
  const result = baseline("SD");
  assert.equal(result.parameters.prompt, "group words\n\nBasic SD");
  result.group_snapshots[0].members[0].weight = 0;
  assert.equal(groups[0].members[0].weight, "0.8137");
});
test("prompt edit detaches old prefix exactly once without rewriting frozen groups", () => {
  const draft = createXYZDraft(baseline()), before = structuredClone(draft);
  const next = patchXYZParameters(draft, { prompt: "red, blue" });
  assert.equal(next.baseline.prompt_without_groups, "red, blue"); assert.equal(next.baseline.group_prompt, "");
  assert.deepEqual(next.baseline.group_snapshots, before.baseline.group_snapshots); assert.deepEqual(draft, before);
  assert.throws(() => patchXYZParameters(draft, { engine: "Flux" }));
});
test("mount add/reorder/toggle/negative/zero preserves full strong evidence and ordered snapshot", () => {
  const draft = createXYZDraft(baseline()), before = structuredClone(draft);
  let next = editXYZMounts(draft, { kind: "add", item: member("b") });
  next = editXYZMounts(next, { kind: "patch", index: 1, patch: { weight: "0", precision: 4, enabled: false } });
  next = editXYZMounts(next, { kind: "move", index: 1, delta: -1 });
  assert.equal(next.baseline.loras[0].instanceId, member("b").instanceId); assert.equal(next.baseline.loras[0].weight, "0");
  assert.equal(next.baseline.loras[0].contentRevision, 7); assert.deepEqual(next.baseline.loras[1], before.baseline.loras[0]);
  assert.deepEqual(draft, before);
});
test("mount refuses stale identity, duplicate, out-of-range and >16 without changing original", () => {
  const draft = createXYZDraft(baseline());
  assert.equal(xyzStrongLora(member()), true); assert.equal(xyzStrongLora({ ...member(), contentRevision: null }), false);
  assert.throws(() => editXYZMounts(draft, { kind: "add", item: member() }));
  assert.throws(() => editXYZMounts(draft, { kind: "add", item: { ...member("b"), contentHash: "" } }));
  assert.throws(() => editXYZMounts(draft, { kind: "patch", index: 0, patch: { weight: 5.01 } }));
  assert.throws(() => editXYZMounts(draft, { kind: "patch", index: 0, patch: { instanceId: "wrong" } }));
  let next = draft; for (let i = 1; i < 16; i++) next = editXYZMounts(next, { kind: "add", item: member(String(i)) });
  assert.throws(() => editXYZMounts(next, { kind: "add", item: member("17") })); assert.equal(draft.baseline.loras.length, 1);
});
test("copy group replaces background list but neither library nor actual prompt or axes", () => {
  const draft = createXYZDraft(baseline()), before = structuredClone(groups), group = { ...groups[0], members: [member("b", "0"), member("c", "-1.2345")] };
  const next = copyXYZGroup(draft, group);
  assert.equal(next.baseline.loras[0].weight, "0"); assert.equal(next.baseline.parameters.prompt, draft.baseline.parameters.prompt);
  assert.deepEqual(next.axes, draft.axes); assert.deepEqual(groups, before); assert.equal(draft.baseline.loras.length, 1);
});
test("catalog adapter projects only existing engine component fields, never auto-picks or renames", () => {
  const payload = { engine: "Flux", model_type: "split", assets: Object.fromEntries(["diffusion_model", "text_encoder", "text_encoder_2", "vae"].map((key) => [key, { models: [{ value: `${key}/same.safetensors`, name: "same" }] }])) };
  assert.deepEqual(Object.keys(xyzCatalogChoices(payload)), ["diffusion_model", "text_encoder", "text_encoder_2", "vae"]);
  assert.equal(xyzCatalogChoices(payload).text_encoder_2[0].value, "text_encoder_2/same.safetensors");
  assert.deepEqual(xyzCatalogChoices({ engine: "SD", models: [] }), { checkpoint: [] });
});
test("baseline invalid text is refused rather than normalized, uint64 stays exact", () => {
  const draft = createXYZDraft(baseline()); assert.deepEqual(xyzSidebarIssues(draft), []);
  const next = patchXYZParameters(draft, { steps: "", seed: "18446744073709551616" });
  assert.equal(xyzSidebarIssues(next).length, 2); assert.equal(next.baseline.parameters.seed, "18446744073709551616");
});
test("edited true-current baseline reaches strict transport without UI field caches", async () => {
  const requests = [], api = xyzApi(async (url, options) => { requests.push(JSON.parse(options.body)); return { ok: true, json: async () => ({}) }; });
  const draft = patchXYZParameters(createXYZDraft(baseline()), { steps: 39, cfg: 3, seed: "0" });
  await api.plan(draft); assert.equal(requests[0].baseline.parameters.steps, 39);
  assert.equal(requests[0].baseline.parameters.seed, "0"); assert.equal(Object.hasOwn(requests[0], "previousValueState"), false);
});
test("App chooses independent engine/callbacks without invoking Basic owners", async () => {
  const app = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
  // QPlan2/P2: XYZ and the LoRA mix search share the XYZ-family rail; each writes only its own independent draft owner.
  assert.match(app, /isXYZFamily = selectedGenerationMode === "xyz_plot" \|\| isMixMode/);
  assert.match(app, /isXYZFamily \? setXYZEngine\(item.name\) : selectModel/);
  assert.match(app, /componentChoices=\{xyzCatalog.choices\}/);
  const start = app.indexOf("<XYZBaselineSidebar"), end = app.indexOf(" /></div>", start);
  const sidebar = app.slice(start, end);
  assert.match(sidebar, /onChange=\{setFamilyDraft\}/); assert.doesNotMatch(sidebar, /setSteps|setCfg|setLoras|selectCheckpoint/);
  const family = app.slice(app.indexOf("const setFamilyDraft"), app.indexOf("const xyzCatalog"));
  assert.match(family, /withLoraMixDraft\(old, activeXYZEngine, draft\)/); assert.match(family, /withXYZDraft\(old, activeXYZEngine, draft\)/);
  assert.doesNotMatch(family, /setSteps|setCfg|setLoras|selectCheckpoint|setEngineSettings/);
  assert.match(app, /ModelAssetPicker loading=\{modelLoading\}/); assert.match(app, /MountedLoraSummary loras=\{loras\}/);
});
test("guarded field/range replacement caches live only in editor, results footer has no draft start", async () => {
  const editor = await readFile(new URL("../src/XYZAxisEditor.jsx", import.meta.url), "utf8"), workspace = await readFile(new URL("../src/XYZPlotWorkspace.jsx", import.meta.url), "utf8");
  assert.match(editor, /previousValueState = useRef\(previousStates \|\| new Map\(\)\)/); assert.match(editor, /AxisReplaceDialog/);
  assert.match(workspace, /tab === "config" && <footer/);
  const controls = workspace.slice(workspace.indexOf("const controls ="), workspace.indexOf("return <section", workspace.indexOf("const controls =")));
  assert.doesNotMatch(controls, /start\}|开始 XYZ|导出整网格/);
  assert.match(workspace, /detail\.baseline\?\.steps/);
});
test("SizeGrid reuses root pointer owner and only binds draft generation dimensions; catalog never persists", async () => {
  const canvas = await readFile(new URL("../src/XYZCanvasPanel.jsx", import.meta.url), "utf8"), catalog = await readFile(new URL("../src/use-xyz-catalog.js", import.meta.url), "utf8");
  assert.match(canvas, /<SizeGrid width=\{p.width\} height=\{p.height\} max=\{max\} onChange=\{\(width, height\) => update\(\{ width, height \}\)\}/); assert.match(canvas, /XYZ 生成/);
  assert.doesNotMatch(canvas, /<details|previewSize|exportPercent/);
  assert.doesNotMatch(catalog, /\/api\/ui-state|\/api\/lora-cards|method: "PUT"|localStorage/);
  assert.match(catalog, /controller.signal.aborted/);
});

test("XYZ uses the Basic model picker and compact LoRA card without changing its independent owners", async () => {
  const [sidebar, picker, summary, catalog] = await Promise.all([
    readFile(new URL("../src/XYZBaselineSidebar.jsx", import.meta.url), "utf8"),
    readFile(new URL("../src/ModelAssetPicker.jsx", import.meta.url), "utf8"),
    readFile(new URL("../src/MountedLoraSummary.jsx", import.meta.url), "utf8"),
    readFile(new URL("../src/use-xyz-catalog.js", import.meta.url), "utf8"),
  ]);
  assert.match(sidebar, /section-heading lora-title"><span>02<\/span><h2>全局 LoRA<\/h2>/);
  assert.match(sidebar, /MountedLoraSummary loras=\{draft\.baseline\.loras\}/);
  assert.match(sidebar, /onClick=\{onOpenManager\}/);
  assert.doesNotMatch(sidebar, /setLoras\(|setLoraGroups\(|openBasicLoraManager|instanceId\.slice/);
  const manager = await readFile(new URL("../src/XYZLoraManager.jsx", import.meta.url), "utf8");
  const shared = await readFile(new URL("../src/LoraManagerModal.jsx", import.meta.url), "utf8");
  // P2.1: the library-group copy is one verified transaction (resolve -> preview -> explicit mode -> one draft edit),
  // still applied only through the independent XYZ draft owner and never through a Basic/library writer.
  const copyHook = await readFile(new URL("../src/use-xyz-group-copy.js", import.meta.url), "utf8");
  assert.match(manager, /useXYZGroupCopy\(\{ engine: draft\.engine, draftRef: current, attempt,/);
  assert.match(manager, /libraryCopy=\{\(group\) => void copy\.start\(group\)\}/);
  assert.match(copyHook, /attempt\(\(draft\) => copyXYZManagerGroup\(draft, projection\.group, \{ mode \}\)\)/);
  assert.doesNotMatch(copyHook, /commitMountedLoras|commitLoraGroups|\/api\/ui-state|\/api\/lora-cards|PUT/);
  assert.match(manager, /updateXYZManagerMounts\(d, updater\)/);
  assert.match(manager, /locked=\{!xyzStrongLora\(item\)\}/);
  assert.match(shared, /<LoraMountPanel variant="modal"/);
  assert.match(shared, /<LoraGroupPanel/);
  assert.doesNotMatch(manager, /commitMountedLoras|commitLoraGroups|setActivePromptPresets/);
  assert.match(picker, /checkpoint-picker split-model-picker/);
  assert.match(picker, /split-model-field/);
  assert.match(picker, /<WorkspaceSelect/);
  assert.match(summary, /lora-summary-table/); assert.match(summary, /lora-summary-row/); assert.match(summary, /formatWeight\(lora\.weight\)/);
  assert.match(catalog, /directories/); assert.match(catalog, /const refresh = useCallback/);
  assert.doesNotMatch(catalog, /\/api\/ui-state|\/api\/lora-cards|method: "PUT"|localStorage/);
});
