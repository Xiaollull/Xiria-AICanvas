import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { XYZ_APPLY_DEFAULT_GROUPS, stripPresetBlocks, xyzApplyRows, xyzBasicPrompt, xyzCellGalleryOutput, xyzCellSettings, xyzViewerSource } from "../src/xyz-apply.js";
import { interpretImageMetadata } from "../src/image-metadata.js";
import { imageInfoApplyFields } from "../src/image-info-apply.js";
import { withGalleryLoraRevision } from "../src/gallery-core.js";

const source = (file) => readFile(new URL(`../src/${file}`, import.meta.url), "utf8");
const records = [
  { id: "q", type: "positive", content: "masterpiece, best quality", position: "start" },
  { id: "t", type: "positive", content: "film grain", position: "end" },
  { id: "n", type: "negative", content: "lowres", position: "start" },
];
const parts = (overrides = {}) => ({ prompt: "group words\n\nmasterpiece, best quality\n\ncat\n\nfilm grain", negative_prompt: "lowres\n\nbad hands", rest: "masterpiece, best quality\n\ncat\n\nfilm grain",
  removed: true, lead: "group words", lead_source: "groups", expected_lead: "group words", expected_source: "groups", ...overrides });

test("preset blocks are removed only when Basic's switched-on presets frame the text exactly", () => {
  const blocks = { leading: ["masterpiece, best quality"], trailing: ["film grain"] };
  assert.deepEqual(stripPresetBlocks("masterpiece, best quality\n\ncat\n\nfilm grain", blocks), { text: "cat", removed: true });
  assert.deepEqual(stripPresetBlocks("masterpiece, best quality\n\nfilm grain", blocks), { text: "", removed: true });
  assert.deepEqual(stripPresetBlocks("cat\n\nfilm grain", blocks), { text: "cat\n\nfilm grain", removed: false });
  assert.deepEqual(stripPresetBlocks("masterpiece, best quality, cat", blocks), { text: "masterpiece, best quality, cat", removed: false });
  assert.deepEqual(stripPresetBlocks("  cat  ", {}), { text: "cat", removed: false });
});

test("Apply writes the cell text without combination words and without the presets Basic re-adds", () => {
  const context = { records, activePositive: ["q", "t"], activeNegative: ["n"], groupPrompt: "group words" };
  const prompt = xyzBasicPrompt(parts(), context);
  assert.equal(prompt.positive, "cat");
  assert.equal(prompt.negative, "bad hands");
  assert.equal(prompt.removedLead, "group words");
  assert.deepEqual(prompt.reproduces, { positive: true, negative: true });
  assert.equal(prompt.preview.positive, parts().prompt);
  assert.ok(prompt.notes.some((note) => note.includes("组合预设词")) && prompt.notes.some((note) => note.includes("提示词预设")));
  // Basic without those presets: the preset words stay in the box, nothing is doubled, still identical.
  const bare = xyzBasicPrompt(parts(), { records, activePositive: [], activeNegative: [], groupPrompt: "group words" });
  assert.equal(bare.positive, "masterpiece, best quality\n\ncat\n\nfilm grain");
  assert.deepEqual(bare.reproduces, { positive: true, negative: true });
  // Basic with other combinations enabled: the difference is shown, never silently "fixed".
  const other = xyzBasicPrompt(parts(), { records, activePositive: ["q", "t"], activeNegative: ["n"], groupPrompt: "other words" });
  assert.equal(other.reproduces.positive, false);
  assert.equal(other.preview.positive, "other words\n\nmasterpiece, best quality\n\ncat\n\nfilm grain");
  assert.ok(other.notes.some((note) => note.includes("见下方对照")));
});

test("a prompt the experiment's words do not lead is written as it is and says why", () => {
  const prompt = xyzBasicPrompt(parts({ prompt: "dog in snow", rest: "dog in snow", removed: false, lead: "", lead_source: null }), { groupPrompt: "group words" });
  assert.equal(prompt.positive, "dog in snow");
  assert.equal(prompt.removedLead, "");
  assert.ok(prompt.notes.some((note) => note.includes("按原文写入")));
  assert.equal(prompt.reproduces.positive, false);
  const profile = xyzBasicPrompt(parts({ prompt: "new words\n\ncat", rest: "cat", lead: "new words", lead_source: "profile" }), { groupPrompt: "" });
  assert.ok(profile.notes[0].includes("LoRA 方案"));
});

const MAX = "18446744073709551615";
const cell = (request, extra = {}) => ({
  request: { negative_prompt: "bad", steps: 20, cfg: 6.5, denoise: 1, seed: MAX, sampler: "euler", scheduler: "normal", guidance: "none", pag: { scale: 0.3, applied_layers: "mid" }, width: 512, height: 768, postprocess_chain: [], ...request },
  lora_references: [{ value: "style/a.safetensors", weight: 0.8, precision: 4, enabled: true, identitySchema: 1, instanceId: "baa8eed2-a731-40db-a485-042addfca4c6", contentHash: "1".repeat(64), contentRevision: 1, name: "a.safetensors" }],
  prompt_parts: parts(), ...extra,
});

test("a cell's frozen request becomes the workspace record the gallery apply path reads", () => {
  const prompt = { positive: "cat", negative: "bad", removedLead: "group words" };
  const sd = xyzCellSettings(cell({ engine: "iL", checkpoint: "wai.safetensors", clip_skip: 2 }), prompt);
  assert.deepEqual({ model: sd.model, checkpoint: sd.checkpoint, seed: sd.seed, seedMode: sd.seedMode, clipSkip: sd.clipSkip, size: sd.size, pag: sd.pag }, { model: "iL", checkpoint: "wai.safetensors", seed: MAX, seedMode: "fixed", clipSkip: 2, size: { width: 512, height: 768 }, pag: { scale: 0.3, appliedLayers: "mid" } });
  assert.equal(typeof sd.seed, "string");
  assert.equal(sd.loras[0].instanceId, "baa8eed2-a731-40db-a485-042addfca4c6");
  assert.equal(sd.loraGroupPrompt, "group words");
  assert.equal(sd.composedPrompt, parts().prompt);
  assert.equal("imagesPerBatch" in sd || "batchCount" in sd, false, "a cell never resets Basic's batch settings");
  const original = cell({ engine: "iL", checkpoint: "wai.safetensors" });
  xyzCellSettings(original, prompt).loras[0].weight = 5;
  assert.equal(original.lora_references[0].weight, 0.8, "the record is a copy");
  const anima = xyzCellSettings(cell({ engine: "Anima", diffusion_model: "anima.safetensors", text_encoder: "qwen.safetensors", vae: "vae.safetensors" }), prompt);
  assert.deepEqual([anima.checkpoint, anima.diffusionModel, anima.textEncoder, anima.textEncoder2, anima.vae, anima.clipSkip], ["", "anima.safetensors", "qwen.safetensors", "", "vae.safetensors", null]);
  const flux = xyzCellSettings(cell({ engine: "Flux", diffusion_model: "f.safetensors", text_encoder: "clip.safetensors", text_encoder_2: "t5.safetensors", vae: "ae.safetensors", negative_prompt: "" }), prompt);
  assert.equal(flux.textEncoder2, "t5.safetensors");
  const chained = xyzCellSettings(cell({ engine: "iL", checkpoint: "w", postprocess_chain: [{ type: "resample", enabled: true, upscaler: "4x.pth", scale: 1.5, resize_algorithm: "lanczos", denoise: 0.35, steps: null, cfg: null, seed_mode: "fixed", seed: "42", tile_size: 512, tile_overlap: 64, execution_mode: "full_frame", padding: 32, mask_blur: 8, uniform_tiles: true, tiled_decode: true, exact_output: false }] }), prompt);
  assert.deepEqual([chained.postprocessChain[0].type, chained.postprocessChain[0].upscaler, chained.postprocessChain[0].seedMode, chained.postprocessChain[0].seed], ["resample", "4x.pth", "fixed", "42"]);
});

test("every group is offered with what it would write; only Prompt starts selected", () => {
  assert.deepEqual([...XYZ_APPLY_DEFAULT_GROUPS], ["prompts"]);
  const rows = xyzApplyRows(xyzCellSettings(cell({ engine: "iL", checkpoint: "wai.safetensors" }, { lora_references: [] }), { positive: "cat", negative: "", removedLead: "" }));
  assert.deepEqual(rows.map((row) => row.id), ["prompts", "model", "loras", "sampling", "canvas", "chain", "auxiliary"]);
  assert.match(rows.find((row) => row.id === "loras").summary, /清空/);
  assert.match(rows.find((row) => row.id === "chain").summary, /清空/);
  assert.match(rows.find((row) => row.id === "sampling").summary, new RegExp(`Seed ${MAX}`));
});

test("a cell enters the gallery as one output with its frozen seed; viewer sources need an asset id", () => {
  const output = xyzCellGalleryOutput({ ...cell({ engine: "iL" }), thumbnail_url: "/thumb", output: { asset_id: "tok", asset_url: "/a/tok", width: 512, height: 768 } });
  assert.deepEqual(output, { asset_id: "tok", url: "/thumb", batch_index: 1, image_index: 1, seed: MAX, base_seed: MAX, width: 512, height: 768 });
  assert.deepEqual(xyzViewerSource({ assetId: "tok", name: "" }), { assetId: "tok", name: "XYZ.png" });
  assert.equal(xyzViewerSource({ assetId: "" }), null);
});

test("a grid PNG reads as a comparison sheet with nothing to apply; a cell PNG names its experiment", () => {
  const grid = { schema_version: 1, kind: "xyz_plot_grid", export_id: "e1", experiment_id: "a".repeat(32), definition_digest: "d".repeat(64), manifest: { sha256: "f".repeat(64) },
    recipe: { scope: "all", mode: "scaled", percent: 50, algorithm: "Pillow LANCZOS", canvas: { width: 585, height: 974 } }, cells: [{ id: "0:0:0:0", rect: [0, 0, 1, 1] }, { id: "1:0:0:0", rect: [1, 0, 1, 1] }] };
  const info = interpretImageMetadata({ xyz_plot: JSON.stringify(grid) });
  assert.equal(info.source, "xyz_grid");
  assert.deepEqual({ ...info.xyzGrid }, { experimentId: "a".repeat(32), definitionDigest: "d".repeat(64), exportId: "e1", manifestSha256: "f".repeat(64), mode: "scaled", percent: 50, width: 585, height: 974, cellIds: ["0:0:0:0", "1:0:0:0"], overflow: false, search: null }, "a plain XYZ grid is explicitly not a search grid (QPlan2/P2)");
  assert.deepEqual(imageInfoApplyFields(info), []);
  assert.equal(info.parameters.some((item) => item.label === "Size"), false, "the sheet's size is never offered as a canvas");
  assert.equal(interpretImageMetadata({ xyz_plot: JSON.stringify({ ...grid, cells: undefined, metadata_overflow: { see: "manifest" } }) }).xyzGrid.overflow, true);
  const cellInfo = interpretImageMetadata({ prompt: "cat", negative_prompt: "", parameters: JSON.stringify({ job_id: "j", engine: "iL", checkpoint: "w.safetensors", steps: 20, xyz_plot: { experiment_id: "b".repeat(32), cell_id: "1:0:0:0", attempt: 1 } }) });
  assert.deepEqual(cellInfo.xyzCell, { experimentId: "b".repeat(32), cellId: "1:0:0:0" });
  assert.ok(imageInfoApplyFields(cellInfo).length > 0, "a cell original keeps the ordinary apply");
  assert.equal(interpretImageMetadata({ parameters: JSON.stringify({ job_id: "j", engine: "iL", xyz_plot: { experiment_id: "../x", cell_id: "1" } }) }).xyzCell, null);
});

test("new gallery content carries the client's LoRA publication; a card read at a revision keeps it", async () => {
  assert.deepEqual(withGalleryLoraRevision({ positive: "a" }, 3), { positive: "a", loraLibraryRevision: 3 });
  assert.deepEqual(withGalleryLoraRevision({ positive: "a", loraLibraryRevision: 2 }, 3), { positive: "a", loraLibraryRevision: 2 }, "an editor never claims a newer publication than it read");
  assert.deepEqual(withGalleryLoraRevision({ positive: "a" }, 0), { positive: "a" }, "nothing to stamp before any publication");
  assert.equal(withGalleryLoraRevision(null, 3), null);
  assert.match(await source("Gallery.jsx"), /return withGalleryLoraRevision\(galleryOutputSeedSettings\(/);
  assert.match(await source("GalleryPage.jsx"), /settings: withGalleryLoraRevision\(\{\n\s+\.\.\.savedSettings,/);
});

test("Viewer, Gallery and Apply reach App's own owners; nothing applies without a click", async () => {
  const app = await source("App.jsx");
  assert.match(app, /const openImageViewer = async \(source\) => \{\n\s+const explicit = source && typeof source === "object" && typeof source\.assetId === "string"/);
  assert.match(app, /onApplyToBasic: \(settings, groups\) => applyGallerySettings\(settings, groups, \{ page: null, label: "XYZ 格参数" \}\)/);
  assert.match(app, /integration=\{xyzIntegration\}/);
  assert.match(app, /xyzIntegration=\{xyzIntegration\}/);
  assert.match(app, /outputs=\{galleryAddSource \? galleryAddSource\.outputs : generatedOutputs\}/);
  assert.equal((app.match(/setGalleryAddSource\(null\); setGalleryAddOpen\(true\)/g) || []).length, 2, "ordinary gallery adds never show a cell");
  assert.match(app, /disableUnmountedGroups\(activeLoraGroups, loras\)/);
  const toolbox = await source("ToolboxPage.jsx");
  assert.match(toolbox, /xyzIntegration=\{xyzIntegration\}/);
  const workspace = await source("XYZPlotWorkspace.jsx");
  for (const label of ["预览 ${cell.id} 原图", "把 ${cell.id} 加入画廊", "把 ${cell.id} 的参数应用到基础生图", "下载 ${cell.id} 原图"]) assert.ok(workspace.includes(label), label);
  assert.match(workspace, /const full = await api\.cell\(experimentId, cell\.id\)/);
  assert.match(workspace, /<XYZCellApplyDialog experimentId=\{detailId\}/);
  assert.match(workspace, /<= VIEWER_MAX_CANVAS_EDGE/);
  const actions = await source("XYZCellActions.jsx");
  assert.match(actions, /useState\(\(\) => new Set\(XYZ_APPLY_DEFAULT_GROUPS\)\)/);
  assert.match(actions, /await onApply\(settings, chosen\)/);
  assert.equal((actions.match(/onApply\(/g) || []).length, 1, "the only apply call is the button handler");
  assert.match(actions, /读取网格不会自动应用任何参数/);
  const reader = await source("ImageInfoReader.jsx");
  assert.match(reader, /info\.xyzGrid && <XYZGridRecordPanel/);
  assert.match(reader, /按实验记录应用…/);
});
