import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  IMAGE_INFO_APPLY_TARGETS,
  buildImageInfoApplyPlan,
  imageInfoApplyAllFields,
  imageInfoApplyFields,
  imageInfoApplySummary,
  imageInfoCanvas,
  imageInfoDefaultFields,
  imageInfoSampling,
} from "../src/image-info-apply.js";

const readSource = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");
const found = (value, engine) => ({ status: "local", name: value, match: { value, engine, label: `${engine} root` } });
const missing = (name) => ({ status: "missing", name });

const record = (overrides = {}) => ({
  status: "ok",
  positive: "a quiet observatory",
  negative: "blurry",
  checkpoint: "waiIllustrious_v17.safetensors",
  checkpointMatch: found("iL/waiIllustrious_v17.safetensors", "iL"),
  parameters: [
    { label: "Sampler", value: "euler_ancestral" },
    { label: "Scheduler", value: "karras" },
    { label: "Steps", value: "25" },
    { label: "CFG scale", value: "5.5" },
    { label: "Seed", value: "800220165636076" },
    { label: "Size", value: "1024x1280" },
  ],
  loras: [
    { name: "charactor/kuro.safetensors", weight: 0.75, match: found("iL/charactor/kuro.safetensors", "iL") },
    { name: "style/ghost.safetensors", weight: 0.6, match: missing("style/ghost.safetensors") },
  ],
  ...overrides,
});

test("only the prompt is selected by default, and only rows the record carries are offered", () => {
  assert.deepEqual(imageInfoDefaultFields(record()), ["prompts"]);
  assert.deepEqual(imageInfoApplyAllFields(record()), ["prompts", "sampling", "canvas", "model", "loras"]);

  // A record with nothing but a prompt offers exactly one row, and a record the
  // reader could not parse offers none — the button has nothing to apply.
  const bare = { status: "ok", positive: "just words", negative: "", parameters: [], loras: [] };
  assert.deepEqual(imageInfoApplyAllFields(bare), ["prompts"]);
  assert.deepEqual(imageInfoApplyFields({ status: "empty" }), []);
  assert.deepEqual(imageInfoApplyFields(null), []);
  // Defaulting cannot select a row that is not on offer.
  assert.deepEqual(imageInfoDefaultFields({ status: "ok", positive: "", negative: "", parameters: [{ label: "Steps", value: "20" }], loras: [] }), []);
});

test("a producer's own vocabulary is only applied where this workspace offers the same one", () => {
  const { values, skipped } = imageInfoSampling(record());
  assert.equal(values.sampler, "euler_ancestral");
  assert.equal(values.scheduler, "karras");
  assert.equal(values.steps, 25);
  assert.equal(values.cfg, 5.5);
  assert.equal(values.seed, "800220165636076");
  // A recorded seed that is not then held would roll a new one on the next run.
  assert.equal(values.seedMode, "fixed");
  assert.deepEqual(skipped, []);

  // A1111 writes one string for both, and neither half is a name this workspace
  // has. Writing it anyway would store a value no picker can show.
  const a1111 = imageInfoSampling(record({ parameters: [{ label: "Sampler", value: "DPM++ 2M Karras" }, { label: "Steps", value: "20" }] }));
  assert.equal("sampler" in a1111.values, false);
  assert.deepEqual(a1111.skipped, ["采样器 DPM++ 2M Karras"]);
  assert.equal(a1111.values.steps, 20);
  // `Number("")` is 0: a record that never mentioned CFG must not apply a CFG of zero.
  assert.equal("cfg" in a1111.values, false);
  assert.equal("seed" in a1111.values, false);

  // Out-of-range values are clamped to what the controls accept.
  const wild = imageInfoSampling(record({ parameters: [{ label: "Steps", value: "9000" }, { label: "CFG scale", value: "-4" }] }));
  assert.equal(wild.values.steps, 100);
  assert.equal(wild.values.cfg, 0);
});

test("a canvas is snapped to what the size control accepts, or not offered at all", () => {
  assert.deepEqual(imageInfoCanvas(record()), { width: 1024, height: 1280 });
  assert.deepEqual(imageInfoCanvas(record({ parameters: [{ label: "Size", value: "1000x500" }] })), { width: 1024, height: 512 });
  assert.deepEqual(imageInfoCanvas(record({ parameters: [{ label: "Size", value: "9000x9000" }] })), { width: 2048, height: 2048 });
  assert.equal(imageInfoCanvas(record({ parameters: [] })), null);
  assert.equal(imageInfoCanvas(record({ parameters: [{ label: "Size", value: "wide" }] })), null);
});

test("a model that is not installed is reported rather than written", () => {
  const fields = imageInfoApplyFields(record());
  assert.deepEqual(fields.find((field) => field.id === "loras").missing, ["style/ghost.safetensors"]);
  assert.deepEqual(fields.find((field) => field.id === "model").missing, []);

  const plan = buildImageInfoApplyPlan(record(), imageInfoApplyAllFields(record()), "t2i");
  // The four that exist are applied; the fifth is named, not mounted. The
  // mounted library is reconciled against the scanned catalogue, so mounting a
  // file that is not there would be pruned moments later anyway.
  assert.deepEqual(plan.overlay.loras.map((item) => item.value), ["iL/charactor/kuro.safetensors"]);
  assert.deepEqual(plan.missing, [{ kind: "lora", name: "style/ghost.safetensors" }]);
  assert.equal(plan.overlay.checkpoint, "iL/waiIllustrious_v17.safetensors");
  // Naming the engine is what lets the apply path switch to the right catalogue.
  assert.equal(plan.overlay.model, "iL");

  const withoutCheckpoint = buildImageInfoApplyPlan(record({ checkpointMatch: missing("waiIllustrious_v17.safetensors") }), ["model", "loras"], "t2i");
  assert.equal("checkpoint" in withoutCheckpoint.overlay, false);
  assert.equal(withoutCheckpoint.groups.includes("model"), false);
  assert.deepEqual(withoutCheckpoint.missing[0], { kind: "checkpoint", name: "waiIllustrious_v17.safetensors" });

  // Every LoRA missing means there is no mount to write, so the group is not
  // claimed — an empty list would read as "unmount everything".
  const allMissing = buildImageInfoApplyPlan(record({ loras: [{ name: "a.safetensors", weight: 1, match: missing("a.safetensors") }] }), ["loras"], "t2i");
  assert.deepEqual(allMissing.groups, []);
  assert.equal("loras" in allMissing.overlay, false);
});

test("the target decides which state the settings land in", () => {
  assert.deepEqual(IMAGE_INFO_APPLY_TARGETS.map((item) => item.id), ["t2i", "i2i"]);
  const fields = ["prompts", "sampling", "canvas", "model", "loras"];

  const t2i = buildImageInfoApplyPlan(record(), fields, "t2i");
  assert.equal(t2i.overlay.positive, "a quiet observatory");
  assert.deepEqual(t2i.overlay.size, { width: 1024, height: 1280 });
  assert.deepEqual(t2i.imageToImage, {});

  const i2i = buildImageInfoApplyPlan(record(), fields, "i2i");
  assert.equal(i2i.target, "i2i");
  assert.equal(i2i.imageToImage.positive, "a quiet observatory");
  assert.equal(i2i.imageToImage.steps, 25);
  // That page sizes from its source picture unless told otherwise, so applying
  // a canvas has to say which mode it is in.
  assert.equal(i2i.imageToImage.sizeMode, "custom");
  assert.equal(i2i.imageToImage.width, 1024);
  assert.equal("positive" in i2i.overlay, false);
  // The model and the LoRA mount are one shared state whichever page is chosen.
  assert.equal(i2i.overlay.checkpoint, "iL/waiIllustrious_v17.safetensors");
  assert.ok(i2i.overlay.loras.length);

  // An unselected row writes nothing at all.
  const promptOnly = buildImageInfoApplyPlan(record(), ["prompts"], "t2i");
  assert.deepEqual(Object.keys(promptOnly.overlay).sort(), ["negative", "positive"]);
  assert.deepEqual(promptOnly.groups, ["prompts"]);
  assert.deepEqual(buildImageInfoApplyPlan(record(), [], "t2i").groups, []);
});

test("the summary states what happened, including what could not", () => {
  const plan = buildImageInfoApplyPlan(record(), imageInfoApplyAllFields(record()), "t2i");
  assert.match(imageInfoApplySummary(plan), /已应用 5 项/);
  assert.match(imageInfoApplySummary(plan), /1 个模型本地缺失/);
  const a1111 = buildImageInfoApplyPlan(record({ parameters: [{ label: "Sampler", value: "DPM++ 2M Karras" }, { label: "Steps", value: "20" }] }), ["sampling"], "t2i");
  assert.match(imageInfoApplySummary(a1111), /1 项本工作区不支持/);
  assert.match(imageInfoApplySummary(buildImageInfoApplyPlan(record(), [], "t2i")), /没有可应用的项目/);
});

test("the panel keeps its marks on screen and the workspace writes go through one path", async () => {
  const panel = await readSource("src/ImageInfoApply.jsx");
  const app = await readSource("src/App.jsx");
  const reader = await readSource("src/ImageInfoReader.jsx");
  const toolbox = await readSource("src/ToolboxPage.jsx");
  const styles = await readSource("src/styles.css");

  // Which rows were unavailable comes from the applied plan, never from the
  // record alone: the mark means "this apply could not carry it".
  assert.match(panel, /const missingNames = new Set\(\(result\?\.missing \|\| \[\]\)\.map\(\(item\) => item\.name\)\)/);
  assert.match(panel, /className=\{rowMissing \? "missing" : ""\}/);
  assert.match(panel, /className=\{result && missingNames\.has\(entry\.name\) \? "missing" : ""\}/);
  assert.match(panel, /imageInfoDefaultFields\(info\)/);
  assert.match(panel, /全部应用/);

  // The reader threads the callback through, and offers the button only when
  // the host supplied one — the tool renders standalone in its own chunk.
  assert.match(toolbox, /<ImageInfoReader onApplyParameters=\{onApplyImageParameters\} \/>/);
  assert.match(reader, /onApplyParameters &&/);
  assert.match(reader, /disabled=\{!applicableFields\.length\}/);

  // Model writes reuse the gallery apply path rather than a second copy of the
  // engine-switch, catalogue-validation and missing-checkpoint logic.
  assert.match(app, /const applyImageInfoParameters = async \(plan\) => \{/);
  assert.match(app, /await applyGallerySettings\(\{ \.\.\.snapshot, \.\.\.overlay \}, workspaceGroups, \{ page: null, label: "图片参数" \}\)/);
  // Staying put is what keeps the panel's marks visible after applying.
  assert.match(app, /if \(page\) \{\s*\n\s*setActivePage\(page\);/);
  assert.match(app, /onApplyImageParameters=\{applyImageInfoParameters\}/);
  // A wrong-engine LoRA would be pruned by the next scan; dropping the group is
  // not the same as mounting an empty list.
  assert.match(app, /if \(!overlay\.loras\.length\) \{[\s\S]*?delete overlay\.loras;[\s\S]*?workspaceGroups\.splice\(index, 1\)/);

  for (const rule of [".image-info-apply-fields > li.missing", ".image-info-apply-entries span.missing", ".info-identity-model > .info-identity-model-name"]) {
    assert.ok(styles.includes(rule), `${rule} must be styled`);
  }
  // The base model is read first, so it is sized and coloured to be found first.
  assert.match(styles, /\.info-identity-model > \.info-identity-model-name \{[^}]*font-size: 14px/);
  assert.match(styles, /\.info-identity-model > \.model-name-found\.info-identity-model-name \{ color: var\(--lime\)/);
  assert.match(styles, /html\[data-theme-mode="light"\] \.image-info-apply-fields > li\.missing/);
});
