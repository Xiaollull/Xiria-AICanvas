import assert from "node:assert/strict";
import test from "node:test";

import { comfyNodeRoles } from "../src/image-metadata.js";
import {
  buildGraphModules,
  countByCategory,
  describeChunk,
  describeChunks,
  describeField,
  groupParameters,
  isComfyGraph,
  jsonBranch,
  jsonSummary,
  matchesQuery,
  nodeCategory,
} from "../src/metadata-explorer.js";

function graph() {
  return {
    "5": { class_type: "KSampler", _meta: { title: "K采样器" }, inputs: { steps: 30, cfg: 5, model: ["48", 1], positive: ["19", 1] } },
    "19": { class_type: "WeiLinPromptUIWithoutLora", _meta: { title: "补充tag" }, inputs: { positive: "1girl, standing in the rain", opt_text: ["21", 0], 打开提示词编辑器: "" } },
    "21": { class_type: "String Literal (Image Saver)", inputs: { string: "masterpiece" } },
    "48": { class_type: "WeiLinPromptUIOnlyLoraStack", _meta: { title: "Lora2" }, inputs: { lora_str: JSON.stringify([{ name: "ink", weight: 0.75, lora: "ink.safetensors" }]) } },
    "100": { class_type: "Image Save", inputs: { images: ["5", 0] } },
  };
}

test("a graph arrives as one block per node, in file order and never as one wall", () => {
  const modules = buildGraphModules(graph(), comfyNodeRoles(graph()));

  assert.equal(modules.length, 5);
  // Numeric, so #100 sorts after #19 rather than between #1 and #2.
  assert.deepEqual(modules.map((item) => item.id), ["5", "19", "21", "48", "100"]);

  const sampler = modules[0];
  assert.equal(sampler.classType, "KSampler");
  assert.equal(sampler.title, "K采样器");
  assert.deepEqual(sampler.roles, ["采样"]);
  // Every input is its own row, so nothing is summarised away.
  assert.deepEqual(sampler.fields.map((field) => field.name), ["steps", "cfg", "model", "positive"]);

  // A title that only repeats the class name is noise, not a label.
  assert.equal(buildGraphModules({ "1": { class_type: "KSampler", _meta: { title: "KSampler" }, inputs: {} } })[0].title, "");
  // A node with no inputs is still a block; it just has nothing in it.
  assert.deepEqual(buildGraphModules({ "1": { class_type: "PreviewImage" } })[0].fields, []);
});

test("a field knows whether it is a jump, a value or a body of text", () => {
  const link = describeField("model", ["48", 1]);
  assert.equal(link.kind, "link");
  assert.equal(link.target, "48");
  assert.equal(link.text, "#48 · 1");

  assert.deepEqual(describeField("steps", 30), { name: "steps", kind: "value", text: "30" });
  assert.equal(describeField("打开提示词编辑器", "").kind, "text");
  assert.equal(describeField("打开提示词编辑器", "").text, "");

  // The 20 KB of editor state a prompt plugin parks in a widget must not decide
  // how tall a card is, so a long value ships clipped with the full text beside it.
  const long = describeField("temp_str", "x".repeat(900));
  assert.equal(long.truncated, true);
  assert.equal(long.chars, 900);
  assert.ok(long.preview.length < 260 && long.preview.endsWith("…"));
  assert.equal(long.text.length, 900);
  assert.equal(describeField("string", "short").truncated, false);

  // An object widget is JSON, not "[object Object]".
  assert.match(describeField("loras", { __value__: [1, 2] }).text, /__value__/);
});

test("a node is grouped by what it carries, not only by what it is called", () => {
  // Both plugin classes have "Prompt" in the name and one of them has "Lora"
  // too; the widgets are the reliable signal.
  assert.equal(nodeCategory({ class_type: "WeiLinPromptUIWithoutLora", inputs: { positive: "text" } }), "prompt");
  assert.equal(nodeCategory({ class_type: "WeiLinPromptUIOnlyLoraStack", inputs: { lora_str: '[{"lora":"ink.safetensors"}]' } }), "lora");
  assert.equal(nodeCategory({ class_type: "Lora Loader (LoraManager)", inputs: { loras: { __value__: [{ name: "ink", active: true }] } } }), "lora");

  assert.equal(nodeCategory({ class_type: "KSampler", inputs: {} }), "sampling");
  assert.equal(nodeCategory({ class_type: "CLIPTextEncode", inputs: {} }), "prompt");
  assert.equal(nodeCategory({ class_type: "UNETLoader", inputs: {} }), "model");
  assert.equal(nodeCategory({ class_type: "EmptyLatentImage", inputs: {} }), "latent");
  // An output before it is an image, or every save node lands in the wrong list.
  assert.equal(nodeCategory({ class_type: "Image Save", inputs: {} }), "output");
  assert.equal(nodeCategory({ class_type: "UltimateSDUpscale", inputs: {} }), "image");
  assert.equal(nodeCategory({ class_type: "GroupIgnoreManager", inputs: {} }), "other");

  const counts = countByCategory(buildGraphModules(graph()));
  assert.equal(counts.get("prompt"), 2);
  assert.equal(counts.get("lora"), 1);
});

test("search reaches node ids, class names, titles, parameter names and values", () => {
  const [sampler, prompt] = buildGraphModules(graph());

  assert.equal(matchesQuery(prompt, ""), true);
  assert.equal(matchesQuery(prompt, "#19"), true);
  assert.equal(matchesQuery(prompt, "weilin"), true);
  assert.equal(matchesQuery(prompt, "补充tag"), true);
  assert.equal(matchesQuery(prompt, "standing in the rain"), true);
  assert.equal(matchesQuery(sampler, "cfg"), true);
  assert.equal(matchesQuery(sampler, "standing in the rain"), false);
});

test("a chunk is classified by shape, because a graph can arrive under any keyword", () => {
  const chunks = describeChunks({
    prompt: JSON.stringify(graph()),
    workflow: JSON.stringify({ nodes: [{ id: 5 }], links: [] }),
    parameters: "a prompt\nSteps: 20, Seed: 1",
  });
  assert.deepEqual(chunks.map((chunk) => chunk.kind), ["graph", "json", "text"]);
  assert.ok(chunks[0].graph && !chunks[0].json);
  assert.ok(chunks[1].json && !chunks[1].graph);
  assert.equal(chunks[2].json, null);
  assert.equal(chunks[2].chars, "a prompt\nSteps: 20, Seed: 1".length);

  assert.equal(isComfyGraph({ 1: { class_type: "KSampler" } }), true);
  assert.equal(isComfyGraph({ nodes: [], links: [] }), false);
  assert.equal(isComfyGraph([{ class_type: "KSampler" }]), false);
  // ComfyUI writes `NaN` into widgets, so chunk detection has to survive it too.
  assert.equal(describeChunk("prompt", '{"1": {"class_type": "KSampler", "inputs": {"cfg": NaN}}}').kind, "graph");
  assert.equal(describeChunks({}).length, 0);
});

test("a payload that is JSON but not a graph still folds per block", () => {
  assert.deepEqual(jsonBranch([1, 2, 3]), { kind: "array", size: 3, entries: [["0", 1], ["1", 2], ["2", 3]] });
  assert.equal(jsonBranch({ a: 1, b: 2 }).kind, "object");
  assert.equal(jsonBranch("scalar").kind, "scalar");
  assert.equal(jsonBranch(null).kind, "scalar");

  // The collapsed line has to say how much is inside, or folding hides the size.
  assert.equal(jsonSummary([1, 2, 3]), "[ 3 项 ]");
  assert.equal(jsonSummary({ a: 1 }), "{ 1 字段 }");
  assert.match(jsonSummary("x".repeat(900)), /…$/);
  assert.equal(jsonSummary(42), "42");
});

test("a flat settings line becomes the modules the producer described", () => {
  // Four ADetailer passes plus Hires: A1111 writes them as one line whose names
  // are nine tenths repeated prefix.
  const parameters = [
    { label: "Steps", value: "32" },
    { label: "CFG scale", value: "6" },
    { label: "ADetailer model", value: "hand_yolov8n.pt" },
    { label: "ADetailer confidence", value: "0.3" },
    { label: "ADetailer model 2nd", value: "person_yolov8n-seg.pt" },
    { label: "ADetailer denoising strength 2nd", value: "0.4" },
    { label: "ADetailer model 4th", value: "Eyes.pt" },
    { label: "Hires upscaler", value: "R-ESRGAN 4x+ Anime6B" },
    { label: "Hires Module 1", value: "Use same choices" },
  ];
  const groups = groupParameters(parameters);

  assert.deepEqual(groups.map((group) => group.label), ["基础参数", "ADetailer · 1", "ADetailer · 2", "ADetailer · 4", "Hires.fix"]);
  // The prefix moved into the heading, so each cell reads as the one thing it says.
  assert.deepEqual(groups[1].items, [{ label: "model", value: "hand_yolov8n.pt" }, { label: "confidence", value: "0.3" }]);
  assert.deepEqual(groups[2].items.map((item) => item.label), ["model", "denoising strength"]);
  // A bare trailing digit is a widget's own name, not an ordinal.
  assert.deepEqual(groups[4].items.map((item) => item.label), ["upscaler", "Module 1"]);
  // Nothing is dropped on the way through.
  assert.equal(groups.reduce((total, group) => total + group.items.length, 0), parameters.length);

  // A record with nothing to group is one base module, so the caller can keep
  // the plain grid.
  const plain = groupParameters([{ label: "Steps", value: "30" }, { label: "Seed", value: "1" }]);
  assert.equal(plain.length, 1);
  assert.equal(plain[0].label, "基础参数");
  assert.deepEqual(groupParameters([]), []);
  assert.deepEqual(groupParameters(null), []);

  // A group that ran once is just itself; only a repeated one is numbered.
  const single = groupParameters([{ label: "ADetailer model", value: "face.pt" }]);
  assert.equal(single[0].label, "ADetailer");
});
