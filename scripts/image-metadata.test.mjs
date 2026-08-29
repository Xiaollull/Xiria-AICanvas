import assert from "node:assert/strict";
import test from "node:test";
import { deflateSync } from "node:zlib";

import { isPng, readPngMetadataChunks, readPngTextChunks } from "./png-text-chunks.mjs";
import {
  IMAGE_INFO_STATUS,
  comfyNodeRoles,
  extractPromptLoras,
  interpretImageMetadata,
  matchModelName,
  parseA1111,
  parseComfyUi,
  parseLooseJson,
  splitSettingsLine,
} from "../src/image-metadata.js";

const SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

function chunk(type, data) {
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length);
  // CRC is not validated on read: this reader never executes or renders the
  // pixels, so a wrong checksum is not a reason to withhold the prompt.
  return Buffer.concat([length, Buffer.from(type, "latin1"), data, Buffer.alloc(4)]);
}

function tEXt(keyword, text) {
  return chunk("tEXt", Buffer.concat([Buffer.from(`${keyword}\0`, "latin1"), Buffer.from(text, "latin1")]));
}

function zTXt(keyword, text) {
  return chunk("zTXt", Buffer.concat([Buffer.from(`${keyword}\0`, "latin1"), Buffer.from([0]), deflateSync(Buffer.from(text, "latin1"))]));
}

function iTXt(keyword, text, compressed = false) {
  const body = Buffer.from(text, "utf8");
  return chunk("iTXt", Buffer.concat([
    Buffer.from(`${keyword}\0`, "latin1"),
    Buffer.from([compressed ? 1 : 0, 0]),
    Buffer.from("\0\0", "latin1"),
    compressed ? deflateSync(body) : body,
  ]));
}

function png(...textChunks) {
  const ihdr = chunk("IHDR", Buffer.alloc(13));
  return Buffer.concat([SIGNATURE, ihdr, ...textChunks, chunk("IDAT", Buffer.alloc(8)), chunk("IEND", Buffer.alloc(0))]);
}

test("text chunks are read in every PNG encoding, including compressed UTF-8", () => {
  const buffer = png(
    tEXt("parameters", "plain"),
    zTXt("workflow", "compressed-latin1"),
    iTXt("prompt", "少女，白发，红色眼睛"),
    iTXt("negative_prompt", "低质量", true),
  );
  const chunks = readPngTextChunks(buffer);
  assert.equal(chunks.parameters, "plain");
  assert.equal(chunks.workflow, "compressed-latin1");
  // iTXt is the one text chunk defined as UTF-8, so CJK survives it intact.
  assert.equal(chunks.prompt, "少女，白发，红色眼睛");
  assert.equal(chunks.negative_prompt, "低质量");

  assert.equal(readPngTextChunks(Buffer.from("not a png at all")), null);
  assert.equal(isPng(Buffer.alloc(4)), false);
  // A tEXt chunk is Latin-1 by spec but both A1111 and ComfyUI write UTF-8
  // bytes into it anyway; the repair recovers them.
  const mojibake = png(tEXt("parameters", Buffer.from("杰作", "utf8").toString("latin1")));
  assert.equal(readPngMetadataChunks(mojibake).parameters, "杰作");
});

test("a repeated keyword keeps the first copy and a corrupt chunk costs only itself", () => {
  const chunks = readPngTextChunks(png(tEXt("parameters", "first"), tEXt("parameters", "second")));
  assert.equal(chunks.parameters, "first");

  const broken = chunk("zTXt", Buffer.concat([Buffer.from("workflow\0", "latin1"), Buffer.from([0]), Buffer.from("not zlib")]));
  const survived = readPngTextChunks(png(tEXt("parameters", "kept"), broken));
  assert.equal(survived.parameters, "kept");
  assert.equal(survived.workflow, undefined);
});

test("an image with no generation metadata reports empty rather than guessing", () => {
  assert.equal(interpretImageMetadata(readPngTextChunks(png())).status, IMAGE_INFO_STATUS.empty);
  assert.equal(interpretImageMetadata({}).status, IMAGE_INFO_STATUS.empty);

  // Chunks that exist but describe nothing generative are shown as-is: saying
  // "no metadata" there would be a lie.
  const other = interpretImageMetadata(readPngTextChunks(png(tEXt("Software", "Photoshop"))));
  assert.equal(other.status, IMAGE_INFO_STATUS.ok);
  assert.equal(other.source, "unknown");
  assert.deepEqual(other.parameters, [{ label: "Software", value: "Photoshop" }]);
});

test("A1111 settings survive quoted commas and the prompt split", () => {
  const text = [
    "masterpiece, 1girl, <lora:styleA:0.7>, <lora:styleB:0.35>",
    "Negative prompt: worst quality, bad hands",
    'Steps: 28, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 12345, Size: 832x1216, Model hash: a1b2c3, Model: animagineXL31, Denoising strength: 0.4, Hires upscale: 2, Hires upscaler: 4x-UltraSharp, Lora hashes: "styleA: aaa111, styleB: bbb222", ADetailer model: face_yolov8n.pt, Version: v1.9.3',
  ].join("\n");
  const info = parseA1111(text);

  assert.equal(info.positive, "masterpiece, 1girl, <lora:styleA:0.7>, <lora:styleB:0.35>");
  assert.equal(info.negative, "worst quality, bad hands");
  assert.equal(info.checkpoint, "animagineXL31");
  assert.equal(info.checkpointHash, "a1b2c3");

  // The quoted Lora hashes value contains commas; a naive split would shred it.
  const settings = splitSettingsLine('Steps: 28, Lora hashes: "a: 1, b: 2", Seed: 5');
  assert.equal(settings.get("Lora hashes"), "a: 1, b: 2");
  assert.equal(settings.get("Seed"), "5");

  // Weights only exist in the prompt tags, so those win over Lora hashes.
  assert.deepEqual(info.loras, [{ name: "styleA", weight: 0.7 }, { name: "styleB", weight: 0.35 }]);
  const flags = Object.fromEntries(info.flags.map((flag) => [flag.id, flag]));
  assert.equal(flags.hires.enabled, true);
  assert.equal(flags.hires.detail, "4x-UltraSharp · x2");
  assert.equal(flags.adetailer.enabled, true);
  assert.equal(flags.controlnet.enabled, false);
  // Model and hash render in their own fields, not twice in the table.
  assert.equal(info.parameters.some((item) => item.label === "Model"), false);
  assert.equal(info.parameters.find((item) => item.label === "Steps").value, "28");
});

test("A1111 without a negative prompt or LoRA tags still parses", () => {
  const info = parseA1111("a lone prompt\nSteps: 20, Sampler: Euler a, Seed: 1");
  assert.equal(info.positive, "a lone prompt");
  assert.equal(info.negative, "");
  assert.deepEqual(info.loras, []);
  assert.equal(info.flags.find((flag) => flag.id === "hires").enabled, false);

  // A prompt containing a colon must not be mistaken for the settings line.
  const tricky = parseA1111("portrait: dramatic lighting\nSteps: 20, Seed: 2");
  assert.equal(tricky.positive, "portrait: dramatic lighting");
  assert.deepEqual(extractPromptLoras("<lora:a:1.2> and <lyco:b>"), [{ name: "a", weight: 1.2 }, { name: "b", weight: 1 }]);
});

test("ComfyUI prompts follow the sampler's wiring, not node order", () => {
  // The negative encoder is deliberately first in the object: anything that
  // reads "the first CLIPTextEncode" gets this backwards.
  const graph = {
    "7": { class_type: "CLIPTextEncode", inputs: { text: "worst quality", clip: ["4", 1] } },
    "6": { class_type: "CLIPTextEncode", inputs: { text: "a scenic vista", clip: ["4", 1] } },
    "4": { class_type: "CheckpointLoaderSimple", inputs: { ckpt_name: "SDXL/animagine.safetensors" } },
    "5": { class_type: "EmptyLatentImage", inputs: { width: 1024, height: 1024 } },
    "10": { class_type: "LoraLoader", inputs: { lora_name: "styles/ink.safetensors", strength_model: 0.8, strength_clip: 0.8 } },
    "12": { class_type: "ImageUpscaleWithModel", inputs: {} },
    "3": { class_type: "KSampler", inputs: { seed: 987, steps: 25, cfg: 6.5, sampler_name: "dpmpp_2m", scheduler: "karras", denoise: 1, positive: ["6", 0], negative: ["7", 0] } },
  };
  const info = parseComfyUi(graph);

  assert.equal(info.positive, "a scenic vista");
  assert.equal(info.negative, "worst quality");
  assert.equal(info.checkpoint, "SDXL/animagine.safetensors");
  assert.deepEqual(info.loras, [{ name: "styles/ink.safetensors", weight: 0.8 }]);
  assert.equal(info.parameters.find((item) => item.label === "Size").value, "1024x1024");
  assert.equal(info.parameters.find((item) => item.label === "CFG scale").value, "6.5");
  assert.equal(info.flags.find((flag) => flag.id === "hires").enabled, true);
  assert.equal(info.flags.find((flag) => flag.id === "controlnet").enabled, false);
});

test("ComfyUI text reached through a primitive link is still found", () => {
  const graph = {
    "1": { class_type: "CheckpointLoaderSimple", inputs: { ckpt_name: "base.safetensors" } },
    "2": { class_type: "PrimitiveNode", inputs: { value: "linked prompt text" } },
    "3": { class_type: "CLIPTextEncode", inputs: { text: ["2", 0] } },
    "4": { class_type: "KSampler", inputs: { steps: 10, positive: ["3", 0], negative: ["3", 0] } },
  };
  assert.equal(parseComfyUi(graph).positive, "linked prompt text");
  // A graph with no sampler at all falls back to the encoders it can see.
  const noSampler = parseComfyUi({ "3": { class_type: "CLIPTextEncode", inputs: { text: "orphan" } } });
  assert.equal(noSampler.positive, "orphan");
});

// Distilled from a real ComfyUI export. Every shape here is one a plugin
// actually writes: a prompt editor that calls its widget `positive` on the node
// feeding the *negative* inlet, a preset string linked in through `opt_text`, a
// resolution picker and a seed node behind what look like plain numbers, and
// three LoRA stacks of which only one is wired to the model.
function pluginWorkflow() {
  return {
    "2": { class_type: "CLIPLoader", inputs: { clip_name: "qwen.safetensors" } },
    "3": { class_type: "UNETLoader", inputs: { unet_name: "anima-base-v1.0.safetensors" } },
    "5": { class_type: "KSampler", inputs: { seed: ["40", 0], steps: 30, cfg: 5, sampler_name: "euler_ancestral", scheduler: "normal", denoise: 1, model: ["48", 1], positive: ["19", 1], negative: ["32", 1], latent_image: ["24", 0] } },
    "19": { class_type: "WeiLinPromptUIWithoutLora", _meta: { title: "补充tag" }, inputs: { positive: "1girl, standing", temp_str: '[{"text":"editor undo state, never the prompt"}]', opt_text: ["21", 0], opt_clip: ["48", 0] } },
    "21": { class_type: "String Literal (Image Saver)", inputs: { string: "masterpiece, best quality," } },
    "23": { class_type: "ResolutionMasterSimplify", inputs: { width: 1088, height: 1408 } },
    "24": { class_type: "EmptyLatentImage", inputs: { width: ["23", 0], height: ["23", 1], batch_size: 1 } },
    "32": { class_type: "WeiLinPromptUIWithoutLora", _meta: { title: "Negative" }, inputs: { positive: "worst quality, bad hands", opt_clip: ["48", 0] } },
    "37": { class_type: "WeiLinPromptUIOnlyLoraStack", inputs: { lora_str: JSON.stringify([{ name: "left/on/canvas", weight: 0.9, lora: "left/on/canvas.safetensors" }]) } },
    "40": { class_type: "Seed (rgthree)", inputs: { seed: 80451992085589 } },
    "48": { class_type: "WeiLinPromptUIOnlyLoraStack", inputs: {
      lora_str: JSON.stringify([
        { name: "Anima/Style/kazutake", weight: 0.75, lora: "Anima/Style/kazutake.safetensors", hidden: false },
        { name: "Anima/Style/switched-off", weight: 0.4, lora: "Anima/Style/switched-off.safetensors", hidden: true },
      ]),
      clip: ["2", 0],
      model: ["3", 0],
    } },
    "50": { class_type: "CLIPTextEncode", inputs: { text: "a disconnected encoder nobody wired up" } },
  };
}

test("a prompt-editor plugin's text is found under its own widget name", () => {
  const info = parseComfyUi(pluginWorkflow());

  // `positive`/`negative` is what the node calls its widget, not what the node
  // does; only the sampler's wiring says which is which.
  assert.equal(info.positive, "masterpiece, best quality, 1girl, standing");
  assert.equal(info.negative, "worst quality, bad hands");
  // The editor's own undo state sits in the same node and is not a prompt.
  assert.doesNotMatch(info.positive, /undo state/);
  // Falling back to "the first CLIPTextEncode" would report a node the sampler
  // never read, which is worse than reporting nothing.
  assert.doesNotMatch(info.positive, /disconnected/);
  assert.doesNotMatch(info.negative, /disconnected/);
});

test("linked widget values are followed by name instead of printed as link ids", () => {
  const parameters = Object.fromEntries(parseComfyUi(pluginWorkflow()).parameters.map((item) => [item.label, item.value]));

  // Both of these arrive as `[nodeId, slot]`; interpolating them would print
  // "23,0x23,1" and "40,0", which read as plausible values and are not.
  assert.equal(parameters.Size, "1088x1408");
  assert.equal(parameters.Seed, "80451992085589");
  assert.equal(parameters.Steps, "30");

  // A size that genuinely cannot be resolved is omitted, never guessed.
  const live = parseComfyUi({
    "1": { class_type: "GetImageSize", inputs: { image: ["9", 0] } },
    "2": { class_type: "EmptyLatentImage", inputs: { width: ["1", 0], height: ["1", 1] } },
    "3": { class_type: "KSampler", inputs: { steps: 8, latent_image: ["2", 0] } },
  });
  assert.equal(live.parameters.some((item) => item.label === "Size"), false);
});

test("a LoRA stack widget is read, and only the stack the sampler was wired to", () => {
  const info = parseComfyUi(pluginWorkflow());

  // One widget holds the whole stack, so there is no `LoraLoader` to count.
  assert.deepEqual(info.loras, [{ name: "Anima/Style/kazutake.safetensors", weight: 0.75 }]);
  assert.equal(info.checkpoint, "anima-base-v1.0.safetensors");

  // LoraManager spells the same idea differently, and switched-off rows stay in
  // the payload after the user turns them off.
  const managed = parseComfyUi({
    "1": { class_type: "UNETLoader", inputs: { unet_name: "wai.safetensors" } },
    "2": { class_type: "Lora Loader (LoraManager)", inputs: {
      text: "<lora:ashima_v4:1.00> <lora:switched_off:1.00>",
      loras: { __value__: [{ name: "ashima_v4", strength: 0.6, active: true }, { name: "switched_off", strength: 1, active: false }] },
      model: ["1", 0],
    } },
    "3": { class_type: "KSampler", inputs: { steps: 20, model: ["2", 0] } },
  });
  assert.deepEqual(managed.loras, [{ name: "ashima_v4", weight: 0.6 }]);
});

test("Python's non-finite floats cost the widget that holds one, not the graph", () => {
  // `json.dumps` writes these bare and `JSON.parse` rejects them, which used to
  // drop a whole ComfyUI export to "unknown source".
  assert.deepEqual(parseLooseJson('{"a": NaN, "b": -Infinity, "c": Infinity}'), { a: null, b: null, c: null });
  // Only outside string literals, escapes included.
  assert.deepEqual(parseLooseJson('{"a": "NaN", "b": "quote \\" then Infinity"}'), { a: "NaN", b: 'quote " then Infinity' });
  assert.equal(parseLooseJson("not json at all"), null);
  assert.equal(parseLooseJson('{"a": 1}').a, 1);

  const info = interpretImageMetadata({
    prompt: '{"1": {"class_type": "KSampler", "inputs": {"steps": 20, "cfg": NaN, "positive": ["2", 0]}}, "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a NaN inside a string survives"}}}',
  });
  assert.equal(info.source, "comfyui");
  assert.equal(info.positive, "a NaN inside a string survives");
});

test("every summary field says which node it was read out of", () => {
  const roles = comfyNodeRoles(pluginWorkflow());

  assert.deepEqual(roles["19"], ["正向提示词"]);
  assert.deepEqual(roles["32"], ["负向提示词"]);
  assert.deepEqual(roles["3"], ["底模"]);
  assert.deepEqual(roles["24"], ["尺寸"]);
  assert.deepEqual(roles["5"], ["采样"]);
  assert.deepEqual(roles["48"], ["LoRA"]);
  // The stack left on the canvas contributed nothing and is not credited.
  assert.equal(roles["37"], undefined);
});

test("this project's own outputs read back with engine, flags and Anima assets", () => {
  const parameters = {
    job_id: "job-1", engine: "iL", checkpoint: "waiNSFW.safetensors", checkpoint_sha256: "deadbeef",
    steps: 30, cfg: 5, sampler: "DPM++ 2M", scheduler: "Karras", seed: "42", base_seed: "42",
    width: 832, height: 1216, output_width: 1664, output_height: 2432, denoise: 1,
    hires: { upscaler: "4x-UltraSharp" }, adetailer: { detector: "face_yolov8n.pt" }, rtx: { quality: 3 },
    loras: [{ name: "ink.safetensors", weight: 0.6 }],
  };
  const info = interpretImageMetadata({ parameters: JSON.stringify(parameters), prompt: "positive text", negative_prompt: "negative text" });

  assert.equal(info.status, IMAGE_INFO_STATUS.ok);
  assert.equal(info.source, "xiria");
  assert.equal(info.engine, "iL");
  assert.equal(info.checkpoint, "waiNSFW.safetensors");
  assert.equal(info.positive, "positive text");
  assert.equal(info.negative, "negative text");
  assert.deepEqual(info.loras, [{ name: "ink.safetensors", weight: 0.6 }]);
  const enabled = info.flags.filter((flag) => flag.enabled).map((flag) => flag.id);
  assert.deepEqual(enabled, ["hires", "adetailer", "rtx"]);
  // Hires changed the size, so both the requested and final sizes are shown.
  assert.equal(info.parameters.find((item) => item.label === "Size").value, "1664x2432");
  assert.equal(info.parameters.find((item) => item.label === "Requested size").value, "832x1216");

  const anima = interpretImageMetadata({ parameters: JSON.stringify({ engine: "Anima", diffusion_model: "anima.safetensors", text_encoder: "qwen.safetensors", vae: "anima-vae.safetensors" }) });
  assert.equal(anima.checkpoint, "anima.safetensors");
  assert.equal(anima.animaAssets.text_encoder, "qwen.safetensors");
});

test("detection is by shape, so ComfyUI and A1111 never collide on `parameters`", () => {
  // Both write a `parameters` key; only one of them writes JSON.
  const a1111 = interpretImageMetadata({ parameters: "prompt\nSteps: 20, Seed: 1" });
  assert.equal(a1111.source, "a1111");
  const comfy = interpretImageMetadata({ prompt: JSON.stringify({ "1": { class_type: "KSampler", inputs: { steps: 8 } } }) });
  assert.equal(comfy.source, "comfyui");
  // A ComfyUI file also carries a `prompt` chunk, but so does ours — ours is
  // plain text, so a JSON graph is what tells them apart.
  const ours = interpretImageMetadata({ prompt: "just a prompt", parameters: JSON.stringify({ engine: "SD", steps: 20 }) });
  assert.equal(ours.source, "xiria");
});

test("model matching prefers local over shared at every precision level", () => {
  const catalog = [
    { origin: "shared", path: "sdxl/animagine.safetensors", base: "animagine.safetensors", stem: "animagine", label: "ComfyUI" },
    { origin: "local", path: "il/animagine.safetensors", base: "animagine.safetensors", stem: "animagine", label: "models/checkpoints/illustrious" },
    { origin: "shared", path: "loose.ckpt", base: "loose.ckpt", stem: "loose", label: "sd-webui" },
  ];

  // A bare filename is the common case and must still resolve locally first.
  const byName = matchModelName("animagine.safetensors", catalog);
  assert.equal(byName.status, "local");
  assert.equal(byName.precision, "filename");

  // An exact relative path that only the shared root has still resolves.
  const shared = matchModelName("SDXL\\animagine.safetensors", catalog);
  assert.equal(shared.status, "shared");
  assert.equal(shared.precision, "exact");

  // Extension differences are the last resort, not the first.
  assert.equal(matchModelName("loose.safetensors", catalog).status, "shared");
  assert.equal(matchModelName("nothing-like-this.safetensors", catalog).status, "missing");
  assert.equal(matchModelName("", catalog).status, "unknown");
});
