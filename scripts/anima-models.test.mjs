import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, mkdir, readFile, rm, utimes, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  ANIMA_RUNTIME_ARTIFACTS,
  animaRuntimeArtifactStatuses,
  bundledAnimaTokenizerDirectory,
  classifyAnimaSafetensorsHeader,
  discoverAnimaModels,
  MAX_SAFETENSORS_HEADER_BYTES,
  normalizeAnimaSafetensorsKey,
  readSafetensorsHeader,
  requireBundledAnimaTokenizers,
} from "./anima-models.mjs";

const tensor = (shape = [1]) => ({ dtype: "F32", shape });

function completeHeader(entries, minimumTensors = Object.keys(entries).length) {
  const header = { ...entries };
  for (let index = Object.keys(header).length; index < minimumTensors; index += 1) {
    header[`fixture.${index}`] = tensor();
  }
  let offset = 0;
  for (const descriptor of Object.values(header)) {
    const bytes = descriptor.shape.reduce((total, dimension) => total * dimension, 1) * 4;
    descriptor.data_offsets = [offset, offset + bytes];
    offset += bytes;
  }
  return header;
}

const diffusionHeader = (prefix = "") => completeHeader({
  [`${prefix}llm_adapter.embed.weight`]: tensor([32128, 1024]),
  [`${prefix}x_embedder.proj.1.weight`]: tensor([2048, 68]),
  [`${prefix}blocks.0.self_attn.q_proj.weight`]: tensor([2048, 2048]),
  [`${prefix}blocks.27.self_attn.q_proj.weight`]: tensor([2048, 2048]),
  [`${prefix}final_layer.linear.weight`]: tensor([64, 2048]),
  [`${prefix}llm_adapter.blocks.5.mlp.2.weight`]: tensor([1024, 4096]),
}, 680);

const textEncoderHeader = (prefix = "model.") => completeHeader({
  [`${prefix}embed_tokens.weight`]: tensor([151936, 1024]),
  [`${prefix}layers.0.self_attn.q_proj.weight`]: tensor([2048, 1024]),
  [`${prefix}layers.27.self_attn.q_proj.weight`]: tensor([2048, 1024]),
  [`${prefix}norm.weight`]: tensor([1024]),
}, 300);

const diffusersVaeHeader = () => completeHeader({
  "encoder.conv_in.weight": tensor([96, 3, 3, 3, 3]),
  "decoder.conv_out.weight": tensor([3, 96, 3, 3, 3]),
  "quant_conv.weight": tensor([32, 32, 1, 1, 1]),
  "post_quant_conv.weight": tensor([16, 16, 1, 1, 1]),
}, 190);

const comfyVaeHeader = () => completeHeader({
  "encoder.conv1.weight": tensor([96, 3, 3, 3, 3]),
  "decoder.head.2.weight": tensor([3, 96, 3, 3, 3]),
  "conv1.weight": tensor([32, 32, 1, 1, 1]),
  "conv1.bias": tensor([32]),
  "conv2.weight": tensor([16, 16, 1, 1, 1]),
  "encoder.middle.0.residual.0.gamma": tensor([96]),
  "decoder.middle.0.residual.0.gamma": tensor([384]),
}, 190);

async function writeSafetensors(filePath, header) {
  const json = Buffer.from(JSON.stringify(header));
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(json.length));
  const dataBytes = Math.max(0, ...Object.values(header).filter((value) => value?.data_offsets).map((value) => value.data_offsets[1]));
  await writeFile(filePath, Buffer.concat([prefix, json, Buffer.alloc(dataBytes)]));
}

test("normalizes exactly one known Anima diffusion prefix", () => {
  assert.equal(normalizeAnimaSafetensorsKey("net.blocks.0.weight"), "blocks.0.weight");
  assert.equal(normalizeAnimaSafetensorsKey("model.diffusion_model.blocks.0.weight"), "blocks.0.weight");
  assert.equal(normalizeAnimaSafetensorsKey("diffusion_model.net.blocks.0.weight"), "net.blocks.0.weight");
  assert.equal(normalizeAnimaSafetensorsKey("blocks.0.weight"), "blocks.0.weight");
});

test("classifies Anima diffusion, Qwen3-0.6B, and both supported Qwen Image VAE formats", () => {
  for (const prefix of ["", "net.", "model.diffusion_model.", "diffusion_model."]) {
    assert.equal(classifyAnimaSafetensorsHeader(diffusionHeader(prefix)), "diffusion_model");
  }
  assert.equal(classifyAnimaSafetensorsHeader(textEncoderHeader()), "text_encoder");
  assert.equal(classifyAnimaSafetensorsHeader(textEncoderHeader("")), "text_encoder");
  assert.equal(classifyAnimaSafetensorsHeader(diffusersVaeHeader()), "vae");
  assert.equal(classifyAnimaSafetensorsHeader(comfyVaeHeader()), "vae");
});

test("rejects incompatible, malformed, colliding, and ambiguous headers", () => {
  assert.equal(classifyAnimaSafetensorsHeader(completeHeader({ "double_blocks.0.img_attn.qkv.weight": tensor([3072, 3072]) })), null);
  assert.equal(classifyAnimaSafetensorsHeader(completeHeader({ "model.embed_tokens.weight": tensor([152064, 3584]) })), null);
  assert.equal(classifyAnimaSafetensorsHeader(completeHeader({ "net.llm_adapter.embed.weight": tensor([32128, 1024]), "llm_adapter.embed.weight": tensor([32128, 1024]) })), null);
  assert.equal(classifyAnimaSafetensorsHeader({ ...textEncoderHeader(), ...diffusersVaeHeader() }), null);
  assert.equal(classifyAnimaSafetensorsHeader({ broken: { dtype: "F32", shape: [1] } }), null);
});

test("reads only a bounded Safetensors JSON header", async (context) => {
  const temporary = await mkdtemp(path.join(os.tmpdir(), "xirai-anima-header-"));
  context.after(() => rm(temporary, { recursive: true, force: true }));
  const valid = path.join(temporary, "valid.safetensors");
  const validHeader = completeHeader({ value: tensor([1]) });
  await writeSafetensors(valid, validHeader);
  assert.deepEqual(await readSafetensorsHeader(valid), validHeader);

  const excessive = path.join(temporary, "excessive.safetensors");
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(MAX_SAFETENSORS_HEADER_BYTES + 1));
  await writeFile(excessive, prefix);
  await assert.rejects(readSafetensorsHeader(excessive), /exceeds the limit/);

  const corrupt = path.join(temporary, "corrupt.safetensors");
  const corruptJson = Buffer.from("{not-json}");
  const corruptPrefix = Buffer.alloc(8);
  corruptPrefix.writeBigUInt64LE(BigInt(corruptJson.length));
  await writeFile(corrupt, Buffer.concat([corruptPrefix, corruptJson]));
  await assert.rejects(readSafetensorsHeader(corrupt), SyntaxError);

  const truncated = path.join(temporary, "truncated.safetensors");
  const truncatedHeader = completeHeader({ value: tensor([2]) });
  const truncatedJson = Buffer.from(JSON.stringify(truncatedHeader));
  const truncatedPrefix = Buffer.alloc(8);
  truncatedPrefix.writeBigUInt64LE(BigInt(truncatedJson.length));
  await writeFile(truncated, Buffer.concat([truncatedPrefix, truncatedJson, Buffer.alloc(4)]));
  await assert.rejects(readSafetensorsHeader(truncated), /data section/);
});

test("recursively discovers only compatible Safetensors in their configured component roots", async (context) => {
  const temporary = await mkdtemp(path.join(os.tmpdir(), "xirai-anima-discovery-"));
  context.after(() => rm(temporary, { recursive: true, force: true }));
  const roots = {
    diffusion_model: path.join(temporary, "diffusion"),
    text_encoder: path.join(temporary, "text"),
    vae: path.join(temporary, "vae"),
  };
  await Promise.all([mkdir(path.join(roots.diffusion_model, "nested"), { recursive: true }), mkdir(roots.text_encoder), mkdir(roots.vae)]);
  const diffusionPath = path.join(roots.diffusion_model, "nested", "Anima.safetensors");
  await Promise.all([
    writeFile(diffusionPath, "fixture"),
    writeFile(path.join(roots.diffusion_model, "flux.safetensors"), "fixture"),
    writeFile(path.join(roots.diffusion_model, "wrong-role.safetensors"), "fixture"),
    writeFile(path.join(roots.text_encoder, "qwen.safetensors"), "fixture"),
    writeFile(path.join(roots.vae, "qwen-image.safetensors"), "fixture"),
    writeFile(path.join(roots.diffusion_model, "named-anima.ckpt"), "not executable"),
    writeFile(path.join(roots.vae, "corrupt.safetensors"), "broken"),
  ]);
  const modified = new Date("2026-07-20T12:34:56.000Z");
  await utimes(diffusionPath, modified, modified);

  const headers = new Map([
    ["Anima.safetensors", diffusionHeader("net.")],
    ["flux.safetensors", completeHeader({ "double_blocks.0.img_attn.qkv.weight": tensor([3072, 3072]) })],
    ["wrong-role.safetensors", textEncoderHeader()],
    ["qwen.safetensors", textEncoderHeader()],
    ["qwen-image.safetensors", comfyVaeHeader()],
  ]);
  const discovered = await discoverAnimaModels(roots, async (filePath) => {
    if (path.basename(filePath) === "corrupt.safetensors") throw new Error("broken");
    return headers.get(path.basename(filePath));
  });
  assert.deepEqual(discovered.diffusion_model.map((model) => model.value), ["nested/Anima.safetensors"]);
  assert.deepEqual(discovered.text_encoder.map((model) => model.value), ["qwen.safetensors"]);
  assert.deepEqual(discovered.vae.map((model) => model.value), ["qwen-image.safetensors"]);
  assert.equal(discovered.diffusion_model[0].name, "nested/Anima.safetensors");
  assert.equal(discovered.diffusion_model[0].size, (await readFile(diffusionPath)).length);
  assert.equal(discovered.diffusion_model[0].modifiedAt, modified.getTime());
});

test("Anima tokenizers are bundled program resources and not recommended downloads", async () => {
  const root = path.resolve(import.meta.dirname, "..");
  const catalog = JSON.parse(await readFile(path.join(root, "models", "recommended-models.json"), "utf8"));
  const byId = new Map(catalog.artifacts.map((artifact) => [artifact.id, artifact]));
  assert.equal(byId.has("anima-qwen3-tokenizer"), false);
  assert.equal(byId.has("anima-qwen3-tokenizer-config"), false);
  assert.equal(byId.has("anima-t5-tokenizer"), false);
  for (const family of [...catalog.civitaiFamilies, ...catalog.staticFamilies].filter((item) => item.group === "Anima")) {
    assert.deepEqual(family.runtimeConfigs || [], [], family.id);
  }

  const expected = {
    qwen_tokenizer: {
      canonical: ["anima-qwen3-tokenizer.json", 7031645, "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"],
      alternates: [[7334926, "47ec9be242d3ef39b9c97ac0a3f06c1752f061b234e295bc0a2842067a3fe4f9"]],
    },
    qwen_tokenizer_config: {
      canonical: ["anima-qwen3-tokenizer-config.json", 9678, "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5"],
      alternates: [[9916, "7992a7924330571ac9b97d58e39d4a4993ccdb865335034cec29cf2c482fd460"]],
    },
    t5_tokenizer: {
      canonical: ["anima-t5-tokenizer.json", 1389353, "d2acde0d8d71dd30a711834b07781b9c89feaac33fd332f60507699282740066"],
      alternates: [],
    },
  };
  const statuses = await requireBundledAnimaTokenizers(root);
  assert.equal(path.normalize(bundledAnimaTokenizerDirectory(root)), path.normalize(path.join(root, "backend", "resources", "anima-tokenizers")));
  for (const [name, { canonical: [filename, size, sha256], alternates }] of Object.entries(expected)) {
    const artifact = ANIMA_RUNTIME_ARTIFACTS[name];
    assert.deepEqual([artifact.filename, artifact.size, artifact.sha256], [filename, size, sha256], `${name} canonical`);
    assert.deepEqual((artifact.alternates || []).map((variant) => [variant.size, variant.sha256]), alternates, `${name} alternates`);
    const bytes = await readFile(path.join(bundledAnimaTokenizerDirectory(root), filename));
    const canonicalBytes = Buffer.from(bytes.toString("utf8").replace(/\r\n/g, "\n"), "utf8");
    assert.deepEqual([canonicalBytes.length, createHash("sha256").update(canonicalBytes).digest("hex")], [size, sha256], `${name} canonical bytes`);
    const digest = createHash("sha256").update(bytes).digest("hex");
    assert.ok([[size, sha256], ...alternates].some(([variantSize, variantDigest]) => bytes.length === variantSize && digest === variantDigest), name);
    assert.equal(statuses[name].installed, true, name);
  }

  const vite = await readFile(path.join(root, "vite.config.js"), "utf8");
  assert.match(vite, /const checkpointEnginePathKeys = \{ SD: "sd", iL: "illustrious" \}/);
  assert.match(vite, /const loraEnginePathKeys = \{ SD: "sd", iL: "illustrious", Anima: "anima", Flux: "flux", Flux2: "flux2", Krea2: "krea2" \}/);
  assert.doesNotMatch(vite, /checkpointEnginePathKeys = \{[^}]*Anima/);
  // A request that names no engine is a caller mistake, and the refusal has to read as one: the
  // bare "Unsupported model engine" was taken as evidence of a corrupt model library.
  assert.match(vite, /Unsupported model engine \$\{engine \? `"\$\{engine\}"` : "\(missing\)"\} for \$\{type\}; expected one of: \$\{supported\}/);
  assert.match(vite, /Unsupported LoRA engine \$\{engine \? `"\$\{engine\}"` : "\(missing\)"\}; expected one of: /);
  assert.match(vite, /result\.loras\[engine\].*configuredModelDirectory\(config, "loras"/);
  assert.match(vite, /nativeEngines\.has\(engine\) \? animaLoraExtensions : loraExtensions/);
  assert.match(vite, /nativeEngines\.has\(payload\.engine\) && kind !== "lora"/);
  assert.match(vite, /kind === "lora" && nativeEngines\.has\(payload\.engine\) && !animaLoraExtensions/);
  assert.doesNotMatch(vite, /family\.runtimeConfigs/);
  assert.match(vite, /bundledAnimaTokenizerDirectory\(projectRoot\)/);
  assert.match(vite, /filterPendingRecommendedArtifacts\(uniqueArtifacts, installed/);
  assert.match(vite, /enqueueModelDownloadBatch\(/);
  assert.match(vite, /\/api\/model-download\/retry/);
  assert.match(vite, /role === "config" \? "configs"/);
  assert.match(vite, /artifact\.filename \|\| path\.posix\.basename\(location\.filePath\)/);
  assert.match(vite, /apiHosts = url\.hostname\.toLowerCase\(\) === "hf-mirror\.com"/);
  assert.doesNotMatch(vite, /!apiKeys?\.huggingface \? \[\{ id: "hf-mirror"/);
  assert.match(vite, /\{ id: "hf-mirror", label: "HF-Mirror", url: `https:\/\/hf-mirror\.com\/\$\{relative\}` \}/);

  // The downloader lives inside the lazily loaded toolbox route and splits again into its own
  // chunk; the workspace only keeps the tab switch.
  const [app, downloader, toolbox] = await Promise.all([
    readFile(path.join(root, "src", "App.jsx"), "utf8"),
    readFile(path.join(root, "src", "ModelDownloader.jsx"), "utf8"),
    readFile(path.join(root, "src", "ToolboxPage.jsx"), "utf8"),
  ]);
  assert.match(toolbox, /lazy\(\(\) => import\("\.\/ModelDownloader"\)\)/);
  assert.match(app, /fetch\("\/api\/inference\/jobs\/active"/);
  // Active-job polling still pauses while the toolbox is open, which is where the downloader
  // now lives and runs its own queue polling.
  assert.match(app, /if \(activePage === "toolbox"\) return undefined/);
  assert.match(downloader, /fetch\("\/api\/model-download\/job"/);
  assert.match(downloader, /endpoint: "\/api\/model-download\/retry"/);
  assert.match(downloader, /isDownloading \? "添加到队列" : "开始下载"/);
  assert.doesNotMatch(downloader, /recommended-zone-trigger" type="button" disabled=\{isDownloading\}/);
  for (const source of [app, downloader]) {
    assert.doesNotMatch(source, /Runtime Resources/);
    assert.doesNotMatch(source, /failures >= 12/);
  }

  const [start, setup, updateValidation] = await Promise.all([
    readFile(path.join(root, "scripts", "start.mjs"), "utf8"),
    readFile(path.join(root, "scripts", "setup.mjs"), "utf8"),
    readFile(path.join(root, "scripts", "update-validation.mjs"), "utf8"),
  ]);
  assert.match(start, /requireBundledAnimaTokenizers\(projectRoot\)/);
  assert.match(setup, /requireBundledAnimaTokenizers\(projectRoot\)/);
  assert.match(updateValidation, /inference_server\.anima_tokenizer_sources\(\)/);
});

test("bundled runtime artifacts require their canonical filenames and exact digests", async (context) => {
  const temporary = await mkdtemp(path.join(os.tmpdir(), "xirai-anima-runtime-"));
  context.after(() => rm(temporary, { recursive: true, force: true }));
  const nested = path.join(temporary, "nested");
  await mkdir(nested);
  await writeFile(path.join(nested, "renamed.json"), "valid");
  const artifacts = {
    fixture: { filename: "canonical.json", size: 5, sha256: "valid-digest" },
  };
  const hash = async (file) => (await readFile(file.path, "utf8")) === "valid" ? "valid-digest" : "wrong-digest";

  let statuses = await animaRuntimeArtifactStatuses(temporary, hash, artifacts);
  assert.equal(statuses.fixture.installed, false);
  assert.equal(statuses.fixture.reason, "not installed");

  await writeFile(path.join(temporary, "canonical.json"), "wrong");
  statuses = await animaRuntimeArtifactStatuses(temporary, hash, artifacts);
  assert.equal(statuses.fixture.installed, false);
  assert.equal(statuses.fixture.reason, "SHA-256 mismatch");

  await writeFile(path.join(temporary, "canonical.json"), "valid");
  statuses = await animaRuntimeArtifactStatuses(temporary, hash, artifacts);
  assert.equal(statuses.fixture.installed, true);
  assert.equal(statuses.fixture.reason, null);
});

test("bundled tokenizer guard rejects missing or corrupt program resources", async (context) => {
  const root = await mkdtemp(path.join(os.tmpdir(), "xirai-anima-bundled-"));
  context.after(() => rm(root, { recursive: true, force: true }));
  const directory = bundledAnimaTokenizerDirectory(root);
  await mkdir(directory, { recursive: true });
  await writeFile(path.join(directory, "anima-qwen3-tokenizer.json"), "{}");
  await assert.rejects(requireBundledAnimaTokenizers(root), /missing or corrupt/);
});
