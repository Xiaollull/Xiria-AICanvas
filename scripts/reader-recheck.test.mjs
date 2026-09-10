import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { stripImageInfoMatches } from "../src/image-metadata.js";

// "Apply All" kept failing with "Anima 扩散模型未安装或已移动" after the catalogue was fixed, because
// the image reader restored the model match it had saved -- computed by the old catalogue, which
// tagged every shared diffusion model as Anima -- and applied it without asking again. A match is a
// fact about this machine at one moment, not about the picture; these pin that it is re-checked.

const readSource = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("a resolved record loses every match and can be matched again", () => {
  const resolved = {
    status: "ok",
    source: "comfyui",
    checkpoint: "krea2_turbo_fp8_scaled.safetensors",
    checkpointMatch: { status: "local", match: { value: "krea2_turbo_fp8_scaled.safetensors", engine: "Anima" } },
    loras: [{ name: "NekoXstyle.safetensors", weight: 1, match: { status: "local", match: { engine: "Krea2" } } }],
    // Matching turns these names into { name, match } pairs, which a second pass could not read.
    animaAssets: { text_encoder: { name: "qwen_3_06b.safetensors", match: {} }, vae: { name: "qwen_image_vae.safetensors", match: {} } },
  };
  const stripped = stripImageInfoMatches(resolved);
  assert.equal("checkpointMatch" in stripped, false, "the stale engine tag must not survive");
  assert.deepEqual(stripped.loras, [{ name: "NekoXstyle.safetensors", weight: 1 }]);
  assert.deepEqual(stripped.animaAssets, { text_encoder: "qwen_3_06b.safetensors", vae: "qwen_image_vae.safetensors" });
  // Everything that describes the picture itself is kept.
  assert.equal(stripped.checkpoint, "krea2_turbo_fp8_scaled.safetensors");
  assert.equal(stripped.source, "comfyui");
  // The input is not mutated, so the caller's copy still renders until the fresh one arrives.
  assert.equal(resolved.checkpointMatch.match.engine, "Anima");
});

test("an unresolved record passes through unchanged in shape", () => {
  const plain = stripImageInfoMatches({ status: "ok", checkpoint: "a.safetensors", loras: [{ name: "b" }], animaAssets: { text_encoder: "t", vae: "v" } });
  assert.deepEqual(plain.animaAssets, { text_encoder: "t", vae: "v" });
  assert.deepEqual(plain.loras, [{ name: "b" }]);
  assert.deepEqual(stripImageInfoMatches(null).loras, []);
  assert.equal(stripImageInfoMatches({ status: "ok" }).animaAssets, null);
});

test("the control plane re-checks a parsed record against what is installed now", async () => {
  const vite = await readSource("vite.config.js");
  const route = vite.slice(vite.indexOf('url.pathname === "/api/image-info/resolve"'), vite.indexOf('url.pathname === "/api/image-info/read"'));
  assert.match(route, /request\.method !== "POST"/);
  assert.match(route, /resolveImageInfoModels\(stripImageInfoMatches\(payload\.info\)\)/);
  // Registered, or the plugin never routes the path to the handler at all.
  assert.match(vite, /"\/api\/image-info\/read", "\/api\/image-info\/resolve",/);
});

test("the reader re-checks a restored record, and Apply re-checks before it builds the plan", async () => {
  const reader = await readSource("src/ImageInfoReader.jsx");
  assert.match(reader, /fetch\("\/api\/image-info\/resolve"/);
  assert.match(reader, /if \(restored\.info\) void resolveMatches\(restored\.info\)/);
  assert.match(reader, /if \(token === requestRef\.current\) setInfo\(payload\.info\);/, "a newer read must not be overwritten");
  assert.match(reader, /onRefresh=\{\(\) => resolveMatches\(info\)\}/);

  const apply = await readSource("src/ImageInfoApply.jsx");
  const run = apply.slice(apply.indexOf("const run = async"), apply.indexOf("const missingNames"));
  const refreshed = run.indexOf("await onRefresh()");
  const planned = run.indexOf("buildImageInfoApplyPlan(current");
  assert.ok(refreshed > 0 && planned > refreshed, "the plan must be built from the re-checked record");
  // Re-checking replaces the record object; the selection is keyed on the picture so it survives.
  assert.match(apply, /\}, \[pictureKey\]\);/);
});

test("a refusal names what was being applied instead of always saying Gallery card", async () => {
  const app = await readSource("src/App.jsx");
  const apply = app.slice(app.indexOf("const applyGallerySettings = async"), app.indexOf("const applyImageInfoParameters = async"));
  assert.doesNotMatch(apply, /精选卡片的/);
  assert.match(apply, /throw new Error\(`\$\{label\}的 \$\{normalized\.model\} \$\{assetLabel\}未安装或已移动，当前模型缓存保持不变`\);/);
});
