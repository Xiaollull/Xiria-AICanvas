import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

// Applying a picture read by the image reader failed for a Krea 2 image with "the card's Anima
// diffusion model is not installed". Three faults were stacked; the model field itself is covered in
// `image-info-apply.test.mjs`, and these pin the two that live in the control plane and the workspace.

const readSource = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("the reader's catalogue tags a diffusion model with the engine that owns it", async () => {
  const vite = await readSource("vite.config.js");
  const catalogs = vite.slice(vite.indexOf("async function buildModelCatalogs"), vite.indexOf("for (const engine of Object.keys(loraEnginePathKeys))"));
  // Every split engine shares one diffusion folder, so the folder cannot name the engine. The old
  // code tagged all of it as Anima, which is how a Krea 2 model became an "Anima" model.
  assert.doesNotMatch(catalogs, /\{ engine: "Anima" \}/);
  for (const discover of ["discoverAnimaModels", "discoverFluxModels", "discoverFlux2Models", "discoverKrea2Models"]) {
    assert.match(catalogs, new RegExp(`\\["\\w+", ${discover}\\]`), `${discover} is not consulted`);
  }
  assert.match(catalogs, /catalogEntry\("local", model\.value, `\$\{relative\} · \$\{engine\}`, \{ engine \}\)/);
  // Claimed in a fixed order, so a file two engines accept always lands on the same one.
  assert.match(catalogs, /if \(claimed\.has\(model\.value\)\) continue;/);
  // A file no engine recognises is still reported as found, but names no engine.
  assert.match(catalogs, /if \(!claimed\.has\(model\.value\)\) checkpoints\.push\(catalogEntry\("local", model\.value, relative\)\);/);
});

test("applying another engine's picture takes that engine's encoder and VAE", async () => {
  const app = await readSource("src/App.jsx");
  const apply = app.slice(app.indexOf("const applyImageInfoParameters = async"), app.indexOf('label: "图片参数"'));
  // Not the selected engine's: Anima's Qwen3 0.6B under a Krea 2 diffusion model would be wrong.
  assert.match(apply, /if \(targetEngine !== snapshot\.model\) \{/);
  assert.match(apply, /engineSettingsFor\(workspaceSnapshot\.current\.engineSettingsByEngine, targetEngine\)/);
  for (const field of ["diffusionModel", "textEncoder", "textEncoder2", "vae"]) {
    assert.match(apply, new RegExp(`${field}: record\\.${field},`), `${field} is not taken from the target engine`);
  }
});

test("an empty encoder or VAE is filled only when exactly one is installed", async () => {
  const app = await readSource("src/App.jsx");
  // An engine never opened here has an empty record; with a single candidate there is no choice
  // to make, so the picture applies instead of being refused. Two or more still refuse.
  assert.match(app, /if \(!normalized\[field\] && catalog\.length === 1\) normalized\[field\] = catalog\[0\]\.value;/);
});

test("a Gallery card never carries the per-engine parameter library", async () => {
  // The workspace snapshot gained every engine's parameters with the per-engine split. Without
  // these, each card saved would embed all of them, and applying it could rewrite every engine.
  const app = await readSource("src/App.jsx");
  assert.match(app, /engineSettingsByEngine: _engineSettingsByEngine, \.\.\.gallerySettings \} = settings;/);
  assert.match(app, /delete source\.engineSettingsByEngine;/);
  const core = await readSource("src/gallery-core.js");
  assert.match(core, /delete normalized\.engineSettingsByEngine;/);
});

test("copying an image outside a secure context says why instead of throwing a TypeError", async () => {
  const app = await readSource("src/App.jsx");
  assert.match(app, /if \(!globalThis\.isSecureContext \|\| !navigator\.clipboard\?\.write \|\| typeof ClipboardItem === "undefined"\)/);
  assert.match(app, /浏览器只在 HTTPS 或 localhost 页面允许复制图片/);
});

test("a picture's borrowed encoder or VAE can be replaced; its diffusion model and a card's files cannot", async () => {
  const app = await readSource("src/App.jsx");
  // Only the image reader asks for this; a Gallery card's recorded files are still checked as-is.
  assert.match(app, /label: "图片参数", fillMissingAssets: true \}\);/);
  assert.equal((app.match(/fillMissingAssets: true/g) || []).length, 1, "only the picture path may opt in");
  assert.match(app, /else if \(fillMissingAssets && field !== "diffusionModel" && catalog\.length === 1 && !catalog\.some\(\(item\) => item\.value === normalized\[field\]\)\) normalized\[field\] = catalog\[0\]\.value;/);
});
