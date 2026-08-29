import assert from "node:assert/strict";
import path from "node:path";
import test from "node:test";

import {
  configuredModelDirectories,
  configuredModelDirectory,
  groupLoraModels,
  mergeModelPaths,
} from "./model-paths.mjs";

test("model roots accept custom names and arbitrary nesting inside models", () => {
  const root = path.resolve("fixture-project");
  const config = {
    checkpoints: { sd: "models/My Models/SD/checkpoints/deep" },
    loras: { anima: "models/自定义/Anima LoRA/多层" },
    upscalers: "models/tools/upscale/custom",
  };

  assert.equal(configuredModelDirectory(config, "checkpoints", root, "sd"), path.join(root, "models", "My Models", "SD", "checkpoints", "deep"));
  assert.equal(configuredModelDirectory(config, "loras", root, "anima"), path.join(root, "models", "自定义", "Anima LoRA", "多层"));
  assert.equal(configuredModelDirectory(config, "upscalers", root), path.join(root, "models", "tools", "upscale", "custom"));
  assert.equal(mergeModelPaths({}).loras.anima, "models/loras/anima");
  assert.equal(configuredModelDirectories(config, root).length, 13);
});

test("model roots reject empty, absolute, root, and traversal paths", () => {
  const root = path.resolve("fixture-project");
  for (const value of ["", "models", "../outside", path.resolve("outside")]) {
    assert.throws(() => configuredModelDirectory({ upscalers: value }, "upscalers", root), /模型目录/);
  }
  assert.throws(() => mergeModelPaths({ checkpoints: "models/checkpoints" }), /checkpoints/);
});

test("Anima LoRA discovery groups arbitrary top-level folders and keeps nested values", () => {
  const projectRoot = path.resolve("fixture-project");
  const loraRoot = path.join(projectRoot, "models", "loras", "anima");
  const categories = groupLoraModels([
    { name: "Loose.safetensors", value: "Loose.safetensors", size: 1 },
    { name: "character/deep/Hero.safetensors", value: "character/deep/Hero.safetensors", size: 2 },
    { name: "My Folder/level/two/Style.safetensors", value: "My Folder/level/two/Style.safetensors", size: 3 },
  ], loraRoot, projectRoot);

  assert.equal(categories.find((item) => item.id === "root").models[0].value, "Loose.safetensors");
  assert.equal(categories.find((item) => item.id === "character").models[0].name, "deep/Hero.safetensors");
  const custom = categories.find((item) => item.label === "My Folder");
  assert.equal(custom.models[0].name, "level/two/Style.safetensors");
  assert.equal(custom.models[0].value, "My Folder/level/two/Style.safetensors");
});
