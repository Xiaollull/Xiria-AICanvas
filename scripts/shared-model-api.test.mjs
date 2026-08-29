import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { SHARED_CATEGORY_ID, SHARED_ROOT_FOLDER_LABEL, shapeSharedLoraCategory } from "./shared-model-paths.mjs";
import { defaultLoraCategories } from "./model-paths.mjs";

const projectDirectory = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const viteConfig = await readFile(path.join(projectDirectory, "vite.config.js"), "utf8");

function functionBody(source, signature) {
  const start = source.indexOf(signature);
  assert.notEqual(start, -1, `${signature} is gone from vite.config.js`);
  let depth = 0;
  let index = source.indexOf("{", start);
  const open = index;
  for (; index < source.length; index += 1) {
    if (source[index] === "{") depth += 1;
    else if (source[index] === "}" && (depth -= 1) === 0) return source.slice(open, index + 1);
  }
  throw new Error(`${signature} body is unbalanced`);
}

function sharedModel(rootId, name, size) {
  return { name, value: `shared:${rootId}/${name}`, size, modifiedAt: 0, shared: true, rootId };
}

test("the shared category exposes a folder tree and a flat list at the same time", () => {
  const category = shapeSharedLoraCategory([
    {
      root: { id: "95746e6dad38", label: "sd-webui Lora", path: "F:/AI/Lora", kind: "loras" },
      models: [
        sharedModel("95746e6dad38", "style/ink.safetensors", 3),
        sharedModel("95746e6dad38", "loose.safetensors", 1),
        sharedModel("95746e6dad38", "character/hero.safetensors", 5),
        sharedModel("95746e6dad38", "character/deep/nested.safetensors", 7),
      ],
    },
    {
      root: { id: "5dff1a102833", label: "ComfyUI", path: "D:/ComfyUI/models", kind: "auto" },
      models: [sharedModel("5dff1a102833", "loras/extra.safetensors", 2)],
    },
  ]);

  assert.equal(category.id, SHARED_CATEGORY_ID);
  assert.equal(category.label, "共享");
  assert.equal(category.shared, true);
  // The flat list is what the category count, the analytics roll-up and the
  // mounted-value lookups all read, so it has to hold every shared file.
  assert.equal(category.models.length, 5);
  assert.equal(category.directory, "2 个共享目录");

  const [webui, comfy] = category.roots;
  assert.equal(webui.files, 4);
  assert.equal(webui.bytes, 16);
  // Loose files first, then folders alphabetically — the tree opens on the
  // shallowest thing rather than burying it under the subfolders.
  assert.deepEqual(webui.folders.map((folder) => folder.name), ["", "character", "character/deep", "style"]);
  assert.equal(webui.folders[0].label, SHARED_ROOT_FOLDER_LABEL);
  assert.equal(webui.folders[1].models[0].value, "shared:95746e6dad38/character/hero.safetensors");
  // An `auto` root keeps the kind folder in the name, so the tree shows where
  // the file actually sits inside the other tool's install.
  assert.equal(comfy.folders[0].name, "loras");
  assert.equal(comfy.label, "ComfyUI");

  const single = shapeSharedLoraCategory([{ root: { id: "95746e6dad38", label: "L", path: "F:/AI/Lora", kind: "loras" }, models: [] }]);
  assert.equal(single.directory, "F:/AI/Lora");
  assert.deepEqual(single.models, []);
});

test("the shared category is served after the four local ones", () => {
  const body = functionBody(viteConfig, "async function modelApi(");
  const grouped = body.indexOf("groupLoraModels(local, directory, projectRoot)");
  const appended = body.indexOf("categories.push(shapeSharedLoraCategory(sharedListings))");
  assert.ok(grouped > 0 && appended > grouped, "shared LoRAs must be appended after the local categories");
  assert.deepEqual(defaultLoraCategories.map((category) => category.id), ["character", "style", "concept", "other"]);

  // Both routes have to be listed or the middleware hands them to `next()`
  // and the whole feature 404s.
  const apiPaths = viteConfig.match(/const apiPaths = \[[^\]]+\]/)[0];
  assert.match(apiPaths, /"\/api\/shared-paths"/);
  assert.match(apiPaths, /"\/api\/shared-paths\/inspect"/);
});

test("a local listing can never claim the shared namespace", () => {
  const body = functionBody(viteConfig, "async function modelApi(");
  // Both listings filter local results through parseSharedRef, so a folder
  // named `shared:` on disk cannot shadow a registered root's reference.
  assert.equal(body.match(/filter\(\(model\) => !parseSharedRef\(model\.value\)\)/g)?.length, 2);
});

test("shared folders are read-only: downloads only ever resolve to local roots", () => {
  const body = functionBody(viteConfig, "async function getDownloadDestination(");
  for (const forbidden of ["shared", "Shared"]) {
    assert.equal(body.includes(forbidden), false, `download destinations must not consider ${forbidden} roots`);
  }
  assert.match(body, /getConfiguredDirectory|getAuxiliaryModelDirectory/);

  // The one place that writes model files must stay pointed at model-paths.json.
  const auxiliary = functionBody(viteConfig, "async function getAuxiliaryModelDirectory(");
  assert.match(auxiliary, /模型目录必须位于项目 models 文件夹中/);
});

test("shared LoRA lookups resolve through the registered roots, not the local one", () => {
  const body = functionBody(viteConfig, "async function validateLoraPath(");
  const sharedBranch = body.indexOf("parseSharedRef(requestedPath)");
  const localResolve = body.indexOf("getConfiguredDirectory(engine, \"loras\")");
  assert.ok(sharedBranch > 0 && sharedBranch < localResolve, "shared refs must be handled before the local root resolve");
  // Engine scoping is applied at resolve time so a root the user detached from
  // Anima cannot be reached through an Anima request.
  assert.match(body, /root\.engines\.includes\(engine\)/);
  assert.match(body, /extensions/);
});

test("a damaged shared config degrades to local-only instead of breaking the library", () => {
  const body = functionBody(viteConfig, "async function sharedListingsFor(");
  assert.match(body, /catch\s*{[\s\S]*return \[\]/);
  assert.match(body, /if \(!root\.enabled\) continue/);
});
