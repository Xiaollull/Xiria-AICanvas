import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  SHARED_PATHS_CASE_INSENSITIVE,
  classifyDirectoryName,
  formatSharedRef,
  inspectSharedDirectory,
  parseSharedRef,
  readSharedRoots,
  resolveSharedFile,
  sharedKindDirectories,
  sharedRootId,
  sharedRootOverlapsProject,
  upsertSharedRoot,
  writeSharedRoots,
} from "./shared-model-paths.mjs";
import {
  MAX_SHARED_ROOTS,
  normalizeSharedRoots,
  normalizeSharedRootPath,
  sharedKindSupported,
} from "../src/shared-model-refs.js";

async function fixture() {
  const base = await mkdtemp(path.join(os.tmpdir(), "xiria-shared-"));
  return {
    base,
    async cleanup() { await rm(base, { recursive: true, force: true }); },
  };
}

async function seed(root, files) {
  for (const [relative, size] of Object.entries(files)) {
    const target = path.join(root, relative);
    await mkdir(path.dirname(target), { recursive: true });
    await writeFile(target, Buffer.alloc(size));
  }
}

test("a shared reference keeps the relative shape and refuses every escape", () => {
  const id = "0123456789ab";
  assert.equal(formatSharedRef(id, "character/hero.safetensors"), "shared:0123456789ab/character/hero.safetensors");
  // Backslashes survive a round trip through a Windows-shaped listing.
  assert.equal(formatSharedRef(id, "character\\hero.safetensors"), "shared:0123456789ab/character/hero.safetensors");
  assert.deepEqual(parseSharedRef("shared:0123456789ab/a/b.safetensors"), { rootId: id, relativePath: "a/b.safetensors" });

  for (const invalid of [
    "character/hero.safetensors",           // a plain local value is not a shared ref
    "shared:0123456789ab/../../secrets.env",
    "shared:0123456789ab/./x.safetensors",
    "shared:0123456789ab/",
    "shared:0123456789ab",
    "shared:/x.safetensors",
    "shared:XYZ456789abc/x.safetensors",    // ids are lowercase hex only
    "shared:0123456789/x.safetensors",      // wrong length
  ]) {
    assert.equal(parseSharedRef(invalid), null, invalid);
  }
});

test("root ids follow the folder, not the spelling, and match the Python twin", () => {
  // Pinned against backend/test_shared_model_paths.py: both planes must derive
  // the same id or every mounted shared LoRA breaks at generation time. The
  // folding mode is passed explicitly so the vectors hold on either host.
  const windows = { caseInsensitive: true };
  assert.equal(sharedRootId("F:\\AI\\Ai SD\\sd-webui-aki-v4.10\\models\\Lora", windows), "95746e6dad38");
  assert.equal(sharedRootId("D:\\.XAIG\\XiriaCanvas AI\\models", windows), "5dff1a102833");

  // Separator flavour, repeats and a trailing slash are spelling on both
  // platforms, so they never change the id.
  assert.equal(normalizeSharedRootPath("F:\\AI\\Ai SD\\sd-webui-aki-v4.10\\models\\Lora"), "F:/AI/Ai SD/sd-webui-aki-v4.10/models/Lora");
  assert.equal(normalizeSharedRootPath("/srv/models/Lora//"), "/srv/models/Lora");
  for (const spelling of ["F:/AI/Ai SD/sd-webui-aki-v4.10/models/Lora", "f:\\ai\\ai sd\\sd-webui-aki-v4.10\\models\\lora\\", "F:\\AI\\Ai SD\\sd-webui-aki-v4.10\\models\\\\Lora"]) {
    assert.equal(sharedRootId(spelling, windows), "95746e6dad38", spelling);
  }
  assert.notEqual(sharedRootId("F:/AI/other/Lora", windows), sharedRootId("F:/AI/Ai SD/sd-webui-aki-v4.10/models/Lora", windows));
});

test("case only decides identity where the filesystem says it does", () => {
  // On Linux these are two real directories. Folding them would give both the
  // same id, and `normalizeSharedRoots` would then drop the second as a
  // duplicate of the first — the user's folder would silently not register.
  assert.notEqual(
    sharedRootId("/srv/models/Lora", { caseInsensitive: false }),
    sharedRootId("/srv/models/lora", { caseInsensitive: false }),
  );
  // On Windows they are one folder typed two ways, so the id must survive it.
  assert.equal(
    sharedRootId("F:\\Models\\Lora", { caseInsensitive: true }),
    sharedRootId("f:\\models\\lora", { caseInsensitive: true }),
  );
  assert.equal(SHARED_PATHS_CASE_INSENSITIVE, process.platform === "win32");
  assert.equal(sharedRootId("/srv/models/Lora"), sharedRootId("/srv/models/Lora", { caseInsensitive: SHARED_PATHS_CASE_INSENSITIVE }));
});

test("folder names from sd-webui, ComfyUI and this project all classify", () => {
  assert.equal(classifyDirectoryName("Stable-diffusion"), "checkpoints");   // sd-webui
  assert.equal(classifyDirectoryName("Lora"), "loras");                    // sd-webui
  assert.equal(classifyDirectoryName("ESRGAN"), "upscalers");              // sd-webui
  assert.equal(classifyDirectoryName("checkpoints"), "checkpoints");       // ComfyUI
  assert.equal(classifyDirectoryName("loras"), "loras");                   // ComfyUI
  assert.equal(classifyDirectoryName("upscale_models"), "upscalers");      // ComfyUI
  assert.equal(classifyDirectoryName("unet"), "diffusion_models");         // ComfyUI
  assert.equal(classifyDirectoryName("VAE"), "vae");
  assert.equal(classifyDirectoryName("hypernetworks"), "hypernetworks");
  assert.equal(classifyDirectoryName("random-folder"), "");
  assert.equal(classifyDirectoryName(""), "");

  // Detection is wider than support, and the difference has to stay visible.
  assert.equal(sharedKindSupported("loras"), true);
  assert.equal(sharedKindSupported("checkpoints"), true);
  assert.equal(sharedKindSupported("controlnet"), false);
  assert.equal(sharedKindSupported("vae"), false);
});

test("a models root is verified by scanning it, not by trusting folder names", async () => {
  const { base, cleanup } = await fixture();
  try {
    const shared = path.join(base, "ComfyUI", "models");
    await seed(shared, {
      "checkpoints/base.safetensors": 12,
      "loras/character/hero.safetensors": 8,
      "loras/style/ink.safetensors": 4,
      "controlnet/canny.safetensors": 6,
      "vae/empty-marker.txt": 3,   // named `vae` but holds nothing loadable
      "notes/readme.md": 2,
    });

    const report = await inspectSharedDirectory(base, shared, "auto");
    const kinds = Object.fromEntries(report.entries.map((entry) => [entry.kind, entry]));
    assert.equal(kinds.loras.files, 2);
    assert.equal(kinds.loras.bytes, 12);
    assert.equal(kinds.checkpoints.files, 1);
    assert.equal(kinds.controlnet.supported, false);
    // A folder whose name promises VAEs but contains no model file is not
    // reported as a VAE library — the claim has to be backed by a scan.
    assert.equal(kinds.vae, undefined);
    assert.equal(report.entries.some((entry) => entry.kind === "notes"), false);
    assert.match(report.warnings.join(" "), /ControlNet/);
    assert.equal(report.id, sharedRootId(report.path));
  } finally {
    await cleanup();
  }
});

test("a leaf LoRA folder registers as itself and reports the project overlap", async () => {
  const { base, cleanup } = await fixture();
  try {
    const leaf = path.join(base, "sd-webui", "models", "Lora");
    await seed(leaf, { "character/hero.safetensors": 5, "notes.txt": 1 });

    const report = await inspectSharedDirectory(base, leaf, "loras");
    assert.equal(report.entries.length, 1);
    assert.equal(report.entries[0].kind, "loras");
    assert.equal(report.entries[0].directory, ".");
    assert.equal(report.entries[0].files, 1);
    assert.equal(report.warnings.length, 0);

    // The user's own example points at this project's models folder. That is
    // allowed, but it double-lists every file, so it must be said out loud.
    const local = path.join(base, "models", "loras");
    await seed(local, { "a.safetensors": 2 });
    const overlap = await inspectSharedDirectory(base, local, "loras");
    assert.equal(sharedRootOverlapsProject(base, overlap.path), true);
    assert.match(overlap.warnings.join(" "), /本地模型目录/);

    // Pointing at a whole tool install must not declare the entire tree to be
    // checkpoints: an unrecognised folder name only counts files directly
    // inside it, so `sd-webui/` (whose models are two levels down) finds none.
    const ancestor = await inspectSharedDirectory(base, path.join(base, "sd-webui"), "auto");
    assert.deepEqual(ancestor.entries, []);
    assert.match(ancestor.warnings.join(" "), /没有找到可识别的模型文件/);

    // A custom folder name with model files sitting in it is still a library.
    const custom = path.join(base, "my-models");
    await seed(custom, { "base.safetensors": 7, "nested/deep.safetensors": 7 });
    const customReport = await inspectSharedDirectory(base, custom, "auto");
    assert.equal(customReport.entries.length, 1);
    assert.equal(customReport.entries[0].kind, "checkpoints");
    assert.equal(customReport.entries[0].files, 1);   // depth 0 only
  } finally {
    await cleanup();
  }
});

test("registration refuses anything that is not a real, specific folder", async () => {
  const { base, cleanup } = await fixture();
  try {
    await assert.rejects(inspectSharedDirectory(base, ""), /请输入目录路径/);
    await assert.rejects(inspectSharedDirectory(base, "models/loras"), /绝对路径/);
    await assert.rejects(inspectSharedDirectory(base, path.join(base, "nope")), /不存在/);
    await assert.rejects(inspectSharedDirectory(base, path.parse(base).root), /磁盘根目录/);

    await seed(base, { "a-file.safetensors": 1 });
    await assert.rejects(inspectSharedDirectory(base, path.join(base, "a-file.safetensors")), /必须是文件夹/);
  } finally {
    await cleanup();
  }
});

test("the config drops damaged entries, dedupes by id and stays capped", async () => {
  const { base, cleanup } = await fixture();
  try {
    await mkdir(path.join(base, "models"), { recursive: true });
    assert.deepEqual(await readSharedRoots(base), []);   // absent file is the default state

    const roots = normalizeSharedRoots({
      roots: [
        { id: "95746e6dad38", path: "F:/AI/Lora", label: "  sd-webui   Lora ", enabled: false, engines: ["SD"] },
        { id: "95746e6dad38", path: "F:/AI/Duplicate" },
        { id: "not-an-id", path: "F:/AI/Bad" },
        { id: "5dff1a102833", path: "" },
        { id: "5dff1a102833", path: "D:/models", engines: [] },
        "nonsense",
      ],
    });
    assert.equal(roots.length, 2);
    assert.equal(roots[0].label, "sd-webui Lora");
    assert.equal(roots[0].enabled, false);
    assert.deepEqual(roots[0].engines, ["SD"]);
    // An empty engine list would hide the folder the user just added, so it
    // reads as "not configured" rather than "attached to nothing".
    // An unset list is every engine, which now includes both Flux generations and Krea 2.
    assert.deepEqual(roots[1].engines, ["SD", "iL", "Anima", "Flux", "Flux2", "Krea2"]);
    assert.equal(roots[1].label, "models");

    await writeSharedRoots(base, roots);
    assert.deepEqual(await readSharedRoots(base), roots);

    let many = [];
    for (let index = 0; index < MAX_SHARED_ROOTS; index += 1) {
      many = upsertSharedRoot(many, { id: String(index).padStart(12, "0"), path: `F:/AI/${index}`, kind: "loras", label: `${index}`, enabled: true, engines: ["SD"] });
    }
    assert.equal(many.length, MAX_SHARED_ROOTS);
    assert.throws(() => upsertSharedRoot(many, { id: "ffffffffffff", path: "F:/AI/extra" }), /最多只能共享/);
    // Re-registering an existing folder updates it instead of filling a slot.
    assert.equal(upsertSharedRoot(many, { ...many[0], label: "renamed" }).length, MAX_SHARED_ROOTS);
  } finally {
    await cleanup();
  }
});

test("resolving a shared file is contained by the registered root", async () => {
  const { base, cleanup } = await fixture();
  try {
    const shared = path.join(base, "elsewhere", "Lora");
    await seed(shared, { "character/hero.safetensors": 4 });
    await seed(base, { "secret/keys.safetensors": 4 });

    const id = sharedRootId(await inspectSharedDirectory(base, shared, "loras").then((report) => report.path));
    const roots = [{ id, path: shared, kind: "loras", label: "Lora", enabled: true, engines: ["SD"] }];
    const extensions = new Set([".safetensors"]);

    const resolved = await resolveSharedFile(roots, `shared:${id}/character/hero.safetensors`, { extensions });
    assert.equal(path.basename(resolved), "hero.safetensors");

    await assert.rejects(resolveSharedFile(roots, `shared:${id}/../secret/keys.safetensors`, { extensions }), /格式无效/);
    await assert.rejects(resolveSharedFile(roots, `shared:ffffffffffff/character/hero.safetensors`, { extensions }), /未注册/);
    await assert.rejects(resolveSharedFile(roots, `shared:${id}/character/hero.exe`, { extensions }), /类型不受支持/);
    await assert.rejects(resolveSharedFile(roots, `shared:${id}/character/missing.safetensors`, { extensions }), /不存在/);
    await assert.rejects(resolveSharedFile([{ ...roots[0], enabled: false }], `shared:${id}/character/hero.safetensors`, { extensions }), /已停用/);
    await assert.rejects(resolveSharedFile(roots, `shared:${id}/character`, { extensions }), /类型不受支持/);

    // A symlink planted inside somebody else's model folder must not become a
    // way out of it: containment is re-checked against the resolved path.
    try {
      await symlink(path.join(base, "secret"), path.join(shared, "escape"), "junction");
      await assert.rejects(resolveSharedFile(roots, `shared:${id}/escape/keys.safetensors`, { extensions }), /不在已注册的共享目录内/);
    } catch (error) {
      if (error?.code !== "EPERM" && error?.code !== "ENOSYS") throw error;   // unprivileged Windows cannot link
    }
  } finally {
    await cleanup();
  }
});

test("kind directories are re-derived so later additions need no re-registering", async () => {
  const { base, cleanup } = await fixture();
  try {
    const shared = path.join(base, "ComfyUI", "models");
    await seed(shared, { "checkpoints/base.safetensors": 1 });
    const root = { id: sharedRootId(shared), path: shared, kind: "auto", label: "ComfyUI", enabled: true, engines: ["SD"] };

    assert.deepEqual(await sharedKindDirectories(root, "loras"), []);
    await seed(shared, { "loras/hero.safetensors": 1 });
    const loras = await sharedKindDirectories(root, "loras");
    assert.equal(loras.length, 1);
    assert.equal(loras[0].prefix, "loras");

    const leaf = { id: "0123456789ab", path: path.join(shared, "loras"), kind: "loras", label: "L", enabled: true, engines: ["SD"] };
    assert.equal((await sharedKindDirectories(leaf, "loras"))[0].prefix, "");
    assert.deepEqual(await sharedKindDirectories(leaf, "checkpoints"), []);
    assert.deepEqual(await sharedKindDirectories({ ...leaf, path: path.join(base, "gone") }, "loras"), []);
  } finally {
    await cleanup();
  }
});
