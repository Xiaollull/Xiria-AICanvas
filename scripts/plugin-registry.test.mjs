import assert from "node:assert/strict";
import { renameSync, rmSync, writeFileSync } from "node:fs";
import { mkdir, mkdtemp, readdir, readFile, rm, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import {
  PLUGIN_DIAGNOSTIC_CODES,
  PLUGIN_EXECUTION_SUPPORT,
  PLUGIN_HOST_API_VERSION,
  PLUGIN_MANIFEST_MAXIMUM_BYTES,
  PLUGIN_MANIFEST_SCHEMA_VERSION,
  PLUGIN_REGISTRY_SCHEMA_VERSION,
  createPluginRegistry,
  decodePluginManifest,
  discoverPlugins,
  displayFolderName,
  hostApiCompatible,
  parsePluginManifest,
  pluginsRootFor,
  servesPluginContent,
  validEntrypointPath,
  validPluginId,
  validPluginVersion,
} from "./plugin-registry.mjs";

const scriptsDirectory = path.dirname(fileURLToPath(import.meta.url));
const projectDirectory = path.dirname(scriptsDirectory);

function baseManifest(overrides = {}) {
  return { schemaVersion: 1, id: "sample-plugin", version: "1.0.0", hostApi: { min: 1, max: 1 }, permissions: [], ...overrides };
}

async function temporaryProject(prefix) {
  const root = await mkdtemp(path.join(os.tmpdir(), prefix));
  return {
    root,
    async plugin(name, manifest, extraFiles = {}) {
      const folder = path.join(root, "plugins", name);
      await mkdir(folder, { recursive: true });
      if (manifest !== null) {
        const body = typeof manifest === "string" ? manifest : JSON.stringify(manifest, null, 2);
        await writeFile(path.join(folder, "plugin.json"), body, "utf8");
      }
      for (const [relativePath, contents] of Object.entries(extraFiles)) {
        const target = path.join(folder, relativePath);
        await mkdir(path.dirname(target), { recursive: true });
        await writeFile(target, contents, "utf8");
      }
      return folder;
    },
    async dispose() {
      await rm(root, { recursive: true, force: true });
    },
  };
}

function codesFor(snapshot, id) {
  return snapshot.plugins.find((entry) => entry.id === id)?.diagnostics || null;
}

async function manifestCode(manifest, folderName = "sample-plugin") {
  const project = await temporaryProject("xirai-plugin-manifest-");
  try {
    await project.plugin(folderName, manifest);
    const snapshot = await discoverPlugins({ projectRoot: project.root });
    assert.equal(snapshot.plugins.length, 1);
    return { state: snapshot.plugins[0].state, code: snapshot.plugins[0].diagnostics[0] || null, entry: snapshot.plugins[0] };
  } finally {
    await project.dispose();
  }
}

test("a missing plugin root is an empty registry and is never created", async () => {
  const project = await temporaryProject("xirai-plugin-missing-");
  try {
    const snapshot = await discoverPlugins({ projectRoot: project.root });
    assert.deepEqual(snapshot, {
      schemaVersion: PLUGIN_REGISTRY_SCHEMA_VERSION,
      hostApiVersion: PLUGIN_HOST_API_VERSION,
      execution: PLUGIN_EXECUTION_SUPPORT,
      plugins: [],
      diagnostics: [],
    });
    assert.deepEqual(await readdir(project.root), [], "discovery must not create the plugin root");
  } finally {
    await project.dispose();
  }
});

test("an empty plugin root and a root holding only files stay empty registries", async () => {
  const project = await temporaryProject("xirai-plugin-empty-");
  try {
    const pluginsRoot = pluginsRootFor(project.root);
    await mkdir(pluginsRoot, { recursive: true });
    assert.deepEqual((await discoverPlugins({ projectRoot: project.root })).plugins, []);

    await writeFile(path.join(pluginsRoot, "README.md"), "# placeholder\n", "utf8");
    await writeFile(path.join(pluginsRoot, "plugin.json"), "{}", "utf8");
    const snapshot = await discoverPlugins({ projectRoot: project.root });
    assert.deepEqual(snapshot.plugins, []);
    assert.deepEqual(snapshot.diagnostics, []);
    assert.deepEqual((await readdir(pluginsRoot)).sort(), ["README.md", "plugin.json"]);
  } finally {
    await project.dispose();
  }
});

test("a minimal valid manifest is discovered with the exact registry DTO", async () => {
  const project = await temporaryProject("xirai-plugin-minimal-");
  try {
    await project.plugin("sample-plugin", baseManifest());
    const snapshot = await discoverPlugins({ projectRoot: project.root });
    assert.deepEqual(snapshot, {
      schemaVersion: 1,
      hostApiVersion: 1,
      execution: "not-supported",
      plugins: [{
        id: "sample-plugin",
        name: "sample-plugin",
        description: "",
        version: "1.0.0",
        enabled: false,
        state: "discovered",
        compatible: true,
        execution: "not-supported",
        contributions: { panels: [], commands: [] },
        diagnostics: [],
      }],
      diagnostics: [],
    });
  } finally {
    await project.dispose();
  }
});

test("optional display fields and contributions are reported as validated metadata", async () => {
  const project = await temporaryProject("xirai-plugin-display-");
  try {
    await project.plugin("sample-plugin", baseManifest({
      name: "Sample Plugin",
      description: "A discovery-only sample.",
      developer: { name: "XiriaCanvas", homepage: "https://example.com/plugins" },
      entrypoints: { frontend: "ui/index.js", backend: "server/main.py" },
      contributes: { panels: [{ id: "sample", title: "Sample panel" }], commands: [{ id: "run", title: "Run" }] },
    }));
    const snapshot = await discoverPlugins({ projectRoot: project.root });
    assert.equal(snapshot.plugins[0].name, "Sample Plugin");
    assert.equal(snapshot.plugins[0].description, "A discovery-only sample.");
    assert.deepEqual(snapshot.plugins[0].contributions, {
      panels: [{ id: "sample", title: "Sample panel" }],
      commands: [{ id: "run", title: "Run" }],
    });
    // Entrypoints and developer details stay host-side: the DTO exposes neither.
    assert.deepEqual(Object.keys(snapshot.plugins[0]).sort(), [
      "compatible", "contributions", "description", "diagnostics", "enabled", "execution", "id", "name", "state", "version",
    ]);
  } finally {
    await project.dispose();
  }
});

test("registry snapshots are deeply frozen and sorted by identifier", async () => {
  const project = await temporaryProject("xirai-plugin-frozen-");
  try {
    await project.plugin("zulu-plugin", baseManifest({ id: "zulu-plugin" }));
    await project.plugin("alpha-plugin", baseManifest({ id: "alpha-plugin" }));
    const snapshot = await discoverPlugins({ projectRoot: project.root });
    assert.deepEqual(snapshot.plugins.map((entry) => entry.id), ["alpha-plugin", "zulu-plugin"]);
    assert.equal(Object.isFrozen(snapshot), true);
    assert.equal(Object.isFrozen(snapshot.plugins), true);
    assert.equal(Object.isFrozen(snapshot.plugins[0]), true);
    assert.equal(Object.isFrozen(snapshot.plugins[0].contributions.panels), true);
    assert.throws(() => { "use strict"; snapshot.plugins[0].enabled = true; }, TypeError);
  } finally {
    await project.dispose();
  }
});

test("identifier syntax follows lowercase ASCII kebab-case with Windows reserved names refused", () => {
  for (const value of ["abc", "sample-plugin", "a1-b2-c3", "x".repeat(64)]) {
    assert.equal(validPluginId(value), true, value);
  }
  for (const value of [
    "ab", "x".repeat(65), "Sample", "sample_plugin", "sample.plugin", "sample plugin", "-sample", "sample-",
    "sample--plugin", "sample/plugin", "sample\\plugin", "sämple", "nul", "com1", "lpt9", "aux", "prn", "con",
    "", null, 5,
  ]) {
    assert.equal(validPluginId(value), false, String(value));
  }
});

test("a folder name that is not a legal identifier is reported without echoing an unbounded name", async () => {
  const project = await temporaryProject("xirai-plugin-badfolder-");
  try {
    await project.plugin("Sample", baseManifest());
    await project.plugin("ab", baseManifest());
    const snapshot = await discoverPlugins({ projectRoot: project.root });
    assert.deepEqual(snapshot.plugins.map((entry) => [entry.id, entry.state, entry.diagnostics[0]]).sort(), [
      ["Sample", "invalid", "invalid_plugin_id"],
      ["ab", "invalid", "invalid_plugin_id"],
    ]);
  } finally {
    await project.dispose();
  }
  // Names outside the bounded ASCII display shape collapse to a stable digest label instead.
  assert.equal(displayFolderName("plugin"), "plugin");
  assert.equal(displayFolderName("插件"), displayFolderName("插件"));
  assert.match(displayFolderName("插件"), /^unsupported-[0-9a-f]{12}$/);
  assert.notEqual(displayFolderName("插件"), displayFolderName("挿件"));
  assert.match(displayFolderName("a\u0000b"), /^unsupported-[0-9a-f]{12}$/);
});

test("manifest identifier must equal the folder name", async () => {
  const mismatch = await manifestCode(baseManifest({ id: "other-plugin" }), "sample-plugin");
  assert.equal(mismatch.state, "invalid");
  assert.equal(mismatch.code, "id_folder_mismatch");
  assert.equal(mismatch.entry.id, "sample-plugin", "the folder name, not the claimed identifier, keys the entry");
  assert.equal(mismatch.entry.version, null);
});

test("manifest validation rejects every structural violation with a stable code", async () => {
  const cases = [
    [baseManifest({ schemaVersion: 2 }), "unsupported_schema_version"],
    [baseManifest({ schemaVersion: "1" }), "invalid_manifest"],
    [baseManifest({ schemaVersion: 1.5 }), "invalid_manifest"],
    [{ ...baseManifest(), extra: true }, "invalid_manifest"],
    [baseManifest({ version: "1.0" }), "invalid_version"],
    [baseManifest({ version: "v1.0.0" }), "invalid_version"],
    [baseManifest({ version: "1.0.0-beta.1" }), "invalid_version"],
    [baseManifest({ version: "01.0.0" }), "invalid_version"],
    [baseManifest({ hostApi: { min: 1 } }), "invalid_manifest"],
    [baseManifest({ hostApi: { min: 1, max: 1, exact: 1 } }), "invalid_manifest"],
    [baseManifest({ hostApi: { min: 2, max: 1 } }), "invalid_manifest"],
    [baseManifest({ hostApi: { min: 0, max: 1 } }), "invalid_manifest"],
    [baseManifest({ hostApi: "1" }), "invalid_manifest"],
    [baseManifest({ permissions: "none" }), "invalid_manifest"],
    [baseManifest({ permissions: [1] }), "invalid_manifest"],
    [baseManifest({ permissions: ["Filesystem"] }), "invalid_manifest"],
    [baseManifest({ name: "" }), "invalid_manifest"],
    [baseManifest({ name: "x".repeat(65) }), "invalid_manifest"],
    [baseManifest({ name: "line\nbreak" }), "invalid_manifest"],
    [baseManifest({ description: "x".repeat(281) }), "invalid_manifest"],
    [baseManifest({ developer: { name: "x", homepage: "http://example.com" } }), "invalid_manifest"],
    [baseManifest({ developer: { name: "x", homepage: "javascript:alert(1)" } }), "invalid_manifest"],
    [baseManifest({ developer: { name: "x", homepage: "https://user:pass@example.com" } }), "invalid_manifest"],
    [baseManifest({ developer: { name: "x", email: "a@b.c" } }), "invalid_manifest"],
    [baseManifest({ contributes: { panels: [{ id: "a", title: "A" }, { id: "a", title: "B" }] } }), "invalid_manifest"],
    [baseManifest({ contributes: { panels: [{ id: "a", title: "A", command: "x" }] } }), "invalid_manifest"],
    [baseManifest({ contributes: { routes: [] } }), "invalid_manifest"],
    [baseManifest({ entrypoints: { worker: "w.js" } }), "invalid_manifest"],
  ];
  for (const [manifest, expected] of cases) {
    const result = await manifestCode(manifest);
    assert.equal(result.code, expected, JSON.stringify(manifest));
    assert.equal(result.state, "invalid");
  }

  // Required keys must be present.
  for (const key of ["schemaVersion", "id", "version", "hostApi", "permissions"]) {
    const manifest = baseManifest();
    delete manifest[key];
    assert.equal((await manifestCode(manifest)).code, "invalid_manifest", key);
  }

  // `JSON.parse` materialises `__proto__` as an own key, so the allow-list has to reject it. An
  // object literal cannot stage this case because `__proto__:` is a setter there, not a property.
  const polluted = '{"schemaVersion":1,"id":"sample-plugin","version":"1.0.0","hostApi":{"min":1,"max":1},'
    + '"permissions":[],"__proto__":{"polluted":true}}';
  assert.equal((await manifestCode(polluted)).code, "invalid_manifest");
  assert.equal({}.polluted, undefined, "manifest parsing must not reach Object.prototype");
  assert.equal((await manifestCode('{"constructor":{"prototype":{}}}')).code, "invalid_manifest");
});

test("entrypoint paths are validated metadata and never opened", async () => {
  for (const value of ["a.js", "ui/index.js", "server/main.py", "a/b/c/d.mjs"]) {
    assert.equal(validEntrypointPath(value), true, value);
  }
  for (const value of [
    "/abs.js", "C:/x.js", "c:x.js", "..\\x.js", "../x.js", "a/../b.js", "a/./b.js", "a//b.js",
    "a\\b.js", "//server/share/x.js", "x.js:stream", "a\u0000.js", "a/b.js ", "a/b.", "nul.js", "a/COM1.js",
    "x".repeat(201), "", ".", "..", null,
  ]) {
    assert.equal(validEntrypointPath(value), false, String(value));
  }

  const project = await temporaryProject("xirai-plugin-entrypoint-");
  try {
    // A declared entrypoint that does not exist on disk must not affect discovery: the host never
    // stats or opens it.
    await project.plugin("sample-plugin", baseManifest({ entrypoints: { frontend: "does/not/exist.js" } }));
    assert.equal((await discoverPlugins({ projectRoot: project.root })).plugins[0].state, "discovered");
    await project.plugin("bad-entrypoint", baseManifest({ id: "bad-entrypoint", entrypoints: { frontend: "../escape.js" } }));
    assert.deepEqual(codesFor(await discoverPlugins({ projectRoot: project.root }), "bad-entrypoint"), ["invalid_entrypoint"]);
  } finally {
    await project.dispose();
  }
});

test("manifest bytes must be bounded, BOM-free UTF-8 holding a JSON object", async () => {
  const project = await temporaryProject("xirai-plugin-bytes-");
  try {
    await project.plugin("no-manifest", null);
    await project.plugin("bad-json", "{ not json");
    await project.plugin("array-root", "[]");
    await project.plugin("null-root", "null");
    await project.plugin("bom-manifest", `\uFEFF${JSON.stringify(baseManifest({ id: "bom-manifest" }))}`);
    const oversize = await project.plugin("oversize", baseManifest({ id: "oversize" }));
    await writeFile(path.join(oversize, "plugin.json"), "x".repeat(PLUGIN_MANIFEST_MAXIMUM_BYTES + 1), "utf8");
    const invalidUtf8 = await project.plugin("bad-utf8", baseManifest({ id: "bad-utf8" }));
    await writeFile(path.join(invalidUtf8, "plugin.json"), Buffer.from([0x7b, 0xff, 0xfe, 0x7d]));

    const snapshot = await discoverPlugins({ projectRoot: project.root });
    assert.deepEqual(codesFor(snapshot, "no-manifest"), ["manifest_missing"]);
    assert.deepEqual(codesFor(snapshot, "bad-json"), ["manifest_not_json"]);
    assert.deepEqual(codesFor(snapshot, "array-root"), ["manifest_not_json"]);
    assert.deepEqual(codesFor(snapshot, "null-root"), ["manifest_not_json"]);
    assert.deepEqual(codesFor(snapshot, "bom-manifest"), ["manifest_not_utf8"]);
    assert.deepEqual(codesFor(snapshot, "bad-utf8"), ["manifest_not_utf8"]);
    assert.deepEqual(codesFor(snapshot, "oversize"), ["manifest_too_large"]);
  } finally {
    await project.dispose();
  }
  // The pure decoder enforces the same limit without touching the filesystem.
  assert.throws(() => decodePluginManifest(Buffer.alloc(PLUGIN_MANIFEST_MAXIMUM_BYTES + 1), "sample-plugin"), /manifest_too_large/);
});

test("a manifest file that is a directory is refused before any read", async () => {
  const project = await temporaryProject("xirai-plugin-typechange-");
  try {
    await mkdir(path.join(project.root, "plugins", "sample-plugin", "plugin.json"), { recursive: true });
    assert.deepEqual(codesFor(await discoverPlugins({ projectRoot: project.root }), "sample-plugin"), ["unsafe_reparse_point"]);
  } finally {
    await project.dispose();
  }
});

test("host API compatibility gates on the declared range", async () => {
  assert.equal(hostApiCompatible({ min: 1, max: 1 }), true);
  assert.equal(hostApiCompatible({ min: 1, max: 5 }), true);
  assert.equal(hostApiCompatible({ min: 2, max: 5 }), false);
  assert.equal(hostApiCompatible({ min: 2, max: 5 }, 3), true);
  assert.equal(hostApiCompatible({ min: 1, max: 1001 }), false);
  assert.equal(hostApiCompatible(null), false);

  const incompatible = await manifestCode(baseManifest({ hostApi: { min: 2, max: 9 } }));
  assert.equal(incompatible.state, "incompatible");
  assert.equal(incompatible.code, "host_api_incompatible");
  assert.equal(incompatible.entry.compatible, false);
});

test("any declared permission blocks the plugin because permissions are not a security mechanism", async () => {
  const blocked = await manifestCode(baseManifest({ permissions: ["filesystem"] }));
  assert.equal(blocked.state, "blocked");
  assert.equal(blocked.code, "permissions_not_supported");
  assert.equal(blocked.entry.compatible, true);
  assert.equal(blocked.entry.enabled, false);

  // An incompatible host API is reported ahead of an unsupported permission so a manifest can never
  // present two different states on two hosts.
  const both = await manifestCode(baseManifest({ hostApi: { min: 9, max: 9 }, permissions: ["filesystem"] }));
  assert.equal(both.state, "incompatible");
  assert.deepEqual(both.entry.diagnostics, ["host_api_incompatible"]);
});

test("an unsafe plugin root fails the whole registry closed without throwing", async () => {
  const project = await temporaryProject("xirai-plugin-rootfile-");
  try {
    await writeFile(pluginsRootFor(project.root), "not a directory", "utf8");
    const snapshot = await discoverPlugins({ projectRoot: project.root });
    assert.deepEqual(snapshot.plugins, []);
    assert.deepEqual(snapshot.diagnostics, [{ id: null, code: "plugins_root_unsafe" }]);
  } finally {
    await project.dispose();
  }

  const missingProject = path.join(os.tmpdir(), `xirai-plugin-absent-${process.pid}-${Date.now()}`);
  const snapshot = await discoverPlugins({ projectRoot: missingProject });
  assert.deepEqual(snapshot.diagnostics, [{ id: null, code: "plugins_root_unavailable" }]);
  assert.deepEqual(snapshot.plugins, []);
});

test("an injected reparse-point probe rejects the root, a folder, and a manifest decisively", async () => {
  const project = await temporaryProject("xirai-plugin-reparse-");
  try {
    await project.plugin("sample-plugin", baseManifest());
    const pluginsRoot = pluginsRootFor(project.root);

    const rootBlocked = await discoverPlugins({
      projectRoot: project.root,
      probeReparsePoint: (absolutePath) => path.basename(absolutePath) === "plugins",
    });
    assert.deepEqual(rootBlocked.diagnostics, [{ id: null, code: "plugins_root_unsafe" }]);
    assert.deepEqual(rootBlocked.plugins, []);

    const folderBlocked = await discoverPlugins({
      projectRoot: project.root,
      probeReparsePoint: (absolutePath) => absolutePath !== pluginsRoot && path.basename(absolutePath) === "sample-plugin",
    });
    assert.deepEqual(codesFor(folderBlocked, "sample-plugin"), ["unsafe_reparse_point"]);

    const manifestBlocked = await discoverPlugins({
      projectRoot: project.root,
      probeReparsePoint: (absolutePath) => path.basename(absolutePath) === "plugin.json",
    });
    assert.deepEqual(codesFor(manifestBlocked, "sample-plugin"), ["unsafe_reparse_point"]);
  } finally {
    await project.dispose();
  }
});

test("a symlinked plugin folder, manifest, and plugin root are all refused", async (context) => {
  const project = await temporaryProject("xirai-plugin-symlink-");
  try {
    const real = await project.plugin("sample-plugin", baseManifest());
    const pluginsRoot = pluginsRootFor(project.root);
    // Windows creates junctions without elevation; POSIX uses an ordinary directory symlink.
    const directoryLinkType = process.platform === "win32" ? "junction" : "dir";
    try {
      await symlink(real, path.join(pluginsRoot, "linked-plugin"), directoryLinkType);
    } catch (error) {
      context.skip(`this environment cannot create directory links: ${error.code}`);
      return;
    }
    const snapshot = await discoverPlugins({ projectRoot: project.root });
    assert.deepEqual(codesFor(snapshot, "linked-plugin"), ["unsafe_reparse_point"]);
    assert.deepEqual(codesFor(snapshot, "sample-plugin"), [], "a linked sibling never invalidates a real plugin");

    const outside = await temporaryProject("xirai-plugin-symlink-target-");
    try {
      const linkedRootProject = await temporaryProject("xirai-plugin-symlink-root-");
      try {
        await mkdir(pluginsRootFor(outside.root), { recursive: true });
        await symlink(pluginsRootFor(outside.root), pluginsRootFor(linkedRootProject.root), directoryLinkType);
        const linkedRoot = await discoverPlugins({ projectRoot: linkedRootProject.root });
        assert.deepEqual(linkedRoot.diagnostics, [{ id: null, code: "plugins_root_unsafe" }]);
      } finally {
        await linkedRootProject.dispose();
      }
    } finally {
      await outside.dispose();
    }

    const manifestProject = await temporaryProject("xirai-plugin-symlink-manifest-");
    try {
      const folder = await manifestProject.plugin("linked-manifest", null);
      const target = path.join(manifestProject.root, "outside.json");
      await writeFile(target, JSON.stringify(baseManifest({ id: "linked-manifest" })), "utf8");
      try {
        await symlink(target, path.join(folder, "plugin.json"), "file");
      } catch (error) {
        context.diagnostic(`file symlink unavailable: ${error.code}`);
        return;
      }
      assert.deepEqual(
        codesFor(await discoverPlugins({ projectRoot: manifestProject.root }), "linked-manifest"),
        ["unsafe_reparse_point"],
      );
    } finally {
      await manifestProject.dispose();
    }
  } finally {
    await project.dispose();
  }
});

test("a manifest replaced or deleted between validation and read is refused", async () => {
  const project = await temporaryProject("xirai-plugin-toctou-");
  try {
    await project.plugin("sample-plugin", baseManifest());
    const manifestPath = path.join(pluginsRootFor(project.root), "sample-plugin", "plugin.json");
    const replacement = path.join(project.root, "replacement.json");

    // The probe runs after the manifest has been validated and before it is opened, which is the
    // exact window an attacker would use. Renaming over the path also changes the inode.
    let swapped = false;
    const swapProbe = (absolutePath) => {
      if (path.basename(absolutePath) === "plugin.json" && !swapped) {
        swapped = true;
        writeFileSync(replacement, `${JSON.stringify(baseManifest({ id: "sample-plugin", name: "swapped" }))}\n\n\n`, "utf8");
        renameSync(replacement, absolutePath);
      }
      return false;
    };
    assert.deepEqual(
      codesFor(await discoverPlugins({ projectRoot: project.root, probeReparsePoint: swapProbe }), "sample-plugin"),
      ["manifest_changed_during_read"],
    );

    let removed = false;
    const removeProbe = (absolutePath) => {
      if (path.basename(absolutePath) === "plugin.json" && !removed) {
        removed = true;
        rmSync(absolutePath, { force: true });
      }
      return false;
    };
    assert.deepEqual(
      codesFor(await discoverPlugins({ projectRoot: project.root, probeReparsePoint: removeProbe }), "sample-plugin"),
      ["manifest_missing"],
    );
  } finally {
    await project.dispose();
  }
});

test("one invalid plugin is isolated from its healthy neighbours", async () => {
  const project = await temporaryProject("xirai-plugin-isolation-");
  try {
    await project.plugin("good-plugin", baseManifest({ id: "good-plugin" }));
    await project.plugin("broken-plugin", "{{{");
    await project.plugin("blocked-plugin", baseManifest({ id: "blocked-plugin", permissions: ["gpu"] }));
    const snapshot = await discoverPlugins({ projectRoot: project.root });
    assert.deepEqual(snapshot.plugins.map((entry) => [entry.id, entry.state]), [
      ["blocked-plugin", "blocked"],
      ["broken-plugin", "invalid"],
      ["good-plugin", "discovered"],
    ]);
    assert.deepEqual(snapshot.diagnostics, [
      { id: "blocked-plugin", code: "permissions_not_supported" },
      { id: "broken-plugin", code: "manifest_not_json" },
    ]);
  } finally {
    await project.dispose();
  }
});

test("case-insensitive folder collisions invalidate every colliding candidate", async (context) => {
  const project = await temporaryProject("xirai-plugin-duplicate-");
  try {
    await project.plugin("sample-plugin", baseManifest());
    const upper = path.join(pluginsRootFor(project.root), "Sample-Plugin");
    await mkdir(upper, { recursive: true });
    const entries = await readdir(pluginsRootFor(project.root));
    if (entries.length !== 2) {
      context.skip("this filesystem folds folder name case, so a collision cannot be staged");
      return;
    }
    await writeFile(path.join(upper, "plugin.json"), JSON.stringify(baseManifest()), "utf8");
    const snapshot = await discoverPlugins({ projectRoot: project.root });
    assert.deepEqual(snapshot.plugins.map((entry) => [entry.id, entry.diagnostics[0]]), [
      ["Sample-Plugin", "duplicate_id"],
      ["sample-plugin", "duplicate_id"],
    ]);
  } finally {
    await project.dispose();
  }
});

test("discovery reads direct children only and never recurses into a plugin folder", async () => {
  const project = await temporaryProject("xirai-plugin-recursion-");
  try {
    await project.plugin("outer-plugin", baseManifest({ id: "outer-plugin" }), {
      "nested/plugin.json": JSON.stringify(baseManifest({ id: "nested" })),
      "nested/deep/plugin.json": JSON.stringify(baseManifest({ id: "deep" })),
    });
    const snapshot = await discoverPlugins({ projectRoot: project.root });
    assert.deepEqual(snapshot.plugins.map((entry) => entry.id), ["outer-plugin"]);
  } finally {
    await project.dispose();
  }
});

test("plugin program code stays inert: nothing is imported, executed, or opened", async () => {
  const project = await temporaryProject("xirai-plugin-inert-");
  try {
    await project.plugin("sample-plugin", baseManifest({ entrypoints: { frontend: "index.js", backend: "main.py" } }), {
      "index.js": "process.exit(97);\n",
      "index.mjs": "throw new Error('plugin module must never be imported');\n",
      "main.py": "import os\nos._exit(97)\n",
      "package.json": JSON.stringify({ name: "hostile", scripts: { postinstall: "exit 97" } }),
    });
    const snapshot = await discoverPlugins({ projectRoot: project.root });
    assert.equal(snapshot.plugins[0].state, "discovered");
    assert.equal(snapshot.plugins[0].execution, "not-supported");
    assert.equal(snapshot.execution, "not-supported");
  } finally {
    await project.dispose();
  }

  // Source contract: the registry has no mechanism capable of running plugin code at all.
  const source = await readFile(path.join(scriptsDirectory, "plugin-registry.mjs"), "utf8");
  for (const forbidden of [
    "import(", "require(", "createRequire", "child_process", "node:vm", "new Function", "eval(",
    "spawn", "execFile", "readFileSync",
  ]) {
    assert.equal(source.includes(forbidden), false, `plugin-registry.mjs must not contain ${forbidden}`);
  }
  // The only filesystem verbs the registry may use.
  assert.match(source, /import \{ lstat, open, readdir, realpath \} from "node:fs\/promises";/);
});

test("the plugin root is fixed at <projectRoot>/plugins and reads no environment variable", async () => {
  assert.equal(pluginsRootFor("/anywhere"), path.join("/anywhere", "plugins"));
  const source = await readFile(path.join(scriptsDirectory, "plugin-registry.mjs"), "utf8");
  assert.equal(source.includes("process.env"), false, "the plugin root must not be configurable");
});

test("the registry is lazy, single-flight, and refreshes after a scan settles", async () => {
  const project = await temporaryProject("xirai-plugin-registry-");
  try {
    let scans = 0;
    const registry = createPluginRegistry({
      projectRoot: project.root,
      probeReparsePoint: (absolutePath, stats) => {
        if (path.basename(absolutePath) === "plugins") scans += 1;
        return stats.isSymbolicLink();
      },
    });
    await project.plugin("first-plugin", baseManifest({ id: "first-plugin" }));
    assert.equal(scans, 0, "nothing is scanned before the first read");

    const [left, right] = await Promise.all([registry.read(), registry.read()]);
    assert.equal(left, right, "concurrent reads share one scan");
    assert.equal(scans, 1);

    await project.plugin("second-plugin", baseManifest({ id: "second-plugin" }));
    const refreshed = await registry.read();
    assert.deepEqual(refreshed.plugins.map((entry) => entry.id), ["first-plugin", "second-plugin"]);
    assert.equal(scans, 2);
  } finally {
    await project.dispose();
  }
});

test("plugin content is never servable over HTTP, including traversal and encoded variants", () => {
  const root = process.platform === "win32" ? "D:\\project" : "/project";
  const blocked = [
    "/plugins",
    "/plugins/",
    "/plugins/sample-plugin/index.js",
    "/plugins/README.md",
    "//plugins/sample-plugin/index.js",
    "/plugins/../plugins/sample-plugin/index.js",
    "/assets/../plugins/sample-plugin/index.js",
    "/%70lugins/sample-plugin/index.js",
    "/plugins%2Fsample-plugin/index.js",
    "/plugins/sample%2Dplugin/index.js",
    "/assets/..%2Fplugins/index.js",
    "/plugins/sample-plugin/index.js?import&t=1",
    "/plugins/sample-plugin/index.js#fragment",
    "/plugins\\sample-plugin\\index.js",
    "http://localhost:7709/plugins/sample-plugin/index.js",
    "/%ff",
    "/plugins/sample-plugin/ .js",
    process.platform === "win32" ? "/@fs/D:/project/plugins/sample-plugin/index.js" : "/@fs/project/plugins/sample-plugin/index.js",
    process.platform === "win32" ? "/@fs/d:/PROJECT/Plugins/x.js" : "/@fs/project/plugins",
  ];
  for (const value of blocked) assert.equal(servesPluginContent(value, root), true, value);

  const allowed = [
    "/", "/index.html", "/api/plugins", "/api/plugins?refresh=1", "/src/App.jsx", "/pluginsomething/x.js",
    "/assets/plugins.js", "/@fs/elsewhere/x.js", "/@vite/client",
  ];
  for (const value of allowed) assert.equal(servesPluginContent(value, root), false, value);
});

test("published constants and diagnostic codes stay stable", () => {
  assert.equal(PLUGIN_HOST_API_VERSION, 1);
  assert.equal(PLUGIN_MANIFEST_SCHEMA_VERSION, 1);
  assert.equal(PLUGIN_REGISTRY_SCHEMA_VERSION, 1);
  assert.equal(PLUGIN_EXECUTION_SUPPORT, "not-supported");
  assert.equal(PLUGIN_MANIFEST_MAXIMUM_BYTES, 65536);
  assert.equal(new Set(PLUGIN_DIAGNOSTIC_CODES).size, PLUGIN_DIAGNOSTIC_CODES.length);
  assert.equal(validPluginVersion("0.0.0"), true);
  assert.equal(validPluginVersion("1.0.0+build"), false);
  assert.deepEqual(parsePluginManifest(baseManifest(), "sample-plugin").contributes, { panels: [], commands: [] });
});

test("the shipped plugin root holds only the documented placeholder", async () => {
  const entries = await readdir(pluginsRootFor(projectDirectory), { withFileTypes: true });
  assert.deepEqual(entries.filter((entry) => entry.isDirectory()).map((entry) => entry.name), [],
    "no example plugin may ship: the default registry state is empty");
  const snapshot = await discoverPlugins({ projectRoot: projectDirectory });
  assert.deepEqual(snapshot.plugins, []);
  assert.deepEqual(snapshot.diagnostics, []);
});
