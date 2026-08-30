import assert from "node:assert/strict";
import { access, mkdir, mkdtemp, readFile, readdir, realpath, rm, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { removePluginFolder, revealFolderCommand, revealPluginFolder } from "./plugin-actions.mjs";
import { assertSafePluginFolder, pluginsRootFor } from "./plugin-registry.mjs";
import { pluginRemoveConfirmation } from "../src/plugin-presentation.js";

const scriptsDirectory = path.dirname(fileURLToPath(import.meta.url));
const projectDirectory = path.dirname(scriptsDirectory);

function baseManifest(overrides = {}) {
  return { schemaVersion: 1, id: "sample-plugin", version: "1.0.0", hostApi: { min: 1, max: 1 }, permissions: [], ...overrides };
}

async function temporaryProject(prefix) {
  // Resolved so the reveal path this test asserts on is the same spelling the path-safety contract
  // produces. `os.tmpdir()` is the 8.3 short form on a Windows CI runner, and the symlink cases
  // below need the escape to be the only thing `realpath` changes.
  const root = await realpath(await mkdtemp(path.join(os.tmpdir(), prefix)));
  return {
    root,
    async plugin(name, manifest, extraFiles = {}) {
      const folder = path.join(root, "plugins", name);
      await mkdir(folder, { recursive: true });
      if (manifest !== null) await writeFile(path.join(folder, "plugin.json"), JSON.stringify(manifest, null, 2), "utf8");
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

function recordingSpawn() {
  const calls = [];
  const spawnProcess = (command, args, options) => {
    calls.push({ command, args, options });
    return { on() {}, unref() {} };
  };
  return { calls, spawnProcess };
}

test("the reveal command opens a file manager per platform without a shell", () => {
  assert.deepEqual(revealFolderCommand("win32", "D:\\project\\plugins\\sample-plugin"), {
    command: "explorer.exe", args: ["D:\\project\\plugins\\sample-plugin"],
  });
  assert.deepEqual(revealFolderCommand("darwin", "/project/plugins/sample-plugin"), {
    command: "open", args: ["/project/plugins/sample-plugin"],
  });
  assert.deepEqual(revealFolderCommand("linux", "/project/plugins/sample-plugin"), {
    command: "xdg-open", args: ["/project/plugins/sample-plugin"],
  });

  // The path is always a separate argument, so a folder name can never become shell syntax.
  for (const platform of ["win32", "darwin", "linux"]) {
    const resolved = revealFolderCommand(platform, "/a b/c&d");
    assert.equal(resolved.args.length, 1);
    assert.equal(resolved.args[0], "/a b/c&d");
  }
});

test("reveal resolves the folder from the id and never from client input", async () => {
  const project = await temporaryProject("xirai-plugin-reveal-");
  try {
    const folder = await project.plugin("sample-plugin", baseManifest());
    const { calls, spawnProcess } = recordingSpawn();
    assert.deepEqual(await revealPluginFolder({ projectRoot: project.root, id: "sample-plugin", platform: "linux", spawnProcess }), {
      revealed: true,
    });
    assert.equal(calls.length, 1);
    assert.equal(calls[0].command, "xdg-open");
    assert.deepEqual(calls[0].args, [folder]);
    assert.deepEqual(calls[0].options, { detached: true, stdio: "ignore", windowsHide: true });

    // Traversal, absolute paths and unknown ids never reach a spawn.
    for (const id of ["../..", "..", "sample-plugin/../../etc", "/etc", "C:\\Windows", "Sample-Plugin", "ab", "nul", ""]) {
      await assert.rejects(
        revealPluginFolder({ projectRoot: project.root, id, platform: "linux", spawnProcess }),
        (error) => ["invalid_plugin_id", "unsafe_reparse_point"].includes(error.code),
        id,
      );
    }
    await assert.rejects(
      revealPluginFolder({ projectRoot: project.root, id: "missing-plugin", platform: "linux", spawnProcess }),
      (error) => error.code === "ENOENT",
    );
    assert.equal(calls.length, 1, "no rejected identifier may spawn anything");
  } finally {
    await project.dispose();
  }
});

test("removal deletes exactly the requested folder and nothing else", async () => {
  const project = await temporaryProject("xirai-plugin-remove-");
  try {
    await project.plugin("sample-plugin", baseManifest(), { "ui/index.js": "code", "assets/logo.svg": "<svg/>" });
    const keeper = await project.plugin("keep-plugin", baseManifest({ id: "keep-plugin" }));
    const outside = path.join(project.root, "outside.txt");
    await writeFile(outside, "untouched", "utf8");

    assert.deepEqual(await removePluginFolder({ projectRoot: project.root, id: "sample-plugin" }), { removed: true });
    await assert.rejects(access(path.join(pluginsRootFor(project.root), "sample-plugin")));
    assert.deepEqual(await readdir(pluginsRootFor(project.root)), ["keep-plugin"]);
    assert.equal(await readFile(path.join(keeper, "plugin.json"), "utf8").then(Boolean), true);
    assert.equal(await readFile(outside, "utf8"), "untouched");
  } finally {
    await project.dispose();
  }
});

test("removal refuses traversal, unknown identifiers and a missing folder", async () => {
  const project = await temporaryProject("xirai-plugin-remove-guard-");
  try {
    await project.plugin("sample-plugin", baseManifest());
    const outside = path.join(project.root, "outside.txt");
    await writeFile(outside, "untouched", "utf8");

    for (const id of ["..", "../..", "sample-plugin/../../..", "/etc", "C:\\Windows", "Sample-Plugin", "ab", ""]) {
      await assert.rejects(
        removePluginFolder({ projectRoot: project.root, id }),
        (error) => ["invalid_plugin_id", "unsafe_reparse_point"].includes(error.code),
        id,
      );
    }
    await assert.rejects(removePluginFolder({ projectRoot: project.root, id: "missing-plugin" }), (error) => error.code === "ENOENT");

    assert.equal(await readFile(outside, "utf8"), "untouched");
    assert.deepEqual(await readdir(pluginsRootFor(project.root)), ["sample-plugin"], "nothing was deleted");
  } finally {
    await project.dispose();
  }
});

test("a linked plugin folder is never revealed or removed", async (context) => {
  const project = await temporaryProject("xirai-plugin-action-link-");
  try {
    const real = await project.plugin("sample-plugin", baseManifest());
    const linkType = process.platform === "win32" ? "junction" : "dir";
    try {
      await symlink(real, path.join(pluginsRootFor(project.root), "linked-plugin"), linkType);
    } catch (error) {
      context.skip(`this environment cannot create directory links: ${error.code}`);
      return;
    }
    const { calls, spawnProcess } = recordingSpawn();
    await assert.rejects(
      revealPluginFolder({ projectRoot: project.root, id: "linked-plugin", platform: "linux", spawnProcess }),
      (error) => error.code === "unsafe_reparse_point",
    );
    await assert.rejects(
      removePluginFolder({ projectRoot: project.root, id: "linked-plugin" }),
      (error) => error.code === "unsafe_reparse_point",
    );
    assert.equal(calls.length, 0);
    // The link and, crucially, its target both survive.
    assert.equal(await readFile(path.join(real, "plugin.json"), "utf8").then(Boolean), true);
  } finally {
    await project.dispose();
  }
});

test("a link inside a plugin folder is unlinked, never followed", async (context) => {
  const project = await temporaryProject("xirai-plugin-inner-link-");
  try {
    await project.plugin("sample-plugin", baseManifest());
    const treasure = path.join(project.root, "treasure");
    await mkdir(treasure, { recursive: true });
    await writeFile(path.join(treasure, "keep.txt"), "must survive", "utf8");
    const linkType = process.platform === "win32" ? "junction" : "dir";
    try {
      await symlink(treasure, path.join(pluginsRootFor(project.root), "sample-plugin", "escape"), linkType);
    } catch (error) {
      context.skip(`this environment cannot create directory links: ${error.code}`);
      return;
    }

    await removePluginFolder({ projectRoot: project.root, id: "sample-plugin" });
    await assert.rejects(access(path.join(pluginsRootFor(project.root), "sample-plugin")));
    assert.equal(await readFile(path.join(treasure, "keep.txt"), "utf8"), "must survive",
      "deletion must not traverse a link planted inside the plugin folder");
  } finally {
    await project.dispose();
  }
});

test("an unsafe plugin root blocks both folder actions", async () => {
  const project = await temporaryProject("xirai-plugin-action-root-");
  try {
    await writeFile(pluginsRootFor(project.root), "not a directory", "utf8");
    const { calls, spawnProcess } = recordingSpawn();
    await assert.rejects(
      revealPluginFolder({ projectRoot: project.root, id: "sample-plugin", platform: "linux", spawnProcess }),
      (error) => error.code === "plugins_root_unsafe",
    );
    await assert.rejects(
      removePluginFolder({ projectRoot: project.root, id: "sample-plugin" }),
      (error) => error.code === "plugins_root_unsafe",
    );
    assert.equal(calls.length, 0);
  } finally {
    await project.dispose();
  }
});

test("folder resolution reuses the discovery path-safety contract", async () => {
  const project = await temporaryProject("xirai-plugin-resolve-");
  try {
    const folder = await project.plugin("sample-plugin", baseManifest());
    assert.equal(await assertSafePluginFolder({ projectRoot: project.root, id: "sample-plugin" }), folder);

    // The injected reparse probe rejects here exactly as it does during discovery.
    await assert.rejects(
      assertSafePluginFolder({
        projectRoot: project.root,
        id: "sample-plugin",
        probeReparsePoint: (absolutePath) => path.basename(absolutePath) === "sample-plugin",
      }),
      (error) => error.code === "unsafe_reparse_point",
    );
  } finally {
    await project.dispose();
  }

  // The module that performs filesystem actions is separate from the read-only registry, so the
  // registry keeps its no-spawn source contract.
  const registry = await readFile(path.join(scriptsDirectory, "plugin-registry.mjs"), "utf8");
  for (const forbidden of ["child_process", "spawn(", "rm(", "unlink", "writeFile"]) {
    assert.equal(registry.includes(forbidden), false, `plugin-registry.mjs must not contain ${forbidden}`);
  }
  const actions = await readFile(path.join(scriptsDirectory, "plugin-actions.mjs"), "utf8");
  // The only spawn in the module passes an argument array with no `shell` option.
  const spawnCall = /spawnProcess\(command, args, \{([^}]*)\}\)/.exec(actions);
  assert.notEqual(spawnCall, null, "the reveal spawn must keep its documented shape");
  assert.equal(/\bshell\b/.test(spawnCall[1]), false, "no action may run through a shell");
  assert.equal(actions.includes("exec("), false);
  assert.equal(actions.includes("execSync"), false);
  // Every action derives its path from the shared safety check rather than from a caller.
  assert.equal((actions.match(/assertSafePluginFolder\(\{ projectRoot, id \}\)/g) || []).length, 2);
});

test("removal is confirmed with the plugin name, folder and consequence", () => {
  const message = pluginRemoveConfirmation({ id: "sample-plugin", name: "Sample Plugin" });
  assert.match(message, /Sample Plugin/);
  assert.match(message, /plugins\/sample-plugin/);
  assert.match(message, /不可恢复/);
});

test("the settings page confirms before removing and never sends a path", async () => {
  const app = await readFile(path.join(projectDirectory, "src", "App.jsx"), "utf8");
  const handlers = /const runPluginAction = async[\s\S]*?const clearDiagnosticLogs =/.exec(app);
  assert.notEqual(handlers, null, "the plugin action handlers must stay together");
  assert.match(handlers[0], /if \(!window\.confirm\(pluginRemoveConfirmation\(plugin\)\)\) return;/);
  assert.match(handlers[0], /fetch\(`\/api\/plugins\/\$\{encodeURIComponent\(plugin\.id\)\}`, \{ method: "DELETE" \}\)/);
  assert.match(handlers[0], /fetch\(`\/api\/plugins\/\$\{encodeURIComponent\(id\)\}\/reveal`, \{ method: "POST" \}\)/);
  // Only identifiers travel to the server: no plugin request carries a filesystem path.
  assert.equal(/path/.test(handlers[0]), false, "a plugin request must never send a path");
  // Reveal must not adopt a snapshot, because it does not return one.
  assert.match(handlers[0], /failure: "无法打开插件文件夹",\n\s+adoptSnapshot: false,/);

  const section = /settingsTab === "plugins" && <section[\s\S]*?<\/section>}/.exec(app);
  assert.match(section[0], /plugin\.description && <p className="plugin-card-description">\{plugin\.description\}<\/p>/);
  assert.match(section[0], /打开文件夹/);
  assert.match(section[0], /className="plugin-remove"/);
  assert.match(section[0], /不会运行其中任何文件/);
});
