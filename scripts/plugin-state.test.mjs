import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import {
  MAXIMUM_ENABLED_PLUGINS,
  PLUGIN_STATE_SCHEMA_VERSION,
  applyPluginEnabled,
  normalizePluginState,
  pluginStatePathFor,
  pluginToggleAdmission,
  readPluginState,
  serializePluginState,
  writePluginState,
} from "./plugin-state.mjs";
import { discoverPlugins, createPluginRegistry, pluginsRootFor } from "./plugin-registry.mjs";
import {
  PLUGIN_STATE_PRESENTATION,
  pluginDiagnosticMessage,
  pluginRegistrySummary,
  pluginStatePresentation,
  pluginToggleAvailable,
} from "../src/plugin-presentation.js";
import { PLUGIN_DIAGNOSTIC_CODES } from "./plugin-registry.mjs";

const projectDirectory = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

function baseManifest(overrides = {}) {
  return { schemaVersion: 1, id: "sample-plugin", version: "1.0.0", hostApi: { min: 1, max: 1 }, permissions: [], ...overrides };
}

async function temporaryProject(prefix) {
  const root = await mkdtemp(path.join(os.tmpdir(), prefix));
  const stateDirectory = path.join(root, "state-cache");
  return {
    root,
    stateDirectory,
    statePath: pluginStatePathFor(stateDirectory),
    async plugin(name, manifest) {
      const folder = path.join(root, "plugins", name);
      await mkdir(folder, { recursive: true });
      if (manifest !== null) await writeFile(path.join(folder, "plugin.json"), JSON.stringify(manifest, null, 2), "utf8");
      return folder;
    },
    async dispose() {
      await rm(root, { recursive: true, force: true });
    },
  };
}

test("persisted state is opt-in, bounded, and sorted", () => {
  assert.deepEqual(normalizePluginState({ schemaVersion: 1, enabled: ["zulu-plugin", "alpha-plugin", "alpha-plugin"] }), [
    "alpha-plugin", "zulu-plugin",
  ]);
  assert.deepEqual(normalizePluginState({ schemaVersion: 1, enabled: [] }), []);
  assert.deepEqual(serializePluginState(["b-plugin", "a-plugin", "a-plugin"]), {
    schemaVersion: PLUGIN_STATE_SCHEMA_VERSION, enabled: ["a-plugin", "b-plugin"],
  });

  // Fail closed on anything this build cannot vouch for.
  for (const document of [
    null,
    [],
    "enabled",
    { schemaVersion: 2, enabled: [] },
    { schemaVersion: "1", enabled: [] },
    { enabled: [] },
    { schemaVersion: 1 },
    { schemaVersion: 1, enabled: {} },
    { schemaVersion: 1, enabled: ["Bad-Id"] },
    { schemaVersion: 1, enabled: ["ab"] },
    { schemaVersion: 1, enabled: ["../escape"] },
    { schemaVersion: 1, enabled: [1] },
    { schemaVersion: 1, enabled: [], extra: true },
    { schemaVersion: 1, enabled: new Array(MAXIMUM_ENABLED_PLUGINS + 1).fill("sample-plugin") },
  ]) {
    assert.equal(normalizePluginState(document), null, JSON.stringify(document));
  }
});

test("toggling an id list is pure and idempotent", () => {
  assert.deepEqual(applyPluginEnabled([], "sample-plugin", true), ["sample-plugin"]);
  assert.deepEqual(applyPluginEnabled(["sample-plugin"], "sample-plugin", true), ["sample-plugin"]);
  assert.deepEqual(applyPluginEnabled(["sample-plugin"], "sample-plugin", false), []);
  assert.deepEqual(applyPluginEnabled(["b-plugin"], "a-plugin", true), ["a-plugin", "b-plugin"]);
  assert.deepEqual(applyPluginEnabled(["b-plugin"], "missing-plugin", false), ["b-plugin"]);
});

test("a missing state file is the normal first-run state and nothing is enabled", async () => {
  const project = await temporaryProject("xirai-plugin-state-missing-");
  try {
    assert.deepEqual(await readPluginState({ statePath: project.statePath }), { enabled: [], readable: true, present: false });
    await assert.rejects(readdir(project.stateDirectory), "reading state must not create the directory");
  } finally {
    await project.dispose();
  }
});

test("state is written atomically and read back exactly", async () => {
  const project = await temporaryProject("xirai-plugin-state-write-");
  try {
    await writePluginState({ stateDirectory: project.stateDirectory, statePath: project.statePath, enabled: ["zulu-plugin", "alpha-plugin"] });
    assert.deepEqual(await readPluginState({ statePath: project.statePath }), {
      enabled: ["alpha-plugin", "zulu-plugin"], readable: true, present: true,
    });
    const raw = await readFile(project.statePath, "utf8");
    assert.equal(raw, `${JSON.stringify({ schemaVersion: 1, enabled: ["alpha-plugin", "zulu-plugin"] }, null, 2)}\n`);
    assert.deepEqual(await readdir(project.stateDirectory), ["plugins.json"], "no temporary file may survive");
  } finally {
    await project.dispose();
  }
});

test("a malformed or future state file disables everything and is never overwritten", async () => {
  const project = await temporaryProject("xirai-plugin-state-broken-");
  try {
    await mkdir(project.stateDirectory, { recursive: true });
    for (const contents of ["{{{", "[]", JSON.stringify({ schemaVersion: 2, enabled: ["sample-plugin"] })]) {
      await writeFile(project.statePath, contents, "utf8");
      assert.deepEqual(await readPluginState({ statePath: project.statePath }), { enabled: [], readable: false, present: true });
      // The caller is expected to refuse writes while `readable` is false; the bytes stay untouched.
      assert.equal(await readFile(project.statePath, "utf8"), contents);
    }
  } finally {
    await project.dispose();
  }
});

test("only a healthy discovered plugin may be switched on", () => {
  const entry = (state) => ({ id: "sample-plugin", state });
  assert.deepEqual(pluginToggleAdmission(entry("discovered"), true), { allowed: true });
  assert.deepEqual(pluginToggleAdmission(entry("discovered"), false), { allowed: true });

  // Switching off is always admitted so a plugin that broke while enabled can still be cleared.
  for (const state of ["invalid", "incompatible", "blocked"]) {
    assert.deepEqual(pluginToggleAdmission(entry(state), false), { allowed: true }, state);
  }

  assert.equal(pluginToggleAdmission(entry("blocked"), true).code, "permissions_not_supported");
  assert.equal(pluginToggleAdmission(entry("blocked"), true).statusCode, 409);
  assert.equal(pluginToggleAdmission(entry("invalid"), true).code, "plugin_not_enableable");
  assert.equal(pluginToggleAdmission(entry("incompatible"), true).code, "plugin_not_enableable");
  assert.equal(pluginToggleAdmission(undefined, true).statusCode, 404);
  assert.equal(pluginToggleAdmission(undefined, false).code, "plugin_not_found");
});

test("the registry reports enabled only for a stored, healthy plugin", async () => {
  const project = await temporaryProject("xirai-plugin-state-registry-");
  try {
    await project.plugin("sample-plugin", baseManifest());
    await project.plugin("blocked-plugin", baseManifest({ id: "blocked-plugin", permissions: ["gpu"] }));
    await project.plugin("broken-plugin", null);

    const off = await discoverPlugins({ projectRoot: project.root });
    assert.deepEqual(off.plugins.map((entry) => entry.enabled), [false, false, false], "opt-in: nothing is on by default");

    const on = await discoverPlugins({
      projectRoot: project.root,
      // A stored id for a plugin that is blocked or invalid must never present as enabled.
      enabledIds: ["sample-plugin", "blocked-plugin", "broken-plugin"],
    });
    assert.deepEqual(on.plugins.map((entry) => [entry.id, entry.state, entry.enabled]), [
      ["blocked-plugin", "blocked", false],
      ["broken-plugin", "invalid", false],
      ["sample-plugin", "discovered", true],
    ]);
    assert.equal(on.execution, "not-supported");
    assert.equal(on.plugins.find((entry) => entry.id === "sample-plugin").execution, "not-supported");
  } finally {
    await project.dispose();
  }
});

test("a stored preference survives a manifest breaking and returns when it is repaired", async () => {
  const project = await temporaryProject("xirai-plugin-state-sticky-");
  try {
    const folder = await project.plugin("sample-plugin", baseManifest());
    const enabledIds = ["sample-plugin"];
    assert.equal((await discoverPlugins({ projectRoot: project.root, enabledIds })).plugins[0].enabled, true);

    await writeFile(path.join(folder, "plugin.json"), "{{{", "utf8");
    const broken = await discoverPlugins({ projectRoot: project.root, enabledIds });
    assert.equal(broken.plugins[0].enabled, false);
    assert.deepEqual(broken.plugins[0].diagnostics, ["manifest_not_json"]);

    await writeFile(path.join(folder, "plugin.json"), JSON.stringify(baseManifest()), "utf8");
    assert.equal((await discoverPlugins({ projectRoot: project.root, enabledIds })).plugins[0].enabled, true);
  } finally {
    await project.dispose();
  }
});

test("an unreadable preference file is reported once at registry scope", async () => {
  const project = await temporaryProject("xirai-plugin-state-diagnostic-");
  try {
    await project.plugin("sample-plugin", baseManifest());
    const snapshot = await discoverPlugins({ projectRoot: project.root, enabledIds: ["sample-plugin"], stateReadable: false });
    assert.deepEqual(snapshot.diagnostics, [{ id: null, code: "plugin_state_unreadable" }]);
    assert.equal(snapshot.plugins[0].enabled, true, "the loader already emptied the set; the flag only reports");

    // The registry-scope diagnostic survives an early fail-closed return.
    await rm(pluginsRootFor(project.root), { recursive: true, force: true });
    await writeFile(pluginsRootFor(project.root), "not a directory", "utf8");
    const unsafe = await discoverPlugins({ projectRoot: project.root, stateReadable: false });
    assert.deepEqual(unsafe.diagnostics, [
      { id: null, code: "plugin_state_unreadable" },
      { id: null, code: "plugins_root_unsafe" },
    ]);
  } finally {
    await project.dispose();
  }
});

test("the lazy registry reloads persisted state on every scan", async () => {
  const project = await temporaryProject("xirai-plugin-state-reload-");
  try {
    await project.plugin("sample-plugin", baseManifest());
    let loads = 0;
    const registry = createPluginRegistry({
      projectRoot: project.root,
      loadState: async () => {
        loads += 1;
        return readPluginState({ statePath: project.statePath });
      },
    });
    assert.equal(loads, 0, "nothing is read before the first request");
    assert.equal((await registry.read()).plugins[0].enabled, false);

    await writePluginState({ stateDirectory: project.stateDirectory, statePath: project.statePath, enabled: ["sample-plugin"] });
    assert.equal((await registry.read()).plugins[0].enabled, true, "a later read sees the new preference");
    assert.equal(loads, 2);

    const [left, right] = await Promise.all([registry.read(), registry.read()]);
    assert.equal(left, right, "concurrent reads still share one scan");
    assert.equal(loads, 3);
  } finally {
    await project.dispose();
  }
});

test("every diagnostic code and state has user-facing text", () => {
  for (const code of PLUGIN_DIAGNOSTIC_CODES) {
    const message = pluginDiagnosticMessage(code);
    assert.equal(message.includes("未识别的诊断码"), false, code);
    assert.equal(message.length > 0, true, code);
  }
  assert.match(pluginDiagnosticMessage("brand_new_code"), /未识别的诊断码/);

  for (const state of ["discovered", "invalid", "incompatible", "blocked"]) {
    assert.equal(typeof PLUGIN_STATE_PRESENTATION[state].label, "string");
    assert.equal(pluginStatePresentation(state).label.length > 0, true, state);
  }
  assert.equal(pluginStatePresentation("brand-new-state").tone, "error");
});

test("only a discovered plugin exposes an interactive switch", () => {
  assert.equal(pluginToggleAvailable({ state: "discovered" }), true);
  for (const state of ["invalid", "incompatible", "blocked", "unknown"]) {
    assert.equal(pluginToggleAvailable({ state }), false, state);
  }
  assert.equal(pluginToggleAvailable(null), false);
  assert.equal(pluginToggleAvailable(undefined), false);
});

test("the settings summary counts total, enabled and unhealthy plugins", () => {
  assert.deepEqual(pluginRegistrySummary(null), { total: 0, enabled: 0, needsAttention: 0 });
  assert.deepEqual(pluginRegistrySummary({ plugins: [] }), { total: 0, enabled: 0, needsAttention: 0 });
  assert.deepEqual(pluginRegistrySummary({
    plugins: [
      { state: "discovered", enabled: true },
      { state: "discovered", enabled: false },
      { state: "blocked", enabled: false },
      { state: "invalid", enabled: false },
    ],
  }), { total: 4, enabled: 1, needsAttention: 2 });
});

test("the settings page never presents enabling as an execution grant", async () => {
  const app = await readFile(path.join(projectDirectory, "src", "App.jsx"), "utf8");
  const section = /settingsTab === "plugins" && <section[\s\S]*?<\/section>}/.exec(app);
  assert.notEqual(section, null, "the plugins settings section must exist");
  assert.match(section[0], /本版本不执行任何插件代码/);
  assert.match(section[0], /不是执行授权/);
  assert.match(section[0], /默认关闭/);
  assert.match(section[0], /state-cache\/plugins\.json/);
  assert.match(section[0], /已启用（仍不会执行）/);

  // The nav entry and its lazy load effect are wired.
  assert.match(app, /settingsTab === "plugins" \? "active" : ""/);
  assert.match(app, /if \(!settingsOpen \|\| settingsTab !== "plugins"\) return;\n\s+void refreshPlugins\(\);/);
  // Toggles go through the dedicated route and adopt the returned snapshot.
  assert.match(app, /fetch\(`\/api\/plugins\/\$\{encodeURIComponent\(id\)\}`, \{\n\s+method: "PUT",/);
  assert.match(app, /setPluginRegistry\(payload\);/);
  // The switch is disabled for anything the host refuses to enable.
  assert.match(section[0], /disabled=\{!toggleAvailable \|\| Boolean\(pluginPendingId\)\}/);
});
