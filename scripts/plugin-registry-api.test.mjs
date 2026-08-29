import assert from "node:assert/strict";
import http from "node:http";
import { readFile, readdir, stat } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { pluginRegistryApiPlugin } from "../vite.config.js";
import { PLUGIN_DIAGNOSTIC_CODES, PLUGIN_HOST_API_VERSION } from "./plugin-registry.mjs";

const scriptsDirectory = path.dirname(fileURLToPath(import.meta.url));
const projectDirectory = path.dirname(scriptsDirectory);
const HOOKS = ["configureServer", "configurePreviewServer"];

// Only `plugins/README.md` carries this sentence, so finding it inside `dist/` would mean build
// output picked up plugin-port content.
const PLUGIN_README_SENTINEL = "This folder is the plugin extension port.";

const sampleRegistry = Object.freeze({
  schemaVersion: 1,
  hostApiVersion: PLUGIN_HOST_API_VERSION,
  execution: "not-supported",
  plugins: [Object.freeze({
    id: "sample-plugin",
    name: "Sample Plugin",
    version: "1.0.0",
    enabled: false,
    state: "discovered",
    compatible: true,
    execution: "not-supported",
    contributions: { panels: [], commands: [] },
    diagnostics: [],
  })],
  diagnostics: [],
});

function middlewareServer() {
  const handlers = [];
  return {
    middlewares: { use(handler) { handlers.push(handler); } },
    handlers,
  };
}

async function requestThrough(handlers, target, { method = "GET", headers = {}, body } = {}) {
  const listener = http.createServer((request, response) => {
    let index = 0;
    const next = (error) => {
      if (error) { response.statusCode = 500; response.end(error.message); return; }
      const handler = handlers[index++];
      if (!handler) { response.statusCode = 599; response.end("no downstream"); return; }
      Promise.resolve(handler(request, response, next)).catch(next);
    };
    next();
  });
  await new Promise((resolve) => listener.listen(0, "127.0.0.1", resolve));
  const { port } = listener.address();
  try {
    return await new Promise((resolve, reject) => {
      const request = http.request({ host: "127.0.0.1", port, path: target, method, headers }, (response) => {
        let payload = "";
        response.setEncoding("utf8");
        response.on("data", (chunk) => { payload += chunk; });
        response.on("end", () => resolve({ statusCode: response.statusCode, headers: response.headers, body: payload }));
      });
      request.once("error", reject);
      if (body !== undefined) request.write(typeof body === "string" ? body : JSON.stringify(body));
      request.end();
    });
  } finally {
    await new Promise((resolve, reject) => listener.close((error) => error ? reject(error) : resolve()));
  }
}

async function mountedPlugin({ readRegistry, setPluginEnabled, revealPlugin, removePlugin } = {}) {
  const plugin = pluginRegistryApiPlugin({
    ...(readRegistry ? { readRegistry } : {}),
    ...(setPluginEnabled ? { setPluginEnabled } : {}),
    ...(revealPlugin ? { revealPlugin } : {}),
    ...(removePlugin ? { removePlugin } : {}),
  });
  const server = middlewareServer();
  await plugin.configureServer(server);
  let downstreamHits = 0;
  server.middlewares.use((_request, response) => {
    downstreamHits += 1;
    response.statusCode = 204;
    response.end();
  });
  return { handlers: server.handlers, downstream: () => downstreamHits };
}

for (const hook of HOOKS) {
  test(`${hook} serves the plugin registry as a no-store JSON snapshot`, async () => {
    const plugin = pluginRegistryApiPlugin({ readRegistry: async () => sampleRegistry });
    const server = middlewareServer();
    await plugin[hook](server);

    const response = await requestThrough(server.handlers, "/api/plugins");
    assert.equal(response.statusCode, 200);
    assert.equal(response.headers["cache-control"], "no-store");
    assert.equal(response.headers["content-type"], "application/json; charset=utf-8");
    assert.deepEqual(JSON.parse(response.body), sampleRegistry);
  });

  test(`${hook} refuses to serve any plugin file`, async () => {
    const plugin = pluginRegistryApiPlugin({ readRegistry: async () => sampleRegistry });
    const server = middlewareServer();
    await plugin[hook](server);
    let downstreamHits = 0;
    server.middlewares.use((_request, response) => {
      downstreamHits += 1;
      response.statusCode = 204;
      response.end();
    });

    for (const target of [
      "/plugins",
      "/plugins/",
      "/plugins/sample-plugin/index.js",
      "/plugins/sample-plugin/plugin.json",
      "/plugins/README.md",
      "//plugins/sample-plugin/index.js",
      "/assets/../plugins/sample-plugin/index.js",
      "/%70lugins/sample-plugin/index.js",
      "/plugins%2Fsample-plugin/index.js",
      "/plugins/sample-plugin/index.js?import&t=123",
      "/plugins\\sample-plugin\\index.js",
    ]) {
      const blocked = await requestThrough(server.handlers, target);
      assert.equal(blocked.statusCode, 404, target);
      assert.equal(blocked.headers["cache-control"], "no-store", target);
      assert.deepEqual(JSON.parse(blocked.body), { error: "Plugin files are not served" }, target);
    }
    assert.equal(downstreamHits, 0, "no plugin request may reach Vite's static or fallback layer");

    // Everything else still flows to the rest of the stack untouched.
    for (const target of ["/", "/index.html", "/src/App.jsx", "/pluginsomething/x.js", "/assets/plugins.js"]) {
      assert.equal((await requestThrough(server.handlers, target)).statusCode, 204, target);
    }
    assert.equal(downstreamHits, 5);
  });
}

test("the registry route accepts GET only and enforces same-origin", async () => {
  const mounted = await mountedPlugin({ readRegistry: async () => sampleRegistry });

  for (const method of ["POST", "PUT", "DELETE", "PATCH"]) {
    const response = await requestThrough(mounted.handlers, "/api/plugins", { method });
    assert.equal(response.statusCode, 405, method);
    assert.deepEqual(JSON.parse(response.body), { error: "Method not allowed" });
  }

  const crossOrigin = await requestThrough(mounted.handlers, "/api/plugins", { headers: { Origin: "http://evil.example" } });
  assert.equal(crossOrigin.statusCode, 403);
  assert.equal(mounted.downstream(), 0);
});

test("the enable route persists a toggle and answers with a fresh snapshot", async () => {
  const calls = [];
  const mounted = await mountedPlugin({
    setPluginEnabled: async (id, enabled) => {
      calls.push([id, enabled]);
      return { ...sampleRegistry, plugins: [{ ...sampleRegistry.plugins[0], enabled }] };
    },
  });

  const enable = await requestThrough(mounted.handlers, "/api/plugins/sample-plugin", { method: "PUT", body: { enabled: true } });
  assert.equal(enable.statusCode, 200);
  assert.equal(enable.headers["cache-control"], "no-store");
  assert.equal(enable.headers["content-type"], "application/json; charset=utf-8");
  assert.equal(JSON.parse(enable.body).plugins[0].enabled, true);
  // Execution support is never implied by enabling.
  assert.equal(JSON.parse(enable.body).execution, "not-supported");

  const disable = await requestThrough(mounted.handlers, "/api/plugins/sample-plugin", { method: "PUT", body: { enabled: false } });
  assert.equal(JSON.parse(disable.body).plugins[0].enabled, false);
  assert.deepEqual(calls, [["sample-plugin", true], ["sample-plugin", false]]);

  // Percent-encoded identifiers reach the handler decoded.
  await requestThrough(mounted.handlers, "/api/plugins/sample%2Dplugin", { method: "PUT", body: { enabled: true } });
  assert.deepEqual(calls[2], ["sample-plugin", true]);
  assert.equal(mounted.downstream(), 0);
});

test("the enable route rejects bad methods, bodies and refused toggles", async () => {
  const mounted = await mountedPlugin({
    setPluginEnabled: async () => {
      throw Object.assign(new Error("This plugin declares permissions, which are not supported"), {
        statusCode: 409, publicCode: "permissions_not_supported",
      });
    },
  });

  // `PUT` toggles and `DELETE` removes; every other method on the plugin path is refused.
  for (const method of ["GET", "POST", "PATCH"]) {
    const response = await requestThrough(mounted.handlers, "/api/plugins/sample-plugin", { method });
    assert.equal(response.statusCode, 405, method);
  }

  for (const body of [{}, { enabled: "true" }, { enabled: 1 }, { enabled: null }]) {
    const response = await requestThrough(mounted.handlers, "/api/plugins/sample-plugin", { method: "PUT", body });
    assert.equal(response.statusCode, 400, JSON.stringify(body));
    assert.deepEqual(JSON.parse(response.body), { error: "Field enabled must be a boolean", code: "invalid_request" });
  }

  const malformed = await requestThrough(mounted.handlers, "/api/plugins/sample-plugin", { method: "PUT", body: "{{{" });
  assert.equal(malformed.statusCode, 400);

  const refused = await requestThrough(mounted.handlers, "/api/plugins/sample-plugin", { method: "PUT", body: { enabled: true } });
  assert.equal(refused.statusCode, 409);
  assert.deepEqual(JSON.parse(refused.body), {
    error: "This plugin declares permissions, which are not supported", code: "permissions_not_supported",
  });
  assert.equal(mounted.downstream(), 0);
});

test("the reveal route opens a folder by identifier and returns no snapshot", async () => {
  const revealed = [];
  const mounted = await mountedPlugin({ revealPlugin: async (id) => { revealed.push(id); return { id, revealed: true }; } });

  const response = await requestThrough(mounted.handlers, "/api/plugins/sample-plugin/reveal", { method: "POST" });
  assert.equal(response.statusCode, 200);
  assert.equal(response.headers["cache-control"], "no-store");
  assert.deepEqual(JSON.parse(response.body), { id: "sample-plugin", revealed: true });
  assert.deepEqual(revealed, ["sample-plugin"]);

  for (const method of ["GET", "PUT", "DELETE"]) {
    assert.equal((await requestThrough(mounted.handlers, "/api/plugins/sample-plugin/reveal", { method })).statusCode, 405, method);
  }
  // `reveal` is the only recognised action. Anything else is not a plugin route at all: it falls
  // through untouched rather than being treated as an unnamed action.
  for (const target of ["/api/plugins/sample-plugin/reveal/extra", "/api/plugins/sample-plugin/open", "/api/plugins/a/b/c"]) {
    assert.equal((await requestThrough(mounted.handlers, target, { method: "POST" })).statusCode, 204, target);
  }
  assert.deepEqual(revealed, ["sample-plugin"], "no unrecognised path may reach a folder action");
});

test("the remove route deletes by identifier and answers with a rescanned snapshot", async () => {
  const removed = [];
  const mounted = await mountedPlugin({
    removePlugin: async (id) => { removed.push(id); return { ...sampleRegistry, plugins: [] }; },
  });

  const response = await requestThrough(mounted.handlers, "/api/plugins/sample-plugin", { method: "DELETE" });
  assert.equal(response.statusCode, 200);
  assert.equal(response.headers["cache-control"], "no-store");
  assert.deepEqual(JSON.parse(response.body).plugins, []);
  assert.deepEqual(removed, ["sample-plugin"]);
});

test("folder-action failures surface a stable code and never a path or stack", async () => {
  const mounted = await mountedPlugin({
    revealPlugin: async () => {
      throw Object.assign(new Error("Plugin was not found"), { statusCode: 404, publicCode: "plugin_not_found" });
    },
    removePlugin: async () => {
      throw Object.assign(new Error("Could not remove the plugin folder"), { statusCode: 409, publicCode: "unsafe_reparse_point" });
    },
  });

  const reveal = await requestThrough(mounted.handlers, "/api/plugins/ghost-plugin/reveal", { method: "POST" });
  assert.equal(reveal.statusCode, 404);
  assert.deepEqual(JSON.parse(reveal.body), { error: "Plugin was not found", code: "plugin_not_found" });

  const remove = await requestThrough(mounted.handlers, "/api/plugins/linked-plugin", { method: "DELETE" });
  assert.equal(remove.statusCode, 409);
  assert.deepEqual(JSON.parse(remove.body), { error: "Could not remove the plugin folder", code: "unsafe_reparse_point" });
  for (const body of [reveal.body, remove.body]) {
    assert.equal(/[A-Za-z]:\\|\/home\/|at Object|Error:/.test(body), false, body);
  }
});

test("the enable route is loopback and same-origin only", async () => {
  let calls = 0;
  const mounted = await mountedPlugin({ setPluginEnabled: async () => { calls += 1; return sampleRegistry; } });
  const crossOrigin = await requestThrough(mounted.handlers, "/api/plugins/sample-plugin", {
    method: "PUT", headers: { Origin: "http://evil.example" }, body: { enabled: true },
  });
  assert.equal(crossOrigin.statusCode, 403);
  assert.equal(calls, 0, "a cross-origin request must never reach the writer");

  // Reveal and remove sit behind the same check, so no folder action is reachable cross-origin.
  let actions = 0;
  const guarded = await mountedPlugin({
    revealPlugin: async () => { actions += 1; return {}; },
    removePlugin: async () => { actions += 1; return sampleRegistry; },
  });
  for (const [target, method] of [["/api/plugins/sample-plugin/reveal", "POST"], ["/api/plugins/sample-plugin", "DELETE"]]) {
    const response = await requestThrough(guarded.handlers, target, { method, headers: { Origin: "http://evil.example" } });
    assert.equal(response.statusCode, 403, target);
  }
  assert.equal(actions, 0, "no cross-origin request may reach a folder action");

  const source = await readFile(path.join(projectDirectory, "vite.config.js"), "utf8");
  assert.match(source, /requireLocalRequest\(request, "Plugin management is available only from this computer"\)/);
  // One guard covers every plugin write; a new action cannot be added outside it.
  assert.equal((source.match(/requireLocalRequest\(request, "Plugin management/g) || []).length, 1);
});

test("a failing registry returns a stable error and never leaks a path or a stack", async () => {
  const mounted = await mountedPlugin({
    readRegistry: async () => {
      throw Object.assign(new Error("EACCES: permission denied, scandir 'D:\\.XAIG\\XiriaCanvas AI\\plugins'"), { code: "EACCES" });
    },
  });
  const response = await requestThrough(mounted.handlers, "/api/plugins");
  assert.equal(response.statusCode, 500);
  assert.deepEqual(JSON.parse(response.body), { error: "Plugin registry is unavailable" });
  assert.equal(response.headers["cache-control"], "no-store");
  assert.equal(response.body.includes("XiriaCanvas"), false);
  assert.equal(response.body.includes("EACCES"), false);
});

test("a failing registry leaves every other control-plane route working", async () => {
  const mounted = await mountedPlugin({ readRegistry: async () => { throw new Error("registry down"); } });
  assert.equal((await requestThrough(mounted.handlers, "/api/plugins")).statusCode, 500);
  for (const target of ["/api/models", "/api/ui-state", "/api/inference/health", "/api/logs", "/api/update/state"]) {
    assert.equal((await requestThrough(mounted.handlers, target)).statusCode, 204, target);
  }
  assert.equal(mounted.downstream(), 5);
});

test("registry diagnostics reach the client as stable codes only", async () => {
  const mounted = await mountedPlugin({
    readRegistry: async () => ({
      schemaVersion: 1,
      hostApiVersion: 1,
      execution: "not-supported",
      plugins: [{
        id: "broken-plugin",
        name: "broken-plugin",
        description: "",
        version: null,
        enabled: false,
        state: "invalid",
        compatible: false,
        execution: "not-supported",
        contributions: { panels: [], commands: [] },
        diagnostics: ["manifest_not_json"],
      }],
      diagnostics: [{ id: "broken-plugin", code: "manifest_not_json" }],
    }),
  });
  const payload = JSON.parse((await requestThrough(mounted.handlers, "/api/plugins")).body);
  for (const diagnostic of payload.diagnostics) {
    assert.equal(PLUGIN_DIAGNOSTIC_CODES.includes(diagnostic.code), true, diagnostic.code);
  }
  assert.deepEqual(Object.keys(payload.plugins[0]).sort(), [
    "compatible", "contributions", "description", "diagnostics", "enabled", "execution", "id", "name", "state", "version",
  ]);
});

test("the registry plugin is wired into the control plane behind the setup gate", async () => {
  const source = await readFile(path.join(projectDirectory, "vite.config.js"), "utf8");

  // Registered in both the dev and preview middleware stacks, following the existing API plugins.
  assert.match(source, /name: "local-plugin-registry-api",\n\s+configureServer\(server\) \{ server\.middlewares\.use\(middleware\); \},\n\s+configurePreviewServer\(server\) \{ server\.middlewares\.use\(middleware\); \},/);

  // The setup gate stays `enforce: "pre"`, and the registry only mounts once setup is complete, so
  // an unconfigured install answers `/config` instead of exposing the plugin port.
  assert.match(source, /name: "setup-gate",\n\s+enforce: "pre",/);
  const registration = /plugins: \[setupGatePlugin\(\), themedLogoPlugin\(\), react\(\), \.\.\.\(setupComplete \? \[([^\]]+)\]/.exec(source);
  assert.notEqual(registration, null, "the plugin array must keep its documented shape");
  assert.equal(registration[1].includes("pluginRegistryApiPlugin()"), true);

  // The plugin root is fixed: the API never reads a root from the request, the manifest, or an
  // environment variable, and every derivation goes through `pluginsRootFor(projectRoot)`.
  assert.match(source, /servesPluginContent\(request\.url, projectRoot\)/);
  assert.equal(/XIRAI_PLUGIN|PLUGIN_DIR|pluginsRoot\s*=/.test(source), false);
  for (const call of source.match(/pluginsRootFor\([^)]*\)/g) || []) {
    assert.equal(call, "pluginsRootFor(projectRoot)", call);
  }

  // Dropping a plugin folder in must not churn the dev watcher; plugin files are never modules.
  assert.match(source, /const runtimeWatchDirectories = \[[^\]]*pluginsRootFor\(projectRoot\),\n\];/);

  // Discovery stays in the Node control plane; the FastAPI data plane is not involved.
  const backend = await readFile(path.join(projectDirectory, "backend", "inference_server.py"), "utf8");
  assert.equal(backend.includes("/api/plugins"), false);
  assert.match(source, /const inferenceProtocol = 34;/);
  assert.match(backend, /^INFERENCE_PROTOCOL = 34$/m);
});

test("the build never transpiles, copies, or bundles plugin content", async (context) => {
  const source = await readFile(path.join(projectDirectory, "vite.config.js"), "utf8");
  assert.equal(/publicDir/.test(source), false, "the public directory stays the Vite default of public/");

  const distDirectory = path.join(projectDirectory, "dist");
  try {
    await stat(distDirectory);
  } catch {
    context.skip("dist/ is absent; run npm run build before this gate");
    return;
  }

  const files = [];
  const pending = [distDirectory];
  while (pending.length) {
    for (const entry of await readdir(pending.pop(), { withFileTypes: true })) {
      assert.notEqual(entry.name, "plugins", "dist/ must never contain a plugins directory");
      const absolute = path.join(entry.parentPath || entry.path, entry.name);
      if (entry.isDirectory()) pending.push(absolute);
      else if (entry.isFile()) files.push(absolute);
    }
  }
  assert.notEqual(files.length, 0, "dist/ must hold build output");
  for (const file of files) {
    const contents = await readFile(file);
    assert.equal(contents.includes(PLUGIN_README_SENTINEL), false, `${path.relative(projectDirectory, file)} contains plugin-port content`);
  }
});

test("no application source imports anything from the plugin port", async () => {
  const roots = ["src", "backend", "scripts"];
  for (const root of roots) {
    const pending = [path.join(projectDirectory, root)];
    while (pending.length) {
      for (const entry of await readdir(pending.pop(), { withFileTypes: true })) {
        const absolute = path.join(entry.parentPath || entry.path, entry.name);
        if (entry.isDirectory()) {
          if (entry.name !== "__pycache__") pending.push(absolute);
          continue;
        }
        if (!entry.isFile() || !/\.(m?js|jsx|py)$/.test(entry.name)) continue;
        const contents = await readFile(absolute, "utf8");
        assert.equal(/from ["']\.{0,2}\/?plugins\//.test(contents), false, `${absolute} imports from the plugin port`);
        assert.equal(/import\(["'][^"']*\/plugins\//.test(contents), false, `${absolute} dynamically imports plugin code`);
      }
    }
  }
});

test("the plugin port README documents the discovery-only contract", async () => {
  const readme = await readFile(path.join(projectDirectory, "plugins", "README.md"), "utf8");
  assert.equal(readme.includes(PLUGIN_README_SENTINEL), true);
  assert.match(readme, /12-PLUGIN-ARCHITECTURE-API\.md/);
  assert.match(readme, /13-PLUGIN-DEVELOPMENT-GUIDELINES-ZH\.md/);

  const ignore = await readFile(path.join(projectDirectory, ".gitignore"), "utf8");
  assert.match(ignore, /^plugins\/\*$/m, "installed plugin folders stay untracked and out of npm pack");
  assert.match(ignore, /^!plugins\/README\.md$/m);
});
