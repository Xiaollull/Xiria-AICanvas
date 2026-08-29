import assert from "node:assert/strict";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { assistantApiPlugin } from "../vite.config.js";
import { strengthPayload } from "../src/ai-assistant-providers.js";

const projectDirectory = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const HOOKS = ["configureServer", "configurePreviewServer"];
const SECRET = "sk-live-should-never-be-echoed-9911";

const validSettings = Object.freeze({
  provider: "deepseek",
  baseUrl: "https://api.deepseek.com/v1",
  apiKey: SECRET,
  model: "deepseek-v4-pro",
  strength: 1,
  personaId: "prompt-architect",
});

function middlewareServer() {
  const handlers = [];
  return { middlewares: { use(handler) { handlers.push(handler); } }, handlers };
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
      const outbound = http.request({ host: "127.0.0.1", port, path: target, method, headers }, (response) => {
        let payload = "";
        response.setEncoding("utf8");
        response.on("data", (chunk) => { payload += chunk; });
        response.on("end", () => resolve({ statusCode: response.statusCode, headers: response.headers, body: payload }));
      });
      outbound.once("error", reject);
      if (body !== undefined) outbound.write(typeof body === "string" ? body : JSON.stringify(body));
      outbound.end();
    });
  } finally {
    await new Promise((resolve, reject) => listener.close((error) => error ? reject(error) : resolve()));
  }
}

// Records what the proxy would have sent upstream, so provider-shape assertions never need a
// network call or a real key.
function stubUpstream({ status = 200, payload = "", capture = {} } = {}) {
  return async (settings, messages, options) => {
    capture.settings = settings;
    capture.messages = messages;
    capture.options = options;
    return {
      ok: status >= 200 && status < 300,
      status,
      async text() { return typeof payload === "string" ? payload : JSON.stringify(payload); },
      body: (async function* stream() { yield Buffer.from(typeof payload === "string" ? payload : JSON.stringify(payload)); })(),
    };
  };
}

async function mounted(hook, { stateDirectory, upstreamRequest, projectRoot = projectDirectory } = {}) {
  const plugin = assistantApiPlugin({ stateDirectory, projectRoot, ...(upstreamRequest ? { upstreamRequest } : {}) });
  const server = middlewareServer();
  await plugin[hook](server);
  let downstreamHits = 0;
  server.middlewares.use((_request, response) => { downstreamHits += 1; response.statusCode = 204; response.end(); });
  return { handlers: server.handlers, downstream: () => downstreamHits };
}

async function temporaryState(prefix) {
  return mkdtemp(path.join(os.tmpdir(), prefix));
}

for (const hook of HOOKS) {
  test(`${hook} stores settings and never returns the API key`, async () => {
    const stateDirectory = await temporaryState("xirai-assistant-settings-");
    try {
      const { handlers } = await mounted(hook, { stateDirectory });
      const saved = await requestThrough(handlers, "/api/assistant/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: validSettings,
      });
      assert.equal(saved.statusCode, 200);
      const savedBody = JSON.parse(saved.body);
      assert.equal(savedBody.settings.hasApiKey, true);
      assert.equal(savedBody.settings.apiKeyHint, "••••9911");
      assert.equal(savedBody.settings.apiKey, undefined);
      assert.ok(!saved.body.includes(SECRET), "PUT response must not echo the API key");

      const fetched = await requestThrough(handlers, "/api/assistant/settings");
      assert.equal(fetched.statusCode, 200);
      assert.ok(!fetched.body.includes(SECRET), "GET response must not echo the API key");
      assert.equal(JSON.parse(fetched.body).settings.model, "deepseek-v4-pro");

      // The secret is on disk, which is exactly why it must never be in a response body. Settings
      // are stored as the active entry of the profile set, so the key lives one level in.
      const onDisk = JSON.parse(await readFile(path.join(stateDirectory, "assistant-settings.json"), "utf8"));
      const live = onDisk.profiles.find((profile) => profile.id === onDisk.activeId);
      assert.equal(live.settings.apiKey, SECRET);
    } finally {
      await rm(stateDirectory, { recursive: true, force: true });
    }
  });
}

test("an omitted API key preserves the stored one and an empty string clears it", async () => {
  const stateDirectory = await temporaryState("xirai-assistant-merge-");
  try {
    const { handlers } = await mounted("configureServer", { stateDirectory });
    const put = (body) => requestThrough(handlers, "/api/assistant/settings", {
      method: "PUT", headers: { "Content-Type": "application/json" }, body,
    });
    await put(validSettings);

    // The settings form only ever shows a masked key, so saving an unrelated field must not wipe it.
    const preserved = await put({ ...validSettings, apiKey: undefined, strength: 0.7 });
    assert.equal(preserved.statusCode, 200);
    assert.equal(JSON.parse(preserved.body).settings.hasApiKey, true);
    assert.equal(JSON.parse(preserved.body).settings.strength, 0.7);

    const cleared = await put({ ...validSettings, apiKey: "" });
    assert.equal(cleared.statusCode, 400, "clearing the key of a provider that requires one is invalid");
    assert.deepEqual(JSON.parse(cleared.body).errors.map((entry) => entry.code), ["api_key_missing"]);
  } finally {
    await rm(stateDirectory, { recursive: true, force: true });
  }
});

test("invalid configuration is refused with per-field codes and never written", async () => {
  const stateDirectory = await temporaryState("xirai-assistant-invalid-");
  try {
    const { handlers } = await mounted("configureServer", { stateDirectory });
    const rejected = await requestThrough(handlers, "/api/assistant/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: { provider: "deepseek", baseUrl: "not-a-url", apiKey: "", model: "", strength: 99 },
    });
    assert.equal(rejected.statusCode, 400);
    const codes = JSON.parse(rejected.body).errors.map((entry) => entry.code).sort();
    assert.deepEqual(codes, ["api_key_missing", "base_url_invalid", "model_missing", "strength_out_of_range"]);
    await assert.rejects(readFile(path.join(stateDirectory, "assistant-settings.json"), "utf8"));
  } finally {
    await rm(stateDirectory, { recursive: true, force: true });
  }
});

test("a URL carrying embedded credentials is refused", async () => {
  const stateDirectory = await temporaryState("xirai-assistant-credentials-");
  try {
    const { handlers } = await mounted("configureServer", { stateDirectory });
    const rejected = await requestThrough(handlers, "/api/assistant/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: { ...validSettings, baseUrl: "https://user:pass@api.deepseek.com/v1" },
    });
    assert.equal(rejected.statusCode, 400);
    assert.ok(JSON.parse(rejected.body).errors.some((entry) => entry.code === "base_url_credentials"));
  } finally {
    await rm(stateDirectory, { recursive: true, force: true });
  }
});

test("chat is refused before the client is configured", async () => {
  const stateDirectory = await temporaryState("xirai-assistant-unconfigured-");
  try {
    const { handlers } = await mounted("configureServer", { stateDirectory, upstreamRequest: stubUpstream() });
    const blocked = await requestThrough(handlers, "/api/assistant/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { history: [{ role: "user", content: "hi" }] },
    });
    assert.equal(blocked.statusCode, 400);
    assert.ok(JSON.parse(blocked.body).errors.some((entry) => entry.code === "api_key_missing"));
  } finally {
    await rm(stateDirectory, { recursive: true, force: true });
  }
});

test("chat streams provider frames through and sends the persona as the system message", async () => {
  const stateDirectory = await temporaryState("xirai-assistant-chat-");
  try {
    const capture = {};
    const frames = 'data: {"choices":[{"delta":{"content":"好"}}]}\n\ndata: [DONE]\n\n';
    const { handlers } = await mounted("configureServer", {
      stateDirectory,
      upstreamRequest: stubUpstream({ payload: frames, capture }),
    });
    await requestThrough(handlers, "/api/assistant/settings", {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: validSettings,
    });
    const streamed = await requestThrough(handlers, "/api/assistant/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { history: [{ role: "user", content: "画一只猫" }] },
    });
    assert.equal(streamed.statusCode, 200);
    assert.match(streamed.headers["content-type"], /text\/event-stream/);
    assert.equal(streamed.headers["x-accel-buffering"], "no");
    assert.equal(streamed.body, frames);

    assert.equal(capture.messages[0].role, "system");
    // Proves the selected persona's systemPrompt was applied, and that only the prompt travels:
    // the record's description is metadata for the picker and must not reach the provider.
    assert.ok(capture.messages[0].content.includes("权重语法"), "persona systemPrompt must be sent");
    assert.ok(!capture.messages[0].content.includes("通用提示词顾问"), "persona description must not be sent");
    assert.ok(capture.messages[0].content.includes("```prompt"), "output protocol must always be appended");
    assert.deepEqual(capture.messages.slice(1), [{ role: "user", content: "画一只猫" }]);
    assert.equal(capture.options.stream, true);
  } finally {
    await rm(stateDirectory, { recursive: true, force: true });
  }
});

test("strength is resolved per model: a chat model sends temperature, a reasoning model does not", async () => {
  const stateDirectory = await temporaryState("xirai-assistant-strength-");
  try {
    const capture = {};
    const { handlers } = await mounted("configureServer", {
      stateDirectory,
      upstreamRequest: stubUpstream({ payload: "data: [DONE]\n\n", capture }),
    });
    const chat = (body) => requestThrough(handlers, "/api/assistant/chat", {
      method: "POST", headers: { "Content-Type": "application/json" }, body,
    });
    const save = (body) => requestThrough(handlers, "/api/assistant/settings", {
      method: "PUT", headers: { "Content-Type": "application/json" }, body,
    });

    await save(validSettings);
    await chat({ history: [{ role: "user", content: "hi" }] });
    assert.equal(capture.settings.model, "deepseek-v4-pro");
    assert.deepEqual(strengthPayload(capture.settings.provider, capture.settings.model, capture.settings.strength),
      { temperature: capture.settings.strength });

    // An OpenAI reasoning model rejects temperature, so the request must carry reasoning_effort and
    // no temperature key at all.
    await save({ provider: "openai", baseUrl: "https://api.openai.com/v1", apiKey: SECRET, model: "gpt-5.6-sol", strength: "high" });
    await chat({ history: [{ role: "user", content: "hi" }] });
    assert.equal(capture.settings.model, "gpt-5.6-sol");
    const payload = strengthPayload(capture.settings.provider, capture.settings.model, capture.settings.strength);
    assert.deepEqual(payload, { reasoning_effort: "high" });
    assert.equal("temperature" in payload, false);
  } finally {
    await rm(stateDirectory, { recursive: true, force: true });
  }
});

test("an upstream rejection surfaces the provider message as 502 and not as a stream", async () => {
  const stateDirectory = await temporaryState("xirai-assistant-upstream-error-");
  try {
    const { handlers } = await mounted("configureServer", {
      stateDirectory,
      upstreamRequest: stubUpstream({ status: 401, payload: { error: { message: "Authentication Fails" } } }),
    });
    await requestThrough(handlers, "/api/assistant/settings", {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: validSettings,
    });
    const failed = await requestThrough(handlers, "/api/assistant/chat", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: { history: [{ role: "user", content: "hi" }] },
    });
    assert.equal(failed.statusCode, 502);
    const payload = JSON.parse(failed.body);
    assert.equal(payload.status, 401);
    assert.equal(payload.error, "Authentication Fails");
    assert.ok(!failed.body.includes(SECRET));
  } finally {
    await rm(stateDirectory, { recursive: true, force: true });
  }
});

test("shipped personas load and a malformed file is skipped rather than failing the list", async () => {
  const { handlers } = await mounted("configureServer", { stateDirectory: await temporaryState("xirai-assistant-personas-") });
  const listed = await requestThrough(handlers, "/api/assistant/personas");
  assert.equal(listed.statusCode, 200);
  const { personas, available } = JSON.parse(listed.body);
  assert.equal(available, true);
  const ids = personas.map((persona) => persona.id).sort();
  assert.deepEqual(ids, ["anime-tagger", "photo-director", "prompt-architect"]);
  for (const persona of personas) {
    assert.ok(persona.systemPrompt.length > 0);
    assert.deepEqual(persona.knowledge, [], "knowledge retrieval is not implemented and must report empty");
  }

  const root = await temporaryState("xirai-assistant-persona-root-");
  try {
    await mkdir(path.join(root, "assistant", "personas"), { recursive: true });
    await writeFile(path.join(root, "assistant", "personas", "good.json"), JSON.stringify({ id: "good", name: "G", systemPrompt: "x" }));
    await writeFile(path.join(root, "assistant", "personas", "broken.json"), "{ not json");
    await writeFile(path.join(root, "assistant", "personas", "mismatch.json"), JSON.stringify({ id: "other", systemPrompt: "x" }));
    const mixed = await mounted("configureServer", { stateDirectory: root, projectRoot: root });
    const response = await requestThrough(mixed.handlers, "/api/assistant/personas");
    const body = JSON.parse(response.body);
    assert.deepEqual(body.personas.map((persona) => persona.id), ["good"]);
    assert.deepEqual(body.diagnostics.map((entry) => entry.code).sort(), ["persona_id_mismatch", "persona_not_json"]);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("profiles are listed, created, renamed, switched, duplicated and deleted", async () => {
  const stateDirectory = await temporaryState("xirai-assistant-profiles-");
  try {
    const { handlers } = await mounted("configureServer", { stateDirectory });
    const call = (target, options) => requestThrough(handlers, target, options);
    const post = (target, body) => call(target, { method: "POST", headers: { "Content-Type": "application/json" }, body: body || {} });

    // A client that has never saved anything still gets a profile to edit, not an empty page.
    const initial = JSON.parse((await call("/api/assistant/profiles")).body);
    assert.equal(initial.profiles.length, 1);
    assert.equal(initial.profiles[0].active, true);
    assert.equal(initial.activeId, initial.profiles[0].id);
    const first = initial.profiles[0].id;

    await call("/api/assistant/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: validSettings });

    // Creating is deliberately unvalidated: a profile is configured after it exists, so refusing it
    // for a missing API key would make "new profile" impossible.
    const created = await post("/api/assistant/profiles");
    assert.equal(created.statusCode, 201);
    const afterCreate = JSON.parse(created.body);
    assert.equal(afterCreate.profiles.length, 2);
    const second = afterCreate.createdId;
    assert.equal(afterCreate.activeId, second, "a new profile is the one you are about to configure");
    assert.equal(afterCreate.profiles.find((profile) => profile.id === second).settings.hasApiKey, false);

    // The decisive isolation property: saving a profile resolves the "keep the stored key"
    // shorthand against *its own* secret, never against whichever profile was active before.
    const inherited = await call(`/api/assistant/profiles/${second}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: { settings: { provider: "deepseek", baseUrl: "https://api.deepseek.com/v1", model: "deepseek-v4-pro", strength: 1 } },
    });
    assert.equal(inherited.statusCode, 400);
    assert.deepEqual(JSON.parse(inherited.body).errors.map((entry) => entry.code), ["api_key_missing"]);

    const renamed = await call(`/api/assistant/profiles/${second}`, {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: { name: "本地 Ollama" },
    });
    assert.equal(renamed.statusCode, 200);
    const named = JSON.parse(renamed.body).profiles.find((profile) => profile.id === second);
    assert.equal(named.name, "本地 Ollama");
    assert.equal(named.settings.model, "deepseek-v4-pro", "a rename must not disturb the configuration");

    // Switching is what makes the list a chooser rather than a filing cabinet.
    const switched = JSON.parse((await post(`/api/assistant/profiles/${first}/activate`)).body);
    assert.equal(switched.activeId, first);
    assert.equal(switched.profiles.find((profile) => profile.id === first).settings.hasApiKey, true);

    const duplicated = JSON.parse((await post(`/api/assistant/profiles/${first}/duplicate`)).body);
    assert.equal(duplicated.profiles.length, 3);
    const copy = duplicated.profiles.find((profile) => profile.id === duplicated.createdId);
    assert.equal(copy.settings.hasApiKey, true, "a copy without the key would look broken with no way to explain why");
    assert.equal(copy.settings.apiKeyHint, "••••9911");

    const removed = await call(`/api/assistant/profiles/${second}`, { method: "DELETE" });
    assert.equal(removed.statusCode, 200);
    assert.equal(JSON.parse(removed.body).profiles.length, 2);
    assert.equal(JSON.parse(removed.body).deleted, second);

    // Nothing in any of those replies may carry a key, and the store on disk still holds two.
    for (const body of [initial, afterCreate, switched, duplicated]) {
      assert.ok(!JSON.stringify(body).includes(SECRET), "a profile reply must never echo a key");
    }
    const onDisk = JSON.parse(await readFile(path.join(stateDirectory, "assistant-settings.json"), "utf8"));
    assert.equal(onDisk.schemaVersion, 2);
    assert.equal(onDisk.profiles.length, 2);
    assert.equal(onDisk.profiles.filter((profile) => profile.settings.apiKey === SECRET).length, 2);
  } finally {
    await rm(stateDirectory, { recursive: true, force: true });
  }
});

test("the last profile cannot be deleted and an unknown one is a 404, never a silent create", async () => {
  const stateDirectory = await temporaryState("xirai-assistant-profile-guards-");
  try {
    const { handlers } = await mounted("configureServer", { stateDirectory });
    const call = (target, options) => requestThrough(handlers, target, options);
    const only = JSON.parse((await call("/api/assistant/profiles")).body).profiles[0].id;

    // An empty store would leave the chat surface with no configuration to read and the settings
    // page with no row to edit.
    const refused = await call(`/api/assistant/profiles/${only}`, { method: "DELETE" });
    assert.equal(refused.statusCode, 409);
    assert.equal(JSON.parse((await call("/api/assistant/profiles")).body).profiles.length, 1);

    for (const candidate of ["..", "%2e%2e", "not-a-uuid", "ffffffff-ffff-4fff-8fff-ffffffffffff", "..%2F..%2Fui-state"]) {
      for (const [method, body] of [["PUT", { name: "x" }], ["DELETE", undefined]]) {
        const response = await call(`/api/assistant/profiles/${candidate}`, {
          method, headers: { "Content-Type": "application/json" }, ...(body ? { body } : {}),
        });
        assert.ok(response.statusCode >= 400, `${method} ${candidate} must be refused, got ${response.statusCode}`);
      }
      const activated = await call(`/api/assistant/profiles/${candidate}/activate`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: {},
      });
      assert.ok(activated.statusCode >= 400, `activating ${candidate} must be refused`);
    }
    await assert.rejects(readFile(path.join(stateDirectory, "ui-state.json"), "utf8"));

    // A PUT that asks for nothing is a client bug, not a no-op success.
    const empty = await call(`/api/assistant/profiles/${only}`, {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: {},
    });
    assert.equal(empty.statusCode, 400);
  } finally {
    await rm(stateDirectory, { recursive: true, force: true });
  }
});

test("switching profile switches the service the chat actually uses", async () => {
  const stateDirectory = await temporaryState("xirai-assistant-profile-live-");
  try {
    const capture = {};
    const { handlers } = await mounted("configureServer", {
      stateDirectory,
      upstreamRequest: stubUpstream({ payload: "data: [DONE]\n\n", capture }),
    });
    const call = (target, options) => requestThrough(handlers, target, options);
    const put = (target, body) => call(target, { method: "PUT", headers: { "Content-Type": "application/json" }, body });
    const post = (target, body) => call(target, { method: "POST", headers: { "Content-Type": "application/json" }, body: body || {} });

    await put("/api/assistant/settings", validSettings);
    const cloud = JSON.parse((await call("/api/assistant/profiles")).body).activeId;

    const local = JSON.parse((await post("/api/assistant/profiles")).body).createdId;
    await put(`/api/assistant/profiles/${local}`, {
      name: "本地",
      settings: { provider: "ollama", baseUrl: "http://localhost:11434/v1", apiKey: "", model: "qwen3:8b", strength: 0.8 },
    });

    await post("/api/assistant/chat", { history: [{ role: "user", content: "hi" }] });
    assert.equal(capture.settings.model, "qwen3:8b");
    assert.equal(capture.settings.apiKey, "", "a key-free provider must not inherit another profile's credential");

    await post(`/api/assistant/profiles/${cloud}/activate`);
    await post("/api/assistant/chat", { history: [{ role: "user", content: "hi" }] });
    assert.equal(capture.settings.model, "deepseek-v4-pro");
    assert.equal(capture.settings.apiKey, SECRET);

    // Saving through the compatibility route edits the live profile and leaves the other alone.
    await put("/api/assistant/settings", { ...validSettings, apiKey: undefined, model: "deepseek-v4-flash" });
    const stored = JSON.parse((await call("/api/assistant/profiles")).body).profiles;
    assert.equal(stored.find((profile) => profile.id === cloud).settings.model, "deepseek-v4-flash");
    assert.equal(stored.find((profile) => profile.id === local).settings.model, "qwen3:8b");
    assert.equal(stored.find((profile) => profile.id === local).name, "本地");
  } finally {
    await rm(stateDirectory, { recursive: true, force: true });
  }
});

test("characters are created, edited, copied and deleted while the shipped ones stay read-only", async () => {
  const stateDirectory = await temporaryState("xirai-assistant-characters-");
  try {
    const { handlers } = await mounted("configureServer", { stateDirectory });
    const call = (target, options) => requestThrough(handlers, target, options);
    const send = (target, method, body) => call(target, { method, headers: { "Content-Type": "application/json" }, body: body || {} });

    const shipped = JSON.parse((await call("/api/assistant/personas")).body).personas;
    assert.deepEqual(shipped.map((persona) => persona.id).sort(), ["anime-tagger", "photo-director", "prompt-architect"]);
    assert.ok(shipped.every((persona) => persona.builtIn === true), "a shipped character must announce that it cannot be edited");

    const created = await send("/api/assistant/personas", "POST", {
      name: "赛博朋克摄影师",
      description: "偏爱霓虹夜景的摄影指导",
      traits: { tone: "professional", detail: "balanced", language: "zh", initiative: "ask-first" },
      specialties: ["电影感人像"],
      rules: ["先给提示词，再说明改动重点"],
      systemPrompt: "你是一位擅长霓虹夜景的摄影指导。",
      starters: ["帮我写一段雨后街道的提示词"],
    });
    assert.equal(created.statusCode, 201);
    const authored = JSON.parse(created.body);
    assert.match(authored.createdId, UUID, "the server owns id generation, because the id becomes a filename");
    assert.equal(authored.persona.builtIn, false);
    assert.equal(authored.personas.length, 4);
    assert.equal(authored.personas[0].id, authored.createdId, "authored characters sort above the built-ins");

    // Stored as plain JSON in the state directory, never in the shipped asset folder — an app
    // update replaces that folder wholesale and would take the user's work with it.
    const onDisk = JSON.parse(await readFile(path.join(stateDirectory, "assistant-personas", `${authored.createdId}.json`), "utf8"));
    assert.equal(onDisk.name, "赛博朋克摄影师");
    assert.deepEqual(onDisk.rules, ["先给提示词，再说明改动重点"]);
    assert.equal(onDisk.builtIn, undefined, "where a record came from is decided by which folder holds it, not by a stored flag");
    assert.deepEqual(onDisk.knowledge, []);

    const edited = await send(`/api/assistant/personas/${authored.createdId}`, "PUT", {
      ...authored.persona,
      name: "夜景摄影师",
      rules: ["先给提示词，再说明改动重点", "不要堆砌无意义的画质词"],
    });
    assert.equal(edited.statusCode, 200);
    assert.equal(JSON.parse(edited.body).persona.name, "夜景摄影师");
    assert.equal(JSON.parse(edited.body).persona.rules.length, 2);

    // Copying is the only way to start from a built-in, and the copy is a fully editable record.
    const copied = JSON.parse((await send("/api/assistant/personas", "POST", { fromId: "prompt-architect" })).body);
    const copy = copied.personas.find((persona) => persona.id === copied.createdId);
    assert.equal(copy.builtIn, false);
    assert.equal(copy.name, "提示词架构师", "a copy keeps the name it was made from");
    assert.ok(copy.systemPrompt.includes("权重语法"), "a copy carries the whole character, not just its name");

    const removed = await call(`/api/assistant/personas/${authored.createdId}`, { method: "DELETE" });
    assert.equal(removed.statusCode, 200);
    assert.equal(JSON.parse(removed.body).personas.length, 4);
    assert.equal(JSON.parse(removed.body).deleted, authored.createdId);
  } finally {
    await rm(stateDirectory, { recursive: true, force: true });
  }
});

test("a shipped character cannot be edited or deleted, and an invalid one is never written", async () => {
  const stateDirectory = await temporaryState("xirai-assistant-character-guards-");
  try {
    const { handlers } = await mounted("configureServer", { stateDirectory });
    const call = (target, options) => requestThrough(handlers, target, options);
    const send = (target, method, body) => call(target, { method, headers: { "Content-Type": "application/json" }, body: body || {} });

    // 409 rather than 404: the character exists, and "copy it first" is only useful advice when it
    // does. An update replaces the shipped folder wholesale, so an edit there would be lost.
    // DELETE carries no body — `http.request` frames neither `Content-Length` nor chunking for it,
    // so bytes written on a DELETE would be parsed as the start of a second request.
    for (const [method, body] of [["PUT", { name: "改名" }], ["DELETE", undefined]]) {
      const refused = await call("/api/assistant/personas/prompt-architect", {
        method, headers: { "Content-Type": "application/json" }, ...(body ? { body } : {}),
      });
      assert.equal(refused.statusCode, 409, `${method} on a built-in must be refused`);
      assert.equal(JSON.parse(refused.body).errors?.[0]?.code, "persona_readonly");
    }

    // A character that would send nothing but the appended protocol is refused: selecting it would
    // look like the choice had no effect.
    const empty = await send("/api/assistant/personas", "POST", { name: "空角色" });
    assert.equal(empty.statusCode, 400);
    assert.deepEqual(JSON.parse(empty.body).errors.map((entry) => entry.code), ["persona_empty"]);

    // Restating the output protocol would break the apply button, which is the whole point of it.
    const conflicting = await send("/api/assistant/personas", "POST", { name: "越权", systemPrompt: "请用 ```prompt 围栏输出。" });
    assert.equal(conflicting.statusCode, 400);
    assert.ok(JSON.parse(conflicting.body).errors.some((entry) => entry.code === "persona_protocol_conflict"));

    for (const candidate of ["..", "%2e%2e", "prompt-architect", "3f4a2b1c-1111-2222-3333-44445555666", "..%2F..%2Fui-state"]) {
      for (const [method, body] of [["PUT", { name: "x", systemPrompt: "y" }], ["DELETE", undefined]]) {
        const response = await call(`/api/assistant/personas/${candidate}`, {
          method, headers: { "Content-Type": "application/json" }, ...(body ? { body } : {}),
        });
        assert.ok(response.statusCode >= 400, `${method} ${candidate} must be refused, got ${response.statusCode}`);
      }
    }
    await assert.rejects(readFile(path.join(stateDirectory, "ui-state.json"), "utf8"));
    // Nothing above was valid, so nothing may have been stored.
    await assert.rejects(readdir(path.join(stateDirectory, "assistant-personas")));
  } finally {
    await rm(stateDirectory, { recursive: true, force: true });
  }
});

test("a character's attributes and rules reach the provider, and its description does not", async () => {
  const stateDirectory = await temporaryState("xirai-assistant-character-chat-");
  try {
    const capture = {};
    const { handlers } = await mounted("configureServer", {
      stateDirectory,
      upstreamRequest: stubUpstream({ payload: "data: [DONE]\n\n", capture }),
    });
    const call = (target, options) => requestThrough(handlers, target, options);
    const send = (target, method, body) => call(target, { method, headers: { "Content-Type": "application/json" }, body: body || {} });

    const created = JSON.parse((await send("/api/assistant/personas", "POST", {
      name: "夜景摄影师",
      description: "只在角色库里显示的一句话",
      traits: { tone: "blunt", language: "zh" },
      specialties: ["霓虹夜景"],
      rules: ["先给提示词"],
      systemPrompt: "你是一位摄影指导。",
    })).body);

    await send("/api/assistant/settings", "PUT", { ...validSettings, personaId: created.createdId });
    await send("/api/assistant/chat", "POST", { history: [{ role: "user", content: "hi" }] });

    const system = capture.messages[0].content;
    assert.equal(capture.messages[0].role, "system");
    assert.ok(system.includes("你现在扮演「夜景摄影师」"), "the identity block must be sent");
    assert.ok(system.includes("语气直接"), "a chosen attribute must be sent as its sentence");
    assert.ok(system.includes("【擅长领域】霓虹夜景"));
    assert.ok(system.includes("1. 先给提示词"));
    assert.ok(system.includes("你是一位摄影指导。"));
    assert.ok(!system.includes("只在角色库里显示的一句话"), "the description is picker metadata and must never be sent");
    assert.ok(system.includes("```prompt"), "the output protocol is still appended to every character");

    // A character deleted while it was selected resolves to nothing rather than failing the chat.
    await call(`/api/assistant/personas/${created.createdId}`, { method: "DELETE" });
    await send("/api/assistant/chat", "POST", { history: [{ role: "user", content: "hi" }] });
    assert.ok(!capture.messages[0].content.includes("夜景摄影师"));
    assert.ok(capture.messages[0].content.includes("```prompt"), "the protocol survives a missing character");
  } finally {
    await rm(stateDirectory, { recursive: true, force: true });
  }
});

test("a malformed authored character is skipped rather than hiding the rest of the library", async () => {
  const stateDirectory = await temporaryState("xirai-assistant-character-corrupt-");
  try {
    const directory = path.join(stateDirectory, "assistant-personas");
    await mkdir(directory, { recursive: true });
    const good = "3f4a2b1c-1111-2222-3333-444455556666";
    await writeFile(path.join(directory, `${good}.json`), JSON.stringify({ id: good, name: "可用", systemPrompt: "你是…" }));
    await writeFile(path.join(directory, "3f4a2b1c-2222-3333-4444-555566667777.json"), "{ not json");
    // The filename is the id, so a record claiming a different one is a mismatch, not a rename.
    await writeFile(path.join(directory, "3f4a2b1c-3333-4444-5555-666677778888.json"), JSON.stringify({ id: good, name: "冒名", systemPrompt: "x" }));
    // A slug id in the user folder would collide with the shipped namespace.
    await writeFile(path.join(directory, "prompt-architect.json"), JSON.stringify({ id: "prompt-architect", name: "假冒", systemPrompt: "x" }));

    const { handlers } = await mounted("configureServer", { stateDirectory });
    const body = JSON.parse((await requestThrough(handlers, "/api/assistant/personas")).body);
    assert.deepEqual(body.personas.filter((persona) => !persona.builtIn).map((persona) => persona.name), ["可用"]);
    assert.equal(body.personas.length, 4, "the three shipped characters are unaffected");
    assert.ok(body.diagnostics.length >= 3);
    assert.ok(body.diagnostics.every((entry) => !entry.file?.includes(stateDirectory)), "a diagnostic must not carry an absolute path");
  } finally {
    await rm(stateDirectory, { recursive: true, force: true });
  }
});

test("unrelated paths fall through to the next middleware", async () => {
  const { handlers, downstream } = await mounted("configureServer", { stateDirectory: await temporaryState("xirai-assistant-passthrough-") });
  const response = await requestThrough(handlers, "/api/ui-state");
  assert.equal(response.statusCode, 204);
  assert.equal(downstream(), 1);
});

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

test("sessions are created, listed, reopened and deleted", async () => {
  const stateDirectory = await temporaryState("xirai-assistant-sessions-");
  try {
    const { handlers } = await mounted("configureServer", { stateDirectory });
    const call = (target, options) => requestThrough(handlers, target, options);

    assert.deepEqual(JSON.parse((await call("/api/assistant/sessions")).body).sessions, []);

    const created = await call("/api/assistant/sessions", { method: "POST", headers: { "Content-Type": "application/json" }, body: {} });
    assert.equal(created.statusCode, 201);
    const session = JSON.parse(created.body).session;
    assert.match(session.id, UUID, "the server owns id generation");
    assert.equal(session.title, "新对话");

    const saved = await call(`/api/assistant/sessions/${session.id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: { messages: [{ role: "user", content: "画一只猫" }, { role: "assistant", content: "好的" }] },
    });
    assert.equal(saved.statusCode, 200);
    // The title stops being a placeholder once the conversation has a first message.
    assert.equal(JSON.parse(saved.body).session.title, "画一只猫");

    const listed = JSON.parse((await call("/api/assistant/sessions")).body).sessions;
    assert.equal(listed.length, 1);
    assert.equal(listed[0].messageCount, 2);
    assert.equal(listed[0].messages, undefined, "listing carries metadata only");

    const reopened = JSON.parse((await call(`/api/assistant/sessions/${session.id}`)).body).session;
    assert.equal(reopened.messages.length, 2);
    assert.equal(reopened.createdAt, session.createdAt, "reopening must not reset the creation time");

    assert.equal((await call(`/api/assistant/sessions/${session.id}`, { method: "DELETE" })).statusCode, 200);
    assert.deepEqual(JSON.parse((await call("/api/assistant/sessions")).body).sessions, []);
    assert.equal((await call(`/api/assistant/sessions/${session.id}`)).statusCode, 404);
  } finally {
    await rm(stateDirectory, { recursive: true, force: true });
  }
});

test("a session id that is not a generated uuid never reaches the filesystem", async () => {
  const stateDirectory = await temporaryState("xirai-assistant-traversal-");
  try {
    const { handlers } = await mounted("configureServer", { stateDirectory });
    // Some of these are rejected by the id guard (400) and some never match the route at all
    // because URL parsing normalises the dot segments away (405). Either is acceptable; what must
    // never happen is a 2xx, or any filesystem effect.
    const hostile = ["..", "%2e%2e", "not-a-uuid", "3f4a2b1c-1111-2222-3333-44445555666", "..%2F..%2Fui-state"];
    for (const candidate of hostile) {
      for (const method of ["GET", "PUT", "DELETE"]) {
        const response = await requestThrough(handlers, `/api/assistant/sessions/${candidate}`, {
          method, headers: { "Content-Type": "application/json" }, ...(method === "PUT" ? { body: { messages: [] } } : {}),
        });
        assert.ok(response.statusCode >= 400, `${method} ${candidate} must be refused, got ${response.statusCode}`);
      }
    }
    // The decisive assertion: nothing was created, read into existence, or removed.
    await assert.rejects(readFile(path.join(stateDirectory, "ui-state.json"), "utf8"));
    assert.deepEqual(JSON.parse((await requestThrough(handlers, "/api/assistant/sessions")).body).sessions, []);
  } finally {
    await rm(stateDirectory, { recursive: true, force: true });
  }
});

test("saving an unknown session is a 404 rather than a silent create", async () => {
  const stateDirectory = await temporaryState("xirai-assistant-missing-");
  try {
    const { handlers } = await mounted("configureServer", { stateDirectory });
    const response = await requestThrough(handlers, "/api/assistant/sessions/3f4a2b1c-1111-2222-3333-444455556666", {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: { messages: [] },
    });
    assert.equal(response.statusCode, 404);
  } finally {
    await rm(stateDirectory, { recursive: true, force: true });
  }
});

test("a corrupt session file opens empty instead of making its slot unreachable", async () => {
  const stateDirectory = await temporaryState("xirai-assistant-corrupt-");
  try {
    const id = "3f4a2b1c-1111-2222-3333-444455556666";
    await mkdir(path.join(stateDirectory, "assistant-sessions"), { recursive: true });
    await writeFile(path.join(stateDirectory, "assistant-sessions", `${id}.json`), "{ not json");
    const { handlers } = await mounted("configureServer", { stateDirectory });
    const response = await requestThrough(handlers, `/api/assistant/sessions/${id}`);
    assert.equal(response.statusCode, 200);
    assert.deepEqual(JSON.parse(response.body).session.messages, []);
    // It still lists, so the user can delete it from the picker.
    assert.equal(JSON.parse((await requestThrough(handlers, "/api/assistant/sessions")).body).sessions.length, 1);
  } finally {
    await rm(stateDirectory, { recursive: true, force: true });
  }
});

test("the model list is fetched from the configured service and needs no model name", async () => {
  const stateDirectory = await temporaryState("xirai-assistant-models-");
  try {
    const { handlers } = await mounted("configureServer", { stateDirectory });
    // Before configuration the route reports the blocking fields rather than calling out.
    const blocked = await requestThrough(handlers, "/api/assistant/models", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: {},
    });
    assert.equal(blocked.statusCode, 400);
    assert.deepEqual(JSON.parse(blocked.body).errors.map((entry) => entry.field), ["apiKey"],
      "only endpoint and credential problems may block a listing; the model is the thing being looked up");
  } finally {
    await rm(stateDirectory, { recursive: true, force: true });
  }
});
