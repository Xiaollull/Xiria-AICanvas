import test from "node:test";
import assert from "node:assert/strict";
import { deletionBody, deletionResult, removedExperiments } from "../src/task-deletion.js";
import { isTaskDeletionRequest, taskDeletionAdmission } from "./task-deletion-access.mjs";
import { xyzApi } from "../src/xyz-client.js";
import { loraMixApi } from "../src/lora-mix-client.js";
import { readFileSync } from "node:fs";

test("all is server-enumerated; selected is bounded and path-free", () => {
  const id = "a".repeat(32);
  assert.deepEqual(deletionBody("all", [id]), { scope: "all" });
  assert.deepEqual(deletionBody("selected", [id]), { scope: "selected", ids: [id] });
  for (const ids of [[], ["../oops"], [id, id], Array.from({ length: 201 }, (_, i) => i.toString(16).padStart(32, "0"))]) assert.throws(() => deletionBody("selected", ids));
});
test("partial receipt cannot claim whole success", () => {
  const value = { deleted: 1, skipped: 1, failed: 1, results: ["deleted", "skipped", "failed"].map((outcome, i) => ({ id: String(i).repeat(32), outcome, experiment_ids: [String(i).repeat(32)] })) };
  assert.equal(deletionResult(value), value);
  assert.deepEqual(removedExperiments(value), ["0".repeat(32)]);
  assert.deepEqual(removedExperiments({ results: [], visibilityRemoved: ["a".repeat(32)] }), ["a".repeat(32)]);
  assert.throws(() => deletionResult({ ...value, deleted: 3 }));
  assert.throws(() => deletionResult(null));
});
test("trusted local write admission including DNS rebinding and CSRF", () => {
  const request = { method: "POST", url: "/api/inference/xyz-plots/task-deletions", headers: { host: "localhost:9017", origin: "http://localhost:9017" }, socket: { remoteAddress: "127.0.0.1" } };
  assert.equal(isTaskDeletionRequest(request), true);
  assert.equal(taskDeletionAdmission(request).allowed, true);
  for (const changed of [{ socket: { remoteAddress: "192.168.1.4" } }, { headers: { host: "evil.test:9017", origin: "http://evil.test:9017" } }, { headers: { host: "localhost:9017" } }, { headers: { ...request.headers, origin: "http://evil.test:9017" } }]) assert.equal(taskDeletionAdmission({ ...request, ...changed }).allowed, false);
  assert.equal(taskDeletionAdmission({ ...request, method: "GET", url: "/api/inference/xyz-plots/tasks" }).allowed, true);
});
test("both clients use fixed module endpoints without current-engine all IDs", async () => {
  for (const [factory, module] of [[xyzApi, "xyz-plots"], [loraMixApi, "lora-mix"]]) {
    const calls = [], api = factory(async (url, options) => { calls.push([url, options]); return { ok: true, json: async () => ({}) }; });
    await api.tasks("Anima"); await api.previewTaskDeletion({ scope: "all" }); await api.deleteTasks({ scope: "all", fingerprint: "x" });
    assert.equal(calls[0][0], `/api/inference/${module}/tasks?engine=Anima`);
    assert.equal(calls[1][0], `/api/inference/${module}/task-deletions/preview`);
    assert.equal(calls[2][0], `/api/inference/${module}/task-deletions`);
    assert.deepEqual(JSON.parse(calls[2][1].body), { scope: "all", fingerprint: "x" });
  }
});
test("the real Node gate protects deletion before identity admission; proof is still server-minted", () => {
  const source = readFileSync(new URL("../vite.config.js", import.meta.url), "utf8");
  const guard = source.indexOf("const taskAdmission = taskDeletionAdmission(request);");
  assert.ok(guard > 0 && source.indexOf("const health = await probeBackendForGate();", guard) > guard);
  assert.equal((source.match(/forward\.removeHeader\("x-xirai-wd14-proxy-proof"\)/g) || []).length, 2);
  assert.equal((source.match(/isTaskDeletionRequest\(request\) \|\|/g) || []).length, 2);
});
