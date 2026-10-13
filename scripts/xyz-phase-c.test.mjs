import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { xyzApi } from "../src/xyz-client.js";
import { createXYZDraft, xyzLocalSummary } from "../src/xyz-draft.js";

const baseline = { parameters: { engine: "SD", prompt: "cat", negative_prompt: "", checkpoint: "m.safetensors", steps: 20, cfg: 6, width: 512, height: 512,
  seed: "1", sampler: "euler", scheduler: "normal", guidance: "none", pag: { scale: 0.3, applied_layers: "mid" }, postprocess_chain: [] },
  seed_mode: "fixed", prompt_without_groups: "cat", group_prompt: "", group_snapshots: [], loras: [], library_revision: 0 };

test("draft: concurrency 1 by default, resource grouping off, 1–10 validated without coercion", () => {
  const draft = createXYZDraft(baseline);
  assert.equal(draft.requested_concurrency, 1);
  assert.equal(draft.group_by_resources, false);
  draft.axes.x = { field: "seed", values: [{ value: "1" }, { value: "2" }] };
  draft.requested_concurrency = 4;
  assert.equal(xyzLocalSummary(draft).total, 2, "concurrency never multiplies the cell count");
  draft.requested_concurrency = 11;
  assert.match(xyzLocalSummary(draft).error, /1–10/);
  assert.equal(draft.requested_concurrency, 11, "the draft keeps what was typed");
});

test("client: continuation carries a new limit only when it is an integer", async () => {
  const bodies = [];
  const api = xyzApi(async (_path, options) => { bodies.push(JSON.parse(options.body)); return { ok: true, json: async () => ({}) }; });
  await api.continue("e", "none", { requestedConcurrency: 4 });
  await api.continue("e", "failed");
  await api.continue("e", "none", { requestedConcurrency: "4" });
  assert.deepEqual(bodies, [{ retry: "none", requested_concurrency: 4 }, { retry: "failed" }, { retry: "none" }]);
});

test("workspace: server maximum bounds the controls; batches, reasons and per-cell batch facts are shown", async () => {
  const source = await readFile(new URL("../src/XYZPlotWorkspace.jsx", import.meta.url), "utf8");
  assert.match(source, /XYZCountField label="请求并发"[^\n]*max=\{maxConcurrency\}/);
  const controls = await readFile(new URL("../src/SamplingControls.jsx", import.meta.url), "utf8");
  assert.match(controls, /value >= max/);
  assert.match(controls, /onChange\(\/\^\\d\+\$\//);
  assert.match(source, /group_by_resources: event\.target\.checked/);
  assert.match(source, /preview\.batches\.count/);
  assert.match(source, /detail\.batch_log/);
  assert.match(source, /批 \$\{cell\.batch\.index \+ 1\}\/\$\{cell\.batch\.size\}/);
  assert.match(source, /api\.continue\(experimentId, value, \{ requestedConcurrency: limit \}\)/);
  assert.doesNotMatch(source, /阶段 C 的真实批处理尚未实现/);
});
