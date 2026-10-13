import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { xyzApi, xyzSubmissionKey } from "../src/xyz-client.js";
import { initialXYZAxisValue, initialXYZRange, parseXYZStructuredText, validXYZFiniteDecimal, validXYZSeed, xyzStructuredText } from "../src/xyz-axis-editor.js";

const baseline = { parameters: { engine: "SD", prompt: "cat", negative_prompt: "", steps: 20, cfg: 6, width: 512, height: 512, seed: "18446744073709551615", pag: null },
  loras: [], group_snapshots: [] };
const draft = { engine: "SD", baseline };

test("typed axis helpers keep uint64 strings exact, tolerate a missing PAG object, and reject non-finite decimals", () => {
  assert.equal(initialXYZAxisValue("pag.scale", draft), "");
  assert.deepEqual(initialXYZRange("pag.scale", draft), { start: "1", end: "1.2", step: "0.1" });
  assert.equal(initialXYZAxisValue("seed", draft), "18446744073709551615");
  assert.equal(validXYZSeed("18446744073709551615"), true);
  assert.equal(validXYZSeed("18446744073709551616"), false);
  assert.equal(validXYZSeed("-1"), false);
  assert.equal(validXYZFiniteDecimal("-0.125"), true);
  assert.equal(validXYZFiniteDecimal("Infinity"), false);
  assert.equal(validXYZFiniteDecimal(""), false);
});

test("advanced structured editor preserves malformed text directly instead of wrapping it", () => {
  const raw = '{"find":"unfinished';
  const parsed = parseXYZStructuredText(raw);
  assert.equal(parsed.valid, false);
  assert.equal(parsed.value, raw);
  assert.equal(xyzStructuredText(parsed.value), raw);
  assert.deepEqual(parseXYZStructuredText('{"find":"cat","replace":"fox","expected_matches":1}'), {
    valid: true, value: { find: "cat", replace: "fox", expected_matches: 1 },
  });
});

test("submission id key varies with actual error policy but is stable for an identical transport retry", () => {
  const draftKey = JSON.stringify({ engine: "SD", repeat_count: 2 });
  assert.equal(xyzSubmissionKey(draftKey, "pause"), xyzSubmissionKey(draftKey, "pause"));
  assert.notEqual(xyzSubmissionKey(draftKey, "pause"), xyzSubmissionKey(draftKey, "continue"));
  assert.notEqual(xyzSubmissionKey(draftKey, "pause"), xyzSubmissionKey(`${draftKey} `, "pause"));
});

test("submit transport keeps editor-only state out of the strict XYZ wire envelope", async () => {
  const requests = [];
  const api = xyzApi(async (path, options) => { requests.push({ path, options }); return { ok: true, json: async () => ({ experiment: { id: "fixture" } }) }; });
  const typedDraft = { schema_version: 1, kind: "xyz_draft", engine: "SD", axes: { x: { field: "prompt", values: [{ value: "red, blue" }] } },
    baseline, repeat_count: 1, requested_concurrency: 1, inherit_postprocess: false, group_by_resources: false };
  await api.submit(typedDraft, { submissionId: "one-id", onError: "continue" });
  const body = JSON.parse(requests[0].options.body);
  assert.deepEqual(Object.keys(body).sort(), ["draft", "kind", "on_error", "schema_version", "submission_id"]);
  assert.deepEqual(body.draft.axes.x.values[0].value, "red, blue");
  assert.equal(body.on_error, "continue");
  assert.equal(Object.hasOwn(body.draft, "editorState"), false);
  assert.equal(Object.hasOwn(body.draft.axes.x.values[0], "invalid_raw"), false);
});

test("workspace wiring keeps async snapshots/old experiments separate and adds only live baseline component choices", async () => {
  const [workspace, editor, app, styles] = await Promise.all([
    readFile(new URL("../src/XYZPlotWorkspace.jsx", import.meta.url), "utf8"),
    readFile(new URL("../src/XYZAxisEditor.jsx", import.meta.url), "utf8"),
    readFile(new URL("../src/App.jsx", import.meta.url), "utf8"),
    readFile(new URL("../src/styles.css", import.meta.url), "utf8"),
  ]);
  assert.match(workspace, /draftKeyRef\.current === submittedKey/);
  assert.match(workspace, /engineRef\.current === submittedEngine/);
  assert.match(workspace, /setSubmittedNotice\(\{ id: result\.experiment\.id, key: submittedKey \}\)/);
  assert.match(workspace, /detailRequest\.current/);
  assert.match(workspace, /\["pausing", "paused"\]\.includes\(detail\.status\)/);
  assert.match(workspace, /已提交实验是服务端冻结快照/);
  assert.match(workspace, /xyz-results-context/);
  assert.match(editor, /onBlur=\{commit\}/);
  assert.match(editor, /members\.map\(\(member, index\) =>/);
  assert.match(editor, /同时使用组合预设词（仅本方案前缀一次）/);
  assert.match(editor, /完全无 LoRA/);
  assert.match(editor, /输入冻结资产值/);
  assert.match(app, /componentChoices=\{xyzCatalog\.choices\}/);
  assert.match(styles, /\.xyz-axis-unused > summary/);
  assert.match(styles, /\.xyz-run-bar \{ position: sticky; bottom: 0/);
  assert.match(styles, /--accent-rgb/);
});
