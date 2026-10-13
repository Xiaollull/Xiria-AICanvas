import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import {
  initialXYZAxisValue, runtimeGuidanceHealthStatus, xyzAxisRuntimeIssues, xyzGuidanceOption, xyzGuidanceRuntimeStatus, xyzRuntimeGuidanceForEngine,
} from "../src/xyz-axis-editor.js";

const health = {
  // Publisher shape: SD/iL intentionally have no health.engines entries; their implementation map is authoritative.
  engines: { Anima: { features: { pag: true } } },
  guidance: {
    pag: { available: true, engines: ["SD", "iL", "Anima"], implementations: {
      SD: "diffusers_pag", iL: "diffusers_pag", Anima: "native_cosmos_identity_self_attention",
    } },
    cfg_zero_star: { available: true, engines: ["Anima", "Krea2"] },
  },
};

test("runtime guidance maps only explicit current-engine capability as available", () => {
  const sd = xyzRuntimeGuidanceForEngine(health, "SD");
  const ill = xyzRuntimeGuidanceForEngine(health, "iL");
  const anima = xyzRuntimeGuidanceForEngine(health, "Anima");
  assert.equal(xyzGuidanceRuntimeStatus(sd, "SD", "pag").status, "available");
  assert.equal(xyzGuidanceRuntimeStatus(ill, "iL", "pag").status, "available");
  assert.equal(runtimeGuidanceHealthStatus(health, "SD", "pag").status, "available", "Basic and XYZ share the published SD contract");
  assert.equal(runtimeGuidanceHealthStatus(health, "iL", "pag").status, "available", "Basic and XYZ share the published iL contract");
  assert.equal(xyzRuntimeGuidanceForEngine({ ...health, engines: {} }, "Anima").modes.pag.status, "available", "the recognized native implementation is sufficient when no engine feature row exists");
  assert.equal(xyzGuidanceRuntimeStatus(sd, "SD", "cfg_zero_star").status, "unavailable");
  assert.equal(xyzGuidanceRuntimeStatus(anima, "Anima", "cfg_zero_star").status, "available");
  assert.equal(xyzGuidanceRuntimeStatus(xyzRuntimeGuidanceForEngine({ guidance: { pag: { available: true, engines: ["Anima"] } }, engines: { Anima: { features: { pag: true } } } }, "Anima"), "Anima", "cfg_zero_star").status, "unknown");
  assert.equal(xyzRuntimeGuidanceForEngine({ ...health, engines: { ...health.engines, iL: { features: { pag: false } } } }, "iL").modes.pag.status, "unavailable", "explicit false overrides implementation evidence");
  assert.equal(xyzRuntimeGuidanceForEngine({ ...health, engines: { Anima: { features: { pag: false } } }, guidance: { ...health.guidance, pag: { ...health.guidance.pag, implementations: { ...health.guidance.pag.implementations, Anima: "native_cosmos_identity_self_attention" } } } }, "Anima").modes.pag.status, "unavailable");
  assert.equal(xyzGuidanceRuntimeStatus(xyzRuntimeGuidanceForEngine({ guidance: { pag: { available: true, engines: ["SD", "iL"] } } }, "SD"), "SD", "pag").status, "unknown", "missing implementation/feature proof stays unknown");
  assert.equal(xyzGuidanceRuntimeStatus(xyzRuntimeGuidanceForEngine({ guidance: { pag: { available: false, engines: [] } } }, "SD"), "SD", "pag").status, "unavailable");
  assert.equal(xyzGuidanceRuntimeStatus(xyzRuntimeGuidanceForEngine(null, "SD"), "SD", "pag").status, "unknown");
  assert.equal(xyzGuidanceRuntimeStatus(xyzRuntimeGuidanceForEngine({ guidance: { pag: { available: true, engines: ["SD"] } }, engines: {} }, "SD"), "SD", "pag").status, "unknown");
  assert.equal(xyzGuidanceRuntimeStatus(sd, "iL", "pag").status, "unknown", "an old engine-tagged snapshot cannot authorize the newly selected engine");
  const blocked = xyzGuidanceOption(xyzRuntimeGuidanceForEngine(null, "SD"), "SD", "pag", "PAG");
  assert.equal(blocked.disabled, true);
  assert.match(blocked.disabledReason, /能力尚未提供/);
});

test("saved runtime-unsupported guidance and PAG axes remain byte-for-byte present and block plan admission", () => {
  const runtime = xyzRuntimeGuidanceForEngine({
    engines: { SD: { features: { pag: false } } },
    guidance: { pag: { available: false, engines: [] }, cfg_zero_star: { available: true, engines: ["Anima"] } },
  }, "SD");
  const draft = { engine: "SD", axes: {
    x: { field: "guidance", values: [{ value: "pag", label: "saved mode" }], range: null },
    y: { field: "pag.scale", values: [{ value: "0.8", label: "saved scale" }], range: null },
    z: null,
  } };
  const before = structuredClone(draft);
  const issues = xyzAxisRuntimeIssues(draft, runtime);
  assert.equal(issues.length, 2);
  assert.match(issues[0], /X 轴第 1 项.*PAG.*不可用/);
  assert.match(issues[1], /Y 轴 PAG 强度.*不可用/);
  assert.deepEqual(draft, before);
  assert.deepEqual(xyzAxisRuntimeIssues({ engine: "SD", axes: { x: { field: "cfg", values: [{ value: "6" }] }, y: null, z: null } }, runtime), []);
  assert.deepEqual(xyzAxisRuntimeIssues({ engine: "SD", axes: { x: { field: "guidance", values: [{ value: "none" }] }, y: null, z: null } }, runtime), []);
  const invalidSavedMode = { engine: "SD", axes: { x: { field: "guidance", values: [{ value: "future-guidance" }] }, y: null, z: null } };
  assert.match(xyzAxisRuntimeIssues(invalidSavedMode, runtime)[0], /无法识别；原值保留/);
});

test("null baseline PAG still seeds an empty scale, and live App/workspace/editor wiring is engine-scoped", async () => {
  assert.equal(initialXYZAxisValue("pag.scale", { baseline: { parameters: { pag: null } } }), "");
  const [app, workspace, editor] = await Promise.all([
    readFile(new URL("../src/App.jsx", import.meta.url), "utf8"),
    readFile(new URL("../src/XYZPlotWorkspace.jsx", import.meta.url), "utf8"),
    readFile(new URL("../src/XYZAxisEditor.jsx", import.meta.url), "utf8"),
  ]);
  assert.match(app, /xyzRuntimeGuidanceForEngine\(inferenceHealth, activeXYZEngine\)/);
  assert.match(app, /runtimeGuidanceHealthStatus\(health, engine, "pag"\)\.status === "available"/);
  assert.match(app, /runtimeGuidance=\{xyzRuntimeGuidance\}/);
  assert.match(workspace, /runtimeGuidance=\{runtimeGuidance\}/);
  assert.match(workspace, /axisRuntimeIssues\.length > 0/);
  assert.match(workspace, /disabled=\{planBusy \|\| !summary\.valid \|\| baselineIssues\.length > 0 \|\| Boolean\(concurrencyError\) \|\| submitBusy \|\| axisRuntimeIssues\.length > 0\}/);
  assert.match(editor, /disabledReason: `\$\{pagIssue\}/);
  assert.match(editor, /当前运行时不可用/);
});
