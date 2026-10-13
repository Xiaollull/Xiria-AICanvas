import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { xyzApi, xyzExportBody, xyzExportActive, formatBytes } from "../src/xyz-client.js";

test("export body: original is always 100 %, page scope carries the page, nothing else is invented", () => {
  const options = { scope: "all", mode: "original", percent: 37, theme: "light", arrangement: "vertical", captions: true };
  assert.deepEqual(xyzExportBody(options, { z: 2, sample: 1 }), { scope: "all", page: null, mode: "original", percent: 100, theme: "light", arrangement: "vertical", captions: true });
  assert.deepEqual(xyzExportBody({ ...options, scope: "page", mode: "scaled" }, { z: 2, sample: 1 }).page, { z_index: 2, sample_index: 1 });
  assert.equal(xyzExportBody({ ...options, mode: "scaled" }, null).percent, 37);
  assert.equal(xyzExportActive({ status: "running" }), true);
  assert.equal(xyzExportActive({ status: "interrupted" }), false);
  assert.equal(formatBytes(1536), "1.5 KB");
});

test("client: export routes are POST/GET on the experiment resource", async () => {
  const calls = [];
  const api = xyzApi(async (path, options) => { calls.push([options.method, path, options.body && JSON.parse(options.body)]); return { ok: true, json: async () => ({}) }; });
  await api.estimateExport("e", { mode: "scaled", percent: 50 });
  await api.createExport("e", { mode: "original" });
  await api.exportStatus("e", "x");
  await api.cancelExport("e", "x");
  assert.deepEqual(calls.map(([method, path]) => `${method} ${path}`), [
    "POST /api/inference/xyz-plots/e/export/estimate", "POST /api/inference/xyz-plots/e/exports",
    "GET /api/inference/xyz-plots/e/exports/x", "POST /api/inference/xyz-plots/e/exports/x/cancel"]);
});

test("workspace: dialog offers 1–100 % with 25/50/75/100 shortcuts, 50 % default for downscale, live budget and suggestions", async () => {
  const source = await readFile(new URL("../src/XYZPlotWorkspace.jsx", import.meta.url), "utf8");
  assert.match(source, /\[25, 50, 75, 100\]\.map/);
  assert.match(source, /percent: mode === "scaled" \? \(options\.mode === "scaled" \? options\.percent : 50\) : 100/);
  // QPlan2/P2: the dialog is shared with the search candidate grids; XYZ still defaults to its own export API.
  assert.match(source, /exportApi = api, extraBody = null/);
  assert.match(source, /exportApi\.estimateExport\(detail\.id/);
  assert.match(source, /estimate\.suggestions\.map/);
  assert.match(source, /不会自动缩小、分页或省略/);
  assert.match(source, /disabled=\{busy \|\| running \|\| !estimate \|\| !budget\?\.ok \|\| !percentValid\}/);
  assert.match(source, /useDialogLifecycle\(true, onClose\)/);
  assert.match(source, /cell\.thumbnail_url \|\| cell\.asset_url/);
  assert.match(source, /屏幕格大小（不影响导出）/);
});
