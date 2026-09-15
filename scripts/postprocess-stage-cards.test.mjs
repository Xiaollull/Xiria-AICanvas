import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

const source = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("text-to-image postprocess stages reuse the image-to-image card contract", async () => {
  const [app, css] = await Promise.all([source("src/App.jsx"), source("src/styles.css")]);

  assert.match(app, /function PostprocessStageHeader\([\s\S]*?aria-expanded=\{expanded\}/);
  assert.match(app, /function PostprocessStageHeader\([\s\S]*?role="switch"/);
  assert.match(app, /<section className="postprocess-stage-stack" aria-labelledby="postprocess-stage-title">/);
  for (const title of ["Hires\.fix", "ADetailer", "RTX VSR"]) {
    assert.match(app, new RegExp(`title="${title}"[\\s\\S]*?expanded=\\{(?:hires|adetailer|rtx)\\.expanded\\}`));
  }
  assert.doesNotMatch(app, /parameter-title parameter-toggle \$\{(?:hires|adetailer|rtx)\.expanded/);
  assert.match(app, /onToggleEnabled=\{\(\) => \(engineAllowsHires \|\| hires\.enabled\) && updateHires/);
  assert.match(app, /onToggleEnabled=\{\(\) => \(engineAllowsADetailer \|\| adetailer\.enabled\) && updateADetailer/);
  assert.match(app, /onToggleEnabled=\{\(\) => \(engineAllowsRtx \|\| rtx\.enabled\) && updateRtx/);
  assert.match(app, /const hiresControlsLocked = modelSwitching;/);
  assert.match(app, /ariaLabel="超分模型" value=\{hires\.model\} disabled=\{modelSwitching \|\| compatibleUpscalers\.length === 0\}/);
  assert.match(app, /const unitLocked = adetailerLocked;/);
  assert.doesNotMatch(app, /const hiresControlsLocked = !hires\.enabled/);
  assert.doesNotMatch(app, /const unitLocked = !adetailer\.enabled/);

  assert.match(css, /\.i2i-stage-card, \.postprocess-stage-card \{/);
  assert.match(css, /\.i2i-stage-head, \.postprocess-stage-head \{/);
  assert.match(css, /\.postprocess-stage-card > \.hires-parameters, \.postprocess-stage-card > \.adetailer-parameters, \.postprocess-stage-card > \.rtx-parameters \{[^}]*border-top: 1px solid var\(--line\)/);
  assert.match(css, /\.postprocess-stage-card\.expanded \.postprocess-stage-expand svg \{ transform: rotate\(180deg\); \}/);
  assert.doesNotMatch(css, /\.hires-parameters:not\(\.enabled\)|\.adetailer-parameters:not\(\.enabled\)|\.rtx-parameters:not\(\.enabled\)/);
});
