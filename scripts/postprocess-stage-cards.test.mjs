import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

const source = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("both generate pages draw the chain through one component and their own card chrome", async () => {
  const [app, page, panel] = await Promise.all([
    source("src/App.jsx"), source("src/ImageToImagePage.jsx"), source("src/ResampleChainPanel.jsx"),
  ]);

  // One editor, one state shape: the two pages differ only in the card they wrap a stage in.
  for (const [name, text] of [["App", app], ["ImageToImagePage", page]]) {
    assert.match(text, /<ResampleChainPanel/, `${name} must render the shared chain panel`);
    assert.match(text, /renderCard=\{/, `${name} must supply its own card chrome`);
    assert.match(text, /renderStage=\{\(stage, index, actions\) => \(stage\.type === "adetailer" \? \(/, `${name} keeps its ADetailer and RTX bodies`);
  }
  assert.match(app, /chain=\{postprocessChain\}[\s\S]*?onChange=\{setPostprocessChain\}/);
  assert.match(page, /chain=\{config\.postprocessChain\}[\s\S]*?onChange=\{setChain\}/);

  // The order lives in the list itself, so neither page carries a second list of stage names. The
  // one surviving mention of the old field is the migration that reads it off a saved workspace.
  for (const [name, text] of [["App", app], ["ImageToImagePage", page]]) {
    assert.doesNotMatch(text, /setPostprocessOrder|normalizePostprocessOrder|movePostprocessStage/, `${name} must not keep a separate order list`);
    assert.doesNotMatch(text, /className="postprocess-order"|className="i2i-order-card"/, name);
  }
  assert.match(app, /postprocessOrder: saved\.postprocessOrder \?\? saved\.postprocess_order/, "the old order is still read on the way in");
});

test("text-to-image postprocess stages reuse the image-to-image card contract", async () => {
  const [app, css] = await Promise.all([source("src/App.jsx"), source("src/styles.css")]);

  assert.match(app, /function PostprocessStageHeader\([\s\S]*?aria-expanded=\{expanded\}/);
  assert.match(app, /function PostprocessStageHeader\([\s\S]*?role="switch"/);
  assert.match(app, /<section className="postprocess-stage-stack" aria-labelledby="postprocess-stage-title">/);
  // ADetailer and RTX keep their own headers; a resample unit's comes from the panel.
  for (const title of ["ADetailer", "RTX VSR"]) {
    assert.match(app, new RegExp(`title="${title}"[\\s\\S]*?expanded=\\{(?:adetailer|rtx)\\.expanded\\}`));
  }
  assert.match(app, /onToggleEnabled=\{\(\) => \(engineAllowsHires \|\| enabled\) && onToggleEnabled\(\)\}/);
  assert.match(app, /onToggleEnabled=\{\(\) => \(engineAllowsADetailer \|\| adetailer\.enabled\) && updateADetailer/);
  assert.match(app, /onToggleEnabled=\{\(\) => \(engineAllowsRtx \|\| rtx\.enabled\) && updateRtx/);
  assert.match(app, /const hiresControlsLocked = modelSwitching;/);
  assert.match(app, /const unitLocked = adetailerLocked;/);
  assert.doesNotMatch(app, /const unitLocked = !adetailer\.enabled/);

  assert.match(css, /\.i2i-stage-card, \.postprocess-stage-card \{/);
  assert.match(css, /\.i2i-stage-head, \.postprocess-stage-head \{/);
  assert.match(css, /\.postprocess-stage-card > \.resample-parameters[^{]*\{[^}]*border-top: 1px solid var\(--line\)/);
  assert.match(css, /\.postprocess-stage-card\.expanded \.postprocess-stage-expand svg \{ transform: rotate\(180deg\); \}/);
  assert.doesNotMatch(css, /\.resample-parameters:not\(\.enabled\)|\.adetailer-parameters:not\(\.enabled\)|\.rtx-parameters:not\(\.enabled\)/);
});

test("a unit card carries its own ordering and deletion, and the add button is bounded", async () => {
  const panel = await source("src/ResampleChainPanel.jsx");
  // Keyboard-reachable buttons rather than a drag handle: the list has to be operable without a
  // pointer, and every control here is a real <button> with a label.
  assert.match(panel, /aria-label=\{`上移\$\{/);
  assert.match(panel, /aria-label=\{`下移\$\{/);
  assert.match(panel, /aria-label=\{`删除重采样 \$\{resampleUnitNumber\(chain, stage\)\}`\}/);
  assert.match(panel, /disabled=\{locked \|\| index === 0\}/);
  assert.match(panel, /disabled=\{locked \|\| index === chain\.length - 1\}/);
  assert.match(panel, /units\.length < RESAMPLE_UNIT_LIMIT/);
  // The one fact no single card can show: what the whole chain does to the picture.
  assert.match(panel, /chainSizeLine\(baseSize, chain, engine\)/);
});
