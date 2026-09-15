import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const read = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("image workspace has responsive layout contracts", async () => {
  const css = await read("src/styles.css");
  // One base rule, not two: the grid was declared twice, and the second copy only won because it
  // came later — the sort of agreement that survives until someone reorders the file.
  const bases = [...css.matchAll(/^\.image-workspace \{/gm)];
  assert.equal(bases.length, 1, "the grid is defined once");
  // The fixed parameter rail lives on the left; row 2 is the scrollport the rail and the stage use
  // independently. Image-to-image deliberately has no resize seam or collapsed state.
  assert.match(css, /^\.image-workspace \{[^}]*display: grid;[^}]*grid-template-columns: minmax\(360px, 400px\) minmax\(0, 1fr\);[^}]*grid-template-rows: auto minmax\(0, 1fr\)/m);
  assert.match(css, /@media \(max-width: 1279px\)[\s\S]*?minmax\(360px, 400px\) minmax\(0, 1fr\)/);
  assert.match(css, /@media \(max-width: 959px\)[\s\S]*?\.image-workspace \{ display: block/);
  assert.match(css, /@media \(max-width: 640px\)[\s\S]*?overflow-x: auto/);
});

test("each column scrolls on its own, and neither carries the other", async () => {
  const css = await read("src/styles.css");
  // The stage holds a full-size prompt and two square previews; the rail holds a long parameter
  // stack. Both overflow, and reading one has no business moving the other — so the workspace is
  // pinned to the viewport and each column is its own scrollport.
  const workspace = css.match(/^\.image-workspace \{[^}]*\}/m)[0];
  assert.match(workspace, /height: calc\(100dvh - 62px\)/);
  assert.match(workspace, /overflow: hidden/, "the workspace itself must not scroll");
  assert.match(workspace, /grid-template-rows: auto minmax\(0, 1fr\)/, "row 2 needs a definite height to scroll within");
  assert.match(css, /\.i2i-stage-panel \{[^}]*overflow: auto;[^}]*overscroll-behavior: contain/);
  assert.match(css, /\.i2i-controls-panel \.panel-scroll \{[^}]*overflow-y: auto;[^}]*overscroll-behavior: contain/,
    "reaching the end of a column must not chain the gesture out to the document");
  // Below the two-column breakpoint there are no columns to scroll independently, so the page takes
  // it back and the sticky bar has nothing to stick to.
  assert.match(css, /@media \(max-width: 959px\)[\s\S]*?\.i2i-run-bar \{ position: static/);
});

test("the prompt and the pictures share one left-aligned measure and the previews are square", async () => {
  const css = await read("src/styles.css");
  // Capping the two blocks separately left a band of dead stage to the right of the images that the
  // prompt above them filled — visible as a misalignment rather than as a deliberate margin.
  assert.match(css, /\.i2i-stage-panel \{ --stage-measure: min\(100%, 1600px\); \}/,
    "the stage measure fills the central column until it reaches the wide-screen reading cap");
  for (const block of [".i2i-prompt-deck", ".i2i-compare"]) {
    assert.match(css, new RegExp(`\\${block} \\{[^}]*max-width: var\\(--stage-measure\\)`), `${block} shares the measure`);
  }
  // Keep the bounded stage content on the left edge so the unused width remains on the right rather
  // than becoming an unexplained blank band before it.
  assert.match(css, /\.i2i-prompt-deck, \.i2i-compare[^{]*\{[^}]*margin-inline: 0/);
  assert.match(css, /\.i2i-run-bar > \* \{[^}]*min-width: 0;[^}]*max-width: var\(--stage-measure\);[^}]*margin-inline: 0/,
    "the bar's contents line up with the left-anchored blocks below while the bar stays full-bleed for its sticky background");
  assert.match(css, /\.i2i-compare-pane \.i2i-compare-figure \{[^}]*aspect-ratio: 1 \/ 1/);
  // `.preview-stage` carries a 340px floor that would beat the aspect ratio on a narrow pane.
  assert.match(css, /\.i2i-compare-pane \.i2i-compare-figure \{[^}]*min-height: 0/);
  assert.match(css, /\.i2i-prompt-column \.prompt-field textarea[^{]*\{[^}]*resize: vertical/,
    "the stage scrolls, so the user can make the primary block taller themselves");
});

test("the two pictures are one stage at equal size, and only collapse when neither would be useful", async () => {
  const css = await read("src/styles.css");
  // Equal columns plus fixed head and foot heights: without the second half, the pane whose caption
  // wraps to two lines gets a shorter picture than the one beside it, and the comparison the page
  // exists for stops being like-for-like.
  assert.match(css, /\.i2i-compare \{[^}]*grid-template-columns: repeat\(2, minmax\(0, 1fr\)\)/);
  assert.match(css, /\.i2i-compare-head \{[^}]*min-height: \d+px/);
  assert.match(css, /\.i2i-compare-foot \{[^}]*min-height: \d+px/);
  assert.match(css, /\.i2i-compare-pane \.i2i-compare-figure \{[^}]*width: 100%/,
    "the pane is capped at the square's width, so head, frame and caption align to one edge");
  // Side by side below 640px would be two thumbnails, so the comparison turns vertical instead.
  assert.match(css, /@media \(max-width: 639px\)[\s\S]*?\.i2i-compare[^{]*\{ grid-template-columns: minmax\(0, 1fr\)/);
});

test("the run bar holds the top of the stage and the prompt lives under the pictures", async () => {
  const css = await read("src/styles.css");
  const page = await read("src/ImageToImagePage.jsx");
  const stage = page.slice(page.indexOf('<section className="i2i-stage-panel panel">'), page.indexOf('<aside className="i2i-controls-panel panel">'));
  // Order inside the stage is the contract: the run bar first so nothing below can displace it,
  // then the prompt as the primary block, then the pictures it produces.
  const order = ["i2i-run-bar", "i2i-prompt-deck", "i2i-compare"].map((name) => stage.indexOf(name));
  assert.ok(order.every((at, index) => at >= 0 && (index === 0 || at > order[index - 1])), `stage order is ${order}`);
  assert.match(css, /\.i2i-run-bar \{[^}]*position: sticky;[^}]*top: 0;[^}]*flex: 0 0 auto/,
    "first in order is not enough: the stage scrolls and would carry the bar away with it");
  assert.match(css, /\.i2i-stage-panel \{[^}]*display: flex;[^}]*flex-direction: column/);
  // A sticky bar over transparent background shows whatever scrolls beneath it.
  assert.match(css, /\.i2i-run-bar \{[^}]*background: var\(--n-\d\d\)/);
  // A prompt box is the widest thing a user types into; the rail is 400px at its widest.
  const rail = page.slice(page.indexOf('<aside className="i2i-controls-panel panel">'));
  assert.ok(!rail.includes("i2i-prompt"), "the prompt does not belong in the rail");
});

test("panel placement is keyed on the page's own classes, never on the shared panel names", async () => {
  // `.image-workspace .left-panel` is two class selectors and outranks `.i2i-asset-panel` however
  // late that appears. An earlier generation placed the panels that way, and once the page grew a
  // header row those rules kept winning: between 960 and 1100 px the asset panel landed on the
  // header's row and the result panel shared the composer's cell.
  const css = await read("src/styles.css");
  const page = await read("src/ImageToImagePage.jsx");
  const placement = /(grid-column|grid-row|order)\s*:/;
  for (const rule of css.match(/\.image-workspace\s+\.(left|center|preview)-panel[^{]*\{[^}]*\}/g) || []) {
    assert.ok(!placement.test(rule), `placement must not be keyed on a shared panel name: ${rule}`);
  }
  for (const panel of ["i2i-stage-panel", "i2i-controls-panel"]) {
    assert.match(css, new RegExp(`\\.${panel} \\{[^}]*grid-column:`), `${panel} places itself`);
  }
  // The page stopped borrowing the generate page's panel names entirely, so the class of conflict
  // above cannot recur through markup either.
  for (const shared of ["left-panel", "center-panel", "preview-panel"]) {
    assert.ok(!page.includes(shared), `${shared} is the generate page's, not this one's`);
  }
});

test("shared SizeGrid exposes pointer capture and keyboard controls", async () => {
  const grid = await read("src/SizeGrid.jsx");
  for (const anchor of ["setPointerCapture", 'role="slider"', "ArrowRight", "ArrowLeft", "ArrowUp", "ArrowDown", "Home", "End", "tabIndex={disabled ? -1 : 0}"]) assert.ok(grid.includes(anchor), `${anchor} is required`);
  assert.match(grid, /min = 0, max = 2048, step = 64/);
  assert.match(grid, /grid\.setPointerCapture\?\.\(event\.pointerId\)/, "the grid owns pointer capture after the nested handle starts a drag");
  assert.match(grid, /drag\.startWidth \+ widthDelta/, "handle drags preserve the initial width instead of jumping to the pointer");
  assert.match(grid, /className="grid-handle-hitbox"/, "the visual handle has an expanded pointer target");
});

test("image controls use a fixed left rail with no resize or collapse interaction", async () => {
  const page = await read("src/ImageToImagePage.jsx");
  const layout = await read("src/workspace-layout.js");
  const css = await read("src/styles.css");
  assert.match(page, /<section className="image-workspace">/);
  assert.match(css, /\.image-workspace \{[^}]*grid-template-columns: minmax\(360px, 400px\) minmax\(0, 1fr\)/);
  assert.match(css, /^\.workspace \{[^}]*grid-template-columns: minmax\(360px, 400px\)/m,
    "the image rail starts at the same width as the text-to-image parameter rail");
  assert.match(css, /\.i2i-controls-panel \{[^}]*grid-column: 1;[^}]*border-right: 1px solid var\(--line\)/);
  assert.match(css, /\.i2i-stage-panel \{[^}]*grid-column: 2/);
  for (const removed of ["imageWorkspaceLayout", "i2i-controls-resizer", "beginControlsResize", "toggleImageControlsPanel"]) assert.ok(!page.includes(removed), `${removed} must be removed`);
  assert.doesNotMatch(css, /i2i-controls-resizer|controls-sized|controls-collapsed/);
  assert.doesNotMatch(layout, /IMAGE_WORKSPACE_LAYOUT|ImageControlsPanel|imageWorkspaceLayout/);
});

test("i2i custom dimensions use the shared 64..2048 grid and latest-wins source token", async () => {
  const page = await read("src/ImageToImagePage.jsx");
  assert.match(page, /<SizeGrid[\s\S]*?min=\{64\}[\s\S]*?max=\{2048\}[\s\S]*?step=\{64\}/);
  assert.match(page, /sourceTokenRef/);
  assert.match(page, /const editingLocked = modelPicker\.switching;/);
  assert.match(page, /if \(editingLocked\) return/);
});

test("image-to-image keeps the active run visible while drafting a queued request", async () => {
  const [page, app, logic] = await Promise.all([read("src/ImageToImagePage.jsx"), read("src/App.jsx"), read("src/image-to-image.js")]);
  const submit = app.slice(app.indexOf("  const generateFromImage = async () => {"), app.indexOf("  const releaseLoadedModel = async () => {"));
  assert.match(submit, /if \(submissionInFlight\.current \|\| !imageSource\) return;/);
  assert.match(submit, /const watching = queueBusy;/);
  assert.match(submit, /if \(!watching\) \{[\s\S]{0,500}beginGenerationRun\(/);
  assert.match(submit, /gallerySettings,/);
  assert.doesNotMatch(submit, /status === "running" \|\| !imageSource/);
  assert.match(page, /queueBusy = false,/);
  assert.match(page, /className=\{`generate-button \$\{queueBusy \? "queueing" : ""\}`\}/);
  assert.match(page, /<PostprocessControls[\s\S]*?running=\{editingLocked\}/);
  assert.doesNotMatch(logic, /if \(running\) return "已有任务正在生成"/);
});

test("image controls use an accessible unclipped dropdown and page-scoped scrollbars", async () => {
  const app = await read("src/App.jsx");
  const page = await read("src/ImageToImagePage.jsx");
  const select = await read("src/WorkspaceSelect.jsx");
  const css = await read("src/styles.css");
  assert.ok(!page.includes("<select"), "native selects do not match the page's scroll and focus treatment");
  for (const anchor of ["createPortal", 'aria-haspopup="listbox"', 'role="option"', "aria-activedescendant", 'event.key === "Escape"', 'event.key === "Home"', 'event.key === "End"']) {
    assert.ok(select.includes(anchor), `${anchor} is required for the custom dropdown`);
  }
  assert.match(app, /import WorkspaceSelect from "\.\/WorkspaceSelect"/);
  for (const label of ['ariaLabel="采样器"', 'ariaLabel="Hires Seed 模式"', 'ariaLabel={`第 ${index + 1} 个 ADetailer 检测模型`}', 'ariaLabel="RTX VSR 处理质量"']) assert.ok(app.includes(`<WorkspaceSelect ${label}`), `${label} uses the shared dropdown on text-to-image`);
  assert.match(page, /import WorkspaceSelect from "\.\/WorkspaceSelect"/);
  assert.match(css, /\.workspace-select-menu \{[^}]*position: fixed;[^}]*overflow-y: auto;[^}]*scrollbar-width: thin/,
    "the dropdown escapes both independent column scrollports and keeps long lists usable");
  assert.match(css, /\.i2i-stage-panel \{[^}]*scrollbar-gutter: stable;[^}]*scrollbar-width: thin/);
  assert.match(css, /\.i2i-controls-panel \.panel-scroll \{[^}]*scrollbar-gutter: stable;[^}]*scrollbar-width: thin/);
});

test("prompt and picture columns share the same vertical guides", async () => {
  const css = await read("src/styles.css");
  assert.match(css, /\.i2i-prompt-deck \{[^}]*grid-template-columns: repeat\(2, minmax\(0, 1fr\)\);[^}]*gap: 8px 16px/);
  assert.match(css, /\.i2i-compare \{[^}]*grid-template-columns: repeat\(2, minmax\(0, 1fr\)\);[^}]*gap: 16px/);
});

test("image prompts reuse the text-to-image cards and keep transparency below the pair", async () => {
  const [page, app, control, css] = await Promise.all([
    read("src/ImageToImagePage.jsx"),
    read("src/App.jsx"),
    read("src/TransparentBackgroundControl.jsx"),
    read("src/styles.css"),
  ]);
  assert.match(page, /className=\{`prompt-field positive-field/);
  assert.match(page, /className="prompt-field negative-field"/);
  assert.match(page, /<Sparkles className="field-watermark" size=\{46\}/);
  assert.match(page, /spellCheck=\{false\}/);
  assert.match(page, /<TransparentBackgroundControl[\s\S]*?className="i2i-transparent-control"/);
  assert.match(page, /\{positivePromptPresets\}/);
  assert.match(page, /\{negativePromptPresets\}/);
  assert.match(app, /<TransparentBackgroundControl[\s\S]*?onToggle=\{toggleTextToImageTransparentBackground\}/);
  assert.match(app, /positivePromptPresets=\{<PresetBox title="预设正向 Prompt" type="positive"/);
  assert.match(app, /negativePromptPresets=\{<PresetBox title="预设负向 Prompt" type="negative"/);
  assert.match(control, /className=\{`transparent-tag-control/);
  assert.match(css, /\.i2i-prompt-column \{[^}]*display: grid/);
  assert.match(css, /\.i2i-prompt-column > \.preset-box \{ min-width: 0; \}/);
  assert.match(css, /\.i2i-prompt-deck > \.i2i-transparent-control \{ grid-column: 1 \/ -1; margin: 0; \}/);
  assert.match(css, /\.transparent-tag-control \{[^}]*position: relative;[^}]*z-index: 1;/,
    "the ordinary control stays below the sticky image-to-image run bar while scrolling");
  assert.match(css, /\.i2i-run-bar \{[^}]*position: sticky;[^}]*z-index: 3;/,
    "the run bar owns the upper stacking layer");
  assert.doesNotMatch(page, /className="i2i-prompt"/);
});

test("the run bar's summary strip and its button share top and bottom edges", async () => {
  const css = await read("src/styles.css");
  // The button carries a 58px floor and the strip beside it is a few pixels shorter. Centred, the
  // button overhung it at both ends; stretched, the two read as one bar.
  const row = css.match(/^\.i2i-run-actions \{[^}]*\}/m)[0];
  assert.match(row, /align-items: stretch/);
  assert.doesNotMatch(row, /align-items: center/);
  // Stretching the strip is only half of it: without this its labels would sit against the top of
  // a cell that is now taller than the text it holds.
  assert.match(css, /\.i2i-run-actions \.generation-info > div \{[^}]*display: flex;[^}]*flex-direction: column;[^}]*justify-content: center/);
});

test("the run bar adapts to the fixed stage width and exposes the shared run diagnostics", async () => {
  const css = await read("src/styles.css");
  const page = await read("src/ImageToImagePage.jsx");
  const app = await read("src/App.jsx");
  assert.match(css, /\.i2i-stage-panel \{[^}]*min-width: 0;[^}]*container-type: inline-size/);
  assert.match(css, /\.i2i-run-bar \{[^}]*width: 100%;[^}]*min-width: 0/);
  assert.match(css, /@container \(max-width: 900px\) \{\s*\.i2i-run-actions \{ grid-template-columns: minmax\(0, 1fr\); \}/,
    "run controls respond to the stage width when the rail is widened on desktop");
  assert.match(page, /aria-controls="generation-run-info"/);
  assert.match(page, /<div id="generation-run-info" className="i2i-run-info" hidden=\{!runInfoVisible\}>/);
  assert.match(page, /runInfoVisible \? <Eye size=\{15\} \/> : <EyeOff size=\{15\} \/>/);
  assert.equal((page.match(/<span>当前模型<\/span>/g) || []).length, 1,
    "the always-visible run summary owns the model name; diagnostics must not repeat it");
  assert.match(css, /\.i2i-run-info \.generation-info \{ grid-template-columns: repeat\(2, minmax\(0, 1fr\)\); margin: 0; \}/);
  assert.match(app, /runInfoVisible=\{runInfoVisible\}/);
  assert.match(app, /onToggleRunInfo=\{\(\) => setRunInfoVisible\(\(current\) => !current\)\}/);
});

test("image postprocess stages have independent collapse controls", async () => {
  const [page, css] = await Promise.all([read("src/ImageToImagePage.jsx"), read("src/styles.css")]);
  assert.match(page, /ChevronDown,/);
  assert.match(page, /const \[expandedStages, setExpandedStages\] = useState\(\{ hires: false, adetailer: false, rtx: false \}\)/);
  assert.match(page, /const toggleStageExpanded = \(stage\) => setExpandedStages\(\(current\) => \(\{ \.\.\.current, \[stage\]: !current\[stage\] \}\)\)/);
  for (const stage of ["hires", "adetailer", "rtx"]) {
    assert.match(page, new RegExp(`expanded=\\{expandedStages\\.${stage}\\}[\\s\\S]*?onToggleExpanded=\\{\\(\\) => toggleStageExpanded\\("${stage}"\\)\\}`));
    assert.match(page, new RegExp(`expandedStages\\.${stage} &&`));
    assert.doesNotMatch(page, new RegExp(`config\\.${stage}\\.enabled && expandedStages\\.${stage}`));
  }
  assert.match(page, /className="i2i-stage-summary" aria-expanded=\{expanded\}/);
  assert.match(page, /className="i2i-stage-expand"/);
  assert.match(css, /\.i2i-stage-head-actions > \.i2i-stage-expand, \.postprocess-stage-head-actions > \.postprocess-stage-expand \{/);
  assert.match(css, /\.i2i-stage-card\.expanded \.i2i-stage-expand svg, \.postprocess-stage-card\.expanded \.postprocess-stage-expand svg \{ transform: rotate\(180deg\); \}/);
});

test("image postprocess parameters reuse the text-to-image visual primitives", async () => {
  const [page, css] = await Promise.all([read("src/ImageToImagePage.jsx"), read("src/styles.css")]);
  assert.match(page, /function PostprocessSlider\([\s\S]*?className="slider-field"/);
  for (const panel of ["hires", "adetailer", "rtx"]) assert.match(page, new RegExp(`className="${panel}-parameters"`));
  for (const primitive of ["hires-scale", "hires-sliders", "adetailer-units", "adetailer-pager", "adetailer-sliders", "adetailer-mask-grid", "rtx-runtime-grid", "rtx-scale"]) {
    assert.match(page, new RegExp(`className="${primitive}"`), `${primitive} is shared with text-to-image`);
  }
  assert.match(page, /className=\{`adetailer-unit \$\{unit\.enabled \? "on" : ""\}`\}/);
  assert.match(page, /className="bounded-number"/);
  assert.doesNotMatch(page, /i2i-hires-body|i2i-adetailer-body|i2i-adetailer-pager|i2i-adetailer-unit/);
  assert.match(css, /\.i2i-stage-card > \.hires-parameters, \.i2i-stage-card > \.adetailer-parameters, \.i2i-stage-card > \.rtx-parameters, \.postprocess-stage-card > \.hires-parameters/);
  assert.match(css, /\.workspace input\[type="range"\]::-webkit-slider-thumb, \.i2i-stage-card \.slider-field > input\[type="range"\]::-webkit-slider-thumb \{[^}]*border-radius: 50%/);
  assert.match(css, /\.workspace input\[type="range"\]::-moz-range-thumb, \.i2i-stage-card \.slider-field > input\[type="range"\]::-moz-range-thumb \{[^}]*border-radius: 50%/);
});
