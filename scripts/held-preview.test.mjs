import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const read = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("the last finished picture is kept past the end of its own run", async () => {
  const app = await read("src/App.jsx");
  assert.match(app, /const \[heldPreview, setHeldPreview\] = useState\(null\);/);
  // Recorded from what the panel is already showing as finished, so a run that completed here and a
  // finished job picked out of the queue both land in it without either path remembering to.
  assert.match(app, /if \(status !== "complete" \|\| !generatedImage\) return;\s*setHeldPreview\(\{ url: generatedImage, name: generatedName \}\);/);

  // The board a new run starts from clears this run's result -- and must not clear the held one,
  // which is the whole point: queueing one more image used to cost you the sight of the one you
  // were still looking at.
  const begin = app.slice(app.indexOf("  const beginGenerationRun = ("), app.indexOf("  const generate = async () => {"));
  assert.ok(begin.length > 0, "beginGenerationRun is still there to check");
  assert.match(begin, /setGeneratedImage\(""\)/, "the current result is still cleared");
  assert.doesNotMatch(begin, /setHeldPreview/, "the held picture survives the next submission");
});

test("the held picture fills the gap until the new run has something of its own", async () => {
  const shared = await read("src/StagePreviews.jsx");
  // The placeholder is only for when there is genuinely nothing to show.
  assert.match(shared, /\{stage \|\| held\?\.url \? null : <><ImageIcon size=\{28\} \/><span>\{placeholder\}<\/span><\/>\}/);
  assert.match(shared, /\{!stage && held\?\.url && <div className="stage-preview-image held-preview">/);
  // A picture with no caption in a box that says "generating" would read as this run's output.
  assert.match(shared, /<span className="stage-preview-tag">上一次结果 · \{placeholder\}<\/span>/);
  // The moment this run publishes a stage, that stage takes over.
  assert.match(shared, /\{stage && <button/);

  const app = await read("src/App.jsx");
  assert.match(app, /<StagePreviewLive stage=\{latestStagePreview\}[^/]*held=\{heldPreview\} \/>/);
  const page = await read("src/ImageToImagePage.jsx");
  assert.match(page, /<StagePreviewLive stage=\{latestStagePreview\}[^/]*held=\{job\.held\} \/>/);
  // Both pages watch one run, so they hold one picture; the image page reads it off the job.
  assert.match(app, /\/\/ Both pages watch the same run, so they hold the same last picture\.\s*held: heldPreview,/);
});

test("a cancelled run keeps the last finished picture instead of replacing it with a notice", async () => {
  for (const file of ["src/App.jsx", "src/ImageToImagePage.jsx"]) {
    const source = await read(file);
    assert.match(source, /className="stage-preview-image held-preview cancelled-held"/, file);
    assert.match(source, /<span className="stage-preview-tag">已终止 · 显示上一次完成的结果<\/span>/, file);
    // There is still a way back to an empty board; it is a control on the picture rather than a
    // panel in place of it.
    assert.match(source, /className="held-preview-reset"/, file);
    // And with nothing held -- a cancel before anything ever finished -- the original notice stands.
    assert.match(source, /<div className="generation-cancelled"><Square size=\{30\} \/><strong>生成已终止<\/strong>/, file);
  }
  const css = await read("src/styles.css");
  // Dimmed, so it reads as context rather than as this run's result.
  assert.match(css, /\.held-preview \{ cursor: default; opacity: \.62;/);
  assert.match(css, /\.held-preview-reset \{ position: absolute;/);
});

test("the background-removal menu leaves the stacking context that was burying it", async () => {
  const control = await read("src/TransparentBackgroundControl.jsx");
  const css = await read("src/styles.css");
  // `.transparent-tag-control` keeps `z-index: 1` on purpose, to stay under the sticky
  // image-to-image run bar. That makes it a stacking context, which is a ceiling: the menu's own
  // z-index counted only against its siblings inside the control, so the prompt boxes and the size
  // grid painted straight over the open menu. No number fixes that from inside it.
  assert.match(css, /\.transparent-tag-control \{ position: relative; z-index: 1;/);
  assert.match(control, /createPortal\(<div ref=\{menuRef\} className="transparent-model-menu"/);
  assert.match(control, /<\/div>, document\.body\)\}/);
  assert.match(css, /\.transparent-model-menu \{ position: fixed; z-index: var\(--z-modal-popover\);/);
  assert.doesNotMatch(css, /\.transparent-model-menu \{ position: absolute/);
  // Geometry comes from the trigger now, so no breakpoint may restate the width or the edges.
  assert.doesNotMatch(css, /\.transparent-model-menu \{ width:/);
  assert.match(control, /const rect = pickerRef\.current\?\.getBoundingClientRect\(\);/);
  assert.match(control, /placeAbove \? \{ bottom: window\.innerHeight - rect\.top \+ 6 \} : \{ top: rect\.bottom \+ 6 \}/,
    "a menu that will not fit below the trigger opens above it");
  // Portalled means a press inside the menu is no longer a press inside the picker; without this
  // the dismiss listener would close it before a model could be chosen.
  assert.match(control, /pickerRef\.current\?\.contains\(event\.target\) \|\| menuRef\.current\?\.contains\(event\.target\)/);
  // It follows the trigger while the page behind it scrolls.
  assert.match(control, /document\.addEventListener\("scroll", onScroll, true\)/);
  assert.match(control, /window\.addEventListener\("resize", positionMenu\)/);
});
