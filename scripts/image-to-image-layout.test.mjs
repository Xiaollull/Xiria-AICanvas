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
  // The stage takes everything the rail does not need; row 2 is the scrollport each column lives in.
  assert.match(css, /^\.image-workspace \{[^}]*display: grid;[^}]*grid-template-columns: minmax\(0, 1fr\) minmax\(330px, 400px\);[^}]*grid-template-rows: auto minmax\(0, 1fr\)/m);
  assert.match(css, /@media \(max-width: 1279px\)[\s\S]*?minmax\(0, 1fr\) minmax\(300px, 340px\)/);
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

test("the prompt and the pictures share one measure and the previews are square", async () => {
  const css = await read("src/styles.css");
  // Capping the two blocks separately left a band of dead stage to the right of the images that the
  // prompt above them filled — visible as a misalignment rather than as a deliberate margin.
  assert.match(css, /\.i2i-stage-panel \{ --stage-measure: \d+px; \}/);
  for (const block of [".i2i-prompt-deck", ".i2i-compare"]) {
    assert.match(css, new RegExp(`\\${block} \\{[^}]*max-width: var\\(--stage-measure\\)`), `${block} shares the measure`);
  }
  // Centred, so the leftover on a wide monitor reads as a margin rather than as a block that failed
  // to reach the edge — which is what the separate caps looked like.
  assert.match(css, /\.i2i-prompt-deck, \.i2i-compare[^{]*\{[^}]*margin-inline: auto/);
  assert.match(css, /\.i2i-run-bar > \* \{[^}]*max-width: var\(--stage-measure\);[^}]*margin-inline: auto/,
    "the bar's contents line up with the blocks below while the bar stays full-bleed for its sticky background");
  assert.match(css, /\.i2i-compare-pane \.i2i-compare-figure \{[^}]*aspect-ratio: 1 \/ 1/);
  // `.preview-stage` carries a 340px floor that would beat the aspect ratio on a narrow pane.
  assert.match(css, /\.i2i-compare-pane \.i2i-compare-figure \{[^}]*min-height: 0/);
  assert.match(css, /\.i2i-prompt-deck \.i2i-prompt textarea[^{]*\{[^}]*resize: vertical/,
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
  assert.match(css, /\.i2i-run-bar \{[^}]*background: #/);
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
});

test("i2i custom dimensions use the shared 64..2048 grid and latest-wins source token", async () => {
  const page = await read("src/ImageToImagePage.jsx");
  assert.match(page, /<SizeGrid[\s\S]*?min=\{64\}[\s\S]*?max=\{2048\}[\s\S]*?step=\{64\}/);
  assert.match(page, /sourceTokenRef/);
  assert.match(page, /if \(running\) return/);
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
