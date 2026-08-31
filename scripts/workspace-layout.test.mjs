import assert from "node:assert/strict";
import test from "node:test";
import path from "node:path";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import {
  DEFAULT_IMAGE_WORKSPACE_LAYOUT,
  DEFAULT_WORKSPACE_LAYOUT,
  IMAGE_CONTROLS_COLLAPSE_WIDTH,
  IMAGE_CONTROLS_MAXIMUM_WIDTH,
  IMAGE_CONTROLS_MINIMUM_WIDTH,
  IMAGE_CONTROLS_WIDTH_STEP,
  IMAGE_WORKSPACE_LAYOUT_STORAGE_KEY,
  LEFT_PANEL_COLLAPSE_WIDTH,
  LEFT_PANEL_MAXIMUM_WIDTH,
  LEFT_PANEL_MINIMUM_WIDTH,
  LEFT_PANEL_WIDTH_STEP,
  WORKSPACE_LAYOUT_STORAGE_KEY,
  clampImageControlsWidth,
  clampLeftPanelWidth,
  imageWorkspaceLayoutClassName,
  imageWorkspaceLayoutStyle,
  isDefaultWorkspaceLayout,
  normalizeImageWorkspaceLayout,
  normalizeWorkspaceLayout,
  readImageWorkspaceLayout,
  readWorkspaceLayout,
  resizeImageControlsPanel,
  resizeLeftPanel,
  steppedImageControlsPanel,
  steppedLeftPanel,
  toggleImageControlsPanel,
  toggleLeftPanel,
  writeImageWorkspaceLayout,
  workspaceLayoutClassName,
  workspaceLayoutStyle,
  writeWorkspaceLayout,
} from "../src/workspace-layout.js";

const projectDirectory = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const source = (name) => readFile(path.join(projectDirectory, "src", name), "utf8");

function fakeStorage({ throws = false } = {}) {
  const map = new Map();
  return {
    map,
    getItem: (key) => { if (throws) throw new Error("blocked"); return map.has(key) ? map.get(key) : null; },
    setItem: (key, value) => { if (throws) throw new Error("blocked"); map.set(key, String(value)); },
    removeItem: (key) => { if (throws) throw new Error("blocked"); map.delete(key); },
  };
}

test("the default layout carries no width, so the stylesheet decides what default means", () => {
  // The point of `leftWidth: 0`: no number in the module has to be kept in sync with the grid
  // template, so "the default width is the current one" cannot drift as the stylesheet changes.
  assert.deepEqual({ ...DEFAULT_WORKSPACE_LAYOUT }, { leftWidth: 0, leftCollapsed: false });
  assert.equal(workspaceLayoutClassName(DEFAULT_WORKSPACE_LAYOUT), "workspace");
  assert.equal(workspaceLayoutStyle(DEFAULT_WORKSPACE_LAYOUT), undefined);
  assert.equal(isDefaultWorkspaceLayout(DEFAULT_WORKSPACE_LAYOUT), true);
});

test("a resized layout emits both the marker class and the width, and a collapsed one emits neither", () => {
  const sized = { leftWidth: 420, leftCollapsed: false };
  assert.equal(workspaceLayoutClassName(sized), "workspace left-sized");
  assert.deepEqual(workspaceLayoutStyle(sized), { "--left-panel-width": "420px" });

  const collapsed = { leftWidth: 420, leftCollapsed: true };
  assert.equal(workspaceLayoutClassName(collapsed), "workspace left-collapsed");
  // A collapsed panel is laid out by its own rule; publishing a width would let the two disagree.
  assert.equal(workspaceLayoutStyle(collapsed), undefined);
});

test("width is bounded wherever it enters, not only where it is dragged", () => {
  assert.equal(clampLeftPanelWidth(10), LEFT_PANEL_MINIMUM_WIDTH);
  assert.equal(clampLeftPanelWidth(9000), LEFT_PANEL_MAXIMUM_WIDTH);
  assert.equal(clampLeftPanelWidth(413.6), 414);
  // Zero and nonsense both mean "unset", which is the stylesheet default rather than a clamp.
  for (const junk of [0, -5, null, undefined, "wide", NaN, Infinity, {}]) {
    assert.equal(clampLeftPanelWidth(junk), 0, `${String(junk)} must read as unset`);
  }
  // A hand-edited storage value goes through the same gate a gesture does.
  assert.deepEqual(normalizeWorkspaceLayout({ leftWidth: 9000, leftCollapsed: "yes" }), { leftWidth: LEFT_PANEL_MAXIMUM_WIDTH, leftCollapsed: false });
  for (const junk of [null, undefined, 7, "text", []]) {
    assert.deepEqual(normalizeWorkspaceLayout(junk), { leftWidth: 0, leftCollapsed: false });
  }
});

test("a drag is measured from the starting rect, so a clamped axis cannot accumulate drift", () => {
  const start = 360;
  assert.deepEqual(resizeLeftPanel(DEFAULT_WORKSPACE_LAYOUT, start, 60), { leftWidth: 420, leftCollapsed: false });
  // Past the maximum and back again returns to the same width it would have had without the
  // excursion — the value is recomputed from the start each time, never accumulated.
  assert.deepEqual(resizeLeftPanel(DEFAULT_WORKSPACE_LAYOUT, start, 9000), { leftWidth: LEFT_PANEL_MAXIMUM_WIDTH, leftCollapsed: false });
  assert.deepEqual(resizeLeftPanel(DEFAULT_WORKSPACE_LAYOUT, start, 60), { leftWidth: 420, leftCollapsed: false });
  assert.deepEqual(resizeLeftPanel(DEFAULT_WORKSPACE_LAYOUT, start, -60), { leftWidth: 300, leftCollapsed: false });
  // Narrower than the minimum but not yet a collapse: clamped, and still not accumulating.
  assert.deepEqual(resizeLeftPanel(DEFAULT_WORKSPACE_LAYOUT, start, -160), { leftWidth: LEFT_PANEL_MINIMUM_WIDTH, leftCollapsed: false });
  for (const junk of [undefined, NaN]) {
    assert.equal(resizeLeftPanel(DEFAULT_WORKSPACE_LAYOUT, start, junk).leftWidth, clampLeftPanelWidth(start));
  }
});

test("dragging the panel shut collapses it and remembers the width to come back to", () => {
  // Otherwise the gesture would stop dead at the minimum and the only way to close the panel
  // would be to find the toggle.
  const collapsed = resizeLeftPanel(DEFAULT_WORKSPACE_LAYOUT, 360, -(360 - LEFT_PANEL_COLLAPSE_WIDTH) - 1);
  assert.equal(collapsed.leftCollapsed, true);
  assert.equal(collapsed.leftWidth, 360, "expanding must return to the width it had, not to the default");
  // One pixel short of the threshold is still a resize.
  assert.equal(resizeLeftPanel(DEFAULT_WORKSPACE_LAYOUT, 360, -(360 - LEFT_PANEL_COLLAPSE_WIDTH) + 1).leftCollapsed, false);
  // Dragging out from a collapsed panel measures 0, so the width comes from the gesture alone.
  assert.deepEqual(resizeLeftPanel({ leftWidth: 300, leftCollapsed: true }, 0, 400), { leftWidth: 400, leftCollapsed: false });
});

test("the keyboard reaches everything the pointer does", () => {
  // A pointer-only affordance would put the panel out of reach of anyone not using one.
  assert.deepEqual(steppedLeftPanel(DEFAULT_WORKSPACE_LAYOUT, 1, 360), { leftWidth: 360 + LEFT_PANEL_WIDTH_STEP, leftCollapsed: false });
  assert.deepEqual(steppedLeftPanel({ leftWidth: 400, leftCollapsed: false }, -1, 0), { leftWidth: 400 - LEFT_PANEL_WIDTH_STEP, leftCollapsed: false });
  assert.equal(steppedLeftPanel({ leftWidth: LEFT_PANEL_MAXIMUM_WIDTH, leftCollapsed: false }, 1, 0).leftWidth, LEFT_PANEL_MAXIMUM_WIDTH);
  // The first keystroke out of a collapsed panel reopens it rather than nudging a hidden edge.
  assert.deepEqual(steppedLeftPanel({ leftWidth: 300, leftCollapsed: true }, 1, 0), { leftWidth: 300, leftCollapsed: false });
  assert.deepEqual(steppedLeftPanel({ leftWidth: 300, leftCollapsed: true }, -1, 0), { leftWidth: 300, leftCollapsed: true });
  assert.deepEqual(toggleLeftPanel(DEFAULT_WORKSPACE_LAYOUT), { leftWidth: 0, leftCollapsed: true });
  assert.deepEqual(toggleLeftPanel({ leftWidth: 400, leftCollapsed: true }), { leftWidth: 400, leftCollapsed: false });
});

test("the image controls rail has independent right-side resize and collapse semantics", () => {
  assert.deepEqual({ ...DEFAULT_IMAGE_WORKSPACE_LAYOUT }, { controlsWidth: 0, controlsCollapsed: false });
  assert.equal(imageWorkspaceLayoutClassName(DEFAULT_IMAGE_WORKSPACE_LAYOUT), "image-workspace");
  assert.equal(imageWorkspaceLayoutStyle(DEFAULT_IMAGE_WORKSPACE_LAYOUT), undefined);

  assert.equal(clampImageControlsWidth(10), IMAGE_CONTROLS_MINIMUM_WIDTH);
  assert.equal(clampImageControlsWidth(9000), IMAGE_CONTROLS_MAXIMUM_WIDTH);
  assert.deepEqual(normalizeImageWorkspaceLayout({ controlsWidth: 9000, controlsCollapsed: "yes" }), { controlsWidth: IMAGE_CONTROLS_MAXIMUM_WIDTH, controlsCollapsed: false });

  // This rail sits on the right: moving its left seam left is a negative delta and makes it wider.
  assert.deepEqual(resizeImageControlsPanel(DEFAULT_IMAGE_WORKSPACE_LAYOUT, 360, -60), { controlsWidth: 420, controlsCollapsed: false });
  assert.deepEqual(resizeImageControlsPanel(DEFAULT_IMAGE_WORKSPACE_LAYOUT, 360, 160), { controlsWidth: 360, controlsCollapsed: true });
  assert.ok(360 - 160 < IMAGE_CONTROLS_COLLAPSE_WIDTH);
  assert.deepEqual(steppedImageControlsPanel(DEFAULT_IMAGE_WORKSPACE_LAYOUT, 1, 360), { controlsWidth: 360 + IMAGE_CONTROLS_WIDTH_STEP, controlsCollapsed: false });
  assert.deepEqual(steppedImageControlsPanel({ controlsWidth: 400, controlsCollapsed: true }, 1, 0), { controlsWidth: 400, controlsCollapsed: false });
  assert.deepEqual(toggleImageControlsPanel({ controlsWidth: 400, controlsCollapsed: false }), { controlsWidth: 400, controlsCollapsed: true });

  const storage = fakeStorage();
  writeWorkspaceLayout(storage, { leftWidth: 420, leftCollapsed: false });
  writeImageWorkspaceLayout(storage, { controlsWidth: 440, controlsCollapsed: false });
  assert.deepEqual(readImageWorkspaceLayout(storage), { controlsWidth: 440, controlsCollapsed: false });
  writeImageWorkspaceLayout(storage, DEFAULT_IMAGE_WORKSPACE_LAYOUT);
  assert.equal(storage.map.has(IMAGE_WORKSPACE_LAYOUT_STORAGE_KEY), false, "resetting the image rail only removes its own key");
  assert.equal(storage.map.has(WORKSPACE_LAYOUT_STORAGE_KEY), true, "the generate page layout remains intact");
});

test("the default is stored as absence, so a reset leaves nothing to be read back", () => {
  const storage = fakeStorage();
  writeWorkspaceLayout(storage, { leftWidth: 420, leftCollapsed: false });
  assert.equal(storage.map.size, 1);
  assert.deepEqual(readWorkspaceLayout(storage), { leftWidth: 420, leftCollapsed: false });

  writeWorkspaceLayout(storage, DEFAULT_WORKSPACE_LAYOUT);
  assert.equal(storage.map.has(WORKSPACE_LAYOUT_STORAGE_KEY), false, "resetting must not leave a stored default behind");
  assert.deepEqual(readWorkspaceLayout(storage), { ...DEFAULT_WORKSPACE_LAYOUT });
});

test("a blocked or corrupt storage costs the remembered layout, never the page", () => {
  assert.deepEqual(readWorkspaceLayout(fakeStorage({ throws: true })), { ...DEFAULT_WORKSPACE_LAYOUT });
  assert.doesNotThrow(() => writeWorkspaceLayout(fakeStorage({ throws: true }), { leftWidth: 400, leftCollapsed: true }));
  assert.deepEqual(readWorkspaceLayout(null), { ...DEFAULT_WORKSPACE_LAYOUT });

  const corrupt = fakeStorage();
  corrupt.map.set(WORKSPACE_LAYOUT_STORAGE_KEY, "{ not json");
  assert.deepEqual(readWorkspaceLayout(corrupt), { ...DEFAULT_WORKSPACE_LAYOUT });
});

test("layout is stored apart from every creation parameter, so a reset cannot reach one", async () => {
  const app = await source("App.jsx");
  // The reset writes the layout state and nothing else. If it ever touched a parameter setter the
  // promise this feature makes — "restores the layout without modifying your settings" — would be
  // quietly false, and nothing else in the suite would notice.
  const reset = app.slice(app.indexOf("const resetWorkspaceLayout ="), app.indexOf("const applyAssistantPrompt"));
  assert.match(reset, /commitWorkspaceLayout\(\{ \.\.\.DEFAULT_WORKSPACE_LAYOUT \}\)/);
  assert.ok(!/set(?!WorkspaceLayout)[A-Z]/.test(reset), "the reset may not call any other setter");

  const layout = await source("workspace-layout.js");
  // Its own browser key, and only that one. Checked against the calls rather than against the
  // prose, which necessarily names the stores this module exists to stay out of.
  assert.match(layout, /WORKSPACE_LAYOUT_STORAGE_KEY = "xirai-workspace-layout-v1"/);
  const keys = [...layout.matchAll(/(?:get|set|remove)Item\(([^,)]+)/g)].map((match) => match[1].trim());
  assert.deepEqual([...new Set(keys)], ["WORKSPACE_LAYOUT_STORAGE_KEY", "IMAGE_WORKSPACE_LAYOUT_STORAGE_KEY"], "each workspace may read and write only its dedicated layout key");
  assert.ok(!/^import /m.test(layout), "the layout model is dependency-free, so it cannot reach a parameter store");
  assert.ok(!/fetch\(|\/api\//.test(layout), "layout is browser-local and never posted to the control plane");
});

test("the panel is resizable and collapsible from the gutter, and the grid keeps its shipped default", async () => {
  const app = await source("App.jsx");
  const styles = await readFile(path.join(projectDirectory, "src", "styles.css"), "utf8");

  // The handle lives between the two panels it separates, not inside either of them.
  const gutter = app.slice(app.indexOf('<aside className="left-panel panel"'), app.indexOf('<section className="center-panel panel">'));
  assert.match(gutter, /className="panel-resizer"/);
  assert.match(gutter, /onPointerDown=\{beginPanelResize\}/);
  assert.match(gutter, /onPointerMove=\{continuePanelResize\}/);
  assert.match(gutter, /onPointerCancel=\{endPanelResize\}/);
  assert.match(gutter, /role="separator"/);
  assert.match(gutter, /onKeyDown=\{panelResizeKeyDown\}/);
  // Collapsing is the gutter's own double-click, not a tab parked on it. A button in a 7px strip
  // was chrome for a gesture that reads fine without one — and it had to cancel the drag beneath
  // itself to work at all.
  assert.match(gutter, /onDoubleClick=\{\(\) => commitWorkspaceLayout\(toggleLeftPanel\)\}/);
  assert.ok(!gutter.includes("panel-resizer-toggle"), "the seam carries no button");
  assert.ok(!/stopPropagation/.test(gutter), "nothing on the seam needs to cancel the drag beneath it");
  // With no tab to point at, the separator itself has to report the state it is in.
  assert.match(gutter, /aria-expanded=\{!workspaceLayout\.leftCollapsed\}/);
  assert.match(gutter, /workspaceLayout\.leftCollapsed \? "展开左侧配置面板"/);
  assert.match(app, /setPointerCapture\?\.\(event\.pointerId\)/, "a drag must survive the pointer leaving the 7px strip");

  // The untouched grid template still lays out an unresized workspace, which is what makes the
  // default width exactly the shipped one rather than a number copied into JavaScript.
  assert.match(styles, /\n\.workspace \{[^}]*grid-template-columns: minmax\(320px, 360px\) 7px minmax\(420px, 1fr\) minmax\(350px, 460px\)/);
  assert.match(styles, /\.workspace\.left-sized \{[^}]*grid-template-columns: var\(--left-panel-width\)/);
  assert.match(styles, /\.workspace\.left-collapsed \{[^}]*grid-template-columns: 0 /);
  // A zero-width panel is still in the tab order; a collapsed one must not be reachable by tabbing.
  assert.match(styles, /\.workspace\.left-collapsed \.left-panel \{[^}]*visibility: hidden/);
  assert.ok(styles.includes(".panel-resizer {"), "the handle needs a hit area");
  assert.ok(!styles.includes(".panel-resizer-toggle"), "the removed tab must not leave rules behind");
  // The seam is the only affordance left, so a collapsed panel has to mark where it went.
  assert.match(styles, /\.workspace\.left-collapsed \.panel-resizer \{[^}]*background: color-mix/);
  // Below the two-column breakpoint there is no gutter, and the panel is the only place those
  // controls exist, so a collapse chosen on a wide screen must not hide them on a narrow one.
  assert.match(styles, /@media \(max-width: 1100px\) \{[\s\S]*?\.workspace \.panel-resizer \{ display: none; \}/);
  assert.match(styles, /@media \(max-width: 1100px\) \{[\s\S]*?\.workspace\.left-collapsed \.left-panel \{ visibility: visible/);
});

test("the reset control states its scope and disables itself once there is nothing to reset", async () => {
  const app = await source("App.jsx");
  const option = app.slice(app.indexOf("<strong>界面布局</strong>"), app.indexOf("<strong>界面动态效果</strong>"));
  assert.match(option, /不会改动任何创作参数或其他设置/, "the control must say what it does not touch");
  assert.match(option, /disabled=\{isDefaultWorkspaceLayout\(workspaceLayout\)\}/);
  assert.match(option, /onClick=\{resetWorkspaceLayout\}/);
  const styles = await readFile(path.join(projectDirectory, "src", "styles.css"), "utf8");
  assert.ok(styles.includes(".settings-layout-reset {"));
});
