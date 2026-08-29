import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { VIEWER_TOOLBAR_POPOVER_LAYOUT, VIEWER_TOOLBAR_POPOVER_TEMPLATES, viewerEscapeAction, viewerToolbarPopoverTransition } from "../src/viewer-toolbar.js";

test("toolbar popover transition defines deterministic focus semantics and Escape priority", () => {
  assert.deepEqual(viewerToolbarPopoverTransition("none", VIEWER_TOOLBAR_POPOVER_LAYOUT), { next: "layout", focusTarget: "layout", focusReturn: null });
  assert.deepEqual(viewerToolbarPopoverTransition("layout", VIEWER_TOOLBAR_POPOVER_TEMPLATES), { next: "templates", focusTarget: "templates", focusReturn: null });
  assert.deepEqual(viewerToolbarPopoverTransition("templates", VIEWER_TOOLBAR_POPOVER_LAYOUT), { next: "layout", focusTarget: "layout", focusReturn: null });
  assert.deepEqual(viewerToolbarPopoverTransition("layout", "none", "backdrop"), { next: "none", focusTarget: null, focusReturn: "layout" });
  assert.deepEqual(viewerToolbarPopoverTransition("templates", "none", "escape"), { next: "none", focusTarget: null, focusReturn: "templates" });
  assert.deepEqual(viewerToolbarPopoverTransition("layout", "none", "context-menu"), { next: "none", focusTarget: null, focusReturn: null });
  assert.deepEqual(viewerToolbarPopoverTransition("templates", "none", "viewer-close"), { next: "none", focusTarget: null, focusReturn: null });
  assert.equal(viewerEscapeAction({ historyDelete: {}, contextMenu: {}, popover: VIEWER_TOOLBAR_POPOVER_LAYOUT, historyBatch: {} }), "historyDelete");
  assert.equal(viewerEscapeAction({ contextMenu: {}, popover: VIEWER_TOOLBAR_POPOVER_LAYOUT, historyBatch: {} }), "contextMenu");
  assert.equal(viewerEscapeAction({ popover: VIEWER_TOOLBAR_POPOVER_LAYOUT, historyBatch: {} }), "layout");
  assert.equal(viewerEscapeAction({ popover: VIEWER_TOOLBAR_POPOVER_TEMPLATES, historyBatch: {} }), "templates");
  assert.equal(viewerEscapeAction({ historyBatch: {} }), "historyBatch");
  assert.equal(viewerEscapeAction({}), "viewer");
});

test("toolbar source keeps all controls grouped, accessible, session-only, and canvas-sibling popovers", () => {
  const app = fs.readFileSync(new URL("../src/App.jsx", import.meta.url), "utf8");
  const css = fs.readFileSync(new URL("../src/styles.css", import.meta.url), "utf8");
  for (const control of ["viewerGridEnabled", "viewerGridSize", "viewerEdgeSnapEnabled", "viewerAlignmentGuidesEnabled", "viewerEdgeLine.enabled", "图片尺寸调整", "activeViewerLayerItem", "activeCollageSlotItem", "confirmCollage", "saveCollage", "editCollage", "discardCollage"]) assert.match(app, new RegExp(control.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
  assert.match(app, /对齐与线条/);
  assert.match(app, /aria-expanded=\{viewerEdgePanelOpen\}[\s\S]{0,140}aria-controls="viewer-alignment-panel"[\s\S]{0,100}aria-haspopup="dialog"/);
  assert.match(app, /aria-expanded=\{viewerTemplatesOpen\}[\s\S]{0,140}aria-controls="viewer-template-panel"[\s\S]{0,100}aria-haspopup="dialog"/);
  assert.match(app, /id="viewer-alignment-panel"[\s\S]{0,100}role="dialog"[\s\S]{0,80}aria-label="对齐与线条"/);
  assert.match(app, /id="viewer-template-panel"[\s\S]{0,100}role="dialog"[\s\S]{0,80}aria-label="拼图模板"/);
  assert.ok(app.indexOf("viewer-toolbar-popover-backdrop") < app.indexOf("image-viewer-canvas"), "toolbar backdrop is a workspace sibling before the canvas");
  assert.match(app, /openImageViewer[\s\S]{0,900}setViewerEdgePanelOpen\(false\)[\s\S]{0,100}setViewerTemplatesOpen\(false\)/);
  assert.match(app, /closeImageViewer[\s\S]{0,450}setViewerEdgePanelOpen\(false\)[\s\S]{0,100}setViewerTemplatesOpen\(false\)/);
  assert.match(app, /const openViewerContextMenu[\s\S]{0,160}applyViewerToolbarPopoverTransition\("none", "context-menu"\)/);
  assert.match(app, /disabled=\{!viewerLayerResizeEnabled\}/);
  for (const button of ["左边缘对齐", "水平居中", "右边缘对齐", "顶边缘对齐", "垂直居中", "底边缘对齐"]) assert.match(app, new RegExp(button));
  assert.match(css, /\.viewer-toolbar \{ flex-wrap: wrap; overflow-x: visible;/);
  assert.match(css, /\.viewer-toolbar-group \{[^}]*white-space: nowrap/);
  assert.match(css, /\.viewer-toolbar-group\.collage-slot-adjust \{[^}]*flex-wrap: wrap/);
  assert.match(css, /\.viewer-toolbar-subgroup \{[^}]*white-space: nowrap/);
  assert.match(css, /\.collage-slot-adjust \{ max-width: none; overflow: visible;/);
  assert.doesNotMatch(css, /\.viewer-toolbar \{[^}]*overflow-x: hidden/);
  assert.match(css, /\.viewer-toolbar-popover-backdrop \{ position: absolute/);
  assert.match(css, /width: min\(475px, calc\(100% - 16px\)\)/);
});

test("toolbar measured-height state is declared, setter bound to ResizeObserver, and value read into workspace CSS variable", () => {
  const app = fs.readFileSync(new URL("../src/App.jsx", import.meta.url), "utf8");
  assert.equal((app.match(/\[viewerToolbarHeight, setViewerToolbarHeight\]/g) || []).length, 1, "measured-height state declared exactly once");
  assert.match(app, /const \[viewerToolbarHeight, setViewerToolbarHeight\] = useState\(46\);/);
  assert.match(app, /setViewerToolbarHeight\(Math\.ceil\(viewerToolbarRef\.current\?\.getBoundingClientRect\(\)\.height \|\| 46\)\)/);
  assert.match(app, /const observer = new ResizeObserver\(updateHeight\)[\s\S]{0,120}observer\.observe\(viewerToolbarRef\.current\)/);
  assert.match(app, /"--viewer-toolbar-height": `\$\{viewerToolbarHeight\}px`/);
  const uses = (app.match(/\bviewerToolbarHeight\b/g) || []).length;
  const setters = (app.match(/\bsetViewerToolbarHeight\(/g) || []).length;
  assert.ok(uses >= 3, `measured height read into workspace CSS variable and popover tops (${uses} reads)`);
  assert.equal(setters, 1, "measured height setter appears exactly once, inside the ResizeObserver effect");
});
