// MOSS coordinate layouts as the browser edits them (QPlan2/MOSS/P2): the compiler mirror, the editor state,
// geometry and import. The authoritative side is backend/tests/test_moss_layout.py; the parity fixture both
// suites assert is written by scripts/moss-layout-parity.mjs.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  MOSS_LAYOUT_README_DRAFT,
  MOSS_LAYOUT_TEMPLATES,
  alignMossLayoutItems,
  boxFromDrag,
  centredBox,
  clampBox,
  compileMossLayout,
  compileMossLayoutDraft,
  emptyMossLayout,
  layoutTextProblem,
  moveBox,
  mossLayoutDraft,
  mossLayoutDraftIssues,
  mossLayoutFromDraft,
  mossLayoutFromRecord,
  mossLayoutFromTemplate,
  mossLayoutIssues,
  mossLayoutPayload,
  mossLayoutStats,
  normalizeMossLayout,
  parseMossLayoutImport,
  pyStrip,
  rescaleMossLayout,
  resizeBox,
  snapTargets,
  snapValue,
} from "../src/moss-layout.js";
import { PARITY_FILE, buildParityFixture } from "./moss-layout-parity.mjs";

const README_COMPILED = '{"summary":{"meta":["poster"],"canvas":"single","caption":"A minimalist travel poster with a cream background, a blue mountain illustration, and the title EXPLORE at the top."},"items":[{"label":"text","caption":"Large dark-blue uppercase title centered at the top.","text":"EXPLORE","bbox":[-416,-384,-288,384]},{"label":"object","caption":"A flat blue mountain illustration in the lower center.","bbox":[-192,-288,384,288]}],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}';

test("L1 the parity fixture is current: src/moss-layout.js and backend/moss_layout.py compile the same bytes", async () => {
  const stored = JSON.parse(await readFile(PARITY_FILE, "utf8"));
  assert.deepEqual(stored, JSON.parse(JSON.stringify(buildParityFixture())), "run `node scripts/moss-layout-parity.mjs --write` after changing moss-layout.js");
  assert.ok(stored.cases.length >= 25);
  assert.ok(stored.cases.some((entry) => entry.prompt === null) && stored.cases.some((entry) => entry.prompt?.includes("\n{")));
});

test("L1 the README layout and the official vectors are reproduced exactly", () => {
  assert.equal(compileMossLayoutDraft(MOSS_LAYOUT_README_DRAFT), README_COMPILED);
  assert.deepEqual(centredBox([928, 272, 1408, 448], 1536, 1024), [-240, 160, -64, 640]);
  assert.deepEqual(centredBox([0, 0, 768, 1344], 768, 1344), [-672, -384, 672, 384]);
  assert.deepEqual(centredBox([10.9, 20.1, 100.01, 200.99], 768, 1344), [-652, -374, -471, -283]);
  assert.deepEqual(centredBox([1.00000000001, 2, 100, 200], 768, 1344), [-670, -383, -472, -284]);
  assert.equal(compileMossLayoutDraft(MOSS_LAYOUT_README_DRAFT, "  @nyalia, best quality \n"), `@nyalia, best quality\n${README_COMPILED}`);
});

test("L1 text rules match the server: blank, control characters and lone surrogates are refused", () => {
  assert.equal(layoutTextProblem("A fox."), "");
  assert.equal(layoutTextProblem("第一行\n第二行\t🦊"), "");
  for (const value of ["", " 　\u0085 ", "a\u0001b", "a\rb", "a\ud83db", "a\udc00", 5]) assert.notEqual(layoutTextProblem(value), "", JSON.stringify(value));
  assert.equal(pyStrip(" 　x\u0085 "), "x");
  assert.equal(pyStrip("﻿x"), "﻿x", "U+FEFF is not whitespace to Python");
});

test("L1 issues name their path, and the editor maps them back to its items", () => {
  const draft = { width: 1024, height: 1024, bbox_coord_space: "abs", bbox_format: "xyxy", summary: { canvas: "grid" },
    items: [{ label: "subject", caption: "", bbox: [0, 0, 2000, 10] }, { label: "text", caption: "ok", bbox: [0, 0, 8, 8] }] };
  assert.deepEqual(mossLayoutDraftIssues(draft).map((issue) => issue.path),
    ["summary.caption", "summary.canvas", "items[0].label", "items[0].caption", "items[0].bbox", "items[1].text"]);
  const layout = { ...emptyMossLayout(512, 512), summary: { caption: "Room.", meta: [], canvas: "single", palette: [] },
    items: [{ id: "a", label: "object", caption: "Lamp", text: "", tags: [], palette: [], box: [0, 0, 64, 64], hidden: true, locked: false },
      { id: "b", label: "text", caption: "Sign", text: "", tags: [], palette: [], box: [64, 64, 256, 128], hidden: false, locked: false }] };
  const issues = mossLayoutIssues(layout);
  assert.deepEqual(issues.map((issue) => [issue.path, issue.itemId]), [["items[0].text", "b"]], "hidden items are not compiled, so indices follow the visible ones");
});

test("L2 the editor state compiles to the official draft in canonical order, without editor fields", () => {
  const layout = normalizeMossLayout({
    enabled: true, width: 1024, height: 1024,
    summary: { caption: "Poster.", meta: ["poster", " "], canvas: "single", palette: [] },
    items: [
      { id: "x", label: "text", caption: "Title", text: "EXPLORE", tags: ["title"], palette: ["#123456"], box: [128, 96, 896, 224], locked: true },
      { id: "y", label: "object", caption: "Mountain", text: "stale text from an earlier label", box: [224, 320, 800, 896] },
      { id: "z", label: "panel", caption: "Hidden frame", box: [0, 0, 1024, 1024], hidden: true },
    ],
  });
  const draft = mossLayoutDraft(layout);
  assert.deepEqual(Object.keys(draft.summary), ["meta", "canvas", "caption"]);
  assert.deepEqual(draft.summary.meta, ["poster"]);
  assert.deepEqual(draft.items.map((item) => Object.keys(item)), [["label", "caption", "text", "tags", "palette", "bbox"], ["label", "caption", "bbox"]]);
  const { prompt, issues } = compileMossLayout(layout, "@nyalia");
  assert.deepEqual(issues, []);
  assert.ok(prompt.startsWith("@nyalia\n{\"summary\""));
  assert.deepEqual(mossLayoutPayload(layout, "@nyalia"), { version: 1, prefix: "@nyalia", draft });
  assert.equal(compileMossLayout(emptyMossLayout(1024, 1024)).prompt, null);
});

test("L2 stored layouts are repaired field by field and follow the canvas", () => {
  const repaired = normalizeMossLayout({ enabled: "yes", width: 512, height: 512, summary: { caption: 4, canvas: "grid", meta: ["ok", 3] },
    items: [{ label: "dragon", caption: "x\r\ny", box: [-20, 10, 900, 20] }, "junk"] });
  assert.equal(repaired.enabled, false);
  assert.equal(repaired.summary.caption, "");
  assert.equal(repaired.summary.canvas, "");
  assert.deepEqual(repaired.summary.meta, ["ok"]);
  assert.equal(repaired.items[0].label, "object");
  assert.equal(repaired.items[0].caption, "x\ny");
  assert.deepEqual(repaired.items[0].box, [0, 10, 512, 20]);
  assert.equal(repaired.items.length, 2);
  const moved = normalizeMossLayout({ width: 512, height: 512, items: [{ box: [128, 128, 256, 256] }] }, { width: 1024, height: 768 });
  assert.deepEqual([moved.width, moved.height, moved.items[0].box], [1024, 768, [256, 192, 512, 384]]);
  assert.deepEqual(normalizeMossLayout(null, { width: 640, height: 448 }).items, []);
});

test("L2 geometry: clamp, move, resize from every handle, drag and snap stay inside the canvas", () => {
  assert.deepEqual(clampBox([-10, -10, 5, 5], 100, 100), [0, 0, 16, 16]);
  assert.deepEqual(clampBox([90, 90, 200, 200], 100, 100), [0, 0, 100, 100], "a box larger than the canvas becomes the canvas");
  assert.deepEqual(clampBox([95, 95, 100, 100], 100, 100), [84, 84, 100, 100]);
  assert.deepEqual(moveBox([10, 10, 50, 50], 100, -100, 64, 64), [24, 0, 64, 40]);
  assert.deepEqual(resizeBox([10, 10, 50, 50], "br", 30, 30, 64, 64), [10, 10, 64, 64]);
  assert.deepEqual(resizeBox([10, 10, 50, 50], "tl", 100, 100, 64, 64), [34, 34, 50, 50]);
  assert.deepEqual(resizeBox([10, 10, 50, 50], "left", -50, 0, 64, 64), [0, 10, 50, 50]);
  assert.deepEqual(resizeBox([0, 0, 100, 50], "br", 100, 0, 1000, 1000, { keepRatio: true }), [0, 0, 200, 100]);
  assert.deepEqual(boxFromDrag({ x: 80, y: 60 }, { x: 20, y: 10 }, 100, 100), [20, 10, 80, 60]);
  assert.deepEqual(snapValue(37, [32, 48], 6), { value: 32, target: 32 });
  assert.deepEqual(snapValue(40, [32, 48], 6), { value: 40, target: null });
  const layout = { ...emptyMossLayout(64, 64), items: [{ id: "a", box: [10, 20, 30, 40], hidden: false }, { id: "b", box: [1, 2, 3, 4], hidden: true }] };
  const targets = snapTargets(layout, "x", "none", { useGrid: false });
  assert.deepEqual(targets.sort((a, b) => a - b), [0, 10, 20, 30, 32, 64]);
});

test("L2 rescaling and alignment keep every box whole and inside", () => {
  const layout = mossLayoutFromDraft(MOSS_LAYOUT_README_DRAFT);
  const scaled = rescaleMossLayout(layout, 2048, 1024);
  assert.deepEqual(scaled.items.map((item) => item.box), [[256, 96, 1792, 224], [448, 320, 1600, 896]]);
  assert.equal(rescaleMossLayout(layout, 1024, 1024), layout);
  const aligned = alignMossLayoutItems(layout, [layout.items[1].id], "top");
  assert.deepEqual(aligned.items[1].box, [224, 0, 800, 576]);
  const both = alignMossLayoutItems(layout, layout.items.map((item) => item.id), "left");
  assert.deepEqual(both.items.map((item) => item.box[0]), [128, 128]);
});

test("L2 import: readable drafts, official compiled layouts with a prefix line, and scene graphs refused", () => {
  const readable = parseMossLayoutImport(JSON.stringify(MOSS_LAYOUT_README_DRAFT), 2048, 2048);
  assert.equal(readable.layout.width, 2048);
  assert.match(readable.note, /缩放/);
  const compiled = parseMossLayoutImport(`@nyalia, best quality\n${README_COMPILED}`, 1024, 1024);
  assert.equal(compiled.prefix, "@nyalia, best quality");
  assert.equal(compileMossLayout(compiled.layout).prompt, README_COMPILED, "a compiled layout round-trips exactly at its canvas");
  assert.match(compiled.note, /前缀/);
  const small = parseMossLayoutImport(README_COMPILED, 512, 512);
  assert.match(small.note, /超出/);
  assert.match(parseMossLayoutImport('{"scene_type":"x","elements":[]}', 1024, 1024).error, /场景图/);
  assert.match(parseMossLayoutImport("not json {", 1024, 1024).error, /无法解析/);
  assert.match(parseMossLayoutImport("plain words", 1024, 1024).error, /没有找到/);
  assert.match(parseMossLayoutImport('{"items":[],"bbox_format":"xyxy","bbox_coord_space":"abs_centralized"}', 1024, 1024).error, /bbox_format/);
});

test("L2 templates and PNG records restore editor state", () => {
  assert.deepEqual(MOSS_LAYOUT_TEMPLATES.map((template) => template.id), ["blank", "readme-poster"]);
  const poster = mossLayoutFromTemplate("readme-poster", 1536, 1024);
  assert.equal(poster.enabled, true);
  assert.deepEqual(poster.items[0].box, [192, 96, 1344, 224]);
  assert.equal(mossLayoutFromTemplate("blank", 640, 448).items.length, 0);
  assert.equal(mossLayoutFromTemplate("missing", 640, 448), null);
  const restored = mossLayoutFromRecord({ version: 1, prefix: "@nyalia", draft: MOSS_LAYOUT_README_DRAFT, stages: [] });
  assert.equal(restored.prefix, "@nyalia");
  assert.equal(compileMossLayout(restored.layout).prompt, README_COMPILED);
  assert.equal(mossLayoutFromRecord({ version: 2, draft: {} }), null);
  assert.deepEqual(mossLayoutStats(poster).counts, { object: 1, character: 0, text: 1, panel: 0 });
});
