// Floating windows stack like desktop windows: the last one opened, clicked or dragged is on top.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { FLOATING_WINDOW_BASE_Z, floatingWindowZIndex, raiseInStack, removeFromStack } from "../src/floating-windows.js";

test("raising moves a window to the top and leaves the others in their order", () => {
  let stack = [];
  stack = raiseInStack(stack, "queue");
  stack = raiseInStack(stack, "presets");
  assert.deepEqual(stack, ["queue", "presets"]);
  assert.ok(floatingWindowZIndex(stack, "presets") > floatingWindowZIndex(stack, "queue"));
  stack = raiseInStack(stack, "queue");
  assert.deepEqual(stack, ["presets", "queue"]);
  assert.ok(floatingWindowZIndex(stack, "queue") > floatingWindowZIndex(stack, "presets"));
  // Raising the window already on top changes nothing, so no re-render is published.
  assert.equal(raiseInStack(stack, "queue"), stack);
  stack = removeFromStack(stack, "presets");
  assert.deepEqual(stack, ["queue"]);
  assert.equal(floatingWindowZIndex(stack, "queue"), FLOATING_WINDOW_BASE_Z);
});

test("the layers stay below dialogs, menus and hover previews", async () => {
  const css = await readFile(new URL("../src/styles.css", import.meta.url), "utf8");
  const layer = (name) => Number(new RegExp(`--z-${name}:\\s*(\\d+)`).exec(css)?.[1]);
  assert.equal(layer("popover"), FLOATING_WINDOW_BASE_Z);
  for (const name of ["modal", "hover", "notice"]) assert.ok(floatingWindowZIndex(["queue", "presets"], "presets") < layer(name), name);
});

test("both floating windows take their layer from the shared stack and raise on press or focus", async () => {
  const app = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
  const presets = await readFile(new URL("../src/PromptPresetWindow.jsx", import.meta.url), "utf8");
  assert.match(app, /useFloatingWindowLayer\("queue"\)/);
  assert.match(presets, /useFloatingWindowLayer\("presets"\)/);
  for (const source of [app, presets]) {
    assert.match(source, /zIndex: layer\.zIndex/);
    assert.match(source, /\{\.\.\.layer\.bind\}/);
  }
});
