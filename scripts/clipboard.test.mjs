import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { copyText } from "../src/clipboard.js";

// `navigator.clipboard` exists only in a secure context. Opened over plain HTTP from another machine
// -- a LAN address, or a cloud GPU instance -- it is undefined, and the image reader's copy buttons,
// which called it through `?.`, silently did nothing. These tests drive the helper that replaced
// them through both the secure path and the fallback.

function fakeDocument({ execResult = true, execThrows = false } = {}) {
  const calls = { appended: 0, removed: 0, selected: 0, commands: [], refocused: false };
  const opener = { isConnected: true, focus() { calls.refocused = true; } };
  let active = opener;
  const document = {
    body: { appendChild() { calls.appended += 1; } },
    get activeElement() { return active; },
    createElement(tag) {
      return {
        tag, value: "", style: {}, attributes: {},
        setAttribute(name, value) { this.attributes[name] = value; },
        focus() { active = this; },
        select() { calls.selected += 1; },
        setSelectionRange() {},
        remove() { calls.removed += 1; },
      };
    },
    execCommand(command) {
      calls.commands.push(command);
      if (execThrows) throw new Error("not allowed");
      return execResult;
    },
  };
  return { document, calls };
}

async function withGlobals(values, run) {
  const saved = new Map();
  for (const [key, value] of Object.entries(values)) {
    saved.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { value, configurable: true, writable: true });
  }
  try {
    return await run();
  } finally {
    for (const [key, descriptor] of saved) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  }
}

test("over plain HTTP the text is still copied, through the selection fallback", async () => {
  // The reported bug: no secure context, so no `navigator.clipboard`, so nothing happened.
  const { document, calls } = fakeDocument();
  const copied = await withGlobals({ isSecureContext: false, document, navigator: {} }, () => copyText("anime coloring, pastel"));
  assert.equal(copied, true);
  assert.deepEqual(calls.commands, ["copy"]);
  assert.equal(calls.appended, 1);
  assert.equal(calls.removed, 1, "the helper textarea must not be left in the page");
  assert.equal(calls.refocused, true, "focus goes back to the button, so a dialog keeps its focus trap");
});

test("in a secure context the clipboard API is used and the page is not touched", async () => {
  const written = [];
  const { document, calls } = fakeDocument();
  const navigator = { clipboard: { writeText: async (text) => { written.push(text); } } };
  const copied = await withGlobals({ isSecureContext: true, document, navigator }, () => copyText("seed 758167733102762"));
  assert.equal(copied, true);
  assert.deepEqual(written, ["seed 758167733102762"]);
  assert.deepEqual(calls.commands, []);
});

test("a refused clipboard permission still falls back rather than failing", async () => {
  const { document, calls } = fakeDocument();
  const navigator = { clipboard: { writeText: async () => { throw new Error("NotAllowedError"); } } };
  const copied = await withGlobals({ isSecureContext: true, document, navigator }, () => copyText("x"));
  assert.equal(copied, true);
  assert.deepEqual(calls.commands, ["copy"]);
});

test("a copy that cannot happen reports false, so the button can say so", async () => {
  for (const options of [{ execResult: false }, { execThrows: true }]) {
    const { document, calls } = fakeDocument(options);
    const copied = await withGlobals({ isSecureContext: false, document, navigator: {} }, () => copyText("x"));
    assert.equal(copied, false, JSON.stringify(options));
    assert.equal(calls.removed, 1, "the textarea is removed even when the copy fails");
  }
});

test("nothing to copy touches nothing", async () => {
  const { document, calls } = fakeDocument();
  const copied = await withGlobals({ isSecureContext: false, document, navigator: {} }, () => copyText(""));
  assert.equal(copied, false);
  assert.equal(calls.appended, 0);
});

test("no copy button reaches for the clipboard API directly any more", async () => {
  for (const file of ["ImageInfoFields.jsx", "GalleryPage.jsx", "LoraDetailsDialog.jsx"]) {
    const source = await readFile(new URL(`../src/${file}`, import.meta.url), "utf8");
    assert.match(source, /import \{ copyText \} from "\.\/clipboard";/, `${file} does not use the shared helper`);
    assert.doesNotMatch(source, /navigator\.clipboard/, `${file} calls the clipboard API itself`);
    assert.doesNotMatch(source, /async function copyText/, `${file} still carries a private copy`);
  }
  const fields = await readFile(new URL("../src/ImageInfoFields.jsx", import.meta.url), "utf8");
  // A failed copy is visible instead of silent.
  assert.match(fields, /"复制失败"/);
});
