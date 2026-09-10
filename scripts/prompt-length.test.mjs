import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { normalizeADetailerUnit } from "../src/adetailer-units.js";

const readSource = (file) => readFile(new URL(`../${file}`, import.meta.url), "utf8");

// A prompt has a floor and no ceiling. The backend used to refuse one past 8000 characters
// outright, and several of these fields carried a `maxLength` that stopped the typing before the
// request was ever made. Both are gone, and these tests are what stops either coming back: a cap
// on a prompt field is invisible until someone loses the end of a prompt they were writing.

// Every textarea that holds prompt text, wherever it is edited.
const PROMPT_FIELD_ANCHORS = [
  ["src/App.jsx", "ref={positivePromptRef}"],
  ["src/App.jsx", "ref={negativePromptRef}"],
  ["src/App.jsx", 'updateADetailerUnit(unit.id, { prompt: event.target.value })'],
  ["src/App.jsx", 'updateADetailerUnit(unit.id, { negativePrompt: event.target.value })'],
  ["src/App.jsx", 'update("content", event.target.value)'],
  ["src/ImageToImagePage.jsx", 'update({ positive: event.target.value })'],
  ["src/ImageToImagePage.jsx", 'update({ negative: event.target.value })'],
  ["src/ImageToImagePage.jsx", 'updateUnit(unit.id, { prompt: event.target.value })'],
  ["src/ImageToImagePage.jsx", 'updateUnit(unit.id, { negativePrompt: event.target.value })'],
  ["src/GalleryPage.jsx", "setPositivePrompt(event.target.value)"],
  ["src/GalleryPage.jsx", "setNegativePrompt(event.target.value)"],
  ["src/GalleryPage.jsx", 'setField("positive", event.target.value)'],
  ["src/GalleryPage.jsx", 'setField("negative", event.target.value)'],
];

/** The single JSX element containing an anchor, read back to its opening `<`. */
function elementAround(source, anchor, file) {
  const index = source.indexOf(anchor);
  assert.notEqual(index, -1, `missing anchor in ${file}: ${anchor}`);
  const start = source.lastIndexOf("<", index);
  assert.notEqual(start, -1, `no element opens before the anchor in ${file}: ${anchor}`);
  const end = source.indexOf(">", index);
  assert.notEqual(end, -1, `no element closes after the anchor in ${file}: ${anchor}`);
  return source.slice(start, end + 1);
}

test("no prompt field caps what can be typed into it", async () => {
  const sources = new Map();
  for (const [file] of PROMPT_FIELD_ANCHORS) {
    if (!sources.has(file)) sources.set(file, await readSource(file));
  }
  for (const [file, anchor] of PROMPT_FIELD_ANCHORS) {
    const element = elementAround(sources.get(file), anchor, file);
    assert.ok(element.startsWith("<textarea"), `${file} :: ${anchor} is not the textarea it should be`);
    assert.doesNotMatch(element, /maxLength/i, `${file} :: ${anchor} caps the prompt`);
  }
});

test("the character counters report a count rather than advertising a limit", async () => {
  // They read "1234 / 2000" while nothing in the client enforced 2000 and the server enforced
  // 8000 — so the number shown was not a limit, and the limit that existed was not shown.
  const app = await readSource("src/App.jsx");
  assert.doesNotMatch(app, /\{positive\.length\} \/ \d+/);
  assert.doesNotMatch(app, /\$\{negative\.length\} \/ \d+/);
  assert.match(app, /\{positive\.length\} 字符/);
  assert.match(app, /\$\{negative\.length\} 字符/);
});

test("normalising an ADetailer unit keeps the whole prompt", () => {
  // The normaliser ran on every stored and every submitted unit, and used to `slice(0, 8000)`.
  const prompt = "a lantern in the rain, ".repeat(2000);
  const unit = normalizeADetailerUnit({ prompt, negativePrompt: prompt });
  assert.equal(unit.prompt, prompt);
  assert.equal(unit.negativePrompt, prompt);
  // A non-string is still not a prompt.
  assert.equal(normalizeADetailerUnit({ prompt: 42 }).prompt, "");
});

test("the saved workspace body limit leaves room for the prompt it carries", async () => {
  // The workspace state holds both prompts and every prompt preset, and the client sends it
  // without reading the reply — so a 413 here would stop the workspace persisting in silence.
  const config = await readSource("vite.config.js");
  assert.match(config, /const maximumUiStateBytes = \d+ \* 1024 \* 1024;/);
  assert.match(config, /readJsonRequest\(request, maximumUiStateBytes\)/);
});
