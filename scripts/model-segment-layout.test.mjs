import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const read = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("all six model engines occupy a balanced three-by-two segment", async () => {
  const [app, css] = await Promise.all([read("src/App.jsx"), read("src/styles.css")]);
  const modelBlock = app.slice(app.indexOf("const models = ["), app.indexOf("];", app.indexOf("const models = [")));
  assert.equal((modelBlock.match(/name: /g) || []).length, 6, "the engine list has six entries");
  assert.match(css, /\.model-segment \{[^}]*grid-template-columns: repeat\(3, minmax\(0, 1fr\)\)/);
  assert.match(css, /\.model-seg \{[^}]*border-right: 1px solid var\(--line\);[^}]*border-bottom: 1px solid var\(--line\)/);
  assert.match(css, /\.model-seg:nth-child\(3n\) \{ border-right: 0; \}/);
  assert.match(css, /\.model-seg:nth-last-child\(-n \+ 3\) \{ border-bottom: 0; \}/);
});
