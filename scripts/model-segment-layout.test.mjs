import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const read = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("all seven model engines fill a three-column segment with no cell left alone", async () => {
  const [app, css] = await Promise.all([read("src/App.jsx"), read("src/styles.css")]);
  const modelBlock = app.slice(app.indexOf("const models = ["), app.indexOf("];", app.indexOf("const models = [")));
  const engines = (modelBlock.match(/name: /g) || []).length;
  assert.equal(engines, 7, "the engine list has seven entries");
  assert.match(css, /\.model-segment \{[^}]*grid-template-columns: repeat\(3, minmax\(0, 1fr\)\)/);
  assert.match(css, /\.model-seg \{[^}]*border-right: 1px solid var\(--line\);[^}]*border-bottom: 1px solid var\(--line\)/);
  assert.match(css, /\.model-seg:nth-child\(3n\), \.model-seg:last-child \{ border-right: 0; \}/);
  // Only the cells of the real last row lose their bottom border, whatever its length.
  assert.match(css, /\.model-seg:nth-child\(3n \+ 1\):nth-last-child\(-n \+ 3\), \.model-seg:nth-child\(3n \+ 1\):nth-last-child\(-n \+ 3\) ~ \.model-seg \{ border-bottom: 0; \}/);
  // A short last row is filled: seven engines leave MOSS alone, so it spans the row.
  assert.match(css, /\.model-seg:nth-child\(3n \+ 1\):last-child \{ grid-column: 1 \/ -1; \}/);
  assert.match(css, /\.model-seg:nth-child\(3n \+ 2\):last-child \{ grid-column: span 2; \}/);
  // The two-column WD14 tagger grid keeps its own cells in place.
  assert.match(css, /\.wd14-page \.wd14-engine-grid \.model-seg:last-child \{ grid-column: auto; \}/);
  // On a phone the XYZ sidebar shows the engines as one strip, one column per engine.
  assert.match(css, new RegExp(`\\.workspace\\.xyz-mode \\.xyz-sidebar \\.model-segment \\{ grid-template-columns: repeat\\(${engines}, minmax\\(0, 1fr\\)\\);`));
  assert.match(css, /\.workspace\.xyz-mode \.xyz-sidebar \.model-seg \{[^}]*grid-column: auto; border-bottom: 0; \}/);
});
