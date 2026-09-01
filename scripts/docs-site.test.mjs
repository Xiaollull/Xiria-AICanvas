import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { dirname, extname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const docsRoot = resolve(root, "docs");
const pages = [
  "index.html",
  "setup.html",
  "models.html",
  "usage.html",
  "runtime.html",
  "zh/index.html",
  "zh/setup.html",
  "zh/models.html",
  "zh/usage.html",
  "zh/runtime.html",
];

function read(relativePath) {
  return readFileSync(resolve(docsRoot, relativePath), "utf8");
}

function localReferences(source) {
  return [...source.matchAll(/\b(?:href|src)="([^"]+)"/g)]
    .map((match) => match[1])
    .filter((value) => !/^(?:[a-z]+:|\/\/|#)/i.test(value));
}

test("every documentation page keeps its local links and assets resolvable", () => {
  for (const page of pages) {
    const sourcePath = resolve(docsRoot, page);
    assert.ok(existsSync(sourcePath), `${page} must exist`);

    for (const reference of localReferences(read(page))) {
      const [target, fragment] = reference.split("#", 2);
      const targetPath = resolve(dirname(sourcePath), target);
      assert.ok(existsSync(targetPath), `${page} references missing ${reference}`);

      if (fragment && extname(targetPath).toLowerCase() === ".html") {
        const targetSource = readFileSync(targetPath, "utf8");
        const escaped = fragment.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        assert.match(targetSource, new RegExp(`\\bid="${escaped}"`), `${reference} must resolve to an id`);
      }
    }
  }
});

test("English and Chinese home pages keep the same portal structure", () => {
  const english = read("index.html");
  const chinese = read("zh/index.html");
  const requiredClasses = [
    "home-page",
    "home-hero",
    "hero-runtime",
    "principle-list",
    "command-panel",
    "engine-lanes",
    "manual-grid",
    "wiki-callout",
  ];

  for (const className of requiredClasses) {
    assert.match(english, new RegExp(`\\b${className}\\b`));
    assert.match(chinese, new RegExp(`\\b${className}\\b`));
  }

  for (const source of [english, chinese]) {
    assert.match(source, /npm run setup/);
    assert.match(source, /npm start/);
    assert.match(source, /<\/code><\/pre>/);
    assert.equal((source.match(/class="home-section/g) || []).length, 4);
  }
});

test("documentation styling preserves the violet brand and responsive contracts", () => {
  const css = read("assets/docs.css");

  assert.match(css, /--accent-brand:\s*#8b7cff/i);
  assert.match(css, /@media \(prefers-color-scheme: dark\)/);
  assert.match(css, /@media \(max-width: 900px\)/);
  assert.match(css, /@media \(max-width: 520px\)/);
  assert.match(css, /@media \(prefers-reduced-motion: reduce\)/);
  assert.match(css, /\.home-page \.sidebar/);
  assert.doesNotMatch(css, /#b7ff3c|acid[- ]green|lime-green/i);
});
