import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { RELEASE_MANAGED_FILES, RELEASE_REQUIRED_FILES } from "./release-package.mjs";

// Licences the AGPL can absorb into a conveyable combined work. Anything outside this set needs
// either a section 7 additional permission or removal, and both are decisions rather than
// oversights — so the suite should stop on one instead of letting it ship.
const ABSORBABLE = new Set([
  "MIT", "ISC", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "0BSD", "CC0-1.0", "Unlicense",
  "MPL-2.0", "Python-2.0", "BlueOak-1.0.0",
  "AGPL-3.0", "AGPL-3.0-only", "AGPL-3.0-or-later",
  "GPL-3.0", "GPL-3.0-only", "GPL-3.0-or-later", "LGPL-3.0-only", "LGPL-3.0-or-later",
]);

// GSAP's Standard License is free of charge but is not free software: it does not permit
// conveying the library under the AGPL. It was removed rather than carved out with an exception,
// and this is the guard against it coming back through an unrelated `npm install`.
const REJECTED = ["gsap"];

const read = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");
const DOC_PAGES = ["index", "setup", "models", "usage", "runtime"];

test("LICENSE is the unmodified AGPL-3.0 text", async () => {
  const license = await read("LICENSE");
  assert.match(license, /^ {20}GNU AFFERO GENERAL PUBLIC LICENSE\n {23}Version 3, 19 November 2007\n/);
  // Section 13 is the whole reason this licence rather than the GPL: a user who reaches a
  // modified copy over the network has to be offered its source.
  assert.match(license, /^ {2}13\. Remote Network Interaction; Use with the GNU General Public License\./m);
  assert.match(license, /Copyright \(C\) 2007 Free Software Foundation, Inc\./);
  assert.equal(license.split("\n").length - 1, 661, "the FSF text is 661 lines; a shorter one has been edited");
});

test("the declared SPDX identifier matches the licence that ships", async () => {
  const manifest = JSON.parse(await read("package.json"));
  assert.equal(manifest.license, "AGPL-3.0-only");
});

test("packaging cannot produce a release without the licence text", async () => {
  // AGPL-3.0 section 4: every copy conveyed carries the licence. Managed alone would only mean
  // "copied when present"; required means a package built without it does not validate.
  assert.ok(RELEASE_MANAGED_FILES.includes("LICENSE"));
  assert.ok(RELEASE_REQUIRED_FILES.includes("LICENSE"));
});

test("both READMEs state the licence and link the text that ships beside them", async () => {
  for (const path of ["README.md", "README.zh-CN.md"]) {
    const readme = await read(path);
    assert.match(readme, /\]\(LICENSE\)/, `${path} links the licence file`);
    assert.match(readme, /AGPL-3\.0/, `${path} names the licence`);
    // Ultralytics is why the choice is not free: its YOLO detector is AGPL-3.0 and the ADetailer
    // path imports it, so a combined work under any other licence would need a commercial grant.
    assert.match(readme, /[Uu]ltralytics/, `${path} records why AGPL-3.0 is the compatible choice`);
  }
});

test("no dependency carries a licence the AGPL cannot absorb", async (t) => {
  const manifest = JSON.parse(await read("package.json"));
  const dependencies = Object.keys({ ...manifest.dependencies, ...manifest.devDependencies });
  for (const name of REJECTED) {
    assert.ok(!dependencies.includes(name), `${name} is not AGPL-compatible and must not be a dependency`);
  }
  let installed;
  try {
    installed = await Promise.all(dependencies.map(async (name) => [name, JSON.parse(await read(`node_modules/${name}/package.json`))]));
  } catch {
    // A bare checkout has no node_modules; the declaration check above still ran.
    return t.skip("node_modules is not installed");
  }
  for (const [name, meta] of installed) {
    const declared = typeof meta.license === "string" ? meta.license : meta.license?.type;
    assert.ok(
      declared && declared.split(/\s+(?:OR|AND)\s+|[()]/).filter(Boolean).some((part) => ABSORBABLE.has(part.trim())),
      `${name} declares "${declared}", which needs an AGPL-3.0 section 7 permission or removal`,
    );
  }
});

test("the entrance animation uses the platform rather than a non-free library", async () => {
  const source = await read("src/LoraManagerPage.jsx");
  // Prose may still name what the timings were ported from; what must not come back is a use.
  assert.doesNotMatch(source, /from\s+["']gsap["']|require\(["']gsap["']\)|\bgsap\s*\./i);
  assert.match(source, /from "\.\/entrance-animation\.js"/);
});

test("every documentation page carries the licence in its footer", async () => {
  for (const page of DOC_PAGES) {
    for (const path of [`docs/${page}.html`, `docs/zh/${page}.html`]) {
      const html = await read(path);
      const footer = html.slice(html.indexOf('<footer class="page-footer">'));
      assert.match(
        footer.slice(0, footer.indexOf("</footer>")),
        /href="https:\/\/github\.com\/Xiaollull\/Xiria-AICanvas\/blob\/main\/LICENSE"/,
        `${path} links the licence`,
      );
    }
  }
});
