import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

import { RELEASE_NOTES_DELIMITER, RELEASE_NOTES_DIRECTORY, RELEASE_NOTES_MAX_BYTES, readReleaseNotes } from "./release-notes.mjs";

const projectDirectory = path.resolve(import.meta.dirname, "..");

async function withNotes(contents, run) {
  // Under node_modules, like the other fixtures, so no Windows 8.3 temp path is involved.
  const root = await mkdtemp(path.join(projectDirectory, "node_modules", ".release-notes-"));
  try {
    if (contents !== null) {
      await mkdir(path.join(root, RELEASE_NOTES_DIRECTORY), { recursive: true });
      await writeFile(path.join(root, RELEASE_NOTES_DIRECTORY, "v9.8.7.md"), contents);
    }
    return await run(root);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}

test("a release without notes publishes as before", async () => {
  await withNotes(null, async (root) => assert.equal(await readReleaseNotes(root, "v9.8.7"), ""));
});

test("notes are trimmed, LF-only, and must name the version they describe", async () => {
  await withNotes("\r\n# XiriaCanvas AI 9.8.7\r\n\r\n- one\r\n", async (root) => {
    assert.equal(await readReleaseNotes(root, "v9.8.7"), "# XiriaCanvas AI 9.8.7\n\n- one");
  });
  await withNotes("# XiriaCanvas AI 9.8.6\n\n- copied from the last release", async (root) => {
    await assert.rejects(readReleaseNotes(root, "v9.8.7"), /must start with "# XiriaCanvas AI 9\.8\.7"/);
  });
  await withNotes("   \n", async (root) => {
    await assert.rejects(readReleaseNotes(root, "v9.8.7"), /empty/);
  });
});

test("notes cannot break out of the workflow output or outgrow a release body", async () => {
  await withNotes(`# XiriaCanvas AI 9.8.7\n${RELEASE_NOTES_DELIMITER}\nversion=0.0.0`, async (root) => {
    await assert.rejects(readReleaseNotes(root, "v9.8.7"), /delimiter/);
  });
  await withNotes(`# XiriaCanvas AI 9.8.7\n${"x".repeat(RELEASE_NOTES_MAX_BYTES)}`, async (root) => {
    await assert.rejects(readReleaseNotes(root, "v9.8.7"), /larger than/);
  });
  await assert.rejects(readReleaseNotes(projectDirectory, "v1.2"), /stable tags/);
  await assert.rejects(readReleaseNotes(projectDirectory, "../v1.2.0"), /stable tags/);
});

test("every notes file in the repository is publishable under its own tag", async () => {
  const { readdir } = await import("node:fs/promises");
  const directory = path.join(projectDirectory, RELEASE_NOTES_DIRECTORY);
  let names = [];
  try {
    names = await readdir(directory);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
  for (const name of names) {
    assert.match(name, /^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.md$/, `${name} is not named after a stable tag`);
    const notes = await readReleaseNotes(projectDirectory, name.slice(0, -3));
    assert.ok(notes.length > 0, name);
  }
});

test("the release workflow validates the notes early and places them above the generated list", async () => {
  const workflow = await readFile(path.join(projectDirectory, ".github", "workflows", "release.yml"), "utf8");
  assert.match(workflow, /import \{ RELEASE_NOTES_DELIMITER, readReleaseNotes \} from "\.\/scripts\/release-notes\.mjs";/);
  assert.match(workflow, /const releaseNotes = await readReleaseNotes\(process\.cwd\(\), tag\);/);
  assert.match(workflow, /release_notes: \$\{\{ steps\.version\.outputs\.release_notes \}\}/);
  // The publish job checks out nothing, so the notes reach it through the environment, never
  // through an expression interpolated into the script.
  assert.match(workflow, /RELEASE_NOTES: \$\{\{ needs\.verify-and-package\.outputs\.release_notes \}\}/);
  assert.doesNotMatch(workflow, /\$\{\{[^}]*release_notes[^}]*\}\}[^\n]*\n[^\n]*node --input-type/);
  assert.match(workflow, /const releaseBody = \[notice, releaseNotes, notes\.body\]\.filter\(Boolean\)\.join\("\\n\\n"\);/);
  assert.match(workflow, /body: releaseBody/);
});
