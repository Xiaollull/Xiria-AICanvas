import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  FULL_PACKAGE_DECLARATION,
  FULL_PACKAGE_NOTICE,
  assertFullPackageJustified,
  isFullPackageRelease,
  nodeDependenciesChanged,
  readFullPackageDeclaration,
} from "./full-package-release.mjs";

const projectRoot = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const read = (relative) => readFile(path.join(projectRoot, relative), "utf8");

async function withRoot(run) {
  const root = await mkdtemp(path.join(tmpdir(), "full-package-"));
  try {
    return await run(root);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}

const declare = (root, contents) => writeFile(path.join(root, "declaration"), contents).then(async () => {
  await mkdir(path.join(root, ".github"), { recursive: true });
  await writeFile(path.join(root, FULL_PACKAGE_DECLARATION), contents);
});

/** A project tree carrying only the two files the dependency decision is made from. */
async function environment(root, name, { lock, requirements = "torch==2.9.0\n" }) {
  const target = path.join(root, name);
  await mkdir(path.join(target, "backend"), { recursive: true });
  await writeFile(path.join(target, "package-lock.json"), JSON.stringify(lock, null, 2));
  await writeFile(path.join(target, "backend", "requirements.txt"), requirements);
  return target;
}

const lockWith = (dependencies) => ({
  name: "xirai",
  lockfileVersion: 3,
  packages: {
    "": { name: "xirai", version: "1.0.4", dependencies, devDependencies: { vite: "^8.0.0" } },
    ...Object.fromEntries(Object.keys(dependencies).map((key) => [`node_modules/${key}`, { version: "1.0.0" }])),
  },
});

test("no declaration means the release is gated the usual way", async () => {
  await withRoot(async (root) => {
    assert.equal(await readFullPackageDeclaration(root), null);
    assert.equal(await isFullPackageRelease(root, "1.0.4"), false);
  });
});

test("a declaration applies only to the version it names", async () => {
  await withRoot(async (root) => {
    await declare(root, "1.0.4\n");
    assert.equal(await isFullPackageRelease(root, "1.0.4"), true);
    // The file outliving its release must not weaken the next one's gate, which is why it names a
    // version instead of being a flag.
    assert.equal(await isFullPackageRelease(root, "1.0.5"), false);
  });
});

test("comments carry the reason without becoming the version", async () => {
  await withRoot(async (root) => {
    await declare(root, "# GSAP removed for the AGPL move; node_modules changes.\n\n1.0.4\n");
    assert.equal(await readFullPackageDeclaration(root), "1.0.4");
  });
});

test("an unreadable declaration is an error, not an absent one", async () => {
  await withRoot(async (root) => {
    await declare(root, "# only a comment\n");
    await assert.rejects(() => readFullPackageDeclaration(root), /exactly one version/);
    await declare(root, "1.0.4\n1.0.5\n");
    await assert.rejects(() => readFullPackageDeclaration(root), /exactly one version/);
    for (const bad of ["v1.0.4\n", "1.0\n", "1.0.04\n", "latest\n"]) {
      await declare(root, bad);
      await assert.rejects(() => readFullPackageDeclaration(root), /strict MAJOR\.MINOR\.PATCH/);
    }
  });
});

test("a dependency change is what the updater refuses, and what this detects", async () => {
  await withRoot(async (root) => {
    const installed = await environment(root, "installed", { lock: lockWith({ gsap: "3.15.0", react: "19.0.0" }) });
    const same = await environment(root, "same", { lock: lockWith({ gsap: "3.15.0", react: "19.0.0" }) });
    const dropped = await environment(root, "dropped", { lock: lockWith({ react: "19.0.0" }) });
    assert.equal(await nodeDependenciesChanged(installed, same), false);
    assert.equal(await nodeDependenciesChanged(installed, dropped), true);
  });
});

test("a Python requirement change alone is not a full-package release", async () => {
  // The updater replaces those files and repairs the environment afterwards, so it stays an
  // in-place update and must keep being rehearsed as one.
  await withRoot(async (root) => {
    const lock = lockWith({ react: "19.0.0" });
    const installed = await environment(root, "installed", { lock, requirements: "torch==2.9.0\n" });
    const candidate = await environment(root, "candidate", { lock, requirements: "torch==2.10.0\n" });
    assert.equal(await nodeDependenciesChanged(installed, candidate), false);
  });
});

test("a missing lockfile is an error rather than evidence of a change", async () => {
  // `dependencyChangesRequired` reports an unreadable file as a dependency change, which is right
  // for an updater facing a damaged install and wrong here: it would let a missing file excuse the
  // release from the rehearsal.
  await withRoot(async (root) => {
    const installed = await environment(root, "installed", { lock: lockWith({ react: "19.0.0" }) });
    const empty = path.join(root, "empty");
    await mkdir(empty, { recursive: true });
    await assert.rejects(() => nodeDependenciesChanged(installed, empty), { code: "ENOENT" });
  });
});

test("declaring a release that could have updated in place is itself the error", async () => {
  await withRoot(async (root) => {
    const lock = lockWith({ react: "19.0.0" });
    const installed = await environment(root, "installed", { lock });
    const candidate = await environment(root, "candidate", { lock });
    await assert.rejects(
      () => assertFullPackageJustified({ installedRoot: installed, packageRoot: candidate, version: "1.0.4" }),
      /can update in place/,
    );
  });
});

test("a genuine dependency change satisfies the declaration", async () => {
  await withRoot(async (root) => {
    const installed = await environment(root, "installed", { lock: lockWith({ gsap: "3.15.0", react: "19.0.0" }) });
    const candidate = await environment(root, "candidate", { lock: lockWith({ react: "19.0.0" }) });
    await assertFullPackageJustified({ installedRoot: installed, packageRoot: candidate, version: "1.0.4" });
  });
});

test("this release declares itself a full package, and says why", async () => {
  const declared = await readFullPackageDeclaration(projectRoot);
  const manifest = JSON.parse(await read("package.json"));
  assert.equal(declared, manifest.version, "the declaration names the version being released");
  const contents = await read(FULL_PACKAGE_DECLARATION);
  assert.match(contents, /^#/m, "a reviewer reading the file learns why the release needs it");
  assert.match(contents, /GSAP/i);
});

test("the release workflow reads the declaration and refuses to publish without its notice", async () => {
  const workflow = await read(path.join(".github", "workflows", "release.yml"));
  assert.match(workflow, /import \{ FULL_PACKAGE_NOTICE, isFullPackageRelease \} from "\.\/scripts\/full-package-release\.mjs"/);
  assert.match(workflow, /full_package=\$\{fullPackage\}/);
  assert.match(workflow, /assertFullPackageJustified/, "the declaration is checked, not merely read");
  // The in-place rehearsal is skipped only for a declared release, and only that part of it.
  assert.match(workflow, /if \(process\.env\.PREVIOUS_FOUND === "true" && !fullPackage\)/);
  // The publish job holds the only write token and checks out nothing, so the notice reaches it as
  // an output; publishing with the two out of step is refused.
  assert.match(workflow, /full_package_notice: \$\{\{ steps\.version\.outputs\.full_package_notice \}\}/);
  assert.match(workflow, /refusing to publish/);
  assert.match(workflow, /body: releaseBody/);
});

test("the notice tells a reader what to do and what survives", async () => {
  assert.match(FULL_PACKAGE_NOTICE, /requires a fresh installation/i);
  assert.match(FULL_PACKAGE_NOTICE, /Settings → About/);
  for (const kept of ["models", "outputs", "`.env`", "plugins"]) {
    assert.ok(FULL_PACKAGE_NOTICE.includes(kept), `${kept} is named as surviving the reinstall`);
  }
});
