import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { rmSync } from "node:fs";
import { access, mkdir, mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import {
  applyPreparedUpdate,
  archiveExtensionAllowed,
  archiveUpdateInternals,
  prepareUpdate,
  recoverInterruptedUpdate,
} from "./archive-update.mjs";
import { createEnvironmentBackupOwnership, createOfflineUpdateTemp, writeEnvironmentBackupOwnership } from "./offline-update-temp.mjs";
import { acquireOfflineUpdateLock } from "./offline-update-lock.mjs";
import { stageReleasePackage } from "./release-package.mjs";
import { defaultModelPaths } from "./model-paths.mjs";

const {
  FORBIDDEN_TOP_LEVEL_ITEMS,
  createUpdatePlan,
  dependencyEnvironmentEqual,
  identifyProjectRoot,
  managedSevenZipPackage,
  normalizeRequirements,
  validateArchiveMemberPath,
} = archiveUpdateInternals;

// The update code canonicalises every path it is handed before deciding whether it is inside a
// managed root, so a fixture root has to be canonical too or the comparison fails on the machine
// rather than on the behaviour. `os.tmpdir()` is not always canonical: a Windows CI runner exports
// TEMP in its 8.3 short form (C:\Users\RUNNER~1\...), and a temp directory can sit behind a
// junction or a symlink on any platform.
async function temporaryDirectory(prefix) {
  return realpath(await mkdtemp(path.join(os.tmpdir(), prefix)));
}

function pathIsOutside(root, candidate) {
  const relative = path.relative(path.resolve(root), path.resolve(candidate));
  return Boolean(relative) && (path.isAbsolute(relative) || relative.split(path.sep)[0] === "..");
}

const COMMAND_TIMEOUT_MS = 30_000;

function run(command, args, { timeoutMs = COMMAND_TIMEOUT_MS, ...options } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { ...options, windowsHide: true, stdio: "ignore" });
    let settled = false;
    const finish = (callback) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      callback();
    };
    const timer = setTimeout(() => {
      // These helpers invoke a direct archive executable, so terminating the child
      // also releases the fixture's archive handle on Windows.
      child.kill();
      finish(() => reject(new Error(`${command} timed out after ${timeoutMs}ms`)));
    }, timeoutMs);
    timer.unref();
    child.once("error", (error) => finish(() => reject(error)));
    child.once("close", (code) => finish(() => code === 0 ? resolve() : reject(new Error(`${command} exited with ${code}`))));
  });
}

/** Builds a fixture archive under the same constraint the update code works under.
 *
 * GNU tar treats an argument containing a colon as `host:path`, so handing it an absolute Windows
 * path makes it try to reach a machine called `D` and exit 128. Git for Windows ships GNU tar and
 * is ahead of the System32 bsdtar on PATH on a GitHub Windows runner, so the archive is named
 * relative to the directory tar runs in, with separators tar accepts on either implementation.
 */
function createTar(archivePath, parentDirectory, entry) {
  const name = path.relative(parentDirectory, archivePath).split(path.sep).join("/");
  return run("tar", ["-cf", name, entry], { cwd: parentDirectory });
}

test("archive command helper terminates a stalled child", async () => {
  await assert.rejects(
    run(process.execPath, ["-e", "setInterval(() => {}, 1000)"], { timeoutMs: 100 }),
    /timed out after 100ms/,
  );
});

async function createProjectRoot(root) {
  await Promise.all([
    mkdir(path.join(root, "src"), { recursive: true }),
    mkdir(path.join(root, "backend"), { recursive: true }),
    mkdir(path.join(root, "scripts"), { recursive: true }),
  ]);
  await Promise.all([
    writeFile(path.join(root, "package.json"), JSON.stringify({ name: "xiriacanvas-ai" })),
    writeFile(path.join(root, "package-lock.json"), JSON.stringify({ packages: { "": {} } })),
    writeFile(path.join(root, "vite.config.js"), "export default {};\n"),
    // AGPL-3.0 section 4 obliges every conveyed copy to carry the licence, so packaging refuses
    // to build without it.
    writeFile(path.join(root, "LICENSE"), "GNU AFFERO GENERAL PUBLIC LICENSE\n"),
    writeFile(path.join(root, "backend", "requirements.txt"), "fastapi==1\n"),
  ]);
}

test("archive extension allowlist handles compound and case-insensitive names", () => {
  for (const filename of [
    "update.zip",
    "update.7z",
    "update.RAR",
    "update.tar",
    "update.tar.gz",
    "update.tgz",
    "update.tar.xz",
    "update.txz",
  ]) {
    assert.equal(archiveExtensionAllowed(filename), true, filename);
  }
  assert.equal(archiveExtensionAllowed("update.zip.exe"), false);
  assert.equal(archiveExtensionAllowed("update.gz"), false);
  assert.equal(archiveExtensionAllowed(null), false);
});

test("managed 7-Zip packages cover supported Windows and Linux architectures", () => {
  for (const [platform, architecture, extension] of [
    ["win32", "x64", ".exe"],
    ["win32", "arm64", ".exe"],
    ["linux", "x64", ".tar.xz"],
    ["linux", "arm64", ".tar.xz"],
  ]) {
    const packageInfo = managedSevenZipPackage(platform, architecture);
    assert.ok(packageInfo.url.endsWith(extension), `${platform}/${architecture}`);
    assert.match(packageInfo.sha256, /^[a-f0-9]{64}$/i, `${platform}/${architecture}`);
  }
  assert.throws(() => managedSevenZipPackage("linux", "ia32"), /不支持/);
});

test("archive member validation rejects traversal and platform absolute paths", () => {
  for (const unsafe of [
    "../src/App.jsx",
    "wrapper/../../src/App.jsx",
    "/etc/passwd",
    "C:\\Windows\\system.ini",
    "C:relative.txt",
    "\\\\server\\share\\file.txt",
    "src/file\0.txt",
    "src/file:stream",
  ]) {
    assert.throws(() => validateArchiveMemberPath(unsafe), /更新归档|路径/);
  }
  assert.equal(validateArchiveMemberPath("release/src/App.jsx"), "release/src/App.jsx");
});

test("project root detection accepts the root or exactly one wrapper directory", async () => {
  const directStage = await temporaryDirectory("xirai-archive-root-");
  const wrappedStage = await temporaryDirectory("xirai-archive-wrapper-");
  try {
    await createProjectRoot(directStage);
    assert.equal(await identifyProjectRoot(directStage), directStage);

    const wrappedRoot = path.join(wrappedStage, "XiriaCanvas-AI-release");
    await createProjectRoot(wrappedRoot);
    assert.equal(await identifyProjectRoot(wrappedStage), wrappedRoot);

    await writeFile(path.join(wrappedStage, "extra.txt"), "not allowed beside wrapper");
    await assert.rejects(identifyProjectRoot(wrappedStage), /超过一层包裹目录/);
  } finally {
    await Promise.all([
      rm(directStage, { recursive: true, force: true }),
      rm(wrappedStage, { recursive: true, force: true }),
    ]);
  }
});

test("dependency comparison ignores ordering and requirements comments", () => {
  const oldLock = {
    packages: {
      "": {
        dependencies: { react: "19.2.7", "react-dom": "19.2.7" },
        devDependencies: { vite: "8.1.5", plugin: "1.0.0" },
      },
    },
  };
  const reorderedLock = {
    packages: {
      "": {
        dependencies: { "react-dom": "19.2.7", react: "19.2.7" },
        devDependencies: { plugin: "1.0.0", vite: "8.1.5" },
      },
    },
  };
  assert.equal(dependencyEnvironmentEqual(
    oldLock,
    reorderedLock,
    "fastapi==1\n# comment\nnumpy==2\n",
    " numpy==2 # same dependency\n\nfastapi==1\n",
  ), true);
  assert.deepEqual(normalizeRequirements("b==1\n# x\na==1\n"), ["a==1", "b==1"]);
});

test("dependency comparison rejects npm or Python dependency changes", () => {
  const baseline = { packages: { "": { dependencies: { react: "19" }, devDependencies: { vite: "8" } } } };
  const npmChanged = { packages: { "": { dependencies: { react: "20" }, devDependencies: { vite: "8" } } } };
  const missingDependencies = { packages: { "": { devDependencies: { vite: "8" } } } };
  assert.equal(dependencyEnvironmentEqual(baseline, npmChanged, "fastapi==1", "fastapi==1"), false);
  assert.equal(dependencyEnvironmentEqual(baseline, baseline, "fastapi==1", "fastapi==2"), false);
  assert.equal(dependencyEnvironmentEqual(missingDependencies, baseline, "fastapi==1", "fastapi==1"), false);
});

test("dependency comparison detects transitive lockfile and optional dependency changes", () => {
  const baseline = {
    lockfileVersion: 3,
    packages: {
      "": { dependencies: { react: "19" }, optionalDependencies: { optional: "1" } },
      "node_modules/react": { version: "19.2.7", resolved: "https://example.test/react.tgz", integrity: "sha512-old" },
    },
  };
  const reordered = {
    packages: {
      "node_modules/react": { integrity: "sha512-old", resolved: "https://example.test/react.tgz", version: "19.2.7" },
      "": { optionalDependencies: { optional: "1" }, dependencies: { react: "19" } },
    },
    lockfileVersion: 3,
  };
  const transitiveChanged = structuredClone(baseline);
  transitiveChanged.packages["node_modules/react"].integrity = "sha512-new";
  const optionalChanged = structuredClone(baseline);
  optionalChanged.packages[""].optionalDependencies.optional = "2";
  assert.equal(dependencyEnvironmentEqual(baseline, reordered, "fastapi==1", "fastapi==1"), true);
  assert.equal(dependencyEnvironmentEqual(baseline, transitiveChanged, "fastapi==1", "fastapi==1"), false);
  assert.equal(dependencyEnvironmentEqual(baseline, optionalChanged, "fastapi==1", "fastapi==1"), false);
});

test("update plan includes only managed program items and excludes environments and models", async () => {
  const projectRoot = await temporaryDirectory("xirai-plan-current-");
  const packageRoot = await temporaryDirectory("xirai-plan-package-");
  try {
    await Promise.all([
      mkdir(path.join(projectRoot, "public"), { recursive: true }),
      mkdir(path.join(packageRoot, "src"), { recursive: true }),
      mkdir(path.join(packageRoot, "backend"), { recursive: true }),
      mkdir(path.join(packageRoot, "scripts"), { recursive: true }),
      mkdir(path.join(packageRoot, "models"), { recursive: true }),
      mkdir(path.join(packageRoot, "node_modules"), { recursive: true }),
      mkdir(path.join(packageRoot, ".venv"), { recursive: true }),
    ]);
    await Promise.all([
      writeFile(path.join(packageRoot, "README.md"), "updated"),
      writeFile(path.join(packageRoot, ".env"), "SECRET=do-not-copy"),
      writeFile(path.join(packageRoot, "models", "model.bin"), "model"),
    ]);

    const plan = await createUpdatePlan(projectRoot, packageRoot);
    assert.deepEqual(plan.map((item) => item.relativePath), ["src", "backend", "scripts", "public", "README.md"]);
    assert.deepEqual(plan.find((item) => item.relativePath === "public"), {
      relativePath: "public",
      kind: "directory",
      action: "remove",
    });
    for (const forbidden of ["models", "node_modules", ".venv", ".env", ".cache", "dist"] ) {
      assert.equal(plan.some((item) => item.relativePath === forbidden), false, forbidden);
    }
  } finally {
    await Promise.all([
      rm(projectRoot, { recursive: true, force: true }),
      rm(packageRoot, { recursive: true, force: true }),
    ]);
  }
});

test("update plan removes managed files missing from the new project but preserves unmanaged data", async () => {
  const projectRoot = await temporaryDirectory("xirai-plan-remove-current-");
  const packageRoot = await temporaryDirectory("xirai-plan-remove-package-");
  try {
    await Promise.all([
      mkdir(path.join(packageRoot, "src"), { recursive: true }),
      mkdir(path.join(packageRoot, "backend"), { recursive: true }),
      mkdir(path.join(packageRoot, "scripts"), { recursive: true }),
    ]);
    await Promise.all([
      writeFile(path.join(projectRoot, "README.md"), "obsolete readme"),
      writeFile(path.join(projectRoot, "Start-XirAI.sh"), "obsolete launcher"),
      writeFile(path.join(projectRoot, "user-notes.txt"), "keep me"),
    ]);
    const plan = await createUpdatePlan(projectRoot, packageRoot);
    assert.deepEqual(plan.find((item) => item.relativePath === "README.md"), {
      relativePath: "README.md",
      kind: "file",
      action: "remove",
    });
    assert.deepEqual(plan.find((item) => item.relativePath === "Start-XirAI.sh"), {
      relativePath: "Start-XirAI.sh",
      kind: "file",
      action: "remove",
    });
    assert.equal(plan.some((item) => item.relativePath === "user-notes.txt"), false);
  } finally {
    await Promise.all([
      rm(projectRoot, { recursive: true, force: true }),
      rm(packageRoot, { recursive: true, force: true }),
    ]);
  }
});

test("update plan preserves user model paths and weights while refreshing catalogs", async () => {
  const projectRoot = await temporaryDirectory("xirai-plan-manifest-current-");
  const packageRoot = await temporaryDirectory("xirai-plan-manifest-package-");
  try {
    await Promise.all([
      mkdir(path.join(projectRoot, "models", "background-removal"), { recursive: true }),
      mkdir(path.join(packageRoot, "models"), { recursive: true }),
    ]);
    await Promise.all([
      writeFile(path.join(projectRoot, "models", "background-removal", "u2netp.onnx"), "weights"),
      writeFile(path.join(packageRoot, "models", "model-paths.json"), "{}"),
      writeFile(path.join(packageRoot, "models", "recommended-models.json"), "{}"),
      writeFile(path.join(packageRoot, "models", "yolo-models.json"), "{}"),
      writeFile(path.join(packageRoot, "models", "background-removal-models.json"), "{}"),
      writeFile(path.join(packageRoot, "models", "README.md"), "models"),
      writeFile(path.join(packageRoot, "models", "untrusted.bin"), "not managed"),
    ]);

    const plan = await createUpdatePlan(projectRoot, packageRoot);
    assert.deepEqual(plan.map((item) => item.relativePath), [
      "models/model-paths.json",
      "models/recommended-models.json",
      "models/yolo-models.json",
      "models/background-removal-models.json",
      "models/README.md",
    ]);
    assert.equal(plan.some((item) => item.relativePath.endsWith(".onnx") || item.relativePath.endsWith(".bin")), false);
  } finally {
    await Promise.all([
      rm(projectRoot, { recursive: true, force: true }),
      rm(packageRoot, { recursive: true, force: true }),
    ]);
  }
});

test("a usable model path configuration survives the update, an unusable one is restored", async () => {
  const projectRoot = await temporaryDirectory("xirai-plan-preserve-current-");
  const packageRoot = await temporaryDirectory("xirai-plan-preserve-package-");
  const configured = path.join(projectRoot, "models", "model-paths.json");
  const shipped = { ...defaultModelPaths };
  const planned = async () => (await createUpdatePlan(projectRoot, packageRoot))
    .some((item) => item.relativePath === "models/model-paths.json");
  try {
    await Promise.all([
      mkdir(path.join(projectRoot, "models"), { recursive: true }),
      mkdir(path.join(packageRoot, "models"), { recursive: true }),
    ]);
    await writeFile(path.join(packageRoot, "models", "model-paths.json"), JSON.stringify(shipped));

    // Nothing to keep yet: a fresh or repaired project takes the release default.
    assert.equal(await planned(), true);

    await writeFile(configured, JSON.stringify(shipped));
    assert.equal(await planned(), false);

    // The whole point: a hand-edited model root must still be there after the update.
    await writeFile(configured, JSON.stringify({
      ...shipped,
      checkpoints: { ...shipped.checkpoints, illustrious: "models/checkpoints/my-illustrious" },
    }));
    assert.equal(await planned(), false);

    // Anything the post-update validation would reject is replaced instead of kept, so preserving
    // a file can never turn into an update that rolls back.
    for (const unusable of [
      "{ not json",
      JSON.stringify({ checkpoints: {}, loras: {} }),
      JSON.stringify({ ...shipped, upscalers: "../../elsewhere" }),
      JSON.stringify({ ...shipped, configs: 7 }),
    ]) {
      await writeFile(configured, unusable);
      assert.equal(await planned(), true, `expected replacement for ${unusable.slice(0, 40)}`);
    }
  } finally {
    await Promise.all([
      rm(projectRoot, { recursive: true, force: true }),
      rm(packageRoot, { recursive: true, force: true }),
    ]);
  }
});

test("update planning treats the plugin port as user-owned and never manages it", async () => {
  // The guard is structural: `plugins` sits in the forbidden top-level set, so adding it to the
  // managed lists by mistake would make `createUpdatePlan` throw instead of quietly replacing a
  // user's installed plugins.
  assert.equal(FORBIDDEN_TOP_LEVEL_ITEMS.has("plugins"), true);

  const projectRoot = await temporaryDirectory("xirai-plan-plugins-current-");
  const packageRoot = await temporaryDirectory("xirai-plan-plugins-package-");
  try {
    await Promise.all([
      mkdir(path.join(projectRoot, "plugins", "user-plugin"), { recursive: true }),
      mkdir(path.join(packageRoot, "src"), { recursive: true }),
      mkdir(path.join(packageRoot, "backend"), { recursive: true }),
      mkdir(path.join(packageRoot, "scripts"), { recursive: true }),
      mkdir(path.join(packageRoot, "plugins", "shipped-plugin"), { recursive: true }),
    ]);
    await Promise.all([
      writeFile(path.join(projectRoot, "plugins", "user-plugin", "plugin.json"), "{}"),
      writeFile(path.join(packageRoot, "plugins", "shipped-plugin", "plugin.json"), "{}"),
      writeFile(path.join(packageRoot, "plugins", "README.md"), "packaged placeholder"),
    ]);

    const plan = await createUpdatePlan(projectRoot, packageRoot);
    assert.equal(plan.some((item) => item.relativePath.split("/")[0] === "plugins"), false);
  } finally {
    await Promise.all([
      rm(projectRoot, { recursive: true, force: true }),
      rm(packageRoot, { recursive: true, force: true }),
    ]);
  }
});

test("an applied update leaves installed plugins untouched and never delivers packaged plugins", async (context) => {
  try {
    await run("tar", ["--version"]);
  } catch {
    context.skip("system tar is unavailable");
    return;
  }

  const workspace = await temporaryDirectory("xirai-archive-plugins-");
  const projectRoot = path.join(workspace, "current");
  const packageParent = path.join(workspace, "package");
  const packageRoot = path.join(packageParent, "XiriaCanvas-AI-release");
  const archivePath = path.join(workspace, "update.tar");
  try {
    await Promise.all([createProjectRoot(projectRoot), createProjectRoot(packageRoot)]);
    await Promise.all([
      mkdir(path.join(projectRoot, "plugins", "user-plugin"), { recursive: true }),
      mkdir(path.join(packageRoot, "plugins", "attacker-plugin"), { recursive: true }),
    ]);
    await Promise.all([
      writeFile(path.join(projectRoot, "plugins", "user-plugin", "plugin.json"), "{\"id\":\"user-plugin\"}"),
      writeFile(path.join(projectRoot, "plugins", "user-plugin", "index.js"), "user code"),
      writeFile(path.join(packageRoot, "plugins", "attacker-plugin", "plugin.json"), "{\"id\":\"attacker-plugin\"}"),
      writeFile(path.join(packageRoot, "plugins", "attacker-plugin", "index.js"), "attacker code"),
      writeFile(path.join(packageRoot, "src", "new.jsx"), "new"),
    ]);
    await createTar(archivePath, packageParent, path.basename(packageRoot));

    const prepared = await prepareUpdate({ projectRoot, archivePath });
    await applyPreparedUpdate({ projectRoot, prepared });

    assert.equal(await readFile(path.join(projectRoot, "src", "new.jsx"), "utf8"), "new");
    assert.equal(await readFile(path.join(projectRoot, "plugins", "user-plugin", "index.js"), "utf8"), "user code");
    assert.equal(await readFile(path.join(projectRoot, "plugins", "user-plugin", "plugin.json"), "utf8"), "{\"id\":\"user-plugin\"}");
    await assert.rejects(access(path.join(projectRoot, "plugins", "attacker-plugin")));
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

test("a small tar update is staged externally and mirrors managed directories", async (context) => {
  try {
    await run("tar", ["--version"]);
  } catch {
    context.skip("system tar is unavailable");
    return;
  }

  const workspace = await temporaryDirectory("xirai-archive-e2e-");
  const projectRoot = path.join(workspace, "current");
  const packageParent = path.join(workspace, "package");
  const packageRoot = path.join(packageParent, "XiriaCanvas-AI-release");
  const archivePath = path.join(workspace, "update.tar");
  try {
    await Promise.all([createProjectRoot(projectRoot), createProjectRoot(packageRoot)]);
    await Promise.all([
      mkdir(path.join(projectRoot, "public"), { recursive: true }),
      mkdir(path.join(projectRoot, "models"), { recursive: true }),
    ]);
    await Promise.all([
      writeFile(path.join(projectRoot, "src", "old.jsx"), "old"),
      writeFile(path.join(projectRoot, "public", "old.txt"), "old public"),
      writeFile(path.join(projectRoot, "models", "keep.bin"), "model"),
      writeFile(path.join(projectRoot, "Start-XirAI.sh"), "obsolete launcher"),
      writeFile(path.join(packageRoot, "src", "new.jsx"), "new"),
      writeFile(path.join(packageRoot, "README.md"), "new readme"),
    ]);
    await createTar(archivePath, packageParent, path.basename(packageRoot));

    const prepared = await prepareUpdate({ projectRoot, archivePath });
    assert.equal(pathIsOutside(projectRoot, prepared.stagingDirectory), true);
    assert.equal(await readFile(path.join(projectRoot, "src", "old.jsx"), "utf8"), "old");
    await applyPreparedUpdate({ projectRoot, prepared });

    assert.equal(await readFile(path.join(projectRoot, "src", "new.jsx"), "utf8"), "new");
    await assert.rejects(access(path.join(projectRoot, "src", "old.jsx")));
    await assert.rejects(access(path.join(projectRoot, "public")));
    await assert.rejects(access(path.join(projectRoot, "Start-XirAI.sh")));
    assert.equal(await readFile(path.join(projectRoot, "models", "keep.bin"), "utf8"), "model");
    await assert.rejects(access(archivePath));
    await assert.rejects(access(prepared.stagingDirectory));
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

test("apply rolls back files already replaced when a later copy fails", async (context) => {
  try {
    await run("tar", ["--version"]);
  } catch {
    context.skip("system tar is unavailable");
    return;
  }

  const workspace = await temporaryDirectory("xirai-archive-rollback-");
  const projectRoot = path.join(workspace, "current");
  const packageParent = path.join(workspace, "package");
  const packageRoot = path.join(packageParent, "release");
  const archivePath = path.join(workspace, "update.tar");
  let prepared;
  try {
    await Promise.all([createProjectRoot(projectRoot), createProjectRoot(packageRoot)]);
    await Promise.all([
      writeFile(path.join(projectRoot, "src", "old.jsx"), "old"),
      writeFile(path.join(packageRoot, "src", "new.jsx"), "new"),
    ]);
    await createTar(archivePath, packageParent, path.basename(packageRoot));
    prepared = await prepareUpdate({ projectRoot, archivePath });

    let faultInjected = false;
    await assert.rejects(applyPreparedUpdate({
      projectRoot,
      prepared,
      report: (event) => {
        if (!faultInjected && event.phase === "apply" && event.item?.relativePath === "backend") {
          faultInjected = true;
          rmSync(path.join(prepared.packageRoot, "backend"), { recursive: true, force: true });
        }
      },
    }), /已自动回滚/);

    assert.equal(faultInjected, true);
    assert.equal(await readFile(path.join(projectRoot, "src", "old.jsx"), "utf8"), "old");
    await assert.rejects(access(path.join(projectRoot, "src", "new.jsx")));
    assert.equal(await readFile(path.join(projectRoot, "backend", "requirements.txt"), "utf8"), "fastapi==1\n");
    await access(archivePath);
  } finally {
    if (prepared?.stagingDirectory) await rm(prepared.stagingDirectory, { recursive: true, force: true });
    await rm(workspace, { recursive: true, force: true });
  }
});

test("post-apply validation failure restores the previous managed files", async (context) => {
  try {
    await run("tar", ["--version"]);
  } catch {
    context.skip("system tar is unavailable");
    return;
  }

  const workspace = await temporaryDirectory("xirai-archive-validation-rollback-");
  const projectRoot = path.join(workspace, "current");
  const packageParent = path.join(workspace, "package");
  const packageRoot = path.join(packageParent, "release");
  const archivePath = path.join(workspace, "update.tar");
  let prepared;
  try {
    await Promise.all([createProjectRoot(projectRoot), createProjectRoot(packageRoot)]);
    await Promise.all([
      writeFile(path.join(projectRoot, "src", "version.txt"), "old"),
      writeFile(path.join(projectRoot, "Start-XirAI.sh"), "old launcher"),
      writeFile(path.join(packageRoot, "src", "version.txt"), "new"),
    ]);
    await createTar(archivePath, packageParent, path.basename(packageRoot));
    prepared = await prepareUpdate({ projectRoot, archivePath });
    await assert.rejects(applyPreparedUpdate({
      projectRoot,
      prepared,
      validate: async () => { throw new Error("build failed"); },
    }), /已自动回滚/);
    assert.equal(await readFile(path.join(projectRoot, "src", "version.txt"), "utf8"), "old");
    assert.equal(await readFile(path.join(projectRoot, "Start-XirAI.sh"), "utf8"), "old launcher");
    await access(archivePath);
  } finally {
    if (prepared?.stagingDirectory) await rm(prepared.stagingDirectory, { recursive: true, force: true });
    await rm(workspace, { recursive: true, force: true });
  }
});

test("transaction remains applying until complete validation succeeds", async (context) => {
  try {
    await run("tar", ["--version"]);
  } catch {
    context.skip("system tar is unavailable");
    return;
  }
  const workspace = await temporaryDirectory("xirai-archive-commit-order-");
  const projectRoot = path.join(workspace, "current");
  const packageParent = path.join(workspace, "package");
  const packageRoot = path.join(packageParent, "release");
  const archivePath = path.join(workspace, "update.tar");
  const transactionPath = path.join(workspace, "transaction.json");
  let prepared;
  try {
    await Promise.all([createProjectRoot(projectRoot), createProjectRoot(packageRoot)]);
    await writeFile(path.join(packageRoot, "src", "version.txt"), "new");
    await createTar(archivePath, packageParent, path.basename(packageRoot));
    prepared = await prepareUpdate({ projectRoot, archivePath });
    let phaseDuringValidation;
    let phaseDuringCommit;
    await applyPreparedUpdate({
      projectRoot,
      prepared,
      transactionPath,
      validate: async () => {
        phaseDuringValidation = JSON.parse(await readFile(transactionPath, "utf8")).phase;
        return {
          verified: true,
          commit: async () => { phaseDuringCommit = JSON.parse(await readFile(transactionPath, "utf8")).phase; },
        };
      },
    });
    assert.equal(phaseDuringValidation, "applying");
    assert.equal(phaseDuringCommit, "committed");
    await assert.rejects(access(transactionPath));
  } finally {
    if (prepared?.stagingDirectory) await rm(prepared.stagingDirectory, { recursive: true, force: true });
    await rm(workspace, { recursive: true, force: true });
  }
});

test("transactional apply rejects a missing validation receipt and rolls back", async (context) => {
  try {
    await run("tar", ["--version"]);
  } catch {
    context.skip("system tar is unavailable");
    return;
  }
  const workspace = await temporaryDirectory("xirai-archive-validation-receipt-");
  const projectRoot = path.join(workspace, "current");
  const packageParent = path.join(workspace, "package");
  const packageRoot = path.join(packageParent, "release");
  const archivePath = path.join(workspace, "update.tar");
  const transactionPath = path.join(workspace, "transaction.json");
  let prepared;
  try {
    await Promise.all([createProjectRoot(projectRoot), createProjectRoot(packageRoot)]);
    await Promise.all([
      writeFile(path.join(projectRoot, "src", "version.txt"), "old"),
      writeFile(path.join(packageRoot, "src", "version.txt"), "new"),
    ]);
    await createTar(archivePath, packageParent, path.basename(packageRoot));
    prepared = await prepareUpdate({ projectRoot, archivePath });
    await assert.rejects(applyPreparedUpdate({
      projectRoot,
      prepared,
      transactionPath,
      validate: async () => undefined,
    }), /未返回有效验证结果/);
    assert.equal(await readFile(path.join(projectRoot, "src", "version.txt"), "utf8"), "old");
  } finally {
    if (prepared?.stagingDirectory) await rm(prepared.stagingDirectory, { recursive: true, force: true });
    await rm(workspace, { recursive: true, force: true });
  }
});

test("program files still roll back when Python environment rollback fails", async (context) => {
  try {
    await run("tar", ["--version"]);
  } catch {
    context.skip("system tar is unavailable");
    return;
  }
  const workspace = await temporaryDirectory("xirai-archive-partial-env-rollback-");
  const projectRoot = path.join(workspace, "current");
  const packageParent = path.join(workspace, "package");
  const packageRoot = path.join(packageParent, "release");
  const archivePath = path.join(workspace, "update.tar");
  const transactionPath = path.join(workspace, "transaction.json");
  let prepared;
  try {
    await Promise.all([createProjectRoot(projectRoot), createProjectRoot(packageRoot)]);
    await Promise.all([
      writeFile(path.join(projectRoot, "src", "version.txt"), "old"),
      writeFile(path.join(packageRoot, "src", "version.txt"), "new"),
    ]);
    await createTar(archivePath, packageParent, path.basename(packageRoot));
    prepared = await prepareUpdate({ projectRoot, archivePath });
    await assert.rejects(applyPreparedUpdate({
      projectRoot,
      prepared,
      transactionPath,
      validate: async () => ({
        verified: false,
        rollback: async () => { throw new Error("venv restore failed"); },
      }),
    }), (error) => error.rollbackIncomplete === true && /程序文件已回滚.*Python 环境回滚不完整/.test(error.message));
    assert.equal(await readFile(path.join(projectRoot, "src", "version.txt"), "utf8"), "old");
    await access(transactionPath);
  } finally {
    if (prepared?.stagingDirectory) await rm(prepared.stagingDirectory, { recursive: true, force: true });
    await rm(workspace, { recursive: true, force: true });
  }
});

test("Python requirement changes are accepted and require automatic environment repair", async (context) => {
  try {
    await run("tar", ["--version"]);
  } catch {
    context.skip("system tar is unavailable");
    return;
  }
  const workspace = await temporaryDirectory("xirai-archive-python-change-");
  const projectRoot = path.join(workspace, "current");
  const packageParent = path.join(workspace, "package");
  const packageRoot = path.join(packageParent, "release");
  const archivePath = path.join(workspace, "update.tar");
  let prepared;
  try {
    await Promise.all([createProjectRoot(projectRoot), createProjectRoot(packageRoot)]);
    await writeFile(path.join(packageRoot, "backend", "requirements.txt"), "fastapi==2\n");
    await createTar(archivePath, packageParent, path.basename(packageRoot));
    prepared = await prepareUpdate({ projectRoot, archivePath });
    assert.equal(prepared.environmentRepairRequired, true);
  } finally {
    if (prepared?.stagingDirectory) await rm(prepared.stagingDirectory, { recursive: true, force: true });
    await rm(workspace, { recursive: true, force: true });
  }
});

test("npm lock environment changes are rejected before replacing live Node modules", async (context) => {
  try {
    await run("tar", ["--version"]);
  } catch {
    context.skip("system tar is unavailable");
    return;
  }
  const workspace = await temporaryDirectory("xirai-archive-node-change-");
  const projectRoot = path.join(workspace, "current");
  const packageParent = path.join(workspace, "package");
  const packageRoot = path.join(packageParent, "release");
  const archivePath = path.join(workspace, "update.tar");
  try {
    await Promise.all([createProjectRoot(projectRoot), createProjectRoot(packageRoot)]);
    await writeFile(path.join(packageRoot, "package-lock.json"), JSON.stringify({ packages: { "": { dependencies: { react: "20" } } } }));
    await createTar(archivePath, packageParent, path.basename(packageRoot));
    await assert.rejects(prepareUpdate({ projectRoot, archivePath }), /前端依赖变化/);
    await access(path.join(projectRoot, "package-lock.json"));
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

test("interrupted update journal restores program files and the previous Python environment", async () => {
  const workspace = await temporaryDirectory("xirai-update-recovery-");
  const projectRoot = path.join(workspace, "project");
  const transactionPath = path.join(workspace, "manual-update-transaction.json");
  try {
    const backupOwnership = await createOfflineUpdateTemp({ projectRoot, prefix: "xirai-update-backup-", kind: "backup" });
    const backupRoot = backupOwnership.path;
    const environmentBackup = createEnvironmentBackupOwnership(projectRoot);
    await Promise.all([
      createProjectRoot(projectRoot),
      mkdir(path.join(backupRoot, "src"), { recursive: true }),
      mkdir(environmentBackup.path, { recursive: true }),
      mkdir(path.join(projectRoot, ".venv"), { recursive: true }),
    ]);
    await writeEnvironmentBackupOwnership(projectRoot, environmentBackup.path, environmentBackup);
    await Promise.all([
      writeFile(path.join(projectRoot, "src", "version.txt"), "partial"),
      writeFile(path.join(backupRoot, "src", "version.txt"), "old"),
      writeFile(path.join(projectRoot, ".venv", "new.txt"), "new environment"),
      writeFile(path.join(environmentBackup.path, "old.txt"), "old environment"),
      writeFile(transactionPath, JSON.stringify({
        schema: 1,
        product: "XiriaCanvas AI",
        projectRoot,
        phase: "applying",
        backupRoot,
        backupOwnership,
        environmentBackup,
        setupMarker: null,
        manifest: [{ relativePath: "src", kind: "directory", action: "replace", existed: true }],
      })),
    ]);
    const lock = await acquireOfflineUpdateLock({ projectRoot, operation: "test-recovery" });
    const result = await recoverInterruptedUpdate({ projectRoot, transactionPath, lock });
    await lock.release();
    assert.deepEqual(result, { recovered: true, rolledBack: true });
    assert.equal(await readFile(path.join(projectRoot, "src", "version.txt"), "utf8"), "old");
    assert.equal(await readFile(path.join(projectRoot, ".venv", "old.txt"), "utf8"), "old environment");
    await assert.rejects(access(transactionPath));
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

test("recovery requires the current project's offline update lock", async () => {
  const workspace = await temporaryDirectory("xirai-update-live-owner-");
  const projectRoot = path.join(workspace, "project");
  const transactionPath = path.join(workspace, "transaction.json");
  try {
    await createProjectRoot(projectRoot);
    const backupOwnership = await createOfflineUpdateTemp({ projectRoot, prefix: "xirai-update-backup-", kind: "backup" });
    await mkdir(path.join(backupOwnership.path, "src"), { recursive: true });
    await Promise.all([
      writeFile(path.join(projectRoot, "src", "version.txt"), "updating"),
      writeFile(path.join(backupOwnership.path, "src", "version.txt"), "old"),
      writeFile(transactionPath, JSON.stringify({
        schema: 1,
        product: "XiriaCanvas AI",
        projectRoot,
        ownerPid: process.pid,
        phase: "applying",
        backupRoot: backupOwnership.path,
        backupOwnership,
        environmentBackup: null,
        setupMarker: null,
        manifest: [{ relativePath: "src", kind: "directory", action: "replace", existed: true }],
      })),
    ]);
    await assert.rejects(recoverInterruptedUpdate({ projectRoot, transactionPath }), /必须持有/);
    assert.equal(await readFile(path.join(projectRoot, "src", "version.txt"), "utf8"), "updating");
    await access(transactionPath);
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

test("recovery rejects unmanaged paths before modifying the project", async () => {
  const workspace = await temporaryDirectory("xirai-update-invalid-manifest-");
  const projectRoot = path.join(workspace, "project");
  const transactionPath = path.join(workspace, "transaction.json");
  try {
    await createProjectRoot(projectRoot);
    const backupOwnership = await createOfflineUpdateTemp({ projectRoot, prefix: "xirai-update-backup-", kind: "backup" });
    await writeFile(path.join(projectRoot, ".env"), "KEEP=1");
    await writeFile(transactionPath, JSON.stringify({
      schema: 1,
      product: "XiriaCanvas AI",
      projectRoot,
      phase: "applying",
      backupRoot: backupOwnership.path,
      backupOwnership,
      environmentBackup: null,
      setupMarker: null,
      manifest: [{ relativePath: ".env", kind: "file", action: "remove", existed: true }],
    }));
    const lock = await acquireOfflineUpdateLock({ projectRoot, operation: "test-invalid-recovery" });
    await assert.rejects(recoverInterruptedUpdate({ projectRoot, transactionPath, lock }), /恢复清单/);
    await lock.release();
    assert.equal(await readFile(path.join(projectRoot, ".env"), "utf8"), "KEEP=1");
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

test("interrupted standalone environment repair restores the previous venv", async () => {
  const workspace = await temporaryDirectory("xirai-environment-recovery-");
  const projectRoot = path.join(workspace, "project");
  const transactionPath = path.join(workspace, "environment-transaction.json");
  try {
    await createProjectRoot(projectRoot);
    const environmentBackup = createEnvironmentBackupOwnership(projectRoot);
    await Promise.all([
      mkdir(environmentBackup.path, { recursive: true }),
      mkdir(path.join(projectRoot, ".venv"), { recursive: true }),
    ]);
    await writeEnvironmentBackupOwnership(projectRoot, environmentBackup.path, environmentBackup);
    await Promise.all([
      writeFile(path.join(environmentBackup.path, "old.txt"), "old environment"),
      writeFile(path.join(projectRoot, ".venv", "new.txt"), "partial environment"),
      writeFile(transactionPath, JSON.stringify({
        schema: 1,
        product: "XiriaCanvas AI",
        projectRoot,
        phase: "environment-repair",
        environmentBackup,
        setupMarker: null,
      })),
    ]);
    const lock = await acquireOfflineUpdateLock({ projectRoot, operation: "test-environment-recovery" });
    const result = await recoverInterruptedUpdate({ projectRoot, transactionPath, lock });
    await lock.release();
    assert.deepEqual(result, { recovered: true, rolledBack: true });
    assert.equal(await readFile(path.join(projectRoot, ".venv", "old.txt"), "utf8"), "old environment");
    await assert.rejects(access(path.join(projectRoot, ".venv", "new.txt")));
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

test("committed standalone environment repair keeps the new venv", async () => {
  const workspace = await temporaryDirectory("xirai-environment-commit-recovery-");
  const projectRoot = path.join(workspace, "project");
  const transactionPath = path.join(workspace, "environment-transaction.json");
  try {
    await createProjectRoot(projectRoot);
    const environmentBackup = createEnvironmentBackupOwnership(projectRoot);
    await Promise.all([
      mkdir(environmentBackup.path, { recursive: true }),
      mkdir(path.join(projectRoot, ".venv"), { recursive: true }),
    ]);
    await writeEnvironmentBackupOwnership(projectRoot, environmentBackup.path, environmentBackup);
    await Promise.all([
      writeFile(path.join(environmentBackup.path, "old.txt"), "old environment"),
      writeFile(path.join(projectRoot, ".venv", "new.txt"), "verified environment"),
      writeFile(transactionPath, JSON.stringify({
        schema: 1,
        product: "XiriaCanvas AI",
        projectRoot,
        phase: "environment-committed",
        environmentBackup,
        setupMarker: null,
      })),
    ]);
    const lock = await acquireOfflineUpdateLock({ projectRoot, operation: "test-environment-commit" });
    const result = await recoverInterruptedUpdate({ projectRoot, transactionPath, lock });
    await lock.release();
    assert.deepEqual(result, { recovered: true, rolledBack: false });
    assert.equal(await readFile(path.join(projectRoot, ".venv", "new.txt"), "utf8"), "verified environment");
    await assert.rejects(access(environmentBackup.path));
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

test("real Windows bootstrap prepares a 7z update with the verified portable tool", {
  skip: process.platform !== "win32" || process.env.XIRAI_TEST_REAL_7ZIP !== "1",
}, async () => {
  const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const workspace = await temporaryDirectory("xirai-real-7zip-");
  const packageParent = path.join(workspace, "package");
  const packageRoot = path.join(packageParent, "release");
  const invalidArchive = path.join(workspace, "bootstrap.7z");
  const archivePath = path.join(workspace, "update.7z");
  const portableExecutable = path.join(repositoryRoot, ".cache", "tools", "7zip", "portable", "7z.exe");
  let prepared;
  try {
    await createProjectRoot(packageRoot);
    const [packageLock, requirements] = await Promise.all([
      readFile(path.join(repositoryRoot, "package-lock.json"), "utf8"),
      readFile(path.join(repositoryRoot, "backend", "requirements.txt"), "utf8"),
    ]);
    await Promise.all([
      writeFile(path.join(packageRoot, "package-lock.json"), packageLock),
      writeFile(path.join(packageRoot, "backend", "requirements.txt"), requirements),
      writeFile(path.join(packageRoot, "src", "real-7zip-test.txt"), "ready\n"),
      writeFile(invalidArchive, "bootstrap the verified tools"),
    ]);

    // The host has tar but no 7z command. A 7z archive must still use the verified portable 7-Zip chain.
    await assert.rejects(prepareUpdate({ projectRoot: repositoryRoot, archivePath: invalidArchive }));
    await access(portableExecutable);
    await run(portableExecutable, ["i"]);
    await run(portableExecutable, [
      "a", "-t7z", "-mx=1", "-bd", "-bso0", "-bsp0", archivePath, path.basename(packageRoot),
    ], { cwd: packageParent });
    await writeFile(portableExecutable, "tampered portable cache");

    const events = [];
    prepared = await prepareUpdate({
      projectRoot: repositoryRoot,
      archivePath,
      report: (event) => events.push(event),
    });
    assert.equal(path.resolve(prepared.archiveTool.command), path.resolve(portableExecutable));
    assert.equal(prepared.archiveTool.kind, "7zip");
    assert.equal(pathIsOutside(repositoryRoot, prepared.stagingDirectory), true);
    assert.equal(await readFile(path.join(prepared.packageRoot, "src", "real-7zip-test.txt"), "utf8"), "ready\n");
    await run(portableExecutable, ["i"]);
    assert.ok(events.some((event) => event.phase === "tools"));
    assert.ok(events.some((event) => event.phase === "ready" && event.progress === 100));
  } finally {
    if (prepared?.stagingDirectory) await rm(prepared.stagingDirectory, { recursive: true, force: true });
    await rm(workspace, { recursive: true, force: true });
  }
});

test("both README translations are replaced by an update, not just the English one", () => {
  // The English README was managed and the Chinese one was not, so every documentation change
  // reached English readers through an update and never reached Chinese readers at all — the
  // shipped README.zh-CN.md stayed at whatever version was first installed.
  const { MANAGED_FILES } = archiveUpdateInternals;
  assert.ok(MANAGED_FILES.includes("README.md"));
  assert.ok(MANAGED_FILES.includes("README.zh-CN.md"));
  // A managed file must never be one an update is forbidden to touch, or the plan throws.
  for (const relativePath of MANAGED_FILES) {
    const topLevel = relativePath.split("/")[0];
    const exempt = relativePath.startsWith("models/");
    assert.equal(FORBIDDEN_TOP_LEVEL_ITEMS.has(topLevel) && !exempt, false, relativePath);
  }
});

test("release archives require one wrapper and reject unknown/test package members", async (context) => {
  try { await run("tar", ["--version"]); } catch { context.skip("system tar is unavailable"); return; }
  const workspace = await temporaryDirectory("xirai-release-archive-");
  const current = path.join(workspace, "current");
  const source = path.join(workspace, "source");
  const staging = path.join(workspace, "staging");
  const archivePath = path.join(workspace, "release.tar");
  const directArchivePath = path.join(workspace, "release-direct.tar");
  try {
    await Promise.all([createProjectRoot(current), createProjectRoot(source), mkdir(path.join(source, "models"), { recursive: true })]);
    await mkdir(path.join(source, "assistant", "personas"), { recursive: true });
    await Promise.all([
      writeFile(path.join(source, "index.html"), "<main></main>"),
      writeFile(path.join(source, "src", "App.jsx"), "export default null;"),
      writeFile(path.join(source, "scripts", "start.mjs"), "export {};"),
      writeFile(path.join(source, "assistant", "personas", "built-in.json"), "{}"),
      writeFile(path.join(source, "models", "model-paths.json"), "{}"),
      writeFile(path.join(source, "models", "recommended-models.json"), "{}"),
      writeFile(path.join(source, "models", "yolo-models.json"), "{}"),
      writeFile(path.join(source, "models", "background-removal-models.json"), "{}"),
      writeFile(path.join(source, "models", "README.md"), "# models"),
    ]);
    const staged = await stageReleasePackage({ projectRoot: source, stagingDirectory: staging, wrapperName: "release" });
    await run("tar", ["-cf", "../release-direct.tar", "-C", "release", ".xirai-release-manifest.json", ...staged.manifest.files], { cwd: staging });
    await assert.rejects(prepareUpdate({ projectRoot: current, archivePath: directArchivePath }), /恰有一层 wrapper/);
    await writeFile(path.join(staged.packageRoot, "src", "injected.test.jsx"), "bad");
    await createTar(archivePath, staging, "release");
    await assert.rejects(prepareUpdate({ projectRoot: current, archivePath }), /未知或禁止/);
  } finally { await rm(workspace, { recursive: true, force: true }); }
});

test("assistant assets replace as a managed directory while user personas and model weights remain", async (context) => {
  try { await run("tar", ["--version"]); } catch { context.skip("system tar is unavailable"); return; }
  const workspace = await temporaryDirectory("xirai-assistant-update-");
  const current = path.join(workspace, "current");
  const packageParent = path.join(workspace, "package");
  const packageRoot = path.join(packageParent, "release");
  const archivePath = path.join(workspace, "update.tar");
  try {
    await Promise.all([createProjectRoot(current), createProjectRoot(packageRoot)]);
    await Promise.all([
      mkdir(path.join(current, "assistant", "personas"), { recursive: true }),
      mkdir(path.join(current, "state-cache", "assistant-personas"), { recursive: true }),
      mkdir(path.join(current, "models", "checkpoints"), { recursive: true }),
      mkdir(path.join(packageRoot, "assistant", "personas"), { recursive: true }),
    ]);
    await Promise.all([
      writeFile(path.join(current, "assistant", "personas", "obsolete.json"), "old"),
      writeFile(path.join(current, "state-cache", "assistant-personas", "user.json"), "user"),
      writeFile(path.join(current, "models", "checkpoints", "user.safetensors"), "weight"),
      writeFile(path.join(packageRoot, "assistant", "personas", "new.json"), "new"),
    ]);
    await createTar(archivePath, packageParent, "release");
    const prepared = await prepareUpdate({ projectRoot: current, archivePath });
    await applyPreparedUpdate({ projectRoot: current, prepared });
    assert.equal(await readFile(path.join(current, "assistant", "personas", "new.json"), "utf8"), "new");
    await assert.rejects(access(path.join(current, "assistant", "personas", "obsolete.json")));
    assert.equal(await readFile(path.join(current, "state-cache", "assistant-personas", "user.json"), "utf8"), "user");
    assert.equal(await readFile(path.join(current, "models", "checkpoints", "user.safetensors"), "utf8"), "weight");
  } finally { await rm(workspace, { recursive: true, force: true }); }
});
