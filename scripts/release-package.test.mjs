import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { access, mkdir, mkdtemp, readFile, realpath, rm, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { createReleaseManifest, RELEASE_MANIFEST_FILE, stageReleasePackage, validateReleasePackageDirectory } from "./release-package.mjs";

const RELEASE_PACKAGE_SCRIPT = fileURLToPath(new URL("./release-package.mjs", import.meta.url));

function comparePortablePaths(left, right) {
  const first = left.replace(/\\/g, "/");
  const second = right.replace(/\\/g, "/");
  return first < second ? -1 : first > second ? 1 : 0;
}

function spawnNode(args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, args, {
      ...options,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.once("error", reject);
    child.once("close", (code, signal) => resolve({ code, signal, stdout, stderr }));
  });
}

async function temporaryDirectory(prefix) {
  return realpath(await mkdtemp(path.join(os.tmpdir(), prefix)));
}

async function createReleaseSource(root) {
  for (const directory of ["src", "backend", "scripts", "public", "assistant/personas", "models"]) {
    await mkdir(path.join(root, directory), { recursive: true });
  }
  const files = {
    "package.json": JSON.stringify({ name: "xiriacanvas-ai" }), "package-lock.json": "{}", "vite.config.js": "export default {};\n",
    "index.html": "<main></main>", "backend/requirements.txt": "fastapi==1\n", "src/App.jsx": "export default null;\n",
    "scripts/start.mjs": "export {};\n", "public/logo.svg": "<svg/>", "assistant/personas/builtin.json": "{}",
    "models/model-paths.json": "{}", "models/recommended-models.json": "{}", "models/yolo-models.json": "{}",
    "models/background-removal-models.json": "{}", "models/README.md": "# Models\n",
  };
  await Promise.all(Object.entries(files).map(([relative, contents]) => writeFile(path.join(root, ...relative.split("/")), contents)));
}

test("release staging uses a deterministic allowlist and nested model config files only", async () => {
  const workspace = await temporaryDirectory("xirai-release-package-");
  const source = path.join(workspace, "source");
  const staging = path.join(workspace, "staging");
  try {
    await createReleaseSource(source);
    await Promise.all([
      mkdir(path.join(source, "models", "checkpoints"), { recursive: true }),
      mkdir(path.join(source, "state-cache", "assistant-personas"), { recursive: true }),
    ]);
    await Promise.all([
      writeFile(path.join(source, "models", "checkpoints", "user.safetensors"), "weight"),
      writeFile(path.join(source, "src", "App.test.jsx"), "test"),
      writeFile(path.join(source, "opencode.json"), "{}"),
      writeFile(path.join(source, "state-cache", "assistant-personas", "user.json"), "{}"),
    ]);
    const result = await stageReleasePackage({ projectRoot: source, stagingDirectory: staging, wrapperName: "release" });
    assert.deepEqual(result.manifest.files, [...result.manifest.files].sort(comparePortablePaths));
    assert.ok(result.manifest.files.includes("assistant/personas/builtin.json"));
    assert.ok(result.manifest.files.includes("models/model-paths.json"));
    for (const forbidden of ["models/checkpoints/user.safetensors", "src/App.test.jsx", "opencode.json", "state-cache/assistant-personas/user.json"]) {
      assert.equal(result.manifest.files.includes(forbidden), false, forbidden);
      await assert.rejects(access(path.join(result.packageRoot, ...forbidden.split("/"))));
    }
    assert.deepEqual(await validateReleasePackageDirectory({ packageRoot: result.packageRoot }), result.manifest);
  } finally { await rm(workspace, { recursive: true, force: true }); }
});

test("release package validation rejects unknown/test members and missing critical files", async () => {
  const workspace = await temporaryDirectory("xirai-release-validation-");
  const source = path.join(workspace, "source");
  const staging = path.join(workspace, "staging");
  try {
    await createReleaseSource(source);
    const { packageRoot } = await stageReleasePackage({ projectRoot: source, stagingDirectory: staging });
    await writeFile(path.join(packageRoot, "src", "unexpected.test.jsx"), "bad");
    await assert.rejects(validateReleasePackageDirectory({ packageRoot }), /未知或禁止/);
    await rm(path.join(packageRoot, "src", "unexpected.test.jsx"));
    await rm(path.join(packageRoot, "index.html"));
    await assert.rejects(validateReleasePackageDirectory({ packageRoot }), /关键文件|缺少 manifest|不是安全普通文件/);
  } finally { await rm(workspace, { recursive: true, force: true }); }
});

test("release manifest is explicit JSON rather than a repository snapshot", async () => {
  const workspace = await temporaryDirectory("xirai-release-manifest-");
  try {
    await createReleaseSource(workspace);
    const manifest = await createReleaseManifest({ projectRoot: workspace });
    assert.equal(manifest.files.includes(RELEASE_MANIFEST_FILE), false);
    assert.equal(manifest.files.some((file) => file.startsWith("models/") && file.endsWith(".safetensors")), false);
    assert.equal(JSON.parse(await readFile(path.join(workspace, "package.json"), "utf8")).name, "xiriacanvas-ai");
  } finally { await rm(workspace, { recursive: true, force: true }); }
});

test("release manifest uses fixed code-unit ordering under explicit locale simulations", async () => {
  const workspace = await temporaryDirectory("xirai-release-locale-");
  const source = path.join(workspace, "source");
  const staging = path.join(workspace, "staging");
  const mixedCaseFiles = [
    "src/Alpha.jsx",
    "src/Istanbul.jsx",
    "src/Zebra.jsx",
    "src/_under.jsx",
    "src/apple-two.jsx",
    "src/istanbul-two.jsx",
  ];
  try {
    await createReleaseSource(source);
    await Promise.all(mixedCaseFiles.map((relativePath) => writeFile(path.join(source, ...relativePath.split("/")), relativePath)));
    const result = await stageReleasePackage({ projectRoot: source, stagingDirectory: staging });
    assert.deepEqual(
      result.manifest.files.filter((relativePath) => mixedCaseFiles.includes(relativePath)),
      mixedCaseFiles,
    );

    const validationScript = `
      const collator = new Intl.Collator(process.env.XIRAI_TEST_LOCALE);
      String.prototype.localeCompare = function (other) { return collator.compare(String(this), String(other)); };
      const { validateReleasePackageDirectory } = await import(process.env.XIRAI_RELEASE_MODULE_URL);
      await validateReleasePackageDirectory({ packageRoot: process.env.XIRAI_PACKAGE_ROOT });
    `;
    for (const locale of ["en", "tr", "sv"]) {
      const child = await spawnNode(["--input-type=module", "--eval", validationScript], {
        env: {
          ...process.env,
          XIRAI_TEST_LOCALE: locale,
          XIRAI_RELEASE_MODULE_URL: new URL("./release-package.mjs", import.meta.url).href,
          XIRAI_PACKAGE_ROOT: result.packageRoot,
        },
      });
      assert.equal(child.code, 0, `${locale}: ${child.stderr}`);
    }
  } finally { await rm(workspace, { recursive: true, force: true }); }
});

test("release staging never removes an existing caller-supplied directory", async () => {
  const workspace = await temporaryDirectory("xirai-release-existing-stage-");
  const source = path.join(workspace, "source");
  const staging = path.join(workspace, "staging");
  const sentinel = path.join(staging, "keep.txt");
  try {
    await createReleaseSource(source);
    await mkdir(staging);
    await writeFile(sentinel, "keep");
    await assert.rejects(stageReleasePackage({ projectRoot: source, stagingDirectory: staging }), /必须不存在/);
    assert.equal(await readFile(sentinel, "utf8"), "keep");
  } finally { await rm(workspace, { recursive: true, force: true }); }
});

test("release staging rejects the project root, its parent, and a path inside it", async () => {
  const workspace = await temporaryDirectory("xirai-release-overlap-");
  const source = path.join(workspace, "source");
  const sentinel = path.join(source, "src", "App.jsx");
  try {
    await createReleaseSource(source);
    for (const stagingDirectory of [source, workspace, path.join(source, "release-stage")]) {
      await assert.rejects(
        stageReleasePackage({ projectRoot: source, stagingDirectory }),
        /相同|内部|包含/,
      );
      assert.equal(await readFile(sentinel, "utf8"), "export default null;\n");
    }
  } finally { await rm(workspace, { recursive: true, force: true }); }
});

test("release staging rejects a symlink or junction parent", async (context) => {
  const workspace = await temporaryDirectory("xirai-release-linked-stage-");
  const source = path.join(workspace, "source");
  const externalParent = path.join(workspace, "external-parent");
  const linkedParent = path.join(workspace, "linked-parent");
  try {
    await createReleaseSource(source);
    await mkdir(externalParent);
    try {
      await symlink(externalParent, linkedParent, process.platform === "win32" ? "junction" : "dir");
    } catch (error) {
      if (["EACCES", "EPERM", "ENOSYS", "ENOTSUP"].includes(error.code)) {
        context.skip(`directory links are unavailable: ${error.code}`);
        return;
      }
      throw error;
    }
    const staging = path.join(linkedParent, "staging");
    await assert.rejects(stageReleasePackage({ projectRoot: source, stagingDirectory: staging }), /符号链接|目录联接|路径别名/);
    await assert.rejects(access(path.join(externalParent, "staging")));
  } finally { await rm(workspace, { recursive: true, force: true }); }
});

test("release staging treats Windows case aliases as overlapping project paths", {
  skip: process.platform !== "win32",
}, async () => {
  const workspace = await temporaryDirectory("xirai-release-case-alias-");
  const source = path.join(workspace, "source");
  try {
    await createReleaseSource(source);
    for (const stagingDirectory of [
      source.toUpperCase(),
      path.join(source.toUpperCase(), "RELEASE-STAGE"),
      workspace.toUpperCase(),
    ]) {
      await assert.rejects(
        stageReleasePackage({ projectRoot: source, stagingDirectory }),
        /相同|内部|包含/,
      );
    }
    assert.equal(await readFile(path.join(source, "src", "App.jsx"), "utf8"), "export default null;\n");
  } finally { await rm(workspace, { recursive: true, force: true }); }
});

test("release package CLI reports usage and exits nonzero when required arguments are missing", async () => {
  const workspace = await temporaryDirectory("xirai-release-cli-");
  try {
    const child = await spawnNode([RELEASE_PACKAGE_SCRIPT], { cwd: workspace });
    assert.notEqual(child.code, 0);
    assert.equal(child.signal, null);
    assert.match(child.stderr, /用法：node scripts\/release-package\.mjs/);
    assert.equal(child.stdout, "");
  } finally { await rm(workspace, { recursive: true, force: true }); }
});
