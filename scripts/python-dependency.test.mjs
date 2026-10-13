import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { pythonDependencyOptions } from "./python-dependency.test.helper.mjs";

const python = process.env.P10_PYTHON || "python";
const cwd = import.meta.dirname;

test("a present Python dependency never skips its tests", () => {
  assert.deepEqual(pythonDependencyOptions(python, cwd, ["json"]), { skip: false });
});

test("only missing Python dependencies produce an explicit skip reason", () => {
  assert.deepEqual(pythonDependencyOptions(python, cwd, ["json", "xirai_dependency_that_does_not_exist_130"]),
    { skip: "Python dependencies unavailable: xirai_dependency_that_does_not_exist_130" });
});

test("an invalid Python executable fails rather than skipping", () => {
  assert.throws(() => pythonDependencyOptions(join(tmpdir(), "xirai-no-python-130.exe"), cwd, ["json"]));
});

test("an installed but broken module remains enabled and fails its real import", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xirai-broken-python-dependency-"));
  try {
    await writeFile(join(directory, "xirai_broken_dependency.py"), "raise RuntimeError('installed package is broken')\n");
    assert.deepEqual(pythonDependencyOptions(python, directory, ["xirai_broken_dependency"]), { skip: false });
    const result = spawnSync(python, ["-B", "-c", "import xirai_broken_dependency"],
      { cwd: directory, encoding: "utf8", windowsHide: true, timeout: 15000 });
    assert.equal(result.status, 1);
    assert.match(result.stderr, /RuntimeError: installed package is broken/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
