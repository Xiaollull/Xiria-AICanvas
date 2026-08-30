import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { access, mkdtemp, readFile, realpath, rm, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { assertOfflineUpdateTemp, createOfflineUpdateTemp, removeOfflineUpdateTemp } from "./offline-update-temp.mjs";

/** The 8.3 alias Windows keeps for a directory whose name it cannot fit, or null.
 *
 * `%~sI` is cmd's short-name expansion. The path is passed unquoted because Node's Windows
 * argument escaping is not the quoting cmd.exe parses; a path holding a space therefore comes
 * back as several tokens, which the existence check below turns into a skip rather than a wrong
 * answer. Every path this is called with is a mkdtemp directory under the temp directory.
 */
function shortNameFor(directory) {
  if (process.platform !== "win32") return null;
  try {
    const output = execFileSync("cmd.exe", ["/d", "/c", `for %I in (${directory}) do @echo %~sI`], {
      encoding: "utf8",
      windowsHide: true,
    }).trim();
    return output && existsSync(output) ? output : null;
  } catch {
    return null;
  }
}

async function useTemporaryDirectory(context, directory) {
  const previous = { TMPDIR: process.env.TMPDIR, TEMP: process.env.TEMP, TMP: process.env.TMP };
  context.after(() => {
    for (const [name, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[name];
      else process.env[name] = value;
    }
  });
  for (const name of ["TMPDIR", "TEMP", "TMP"]) process.env[name] = directory;
  assert.equal(os.tmpdir(), directory);
}

test("a short-name temp directory validates the record it created", async (context) => {
  // A short name is not a link, and that distinction is the whole bug: `fs.realpathSync` walks
  // links and hands a short name straight back, while the `realpath` used to record a created
  // directory expands it. Resolving the two ends of the comparison with different implementations
  // passed on every machine whose temp directory is spelled in full and failed on the ones that
  // are not — a Windows profile name too long for 8.3, which is what a CI runner reports as
  // C:\Users\RUNNER~1\AppData\Local\Temp.
  const real = await realpath(await mkdtemp(path.join(os.tmpdir(), "xirai-update-temp-shortname-")));
  context.after(async () => rm(real, { recursive: true, force: true }));
  const short = shortNameFor(real);
  if (!short || short === real) {
    context.skip("this filesystem does not keep 8.3 short names, so the two spellings cannot differ");
    return;
  }
  assert.equal(await realpath(short), real, "the short name is another spelling of the same directory");

  const projectRoot = await realpath(await mkdtemp(path.join(os.tmpdir(), "xirai-update-temp-short-project-")));
  context.after(async () => rm(projectRoot, { recursive: true, force: true }));
  await useTemporaryDirectory(context, short);

  const ownership = await createOfflineUpdateTemp({ projectRoot, prefix: "xirai-update-stage-", kind: "stage" });
  assert.equal(ownership.path, await realpath(ownership.path));
  assert.equal(await assertOfflineUpdateTemp({
    projectRoot,
    record: ownership,
    prefix: "xirai-update-stage-",
    kind: "stage",
  }), ownership.path);
  await removeOfflineUpdateTemp({ projectRoot, record: ownership, prefix: "xirai-update-stage-", kind: "stage" });
});

test("a temp directory reached through a link still validates the record it created", async (context) => {
  // os.tmpdir() is not always its own realpath: Windows writes TEMP as an 8.3 short name for a
  // profile it cannot fit (C:\Users\JOHNSM~1\...), which is also what a CI runner exports, and
  // macOS reaches /var through a symlink. Creation stores the canonical path, so validation has
  // to accept it — this used to reject the record the module had just written, which left offline
  // updates unusable on those machines rather than failing any test.
  const real = await realpath(await mkdtemp(path.join(os.tmpdir(), "xirai-update-temp-real-")));
  const linked = `${real}-link`;
  const projectRoot = await realpath(await mkdtemp(path.join(os.tmpdir(), "xirai-update-temp-linked-project-")));
  await symlink(real, linked, "junction");
  context.after(async () => {
    await rm(linked, { recursive: true, force: true }).catch(() => {});
    await rm(real, { recursive: true, force: true });
    await rm(projectRoot, { recursive: true, force: true });
  });
  await useTemporaryDirectory(context, linked);
  assert.notEqual(await realpath(linked), linked);

  const ownership = await createOfflineUpdateTemp({ projectRoot, prefix: "xirai-update-stage-", kind: "stage" });
  assert.equal(ownership.path, await realpath(ownership.path), "the record stores the canonical path");
  const staged = await assertOfflineUpdateTemp({
    projectRoot,
    record: ownership,
    prefix: "xirai-update-stage-",
    kind: "stage",
  });
  assert.equal(staged, ownership.path);
  await removeOfflineUpdateTemp({ projectRoot, record: ownership, prefix: "xirai-update-stage-", kind: "stage" });
  await assert.rejects(access(ownership.path));
});

test("offline update cleanup requires the matching ownership token", async () => {
  // Canonical, because the ownership check compares a resolved staging path against this root and
  // `os.tmpdir()` is a short name on a Windows CI runner (C:\Users\RUNNER~1\...).
  const projectRoot = await realpath(await mkdtemp(path.join(os.tmpdir(), "xirai-update-temp-project-")));
  let ownership;
  try {
    ownership = await createOfflineUpdateTemp({ projectRoot, prefix: "xirai-update-stage-", kind: "stage" });
    await writeFile(path.join(ownership.path, "keep.txt"), "owned");
    await assert.rejects(removeOfflineUpdateTemp({
      projectRoot,
      record: { ...ownership, token: "wrong" },
      prefix: "xirai-update-stage-",
      kind: "stage",
    }), /所有权不匹配/);
    assert.equal(await readFile(path.join(ownership.path, "keep.txt"), "utf8"), "owned");
    await removeOfflineUpdateTemp({ projectRoot, record: ownership, prefix: "xirai-update-stage-", kind: "stage" });
    await assert.rejects(access(ownership.path));
    ownership = null;
  } finally {
    if (ownership) await rm(ownership.path, { recursive: true, force: true });
    await rm(projectRoot, { recursive: true, force: true });
  }
});
