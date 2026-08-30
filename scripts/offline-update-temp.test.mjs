import assert from "node:assert/strict";
import { access, mkdtemp, readFile, realpath, rm, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { assertOfflineUpdateTemp, createOfflineUpdateTemp, removeOfflineUpdateTemp } from "./offline-update-temp.mjs";

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
  const previous = { TMPDIR: process.env.TMPDIR, TEMP: process.env.TEMP, TMP: process.env.TMP };
  context.after(async () => {
    for (const [name, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[name];
      else process.env[name] = value;
    }
    await rm(linked, { recursive: true, force: true }).catch(() => {});
    await rm(real, { recursive: true, force: true });
    await rm(projectRoot, { recursive: true, force: true });
  });
  for (const name of ["TMPDIR", "TEMP", "TMP"]) process.env[name] = linked;
  assert.equal(os.tmpdir(), linked);
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
