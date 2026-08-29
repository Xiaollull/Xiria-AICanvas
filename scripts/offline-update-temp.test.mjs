import assert from "node:assert/strict";
import { access, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { createOfflineUpdateTemp, removeOfflineUpdateTemp } from "./offline-update-temp.mjs";

test("offline update cleanup requires the matching ownership token", async () => {
  const projectRoot = await mkdtemp(path.join(os.tmpdir(), "xirai-update-temp-project-"));
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
