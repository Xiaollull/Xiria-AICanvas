import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { acquireOfflineUpdateLock } from "./offline-update-lock.mjs";

test("offline update lock excludes concurrent local operations", async () => {
  const projectRoot = await mkdtemp(path.join(os.tmpdir(), "xirai-update-lock-project-"));
  const stateDirectory = path.join(projectRoot, "state-cache");
  try {
    const first = await acquireOfflineUpdateLock({ stateDirectory, projectRoot, operation: "apply" });
    await assert.rejects(
      acquireOfflineUpdateLock({ stateDirectory, projectRoot, operation: "recovery" }),
      (error) => error.code === "UPDATE_BUSY",
    );
    await first.release();
    const second = await acquireOfflineUpdateLock({ stateDirectory, projectRoot, operation: "recovery" });
    await second.release();
  } finally {
    await rm(projectRoot, { recursive: true, force: true });
  }
});
