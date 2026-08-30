import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";
import {
  UPDATE_BUSY_STATUSES,
  UPDATE_OCCUPIED_STATUSES,
  UPDATE_RESTART_SESSION_KEY,
  clearUpdateRestart,
  markUpdateRestart,
  updateBusy,
  updateRestartPending,
  waitForUpdatedApplication,
} from "../src/update-navigation.js";

function memoryStorage() {
  const values = new Map();
  return {
    getItem: (key) => values.get(key) ?? null,
    removeItem: (key) => values.delete(key),
    setItem: (key, value) => values.set(key, value),
  };
}

test("update restart marker survives a page reload and expires safely", () => {
  const storage = memoryStorage();
  assert.equal(markUpdateRestart(storage, 1000), true);
  assert.equal(storage.getItem(UPDATE_RESTART_SESSION_KEY), "1000");
  assert.equal(updateRestartPending(storage, 2000), true);
  assert.equal(updateRestartPending(storage, 400000), false);
  assert.equal(storage.getItem(UPDATE_RESTART_SESSION_KEY), null);
  markUpdateRestart(storage, 500000);
  clearUpdateRestart(storage);
  assert.equal(updateRestartPending(storage, 500001), false);
});

test("updated application polling retries failures before returning home", async () => {
  const results = [new Error("offline"), false, true];
  const waits = [];
  let returned = 0;
  let time = 0;
  const completed = await waitForUpdatedApplication({
    checkHealth: async () => {
      const result = results.shift();
      if (result instanceof Error) throw result;
      return result;
    },
    returnHome: () => { returned += 1; },
    wait: async (milliseconds) => { waits.push(milliseconds); time += milliseconds; },
    now: () => time,
    timeoutMs: 10000,
    initialDelayMs: 1800,
    retryDelayMs: 600,
  });

  assert.equal(completed, true);
  assert.equal(returned, 1);
  assert.deepEqual(waits, [1800, 600, 600]);
});

test("updated application polling reports a bounded timeout", async () => {
  let time = 0;
  await assert.rejects(waitForUpdatedApplication({
    checkHealth: async () => false,
    returnHome: () => assert.fail("timed out restart must not navigate"),
    wait: async (milliseconds) => { time += milliseconds; },
    now: () => time,
    timeoutMs: 1200,
    initialDelayMs: 0,
    retryDelayMs: 600,
  }), /应用重启超时/);
});

test("an in-flight download holds the update state exactly as an upload does", () => {
  // The online flow added a status. Every gate that refuses a second update has to know it, or a
  // download can be started on top of one already running.
  for (const status of ["uploading", "downloading", "preparing", "applying", "repairing"]) {
    assert.equal(updateBusy(status), true, status);
    assert.ok(UPDATE_OCCUPIED_STATUSES.includes(status), status);
  }
  for (const status of ["idle", "error", "ready", "complete"]) {
    assert.equal(updateBusy(status), false, status);
  }
  // An archive already on disk is not busy, but it does occupy the workspace.
  for (const status of ["ready", "complete"]) {
    assert.ok(UPDATE_OCCUPIED_STATUSES.includes(status), status);
  }
  assert.equal(updateBusy(undefined), false);
});

test("both update gates read the shared status list instead of their own copy", async () => {
  const [server, page] = await Promise.all([
    readFile(new URL("../vite.config.js", import.meta.url), "utf8"),
    readFile(new URL("../src/ManualUpdatePage.jsx", import.meta.url), "utf8"),
  ]);
  for (const source of [server, page]) {
    // A literal list beside the shared one is the shape of the bug this guards: it stays correct
    // until a status is added, and then only one of the copies is updated.
    assert.doesNotMatch(source, /\["uploading", "preparing", "applying", "repairing"\]/);
    assert.match(source, /updateBusy\(/);
  }
  assert.ok(UPDATE_BUSY_STATUSES.includes("downloading"));
});
