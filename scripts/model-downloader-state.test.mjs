import assert from "node:assert/strict";
import test from "node:test";
import { completedDownloadNoticeKey } from "../src/model-downloader-state.js";

const complete = {
  jobId: "download-1", active: false, status: "complete", totalModels: 1,
  completedModels: 1, failedModels: 0, items: [{ status: "complete" }], updatedAt: 100,
};

test("polling and restored timestamps do not revive a dismissed completion", () => {
  const key = completedDownloadNoticeKey(complete);
  assert.ok(key);
  assert.equal(completedDownloadNoticeKey({ ...complete, updatedAt: 200 }), key);
});

test("active workers, pending items and failures must stay visible", () => {
  for (const update of [
    { active: true }, { status: "downloading" }, { status: "metadata" },
    { status: "error" }, { status: "partial" }, { failedModels: 1 },
    { items: [{ status: "complete" }, { status: "waiting" }] },
    { items: [{ status: "error" }] },
  ]) assert.equal(completedDownloadNoticeKey({ ...complete, ...update }), "", JSON.stringify(update));
});

test("new jobs and appended queue completions get their own notice", () => {
  const key = completedDownloadNoticeKey(complete);
  assert.notEqual(completedDownloadNoticeKey({ ...complete, jobId: "download-2" }), key);
  assert.notEqual(completedDownloadNoticeKey({ ...complete, totalModels: 2, completedModels: 2, items: [{ status: "complete" }, { status: "complete" }] }), key);
});

test("already-installed no-op completions can dismiss; absent jobs cannot", () => {
  assert.ok(completedDownloadNoticeKey({ jobId: "already-installed", status: "complete", active: false, totalModels: 0, completedModels: 0, items: [] }));
  assert.equal(completedDownloadNoticeKey(null), "");
  assert.equal(completedDownloadNoticeKey({ status: "idle" }), "");
});
