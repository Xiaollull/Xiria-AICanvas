import assert from "node:assert/strict";
import test from "node:test";

import { appendDownloadQueueState, filterPendingRecommendedArtifacts, itemStatusIsTerminal } from "./model-download-queue.mjs";

const digest = (character) => character.repeat(64);

test("recommended queue omits verified and already queued artifacts", () => {
  const artifacts = [
    { id: "installed", sha256: digest("a") },
    { id: "downloading", sha256: digest("b") },
    { id: "missing", sha256: digest("c") },
    { id: "unverified" },
  ];
  const pending = filterPendingRecommendedArtifacts(artifacts, new Map([[digest("a"), "models/a"]]), [
    { status: "downloading", sha256: digest("b") },
    { status: "error", sha256: digest("c") },
  ]);
  assert.deepEqual(pending.map((item) => item.id), ["unverified"]);
});

test("download batches append stable indexes and refresh targets", () => {
  const initial = {
    status: "downloading",
    totalModels: 1,
    items: [{ index: 1, url: "current", status: "downloading" }],
    targets: [{ kind: "diffusion_model", engine: "Anima" }],
  };
  const first = appendDownloadQueueState(initial, {
    source: "manual",
    kind: "lora",
    engine: "Anima",
    items: [{ label: "one" }, { label: "two" }],
  });
  const second = appendDownloadQueueState(first.state, {
    source: "manual",
    kind: "lora",
    engine: "Anima",
    items: [{ label: "three" }],
  });
  assert.equal(first.startIndex, 2);
  assert.deepEqual(second.state.items.map((item) => item.index), [1, 2, 3, 4]);
  assert.deepEqual(second.state.targets, [
    { kind: "diffusion_model", engine: "Anima" },
    { kind: "lora", engine: "Anima" },
  ]);
  assert.equal(itemStatusIsTerminal("complete"), true);
  assert.equal(itemStatusIsTerminal("downloading"), false);
});
