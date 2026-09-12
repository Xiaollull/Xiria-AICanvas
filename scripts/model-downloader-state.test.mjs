import assert from "node:assert/strict";
import test from "node:test";

import {
  configuredDownloadCredentialCount,
  createLatestRequestGate,
  normalizedDownloadCredentials,
  reconcileRecommendedSelection,
} from "../src/model-downloader-state.js";

test("download credentials use one trimmed representation for count, gates, and payloads", () => {
  assert.deepEqual(normalizedDownloadCredentials({ civitai: "   ", huggingface: " hf_live ", modelscope: "\tms\n" }), {
    civitai: "",
    huggingface: "hf_live",
    modelscope: "ms",
  });
  assert.equal(configuredDownloadCredentialCount({ civitai: "   ", huggingface: " hf_live ", modelscope: "\t" }), 1);
});

test("catalog replacement keeps valid selections and repairs removed model or dependencies", () => {
  const families = [{
    id: "family",
    models: [{ id: "model-new" }, { id: "model-kept" }],
    textEncoders: [{ id: "encoder-new" }],
    vaes: [],
  }];
  assert.deepEqual(reconcileRecommendedSelection(families, "family", {
    modelId: "model-kept",
    textEncoderId: "encoder-removed",
    vaeId: "vae-removed",
  }), {
    familyId: "family",
    selection: { modelId: "model-kept", textEncoderId: "encoder-new", vaeId: "" },
  });
  assert.deepEqual(reconcileRecommendedSelection(families, "family", { modelId: "model-removed" }).selection.modelId, "model-new");
  assert.deepEqual(reconcileRecommendedSelection(families, "removed", { modelId: "stale" }), {
    familyId: "",
    selection: { modelId: "", textEncoderId: "", vaeId: "" },
  });
});

test("a superseded delayed request cannot commit over a newer download job", async () => {
  const gate = createLatestRequestGate();
  const committed = [];
  let releaseOld;
  const oldToken = gate.begin();
  const oldRequest = new Promise((resolve) => { releaseOld = resolve; }).then((value) => gate.commit(oldToken, () => committed.push(value)));
  const newToken = gate.begin();
  assert.equal(gate.commit(newToken, () => committed.push("recommended-post")), true);
  releaseOld("stale-poll");
  assert.equal(await oldRequest, false);
  assert.deepEqual(committed, ["recommended-post"]);
});
