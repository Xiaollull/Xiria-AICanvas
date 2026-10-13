import assert from "node:assert/strict";
import { test } from "node:test";
import { readFile } from "node:fs/promises";
import { loraFetch, publishedUiSnapshot, rebasePublishedTagSpans, rebasePublishedWorkspace } from "../src/lora-library-client.js";
import { createLoraRepairController } from "../src/lora-repair-controller.js";
import { recordedImageLoraReference, resolvedImageLoraIdentity } from "../src/image-info-lora-reference.js";
import { stripImageInfoMatches } from "../src/image-metadata.js";
import { inspectLoraIdentity } from "../src/lora-identity.js";

test("published confirmed name spans survive late local prose/weight edits, not global text replacement", () => {
  const old = "keep <lora:style/A:0.8137> tail <lora:A:0.2> literal style/A";
  const fresh = old.replace("<lora:style/A:", "<lora:concept/A:");
  const live = old.replace("0.8137", "0.9137") + " local note";
  assert.equal(rebasePublishedTagSpans(old, live, fresh), "keep <lora:concept/A:0.9137> tail <lora:A:0.2> literal style/A local note");
  assert.equal(rebasePublishedTagSpans(old, old.replace("style/A:0", "other/B:0"), fresh), old.replace("style/A:0", "other/B:0"));
  assert.equal(rebasePublishedTagSpans(old, live + " <lora:style/A:1>", fresh), "keep <lora:concept/A:0.9137> tail <lora:A:0.2> literal style/A local note <lora:concept/A:1>");
  assert.equal(rebasePublishedTagSpans(old, live, fresh + " server unrelated edit"), live);
  assert.equal(rebasePublishedTagSpans("<lora:A:1>", "<lora:A:1> local", "<lora:B:1>"), "<lora:A:1> local");
});

test("confirmed exact-name evidence survives tag insert/delete/duplicate/reorder without rewriting unconfirmed names", () => {
  const old = "prefix <lora:style/A:0.8> <lora:other/B:te=0.2:unet=0.2> <lora:A:1> tail";
  const fresh = old.replace("<lora:style/A:", "<lora:concept/A:");
  const local = "new <lora:other/Unconfirmed:1> <lora:other/B:0.7> <lora:style/A:te=0.9:unet=0.9> <lora:style/A:-0.2> literal style/A";
  assert.equal(rebasePublishedTagSpans(old, local, fresh), "new <lora:other/Unconfirmed:1> <lora:other/B:0.7> <lora:concept/A:te=0.9:unet=0.9> <lora:concept/A:-0.2> literal style/A");
  assert.equal(rebasePublishedTagSpans(old, "all deleted; literal style/A", fresh), "all deleted; literal style/A");
  const duplicate = "<lora:style/A:1> <lora:style/A:2>";
  assert.equal(rebasePublishedTagSpans(duplicate, "<lora:style/A:3> local", "<lora:concept/A:1> <lora:style/A:2>"), "<lora:style/A:3> local");
  assert.equal(rebasePublishedTagSpans(old, "<lora:other/UserChosen:0.9>", fresh), "<lora:other/UserChosen:0.9>");
});

test("merged PUT representation is not an unrendered local edit basis on the next publication", async () => {
  const originalFetch = globalThis.fetch;
  const local = { theme: { accent: "old" }, workspace: { positive: "unchanged quiet tab", loras: [] } };
  const latest = structuredClone(local); latest.workspace.positive = "later edit in another tab";
  try {
    globalThis.fetch = async () => new Response(JSON.stringify({ state: latest }), { headers: { "Content-Type": "application/json", "X-XirAI-LoRA-Revision": "0", "X-XirAI-Document-Version": "merged-version" } });
    await loraFetch("/api/ui-state", { method: "PUT", body: JSON.stringify(local) });
    assert.deepEqual(publishedUiSnapshot(), local);
    const published = structuredClone(latest); published.workspace.loras = [{ value: "style/restored" }];
    assert.equal(rebasePublishedWorkspace(publishedUiSnapshot(), local, published).workspace.positive, latest.workspace.positive);
  } finally { globalThis.fetch = originalFetch; }
});

test("catalog's actual middleware catches asynchronous published-owner failures", async () => {
  const source = await readFile(new URL("../vite.config.js", import.meta.url), "utf8");
  const body = source.slice(source.indexOf("async function modelApi("), source.indexOf("\nfunction modelApiPlugin()"));
  const api = new Function("getConfiguredDirectory", "loraPublishedStore", `${body};return modelApi;`)(
    async () => "fixture", { access: async () => { throw Object.assign(new Error("lease EPERM"), { code: "EPERM" }); } });
  let value; const response = { headers: {}, setHeader(key, val) { this.headers[key] = val; }, getHeader(key) { return this.headers[key]; }, end(text) { value = JSON.parse(text); } };
  await api({ url: "/api/loras?engine=SD", method: "GET" }, response, () => assert.fail("unmatched route"));
  assert.equal(response.statusCode, 500); assert.equal(value.error, "lease EPERM");
});

test("image read then Apply re-resolves recorded A, not target C paired with A's path", () => {
  const A = "11111111-1111-4111-8111-111111111111", C = "22222222-2222-4222-8222-222222222222";
  const raw = { name: "A", path: "style/A.safetensors", instanceId: A, contentHash: "a".repeat(64), contentRevision: 1, weight: .8137 };
  const verdict = { instanceId: C, contentHash: "c".repeat(64), contentRevision: 2 };
  const read = resolvedImageLoraIdentity(raw, verdict), submitted = stripImageInfoMatches({ loras: [{ ...read, match: { relink: { kind: "replace", promptRewriteAllowed: true } } }] }).loras[0];
  assert.equal(inspectLoraIdentity(read).fatal, false);
  assert.equal(read.instanceId, C); assert.equal(read.contentRevision, 2);
  assert.deepEqual(recordedImageLoraReference(submitted), recordedImageLoraReference(raw));
  assert.deepEqual(resolvedImageLoraIdentity(submitted, verdict), read);
  const legacy = { name: "A", path: raw.path, weight: raw.weight };
  assert.deepEqual(recordedImageLoraReference(resolvedImageLoraIdentity(legacy, verdict)), { value: raw.path, weight: raw.weight });
  // New PNG with current executed path C must not be redirected by an older audit A.
  assert.equal(recordedImageLoraReference({ ...read, path: "concept/C.safetensors" }).instanceId, C);
});

test("publication workspace recursive merge carries confirmed group/preset/engine/image tags and unrelated local fields", () => {
  const text = "<lora:style/A:te=0.8:unet=0.8> words", freshText = text.replace("style/A", "concept/A");
  const base = { theme: { accent: "violet" }, workspace: { positive: text, imageToImage: { positive: text }, engineSettingsByEngine: { byEngine: { SD: { positive: text } } }, promptPresets: { note: text } } };
  const local = structuredClone(base), fresh = structuredClone(base);
  local.theme.accent = "custom"; local.workspace.positive += " local";
  for (const owner of [fresh.workspace, fresh.workspace.imageToImage, fresh.workspace.engineSettingsByEngine.byEngine.SD, fresh.workspace.promptPresets]) owner[owner.note ? "note" : "positive"] = freshText;
  const result = rebasePublishedWorkspace(base, local, fresh);
  assert.equal(result.workspace.positive, freshText + " local");
  assert.equal(result.theme.accent, "custom");
  assert.equal(result.workspace.imageToImage.positive, freshText);
  const presets = { workspace: { promptPresets: { records: [{ id: "preset", content: text }] } } };
  const edited = structuredClone(presets), published = structuredClone(presets);
  edited.workspace.promptPresets.records[0].content += " local";
  published.workspace.promptPresets.records[0].content = freshText;
  assert.equal(rebasePublishedWorkspace(presets, edited, published).workspace.promptPresets.records[0].content, freshText + " local");
});

test("own published revision does not invalidate receipt; external revision retains choices and flags review", async () => {
  let controller;
  const response = (value) => ({ ok: true, json: async () => value });
  controller = createLoraRepairController({ request: async (url) => {
    if (url.endsWith("/state")) return response({ available: true, contractVersion: 1 });
    if (url.endsWith("/scan")) return response({ scanId: "scan", complete: true, revision: 1, rows: [{ sourceId: "a", recommendation: { value: "concept/A", kind: "relocate" }, impacts: { groups: [], cards: [], mounts: 1 } }] });
    if (url.endsWith("/preview")) return response({ conflicts: [], affected: { workspace: 2, cards: 1, gallery: 1 } });
    controller.markRevision(2);
    return response({ operationId: "op", status: "published", revision: 2 });
  } });
  await controller.availability(); await controller.scan(["SD"]); await controller.preview(); await controller.commit();
  controller.markRevision(2);
  assert.equal(controller.state().phase, "result"); assert.equal(controller.state().error, "");
  assert.equal(controller.state().receipt.affected.workspace, 2);
  controller.markRevision(3);
  assert.equal(controller.state().draft.rows[0].target.value, "concept/A");
  assert.equal(controller.state().draft.rows[0].needsReview, true);
});
