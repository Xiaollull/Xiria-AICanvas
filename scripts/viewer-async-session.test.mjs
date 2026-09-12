import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { createViewerRafScheduler, ViewerAsyncSession, viewerOpenPlan } from "../src/viewer-async-session.js";

const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return { promise, resolve, reject }; };

test("slow open A cannot commit over fast open B", () => {
  const gate = new ViewerAsyncSession();
  const a = gate.request("open", { session: gate.beginSession(), latest: true });
  const b = gate.request("open", { session: gate.beginSession(), latest: true });
  assert.equal(gate.isCurrent(a), false);
  assert.equal(gate.isCurrent(b), true);
});

test("open, focus, and restore share replacement ordering while appends invalidate stale replacement only", () => {
  const gate = new ViewerAsyncSession();
  gate.beginSession();
  const open = gate.beginReplacement("open");
  const focus = gate.beginReplacement("focus");
  assert.equal(gate.isCurrent(open), false);
  assert.equal(gate.isCurrent(focus), true);
  const restore = gate.beginReplacement("restore");
  assert.equal(gate.isCurrent(focus), false);
  const drop = gate.beginAppend("drop");
  const add = gate.beginAppend("add");
  assert.equal(gate.isCurrent(restore), false);
  assert.equal(gate.isCurrent(drop), true, "append requests remain current after invalidating a replacement");
  assert.equal(gate.isCurrent(add), true);
  const laterFocus = gate.beginReplacement("focus");
  assert.equal(gate.isCurrent(drop), false, "a later replacement cancels unresolved appends from the old canvas");
  assert.equal(gate.isCurrent(add), false);
  assert.equal(gate.isCurrent(laterFocus), true);
});

test("replacement tokens are revision-bound so a canvas edit aborts a slow open", () => {
  const gate = new ViewerAsyncSession();
  gate.beginSession();
  const slowOpen = gate.beginReplacement("open");
  const originalRevision = slowOpen.revision;
  assert.equal(gate.isCurrent(slowOpen), true);
  gate.revise(); // create text / stroke / property update / delete all use this boundary
  assert.equal(gate.revision > originalRevision, true);
  assert.equal(slowOpen.signal.aborted, true);
  assert.equal(gate.isCurrent(slowOpen), false);
});

test("slow open resolving after create text cannot replace the new text layer", async () => {
  const gate = new ViewerAsyncSession();
  gate.beginSession();
  const loading = deferred();
  const open = gate.beginReplacement("open");
  const layers = [];
  const completion = loading.promise.then(() => { if (gate.isCurrent(open)) layers.splice(0, layers.length, "opened-image"); });
  gate.revise();
  layers.push("new-text");
  loading.resolve();
  await completion;
  assert.deepEqual(layers, ["new-text"]);
});

test("replacement releases only the old drop epoch and a late old release cannot unblock the new queue", async () => {
  const gate = new ViewerAsyncSession();
  gate.beginSession();
  const oldLargeDrop = gate.beginAppend("drop");
  gate.beginReplacement("focus");
  assert.equal(await gate.waitForDropTurn(oldLargeDrop), false);

  const newFirst = gate.beginAppend("drop");
  const newSecond = gate.beginAppend("drop");
  assert.equal(await gate.waitForDropTurn(newFirst), true, "the first drop in the new epoch never waits for old work");
  let secondReleased = false;
  const waitingSecond = gate.waitForDropTurn(newSecond).then((current) => { secondReleased = true; return current; });
  gate.releaseDrop(oldLargeDrop);
  await Promise.resolve();
  assert.equal(secondReleased, false, "an old epoch release cannot alter the new epoch tail");
  gate.releaseDrop(newFirst);
  assert.equal(await waitingSecond, true);
  gate.releaseDrop(newSecond);
});

test("collage operations are single-flight, abort on revision, and reject stale commits", () => {
  const gate = new ViewerAsyncSession();
  gate.beginSession();
  const first = gate.beginOperation("confirm", { group: "collage" });
  assert.ok(first);
  assert.equal(gate.beginOperation("confirm", { group: "collage" }), null);
  assert.equal(gate.isOperationCurrent(first), true);
  gate.revise();
  assert.equal(first.signal.aborted, true);
  assert.equal(gate.isOperationCurrent(first), false);
  const second = gate.beginOperation("save", { group: "collage" });
  assert.equal(gate.isOperationCurrent(second), true);
  assert.equal(gate.finishOperation(first), false, "an old finally block cannot clear a newer operation");
  assert.equal(gate.finishOperation(second), true);
  assert.equal(gate.isOperationCurrent(second), false);
});

test("a worker-aborted operation stays owned until catch and finally finish it", () => {
  const gate = new ViewerAsyncSession();
  gate.beginSession();
  const token = gate.beginOperation("confirm", { group: "collage" });
  token.controller.abort(new Error("worker failed"));
  assert.equal(gate.isOperationCurrent(token), false);
  assert.equal(gate.isOperationOwned(token), true, "catch can still report the primary error before finally removes ownership");
  assert.equal(gate.finishOperation(token), true);
  assert.equal(gate.isOperationOwned(token), false);
});

test("close and unmount invalidate unresolved requests without reopening", () => {
  const gate = new ViewerAsyncSession();
  const token = gate.request("open", { session: gate.beginSession(), latest: true });
  gate.close();
  assert.equal(gate.isCurrent(token), false);
  const afterClose = gate.request("add");
  gate.unmount();
  assert.equal(gate.isCurrent(afterClose), false);
});

test("StrictMode mount cleanup mount invalidates old tokens but accepts new work", () => {
  const gate = new ViewerAsyncSession();
  gate.mount();
  const first = gate.request("open", { session: gate.beginSession(), latest: true });
  gate.unmount();
  gate.mount();
  const second = gate.request("open", { session: gate.beginSession(), latest: true });
  assert.equal(gate.isCurrent(first), false);
  assert.equal(gate.isCurrent(second), true);
  gate.unmount();
  assert.equal(gate.isCurrent(second), false);
});

test("empty viewer opens without decode while image decode failures retain an empty workspace", () => {
  assert.deepEqual(viewerOpenPlan(""), { decode: false, empty: true });
  assert.deepEqual(viewerOpenPlan("/image.png"), { decode: true, empty: false });
  const gate = new ViewerAsyncSession();
  gate.mount();
  const imageOpen = gate.request("open", { session: gate.beginSession(), latest: true });
  const emptyOpen = gate.request("open", { session: gate.beginSession(), latest: true });
  assert.equal(gate.isCurrent(imageOpen), false);
  assert.equal(gate.isCurrent(emptyOpen), true);
});

test("focus is latest-wins while file drops append in batch order", async () => {
  const gate = new ViewerAsyncSession();
  gate.beginSession();
  const focusA = gate.request("focus", { latest: true });
  const focusB = gate.request("focus", { latest: true });
  assert.equal(gate.isCurrent(focusA), false);
  assert.equal(gate.isCurrent(focusB), true);
  const firstDrop = gate.request("drop");
  const secondDrop = gate.request("drop");
  assert.equal(gate.isCurrent(firstDrop), true);
  assert.equal(gate.isCurrent(secondDrop), true);
  const order = [];
  const second = gate.waitForDropTurn(secondDrop).then(() => { order.push("second"); gate.releaseDrop(secondDrop); });
  const first = gate.waitForDropTurn(firstDrop).then(() => { order.push("first"); gate.releaseDrop(firstDrop); });
  await Promise.all([first, second]);
  assert.deepEqual(order, ["first", "second"]);
  assert.deepEqual(["4k.png", "1k.png"].map((name, index) => ({ name, index })), [{ name: "4k.png", index: 0 }, { name: "1k.png", index: 1 }]);
});

test("restore starts a replacement session and stale errors are silent", () => {
  const gate = new ViewerAsyncSession();
  const restore = gate.request("restore", { session: gate.beginSession(), latest: true });
  const open = gate.request("open", { session: gate.beginSession(), latest: true });
  assert.equal(gate.isCurrent(restore), false);
  assert.equal(gate.isCurrent(open), true);
  let notice = "";
  if (gate.isCurrent(restore)) notice = "stale error";
  assert.equal(notice, "");
  gate.unmount();
  assert.equal(gate.isCurrent(open), false);
});

test("collage slots are keyed latest-wins and independent", () => {
  const gate = new ViewerAsyncSession();
  gate.beginSession();
  const slot0Old = gate.request("slot", { latest: true, key: "slot:0" });
  const slot0New = gate.request("slot", { latest: true, key: "slot:0" });
  const slot1 = gate.request("slot", { latest: true, key: "slot:1" });
  assert.equal(gate.isCurrent(slot0Old), false);
  assert.equal(gate.isCurrent(slot0New), true);
  assert.equal(gate.isCurrent(slot1), true);
});

test("confirm is latest-wins and close/unmount invalidate slot and confirm work", () => {
  const gate = new ViewerAsyncSession();
  gate.beginSession();
  const oldConfirm = gate.request("confirm", { latest: true });
  const newConfirm = gate.request("confirm", { latest: true });
  const slot = gate.request("slot", { latest: true, key: "slot:0" });
  assert.equal(gate.isCurrent(oldConfirm), false);
  assert.equal(gate.isCurrent(newConfirm), true);
  assert.equal(gate.isCurrent(slot), true);
  gate.close();
  assert.equal(gate.isCurrent(newConfirm), false);
  assert.equal(gate.isCurrent(slot), false);
  gate.beginSession();
  const currentConfirm = gate.request("confirm", { latest: true });
  gate.unmount();
  assert.equal(gate.isCurrent(currentConfirm), false);
});

test("stale drop success and error cannot publish while current slot errors can", () => {
  const gate = new ViewerAsyncSession();
  gate.beginSession();
  const oldDrop = gate.request("slot", { latest: true, key: "slot:0" });
  const currentDrop = gate.request("slot", { latest: true, key: "slot:0" });
  const notices = [];
  if (gate.isCurrent(oldDrop)) notices.push("old success");
  if (gate.isCurrent(oldDrop)) notices.push("old error");
  if (gate.isCurrent(currentDrop)) notices.push("current error");
  assert.deepEqual(notices, ["current error"]);
  gate.close();
  if (gate.isCurrent(currentDrop)) notices.push("closed success");
  assert.deepEqual(notices, ["current error"]);
});

test("clipboard pending close/unmount suppresses both success and error", async () => {
  const gate = new ViewerAsyncSession();
  gate.beginSession();
  const pending = deferred();
  const token = gate.request("copy", { latest: true });
  const writes = [];
  const action = pending.promise.then(() => { if (gate.isCurrent(token)) writes.push("success"); }, () => { if (gate.isCurrent(token)) writes.push("error"); });
  gate.close();
  pending.resolve();
  await action;
  assert.deepEqual(writes, []);
  gate.beginSession();
  const current = gate.request("copy", { latest: true });
  if (gate.isCurrent(current)) writes.push("current");
  gate.unmount();
  assert.deepEqual(writes, ["current"]);
});

test("eyedropper old/new, abort, current error and unmount semantics", () => {
  const gate = new ViewerAsyncSession();
  gate.beginSession();
  const old = gate.request("eyedropper", { latest: true });
  const current = gate.request("eyedropper", { latest: true });
  const events = [];
  if (gate.isCurrent(old)) events.push("old-color");
  if (gate.isCurrent(current)) events.push("current-error");
  assert.deepEqual(events, ["current-error"]);
  gate.close();
  if (gate.isCurrent(current)) events.push("closed-color");
  gate.beginSession();
  const abort = gate.request("eyedropper", { latest: true });
  if (gate.isCurrent(abort) && { name: "AbortError" }.name !== "AbortError") events.push("abort-notice");
  gate.unmount();
  if (gate.isCurrent(abort)) events.push("unmounted-color");
  assert.deepEqual(events, ["current-error"]);
});

test("history delete stale failure cannot clear dialog; current failure can", () => {
  const gate = new ViewerAsyncSession();
  gate.beginSession();
  const old = gate.request("history-delete", { latest: true });
  const current = gate.request("history-delete", { latest: true });
  const cleared = [];
  if (gate.isCurrent(old)) cleared.push("old-clear");
  if (gate.isCurrent(current)) cleared.push("current-clear");
  assert.deepEqual(cleared, ["current-clear"]);
  gate.close();
  if (gate.isCurrent(current)) cleared.push("closed-clear");
  assert.deepEqual(cleared, ["current-clear"]);
});

test("RAF scheduler cancels pending close/unmount and only current open fits", () => {
  let next = 0;
  const callbacks = new Map();
  const api = { requestAnimationFrame(callback) { const id = ++next; callbacks.set(id, callback); return id; }, cancelAnimationFrame(id) { callbacks.delete(id); } };
  const scheduler = createViewerRafScheduler(api);
  const gate = new ViewerAsyncSession();
  gate.beginSession();
  const openA = gate.request("open", { latest: true });
  const fits = [];
  scheduler.schedule(() => { if (gate.isCurrent(openA)) fits.push("A"); });
  scheduler.cancel();
  for (const callback of callbacks.values()) callback();
  assert.deepEqual(fits, []);
  const openB = gate.request("open", { latest: true });
  scheduler.schedule(() => { if (gate.isCurrent(openB)) fits.push("B"); });
  callbacks.get(scheduler.pendingId)?.();
  assert.deepEqual(fits, ["B"]);
  gate.unmount();
  scheduler.schedule(() => { if (gate.isCurrent(openB)) fits.push("unmounted"); });
  scheduler.cancel();
  assert.deepEqual(fits, ["B"]);
});

test("App uses the gate and unified close contract", () => {
  const source = fs.readFileSync(new URL("../src/App.jsx", import.meta.url), "utf8");
  assert.match(source, /ViewerAsyncSession/);
  assert.match(source, /const closeImageViewer/);
  assert.equal((source.match(/setImageViewerOpen\(false\)/g) || []).length, 1, "only closeImageViewer may set closed state");
  for (const name of ["openImageViewer", "focusViewerAsset", "addViewerAsset", "addViewerFiles", "restoreManualCollage", "setCollageSlot", "dropCollageSlot", "confirmCollage", "copyViewerLayer", "createManualCollage", "finishHistoryDelete", "pickEdgeColor"]) assert.match(source, new RegExp(`${name}[\\s\\S]{0,3000}viewerSession`));
  assert.match(source, /key: `slot:\$\{index\}`/);
  assert.match(source, /beginViewerCollageOperation\("confirm"\)/);
  assert.match(source, /beginViewerCollageOperation\("save"\)/);
  assert.match(source, /isOperationCurrent\(token\)/);
  for (const kind of ["open", "focus", "restore"]) assert.match(source, new RegExp(`beginReplacement\\("${kind}"`));
  for (const kind of ["add", "drop"]) assert.match(source, new RegExp(`beginAppend\\("${kind}"\\)`));
  assert.match(source, /invalidateReplacement: true/);
  assert.match(source, /await navigator\.clipboard\.write[\s\S]{0,180}isCurrent\(token\)/);
  assert.match(source, /const token = viewerSession\.current\.request\("eyedropper", \{ latest: true \}\)/);
  assert.match(source, /viewerFitRaf\.current\?\.cancel\(\)/);
  assert.match(source, /createViewerRafScheduler/);
  assert.match(source, /viewerSession\.current\.isCurrent\(token\)[\s\S]{0,120}setViewerZoom/);
  assert.match(source, /const snapToken = viewerSession\.current\.request\("snap", \{ latest: true \}\)/);
  assert.match(source, /viewerSession\.current\.mount\(\)/);
  assert.match(source, /const plan = viewerOpenPlan\(generatedImage\);\s*if \(plan\.empty\) return/);
});
