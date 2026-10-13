import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { galleryTransferAdmission, isGalleryTransferRequest } from "./gallery-transfer-access.mjs";

const local = { method: "POST", headers: { host: "localhost:7709", origin: "http://localhost:7709" }, socket: { remoteAddress: "127.0.0.1" } };
const remote = { method: "POST", headers: { host: "10.0.0.39:7709", origin: "http://10.0.0.39:7709" }, socket: { remoteAddress: "10.0.0.8" } };
const at = (request, url, changes = {}) => ({ ...request, url, ...changes });

test("only the three transfer writes are gallery transfer requests", () => {
  for (const url of ["/api/inference/gallery/transfer/export", "/api/inference/gallery/transfer/import", "/api/inference/gallery/transfer/import-upload?folder=a"]) {
    assert.equal(isGalleryTransferRequest(at(local, url)), true, url);
  }
  for (const [method, url] of [["GET", "/api/inference/gallery/transfer/exports/XiriusA-Gallery-20261012-220000.zip"], ["POST", "/api/inference/gallery/cards"], ["POST", "/api/inference/gallery/transfer/import/extra"]]) {
    assert.equal(isGalleryTransferRequest(at(local, url, { method })), false, url);
    assert.equal(galleryTransferAdmission(at(local, url, { method })).allowed, true, url);
  }
});

test("export and upload need a same-Origin page; a remote same-Origin page may use them", () => {
  for (const url of ["/api/inference/gallery/transfer/export", "/api/inference/gallery/transfer/import-upload"]) {
    assert.equal(galleryTransferAdmission(at(local, url)).allowed, true);
    assert.equal(galleryTransferAdmission(at(remote, url)).allowed, true);
    for (const headers of [{ host: "localhost:7709" }, { host: "localhost:7709", origin: "http://evil.test:7709" }, { host: "localhost:7709", origin: "http://localhost:7709/path" }, { host: "localhost:7709", origin: "null" }]) {
      assert.deepEqual(galleryTransferAdmission(at(local, url, { headers })), { allowed: false, status: 403 }, JSON.stringify(headers));
    }
  }
});

test("an import by path is loopback-only, including a DNS-rebinding Host", () => {
  const url = "/api/inference/gallery/transfer/import";
  assert.equal(galleryTransferAdmission(at(local, url)).allowed, true);
  assert.deepEqual(galleryTransferAdmission(at(remote, url)), { allowed: false, status: 403, local: true });
  const rebound = at(local, url, { headers: { host: "evil.test:7709", origin: "http://evil.test:7709" } });
  assert.equal(galleryTransferAdmission(rebound).allowed, false);
});

test("Vite gates transfer writes and forwards them with the proxy proof", () => {
  const source = readFileSync(new URL("../vite.config.js", import.meta.url), "utf8");
  assert.match(source, /const galleryAdmission = galleryTransferAdmission\(request\);/);
  assert.equal(source.match(/isTaskDeletionRequest\(request\) \|\| isGalleryTransferRequest\(request\)/g)?.length, 2, "dev server and preview both add the proof");
});

test("the page sends only a quoted-or-plain absolute path that ends in the .zip itself", async () => {
  const { cleanZipPath, isZipFile, zipPathIssue } = await import("../src/gallery-transfer.js");
  assert.equal(cleanZipPath('  "D:\\Exports\\a b.zip"  '), "D:\\Exports\\a b.zip");
  for (const path of ["D:\\Exports\\a.zip", "\"D:\\Exports\\A.ZIP\"", "\\\\nas\\share\\a.zip", "/home/ubuntu/a.zip", ""]) assert.equal(zipPathIssue(path), "", path);
  for (const path of ["D:\\Exports", "D:\\Exports\\a.zip\\", "D:\\Exports\\a.zip.part", "Exports\\a.zip", "a.zip"]) assert.notEqual(zipPathIssue(path), "", path);
  assert.equal(isZipFile({ name: "gallery.ZIP" }), true);
  assert.equal(isZipFile({ name: "gallery.zip.png" }), false);
  assert.equal(isZipFile(null), false);
});
