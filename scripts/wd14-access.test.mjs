import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { wd14Admission } from "./wd14-access.mjs";

function req(path, { peer = "127.0.0.1", host = "127.0.0.1:7709", origin = `http://${host}`, method = "POST", proof = "forged" } = {}) {
  return { url: path, method, socket: { remoteAddress: peer }, headers: { host, origin, "x-xirai-wd14-proxy-proof": proof } };
}
const base = "/api/inference/wd14";

test("local same-origin can batch/install, remote LAN cannot even with forged proof", () => {
  for (const path of [`${base}/batch`, `${base}/download/wd-swinv2-v3`, `${base}/jobs/any/cancel`]) {
    assert.equal(wd14Admission(req(path)).allowed, true);
    assert.equal(wd14Admission(req(path, { peer: "10.0.0.42", host: "10.0.0.1:7709" })).status, 403);
    assert.equal(wd14Admission(req(path, { host: "evil.example:7709" })).status, 403);
    assert.equal(wd14Admission(req(path, { origin: "https://other.example" })).status, 403);
    const missing = req(path);
    delete missing.headers.origin;
    assert.equal(wd14Admission(missing).status, 403);
    assert.equal(wd14Admission(req(path, { origin: "null" })).status, 403);
  }
  assert.equal(wd14Admission(req(`${base}/batch`, { host: "localhost:7709" })).allowed, true);
  assert.equal(wd14Admission(req(`${base}/tag`, { peer: "10.0.0.42", host: "10.0.0.1:7709" })).allowed, true);
  assert.equal(wd14Admission(req(`${base}/tag`, { origin: "https://other.example" })).status, 403);
  assert.equal(wd14Admission(req("/api/inference/%77d14/download/wd-swinv2-v3", { peer: "10.0.0.42" })).status, 403);
});

test("the proxy strips client proof and mints its own; guards run before the identity gate", () => {
  const vite = readFileSync(new URL("../vite.config.js", import.meta.url), "utf8");
  const guard = vite.indexOf("const admission = wd14Admission(request);");
  const identity = vite.indexOf("const health = await probeBackendForGate();", guard);
  assert.ok(guard > 0 && identity > guard);
  assert.equal((vite.match(/forward\.removeHeader\("x-xirai-wd14-proxy-proof"\)/g) || []).length, 2);
  assert.equal((vite.match(/forward\.setHeader\("x-xirai-wd14-proxy-proof", wd14ProxyProof\)/g) || []).length, 2);
  assert.match(vite, /XIRAI_WD14_PROXY_PROOF: wd14ProxyProof/);
});

test("force catalog is a local same-origin action; aliases and duplicate query cannot bypass", () => {
  for (const value of ['true','1','on','yes','TRUE','false','0']) {
    const path=`${base}/models?refresh=${value}`;
    assert.equal(wd14Admission(req(path,{method:'GET'})).allowed,true);
    assert.equal(wd14Admission(req(path,{method:'GET',peer:'10.0.0.42',host:'10.0.0.1:7709'})).status,403);
    assert.equal(wd14Admission(req(path,{method:'GET',origin:'http://evil.example'})).status,403);
    const missing=req(path,{method:'GET'});delete missing.headers.origin;assert.equal(wd14Admission(missing).status,403);
  }
  assert.equal(wd14Admission(req(`${base}/models?%72efresh=true`,{method:'GET',peer:'10.0.0.42'})).status,403);
  assert.equal(wd14Admission(req(`${base}/models?refresh=false&refresh=true`,{method:'GET'})).status,400);
  assert.equal(wd14Admission(req(`${base}/models?refresh=true&%72efresh=false`,{method:'GET'})).status,400);
  assert.equal(wd14Admission(req(`${base}/models`,{method:'GET',peer:'10.0.0.42',host:'10.0.0.1:7709',origin:'http://evil.example'})).allowed,true);
  assert.equal(wd14Admission(req(`${base}/models?refresh=true`,{method:'POST'})).allowed,true);
  assert.equal(wd14Admission(req(`${base}/models?refresh=true`,{method:'POST',peer:'10.0.0.42',host:'10.0.0.1:7709'})).status,403);
});
