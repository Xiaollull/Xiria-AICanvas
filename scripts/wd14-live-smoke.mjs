// Explicit, opt-in isolated-port smoke. Starts the current source through the
// normal Vite supervisor, never touches the running 7709/8718 service.
import { mkdtemp, mkdir, rm } from "node:fs/promises";
import { createServer as netServer } from "node:net";
import { networkInterfaces } from "node:os";
import path from "node:path";
import { getSetupMarkerPath, readSetupMarker } from "./setup-state.mjs";

const parent = process.env.WD14_SMOKE_TEMP;
if (!parent) throw new Error("Set WD14_SMOKE_TEMP to the approved temporary parent directory");
const projectRoot = path.resolve(".");
if (!readSetupMarker(projectRoot)) throw new Error("The existing project setup marker is not valid; cannot start normal Vite service");
// Reuse the real, validated marker read-only when the test isolates its cache.
// Never create a forged marker or bypass the setup gate.
process.env.XIRAI_SETUP_MARKER = getSetupMarkerPath(projectRoot);
const scratch = await mkdtemp(path.join(parent, "wd14-live-"));
async function freePort() {
  const server = netServer();
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = server.address().port;
  await new Promise((resolve) => server.close(resolve));
  return port;
}
const webPort = await freePort();
const inferPort = await freePort();
process.env.WEB_HOST = "0.0.0.0";
process.env.WEB_PORT = String(webPort);
process.env.INFERENCE_HOST = "127.0.0.1";
process.env.INFERENCE_PORT = String(inferPort);
process.env.XIRAI_STATE_DIR = path.join(scratch, "state");
process.env.XIRAI_OUTPUT_DIR = path.join(scratch, "outputs");
process.env.XIRAI_CACHE_DIR = path.join(scratch, "cache");
await mkdir(path.join(scratch, "outputs"));
const input = path.join(scratch, "outside-outputs");
await mkdir(input);
// Zero-byte file suffices to exercise path admission; no real model/download.
await (await import("node:fs/promises")).writeFile(path.join(input, "picture.png"), "not-an-image");
let server;
try {
  const { createServer } = await import("vite");
  server = await createServer({ configFile: path.resolve("vite.config.js"), server: {
    host: "0.0.0.0", port: webPort, strictPort: true,
    watch: { ignored: ["**/state-cache/**", "**/.cache/**", "**/outputs/**", "**/node_modules/**"] },
  } });
  await server.listen();
  const origin = `http://127.0.0.1:${webPort}`;
  const inference = `http://127.0.0.1:${inferPort}`;
  let ready = false;
  let lastHealth = {};
  for (let n = 0; n < 150; n++) {
    try {
      const health = await (await fetch(`${origin}/api/inference/health`)).json();
      lastHealth = { status: health.status, phase: health.phase, error: health.error, protocol: health.protocol };
      if (health.status === "ready" && health.protocol === 39) { ready = true; break; }
    } catch { /* still importing */ }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  if (!ready) throw new Error(`isolated inference service failed to become ready: ${JSON.stringify(lastHealth)}`);
  async function check(url, options, expected, label) {
    const response = await fetch(url, options);
    console.log(`${label} ${response.status}`);
    if (response.status !== expected) throw new Error(`${label}: expected ${expected}, got ${response.status}`);
    return response.json();
  }
  const catalog = await check(`${origin}/api/inference/wd14/models`, {}, 200, "new-source models");
   if (catalog.models.length !== 19 || catalog.engines?.length !== 4 ||
       catalog.models.filter((model) => model.engine === "wd").length !== 16) throw new Error("four-engine registry mismatch");
  const body = JSON.stringify({ folder: input, recursive: false });
  const json = { "Content-Type": "application/json", Origin: origin };
  const admitted = await check(`${origin}/api/inference/wd14/batch`, { method: "POST", headers: json, body }, 202, "local outside-outputs batch");
  await check(`${origin}/api/inference/wd14/batch`, { method: "POST", headers: { ...json, Origin: "http://evil.example" }, body }, 403, "cross-origin batch");
  await check(`${origin}/api/inference/wd14/download/wd-swinv2-v3`, { method: "POST", headers: { Origin: "http://evil.example" } }, 403, "cross-origin download");
  await check(`${inference}/api/inference/wd14/batch`, { method: "POST", headers: { ...json, Origin: "http://evil.example", "x-xirai-wd14-proxy-proof": "forged" }, body }, 403, "direct Python forged Origin/proof");
  const remoteAddress = Object.values(networkInterfaces()).flat().find((entry) => entry.family === "IPv4" && !entry.internal)?.address;
  if (remoteAddress) {
    const remote = `http://${remoteAddress}:${webPort}`;
    await check(`${remote}/api/inference/wd14/batch`, { method: "POST", headers: { Origin: remote, "Content-Type": "application/json", "x-xirai-wd14-proxy-proof": "forged" }, body }, 403, "LAN batch/proof");
    await check(`${remote}/api/inference/wd14/download/wd-swinv2-v3`, { method: "POST", headers: { Origin: remote } }, 403, "LAN download");
  } else console.log("LAN: no non-loopback IPv4 interface; skipped");
  let job;
  for (let n = 0; n < 30; n++) {
    job = await (await fetch(`${origin}/api/inference/wd14/jobs/${admitted.id}`)).json();
    if (!["queued", "running"].includes(job.state)) break;
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
   console.log(`outside-outputs job ${job.state} (intentionally invalid image; no caption should be written)`);
   if ((job.done || 0) !== 0 || !["done", "error"].includes(job.state)) throw new Error("invalid-image batch unexpectedly wrote or remained active");
  if (process.env.WD14_BROWSER_SMOKE === "1") {
    const { browserSmoke } = await import("./wd14-browser-smoke.mjs");
    await browserSmoke({ origin, scratch, evidenceDirectory: parent });
  }
} finally {
  if (server) await server.close();
  await rm(scratch, { recursive: true, force: true });
}
