import assert from "node:assert/strict";
import { randomBytes } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import net from "node:net";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { createServer } from "vite";

// The running package sometimes dropped back to its boot screen while idle or mid-generation.
// Nothing on the server asked for that: Vite's browser client reloads the page by itself whenever
// its live-update socket drops and a "vite-ping" handshake then succeeds. The service therefore
// offers no socket at all. These tests make the handshake the browser makes, against the Vite this
// project installs, so a Vite upgrade that changes what `ws: false` means is caught here.

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function firstReplyLine(port) {
  return new Promise((resolve) => {
    let reply = "";
    const done = () => { socket.destroy(); resolve(reply.split("\r\n")[0]); };
    const socket = net.connect(port, "127.0.0.1", () => socket.write([
      "GET / HTTP/1.1",
      `Host: 127.0.0.1:${port}`,
      "Upgrade: websocket",
      "Connection: Upgrade",
      `Sec-WebSocket-Key: ${randomBytes(16).toString("base64")}`,
      "Sec-WebSocket-Version: 13",
      "Sec-WebSocket-Protocol: vite-ping",
      "", "",
    ].join("\r\n")));
    socket.setTimeout(3000, done);
    socket.on("data", (chunk) => { reply += chunk.toString(); if (reply.includes("\r\n")) done(); });
    socket.on("close", done);
    socket.on("error", done);
  });
}

function readPageDirect(port) {
  return new Promise((resolve, reject) => {
    let reply = "";
    const socket = net.connect(port, "127.0.0.1", () => socket.write([
      "GET / HTTP/1.1",
      `Host: 127.0.0.1:${port}`,
      "Connection: close",
      "", "",
    ].join("\r\n")));
    socket.setEncoding("utf8");
    socket.setTimeout(3000, () => socket.destroy(new Error("Timed out reading the local Vite page")));
    socket.on("data", (chunk) => { reply += chunk; });
    socket.on("end", () => {
      const [head, body = ""] = reply.split("\r\n\r\n", 2);
      const status = Number(/^HTTP\/\d(?:\.\d)? (\d{3})/.exec(head)?.[1]);
      resolve({ status, body });
    });
    socket.on("error", reject);
  });
}

async function withServer(server, check) {
  // Vite refuses Windows 8.3 paths containing `~`; hosted runners can expose os.tmpdir() that way.
  const root = await mkdtemp(path.join(projectRoot, "node_modules", ".xirai-no-reload-"));
  await writeFile(path.join(root, "index.html"), "<!doctype html><title>probe</title><p>page</p>");
  // Host filtering is a separate Vite feature and varies with the hosted runner's network setup.
  // This loopback fixture isolates the HMR contract it exists to test.
  const vite = await createServer({ configFile: false, root, logLevel: "silent", server: { host: "127.0.0.1", allowedHosts: true, port: 0, ...server } });
  try {
    await vite.listen();
    const { port } = vite.httpServer.address();
    await check(port);
  } finally {
    await vite.close();
    await rm(root, { recursive: true, force: true });
  }
}

test("a default Vite server accepts the handshake the reload waits for", async () => {
  // The control: without it, a probe that could never succeed would pass the next test vacuously.
  await withServer({}, async (port) => {
    assert.match(await firstReplyLine(port), / 101 /);
  });
});

test("the service settings refuse that handshake and still serve the page", async () => {
  await withServer({ hmr: false, ws: false }, async (port) => {
    assert.doesNotMatch(await firstReplyLine(port), / 101 /, "the socket is still offered, so a dropped connection still reloads the page");
    const page = await readPageDirect(port);
    assert.equal(page.status, 200);
    assert.match(page.body, /<p>page<\/p>/);
  });
});

test("the supervised service runs with those settings and a developer's plain vite does not", async () => {
  const config = await readFile(path.join(projectRoot, "vite.config.js"), "utf8");
  const start = await readFile(path.join(projectRoot, "scripts", "start.mjs"), "utf8");
  assert.match(config, /const supervisedService = process\.env\.XIRAI_SERVICE_SUPERVISOR === "1";/);
  const devServer = config.slice(config.indexOf("\n  server: {"), config.indexOf("\n  preview: {"));
  assert.match(devServer, /\.\.\.\(supervisedService \? \{ hmr: false, ws: false \} : \{\}\)/);
  assert.match(start, /env: \{ \.\.\.process\.env, XIRAI_SERVICE_SUPERVISOR: "1" \}/);
});
