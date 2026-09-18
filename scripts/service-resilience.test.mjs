import assert from "node:assert/strict";
import http from "node:http";
import net from "node:net";
import { once } from "node:events";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  SERVICE_RESTART_LIMIT,
  SERVICE_RESTART_WINDOW_MS,
  serviceRestartDecision,
  serviceRestartMessage,
} from "./service-supervisor.js";

const read = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

test("a crash brings the service back; a stop request does not", () => {
  // The exit that started this: the server threw on an unhandled socket error and left with 1.
  const crash = serviceRestartDecision({ code: 1, signal: null });
  assert.equal(crash.restart, true);
  assert.equal(crash.reason, "crashed");
  assert.equal(crash.attempt, 1);
  // Killed by a fault rather than an exit code -- still a crash.
  assert.equal(serviceRestartDecision({ code: null, signal: "SIGSEGV" }).restart, true);

  // Ctrl-C, a deploy script, or a service manager asking it to stop. Coming back up would be
  // disobeying an instruction, not surviving a fault.
  for (const signal of ["SIGINT", "SIGTERM", "SIGQUIT"]) {
    const decision = serviceRestartDecision({ code: null, signal });
    assert.equal(decision.restart, false, signal);
    assert.equal(decision.reason, "signalled");
  }
  assert.equal(serviceRestartDecision({ code: 0, signal: null }).restart, false);
  assert.equal(serviceRestartDecision({ code: 1, signal: null, shuttingDown: true }).restart, false);
});

test("a crash loop stops instead of scrolling the error away", () => {
  const now = 1_000_000;
  let history = [];
  for (let attempt = 1; attempt <= SERVICE_RESTART_LIMIT; attempt += 1) {
    const decision = serviceRestartDecision({ code: 1, history, now: now + attempt });
    assert.equal(decision.restart, true, `attempt ${attempt}`);
    assert.equal(decision.attempt, attempt);
    history = decision.history;
  }
  // The next one inside the window gives up, so the error that says what is actually wrong stays
  // on screen instead of being buried under endless restarts.
  const exhausted = serviceRestartDecision({ code: 1, history, now: now + SERVICE_RESTART_LIMIT + 1 });
  assert.equal(exhausted.restart, false);
  assert.equal(exhausted.reason, "crash-loop");
  assert.match(serviceRestartMessage(exhausted, { code: 1, signal: null }), /已停止自动重启/);

  // A service that ran fine for longer than the window and then fell over is a fresh incident,
  // not the next strike against an old one.
  const later = serviceRestartDecision({ code: 1, history, now: now + SERVICE_RESTART_WINDOW_MS + 5_000 });
  assert.equal(later.restart, true);
  assert.equal(later.attempt, 1);
  assert.match(serviceRestartMessage(later, { code: 1, signal: null }), /正在自动重启/);
});

test("a client that resets its connection cannot stop the server", async () => {
  // The mechanism the plugin installs, exercised for real over loopback. Without the listener this
  // socket's `'error'` has nowhere to go and Node throws out of the event loop, which is what took
  // the whole service down; with it, the reset is ordinary traffic and the next request is served.
  const server = http.createServer((request, response) => {
    response.writeHead(200, { "Content-Type": "text/plain" });
    response.end("ok");
  });
  server.on("connection", (socket) => { socket.on("error", () => {}); });
  server.on("clientError", (error, socket) => {
    if (socket.writable) socket.end("HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n");
    else socket.destroy();
  });

  let fatal = null;
  const onFatal = (error) => { fatal = error; };
  process.once("uncaughtException", onFatal);
  try {
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    const { port } = server.address();

    // Open a connection, send nothing, and reset it — the idle keep-alive case that has neither a
    // request nor a response to carry the error.
    const idle = net.connect({ port, host: "127.0.0.1" });
    await once(idle, "connect");
    idle.resetAndDestroy();

    // And one that resets midway through a request line, which lands on `clientError` instead.
    const partial = net.connect({ port, host: "127.0.0.1" });
    await once(partial, "connect");
    partial.write("GET / HTTP/1.1\r\nHost: 127.0.0.1\r\n");
    partial.resetAndDestroy();

    await new Promise((resolve) => setTimeout(resolve, 60));
    assert.equal(fatal, null, `a reset client must not raise: ${fatal?.message}`);

    // The server is still there and still answering, which is the whole point.
    const answer = await new Promise((resolve, reject) => {
      const request = http.get({ port, host: "127.0.0.1", path: "/" }, (response) => {
        let body = "";
        response.setEncoding("utf8");
        response.on("data", (chunk) => { body += chunk; });
        response.on("end", () => resolve({ status: response.statusCode, body }));
      });
      request.on("error", reject);
    });
    assert.equal(answer.status, 200);
    assert.equal(answer.body, "ok");
  } finally {
    process.off("uncaughtException", onFatal);
    server.close();
    await once(server, "close");
  }
});

test("the guard is installed on the real server, ahead of everything that serves a request", async () => {
  const config = await read("vite.config.js");
  assert.match(config, /function connectionResiliencePlugin\(\)/);
  assert.match(config, /httpServer\.on\("connection", \(socket\) => \{ socket\.on\("error", \(\) => \{\}\); \}\);/);
  assert.match(config, /httpServer\.on\("clientError", \(error, socket\) => \{/);
  assert.match(config, /name: "connection-resilience",\s*configureServer: guard,\s*configurePreviewServer: guard,/,
    "the preview server is the same service and needs the same guard");
  // First in the list: a connection is accepted before any middleware sees it, so a plugin that
  // installs the listener later than the one handling the request would still be racing it.
  assert.match(config, /plugins: \[connectionResiliencePlugin\(\), setupGatePlugin\(\)/);
});

test("the supervisor asks the policy instead of treating every exit as final", async () => {
  const start = await read("scripts/start.mjs");
  assert.match(start, /import \{ serviceRestartDecision, serviceRestartMessage \} from "\.\/service-supervisor\.js";/);
  assert.match(start, /const decision = serviceRestartDecision\(\{ code: result\.code, signal: result\.signal, shuttingDown, history: restartHistory \}\);/);
  assert.match(start, /if \(decision\.restart\) \{\s*openOnReady = false;\s*continue;\s*\}/);
  // The three instruction codes still mean what they meant; the restart policy only sees what is
  // left over after them.
  for (const code of [75, 76, 77]) assert.ok(start.includes(`result.code === ${code}`), `exit code ${code} keeps its meaning`);
  const loop = start.slice(start.indexOf("let service = \"main\";"));
  assert.ok(loop.indexOf("serviceRestartDecision") > loop.indexOf("result.code === 77"),
    "the instruction codes are read before the crash policy");
  // Re-opening the browser on every restart would pile up tabs on the machine nobody is sitting at.
  assert.doesNotMatch(loop, /decision\.restart\) \{\s*openOnReady = true/);
});
