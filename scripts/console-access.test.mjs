import assert from "node:assert/strict";
import test from "node:test";
import { inferenceBackendPlugin } from "../vite.config.js";

// Reading the console log and running a shell command in it are separate permissions. Holding both
// to the host left the drawer blank on every other device — a phone on the LAN could not see the
// inference log the drawer exists to show, and the reason it gave was about commands.

function consoleHandler() {
  const handlers = [];
  const plugin = inferenceBackendPlugin({ testOnly: true });
  plugin.configureServer({ middlewares: { use(handler) { handlers.push(handler); } } });
  return handlers[0];
}

function fakeRequest({ url = "/api/console", method = "GET", address = "127.0.0.1", headers = {} } = {}) {
  return { url, method, headers, socket: { remoteAddress: address } };
}

function fakeResponse() {
  const state = { statusCode: 0, body: "", headers: {} };
  return {
    state,
    setHeader(name, value) { state.headers[name] = value; },
    end(body) { state.body = body || ""; state.ended = true; },
    set statusCode(value) { state.statusCode = value; },
    get statusCode() { return state.statusCode; },
  };
}

async function call(request) {
  const handler = consoleHandler();
  const response = fakeResponse();
  await handler(request, response, () => { response.state.passedThrough = true; });
  return response.state;
}

test("the console log is readable from another device, because it is the log the app already streams", async () => {
  const state = await call(fakeRequest({ address: "192.168.1.40" }));
  assert.equal(state.statusCode, 200);
  const payload = JSON.parse(state.body);
  assert.ok(Array.isArray(payload.entries));
  assert.equal(payload.commands_allowed, false);
});

test("the same read from the host reports that commands are available", async () => {
  const state = await call(fakeRequest({ address: "127.0.0.1" }));
  assert.equal(state.statusCode, 200);
  assert.equal(JSON.parse(state.body).commands_allowed, true);
});

test("every loopback spelling counts as the host", async () => {
  for (const address of ["127.0.0.1", "::1", "::ffff:127.0.0.1"]) {
    const state = await call(fakeRequest({ address }));
    assert.equal(JSON.parse(state.body).commands_allowed, true, address);
  }
});

test("running a command from another device is refused, and the message says how to allow it", async () => {
  const state = await call(fakeRequest({ url: "/api/console/commands", method: "POST", address: "10.0.0.5" }));
  assert.equal(state.statusCode, 403);
  const message = JSON.parse(state.body).error;
  assert.match(message, /only from this computer/);
  // A refusal that does not name the setting is a dead end for the person who owns the machine.
  assert.match(message, /XIRAI_REMOTE_CONSOLE=1/);
});

test("the host may still run a command", async () => {
  const state = await call(fakeRequest({ url: "/api/console/commands", method: "GET", address: "127.0.0.1" }));
  // Past the address gate, so the method check is what answers — not a 403.
  assert.equal(state.statusCode, 405);
});

test("the opt-in is read from the environment of the machine running the app", async () => {
  const previous = process.env.XIRAI_REMOTE_CONSOLE;
  process.env.XIRAI_REMOTE_CONSOLE = "1";
  try {
    const read = await call(fakeRequest({ address: "10.0.0.5" }));
    assert.equal(JSON.parse(read.body).commands_allowed, true);
    const run = await call(fakeRequest({ url: "/api/console/commands", method: "GET", address: "10.0.0.5" }));
    assert.equal(run.statusCode, 405, "the address gate should no longer be what refuses");
  } finally {
    if (previous === undefined) delete process.env.XIRAI_REMOTE_CONSOLE;
    else process.env.XIRAI_REMOTE_CONSOLE = previous;
  }
});

test("only an exact 1 opts in, so a stray value cannot open a shell", async () => {
  const previous = process.env.XIRAI_REMOTE_CONSOLE;
  for (const value of ["0", "", "true", "yes"]) {
    process.env.XIRAI_REMOTE_CONSOLE = value;
    const state = await call(fakeRequest({ url: "/api/console/commands", method: "POST", address: "10.0.0.5" }));
    assert.equal(state.statusCode, 403, `XIRAI_REMOTE_CONSOLE=${value || "(empty)"}`);
  }
  if (previous === undefined) delete process.env.XIRAI_REMOTE_CONSOLE;
  else process.env.XIRAI_REMOTE_CONSOLE = previous;
});

test("a cross-origin read is refused wherever it comes from", async () => {
  const state = await call(fakeRequest({
    address: "127.0.0.1",
    headers: { origin: "http://evil.example", host: "127.0.0.1:7709" },
  }));
  assert.equal(state.statusCode, 403);
});
