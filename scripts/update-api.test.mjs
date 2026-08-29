import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import test from "node:test";

test("offline update upload releases its process lock after an empty archive", async () => {
  const stateDirectory = await mkdtemp(path.join(os.tmpdir(), "xirai-update-api-state-"));
  const previousStateDirectory = process.env.XIRAI_STATE_DIR;
  process.env.XIRAI_STATE_DIR = stateDirectory;
  try {
    const { updateApiPlugin } = await import(`../vite.config.js?update-api-test=${Date.now()}`);
    let middleware;
    const server = {
      middlewares: { use: (handler) => { middleware = handler; } },
    };
    updateApiPlugin().configureServer(server);

    const invoke = async () => {
      const request = Readable.from([]);
      Object.assign(request, {
        url: "/api/system/update/archive",
        method: "POST",
        headers: {
          host: "127.0.0.1:7709",
          "content-length": "0",
          "x-archive-name": "update.zip",
        },
        socket: { remoteAddress: "127.0.0.1" },
      });
      let body = "";
      const response = {
        statusCode: 200,
        setHeader() {},
        end(value = "") { body += value; },
      };
      await middleware(request, response, () => assert.fail("update middleware must handle the request"));
      return { body: JSON.parse(body), statusCode: response.statusCode };
    };

    const first = await invoke();
    const second = await invoke();
    assert.equal(first.statusCode, 400);
    assert.equal(second.statusCode, 400);
    assert.match(first.body.error, /更新包为空/);
    assert.match(second.body.error, /更新包为空/);
  } finally {
    if (previousStateDirectory === undefined) delete process.env.XIRAI_STATE_DIR;
    else process.env.XIRAI_STATE_DIR = previousStateDirectory;
    await rm(stateDirectory, { recursive: true, force: true });
  }
});
