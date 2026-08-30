import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { createServer } from "node:http";
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

/** Drives the update middleware the way the dev server does. */
async function updateMiddleware() {
  const { updateApiPlugin } = await import(`../vite.config.js?update-api-test=${Date.now()}-${Math.random()}`);
  let middleware;
  updateApiPlugin().configureServer({ middlewares: { use: (handler) => { middleware = handler; } } });
  return async (pathname, method = "GET") => {
    const request = Readable.from([]);
    Object.assign(request, {
      url: pathname,
      method,
      headers: { host: "127.0.0.1:7709" },
      socket: { remoteAddress: "127.0.0.1" },
    });
    let body = "";
    const response = { statusCode: 200, setHeader() {}, end(value = "") { body += value; } };
    await middleware(request, response, () => assert.fail("update middleware must handle the request"));
    return { body: body ? JSON.parse(body) : {}, statusCode: response.statusCode };
  };
}

function releasePayload(version, assetName = `XirAI-${version}.7z`) {
  return {
    tag_name: `v${version}`,
    published_at: "2026-09-01T10:00:00Z",
    body: "示例更新说明",
    assets: [{
      name: assetName,
      browser_download_url: `https://github.com/o/r/releases/download/v${version}/${assetName}`,
      size: 1024,
      digest: `sha256:${"a".repeat(64)}`,
    }],
  };
}

async function withUpdateEnvironment(run) {
  const stateDirectory = await mkdtemp(path.join(os.tmpdir(), "xirai-update-online-"));
  const previous = {
    XIRAI_STATE_DIR: process.env.XIRAI_STATE_DIR,
    XIRAI_UPDATE_FEED: process.env.XIRAI_UPDATE_FEED,
  };
  process.env.XIRAI_STATE_DIR = stateDirectory;
  try {
    await run();
  } finally {
    for (const [name, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[name];
      else process.env[name] = value;
    }
    await rm(stateDirectory, { recursive: true, force: true });
  }
}

test("the update state reports the running version so the client compares against the build", async () => {
  await withUpdateEnvironment(async () => {
    const invoke = await updateMiddleware();
    const { body, statusCode } = await invoke("/api/system/update");
    assert.equal(statusCode, 200);
    assert.equal(body.current_version, JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8")).version);
    assert.equal(body.online_release, null);
    assert.equal(body.online_checked_at, null);
  });
});

test("a newer release is reported as available and an equal one is not", async (context) => {
  const current = JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8")).version;
  const [major] = current.split(".");
  let served = releasePayload(`${Number(major) + 1}.0.0`);
  const feed = createServer((_request, response) => {
    response.setHeader("Content-Type", "application/json");
    response.end(JSON.stringify(served));
  });
  await new Promise((resolve) => feed.listen(0, "127.0.0.1", resolve));
  context.after(() => new Promise((resolve) => feed.close(resolve)));

  await withUpdateEnvironment(async () => {
    process.env.XIRAI_UPDATE_FEED = `http://127.0.0.1:${feed.address().port}/latest`;
    const invoke = await updateMiddleware();

    const newer = await invoke("/api/system/update/check", "POST");
    assert.equal(newer.statusCode, 200);
    assert.equal(newer.body.online_release.version, `${Number(major) + 1}.0.0`);
    assert.equal(newer.body.online_release.update_available, true);
    assert.equal(newer.body.online_release.verified, true);
    assert.ok(newer.body.online_checked_at);

    // The same version the build already runs is not an update, so the client is never prompted.
    served = releasePayload(current);
    const same = await invoke("/api/system/update/check", "POST");
    assert.equal(same.body.online_release.update_available, false);
  });
});

test("a download is refused until a check has named a newer release", async (context) => {
  const current = JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8")).version;
  let served = releasePayload(current);
  const feed = createServer((_request, response) => {
    response.setHeader("Content-Type", "application/json");
    response.end(JSON.stringify(served));
  });
  await new Promise((resolve) => feed.listen(0, "127.0.0.1", resolve));
  context.after(() => new Promise((resolve) => feed.close(resolve)));

  await withUpdateEnvironment(async () => {
    process.env.XIRAI_UPDATE_FEED = `http://127.0.0.1:${feed.address().port}/latest`;
    const invoke = await updateMiddleware();

    // Nothing has been checked, so there is no release the server would agree to fetch.
    const unchecked = await invoke("/api/system/update/download", "POST");
    assert.equal(unchecked.statusCode, 409);
    assert.match(unchecked.body.error, /请先检查更新/);

    // Checking finds only the running version, which is still not something to download.
    await invoke("/api/system/update/check", "POST");
    const current_ = await invoke("/api/system/update/download", "POST");
    assert.equal(current_.statusCode, 409);
    assert.match(current_.body.error, /已是最新版本/);
  });
});

test("an unreachable or unusable feed is reported rather than silently offering nothing", async (context) => {
  const feed = createServer((_request, response) => {
    response.statusCode = 404;
    response.end("{}");
  });
  await new Promise((resolve) => feed.listen(0, "127.0.0.1", resolve));
  context.after(() => new Promise((resolve) => feed.close(resolve)));

  await withUpdateEnvironment(async () => {
    process.env.XIRAI_UPDATE_FEED = `http://127.0.0.1:${feed.address().port}/latest`;
    const invoke = await updateMiddleware();
    const missing = await invoke("/api/system/update/check", "POST");
    assert.equal(missing.statusCode, 502);
    assert.match(missing.body.error, /未找到发布源/);
  });
});
