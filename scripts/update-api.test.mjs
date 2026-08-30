import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { Readable } from "node:stream";
import test from "node:test";

const sha = "a".repeat(64);

test("offline update upload releases its process lock after an empty archive", async () => {
  const stateDirectory = await mkdtemp(path.join(os.tmpdir(), "xirai-update-api-state-"));
  const previousStateDirectory = process.env.XIRAI_STATE_DIR;
  process.env.XIRAI_STATE_DIR = stateDirectory;
  try {
    const { updateApiPlugin } = await import(`../vite.config.js?update-api-test=${Date.now()}`);
    let middleware;
    updateApiPlugin().configureServer({ middlewares: { use: (handler) => { middleware = handler; } } });

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
      const response = { statusCode: 200, setHeader() {}, end(value = "") { body += value; } };
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

function releasePayload(version, overrides = {}) {
  const assetName = `XirAI-${version}.7z`;
  return {
    tag_name: `v${version}`,
    draft: false,
    prerelease: false,
    published_at: "2026-09-01T10:00:00Z",
    body: "示例更新说明",
    assets: [
      {
        name: assetName,
        state: "uploaded",
        browser_download_url: `https://github.com/o/r/releases/download/v${version}/${assetName}`,
        url: `https://api.github.com/repos/o/r/releases/assets/archive-${version}`,
        size: 1024,
        digest: `sha256:${sha}`,
      },
      {
        name: `${assetName}.sha256`,
        state: "uploaded",
        browser_download_url: `https://github.com/o/r/releases/download/v${version}/${assetName}.sha256`,
        url: `https://api.github.com/repos/o/r/releases/assets/checksum-${version}`,
        size: 89,
        digest: null,
      },
    ],
    ...overrides,
  };
}

function response(body = "", { status = 200, headers = {} } = {}) {
  return new Response(body, { status, headers });
}

function hangingResponse({ status = 200 } = {}) {
  return new Response(new ReadableStream({ start() {} }), { status });
}

function requestAuthorization(options) {
  return new Headers(options?.headers || {}).get("authorization");
}

function feedFetcher({
  getPayload,
  feedUrl = "https://updates.example/latest.json",
  sidecarBody,
  sidecarStatus = 200,
  requests = [],
} = {}) {
  return async (input, options = {}) => {
    const url = String(input);
    requests.push({ url, authorization: requestAuthorization(options), headers: new Headers(options.headers || {}) });
    if (url === feedUrl) return response(JSON.stringify(getPayload()));
    if (url.includes("checksum-") || url.endsWith(".sha256")) {
      const payload = getPayload();
      const version = String(payload.tag_name || "").replace(/^v/, "");
      return response(sidecarBody ?? `${sha}  XirAI-${version}.7z\n`, { status: sidecarStatus });
    }
    throw new Error(`unexpected fake request: ${url}`);
  };
}

async function withUpdateMiddleware({ environment, fetcher, ...pluginOptions }, run) {
  const stateDirectory = await mkdtemp(path.join(os.tmpdir(), "xirai-update-online-"));
  const previousStateDirectory = process.env.XIRAI_STATE_DIR;
  process.env.XIRAI_STATE_DIR = stateDirectory;
  try {
    const { updateApiPlugin } = await import(`../vite.config.js?update-api-test=${Date.now()}-${Math.random()}`);
    let middleware;
    updateApiPlugin({ environment, fetcher, ...pluginOptions }).configureServer({
      middlewares: { use: (handler) => { middleware = handler; } },
    });
    const invoke = async (pathname, method = "GET") => {
      const request = Readable.from([]);
      Object.assign(request, {
        url: pathname,
        method,
        headers: { host: "127.0.0.1:7709" },
        socket: { remoteAddress: "127.0.0.1" },
      });
      let body = "";
      const apiResponse = { statusCode: 200, setHeader() {}, end(value = "") { body += value; } };
      await middleware(request, apiResponse, () => assert.fail("update middleware must handle the request"));
      return { body: body ? JSON.parse(body) : {}, statusCode: apiResponse.statusCode };
    };
    await run(invoke);
  } finally {
    if (previousStateDirectory === undefined) delete process.env.XIRAI_STATE_DIR;
    else process.env.XIRAI_STATE_DIR = previousStateDirectory;
    await rm(stateDirectory, { recursive: true, force: true });
  }
}

test("the update state reports the running version before any release has been trusted", async () => {
  await withUpdateMiddleware({ environment: {}, fetcher: async () => assert.fail("network must not be used") }, async (invoke) => {
    const { body, statusCode } = await invoke("/api/system/update");
    assert.equal(statusCode, 200);
    assert.equal(body.current_version, JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8")).version);
    assert.equal(body.online_release, null);
    assert.equal(body.online_checked_at, null);
  });
});

test("a release is exposed only after its required sidecar has resolved the checksum", async () => {
  const current = JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8")).version;
  let served = releasePayload("2.0.0");
  const environment = { XIRAI_UPDATE_FEED: "https://updates.example/latest.json" };
  const fetcher = feedFetcher({ getPayload: () => served });
  await withUpdateMiddleware({ environment, fetcher }, async (invoke) => {
    const newer = await invoke("/api/system/update/check", "POST");
    assert.equal(newer.statusCode, 200);
    assert.equal(newer.body.online_release.version, "2.0.0");
    assert.equal(newer.body.online_release.update_available, true);
    assert.equal(newer.body.online_release.verified, true);
    assert.ok(newer.body.online_checked_at);

    served = releasePayload(current);
    const same = await invoke("/api/system/update/check", "POST");
    assert.equal(same.statusCode, 200);
    assert.equal(same.body.online_release.update_available, false);
    assert.equal(same.body.online_release.verified, true);
  });
});

test("a strict sidecar resolves verification when GitHub advertises no asset digest", async () => {
  const payload = releasePayload("2.0.0");
  payload.assets[0].digest = null;
  await withUpdateMiddleware({
    environment: { XIRAI_UPDATE_FEED: "https://updates.example/latest.json" },
    fetcher: feedFetcher({ getPayload: () => payload }),
  }, async (invoke) => {
    const checked = await invoke("/api/system/update/check", "POST");
    assert.equal(checked.statusCode, 200);
    assert.equal(checked.body.online_release.version, "2.0.0");
    assert.equal(checked.body.online_release.verified, true);
  });
});

test("a download is refused until the server has trusted a newer checked release", async () => {
  const current = JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8")).version;
  const payload = releasePayload(current);
  await withUpdateMiddleware({
    environment: { XIRAI_UPDATE_FEED: "https://updates.example/latest.json" },
    fetcher: feedFetcher({ getPayload: () => payload }),
  }, async (invoke) => {
    const unchecked = await invoke("/api/system/update/download", "POST");
    assert.equal(unchecked.statusCode, 409);
    assert.match(unchecked.body.error, /请先检查更新/);

    const checked = await invoke("/api/system/update/check", "POST");
    assert.equal(checked.statusCode, 200);
    const currentDownload = await invoke("/api/system/update/download", "POST");
    assert.equal(currentDownload.statusCode, 409);
    assert.match(currentDownload.body.error, /已是最新版本/);
  });
});

test("invalid release metadata is a visible API error rather than an up-to-date result", async (t) => {
  const cases = [
    ["wrong archive", releasePayload("2.0.0", { assets: [
      { ...releasePayload("2.0.0").assets[0], name: "prefix-XirAI-2.0.0.7z" },
      releasePayload("2.0.0").assets[1],
    ] }), /缺少精确命名的更新包/],
    ["no sidecar", releasePayload("2.0.0", { assets: [releasePayload("2.0.0").assets[0]] }), /缺少精确命名的校验和文件/],
    ["non-uploaded archive", releasePayload("2.0.0", { assets: [
      { ...releasePayload("2.0.0").assets[0], state: "new" },
      releasePayload("2.0.0").assets[1],
    ] }), /尚未完成上传/],
    ["prerelease tag", releasePayload("2.0.0", { tag_name: "v2.0.0-beta.1", prerelease: true }), /预发布/],
  ];
  for (const [name, payload, message] of cases) {
    await t.test(name, async () => {
      await withUpdateMiddleware({
        environment: { XIRAI_UPDATE_FEED: "https://updates.example/latest.json" },
        fetcher: feedFetcher({ getPayload: () => payload }),
      }, async (invoke) => {
        const result = await invoke("/api/system/update/check", "POST");
        assert.equal(result.statusCode, 502);
        assert.match(result.body.error, message);
        const state = await invoke("/api/system/update");
        assert.equal(state.body.online_release, null);
      });
    });
  }
});

test("malformed, unavailable, or contradictory sidecars fail closed", async (t) => {
  const payload = releasePayload("2.0.0");
  const cases = [
    ["malformed", { sidecarBody: `${sha}  prefix-XirAI-2.0.0.7z\n` }, /格式无效|文件名不匹配/],
    ["unavailable", { sidecarStatus: 404 }, /HTTP 404/],
    ["digest mismatch", { sidecarBody: `${"b".repeat(64)}  XirAI-2.0.0.7z\n` }, /不一致/],
  ];
  for (const [name, options, message] of cases) {
    await t.test(name, async () => {
      await withUpdateMiddleware({
        environment: { XIRAI_UPDATE_FEED: "https://updates.example/latest.json" },
        fetcher: feedFetcher({ getPayload: () => payload, ...options }),
      }, async (invoke) => {
        const result = await invoke("/api/system/update/check", "POST");
        assert.equal(result.statusCode, 502);
        assert.match(result.body.error, message);
        const download = await invoke("/api/system/update/download", "POST");
        assert.equal(download.statusCode, 409);
        assert.match(download.body.error, /请先检查更新/);
      });
    });
  }
});

test("release JSON and sidecar bodies are capped at 1 MiB and 16 KiB", async (t) => {
  await t.test("release payload cap", async () => {
    const fetcher = async () => response("x".repeat(1024 ** 2 + 1));
    await withUpdateMiddleware({
      environment: { XIRAI_UPDATE_FEED: "https://updates.example/latest.json" }, fetcher,
    }, async (invoke) => {
      const result = await invoke("/api/system/update/check", "POST");
      assert.equal(result.statusCode, 502);
      assert.match(result.body.error, /1048576 字节限制/);
    });
  });

  await t.test("sidecar cap", async () => {
    const payload = releasePayload("2.0.0");
    const fetcher = feedFetcher({ getPayload: () => payload, sidecarBody: "x".repeat(16 * 1024 + 1) });
    await withUpdateMiddleware({
      environment: { XIRAI_UPDATE_FEED: "https://updates.example/latest.json" }, fetcher,
    }, async (invoke) => {
      const result = await invoke("/api/system/update/check", "POST");
      assert.equal(result.statusCode, 502);
      assert.match(result.body.error, /16384 字节限制/);
    });
  });
});

test("release and sidecar deadlines remain active while response bodies are read", async (t) => {
  await t.test("release body timeout", async () => {
    await withUpdateMiddleware({
      environment: { XIRAI_UPDATE_FEED: "https://updates.example/latest.json" },
      fetcher: async () => hangingResponse(),
      releaseTimeoutMs: 20,
    }, async (invoke) => {
      const result = await invoke("/api/system/update/check", "POST");
      assert.equal(result.statusCode, 502);
      assert.match(result.body.error, /响应超时/);
    });
  });

  await t.test("sidecar body timeout", async () => {
    const payload = releasePayload("2.0.0");
    const fetcher = async (input) => String(input).endsWith("latest.json")
      ? response(JSON.stringify(payload))
      : hangingResponse();
    await withUpdateMiddleware({
      environment: { XIRAI_UPDATE_FEED: "https://updates.example/latest.json" },
      fetcher,
      sidecarTimeoutMs: 20,
    }, async (invoke) => {
      const result = await invoke("/api/system/update/check", "POST");
      assert.equal(result.statusCode, 502);
      assert.match(result.body.error, /校验和.*响应超时/);
    });
  });
});

test("custom feeds never receive or reuse the GitHub PAT", async () => {
  const requests = [];
  const payload = releasePayload("2.0.0");
  const environment = {
    XIRAI_UPDATE_FEED: "https://api.github.com/custom-feed.json",
    XIRAI_UPDATE_TOKEN: "private-pat",
  };
  await withUpdateMiddleware({
    environment,
    fetcher: feedFetcher({ getPayload: () => payload, feedUrl: environment.XIRAI_UPDATE_FEED, requests }),
  }, async (invoke) => {
    const result = await invoke("/api/system/update/check", "POST");
    assert.equal(result.statusCode, 200);
    assert.equal(result.body.online_release.verified, true);
  });
  assert.equal(requests.length, 2);
  assert.ok(requests.every((request) => request.authorization == null));
  assert.ok(requests[1].url.endsWith(".sha256"), "custom feed sidecar must use the unauthenticated browser URL");
});

test("private GitHub feed and sidecar API requests receive PAT only on trusted hosts", async () => {
  const requests = [];
  const payload = releasePayload("2.0.0");
  const environment = { XIRAI_UPDATE_REPO: "o/r", XIRAI_UPDATE_TOKEN: "private-pat" };
  const feedUrl = "https://api.github.com/repos/o/r/releases/latest";
  await withUpdateMiddleware({
    environment,
    fetcher: feedFetcher({ getPayload: () => payload, feedUrl, requests }),
  }, async (invoke) => {
    const result = await invoke("/api/system/update/check", "POST");
    assert.equal(result.statusCode, 200);
    assert.equal(result.body.online_release.verified, true);
  });
  assert.deepEqual(requests.map((request) => request.url), [
    feedUrl,
    "https://api.github.com/repos/o/r/releases/assets/checksum-2.0.0",
  ]);
  assert.ok(requests.every((request) => request.authorization === "Bearer private-pat"));
  assert.equal(requests[1].headers.get("accept"), "application/octet-stream");
});

test("private GitHub repository probe is authenticated and distinguishes a missing release", async () => {
  const requests = [];
  const environment = { XIRAI_UPDATE_REPO: "o/private", XIRAI_UPDATE_TOKEN: "private-pat" };
  const fetcher = async (input, options = {}) => {
    const url = String(input);
    requests.push({ url, authorization: requestAuthorization(options) });
    if (url.endsWith("/releases/latest")) return response("{}", { status: 404 });
    if (url === "https://api.github.com/repos/o/private") return response("{}");
    throw new Error(`unexpected fake request: ${url}`);
  };
  await withUpdateMiddleware({ environment, fetcher }, async (invoke) => {
    const result = await invoke("/api/system/update/check", "POST");
    assert.equal(result.statusCode, 200);
    assert.equal(result.body.online_release, null);
    assert.ok(result.body.online_checked_at);
  });
  assert.equal(requests.length, 2);
  assert.ok(requests.every((request) => request.authorization === "Bearer private-pat"));
});

test("invalid nonempty repository configuration is surfaced without any request", async () => {
  let fetched = false;
  await withUpdateMiddleware({
    environment: { XIRAI_UPDATE_REPO: "not a repo" },
    fetcher: async () => { fetched = true; throw new Error("must not fetch"); },
  }, async (invoke) => {
    const result = await invoke("/api/system/update/check", "POST");
    assert.equal(result.statusCode, 500);
    assert.match(result.body.error, /XIRAI_UPDATE_REPO 格式无效/);
  });
  assert.equal(fetched, false);
});

test("a failed refresh revokes the previously verified onlineRelease download authority", async () => {
  let payload = releasePayload("2.0.0");
  let sidecarBody = `${sha}  XirAI-2.0.0.7z\n`;
  const dynamicFetcher = async (input, options) => {
    const url = String(input);
    if (url.endsWith("latest.json")) return response(JSON.stringify(payload));
    return response(sidecarBody);
  };
  await withUpdateMiddleware({
    environment: { XIRAI_UPDATE_FEED: "https://updates.example/latest.json" },
    fetcher: dynamicFetcher,
  }, async (invoke) => {
    const first = await invoke("/api/system/update/check", "POST");
    assert.equal(first.statusCode, 200);
    assert.equal(first.body.online_release.update_available, true);

    sidecarBody = `${sha}  attacker-XirAI-2.0.0.7z\n`;
    const failed = await invoke("/api/system/update/check", "POST");
    assert.equal(failed.statusCode, 502);
    const download = await invoke("/api/system/update/download", "POST");
    assert.equal(download.statusCode, 409);
    assert.match(download.body.error, /请先检查更新/);
    const state = await invoke("/api/system/update");
    assert.equal(state.body.online_release, null);
    assert.equal(state.body.online_checked_at, null);
  });
});

test("token-bearing asset redirects strip Authorization before object storage", async () => {
  const { onlineUpdateNetworkInternals } = await import(`../vite.config.js?redirect-test=${Date.now()}`);
  const calls = [];
  const fetcher = async (input, options = {}) => {
    calls.push({ url: String(input), authorization: requestAuthorization(options), redirect: options.redirect });
    if (calls.length === 1) {
      return response("", {
        status: 302,
        headers: { Location: "https://objects.githubusercontent.com/private/signed-asset" },
      });
    }
    return response("archive bytes", { status: 206 });
  };
  const result = await onlineUpdateNetworkInternals.fetchDownload(
    "https://api.github.com/repos/o/r/releases/assets/123",
    { headers: { Authorization: "Bearer private-pat" } },
    { fetcher, environment: { XIRAI_UPDATE_TOKEN: "private-pat" } },
  );
  assert.equal(result.status, 206);
  assert.deepEqual(calls, [
    {
      url: "https://api.github.com/repos/o/r/releases/assets/123",
      authorization: "Bearer private-pat",
      redirect: "manual",
    },
    {
      url: "https://objects.githubusercontent.com/private/signed-asset",
      authorization: null,
      redirect: "manual",
    },
  ]);
  await assert.rejects(
    onlineUpdateNetworkInternals.fetchDownload(
      "https://evil.example/archive",
      { headers: { Authorization: "Bearer private-pat" } },
      { fetcher, environment: { XIRAI_UPDATE_TOKEN: "private-pat" } },
    ),
    /拒绝向非受信任地址发送更新令牌/,
  );
});

test("Vite download sends private route headers only to the returned official route", async () => {
  const [{ onlineUpdateNetworkInternals }, { parseRelease, releaseDownloadRoutes }] = await Promise.all([
    import(`../vite.config.js?route-header-test=${Date.now()}`),
    import("./release-feed.mjs"),
  ]);
  const token = "private-pat";
  const environment = {
    XIRAI_UPDATE_TOKEN: token,
    XIRAI_UPDATE_MIRROR: "https://mirror.example/gh",
  };
  const routes = releaseDownloadRoutes(parseRelease(releasePayload("2.0.0")), {
    checksum: sha,
    environment,
    privateContext: true,
  });
  const calls = [];
  const fetcher = async (input, options = {}) => {
    calls.push({ url: String(input), authorization: requestAuthorization(options) });
    return response("sample", { status: 206 });
  };
  for (const route of routes) {
    await onlineUpdateNetworkInternals.fetchDownload(
      route.url,
      { headers: route.headers },
      { fetcher, environment },
    );
  }
  assert.deepEqual(calls, [
    {
      url: "https://mirror.example/gh/https://github.com/o/r/releases/download/v2.0.0/XirAI-2.0.0.7z",
      authorization: null,
    },
    {
      url: "https://api.github.com/repos/o/r/releases/assets/archive-2.0.0",
      authorization: `Bearer ${token}`,
    },
  ]);
});

test("token-bearing sidecar body redirects also strip Authorization before object storage", async () => {
  const { onlineUpdateNetworkInternals } = await import(`../vite.config.js?sidecar-redirect-test=${Date.now()}`);
  const calls = [];
  const fetcher = async (input, options = {}) => {
    calls.push({ url: String(input), authorization: requestAuthorization(options) });
    if (calls.length === 1) {
      return response("", {
        status: 302,
        headers: { Location: "https://objects.githubusercontent.com/private/signed-sidecar" },
      });
    }
    return response(`${sha}  XirAI-2.0.0.7z\n`);
  };
  const result = await onlineUpdateNetworkInternals.fetchBoundedUpdateBody(
    "https://api.github.com/repos/o/r/releases/assets/456",
    {
      headers: { Authorization: "Bearer private-pat", Accept: "application/octet-stream" },
      timeoutMs: 1000,
      maximumBytes: 16 * 1024,
      fetcher,
      environment: { XIRAI_UPDATE_TOKEN: "private-pat" },
      label: "校验和文件",
    },
  );
  assert.equal(result.body.toString("utf8"), `${sha}  XirAI-2.0.0.7z\n`);
  assert.deepEqual(calls, [
    { url: "https://api.github.com/repos/o/r/releases/assets/456", authorization: "Bearer private-pat" },
    { url: "https://objects.githubusercontent.com/private/signed-sidecar", authorization: null },
  ]);
});
