import assert from "node:assert/strict";
import http from "node:http";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { inferenceBackendPlugin, inferenceWorkspaceId } from "../vite.config.js";
import { INFERENCE_IDENTITY_GATE_ERROR, guardedInferenceResponse, inferenceIdentityGate } from "./inference-identity-gate.mjs";

// The health fixture has to agree with the shipped `inferenceProtocol`, because the tests below
// drive the real plugin from `vite.config.js`. Drift between the four declarations of the
// protocol is owned by `inference-protocol.test.mjs`.
const protocol = 34;
const expected = { status: "ready", protocol, workspace_id: inferenceWorkspaceId };

function middlewareServer() {
  const handlers = [];
  return {
    middlewares: { use(handler) { handlers.push(handler); } },
    handlers,
  };
}

async function requestThrough(handlers, pathname) {
  const listener = http.createServer((request, response) => {
    let index = 0;
    const next = (error) => {
      if (error) { response.statusCode = 500; response.end(error.message); return; }
      const handler = handlers[index++];
      if (!handler) { response.statusCode = 404; response.end("no downstream"); return; }
      Promise.resolve(handler(request, response, next)).catch(next);
    };
    next();
  });
  await new Promise((resolve) => listener.listen(0, "127.0.0.1", resolve));
  const { port } = listener.address();
  try {
    return await new Promise((resolve, reject) => {
      const request = http.get({ host: "127.0.0.1", port, path: pathname }, (response) => {
        let body = "";
        response.setEncoding("utf8");
        response.on("data", (chunk) => { body += chunk; });
        response.on("end", () => resolve({ statusCode: response.statusCode, body }));
      });
      request.once("error", reject);
    });
  } finally {
    await new Promise((resolve, reject) => listener.close((error) => error ? reject(error) : resolve()));
  }
}

async function requestsThrough(handlers, pathnames) {
  const listener = http.createServer((request, response) => {
    let index = 0;
    const next = (error) => {
      if (error) { response.statusCode = 500; response.end(error.message); return; }
      const handler = handlers[index++];
      if (!handler) { response.statusCode = 404; response.end("no downstream"); return; }
      Promise.resolve(handler(request, response, next)).catch(next);
    };
    next();
  });
  await new Promise((resolve) => listener.listen(0, "127.0.0.1", resolve));
  const { port } = listener.address();
  try {
    return await Promise.all(pathnames.map((pathname) => new Promise((resolve, reject) => {
      const request = http.get({ host: "127.0.0.1", port, path: pathname }, (response) => {
        let body = "";
        response.setEncoding("utf8");
        response.on("data", (chunk) => { body += chunk; });
        response.on("end", () => resolve({ statusCode: response.statusCode, body }));
      });
      request.once("error", reject);
    })));
  } finally {
    await new Promise((resolve, reject) => listener.close((error) => error ? reject(error) : resolve()));
  }
}

/** Every <img> in the source, including the multi-line JSX ones. */
function imageTags(source) {
  return [...source.matchAll(/<img\b[\s\S]*?\/>/g)].map((match) => match[0]);
}

test("inference identity gate allows only a ready matching backend and recovers", () => {
  assert.deepEqual(inferenceIdentityGate(expected, protocol, inferenceWorkspaceId), { allowed: true, reason: "matched" });
  assert.deepEqual(inferenceIdentityGate({ ...expected, protocol: protocol - 1 }, protocol, inferenceWorkspaceId), { allowed: false, reason: "protocol_mismatch" });
  assert.deepEqual(inferenceIdentityGate({ ...expected, workspace_id: "workspace-b" }, protocol, inferenceWorkspaceId), { allowed: false, reason: "workspace_mismatch" });
  assert.deepEqual(inferenceIdentityGate(null, protocol, inferenceWorkspaceId), { allowed: false, reason: "offline_or_not_ready" });
  assert.equal(INFERENCE_IDENTITY_GATE_ERROR, "INFERENCE_BACKEND_IDENTITY_UNAVAILABLE");
});

for (const hook of ["configureServer", "configurePreviewServer"]) {
  test(`${hook} registers the production identity gate before downstream inference proxy middleware`, async () => {
    let health = expected;
    let downstreamHits = 0;
    // TTL 0 keeps this suite probing per request, so every identity flip below
    // is observed immediately; the shared-observation window has its own test.
    const plugin = inferenceBackendPlugin({ testOnly: true, gateProbeTtlMs: 0, probeBackend: async () => {
      if (health instanceof Error) throw health;
      return health;
    } });
    const server = middlewareServer();
    await plugin[hook](server);
    server.middlewares.use((_request, response) => {
      downstreamHits += 1;
      response.statusCode = 204;
      response.end();
    });

    const allowed = await requestThrough(server.handlers, "/api/inference/jobs?resume=1");
    assert.equal(allowed.statusCode, 204);
    assert.equal(downstreamHits, 1, "matching identity reaches business proxy sentinel");

    for (const [value, reason] of [
      [null, "offline_or_not_ready"],
      [new Error("timeout"), "offline_or_not_ready"],
      [new Error("invalid JSON"), "offline_or_not_ready"],
      [{ ...expected, protocol: 26 }, "protocol_mismatch"],
      [{ ...expected, workspace_id: "other" }, "workspace_mismatch"],
      [{ ...expected, status: "starting" }, "offline_or_not_ready"],
    ]) {
      health = value;
      const blocked = await requestThrough(server.handlers, "/api/inference/jobs/");
      assert.equal(blocked.statusCode, 503);
      assert.deepEqual(JSON.parse(blocked.body), { error: INFERENCE_IDENTITY_GATE_ERROR, reason });
      assert.equal(downstreamHits, 1, `${reason} must not reach business proxy sentinel`);
    }

    health = expected;
    const recovered = await requestThrough(server.handlers, "/api/inference/jobs/active?after=recovery");
    assert.equal(recovered.statusCode, 204);
    assert.equal(downstreamHits, 2, "matching health recovery reopens the gate");

    health = null;
    const aggregateHealth = await requestThrough(server.handlers, "/api/inference/health?aggregate=1");
    assert.equal(aggregateHealth.statusCode, 200);
    assert.equal(downstreamHits, 2, "exact health pathname is served by aggregate handler, never business proxy");

    const trailingHealth = await requestThrough(server.handlers, "/api/inference/health/");
    assert.equal(trailingHealth.statusCode, 503, "only exact health pathname bypasses the gate");
    assert.equal(downstreamHits, 2);
  });
}

test("pure gate response retains stable fail-closed error", () => {
  assert.deepEqual(guardedInferenceResponse("/api/inference/jobs", null, protocol, inferenceWorkspaceId), {
    statusCode: 503, body: { error: INFERENCE_IDENTITY_GATE_ERROR, reason: "offline_or_not_ready" },
  });
});

test("the gate shares one backend probe across a burst instead of scanning per request", async () => {
  // `/api/inference/health` walks the ADetailer, upscaler and Anima model trees
  // on every call — roughly 300 ms against a populated models/ directory. The
  // gate ran it per proxied request, so opening a gallery card paid a directory
  // scan for the image and one more for every thumbnail beside it.
  let probes = 0;
  const plugin = inferenceBackendPlugin({ testOnly: true, gateProbeTtlMs: 1000, probeBackend: async () => {
    probes += 1;
    await new Promise((resolve) => setTimeout(resolve, 5));
    return expected;
  } });
  const server = middlewareServer();
  await plugin.configureServer(server);
  let downstreamHits = 0;
  server.middlewares.use((_request, response) => {
    downstreamHits += 1;
    response.statusCode = 204;
    response.end();
  });

  // A card opening: one full image plus a thumbnail strip, all at once.
  const burst = await requestsThrough(server.handlers, Array.from({ length: 8 }, (_, index) => `/api/inference/gallery/images/img-${index}`));
  assert.deepEqual([...new Set(burst.map((item) => item.statusCode))], [204], "every image is served");
  assert.equal(downstreamHits, 8, "the gate still admits each request individually");
  assert.equal(probes, 1, "a concurrent burst shares one identity observation");

  // Still inside the window: reuse rather than re-scan.
  const later = await requestThrough(server.handlers, "/api/inference/gallery/images/img-again");
  assert.equal(later.statusCode, 204);
  assert.equal(probes, 1, "a result inside the window is reused");
});

test("a shared probe expires, is never cached on failure, and never outlives a restart", async () => {
  let health = expected;
  let probes = 0;
  const plugin = inferenceBackendPlugin({ testOnly: true, gateProbeTtlMs: 30, probeBackend: async () => {
    probes += 1;
    return health;
  } });
  const server = middlewareServer();
  await plugin.configureServer(server);
  server.middlewares.use((_request, response) => { response.statusCode = 204; response.end(); });

  assert.equal((await requestThrough(server.handlers, "/api/inference/jobs")).statusCode, 204);
  assert.equal(probes, 1);

  // The window is a sharing window, not a trust window: once it lapses the next
  // request re-verifies, so a backend that changed identity is caught.
  await new Promise((resolve) => setTimeout(resolve, 45));
  health = { ...expected, protocol: 26 };
  const stale = await requestThrough(server.handlers, "/api/inference/jobs");
  assert.equal(stale.statusCode, 503);
  assert.deepEqual(JSON.parse(stale.body), { error: INFERENCE_IDENTITY_GATE_ERROR, reason: "protocol_mismatch" });
  assert.equal(probes, 2, "an expired observation is re-probed");

  // A refusal is never cached, so recovery is immediate rather than delayed by
  // a stale negative sitting in the window.
  health = expected;
  assert.equal((await requestThrough(server.handlers, "/api/inference/jobs")).statusCode, 204);
  assert.equal(probes, 3, "a failed probe leaves nothing behind to reuse");

  // The aggregate health route keeps its own live probe: it is the liveness
  // display and the first thing to notice a restart.
  const before = probes;
  await requestThrough(server.handlers, "/api/inference/health");
  assert.ok(probes > before, "the health route never serves a shared observation");
});

test("only the opened picture loads a full-resolution original", async () => {
  const gallery = await readFile(new URL("../src/GalleryPage.jsx", import.meta.url), "utf8");
  // Curated images are stored at generation size — a 2048x2944 PNG is six
  // megapixels and five megabytes. A grid tile renders one at ~300 px and the
  // focus strip at ~60 px, so anything drawn small must ask for the derivative.
  assert.match(gallery, /function thumbUrl\(image\)/);
  assert.match(gallery, /image\?\.thumb_url \|\| image\?\.url \|\| ""/,
    "records written before thumbnails existed must still render");
  for (const [what, pattern] of [
    ["grid tile", /\{first \? <img src=\{thumbUrl\(first\)\}/],
    ["focus strip", /<img src=\{thumbUrl\(item\)\}/],
    ["editor preview", /src=\{image\.kind === "existing" \? thumbUrl\(image\) : image\.url\}/],
  ]) {
    assert.match(gallery, pattern, `${what} must request the derivative`);
  }
  // The stage paints the cached thumbnail first, then swaps in the original once
  // it has decoded, so opening a card is never a blank wait.
  assert.match(gallery, /function FocusImage\(/);
  assert.match(gallery, /className="gallery-focus-preview"/);
  assert.match(gallery, /onLoad=\{\(\) => setLoaded\(true\)\}/);
  assert.match(gallery, /onError=\{\(\) => setLoaded\(true\)\}/,
    "a failed original must not leave the stage stuck on the placeholder");
  assert.match(gallery, /useEffect\(\(\) => setLoaded\(false\), \[image\.url\]\)/,
    "switching image resets the swap");

  const styles = await readFile(new URL("../src/styles.css", import.meta.url), "utf8");
  // Originals can exceed 20 MP. A per-pixel filter can exhaust the browser's
  // GPU texture budget and paint the decoded image as a solid black rectangle.
  assert.doesNotMatch(styles, /\.gallery-focus-preview \{[^}]*drop-shadow/);
  assert.match(styles, /\.gallery-focus-frame\.ready \.gallery-focus-full \{ opacity: 1; \}/);
  assert.doesNotMatch(styles, /\.gallery-focus-frame\.ready \.gallery-focus-full \{[^}]*drop-shadow/);
  assert.match(styles, /prefers-reduced-motion[\s\S]*?\.gallery-focus-full \{ opacity: 1; \}/,
    "the picture must still become visible without the cross-fade");
});

test("gallery images never block the panel beside them on a synchronous decode", async () => {
  const gallery = await readFile(new URL("../src/GalleryPage.jsx", import.meta.url), "utf8");
  // Curated images are full-resolution originals — multiple megabytes each in a
  // normal library. Decoded synchronously they hold up the frame, so the card's
  // parameters appeared only once the picture was ready.
  const images = imageTags(gallery);
  assert.ok(images.length >= 3, "the grid tile, focus stage and thumbnail strip all render images");
  for (const tag of images) {
    // The editor previews object URLs of files the user just picked; everything
    // that renders a stored gallery asset must decode off the main thread.
    if (!/src=\{(?:first|item|image)\.url\}/.test(tag)) continue;
    assert.match(tag, /decoding="async"/, `blocking decode: ${tag}`);
  }
  // Only the picture the user actually opened is worth prioritising; the strip
  // and the grid stay lazy so opening a card does not fetch every original.
  // The stage renders FocusImage now, so the hint lives on its full-size layer.
  assert.match(gallery, /className="gallery-focus-full"[\s\S]*?fetchPriority="high"/);
  assert.equal([...gallery.matchAll(/fetchPriority="high"/g)].length, 1);
  const previewTag = imageTags(gallery).find((tag) => tag.includes("gallery-focus-preview"));
  assert.ok(previewTag, "the placeholder layer must exist");
  assert.ok(!previewTag.includes("fetchPriority"),
    "the placeholder must never compete with the original it stands in for");
  for (const tag of images.filter((item) => /src=\{(?:first|item|image)\.url\}/.test(item) && !/fetchPriority/.test(item))) {
    assert.match(tag, /loading="lazy"/, `eager offscreen image: ${tag}`);
  }
});
